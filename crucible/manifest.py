"""The run manifest: the one record every job writes, and its validator.

Normative source: `crucible_v2_rebuild_plan_260901.md` §4.2 and §9.2.

A manifest is not a log line — it is the durable artifact from which a
verdict is reconstructed (principle 1) and the surface the two page
conditions read (§4.6). Three properties follow, and every one of them is
enforced here rather than requested:

1. **Two statuses, exhaustively.** `ok` or `failed`. `reason` is mandatory
   and must be non-empty on failure and empty on success. There is no
   `partial`, `skipped`, `degraded` or `unknown` to fall into.
2. **Trading day is the key.** `trading_day` and `calendar_date` are
   separate typed fields; only the first is ever a key.
3. **All five signal classes in one document**, under one `run_id`.

Validation is a runtime dependency, not a test-only one: a writer that could
emit an invalid manifest would defeat the schema entirely.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

RUN_MANIFEST_SCHEMA_VERSION = "run_manifest.v1"

SCHEMA_DIR = Path(__file__).parent / "schemas"
SCHEMA_PATH = SCHEMA_DIR / "run_manifest.v1.json"

#: The exhaustive status set. Imported by the runner and by tests so that
#: adding a third state requires editing this line, where the reason it must
#: not happen is written down.
STATUSES: tuple[str, ...] = ("ok", "failed")

#: The declared transient-retry class (§11 risk 2). A failure outside this
#: set pages immediately. The set grows only by PR with the failure named.
TRANSIENT_RETRY_REASONS: tuple[str, ...] = (
    "spot_interruption",
    "provider_5xx",
    "provider_timeout",
    "s3_throttling",
)


class ManifestValidationError(ValueError):
    """A manifest that does not conform. Always raised, never logged and
    swallowed: an unvalidated manifest read as valid is worse than none, and
    the whole design rests on the manifest being trustworthy."""


@lru_cache(maxsize=1)
def load_schema() -> dict[str, Any]:
    """The v1 run-manifest schema, as a dict.

    Cached because the validator is constructed per process, and the file
    never changes within one.
    """
    if not SCHEMA_PATH.is_file():
        raise FileNotFoundError(
            f"run manifest schema missing at {SCHEMA_PATH}. It ships inside the "
            "package; a missing schema means a broken build, not a degraded run."
        )
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    schema = load_schema()
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def validate(manifest: dict[str, Any]) -> None:
    """Raise :class:`ManifestValidationError` unless ``manifest`` conforms.

    Every error is reported, not just the first: a writer fixing one field at
    a time against a validator that reports one error at a time is how a
    half-conformant producer ships.
    """
    errors = sorted(_validator().iter_errors(manifest), key=lambda e: list(e.absolute_path))
    if not errors:
        return
    detail = "\n".join(
        f"  - {'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}" for e in errors
    )
    raise ManifestValidationError(
        f"run manifest does not conform to {RUN_MANIFEST_SCHEMA_VERSION}:\n{detail}"
    )


#: A discriminator is a path segment, not free text: it must round-trip
#: through every path-shaped tool a key passes through (a `LocalStore` path,
#: `aws s3 cp`, a URL) the same way `crucible.keys.arm_key_segment` protects
#: arm ids. Slot letters (`u`/`r`/`m`/`s`) and ISO calendar dates both already
#: satisfy this, so no translation table is needed — only a refusal of
#: anything that would not.
_DISCRIMINATOR_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def manifest_key(job: str, trading_day: str, *, discriminator: str | None = None) -> str:
    """The store key a manifest is written under.

    The trading day is the key (§4.12). This function is the single place
    that shape is expressed, so the trading-day contract test has one thing
    to walk.

    ``discriminator`` distinguishes multiple manifests a single job
    legitimately writes for one trading day — the slot letter for
    `experiment.run`/`experiment.grade` (four slots, one job name, one
    trading day, four writers), or a firing's own `calendar_date` for a job
    like `alerts.sweep` that runs more than once between two trading-day
    rollovers. Omitted (the default) for every job that writes at most one
    manifest per trading day, which keeps the key shape unchanged for every
    existing caller (alpha-engine-config-I9781).

    The alternative weighed in I9781 — folding the slot into the job name
    itself (`experiment.run:r`) — was rejected: `job` is a closed enum the
    schema validates and `components.yaml` keys its one-row-per-job registry
    off it, so multiplying it by slot would multiply the registry too. A
    discriminator is an orthogonal path segment instead, so `job` keeps
    meaning "which of the thirteen CLI jobs wrote this" and nothing else.

    ``discriminator`` is never an arm id, so it does not go through
    `crucible.keys.arm_key_segment` — it is validated directly against a
    plain path-segment charset instead.
    """
    if not job:
        raise ValueError("job must be non-empty")
    if not trading_day:
        raise ValueError("trading_day must be non-empty")
    if discriminator is None:
        return f"runs/{job}/{trading_day}/run.json"
    if not _DISCRIMINATOR_RE.match(discriminator):
        raise ValueError(
            f"discriminator {discriminator!r} must be 1-64 characters of "
            "[A-Za-z0-9_.-] — it is a path segment, and this is the one place "
            "that is enforced so a future writer cannot orphan a manifest under "
            "a key no path-shaped tool can address."
        )
    return f"runs/{job}/{trading_day}/{discriminator}/run.json"


def manifest_prefix(job: str, trading_day: str) -> str:
    """The prefix under which every manifest for ``job`` on ``trading_day``
    lives, discriminated or not.

    `manifest_key(job, trading_day)` (bare) and
    `manifest_key(job, trading_day, discriminator=d)` (any ``d``) both start
    with this prefix, so a reader that does not know in advance whether a
    job writes one manifest or several per trading day — `crucible.alerts`,
    chiefly — lists this prefix rather than guessing a discriminator.
    """
    if not job:
        raise ValueError("job must be non-empty")
    if not trading_day:
        raise ValueError("trading_day must be non-empty")
    return f"runs/{job}/{trading_day}/"


def read_manifest(
    store: Any, job: str, trading_day: str, *, discriminator: str | None = None
) -> dict[str, Any]:
    """Read and validate one manifest from ``store``.

    Validated on READ as well as on write. A manifest written by an older
    release is still refused if it does not conform: a consumer that read a
    document it could not check would be reasoning from a shape nobody
    guarantees, which is the whole thing the schema exists to prevent.

    Raises :class:`KeyError` when the manifest is absent — never returns
    ``None``, because absence is one of the two page conditions (§4.6) and a
    caller that cannot tell "absent" from "empty" cannot raise it.
    """
    payload = store.get_bytes(manifest_key(job, trading_day, discriminator=discriminator))
    document = json.loads(payload.decode("utf-8"))
    validate(document)
    return document
