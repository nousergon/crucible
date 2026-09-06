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

**THE TYPED BOUNDARY (`alpha-engine-config-I10045` row 1).** A
`run_manifest.v2` document is validated whole, once, through
`crucible.models.RunManifestV2` — never by hand-walking `dict` keys — so a
malformed manifest surfaces here, naming the row and the field, rather than
as a `KeyError` in a consumer several functions away. `run_manifest.v2.json`
is GENERATED from that model (`tests/test_manifest_schema.py` fails when the
committed file and the generated one differ); the two status<->reason
cross-field rules stay in the model as `RunManifestV2._status_and_reason_agree`
and are deliberately not re-encoded as the schema's `allOf` — see
`crucible/models.py`'s module docstring. `run_manifest.v1` stays on the
original jsonschema-only path below: it is FROZEN and carries no model.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from pydantic import ValidationError as PydanticValidationError

from crucible.documents import load_store_document
from crucible.keys import manifest_key, manifest_prefix  # noqa: F401 - re-exported
from crucible.models import RunManifestV2

#: The version every producer writes TODAY. Bumped to v2 by
#: alpha-engine-config-I9918: v1 declared no live/replay field and set
#: `additionalProperties: false`, so no producer could say whether a run was
#: a live Saturday or a replay of a historical one, and phase 2's exit gate
#: was permanently unmeasurable. v2 adds one REQUIRED field, `run_mode`, and
#: is otherwise v1's contract unchanged.
#:
#: **A new version rather than a v1 addition, and nothing is backfilled.** A
#: required field added to v1 would retroactively invalidate every manifest
#: already in the store, and `read_manifest` validates on READ — so the board,
#: the morning report and the alert sweep would all start refusing documents
#: that were correct when they were written. Instead each document is checked
#: against the version it declares (see :func:`validate`): a v1 object stays
#: valid as v1, forever, and is never rewritten. Backfilling one would mean
#: inventing a `run_mode` for a run nobody observed, which is the false
#: liveness claim the field exists to prevent.
RUN_MANIFEST_SCHEMA_VERSION = "run_manifest.v2"

#: v2's predecessor. Frozen: it gains no fields and no producer writes it.
#: It stays in the package because the objects written under it are still in
#: the store and are still read.
PREDECESSOR_SCHEMA_VERSION = "run_manifest.v1"

SCHEMA_DIR = Path(__file__).parent / "schemas"

#: Every run-manifest version this package can check a document against,
#: newest first. Not a suppression list and not an exemption list — each entry
#: is a real, strict schema that shipped, and a document is graded against the
#: one it declares rather than waved through.
SCHEMA_PATHS: dict[str, Path] = {
    RUN_MANIFEST_SCHEMA_VERSION: SCHEMA_DIR / "run_manifest.v2.json",
    PREDECESSOR_SCHEMA_VERSION: SCHEMA_DIR / "run_manifest.v1.json",
}

SCHEMA_PATH = SCHEMA_PATHS[RUN_MANIFEST_SCHEMA_VERSION]

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
    """The CURRENT run-manifest schema, as a dict.

    Cached because the validator is constructed per process, and the file
    never changes within one. A test that mutates what this returns must call
    ``load_schema.cache_clear()``.
    """
    return load_schema_for(RUN_MANIFEST_SCHEMA_VERSION)


@lru_cache(maxsize=len(SCHEMA_PATHS))
def load_schema_for(version: str) -> dict[str, Any]:
    """The run-manifest schema for ``version``.

    An unrecognised version is refused rather than checked against the newest
    schema on the assumption that it is close enough — a document declaring a
    version this build does not carry is a document this build cannot make any
    claim about.
    """
    path = SCHEMA_PATHS.get(version)
    if path is None:
        raise ManifestValidationError(
            f"run manifest declares schema_version {version!r}; this build carries "
            f"{sorted(SCHEMA_PATHS)}. A document whose contract this process does not "
            "ship cannot be checked, and an unchecked manifest read as valid is worse "
            "than none."
        )
    if not path.is_file():
        raise FileNotFoundError(
            f"run manifest schema missing at {path}. It ships inside the "
            "package; a missing schema means a broken build, not a degraded run."
        )
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=len(SCHEMA_PATHS))
def _validator(version: str) -> Draft202012Validator:
    """A checked validator per version.

    FIVE caches sit over these schema files, and a test that mutates one must
    clear every cache that could still be holding the unmutated copy: this
    one, :func:`load_schema`, :func:`load_schema_for`, and — over the same
    documents, in another module — `crucible.gate._manifest_property_names`
    and `crucible.gate._manifest_status_values`. Clearing three of the five
    grades a mutation against the file on disk and passes for the wrong
    reason.
    """
    schema = load_schema_for(version)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def validate(manifest: dict[str, Any]) -> None:
    """Raise :class:`ManifestValidationError` unless ``manifest`` conforms.

    **Checked against the version the document itself declares.** Producers
    write :data:`RUN_MANIFEST_SCHEMA_VERSION` and nothing else, so on the
    write path this is always the current schema; on the read path it is what
    lets a manifest written before v2 stay readable at its own contract
    instead of being condemned by a field that did not exist when it was
    written (alpha-engine-config-I9918). Each version is checked strictly —
    a v1 document is held to every one of v1's rules.

    Every error is reported, not just the first: a writer fixing one field at
    a time against a validator that reports one error at a time is how a
    half-conformant producer ships.

    **The current version goes through `crucible.models.RunManifestV2`**
    (`alpha-engine-config-I10045` row 1), which also enforces the
    status<->reason cross-field rule the published schema no longer carries
    (see this module's and `crucible.models`' docstrings). `run_manifest.v1`
    is unaffected — it stays on the jsonschema-only path it always used,
    frozen.
    """
    declared = manifest.get("schema_version")
    if not isinstance(declared, str):
        raise ManifestValidationError(
            f"run manifest declares no schema_version (got {declared!r}). The version "
            "is what says which contract the document was written to; a consumer that "
            "cannot read it refuses the document rather than guessing."
        )
    if declared == RUN_MANIFEST_SCHEMA_VERSION:
        try:
            RunManifestV2.model_validate(manifest)
        except PydanticValidationError as exc:
            detail = "\n".join(
                f"  - {'/'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}"
                for e in exc.errors()
            )
            raise ManifestValidationError(
                f"run manifest does not conform to {declared}:\n{detail}"
            ) from exc
        return
    errors = sorted(_validator(declared).iter_errors(manifest), key=lambda e: list(e.absolute_path))
    if not errors:
        return
    detail = "\n".join(
        f"  - {'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}" for e in errors
    )
    raise ManifestValidationError(f"run manifest does not conform to {declared}:\n{detail}")


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
    document = load_store_document(
        store, manifest_key(job, trading_day, discriminator=discriminator)
    )
    validate(document)
    return document
