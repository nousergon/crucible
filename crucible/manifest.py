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
AND are mirrored into the published schema's `allOf` by
`_run_manifest_v2_json_schema_extra` (PR123 review finding 3) — see
`crucible/models.py`'s module docstring. `run_manifest.v1` stays on the
original jsonschema-only path below: it is FROZEN and carries no model.

**THE MONEY-PATH HASH CHAIN (`alpha-engine-config-I10414`, plan §9.5).**
:func:`write_manifest` is the single writer, and it is the only place a
`money_path_link` is produced. A run that wrote a money-path artifact is
appended to a hash chain whose every record carries the sha256 of its
predecessor's *stored bytes*; :func:`crucible.explain.verify_money_path_chain`
walks it and reports a break as `status: failed`. The predicates that decide
what "money path" means live here, in :data:`MONEY_PATH_PREDICATES`, built
out of `crucible.keys` functions rather than restated literals — the key
shapes stay `crucible.keys`' to own, and which of them carry money is this
module's.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from pydantic import ValidationError as PydanticValidationError

from crucible.documents import load_store_document
from crucible.keys import (  # noqa: F401 - manifest_key/manifest_prefix re-exported
    RUNS_ROOT,
    champion_key,
    holdout_unseal_prefix,
    is_manifest_key,
    manifest_key,
    manifest_prefix,
    predictions_key,
    strategy_holdout_key,
)
from crucible.models import RunManifestV2
from crucible.store import Store, sha256_hex

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
    (`alpha-engine-config-I10045` row 1), which enforces the status<->reason
    cross-field rule — also mirrored into the published schema's `allOf`
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


# ── The money-path hash chain (alpha-engine-config-I10414, plan §9.5) ───────

#: What "on the money path" means, as predicates over a store key rather than
#: as a list of prefix literals.
#:
#: Built out of `crucible.keys` functions on purpose. The key SHAPES are
#: `crucible.keys`' to own (`tests/test_no_inline_store_keys.py`,
#: `tests/test_key_construction_placement.py`); which of those shapes carry
#: money is a judgement about the system, and it belongs next to the writer
#: that acts on it. Restating `"champions/"` here would be the second
#: declaration that drifts the first time a prefix moves — which it has: the
#: arm-prediction prefix moved out of `predictions/` on 2026-09-11
#: (`alpha-engine-config-I9822`), and a literal here would have silently
#: started chaining every arm's predictions along with the serving feed.
#:
#: **The set as it stands, and what is deliberately not in it.** Plan §9.5
#: names orders, fills, reconciliation results and the NAV series. None of
#: those has a key in `crucible.keys` yet — the trader is a separate system
#: and phase 4 is where it lands — so chaining them is not something this
#: change can do, and pretending otherwise by inventing their keys here would
#: put a shape in the chain that no producer writes. What exists today and
#: decides where money goes is the champion contract the trader reads
#: (`champions/{slot}/current.json` plus `predictions/{trading_day}.json`,
#: this repo's `AGENTS.md`) and the sealed holdout that bounds what may be
#: graded into it. Those are chained here; each order/fill/NAV key joins by
#: adding one predicate below, and `tests/test_money_path_chain.py` pins the
#: membership so an addition is a deliberate edit rather than a silent widen.
#:
#: This is NOT a suppression collection (`AGENTS.md` rule 4): it is a
#: positive membership rule that only ever ADMITS keys to a check. A key
#: absent from it is not exempted from anything — it is outside the domain
#: this chain makes a claim about, and `crucible explain` never reports a
#: chain verdict over a walk that does not cross one of these keys.
MONEY_PATH_PREDICATES: tuple[Callable[[str], bool], ...] = (
    # The four champion pointers — the trader's whole read surface.
    lambda key: key in {champion_key(slot) for slot in ("u", "r", "m", "s")},
    # The champion's serving feed for one trading day. Matched by rebuilding
    # the key from the day the candidate claims, so `arm_predictions/` (a
    # different artifact answering a different question, I9822) can never
    # match by sharing a prefix.
    lambda key: (
        key == predictions_key(key.rsplit("/", 1)[-1].removesuffix(".json"))
        if key.endswith(".json")
        else False
    ),
    # The sealed holdout: the standing reservation every grading that reaches
    # the money path is measured outside of.
    lambda key: key == strategy_holdout_key(),
    # Every unseal audit record: WHO ruled, on WHICH session, unsealing WHAT.
    lambda key: key.startswith(holdout_unseal_prefix()),
)


class MoneyPathChainError(RuntimeError):
    """The money-path chain could not be extended or could not be verified.

    A `RuntimeError`, not a validation error: a chain that cannot be read is
    not a malformed document, it is a store that cannot answer the one
    question the chain exists to answer.
    """


def on_money_path(key: str) -> bool:
    """Whether ``key`` is a money-path artifact (:data:`MONEY_PATH_PREDICATES`)."""
    return any(predicate(key) for predicate in MONEY_PATH_PREDICATES)


def money_path_writes(manifest: dict[str, Any]) -> tuple[str, ...]:
    """The money-path keys ``manifest`` claims as OUTPUTS, sorted.

    Outputs only. A run that READ the champion pointer has not changed where
    money goes and does not belong in the chain; a chain that grew a record
    per reader would make its own length meaningless.

    Tolerant of a malformed document ON PURPOSE: this is the membership test
    :func:`money_path_manifests` applies BEFORE validating, so it runs over
    documents nothing has checked yet. An entry that is not an object with a
    string `key` cannot be a money-path output, and answering "no" for it is
    not a swallow — the document is still validated the moment it is admitted,
    and a malformed manifest that DOES claim a money-path key is refused
    there, loudly, with the key named.
    """
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list):
        return ()
    return tuple(
        sorted(
            {
                output["key"]
                for output in outputs
                if isinstance(output, dict)
                and isinstance(output.get("key"), str)
                and on_money_path(output["key"])
            }
        )
    )


def money_path_manifests(store: Store) -> list[tuple[str, dict[str, Any]]]:
    """Every money-path run manifest in ``store`` as ``(key, manifest)``,
    oldest first by ``(finished, run_id)``.

    Every ADMITTED document is validated, like everywhere else a manifest is
    read: a chain assembled out of documents nobody checked is a chain nobody
    should act on, and a malformed money-path manifest is a predecessor the
    chain cannot honestly be extended over.

    **Membership is decided before validation, and that ordering is
    load-bearing.** This function runs on the WRITE path
    (:func:`money_path_link_for`), so validating every manifest under `runs/`
    would let one malformed document anywhere in the store veto the write of
    an unrelated run's manifest — inverting `AGENTS.md` rule 1
    (manifest-or-it-didn't-happen) for a document the chain makes no claim
    about. It is the same failure `alpha-engine-config-I9900` recorded from
    the other side, where one stray file beside a run manifest took the whole
    board down. So a document is read, asked whether it claims a money-path
    OUTPUT, and only then held to the schema.

    ``run_id`` breaks a `finished` tie. Two manifests can share a
    whole-second timestamp, and a tie broken by dict order would give the same
    store two different chains on two reads — the verifier would then call one
    of them broken at random.
    """
    found: list[tuple[str, dict[str, Any]]] = []
    for candidate in store.list_keys(RUNS_ROOT):
        if not is_manifest_key(candidate):
            continue
        document = load_store_document(store, candidate)
        if not money_path_writes(document):
            continue
        validate(document)
        found.append((candidate, document))
    return sorted(found, key=lambda pair: (pair[1]["finished"], pair[1]["run_id"]))


def digest_of_stored(store: Store, key: str) -> str:
    """The sha256 of what ``store`` actually holds at ``key``.

    **The whole chain rests on this being a content digest and nothing else.**
    `nous-ergon-ops-I1145`: a backend version token — `Store.etag`, whose own
    docstring calls itself opaque — was written into a field named `sha256`.
    It agreed with every local test, because the local backend's token happens
    to be a content hash, and it was wrong on S3 the moment an object was
    uploaded multipart. So this function reads the bytes and hashes them, and
    `tests/test_money_path_chain.py` asserts that no chain code path calls
    `Store.etag` at all.
    """
    return sha256_hex(store.get_bytes(key))


def money_path_link_for(store: Store, manifest: dict[str, Any]) -> dict[str, Any] | None:
    """The `money_path_link` ``manifest`` should carry, or ``None``.

    ``None`` for a run that wrote nothing on the money path — which is the
    overwhelming majority, and which costs no store round-trip at all: the
    membership test is :func:`money_path_writes` over the manifest already in
    hand, so the listing below only ever happens on a run that is joining the
    chain.

    **Failures propagate.** A store whose head cannot be listed or read is a
    store the manifest write that follows would not survive either, and a
    linkless money-path manifest written "to be safe" is precisely the break
    :func:`crucible.explain.verify_money_path_chain` is built to refuse. There
    is no degraded link.
    """
    writes = money_path_writes(manifest)
    if not writes:
        return None
    chain = money_path_manifests(store)
    # A rerun of the same job on the same trading day writes the SAME key, so
    # a predecessor that is this very manifest's key would be chaining a
    # record to the bytes it is about to overwrite. Drop it: this run replaces
    # that record rather than following it.
    key = manifest_key(
        manifest["job"], manifest["trading_day"], discriminator=manifest.get("discriminator")
    )
    chain = [pair for pair in chain if pair[0] != key]
    if not chain:
        return {
            "index": 0,
            "prev_sha256": None,
            "prev_run_id": None,
            "money_path_writes": list(writes),
        }
    head_key, head = chain[-1]
    head_link = head.get("money_path_link")
    if head_link is None:
        raise MoneyPathChainError(
            f"the latest money-path manifest in this store ({head['run_id']} at "
            f"{head_key}) carries no `money_path_link`, so this run has nothing to "
            "chain to. Either it predates the chain — in which case the chain must be "
            "started deliberately against a store whose money-path history is known, "
            "not by silently electing this run the genesis — or the link was removed, "
            "which is the tamper this record exists to make visible."
        )
    return {
        "index": head_link["index"] + 1,
        "prev_sha256": digest_of_stored(store, head_key),
        "prev_run_id": head["run_id"],
        "money_path_writes": list(writes),
    }


def write_manifest(store: Store, key: str, manifest: dict[str, Any]) -> dict[str, Any]:
    """Attach the money-path chain link, validate, and write. The single writer.

    `crucible.runner._write_manifest` assembles the document and calls this;
    nothing else writes a run manifest. Validation happens AFTER the link is
    attached, so the bytes that land in the store are the bytes that were
    checked — a link attached after validation would be an unvalidated field
    on the one document the whole system's trust rests on.

    Returns the manifest as written, link included.
    """
    link = money_path_link_for(store, manifest)
    if link is not None:
        manifest = {**manifest, "money_path_link": link}
    validate(manifest)
    store.put_bytes(key, json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"))
    return manifest
