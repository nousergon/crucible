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

import datetime as dt
import json
import time
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
    execution_shortfall_key,
    holdout_unseal_prefix,
    is_manifest_key,
    manifest_key,
    manifest_prefix,
    money_path_claim_key,
    predictions_key,
    strategy_holdout_key,
    trader_reconciliation_key,
)
from crucible.models import RunManifestV2
from crucible.store import ETAG_ABSENT, PointerConflictError, Store, sha256_hex

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
#: names orders, fills, reconciliation results and the NAV series. What
#: decides where money goes on the harness side is the champion contract the
#: trader reads (`champions/{slot}/current.json` plus
#: `predictions/{trading_day}.json`, this repo's `AGENTS.md`) and the sealed
#: holdout that bounds what may be graded into it. Phase 4 adds the trader's
#: side as its keys land in `crucible.keys` (`alpha-engine-config-I10651`):
#:
#: * the daily broker RECONCILIATION result — plan §9.5 by name;
#: * the per-session EXECUTION SHORTFALL document — the one artifact carrying
#:   the trader's real fills (decision price, fill price, filled quantity per
#:   order), so it is the fills record §9.5 names until a separate order/fill
#:   log exists.
#:
#: **Deliberately NOT chained:** the shadow books (`trader/shadow_books/`) are
#: SIMULATED paper books per challenger — evidence beside a promotion, never
#: an input to one (`alpha-engine-config-I10653` deliverable 5) — and a chain
#: that claims tamper-evidence over money must not grow a record for a book no
#: money follows. The broker STATEMENT is the reconciliation's input and is
#: content-hashed into the reconciliation run's manifest already; chaining the
#: result is what §9.5 asks. The NAV series has no key yet and joins by one
#: predicate when it does. `tests/test_money_path_chain.py` pins the
#: membership, both directions, so an addition is a deliberate edit rather
#: than a silent widen.
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
    # The trader's daily broker reconciliation and its per-session fills
    # (execution shortfall). Matched by rebuilding the key from the day the
    # candidate claims, like the serving feed, so a sibling artifact under
    # `trader/` (the shadow books, the evidence document) never matches by
    # sharing a prefix. `_day_key_matches` refuses a non-date day rather than
    # letting the helper's ISO-date check raise out of a membership test.
    lambda key: _day_key_matches(key, trader_reconciliation_key),
    lambda key: _day_key_matches(key, execution_shortfall_key),
)


def _day_key_matches(key: str, helper: Callable[[str], str]) -> bool:
    """Whether ``key`` is exactly ``helper(day)`` for the day its basename names."""
    if not key.endswith(".json"):
        return False
    day = key.rsplit("/", 1)[-1].removesuffix(".json")
    try:
        return key == helper(day)
    except ValueError:
        return False


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

    **A read, not a reservation.** Two writers calling this at the same moment
    get the same index; :func:`write_manifest` is what makes only one of them
    able to USE it (`alpha-engine-config-I12020`).
    """
    writes = money_path_writes(manifest)
    if not writes:
        return None
    return _next_link(store, manifest, writes)[0]


def _next_link(
    store: Store, manifest: dict[str, Any], writes: tuple[str, ...]
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """The link ``manifest`` would carry against the chain as it reads NOW,
    and the head manifest it chains to (``None`` for the genesis)."""
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
        genesis = {
            "index": 0,
            "prev_sha256": None,
            "prev_run_id": None,
            "money_path_writes": list(writes),
        }
        return genesis, None
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
    link = {
        "index": head_link["index"] + 1,
        "prev_sha256": digest_of_stored(store, head_key),
        "prev_run_id": head["run_id"],
        "money_path_writes": list(writes),
    }
    return link, head


# ── Index claims: one writer per chain index (alpha-engine-config-I12020) ───
#
# Measured 2026-10-07: two `promote` runs (slots m and s) dispatched ~3 s
# apart both read head index 10 and both wrote index 11, forking the chain;
# `verify_money_path_chain` correctly failed and the trader refused its
# session. Reading the head and then writing `head.index + 1` is a
# read-modify-write with no concurrency control, and the manifests themselves
# live at DIFFERENT keys, so no conditional write on the manifest can see the
# collision. The collision is on the INDEX, so the index is what gets a key:
# before a money-path manifest is written, its writer creates
# `money_path_claim_key(index)` with create-if-absent
# (`Store.compare_and_swap(..., ETAG_ABSENT, ...)`: `IfNoneMatch='*'` on S3,
# an `os.link` exclusive create locally). Exactly one writer can create it.
# The loser re-reads the head — which by then is, or is about to be, the
# winner — and claims the next index, a bounded number of times.

#: How many times :func:`write_manifest` reads the head and tries to claim the
#: next index before it gives up. A winner's claim is followed by its manifest
#: write within the same call — validation is done BEFORE the claim — so the
#: window a loser waits through is milliseconds, and six attempts over the
#: backoff below (~7.75 s of sleeping) absorbs a burst of several concurrent
#: money-path writers, not just two.
MONEY_PATH_CLAIM_ATTEMPTS = 6

#: The first retry's sleep; each later one doubles it.
MONEY_PATH_CLAIM_BACKOFF_S = 0.25

#: The longest :func:`write_manifest` will wait for the wall clock to pass
#: the head's `finished` second (see :func:`_ordered_after_head`). A head
#: stamped further in the future than this is not a race to wait out, it is a
#: clock or a document to look at, and the write refuses with both named.
MONEY_PATH_ORDER_MAX_WAIT_S = 5.0

#: The claim object's declared shape. Small and self-describing: an operator
#: looking at `runs/_money_path/claims/` can see which run took which index, and the
#: exhaustion message below names the holder from it.
MONEY_PATH_CLAIM_SCHEMA_VERSION = "money_path_claim.v1"

_FINISHED_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _parse_finished(value: str) -> dt.datetime:
    try:
        return dt.datetime.strptime(value, _FINISHED_FORMAT).replace(tzinfo=dt.UTC)
    except ValueError as exc:
        raise MoneyPathChainError(
            f"`finished`={value!r} is not a {_FINISHED_FORMAT} UTC instant, so this "
            "write cannot tell whether its record would sort after the chain head."
        ) from exc


def _ordered_after_head(
    candidate: dict[str, Any],
    head: dict[str, Any],
    *,
    sleep: Callable[[float], None],
    clock: Callable[[], dt.datetime],
) -> dict[str, Any]:
    """``candidate``, guaranteed to sort strictly after ``head`` by
    `(finished, run_id)`.

    The verifier — whose rules this change does not touch — orders the chain
    by `(finished, run_id)` and requires that order to equal index order. A
    writer that LOST an index race was not necessarily the later finisher:
    two runs finishing in the same second sort by `run_id`, which is minted at
    START, so the run that started first and finished last can lose the claim
    to its sibling and still sort before it. Writing index N+1 under a
    `(finished, run_id)` that sorts before index N would fork the chain all
    over again, one step later.

    So when the record would not sort after its predecessor, `finished` is
    restamped FORWARD to the wall clock — waiting, at most
    :data:`MONEY_PATH_ORDER_MAX_WAIT_S`, for the clock to pass the head's
    second when it has not yet. That is not a fabricated time: `finished` is
    when the run's manifest lands, and a run that had to wait for a sibling's
    record to land before its own could did not finish any earlier. It never
    moves backward, so `started <= finished` holds. A record that already
    sorts after the head — every sequential write — is returned unchanged.
    """
    if (candidate["finished"], candidate["run_id"]) > (head["finished"], head["run_id"]):
        return candidate
    earliest = _parse_finished(head["finished"])
    if candidate["run_id"] <= head["run_id"]:
        # Same second would sort by run_id, and ours does not sort after.
        earliest += dt.timedelta(seconds=1)
    now = clock()
    if now < earliest:
        wait = (earliest - now).total_seconds()
        if wait > MONEY_PATH_ORDER_MAX_WAIT_S:
            raise MoneyPathChainError(
                f"the money-path chain head ({head['run_id']}) finished at "
                f"{head['finished']}, {wait:.1f}s after this process's clock "
                f"({now.strftime(_FINISHED_FORMAT)}). This run's record must sort after "
                "its predecessor, and waiting longer than "
                f"{MONEY_PATH_ORDER_MAX_WAIT_S}s for a clock to catch up is not a race to "
                "wait out — check this host's clock and that record's `finished`."
            )
        sleep(wait)
        now = max(clock(), earliest)
    return {**candidate, "finished": now.strftime(_FINISHED_FORMAT)}


def _claim_index(store: Store, key: str, candidate: dict[str, Any]) -> dict[str, Any] | None:
    """Claim ``candidate``'s chain index for the manifest at ``key``.

    Returns ``None`` when the index is this writer's to write, or the claim
    document of whoever holds it.

    **The rerun case.** A rerun of the same job on the same trading day
    rewrites the same manifest key and replaces its own record at its own
    index (:func:`_next_link` drops that key from the chain before reading the
    head). Its claim already exists — it is the one the first run created —
    and names this same manifest key, so it is honoured as this writer's own.
    That is also how an orphaned claim heals: a run that claimed an index and
    died before its manifest landed is re-run, which writes the same key and
    takes the same index. Honoured only while no SUCCESSOR has claimed the
    next index: rewriting a record somebody is already chaining past would
    change the bytes their `prev_sha256` was computed over.
    """
    link = candidate["money_path_link"]
    index = link["index"]
    claim_key = money_path_claim_key(index)
    claim = {
        "schema_version": MONEY_PATH_CLAIM_SCHEMA_VERSION,
        "index": index,
        "run_id": candidate["run_id"],
        "manifest_key": key,
        "prev_run_id": link["prev_run_id"],
        "prev_sha256": link["prev_sha256"],
    }
    try:
        store.compare_and_swap(
            claim_key,
            ETAG_ABSENT,
            json.dumps(claim, indent=2, sort_keys=True).encode("utf-8"),
        )
    except PointerConflictError:
        holder = load_store_document(store, claim_key)
        if holder.get("manifest_key") == key and not store.exists(money_path_claim_key(index + 1)):
            return None
        return holder
    return None


def write_manifest(
    store: Store,
    key: str,
    manifest: dict[str, Any],
    *,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], dt.datetime] = _utc_now,
) -> dict[str, Any]:
    """Attach the money-path chain link, validate, and write. The single writer.

    `crucible.runner._write_manifest` assembles the document and calls this;
    nothing else writes a run manifest. Validation happens AFTER the link is
    attached, so the bytes that land in the store are the bytes that were
    checked — a link attached after validation would be an unvalidated field
    on the one document the whole system's trust rests on.

    **A money-path manifest claims its index before it is written**
    (`alpha-engine-config-I12020`; see the section comment above
    :data:`MONEY_PATH_CLAIM_ATTEMPTS`). The document is linked, ordered after
    its predecessor (:func:`_ordered_after_head`) and VALIDATED first, so a
    manifest that would be refused never takes an index it cannot fill. A lost
    claim re-reads the head and tries the next index, up to
    :data:`MONEY_PATH_CLAIM_ATTEMPTS` times, then raises
    :class:`MoneyPathChainError` — never writes a record whose index somebody
    else owns. ``sleep`` and ``clock`` exist for the tests that drive that
    interleaving deterministically.

    Returns the manifest as written, link included.
    """
    writes = money_path_writes(manifest)
    if writes:
        manifest = _claim_chain_link(store, key, manifest, writes, sleep=sleep, clock=clock)
    else:
        validate(manifest)
    store.put_bytes(key, json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"))
    return manifest


def _claim_chain_link(
    store: Store,
    key: str,
    manifest: dict[str, Any],
    writes: tuple[str, ...],
    *,
    sleep: Callable[[float], None],
    clock: Callable[[], dt.datetime],
) -> dict[str, Any]:
    """``manifest`` linked at an index this writer has claimed, validated."""
    lost: list[dict[str, Any]] = []
    for attempt in range(MONEY_PATH_CLAIM_ATTEMPTS):
        if attempt:
            sleep(MONEY_PATH_CLAIM_BACKOFF_S * 2 ** (attempt - 1))
        link, head = _next_link(store, manifest, writes)
        candidate = {**manifest, "money_path_link": link}
        if head is not None:
            candidate = _ordered_after_head(candidate, head, sleep=sleep, clock=clock)
        validate(candidate)
        holder = _claim_index(store, key, candidate)
        if holder is None:
            return candidate
        lost.append(holder)
    last = lost[-1]
    holder_key = last.get("manifest_key")
    # Failure path only, so the second listing costs nothing a success pays.
    landed = any(doc["run_id"] == last.get("run_id") for _, doc in money_path_manifests(store))
    raise MoneyPathChainError(
        f"run {manifest['run_id']} ({key}) could not claim a money-path chain index in "
        f"{MONEY_PATH_CLAIM_ATTEMPTS} attempts; every index it read as next was already "
        f"claimed. Last lost: index {last.get('index')!r} at "
        f"{money_path_claim_key(int(last.get('index', 0)))}, held by run "
        f"{last.get('run_id')!r} for {holder_key!r}, whose manifest "
        f"{'is in the chain' if landed else 'is NOT in the chain'}. "
        + (
            "Concurrent money-path writers outran the retry budget; re-run this job."
            if landed
            else "A claim whose manifest never landed is a writer that died between "
            "claiming and writing: re-running THAT job for that trading day rewrites "
            "the same key, reuses its claim and unblocks the chain."
        )
        + " This manifest was NOT written: an index another run owns cannot be "
        "written over, and a linkless money-path manifest is the break the verifier "
        "refuses."
    )
