"""The sealed holdout, and the ruling-gated `--unseal` that is the only way in.

Normative source: `alpha-engine-config-I10502` (phase-3 deliverable, quoted
verbatim from `crucible/gate.py::PHASE3_DELIVERABLES`):

    sealed_holdout
      "sealed holdout `strategy/holdout.json` with `--unseal` requiring a
      ruling reference"

Plan §9.4 reserved the seam; `crucible/gate.py`'s runbook clause has read
"unseal: reserved, no command, no such job" since phase 2, honestly, because
there was no command and no such job. This module is the mechanism.

── WHY A HOLDOUT NEEDS A LOCK AND NOT A CONVENTION ───────────────────────

Plan §11 risk 1 is "v2 passes its gates because its tests were written to
pass". A holdout is the principal defence against it — data the harness
grades itself *outside* of, so a result on it is a result nobody tuned
towards. That defence is worth exactly as much as the difficulty of looking
at the data. A holdout anyone may read while promising not to is a test set,
and the promise leaves no artifact, so nobody can ever tell afterwards
whether it held.

So: the payload is not reachable by reading the document. :func:`read_sealed_holdout`
returns the seal's METADATA and nothing else — no code path in this package
hands a caller the payload except :func:`unseal`, which REFUSES without a
ruling reference and returns the record that must be filed with it.

── WHAT THE SEAL IS, AND WHAT IT IS NOT ──────────────────────────────────

The seal is a **content address**: ``seal.digest`` is ``sha256:`` + the
SHA-256 of the canonical JSON (sorted keys, no whitespace) of ``payload``
alone — the same canonicalisation `crucible.portfolio.params_digest` and
`crucible.attribution.params_digest` already use, so whitespace and key order
in the published file cannot change a holdout's identity.

It is **not encryption**, deliberately: encrypting the payload would put a
KMS key, a grant and a rotation schedule between every grading run and its
own data, and the thing being defended against is not an attacker — it is the
much likelier failure where a well-meaning process reads the holdout because
nothing stopped it. What the content address buys is that every unseal record
names the exact bytes released (`holdout_unseal.v1::holdout_digest`), and an
edit to the payload that does not re-seal is refused by the reader. What it
does not buy is protection against someone who rewrites the payload AND the
seal together; that is what the strategy tree's git history and the store's
own object retention are for, and this module claims nothing more.

── UNSEALING IS A RESERVED MATTER ────────────────────────────────────────

`principles.md` §3.2: unsealing waits for a human ruling no matter how
confident the system is. :func:`assert_ruling_reference` refuses anything that
is not a fleet ruling reference (``alpha-engine-config-I<N>``), non-zero exit,
named reason — it does not warn and proceed, and the refusal is the tested
behaviour, not the happy path.

**No parallel reserved-action list lives here.** `alpha-engine-config-I10416`
§11 row 9 already declares one config surface for this class —
`crucible.config.Settings.autonomy_reserved_events`, which
`crucible.autonomy.count_operator_actions` applies so a ruled unseal's
CloudTrail event is excluded from phase 2's human-mutating-call count. A
second list here would be a second thing to widen, in a repository that goes
public, out of sight of the operator who audits the first. Whether an unseal's
event name belongs in that set is an operator's `gh variable set`; this module
declares no event names at all, and `tests/test_holdout.py` asserts it.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from crucible.documents import read_listed_document, read_store_document
from crucible.keys import holdout_unseal_key, holdout_unseal_prefix, strategy_holdout_key
from crucible.store import Store

__all__ = [
    "HOLDOUT_JOB",
    "RULING_REFERENCE_PATTERN",
    "SEALED_HOLDOUT_SCHEMA_VERSION",
    "UNSEAL_RECORD_SCHEMA_VERSION",
    "HoldoutAbsentError",
    "HoldoutError",
    "HoldoutSealBrokenError",
    "HoldoutUnsealedError",
    "SealedHoldout",
    "UnsealRulingRequiredError",
    "UnsealRecord",
    "add_holdout_arguments",
    "assert_ruling_reference",
    "holdout_handler",
    "holdout_key_readers",
    "load_holdout_payload_file",
    "payload_digest",
    "read_sealed_holdout",
    "seal_document",
    "unseal",
    "unseal_record_bytes",
    "unseal_records",
]

SEALED_HOLDOUT_SCHEMA_VERSION = "sealed_holdout.v1"
UNSEAL_RECORD_SCHEMA_VERSION = "holdout_unseal.v1"

#: The CLI job name. One job, two modes: the default READ (seal state, no
#: payload, no write) and `--unseal` (the ruled read, which writes the record).
HOLDOUT_JOB = "holdout"

#: A fleet ruling reference, `<repo>-I<N>` with the repo fixed: the
#: alpha-engine tracker is `alpha-engine-config` and no other repo's issue
#: number means anything to it (see `~/.claude/CLAUDE.md`, "Reference
#: format"). Anchored and digit-led so `alpha-engine-config-I0` and
#: `see alpha-engine-config-I123 maybe` are both refused — a reference a
#: reader has to interpret is a reference that will be interpreted wrongly
#: once.
RULING_REFERENCE_PATTERN = r"^alpha-engine-config-I[1-9][0-9]*$"
_RULING_RE = re.compile(RULING_REFERENCE_PATTERN)

_SCHEMA_DIR = Path(__file__).parent / "schemas"


class HoldoutError(ValueError):
    """Anything that stops a holdout being read as what it claims to be."""


class HoldoutAbsentError(HoldoutError):
    """No holdout document is published. UNMEASURABLE, never "no holdout"."""


class HoldoutUnsealedError(HoldoutError):
    """A document exists at the holdout key and carries no valid seal."""


class HoldoutSealBrokenError(HoldoutError):
    """The payload does not hash to the digest the seal claims for it."""


class UnsealRulingRequiredError(HoldoutError):
    """An unseal was attempted with no ruling reference, or a malformed one."""


def _validator(schema_version: str) -> Draft202012Validator:
    schema = json.loads((_SCHEMA_DIR / f"{schema_version}.json").read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _validate(document: Mapping[str, Any], schema_version: str, *, source: str) -> dict[str, Any]:
    errors = sorted(_validator(schema_version).iter_errors(document), key=lambda e: list(e.path))
    if errors:
        paths = "; ".join(
            f"{'/'.join(str(part) for part in e.path) or '<root>'}: {e.message}" for e in errors
        )
        raise HoldoutUnsealedError(f"{source} does not satisfy {schema_version}: {paths}")
    return dict(document)


def assert_ruling_reference(ruling: str | None, *, action: str) -> str:
    """``ruling`` if it is a fleet ruling reference; raise otherwise.

    The refusal, not a warning. `principles.md` §3.2 reserves unsealing to a
    human ruling, and a reserved decision that proceeds with a complaint in
    the log has not been reserved — it has been announced.
    """
    if not ruling:
        raise UnsealRulingRequiredError(
            f"{action} requires --ruling <alpha-engine-config-I<N>>, and none was given. "
            "Unsealing the holdout is a RESERVED matter (principles.md §3.2): it waits "
            "for a human ruling, and the ruling's reference is what the audit record at "
            f"{holdout_unseal_prefix()} is built around. There is no proceed-anyway form "
            "of this command."
        )
    if not _RULING_RE.match(ruling):
        raise UnsealRulingRequiredError(
            f"{ruling!r} is not a ruling reference. Expected {RULING_REFERENCE_PATTERN} — "
            "the alpha-engine tracker is `alpha-engine-config`, so a bare number, another "
            "repository's issue or a sentence naming one does not identify a ruling anyone "
            "can look up later."
        )
    return ruling


def payload_digest(payload: Mapping[str, Any]) -> str:
    """The content address of ``payload``: ``sha256:`` + SHA-256 of its
    canonical JSON (sorted keys, no whitespace).

    Canonical, so the published file's formatting is not part of the
    holdout's identity — a re-indent is not a re-seal. Mirrors
    `crucible.portfolio.params_digest` exactly rather than inventing a second
    canonicalisation for the same job.
    """
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SealedHoldout:
    """What a reader is allowed to know about the holdout: the seal, not the data.

    **There is deliberately no ``payload`` attribute.** This is the whole
    mechanism: a grading path that resolves the holdout through
    :func:`read_sealed_holdout` holds an object that cannot leak the
    reservation, however it is logged, rendered or serialised. Releasing the
    payload goes through :func:`unseal` and produces an audit record.
    """

    digest: str
    sealed_trading_day: str
    sealed_by: str
    purpose: str
    ruling: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "digest": self.digest,
            "sealed_trading_day": self.sealed_trading_day,
            "sealed_by": self.sealed_by,
            "purpose": self.purpose,
            "ruling": self.ruling,
        }

    def render(self) -> str:
        ruled = f", sealed under {self.ruling}" if self.ruling else ""
        return (
            f"holdout SEALED {self.digest} on {self.sealed_trading_day} "
            f"by {self.sealed_by}{ruled} — {self.purpose}"
        )


@dataclass(frozen=True)
class UnsealRecord:
    """One filed unseal, as read back from the store."""

    #: The store key this record was read from. Named `store_key` rather than
    #: `key` so a reader of `record.store_key` cannot be misread as the
    #: holdout's own identity, which is `holdout_digest`.
    store_key: str
    trading_day: str
    ruling: str
    operator: str
    reason: str
    holdout_digest: str
    unsealed_at_utc: str


def seal_document(
    payload: Mapping[str, Any],
    *,
    sealed_trading_day: str,
    sealed_by: str,
    purpose: str,
    ruling: str | None = None,
) -> dict[str, Any]:
    """Build the `sealed_holdout.v1` document for ``payload``.

    The authoring half, run once when a holdout is reserved; the result is
    committed to the private strategy tree as `strategy/holdout.json` and
    published to :func:`crucible.keys.strategy_holdout_key`. This function
    writes nothing: the holdout's contents are strategy edge and this
    repository neither stores them nor invents them (the same boundary
    `crucible.attribution.load_attribution_params` holds for the factor
    spec).

    ``ruling`` is optional HERE and mandatory on the way out: sealing
    reserves data and needs no ruling; unsealing releases it and always does.
    """
    if not payload:
        raise HoldoutError(
            "an empty holdout payload reserves nothing, and a sealed document over it "
            "would read as a holdout to every clause while withholding no data at all."
        )
    seal: dict[str, Any] = {
        "algorithm": "sha256",
        "digest": payload_digest(payload),
        "sealed_trading_day": sealed_trading_day,
        "sealed_by": sealed_by,
        "purpose": purpose,
    }
    if ruling is not None:
        seal["ruling"] = assert_ruling_reference(ruling, action="sealing a holdout")
    document = {
        "schema_version": SEALED_HOLDOUT_SCHEMA_VERSION,
        "seal": seal,
        "payload": dict(payload),
    }
    return _validate(
        document, SEALED_HOLDOUT_SCHEMA_VERSION, source="the holdout document just built"
    )


def _read_document(store: Store) -> dict[str, Any]:
    """The raw sealed document, verified against its own seal.

    Private on purpose: it is the one function in this package that holds the
    payload and a store handle at the same time, and everything public above
    it either drops the payload (:func:`read_sealed_holdout`) or demands a
    ruling for it (:func:`unseal`).
    """
    key = strategy_holdout_key()
    read = read_store_document(store, key)
    if read.problem is not None:
        raise HoldoutUnsealedError(read.problem)
    if read.absent:
        raise HoldoutAbsentError(
            f"no holdout document at {key}. `strategy/holdout.json` is authored in the "
            "private strategy tree and published into the store; until it is, the "
            "harness holds no reservation — which is UNMEASURABLE, never a holdout of "
            "zero sessions."
        )
    document = _validate(read.document or {}, SEALED_HOLDOUT_SCHEMA_VERSION, source=key)
    seal = document["seal"]
    recomputed = payload_digest(document["payload"])
    if recomputed != seal["digest"]:
        raise HoldoutSealBrokenError(
            f"{key}: the payload hashes to {recomputed} and the seal claims "
            f"{seal['digest']}. The document was edited after it was sealed — a holdout "
            "that does not match its own seal is not a holdout, and reading it as one "
            "would grade against data of unknown provenance."
        )
    return document


def read_sealed_holdout(store: Store) -> SealedHoldout:
    """The seal state, with the payload dropped. The ONLY reader a grading path calls.

    Raises :class:`HoldoutAbsentError` when nothing is published,
    :class:`HoldoutUnsealedError` when the document is not a
    `sealed_holdout.v1`, and :class:`HoldoutSealBrokenError` when the payload
    does not match its own seal. None of the three returns a value: a caller
    that cannot tell "absent" from "broken" from "fine" is the shape of a
    silent no-op (AGENTS.md rule 5).
    """
    document = _read_document(store)
    seal = document["seal"]
    return SealedHoldout(
        digest=seal["digest"],
        sealed_trading_day=seal["sealed_trading_day"],
        sealed_by=seal["sealed_by"],
        purpose=seal["purpose"],
        ruling=seal.get("ruling"),
    )


def unseal(
    store: Store,
    *,
    ruling: str | None,
    trading_day: str,
    operator: str,
    reason: str,
    now_utc: dt.datetime,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Release the holdout payload under ``ruling``, with the record to file.

    Returns ``(payload, record)``. The record is a validated
    `holdout_unseal.v1` document and its caller writes it at
    :func:`crucible.keys.holdout_unseal_key` — through
    `RunContext.record_output`, so the release also lands in the manifest's
    lineage rather than only in a prefix somebody has to know to look at.

    The ruling is checked FIRST, before the store is touched: a refused
    unseal must not depend on whether the document happened to be readable,
    or the refusal would be a different answer on a bad day.

    ``now_utc`` is passed, never read from a clock here — the same rule
    `crucible.attribution.attribution_metric_record` follows, so a test
    asserts the record's content rather than tolerating it.
    """
    assert_ruling_reference(ruling, action="`crucible holdout --unseal`")
    if not operator:
        raise HoldoutError(
            "an unseal needs an operator. The ruling says the read was allowed; the "
            "operator says who performed it, and a record missing that names a decision "
            "with nobody attached to it."
        )
    if not reason:
        raise HoldoutError(
            "an unseal needs a --reason. The ruling records that it was allowed; the "
            "reason records what it was for, and an unexplained release of the holdout "
            "is the one read nobody can reconstruct later (principle 1)."
        )
    document = _read_document(store)
    record = {
        "schema_version": UNSEAL_RECORD_SCHEMA_VERSION,
        "trading_day": trading_day,
        "ruling": ruling,
        "operator": operator,
        "reason": reason,
        "holdout_digest": document["seal"]["digest"],
        # `...Z`, not `+00:00`: this string is also the metric row's
        # `last_updated_utc`, and `run_manifest.v2` pins that spelling — a
        # `+00:00` offset fails validation at write time and routes the whole
        # manifest through `_minimal_failed_manifest`, which drops every
        # job-contributed field. Measured on the first run of this job.
        "unsealed_at_utc": now_utc.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    _validate(record, UNSEAL_RECORD_SCHEMA_VERSION, source="the unseal record just built")
    return dict(document["payload"]), record


def unseal_record_bytes(record: Mapping[str, Any]) -> bytes:
    """``record`` as the bytes filed at :func:`crucible.keys.holdout_unseal_key`."""
    return json.dumps(record, indent=2, sort_keys=True).encode("utf-8")


def unseal_records(store: Store) -> tuple[list[UnsealRecord], list[str]]:
    """Every filed unseal, and every record that could not be read as one.

    Returns ``(records, problems)``. A malformed record is a PROBLEM, never a
    dropped row: the prefix exists so an unseal cannot happen unobserved, and
    a reader that silently skips what it cannot parse reintroduces exactly
    the blindness the prefix removes.
    """
    records: list[UnsealRecord] = []
    problems: list[str] = []
    for listed in sorted(store.list_keys(holdout_unseal_prefix())):
        read = read_listed_document(store, listed)
        if read.problem is not None:
            problems.append(read.problem)
            continue
        try:
            document = _validate(read.document or {}, UNSEAL_RECORD_SCHEMA_VERSION, source=listed)
        except HoldoutUnsealedError as exc:
            problems.append(str(exc))
            continue
        records.append(
            UnsealRecord(
                store_key=listed,
                trading_day=document["trading_day"],
                ruling=document["ruling"],
                operator=document["operator"],
                reason=document["reason"],
                holdout_digest=document["holdout_digest"],
                unsealed_at_utc=document["unsealed_at_utc"],
            )
        )
    return records, problems


def load_holdout_payload_file(path: Path | str) -> dict[str, Any]:
    """A plaintext candidate holdout payload, for :func:`seal_document`.

    Used by `crucible holdout --seal <path>`, the authoring step, which reads
    a candidate from disk and prints the sealed document for committing to
    the private strategy tree. An absent file RAISES: there is no default
    holdout this repository can supply, for the same reason there is no
    default factor spec (`crucible.attribution.load_attribution_params`).
    """
    resolved = Path(path)
    if not resolved.exists():
        raise HoldoutError(
            f"no candidate holdout payload at {resolved}. Which sessions are reserved is "
            "strategy edge and lives in the private strategy tree; there is no default "
            "this repository can invent."
        )
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise HoldoutError(
            f"{resolved}: expected a JSON object as the holdout payload, got "
            f"{type(payload).__name__}. A holdout is a named reservation, not a bare list "
            "whose meaning depends on who is reading it."
        )
    return payload


def holdout_key_readers() -> tuple[str, ...]:
    """The modules allowed to name the holdout key, exhaustively.

    Read by `tests/test_holdout.py`, which walks the tree and fails when any
    other module calls `crucible.keys.strategy_holdout_key` — the sealed
    reader is only the only way in while nothing else opens a second door
    (the deliverable's own words: read "only through the sealed reader").

    `crucible/gate.py` is the second entry and it is not an exception to the
    rule: `_clause_sealed_holdout` names the key in its REQUIREMENT string and
    its evidence list, so an operator reading the gate knows which document
    was graded, and reads the document itself only through
    :func:`read_sealed_holdout` like everyone else. A member of this tuple
    that ever calls `store.get_bytes` on the holdout key is a defect; naming
    the key is not.
    """
    return ("crucible/holdout.py", "crucible/gate.py")


def add_holdout_arguments(parser: Any) -> None:
    """The `crucible holdout` flag surface.

    Lives here rather than inline in `crucible.cli.build_parser` for the
    reason `crucible.track_a.add_track_a_arguments` already exists: the flags
    and the handler that reads them are one contract, and splitting them
    across two files is how a flag ends up accepted and unread.
    """
    parser.add_argument(
        "--unseal",
        action="store_true",
        help=(
            "Release the holdout payload. REFUSED without --ruling and --reason: "
            "unsealing is a reserved matter (principles.md §3.2), and the refusal has "
            "no proceed-anyway form. Writes the audit record naming the ruling."
        ),
    )
    parser.add_argument(
        "--ruling",
        default=None,
        metavar="alpha-engine-config-I<N>",
        help=(
            "The human ruling authorising the unseal. Recorded in the audit record; "
            "anything that is not a fleet ruling reference is refused."
        ),
    )
    parser.add_argument(
        "--reason",
        default=None,
        help=(
            "What the unseal is for, in your own words. Mandatory with --unseal: the "
            "ruling records that it was allowed, the reason records what it was for."
        ),
    )
    parser.add_argument(
        "--operator",
        default=None,
        help="Who is unsealing. Defaults to $USER; recorded in the audit record.",
    )
    parser.add_argument(
        "--seal",
        default=None,
        metavar="PAYLOAD.json",
        help=(
            "Authoring form: read a candidate holdout payload from this file and print "
            "the sealed `sealed_holdout.v1` document for committing to the private "
            "strategy tree. Reads and writes no store."
        ),
    )
    parser.add_argument(
        "--sealed-by",
        default=None,
        help="Who is sealing, with --seal. Defaults to $USER.",
    )
    parser.add_argument(
        "--purpose",
        default=None,
        help="What the holdout is reserved for, with --seal. Mandatory there.",
    )


def holdout_handler(args: Any) -> int:
    """`crucible holdout [--seal PAYLOAD.json | --unseal --ruling REF --reason WHY]`.

    **Three modes, and only one of them writes.**

    * default — READ the seal state and print it. The payload is not reachable
      on this path at all; what is printed is `SealedHoldout.render()` plus the
      filed unseal records. No store write and **no manifest**, the same shape
      and the same reason as `crucible gate` without `--publish`
      (`alpha-engine-config-I10576`): a read publishes no shared artifact, so it
      has no lineage to record, and a manifest over nothing is a write like any
      other — measured there to have made the very gate it was reading
      unmeetable.
    * ``--seal`` — AUTHOR a sealed document from a plaintext candidate and print
      it. Touches no store either: the holdout's contents are strategy edge and
      this repository is the mechanism, not the custodian.
    * ``--unseal`` — the ruled read. This one runs through `run_job`, writes the
      `holdout_unseal.v1` record through `RunContext.record_output` (so the
      release is in the manifest's lineage, not only in a prefix somebody has
      to know to look at) and exits 0.

    **A refused unseal is a USAGE error, not a failed run.** No ruling, a
    malformed one, or a missing reason means the dispatch was malformed — the
    holdout is exactly as sealed afterwards as before, nothing in the system is
    wrong, and writing a `status: failed` manifest for it would fire the failure
    page condition at an operator typo. `crucible.cli.UsageError` is the class
    that already means this, so the refusal exits 2 with its reason on stderr,
    writes nothing, and is not a page.
    """
    import os  # noqa: PLC0415 - a handler-local import, not a package-level dependency

    from crucible.cli import UsageError  # noqa: PLC0415 - avoid an import cycle at module load
    from crucible.config import settings  # noqa: PLC0415 - avoid an import cycle at module load
    from crucible.runner import RunContext, run_job  # noqa: PLC0415 - same cycle

    unsealing = bool(getattr(args, "unseal", False))
    sealing = getattr(args, "seal", None)
    if unsealing and sealing:
        raise UsageError(
            "`--seal` authors a sealed document and `--unseal` releases one; asking for "
            "both in one invocation names no single intent, and guessing which was meant "
            "is the guess that releases a holdout nobody asked to release."
        )

    if sealing:
        purpose = getattr(args, "purpose", None)
        if not purpose:
            raise UsageError(
                "`--seal` requires --purpose. A holdout whose purpose nobody wrote down "
                "is one somebody re-purposes (plan §11 risk 1)."
            )
        payload = load_holdout_payload_file(sealing)
        document = seal_document(
            payload,
            sealed_trading_day=args.trading_day.isoformat(),
            sealed_by=getattr(args, "sealed_by", None) or os.environ.get("USER", "unknown"),
            purpose=purpose,
            ruling=getattr(args, "ruling", None),
        )
        print(json.dumps(document, indent=2, sort_keys=True))
        return 0

    dry_run = bool(getattr(args, "dry_run", False))
    cfg = settings(store_uri=getattr(args, "store", None), dry_run=dry_run)
    store = cfg.store()

    if not unsealing:
        try:
            sealed = read_sealed_holdout(store)
        except HoldoutError as exc:
            # Not raised: this is the READ form, which writes no manifest, so a
            # traceback here would be the whole record of a reading that is a
            # legitimate answer ("nothing is published yet"). The reason is
            # printed and the exit code carries it. The phase-3 `sealed_holdout`
            # clause is what grades the same fact durably.
            print(str(exc))
            return 1
        print(sealed.render())
        records, problems = unseal_records(store)
        for record in records:
            print(f"  unsealed {record.trading_day} under {record.ruling} by {record.operator}")
        for problem in problems:
            print(f"  UNREADABLE unseal record: {problem}")
        if not records:
            print("  no unseal has ever been filed")
        return 1 if problems else 0

    try:
        assert_ruling_reference(getattr(args, "ruling", None), action="`crucible holdout --unseal`")
    except UnsealRulingRequiredError as exc:
        raise UsageError(str(exc)) from exc
    reason = getattr(args, "reason", None)
    if not reason:
        raise UsageError(
            "`crucible holdout --unseal` requires --reason. The ruling records that the "
            "read was allowed; the reason records what it was for, and an unexplained "
            "release of the holdout is the one read nobody can reconstruct later."
        )

    operator = getattr(args, "operator", None) or os.environ.get("USER", "unknown")
    released: dict[str, Any] = {}

    def body(ctx: RunContext) -> None:
        payload, record = unseal(
            store,
            ruling=args.ruling,
            trading_day=ctx.trading_day.isoformat(),
            operator=operator,
            reason=reason,
            now_utc=ctx.started,
        )
        released.update(payload)
        ctx.record_metric(
            {
                "name": "holdout_unseal",
                "module": "crucible.holdout",
                "metric_type": "governance",
                "n_floor": 1,
                "status": "OK",
                "status_reason": (
                    f"holdout {record['holdout_digest']} released under {record['ruling']} "
                    f"by {operator}: {reason}"
                ),
                "source_path": "crucible/holdout.py",
                "last_updated_utc": record["unsealed_at_utc"],
                "value": 1.0,
                "unit": "count",
            }
        )
        if dry_run:
            print(json.dumps(record, indent=2, sort_keys=True))
            return
        ctx.record_output(
            holdout_unseal_key(ctx.trading_day.isoformat(), record["ruling"]),
            unseal_record_bytes(record),
            schema_version=UNSEAL_RECORD_SCHEMA_VERSION,
        )

    result = run_job(
        HOLDOUT_JOB,
        body,
        store=store,
        trading_day=args.trading_day,
        dry_run=dry_run,
        run_mode=getattr(args, "run_mode", None),
    )
    _ = result
    print(f"holdout unsealed under {args.ruling}: {len(released)} top-level key(s) released")
    return 0
