"""A slot temporarily benched at zero weight by ruling, a cutover mechanic.

Normative source: Brian's ruling of 2026-10-18 14:44Z on the Crucible v2
phase-4 cutover (`alpha-engine-config-I12024`, option (a)): *"keep the date and
cut over to v2 with the M slot benched at zero weight, recording 'no lift'"*.
The 10-17 confirmation (alpha-engine-config PR 12406) found no M arm that
passes its serving preconditions; every near-miss carried a negative
out-of-sample IC.

── WHAT A BENCH IS ───────────────────────────────────────────────────────

One document, `slots/{slot}/bench.json` (`slot_bench.v1`,
:func:`crucible.keys.slot_bench_key`), filed by ``crucible bench`` under its
own run manifest. It is a **ruled record, never a cleared one** — the pattern
nousergon-data PR 2141 and nous-ergon-ops PR 1670 established for a measured
failure a human has ruled on: the reading it stands beside is still printed,
and the record names who ruled, when, where the words are (`ruling_ref`) and
what they were ruled on (`evidence`). Nothing here turns a failed reading
green; it records why the system serves what it serves.

── WHAT IT DOES, AT ONE CHOKEPOINT EACH ──────────────────────────────────

* **The alpha.** `crucible.slots.strategy.resolve_session` serves a ZERO alpha
  vector with ``alpha_source = slots/m/bench.json`` for every session on or
  after `effective_trading_day`. That function is the one place the S slot's
  construction inputs are resolved, so the S grade, the shadow books and the
  traded book stay one object. The zero is EXPLICIT: the M pointer and its
  daily feed stay in force on the benched champion, so the priced universe is
  still that arm's and the trader's hold-book safeguard, which reads the feed,
  still sees a real cross-section — an all-zero prediction arm would instead
  trip its degeneracy halt and serve no day at all.
* **The pointer.** `promote[m]` still reads the graded cycle and still decides,
  but writes nothing while benched, and files `pointer_moved = 0` with a
  `status_reason` naming the bench (`crucible.cli._promote`). The pointer the
  trader's clock is counted on stays frozen, and the phase-3
  `m_promotion_or_verdict_backed_non_promotion` clause reads a reasoned
  non-promotion every week through its existing reader.
* **The board** renders a grey `DECLARED_OFF` row, `slot:m:bench`
  (`crucible.board`), and the phase-4 `trader_one_week_on_v2_champion` detail
  names the bench (`crucible.gate`).

── WHO MAY WRITE IT ──────────────────────────────────────────────────────

The operator CLI identity, the way `faults/` and `holdout/` are written. The
box runtime that is benched only READS it: an identity that could write its own
bench could also clear it.

**v1 benches the M slot only.** M's predictions are S's alpha vector, so a
zero weight there has exactly one consumer and one meaning. No other slot has a
weight anything reads; recording one would be a field nothing honours.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import os
import re
from collections.abc import Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from crucible.calendar import assert_trading_day
from crucible.documents import read_store_document
from crucible.holdout import RULING_REFERENCE_PATTERN, payload_digest
from crucible.keys import champion_key, slot_bench_key
from crucible.store import Store

__all__ = [
    "BENCHABLE_SLOTS",
    "BENCH_JOB",
    "NO_LIFT",
    "SLOT_BENCH_SCHEMA_VERSION",
    "BenchRulingRequiredError",
    "SlotBench",
    "SlotBenchError",
    "active_bench",
    "add_bench_arguments",
    "bench_document",
    "bench_handler",
    "bench_record_bytes",
    "read_slot_bench",
]

SLOT_BENCH_SCHEMA_VERSION = "slot_bench.v1"

#: The CLI job. On demand and never scheduled: benching a slot is a ruling, and
#: a clock that could file one would hold an authority reserved to a human.
BENCH_JOB = "bench"

#: The one disposition v1 records.
NO_LIFT = "no_lift"

#: The slots a bench may be filed for. See the module docstring.
BENCHABLE_SLOTS: tuple[str, ...] = ("m",)

_SCHEMA_PATH = Path(__file__).parent / "schemas" / f"{SLOT_BENCH_SCHEMA_VERSION}.json"
_RULING_RE = re.compile(RULING_REFERENCE_PATTERN)
_RULED_AT_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}(:[0-9]{2})?Z$")
_HTTPS_RE = re.compile(r"^https://\S+$")


class SlotBenchError(ValueError):
    """A bench record exists and cannot be honoured as what it claims to be.

    Raised, never folded into "no bench": a reader that treated a malformed
    ruling as absent would serve the slot's alpha a human ruled to zero.
    """


class BenchRulingRequiredError(SlotBenchError):
    """A bench was asked for without the ruling fields that make it a ruling."""


@cache
def _validator() -> Draft202012Validator:
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _validate(document: Mapping[str, Any], *, source: str) -> None:
    errors = sorted(_validator().iter_errors(document), key=lambda e: list(e.path))
    if errors:
        paths = "; ".join(
            f"{'/'.join(str(part) for part in e.path) or '<root>'}: {e.message}" for e in errors
        )
        raise SlotBenchError(f"{source} does not satisfy {SLOT_BENCH_SCHEMA_VERSION}: {paths}")


@dataclasses.dataclass(frozen=True)
class SlotBench:
    """One validated bench record, and the digest that identifies it.

    ``digest`` is the content address of the document (``sha256:`` + the
    SHA-256 of its canonical JSON, `crucible.holdout.payload_digest`), so a
    re-filed bench — a new effective day, a new re-exam date, a new champion —
    is a different identity to every reader that counts days under it, the
    trader's served-day evidence first.
    """

    key: str
    digest: str
    document: Mapping[str, Any]

    @property
    def slot(self) -> str:
        return str(self.document["slot"])

    @property
    def champion_at_bench(self) -> str:
        return str(self.document["champion_at_bench"])

    @property
    def effective_trading_day(self) -> str:
        return str(self.document["effective_trading_day"])

    @property
    def reexam_date(self) -> str:
        return str(self.document["reexam_date"])

    @property
    def tracker(self) -> str:
        return str(self.document["tracker"])

    def covers(self, trading_day: str) -> bool:
        """Whether the bench applies to ``trading_day``: on or after its effective session."""
        return trading_day >= self.effective_trading_day

    def headline(self) -> str:
        """The one line every reader prints: "M temporarily benched at zero weight until a
        rebuilt arm passes the gate (ruled …)"."""
        ruled_on = str(self.document["ruled_at"])[:10]
        return (
            f"{self.slot.upper()} temporarily benched at zero weight until a rebuilt arm passes "
            f"the gate (ruled {ruled_on} by "
            f"{self.document['ruled_by']}, {self.tracker})"
        )

    def describe(self) -> str:
        """:meth:`headline` plus what it covers and where the ruling is."""
        return (
            f"{self.headline()}: champion {self.champion_at_bench} served at weight 0 from "
            f"{self.effective_trading_day}, re-exam {self.reexam_date}. "
            f"{str(self.document['reason']).rstrip('.')}. Ruling: {self.document['ruling_ref']}; "
            "record "
            f"{self.key} ({self.digest})"
        )


def read_slot_bench(store: Store, slot: str) -> SlotBench | None:
    """``slot``'s bench record, or ``None`` when none is filed.

    ``None`` means ABSENT and nothing else. A record that exists and does not
    parse or validate raises :class:`SlotBenchError`; a store that cannot
    answer raises whatever it raised. Neither is "not benched".
    """
    key = slot_bench_key(slot)
    read = read_store_document(store, key)
    if read.absent:
        return None
    if read.problem is not None:
        if read.access_problem:
            raise PermissionError(read.problem)
        raise SlotBenchError(read.problem)
    document = read.document or {}
    _validate(document, source=key)
    if document["slot"] != slot:
        raise SlotBenchError(
            f"{key} records slot {document['slot']!r}; a bench filed at another slot's key "
            "is not this slot's ruling."
        )
    return SlotBench(key=key, digest=payload_digest(document), document=document)


def active_bench(store: Store, slot: str, trading_day: str) -> SlotBench | None:
    """``slot``'s bench when it covers ``trading_day``, else ``None``.

    The single question every honouring reader asks. A slot outside
    :data:`BENCHABLE_SLOTS` has no bench key anything files, and reads
    ``None`` without touching the store.
    """
    if slot not in BENCHABLE_SLOTS:
        return None
    bench = read_slot_bench(store, slot)
    if bench is None or not bench.covers(trading_day):
        return None
    return bench


def _required(value: str | None, flag: str, why: str) -> str:
    if value is None or not str(value).strip():
        raise BenchRulingRequiredError(
            f"`crucible bench` requires {flag}: {why} A bench is a ruled record, and "
            "a record missing any ruling field is refused before anything is written — "
            "there is no proceed-anyway form of this command."
        )
    return str(value).strip()


def bench_document(
    *,
    slot: str,
    champion_at_bench: str | None,
    ruled_by: str | None,
    ruled_at: str | None,
    ruling_ref: str | None,
    tracker: str | None,
    reason: str | None,
    evidence: Sequence[str] | None,
    effective_trading_day: str | None,
    reexam_date: str | None,
    filed_by: str,
    filed_at_utc: str,
    filed_on: str,
) -> dict[str, Any]:
    """Build and validate one `slot_bench.v1` document, or refuse.

    Every ruling field is checked by name BEFORE the schema runs, so the
    refusal says which flag is missing rather than which JSON path failed.
    ``filed_on`` is the trading day the filing runs for: a bench may not take
    effect before it, because sessions already resolved stay as they were.
    """
    if slot not in BENCHABLE_SLOTS:
        raise SlotBenchError(
            f"slot {slot!r} cannot be benched: v1 benches {list(BENCHABLE_SLOTS)} only, the "
            "slot whose weight has a consumer (S's alpha vector)."
        )
    if not champion_at_bench:
        raise SlotBenchError(
            f"{champion_key(slot)} names no arm, so there is no champion to bench. The bench "
            "keeps the pointer and its feed in force and zeroes only the alpha; with no "
            "pointer there is nothing for it to keep."
        )
    document = {
        "schema_version": SLOT_BENCH_SCHEMA_VERSION,
        "slot": slot,
        "weight": 0,
        "disposition": NO_LIFT,
        "ruled_by": _required(ruled_by, "--ruled-by", "who ruled."),
        "ruled_at": _required(ruled_at, "--ruled-at", "when the ruling was made (UTC)."),
        "ruling_ref": _required(ruling_ref, "--ruling-ref", "an https link to the ruling."),
        "tracker": _required(tracker, "--tracker", "the tracked issue the ruling is on."),
        "reason": _required(reason, "--reason", "the no-lift finding in one sentence."),
        "evidence": [str(e).strip() for e in (evidence or ()) if str(e).strip()],
        "champion_at_bench": champion_at_bench,
        "effective_trading_day": _required(
            effective_trading_day, "--effective", "the first session the bench covers."
        ),
        "reexam_date": _required(reexam_date, "--reexam", "when the ruling is re-examined."),
        "filed_by": filed_by,
        "filed_at_utc": filed_at_utc,
    }
    if not document["evidence"]:
        raise BenchRulingRequiredError(
            "`crucible bench` requires at least one --evidence: a no-lift finding with no "
            "evidence named is an opinion, not a ruled record."
        )
    if not _RULED_AT_RE.match(document["ruled_at"]):
        raise BenchRulingRequiredError(
            f"--ruled-at {document['ruled_at']!r} is not a UTC instant (YYYY-MM-DDTHH:MM[:SS]Z)."
        )
    if not _HTTPS_RE.match(document["ruling_ref"]):
        raise BenchRulingRequiredError(
            f"--ruling-ref {document['ruling_ref']!r} is not an https link. The reference is "
            "where a reader goes to read the words that were ruled."
        )
    if not _RULING_RE.match(document["tracker"]):
        raise BenchRulingRequiredError(
            f"--tracker {document['tracker']!r} is not a fleet reference "
            f"({RULING_REFERENCE_PATTERN})."
        )
    effective = document["effective_trading_day"]
    try:
        assert_trading_day(effective, context="crucible bench --effective")
        reexam = dt.date.fromisoformat(document["reexam_date"])
    except ValueError as exc:
        raise SlotBenchError(str(exc)) from exc
    if effective < filed_on:
        raise SlotBenchError(
            f"--effective {effective} is before the filing session {filed_on}. A bench covers "
            "sessions from its effective day on, and a session already resolved stays "
            "resolved as it was: benching the past would rewrite what was served."
        )
    if reexam.isoformat() < effective:
        raise SlotBenchError(
            f"--reexam {reexam.isoformat()} is before --effective {effective}; a ruling is "
            "re-examined after it takes effect."
        )
    _validate(document, source="the bench record built from these flags")
    return document


def bench_record_bytes(document: Mapping[str, Any]) -> bytes:
    """The stored bytes: indented, sorted keys — the digest is over the canonical form."""
    return json.dumps(dict(document), indent=2, sort_keys=True).encode("utf-8")


def add_bench_arguments(parser: Any) -> None:
    """The `crucible bench` flag surface, beside the handler that reads it."""
    parser.add_argument(
        "--slot",
        choices=list(BENCHABLE_SLOTS),
        required=True,
        help="The slot to bench. v1 benches M only.",
    )
    parser.add_argument("--ruled-by", default=None, help="Who ruled. Required.")
    parser.add_argument(
        "--ruled-at",
        default=None,
        metavar="YYYY-MM-DDTHH:MMZ",
        help="UTC instant of the ruling. Required.",
    )
    parser.add_argument(
        "--ruling-ref", default=None, metavar="URL", help="https link to the ruling. Required."
    )
    parser.add_argument(
        "--tracker",
        default=None,
        metavar="alpha-engine-config-I<N>",
        help="The tracked issue the ruling sits on. Required.",
    )
    parser.add_argument(
        "--reason", default=None, help="The no-lift finding in one sentence. Required."
    )
    parser.add_argument(
        "--evidence",
        action="append",
        default=None,
        metavar="REF",
        help="What the ruling was made on: a link or a store key. Repeat; at least one.",
    )
    parser.add_argument(
        "--effective",
        default=None,
        metavar="YYYY-MM-DD",
        help="First NYSE session the bench covers; not before --date. Required.",
    )
    parser.add_argument(
        "--reexam",
        default=None,
        metavar="YYYY-MM-DD",
        help="When the ruling is due to be re-examined. Required.",
    )
    parser.add_argument(
        "--operator",
        default=None,
        help="Who is filing. Defaults to $USER; recorded as filed_by.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help=(
            "Replace a bench already filed for the slot. Refused without it: a re-filed "
            "bench is a new identity, and the trader's served-day count restarts on it."
        ),
    )


def bench_handler(args: Any) -> int:
    """`crucible bench --slot m --ruled-by … --ruled-at … --ruling-ref … --tracker …
    --reason … --evidence … --effective … --reexam …`.

    **A refused bench is a USAGE error, not a failed run** — the same shape as
    `crucible holdout --unseal` without a ruling: nothing was benched, nothing
    in the system is wrong, and a `failed` manifest would page an operator over
    a missing flag. Exit 2, the reason on stderr, nothing written.

    A valid bench runs through `run_job` and writes the record through
    `RunContext.record_output`, so the filing is in the manifest's lineage and,
    since the bench decides what money follows, on the money-path chain
    (`crucible.manifest.MONEY_PATH_PREDICATES`).
    """
    from crucible.cli import UsageError  # noqa: PLC0415 - avoid an import cycle at module load
    from crucible.config import settings  # noqa: PLC0415 - avoid an import cycle at module load
    from crucible.documents import load_store_document  # noqa: PLC0415 - same cycle
    from crucible.runner import (  # noqa: PLC0415 - same cycle
        RunContext,
        invocation_discriminator,
        run_job,
    )

    dry_run = bool(getattr(args, "dry_run", False))
    cfg = settings(store_uri=getattr(args, "store", None), dry_run=dry_run)
    store = cfg.store()
    slot = args.slot
    filed_on = args.trading_day.isoformat()
    operator = getattr(args, "operator", None) or os.environ.get("USER", "unknown")
    pointer = champion_key(slot)
    champion = load_store_document(store, pointer).get("arm_id") if store.exists(pointer) else None
    try:
        document = bench_document(
            slot=slot,
            champion_at_bench=str(champion) if champion else None,
            ruled_by=getattr(args, "ruled_by", None),
            ruled_at=getattr(args, "ruled_at", None),
            ruling_ref=getattr(args, "ruling_ref", None),
            tracker=getattr(args, "tracker", None),
            reason=getattr(args, "reason", None),
            evidence=getattr(args, "evidence", None),
            effective_trading_day=getattr(args, "effective", None),
            reexam_date=getattr(args, "reexam", None),
            filed_by=operator,
            filed_at_utc=dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            filed_on=filed_on,
        )
    except SlotBenchError as exc:
        raise UsageError(str(exc)) from exc
    key = slot_bench_key(slot)
    if store.exists(key) and not getattr(args, "replace", False):
        raise UsageError(
            f"{key} is already filed. Pass --replace to file a new bench: the trader's "
            "served-day count is keyed to the bench's digest and restarts on a new one, so "
            "replacing it is a decision, not a retry."
        )
    payload = bench_record_bytes(document)
    bench = SlotBench(key=key, digest=payload_digest(document), document=document)

    def body(ctx: RunContext) -> None:
        ctx.record_rows(rows_in=1, rows_out=0 if dry_run else 1)
        ctx.record_metric(
            {
                "name": "slot_bench_filed",
                "module": "crucible.bench",
                "metric_type": "governance",
                "value": 1.0,
                "unit": "count",
                "n_floor": 1,
                "status": "OK",
                "status_reason": bench.describe(),
                "source_path": key,
                "last_updated_utc": document["filed_at_utc"],
            }
        )
        if dry_run:
            print(payload.decode("utf-8"))
            return
        ctx.record_output(key, payload, schema_version=SLOT_BENCH_SCHEMA_VERSION)

    run_job(
        BENCH_JOB,
        body,
        store=store,
        trading_day=args.trading_day,
        dry_run=dry_run,
        run_mode=getattr(args, "run_mode", None),
        # On demand, so one manifest per INVOCATION (`invocation_discriminator`),
        # the same reason `holdout --unseal` gives.
        discriminator=invocation_discriminator,
    )
    print(f"{'would file' if dry_run else 'filed'} {key}: {bench.describe()}")
    return 0
