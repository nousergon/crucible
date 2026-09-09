"""`crucible <job>` — the one entry point.

Normative source: plan §4.1, §10.8.

    crucible <job> [--date YYYY-MM-DD] [--dry-run] [...job options]

Identical behaviour on a laptop, in a Lambda and on a spot instance: a job is
a Python function, and the scheduler only calls it. Control flow lives in
Python and is tested with pytest, not in a state machine — this is what
replaces the 465 states.

**argparse, not typer.** Typer is a nicer surface and a dependency we would
carry forever for thirteen subcommands with a handful of flags each. Fewer
things to maintain, and one less import on a Lambda cold start (principle 6).
The delta is hand-written `--help` text; the parser below carries it.

**Every job runs through `crucible.runner.run_job`.** That is not a
convention: it is where the manifest guarantee lives, and a job invoked
around it would produce no telemetry on the path where telemetry matters.

The handlers below raise `NotImplementedError` with the owning track named.
The dispatch table, the argument surface and the date resolution are real
today, so three tracks build against a fixed shape.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from crucible import __version__, morning, track_c, track_e, track_f  # track-C, track-E, track-F
from crucible.calendar import resolve_trading_day
from crucible.fault_probe import FAULT_PROBE_JOB, fault_probe_handler
from crucible.faults import FAULT_RECORD_JOB, record_fault
from crucible.gate import SCRIPTED_FAULTS
from crucible.keys import arena_cycle_key, champion_key
from crucible.keys import manifest_key as _promote_manifest_key
from crucible.llm import FAULT_INJECTION_CAPABILITY_CLASSES
from crucible.models import FAULT_OUTCOME_VALUES
from crucible.release_retention import RELEASE_LOCK_JOB, release_lock_handler
from crucible.runmode import RUN_MODES, resolve_run_mode
from crucible.track_a import HANDLERS as TRACK_A_HANDLERS
from crucible.track_a import add_track_a_arguments

__all__ = [
    "FAULT_CAPABILITY_CLASS_JOBS",
    "HANDLERS",
    "JOBS",
    "JobSpec",
    "build_parser",
    "is_stub",
    "main",
]


@dataclass(frozen=True)
class JobSpec:
    """One CLI job: its name, its one-line help, and whether it is scheduled.

    ``scheduled`` is here rather than only in `components.yaml` so the two
    can be checked against each other. A scheduled job whose absence nothing
    watches is the blindness §4.6 exists to remove.
    """

    name: str
    help: str
    scheduled: bool


#: NOT a phase. A `_todo` stub used to cite phase 1's tracker
#: (`alpha-engine-config-I9757`) — structurally derived from `PHASES` so a
#: renumbering couldn't leave it stale, but phase 1 CLOSED 2026-09-02T01:31Z
#: while these stubs remained, so deriving from `PHASES` still pointed every
#: user at a finished issue: `PHASES` carries no open/closed notion, and no
#: amount of deriving from it fixes that. The `crucible` skill states the
#: right anchor: "the epic is the durable anchor to cite; individual phase
#: issue numbers churn as phases close, the epic number does not." A plain
#: `int`, not a string, so no `ast.Constant` anywhere carries the literal
#: (alpha-engine-config-I9839).
_EPIC_ISSUE = 9751


def _epic_tracker() -> str:
    return f"alpha-engine-config-I{_EPIC_ISSUE}"


def _todo(job: str, track: str, note: str) -> Callable[[argparse.Namespace], int]:
    """A handler that refuses loudly, naming who owns it.

    Not a no-op returning 0. A stub that exits cleanly is indistinguishable
    from a job that ran and had nothing to do — which is the shape the whole
    plan exists to make unrepresentable.
    """

    def handler(args: argparse.Namespace) -> int:
        raise NotImplementedError(
            f"`crucible {job}` is not implemented yet — {track}, {_epic_tracker()}. {note}"
        )

    # Marked so the "a stub never returns 0" guard can enumerate stubs from the
    # CODE rather than from a list a track has to remember to edit. A
    # hard-coded exclusion list is itself the suppression-collection bug class
    # (§11.1): it goes stale in the direction of exempting more.
    handler.is_stub = True  # type: ignore[attr-defined]
    return handler


def is_stub(handler: Callable[[argparse.Namespace], int]) -> bool:
    """Whether ``handler`` is an unimplemented placeholder."""
    return bool(getattr(handler, "is_stub", False))


# --------------------------------------------------------------------------
# track-B handlers.
# --------------------------------------------------------------------------


def _resolve_store(args: argparse.Namespace):
    """The store this invocation writes to.

    `--store` wins, then `CRUCIBLE_STORE`. An `s3://` URI is the S3 backend; a
    path is the laptop backend. There is no default root: a job that silently
    wrote into the current working directory would produce artifacts nobody
    could find and a manifest that named them confidently.

    The resolution itself lives in `crucible.store.open_store` — one factory,
    so `--store` means the same thing to every job and to
    `python -m crucible.deploy`, which does not go through this CLI. All this
    adds is the CLI's exit convention: a missing store is a usage error, and
    a traceback for one is noise in front of a one-line fix.

    Read-only when `--dry-run` is set (alpha-engine-config-I9922 N1) — the
    store this returns is the one every handler writes through, so this is
    where `--dry-run`'s own CLI help ("write nothing") becomes true for every
    job rather than only the ones whose handler body happened to check it.
    """
    from crucible.store import open_store

    try:
        return open_store(
            getattr(args, "store", None), dry_run=bool(getattr(args, "dry_run", False))
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _promote(args: argparse.Namespace) -> int:
    """`crucible promote --slot <s> [--revert-to <arm> --reason <why>]`.

    Runs through `run_job` like every other job, so the manifest exists on
    both paths — including the one where the arena raises
    `TrainingIntegrityError` and the whole slot fails (policy §3).
    """
    import os

    from crucible.promote import load_slot_inputs, revert_champion, run_promotion
    from crucible.runner import run_job
    from crucible.slots import get_slot

    spec = get_slot(args.slot)
    revert_to = getattr(args, "revert_to", None)
    if revert_to and not getattr(args, "reason", None):
        raise SystemExit(
            "--revert-to requires --reason: an operator override with no recorded "
            "reason cannot be reviewed later (principles.md §2.1)."
        )
    if revert_to and getattr(args, "dry_run", False):
        # Refused BEFORE the store is even opened (alpha-engine-config-I9935):
        # a revert is a real pointer move with a recorded reason, a dry run
        # writes nothing, and the two flags together are not a request the CLI
        # can honour either way. Before this refusal the combination silently
        # reverted for real (then, after the read-only store landed, died at
        # the first write with a store error that named neither flag).
        raise SystemExit(
            "--revert-to is an operator action; --dry-run has no effect on it — drop one "
            "flag. A revert moves the champion pointer for real and records why; a dry "
            "run writes nothing. Asking for both at once is refused rather than guessed."
        )
    store = _resolve_store(args)

    def job(ctx) -> None:
        as_of = ctx.trading_day.isoformat()
        if revert_to:
            pointer = revert_champion(
                spec=spec,
                register=load_slot_inputs(store, args.slot).register,
                store=store,
                arm_id=revert_to,
                as_of=as_of,
                operator=getattr(args, "operator", None) or os.environ.get("USER", "unknown"),
                reason=args.reason,
                manifest_key=_promote_manifest_key("promote", as_of),
                run_id=ctx.run_id,
            )
            _record_written(ctx, store, (champion_key(pointer.slot),))
            return

        inputs = load_slot_inputs(store, args.slot)
        result = run_promotion(
            spec=spec,
            as_of=as_of,
            register=inputs.register,
            series_by_arm=inputs.series_by_arm,
            incumbent=inputs.incumbent,
            store=None if args.dry_run else store,
            manifest_key=_promote_manifest_key("promote", as_of),
            run_id=ctx.run_id,
        )
        if args.dry_run:
            # `run_promotion(store=None)` already wrote nothing (promote.py's
            # own docstring: "a caller grading in memory ... gets the same
            # decision and writes nothing"); `result.keys_written` is empty,
            # so `_record_written` would be a no-op here regardless. Skipped
            # explicitly rather than relied on implicitly, since `run_job`
            # itself now also skips the manifest write on `dry_run=True`
            # (alpha-engine-config-I9922) — before that fix this branch's
            # sibling call still wrote an `ok` manifest for a promotion that
            # never happened.
            return
        _record_written(ctx, store, result.keys_written)
        # §11: the console renders "cycles since the pointer last moved", and
        # a pointer that has never moved on evidence is a FINDING. It can only
        # be that if the movement is a metric rather than a log line.
        ctx.record_metric(
            {
                "name": "pointer_moved",
                "module": "crucible.promote",
                "metric_type": "gauge",
                "value": 1.0 if result.decision.moved else 0.0,
                "unit": "count",
                "n_floor": 1,
                "status": result.decision.status,
                "status_reason": result.decision.reason or "pointer held",
                "source_path": arena_cycle_key(args.slot, as_of),
                "last_updated_utc": ctx.started.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )

    # `--revert-to` always writes for real — an explicit operator-authority
    # action, `reason` mandatory. `--revert-to --dry-run` is refused above
    # before the store opens, so by this line `dry_run` implies "not a
    # revert": a revert always reaches `run_job` with `dry_run=False` and gets
    # its manifest, and a dry run never reaches the revert branch at all. The
    # `and not revert_to` this used to carry was the guard against a real
    # pointer move with no manifest; the refusal makes that state unreachable.
    run_job(
        "promote",
        job,
        store=store,
        trading_day=args.trading_day,
        dry_run=bool(args.dry_run),
        run_mode=getattr(args, "run_mode", None),
    )
    return 0


def _fault_record(args: argparse.Namespace) -> int:
    """`crucible fault.record --fault <id> --outcome <kind> [--target-job <job>
    --run-id <ulid> --bus-key <key>] --date <trading-day>`.

    Files the durable record `crucible.gate._clause_fault_injection_against_
    scheduled_path` reads (`alpha-engine-config-I10320`/`-I10322`/`-I10327`).
    Runs through `run_job` like every other job (AGENTS.md rule 1), so a
    refusal — no matching manifest, one whose status the outcome forbids, an
    `ok` manifest with no transient-class retry, a bus key the store does not
    hold, a bus key on an `absorbed` record, or a probe that did not observe
    what it required — is itself a fully-telemetered `status: failed` run of
    THIS job, not a bare traceback.

    **Which flags are legal is decided by `--outcome`, in
    `crucible.faults.record_fault`, not here.** argparse cannot express "one
    of these three field sets", and a second copy of the matrix in the parser
    would be the half that drifts; the CLI passes what it was given and the
    producer refuses. The refusal messages name the flag.

    `--date` is required explicitly rather than defaulting to "the last
    completed trading day" the way every other job's does: this record must
    name the EXACT day the fault was induced against, and a wall-clock
    default would silently file it under the wrong key the first time this
    command is run on a different day than the exercise it describes.
    """
    if not getattr(args, "date", None):
        raise SystemExit(
            "crucible fault.record requires --date YYYY-MM-DD naming the trading day the "
            "fault was induced against. The default every other job uses (the last "
            "completed trading day) is a wall-clock guess, and this record must name the "
            "exact day its excused manifest lives under."
        )
    store = _resolve_store(args)
    trading_day = args.trading_day.isoformat()

    def job(ctx) -> None:
        record_fault(
            ctx,
            store,
            fault_id=args.fault,
            outcome=args.outcome,
            target_job=getattr(args, "target_job", None),
            trading_day=trading_day,
            run_id=getattr(args, "run_id", None),
            bus_key=getattr(args, "bus_key", None),
        )

    from crucible.runner import run_job

    ctx = run_job(
        FAULT_RECORD_JOB,
        job,
        store=store,
        trading_day=args.trading_day,
        dry_run=bool(args.dry_run),
        run_mode=getattr(args, "run_mode", None),
    )
    print(json.dumps({"run_id": ctx.run_id, "outputs": [o["key"] for o in ctx.outputs]}, indent=2))
    return 0


def _record_written(ctx, store, keys) -> None:
    """Record artifacts the job wrote through the store directly.

    Re-reads each key and records it through `ctx.record_output`, which
    re-PUTs identical bytes — a no-op by content addressing. The alternative
    is appending to `ctx.outputs` by hand, which would let a caller record an
    output it never actually wrote; here the manifest can only name bytes
    that are really in the store.
    """
    for key in keys:
        ctx.record_output(key, store.get_bytes(key))


JOBS: dict[str, JobSpec] = {
    "data.daily": JobSpec("data.daily", "Compile one trading day of market data", True),
    "data.weekly": JobSpec("data.weekly", "Weekly data refresh and coverage pass", True),
    "data.heal": JobSpec("data.heal", "Repair a named gap in the store, idempotently", False),
    "experiment.new": JobSpec("experiment.new", "Register an immutable arm from a recipe", False),
    "experiment.run": JobSpec("experiment.run", "Score one arm for one trading day", True),
    "experiment.grade": JobSpec("experiment.grade", "Run one slot's arena cycle", True),
    "promote": JobSpec("promote", "Move a slot's champion pointer, evidence-gated", False),
    "report": JobSpec("report", "Reduce the week's manifests into the attribution table", True),
    "explain": JobSpec("explain", "Walk a run_id or verdict back to what produced it", False),
    "migrate.history": JobSpec(
        "migrate.history", "Import v1 arm history with its provenance", False
    ),
    "release.pin": JobSpec("release.pin", "Repoint a release, or pin the trader to one", False),
    # alpha-engine-config-I9898: the repair for a release published before
    # I9787's write-time Object Lock fix. On-demand, like release.pin — an
    # operator runs it against a named sha, never a schedule.
    RELEASE_LOCK_JOB: JobSpec(
        RELEASE_LOCK_JOB, "Apply Object Lock retention to a published release", False
    ),
    "smoke": JobSpec("smoke", "A real end-to-end run that gates a release flip", False),
    # track-C (alpha-engine-config-I9757): the observing surfaces themselves.
    # They are jobs like any other, so they write manifests like any other and
    # the thing that watches the fleet is watched on the same terms.
    "alerts.sweep": JobSpec("alerts.sweep", "Evaluate the two page conditions and page", True),
    "heartbeat": JobSpec("heartbeat", "Weekly proof the alerting path itself is alive", True),
    "drift": JobSpec("drift", "Feature PSI, prediction drift and IC decay", True),
    "console": JobSpec("console", "Render the static console page from the manifests", True),
    # alpha-engine-config-I9837. Its own job on its own DAILY schedule, not a
    # stage of the weekly arc: a board that only refreshes once phase 2 opens
    # could not have rendered the gap that closed phase 1.
    "board": JobSpec(
        "board", "Render the fully-declared board — every objective, gate and component", True
    ),
    # track-F (alpha-engine-config-I9757). `weekly` is what the Saturday
    # schedule dispatches: `components.yaml` declares six jobs as weekly and
    # the scheduler started exactly one of them, so five components were
    # deadlined, watched for absence, and triggered by nobody. `gate` reads
    # what a phase produced and says whether it may exit — a phase gate that
    # is a MEASUREMENT cannot be satisfied by a merge.
    "weekly": JobSpec("weekly", "Run the declared weekly arc for one trading day", True),
    "gate": JobSpec("gate", "Read a phase's artifacts and report its exit gate", False),
    # alpha-engine-config-I10095. `gate` stays on-demand — a gate on a schedule
    # would be a gate whose absence pages between phases — but the RECORD of a
    # phase's exit cannot wait for somebody to run one. This job reads every
    # registered phase's gate daily and files the closing record for each that
    # reads MET and has none, under a writer identity the board deliberately
    # is not.
    track_f.GATE_CLOSE_JOB: JobSpec(
        track_f.GATE_CLOSE_JOB,
        "File the closing record of every phase whose exit gate reads MET",
        True,
    ),
    # alpha-engine-config-I9896. The daily accountability delivery Brian
    # believed existed on 2026-09-02 and did not: the board was rendered into
    # the store every day and handed to nobody. It READS that board -- it does
    # not render one -- so the reporting surface cannot produce the artifact
    # it reports on.
    morning.MORNING_JOB: JobSpec(
        morning.MORNING_JOB,
        "Deliver the board's reading to the operator channel at 06:00 PT",
        True,
    ),
    # alpha-engine-config-I10320/-I10322. On-demand, like data.heal and
    # experiment.new: an operator/procedure runs it once per fault induced,
    # never on a schedule.
    FAULT_RECORD_JOB: JobSpec(
        FAULT_RECORD_JOB,
        "File the durable record of one exercised scripted fault, or refuse",
        False,
    ),
    # alpha-engine-config-I10343. The INDUCER for plan §10.7 fault 3, kept
    # separate from `fault.record` (the attester) on purpose: a job that both
    # induced a fault and attested to it is the rubber stamp
    # `crucible.faults`' refusals exist to prevent. On-demand — a permanently
    # failing job on a schedule would page every cycle forever.
    FAULT_PROBE_JOB: JobSpec(
        FAULT_PROBE_JOB,
        "Induce a router transport failure on the real dispatched path (§10.7 fault 3)",
        False,
    ),
}

#: The jobs that carry `--fault-capability-class`, exhaustively.
#:
#: **A closed set rather than every job, because the alternative is a silent
#: no-op** (rule 5). The flag only does anything for a job that reaches a
#: model: `crucible.llm.call` is what reads the override off the run context,
#: so passing it to `board` or `heartbeat` would be accepted, recorded on the
#: manifest and change nothing — an operator would have "induced" a fault that
#: never had a call site to fire in. A job absent from this set refuses the
#: flag as an unknown argument instead.
#:
#: `fault.probe` is the only member today, and the flag is REQUIRED there: the
#: job exists for no other purpose. Phase 5 adds `experiment.run` — which is
#: the point of building the override rather than hard-wiring the probe's
#: class, since faulting a real arm run is what proves a job that knows
#: nothing about fault injection fails correctly. That addition is a
#: deliberate edit here, on the day such an arm exists.
FAULT_CAPABILITY_CLASS_JOBS: frozenset[str] = frozenset({FAULT_PROBE_JOB})


HANDLERS: dict[str, Callable[[argparse.Namespace], int]] = {
    "data.daily": _todo("data.daily", "track A", "Lifts the ingest core from nousergon-data."),
    "data.weekly": _todo("data.weekly", "track A", "Weekly refresh + coverage MetricRecords."),
    "data.heal": _todo(
        "data.heal",
        "track A",
        "Must record what it repaired in rows_in/rows_out/rows_rejected, not just log it.",
    ),
    "experiment.new": _todo(
        "experiment.new",
        "track A",
        "Arm id is the hash of its spec; an edited recipe is a NEW arm carrying `supersedes`.",
    ),
    "experiment.run": _todo(
        "experiment.run",
        "track A",
        "Seeded by crucible-research/scripts/run_experiment.py (PR784).",
    ),
    "experiment.grade": _todo(
        "experiment.grade",
        "track A",
        "Calls nousergon_lib.arena.engine.run_cycle; re-implementing §§3-6 is a defect.",
    ),
    # track-B
    "promote": _promote,
    "report": track_e.report_handler,
    "explain": _todo(
        "explain",
        "track A",
        "§10.8: walks the manifest lineage from a verdict to the arms, features, data "
        "snapshot, code sha, cost and LLM calls that produced it.",
    ),
    "migrate.history": _todo(
        "migrate.history",
        "track A",
        "Carries R's `operator_bootstrap` champion flag into v2 so the first "
        "evidence-won promotion is visible as such.",
    ),
    # track-C handlers live in crucible/track_c.py so three tracks can land
    # code in parallel without editing one another's lines.
    "release.pin": track_c.release_pin_handler,
    RELEASE_LOCK_JOB: release_lock_handler,
    "smoke": track_c.smoke_handler,
    "alerts.sweep": track_c.sweep_handler,
    "heartbeat": track_c.heartbeat_handler,
    "drift": track_c.drift_handler,
    "console": track_c.console_handler,
    "board": track_c.board_handler,
    # track-F
    "weekly": track_f.weekly_handler,
    "gate": track_f.gate_handler,
    track_f.GATE_CLOSE_JOB: track_f.gate_close_handler,
    morning.MORNING_JOB: morning.morning_handler,
    FAULT_RECORD_JOB: _fault_record,
    FAULT_PROBE_JOB: fault_probe_handler,
}


# track-A: the implemented handlers replace their `_todo` placeholders. Done by
# assignment rather than by editing the table above so each track owns one
# import line, and so a handler that failed to import is a loud ImportError at
# start-up rather than a job that is quietly still a stub.
HANDLERS.update(TRACK_A_HANDLERS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crucible",
        description=(
            "The experiment harness. One command per job, identical on a laptop, in a "
            "Lambda and on a spot instance. Every job writes a run manifest, including "
            "the ones that fail."
        ),
    )
    parser.add_argument("--version", action="version", version=f"crucible {__version__}")
    subparsers = parser.add_subparsers(dest="job", metavar="<job>", required=True)

    for spec in JOBS.values():
        sub = subparsers.add_parser(spec.name, help=spec.help, description=spec.help)
        sub.add_argument(
            "--date",
            metavar="YYYY-MM-DD",
            help=(
                "Trading day to run for. Defaults to the last completed session — a "
                "Saturday run binds to Friday's close (§4.12). A non-trading day is "
                "REFUSED, not silently resolved."
            ),
        )
        sub.add_argument(
            "--run-mode",
            choices=list(RUN_MODES),
            default=None,
            help=(
                "Whether this invocation is a LIVE run or a REPLAY of a historical "
                "trading day. Recorded in the manifest and read by phase 2's exit "
                "gate. No default: omitted, $CRUCIBLE_RUN_MODE is consulted, and an "
                "invocation that declares neither is REFUSED. The date cannot answer "
                "this — a replay replays a real past Saturday."
            ),
        )
        sub.add_argument(
            "--dry-run",
            action="store_true",
            # Enforced at the store (`crucible.store.open_store`,
            # `crucible.config.Settings.store` — see `read_only`), not in this
            # help text or in each handler: a job whose own body does not
            # check this flag now reports and writes nothing where it can, or
            # raises loudly on the first write it attempts otherwise, rather
            # than the silent real write this closed (alpha-engine-config-
            # I9922 N1, independent review of crucible-PR74, 2026-09-03).
            help="Resolve inputs and report what would be written; write nothing.",
        )
        sub.add_argument(
            "--store",
            metavar="URI",
            help="Store root: an s3://bucket/prefix URI or a local directory path.",
        )
        if spec.name in ("experiment.run", "experiment.grade", "promote", "experiment.new"):
            sub.add_argument("--slot", choices=["u", "r", "m", "s"], required=True)
        if spec.name == "promote":
            # track-B. The revert is one command by design: a rollback that
            # needs three steps is a rollback nobody performs under pressure.
            sub.add_argument(
                "--revert-to",
                metavar="ARM_ID",
                help=(
                    "Point the slot at ARM_ID by operator authority instead of running "
                    "the cycle. Recorded as promotion_source=operator_bootstrap, never "
                    "as evidence. Requires --reason."
                ),
            )
            sub.add_argument(
                "--operator",
                default=None,
                help="Who is reverting. Defaults to $USER; recorded in the pointer.",
            )
            sub.add_argument(
                "--reason",
                default=None,
                help="Why. Mandatory with --revert-to: an unexplained operator "
                "override is the one pointer movement nobody can reconstruct later.",
            )
        if spec.name in ("experiment.run", "experiment.new"):
            # track-A: NOT required for `experiment.run`. Policy §3 scores every
            # registered arm every cycle, so the default is "all of them"; naming
            # one narrows the run to it, which is a debugging affordance rather
            # than the normal path. `experiment.new` still requires it, because
            # registering "whichever arms happen to be on disk" is not a
            # deliberate act.
            sub.add_argument(
                "--arm",
                metavar="NAME|ARM_ID",
                required=spec.name == "experiment.new",
                help=(
                    "Restrict to one arm, by bare name or by registered "
                    "`{slot}:{name}:{spec_hash}` id — `experiment.new` prints ids, so "
                    "the id is the form an operator is holding. Omitted, every "
                    "registered arm in the slot is produced — an arm that skipped a "
                    "cycle records a MISS, and a miss is data."
                ),
            )
        if spec.name == "explain":
            sub.add_argument("target", metavar="RUN_ID|VERDICT_KEY")
        if spec.name == "release.pin":
            sub.add_argument("sha", metavar="RELEASE_SHA")
            sub.add_argument("--target", choices=["current", "trader"], default="current")
        if spec.name == RELEASE_LOCK_JOB:
            sub.add_argument("sha", metavar="RELEASE_SHA")
        if spec.name == "smoke":  # track-C
            sub.add_argument(
                "--release",
                metavar="SHA",
                required=True,
                help=(
                    "The release sha this smoke is verifying. Recorded as the "
                    "manifest's release_sha; the pointer flip refuses a smoke "
                    "manifest belonging to another build."
                ),
            )
        if spec.name == "gate":  # track-F
            sub.add_argument(
                "--gate",
                required=True,
                choices=track_f.gate_names(),
                help=(
                    "Which phase gate to read. A gate not registered in "
                    "`crucible.gate.GATES` has no clause list, and running it would "
                    "report a pass over nothing."
                ),
            )
            sub.add_argument(
                "--weeks",
                type=int,
                default=None,
                help=(
                    "Override the gate's declared window, in weekly trading days. "
                    "The default is the gate's own width; a narrower window is a "
                    "debugging affordance and is recorded in the gate artifact."
                ),
            )
            sub.add_argument(
                "--closing-comment",
                action="store_true",
                help=(
                    "Also print the `phase_closing_reading.v1` block this reading "
                    "justifies, for pasting into the phase issue's closing comment. "
                    "`alpha-engine-config`'s phase-tracker consistency sweep refuses a "
                    "CLOSED phase issue that carries no such block or one that does not "
                    "read MET."
                ),
            )
        if spec.name == "data.heal":
            sub.add_argument("--gap", required=True, help="The named gap to repair.")
        if spec.name == FAULT_RECORD_JOB:
            sub.add_argument(
                "--fault",
                required=True,
                choices=SCRIPTED_FAULTS,
                help="Which scripted fault (plan §10.7) this record is for.",
            )
            sub.add_argument(
                "--outcome",
                required=True,
                choices=FAULT_OUTCOME_VALUES,
                help=(
                    "How the fault ended. induced: it fired and the job FAILED "
                    "(--target-job, --run-id and --bus-key all required). absorbed: it "
                    "fired and the declared transient class handled it, so the manifest "
                    "reads ok and records the retry (--target-job and --run-id required, "
                    "--bus-key REFUSED — a page would mean the retry did not work). "
                    "unreachable: the state cannot be entered, evidenced by an executed "
                    "probe per closed path (no --target-job, --run-id or --bus-key, so "
                    "the record cannot excuse any manifest)."
                ),
            )
            sub.add_argument(
                "--target-job",
                dest="target_job",
                default=None,
                help=(
                    "The CLI job whose manifest this record describes — e.g. data.weekly. "
                    "Required for induced and absorbed, refused for unreachable. "
                    "Not `--job`: that positional slot is this command's own name."
                ),
            )
            sub.add_argument(
                "--run-id",
                dest="run_id",
                default=None,
                help=(
                    "The run_id of the manifest this record describes. Required for "
                    "induced (which needs it to read status: failed) and absorbed (status: "
                    "ok with a transient-class retry in attempts[]); REFUSED for "
                    "unreachable, which names no run and so cannot excuse one. This "
                    "command never invents a run_id."
                ),
            )
            sub.add_argument(
                "--bus-key",
                dest="bus_key",
                default=None,
                metavar="alerts/{day}/{incident}.json",
                help=(
                    "The alert bus row this fault produced. REQUIRED for induced and "
                    "refused unless the store holds it — a failed job that paged nobody "
                    "is half of plan §10.7's exercise. REFUSED for absorbed and "
                    "unreachable: on an absorbed fault a page would mean the retry did "
                    "not work, so the row's ABSENCE is part of what the record asserts, "
                    "and it is never a field borrowed from an unrelated incident to make "
                    "a clause read better."
                ),
            )
        if spec.name in FAULT_CAPABILITY_CLASS_JOBS:
            sub.add_argument(
                "--fault-capability-class",
                dest="fault_capability_class",
                required=spec.name == FAULT_PROBE_JOB,
                choices=sorted(FAULT_INJECTION_CAPABILITY_CLASSES),
                default=None,
                help=(
                    "FAULT INJECTION, not the normal path: route every LLM call this run "
                    "makes to a capability class whose router group is contracted NEVER to "
                    "serve, so plan §10.7 fault 3 (the router returns an error) is induced "
                    "against the real dispatched path. Only a fault-injection class is "
                    "accepted — a real router group is refused, because a flag that could "
                    "choose which model serves a graded run is a second routing plane. "
                    "Recorded verbatim as `fault_capability_class` on the manifest, so an "
                    "arranged transport failure is never mistakable for a real one. No "
                    "other consumer's routing is touched: the redirect is this run's "
                    "context and nothing else."
                ),
            )
        if spec.name == "alerts.sweep":
            sub.add_argument(
                "--now",
                metavar="YYYY-MM-DD",
                default=None,
                help=(
                    "OPERATOR OVERRIDE, not the normal path: evaluate the catch-up and "
                    "ceiling windows as though this trading day were the sweep's own "
                    "close, instead of the real wall clock — so a fault on a historical "
                    "day can be swept for real without landing inside "
                    "crucible.gate's live pages_within_ceiling window (I10125). Must be "
                    "a trading day and not in the future; refused (not silently "
                    "resolved) otherwise. Recorded "
                    "verbatim as now_override_utc on every manifest this run writes, so "
                    "an overridden sweep is never mistaken for a natural one. The "
                    "scheduled/unattended sweep never passes this flag and is "
                    "unaffected by its existence."
                ),
            )

        # track-A: the data, feature, U/R, explain and migrate jobs' own flags.
        add_track_a_arguments(spec.name, sub)

    return parser


def resolve_date(raw: str | None, *, now: dt.datetime | None = None) -> dt.date:
    """`--date` to a trading day.

    An explicit date is parsed and returned as given; it is the runner that
    refuses a non-trading day, so the refusal happens in one place and a
    backfill cannot route around it by calling a different entry point.
    """
    if raw is None:
        return resolve_trading_day(now)
    try:
        return dt.date.fromisoformat(raw)
    except ValueError as exc:
        raise SystemExit(f"--date must be YYYY-MM-DD; got {raw!r} ({exc})") from exc


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code.

    Exceptions are NOT caught here. A job that failed must exit non-zero with
    its traceback intact — the manifest already records the cause, and a
    tidy error message that returns 0 is a degraded-SUCCEEDED by another name.
    """
    args = build_parser().parse_args(argv)
    args.trading_day = resolve_date(getattr(args, "date", None))
    # Resolved once, here, so every handler passes the SAME value to `run_job`
    # and a `--run-mode` typo is a usage error before any job starts. Resolved
    # even for the handlers that never write a manifest: an invocation is
    # live or a replay regardless of what it happens to produce.
    args.run_mode = resolve_run_mode(getattr(args, "run_mode", None))
    return HANDLERS[args.job](args)


if __name__ == "__main__":
    sys.exit(main())
