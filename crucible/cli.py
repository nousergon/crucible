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
from crucible.gate import SCRIPTED_FAULTS, missing_required_env
from crucible.holdout import (
    HOLDOUT_JOB,
    add_holdout_arguments,
    holdout_handler,
)
from crucible.iac_conformance import IAC_CONFORMANCE_JOB, iac_conformance_handler
from crucible.integration_summary import INTEGRATION_TEST_JOB, integration_test_handler
from crucible.keys import arena_cycle_key, champion_key
from crucible.keys import manifest_key as _promote_manifest_key
from crucible.llm import FAULT_INJECTION_CAPABILITY_CLASSES
from crucible.models import FAULT_OUTCOME_VALUES
from crucible.release_retention import RELEASE_LOCK_JOB, release_lock_handler
from crucible.runmode import RUN_MODES, RunModeError, resolve_run_mode
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


#: The money-path hash-chain verifier's own tracker issue (`crucible-PR240`)
#: — a HISTORICAL, non-phase issue that `crucible.gate.phase_tracker` can
#: never derive, so `tests/test_no_stale_tracker_literals.py` requires this
#: shape (a plain `int`, read at f-string time) rather than a literal
#: `"alpha-engine-config-I10414"` anywhere a user-facing string can carry it.
_MONEY_PATH_CHAIN_ISSUE = 10414


def _money_path_chain_tracker() -> str:
    return f"alpha-engine-config-I{_MONEY_PATH_CHAIN_ISSUE}"


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

    from crucible.promote import (
        load_slot_inputs,
        read_graded_cycle,
        revert_champion,
        run_promotion,
    )
    from crucible.runner import resolve_code_sha, run_job
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
    # `alpha-engine-config-I9759`: four slots, one job name, one trading day,
    # four writers — exactly `crucible.keys.manifest_key`'s discriminator case,
    # and the one `experiment.run`/`experiment.grade` already take. Without it
    # every slot's promote wrote `runs/promote/{day}/run.json` and the last one
    # to run silently erased the other three; once promote became an arc stage
    # (`crucible.weekly.ARC_SLOT_JOBS`) that collision would happen every
    # Saturday. It is also what the pointer's own `manifest_key` must name, or
    # `explain` walks a promotion back to a manifest belonging to another slot.
    promote_manifest = _promote_manifest_key("promote", args.trading_day, discriminator=args.slot)

    def job(ctx) -> None:
        as_of = ctx.trading_day.isoformat()
        # `alpha-engine-config-I10506`: neither call below used to receive a
        # `code_sha` at all, so `crucible.promote`'s own `code_sha or "0" * 40`
        # fallback fired on EVERY real invocation and every champion pointer
        # this harness ever wrote carried the placeholder. `run_job` already
        # resolves the same value for THIS run's own manifest
        # (`crucible.runner.resolve_code_sha`, before `job(ctx)` runs — so a
        # box that cannot measure it never reaches here at all); resolved a
        # second time here because `RunContext` does not carry it.
        code_sha = resolve_code_sha()
        if revert_to:
            pointer = revert_champion(
                spec=spec,
                register=load_slot_inputs(store, args.slot).register,
                store=store,
                arm_id=revert_to,
                as_of=as_of,
                operator=getattr(args, "operator", None) or os.environ.get("USER", "unknown"),
                reason=args.reason,
                manifest_key=promote_manifest,
                run_id=ctx.run_id,
                code_sha=code_sha,
            )
            _record_written(ctx, store, (champion_key(pointer.slot),))
            return

        inputs = load_slot_inputs(store, args.slot)
        # `alpha-engine-config-I10679`: the cycle `experiment.grade` computed
        # an hour earlier, read back rather than recomputed — the pointer
        # decision, the retirement verdicts and the serving preconditions
        # (the M behavioural veto, the S contamination attestation) are ALL
        # already inside it. `read_graded_cycle` also refuses loudly if that
        # run's own manifest is absent, failed, or does not claim the cycle.
        graded_cycle = read_graded_cycle(store, args.slot, as_of)
        result = run_promotion(
            spec=spec,
            register=inputs.register,
            cycle=graded_cycle,
            pointer_etag=inputs.pointer_etag,
            store=None if args.dry_run else store,
            manifest_key=promote_manifest,
            run_id=ctx.run_id,
            code_sha=code_sha,
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
        discriminator=args.slot,
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


def _migrate_code_sha(args: argparse.Namespace) -> int:
    """`crucible migrate.code_sha [--dry-run] [--store URI]`.

    Not run through `run_job` (see the note beside its subparser in
    `build_parser`) — it patches other jobs' manifests, so it uses the same
    `--store`/`--dry-run` resolution every job uses (`_resolve_store`) without
    claiming a manifest of its own. See `crucible.migrate.run_migrate_code_sha`
    for the derivation and refusal rules; this is the printed report only.
    """
    from crucible.migrate import run_migrate_code_sha

    store = _resolve_store(args)
    report = run_migrate_code_sha(store, dry_run=bool(getattr(args, "dry_run", False)))
    print(f"crucible migrate.code_sha: {report.summary_line()}")
    for row in report.rewritten:
        print(
            f"  rewrote {row['key']}: {row['old_code_sha']} -> {row['new_code_sha']} "
            f"(from {row['source_key']})"
        )
    for row in report.refused:
        print(f"  refused {row['key']}: {row['reason']}")
    return 0


def _migrate_arm_filed_on(args: argparse.Namespace) -> int:
    """`crucible migrate.arm_filed_on [--dry-run] [--store URI]`.

    One-time, audited, DATE-ONLY repair of the `registered` rows whose event
    date was stamped from the recipe's `created_date` instead of the day the
    append happened. Not run through `run_job`, for the same reason
    `migrate.code_sha` is not: it corrects documents another job wrote and
    claims no manifest of its own. See
    `crucible.migrate.run_migrate_arm_filed_on` and the table above it for
    the derivation and the refusal rules; this is the printed report only.
    """
    from crucible.migrate import run_migrate_arm_filed_on

    store = _resolve_store(args)
    report = run_migrate_arm_filed_on(store, dry_run=bool(getattr(args, "dry_run", False)))
    print(f"crucible migrate.arm_filed_on: {report.summary_line()}")
    for row in report.corrected:
        print(
            f"  corrected {row['arm_id']} in {row['key']}: {row['old_date']} -> "
            f"{row['new_date']} (evidence: object version {row['evidence_version_id']} "
            f"LastModified {row['evidence_last_modified']})"
        )
    for row in report.refused:
        print(f"  refused {row['arm_id']}: {row['reason']}")
    return 0


#: The slots whose v1 champion pointer `migrate.history` imports. M's v1
#: promotions are a dated lineage series, read and counted but never turned
#: into a pointer (see `crucible.migrate.SOURCES`), and S has no v1 pointer.
_MIGRATE_HISTORY_SLOTS: tuple[str, ...] = ("u", "r")


def _migrate_history(args: argparse.Namespace) -> int:
    """`crucible migrate.history [--v1-store URI] [--strategy-dir DIR] [--allow-missing]`.

    Imports v1's U and R champion pointers into the v2 register and
    `champions/{slot}/current.json` (plan §9.8, `alpha-engine-config-I10713`).

    * **Recipes come from the published strategy tree**, through the same
      `load_arm_specs` the U/R cycle uses (`--strategy-dir` checkout first,
      else `strategy/current/arms/{slot}/` in the store). A v1 champion name
      maps to the recipe of the same name — `momentum_sleeve` for U,
      `scanner_predictor_direct` for R — and a champion with no recipe fails
      the run (`MigrationSourceMissing`). A recipe directory that cannot be
      read is not skipped: the previous handler swallowed it, which turned a
      missing tree into a "no recipe supplied" error naming the wrong cause.
    * **v1 is read through a read-only store.** `--v1-store` wins; absent, the
      data bucket setting (`CRUCIBLE_ARCTIC_BUCKET`, which the box exports
      from the stack's `DataBucketName` — the bucket v1 writes its `config/`
      pointers to) is read at its root. Neither resolved is a usage error,
      and so is a v1 store equal to the v2 store: that was the previous
      handler's default, under which every v1 source read as absent.
    * `--dry-run` resolves every source, recipe and existing pointer and
      prints what would be written; nothing is written, including the
      manifest.
    """
    from crucible.config import settings as resolve_settings
    from crucible.config import store_from_uri
    from crucible.migrate import run_migrate_history
    from crucible.runner import run_job
    from crucible.slots.arms import load_arm_specs
    from crucible.store import read_only

    dry_run = bool(getattr(args, "dry_run", False))
    config = resolve_settings(
        store_uri=getattr(args, "store", None),
        strategy_dir=getattr(args, "strategy_dir", None),
        dry_run=dry_run,
    )
    try:
        store = config.store()
    except ValueError as exc:
        raise UsageError(str(exc)) from exc
    v1_uri = getattr(args, "v1_store", None) or (
        f"s3://{config.arctic_bucket}" if config.arctic_bucket else None
    )
    if not v1_uri:
        raise UsageError(
            "crucible migrate.history needs the v1 source: pass --v1-store URI, or set "
            "CRUCIBLE_ARCTIC_BUCKET (the data bucket v1 writes its champion pointers to). "
            "There is no default — reading v1 from the v2 store finds nothing and reports "
            "every source absent."
        )
    if v1_uri.rstrip("/") == str(config.store_uri).rstrip("/"):
        raise UsageError(
            f"crucible migrate.history: --v1-store {v1_uri!r} is the v2 store itself. The "
            "v1 pointers live in the v1 data bucket; reading them here finds nothing."
        )
    v1_store = read_only(store_from_uri(v1_uri), reason="migrate.history reads v1 read-only")

    recipes = {}
    for slot in _MIGRATE_HISTORY_SLOTS:
        for recipe in load_arm_specs(slot, store=store, strategy_dir=config.strategy_dir):
            if recipe.name in recipes:
                raise ValueError(
                    f"recipe name {recipe.name!r} is published in both slot "
                    f"{recipes[recipe.name].slot!r} and slot {slot!r}; a v1 champion name "
                    "cannot be mapped to one of them without guessing."
                )
            recipes[recipe.name] = recipe

    def job(ctx) -> None:
        result = run_migrate_history(
            ctx,
            v1_store=v1_store,
            slots=_MIGRATE_HISTORY_SLOTS,
            arm_recipes=recipes,
            allow_missing=bool(getattr(args, "allow_missing", False)),
            dry_run=dry_run,
        )
        print(json.dumps(result, indent=2, sort_keys=True))

    run_job(
        "migrate.history",
        job,
        store=store,
        trading_day=args.trading_day,
        dry_run=dry_run,
        run_mode=getattr(args, "run_mode", None),
    )
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
    # alpha-engine-config-I10927. The DERIVED sibling of `experiment.new`: it
    # registers every recipe the pinned release declares and the slot's
    # register lacks, and it is an arc stage (11:00, before `experiment.run`
    # at 12:00) so a recipe merged during the week is scored on the next
    # Saturday with no operator in the path. `experiment.new --arm` keeps its
    # required flag and its meaning — see `crucible.registration` for why
    # deriving from the release pin is a deliberate act and a list of arms
    # would not be.
    "experiment.register": JobSpec(
        "experiment.register",
        "Register every recipe the release in force declares and the register lacks",
        True,
    ),
    "experiment.run": JobSpec("experiment.run", "Score one arm for one trading day", True),
    "experiment.grade": JobSpec("experiment.grade", "Run one slot's arena cycle", True),
    # alpha-engine-config-I10696 (Brian's ruling (a), 2026-09-14). On-demand,
    # like `data.heal`: it repairs a HISTORY, and a schedule that produced an
    # arm's past on a clock would be a second producer of the artifacts the
    # weekly arc already writes. `deadline: null` in components.yaml follows
    # from that — a dispatch record is what makes its absence gradeable.
    "experiment.backfill": JobSpec(
        "experiment.backfill",
        "Produce one arm's history over a session range, point-in-time",
        False,
    ),
    "promote": JobSpec("promote", "Move a slot's champion pointer, evidence-gated", False),
    "report": JobSpec("report", "Reduce the week's manifests into the attribution table", True),
    "explain": JobSpec("explain", "Walk a run_id or verdict back to what produced it", False),
    # `alpha-engine-config-I10502` (phase-3 `sealed_holdout`). NOT scheduled,
    # and it must never be: unsealing the holdout is a RESERVED matter
    # (`principles.md` §3.2), so a clock that could invoke this would be an
    # automation holding an authority reserved to a human ruling. The read
    # form writes no manifest at all (see `holdout_handler`).
    HOLDOUT_JOB: JobSpec(
        HOLDOUT_JOB,
        "Read the sealed holdout's seal state, or unseal it under a ruling",
        False,
    ),
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
    # alpha-engine-config-I10418. `dispatch: arc` in components.yaml, so
    # `crucible.weekly.arc_stages` picks it up as a stage of the SAME weekly
    # run — no new schedule, no separate detector fleet, per the issue's own
    # constraint. Two comparisons, both required: account vs template
    # (crucible.iac_conformance.audit_account_vs_template) and template vs
    # the plan's own declared inventory
    # (crucible.iac_conformance.audit_template_vs_declared) — the second is
    # what catches a stale plan objective the first renders GREEN against by
    # construction. A finding pages only once it persists two consecutive
    # weekly cycles (crucible.iac_conformance.IacConformanceDrift).
    IAC_CONFORMANCE_JOB: JobSpec(
        IAC_CONFORMANCE_JOB,
        "Account-vs-template and template-vs-declared-inventory IaC conformance",
        True,
    ),
    # alpha-engine-config-I10459. Promotes the integration tier's summary
    # artifact from a hand-rolled `store.put_bytes` call inside
    # `.github/workflows/integration-nightly.yml` to a real registered job:
    # shells out to `pytest tests/integration` and writes a real
    # `run_manifest.v2` document, like every other job. NOT scheduled — it
    # stays workflow-triggered by that workflow's own
    # `schedule`/`workflow_call`/`workflow_dispatch` triggers, never a second
    # independent starter for the same nightly run.
    INTEGRATION_TEST_JOB: JobSpec(
        INTEGRATION_TEST_JOB,
        "Run tests/integration (real S3 + real ArcticDB) and report pass/fail",
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

#: Handlers wired into `HANDLERS` that are deliberately NOT in `JOBS`: a
#: one-off repair over documents another job already wrote (patches a field
#: in place rather than writing a new `run_manifest.v2` document), so it
#: cannot honestly claim a `job` enum slot or a `components.yaml` row meant
#: for a manifest-producing job (see the note beside `migrate.code_sha`'s
#: subparser in `build_parser`, alpha-engine-config-I10626).
#: `tests/test_cli_and_alerts.py::TestJobSurface::test_every_job_has_a_handler`
#: reads this set rather than requiring `set(HANDLERS) == set(JOBS)`, so a
#: FUTURE handler that silently drops out of `JOBS` by accident is still
#: caught — only a name listed here is exempt.
NON_JOB_HANDLERS: frozenset[str] = frozenset({"migrate.code_sha", "migrate.arm_filed_on"})


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
    "experiment.register": _todo(
        "experiment.register",
        "track A",
        "Derives the set from the release in force; never a hand-written list of arms.",
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
    "migrate.history": _migrate_history,
    # Not in `JOBS` (see the note beside its subparser in `build_parser`),
    # but `dest="job"` is shared across every subparser this module builds,
    # so `migrate.code_sha` reaches `main`'s `HANDLERS[args.job](args)`
    # dispatch the same way every real job does.
    "migrate.code_sha": _migrate_code_sha,
    # alpha-engine-config-I10948 — same shape, same reason (see NON_JOB_HANDLERS).
    "migrate.arm_filed_on": _migrate_arm_filed_on,
    # track-C handlers live in crucible/track_c.py so three tracks can land
    # code in parallel without editing one another's lines.
    HOLDOUT_JOB: holdout_handler,
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
    IAC_CONFORMANCE_JOB: iac_conformance_handler,
    INTEGRATION_TEST_JOB: integration_test_handler,
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
        if spec.name in (
            "experiment.run",
            "experiment.grade",
            "experiment.backfill",
            "promote",
            "experiment.new",
            "experiment.register",
        ):
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
        if spec.name in ("experiment.run", "experiment.new", "experiment.backfill"):
            # track-A: NOT required for `experiment.run`. Policy §3 scores every
            # registered arm every cycle, so the default is "all of them"; naming
            # one narrows the run to it, which is a debugging affordance rather
            # than the normal path. `experiment.new` still requires it, because
            # registering "whichever arms happen to be on disk" is not a
            # deliberate act — and `experiment.backfill` requires it for the
            # same reason one layer on: a backfill of "every arm, for two
            # years" is a bill and a blast radius nobody chose
            # (alpha-engine-config-I10696).
            sub.add_argument(
                "--arm",
                metavar="NAME|ARM_ID",
                required=spec.name in ("experiment.new", "experiment.backfill"),
                help=(
                    "Restrict to one arm, by bare name or by registered "
                    "`{slot}:{name}:{spec_hash}` id — `experiment.new` prints ids, so "
                    "the id is the form an operator is holding. Omitted, every "
                    "registered arm in the slot is produced — an arm that skipped a "
                    "cycle records a MISS, and a miss is data."
                ),
            )
        if spec.name == "explain":
            sub.add_argument("target", metavar="RUN_ID|VERDICT_KEY", nargs="?")
            sub.add_argument(
                "--verify-chain",
                action="store_true",
                help=(
                    "Verify the money-path hash chain (plan §9.5, "
                    f"{_money_path_chain_tracker()}) and exit non-zero on a break. A "
                    "no-op when the walk never crossed the money path."
                ),
            )
            # The scheduled Saturday arc stage's own flag; see
            # `crucible.weekly.SELECT_NEWEST_VERDICT_JOBS` and this file's own
            # tracker note there (alpha-engine-config-I10858) — never restated
            # in a runtime string per `tests/test_no_stale_tracker_literals.py`.
            sub.add_argument(
                "--select-newest-verdict",
                action="store_true",
                help=(
                    "Walk the newest SETTLED verdict.json across every slot/arm "
                    "(`crucible.explain.select_newest_settled_verdict`), deterministically "
                    "— for the scheduled Saturday arc stage, "
                    "never for an operator holding a specific target. Mutually exclusive "
                    "with the positional target; exactly one is required."
                ),
            )
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
                "--publish",
                action="store_true",
                # alpha-engine-config-I10492: `crucible gate` is the command
                # every session and every runbook uses to READ where a phase
                # stands, and it used to write `gates/{gate}/{day}/gate.json`
                # and `gates/ladder.json` on every non-`--dry-run` invocation
                # regardless — so a laptop read with a required var unset
                # clobbered a correct CI reading with a false negative. The
                # default is now the structural fix the issue names: no write
                # unless this flag says so, on top of `--dry-run`'s existing
                # (and unaffected) read-only guarantee. Nothing in this repo's
                # CI (`board.yml`, `gate-close.yml`) calls `crucible gate`
                # directly today, so no workflow needs this flag; a future
                # scheduled publisher passes it explicitly.
                help=(
                    "Write the dated gate reading, the ladder and (if due) the "
                    "phase's closing record to the store. Without it, `crucible gate` "
                    "only reads and reports — the default, since this command is the "
                    "one every session uses to check where a phase stands. Implies "
                    "nothing about --dry-run: --dry-run always wins."
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
        if spec.name == "migrate.history":
            sub.add_argument(
                "--strategy-dir",
                help=(
                    "Checkout of alpha-engine-config/strategy/. Absent, recipes are read "
                    "from the strategy tree published into the store."
                ),
            )
            sub.add_argument(
                "--v1-store",
                metavar="URI",
                help=(
                    "Store URI of the v1 artifacts, opened read-only. Absent, the data "
                    "bucket setting (CRUCIBLE_ARCTIC_BUCKET) is read at its root; neither "
                    "set, or a URI equal to --store, is refused."
                ),
            )
            sub.add_argument(
                "--allow-missing",
                action="store_true",
                help=(
                    "Import the sources that are present. Every absent source is named in "
                    "the result; without this flag an absent source fails the run."
                ),
            )
        if spec.name == HOLDOUT_JOB:
            add_holdout_arguments(sub)
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

    # `migrate.code_sha` (alpha-engine-config-I10626): a one-off REPAIR of
    # manifests another job already wrote, not a job in its own right, so it
    # is deliberately NOT in `JOBS` — it patches an existing `code_sha`
    # rather than writing a new `run_manifest.v2` document, and that schema's
    # `job` enum is closed (`crucible/models.py`, out of this change's
    # ownership). Wired here, by hand, alongside `JOBS`-driven dispatch
    # rather than through it: `tests/test_components_registry.py` derives its
    # checks from `JOBS`, and a row there would wrongly demand a
    # `components.yaml` entry and a schema-enum slot for a tool that writes
    # no manifest of its own. `--run-mode`/`--store`/`--dry-run` are repeated
    # here (not looped, since this parser is built outside the `JOBS` loop
    # above) so the invocation looks and behaves like every other `crucible`
    # command.
    # Tracker: alpha-engine-config-I10626 (cited here, not in the help
    # string itself — tests/test_no_stale_tracker_literals.py forbids a
    # hardcoded tracker reference in any non-docstring string).
    migrate_code_sha_help = (
        "One-off: derive and rewrite the all-zero code_sha placeholder on existing run manifests"
    )
    migrate_code_sha_sub = subparsers.add_parser(
        "migrate.code_sha", help=migrate_code_sha_help, description=migrate_code_sha_help
    )
    migrate_code_sha_sub.add_argument(
        "--run-mode",
        choices=list(RUN_MODES),
        default=None,
        help=(
            "Required by every `crucible` invocation (see `crucible.runmode`); unused by "
            "this repair, which writes no run-manifest-schema document of its own."
        ),
    )
    migrate_code_sha_sub.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be rewritten and refused; write nothing.",
    )
    migrate_code_sha_sub.add_argument(
        "--store",
        metavar="URI",
        help="Store root: an s3://bucket/prefix URI or a local directory path.",
    )

    # `migrate.arm_filed_on`: the same NON-job shape as `migrate.code_sha`
    # directly above, for the same reason — it corrects a field on rows
    # another job already wrote and writes no run-manifest document of its
    # own, so it is deliberately not in `JOBS` and has no `components.yaml`
    # row. Its durable record is the migration report it files under
    # `migrations/{trading_day}/{run_id}.json`, the same artifact
    # `migrate.code_sha` and `migrate.history` file.
    # Tracker: alpha-engine-config-I10948 (cited here, not in the help
    # string — tests/test_no_stale_tracker_literals.py forbids a hardcoded
    # tracker reference in any non-docstring string).
    migrate_arm_filed_on_help = (
        "One-off: correct the registered-event date on the arm-register rows that were "
        "stamped from the recipe's created_date instead of the day they were filed"
    )
    migrate_arm_filed_on_sub = subparsers.add_parser(
        "migrate.arm_filed_on",
        help=migrate_arm_filed_on_help,
        description=migrate_arm_filed_on_help,
    )
    migrate_arm_filed_on_sub.add_argument(
        "--run-mode",
        choices=list(RUN_MODES),
        default=None,
        help=(
            "Required by every `crucible` invocation (see `crucible.runmode`); unused by "
            "this repair, which writes no run-manifest-schema document of its own."
        ),
    )
    migrate_arm_filed_on_sub.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be corrected and refused; write nothing.",
    )
    migrate_arm_filed_on_sub.add_argument(
        "--store",
        metavar="URI",
        help="Store root: an s3://bucket/prefix URI or a local directory path.",
    )

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


#: The exit code an invocation the CLI REFUSED leaves behind, and the reason
#: it is not 1.
#:
#: The crucible-v2 box wrapper (`nous-ergon-ops`
#: `infrastructure/cloudformation/crucible-v2.yaml`, `_finish`) already
#: separates the two cases and has since `alpha-engine-config-I10134`: exit 2
#: pages "MALFORMED DISPATCH ... no run manifest was written by this attempt",
#: and any other non-zero exit pages "job exited $code ... **if no run manifest
#: exists the harness died before writing one**". Argparse's own errors exit 2
#: and land in the first branch correctly; every refusal this module raises by
#: hand exited 1 and landed in the second, so an operator who mistyped a flag
#: was told the harness had died.
#:
#: Measured 2026-09-09: `fault.probe --fault-capability-class chaos_probe`
#: paged the generic branch. The probe's manifest happened to exist, so the
#: page's speculation was merely wrong rather than misleading -- but the same
#: page is emitted for `--fault-capability-class typo`, which cannot write a
#: manifest at all because `crucible.llm.parse_fault_capability_class` refuses
#: it before a store is opened. That is a malformed dispatch by every property
#: the wrapper's own branch names, and it is now labelled as one.
USAGE_EXIT_CODE = 2


class UsageError(SystemExit):
    """An invocation this CLI refused, before any job ran.

    A `SystemExit` subclass carrying :data:`USAGE_EXIT_CODE`, so the message
    still reaches stderr and the process still exits non-zero -- the only
    thing that changes is which of the box wrapper's two page classes claims
    it.

    Deliberately NOT used for a job that ran and failed: the manifest is the
    record there, exit 1 is correct, and widening this to "any error the CLI
    can name" is how "no manifest was written" would start appearing on runs
    that wrote one.
    """

    def __init__(self, message: str) -> None:
        super().__init__(USAGE_EXIT_CODE)
        self.message = message

    def __str__(self) -> str:
        return self.message


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code.

    Exceptions from the JOB are NOT caught here. A job that failed must exit
    non-zero with its traceback intact -- the manifest already records the
    cause, and a tidy error message that returns 0 is a degraded-SUCCEEDED by
    another name.

    The three OPERATOR-INPUT validations below are different in kind: each
    runs before any handler, none of them can write a manifest, and each one
    firing means the dispatch was malformed rather than that anything in the
    system is wrong. They are collected into one block and re-raised as
    :class:`UsageError`, so they exit :data:`USAGE_EXIT_CODE` and are reported
    as what they are. Nothing is swallowed: the message is preserved verbatim
    and the original exception is chained.
    """
    args = build_parser().parse_args(argv)
    # The invocation's capture ledger (alpha-engine-config-I11012). Opened
    # here, around EVERY handler, so the report is the same whether the job
    # goes through `run_job` or returns from its own `--dry-run` branch
    # first, and so a raise on the way through still reports the writes the
    # rehearsal got as far as.
    #
    # `owns_ledger` is what makes `crucible weekly --dry-run` report once:
    # `weekly` re-enters this function per stage in the same process, and a
    # nested call sees the ledger already open, adds to it, and leaves both
    # the render and the reset to the outermost call.
    from crucible.llm import begin_provider_capture, end_provider_capture
    from crucible.store import active_capture_ledger, begin_capture, end_capture

    dry_run = bool(getattr(args, "dry_run", False))
    owns_ledger = dry_run and active_capture_ledger() is None
    ledger = begin_capture() if dry_run else None
    # The EGRESS ledger, opened beside the store ledger and owned by the same
    # caller (alpha-engine-config-I11012). A rehearsal makes two promises —
    # it writes nothing and it reaches no provider — and both are reported,
    # because a promise with no surface is one nobody can check.
    provider_ledger = begin_provider_capture() if dry_run else None
    try:
        return _resolve_operator_input(args)
    except UsageError as exc:
        # Root cause (`alpha-engine-config-I10517`): `UsageError` is a
        # `SystemExit` subclass constructed with an INT `.code`. Python's own
        # top-level handling of an uncaught `SystemExit` only prints
        # something when `.code` is a *string* -- an int code (which this
        # class always carries) exits silently, so `exc.message`/`__str__`
        # were never reaching stderr on the installed console script. `main`
        # is the entry point named in `pyproject.toml`
        # (`crucible = "crucible.cli:main"`), so catching and printing here,
        # rather than relying on whatever wraps `main`, fixes every caller at
        # once instead of only `python -m crucible.cli`.
        print(str(exc), file=sys.stderr)
        return USAGE_EXIT_CODE
    finally:
        if owns_ledger:
            # Rendered on the failure path too, deliberately: a rehearsal
            # that got three keys in and then raised is telling the operator
            # both things, and printing only on success would hide the half
            # the failing command most needs to explain. That applies twice
            # over to the provider ledger, whose whole normal shape is
            # "recorded one call, then stopped".
            print(ledger.render())
            print(provider_ledger.render())
            end_capture()
            end_provider_capture()


def _resolve_operator_input(args: argparse.Namespace) -> int:
    """The OPERATOR-INPUT validations, resolved before any handler runs.

    Split out of :func:`main` so `main` itself can be the single place that
    converts a raised :class:`UsageError` into stderr output plus an exit
    code -- see `main`'s own `except UsageError` clause.
    """
    try:
        args.trading_day = resolve_date(getattr(args, "date", None))
        # Resolved once, here, so every handler passes the SAME value to
        # `run_job` and a `--run-mode` typo is a usage error before any job
        # starts. Resolved even for the handlers that never write a manifest:
        # an invocation is live or a replay regardless of what it produces.
        args.run_mode = resolve_run_mode(getattr(args, "run_mode", None))
        # Moved here from `crucible.fault_probe.fault_probe_handler` so it
        # sits with the other two: its own docstring says it is "pure and
        # offline, so a bad invocation is a usage error before a store, a
        # trading day or a provider is touched", and this is where the CLI's
        # usage errors are. The handler reads the validated value off `args`.
        raw_class = getattr(args, "fault_capability_class", None)
        if raw_class is not None:
            from crucible.llm import parse_fault_capability_class

            args.fault_capability_class = parse_fault_capability_class(raw_class)
        # alpha-engine-config-I10492: `gate` and `gate.close` both live-evaluate
        # every registered gate (`crucible.gate.build_ladder`), and two clauses
        # need environment this tree deliberately carries no default for
        # (`crucible.gate.GATE_REQUIRED_ENV`). A missing one used to fold into
        # that clause's own UNMEASURABLE reading — indistinguishable from a
        # real read failure — and get WRITTEN to the shared gate artifact,
        # clobbering a correct CI reading. Refused here, before either job's
        # `run_job` opens a store or attempts a read, exactly like the other
        # operator-input validations in this block.
        if args.job in ("gate", track_f.GATE_CLOSE_JOB):
            missing = missing_required_env()
            if missing:
                # Not cited by number in this message: `tests/test_no_stale_
                # tracker_literals.py` forbids a hardcoded tracker literal
                # outside a docstring/comment (`alpha-engine-config-I9839`) —
                # see the comment above for the issue this refusal exists for.
                raise UsageError(
                    f"crucible {args.job} refuses to read: "
                    + ", ".join(missing)
                    + " unset. A missing required variable is a refusal, never an "
                    "UNMEASURABLE clause — export it per crucible/AGENTS.md's 'reading "
                    "a gate from the laptop' recipe before re-running."
                )
    except UsageError:
        raise
    except SystemExit as exc:
        raise UsageError(str(exc)) from exc
    except (RunModeError, ValueError) as exc:
        raise UsageError(str(exc)) from exc
    return HANDLERS[args.job](args)


if __name__ == "__main__":
    sys.exit(main())
