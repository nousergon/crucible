"""Track F's handlers: `crucible weekly`, `crucible gate` and `crucible gate.close`.

Its own module rather than a branch inside another track's, so tracks land
code in the same release without editing one another's lines. `cli.py` carries
one line naming each.

Two of them are the two halves of one claim. `weekly` RUNS the declared arc —
the six components `components.yaml` marks `dispatch: arc`, in the order their
own deadlines imply. `gate` READS what the arc produced and reports, clause by
clause, whether a phase may exit. Neither does the other's work: a gate that
ran the thing it grades could not distinguish a run made under gate conditions
from a run made in production, and a driver that graded itself is the shape
that closed phase 1 on 2026-09-01 with zero replays performed.

`gate.close` is the third, and it exists because the FILING half of that loop
had no cadence (`alpha-engine-config-I10095`). `crucible-PR121` made a phase's
exit a durable record — the first live gate reading that reads MET posts the
reading to the phase's tracker issue and files `gates/{phase}/closing.json`,
once, by compare-and-swap — but nothing ran `crucible gate` on a schedule, so
the record was written only when a human or an agent happened to run it. The
daily board DETECTS the gap (a phase issue closed with no record renders red)
and deliberately cannot close it: the board identity is read-only over
everything it grades, because a grading surface that could satisfy the clauses
it grades is not a measurement. So the filing runs as its own daily job under
its own writer identity, the gate-close role, which may write
exactly the closing records and its own manifest and may write no gate reading
at all.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from krepis.metrics import derive_status

from crucible import gate as gate_module
from crucible import tracker
from crucible.documents import UnreadableDocumentError, read_store_document
from crucible.gate import (
    CLOSING_READING_SCHEMA_VERSION,
    GATES,
    LADDER_KEY,
    LADDER_SCHEMA_VERSION,
    PHASES,
    TRACKER_REPO,
    Phase,
    build_ladder,
    closing_reading,
    evaluate,
    gate_key,
    gate_state_for,
    ladder_payload,
    parse_closing_comment,
    phase_for_gate,
    render_closing_comment,
)
from crucible.keys import closing_record_key, manifest_key
from crucible.runmode import RUN_MODE_LIVE
from crucible.runner import RunContext, run_job
from crucible.store import ETAG_ABSENT, Store, open_store, resolve_store_uri
from crucible.weekly import arc_stages, run_arc

__all__ = [
    "CLOSE_OUTCOMES",
    "GATE_CLOSE_JOB",
    "closing_record_line",
    "file_closing_record",
    "gate_close_handler",
    "gate_handler",
    "post_closing_comment",
    "reading_commit",
    "weekly_handler",
]


def reading_commit() -> str:
    """The commit of the crucible tree taking a gate reading.

    Three sources, most authoritative first, and a raise if none answers:

    1. ``$GITHUB_SHA`` — what Actions sets, so a reading taken by the board or
       by any workflow carries the sha of the checkout that took it without
       anybody wiring it.
    2. ``$CRUCIBLE_COMMIT`` — the explicit override, for a context that knows
       its commit and is not a checkout (a spot box running an installed
       wheel, whose commit is the one its release was built from).
    3. ``git rev-parse HEAD`` in the working tree the package is imported
       from — the laptop case.

    It RAISES rather than returning a placeholder. A closing reading exists so
    a later reader can re-run the same clause definitions; ``"unknown"`` in
    that field would be a block that looks complete and cannot be checked,
    which is the shape of overclaim this whole mechanism removes.
    """
    for name in ("GITHUB_SHA", "CRUCIBLE_COMMIT"):
        value = (os.environ.get(name) or "").strip().lower()
        if value:
            return value
    root = Path(__file__).resolve().parent.parent
    # Fixed argv, no shell, no caller-supplied component.
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    value = completed.stdout.strip().lower()
    if completed.returncode == 0 and value:
        return value
    raise RuntimeError(
        "cannot determine the commit this reading was taken at: $GITHUB_SHA and "
        f"$CRUCIBLE_COMMIT are unset and `git -C {root} rev-parse HEAD` failed "
        f"({completed.stderr.strip() or 'no output'}). Set CRUCIBLE_COMMIT to the sha "
        "this build came from. A closing reading with no commit cannot be re-run "
        "against the clause definitions that produced it."
    )


def post_closing_comment(phase: Phase, body: str) -> str:
    """Post ``body`` to ``phase``'s tracker issue, at most once, ever.

    Idempotent AT THE TRACKER, not only in this process: the comments already
    on the issue are read first, and a comment already carrying a
    `crucible-gate-reading` block rendered for THIS phase means the record has
    been posted and its URL is returned unchanged. That check is what makes
    the pair of writes safe in either order — the store record can be lost, a
    race can be lost, this job can be re-run by hand, and the issue still ends
    up with exactly one closing reading on it.

    It comments and it stops. Closing the issue is Brian's authority
    (`alpha-engine-config-I9967` deliverable 1 says so in as many words), and
    `crucible.tracker` cannot construct any other mutating request.

    Raises `crucible.tracker.TrackerError` when the tracker cannot be reached
    or the credential is not granted. Deliberately loud: this function is
    reached only when a phase gate reads MET for the first time on a live run,
    which happens six times in the life of the rebuild, and a record filed to
    the store while the tracker was never told is the same two-instruments
    defect one instrument along.
    """
    for existing in tracker.comment_bodies(TRACKER_REPO, phase.issue):
        block = parse_closing_comment(existing)
        if block is not None and block.get("phase") == phase.id:
            return phase.tracker_url
    return tracker.post_comment(TRACKER_REPO, phase.issue, body)


def file_closing_record(
    ctx: RunContext,
    store: Store,
    reading: Any,
    *,
    store_uri: str,
) -> str | None:
    """File the ONE durable record of a phase's exit, or ``None`` if not due.

    `alpha-engine-config-I9967` deliverable 2, and the whole design constraint
    in one function: **the phase's recorded state is DERIVED from the gate
    reading**, written where a later reader can fetch it, and it is not a
    convention anybody has to remember.

    Three conditions, all of them facts rather than judgements:

    1. the gate reads `MET` — `gate_state_for`, the same derivation the ladder
       row and the closing block use, so the three cannot disagree;
    2. the run is LIVE — a replay of a historical day may legitimately read
       MET and does not exit a phase (`crucible.runmode`, plan §6 row 2);
    3. nothing is filed at the key yet — compare-and-swap against
       :data:`~crucible.store.ETAG_ABSENT`, so the first write wins and every
       later reading leaves it exactly as it was. A phase exits once.

    A record that is PRESENT and unreadable raises rather than being
    overwritten: "the record is corrupt" and "there is no record" call for
    opposite actions, and quietly replacing the first with a fresh reading
    would destroy the only evidence of what was actually filed.

    The tracker comment is posted BEFORE the store write, on purpose. If the
    post fails the record is not written, the job fails loud, and re-running
    files both; if the write fails the comment is already on the issue and
    :func:`post_closing_comment` will not repeat it. The reverse order has a
    state — recorded but never announced — that nothing would ever correct.
    """
    if gate_state_for(reading) != "MET":
        return None
    if ctx.run_mode != RUN_MODE_LIVE:
        return None
    phase = phase_for_gate(reading.gate)
    key = closing_record_key(phase.id)
    filed = read_store_document(store, key)
    if filed.problem is not None:
        raise UnreadableDocumentError(
            f"{filed.problem}. A closing record that is present and unreadable is not an "
            "absent one, and this run will not overwrite it with a fresh reading — that "
            "would destroy the only evidence of what was filed when the phase exited."
        )
    if not filed.absent:
        return key
    document = closing_reading(reading, store_uri=store_uri, commit=reading_commit())
    post_closing_comment(phase, render_closing_comment(document))
    ctx.record_output_cas(
        key,
        ETAG_ABSENT,
        json.dumps(document, indent=2, sort_keys=True).encode("utf-8"),
        schema_version=CLOSING_READING_SCHEMA_VERSION,
    )
    return key


def closing_record_line(store: Store, reading: Any) -> str:
    """The one line a `crucible gate` reader gets about the closing record.

    Always a line, never silence. `alpha-engine-config-I9967` deliverable 2
    asks the gate to render "closing record filed at <key>" once the record
    exists; the other three answers are printed with equal prominence, because
    a surface that prints something only in the good case teaches its reader
    that no line means nothing to see.
    """
    phase = phase_for_gate(reading.gate)
    key = closing_record_key(phase.id)
    filed = read_store_document(store, key)
    if filed.document is not None:
        return (
            f"closing record filed at {key} — {phase.tracker} exited on trading day "
            f"{filed.document.get('trading_day')}, {filed.document.get('gate_state')} at "
            f"commit {filed.document.get('commit')}"
        )
    if filed.problem is not None:
        return f"closing record at {key} could not be read: {filed.problem}"
    if gate_state_for(reading) == "MET":
        return (
            f"no closing record at {key}, and this gate reads MET. Re-run this command "
            f"with --run-mode {RUN_MODE_LIVE} and without --dry-run to file it and post "
            f"the reading to {phase.tracker}."
        )
    return f"no closing record at {key} — {phase.tracker} has not exited"


def weekly_handler(args: argparse.Namespace) -> int:
    """`crucible weekly [--date YYYY-MM-DD]` — run the arc for one trading day.

    The arc's own manifest is written like every other job's, and it records
    each stage's manifest as an INPUT. That is real lineage rather than a log
    line: `crucible explain` walks from the arc to the stage that failed, and
    the arc's `rows_out` is the number of stages that completed — so a run
    that stopped at stage two is legible as such without opening anything.

    A stage failure propagates. `run_job` writes the arc's `failed` manifest in
    its `finally`, naming the stage in `reason`, and the process exits
    non-zero.
    """
    store_uri = getattr(args, "store", None)
    dry_run = bool(getattr(args, "dry_run", False))
    store = open_store(store_uri, dry_run=dry_run)

    def body(ctx: RunContext) -> None:
        planned = arc_stages(ctx.trading_day)
        # `dry_run` is passed to `run_arc` so every stage's own argv carries
        # `--dry-run` too (alpha-engine-config-I9922 N1) — each stage is a
        # fresh `crucible.cli.main` invocation with no shared `args`, so
        # `weekly --dry-run` previously ran every stage for real.
        #
        # `run_mode=ctx.run_mode`, not `args` and not the environment: the
        # arc's mode is whatever `run_job` resolved for THIS invocation, and
        # every stage is run under that one answer. A stage left to
        # re-resolve would let the arc manifest and its stage manifests
        # disagree about the same week.
        ran = run_arc(ctx.trading_day, store=store_uri, dry_run=dry_run, run_mode=ctx.run_mode)
        for stage in ran:
            key = manifest_key(stage.job, ctx.trading_day.isoformat(), discriminator=stage.slot)
            ctx.record_input(key, store.get_bytes(key))
        ctx.record_rows(rows_in=len(planned), rows_out=len(ran))
        ctx.record_metric(
            {
                "name": "arc_stages_completed",
                "module": "crucible.weekly",
                "metric_type": "gauge",
                "value": float(len(ran)),
                "unit": "count",
                "n_floor": 1,
                "status": "OK" if len(ran) == len(planned) else "FAIL",
                "status_reason": f"{len(ran)} of {len(planned)} declared arc stages completed",
                "source_path": manifest_key("weekly", ctx.trading_day.isoformat()),
                "last_updated_utc": ctx.started.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )

    # `transient_retry=False`, deliberately. Every stage already runs through
    # its own `run_job` and owns the declared one-retry ladder (§11 risk 2), so
    # an arc-level retry would re-run `data.weekly` through `report` because
    # `console` met a 5xx — twelve jobs repeated for one, each rewriting its
    # own manifest, and the store's answer to "did this week work" decided by
    # write ordering.
    run_job(
        "weekly",
        body,
        store=store,
        trading_day=args.trading_day,
        transient_retry=False,
        dry_run=dry_run,
        run_mode=getattr(args, "run_mode", None),
    )
    return 0


def gate_handler(args: argparse.Namespace) -> int:
    """`crucible gate --gate phase1 [--weeks N] [--date YYYY-MM-DD] [--publish]`.

    **The job succeeds when the MEASUREMENT succeeds; the PROCESS exits
    non-zero when the gate is not met.** The two are different facts and
    collapsing them would make a store outage look like an unmet phase — or,
    worse, make an unmet phase look like a broken run that someone reruns
    until it passes. The manifest records `ok` and the reading; the exit code
    is what a caller branches on.

    **`--publish` gates the shared-artifact writes; it does not exist without
    also reading.** `alpha-engine-config-I10492`: this command is the one
    every session and every runbook uses to READ where a phase stands, and it
    used to write `gates/{gate}/{day}/gate.json` and `gates/ladder.json` on
    every non-`--dry-run` invocation regardless of who ran it or why — so a
    laptop read missing a required var (`crucible.gate.missing_required_env`
    refuses that case separately, before this function is even called) could
    clobber a correct CI reading with a false one. `publish` below is `False`
    unless `--publish` was passed, and `--dry-run` always overrides it: a run
    is never both. This job's OWN manifest at `runs/gate/{day}/run.json`
    still writes on every non-`--dry-run` invocation regardless of `publish`
    — rule 1 is unconditional; only the SHARED artifacts this job is not the
    sole owner of are behind the flag.
    """
    dry_run = bool(getattr(args, "dry_run", False))
    publish = bool(getattr(args, "publish", False)) and not dry_run
    store = open_store(getattr(args, "store", None), dry_run=dry_run)
    store_uri = resolve_store_uri(getattr(args, "store", None))
    result: dict[str, Any] = {}

    def body(ctx: RunContext) -> None:
        reading = evaluate(
            store, gate=args.gate, trading_day=ctx.trading_day, weeks=getattr(args, "weeks", None)
        )
        result["reading"] = reading
        document = reading.to_dict()
        document["run_id"] = ctx.run_id
        payload = json.dumps(document, indent=2, sort_keys=True).encode("utf-8")
        key = gate_key(reading.gate, ctx.trading_day.isoformat())
        # alpha-engine-config-I10492: the shared gate artifact is written only
        # when `--publish` was passed (and `--dry-run` was not) — see the
        # docstring above. `--dry-run`'s own store guard
        # (alpha-engine-config-I9922 R2-1) remains the backstop under
        # `--dry-run` specifically: a stray write that slipped past `publish`
        # would still hit `DryRunWriteRefusedError` there rather than
        # succeed silently.
        if publish:
            ctx.record_output(key, payload)
        for clause in reading.clauses:
            for evidence in clause.evidence:
                # Guarded, not a bare `exists`/`get_bytes` pair: a clause can
                # already read UNMEASURABLE over this exact key (an access
                # failure `gate.py` itself caught and turned into a red
                # reading), and re-reading it here unguarded raised the
                # `PermissionError` straight out of this job body — the
                # gate's own careful non-raising read, undone one call later
                # by its own lineage recorder (`alpha-engine-config-I9869`
                # round 3, finding 1). An evidence key this job cannot read
                # is simply not recorded as an input; the clause reading
                # itself already carries the fact, on the artifact this job
                # writes regardless.
                read = gate_module._read_store_bytes(store, evidence)
                if read.problem is not None or read.absent or read.raw is None:
                    continue
                ctx.record_input(evidence, read.raw)
        # The phase LADDER, republished on every gate read. The gate above
        # answers "is phase N met"; the ladder answers "which phase is the
        # rebuild on, and is any phase being graded ahead of an earlier one" —
        # the question that lived only in `alpha-engine-config-I9757`'s issue
        # comments, was written by hand, and was wrong twice. Written here
        # rather than by a second command because a surface refreshed by a
        # step somebody has to remember is the defect one layer along; the
        # weekly `console` arc stage republishes it on a cadence as well.
        ladder = build_ladder(
            store,
            trading_day=ctx.trading_day,
            now=ctx.started,
            readings={reading.gate: reading},
        )
        if publish:
            ctx.record_output(
                LADDER_KEY, ladder_payload(ladder), schema_version=LADDER_SCHEMA_VERSION
            )
        result["ladder"] = ladder
        # The closing record, derived from the reading above and written at
        # most once (`alpha-engine-config-I9967` deliverable 2). Inside the
        # job body, so the record enters `outputs[]` as lineage and
        # `crucible explain` can name the run that filed it; behind `publish`
        # for the same reason the gate artifact is (`alpha-engine-config-
        # I10492`), and the printed line below still reports whether one
        # exists.
        if publish:
            file_closing_record(ctx, store, reading, store_uri=store_uri)
        ctx.record_rows(rows_in=len(reading.window), rows_out=len(reading.clauses))
        n_clauses = len(reading.clauses)
        met_count = sum(1 for c in reading.clauses if c.met)
        # `reading.met_ratio` is `None` for two DISTINCT reasons
        # (`crucible.gate.GateResult.met_ratio`, alpha-engine-config-I9824,
        # widened round 3 of I9869) and this job's own outcome metric must
        # tell them apart rather than publish one wording for both:
        #
        # 1. Zero clauses registered — nothing to measure. `derive_status`
        #    reads this from `n_samples < 0.5 * n_floor` (n_clauses == 0).
        # 2. N of M clauses registered but one or more could not be READ (a
        #    store access failure, not a fact about the system graded) —
        #    `n_clauses` is the true registered count, so `derive_status`
        #    would read it as measured and pick GREEN/WATCH/RED off a `None`
        #    value. `input_present=False` routes it to `N/A-MISSING-INPUT`
        #    instead — the vocabulary already has a state for exactly this,
        #    round 4 finding 2: it must not be re-typed as "no clauses
        #    registered, nothing measured" when clauses ARE registered, nor
        #    invented as a new status outside the existing taxonomy.
        if reading.met_ratio is None and n_clauses == 0:
            status = derive_status(value=None, n_samples=n_clauses, n_floor=1)
            status_reason = f"gate {reading.gate}: no clauses registered, nothing measured"
        elif reading.met_ratio is None:
            unmeasurable_names = [c.name for c in reading.clauses if c.unmeasurable]
            status = derive_status(value=None, n_samples=n_clauses, n_floor=1, input_present=False)
            status_reason = (
                f"gate {reading.gate}: {len(unmeasurable_names)} of {n_clauses} clauses "
                f"unmeasurable: {', '.join(unmeasurable_names)}"
            )
        else:
            status = "OK" if reading.met else "FAIL"
            status_reason = f"gate {reading.gate}: {met_count}/{n_clauses} clauses met"
        ctx.record_metric(
            {
                "name": "gate_clauses_met_ratio",
                "module": "crucible.gate",
                "metric_type": "gauge",
                "value": reading.met_ratio,
                "unit": "ratio",
                "n_floor": 1,
                "n_samples": n_clauses,
                "status": status,
                "status_reason": status_reason,
                "source_path": key,
                "last_updated_utc": ctx.started.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )

    # `gate` never checked `--dry-run` (alpha-engine-config-I9922 N1 — measured
    # writing `gates/ladder.json` and `gates/phase1/<day>/gate.json`). The
    # read-only `store` above turns those `ctx.record_output` calls into a
    # loud `DryRunWriteRefusedError`; `dry_run=` here keeps `run_job` from
    # also attempting its own manifest write on top of that.
    run_job(
        "gate",
        body,
        store=store,
        trading_day=args.trading_day,
        dry_run=dry_run,
        run_mode=getattr(args, "run_mode", None),
    )
    reading = result["reading"]
    print(reading.render())
    print(result["ladder"].render())
    print(closing_record_line(store, reading))
    # `alpha-engine-config-I9967` deliverable 3. Printed AFTER the reading and
    # the ladder, and only when asked for: the block is a paste target, not a
    # second rendering everyone reads past. It is emitted for an UNMET gate as
    # readily as a MET one — the honest block on a phase that may not close is
    # the one that says so, and `closing_reading_refusals` is what turns it
    # into a refusal rather than the renderer declining to render.
    if getattr(args, "closing_comment", False):
        print()
        print(
            render_closing_comment(
                closing_reading(reading, store_uri=store_uri, commit=reading_commit())
            )
        )
    return 0 if reading.met else 1


#: The job name, in one place. `cli.py`, `components.yaml`'s row and the
#: manifest schema's `job` enum are the three files
#: `tests/test_components_registry.py` holds in lockstep, and the CLI table
#: takes this constant rather than restating the string.
GATE_CLOSE_JOB = "gate.close"

#: What `gate.close` did about one phase, as a CLOSED vocabulary. Every
#: registered phase gets exactly one of these on every run, so "this phase was
#: not touched" is a stated outcome carrying its reason rather than a line
#: nobody wrote. A seventh value is a design change, not a config value.
CLOSE_OUTCOMES: tuple[str, ...] = (
    "filed",
    "already_filed",
    "not_met",
    "unmeasurable",
    "replay",
    "would_file",
    "no_gate_registered",
)

#: The two outcomes that mean a durable record exists for the phase after this
#: run — the value `closing_record_state_{phase}` reads 1.0 for.
_RECORD_PRESENT = frozenset({"filed", "already_filed"})


def _close_one_phase(
    ctx: RunContext,
    store: Store,
    phase: Phase,
    *,
    store_uri: str,
    dry_run: bool,
) -> tuple[str, str]:
    """Read ``phase``'s gate and file its closing record if one is due.

    Returns ``(outcome, detail)`` — one member of :data:`CLOSE_OUTCOMES` and
    the sentence a reader needs to act on it. It never returns a bare boolean:
    "nothing was filed" has five distinct causes here and they call for
    different actions, which is the whole reason the vocabulary is closed.

    :func:`file_closing_record` remains the ONE writer. This function decides
    nothing about MET-ness, liveness or the compare-and-swap — it re-derives
    the outcome from the same three facts so the manifest can name which one
    applied, and hands the write to the function `crucible gate` already uses.
    A second implementation of the filing rule here is exactly the drift the
    single-writer shape exists to prevent.
    """
    if phase.gate is None:
        return (
            "no_gate_registered",
            f"{phase.tracker} has no registered gate, so there is no reading to file. "
            "It renders UNMEASURED on the ladder; a gate is written for it, never "
            "inferred.",
        )
    reading = evaluate(store, gate=phase.gate, trading_day=ctx.trading_day)
    state = gate_state_for(reading)
    key = closing_record_key(phase.id)
    if state == "UNMEASURABLE":
        names = ", ".join(c.name for c in reading.clauses if c.unmeasurable)
        return (
            "unmeasurable",
            f"gate {phase.gate} could not be read: {names}. An unreadable clause is not "
            "a phase falling short, and nothing is filed on one.",
        )
    if state != "MET":
        met = sum(1 for c in reading.clauses if c.met)
        return (
            "not_met",
            f"gate {phase.gate} reads {met}/{len(reading.clauses)} clauses met; "
            f"{phase.tracker} has not exited.",
        )
    if ctx.run_mode != RUN_MODE_LIVE:
        return (
            "replay",
            f"gate {phase.gate} reads MET on a {ctx.run_mode} of trading day "
            f"{ctx.trading_day.isoformat()}. A replay of a historical day does not exit "
            "a phase, so nothing is filed and nothing is posted to the tracker.",
        )
    if dry_run:
        # Returned BEFORE the record is read, and before `file_closing_record`
        # is reached at all. The store's read-only guard would refuse the
        # write, but the tracker comment is posted FIRST and travels over a
        # different wire — a run asked to change nothing would have left a real
        # comment on a real issue before the guard ever fired.
        return (
            "would_file",
            f"gate {phase.gate} reads MET. Re-run without --dry-run to post the reading "
            f"to {phase.tracker} and file its closing record at {key}; --dry-run neither "
            "writes nor comments, so this run says only that one is due.",
        )
    already = read_store_document(store, key)
    # `already.problem` is deliberately NOT branched on here: a present but
    # unreadable record is `file_closing_record`'s refusal to make, and it
    # raises `UnreadableDocumentError` on the very next line rather than being
    # overwritten. Reading it twice is cheap; owning the refusal twice is how
    # the two copies drift.
    file_closing_record(ctx, store, reading, store_uri=store_uri)
    if already.absent:
        return ("filed", f"closing record filed at {key} and the reading posted to {phase.tracker}")
    return (
        "already_filed",
        f"closing record already at {key}; a phase exits once and this run left it "
        "exactly as it was.",
    )


def gate_close_handler(args: argparse.Namespace) -> int:
    """`crucible gate.close [--date YYYY-MM-DD]` — file every closing record due.

    `alpha-engine-config-I10095`. The DETECTION half of the phase-exit loop was
    closed by the daily board and the FILING half was not: a phase could read
    MET for days with the record written only when somebody happened to run
    `crucible gate`. This is that filing on a cadence — every registered phase,
    every day, one manifest.

    **It writes closing records and nothing else.** It does not write the dated
    gate readings `crucible gate` writes, it does not render the ladder, and
    its identity cannot: the gate-close role grants
    `crucible/gates/*/closing.json` and its own manifest prefix, so a bug that
    tried to publish a gate reading from this job is an AccessDenied rather
    than a grading surface quietly authoring what it grades.

    **It exits 0 whenever the MEASUREMENT succeeded**, exactly as `gate` does,
    and for the same reason one layer along: a phase that has not exited is not
    a failed run, and a non-zero exit here would page daily on a working
    producer until the channel was muted. What went wrong reaches a reader
    through `status: failed` on the manifest — a store outage, a denied
    tracker, an unreadable record all raise.
    """
    dry_run = bool(getattr(args, "dry_run", False))
    store = open_store(getattr(args, "store", None), dry_run=dry_run)
    store_uri = resolve_store_uri(getattr(args, "store", None))
    lines: list[str] = []

    def body(ctx: RunContext) -> None:
        outcomes: dict[str, tuple[str, str]] = {}
        for phase in PHASES:
            outcomes[phase.id] = _close_one_phase(
                ctx, store, phase, store_uri=store_uri, dry_run=dry_run
            )
        now = ctx.started.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        source = manifest_key(GATE_CLOSE_JOB, ctx.trading_day.isoformat())
        filed = [p for p, (outcome, _) in outcomes.items() if outcome == "filed"]
        # A phase reading MET whose record is still absent AFTER this run. On a
        # live run that is zero by construction; it is non-zero on a replay and
        # under --dry-run, and those are the cases the number exists to name.
        # Principle 7: the figure that says this job is working, on the surface
        # it appears on, with `no data` never rendered as green.
        pending = ("replay", "would_file")
        unclosed = [p for p, (outcome, _) in outcomes.items() if outcome in pending]
        for phase in PHASES:
            outcome, detail = outcomes[phase.id]
            lines.append(f"{phase.id} ({phase.tracker}): {outcome} — {detail}")
            ctx.record_metric(
                {
                    "name": f"closing_record_state_{phase.id}",
                    "module": "crucible.track_f",
                    "metric_type": "gauge",
                    "value": 1.0 if outcome in _RECORD_PRESENT else 0.0,
                    "unit": "count",
                    "n_floor": 1,
                    "n_samples": 1,
                    "status": "OK",
                    "status_reason": f"{phase.tracker}: {outcome} — {detail}",
                    "source_path": closing_record_key(phase.id),
                    "last_updated_utc": now,
                }
            )
        ctx.record_metric(
            {
                "name": "closing_records_filed",
                "module": "crucible.track_f",
                "metric_type": "gauge",
                "value": float(len(filed)),
                "unit": "count",
                "n_floor": 1,
                "n_samples": len(PHASES),
                "status": "OK",
                "status_reason": (
                    f"{len(filed)} of {len(PHASES)} registered phases had a closing record "
                    f"filed by this run: "
                    + "; ".join(f"{p}={outcomes[p][0]}" for p in sorted(outcomes))
                ),
                "source_path": source,
                "last_updated_utc": now,
            }
        )
        ctx.record_metric(
            {
                "name": "phases_met_without_a_record",
                "module": "crucible.track_f",
                "metric_type": "gauge",
                "value": float(len(unclosed)),
                "unit": "count",
                "n_floor": 1,
                "n_samples": len(PHASES),
                "status": "OK" if not unclosed else "FAIL",
                "status_reason": (
                    "every phase reading MET has a closing record"
                    if not unclosed
                    else f"{sorted(unclosed)} read MET with no record filed: "
                    + "; ".join(f"{p}={outcomes[p][1]}" for p in sorted(unclosed))
                ),
                "source_path": source,
                "last_updated_utc": now,
            }
        )
        ctx.record_rows(rows_in=len(PHASES), rows_out=len(filed))

    # `transient_retry` left at its default. A re-run of this job is safe by
    # construction — the compare-and-swap and `post_closing_comment`'s read of
    # the issue's existing comments make a second attempt file nothing and post
    # nothing — which is the property that lets it run daily at all.
    run_job(
        GATE_CLOSE_JOB,
        body,
        store=store,
        trading_day=args.trading_day,
        dry_run=dry_run,
        run_mode=getattr(args, "run_mode", None),
    )
    for line in lines:
        print(line)
    return 0


def gate_names() -> list[str]:
    """The registered gates, for the CLI's `--gate` choices."""
    return sorted(GATES)
