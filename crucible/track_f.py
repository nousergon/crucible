"""Track F's handlers: `crucible weekly` and `crucible gate`.

Its own module rather than a branch inside another track's, so tracks land
code in the same release without editing one another's lines. `cli.py` carries
one line naming each.

Two jobs, and they are the two halves of one claim. `weekly` RUNS the declared
arc — the six components `components.yaml` marks `dispatch: arc`, in the order
their own deadlines imply. `gate` READS what the arc produced and reports,
clause by clause, whether a phase may exit. Neither does the other's work: a
gate that ran the thing it grades could not distinguish a run made under gate
conditions from a run made in production, and a driver that graded itself is
the shape that closed phase 1 on 2026-09-01 with zero replays performed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from typing import Any

from krepis.metrics import derive_status

from crucible import gate as gate_module
from crucible.gate import (
    GATES,
    LADDER_KEY,
    LADDER_SCHEMA_VERSION,
    build_ladder,
    evaluate,
    gate_key,
    ladder_payload,
)
from crucible.keys import manifest_key
from crucible.runner import RunContext, run_job
from crucible.store import open_store
from crucible.weekly import arc_stages, run_arc

__all__ = ["gate_handler", "weekly_handler"]


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
    """`crucible gate --gate phase1 [--weeks N] [--date YYYY-MM-DD]`.

    **The job succeeds when the MEASUREMENT succeeds; the PROCESS exits
    non-zero when the gate is not met.** The two are different facts and
    collapsing them would make a store outage look like an unmet phase — or,
    worse, make an unmet phase look like a broken run that someone reruns
    until it passes. The manifest records `ok` and the reading; the exit code
    is what a caller branches on.
    """
    dry_run = bool(getattr(args, "dry_run", False))
    store = open_store(getattr(args, "store", None), dry_run=dry_run)
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
        # alpha-engine-config-I9922 R2-1: the store guard (dry_run-wrapped
        # `store` above) is the backstop, not the primary path — `gate` has a
        # natural report (`reading.render()`/`ladder.render()`, printed
        # below), so under `--dry-run` it skips the write and reaches that
        # print rather than dying on the guard before it ever gets there.
        if not dry_run:
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
        if not dry_run:
            ctx.record_output(
                LADDER_KEY, ladder_payload(ladder), schema_version=LADDER_SCHEMA_VERSION
            )
        result["ladder"] = ladder
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
    return 0 if reading.met else 1


def gate_names() -> list[str]:
    """The registered gates, for the CLI's `--gate` choices."""
    return sorted(GATES)
