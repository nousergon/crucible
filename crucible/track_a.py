"""Track-A CLI handlers: the data, feature, U/R, ledger, explain and migrate jobs.

Normative source: plan §4.1. `crucible/cli.py` owns the argument surface and
the dispatch table; the handler bodies live here so three tracks can land
job implementations without three sets of edits to one file.

**Every handler goes through `crucible.runner.run_job`.** That is not style:
`run_job` is where the manifest guarantee lives, and a job invoked around it
produces no telemetry on precisely the path where telemetry matters — the
one where it failed.

**A handler resolves configuration and then does nothing clever.** The store
URI, the strategy tree and the ArcticDB bucket come from
`crucible.config.settings`, which records where each value came from, so
`explain` can report why a run read what it read.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from typing import Any

from crucible import migrate as migrate_module
from crucible.calendar import is_trading_day
from crucible.config import settings as resolve_settings
from crucible.data import ArcticPriceSource, PriceSource, run_daily, run_heal, run_weekly
from crucible.data.universe import DeclaredUniverse, load_declared_universe, universe_from_argv
from crucible.explain import explain as explain_lineage
from crucible.explain import render as render_lineage
from crucible.gate import PHASES
from crucible.keys import arm_register_key
from crucible.manifest import manifest_key
from crucible.runner import run_job
from crucible.slots import dispatchable_slots
from crucible.slots.arms import (
    ForeignRecipeSchemaError,
    load_arm_specs,
    read_register,
    register_arms,
    write_register,
)

__all__ = ["HANDLERS", "add_track_a_arguments"]

#: Derived from the slot modules themselves (`crucible.slots.dispatchable_slots`),
#: so the arc, the phase-1 gate and this dispatch table cannot disagree about
#: which slots the CLI can run. Today that resolves to `universe` and
#: `research`; `tests/test_weekly.py` pins it.
_SLOT_MODULES = dispatchable_slots()

#: "M and S arrive with track B" names phase 3 ("All three slots" — plan
#: §6). Derived rather than hardcoded so a phase renumbering cannot leave
#: `_slot_module`'s message stale (alpha-engine-config-I9839).
_ALL_SLOTS_PHASE = next(p for p in PHASES if p.id == "phase3")


def _today() -> dt.date:
    """The wall-clock calendar date. A thin, monkeypatchable seam —
    `dt.date.today` is a built-in classmethod tests cannot patch directly —
    used only by the `data.daily` holiday guard below (alpha-engine-config-I9781)."""
    return dt.date.today()


def _settings(args: argparse.Namespace) -> Any:
    # `dry_run` threaded here, once, rather than at each of this module's
    # eight `config.store()` call sites (alpha-engine-config-I9922 N1): every
    # track-A handler that reaches `config.store()` gets a read-only store
    # under `--dry-run` regardless of whether that handler's own body checks
    # the flag.
    return resolve_settings(
        store_uri=getattr(args, "store", None),
        strategy_dir=getattr(args, "strategy_dir", None),
        dry_run=bool(getattr(args, "dry_run", False)),
    )


def _source(args: argparse.Namespace, config: Any) -> PriceSource:
    """The price source named on the command line.

    `arctic` is the only production source. A source that is unavailable
    raises by name from inside `load_panel`; it never falls back, because a
    substituted source measures something other than what the run reports.
    """
    name = getattr(args, "source", None) or "arctic"
    if name == "arctic":
        return ArcticPriceSource(config.arctic_bucket)
    raise SystemExit(
        f"--source {name!r} is not a registered price source. The registered source is "
        "`arctic`; a test supplies its own `PriceSource` by calling the job function "
        "directly, which is the same code path."
    )


def _declared_universe(args: argparse.Namespace, config: Any) -> DeclaredUniverse | None:
    """The denominator of the coverage ratio, with its provenance.

    Resolution order is the one every other setting uses: the explicit
    argument (`--symbols`), then the environment (`CRUCIBLE_UNIVERSE_URI`,
    via `config.universe_uri`), then nothing — and nothing is returned as
    `None` so `run_daily`/`run_weekly` raise their own
    `UndeclaredUniverseError`, which names the fix. A document that exists
    and is malformed raises here, before the source is touched, and never
    degrades to "whatever the source has" (`crucible.data.universe`).
    """
    raw = getattr(args, "symbols", None)
    if raw:
        return universe_from_argv(raw)
    if config.universe_uri:
        return load_declared_universe(config.universe_uri, origin=config.origins["universe_uri"])
    return None


def _expected_symbols(declared: DeclaredUniverse | None, ctx: Any) -> list[str] | None:
    """Record the universe beside the run, then hand its symbols to the job.

    Called from inside the job body so the store copy lands under the run's
    own lineage (`outputs[]`); a copy written before `run_job` would belong
    to no manifest.
    """
    if declared is None:
        return None
    declared.record(ctx)
    return list(declared.symbols)


# -- data -------------------------------------------------------------------


def handle_data_daily(args: argparse.Namespace) -> int:
    # `data.daily`'s EventBridge schedule fires every weekday close, and a
    # market holiday (Thanksgiving, Independence Day observed, ...) is still
    # a weekday: without this guard the job resolves `trading_day` back to
    # the last real session (rule 3 — that resolution is legitimate) and
    # reruns the whole compile for it, overwriting that session's already-
    # good manifest with a second, unrelated writer's output — the same
    # last-writer-wins shape as the slot collision this fix addresses
    # (alpha-engine-config-I9781). Guarded only when `--date` was NOT given:
    # an explicit `--date` is a deliberate backfill/replay and is honoured
    # even on a holiday, same as everywhere else in this repo (rule 3).
    #
    # The guard does not skip the job. Rule 2: "if a job had nothing to do,
    # it produced a complete correct result — that is `ok`." A holiday
    # firing goes through `run_job` like any other invocation and writes a
    # real `ok` manifest, discriminated by the wall-clock calendar date it
    # actually fired on so it cannot collide with — or be mistaken for — the
    # prior session's real manifest at the same `trading_day`. A silent
    # `return 0` with no manifest is rule 1's and rule 2's exact shape: no
    # record exists to distinguish a holiday no-op from a `data.daily` that
    # has stopped working, and the alerting absence check has nothing to see
    # either way.
    today = _today()
    config = _settings(args)
    store = config.store()
    if getattr(args, "date", None) is None and not is_trading_day(today):
        detail = (
            f"{today.isoformat()} is not an NYSE trading day; the schedule fired on a "
            f"holiday. Nothing to compile for trading_day {args.trading_day.isoformat()}."
        )
        if args.dry_run:
            print(f"data.daily --date {args.trading_day} would record a holiday no-op: {detail}")
            return 0

        def _holiday_noop(ctx: Any) -> None:
            ctx.record_metric(
                {
                    "name": "data.daily.holiday_noop",
                    "module": "crucible.track_a",
                    "metric_type": "gauge",
                    "value": 1.0,
                    "unit": "count",
                    "n_floor": 0,
                    "status": "OK",
                    "status_reason": detail,
                    "source_path": manifest_key("data.daily", args.trading_day.isoformat()),
                    "last_updated_utc": ctx.started.astimezone(dt.UTC).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    ),
                }
            )

        ctx = run_job(
            "data.daily",
            _holiday_noop,
            store=store,
            trading_day=args.trading_day,
            run_mode=getattr(args, "run_mode", None),
            discriminator=today.isoformat(),
        )
        print(json.dumps({"run_id": ctx.run_id, "outputs": [], "detail": detail}, indent=2))
        return 0
    source = _source(args, config)
    declared = _declared_universe(args, config)
    if args.dry_run:
        print(
            f"data.daily --date {args.trading_day} would read {source.name} and write to "
            f"{config.store_uri}; universe: "
            + (
                f"{len(declared.symbols)} symbols from {declared.source_uri}"
                if declared
                else "NONE declared"
            )
        )
        return 0
    ctx = run_job(
        "data.daily",
        lambda c: run_daily(
            c,
            source=source,
            expected_symbols=_expected_symbols(declared, c),
        ),
        store=store,
        trading_day=args.trading_day,
        run_mode=getattr(args, "run_mode", None),
    )
    print(json.dumps({"run_id": ctx.run_id, "outputs": [o["key"] for o in ctx.outputs]}, indent=2))
    return 0


def handle_data_weekly(args: argparse.Namespace) -> int:
    config = _settings(args)
    store = config.store()
    source = _source(args, config)
    declared = _declared_universe(args, config)
    if args.dry_run:
        print(
            f"data.weekly --date {args.trading_day} would read {source.name} and write to "
            f"{config.store_uri}; universe: "
            + (
                f"{len(declared.symbols)} symbols from {declared.source_uri}"
                if declared
                else "NONE declared"
            )
        )
        return 0
    ctx = run_job(
        "data.weekly",
        lambda c: run_weekly(
            c,
            source=source,
            expected_symbols=_expected_symbols(declared, c),
        ),
        store=store,
        trading_day=args.trading_day,
        run_mode=getattr(args, "run_mode", None),
    )
    print(json.dumps({"run_id": ctx.run_id, "outputs": [o["key"] for o in ctx.outputs]}, indent=2))
    return 0


def handle_data_heal(args: argparse.Namespace) -> int:
    config = _settings(args)
    store = config.store()
    source = _source(args, config)
    start = dt.date.fromisoformat(args.from_date)
    end = dt.date.fromisoformat(args.to_date)
    declared = _declared_universe(args, config)
    if args.dry_run:
        from crucible.data.heal import in_region, sessions_in_range

        on_ec2, evidence = in_region()
        sessions = sessions_in_range(start, end)
        print(
            f"data.heal would recompile {len(sessions)} session(s) {start}..{end}. "
            f"Host: {evidence} (in region: {on_ec2})."
        )
        return 0
    ctx = run_job(
        "data.heal",
        lambda c: run_heal(
            c,
            source=source,
            start=start,
            end=end,
            gap=args.gap,
            i_am_in_region=getattr(args, "i_am_in_region", False),
            expected_symbols=_expected_symbols(declared, c),
        ),
        store=store,
        trading_day=end,
    )
    print(json.dumps({"run_id": ctx.run_id, "outputs": [o["key"] for o in ctx.outputs]}, indent=2))
    return 0


# -- experiments ------------------------------------------------------------


def _slot_module(slot: str) -> Any:
    try:
        return _SLOT_MODULES[slot]
    except KeyError as exc:
        raise SystemExit(
            f"slot {slot!r} is not implemented in track A. U and R are here; M and S "
            f"arrive with track B ({_ALL_SLOTS_PHASE.tracker}). A handler that returned "
            "0 for an unimplemented slot would be indistinguishable from a cycle that "
            "ran and had nothing to do."
        ) from exc


def handle_experiment_new(args: argparse.Namespace) -> int:
    """Register the slot's recipes, appending only what is new.

    ``--dry-run`` resolves the same inputs a real run would (the arm specs
    and the existing register) and reports what would be added — but
    computing that diff is read-only until `write_register` runs, so the
    dry-run path never calls `run_job` and writes nothing: not the register,
    not a manifest (defect #4b, 2026-09-01 adversarial review — the flag's
    own help text is "report what would be written; write nothing", and
    this handler wrote the register regardless of it).

    **M and S are refused here in the same shape :func:`_slot_module` uses**
    (`alpha-engine-config-I9961`). `--slot` admits all four, and for M and S
    this command used to reach `load_arm_specs`, fail on a missing `ranker`,
    and present as a malformed recipe tree — for recipes that are well-formed
    under the schema their own slot declares. The loader now refuses the slot
    by name; this converts that into the same exit `experiment.run --slot m`
    already produces, so the two commands give one answer about when M and S
    arrive rather than two unrelated failures.
    """
    config = _settings(args)
    store = config.store()
    try:
        specs = load_arm_specs(args.slot, store=store, strategy_dir=config.strategy_dir)
    except ForeignRecipeSchemaError as exc:
        raise SystemExit(
            f"{exc} U and R are here; M and S arrive with track B "
            f"({_ALL_SLOTS_PHASE.tracker}), which is when their recipes gain a register "
            "writer. Registering nothing and exiting 0 would be indistinguishable from a "
            "slot whose arms were all already present."
        ) from exc
    arm = getattr(args, "arm", None)
    if arm:
        specs = [s for s in specs if s.name == arm]
        if not specs:
            raise KeyError(
                f"no arm named {arm!r} in slot {args.slot!r}; registering nothing and "
                "exiting 0 would look exactly like registering it"
            )
    register = read_register(store, args.slot)
    before = set(register.all_arms())
    register, _ = register_arms(register, specs)
    added = sorted(set(register.all_arms()) - before)
    if args.dry_run:
        print(json.dumps({"would_register": added, "already_present": sorted(before)}, indent=2))
        return 0

    def job(ctx: Any) -> None:
        payload = write_register(store, args.slot, register)
        ctx.record_output(arm_register_key(args.slot), payload, schema_version="arm_register.v1")
        ctx.record_rows(rows_in=len(specs), rows_out=len(added))
        print(json.dumps({"registered": added, "already_present": sorted(before)}, indent=2))

    ctx = run_job(
        "experiment.new",
        job,
        store=store,
        trading_day=args.trading_day,
        run_mode=getattr(args, "run_mode", None),
    )
    return 0 if ctx else 0


def handle_experiment_run(args: argparse.Namespace) -> int:
    config = _settings(args)
    store = config.store()
    module = _slot_module(getattr(args, "slot", None) or "r")
    if args.dry_run:
        print(
            f"experiment.run --slot {args.slot} would call {module.__name__}.produce and "
            f"write its feed under the store at {config.store_uri}. Resolve inputs and "
            "report what would be written; write nothing."
        )
        return 0
    ctx = run_job(
        "experiment.run",
        lambda c: module.produce(c, settings=config, arm_name=getattr(args, "arm", None)),
        store=store,
        trading_day=args.trading_day,
        run_mode=getattr(args, "run_mode", None),
        # Four slots share one job name and one trading day; the slot is the
        # discriminator that keeps `--slot u` and `--slot r` from writing the
        # same manifest (alpha-engine-config-I9781).
        discriminator=args.slot,
    )
    print(json.dumps({"run_id": ctx.run_id, "outputs": [o["key"] for o in ctx.outputs]}, indent=2))
    return 0


def handle_experiment_grade(args: argparse.Namespace) -> int:
    config = _settings(args)
    store = config.store()
    module = _slot_module(args.slot)
    if args.dry_run:
        print(
            f"experiment.grade --slot {args.slot} would call {module.__name__}.grade, "
            f"score every settled cut and run the slot's arena cycle against the store at "
            f"{config.store_uri}. Resolve inputs and report what would be written; write "
            "nothing."
        )
        return 0
    result: dict[str, Any] = {}

    def job(ctx: Any) -> None:
        result.update(module.grade(ctx, settings=config))

    ctx = run_job(
        "experiment.grade",
        job,
        store=store,
        trading_day=args.trading_day,
        run_mode=getattr(args, "run_mode", None),
        # Same shape as `experiment.run` above: one job name, four slots
        # (alpha-engine-config-I9781).
        discriminator=args.slot,
    )
    print(
        json.dumps(
            {
                "run_id": ctx.run_id,
                "arena_cycle_key": result.get("arena_cycle_key"),
                "pointer": result.get("pointer", {}).get("status"),
                "champion": result.get("pointer", {}).get("champion"),
                "scored_arms": result.get("scored_arms"),
                "settled_dates": result.get("settled_dates"),
                "controls": result.get("controls"),
                "trial_rows_appended": result.get("trial_rows_appended"),
            },
            indent=2,
        )
    )
    return 0


# -- transparency and migration --------------------------------------------


def handle_explain(args: argparse.Namespace) -> int:
    """Read-only lineage walk. Writes no manifest, because it changes nothing.

    Every other job writes one; this one deliberately does not. A manifest
    per `explain` invocation would put a run row on the console for an
    operator reading a page, and absence-of-`explain` is not a fact about
    the system.
    """
    config = _settings(args)
    print(render_lineage(explain_lineage(config.store(), args.target)))
    return 0


def handle_migrate_history(args: argparse.Namespace) -> int:
    config = _settings(args)
    store = config.store()
    v1_config = resolve_settings(store_uri=getattr(args, "v1_store", None) or config.store_uri)
    v1_store = v1_config.store()
    recipes = {}
    for slot in ("u", "r"):
        try:
            for spec in load_arm_specs(slot, store=store, strategy_dir=config.strategy_dir):
                recipes[spec.name] = spec
        except FileNotFoundError:
            continue

    def job(ctx: Any) -> None:
        result = migrate_module.run_migrate_history(
            ctx,
            v1_store=v1_store,
            arm_recipes=recipes,
            allow_missing=getattr(args, "allow_missing", False),
        )
        print(json.dumps(result, indent=2, sort_keys=True))

    # `migrate.history` never checked `--dry-run` at all (alpha-engine-config-
    # I9922 N1): `store` is read-only under `dry_run` (via `_settings` above),
    # so `job`'s writes now raise `DryRunWriteRefusedError` rather than
    # landing for real, and `dry_run=` here means that raise replaces the
    # write attempt cleanly rather than also failing the manifest write in
    # `run_job`'s own `finally`.
    run_job(
        "migrate.history",
        job,
        store=store,
        trading_day=args.trading_day,
        dry_run=bool(getattr(args, "dry_run", False)),
        run_mode=getattr(args, "run_mode", None),
    )
    return 0


HANDLERS = {
    "data.daily": handle_data_daily,
    "data.weekly": handle_data_weekly,
    "data.heal": handle_data_heal,
    "experiment.new": handle_experiment_new,
    "experiment.run": handle_experiment_run,
    "experiment.grade": handle_experiment_grade,
    "explain": handle_explain,
    "migrate.history": handle_migrate_history,
}


def add_track_a_arguments(name: str, sub: argparse.ArgumentParser) -> None:
    """Track-A-only flags for one subcommand."""
    if name in ("data.daily", "data.weekly", "data.heal"):
        sub.add_argument(
            "--source",
            default="arctic",
            help="Price source. `arctic` is the only production source; it never falls back.",
        )
        sub.add_argument(
            "--symbols",
            help=(
                "Comma-separated expected universe. This is the DENOMINATOR of the "
                "coverage ratio, and `run_daily`/`run_weekly` refuse to run without one "
                "— a ratio computed over whatever arrived always reads 1.0, and the "
                "coverage floor cannot fire with no denominator to measure it against. "
                "Absent, the universe is read from the document CRUCIBLE_UNIVERSE_URI "
                "names (a membership document or a pointer to one; see "
                "crucible.data.universe), and a copy is written beside the run under "
                # Historical citation, not a phase pointer: alpha-engine-config-I9757
                # defect #2 is where this specific gap was first found. Kept in this
                # comment rather than the --help text per alpha-engine-config-I9839.
                "universe/declared/. Neither given: the job refuses."
            ),
        )
    if name == "data.heal":
        sub.add_argument("--from", dest="from_date", required=True, metavar="YYYY-MM-DD")
        sub.add_argument("--to", dest="to_date", required=True, metavar="YYYY-MM-DD")
        sub.add_argument(
            "--i-am-in-region",
            action="store_true",
            help=(
                "Override the in-region guard. The ONLY override — there is no "
                "environment variable and no config key, because the failure mode is an "
                "operator in a hurry."
            ),
        )
    if name in ("experiment.new", "experiment.run", "experiment.grade", "migrate.history"):
        sub.add_argument(
            "--strategy-dir",
            help=(
                "Checkout of alpha-engine-config/strategy/. Absent, arms are read from "
                "the strategy tree synced into the store."
            ),
        )
    if name == "migrate.history":
        sub.add_argument(
            "--v1-store", help="Store URI of the v1 artifacts. Read-only; nothing is written there."
        )
        sub.add_argument(
            "--allow-missing",
            action="store_true",
            help=(
                "Import the sources that are present. Every absent source is named in the "
                "result; without this flag an absent source fails the run."
            ),
        )
