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
from crucible.config import settings as resolve_settings
from crucible.data import ArcticPriceSource, PriceSource, run_daily, run_heal, run_weekly
from crucible.explain import explain as explain_lineage
from crucible.explain import render as render_lineage
from crucible.features import DEFAULT_FEATURE_VERSION
from crucible.runner import run_job
from crucible.slots import research, universe
from crucible.slots.arms import load_arm_specs, read_register, register_arms, write_register

__all__ = ["HANDLERS", "add_track_a_arguments"]

_SLOT_MODULES = {"u": universe, "r": research}


def _settings(args: argparse.Namespace) -> Any:
    return resolve_settings(
        store_uri=getattr(args, "store", None),
        strategy_dir=getattr(args, "strategy_dir", None),
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


def _symbols(args: argparse.Namespace) -> list[str] | None:
    raw = getattr(args, "symbols", None)
    if not raw:
        return None
    return sorted({s.strip().upper() for s in raw.split(",") if s.strip()})


# -- data -------------------------------------------------------------------


def handle_data_daily(args: argparse.Namespace) -> int:
    config = _settings(args)
    store = config.store()
    source = _source(args, config)
    if args.dry_run:
        print(
            f"data.daily --date {args.trading_day} would read {source.name} and write to "
            f"{config.store_uri}"
        )
        return 0
    ctx = run_job(
        "data.daily",
        lambda c: run_daily(
            c,
            source=source,
            expected_symbols=_symbols(args),
            feature_version=getattr(args, "feature_version", None) or DEFAULT_FEATURE_VERSION,
        ),
        store=store,
        trading_day=args.trading_day,
    )
    print(json.dumps({"run_id": ctx.run_id, "outputs": [o["key"] for o in ctx.outputs]}, indent=2))
    return 0


def handle_data_weekly(args: argparse.Namespace) -> int:
    config = _settings(args)
    store = config.store()
    source = _source(args, config)
    if args.dry_run:
        print(
            f"data.weekly --date {args.trading_day} would read {source.name} and write to "
            f"{config.store_uri}"
        )
        return 0
    ctx = run_job(
        "data.weekly",
        lambda c: run_weekly(
            c,
            source=source,
            expected_symbols=_symbols(args),
            feature_version=getattr(args, "feature_version", None) or DEFAULT_FEATURE_VERSION,
            require_full_week=not getattr(args, "allow_week_gap", False),
        ),
        store=store,
        trading_day=args.trading_day,
    )
    print(json.dumps({"run_id": ctx.run_id, "outputs": [o["key"] for o in ctx.outputs]}, indent=2))
    return 0


def handle_data_heal(args: argparse.Namespace) -> int:
    config = _settings(args)
    store = config.store()
    source = _source(args, config)
    start = dt.date.fromisoformat(args.from_date)
    end = dt.date.fromisoformat(args.to_date)
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
            expected_symbols=_symbols(args),
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
            "arrive with track B (alpha-engine-config-I9757). A handler that returned "
            "0 for an unimplemented slot would be indistinguishable from a cycle that "
            "ran and had nothing to do."
        ) from exc


def handle_experiment_new(args: argparse.Namespace) -> int:
    """Register the slot's recipes, appending only what is new."""
    config = _settings(args)
    store = config.store()

    def job(ctx: Any) -> None:
        specs = load_arm_specs(args.slot, store=store, strategy_dir=config.strategy_dir)
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
        payload = write_register(store, args.slot, register)
        ctx.record_output(
            f"arms/{args.slot}/register.jsonl", payload, schema_version="arm_register.v1"
        )
        added = sorted(set(register.all_arms()) - before)
        ctx.record_rows(rows_in=len(specs), rows_out=len(added))
        print(json.dumps({"registered": added, "already_present": sorted(before)}, indent=2))

    ctx = run_job("experiment.new", job, store=store, trading_day=args.trading_day)
    return 0 if ctx else 0


def handle_experiment_run(args: argparse.Namespace) -> int:
    config = _settings(args)
    store = config.store()
    module = _slot_module(getattr(args, "slot", None) or "r")
    ctx = run_job(
        "experiment.run",
        lambda c: module.produce(c, settings=config, arm_name=getattr(args, "arm", None)),
        store=store,
        trading_day=args.trading_day,
    )
    print(json.dumps({"run_id": ctx.run_id, "outputs": [o["key"] for o in ctx.outputs]}, indent=2))
    return 0


def handle_experiment_grade(args: argparse.Namespace) -> int:
    config = _settings(args)
    store = config.store()
    module = _slot_module(args.slot)
    result: dict[str, Any] = {}

    def job(ctx: Any) -> None:
        result.update(module.grade(ctx, settings=config))

    ctx = run_job("experiment.grade", job, store=store, trading_day=args.trading_day)
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

    run_job("migrate.history", job, store=store, trading_day=args.trading_day)
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
                "coverage ratio — without it no ratio is reported, because a ratio over "
                "whatever arrived always reads 1.0."
            ),
        )
        sub.add_argument("--feature-version", help="Override the derived feature version.")
    if name == "data.weekly":
        sub.add_argument(
            "--allow-week-gap",
            action="store_true",
            help=(
                "Run even though a session in the week has no compiled panel. Records the "
                "gap in the manifest; use only when the gap is understood."
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
