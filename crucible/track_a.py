"""Track-A CLI handlers: the data, feature, U/R, ledger and explain jobs.

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

from crucible.backfill import run_backfill
from crucible.calendar import is_trading_day, resolve_trading_day
from crucible.config import settings as resolve_settings
from crucible.data import ArcticPriceSource, PriceSource, run_daily, run_heal, run_weekly
from crucible.data.point_in_time import FilingDatePointInTimeSource, PointInTimeSource
from crucible.data.universe import DeclaredUniverse, load_declared_universe, universe_from_argv
from crucible.explain import explain as explain_lineage
from crucible.explain import render as render_lineage
from crucible.explain import select_newest_settled_verdict
from crucible.gate import PHASES
from crucible.keys import arm_register_key, calendar_day_discriminator
from crucible.manifest import manifest_key
from crucible.registration import load_registrable_recipes
from crucible.runner import run_job
from crucible.slots import arm_name as name_component
from crucible.slots import dispatchable_slots, history_producer
from crucible.slots.arms import (
    ForeignRecipeSchemaError,
    read_register,
    register_arms,
    write_register,
)
from crucible.slots.inputs import SlotUnservableError

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

    `--arctic-library` (`alpha-engine-config-I10457`) is additive and
    production-inert: production never passes it, so `library` resolves to
    `None` and `ArcticPriceSource` reads the production `universe` library
    exactly as before this flag existed. The integration tier is the only
    caller (`tests/integration/test_cli_jobs.py`), and the value it passes is
    resolved through `crucible.required.require_env` in its own conftest —
    RAISE-on-absent already lives there, not here.
    """
    name = getattr(args, "source", None) or "arctic"
    if name == "arctic":
        return ArcticPriceSource(
            config.arctic_bucket, library=getattr(args, "arctic_library", None) or None
        )
    raise SystemExit(
        f"--source {name!r} is not a registered price source. The registered source is "
        "`arctic`; a test supplies its own `PriceSource` by calling the job function "
        "directly, which is the same code path."
    )


def _point_in_time_source(config: Any) -> PointInTimeSource:
    """The fundamentals / sector / 13F source every feature compile reads.

    Rooted at the same data bucket the price source reads
    (`CRUCIBLE_ARCTIC_BUCKET`), opened READ-ONLY: this job never writes the
    bucket it reads inputs from. No bucket name lives in this package, and an
    unset setting refuses by name rather than reading from nowhere
    (`alpha-engine-config-I10721`).
    """
    from crucible.config import store_from_uri  # noqa: PLC0415 - one call site
    from crucible.store import read_only  # noqa: PLC0415 - one call site

    if not config.arctic_bucket:
        raise SystemExit(
            "the point-in-time source reads the data bucket named by CRUCIBLE_ARCTIC_BUCKET, "
            "which is unset; the feature layer's fundamentals, sector and 13F columns have "
            "no other source and are never zero-filled"
        )
    reader = read_only(
        store_from_uri(f"s3://{config.arctic_bucket}"),
        reason="point-in-time inputs are read from the data bucket, never written",
    )
    return FilingDatePointInTimeSource(reader, label=config.arctic_bucket)


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
        # No `--dry-run` branch here either: the holiday no-op writes one
        # manifest and nothing else, and under a capturing store that
        # manifest is recorded rather than written — so a dry run of a
        # holiday firing reports the exact key the real firing files
        # (alpha-engine-config-I11012).

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
            dry_run=bool(getattr(args, "dry_run", False)),
            discriminator=today.isoformat(),
        )
        print(json.dumps({"run_id": ctx.run_id, "outputs": [], "detail": detail}, indent=2))
        return 0
    source = _source(args, config)
    point_in_time = _point_in_time_source(config)
    declared = _declared_universe(args, config)
    # NO `--dry-run` branch (alpha-engine-config-I11012). `data.daily` is the
    # job whose dry run most needed to be a rehearsal: the compile is where a
    # missing feature column, an unsatisfiable declared input or a refusing
    # store grant lives, and every one of those was invisible to a print that
    # only named the source and the store URI. The READS below are real — a
    # dry run of this job talks to ArcticDB and to the point-in-time source,
    # which is the point — and every WRITE is recorded rather than performed.
    ctx = run_job(
        "data.daily",
        lambda c: run_daily(
            c,
            source=source,
            point_in_time=point_in_time,
            expected_symbols=_expected_symbols(declared, c),
        ),
        store=store,
        trading_day=args.trading_day,
        run_mode=getattr(args, "run_mode", None),
        dry_run=bool(getattr(args, "dry_run", False)),
    )
    print(json.dumps({"run_id": ctx.run_id, "outputs": [o["key"] for o in ctx.outputs]}, indent=2))
    return 0


def handle_data_weekly(args: argparse.Namespace) -> int:
    config = _settings(args)
    store = config.store()
    source = _source(args, config)
    point_in_time = _point_in_time_source(config)
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
            point_in_time=point_in_time,
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
    point_in_time = _point_in_time_source(config)
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
            point_in_time=point_in_time,
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
            f"slot {slot!r} is not implemented in track A. U, R, M and S are here "
            f"(`crucible.slots.dispatchable_slots()` reads {sorted(_SLOT_MODULES)} live); "
            f"a slot beyond those four arrives with track B ({_ALL_SLOTS_PHASE.tracker}). "
            "A handler that returned 0 for an unimplemented slot would be "
            "indistinguishable from a cycle that ran and had nothing to do."
        ) from exc


def _recipes_for_registration(slot: str, *, config: Any, store: Any) -> list[Any]:
    """The slot's loaded recipes, in the shape `register_arms` reads.

    One entry point over three recipe SCHEMAS (`alpha-engine-config-I9957`,
    `-I10512`), and since `alpha-engine-config-I10927` that entry point is
    `crucible.registration.load_registrable_recipes` — shared with
    `experiment.register` and with the gate clause that grades the
    recipe-to-register gap, so the three cannot disagree about what a
    registrable recipe is. U and R recipes are `ArmSpec`s; an M recipe is a
    `ModelRecipe`, wrapped in `crucible.slots.model.RegisteredModelArm`; an S
    recipe is a `StrategyRecipe`, wrapped in
    `crucible.slots.strategy.RegisteredStrategyArm`. Each wrapper carries its
    own id — the hash of its own spec — so it is what registers.
    Re-deriving an id here from an `ArmSpec` view would give one arm two
    identities, and the register, the shadow/session-inputs artifacts and the
    series would each speak about a different one.

    The refusals are PRINTED here and returned nowhere, because this is
    `experiment.new`'s reader and `experiment.new` is an operator command
    whose reader is a terminal. `experiment.register` calls the shared loader
    directly and records each refusal on its manifest instead.
    """
    load = load_registrable_recipes(slot, strategy_dir=config.strategy_dir, store=store)
    if load.refusals:
        print(
            json.dumps(
                {
                    "refused": [
                        {"arm": r.arm, "unresolvable": list(r.unresolvable)} for r in load.refusals
                    ]
                },
                indent=2,
            )
        )
    return list(load.specs)


def handle_experiment_new(args: argparse.Namespace) -> int:
    """Register the slot's recipes, appending only what is new.

    ``--dry-run`` resolves the same inputs a real run would (the arm specs
    and the existing register) and reports what would be added — but
    computing that diff is read-only until `write_register` runs, so the
    dry-run path never calls `run_job` and writes nothing: not the register,
    not a manifest (defect #4b, 2026-09-01 adversarial review — the flag's
    own help text is "report what would be written; write nothing", and
    this handler wrote the register regardless of it).

    **Neither M nor S is refused by name any longer**
    (`alpha-engine-config-I9957`, `-I10512`). Both recipe types —
    `ModelRecipe` and `StrategyRecipe` documents — are not `ArmSpec`s, so
    `load_arm_specs` still refuses slots `m` and `s` — that refusal is
    correct and stays, it is simply never reached for them any more: this
    handler resolves each slot's own loader
    (`crucible.slots.model.load_model_recipes`,
    `crucible.slots.strategy.load_strategy_slot`) instead of converting the
    refusal into an exit. Both read the same two sources (a checkout or the
    synced store tree), and each slot's refused arms are reported here rather
    than silently dropped: an arm that will not register is the fact an
    operator running `experiment.new` most needs.

    All four slots admit `--slot` now. The `ForeignRecipeSchemaError` handler
    below is kept as a defensive backstop — `_recipes_for_registration`
    dispatches M and S to their own loaders before `load_arm_specs` is ever
    asked for either, so the exception path is not expected to fire for any
    of the four current slots; it stays in case a future slot adds a fourth
    recipe schema without a loader wired in here yet.
    """
    config = _settings(args)
    store = config.store()
    try:
        specs = _recipes_for_registration(args.slot, config=config, store=store)
    except ForeignRecipeSchemaError as exc:
        raise SystemExit(
            f"{exc} U, R, M and S are all here; a slot beyond those four arrives with "
            f"track B ({_ALL_SLOTS_PHASE.tracker}), which is when its recipes gain a "
            "register writer. Registering nothing and exiting 0 would be "
            "indistinguishable from a slot whose arms were all already present."
        ) from exc
    arm = getattr(args, "arm", None)
    if arm:
        # Bare name or registered id, both resolved by `name_component` — see
        # `crucible.slots.cycle.run_produce` for why the id form has to work:
        # this command is the one that PRINTS the ids.
        selector = name_component(arm)
        specs = [s for s in specs if s.name == selector]
        if not specs:
            raise KeyError(
                f"no arm named {selector!r} in slot {args.slot!r}; registering nothing and "
                "exiting 0 would look exactly like registering it"
            )
    register = read_register(store, args.slot)
    before = set(register.all_arms())
    # The trading day this run binds to, resolved the same way `run_job`
    # resolves it (`crucible.runner.run_job`: explicit when passed, else the
    # last completed session). It is what each new `registered` event is dated
    # with — never the recipe's own `registered_at`
    # (`alpha-engine-config-I10948`).
    filed_on = (args.trading_day or resolve_trading_day()).isoformat()
    register, _ = register_arms(register, specs, filed_on=filed_on)
    added = sorted(set(register.all_arms()) - before)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "would_register": added,
                    "already_present": sorted(before),
                    "filed_on": filed_on,
                },
                indent=2,
            )
        )
        return 0

    def job(ctx: Any) -> None:
        payload = write_register(store, args.slot, register)
        ctx.record_output(arm_register_key(args.slot), payload, schema_version="arm_register.v1")
        ctx.record_rows(rows_in=len(specs), rows_out=len(added))
        ctx.record_metric(
            _arms_appended_metric(slot=args.slot, added=added, filed_on=filed_on, now=ctx.started)
        )
        print(
            json.dumps(
                {"registered": added, "already_present": sorted(before), "filed_on": filed_on},
                indent=2,
            )
        )

    ctx = run_job(
        "experiment.new",
        job,
        store=store,
        trading_day=args.trading_day,
        run_mode=getattr(args, "run_mode", None),
        # One manifest per INVOCATION, not per trading day
        # (`alpha-engine-config-I10948`). This is an operator command that
        # registers one named arm at a time, so registering seven arms is
        # seven invocations — and with the slot alone as the discriminator
        # (or none at all, which is what it had) they all write the same key
        # and each overwrites the last. Measured 2026-09-17: seven runs, one
        # surviving manifest, `rows_out: 1`, and the other six recoverable
        # only from S3 object versions. `experiment.new` is on-demand
        # (`components.yaml`: `schedule: null`, `deadline: null`), so no
        # absence grading keys off a predictable manifest key here.
        #
        # Led by the CALENDAR day this process ran on
        # (`alpha-engine-config-I10999`). Those same seven registrations ran
        # on 2026-09-17 with `--trading-day 2026-09-11`, so every manifest
        # landed under `runs/experiment.new/2026-09-11/`,
        # `runs/experiment.new/2026-09-17/` did not exist, and "what mutated
        # the store today" was answerable only from S3 object-version
        # metadata. The trading-day SEGMENT is unchanged — moving it is what
        # would blind `crucible.alerts`' absence grading for a scheduled job,
        # which is why this is the on-demand half of a deliberate asymmetry.
        # The whole decision is written at
        # `crucible.keys.calendar_day_discriminator`.
        discriminator=lambda ctx: calendar_day_discriminator(
            ctx.calendar_date, suffix=f"{args.slot}-{ctx.run_id}"
        ),
    )
    return 0 if ctx else 0


#: The metric both registration jobs file naming WHICH arms they appended and
#: on which trading day (`alpha-engine-config-I10948`).
#:
#: `rows_out` counts them; this names them. A count is not a durable record of
#: an append: measured 2026-09-17, seven arms were appended across two slots
#: and the only surviving manifest said `rows_out: 1` and named one register
#: key, because each invocation overwrote the last at the same manifest key.
#: Reconstructing which seven arms they were, and when, took S3 object-version
#: archaeology — which is precisely the reconstruction "manifest or it did not
#: happen" exists to make unnecessary.
ARMS_APPENDED_METRIC = "arms_appended"


def _arms_appended_metric(*, slot: str, added: list[str], filed_on: str, now: dt.datetime) -> dict:
    """The `arms_appended` MetricRecord for one registration run.

    `filed_on` is carried as its own field, not only inside the prose: it is
    the value stamped onto each new `ArmEvent.date`, and a reader checking
    that the register agrees with the manifest must not have to parse a
    sentence to find it.
    """
    return {
        "name": ARMS_APPENDED_METRIC,
        "module": "crucible.track_a",
        "metric_type": "provenance",
        "value": float(len(added)),
        "unit": "arms",
        "n_floor": 0,
        "status": "OK",
        "status_reason": (
            f"slot {slot}: {len(added)} arm(s) appended to the register on trading day "
            f"{filed_on}: {', '.join(added) if added else '(none)'}"
        ),
        "source_path": arm_register_key(slot),
        "last_updated_utc": now.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        # Structured provenance a reader keys off without parsing prose.
        "arms_appended": list(added),
        "filed_on": filed_on,
    }


#: The metric `experiment.register` files every run, on every slot, whether or
#: not it registered anything. Four numbers rather than one: "registered 0"
#: and "read 0" are opposite facts that a single count renders identically,
#: and the second is a broken release while the first is a quiet week.
REGISTER_COVERAGE_METRIC = "arm_recipes_registered"


def handle_experiment_register(args: argparse.Namespace) -> int:
    """Register every recipe the release in force declares and the slot's
    register lacks (`alpha-engine-config-I10927`).

    **Not a relaxation of `experiment.new --arm`.** That flag stays required
    and that job's meaning is unchanged: naming one arm is how an operator
    registers one arm deliberately. This job's deliberate act is a different
    one, one layer up — **pinning the release**. What the pinned release
    declares under `strategy/current/arms/{slot}/` is then DERIVED from that
    pin, exactly as `crucible.slots.dispatchable_slots` derives the
    dispatchable slots from the modules and `crucible.weekly.arc_stages`
    derives the arc from the registry. Nobody keeps a list of arms current,
    because there is no list.

    **A registration is never a promotion.** This job appends `registered`
    events to `arms/{slot}/register.jsonl` and touches nothing else: no
    `champions/` pointer, no serving contract. It makes an arm SCORED — the
    arena decides, later and on evidence, whether it is ever served.

    **Idempotent, and it says so per recipe.** Re-running appends nothing —
    `crucible.slots.arms.register_arms` skips an id already in the fold — and
    the run reports `registered` against `already_present` the way `data.heal`
    reports `repaired` against `already_present`, so "nothing to do" is
    legible rather than indistinguishable from "did not look".

    **A refusal is recorded, not swallowed, and does not fail the stage.** An
    arm whose declared inputs no producer can resolve
    (`sota_directional_combine`, ruled expected behaviour in
    `alpha-engine-config-I10695`) comes back as a
    `crucible.slots.inputs.InputRefusal` value, is recorded as a rejection
    with its reason and as an `unservable` MetricRecord, and its siblings
    still register. An unreadable tree or a malformed recipe still raises,
    and the manifest carries the cause (AGENTS.md rule 5).

    **A slot where NOTHING registers outside a cycle is a state, not a
    failure** — the one deviation from the raise-by-default rule above, and
    the reason is measured rather than assumed. `load_strategy_slot` refuses
    every S recipe outside a cycle by design: an S arm declares no
    `registered_at`, and `experiment.run --slot s` stamps that field from the
    first run that registers it. So `experiment.register --slot s` finds
    nothing registrable *whenever S has not yet cycled*, which today is
    always, because S's `produce` builds its book on the M champion and there
    is no M champion (`alpha-engine-config-I10924`).

    `crucible.weekly.run_arc` stops at the first stage that RAISES, and this
    stage runs at 11:00 — ahead of `experiment.run`, `experiment.grade`,
    `promote`, `report`, `console` and `explain`. Raising here would
    therefore take down the whole Saturday arc, including the producer phase
    1's `explain_walks_a_verdict` depends on, over a slot behaving exactly as
    designed. Measured 2026-09-17 against the live store, before it could
    happen on 2026-09-19.

    Per the fail-loud deviation rule, naming the three things it requires:
    (a) the failure mode swallowed is `SlotUnservableError` — every arm in one
    slot refused; (b) the primary deliverable survives because this stage's
    deliverable is "register what the release declares and report the rest",
    and a slot with nothing registrable has nothing to register, while every
    other slot's stage is unaffected; (c) the recording surface is this run's
    own manifest — each refusal as a rejection, an `unservable` MetricRecord,
    `rows_out=0` — plus `gate._clause_every_recipe_registered`, which reads
    RED over precisely this state and is what makes it visible rather than
    quiet. The condition is not lost; only the arc-killing raise is.
    """
    config = _settings(args)
    store = config.store()
    try:
        load = load_registrable_recipes(args.slot, strategy_dir=config.strategy_dir, store=store)
    except SlotUnservableError as unservable:
        return _register_nothing_registrable(args, store, unservable)
    register = read_register(store, args.slot)
    before = set(register.all_arms())
    # See `handle_experiment_new` for why this is the run's trading day and
    # not the recipe's `registered_at` (`alpha-engine-config-I10948`).
    filed_on = (args.trading_day or resolve_trading_day()).isoformat()
    register, _ = register_arms(register, list(load.specs), filed_on=filed_on)
    added = sorted(set(register.all_arms()) - before)
    already_present = sorted(spec.arm_id for spec in load.specs if spec.arm_id in before)
    refused = [
        {"arm": r.arm, "unresolvable": list(r.unresolvable), "reason": r.reason}
        for r in load.refusals
    ]
    report = {
        "slot": args.slot,
        "read": load.n_read,
        "registered": added,
        "already_present": already_present,
        "refused": refused,
        "filed_on": filed_on,
    }
    if args.dry_run:
        print(json.dumps({**report, "registered": [], "would_register": added}, indent=2))
        return 0

    def job(ctx: Any) -> None:
        # Recorded FIRST, before the write that can raise: a refusal that only
        # reaches the manifest of a run that succeeded is a refusal nobody
        # sees on the day it mattered.
        for metric in load.refusal_metrics:
            ctx.record_metric(metric)
        for refusal in load.refusals:
            ctx.record_rejected(
                _refusal_reason(args.slot, refusal),
                1,
            )
        payload = write_register(store, args.slot, register)
        ctx.record_output(arm_register_key(args.slot), payload, schema_version="arm_register.v1")
        ctx.record_rows(rows_in=load.n_read, rows_out=len(added))
        ctx.record_metric(
            _arms_appended_metric(slot=args.slot, added=added, filed_on=filed_on, now=ctx.started)
        )
        ctx.record_metric(
            {
                "name": REGISTER_COVERAGE_METRIC,
                "module": "crucible.track_a",
                "metric_type": "coverage",
                "value": float(len(added)),
                "unit": "arms",
                "n_floor": 0,
                "status": "OK",
                "status_reason": (
                    f"slot {args.slot}: {load.n_read} recipe(s) read from the release in "
                    f"force, {len(added)} registered, {len(already_present)} already "
                    f"present, {len(load.refusals)} refused at registration"
                ),
                "source_path": arm_register_key(args.slot),
                "last_updated_utc": ctx.started.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )
        print(json.dumps(report, indent=2))

    run_job(
        "experiment.register",
        job,
        store=store,
        trading_day=args.trading_day,
        run_mode=getattr(args, "run_mode", None),
        # Four slots share one job name and one trading day; the slot is the
        # discriminator that keeps `--slot u` and `--slot m` from writing the
        # same manifest (alpha-engine-config-I9781).
        discriminator=args.slot,
    )
    return 0


def _refusals_from(unservable: SlotUnservableError) -> list[str]:
    """One rejection string per refused arm, read from the STRUCTURED field.

    `SlotUnservableError` carries `.refusals` — a tuple of
    `crucible.slots.inputs.InputRefusal`, each with `.arm` and `.reason`.
    Reading `args[0]` instead would hand back the formatted message, and
    iterating a str yields CHARACTERS: one rejection per character, which is
    a reading-shaped artifact containing nothing and which
    `record_rejected` raises over at 200.
    """
    return [
        f"{refusal.arm}: {refusal.reason}" for refusal in getattr(unservable, "refusals", ())
    ] or [str(unservable)]


def _register_nothing_registrable(
    args: argparse.Namespace, store: Any, unservable: SlotUnservableError
) -> int:
    """The `SlotUnservableError` branch of :func:`handle_experiment_register`.

    Files a real manifest for a real reading — `status ok`, `rows_out=0`, one
    rejection per refused arm and one `unservable` MetricRecord — rather than
    raising and stopping the arc. See that function's docstring for why this
    is the one sanctioned deviation from raise-by-default.
    """
    # `SlotUnservableError` is constructed with a TUPLE of refusal strings,
    # but it is also raised with a single formatted string by other call
    # sites. Iterating a str yields characters, so the shape is normalised
    # here rather than assumed — a refusal list of 60 one-character rows is
    # not a reading, and `record_rejected` would file every one of them.
    refusals = _refusals_from(unservable)
    reason = str(unservable)
    report = {
        "slot": args.slot,
        "read": len(refusals),
        "registered": [],
        "already_present": [],
        "refused": refusals,
        "nothing_registrable": True,
    }
    if args.dry_run:
        print(json.dumps({**report, "would_register": []}, indent=2))
        return 0

    def job(ctx: Any) -> None:
        for refusal in refusals:
            ctx.record_rejected(f"{args.slot}: {refusal}"[:200], 1)
        ctx.record_rows(rows_in=len(refusals), rows_out=0)
        ctx.record_metric(
            {
                "name": REGISTER_COVERAGE_METRIC,
                "module": "crucible.track_a",
                "metric_type": "coverage",
                "value": 0.0,
                "unit": "arms",
                "n_floor": 0,
                # `unservable` and not `OK`: nothing registered, and a slot
                # that registers nothing must not read like one that had
                # nothing left to do (plan 5.3, 7).
                "status": "unservable",
                "status_reason": (
                    f"slot {args.slot}: no recipe is registrable outside a cycle, so this "
                    f"stage registered nothing and did not fail the arc. {reason}"
                )[:800],
                "source_path": arm_register_key(args.slot),
                "last_updated_utc": ctx.started.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )
        print(json.dumps(report, indent=2))

    run_job(
        "experiment.register",
        job,
        store=store,
        trading_day=args.trading_day,
        run_mode=getattr(args, "run_mode", None),
        discriminator=args.slot,
    )
    return 0


def _refusal_reason(slot: str, refusal: Any) -> str:
    """The rejection string, fitted at the call site.

    `RunContext.record_rejected` RAISES over 200 characters rather than
    truncating (`alpha-engine-config-I10484`), and the overflow would cost the
    manifest its inputs, outputs, metrics and every row count. The job authors
    this string, so the job is what fits it — the arm and its unresolvable
    inputs are what make it actionable, and they come first.
    """
    detail = f"{slot}:{refusal.arm} unresolvable inputs {sorted(refusal.unresolvable)}"
    return detail[:200]


def handle_experiment_run(args: argparse.Namespace) -> int:
    config = _settings(args)
    store = config.store()
    module = _slot_module(getattr(args, "slot", None) or "r")
    # NO `--dry-run` branch (alpha-engine-config-I11012). The one this
    # replaces printed "would call produce and write its feed" — a sentence
    # restating the handler's own source, which could not be wrong and could
    # not be right. Under `--dry-run` the store RECORDS writes instead of
    # performing them (`crucible.store.capturing`), so the body below runs
    # for real against real inputs, fails exactly where the real run fails,
    # and `crucible.cli.main` prints the key set it would have written.
    ctx = run_job(
        "experiment.run",
        lambda c: module.produce(c, settings=config, arm_name=getattr(args, "arm", None)),
        store=store,
        trading_day=args.trading_day,
        run_mode=getattr(args, "run_mode", None),
        # Passed so the run says out loud that it wrote nothing. The manifest
        # is captured either way — the store decides that, not this flag —
        # but a rehearsal that printed nothing would look like a real run
        # until the captured key set lands at the end of the invocation.
        dry_run=bool(getattr(args, "dry_run", False)),
        # Four slots share one job name and one trading day; the slot is the
        # discriminator that keeps `--slot u` and `--slot r` from writing the
        # same manifest (alpha-engine-config-I9781).
        discriminator=args.slot,
    )
    print(json.dumps({"run_id": ctx.run_id, "outputs": [o["key"] for o in ctx.outputs]}, indent=2))
    return 0


def handle_experiment_backfill(args: argparse.Namespace) -> int:
    """`experiment.backfill` — one arm's history over a session range.

    `alpha-engine-config-I10696`. The whole job body is
    :func:`crucible.backfill.run_backfill`, handed the slot's OWN per-session
    history producer — the same fitting path `experiment.run` reaches, minus
    the serving half (`alpha-engine-config-I11005`). This
    handler resolves what that function cannot see for itself (the settings,
    the store, the slot module and the slot's registered specs) and nothing
    else: a second fitting path is the one thing this job must never grow.

    The manifest is keyed to ``--to`` and discriminated by `{slot}.{arm}`, so
    two arms backfilled to the same session are two manifests rather than one
    overwriting the other.
    """
    config = _settings(args)
    store = config.store()
    module = _slot_module(args.slot)
    # Resolved for the dry run TOO, and by the same call: `produce_history`
    # rather than `produce` is what keeps a backfill out of the serving path
    # (alpha-engine-config-I11005), and a dry run that resolved a different
    # entry point from the real run would be rehearsing a different job.
    produce = history_producer(module)
    start = dt.date.fromisoformat(args.from_date)
    end = dt.date.fromisoformat(args.to_date)
    specs = _recipes_for_registration(args.slot, config=config, store=store)
    dry_run = bool(getattr(args, "dry_run", False))
    if dry_run:
        # PR323's PRE-FLIGHT, kept and extended rather than replaced
        # (alpha-engine-config-I11005 then -I11012). Everything here is
        # resolved through the SAME functions the job calls, in the job's
        # order, so a rehearsal refuses what the job refuses — and then the
        # body below EXECUTES, against a store that records its writes
        # instead of performing them. The gap PR323 named in its printed line
        # ("NOT rehearsed: the per-session produce call itself") is what this
        # closes: that call is exactly where the measured failure lived.
        from crucible.backfill import (  # noqa: PLC0415 - one call site
            arm_id_for,
            sessions_in_range,
        )

        sessions = sessions_in_range(start, end)
        # RAISES, exactly as the job does: whether the arm is registered is a
        # property of the store and the recipes, identical on every host, so
        # a dry run that accepted an arm the job would refuse is wrong here
        # and not merely optimistic.
        arm_id = arm_id_for(specs, slot=args.slot, arm=name_component(args.arm))
        print(
            f"experiment.backfill --slot {args.slot} --arm {args.arm} ({arm_id}) will "
            f"rehearse {len(sessions)} session(s) {start}..{end} through "
            f"{module.__name__}.produce_history, which does not enter the serving path. "
            "Every read is real; every write is recorded, not performed. The in-region "
            "guard is REPORTED rather than raised — it is a property of the host, and a "
            "laptop rehearsal of a range that will be dispatched in region is the normal "
            "case."
        )
    result: dict[str, Any] = {}
    ctx = run_job(
        "experiment.backfill",
        lambda c: result.update(
            run_backfill(
                c,
                produce=produce,
                specs=specs,
                settings=config,
                slot=args.slot,
                arm=name_component(args.arm),
                start=start,
                end=end,
                force=bool(getattr(args, "force", False)),
                i_am_in_region=bool(getattr(args, "i_am_in_region", False)),
                rehearsal=dry_run,
            )
        ),
        store=store,
        trading_day=end,
        run_mode=getattr(args, "run_mode", None),
        dry_run=dry_run,
        # One job name, four slots and many arms, all legitimately keyed to
        # the same `--to` session (alpha-engine-config-I9781's shape).
        discriminator=f"{args.slot}.{name_component(args.arm)}",
    )
    print(
        json.dumps(
            {
                "run_id": ctx.run_id,
                "arm_id": result.get("arm_id"),
                "produced": len(result.get("produced", [])),
                "already_present": len(result.get("already_present", [])),
                "refused": result.get("refused", []),
                "in_region_verdict": result.get("in_region_verdict"),
            },
            indent=2,
        )
    )
    return 0


def handle_experiment_grade(args: argparse.Namespace) -> int:
    config = _settings(args)
    store = config.store()
    module = _slot_module(args.slot)
    # NO `--dry-run` branch, for the reason `experiment.run` has none
    # (`alpha-engine-config-I11012`), applied to the one handler that kept it
    # (`alpha-engine-config-I10721`). The branch this replaces printed
    # "would call {module}.grade, score every settled cut and run the slot's
    # arena cycle" and returned 0 having resolved NOTHING — a sentence
    # restating this function's own next four lines, which could not be wrong
    # and could not be right. It was also the blindest possible instance of
    # it: `experiment.grade` is the only writer of
    # `arena/{slot}/{day}/arena_cycle.json`, the artifact `arms_all_scored`
    # grades, so the one question an operator brings to this flag — "would a
    # replay of this past session rewrite that day's cycle?" — was the one
    # question the flag could not answer. Under `--dry-run` the store RECORDS
    # writes instead of performing them (`crucible.store.capturing`), so the
    # body below runs for real against real inputs, fails exactly where the
    # real run fails, and `crucible.cli.main` prints the key set it would
    # have written.
    result: dict[str, Any] = {}

    def job(ctx: Any) -> None:
        result.update(module.grade(ctx, settings=config))

    ctx = run_job(
        "experiment.grade",
        job,
        store=store,
        trading_day=args.trading_day,
        run_mode=getattr(args, "run_mode", None),
        # Passed so the run says out loud that it wrote nothing. The manifest
        # is captured either way — the store decides that, not this flag —
        # but a rehearsal that printed nothing would look like a real run
        # until the captured key set lands at the end of the invocation.
        dry_run=bool(getattr(args, "dry_run", False)),
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


def _lineage_keys(node: Any) -> list[str]:
    """Every key the walk touched, root first, depth-first, de-duplicated."""
    out: list[str] = []
    stack = [node]
    while stack:
        current = stack.pop()
        if current.key not in out:
            out.append(current.key)
        stack.extend(reversed(current.parents))
    return out


def handle_explain(args: argparse.Namespace) -> int:
    """Lineage walk, through `run_job` like every other job (AGENTS.md rule 1).

    It changes nothing in the store — no outputs — but it records every key
    it WALKED as an input, and that record is what plan §10.8 is measured
    by: `crucible.gate._clause_explain_walks_a_verdict` reads
    `runs/explain/{day}/run.json` for a manifest whose inputs include a
    `verdict.json`. Until 2026-09-05 this handler deliberately wrote no
    manifest ("absence-of-explain is not a fact about the system"), which was
    true and also left the clause with nothing it could ever read: the
    registry row (`components.yaml`, `lineage: run.json:inputs[runs/**]`) and
    the gate both expected the manifest the handler refused to write. The
    console-row concern the old docstring raised is answered by the row's
    own `deadline: null` — an on-demand job with no deadline never pages for
    absence, and a run row for a walk an operator asked for is the walk's
    receipt, not noise.

    `--dry-run` prints the walk and files nothing (`run_job(dry_run=True)`).

    **`--verify-chain`** (`alpha-engine-config-I10625`) is checked AFTER the
    walk is printed and the manifest recorded — a broken chain is not a
    failure of the walk itself (`explain` "runs" successfully either way;
    the walk is what lets an operator SEE the break), so it never turns this
    run's own manifest into a `failed` one. It is the caller's refusal:
    `crucible.explain.explain` already sets `Lineage.chain` to a
    `ChainVerification` whenever the walk crosses the money path
    (`alpha-engine-config-I10414`, plan §9.5) and leaves it `None` otherwise;
    `--verify-chain` calls `chain.raise_if_broken()` when a chain was
    computed, which is a no-op on an intact chain and a non-zero exit naming
    the break otherwise.

    **`--select-newest-verdict`** (`alpha-engine-config-I10858`) is the
    scheduled arc stage's own affordance: `components.yaml`'s `explain` row
    joined the Saturday arc (`dispatch: arc`, after `experiment.grade`) so
    the phase-1 clause `explain_walks_a_verdict` — a ROLLING window — has a
    producer on a cadence shorter than its window, instead of relying on an
    operator's memory the way its one qualifying manifest (2026-08-07) did
    before it aged out. Mutually exclusive with a positional `target`: an
    arc stage names no target on its own argv (`crucible.weekly.Stage.argv`),
    and an operator who typed both meant one or the other, not "pick
    whichever". `crucible.explain.select_newest_settled_verdict` does the
    selection and raises `NoSettledVerdictError` when the store holds no
    verdict at all — inside `job()` below, so that failure still files a
    `failed` manifest (rule 1) rather than dying before `run_job` opens one.
    """
    config = _settings(args)
    store = config.store()
    select_newest = bool(getattr(args, "select_newest_verdict", False))
    target = args.target
    if select_newest and target:
        raise SystemExit("--select-newest-verdict and an explicit target are mutually exclusive")
    if not select_newest and not target:
        raise SystemExit(
            "explain requires a target (RUN_ID|VERDICT_KEY) or --select-newest-verdict"
        )
    captured: dict[str, Any] = {}

    def job(ctx: Any) -> None:
        walk_target = select_newest_settled_verdict(store) if select_newest else target
        if select_newest:
            print(f"selected newest settled verdict: {walk_target}")
        lineage = explain_lineage(store, walk_target)
        captured["lineage"] = lineage
        for key in _lineage_keys(lineage):
            if store.exists(key):
                ctx.record_input(key, store.get_bytes(key))
        print(render_lineage(lineage))

    run_job(
        "explain",
        job,
        store=store,
        trading_day=args.trading_day,
        run_mode=getattr(args, "run_mode", None),
        dry_run=bool(getattr(args, "dry_run", False)),
    )
    if getattr(args, "verify_chain", False):
        _verify_chain_or_refuse(captured["lineage"])
    return 0


def _verify_chain_or_refuse(lineage: Any) -> None:
    """`--verify-chain`'s check, isolated so its exit shape is one place.

    `lineage.chain` is `None` for a walk that never crossed the money path
    (nothing to verify — not an error) or the `ChainVerification` computed
    by `crucible.explain.explain` over the whole store. `raise_if_broken()`
    is a no-op when it verified `ok` and an uncaught
    `crucible.manifest.MoneyPathChainError` — non-zero exit, reason attached
    — when it did not.
    """
    chain = lineage.chain
    if chain is not None:
        chain.raise_if_broken()


HANDLERS = {
    "data.daily": handle_data_daily,
    "data.weekly": handle_data_weekly,
    "data.heal": handle_data_heal,
    "experiment.new": handle_experiment_new,
    "experiment.register": handle_experiment_register,
    "experiment.run": handle_experiment_run,
    "experiment.backfill": handle_experiment_backfill,
    "experiment.grade": handle_experiment_grade,
    "explain": handle_explain,
}


def add_track_a_arguments(name: str, sub: argparse.ArgumentParser) -> None:
    """Track-A-only flags for one subcommand."""
    if name in ("data.daily", "data.weekly", "data.heal"):
        sub.add_argument(
            "--source",
            default="arctic",
            help="Price source. `arctic` is the only production source; it never falls back.",
        )
        # alpha-engine-config-I10457: the dedicated-library override.
        sub.add_argument(
            "--arctic-library",
            dest="arctic_library",
            default=None,
            help=(
                "Dedicated ArcticDB library name to read instead of the production "
                "`universe` library. ADDITIVE: absent, behaviour is unchanged "
                "production. Never set outside the integration test tier, which "
                "resolves it from CRUCIBLE_INTEGRATION_ARCTIC_LIBRARY via "
                "crucible.required.require_env — that is the only caller that "
                "declares this flag."
            ),
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
    if name == "experiment.backfill":
        sub.add_argument("--from", dest="from_date", required=True, metavar="YYYY-MM-DD")
        sub.add_argument("--to", dest="to_date", required=True, metavar="YYYY-MM-DD")
        sub.add_argument(
            "--force",
            action="store_true",
            help=(
                "Reproduce a session whose three artifacts already exist. Without it "
                "an already-produced session is skipped and reported as such, which is "
                "what makes an interrupted backfill resumable by rerunning it."
            ),
        )
        sub.add_argument(
            "--i-am-in-region",
            action="store_true",
            help=(
                "Override the in-region guard. The ONLY override — same rule and same "
                "reason as `data.heal`: the failure mode is an operator in a hurry."
            ),
        )
    if name in (
        "experiment.new",
        "experiment.register",
        "experiment.run",
        "experiment.backfill",
        "experiment.grade",
    ):
        sub.add_argument(
            "--strategy-dir",
            help=(
                "Checkout of alpha-engine-config/strategy/. Absent, arms are read from "
                "the strategy tree synced into the store."
            ),
        )
