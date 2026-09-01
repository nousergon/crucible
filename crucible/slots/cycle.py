"""The produce and grade drivers. One engine, used by every slot.

Normative source: plan §4.4 ("four slots, one engine") and §10 component 1.

`crucible/slots/universe.py` and `crucible/slots/research.py` are bindings
over these two functions: they supply the slot key and the key of the feed
the slot's champion writes, and nothing else. A second copy of this loop per
slot is exactly the shape the plan replaces — four drifting implementations
of one policy.

**Produce** (`experiment.run`) reads the feature layer for one trading day,
runs every registered arm's recipe, and writes each arm's selection. It
knows no outcome, and it writes no verdict.

**Grade** (`experiment.grade`) reads the price panel at the cycle date,
finds every shadow whose horizon has since settled, scores each into a
verdict, checks the controls, and hands the assembled series to
`nousergon_lib.arena.engine.run_cycle`. It writes the `arena_cycle`
artifact and one trial-ledger row per graded arm-cycle.

It does NOT move the pointer. `promote` is a separate job with its own
manifest, so "the evidence said X" and "the pointer moved to X" are two
records and a disagreement between them is visible.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import TYPE_CHECKING, Any

from nousergon_lib.arena.engine import ServingPrecondition
from nousergon_lib.arena.window import ArmSeries

from crucible.calendar import assert_trading_day
from crucible.config import Settings
from crucible.features import DEFAULT_FEATURE_VERSION, read_features
from crucible.keys import (
    arena_cycle_key,
    champion_key,
    data_panel_key,
    features_key,
    shadow_key,
    verdict_key,
)
from crucible.ledger import append_trials, trial_rows
from crucible.slots import get_slot, promotable_arms
from crucible.slots.arms import (
    control_specs,
    load_arm_specs,
    read_register,
    register_arms,
    write_register,
)
from crucible.slots.grading import (
    DEFAULT_HORIZON_TRADING_DAYS,
    ForwardReturnWindow,
    GraderControlError,
    PopulationIntegrityError,
    SelectionMissError,
    ShadowSelection,
    assert_label_control,
    control_selection,
    forward_returns,
    grade_slot,
    produce_shadow,
    raise_training_integrity,
    reference_forward_returns,
    score_selection,
    training_ok,
    write_shadow,
    write_verdict,
)
from crucible.slots.rankers import MissingFeatureError

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from crucible.runner import RunContext
    from crucible.store import Store

__all__ = ["MissingArtifactError", "run_grade", "run_produce"]


class MissingArtifactError(RuntimeError):
    """A required upstream artifact is absent. Its exact key is in the message.

    Never a fabricated substitute and never a skip: the operator's next
    action is to produce that key, and a reason that does not name it sends
    them looking.
    """


def _read_features(store: Store, version: str, trading_day: dt.date) -> pd.DataFrame:
    key = features_key(version, trading_day.isoformat())
    if not store.exists(key):
        raise MissingArtifactError(
            f"the feature layer is absent at {key}. U and R read this artifact and "
            "nothing else (§10.4); compile it with:\n"
            f"    crucible data.daily --date {trading_day}"
        )
    return read_features(store.get_bytes(key))


def _read_panel(store: Store, trading_day: dt.date) -> pd.DataFrame:
    import io

    import pandas as pd

    key = data_panel_key(trading_day.isoformat())
    if not store.exists(key):
        raise MissingArtifactError(
            f"the price panel is absent at {key}; forward returns are computed from it. "
            f"Compile it with:\n    crucible data.daily --date {trading_day}"
        )
    return pd.read_parquet(io.BytesIO(store.get_bytes(key)))


def _incumbent(store: Store, slot: str) -> str | None:
    key = champion_key(slot)
    if not store.exists(key):
        return None
    return json.loads(store.get_bytes(key).decode("utf-8")).get("champion")


def _shadow_dates(store: Store, arm_id: str) -> list[str]:
    """Every trading day this arm has a shadow for, ascending.

    Listed from the store rather than derived from a date range: an arm
    registered mid-window has no shadow before its registration, and
    inventing the dates would turn its absence into a run of misses.
    """
    from crucible.keys import arm_key_segment

    prefix = f"experiments/{arm_key_segment(arm_id)}/"
    days = []
    for key in store.list_keys(prefix):
        if key.endswith("/shadow.json"):
            days.append(key[len(prefix) :].split("/", 1)[0])
    return sorted(days)


def run_produce(
    ctx: RunContext,
    *,
    slot: str,
    settings: Settings,
    feed_key_for: Any,
    arm_name: str | None = None,
    feature_version: str = DEFAULT_FEATURE_VERSION,
) -> dict[str, Any]:
    """Run every registered arm's recipe for one trading day and write the shadows."""
    slot_spec = get_slot(slot)
    trading_day = ctx.trading_day
    assert_trading_day(trading_day, context=f"experiment.run --slot {slot} --date {trading_day}")

    features = _read_features(ctx.store, feature_version, trading_day)
    ctx.record_input(
        features_key(feature_version, trading_day.isoformat()),
        ctx.store.get_bytes(features_key(feature_version, trading_day.isoformat())),
        schema_version="features.v1",
    )

    specs = load_arm_specs(slot, store=ctx.store, strategy_dir=settings.strategy_dir)
    if arm_name is not None:
        specs = [s for s in specs if s.name == arm_name]
        if not specs:
            raise MissingArtifactError(
                f"no arm named {arm_name!r} in slot {slot!r}. Producing nothing and "
                "exiting 0 would be indistinguishable from an arm that ran and selected "
                "nothing."
            )

    register = read_register(ctx.store, slot)
    register, _ = register_arms(register, specs + control_specs(slot_spec))
    write_register(ctx.store, slot, register)

    produced: list[ShadowSelection] = []
    for spec in specs:
        try:
            shadow = produce_shadow(spec, features, trading_day)
        except MissingFeatureError as exc:
            # plan §4.4: a defective input fails the whole slot's run. It is
            # never a miss — "this arm had nothing to say" and "this arm's
            # inputs were broken" must not render identically.
            raise_training_integrity(spec.arm_id, exc)
        else:
            write_shadow(ctx.store, shadow)
            ctx.record_output(
                shadow_key(shadow.arm_id, shadow.trading_day),
                json.dumps(shadow.to_dict(), indent=2, sort_keys=True).encode("utf-8"),
                schema_version="shadow.v1",
            )
            produced.append(shadow)

    champion = _incumbent(ctx.store, slot)
    feed_written = None
    if champion is not None:
        served = next((s for s in produced if s.arm_id == champion), None)
        if served is None:
            raise MissingArtifactError(
                f"the champion pointer for slot {slot!r} names {champion!r}, which produced "
                "no shadow this cycle. The serving path resolves the pointer — it never "
                "imports a ranking function directly — so a pointer to an arm that did "
                "not produce means production has no feed today."
            )
        feed_written = feed_key_for(trading_day.isoformat())
        ctx.record_output(
            feed_written,
            json.dumps(
                {
                    "schema_version": "feed.v1",
                    "slot": slot,
                    "trading_day": trading_day.isoformat(),
                    "champion": champion,
                    "members": list(served.selection),
                },
                indent=2,
                sort_keys=True,
            ).encode("utf-8"),
            schema_version="feed.v1",
        )

    ctx.record_rows(rows_in=int(len(features)), rows_out=len(produced))
    ctx.record_metric(
        {
            "name": "arms_produced",
            "module": f"crucible.slots.{slot}",
            "metric_type": "count",
            "value": float(len(produced)),
            "unit": "arms",
            "n_floor": 1,
            "status": "OK",
            "status_reason": (
                f"slot {slot}: {len(produced)} registered arm(s) produced a shadow for "
                f"{trading_day}; population {len(features)} names"
            ),
            "source_path": f"experiments/*/{trading_day.isoformat()}/shadow.json",
            "last_updated_utc": _utc_now(),
        }
    )
    return {
        "slot": slot,
        "trading_day": trading_day.isoformat(),
        "arms": [s.arm_id for s in produced],
        "champion": champion,
        "feed_key": feed_written,
        "feature_version": feature_version,
    }


def run_grade(
    ctx: RunContext,
    *,
    slot: str,
    settings: Settings,
    horizon_trading_days: int = DEFAULT_HORIZON_TRADING_DAYS,
    feature_version: str = DEFAULT_FEATURE_VERSION,
) -> dict[str, Any]:
    """Score every settled shadow, verify the controls, run the cycle, write it."""
    slot_spec = get_slot(slot)
    as_of = ctx.trading_day
    assert_trading_day(as_of, context=f"experiment.grade {slot} --date {as_of}")

    panel = _read_panel(ctx.store, as_of)
    ctx.record_input(
        data_panel_key(as_of.isoformat()),
        ctx.store.get_bytes(data_panel_key(as_of.isoformat())),
        schema_version="panel.v1",
    )

    specs = load_arm_specs(slot, store=ctx.store, strategy_dir=settings.strategy_dir)
    controls = control_specs(slot_spec)
    register = read_register(ctx.store, slot)
    register, _ = register_arms(register, specs + controls)
    write_register(ctx.store, slot, register)

    # Kind -> REGISTERED arm id. A control is addressed by the same
    # `{slot}:{name}:{spec_hash}` identity every other arm carries; using its
    # bare name would look up a series no register row backs.
    control_by_kind = {c.control_kind: c.arm_id for c in controls}
    control_ids = set(control_by_kind.values())
    scored_arms = register.scored_arms(as_of.isoformat(), slot_spec.retired_trailing_cycles)

    # One returns cache per settled decision date, shared by every arm: the
    # forward return of a ticker from date d does not depend on who picked it,
    # and recomputing it per arm is how two arms end up scored against two
    # slightly different benchmarks.
    returns_cache: dict[str, ForwardReturnWindow] = {}
    unsettled_days: dict[str, str] = {}
    verdicts: dict[str, dict[str, float]] = {}
    unsettled: dict[str, list[str]] = {}
    misses: dict[str, list[str]] = {}
    label_control: dict[str, dict[str, Any]] = {}

    def _returns_for(day: str) -> ForwardReturnWindow | None:
        """The settled forward-return window from ``day``, or None if unsettled.

        Cached BOTH ways. An unsettled date is remembered as unsettled, so a
        second arm holding a shadow for the same day does not re-derive the
        same refusal — and, more to the point, cannot land on a different
        answer than the first arm did. Two arms scored against two different
        benchmarks for one date is the defect this cache exists to preclude.

        **The label control runs here, once per settled date, before any arm
        is scored against the date** (§10.1). The planted/null pair cannot
        see this class: both controls are generated from and scored against
        this very mapping, so a label defect moves them identically and their
        margin survives it. `assert_label_control` recomputes the labels
        through a second construction and raises
        :class:`~crucible.slots.grading.GraderControlError` when the two
        disagree or when the span measured is not the span declared — which
        is the reproduced defect: a forced 5-session horizon published a
        clean cycle while every verdict claimed 21.
        """
        if day in returns_cache:
            return returns_cache[day]
        if day in unsettled_days:
            return None
        try:
            window = forward_returns(
                panel,
                start=dt.date.fromisoformat(day),
                horizon_trading_days=horizon_trading_days,
            )
        except ValueError as exc:
            # The horizon has not settled. NOT a miss and NOT a zero: the date
            # is not yet a measurement, and it enters the series on the first
            # cycle after it settles.
            unsettled_days[day] = str(exc)
            return None
        label_control[day] = assert_label_control(
            window,
            reference_forward_returns(
                panel,
                start=dt.date.fromisoformat(day),
                horizon_trading_days=horizon_trading_days,
            ),
            slot=slot,
            declared_horizon_trading_days=horizon_trading_days,
        )
        returns_cache[day] = window
        return window

    for arm_id in scored_arms:
        if arm_id in control_ids:
            continue
        verdicts.setdefault(arm_id, {})
        for day in _shadow_dates(ctx.store, arm_id):
            window = _returns_for(day)
            if window is None:
                unsettled.setdefault(arm_id, []).append(day)
                continue
            shadow = json.loads(ctx.store.get_bytes(shadow_key(arm_id, day)).decode("utf-8"))
            try:
                score, detail = score_selection(
                    tuple(shadow["selection"]), tuple(shadow["population"]), window.returns
                )
            except SelectionMissError:
                # plan §4.4 / policy §3: "a cycle in which an arm legitimately
                # selects nothing is a MISS", and a miss is data. Every name
                # this arm picked delisted or was halted, while the population
                # it drew from is intact and every other arm scores normally.
                #
                # This is the ONLY swallow in this loop and it is not a
                # degrade: the failure mode absorbed is "one arm's picks are
                # all unscoreable on one date", the date is recorded on the
                # `misses` surface of this run's manifest and on the
                # `arm_misses` metric below, and it stays OUT of the arm's
                # series so a miss can never render as a zero. What is NOT
                # absorbed is one line down.
                misses.setdefault(arm_id, []).append(day)
                continue
            except PopulationIntegrityError as exc:
                # The other half of the distinction, and it still fails the
                # slot. The benchmark could not be formed, every arm is scored
                # against it, so the cycle's shared inputs are compromised.
                raise_training_integrity(arm_id, exc)
            else:
                verdicts[arm_id][day] = score
                write_verdict(
                    ctx.store,
                    arm_id=arm_id,
                    trading_day=day,
                    slot=slot,
                    score=score,
                    window=window,
                    benchmark=slot_spec.benchmark,
                    detail=detail,
                    control=False,
                )

    settled_days = sorted(d for d, w in returns_cache.items() if w.returns)
    if not settled_days:
        raise MissingArtifactError(
            f"slot {slot!r} has no shadow whose {horizon_trading_days}-session horizon has "
            f"settled by {as_of}. There is nothing to grade — which is a state, not a "
            "verdict, so this run FAILS rather than publishing an empty cycle. Produce "
            "shadows at least that many sessions back first:\n"
            f"    crucible experiment.run --slot {slot} --date <earlier trading day>"
        )

    # Controls are produced HERE, at grade time, on the same settled dates the
    # real arms were scored on. A planted edge is a look-ahead by
    # construction: generating it in the produce path would put a look-ahead
    # artifact on the real-time write path (§10.1).
    for control in controls:
        verdicts.setdefault(control.arm_id, {})
        for day in settled_days:
            window = returns_cache[day]
            returns = window.returns
            top_n = _control_top_n(specs)
            selection = control_selection(
                control.control_kind,
                returns,
                top_n=top_n,
                seed=int(day.replace("-", "")),
            )
            # No miss handler here, deliberately. A control draws its picks
            # FROM the settled returns, so every pick is settled by
            # construction; a `SelectionMissError` on a control would be a
            # defect in the harness itself and must reach the operator as a
            # failed run rather than as a control that quietly missed.
            score, detail = score_selection(selection, tuple(sorted(returns)), returns)
            verdicts[control.arm_id][day] = score
            detail["control_kind"] = control.control_kind
            write_verdict(
                ctx.store,
                arm_id=control.arm_id,
                trading_day=day,
                slot=slot,
                score=score,
                window=window,
                benchmark=slot_spec.benchmark,
                detail=detail,
                control=True,
            )

    series_by_arm: dict[str, ArmSeries] = {
        arm_id: ArmSeries(arm_id=arm_id, scores=scores)
        for arm_id, scores in sorted(verdicts.items())
    }
    # An arm with NO settled score is still SUPPLIED, carrying an empty
    # series. Deliberate on both counts: `run_cycle` requires a series for
    # every arm the register says to score, and the pairing happens per
    # comparison, so an arm with nothing to say cannot null another arm's
    # figure. That is I9745 closed by construction rather than by a rule.

    # §10.1: a control never serves. Expressed as a SERVING PRECONDITION —
    # the engine's own mechanism for "this arm may not take the pointer" —
    # rather than as an assumption that the engine will not pick it. The
    # planted control ranks on the realized forward return, so it would win
    # the pointer on evidence and be a look-ahead in production; the
    # precondition puts the refusal, and its reason, into the cycle artifact
    # where an operator can read it.
    preconditions = {
        control.arm_id: [
            ServingPrecondition(
                name="not_a_control_arm",
                passed=False,
                reason=(
                    f"{control.control_kind} control: scored every cycle to verify the "
                    "grader, excluded from the pointer (§10.1)"
                ),
            )
        ]
        for control in controls
    }

    cycle, control_detail = grade_slot(
        slot_spec,
        as_of=as_of,
        register=register,
        control_ids=control_by_kind,
        series_by_arm=series_by_arm,
        incumbent=_incumbent(ctx.store, slot),
        preconditions=preconditions,
        training=training_ok(list(register.active_arms())),
    )

    # §10.1: the cycle's control record is BOTH halves of the grader. The
    # planted/null margin checks the scoring half; the label control checks
    # the half that constructs the number being scored, and until it existed
    # a `verdicts void` finding could only ever come from one of the two.
    control_detail["label_control"] = {
        "dates_checked": sorted(label_control),
        "n_dates_checked": len(label_control),
        "declared_horizon_trading_days": horizon_trading_days,
        "measured_horizon_trading_days": sorted(
            {d["horizon_trading_days"] for d in label_control.values()}
        ),
        "max_relative_disagreement": max(
            (d["max_relative_disagreement"] for d in label_control.values()), default=0.0
        ),
        "per_date": {day: dict(detail) for day, detail in sorted(label_control.items())},
    }

    # The horizon EVERY verdict this run wrote actually claims, read back off
    # the measurements rather than off the argument. One value or the label
    # control would already have raised; asserted here so a future edit that
    # bypasses that control cannot quietly reintroduce a mixed-horizon run.
    measured_horizons = {w.horizon_trading_days for w in returns_cache.values()}
    if measured_horizons != {horizon_trading_days}:
        raise GraderControlError(
            f"slot {slot!r} measured horizons {sorted(measured_horizons)} while declaring "
            f"{horizon_trading_days}. A cycle whose verdicts span more than one horizon "
            "compares arms on different axes (policy §4: same benchmark and horizon "
            "across every arm in a slot), so its verdicts are void."
        )
    measured_horizon = measured_horizons.pop()

    cycle_key = arena_cycle_key(slot, as_of.isoformat())
    ctx.record_output(
        cycle_key,
        json.dumps(cycle.to_dict(), indent=2, sort_keys=True).encode("utf-8"),
        schema_version="arena_cycle.v1",
    )

    rows = trial_rows(
        cycle,
        slot=slot,
        as_of=as_of.isoformat(),
        run_id=ctx.run_id,
        control_ids=control_ids,
        series_by_arm=series_by_arm,
        arena_cycle_key=cycle_key,
    )
    append_trials(ctx.store, rows)

    ctx.record_rows(
        rows_in=len(series_by_arm),
        rows_out=sum(len(s.scores) for s in series_by_arm.values()),
    )
    ctx.record_metric(
        {
            "name": "control_margin_ratio",
            "module": f"crucible.slots.{slot}",
            "metric_type": "control",
            "value": float(control_detail["margin_ratio"]),
            "unit": "ratio",
            "n_floor": 1,
            "status": "OK",
            "status_reason": (
                f"planted control led the null control by "
                f"{control_detail['margin_ratio']:.6f} over "
                f"{control_detail['n_paired_dates']} shared date(s); the planted arm's "
                f"signal is constructed with an IC of {control_detail['planted_ic']}"
            ),
            "source_path": cycle_key,
            "last_updated_utc": _utc_now(),
            # The horizon MEASURED, not the one requested. §4.12 wants this
            # number to be a count of trading days; I9757 wants it to be a
            # count of the trading days that were actually walked.
            "horizon_trading_days": measured_horizon,
            "baseline": 0.0,
        }
    )
    ctx.record_metric(
        {
            "name": "label_control_max_disagreement_ratio",
            "module": f"crucible.slots.{slot}",
            "metric_type": "control",
            "value": float(control_detail["label_control"]["max_relative_disagreement"]),
            "unit": "ratio",
            "n_floor": 1,
            "status": "OK",
            "status_reason": (
                f"two independent label constructions agreed on "
                f"{len(label_control)} settled date(s) over a measured horizon of "
                f"{measured_horizon} session(s); the cycle declared "
                f"{horizon_trading_days}. A disagreement, or a measured horizon other "
                "than the declared one, voids the cycle (§10.1)"
            ),
            "source_path": cycle_key,
            "last_updated_utc": _utc_now(),
            "horizon_trading_days": measured_horizon,
            "baseline": 0.0,
        }
    )
    ctx.record_metric(
        {
            # A miss is DATA, so it gets a surface. Policy §3: silent absence
            # and a genuine zero must never render identically — a miss that
            # only ever appeared as a gap in a series would be exactly that.
            "name": "arm_miss_dates",
            "module": f"crucible.slots.{slot}",
            "metric_type": "count",
            "value": float(sum(len(days) for days in misses.values())),
            "unit": "arm_dates",
            "n_floor": 0,
            "status": "OK",
            "status_reason": (
                f"{sum(len(d) for d in misses.values())} arm-date(s) across "
                f"{len(misses)} arm(s) had every selected name unscoreable while the "
                "population was intact: a miss, not a failure and not a zero. Arms: "
                f"{sorted(misses) or 'none'}"
            ),
            "source_path": cycle_key,
            "last_updated_utc": _utc_now(),
        }
    )
    ctx.record_metric(
        {
            "name": "slot_pointer_status",
            "module": f"crucible.slots.{slot}",
            "metric_type": "decision",
            "value": 1.0 if cycle.decision.moved else 0.0,
            "unit": "indicator",
            "n_floor": 1,
            "status": "OK" if cycle.decision.status in ("decided", "held", "bootstrap") else "FAIL",
            "status_reason": f"{cycle.decision.status}: {cycle.decision.reason}",
            "source_path": cycle_key,
            "last_updated_utc": _utc_now(),
        }
    )

    # §10.1 as an EXCLUSION that BINDS here, not as a field this run reports.
    #
    # `promotable_arms` is the slot registry's filter and stays the shared
    # implementation of the rule. What it cannot know is the identity a
    # control actually carries in the register: an arm is
    # `{slot}:{name}:{spec_hash}` everywhere it is scored — `u:control_null_u:
    # 082d1c6a2c86` — while the slot spec holds the bare name `control_null_u`.
    # This call site is the one place that holds BOTH, because it derived the
    # registered ids from `control_specs` to score them, so it passes what it
    # knows rather than leaving the exclusion to a filter matching on the
    # other half of the identity. Once the registry filter resolves registered
    # ids itself this intersection is a no-op, which is the correct end state:
    # the same names removed twice, never a control removed by neither.
    promotable = [
        a for a in promotable_arms(slot_spec, list(cycle.active_arms)) if a not in control_ids
    ]
    leaked = sorted(set(promotable) & control_ids)
    if leaked:
        raise GraderControlError(
            f"control arm(s) {leaked} reached the promotion pool for slot {slot!r}. The "
            "planted control ranks on the realized forward return, so an arm of this "
            "kind in the promotable pool is a look-ahead one promotion away from "
            "production (§10.1)."
        )

    return {
        "slot": slot,
        "as_of": as_of.isoformat(),
        "arena_cycle_key": cycle_key,
        "scored_arms": list(cycle.scored_arms),
        "active_arms": list(cycle.active_arms),
        "promotable_arms": promotable,
        "settled_dates": settled_days,
        "horizon_trading_days": measured_horizon,
        "unsettled": {a: sorted(d) for a, d in unsettled.items()},
        # A miss is recorded, never inferred from a gap. `unsettled` and
        # `misses` are separate keys because they are separate events: the
        # first is "not yet a measurement", the second is "measured, and this
        # arm had nothing scoreable to say" (plan §4.4).
        "misses": {a: sorted(d) for a, d in misses.items()},
        "pointer": cycle.decision.to_dict(),
        "controls": control_detail,
        "trial_rows_appended": len(rows),
        # A DECLARED divergence from §10.1, not an oversight. Controls are
        # excluded from the pointer here (above) but the installed
        # `nousergon_lib.arena` has no notion of a control arm, so
        # `evaluate_retirements` still counts a control's pairwise win
        # against a real arm toward the cap of five. The correct fix is a
        # `control` flag on the library's ArmRecord so the engine skips them
        # — a nousergon-lib change, tracked; `promote` (track B) is what
        # APPLIES a retirement, so nothing acts on this in track A.
        "controls_counted_in_retirement_cap": True,
        "verdict_keys": [
            verdict_key(arm, day)
            for arm, scores in sorted(verdicts.items())
            for day in sorted(scores)
        ],
    }


def _control_top_n(specs: list[Any], default: int = 10) -> int:
    """Controls select as many names as the arms they are checking.

    A control drawing a different count would be compared on a different
    axis: the count-matched benchmark is exact for any size, but a 5-name
    control and a 50-name arm have different dispersion, and the margin
    between them would partly measure that.
    """
    counts = {int(s.params.get("top_n", default)) for s in specs}
    if len(counts) == 1:
        return counts.pop()
    return max(counts) if counts else default


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
