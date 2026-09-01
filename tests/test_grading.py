"""Grading: the I9745 shape, the control ordering, and replay determinism.

These are the three claims the whole harness rests on. The first is the
defect v1 shipped for months; the second is the only evidence the grader
measures anything at all; the third is what makes a replay diff mean
something.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest
from conftest import sessions_ending
from nousergon_lib.arena.window import ArmSeries, pair_on_common_window

from crucible.config import Settings
from crucible.data import run_daily
from crucible.keys import arena_cycle_key, shadow_key, verdict_key
from crucible.runner import run_job
from crucible.slots import universe
from crucible.slots.cycle import MissingArtifactError
from crucible.slots.grading import (
    CONTROL_PLANTED_IC,
    GraderControlError,
    assert_controls_ordered,
    control_selection,
    forward_returns,
    score_selection,
)

#: Enough decision dates that a 21-session horizon has settled for several of
#: them by the cycle date, so the ladder has more than one rung.
DECISION_DATES = 6
HORIZON = 21


def _control_ids() -> dict[str, str]:
    """Kind -> the control's REGISTERED arm id, as the cycle addresses it."""
    from crucible.slots import get_slot
    from crucible.slots.arms import control_specs

    return {c.control_kind: c.arm_id for c in control_specs(get_slot("u"))}


def _settings(strategy_dir, store_root) -> Settings:
    return Settings(
        store_uri=str(store_root),
        arctic_bucket="unused-in-this-test",
        strategy_dir=strategy_dir,
        origins={"store_uri": "test", "strategy_dir": "test"},
    )


def _seed_cycle(store, source, strategy_dir, cycle_date, tmp_path):
    """Compile the panel, produce shadows on several past dates, and return settings."""
    settings = _settings(strategy_dir, tmp_path / "store")
    sessions = sessions_ending(cycle_date, HORIZON + DECISION_DATES + 1)
    decision_days = sessions[:DECISION_DATES]

    for day in decision_days + [cycle_date]:
        run_job(
            "data.daily",
            lambda c: run_daily(c, source=source),
            store=store,
            trading_day=day,
        )
    for day in decision_days:
        run_job(
            "experiment.run",
            lambda c: universe.produce(c, settings=settings),
            store=store,
            trading_day=day,
        )
    return settings, decision_days


class TestScoreIsCountMatchedAgainstThePopulation:
    def test_alpha_is_the_selection_mean_minus_the_population_mean(self) -> None:
        returns = {"A": 0.10, "B": 0.05, "C": -0.05, "D": -0.10}
        score, detail = score_selection(("A", "B"), tuple(returns), returns)
        assert score == pytest.approx(0.075 - 0.0)
        assert detail["n_selected_settled"] == 2
        assert detail["n_population_settled"] == 4

    def test_an_unsettled_name_is_excluded_not_scored_as_flat(self) -> None:
        returns = {"A": 0.10, "B": 0.05, "C": -0.05}
        score, detail = score_selection(("A", "DELISTED"), tuple(returns), returns)
        assert detail["n_selected"] == 2
        assert detail["n_selected_settled"] == 1
        assert score == pytest.approx(0.10 - (0.10 + 0.05 - 0.05) / 3)

    def test_a_selection_with_nothing_settled_raises_rather_than_scoring_zero(self) -> None:
        with pytest.raises(ValueError, match="cannot be scored"):
            score_selection(("GONE",), ("A", "B"), {"A": 0.1, "B": 0.2})

    def test_an_unsettled_horizon_raises_and_names_what_is_missing(
        self, source, cycle_date
    ) -> None:
        panel = source.load_panel(end=cycle_date, lookback_days=1200)
        with pytest.raises(ValueError, match="has not settled"):
            forward_returns(panel, start=cycle_date, horizon_trading_days=HORIZON)


class TestI9745NoCrossArmIntersection:
    def test_an_arm_with_no_shadow_does_not_null_another_arms_paired_figure(self) -> None:
        """The v1 defect, reproduced as a shape and refuted.

        `apply_cohort_intersection` recomputed every arm's figure over the
        dates on which EVERY registered arm had scored. One arm with no
        shadow at all emptied that intersection, and every other arm's
        `topn_alpha_vs_champion` went null with `no_common_cohort` — including
        arms that shared full history with the champion.
        """
        champion = ArmSeries(
            arm_id="u:champion:aaa", scores={"2026-08-03": 0.01, "2026-08-04": 0.02}
        )
        challenger = ArmSeries(
            arm_id="u:challenger:bbb", scores={"2026-08-03": 0.03, "2026-08-04": 0.01}
        )
        starved = ArmSeries(arm_id="u:starved:ccc", scores={})

        window = pair_on_common_window(challenger, champion, min_dates=1)
        assert window.measurable
        assert window.n_dates == 2
        assert window.mean_diff == pytest.approx(0.005)

        # The starved arm is still SUPPLIED. Its presence must change nothing
        # about the pair above — which is the whole of the fix.
        starved_window = pair_on_common_window(starved, champion, min_dates=1)
        assert not starved_window.measurable
        assert window.mean_diff == pytest.approx(0.005)

    def test_a_starved_arm_reaches_the_cycle_with_an_empty_series_not_a_zero(
        self, store, source, strategy_dir, cycle_date, tmp_path
    ) -> None:
        settings, decision_days = _seed_cycle(store, source, strategy_dir, cycle_date, tmp_path)
        # Remove one arm's shadows entirely: the I9745 trigger.
        from crucible.slots.arms import load_arm_specs

        specs = load_arm_specs("u", strategy_dir=strategy_dir)
        starved = specs[0]
        for day in decision_days:
            path = tmp_path / "store" / shadow_key(starved.arm_id, day.isoformat())
            path.unlink()

        result: dict = {}

        def job(ctx):
            result.update(universe.grade(ctx, settings=settings))

        run_job("experiment.grade", job, store=store, trading_day=cycle_date)

        cycle = json.loads(store.get_bytes(arena_cycle_key("u", cycle_date.isoformat())))
        ladders = {ladder["arm_id"]: ladder for ladder in cycle["ladders"]}
        assert starved.arm_id in ladders, "a starved arm is scored, not dropped"

        survivor_ids = {s.arm_id for s in specs if s.arm_id != starved.arm_id}
        verdicts = cycle["ranking"]["verdicts"]

        starved_pairs = [
            v for v in verdicts if starved.arm_id in (v["arm_a"], v["arm_b"]) and v["n_dates"] > 0
        ]
        assert not starved_pairs, (
            "the starved arm shares no dates with anything, so every pair it is in must "
            "be unmeasurable"
        )

        survivor_pairs = [
            v for v in verdicts if v["arm_a"] in survivor_ids and v["arm_b"] in survivor_ids
        ]
        assert survivor_pairs, "the surviving arms must still be compared with each other"
        assert all(v["n_dates"] > 0 for v in survivor_pairs), (
            "every pair between two arms that both produced must keep its own window. "
            "If the starved arm nulled them, this is I9745 all over again: v1 "
            "recomputed every figure over the dates on which EVERY registered arm had "
            "scored, so one arm with no shadow emptied the lot."
        )
        assert all(v["unmeasurable_reason"] is None for v in survivor_pairs)


class TestControlArms:
    def test_the_planted_control_outranks_the_null_control(self) -> None:
        returns = {f"T{i:03d}": (i - 50) / 500.0 for i in range(100)}
        planted = control_selection("planted", returns, top_n=10, seed=20260828)
        null = control_selection("null", returns, top_n=10, seed=20260828)
        planted_score, _ = score_selection(planted, tuple(returns), returns)
        null_score, _ = score_selection(null, tuple(returns), returns)
        assert planted_score > null_score, (
            f"the planted arm's signal is constructed with an IC of {CONTROL_PLANTED_IC} "
            "against the realized return; a grader that cannot see it cannot see a real "
            "edge either"
        )
        assert planted_score > 0

    def test_the_controls_are_seeded_so_a_replay_reproduces_them(self) -> None:
        returns = {f"T{i:03d}": (i - 50) / 500.0 for i in range(100)}
        first = control_selection("planted", returns, top_n=10, seed=42)
        second = control_selection("planted", returns, top_n=10, seed=42)
        assert first == second

    def test_a_broken_grader_voids_the_cycle_rather_than_publishing_it(self) -> None:
        """The negative result the control exists to be able to produce."""
        control_ids = _control_ids()
        planted, null = control_ids["planted"], control_ids["null"]
        inverted = {
            planted: ArmSeries(arm_id=planted, scores={"2026-08-03": -0.05}),
            null: ArmSeries(arm_id=null, scores={"2026-08-03": 0.05}),
        }
        with pytest.raises(GraderControlError, match="did not outrank"):
            assert_controls_ordered(control_ids, inverted, slot="u")

    def test_an_unscored_control_is_an_unverified_grader(self) -> None:
        control_ids = _control_ids()
        planted = control_ids["planted"]
        with pytest.raises(GraderControlError, match="were not scored this cycle"):
            assert_controls_ordered(
                control_ids,
                {planted: ArmSeries(arm_id=planted, scores={"2026-08-03": 0.05})},
                slot="u",
            )

    def test_a_control_never_takes_the_pointer(
        self, store, source, strategy_dir, cycle_date, tmp_path
    ) -> None:
        settings, _ = _seed_cycle(store, source, strategy_dir, cycle_date, tmp_path)
        result: dict = {}
        run_job(
            "experiment.grade",
            lambda c: result.update(universe.grade(c, settings=settings)),
            store=store,
            trading_day=cycle_date,
        )
        champion = result["pointer"]["champion"]
        assert champion not in set(_control_ids().values())
        ineligible = result["pointer"]["ineligible"]
        assert any("control" in str(v) for v in ineligible.values()), (
            "the refusal must appear in the cycle artifact, not only in code"
        )


class TestReplayDeterminism:
    def test_grading_the_same_date_twice_reproduces_the_verdict(
        self, store, source, strategy_dir, cycle_date, tmp_path
    ) -> None:
        settings, decision_days = _seed_cycle(store, source, strategy_dir, cycle_date, tmp_path)
        run_job(
            "experiment.grade",
            lambda c: universe.grade(c, settings=settings),
            store=store,
            trading_day=cycle_date,
        )
        first = store.get_bytes(arena_cycle_key("u", cycle_date.isoformat())).decode()
        first_verdicts = {
            k: store.get_bytes(k)
            for k in store.list_keys("experiments/")
            if k.endswith("verdict.json")
        }

        run_job(
            "experiment.grade",
            lambda c: universe.grade(c, settings=settings),
            store=store,
            trading_day=cycle_date,
        )
        second = store.get_bytes(arena_cycle_key("u", cycle_date.isoformat())).decode()
        second_verdicts = {
            k: store.get_bytes(k)
            for k in store.list_keys("experiments/")
            if k.endswith("verdict.json")
        }

        assert first_verdicts == second_verdicts, "a replayed verdict must be byte-identical"
        # `as_of` is the only thing that legitimately differs between two runs
        # of the same cycle, and it does not: the artifact is keyed by it.
        assert (
            json.loads(first)["decision"]["champion"] == json.loads(second)["decision"]["champion"]
        )
        assert json.loads(first)["ladders"] == json.loads(second)["ladders"]


class TestKeysAndRefusals:
    def test_every_key_the_slot_writes_binds_to_a_trading_day(
        self, store, source, strategy_dir, cycle_date, tmp_path
    ) -> None:
        settings, _ = _seed_cycle(store, source, strategy_dir, cycle_date, tmp_path)
        run_job(
            "experiment.grade",
            lambda c: universe.grade(c, settings=settings),
            store=store,
            trading_day=cycle_date,
        )
        store.assert_keys_bind_to_trading_days()

    def test_a_missing_feature_layer_names_the_key_and_the_command(
        self, store, strategy_dir, cycle_date, tmp_path
    ) -> None:
        settings = _settings(strategy_dir, tmp_path / "store")
        with pytest.raises(MissingArtifactError) as excinfo:
            run_job(
                "experiment.run",
                lambda c: universe.produce(c, settings=settings),
                store=store,
                trading_day=cycle_date,
            )
        assert "features/" in str(excinfo.value)
        assert "crucible data.daily --date" in str(excinfo.value)

    def test_nothing_settled_fails_rather_than_publishing_an_empty_cycle(
        self, store, source, strategy_dir, cycle_date, tmp_path
    ) -> None:
        settings = _settings(strategy_dir, tmp_path / "store")
        run_job(
            "data.daily",
            lambda c: run_daily(c, source=source),
            store=store,
            trading_day=cycle_date,
        )
        run_job(
            "experiment.run",
            lambda c: universe.produce(c, settings=settings),
            store=store,
            trading_day=cycle_date,
        )
        with pytest.raises(MissingArtifactError, match="has settled"):
            run_job(
                "experiment.grade",
                lambda c: universe.grade(c, settings=settings),
                store=store,
                trading_day=cycle_date,
            )


class TestVerdictArtifacts:
    def test_a_verdict_carries_its_horizon_and_benchmark(
        self, store, source, strategy_dir, cycle_date, tmp_path
    ) -> None:
        settings, decision_days = _seed_cycle(store, source, strategy_dir, cycle_date, tmp_path)
        run_job(
            "experiment.grade",
            lambda c: universe.grade(c, settings=settings),
            store=store,
            trading_day=cycle_date,
        )
        from crucible.slots.arms import load_arm_specs

        arm = load_arm_specs("u", strategy_dir=strategy_dir)[0]
        settled = decision_days[0].isoformat()
        verdict = json.loads(store.get_bytes(verdict_key(arm.arm_id, settled)))
        assert verdict["horizon_trading_days"] == HORIZON
        assert verdict["benchmark"] == "population", "never SPY for a selection slot"
        assert isinstance(verdict["score_ratio"], float)
        assert dt.date.fromisoformat(verdict["trading_day"]) == decision_days[0]
