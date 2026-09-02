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
    CROSS_SECTION_MIN_NAMES,
    ForwardReturnWindow,
    GraderControlError,
    PopulationIntegrityError,
    RankICSkip,
    ScoredCrossSection,
    SelectionMissError,
    assert_controls_ordered,
    assert_label_control,
    control_selection,
    cross_section_key,
    cross_section_settled_key,
    forward_returns,
    produce_cross_section,
    reference_forward_returns,
    score_selection,
    settle_cross_section,
    spearman_ic,
    write_cross_section,
    write_cross_section_settled,
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
            lambda c: run_daily(c, source=source, expected_symbols=source.symbols()),
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

    def test_a_selection_with_nothing_settled_is_a_MISS_not_a_slot_failure(self) -> None:
        """Rewritten (§11 risk 1). The previous assertion was
        `pytest.raises(ValueError)`, which is true of any implementation that
        refuses at all — including the one that took the whole slot down. The
        claim under test is the DISTINCTION: an arm whose picks are all
        unscoreable, against an intact population, is a miss."""
        with pytest.raises(SelectionMissError) as exc:
            score_selection(("GONE",), ("A", "B", "C"), {"A": 0.1, "B": 0.2, "C": 0.0})
        assert "MISS" in str(exc.value)
        assert not isinstance(exc.value, PopulationIntegrityError)

    def test_an_unusable_population_is_compromised_inputs_not_a_miss(self) -> None:
        """The other half. Every arm in the slot is benchmarked against the
        population, so a population that cannot form a benchmark is a defect
        in the cycle's shared inputs — plan §4.4, and it still fails the run."""
        with pytest.raises(PopulationIntegrityError, match="not a benchmark"):
            score_selection(("A",), ("A",), {"A": 0.1})

    def test_the_population_is_judged_before_the_selection(self) -> None:
        """Both conditions hold at once when the population has collapsed.
        Reporting that as a miss would file broken inputs under "this arm had
        nothing to say", which is the confusion policy §3 forbids."""
        with pytest.raises(PopulationIntegrityError):
            score_selection(("GONE",), ("A",), {"A": 0.1})

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
        # If the starved arm nulled them, this is alpha-engine-config-I9745 all over
        # again: v1 recomputed every figure over the dates on which EVERY registered
        # arm had scored, so one arm with no shadow emptied the lot. Historical
        # citation, not a phase pointer (alpha-engine-config-I9839).
        assert all(v["n_dates"] > 0 for v in survivor_pairs), (
            "every pair between two arms that both produced must keep its own window."
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


class TestLabelControl:
    """§10.1 over the half of the grader the planted/null pair cannot see.

    Both controls are generated from and scored against the same `returns`
    mapping the real arms are scored against, so a defect in
    :func:`forward_returns` moves planted and null identically and their
    margin survives it. Reproduced before the fix: forcing a 5-session
    horizon published a clean cycle at `control margin=0.025838, n_paired=6`
    while every `verdict.json` in the run claimed `horizon_trading_days: 21`.
    """

    def test_the_two_label_constructions_agree_on_a_healthy_panel(self, source, cycle_date) -> None:
        """The control's positive case: it must not fire on a correct run, or
        it would be a gate nobody could leave switched on."""
        panel = source.load_panel(end=cycle_date, lookback_days=1200)
        start = sessions_ending(cycle_date, HORIZON + 2)[0]
        window = forward_returns(panel, start=start, horizon_trading_days=HORIZON)
        reference = reference_forward_returns(panel, start=start, horizon_trading_days=HORIZON)
        detail = assert_label_control(
            window, reference, slot="u", declared_horizon_trading_days=HORIZON
        )
        assert detail["n_names"] > 0
        assert detail["horizon_trading_days"] == HORIZON
        assert detail["max_relative_disagreement"] == pytest.approx(0.0, abs=1e-12)

    def test_a_horizon_that_is_not_the_declared_one_voids_the_cycle(
        self, source, cycle_date
    ) -> None:
        """The reproduced defect, at the unit: labels measured over 5 sessions
        while the cycle declares 21. Before the fix the horizon was a literal
        passed beside the returns, so nothing in the system could disagree."""
        panel = source.load_panel(end=cycle_date, lookback_days=1200)
        start = sessions_ending(cycle_date, HORIZON + 2)[0]
        short = forward_returns(panel, start=start, horizon_trading_days=5)
        reference = reference_forward_returns(panel, start=start, horizon_trading_days=HORIZON)
        with pytest.raises(GraderControlError, match="span 5 session"):
            assert_label_control(short, reference, slot="u", declared_horizon_trading_days=HORIZON)

    def test_a_corrupted_label_value_voids_the_cycle(self, source, cycle_date) -> None:
        """The class where the session count is right and the NUMBERS are
        wrong — a close paired with the wrong session, a pivot aggregating
        duplicate rows. The planted/null margin is blind to it because both
        controls read the corrupted mapping."""
        panel = source.load_panel(end=cycle_date, lookback_days=1200)
        start = sessions_ending(cycle_date, HORIZON + 2)[0]
        window = forward_returns(panel, start=start, horizon_trading_days=HORIZON)
        reference = reference_forward_returns(panel, start=start, horizon_trading_days=HORIZON)
        victim = sorted(window.returns)[0]
        corrupted = dict(window.returns)
        corrupted[victim] = corrupted[victim] + 0.01
        with pytest.raises(GraderControlError, match="disagree by"):
            assert_label_control(
                ForwardReturnWindow(
                    start=window.start,
                    end=window.end,
                    horizon_trading_days=window.horizon_trading_days,
                    returns=corrupted,
                ),
                reference,
                slot="u",
                declared_horizon_trading_days=HORIZON,
            )

    def test_a_name_present_in_only_one_construction_voids_the_cycle(self) -> None:
        """A silently dropped or silently invented ticker means the selection
        and its benchmark were drawn from different cross-sections."""
        window = ForwardReturnWindow(
            start="2026-07-01",
            end="2026-07-31",
            horizon_trading_days=21,
            returns={"A": 0.1, "B": 0.2},
        )
        with pytest.raises(GraderControlError, match="WHICH names settled"):
            assert_label_control(window, {"A": 0.1}, slot="u", declared_horizon_trading_days=21)

    def test_a_mutated_label_horizon_fails_the_whole_grade_run(
        self, store, source, strategy_dir, cycle_date, tmp_path, monkeypatch
    ) -> None:
        """The mutation harness's own case, driven through the real cycle.

        Before the fix this published a clean cycle: `control margin` positive,
        `n_paired=6`, verdicts written, pointer decided — and every verdict
        claiming a 21-session horizon it had not measured. The run must now
        FAIL rather than publish."""
        settings, _ = _seed_cycle(store, source, strategy_dir, cycle_date, tmp_path)

        from crucible.slots import cycle as cycle_module

        real = cycle_module.forward_returns

        def five_session_labels(panel, *, start, horizon_trading_days):
            return real(panel, start=start, horizon_trading_days=5)

        monkeypatch.setattr(cycle_module, "forward_returns", five_session_labels)
        with pytest.raises(GraderControlError, match="span 5 session"):
            run_job(
                "experiment.grade",
                lambda c: universe.grade(c, settings=settings),
                store=store,
                trading_day=cycle_date,
                transient_retry=False,
            )

    def test_a_verdict_states_the_horizon_THE_PANEL_actually_walked(
        self, store, source, strategy_dir, cycle_date, tmp_path
    ) -> None:
        """The horizon on a verdict is checked against the panel, not against
        the constant the test passed in.

        The previous assertion — `verdict["horizon_trading_days"] == HORIZON`
        — was true for any implementation, because that literal was written
        straight through from the caller. Here the claim is falsifiable: the
        artifact names the session its return settled at, and the count of
        panel sessions between anchor and settle must equal the horizon it
        claims."""
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

        panel = source.load_panel(end=cycle_date, lookback_days=1200)
        sessions = sorted({str(d) for d in panel["trading_day"].unique()})
        walked = sessions.index(verdict["settled_on"]) - sessions.index(verdict["trading_day"])
        assert verdict["horizon_trading_days"] == walked, (
            "the horizon a verdict claims must be a property of the returns behind it"
        )
        assert walked == HORIZON


class TestOneBrokenArmDoesNotTakeTheSlotDown:
    """`I9757` defect 6: the `ChallengerShadowGapError` shape, one step later.

    `score_selection` raised on an all-delisted selection and the `run_grade`
    call site had no handler, so the raise propagated and the whole slot
    failed: no arena cycle and no verdict for any healthy arm.
    """

    @staticmethod
    def _break_one_arm(store, tmp_path, strategy_dir, day, *, population=None):
        """Rewrite one arm's shadow so every name it picked is unscoreable."""
        from crucible.slots.arms import load_arm_specs

        specs = load_arm_specs("u", strategy_dir=strategy_dir)
        broken = specs[0]
        path = tmp_path / "store" / shadow_key(broken.arm_id, day.isoformat())
        shadow = json.loads(path.read_text())
        shadow["selection"] = ["DELISTED1", "DELISTED2"]
        if population is not None:
            shadow["population"] = population
        path.write_text(json.dumps(shadow, indent=2, sort_keys=True))
        return broken, specs

    def test_an_all_delisted_selection_is_a_miss_and_the_slot_still_grades(
        self, store, source, strategy_dir, cycle_date, tmp_path
    ) -> None:
        settings, decision_days = _seed_cycle(store, source, strategy_dir, cycle_date, tmp_path)
        day = decision_days[0]
        broken, specs = self._break_one_arm(store, tmp_path, strategy_dir, day)

        result: dict = {}
        run_job(
            "experiment.grade",
            lambda c: result.update(universe.grade(c, settings=settings)),
            store=store,
            trading_day=cycle_date,
        )

        # The miss is recorded as a miss — on a durable surface, named by arm
        # and date. Policy §3: silent absence and a genuine zero must never
        # render identically, and a gap in a series is silent absence.
        assert result["misses"] == {broken.arm_id: [day.isoformat()]}
        assert day.isoformat() not in result["unsettled"].get(broken.arm_id, []), (
            "a miss is not an unsettled horizon; the two are separate events"
        )

        # ... and it is a miss for THAT ARM ON THAT DATE only.
        cycle = json.loads(store.get_bytes(arena_cycle_key("u", cycle_date.isoformat())))
        ladders = {ladder["arm_id"]: ladder for ladder in cycle["ladders"]}
        assert broken.arm_id in ladders, "a missing arm is still scored, not dropped"
        assert not store.exists(verdict_key(broken.arm_id, day.isoformat())), (
            "an unscoreable date must not get a verdict — scoring it as zero would "
            "credit a delisting as a flat month"
        )
        for other in specs[1:]:
            assert store.exists(verdict_key(other.arm_id, day.isoformat())), (
                "one arm's miss must not cost a healthy arm its verdict for the same "
                "date: that is the whole defect"
            )
        # The arm keeps its other dates. A miss costs one observation, not a series.
        assert store.exists(verdict_key(broken.arm_id, decision_days[1].isoformat()))

    def test_a_population_that_cannot_form_a_benchmark_still_fails_the_slot(
        self, store, source, strategy_dir, cycle_date, tmp_path
    ) -> None:
        """The distinction, from the other side. Making the failure survivable
        must not make it silent: compromised inputs are a TASK FAILURE (plan
        §4.4, Brian's 2026-08-29 ruling), never a miss and never a degraded
        verdict."""
        from nousergon_lib.arena.engine import TrainingIntegrityError

        settings, decision_days = _seed_cycle(store, source, strategy_dir, cycle_date, tmp_path)
        day = decision_days[0]
        self._break_one_arm(store, tmp_path, strategy_dir, day, population=["T000"])

        with pytest.raises(TrainingIntegrityError, match="compromised"):
            run_job(
                "experiment.grade",
                lambda c: universe.grade(c, settings=settings),
                store=store,
                trading_day=cycle_date,
                transient_retry=False,
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
            lambda c: run_daily(c, source=source, expected_symbols=source.symbols()),
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
        # The horizon claim moved to
        # `TestLabelControl::test_a_verdict_states_the_horizon_THE_PANEL_actually_walked`
        # (§11 risk 1). Asserting it equals the constant this test passed in
        # was true for any implementation, because the constant was written
        # straight through to the artifact — which is the defect, not the test.
        assert verdict["benchmark"] == "population", "never SPY for a selection slot"
        assert isinstance(verdict["score_ratio"], float)
        assert dt.date.fromisoformat(verdict["trading_day"]) == decision_days[0]


class TestSpearmanIC:
    """`spearman_ic` — the formula `crucible.report`'s rank-IC rows reduce
    over settled dates (alpha-engine-config-I9778)."""

    def _names(self, n: int) -> list[str]:
        return [f"T{i:02d}" for i in range(n)]

    def test_a_perfectly_agreeing_ranking_scores_plus_one(self) -> None:
        names = self._names(CROSS_SECTION_MIN_NAMES)
        score = {t: float(i) for i, t in enumerate(names)}
        realized = {t: float(i) * 0.01 for i, t in enumerate(names)}
        ic, n = spearman_ic(score, realized)
        assert ic == pytest.approx(1.0)
        assert n == len(names)

    def test_a_perfectly_inverted_ranking_scores_minus_one(self) -> None:
        names = self._names(CROSS_SECTION_MIN_NAMES)
        score = {t: float(i) for i, t in enumerate(names)}
        realized = {t: -float(i) * 0.01 for i, t in enumerate(names)}
        ic, n = spearman_ic(score, realized)
        assert ic == pytest.approx(-1.0)
        assert n == len(names)

    def test_below_the_names_floor_returns_too_few_names(self) -> None:
        names = self._names(CROSS_SECTION_MIN_NAMES - 1)
        score = {t: float(i) for i, t in enumerate(names)}
        realized = {t: float(i) for i, t in enumerate(names)}
        assert spearman_ic(score, realized) is RankICSkip.TOO_FEW_NAMES

    def test_only_the_paired_intersection_counts_toward_the_floor(self) -> None:
        """A ticker scored but never settled (or vice versa) is unpaired and
        does not count toward `CROSS_SECTION_MIN_NAMES` — the same "absent is
        absent" rule the rest of this module holds."""
        names = self._names(CROSS_SECTION_MIN_NAMES)
        score = {t: float(i) for i, t in enumerate(names)}
        realized = {t: float(i) for i, t in enumerate(names[:-1])}  # one short
        assert spearman_ic(score, realized) is RankICSkip.TOO_FEW_NAMES

    def test_tied_scores_are_degenerate_not_a_measured_zero(self) -> None:
        """`alpha-engine-config-I9778` review, finding 1: every score tied (a
        constant ranker) makes the correlation UNDEFINED, not `0.0`. Prior to
        the fix this returned `(0.0, n)` and was counted as a real
        observation — the exact mechanism the review used to turn a `WATCH`
        row `GREEN` on a date that carried no information at all."""
        names = self._names(CROSS_SECTION_MIN_NAMES)
        score = dict.fromkeys(names, 1.0)
        realized = {t: float(i) for i, t in enumerate(names)}
        assert spearman_ic(score, realized) is RankICSkip.DEGENERATE

    def test_tied_realized_returns_are_also_degenerate(self) -> None:
        """The other side of the pairing: every realized return identical
        (e.g. a quiet cross-section) is equally undefined, not `0.0`."""
        names = self._names(CROSS_SECTION_MIN_NAMES)
        score = {t: float(i) for i, t in enumerate(names)}
        realized = dict.fromkeys(names, 0.01)
        assert spearman_ic(score, realized) is RankICSkip.DEGENERATE

    def test_a_random_shuffle_is_not_perfectly_correlated(self) -> None:
        names = self._names(8)
        score = {t: float(i) for i, t in enumerate(names)}
        shuffled_realized = list(range(8))
        shuffled_realized[0], shuffled_realized[-1] = (
            shuffled_realized[-1],
            shuffled_realized[0],
        )
        realized = {t: float(shuffled_realized[i]) for i, t in enumerate(names)}
        ic, _n = spearman_ic(score, realized)
        assert -1.0 < ic < 1.0


class TestScoredCrossSection:
    """`produce_cross_section`/`write_cross_section` — the produce-time half
    of `shadow.v2` (alpha-engine-config-I9778)."""

    def test_produce_ranks_the_whole_population_not_only_top_n(
        self, store, source, strategy_dir, cycle_date, tmp_path
    ) -> None:
        from crucible.features import DEFAULT_FEATURE_VERSION, read_features
        from crucible.keys import features_key
        from crucible.slots.arms import load_arm_specs

        settings, decision_days = _seed_cycle(store, source, strategy_dir, cycle_date, tmp_path)
        day = decision_days[0]
        features = read_features(
            store.get_bytes(features_key(DEFAULT_FEATURE_VERSION, day.isoformat()))
        )
        arm = load_arm_specs("u", strategy_dir=strategy_dir)[0]
        top_n = int(arm.params.get("top_n", 10))

        cross_section = produce_cross_section(arm, features, day)

        assert len(cross_section.ranks) > top_n, (
            "shadow.v1's whole reason for existing was that the top-N selection is "
            "not enough to compute a rank correlation from"
        )
        assert cross_section.ranks[0][2] == 1, "rank 1 is the highest score"
        scores_desc = [score for _ticker, score, _rank in cross_section.ranks]
        assert scores_desc == sorted(scores_desc, reverse=True)

    def test_write_then_read_round_trips(self, store) -> None:
        cross_section = ScoredCrossSection(
            arm_id="r:x:1", trading_day="2026-08-28", ranks=(("AAA", 3.0, 1), ("BBB", 1.0, 2))
        )
        write_cross_section(store, cross_section)
        key = cross_section_key("r:x:1", "2026-08-28")
        assert store.exists(key)
        document = json.loads(store.get_bytes(key).decode("utf-8"))
        assert document["population_size"] == 2
        assert document["ranks"][0]["ticker"] == "AAA"


class TestSettleCrossSection:
    def test_settlement_joins_by_ticker_and_leaves_unsettled_names_null(self) -> None:
        document = ScoredCrossSection(
            arm_id="r:x:1",
            trading_day="2026-08-28",
            ranks=(("AAA", 3.0, 1), ("BBB", 2.0, 2), ("CCC", 1.0, 3)),
        ).to_dict()
        settled = settle_cross_section(
            document,
            returns={"AAA": 0.05, "CCC": -0.01},
            horizon_trading_days=21,
            settled_on="2026-09-28",
        )
        by_ticker = {r["ticker"]: r["realized_forward_return_ratio"] for r in settled["ranks"]}
        assert by_ticker == {"AAA": 0.05, "BBB": None, "CCC": -0.01}
        assert settled["n_settled"] == 2
        assert settled["population_size"] == 3
        assert settled["horizon_trading_days"] == 21
        assert settled["settled_on"] == "2026-09-28"

    def test_write_then_read_round_trips(self, store) -> None:
        document = settle_cross_section(
            ScoredCrossSection(
                arm_id="r:x:1", trading_day="2026-08-28", ranks=(("AAA", 3.0, 1),)
            ).to_dict(),
            returns={"AAA": 0.05},
            horizon_trading_days=21,
            settled_on="2026-09-28",
        )
        write_cross_section_settled(
            store, arm_id="r:x:1", trading_day="2026-08-28", document=document
        )
        key = cross_section_settled_key("r:x:1", "2026-08-28")
        assert store.exists(key)


class TestCycleWritesShadowV2:
    """The wiring inside `crucible.slots.cycle`: `run_produce` writes
    `cross_section.json` beside `shadow.json`, and `run_grade` settles it
    beside `verdict.json`, once its horizon settles."""

    def test_produce_writes_a_cross_section_for_every_shadow(
        self, store, source, strategy_dir, cycle_date, tmp_path
    ) -> None:
        settings, decision_days = _seed_cycle(store, source, strategy_dir, cycle_date, tmp_path)
        from crucible.slots.arms import load_arm_specs

        for arm in load_arm_specs("u", strategy_dir=strategy_dir):
            for day in decision_days:
                assert store.exists(shadow_key(arm.arm_id, day.isoformat()))
                assert store.exists(cross_section_key(arm.arm_id, day.isoformat())), (
                    f"shadow.json exists for {arm.arm_id}/{day} with no cross_section.json "
                    "beside it — shadow.v2 is not actually being produced"
                )

    def test_grade_settles_the_cross_section_alongside_the_verdict(
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
        settled_anything = False
        for day in decision_days:
            verdict_exists = store.exists(verdict_key(arm.arm_id, day.isoformat()))
            settled_exists = store.exists(cross_section_settled_key(arm.arm_id, day.isoformat()))
            assert verdict_exists == settled_exists, (
                f"{day}: verdict.json and cross_section_settled.json must appear together — "
                "one settling without the other is exactly the drift shadow.v2 exists to "
                "prevent between what the row grades and what it reduces"
            )
            settled_anything = settled_anything or settled_exists
        assert settled_anything, "at least one decision date must have settled by cycle_date"

        settled_key = next(
            cross_section_settled_key(arm.arm_id, day.isoformat())
            for day in decision_days
            if store.exists(cross_section_settled_key(arm.arm_id, day.isoformat()))
        )
        document = json.loads(store.get_bytes(settled_key).decode("utf-8"))
        assert document["schema_version"] == "cross_section_settled.v2"
        assert document["n_settled"] >= 1
        assert document["population_size"] >= document["n_settled"]
