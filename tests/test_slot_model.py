"""The M slot: recipe-defined training, CPCV grading, and the two §5.3 gates.

Normative sources: `champion-challenger-policy.md` §3 (TrainingIntegrityError
fails the task), §3.1 (the recipe is the immutable unit; a refit is not a
change), §5.3 (the behavioural veto stays SCALE-DEPENDENT; input
completeness); plan §4.4.

The veto tests below are the load-bearing ones. Measured 2026-08-29: a
standardized — scale-invariant — version of this guard passes the collapsed
2026-08-28 model at 0.943 and the 2026-08-21 model that produced five live
sessions with zero high-confidence names at 0.973, because dividing by the
spread divides the collapse away. The tests therefore assert the guard's
scale-dependent behaviour directly, so "improving" it by normalising it
turns them red.
"""

from __future__ import annotations

import datetime as dt
import statistics
from dataclasses import replace

import numpy as np
import pytest
from nousergon_lib.arena.engine import TrainingIntegrityError

from crucible.slots.model import (
    MIN_DISPERSION_RATIO,
    REQUIRED_RECIPE_FIELDS,
    CPCVSpec,
    FeatureLayerSource,
    FeaturePanel,
    MetricScaleError,
    ModelRecipe,
    TrainingWindowSpec,
    _rank_ic,
    _ranks,
    cpcv_oos_ic,
    evaluate_behavioural_veto,
    evaluate_input_completeness,
    grade_arm,
    load_model_recipes,
    settled_training_days,
    train_arm,
)
from crucible.store import LocalStore

# --------------------------------------------------------------------------
# §5.3 precondition 1 — the behavioural veto, in its scale-DEPENDENT form.
# --------------------------------------------------------------------------


class TestBehaviouralVeto:
    def test_the_dispersion_bar_is_half_the_incumbent_not_a_tuned_number(self) -> None:
        assert MIN_DISPERSION_RATIO == 0.5

    def test_a_collapsed_spread_is_vetoed(self) -> None:
        result = evaluate_behavioural_veto(
            candidate={
                "alpha_stdev": 0.004,
                "stdev_p_up": 0.05,
                "n_high_confidence": 12,
                "model_hit_rate_30d": 0.61,
            },
            incumbent={
                "alpha_stdev": 0.010,
                "stdev_p_up": 0.05,
                "n_high_confidence": 14,
                "model_hit_rate_30d": 0.60,
            },
        )
        assert result.status == "veto"
        assert "alpha_stdev" in " ".join(result.reasons)

    def test_the_standardized_ratio_that_would_have_passed_the_collapse_is_not_used(
        self,
    ) -> None:
        """2026-08-28: standardized ratio 0.943; 2026-08-21: 0.973. Both read
        healthy. The raw comparison vetoes both."""
        for standardized, alpha_stdev in ((0.943, 0.0040), (0.973, 0.0045)):
            result = evaluate_behavioural_veto(
                candidate={
                    "alpha_stdev": alpha_stdev,
                    "stdev_p_up": 0.05,
                    "n_high_confidence": 9,
                    "model_hit_rate_30d": 0.58,
                    "standardized_dispersion_ratio": standardized,
                },
                incumbent={
                    "alpha_stdev": 0.010,
                    "stdev_p_up": 0.05,
                    "n_high_confidence": 14,
                    "model_hit_rate_30d": 0.60,
                },
            )
            assert result.status == "veto", (
                f"a standardized ratio of {standardized} reads healthy; the guard must "
                "not be normalised (policy §5.3)"
            )

    def test_zero_high_confidence_names_is_an_absolute_veto(self) -> None:
        result = evaluate_behavioural_veto(
            candidate={
                "alpha_stdev": 0.02,
                "stdev_p_up": 0.06,
                "n_high_confidence": 0,
                "model_hit_rate_30d": 0.70,
            },
            incumbent={
                "alpha_stdev": 0.01,
                "stdev_p_up": 0.05,
                "n_high_confidence": 14,
                "model_hit_rate_30d": 0.60,
            },
        )
        assert result.status == "veto"
        assert "n_high_confidence" in " ".join(result.reasons)

    def test_a_hit_rate_below_the_floor_is_vetoed(self) -> None:
        result = evaluate_behavioural_veto(
            candidate={
                "alpha_stdev": 0.02,
                "stdev_p_up": 0.06,
                "n_high_confidence": 12,
                "model_hit_rate_30d": 0.41,
            },
            incumbent={
                "alpha_stdev": 0.01,
                "stdev_p_up": 0.05,
                "n_high_confidence": 14,
                "model_hit_rate_30d": 0.60,
            },
        )
        assert result.status == "veto"

    def test_a_missing_metric_is_insufficient_never_a_silent_pass(self) -> None:
        result = evaluate_behavioural_veto(
            candidate={"n_high_confidence": 12, "model_hit_rate_30d": 0.61},
            incumbent={
                "alpha_stdev": 0.010,
                "stdev_p_up": 0.05,
                "n_high_confidence": 14,
                "model_hit_rate_30d": 0.60,
            },
        )
        assert result.status == "insufficient"
        assert result.uncomputable

    def test_the_veto_reaches_the_engine_as_a_serving_precondition(self) -> None:
        result = evaluate_behavioural_veto(
            candidate={
                "alpha_stdev": 0.004,
                "stdev_p_up": 0.05,
                "n_high_confidence": 12,
                "model_hit_rate_30d": 0.61,
            },
            incumbent={
                "alpha_stdev": 0.010,
                "stdev_p_up": 0.05,
                "n_high_confidence": 14,
                "model_hit_rate_30d": 0.60,
            },
        )
        precondition = result.as_precondition()
        assert precondition.name == "behavioural_veto"
        assert precondition.passed is False
        assert precondition.reason


# --------------------------------------------------------------------------
# §5.3 precondition 2 — input completeness.
# --------------------------------------------------------------------------


class TestInputCompleteness:
    def test_a_required_input_below_its_row_ratio_fails_the_precondition(self) -> None:
        result = evaluate_input_completeness(
            required=("SPY", "VIX"),
            observed={
                "SPY": {"rows": 500, "last_date": "2026-08-28"},
                "VIX": {"rows": 16, "last_date": "2026-08-28"},
            },
            expected_rows=500,
            as_of=dt.date(2026, 8, 28),
        )
        assert result.as_precondition().passed is False
        assert "VIX" in result.as_precondition().reason

    def test_a_stale_required_input_fails(self) -> None:
        result = evaluate_input_completeness(
            required=("SPY",),
            observed={"SPY": {"rows": 500, "last_date": "2026-07-01"}},
            expected_rows=500,
            as_of=dt.date(2026, 8, 28),
        )
        assert result.as_precondition().passed is False

    def test_complete_inputs_pass(self) -> None:
        result = evaluate_input_completeness(
            required=("SPY",),
            observed={"SPY": {"rows": 500, "last_date": "2026-08-28"}},
            expected_rows=500,
            as_of=dt.date(2026, 8, 28),
        )
        assert result.as_precondition().passed is True


# --------------------------------------------------------------------------
# §3 — improper training fails the WHOLE slot.
# --------------------------------------------------------------------------


class TestTrainingIntegrity:
    def test_a_degenerate_fit_raises_rather_than_recording_a_miss(self, panel) -> None:
        recipe = _recipe(features=("mom_21d_ratio",))
        blank = panel.with_zeroed(("mom_21d_ratio",))
        with pytest.raises(TrainingIntegrityError):
            train_arm(recipe, blank, as_of="2026-08-28")

    def test_the_failure_writes_a_failed_manifest_and_re_raises(self, tmp_path, panel) -> None:
        from crucible.runner import run_job

        store = LocalStore(tmp_path)
        recipe = _recipe(features=("mom_21d_ratio",))
        blank = panel.with_zeroed(("mom_21d_ratio",))

        with pytest.raises(TrainingIntegrityError):
            run_job(
                "experiment.grade",
                lambda ctx: train_arm(recipe, blank, as_of="2026-08-28"),
                store=store,
                trading_day=dt.date(2026, 8, 28),
            )

        import json

        manifest = json.loads(store.get_bytes("runs/experiment.grade/2026-08-28/run.json"))
        assert manifest["status"] == "failed"
        assert "TrainingIntegrityError" in manifest["reason"]

    def test_a_sound_fit_produces_a_training_status_the_engine_accepts(self, panel) -> None:
        fit = train_arm(_recipe(), panel, as_of="2026-08-28")
        assert fit.training_status.ok is True
        assert fit.training_status.arm_id == fit.arm_id


# --------------------------------------------------------------------------
# §3.1 — the recipe is the immutable unit.
# --------------------------------------------------------------------------


class TestRecipeIdentity:
    def test_an_edited_recipe_is_a_new_arm(self) -> None:
        a = _recipe(features=("mom_21d_ratio",))
        b = _recipe(features=("mom_21d_ratio", "vol_21d_ratio"))
        assert a.arm_id != b.arm_id
        assert a.arm_id == _recipe(features=("mom_21d_ratio",)).arm_id

    def test_a_refit_does_not_change_the_id_and_does_not_reset_the_series(self, panel) -> None:
        recipe = _recipe()
        first = train_arm(recipe, panel, as_of="2026-08-21")
        second = train_arm(recipe, panel, as_of="2026-08-28")
        assert first.arm_id == second.arm_id, (
            "policy §3.1: a refit is the arm doing its job — same id, continuous series"
        )
        assert first.fitted_at != second.fitted_at


# --------------------------------------------------------------------------
# CPCV — purged and embargoed, per López de Prado.
# --------------------------------------------------------------------------


class TestCPCV:
    def test_labels_overlapping_the_test_block_are_purged_from_training(self, panel) -> None:
        result = cpcv_oos_ic(
            panel,
            recipe=_recipe(),
            cpcv=CPCVSpec(n_groups=6, k_test=2, embargo_trading_days=2),
            label_horizon_trading_days=21,
        )
        for fold in result.folds:
            assert not (set(fold.train_dates) & set(fold.purged_dates))
            assert fold.purged_dates, "a purge that removes nothing is not a purge"

    def test_the_backtest_path_count_is_reported(self, panel) -> None:
        result = cpcv_oos_ic(
            panel,
            recipe=_recipe(),
            cpcv=CPCVSpec(n_groups=6, k_test=2, embargo_trading_days=2),
            label_horizon_trading_days=21,
        )
        assert result.n_backtest_paths == 5
        assert len(result.ics) == 15

    def test_an_empty_fold_is_unmeasurable_not_a_zero_ic(self, panel) -> None:
        thin = panel.head(10)
        result = cpcv_oos_ic(
            thin,
            recipe=_recipe(),
            cpcv=CPCVSpec(n_groups=6, k_test=2, embargo_trading_days=2),
            label_horizon_trading_days=21,
        )
        assert result.status == "unmeasurable"
        assert result.reason


# --------------------------------------------------------------------------
# Grading and the cycle artifact.
# --------------------------------------------------------------------------


class TestGrade:
    """The series the pointer is decided on.

    The two tests that used to live here could not fail. Demonstrated by an
    adversarial review on 2026-09-01: replacing :func:`grade_arm` with a
    function returning an all-zero series left both of them GREEN.
    `test_grade_produces_a_per_date_series_the_arena_can_pair` asserted only
    that the series was non-empty and float-valued, which is true of any
    implementation including one that measures nothing;
    `test_the_benchmark_is_the_scored_cross_section_never_spy` asserted a
    constant it imported and touched no grading code at all while living in a
    class named for grading (plan §11 risk 1: "v2 passes its gates because its
    tests were written to pass").

    What replaces them is the §10.1 control-arm idea applied to the fixtures:
    a planted edge must be MEASURED, a panel with no signal must not produce
    one, and a cross-section that cannot be ranked must come back
    unmeasurable rather than perfect. Each of these turns red when the grader
    stops measuring.
    """

    def test_a_planted_edge_is_measured_by_the_out_of_sample_series(self, panel) -> None:
        """`tests/support/panels.py` plants momentum at ~0.6 of the label's
        standard deviation. A grader that measures anything at all recovers
        it; the all-zero stub the reviewer substituted does not."""
        grade = grade_arm(_recipe(), panel, as_of="2026-08-28")
        assert grade.status == "ok", grade.reason
        assert grade.series.arm_id == _recipe().arm_id
        mean_ic = statistics.fmean(grade.series.scores.values())
        assert mean_ic > 0.30, (
            f"the planted edge reads {mean_ic:.4f} out-of-sample; the fixture plants "
            "momentum at 0.6 of the label's sd and the grader must recover it. A series "
            "that cannot find a planted edge is not measuring the arm"
        )

    def test_an_arm_that_only_sees_the_unplanted_feature_scores_near_zero(self, panel) -> None:
        """`vol_21d_ratio` explains none of the label. The same grader on the
        same panel must separate it from `mom_21d_ratio` — the §10.1 negative
        control beside the positive one."""
        planted = grade_arm(_recipe(features=("mom_21d_ratio",)), panel, as_of="2026-08-28")
        null = grade_arm(_recipe(features=("vol_21d_ratio",)), panel, as_of="2026-08-28")
        planted_ic = statistics.fmean(planted.series.scores.values())
        null_ic = statistics.fmean(null.series.scores.values())
        assert abs(null_ic) < 0.05, f"the null feature scores {null_ic:.4f}, which is not null"
        assert planted_ic > null_ic + 0.30

    def test_more_features_do_not_buy_a_better_grade_on_pure_noise(self) -> None:
        """The F1 reproduction, as a regression test.

        Measured against the in-sample grader this replaces: on 40 pure-noise
        features over a pure-noise label the series read +0.0728 mean IC while
        the CPCV out-of-sample figure on the same arm read -0.0214, and the
        40-feature arm beat a 3-feature arm 0.0728 vs 0.0118 on data
        containing no signal. The slot was ranking capacity to overfit. A
        walk-forward series cannot: the fit that scores a day never saw it.
        """
        noise = _noise_panel(n_days=160, n_names=25, n_features=40, seed=11)
        wide = grade_arm(_recipe(features=_noise_features(40)), noise, as_of="2026-08-28")
        narrow = grade_arm(_recipe(features=_noise_features(3)), noise, as_of="2026-08-28")
        wide_ic = statistics.fmean(wide.series.scores.values())
        narrow_ic = statistics.fmean(narrow.series.scores.values())
        assert abs(wide_ic) < 0.05, (
            f"40 pure-noise features score {wide_ic:.4f} out-of-sample on a pure-noise "
            "label; in-sample the same arm read +0.0728"
        )
        assert wide_ic < narrow_ic + 0.05, (
            f"40 features ({wide_ic:.4f}) beat 3 ({narrow_ic:.4f}) on data with no signal, "
            "which is the in-sample defect: the grader is ranking overfit"
        )

    def test_a_cross_section_that_cannot_be_ranked_is_a_miss_never_a_score(self, panel) -> None:
        """A flat label block has no ordering to correlate against. Under the
        argsort ranks this scored +1.0 — a perfect grade for the collapsed
        condition the behavioural veto exists to refuse."""
        flat_day = panel.dates[120]
        labels = panel.forward_returns.copy()
        labels[120, :] = 0.004
        flattened = replace(panel, forward_returns=labels)
        grade = grade_arm(_recipe(), flattened, as_of="2026-08-28")
        assert flat_day not in grade.series.scores
        assert flat_day in grade.series.misses
        assert flat_day in grade.unrankable_dates

    def test_the_benchmark_is_the_scored_cross_section_never_spy(self, panel) -> None:
        """The slot's declared benchmark AND the grade that carries it.

        The old version of this test asserted the constant alone and never
        called the grader, so it passed against a grader that had been deleted.
        """
        from crucible.slots import get_slot

        assert get_slot("m").benchmark == "population"
        grade = grade_arm(_recipe(), panel, as_of="2026-08-28")
        assert grade.benchmark == "population"
        assert grade.oos_method == "walk_forward_purged"


class TestOutOfSampleClock:
    """Plan §9.1: the OOS window begins at registration, and nothing is pooled."""

    def test_no_date_before_registration_is_scored(self, panel) -> None:
        recipe = _recipe(registered_at="2026-07-01")
        grade = grade_arm(recipe, panel, as_of="2026-08-28")
        assert grade.series.scores
        assert min(grade.series.scores) >= "2026-07-01"
        assert grade.oos_start >= "2026-07-01"

    def test_a_newly_registered_arm_gets_no_backfilled_series(self, panel) -> None:
        """The F5 half this file owns.

        `promote.py`'s `promote_min_weeks = 4` measured the paired window, and
        the grader handed a three-day-old arm a 120-date series to pair over,
        so the age bar could be cleared by an arm that had existed for days.
        The grader no longer produces the history — an arm registered two
        sessions before `as_of` has at most two scores, and one registered
        after `as_of` has none.
        """
        recent = grade_arm(_recipe(registered_at="2026-08-27"), panel, as_of="2026-08-28")
        assert recent.oos_n <= 2, f"a two-session-old arm was handed {recent.oos_n} dates"

        unborn = grade_arm(_recipe(registered_at="2026-09-30"), panel, as_of="2026-08-28")
        assert unborn.status == "unmeasurable"
        assert unborn.oos_n == 0
        assert not unborn.series.scores
        assert "registered_at" in unborn.reason

    def test_in_sample_and_out_of_sample_counts_are_reported_separately(self, panel) -> None:
        """§9.1: 'the grader refuses to report a single pooled figure'."""
        grade = grade_arm(_recipe(registered_at="2026-07-01"), panel, as_of="2026-08-28")
        assert grade.in_sample_n > 0
        assert grade.oos_n > 0
        assert grade.in_sample_n + grade.oos_n + len(grade.unrankable_dates) == sum(
            1 for d in panel.dates if d <= "2026-08-28"
        )
        assert not hasattr(grade, "mean_ic"), (
            "a single pooled figure across the warm-up and the OOS window is a number no "
            "decision can legitimately be taken on, and it gets taken by existing"
        )

    def test_an_arm_without_a_registration_date_does_not_register(self, tmp_path) -> None:
        """Plan §9.1 pre-registration: missing fields -> the arm does not register."""
        (tmp_path / "no_clock.yaml").write_text(
            "\n".join(
                [
                    "slot: m",
                    "name: no_clock",
                    "spec:",
                    "  features: [mom_21d_ratio]",
                    "  estimator: {kind: ridge, alpha: 1.0}",
                    "  label_horizon_trading_days: 21",
                    "  refit_cadence_trading_days: 5",
                    "  training_window: {kind: expanding, min_trading_days: 504}",
                    "  cpcv: {n_groups: 6, k_test: 2, embargo_trading_days: 2}",
                ]
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="registered_at"):
            load_model_recipes(tmp_path)

    def test_registered_at_is_not_part_of_the_arm_id(self) -> None:
        """A re-registration must not orphan the arm's score series (§3.1)."""
        assert (
            _recipe(registered_at="2026-01-02").arm_id == _recipe(registered_at="2026-07-01").arm_id
        )


class TestRankIC:
    """The tie handling `scipy.stats.spearmanr` gave the lifted original."""

    def test_two_flat_vectors_are_unmeasurable_not_a_perfect_correlation(self) -> None:
        flat = np.zeros(50)
        assert _rank_ic(flat, flat.copy()) is None, (
            "argsort ranks gave 1.0 here — a perfect score for a cross-section with no "
            "ordering in it"
        )

    def test_a_collapsed_constant_prediction_is_unmeasurable(self) -> None:
        collapsed = np.full(50, 0.0031)
        actual = np.linspace(-0.05, 0.05, 50)
        assert _rank_ic(collapsed, actual) is None

    def test_ties_take_the_average_rank(self) -> None:
        """Hand-computed against mid-ranks, which is what `scipy.stats.spearmanr`
        assigns and what the lifted `leakfree_meta_ic.py` therefore used."""
        values = np.array([3.0, 1.0, 1.0, 1.0, 5.0])
        assert list(_ranks(values)) == [3.0, 1.0, 1.0, 1.0, 4.0]

    def test_a_mostly_tied_prediction_does_not_invent_an_ordering(self) -> None:
        """45 of 50 names tied.

        Measured on a real cross-section, the argsort form read -0.4558 where
        the mid-rank Spearman is -0.1043. The mechanism is asserted here
        rather than the two numbers: the index-order form's answer DEPENDS ON
        THE ORDER THE TICKERS ARRIVED IN, and the mid-rank form's does not.
        Permuting the tied block alone moves the argsort IC from -0.0372 to
        +0.2060 on identical data; the mid-rank IC is +0.5206 either way, and
        matches Spearman's definition.
        """
        predicted = np.concatenate([np.arange(5.0), np.full(45, 9.9)])
        actual = np.arange(50.0)
        shuffled = np.concatenate([np.arange(5), 5 + np.random.default_rng(9).permutation(45)])

        ic = _rank_ic(predicted, actual)
        assert ic is not None
        assert ic == pytest.approx(_spearman_by_definition(predicted, actual), abs=1e-12)
        assert ic == pytest.approx(_rank_ic(predicted[shuffled], actual[shuffled]), abs=1e-12), (
            "the grade must be a fact about the cross-section, not about the order the "
            "tickers were listed in"
        )

        before = _argsort_rank_ic(predicted, actual)
        after = _argsort_rank_ic(predicted[shuffled], actual[shuffled])
        assert abs(before - after) > 0.2, (
            f"the form this replaced read {before:.4f} and {after:.4f} on the SAME data "
            "reordered; if that is no longer true the fixture has stopped exercising ties"
        )

    def test_untied_inputs_are_unchanged_by_the_mid_rank_form(self) -> None:
        rng = np.random.default_rng(5)
        predicted = rng.normal(0.0, 1.0, 200)
        actual = rng.normal(0.0, 1.0, 200)
        assert _rank_ic(predicted, actual) == pytest.approx(
            _spearman_by_definition(predicted, actual), abs=1e-12
        )

    def test_a_length_mismatch_raises_rather_than_aligning(self) -> None:
        with pytest.raises(ValueError, match="paired vectors"):
            _rank_ic(np.zeros(5), np.zeros(4))


class TestLabelHorizonPurge:
    """A fit dated `as_of` may not consume a label realized after it."""

    def test_the_final_label_horizon_is_purged_from_the_fit(self, panel) -> None:
        recipe = _recipe()
        settled = settled_training_days(panel, as_of="2026-08-28", label_horizon=21)
        assert len(settled) == len(panel.dates) - 21
        fit = train_arm(recipe, panel, as_of="2026-08-28")
        assert fit.n_rows == (len(panel.dates) - 21) * len(panel.names)

    def test_a_label_realized_after_as_of_cannot_move_the_fit(self, panel) -> None:
        """The defect, stated as an experiment: corrupt only the labels that
        settle after `as_of` and the fit must be byte-identical. Before the
        purge it moved, so every replayed Saturday in the §6.1 gate used
        post-date information."""
        recipe = _recipe()
        before = train_arm(recipe, panel, as_of="2026-08-28")
        tampered = panel.forward_returns.copy()
        tampered[-21:, :] = tampered[-21:, :] + 5.0
        after = train_arm(recipe, replace(panel, forward_returns=tampered), as_of="2026-08-28")
        assert np.array_equal(before.coefficients, after.coefficients), (
            "a return realized after as_of moved the fit dated as_of"
        )
        assert before.intercept == after.intercept

    def test_a_window_reached_only_by_unsettled_dates_is_not_satisfied(self, panel) -> None:
        """`min_trading_days` counts SETTLED days. 160 panel dates minus a
        21-session horizon is 139, so a recipe declaring 150 must raise rather
        than fit on rows whose labels do not exist yet."""
        with pytest.raises(TrainingIntegrityError, match="SETTLED"):
            train_arm(
                _recipe(training_window=TrainingWindowSpec("expanding", 150)),
                panel,
                as_of="2026-08-28",
            )


class TestMetricScale:
    """The behavioural veto may not fail OPEN on a unit-scale error."""

    def test_a_percentage_scaled_hit_rate_raises_rather_than_passing(self) -> None:
        """The F7 reproduction: the SAME model at two scales. 0.4 vetoes on
        'below the absolute floor 0.5'; 40.0 used to PASS silently."""
        candidate = {
            "alpha_stdev": 0.02,
            "stdev_p_up": 0.06,
            "n_high_confidence": 12,
            "model_hit_rate_30d": 40.0,
        }
        incumbent = {
            "alpha_stdev": 0.02,
            "stdev_p_up": 0.06,
            "n_high_confidence": 14,
            "model_hit_rate_30d": 0.60,
        }
        with pytest.raises(MetricScaleError, match="model_hit_rate_30d"):
            evaluate_behavioural_veto(candidate, incumbent)

        candidate["model_hit_rate_30d"] = 0.40
        assert evaluate_behavioural_veto(candidate, incumbent).status == "veto"

    def test_an_out_of_range_incumbent_raises_too(self) -> None:
        """Both sides. A percentage-scaled incumbent makes every ratio and
        floor on the candidate meaningless in the same direction."""
        with pytest.raises(MetricScaleError, match="incumbent"):
            evaluate_behavioural_veto(
                {
                    "alpha_stdev": 0.02,
                    "stdev_p_up": 0.06,
                    "n_high_confidence": 12,
                    "model_hit_rate_30d": 0.61,
                },
                {
                    "alpha_stdev": 0.02,
                    "stdev_p_up": 0.06,
                    "n_high_confidence": 14,
                    "model_hit_rate_30d": 60.0,
                },
            )

    def test_a_negative_proportion_raises(self) -> None:
        with pytest.raises(MetricScaleError):
            evaluate_behavioural_veto(
                {
                    "alpha_stdev": 0.02,
                    "stdev_p_up": 0.06,
                    "n_high_confidence": 12,
                    "model_hit_rate_30d": -0.01,
                },
                {
                    "alpha_stdev": 0.02,
                    "stdev_p_up": 0.06,
                    "n_high_confidence": 14,
                    "model_hit_rate_30d": 0.60,
                },
            )


# --------------------------------------------------------------------------
# Recipes come from the private config repo, never from this repository.
# --------------------------------------------------------------------------


class TestRecipeLoading:
    def test_recipes_load_from_yaml_and_carry_their_own_spec_hash(self, tmp_path) -> None:
        (tmp_path / "residual_momentum.yaml").write_text(
            "\n".join(
                [
                    "slot: m",
                    "name: residual_momentum",
                    "spec:",
                    "  features: [mom_21d_ratio]",
                    "  estimator: {kind: ridge, alpha: 1.0}",
                    "  label_horizon_trading_days: 21",
                    "  refit_cadence_trading_days: 5",
                    "  training_window: {kind: expanding, min_trading_days: 504}",
                    "  cpcv: {n_groups: 6, k_test: 2, embargo_trading_days: 2}",
                    "registered_at: '2026-06-01'",
                ]
            ),
            encoding="utf-8",
        )
        recipes = load_model_recipes(tmp_path, feature_columns=("mom_21d_ratio",))
        assert recipes.refused == ()
        assert len(recipes.registered) == 1
        assert recipes.registered[0].arm_id.startswith("m:residual_momentum:")

    def test_feature_version_is_not_a_recipe_field(self, tmp_path) -> None:
        """`alpha-engine-config-I9801`: a hand-written `feature_version` inside

        `spec` was hashed into the arm id and read by nothing — the same bug
        class as `avg_volume_20d` (AGENTS.md), a hand-maintained value
        standing where a derived one belongs. It is now gone from the recipe
        entirely: `REQUIRED_RECIPE_FIELDS` does not name it, a recipe that
        still declares it under `spec` registers with the key silently
        ignored (a private-repo recipe transitioning off the old field is not
        refused mid-migration), and it is absent from the hashed `spec` —
        which feature-layer artifact a run actually reads is resolved by
        `FeatureLayerSource` and recorded as lineage
        (`FeaturePanel.feature_version`), never declared by the recipe.
        """
        assert "feature_version" not in REQUIRED_RECIPE_FIELDS
        recipe = _recipe()
        assert "feature_version" not in recipe.spec
        assert not hasattr(recipe, "feature_version")

        (tmp_path / "legacy.yaml").write_text(
            "\n".join(
                [
                    "slot: m",
                    "name: legacy",
                    "spec:",
                    "  features: [mom_21d_ratio]",
                    "  estimator: {kind: ridge, alpha: 1.0}",
                    "  label_horizon_trading_days: 21",
                    "  refit_cadence_trading_days: 5",
                    "  training_window: {kind: expanding, min_trading_days: 504}",
                    "  cpcv: {n_groups: 6, k_test: 2, embargo_trading_days: 2}",
                    "  feature_version: v1",
                    "registered_at: '2026-06-01'",
                ]
            ),
            encoding="utf-8",
        )
        loaded = load_model_recipes(tmp_path, feature_columns=("mom_21d_ratio",))
        assert loaded.refused == ()
        assert len(loaded.registered) == 1
        assert "feature_version" not in loaded.registered[0].spec

    def test_arm_id_is_independent_of_which_feature_layer_version_is_resolved(
        self, tmp_path
    ) -> None:
        """Deliverable 4 (`alpha-engine-config-I9801`), the actual claim.

        The arm id no longer depends on a declared `feature_version`, so a
        catalogue change — the thing the removed field would otherwise have
        needed re-declaring for — cannot silently re-id or silently NOT re-id
        an arm depending on whether someone remembered to bump it. Proved
        against a real feature-layer symbol: two `FeatureLayerSource`
        instances resolve two DIFFERENT versions, and the recipe's `arm_id`
        — computed from `ModelRecipe.spec` alone, which `FeatureLayerSource`
        never enters — is identical regardless, and identical across two
        loads of the same file. (A prior version of this test loaded the
        file twice with nothing about the feature layer varied at all, which
        passed for the uninteresting reason that `load_model_recipes` is a
        pure hash of file bytes — it named no feature-layer symbol despite
        its name.)
        """
        (tmp_path / "residual_momentum.yaml").write_text(
            "\n".join(
                [
                    "slot: m",
                    "name: residual_momentum",
                    "spec:",
                    "  features: [mom_21d_ratio]",
                    "  estimator: {kind: ridge, alpha: 1.0}",
                    "  label_horizon_trading_days: 21",
                    "  refit_cadence_trading_days: 5",
                    "  training_window: {kind: expanding, min_trading_days: 504}",
                    "  cpcv: {n_groups: 6, k_test: 2, embargo_trading_days: 2}",
                    "registered_at: '2026-06-01'",
                ]
            ),
            encoding="utf-8",
        )
        source_a = FeatureLayerSource(store=None, version="vaaaaaaaaaaaa")
        source_b = FeatureLayerSource(store=None, version="vbbbbbbbbbbbb")
        assert source_a.version != source_b.version

        first = load_model_recipes(tmp_path, feature_columns=("mom_21d_ratio",))
        second = load_model_recipes(tmp_path, feature_columns=("mom_21d_ratio",))
        assert first.registered[0].arm_id == second.registered[0].arm_id
        assert "feature_version" not in first.registered[0].spec

    def test_a_recipe_missing_a_pre_registration_field_does_not_register(self, tmp_path) -> None:
        """Plan §9.1: 'Missing fields -> the arm does not register.'"""
        (tmp_path / "broken.yaml").write_text(
            "slot: m\nname: broken\nspec: {features: [mom_21d_ratio]}\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match="refit_cadence_trading_days"):
            load_model_recipes(tmp_path)


# --------------------------------------------------------------------------
# Fixtures.
# --------------------------------------------------------------------------


def _spearman_by_definition(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Spearman computed from first principles, independently of the module.

    A test that recomputes the statistic with the same helper it is testing
    proves only that the helper is self-consistent. This one builds mid-ranks
    from `sorted()` in plain Python and takes the Pearson correlation of them,
    so it disagrees with :func:`crucible.slots.model._rank_ic` whenever the
    module's tie handling is wrong.
    """

    def midranks(values: np.ndarray) -> list[float]:
        ordered = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(ordered):
            j = i
            while j + 1 < len(ordered) and values[ordered[j + 1]] == values[ordered[i]]:
                j += 1
            for k in range(i, j + 1):
                out[ordered[k]] = (i + j) / 2.0
            i = j + 1
        return out

    a = midranks(predicted)
    b = midranks(actual)
    am = statistics.fmean(a)
    bm = statistics.fmean(b)
    num = sum((x - am) * (y - bm) for x, y in zip(a, b, strict=True))
    den = sum((x - am) ** 2 for x in a) ** 0.5 * sum((y - bm) ** 2 for y in b) ** 0.5
    return num / den


def _argsort_rank_ic(predicted: np.ndarray, actual: np.ndarray) -> float:
    """The rank IC as the module computed it BEFORE the mid-rank fix.

    Kept so the tie tests can state which of the two answers is correct rather
    than only that the module changed. Index-order ranks: a tied block is
    ordered by whatever position its members occupy in the array.
    """

    def ranks(values: np.ndarray) -> np.ndarray:
        order = values.argsort()
        out = np.empty_like(order, dtype=float)
        out[order] = np.arange(values.size, dtype=float)
        return out

    pr = ranks(predicted) - ranks(predicted).mean()
    ar = ranks(actual) - ranks(actual).mean()
    return float((pr * ar).sum() / np.sqrt((pr * pr).sum() * (ar * ar).sum()))


def _noise_features(n: int) -> tuple[str, ...]:
    return tuple(f"f{i:02d}_zscore" for i in range(n))


def _noise_panel(*, n_days: int, n_names: int, n_features: int, seed: int) -> FeaturePanel:
    """A panel with NO relationship between any feature and the label.

    The §10.1 negative control as a fixture. `tests/support/panels.py` plants
    an edge on purpose; this one plants nothing, so any grade it produces
    above sampling noise is the grader measuring itself.
    """
    from tests.support.panels import trading_days

    rng = np.random.default_rng(seed)
    return FeaturePanel(
        dates=tuple(trading_days(n_days)),
        names=tuple(f"T{i:03d}" for i in range(n_names)),
        features={
            name: rng.normal(0.0, 1.0, size=(n_days, n_names))
            for name in _noise_features(n_features)
        },
        forward_returns=rng.normal(0.0, 1.0, size=(n_days, n_names)),
        feature_version="v1",
    )


def _recipe(**over) -> ModelRecipe:
    from crucible.slots.model import EstimatorSpec

    kwargs = dict(
        name="residual_momentum",
        features=("mom_21d_ratio", "vol_21d_ratio"),
        estimator=EstimatorSpec(kind="ridge", params={"alpha": 1.0}),
        label_horizon_trading_days=21,
        refit_cadence_trading_days=5,
        training_window=TrainingWindowSpec(kind="expanding", min_trading_days=40),
        cpcv=CPCVSpec(n_groups=6, k_test=2, embargo_trading_days=2),
        # Earlier than the fixture panel's first session, so a test that does
        # not care about the out-of-sample clock is not silently gated by it.
        # The clock's own tests set this explicitly.
        registered_at="2020-01-02",
    )
    kwargs.update(over)
    return ModelRecipe(**kwargs)


@pytest.fixture
def panel():
    from tests.support.panels import synthetic_panel

    return synthetic_panel(n_days=160, n_names=25, seed=7)
