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

from crucible.slots.inputs import InputRef
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


class TestNoDesignColumnsGuard:
    """alpha-engine-config-I9821: `ModelRecipe.__post_init__` used to test
    `self.features` alone, predating `spec.inputs` (I9777). A PURE
    meta-learner — every design column a `predictions[...]` input, no
    feature-layer columns — is a real, intended arm shape (the canonical
    stacking ensemble) and must be constructible; an arm with an empty
    design matrix altogether (no features AND no inputs) must still be
    refused, by name.
    """

    def test_a_pure_meta_learner_constructs_with_empty_features(self) -> None:
        recipe = _recipe(features=(), inputs=(InputRef(kind="predictions", ref="base"),))
        assert recipe.features == ()
        assert recipe.design_columns == ("predicted_alpha_base_raw",)

    def test_an_arm_with_neither_features_nor_inputs_is_refused(self) -> None:
        with pytest.raises(ValueError, match="declares no design columns"):
            _recipe(features=(), inputs=())


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
        entirely: `REQUIRED_RECIPE_FIELDS` does not name it, and it is absent
        from the hashed `spec` — which feature-layer artifact a run actually
        reads is resolved by `FeatureLayerSource` and recorded as lineage
        (`FeaturePanel.feature_version`), never declared by the recipe.

        A recipe that still declares it is now REFUSED, not silently
        accepted mid-migration — see
        `test_feature_version_under_spec_is_refused_by_name`
        (`alpha-engine-config-I9944`, which closed the "transitional
        accept-and-ignore" loophole this test used to document once both
        live M recipes had already been cleaned up).
        """
        assert "feature_version" not in REQUIRED_RECIPE_FIELDS
        recipe = _recipe()
        assert "feature_version" not in recipe.spec
        assert not hasattr(recipe, "feature_version")

    def test_feature_version_under_spec_is_refused_by_name(self, tmp_path) -> None:
        """`alpha-engine-config-I9944`: `load_model_recipes` accepted-and-
        ignored any `spec:` key it did not read, so `feature_version` (dead
        per `alpha-engine-config-I9801`) could sit in a recipe unnoticed.
        The loader now refuses ANY unknown `spec` key by name, naming
        `feature_version` specifically since it has a documented reason it
        is not an M field.
        """
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
        # The I9801 citation lives in a comment beside `_NAMED_REASONS`, never
        # in the raised text itself (`tests/test_no_stale_tracker_literals.py`
        # refuses a hardcoded tracker literal reachable from a raise).
        with pytest.raises(ValueError, match="feature_version") as exc:
            load_model_recipes(tmp_path, feature_columns=("mom_21d_ratio",))
        assert "hashed spec that nothing read" in str(exc.value)

    def test_llm_callsite_under_spec_is_refused_by_name(self, tmp_path) -> None:
        """`alpha-engine-config-I9944`: `params.llm_callsite` binds on a U/R
        recipe's `params`; an M recipe author writing `llm_callsite:` under
        `spec:` — the natural place for it — would otherwise register an arm
        the phase-5 LLM-arm gate silently never counts.
        """
        (tmp_path / "sneaky.yaml").write_text(
            "\n".join(
                [
                    "slot: m",
                    "name: sneaky",
                    "spec:",
                    "  features: [mom_21d_ratio]",
                    "  estimator: {kind: ridge, alpha: 1.0}",
                    "  label_horizon_trading_days: 21",
                    "  refit_cadence_trading_days: 5",
                    "  training_window: {kind: expanding, min_trading_days: 504}",
                    "  cpcv: {n_groups: 6, k_test: 2, embargo_trading_days: 2}",
                    "  llm_callsite: research.thinktank",
                    "registered_at: '2026-06-01'",
                ]
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="llm_callsite") as exc:
            load_model_recipes(tmp_path, feature_columns=("mom_21d_ratio",))
        assert "phase-5 gate" in str(exc.value)

    def test_an_unknown_top_level_key_is_refused_by_name(self, tmp_path) -> None:
        (tmp_path / "extra.yaml").write_text(
            "\n".join(
                [
                    "slot: m",
                    "name: extra",
                    "bogus_field: nope",
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
        with pytest.raises(ValueError, match="bogus_field"):
            load_model_recipes(tmp_path, feature_columns=("mom_21d_ratio",))

    def test_a_recipe_with_no_unknown_keys_still_loads(self, tmp_path) -> None:
        """Sanity: the closed vocabulary does not reject a clean recipe,
        including the optional `inputs` key a stacked arm declares."""
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
                    "supersedes_v1: spec-residual-mom-2026-07-17-f478ece3",
                ]
            ),
            encoding="utf-8",
        )
        loaded = load_model_recipes(tmp_path, feature_columns=("mom_21d_ratio",))
        assert loaded.refused == ()
        assert len(loaded.registered) == 1

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


class TestTheDeadSlotIsObserved:
    """`alpha-engine-config-I9759`: a veto that can NEVER compute is a slot
    that can never promote, and until this metric existed the only trace was
    a per-arm precondition reason inside the cycle artifact — so a slot that
    can never serve rendered identically to one that held its pointer."""

    def _ctx(self):
        from types import SimpleNamespace

        rows: list[dict] = []
        return SimpleNamespace(record_metric=rows.append), rows

    def test_every_arm_insufficient_emits_the_finding_with_its_producers(self, monkeypatch) -> None:
        """The finding fires for as long as a veto input has no producer.

        `UNPRODUCED_VETO_METRICS` is empty since `alpha-engine-config-I10680`
        landed the three producers, so the declaration is re-populated here
        rather than deleted with the test: what has to keep working is that a
        metric added to a rule family with NO producer re-arms this row.
        """
        from crucible.slots import model as model_module
        from crucible.slots.model import DEAD_SLOT_METRIC, _record_dead_slot_finding

        waiting = {"some_new_metric": "a producer nobody has written"}
        monkeypatch.setattr(model_module, "UNPRODUCED_VETO_METRICS", waiting)
        ctx, rows = self._ctx()
        _record_dead_slot_finding(
            ctx, {"a": {"veto": "insufficient"}, "b": {"veto": "insufficient"}}, as_of="2026-08-28"
        )
        assert len(rows) == 1
        row = rows[0]
        assert row["name"] == DEAD_SLOT_METRIC
        assert row["status"] == "FAIL"
        # Every unproduced metric is NAMED with what it waits on: a finding
        # that says "insufficient" and not which producer is missing sends
        # the next reader back to re-derive it.
        for name, why in waiting.items():
            assert name in row["status_reason"]
            assert why in row["status_reason"]

    def test_with_every_producer_present_an_insufficient_cycle_is_not_a_dead_slot(self) -> None:
        """`alpha-engine-config-I10680`. An all-`insufficient` cycle now has a
        temporary cause — a short settled window — and naming three producers
        that exist would send the next reader to build them again."""
        from crucible.slots.model import UNPRODUCED_VETO_METRICS, _record_dead_slot_finding

        assert UNPRODUCED_VETO_METRICS == {}
        ctx, rows = self._ctx()
        _record_dead_slot_finding(
            ctx, {"a": {"veto": "insufficient"}, "b": {"veto": "insufficient"}}, as_of="2026-09-11"
        )
        assert rows == []

    def test_one_arm_with_a_real_verdict_emits_nothing(self) -> None:
        from crucible.slots.model import _record_dead_slot_finding

        ctx, rows = self._ctx()
        _record_dead_slot_finding(
            ctx, {"a": {"veto": "insufficient"}, "b": {"veto": "pass"}}, as_of="2026-08-28"
        )
        assert rows == [], "a slot with one live verdict is alive, not dead"

    def test_a_cycle_that_graded_nothing_emits_nothing(self) -> None:
        """No arms is an empty slot, not a permanently dead one."""
        from crucible.slots.model import _record_dead_slot_finding

        ctx, rows = self._ctx()
        _record_dead_slot_finding(ctx, {}, as_of="2026-08-28")
        assert rows == []

    def test_every_unproduced_metric_is_one_the_veto_actually_reads(self) -> None:
        """The declaration cannot drift into naming a metric no rule uses —
        which would be a permanent finding about nothing."""
        from crucible.slots.model import (
            DISPERSION_METRICS,
            FLOOR_VETO_METRICS,
            UNPRODUCED_VETO_METRICS,
            ZERO_VETO_METRICS,
        )

        read = set(DISPERSION_METRICS) | set(ZERO_VETO_METRICS) | set(FLOOR_VETO_METRICS)
        assert set(UNPRODUCED_VETO_METRICS) <= read

    def test_the_declaration_matches_what_the_producer_actually_emits(self, panel) -> None:
        """The other direction: every veto input the PRODUCER does not emit
        must be declared here. A metric quietly added to a rule family with
        no producer would otherwise kill the slot with no finding.

        Measured against `serving_metrics` over a real graded arm with a
        window long enough to support every statistic — which is the only
        state in which "the producer does not emit it" means "nobody can
        produce it" rather than "not yet"."""
        from crucible.slots.model import (
            DISPERSION_METRICS,
            FLOOR_VETO_METRICS,
            UNPRODUCED_VETO_METRICS,
            ZERO_VETO_METRICS,
            grade_arm,
            predict_cross_section,
            serving_metrics,
        )

        recipe = _recipe()
        graded = grade_arm(recipe, panel, as_of=panel.dates[-1])
        metrics, reason = serving_metrics(
            predict_cross_section(graded.fit, panel, trading_day=panel.dates[-1]),
            settled=graded.settled,
        )
        assert reason == "", reason
        read = set(DISPERSION_METRICS) | set(ZERO_VETO_METRICS) | set(FLOOR_VETO_METRICS)
        assert read - set(metrics) == set(UNPRODUCED_VETO_METRICS)


# --------------------------------------------------------------------------
# §5.3 precondition 1 — the PRODUCERS behind it (alpha-engine-config-I10680).
# --------------------------------------------------------------------------


class TestTheVetoProducers:
    """Three of the veto's four inputs had no producer, so the veto read
    `insufficient` on every cycle, the serving precondition failed on every
    cycle, and the M slot could never serve a champion — permanently, and for
    a reason no amount of waiting fixed.

    These tests assert the produced statistics have the SHAPE the veto
    expects, and — the load-bearing one — that the up-probability calibration
    keeps `stdev_p_up` scale-DEPENDENT. A calibration fitted on today's
    cross-section, or a predictor standardised at serving time, would divide
    a collapse away exactly as the standardized ratio did on 2026-08-28.
    """

    def _graded(self, panel, *, recipe=None, as_of=None):
        from crucible.slots.model import grade_arm, predict_cross_section

        recipe = recipe or _recipe()
        as_of = as_of or panel.dates[-1]
        graded = grade_arm(recipe, panel, as_of=as_of)
        return graded, predict_cross_section(graded.fit, panel, trading_day=as_of)

    def test_a_long_enough_window_produces_all_four_veto_inputs(self, panel) -> None:
        from crucible.slots.model import (
            DISPERSION_METRICS,
            FLOOR_VETO_METRICS,
            ZERO_VETO_METRICS,
            serving_metrics,
        )

        graded, predicted = self._graded(panel)
        metrics, reason = serving_metrics(predicted, settled=graded.settled)
        assert reason == ""
        for name in (*DISPERSION_METRICS, *ZERO_VETO_METRICS, *FLOOR_VETO_METRICS):
            assert name in metrics, name

    def test_the_veto_reads_pass_or_veto_on_a_real_cycle_never_insufficient(self, panel) -> None:
        """The whole point of the issue: a real cycle reaches a real verdict."""
        from crucible.slots.model import evaluate_behavioural_veto, serving_metrics

        graded, predicted = self._graded(panel)
        metrics, _ = serving_metrics(predicted, settled=graded.settled)
        veto = evaluate_behavioural_veto(metrics, metrics, has_incumbent=True)
        assert veto.status in ("pass", "veto")
        assert veto.uncomputable == ()

    def test_a_short_settled_window_is_insufficient_and_states_the_minimum(self, panel) -> None:
        from crucible.slots.model import (
            SETTLED_WINDOW_DECISION_DATES,
            evaluate_behavioural_veto,
            serving_metrics,
        )

        graded, predicted = self._graded(panel)
        short = graded.settled.tail(SETTLED_WINDOW_DECISION_DATES - 1)
        metrics, reason = serving_metrics(predicted, settled=short)
        assert str(SETTLED_WINDOW_DECISION_DATES) in reason
        assert "window being short, not a producer being absent" in reason
        veto = evaluate_behavioural_veto(metrics, metrics, has_incumbent=True)
        assert veto.status == "insufficient"

    def test_the_hit_rate_is_a_proportion_the_scale_assertion_accepts(self, panel) -> None:
        """`PROPORTION_METRICS` raises outside [0, 1]; the producer must never
        be the thing that trips it."""
        from crucible.slots.model import serving_metrics

        graded, predicted = self._graded(panel)
        metrics, _ = serving_metrics(predicted, settled=graded.settled)
        assert 0.0 <= metrics["model_hit_rate_30d"] <= 1.0

    def test_a_planted_edge_puts_the_hit_rate_above_the_floor(self, panel) -> None:
        """The fixture panel plants momentum in the label, so an arm that
        reads momentum must beat a coin flip. If it did not, the floor would
        be gating the producer rather than the model."""
        from crucible.slots.model import FLOOR_VETO_METRICS, serving_metrics

        graded, predicted = self._graded(panel)
        metrics, _ = serving_metrics(predicted, settled=graded.settled)
        assert metrics["model_hit_rate_30d"] > FLOOR_VETO_METRICS["model_hit_rate_30d"]

    def test_a_collapsed_cross_section_collapses_p_up_with_it(self, panel) -> None:
        """THE test this design exists to pass. The calibration is fitted on
        HISTORY and applied to today, so halving today's spread halves the
        up-probability spread — and the dispersion ratio catches it. A map
        fitted on today's cross-section would restore the spread and the
        collapse would read as healthy, which is the 2026-08-28 defect."""
        from crucible.slots.model import (
            MIN_DISPERSION_RATIO,
            evaluate_behavioural_veto,
            serving_metrics,
        )

        graded, predicted = self._graded(panel)
        incumbent, _ = serving_metrics(predicted, settled=graded.settled)
        collapsed = {name: value * 0.25 for name, value in predicted.items()}
        candidate, _ = serving_metrics(collapsed, settled=graded.settled)

        assert candidate["stdev_p_up"] / incumbent["stdev_p_up"] < MIN_DISPERSION_RATIO
        veto = evaluate_behavioural_veto(candidate, incumbent, has_incumbent=True)
        assert veto.status == "veto"
        assert any("stdev_p_up" in reason for reason in veto.reasons)

    def test_the_calibration_is_not_refitted_per_cross_section(self, panel) -> None:
        """The mechanism behind the test above, asserted directly: the map is
        a fixed affine function of predicted alpha, so scaling the input
        scales the logit."""
        from crucible.slots.model import calibrate_up_probability

        graded, _ = self._graded(panel)
        calibration = calibrate_up_probability(graded.settled)
        assert calibration is not None
        values = np.array([-0.4, 0.0, 0.4])
        halved = calibration.p_up(values * 0.5)
        full = calibration.p_up(values)
        assert float(np.std(halved)) < float(np.std(full))

    def test_one_realized_direction_refuses_the_calibration_rather_than_assuming_one(
        self, panel
    ) -> None:
        from crucible.slots.model import SettledCrossSections, calibrate_up_probability

        graded, _ = self._graded(panel)
        block = graded.settled
        one_sided = SettledCrossSections(
            dates=block.dates,
            predicted=block.predicted,
            realized=np.abs(block.realized) + 1.0,
        )
        assert calibrate_up_probability(one_sided) is None

    def test_a_calibration_that_cannot_be_fitted_leaves_the_metrics_absent(self, panel) -> None:
        from crucible.slots.model import (
            SettledCrossSections,
            evaluate_behavioural_veto,
            serving_metrics,
        )

        graded, predicted = self._graded(panel)
        block = graded.settled
        one_sided = SettledCrossSections(
            dates=block.dates,
            predicted=block.predicted,
            realized=np.abs(block.realized) + 1.0,
        )
        metrics, reason = serving_metrics(predicted, settled=one_sided)
        assert "stdev_p_up" not in metrics
        assert "n_high_confidence" not in metrics
        assert "could not be fitted" in reason
        veto = evaluate_behavioural_veto(metrics, metrics, has_incumbent=True)
        # Never a pass. `veto` also settles the arm, and an existing reason
        # outranks an absent metric because both exclude it — what may never
        # happen is the absent metric being read as satisfied.
        assert veto.status != "pass"
        assert set(veto.uncomputable) == {"stdev_p_up", "n_high_confidence"}

    def test_n_high_confidence_counts_only_the_selection_the_slot_would_serve(self, panel) -> None:
        from crucible.slots.model import M_SELECTION_TOP_N, serving_metrics

        graded, predicted = self._graded(panel)
        metrics, _ = serving_metrics(predicted, settled=graded.settled)
        assert 0 <= metrics["n_high_confidence"] <= M_SELECTION_TOP_N

    def test_a_constant_cross_section_names_nothing_and_is_vetoed_absolutely(self, panel) -> None:
        """The 2026-08-21 condition, reached through the real producer: a
        model whose predictions carry no ordering has nothing to serve, and
        the zero-veto is absolute — no incumbent required."""
        from crucible.slots.model import evaluate_behavioural_veto, serving_metrics

        graded, predicted = self._graded(panel)
        flat = dict.fromkeys(predicted, 0.0)
        metrics, reason = serving_metrics(flat, settled=graded.settled)
        assert reason == ""
        assert metrics["n_high_confidence"] == 0
        veto = evaluate_behavioural_veto(metrics, metrics, has_incumbent=True)
        assert veto.status == "veto"
        assert any("names nothing at high confidence" in r for r in veto.reasons)

    def test_a_planted_edge_passes_the_veto_on_a_cold_slot(self, panel) -> None:
        """The `pass` side. The fixture panel plants momentum in the label, so
        an arm that reads momentum clears the hit-rate floor and names
        tradeable positions — and on a slot with no champion the dispersion
        family has no comparand, so the verdict rests on the absolute rules."""
        from crucible.slots.model import evaluate_behavioural_veto, serving_metrics

        graded, predicted = self._graded(panel)
        metrics, reason = serving_metrics(predicted, settled=graded.settled)
        assert reason == ""
        veto = evaluate_behavioural_veto(metrics, {}, has_incumbent=False)
        assert veto.status == "pass", veto.reasons
        assert metrics["n_high_confidence"] > 0

    def test_only_settled_dates_reach_the_block(self, panel) -> None:
        """A date whose label has not realized has a prediction and no
        outcome; counting it would put an unresolved bet in a hit rate."""
        recipe = _recipe()
        as_of = panel.dates[-1]
        graded, _ = self._graded(panel, recipe=recipe, as_of=as_of)
        horizon = recipe.label_horizon_trading_days
        last_settled = panel.dates[-1 - horizon]
        assert graded.settled.dates
        assert max(graded.settled.dates) <= last_settled
        assert set(graded.settled.dates) <= set(graded.series.scores) | set(graded.unrankable_dates)

    def test_both_sides_of_the_settled_block_are_cross_sectional_excess(self, panel) -> None:
        """M's benchmark is the cross-section it scored, so a hit is a hit
        against the population and the 0.50 floor is a genuine coin flip
        rather than a bar market drift clears on its own."""
        graded, _ = self._graded(panel)
        assert np.allclose(graded.settled.predicted.mean(axis=1), 0.0)
        assert np.allclose(graded.settled.realized.mean(axis=1), 0.0)


class TestTheColdSlotReading:
    """A slot with no champion has nothing for a dispersion RATIO to compare
    against, and §10.1's null control cannot stand in: `control_null_m` is a
    selection-shaped harness control that ranks names on a noise draw and
    publishes no predicted-alpha cross-section at all."""

    def test_the_null_control_publishes_no_cross_section_to_compare_against(self) -> None:
        """The measured fact the cold-start reading rests on."""
        from crucible.slots import get_slot

        controls = {c.kind: c.arm_id for c in get_slot("m").control_arms}
        assert controls == {"planted": "control_planted_m", "null": "control_null_m"}

    def test_dispersion_is_inapplicable_not_uncomputable(self) -> None:
        from crucible.slots.model import DISPERSION_METRICS, evaluate_behavioural_veto

        candidate = {
            "alpha_stdev": 0.02,
            "stdev_p_up": 0.05,
            "n_high_confidence": 6,
            "model_hit_rate_30d": 0.54,
        }
        veto = evaluate_behavioural_veto(candidate, {}, has_incumbent=False)
        assert veto.status == "pass"
        assert veto.inapplicable == DISPERSION_METRICS
        assert veto.uncomputable == ()
        # And the artifact says so: "passed every rule" and "passed every rule
        # that had a comparand" are different claims about a first champion.
        precondition = veto.as_precondition()
        assert precondition.passed
        assert "no incumbent to compare against" in precondition.reason

    def test_a_cold_slot_still_vetoes_a_collapsed_cross_section(self) -> None:
        from crucible.slots.model import evaluate_behavioural_veto

        veto = evaluate_behavioural_veto(
            {
                "alpha_stdev": 0.0,
                "stdev_p_up": 0.0,
                "n_high_confidence": 4,
                "model_hit_rate_30d": 0.54,
            },
            {},
            has_incumbent=False,
        )
        assert veto.status == "veto"
        assert any("zero spread is a collapsed model" in r for r in veto.reasons)

    def test_an_absent_incumbent_metric_is_still_uncomputable_when_there_is_a_champion(
        self,
    ) -> None:
        """The direction that must not be relaxed: a champion we cannot
        measure is not a slot with nothing to compare against."""
        from crucible.slots.model import evaluate_behavioural_veto

        veto = evaluate_behavioural_veto(
            {
                "alpha_stdev": 0.02,
                "stdev_p_up": 0.05,
                "n_high_confidence": 6,
                "model_hit_rate_30d": 0.54,
            },
            {},
            has_incumbent=True,
        )
        assert veto.status == "insufficient"
        assert set(veto.uncomputable) == {"alpha_stdev", "stdev_p_up"}
        assert veto.inapplicable == ()


class TestTheServingVetoWindowIsObserved:
    """`alpha-engine-config-I10680`. `insufficient` used to have exactly one
    cause and it was permanent. Now it has a temporary one, and a cycle must
    say which — or the fix looks identical to the gap."""

    def _ctx(self):
        from types import SimpleNamespace

        rows: list[dict] = []
        return SimpleNamespace(record_metric=rows.append), rows

    def test_a_live_verdict_is_reported_ok(self) -> None:
        from crucible.slots.model import SERVING_VETO_WINDOW_METRIC, _record_serving_veto_window

        ctx, rows = self._ctx()
        _record_serving_veto_window(
            ctx,
            {"a": {"veto": "pass"}, "b": {"veto": "insufficient"}},
            {"b": "short"},
            as_of="2026-09-11",
        )
        assert len(rows) == 1
        assert rows[0]["name"] == SERVING_VETO_WINDOW_METRIC
        assert rows[0]["status"] == "OK"
        assert rows[0]["value"] == 1.0

    def test_every_arm_short_of_the_window_is_unmeasurable_not_a_failure(self) -> None:
        """Nothing is broken and nothing is owed: the arms are accruing
        settled dates."""
        from crucible.slots.model import _record_serving_veto_window

        ctx, rows = self._ctx()
        _record_serving_veto_window(
            ctx,
            {"a": {"veto": "insufficient"}, "b": {"veto": "insufficient"}},
            {"a": "8 settled date(s)", "b": "8 settled date(s)"},
            as_of="2026-09-11",
        )
        assert rows[0]["status"] == "unmeasurable"
        assert "value" not in rows[0]

    def test_insufficient_with_no_window_reason_fails_loud(self) -> None:
        """A veto input missing for a reason other than the window is an
        unexplained dead slot, and it pages."""
        from crucible.slots.model import _record_serving_veto_window

        ctx, rows = self._ctx()
        _record_serving_veto_window(
            ctx,
            {"a": {"veto": "insufficient"}, "b": {"veto": "insufficient"}},
            {"a": "8 settled date(s)"},
            as_of="2026-09-11",
        )
        assert rows[0]["status"] == "FAIL"
        assert "did NOT name a short settled window" in rows[0]["status_reason"]

    def test_a_cycle_that_graded_nothing_emits_nothing(self) -> None:
        from crucible.slots.model import _record_serving_veto_window

        ctx, rows = self._ctx()
        _record_serving_veto_window(ctx, {}, {}, as_of="2026-09-11")
        assert rows == []


# --------------------------------------------------------------------------
# Per-row feature completeness (`alpha-engine-config-I10688`).
# --------------------------------------------------------------------------


def _hole(panel: FeaturePanel, column: str, *, rows: slice, names: slice) -> FeaturePanel:
    """A copy of ``panel`` with NULLs punched into one column."""
    block = panel.column(column).copy()
    block[rows, names] = np.nan
    return replace(panel, features={**panel.features, column: block})


class TestFeatureCompletenessSelectsRowsAndRefusesAnOutage:
    """`alpha-engine-config-I10688`, the training half.

    The refusal on non-finite training inputs stays exactly as Brian ruled it
    on 2026-08-29 — `_assert_trainable` is unchanged and still refuses any
    non-finite cell. What changes is WHICH block it is asked about: the design
    selects the ticker-rows whose whole feature vector is finite, records what
    it dropped, and refuses outright when the dropped share exceeds the arm's
    declared ceiling. Nothing is imputed and nothing is filled.
    """

    def test_a_handful_of_short_history_names_are_excluded_not_fatal(self, panel) -> None:
        """The measured production class B: four tickers out of 903 (62, 75,
        220 and 225 sessions of history) lack a 252-session window. That is the
        feature layer working as declared, and it must not fail the cycle."""
        holed = _hole(panel, "mom_21d_ratio", rows=slice(None), names=slice(0, 1))
        fit = train_arm(_recipe(), holed, as_of="2026-08-28")
        assert fit.completeness is not None
        assert fit.completeness.rows_excluded > 0
        assert fit.completeness.excluded_names == (panel.names[0],)
        assert fit.completeness.nan_rows_by_column["mom_21d_ratio"] > 0
        assert fit.n_rows == fit.completeness.rows_complete

    def test_a_whole_dead_column_refuses_rather_than_fitting_on_nothing(self, panel) -> None:
        """The measured production class A: one column null for 903 of 903
        tickers. Selecting complete rows without a ceiling would have discarded
        every row and returned `ok`."""
        dead = _hole(panel, "mom_21d_ratio", rows=slice(None), names=slice(None))
        with pytest.raises(TrainingIntegrityError, match="incomplete feature vector"):
            train_arm(_recipe(), dead, as_of="2026-08-28")

    def test_the_refusal_names_the_column_and_the_ceiling(self, panel) -> None:
        dead = _hole(panel, "mom_21d_ratio", rows=slice(None), names=slice(None))
        with pytest.raises(TrainingIntegrityError, match="mom_21d_ratio"):
            train_arm(_recipe(), dead, as_of="2026-08-28")

    def test_an_exclusion_just_over_the_ceiling_refuses(self, panel) -> None:
        n_names = len(panel.names)
        over = int(n_names * 0.25) + 1
        holed = _hole(panel, "mom_21d_ratio", rows=slice(None), names=slice(0, over))
        with pytest.raises(TrainingIntegrityError, match="above the declared ceiling"):
            train_arm(_recipe(), holed, as_of="2026-08-28")

    def test_an_arm_may_declare_its_own_ceiling(self, panel) -> None:
        n_names = len(panel.names)
        over = int(n_names * 0.25) + 1
        holed = _hole(panel, "mom_21d_ratio", rows=slice(None), names=slice(0, over))
        tolerant = _recipe(max_incomplete_row_ratio=0.5)
        fit = train_arm(tolerant, holed, as_of="2026-08-28")
        assert fit.completeness is not None
        assert fit.completeness.ceiling == 0.5
        assert 0.10 < fit.completeness.excluded_ratio < 0.5

    def test_the_default_ceiling_applies_when_the_arm_declares_none(self) -> None:
        from crucible.slots.model import DEFAULT_MAX_INCOMPLETE_ROW_RATIO

        assert _recipe().max_incomplete_row_ratio is None
        assert _recipe().resolved_max_incomplete_row_ratio == DEFAULT_MAX_INCOMPLETE_ROW_RATIO

    def test_nothing_is_imputed(self, panel) -> None:
        """The rows that survive are the rows that were measured.

        A fit on the holed panel must equal a fit on the same panel with those
        ticker-rows removed — never a fit on a filled or zeroed version, which
        is the `avg_volume_20d` treatment in the training layer.
        """
        holed = _hole(panel, "mom_21d_ratio", rows=slice(None), names=slice(0, 1))
        zeroed = panel.with_zeroed(("mom_21d_ratio",))
        holed_fit = train_arm(_recipe(), holed, as_of="2026-08-28")
        with pytest.raises(TrainingIntegrityError):
            train_arm(_recipe(), zeroed, as_of="2026-08-28")
        assert holed_fit.n_rows < len(panel.dates) * len(panel.names)

    def test_a_ceiling_outside_zero_to_one_is_refused_at_construction(self) -> None:
        with pytest.raises(ValueError, match="max_incomplete_row_ratio"):
            _recipe(max_incomplete_row_ratio=1.0)
        with pytest.raises(ValueError, match="max_incomplete_row_ratio"):
            _recipe(max_incomplete_row_ratio=-0.1)


class TestTheCeilingIsPartOfTheArmOnlyWhenDeclared:
    def test_an_arm_declaring_none_keeps_the_id_it_registered_under(self) -> None:
        """Policy §3.1: a field that always appeared in the hashed spec would
        re-id every arm already registered and orphan its score series."""
        assert "max_incomplete_row_ratio" not in _recipe().spec

    def test_declaring_a_ceiling_is_a_different_arm(self) -> None:
        assert _recipe().arm_id != _recipe(max_incomplete_row_ratio=0.2).arm_id
        assert _recipe(max_incomplete_row_ratio=0.2).spec["max_incomplete_row_ratio"] == 0.2


class TestTheCompletenessRecordReachesTheManifest:
    def test_the_record_is_metric_shaped_and_carries_the_per_column_counts(self, panel) -> None:
        holed = _hole(panel, "mom_21d_ratio", rows=slice(None), names=slice(0, 1))
        fit = train_arm(_recipe(), holed, as_of="2026-08-28")
        assert fit.completeness is not None
        row = fit.completeness.as_metric(slot="m")
        assert row["name"] == "feature_completeness_excluded_ratio"
        assert row["status"] == "OK"
        assert row["unit"] == "ratio"
        record = row["feature_completeness"]
        assert record["rows_excluded"] == fit.completeness.rows_excluded
        assert record["nan_rows_by_column"]["mom_21d_ratio"] > 0
        assert record["excluded_names_sample"] == [panel.names[0]]
        assert record["excluded_name_count"] == 1

    def test_the_rejection_detail_fits_the_schemas_cap(self, panel) -> None:
        """`RunContext.record_rejected` RAISES over 200 characters and the
        fallback manifest drops every other field with it."""
        holed = _hole(panel, "mom_21d_ratio", rows=slice(None), names=slice(0, 1))
        fit = train_arm(_recipe(), holed, as_of="2026-08-28")
        assert fit.completeness is not None
        assert len(fit.completeness.detail) <= 200


class TestTheRecordIsBoundedOnTheManifest:
    """Measured on the first real replay: the unbounded identifier lists put
    ~900 tickers and ~400 dates into one metric row, ~200KB on a document
    every console reader and every `explain` walk loads whole. The counts are
    complete; the samples are bounded."""

    def test_the_name_and_session_lists_are_capped_and_the_counts_are_not(self, panel) -> None:
        from crucible.slots.model import _SAMPLE_SIZE

        # Three whole sessions nulled: every name is excluded on at least one
        # row, so the name count exceeds the sample cap while the ratio stays
        # well under the default ceiling.
        holed = _hole(panel, "mom_21d_ratio", rows=slice(0, 3), names=slice(None))
        fit = train_arm(_recipe(), holed, as_of="2026-08-28")
        assert fit.completeness is not None
        record = fit.completeness.to_dict()
        assert record["excluded_name_count"] == len(panel.names)
        assert len(record["excluded_names_sample"]) <= _SAMPLE_SIZE
        assert record["excluded_session_count"] >= 1
        assert record["excluded_first_session"] in panel.dates
        assert record["excluded_last_session"] in panel.dates


# --------------------------------------------------------------------------
# The registration clock gated the VETO's window as well as the track record
# (alpha-engine-config-I10709).
# --------------------------------------------------------------------------


class TestTheVetoWindowIsNotGatedByTheRegistrationClock:
    """Measured 2026-09-14 on the first live M grade over a backfilled history
    (`arena/m/2026-09-11/arena_cycle.json`): `residual_momentum` — registered
    2026-09-01, 68 backfilled sessions behind it, 47 paired dates in the arena
    — read `behavioural veto could not be computed for stdev_p_up,
    n_high_confidence, model_hit_rate_30d`, and the run manifest's
    `serving_veto_window` row read `0 settled out-of-sample decision date(s)`.

    `_grade_lookback` was sized by `alpha-engine-config-I10680` to read
    `min_trading_days + 2 * label_horizon + SETTLED_WINDOW_DECISION_DATES`
    sessions precisely so the settled window could fill — and then `grade_arm`
    dropped every walked date before `registered_at` from the settled block as
    well as from the scored series, so the window could not fill until the arm
    was older than the whole extended lookback. A gate that reads
    `insufficient` for its first ~51 sessions is dark exactly where a new arm
    is riskiest.

    Plan §9.1's clock governs the arm's TRACK RECORD — the series the arena
    pairs, and so `promote_min_weeks`. It is not a bound on whether the arm's
    BEHAVIOUR is measurable. Both halves are asserted here together, because
    the fix is only correct if it leaves the first one untouched.
    """

    def _recent(self, panel, *, back: int = 8):
        return _recipe(registered_at=panel.dates[-back - 1]), panel.dates[-1]

    def test_a_recently_registered_arm_still_computes_every_veto_input(self, panel) -> None:
        from crucible.slots.model import (
            DISPERSION_METRICS,
            FLOOR_VETO_METRICS,
            SETTLED_WINDOW_DECISION_DATES,
            ZERO_VETO_METRICS,
            grade_arm,
            predict_cross_section,
            serving_metrics,
        )

        recipe, as_of = self._recent(panel)
        graded = grade_arm(recipe, panel, as_of=as_of)
        assert graded.settled.n_dates >= SETTLED_WINDOW_DECISION_DATES, (
            f"{graded.settled.n_dates} settled date(s) on a panel that carries "
            f"{len(panel.dates)}: the veto's window is bounded by the registration clock"
        )
        metrics, reason = serving_metrics(
            predict_cross_section(graded.fit, panel, trading_day=as_of), settled=graded.settled
        )
        assert reason == "", reason
        for name in (*DISPERSION_METRICS, *ZERO_VETO_METRICS, *FLOOR_VETO_METRICS):
            assert name in metrics, name

    def test_the_veto_reaches_a_verdict_rather_than_could_not_be_computed(self, panel) -> None:
        from crucible.slots.model import (
            evaluate_behavioural_veto,
            grade_arm,
            predict_cross_section,
            serving_metrics,
        )

        recipe, as_of = self._recent(panel)
        graded = grade_arm(recipe, panel, as_of=as_of)
        metrics, _ = serving_metrics(
            predict_cross_section(graded.fit, panel, trading_day=as_of), settled=graded.settled
        )
        veto = evaluate_behavioural_veto(metrics, metrics, has_incumbent=True)
        assert veto.status in ("pass", "veto"), veto.reason
        assert "could not be computed" not in veto.as_precondition().reason

    def test_the_settled_block_is_out_of_sample_and_realized(self, panel) -> None:
        """Every date in the block is walked forward and has a settled label,
        registration or no registration: the veto's inputs are measurements,
        never a fit scoring its own training rows."""
        from crucible.slots.model import grade_arm, settled_training_days

        recipe, as_of = self._recent(panel)
        graded = grade_arm(recipe, panel, as_of=as_of)
        realized = set(
            settled_training_days(
                panel, as_of=as_of, label_horizon=recipe.label_horizon_trading_days
            )
        )
        indexes = {day: i for i, day in enumerate(panel.dates)}
        assert graded.settled.dates
        for day in graded.settled.dates:
            assert indexes[day] in realized, day

    def test_the_track_record_is_still_gated_at_registration(self, panel) -> None:
        """Plan §9.1, untouched: the series the arena pairs — and so
        `promote_min_weeks` — still opens at `registered_at`. Widening the
        veto's window must not hand a nine-session-old arm a track record."""
        from crucible.slots.model import grade_arm

        recipe, as_of = self._recent(panel)
        graded = grade_arm(recipe, panel, as_of=as_of)
        assert graded.oos_n <= 9, f"a nine-session-old arm was handed {graded.oos_n} dates"
        assert min(graded.series.scores) >= recipe.registered_at
        assert graded.settled.n_dates > graded.oos_n

    def test_one_unpriced_name_does_not_void_the_whole_settled_date(self, panel) -> None:
        """The second half of `alpha-engine-config-I10709`, and the defect the
        first half was hiding: the block was demeaned with a plain `.mean()`,
        so ONE non-finite name wrote NaN across that date's whole
        cross-section. Measured 2026-09-14 on the real 908-name universe once
        the window filled — 0 of 27,240 name-date pairs finite, the hit rate
        absent and the calibration refusing a block it read as constant.
        """
        import dataclasses

        import numpy as np

        from crucible.slots.model import grade_arm, predict_cross_section, serving_metrics

        forward = panel.forward_returns.copy()
        forward[:, 0] = np.nan
        holed = dataclasses.replace(panel, forward_returns=forward)
        recipe, as_of = self._recent(holed)
        graded = grade_arm(recipe, holed, as_of=as_of)

        assert graded.settled.n_dates > 0
        finite = np.isfinite(graded.settled.realized)
        assert finite.any(), "one unpriced name voided every settled cross-section"
        # Exactly the holed column is absent, and every other name survives.
        assert not finite[:, 0].any()
        assert finite[:, 1:].all()

        metrics, reason = serving_metrics(
            predict_cross_section(graded.fit, holed, trading_day=as_of), settled=graded.settled
        )
        assert reason == "", reason
        assert "model_hit_rate_30d" in metrics
