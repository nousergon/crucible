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

import pytest
from nousergon_lib.arena.engine import TrainingIntegrityError

from crucible.slots.model import (
    MIN_DISPERSION_RATIO,
    CPCVSpec,
    ModelRecipe,
    cpcv_oos_ic,
    evaluate_behavioural_veto,
    evaluate_input_completeness,
    grade_arm,
    load_model_recipes,
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
    def test_grade_produces_a_per_date_series_the_arena_can_pair(self, panel) -> None:
        grade = grade_arm(_recipe(), panel, as_of="2026-08-28")
        assert grade.series.arm_id == _recipe().arm_id
        assert grade.series.scores
        assert all(isinstance(v, float) for v in grade.series.scores.values())

    def test_the_benchmark_is_the_scored_cross_section_never_spy(self) -> None:
        from crucible.slots import get_slot

        assert get_slot("m").benchmark == "population"


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
                    "  feature_version: v1",
                ]
            ),
            encoding="utf-8",
        )
        recipes = load_model_recipes(tmp_path)
        assert len(recipes) == 1
        assert recipes[0].arm_id.startswith("m:residual_momentum:")

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


def _recipe(**over) -> ModelRecipe:
    from crucible.slots.model import EstimatorSpec, TrainingWindowSpec

    kwargs = dict(
        name="residual_momentum",
        features=("mom_21d_ratio", "vol_21d_ratio"),
        estimator=EstimatorSpec(kind="ridge", params={"alpha": 1.0}),
        label_horizon_trading_days=21,
        refit_cadence_trading_days=5,
        training_window=TrainingWindowSpec(kind="expanding", min_trading_days=40),
        cpcv=CPCVSpec(n_groups=6, k_test=2, embargo_trading_days=2),
        feature_version="v1",
    )
    kwargs.update(over)
    return ModelRecipe(**kwargs)


@pytest.fixture
def panel():
    from tests.support.panels import synthetic_panel

    return synthetic_panel(n_days=160, n_names=25, seed=7)
