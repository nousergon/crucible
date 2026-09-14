"""The M slot's estimator vocabulary beyond ridge/OLS (`alpha-engine-config-I10695`).

Porting v1's serving model `v3.0-meta` needs three fits the slot did not
carry: a LightGBM head, a fixed-formula head and the scikit-learn
`BayesianRidge` stacker. Each is held here to the slot's two standing
properties — a recipe describes exactly what is fitted (unknown parameters
refuse), and a refit reproduces byte for byte (plan §9.1) — and to the one
seam every fit goes through, so grading and serving cannot score an arm by
different code.
"""

from __future__ import annotations

import numpy as np
import pytest
from nousergon_lib.arena.engine import TrainingIntegrityError

from crucible.slots import model as model_module
from crucible.slots.model import (
    CPCVSpec,
    EstimatorSpec,
    FeaturePanel,
    ModelRecipe,
    TrainingWindowSpec,
    _rank_ic,
    grade_arm,
    predict_cross_section,
    train_arm,
)

_LGBM = {
    "num_boost_round": 40,
    "seed": 7,
    "num_leaves": 7,
    "min_child_samples": 10,
    "learning_rate": 0.1,
}


def _panel(*, n_days: int, n_names: int, seed: int, label) -> FeaturePanel:
    from tests.support.panels import trading_days

    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=(n_days, n_names))
    v = rng.uniform(0.5, 2.0, size=(n_days, n_names))
    noise = rng.normal(0.0, 1.0, size=(n_days, n_names))
    return FeaturePanel(
        dates=tuple(trading_days(n_days)),
        names=tuple(f"T{i:03d}" for i in range(n_names)),
        features={"x_ratio": x, "v_ratio": v},
        forward_returns=label(x, v, noise),
        feature_version="v1",
    )


def _recipe(kind: str, params: dict, *, features=("x_ratio",), target=None, minimum=20):
    return ModelRecipe(
        name=f"arm_{kind}",
        features=tuple(features),
        estimator=EstimatorSpec(kind=kind, params=params),
        label_horizon_trading_days=1,
        refit_cadence_trading_days=5,
        training_window=TrainingWindowSpec(kind="expanding", min_trading_days=minimum),
        cpcv=CPCVSpec(n_groups=4, k_test=1, embargo_trading_days=1),
        registered_at="2020-01-01",
        target=target,
    )


def _oos_ic(recipe: ModelRecipe, panel: FeaturePanel, *, train_to: int) -> float:
    fit = train_arm(recipe, panel, as_of=panel.dates[train_to])
    ics = []
    for day in range(train_to + 1, len(panel.dates) - 1):
        predicted = predict_cross_section(fit, panel, trading_day=panel.dates[day])
        actual = panel.forward_returns[day]
        ics.append(_rank_ic(np.array([predicted[n] for n in panel.names]), actual))
    return float(np.mean([ic for ic in ics if ic is not None]))


class TestTheVocabularyIsClosed:
    def test_every_estimator_kind_has_a_fitter(self) -> None:
        """A kind added to `_ESTIMATORS` without a fitter would fall through
        `_fit_estimator` to the LightGBM branch; this fails first."""
        panel = _panel(n_days=40, n_names=20, seed=1, label=lambda x, v, e: x + e)
        params = {
            "ridge": {"alpha": 1.0},
            "ols": {},
            "bayesian_ridge": {},
            "fixed_linear": {"weights": {"x_ratio": 1.0}},
            "lightgbm": dict(_LGBM),
        }
        assert set(params) == set(model_module._ESTIMATORS)
        assert set(model_module._ESTIMATOR_PARAMS) == set(model_module._ESTIMATORS)
        for kind, p in params.items():
            fit = train_arm(_recipe(kind, p), panel, as_of=panel.dates[-2])
            assert fit.fitted.kind == kind

    @pytest.mark.parametrize(
        ("kind", "params", "match"),
        [
            ("ridge", {"alpha": 1.0, "lambda": 2.0}, "does not read"),
            ("ols", {"alpha": 1.0}, "does not read"),
            ("lightgbm", {"seed": 1}, "requires"),
            ("lightgbm", {"num_boost_round": 10}, "requires"),
            ("lightgbm", {**_LGBM, "num_threads": 8}, "harness-owned"),
            ("lightgbm", {**_LGBM, "deterministic": False}, "harness-owned"),
            ("lightgbm", {**_LGBM, "num_leafs": 31}, "does not read"),
            ("lightgbm", {**_LGBM, "objective": "lambdarank"}, "objective"),
            ("lightgbm", {**_LGBM, "num_boost_round": 0}, "num_boost_round"),
            ("lightgbm", {**_LGBM, "seed": "42"}, "seed"),
            ("fixed_linear", {"weights": {}}, "non-empty"),
        ],
    )
    def test_a_parameter_the_fitter_would_not_honour_is_refused(self, kind, params, match) -> None:
        with pytest.raises(ValueError, match=match):
            EstimatorSpec(kind=kind, params=params)

    def test_fixed_linear_weights_must_cover_exactly_the_design_columns(self) -> None:
        with pytest.raises(ValueError, match="fixed_linear weights"):
            _recipe("fixed_linear", {"weights": {"x_ratio": 1.0}}, features=("x_ratio", "v_ratio"))

    def test_an_unknown_target_is_refused(self) -> None:
        with pytest.raises(ValueError, match="target"):
            _recipe("ridge", {}, target="log_forward_return")


class TestTheTargetField:
    def test_an_arm_declaring_no_target_keeps_the_id_it_had_before_the_field(self) -> None:
        recipe = _recipe("ridge", {"alpha": 1.0})
        assert "target" not in recipe.spec
        declared = _recipe("ridge", {"alpha": 1.0}, target="abs_forward_return")
        assert declared.spec["target"] == "abs_forward_return"
        assert declared.arm_id != recipe.arm_id

    def test_a_magnitude_head_is_fitted_to_the_absolute_return(self) -> None:
        """v1's volatility head: `v` sets how FAR a name moves, not which way."""
        panel = _panel(n_days=120, n_names=40, seed=3, label=lambda x, v, e: v * e)
        signed = train_arm(_recipe("ols", {}, features=("v_ratio",)), panel, as_of=panel.dates[-2])
        magnitude = train_arm(
            _recipe("ols", {}, features=("v_ratio",), target="abs_forward_return"),
            panel,
            as_of=panel.dates[-2],
        )
        assert abs(float(signed.coefficients[0])) < 0.1
        assert float(magnitude.coefficients[0]) > 0.5


class TestLightGBM:
    def test_a_refit_reproduces_the_same_predictions_byte_for_byte(self) -> None:
        panel = _panel(n_days=80, n_names=30, seed=5, label=lambda x, v, e: x**2 + 0.5 * e)
        recipe = _recipe(
            "lightgbm",
            {**_LGBM, "feature_fraction": 0.5, "bagging_fraction": 0.7, "bagging_freq": 1},
            features=("x_ratio", "v_ratio"),
        )
        first = train_arm(recipe, panel, as_of=panel.dates[60])
        second = train_arm(recipe, panel, as_of=panel.dates[60])
        a = predict_cross_section(first, panel, trading_day=panel.dates[70])
        b = predict_cross_section(second, panel, trading_day=panel.dates[70])
        assert list(a.values()) == list(b.values())

    def test_it_measures_a_non_linear_edge_a_linear_arm_cannot(self) -> None:
        panel = _panel(n_days=120, n_names=40, seed=9, label=lambda x, v, e: x**2 + 0.5 * e)
        tree = _oos_ic(_recipe("lightgbm", dict(_LGBM)), panel, train_to=80)
        linear = _oos_ic(_recipe("ridge", {"alpha": 1.0}), panel, train_to=80)
        assert tree > 0.5
        assert abs(linear) < 0.15

    def test_it_grades_through_the_walk_forward_seam(self) -> None:
        panel = _panel(n_days=90, n_names=25, seed=13, label=lambda x, v, e: x + e)
        grade = grade_arm(_recipe("lightgbm", dict(_LGBM)), panel, as_of=panel.dates[-1])
        assert grade.status == "ok", grade.reason
        assert grade.oos_n > 0

    def test_a_tree_fit_has_no_coefficients_to_read(self) -> None:
        panel = _panel(n_days=40, n_names=20, seed=1, label=lambda x, v, e: x + e)
        fit = train_arm(_recipe("lightgbm", dict(_LGBM)), panel, as_of=panel.dates[-2])
        with pytest.raises(AttributeError, match="no coefficient"):
            _ = fit.coefficients
        with pytest.raises(AttributeError, match="no intercept"):
            _ = fit.intercept


class TestFixedLinear:
    def test_the_prediction_is_the_declared_formula(self) -> None:
        panel = _panel(n_days=30, n_names=15, seed=2, label=lambda x, v, e: e)
        weights = {"x_ratio": 0.4, "v_ratio": -0.3}
        recipe = _recipe(
            "fixed_linear",
            {"weights": weights, "intercept": -0.25},
            features=("x_ratio", "v_ratio"),
            minimum=1,
        )
        fit = train_arm(recipe, panel, as_of=panel.dates[25])
        predicted = predict_cross_section(fit, panel, trading_day=panel.dates[25])
        expected = 0.4 * panel.features["x_ratio"][25] - 0.3 * panel.features["v_ratio"][25] - 0.25
        assert np.allclose([predicted[n] for n in panel.names], expected, rtol=0, atol=1e-12)


class TestBayesianRidge:
    def test_it_reproduces_scikit_learns_bayesian_ridge(self) -> None:
        """Reference values from `sklearn.linear_model.BayesianRidge()` (defaults,
        scikit-learn 1.x) on this exact matrix, computed once and pinned — the
        estimator v1's `v3.0-meta` stacker is fitted with."""
        rng = np.random.default_rng(11)
        matrix = rng.normal(size=(200, 3))
        labels = matrix @ np.array([0.5, -0.2, 0.0]) + rng.normal(scale=0.7, size=200)
        recipe = _recipe("bayesian_ridge", {}, features=("a_ratio", "b_ratio", "c_ratio"))
        coefficients, intercept = model_module._fit_bayesian_ridge(recipe, matrix, labels)
        assert np.allclose(
            coefficients,
            [0.48871459315208776, -0.19603210385956366, -0.034867399267189694],
            rtol=0,
            atol=1e-9,
        )
        assert intercept == pytest.approx(-0.012300448067335618, abs=1e-9)

    def test_an_evidence_iteration_that_does_not_settle_raises(self) -> None:
        rng = np.random.default_rng(11)
        matrix = rng.normal(size=(200, 3))
        labels = matrix @ np.array([0.5, -0.2, 0.0]) + rng.normal(scale=0.7, size=200)
        recipe = _recipe(
            "bayesian_ridge", {"max_iter": 1}, features=("a_ratio", "b_ratio", "c_ratio")
        )
        with pytest.raises(TrainingIntegrityError, match="did not converge"):
            model_module._fit_bayesian_ridge(recipe, matrix, labels)
