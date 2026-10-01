"""The M slot's predictive std and its calibration gate (`alpha-engine-config-I11791`).

The executor's conviction gate returns q = 1 — turnover throttling OFF — when
the champion's predictions carry no `predicted_alpha_std`. Until this change
the M slot's predictions document carried `predicted_alpha` only and its
Bayesian-ridge fit discarded the noise precision it learned, so the first M
champion promotion would have switched that risk control off in silence.

Three properties are held here:

1. every estimator serves a std, positive, with the method that produced it —
   a posterior for `bayesian_ridge`, an out-of-sample residual for the rest —
   or omits it with a reason, never with a zero;
2. the grader measures that std against the walk-forward's out-of-sample
   error, and a std shrunk 3x fails the band;
3. a failing calibration makes the arm INELIGIBLE on the real cycle job.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from crucible.slots import model as model_module
from crucible.slots.inputs import (
    SCHEMA_PATH,
    UNCERTAINTY_OMITTED,
    UNCERTAINTY_OOS_RESIDUAL,
    UNCERTAINTY_POSTERIOR,
    ArmPredictionsContractError,
    PredictionUncertainty,
    write_arm_predictions,
)
from crucible.slots.model import (
    GAUSSIAN_ONE_SIGMA_COVERAGE,
    SETTLED_WINDOW_DECISION_DATES,
    Z_VAR_BAND,
    CPCVSpec,
    EstimatorSpec,
    FeaturePanel,
    ModelRecipe,
    PredictiveStd,
    TrainingWindowSpec,
    evaluate_uncertainty_calibration,
    grade_arm,
    predict_cross_section,
    predict_uncertainty,
    produce_arm_predictions,
    train_arm,
)

#: The noise std planted in every synthetic label below. Known, so a std the
#: harness serves can be held to it.
NOISE = 0.7

_LGBM = {
    "num_boost_round": 40,
    "seed": 7,
    "num_leaves": 7,
    "min_child_samples": 10,
    "learning_rate": 0.1,
}

_PARAMS = {
    "bayesian_ridge": {},
    "ols": {},
    "ridge": {"alpha": 1.0},
    "fixed_linear": {"weights": {"x_ratio": 0.5}},
    "lightgbm": dict(_LGBM),
}


def _panel(*, n_days: int = 90, n_names: int = 40, seed: int = 5) -> FeaturePanel:
    """A label with a planted linear edge and Gaussian noise of std :data:`NOISE`."""
    from tests.support.panels import trading_days

    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=(n_days, n_names))
    return FeaturePanel(
        dates=tuple(trading_days(n_days)),
        names=tuple(f"T{i:03d}" for i in range(n_names)),
        features={"x_ratio": x},
        forward_returns=0.5 * x + rng.normal(0.0, NOISE, size=(n_days, n_names)),
        feature_version="v1",
    )


def _recipe(kind: str, *, minimum: int = 20) -> ModelRecipe:
    return ModelRecipe(
        name=f"arm_{kind}",
        features=("x_ratio",),
        estimator=EstimatorSpec(kind=kind, params=_PARAMS[kind]),
        label_horizon_trading_days=1,
        refit_cadence_trading_days=5,
        training_window=TrainingWindowSpec(kind="expanding", min_trading_days=minimum),
        cpcv=CPCVSpec(n_groups=4, k_test=1, embargo_trading_days=1),
        registered_at="2020-01-01",
    )


class _Ctx:
    """The two recording calls `produce_arm_predictions` makes, captured."""

    def __init__(self) -> None:
        self.outputs: dict[str, bytes] = {}
        self.metrics: list[dict] = []

    def record_metric(self, metric: dict) -> None:
        self.metrics.append(metric)

    def record_output(self, key: str, payload: bytes, schema_version: str = "v1") -> str:
        self.outputs[key] = payload
        return key


def _validator() -> Draft202012Validator:
    return Draft202012Validator(json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))


class TestEveryEstimatorServesAStd:
    @pytest.mark.parametrize("kind", sorted(_PARAMS))
    def test_the_document_carries_a_positive_std_for_every_scored_name(self, kind) -> None:
        panel = _panel()
        day = panel.dates[-2]
        fit = train_arm(_recipe(kind), panel, as_of=day)
        ctx = _Ctx()
        key = produce_arm_predictions(ctx, fit=fit, panel=panel, trading_day=day)
        document = json.loads(ctx.outputs[key])

        assert not list(_validator().iter_errors(document))
        assert set(document["predicted_alpha_std"]) == set(document["predicted_alpha"])
        assert all(v > 0 for v in document["predicted_alpha_std"].values())
        assert document["predicted_alpha_std_note"]
        expected = UNCERTAINTY_POSTERIOR if kind == "bayesian_ridge" else UNCERTAINTY_OOS_RESIDUAL
        assert document["predicted_alpha_std_method"] == expected

    @pytest.mark.parametrize("kind", ["ols", "ridge", "fixed_linear", "bayesian_ridge"])
    def test_a_linear_fit_s_std_recovers_the_planted_noise(self, kind) -> None:
        """Honest, not merely positive: on a label whose noise std is known,
        the served std lands on it."""
        panel = _panel()
        day = panel.dates[-2]
        fit = train_arm(_recipe(kind), panel, as_of=day)
        predicted = predict_cross_section(fit, panel, trading_day=day)
        std = predict_uncertainty(fit, panel, trading_day=day, names=tuple(predicted))
        assert float(np.median(list(std.std.values()))) == pytest.approx(NOISE, rel=0.1)

    def test_the_posterior_splits_into_aleatoric_and_epistemic(self) -> None:
        panel = _panel()
        day = panel.dates[-2]
        fit = train_arm(_recipe("bayesian_ridge"), panel, as_of=day)
        predicted = predict_cross_section(fit, panel, trading_day=day)
        std = predict_uncertainty(fit, panel, trading_day=day, names=tuple(predicted))
        assert std.method == UNCERTAINTY_POSTERIOR
        for name in predicted:
            total, ale, epi = std.std[name], std.aleatoric[name], std.epistemic[name]
            assert total**2 == pytest.approx(ale**2 + epi**2, rel=1e-12)
            assert epi > 0
        # The aleatoric term is the learned noise precision, one number for
        # the batch; the epistemic term moves with how far a name sits from
        # the training centre.
        assert len(set(std.aleatoric.values())) == 1
        assert len(set(std.epistemic.values())) > 1
        assert next(iter(std.aleatoric.values())) == pytest.approx(NOISE, rel=0.1)

    def test_the_posterior_reproduces_scikit_learns_predictive_std(self) -> None:
        """Reference values from `BayesianRidge().predict(q, return_std=True)`
        (scikit-learn 1.9.1, defaults) on the matrix
        `tests/test_slot_model_estimators.py` pins the coefficients on —
        computed once and pinned, since this module does not import sklearn."""
        rng = np.random.default_rng(11)
        matrix = rng.normal(size=(200, 3))
        labels = matrix @ np.array([0.5, -0.2, 0.0]) + rng.normal(scale=0.7, size=200)
        recipe = ModelRecipe(
            name="br",
            features=("a_ratio", "b_ratio", "c_ratio"),
            estimator=EstimatorSpec(kind="bayesian_ridge", params={}),
            label_horizon_trading_days=1,
            refit_cadence_trading_days=5,
            training_window=TrainingWindowSpec(kind="expanding", min_trading_days=20),
            cpcv=CPCVSpec(n_groups=4, k_test=1, embargo_trading_days=1),
            registered_at="2020-01-01",
        )
        _, _, posterior = model_module._fit_bayesian_ridge_posterior(recipe, matrix, labels)
        query = np.array([[0.0, 0.0, 0.0], [1.0, -1.0, 0.5], [3.0, 2.0, -2.0]])
        std = posterior.predict_std(query)
        assert np.allclose(
            std.total, [0.7450169451866122, 0.748531300039945, 0.7746788968530622], atol=1e-9
        )
        assert 1.0 / posterior.noise_variance == pytest.approx(1.8017176250582898, rel=1e-9)

    def test_a_tree_s_std_is_out_of_sample_not_its_in_sample_fit(self) -> None:
        """A boosted ensemble fits its training rows closely; a std read off
        those residuals would be the fabricated small sigma the issue forbids."""
        panel = _panel()
        day = panel.dates[-2]
        fit = train_arm(_recipe("lightgbm"), panel, as_of=day)
        usable = model_module.settled_training_days(panel, as_of=day, label_horizon=1)
        rows = model_module._flatten(np.asarray(usable), len(panel.names))
        in_sample = panel.forward_returns.reshape(-1)[rows] - fit.fitted.predict(
            model_module._design(fit.recipe, panel, rows)
        )
        served = math.sqrt(fit.fitted.uncertainty.residual_variance)
        assert served > float(np.sqrt(np.mean(in_sample**2)))
        assert "purged 4-fold" in fit.fitted.uncertainty.note


class TestAnOmittedStdIsAbsentNeverZero:
    def test_too_few_days_for_the_purged_folds_omits_with_a_reason(self) -> None:
        panel = _panel()
        reading = model_module._purged_kfold_uncertainty(_recipe("ols"), panel, [0, 1, 2])
        assert reading.method == UNCERTAINTY_OMITTED
        assert "cannot be split" in reading.note
        assert reading.predict_std(np.zeros((3, 1))).total is None

    def test_the_document_records_why_and_carries_no_std_field(self) -> None:
        ctx = _Ctx()
        key = write_arm_predictions(
            ctx,
            arm_id="m:arm:0123abcd",
            trading_day="2026-08-28",
            feature_version="v1",
            predicted_alpha={"AAA": 0.01, "BBB": -0.02},
            uncertainty=PredictionUncertainty(method=UNCERTAINTY_OMITTED, note="no fold fitted"),
        )
        document = json.loads(ctx.outputs[key])
        assert "predicted_alpha_std" not in document
        assert document["predicted_alpha_std_method"] == UNCERTAINTY_OMITTED
        assert document["predicted_alpha_std_note"] == "no fold fitted"

    def test_a_zero_std_is_refused_by_the_schema(self) -> None:
        with pytest.raises(ArmPredictionsContractError, match="predicted_alpha_std"):
            write_arm_predictions(
                _Ctx(),
                arm_id="m:arm:0123abcd",
                trading_day="2026-08-28",
                feature_version="v1",
                predicted_alpha={"AAA": 0.01},
                uncertainty=PredictionUncertainty(
                    method=UNCERTAINTY_OOS_RESIDUAL, note="n", std={"AAA": 0.0}
                ),
            )

    def test_a_std_that_does_not_cover_the_cross_section_is_refused(self) -> None:
        with pytest.raises(ArmPredictionsContractError, match="cover exactly"):
            write_arm_predictions(
                _Ctx(),
                arm_id="m:arm:0123abcd",
                trading_day="2026-08-28",
                feature_version="v1",
                predicted_alpha={"AAA": 0.01, "BBB": 0.02},
                uncertainty=PredictionUncertainty(
                    method=UNCERTAINTY_OOS_RESIDUAL, note="n", std={"AAA": 0.1}
                ),
            )

    def test_a_posterior_without_its_components_is_refused_by_the_schema(self) -> None:
        with pytest.raises(ArmPredictionsContractError, match="aleatoric"):
            write_arm_predictions(
                _Ctx(),
                arm_id="m:arm:0123abcd",
                trading_day="2026-08-28",
                feature_version="v1",
                predicted_alpha={"AAA": 0.01},
                uncertainty=PredictionUncertainty(
                    method=UNCERTAINTY_POSTERIOR, note="n", std={"AAA": 0.1}
                ),
            )

    def test_a_document_written_before_the_field_is_still_valid(self) -> None:
        assert not list(
            _validator().iter_errors(
                {
                    "schema_version": "arm_predictions.v1",
                    "arm_id": "m:arm:0123abcd",
                    "trading_day": "2026-08-28",
                    "feature_version": "v1",
                    "predicted_alpha": {"AAA": 0.01},
                }
            )
        )


def _known_noise_block(scale: float, *, n_dates: int = 40, n_names: int = 200, seed: int = 3):
    """Predictions, realized labels and a std of ``scale`` x the true noise."""
    rng = np.random.default_rng(seed)
    signal = rng.normal(0.0, 0.01, size=(n_dates, n_names))
    noise = 0.02
    realized = signal + rng.normal(0.0, noise, size=(n_dates, n_names))
    sigma = np.full((n_dates, n_names), noise * scale)
    return signal, realized, sigma


class TestTheCalibrationStatistics:
    def test_a_calibrated_std_passes_with_gaussian_coverage(self) -> None:
        predicted, realized, sigma = _known_noise_block(1.0)
        reading = evaluate_uncertainty_calibration(predicted, realized, sigma, method="m")
        assert reading.status == "pass", reading.reason
        assert reading.z_var == pytest.approx(1.0, abs=0.05)
        assert reading.coverage_1sigma == pytest.approx(GAUSSIAN_ONE_SIGMA_COVERAGE, abs=0.02)
        assert reading.as_precondition().passed

    def test_the_cross_sectional_ir_tracks_the_out_of_sample_ic(self) -> None:
        """std(signal) / noise = 0.5 and corr(signal, label) = 0.447 here, so
        a calibrated std puts the gate's IR within first-order reach of the IC."""
        predicted, realized, sigma = _known_noise_block(1.0)
        reading = evaluate_uncertainty_calibration(predicted, realized, sigma, method="m")
        assert reading.ir_xs == pytest.approx(0.5, abs=0.03)
        assert reading.oos_ic == pytest.approx(0.43, abs=0.03)
        assert 0.9 < reading.ir_xs_to_ic < 1.4

    def test_a_std_shrunk_three_fold_fails_as_overconfident(self) -> None:
        predicted, realized, sigma = _known_noise_block(1.0 / 3.0)
        reading = evaluate_uncertainty_calibration(predicted, realized, sigma, method="m")
        assert reading.status == "fail"
        assert reading.z_var == pytest.approx(9.0, rel=0.06)
        assert reading.z_var > Z_VAR_BAND[1]
        assert "OVERCONFIDENT" in reading.reason
        assert reading.ir_xs == pytest.approx(1.5, abs=0.1)
        assert not reading.as_precondition().passed

    def test_a_std_inflated_three_fold_fails_as_underconfident(self) -> None:
        predicted, realized, sigma = _known_noise_block(3.0)
        reading = evaluate_uncertainty_calibration(predicted, realized, sigma, method="m")
        assert reading.status == "fail"
        assert "UNDERCONFIDENT" in reading.reason

    def test_no_std_is_insufficient_and_fails_the_precondition(self) -> None:
        predicted, realized, _ = _known_noise_block(1.0)
        reading = evaluate_uncertainty_calibration(
            predicted, realized, None, method=UNCERTAINTY_OMITTED, omitted_reason="no fold"
        )
        assert reading.status == "insufficient"
        assert "no fold" in reading.reason
        assert not reading.as_precondition().passed

    def test_a_short_window_is_insufficient_and_says_so(self) -> None:
        predicted, realized, sigma = _known_noise_block(
            1.0, n_dates=SETTLED_WINDOW_DECISION_DATES - 1
        )
        reading = evaluate_uncertainty_calibration(predicted, realized, sigma, method="m")
        assert reading.status == "insufficient"
        assert "window being short" in reading.reason


class TestTheGraderCalibratesTheServedStd:
    """Through `grade_arm`'s walk-forward, so the std that is calibrated is
    the std the produce path serves (`FittedEstimator.predict_std`)."""

    @pytest.mark.parametrize("kind", ["bayesian_ridge", "ols"])
    def test_a_correct_std_passes_on_out_of_sample_dates(self, kind) -> None:
        panel = _panel()
        graded = grade_arm(_recipe(kind), panel, as_of=panel.dates[-1])
        reading = graded.calibration
        assert reading.n_dates >= SETTLED_WINDOW_DECISION_DATES
        assert reading.status == "pass", reading.reason
        assert reading.method == (
            UNCERTAINTY_POSTERIOR if kind == "bayesian_ridge" else UNCERTAINTY_OOS_RESIDUAL
        )
        payload = reading.to_dict()
        for name in ("z_var", "coverage_1sigma", "ir_xs", "oos_ic", "ir_xs_to_ic"):
            assert payload[name] is not None, name

    def test_the_same_std_shrunk_three_fold_fails(self, monkeypatch) -> None:
        original = model_module.FittedEstimator.predict_std

        def shrunk(self, matrix):
            std = original(self, matrix)
            return PredictiveStd(method=std.method, note=std.note, total=std.total / 3.0)

        monkeypatch.setattr(model_module.FittedEstimator, "predict_std", shrunk)
        panel = _panel()
        reading = grade_arm(_recipe("bayesian_ridge"), panel, as_of=panel.dates[-1]).calibration
        assert reading.status == "fail"
        assert reading.z_var > Z_VAR_BAND[1]
