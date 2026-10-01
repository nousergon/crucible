"""The M slot's factor-residual target (`alpha-engine-config-I11791`).

An arm declaring `target: factor_residual_forward_return` is FITTED to the
21-session forward return with its point-in-time exposure to the attribution
spec's beta/sector/size factors removed, and GRADED — like every other arm —
against the slot's signed forward return. Four properties are held here:

1. the residual carries no factor return (exactly, on a noise-free panel);
2. no look-ahead: a name's betas on session ``t`` are a function of returns
   realized by close ``t`` and nothing later — asserted by truncation and by a
   regime break that a leaky estimator cannot pass;
3. grading is unchanged: an arm with an oracle for the RESIDUAL is scored
   against the RAW return, so an easier label cannot win by being easier;
4. the arm still serves a positive std, and it calibrates.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

from crucible.attribution import AttributionFactorParams, FactorDef
from crucible.slots import model as model_module
from crucible.slots.inputs import PER_ARM_REFUSALS, FactorResidualUnavailableError
from crucible.slots.model import (
    FACTOR_RESIDUAL_TARGET,
    TARGETS,
    CPCVSpec,
    EstimatorSpec,
    FeaturePanel,
    ModelRecipe,
    TrainingWindowSpec,
    design_panel,
    factor_residual_panel,
    grade_arm,
    neutralize_cross_section,
    point_in_time_factor_betas,
    produce_arm_predictions,
    train_arm,
)
from tests.support.panels import trading_days

HORIZON = 21
WINDOW = 60
PROXIES = ("SPY", "IWM", "XLK")

#: Three factors, one per category the attribution spec requires: market
#: (SPY), size as the IWM-minus-SPY spread, and one sector ETF.
PARAMS = AttributionFactorParams(
    factors={
        "market": FactorDef(category="beta", proxy="SPY"),
        "size": FactorDef(category="size", proxy="IWM", short_proxy="SPY"),
        "sector_technology": FactorDef(category="sector", proxy="XLK"),
    },
    benchmark_proxy="SPY",
)
#: `factor_residual_panel` orders factors by name; the generator below does too.
FACTOR_ORDER = tuple(sorted(PARAMS.factors))


def _recipe(
    *,
    kind: str = "ridge",
    params: dict | None = None,
    features: tuple[str, ...] = ("x_ratio",),
    target: str | None = FACTOR_RESIDUAL_TARGET,
    window: int | None = WINDOW,
    neutralize: bool = False,
    minimum: int = 60,
) -> ModelRecipe:
    return ModelRecipe(
        name="factor_residual_arm",
        features=features,
        estimator=EstimatorSpec(kind=kind, params=params if params is not None else {"alpha": 1.0}),
        label_horizon_trading_days=HORIZON,
        refit_cadence_trading_days=5,
        training_window=TrainingWindowSpec(kind="expanding", min_trading_days=minimum),
        cpcv=CPCVSpec(n_groups=4, k_test=1, embargo_trading_days=2),
        registered_at="2020-01-01",
        target=target,
        factor_beta_window_trading_days=window,
        neutralize_features=neutralize,
    )


class _World:
    """A synthetic market: three factors, constant betas, a planted drift.

    Daily LOG returns are ``drift + beta . f + idio``, so the model the
    residualizer fits is the model that generated the data and the 21-session
    decomposition is exact. Proxies carry their factor's cumulative return as
    their log close; IWM carries SPY's plus the size spread, so IWM - SPY IS
    the size factor.
    """

    def __init__(
        self,
        *,
        n_days: int = 320,
        n_names: int = 40,
        seed: int = 7,
        idio_sd: float = 0.004,
        beta_break: int | None = None,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.n_days, self.n_names = n_days, n_names
        factor_sd = {"market": 0.010, "sector_technology": 0.008, "size": 0.006}
        self.factor_daily = np.column_stack(
            [rng.normal(0.0, factor_sd[f], size=n_days) for f in FACTOR_ORDER]
        )
        self.factor_daily[0] = 0.0
        self.betas = np.column_stack(
            [
                rng.uniform(0.0, 2.0, size=n_names),  # market
                rng.normal(0.0, 0.8, size=n_names),  # sector_technology
                rng.normal(0.0, 0.8, size=n_names),  # size
            ]
        )
        self.drift = rng.normal(0.0, 0.0005, size=n_names)
        self.idio = rng.normal(0.0, idio_sd, size=(n_days, n_names)) if idio_sd else 0.0
        self.idio = np.zeros((n_days, n_names)) + self.idio
        self.idio[0] = 0.0
        loading = np.broadcast_to(self.betas, (n_days, n_names, 3)).copy()
        if beta_break is not None:
            # Name 0: NO market exposure before the break, 2.0 from it on.
            loading[:beta_break, 0, :] = 0.0
            loading[beta_break:, 0, :] = (2.0, 0.0, 0.0)
        self.loading = loading
        daily = self.drift + np.einsum("tnk,tk->tn", loading, self.factor_daily) + self.idio
        daily[0] = 0.0
        stock_log = np.cumsum(daily, axis=0) + np.log(50.0)
        market, sector, size = (
            self.factor_daily[:, FACTOR_ORDER.index(f)]
            for f in (
                "market",
                "sector_technology",
                "size",
            )
        )
        spy = np.cumsum(market) + np.log(400.0)
        iwm = spy + np.cumsum(size) + np.log(0.5)
        xlk = np.cumsum(sector) + np.log(200.0)
        self.log_closes = np.column_stack([stock_log, spy, iwm, xlk])
        self.names = tuple(f"T{i:03d}" for i in range(n_names)) + PROXIES

    def idio_forward(self) -> np.ndarray:
        """The stock-specific log return over each label window, drift included."""
        own = self.drift + self.idio
        cumulative = np.cumsum(own, axis=0)
        out = np.full((self.n_days, self.n_names), np.nan)
        out[:-HORIZON] = cumulative[HORIZON:] - cumulative[:-HORIZON]
        return out

    def panel(self, *, x: np.ndarray | None = None) -> FeaturePanel:
        closes = np.exp(self.log_closes)
        forward = np.full_like(closes, np.nan)
        forward[:-HORIZON] = closes[HORIZON:] / closes[:-HORIZON] - 1.0
        features = {"close_raw": closes}
        if x is not None:
            padded = np.zeros_like(closes)
            padded[:, : self.n_names] = x
            # Proxies get an uninformative column on the stocks' scale, never
            # a hole: their residual is ~0 by construction (each IS a factor).
            padded[:, self.n_names :] = np.random.default_rng(1).normal(
                0.0, 0.001, size=(self.n_days, len(PROXIES))
            )
            features["x_ratio"] = padded
        return FeaturePanel(
            dates=tuple(trading_days(self.n_days)),
            names=self.names,
            features=features,
            forward_returns=forward,
            feature_version="v1",
        )


def _oracle_x(world: _World, *, noise: float, seed: int = 3) -> np.ndarray:
    """A feature that knows the RESIDUAL (plus noise) and nothing about factors."""
    rng = np.random.default_rng(seed)
    fwd = np.nan_to_num(world.idio_forward(), nan=0.0)
    return fwd + rng.normal(0.0, noise, size=fwd.shape)


class TestTheRecipe:
    def test_the_target_is_in_the_closed_vocabulary(self) -> None:
        assert FACTOR_RESIDUAL_TARGET in TARGETS

    def test_the_target_requires_a_declared_beta_window(self) -> None:
        with pytest.raises(ValueError, match="factor_beta_window_trading_days"):
            _recipe(window=None)

    def test_a_window_too_short_to_estimate_betas_is_refused(self) -> None:
        with pytest.raises(ValueError, match="below"):
            _recipe(window=5)

    @pytest.mark.parametrize("kwargs", [{"window": WINDOW}, {"window": None, "neutralize": True}])
    def test_factor_fields_on_another_target_are_refused(self, kwargs) -> None:
        with pytest.raises(ValueError, match="only mean something"):
            _recipe(target=None, **kwargs)

    def test_the_new_fields_are_hashed_and_absent_when_undeclared(self) -> None:
        plain = _recipe(target=None, window=None)
        assert "factor_beta_window_trading_days" not in plain.spec
        assert "neutralize_features" not in plain.spec
        residual = _recipe()
        assert residual.spec["target"] == FACTOR_RESIDUAL_TARGET
        assert residual.spec["factor_beta_window_trading_days"] == WINDOW
        assert "neutralize_features" not in residual.spec
        neutral = _recipe(neutralize=True)
        assert neutral.spec["neutralize_features"] is True
        assert len({plain.arm_id, residual.arm_id, neutral.arm_id}) == 3
        assert _recipe(window=126).arm_id != residual.arm_id

    def test_the_loader_reads_the_fields_and_refuses_a_string_boolean(self, tmp_path) -> None:
        document = {
            "slot": "m",
            "name": "factor_residual_stack",
            "registered_at": "2026-10-01",
            "spec": {
                "features": ["x_ratio"],
                "estimator": {"kind": "bayesian_ridge"},
                "label_horizon_trading_days": 21,
                "refit_cadence_trading_days": 5,
                "training_window": {"kind": "expanding", "min_trading_days": 504},
                "cpcv": {"n_groups": 6, "k_test": 2, "embargo_trading_days": 2},
                "target": FACTOR_RESIDUAL_TARGET,
                "factor_beta_window_trading_days": 126,
                "neutralize_features": True,
            },
        }
        recipe = model_module._parse_model_recipe(yaml.safe_dump(document).encode(), "a.yaml")
        assert recipe.target == FACTOR_RESIDUAL_TARGET
        assert recipe.factor_beta_window_trading_days == 126
        assert recipe.neutralize_features is True

        document["spec"]["neutralize_features"] = "false"
        with pytest.raises(ValueError, match="true or false"):
            model_module._parse_model_recipe(yaml.safe_dump(document).encode(), "a.yaml")


class TestTheResidualIsOrthogonalToTheFactors:
    def test_on_a_noise_free_panel_the_residual_is_exactly_the_drift(self) -> None:
        """With no idiosyncratic noise the betas are recovered exactly, so
        what is left of a 21-session log return is 21 sessions of the name's
        own drift — not one basis point of factor return."""
        world = _World(idio_sd=0.0)
        panel, betas = factor_residual_panel(_recipe(), world.panel(), PARAMS)
        label = panel.target_returns[FACTOR_RESIDUAL_TARGET][:, : world.n_names]
        settled = np.isfinite(label).all(axis=1)
        assert settled.sum() > 200
        expected = np.broadcast_to(HORIZON * world.drift, label.shape)
        assert np.allclose(label[settled], expected[settled], atol=1e-10)
        assert np.allclose(betas[:, : world.n_names], world.betas, atol=1e-8)
        # ...and the raw label is NOT the drift: the factor move was real.
        raw = np.log1p(panel.forward_returns[:, : world.n_names])
        assert float(np.nanstd(raw - expected)) > 0.01

    def test_with_noise_the_residual_carries_no_factor_loading(self) -> None:
        world = _World(n_days=600, idio_sd=0.004)
        panel, _ = factor_residual_panel(_recipe(), world.panel(), PARAMS)
        label = panel.target_returns[FACTOR_RESIDUAL_TARGET][:, : world.n_names]
        raw = np.log1p(panel.forward_returns[:, : world.n_names])
        keep = np.isfinite(label).all(axis=1)
        # The factors' own forward log returns over the same windows.
        cumulative = np.cumsum(world.factor_daily, axis=0)
        factor_fwd = np.full_like(world.factor_daily, np.nan)
        factor_fwd[:-HORIZON] = cumulative[HORIZON:] - cumulative[:-HORIZON]
        factor_fwd = factor_fwd[WINDOW:][keep]
        design = np.hstack([np.ones((factor_fwd.shape[0], 1)), factor_fwd])

        def r_squared(label: np.ndarray) -> np.ndarray:
            coef, *_ = np.linalg.lstsq(design, label, rcond=None)
            fitted = design @ coef
            centred = label - label.mean(axis=0)
            return 1.0 - ((label - fitted) ** 2).sum(axis=0) / (centred**2).sum(axis=0)

        on_raw, *_ = np.linalg.lstsq(design, raw[keep], rcond=None)
        # The raw label loads on the factors at the planted betas...
        assert np.abs(on_raw[1:].T - world.betas).mean() < 0.1
        assert float(np.median(r_squared(raw[keep]))) > 0.6
        # ...and what they explain of the residual is the sampling floor of a
        # three-regressor fit on ~25 independent 21-session windows.
        assert float(np.median(r_squared(label[keep]))) < 0.2
        # The factor component is gone up to beta-estimation error: what the
        # residual differs from the planted stock-specific return by is a
        # small fraction of what the raw label differed from it by.
        idio = world.idio_forward()[WINDOW:][keep]
        contamination = float(np.var(label[keep] - idio))
        factor_part = float(np.var(raw[keep] - idio))
        assert contamination < 0.05 * factor_part


class TestNoLookAhead:
    def test_a_beta_is_unchanged_by_every_session_after_it(self) -> None:
        """Truncating the return history at session ``i`` leaves row ``i``'s
        betas bit-identical. An estimator reading ANY later session — a
        centred window, a full-sample fit, the label window — changes them."""
        world = _World(idio_sd=0.004)
        stock = np.diff(world.log_closes[:, : world.n_names], axis=0, prepend=np.nan)
        factors = world.factor_daily.copy()
        factors[0] = np.nan
        full = point_in_time_factor_betas(stock, factors, window=WINDOW)
        for i in (WINDOW, WINDOW + 7, 150, 250, world.n_days - HORIZON - 1):
            truncated = point_in_time_factor_betas(stock[: i + 1], factors[: i + 1], window=WINDOW)
            assert np.array_equal(truncated[i], full[i], equal_nan=True)
            assert np.isfinite(full[i]).all()

    def test_a_beta_regime_break_is_not_seen_before_it_happens(self) -> None:
        """Name 0 has NO market exposure before session B and 2.0 from B on.
        The label starting at B - 1 is realized entirely inside the new
        regime, so an estimator with look-ahead — any window touching the
        label's sessions — reads a market beta pulled toward 2. Point-in-time
        betas on B - 1 have seen only the old regime and read ~0, and the
        residual on that row therefore still carries the 2x market move,
        which is the honest answer on the day."""
        brk = 200
        world = _World(idio_sd=0.002, beta_break=brk)
        panel, betas = factor_residual_panel(_recipe(), world.panel(), PARAMS)
        row = brk - 1 - WINDOW  # panel rows are trimmed by WINDOW
        market = FACTOR_ORDER.index("market")
        assert abs(float(betas[row, 0, market])) < 0.1
        # The leaky reading, for contrast: a window ending at the label's end.
        stock = np.diff(world.log_closes[:, :1], axis=0, prepend=np.nan)
        factors = world.factor_daily.copy()
        factors[0] = np.nan
        leaky = point_in_time_factor_betas(stock, factors, window=WINDOW)[brk - 1 + HORIZON]
        assert float(leaky[0, market]) > 0.5
        # ...and well after the break the point-in-time beta has caught up.
        assert float(betas[brk + WINDOW + 5 - WINDOW, 0, market]) == pytest.approx(2.0, abs=0.1)

    def test_the_label_at_t_reads_nothing_after_t_plus_horizon(self) -> None:
        world = _World(idio_sd=0.004)
        full, _ = factor_residual_panel(_recipe(), world.panel(), PARAMS)
        cut = 240
        short, _ = factor_residual_panel(_recipe(), world.panel().head(cut + 1), PARAMS)
        last_settled = cut - HORIZON - WINDOW
        a = full.target_returns[FACTOR_RESIDUAL_TARGET][: last_settled + 1]
        b = short.target_returns[FACTOR_RESIDUAL_TARGET][: last_settled + 1]
        assert np.allclose(a, b, atol=1e-12, equal_nan=True)
        assert np.isfinite(b[-1, : world.n_names]).all()
        # The rows whose window runs past the cut are unsettled, not guessed.
        assert np.isnan(short.target_returns[FACTOR_RESIDUAL_TARGET][-1]).all()


class TestGradingIsUnchanged:
    def test_an_oracle_for_the_residual_is_scored_against_the_raw_return(self) -> None:
        """The arm's one feature knows the RESIDUAL. It is fitted to the
        residual, but every walked date is scored as the rank IC of its
        prediction against the RAW forward return — the factor noise it does
        not predict is charged to it, exactly as for any arm.

        Exact, not statistical: a one-feature linear fit with a positive slope
        ranks names by the feature, so each date's score must EQUAL the rank
        IC of the feature against the raw return, and differ from its IC
        against the residual it was fitted to."""
        # Deep enough that the unsettled tail stays under the default
        # incomplete-row ceiling inside every CPCV fold.
        world = _World(n_days=480, idio_sd=0.004)
        x = _oracle_x(world, noise=0.001)
        panel, _ = factor_residual_panel(_recipe(), world.panel(x=x), PARAMS)
        grade = grade_arm(_recipe(), panel, as_of=panel.dates[-1])
        assert grade.status == "ok"
        assert float(grade.fit.coefficients[0]) > 0
        scores = grade.series.scores
        assert len(scores) > 50
        feature = panel.column("x_ratio")
        residual = panel.target_returns[FACTOR_RESIDUAL_TARGET]
        against_residual = []
        for day, score in scores.items():
            i = panel.dates.index(day)
            assert score == pytest.approx(
                model_module._rank_ic(feature[i], panel.forward_returns[i]), abs=1e-12
            )
            against_residual.append(model_module._rank_ic(feature[i], residual[i]))
        # The easier label would have scored far higher; it is not what counts.
        assert float(np.mean(against_residual)) > 0.8
        assert float(np.mean(list(scores.values()))) < float(np.mean(against_residual)) - 0.15
        # The CPCV battery is graded on the raw return too.
        assert grade.cpcv.status == "ok"
        assert max(grade.cpcv.ics) < float(np.mean(against_residual))

    def test_the_raw_forward_return_is_left_untouched(self) -> None:
        world = _World()
        raw = world.panel()
        panel, _ = factor_residual_panel(_recipe(), raw, PARAMS)
        assert np.array_equal(panel.forward_returns, raw.forward_returns[WINDOW:], equal_nan=True)
        assert panel.dates == raw.dates[WINDOW:]
        assert "close_raw" not in panel.features


class TestTheStdIsServedAndCalibrated:
    @pytest.mark.parametrize(
        ("kind", "params"), [("bayesian_ridge", {}), ("ridge", {"alpha": 1.0})]
    )
    def test_the_arm_serves_a_calibrated_std_in_label_units(self, kind, params) -> None:
        world = _World(n_days=380, idio_sd=0.004)
        x = _oracle_x(world, noise=0.02)
        panel, _ = factor_residual_panel(
            _recipe(kind=kind, params=params), world.panel(x=x), PARAMS
        )
        recipe = _recipe(kind=kind, params=params)
        day = panel.dates[-1]
        fit = train_arm(recipe, panel, as_of=day)

        class _Ctx:
            def __init__(self) -> None:
                self.outputs: dict[str, bytes] = {}

            def record_metric(self, metric: dict) -> None:
                pass

            def record_output(self, key: str, payload: bytes, schema_version: str = "") -> str:
                self.outputs[key] = payload
                return key

        ctx = _Ctx()
        key = produce_arm_predictions(ctx, fit=fit, panel=panel, trading_day=day)
        document = json.loads(ctx.outputs[key])
        alpha = np.array(list(document["predicted_alpha"].values()))
        std = np.array(list(document["predicted_alpha_std"].values()))
        assert set(document["predicted_alpha_std"]) == set(document["predicted_alpha"])
        assert (std > 0).all()
        # Units: a 21-session log return. The served dispersion sits on the
        # residual label's own scale, not on a z-score's.
        label_sd = float(np.nanstd(panel.target_returns[FACTOR_RESIDUAL_TARGET]))
        assert 0.2 < float(np.std(alpha)) / label_sd < 1.5
        assert float(np.median(std)) < 3 * label_sd

        grade = grade_arm(recipe, panel, as_of=day)
        assert grade.calibration.status == "pass", grade.calibration.reason
        lo, hi = model_module.Z_VAR_BAND
        assert lo <= grade.calibration.z_var <= hi

    def test_the_std_is_measured_against_the_residual_not_the_raw_return(self) -> None:
        """The std is a claim about the error of the number served, and the
        number served predicts the residual. Against the raw return the same
        std would read overconfident by the whole factor variance."""
        world = _World(n_days=380, idio_sd=0.004)
        panel, _ = factor_residual_panel(
            _recipe(), world.panel(x=_oracle_x(world, noise=0.02)), PARAMS
        )
        recipe = _recipe(kind="bayesian_ridge", params={})
        grade = grade_arm(recipe, panel, as_of=panel.dates[-1])
        rows = np.arange(len(panel.dates) * len(panel.names))
        fitted = model_module._target_labels(recipe, panel, rows)
        assert np.array_equal(
            fitted,
            panel.target_returns[FACTOR_RESIDUAL_TARGET].reshape(-1),
            equal_nan=True,
        )
        assert grade.calibration.status == "pass"


class _Source:
    """The two attributes `design_panel` reads off a feature-layer source."""

    def __init__(self, panel: FeaturePanel) -> None:
        self._panel = panel
        self.store = object()
        self.calls: list[dict] = []

    def panel(self, **kwargs) -> FeaturePanel:
        self.calls.append(kwargs)
        return self._panel


class TestDesignPanel:
    def test_it_reads_the_beta_window_deeper_and_trims_it_back_off(self) -> None:
        world = _World()
        source = _Source(world.panel(x=_oracle_x(world, noise=0.02)))
        panel = design_panel(
            _recipe(),
            source=source,
            trading_day=source._panel.dates[-1],
            lookback_trading_days=100,
            attribution=PARAMS,
        )
        (call,) = source.calls
        assert call["lookback_trading_days"] == 100 + WINDOW
        assert call["columns"] == ("x_ratio", "close_raw")
        assert FACTOR_RESIDUAL_TARGET in panel.target_returns
        assert len(panel.dates) == world.n_days - WINDOW

    def test_neutralized_features_carry_no_beta_exposure(self) -> None:
        world = _World()
        # A feature that is mostly market beta: neutralizing must remove it.
        x = np.broadcast_to(world.betas[:, 0], (world.n_days, world.n_names)).copy()
        x += np.random.default_rng(4).normal(0.0, 0.1, size=x.shape)
        source = _Source(world.panel(x=x))
        panel = design_panel(
            _recipe(neutralize=True),
            source=source,
            trading_day=source._panel.dates[-1],
            attribution=PARAMS,
        )
        column = panel.column("x_ratio")[:, : world.n_names]
        for t in (0, 50, len(panel.dates) - 1):
            raw_corr = np.corrcoef(x[t + WINDOW], world.betas[:, 0])[0, 1]
            new_corr = np.corrcoef(column[t], world.betas[:, 0])[0, 1]
            assert raw_corr > 0.9
            assert abs(new_corr) < 0.1

    def test_neutralization_is_ols_on_the_betas_per_session(self) -> None:
        rng = np.random.default_rng(2)
        betas = rng.normal(size=(3, 30, 2))
        values = 0.5 + betas @ np.array([1.0, -2.0]) + rng.normal(0.0, 0.01, size=(3, 30))
        out = neutralize_cross_section(values, betas)
        for t in range(3):
            design = np.hstack([np.ones((30, 1)), betas[t]])
            assert np.abs(design.T @ out[t]).max() < 1e-10

    def test_a_missing_proxy_is_a_per_arm_refusal(self) -> None:
        world = _World()
        full = world.panel()
        keep = [i for i, n in enumerate(full.names) if n != "XLK"]
        thin = FeaturePanel(
            dates=full.dates,
            names=tuple(full.names[i] for i in keep),
            features={k: v[:, keep] for k, v in full.features.items()},
            forward_returns=full.forward_returns[:, keep],
            feature_version="v1",
        )
        with pytest.raises(FactorResidualUnavailableError) as exc:
            factor_residual_panel(_recipe(), thin, PARAMS)
        assert exc.value.unresolvable == ("XLK",)
        assert FactorResidualUnavailableError in PER_ARM_REFUSALS

    def test_an_unpriced_proxy_session_is_refused_not_zero_filled(self) -> None:
        world = _World()
        panel = world.panel()
        closes = panel.features["close_raw"].copy()
        closes[30, panel.names.index("IWM")] = np.nan
        broken = FeaturePanel(
            dates=panel.dates,
            names=panel.names,
            features={"close_raw": closes},
            forward_returns=panel.forward_returns,
            feature_version="v1",
        )
        with pytest.raises(FactorResidualUnavailableError, match="no close") as exc:
            factor_residual_panel(_recipe(), broken, PARAMS)
        assert exc.value.unresolvable == ("IWM",)

    def test_an_absent_attribution_spec_is_a_per_arm_refusal(self) -> None:
        class _EmptyStore:
            def exists(self, key: str) -> bool:
                return False

        world = _World()
        source = _Source(world.panel())
        source.store = _EmptyStore()
        with pytest.raises(FactorResidualUnavailableError) as exc:
            design_panel(_recipe(), source=source, trading_day=source._panel.dates[-1])
        assert exc.value.unresolvable == ("strategy/slots/attribution.yaml",)
        assert not source.calls

    def test_a_panel_built_without_the_label_refuses_to_fit(self) -> None:
        world = _World()
        panel = world.panel(x=_oracle_x(world, noise=0.02))
        with pytest.raises(ValueError, match="design_panel"):
            train_arm(_recipe(), panel, as_of=panel.dates[-1])
