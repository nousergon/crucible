"""Regime-conditioned feature columns (alpha-engine-config-I14084, epic I13842).

`market_stress_zscore` is the day's `crucible.regime.stress_score`; four
interaction columns multiply a per-name cross-sectional z-score by it. They are
INERT: no arm recipe reads them, so no arm id, risk limit or size moves.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from crucible.data.point_in_time import UnavailablePointInTimeSource
from crucible.features import CATALOG, build_features, catalog_column_depths
from crucible.features.compute import _MARKET_STRESS_DEPTH_TRADING_DAYS, LIQUIDITY_FLOOR_VAR
from crucible.regime import stress_score
from crucible.slots.model import _parse_model_recipe  # pyright: ignore[reportPrivateUsage]

STRESS = "market_stress_zscore"
INTERACTIONS = {
    "momentum_20d_zscore_x_stress_zscore": "momentum_20d_zscore",
    "mom_12_1_zscore_x_stress_zscore": "mom_12_1_zscore",
    "residual_momentum_252d_skip21d_zscore_x_stress_zscore": (
        "residual_momentum_252d_skip21d_zscore"
    ),
    "volatility_20d_x_stress_zscore": "volatility_20d_ratio",
}
NEW_COLUMNS = (STRESS, *INTERACTIONS)

TICKERS = 12
SESSIONS = 365
SHOCK_SESSION = 350


@pytest.fixture(autouse=True)
def _liquidity_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(LIQUIDITY_FLOOR_VAR, "1000000")


def _no_point_in_time(panel: pd.DataFrame, as_of: object = None) -> object:
    day = as_of if as_of is not None else max(panel["trading_day"])
    return UnavailablePointInTimeSource(reason="synthetic fixture carries no fundamentals").load(
        trading_day=day, symbols=sorted(set(panel["ticker"]))
    )


def _panel(sessions: int = SESSIONS) -> pd.DataFrame:
    """A synthetic long panel. T11 is illiquid and T10 lists late (short history)."""
    rng = np.random.default_rng(14084)
    start = dt.date(2024, 1, 1)
    market = rng.normal(0.0003, 0.006, size=sessions)
    if SHOCK_SESSION < sessions:
        market[SHOCK_SESSION] = -0.08
    rows: list[dict[str, object]] = []
    for t in range(TICKERS):
        price = 50.0 + t
        first = 200 if t == 10 else 0
        for i in range(first, sessions):
            price *= float(np.exp(market[i] + rng.normal(0.0, 0.015)))
            rows.append(
                {
                    "trading_day": start + dt.timedelta(days=i),
                    "ticker": f"T{t:02d}",
                    "open_raw": price,
                    "high_raw": price * 1.01,
                    "low_raw": price * 0.99,
                    "close_raw": price,
                    "volume_raw": 1.0 if t == 11 else 1.0e6,
                }
            )
    return pd.DataFrame(rows)


def _day(panel: pd.DataFrame, session: int) -> dt.date:
    return sorted(panel["trading_day"].unique())[session]


@pytest.fixture(scope="module")
def built() -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = _panel()
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(LIQUIDITY_FLOOR_VAR, "1000000")
        features, _ = build_features(panel, point_in_time=_no_point_in_time(panel))  # type: ignore[arg-type]
    return panel, features


def test_the_new_columns_are_catalogued_with_declared_shape() -> None:
    by_name = {spec.name: spec for spec in CATALOG}
    depths = catalog_column_depths()
    assert by_name[STRESS].market_wide is True
    for column in NEW_COLUMNS:
        assert column in by_name and column in depths
        assert by_name[column].unit == "zscore"
    for column in INTERACTIONS:
        assert by_name[column].cross_sectional is True
        assert by_name[column].market_wide is False
        assert STRESS in by_name[column].inputs


def test_market_stress_is_identical_across_the_cross_section_and_is_the_regime_score(
    built: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    panel, features = built
    values = features[STRESS]
    assert values.notna().all()
    assert values.nunique() == 1
    # Independent derivation of the score's four panel inputs, then the regime module.
    frame = panel.sort_values(["ticker", "trading_day"]).copy()
    log_close = np.log(frame["close_raw"])
    by_ticker = log_close.groupby(frame["ticker"], sort=False)
    frame["r1"] = by_ticker.diff(1)
    frame["market_return_1d_log_return"] = frame.groupby("trading_day")["r1"].transform("mean")
    frame["momentum_20d_log_return"] = by_ticker.diff(20)
    for window in (50, 200):
        mean = frame.groupby("ticker", sort=False)["close_raw"].transform(
            lambda s, w=window: s.rolling(w, min_periods=w).mean()
        )
        frame[f"close_to_sma{window}_ratio"] = frame["close_raw"] / mean
    expected = stress_score(frame).iloc[-1]
    assert values.iloc[0] == pytest.approx(expected, rel=1e-12)


def test_a_market_fall_raises_the_stress_score(built: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    panel, _ = built
    before, _ = build_features(
        panel[panel["trading_day"] <= _day(panel, SHOCK_SESSION - 1)],
        point_in_time=_no_point_in_time(
            panel[panel["trading_day"] <= _day(panel, SHOCK_SESSION - 1)]
        ),  # type: ignore[arg-type]
    )
    after, _ = build_features(
        panel[panel["trading_day"] <= _day(panel, SHOCK_SESSION)],
        point_in_time=_no_point_in_time(panel[panel["trading_day"] <= _day(panel, SHOCK_SESSION)]),  # type: ignore[arg-type]
    )
    assert after[STRESS].iloc[0] > before[STRESS].iloc[0]


@pytest.mark.parametrize(("column", "component"), sorted(INTERACTIONS.items()))
def test_each_interaction_is_the_product_of_its_two_inputs_and_null_where_either_is(
    built: tuple[pd.DataFrame, pd.DataFrame], column: str, component: str
) -> None:
    _, features = built
    if component == "volatility_20d_ratio":
        liquid = features["liquidity_pass_raw"] == 1.0
        raw = features.loc[liquid, component]
        per_name = pd.Series(np.nan, index=features.index)
        per_name.loc[liquid] = (raw - raw.mean()) / raw.std(ddof=0)
    else:
        per_name = features[component]
    expected = per_name * features[STRESS]
    pd.testing.assert_series_equal(features[column], expected, check_names=False)
    assert features[column].notna().any()
    # Null where either input is null: the illiquid name and the short-history name.
    either_null = per_name.isna() | features[STRESS].isna()
    assert features.loc[either_null, column].isna().all()
    assert either_null.any()
    assert features.loc[features["ticker"] == "T11", column].isna().all()


def test_the_first_stress_score_lands_on_the_declared_depth() -> None:
    depth = catalog_column_depths()[STRESS]
    assert depth == _MARKET_STRESS_DEPTH_TRADING_DAYS
    panel = _panel(sessions=depth + 1)
    panel = panel[panel["ticker"] != "T10"]
    for session, present in ((depth - 2, False), (depth - 1, True)):
        cut = panel[panel["trading_day"] <= _day(panel, session)]
        features, _ = build_features(cut, point_in_time=_no_point_in_time(cut))  # type: ignore[arg-type]
        assert bool(features[STRESS].notna().all()) is present, f"session index {session}"


def test_appending_rows_after_a_day_leaves_that_days_features_unchanged() -> None:
    panel = _panel()
    day = _day(panel, SHOCK_SESSION + 5)
    cut = panel[panel["trading_day"] <= day]
    full, _ = build_features(panel, as_of=day, point_in_time=_no_point_in_time(panel, day))  # type: ignore[arg-type]
    truncated, _ = build_features(cut, point_in_time=_no_point_in_time(cut))  # type: ignore[arg-type]
    pd.testing.assert_frame_equal(full, truncated)
    assert truncated[STRESS].notna().all()


def test_no_existing_column_changes_when_the_regime_columns_are_added(
    built: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    panel, features = built
    old_catalog = tuple(spec for spec in CATALOG if spec.name not in NEW_COLUMNS)
    old, _ = build_features(panel, point_in_time=_no_point_in_time(panel), catalog=old_catalog)  # type: ignore[arg-type]
    shared = [spec.name for spec in old_catalog]
    pd.testing.assert_frame_equal(
        old[["trading_day", "ticker", *shared]], features[["trading_day", "ticker", *shared]]
    )
    assert old.attrs["feature_version"] != features.attrs["feature_version"]


#: Synthetic M recipes in the shape filed arms take. The real recipes are
#: private strategy content; the ids below were computed on `main` at ebb34b26,
#: BEFORE the regime columns existed, and must never move.
_FIXTURE_RECIPES = {
    "fixture_momentum": (
        "[momentum_20d_zscore, mom_12_1_zscore]",
        "{kind: ridge, alpha: 1.0}",
        "m:fixture_momentum:ba557fe38e10",
    ),
    "fixture_residual": (
        "[residual_momentum_252d_skip21d_zscore, momentum_change_21d_zscore, return_60d_zscore]",
        "{kind: ridge, alpha: 2.5}",
        "m:fixture_residual:876bf9f26ab4",
    ),
    "fixture_volatility": (
        "[volatility_20d_ratio, vol_ratio_10_60_ratio, momentum_20d_zscore]",
        "{kind: ridge, alpha: 0.5}",
        "m:fixture_volatility:c2716098c2a5",
    ),
}


@pytest.mark.parametrize("name", sorted(_FIXTURE_RECIPES))
def test_m_arm_ids_are_unchanged_by_the_new_columns(name: str) -> None:
    features, estimator, pinned = _FIXTURE_RECIPES[name]
    text = (
        f"slot: m\nname: {name}\nspec:\n  features: {features}\n  estimator: {estimator}\n"
        "  label_horizon_trading_days: 5\n  refit_cadence_trading_days: 5\n"
        "  training_window: {kind: expanding, min_trading_days: 60}\n"
        "  cpcv: {n_groups: 4, k_test: 1, embargo_trading_days: 1}\n"
        "registered_at: '2026-06-01'\n"
    )
    recipe = _parse_model_recipe(text.encode(), f"<fixture {name}>")
    assert recipe.arm_id == pinned
    assert not set(recipe.design_columns) & set(NEW_COLUMNS)
