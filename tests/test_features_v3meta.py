"""The v3.0-meta L1 input columns (`alpha-engine-config-I10695`).

Each value is checked against an independent pandas computation of the
declared expression on a hand-built panel, and each column's first non-null
session against the depth `catalog_column_depths` declares for it — the
declared depth is what sizes the producer's panel, so a wrong one is a column
null on every ticker in production.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from crucible.features import CATALOG, build_features, catalog_column_depths

V3META_COLUMNS = (
    "momentum_5d_log_return",
    "atr_14_ratio",
    "vol_ratio_10_60_ratio",
    "dist_from_52w_high_ratio",
    "dist_from_52w_low_ratio",
)


@pytest.fixture(autouse=True)
def _liquidity_floor(monkeypatch):
    from crucible.features.compute import LIQUIDITY_FLOOR_VAR

    monkeypatch.setenv(LIQUIDITY_FLOOR_VAR, "1000000")


def _panel(n_sessions: int = 300, *, flat_from: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(20260914)
    start = dt.date(2025, 1, 1)
    rows = []
    for ticker in ("AAA", "BBB"):
        price = 50.0
        for i in range(n_sessions):
            if flat_from is None or i < flat_from:
                price *= float(np.exp(rng.normal(0.0005, 0.02)))
            rows.append(
                {
                    "trading_day": start + dt.timedelta(days=i),
                    "ticker": ticker,
                    "open_raw": price,
                    "high_raw": price * (1.0 + abs(rng.normal(0, 0.01))),
                    "low_raw": price * (1.0 - abs(rng.normal(0, 0.01))),
                    "close_raw": price,
                    "volume_raw": 1.0e6,
                }
            )
    return pd.DataFrame(rows)


def _expected(history: pd.DataFrame) -> dict[str, float]:
    close = history["close_raw"]
    log_return = np.log(close).diff()
    prior = close.shift(1)
    true_range = np.maximum(
        np.maximum(history["high_raw"] - history["low_raw"], (history["high_raw"] - prior).abs()),
        (history["low_raw"] - prior).abs(),
    )
    return {
        "momentum_5d_log_return": float(np.log(close.iloc[-1]) - np.log(close.iloc[-6])),
        "atr_14_ratio": float(true_range.iloc[-14:].mean() / close.iloc[-1]),
        "vol_ratio_10_60_ratio": float(
            log_return.iloc[-10:].std(ddof=1) / log_return.iloc[-60:].std(ddof=1)
        ),
        "dist_from_52w_high_ratio": float(close.iloc[-1] / close.iloc[-252:].max() - 1.0),
        "dist_from_52w_low_ratio": float(close.iloc[-1] / close.iloc[-252:].min() - 1.0),
    }


def test_every_v3meta_column_is_catalogued_with_a_units_suffix_and_a_depth() -> None:
    names = {spec.name for spec in CATALOG}
    depths = catalog_column_depths()
    for column in V3META_COLUMNS:
        assert column in names
        assert column in depths


def test_each_value_is_the_declared_expression() -> None:
    panel = _panel()
    features, _ = build_features(panel)
    for ticker in ("AAA", "BBB"):
        history = panel[panel["ticker"] == ticker].reset_index(drop=True)
        row = features[features["ticker"] == ticker].iloc[0]
        for column, value in _expected(history).items():
            assert row[column] == pytest.approx(value, rel=1e-12, abs=1e-15), column


@pytest.mark.parametrize("column", V3META_COLUMNS)
def test_the_first_value_lands_on_the_declared_depth(column: str) -> None:
    depth = catalog_column_depths()[column]
    panel = _panel(n_sessions=depth)
    days = sorted(panel["trading_day"].unique())
    short, _ = build_features(panel[panel["trading_day"] <= days[depth - 2]])
    full, _ = build_features(panel)
    assert short[column].isna().all(), f"{column} has a value one session before its depth"
    assert full[column].notna().all(), f"{column} has no value at its declared depth"


def test_a_zero_long_window_volatility_is_null_not_one() -> None:
    """v1 filled it with 1.0 — a substituted constant, refused here."""
    features, _ = build_features(_panel(flat_from=200))
    assert features["vol_ratio_10_60_ratio"].isna().all()
