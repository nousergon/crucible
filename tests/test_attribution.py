"""`crucible.attribution` — factor-neutral return decomposition, deterministic.

No network, no S3, no ArcticDB. Every series below is a fixed literal so the
test is not sensitive to a clock or a fixture generator (test-discipline rule:
fixed date literals, never `today` arithmetic).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from crucible.attribution import (
    ATTRIBUTION_FACTOR_CATEGORIES,
    AttributionFactorParams,
    AttributionParamsError,
    FactorDef,
    compute_factor_attribution,
    load_attribution_params,
    params_digest,
)

TRADING_DAY = "2026-07-06"
WINDOW = 6

# Three deterministic, non-collinear factor return series.
MARKET = [0.010, -0.005, 0.008, 0.002, -0.011, 0.006]
SECTOR = [0.004, 0.006, -0.002, 0.009, -0.003, 0.001]
SIZE = [-0.002, 0.001, 0.003, -0.004, 0.005, 0.002]

_PARAMS = AttributionFactorParams(
    factors={
        "market": FactorDef(category="beta", proxy="SPY"),
        "sector_tech": FactorDef(category="sector", proxy="XLK"),
        "size_factor": FactorDef(category="size", proxy="IWM"),
    },
    benchmark_proxy="SPY",
)


def _combine(betas: dict[str, float]) -> list[float]:
    """An exact (no-intercept, no-noise) linear combination of the three factors."""
    return [
        betas["market"] * MARKET[t]
        + betas["sector_tech"] * SECTOR[t]
        + betas["size_factor"] * SIZE[t]
        for t in range(WINDOW)
    ]


HOLDING_BETAS = {
    "AAA": {"market": 1.0, "sector_tech": 0.5, "size_factor": 0.2},
    "BBB": {"market": 0.8, "sector_tech": 0.0, "size_factor": 0.3},
    "CCC": {"market": 1.2, "sector_tech": 0.7, "size_factor": 0.0},
}
HOLDING_RETURNS = {ticker: _combine(betas) for ticker, betas in HOLDING_BETAS.items()}
WEIGHTS = {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}


def _weighted_portfolio_beta(factor: str) -> float:
    return sum(WEIGHTS[t] * HOLDING_BETAS[t][factor] for t in WEIGHTS)


def _gross_return() -> float:
    """Weighted sum of the (exact, no-residual) holding returns over the window."""
    return sum(WEIGHTS[t] * sum(HOLDING_RETURNS[t]) for t in WEIGHTS)


def test_exact_linear_combination_has_near_zero_residual() -> None:
    gross = _gross_return()
    doc = compute_factor_attribution(
        trading_day=TRADING_DAY,
        window_sessions=WINDOW,
        holding_returns=HOLDING_RETURNS,
        weights=WEIGHTS,
        factor_returns={"market": MARKET, "sector_tech": SECTOR, "size_factor": SIZE},
        params=_PARAMS,
        gross_return=gross,
        cost_bps_total=15.0,
    )
    assert doc["schema_version"] == "factor_attribution.v1"
    assert doc["engine"] == "crucible.attribution"
    assert doc["window_sessions"] == WINDOW
    # No noise, no intercept: the factor model explains ~all of the return.
    assert math.isclose(doc["residual_alpha"], 0.0, abs_tol=1e-9)
    assert math.isclose(doc["category_totals"]["residual"], doc["residual_alpha"])
    # The four category totals sum exactly to gross_return.
    total = sum(doc["category_totals"].values())
    assert math.isclose(total, gross, rel_tol=1e-9, abs_tol=1e-9)
    # net = gross - 15bps.
    assert math.isclose(doc["net_return"], gross - 0.0015, rel_tol=1e-12)
    assert doc["cost_bps_total"] == 15.0


def test_recovered_exposures_match_the_weighted_betas() -> None:
    gross = _gross_return()
    doc = compute_factor_attribution(
        trading_day=TRADING_DAY,
        window_sessions=WINDOW,
        holding_returns=HOLDING_RETURNS,
        weights=WEIGHTS,
        factor_returns={"market": MARKET, "sector_tech": SECTOR, "size_factor": SIZE},
        params=_PARAMS,
        gross_return=gross,
        cost_bps_total=0.0,
    )
    by_name = {row["name"]: row for row in doc["factors"]}
    for factor in ("market", "sector_tech", "size_factor"):
        assert math.isclose(
            by_name[factor]["exposure"], _weighted_portfolio_beta(factor), abs_tol=1e-9
        )
    # Category rollup: market is the sole "beta" factor here.
    assert math.isclose(
        doc["category_totals"]["beta"], by_name["market"]["contribution_return"], abs_tol=1e-9
    )


def test_refuses_a_factor_returns_mapping_missing_a_named_factor() -> None:
    with pytest.raises(ValueError, match="does not match the attribution spec"):
        compute_factor_attribution(
            trading_day=TRADING_DAY,
            window_sessions=WINDOW,
            holding_returns=HOLDING_RETURNS,
            weights=WEIGHTS,
            factor_returns={"market": MARKET, "sector_tech": SECTOR},  # size_factor missing
            params=_PARAMS,
            gross_return=0.01,
            cost_bps_total=0.0,
        )


def test_refuses_a_mismatched_series_length() -> None:
    with pytest.raises(ValueError, match="observations, expected"):
        compute_factor_attribution(
            trading_day=TRADING_DAY,
            window_sessions=WINDOW,
            holding_returns=HOLDING_RETURNS,
            weights=WEIGHTS,
            factor_returns={"market": MARKET[:-1], "sector_tech": SECTOR, "size_factor": SIZE},
            params=_PARAMS,
            gross_return=0.01,
            cost_bps_total=0.0,
        )


def test_refuses_a_zero_session_window() -> None:
    with pytest.raises(ValueError, match="spans no session"):
        compute_factor_attribution(
            trading_day=TRADING_DAY,
            window_sessions=0,
            holding_returns={},
            weights={},
            factor_returns={},
            params=_PARAMS,
            gross_return=0.0,
            cost_bps_total=0.0,
        )


def test_params_reject_a_factor_spec_missing_a_category() -> None:
    with pytest.raises(AttributionParamsError, match="missing categor"):
        AttributionFactorParams(
            factors={
                "market": FactorDef(category="beta", proxy="SPY"),
                "sector_tech": FactorDef(category="sector", proxy="XLK"),
                # no "size" category present
            },
            benchmark_proxy="SPY",
        )


def test_factor_def_rejects_an_unknown_category() -> None:
    with pytest.raises(AttributionParamsError, match="not one of"):
        FactorDef(category="momentum", proxy="MTUM")


def test_load_attribution_params_raises_when_file_absent(tmp_path: Path) -> None:
    with pytest.raises(AttributionParamsError, match="no attribution factor spec"):
        load_attribution_params(tmp_path / "does-not-exist.yaml")


def test_load_attribution_params_round_trips_a_written_file(tmp_path: Path) -> None:
    spec = tmp_path / "attribution.yaml"
    spec.write_text(
        """
attribution:
  benchmark_proxy: SPY
  shrinkage: ledoit_wolf
  factors:
    market:
      category: beta
      proxy: SPY
    sector_tech:
      category: sector
      proxy: XLK
    size_factor:
      category: size
      proxy: IWM
""",
        encoding="utf-8",
    )
    params = load_attribution_params(spec)
    assert params.benchmark_proxy == "SPY"
    assert set(params.factors) == {"market", "sector_tech", "size_factor"}
    assert {f.category for f in params.factors.values()} == set(ATTRIBUTION_FACTOR_CATEGORIES)


def test_params_digest_is_stable_and_order_independent() -> None:
    reordered = AttributionFactorParams(
        factors={
            "size_factor": FactorDef(category="size", proxy="IWM"),
            "market": FactorDef(category="beta", proxy="SPY"),
            "sector_tech": FactorDef(category="sector", proxy="XLK"),
        },
        benchmark_proxy="SPY",
    )
    assert params_digest(_PARAMS) == params_digest(reordered)
    assert params_digest(_PARAMS).startswith("sha256:")
