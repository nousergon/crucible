"""crucible.regime: the point-in-time market-state score (aec#14081)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from crucible import regime
from crucible.regime import RegimeParams, RegimeParamsError

TICKERS = 12
SESSIONS = 400


def _frame(
    sessions: int = SESSIONS,
    *,
    fall_at: int | None = None,
    seed: int = 7,
) -> pd.DataFrame:
    """A long panel: one row per (day, ticker). A fall is one ln(0.8) day on every name."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2020-01-01", periods=sessions)
    rows: list[dict[str, object]] = []
    for position, day in enumerate(days):
        market = 0.0005 * math.sin(position / 5.0)
        if fall_at is not None:
            market = math.log(0.8) if position == fall_at else 0.0
        for ticker in range(TICKERS):
            rows.append(
                {
                    "trading_day": day,
                    "ticker": f"T{ticker:02d}",
                    "market_return_1d_log_return": market,
                    "momentum_20d_log_return": float(rng.normal(0.0, 0.05)),
                    "close_to_sma50_ratio": float(rng.uniform(0.9, 1.1)),
                    "close_to_sma200_ratio": float(rng.uniform(0.85, 1.15)),
                }
            )
    return pd.DataFrame(rows)


PARAMS = RegimeParams(normal_at=-0.5, stressed_at=0.5, crisis_at=1.5)


def test_a_constructed_20_percent_fall_reads_minus_20_percent_drawdown() -> None:
    inputs = regime.regime_inputs(_frame(fall_at=290))
    days = inputs.index
    assert inputs["market_drawdown"].loc[days[290]] == pytest.approx(-0.20, abs=1e-9)
    assert inputs["market_drawdown"].loc[days[289]] == pytest.approx(0.0, abs=1e-12)


def test_breadth_is_the_share_of_the_cross_section_above_its_average() -> None:
    frame = _frame(sessions=5)
    frame["close_to_sma50_ratio"] = np.where(frame["ticker"] < "T03", 1.2, 0.8)
    frame["close_to_sma200_ratio"] = 1.1
    inputs = regime.regime_inputs(frame)
    assert inputs["breadth_50"].tolist() == pytest.approx([3 / TICKERS] * 5)
    assert inputs["breadth_200"].tolist() == [1.0] * 5


def test_fewer_than_252_sessions_is_nan() -> None:
    score = regime.stress_score(_frame(sessions=251))
    assert len(score) == 251
    assert score.isna().all()


def test_enough_history_gives_a_finite_score() -> None:
    score = regime.stress_score(_frame())
    assert score.iloc[-1] == score.iloc[-1]
    assert math.isfinite(float(score.iloc[-1]))
    assert score.iloc[:251].isna().all()


def test_appending_rows_after_day_d_does_not_change_day_d() -> None:
    full = _frame(sessions=SESSIONS)
    cut = 380
    days = sorted(full["trading_day"].unique())
    truncated = full[full["trading_day"] <= days[cut]]
    a = regime.stress_score(truncated)
    b = regime.stress_score(full)
    assert math.isfinite(float(a.iloc[-1]))
    pd.testing.assert_series_equal(a, b.loc[a.index])
    pd.testing.assert_frame_equal(
        regime.regime_inputs(truncated), regime.regime_inputs(full).loc[a.index]
    )


def test_shuffling_rows_does_not_change_any_day() -> None:
    frame = _frame()
    shuffled = frame.sample(frac=1.0, random_state=3).reset_index(drop=True)
    pd.testing.assert_series_equal(regime.stress_score(frame), regime.stress_score(shuffled))


def test_a_later_shock_does_not_move_an_earlier_score() -> None:
    calm = _frame(fall_at=None)
    shocked = calm.copy()
    last = shocked["trading_day"].max()
    shocked.loc[shocked["trading_day"] == last, "market_return_1d_log_return"] = -0.3
    before = regime.stress_score(calm).iloc[:-1]
    after = regime.stress_score(shocked).iloc[:-1]
    pd.testing.assert_series_equal(before, after)


def test_missing_columns_refuse_by_name() -> None:
    with pytest.raises(ValueError, match="close_to_sma200_ratio"):
        regime.stress_score(_frame(sessions=3).drop(columns=["close_to_sma200_ratio"]))


@pytest.mark.parametrize(
    ("score", "state"),
    [
        (-0.51, "calm"),
        (-0.5, "normal"),
        (0.49, "normal"),
        (0.5, "stressed"),
        (1.49, "stressed"),
        (1.5, "crisis"),
        (9.0, "crisis"),
    ],
)
def test_band_edges_are_inclusive_on_the_stressed_side(score: float, state: str) -> None:
    assert regime.regime_state(score, PARAMS) == state


def test_an_unmeasured_score_has_no_state() -> None:
    with pytest.raises(ValueError, match="no state"):
        regime.regime_state(float("nan"), PARAMS)


GOOD = {"normal_at": -0.5, "stressed_at": 0.5, "crisis_at": 1.5}


def test_params_build_from_a_complete_mapping() -> None:
    assert RegimeParams.from_mapping(GOOD, source="t.yaml") == PARAMS


@pytest.mark.parametrize("key", sorted(GOOD))
def test_a_missing_key_raises_naming_it(key: str) -> None:
    payload = {name: value for name, value in GOOD.items() if name != key}
    with pytest.raises(RegimeParamsError, match=key):
        RegimeParams.from_mapping(payload, source="t.yaml")


def test_an_unknown_key_raises_naming_it() -> None:
    with pytest.raises(RegimeParamsError, match="panic_at"):
        RegimeParams.from_mapping({**GOOD, "panic_at": 3.0}, source="t.yaml")


@pytest.mark.parametrize(
    "payload",
    [
        {"normal_at": 0.5, "stressed_at": 0.5, "crisis_at": 1.5},
        {"normal_at": 0.0, "stressed_at": 2.0, "crisis_at": 1.0},
    ],
)
def test_misordered_bands_raise_naming_the_keys(payload: dict[str, float]) -> None:
    with pytest.raises(RegimeParamsError, match="normal_at < stressed_at < crisis_at"):
        RegimeParams.from_mapping(payload, source="t.yaml")


@pytest.mark.parametrize("bad", [True, "1.0", None, float("nan")])
def test_a_non_numeric_value_raises_naming_the_key(bad: object) -> None:
    with pytest.raises(RegimeParamsError, match="stressed_at"):
        RegimeParams.from_mapping({**GOOD, "stressed_at": bad}, source="t.yaml")


def test_regime_days_document_carries_the_schema_and_digest() -> None:
    days = regime.regime_days(_frame(), PARAMS)
    assert days[0].state is None and days[0].stress_score is None
    last = days[-1].to_document()
    assert last["schema"] == "crucible_regime.v1"
    assert last["params_digest"] == PARAMS.digest()
    assert last["state"] in regime.STATES
    assert set(last) == {"schema", "day", "inputs", "stress_score", "state", "params_digest"}
    assert isinstance(last["day"], str) and len(last["day"]) == 10
    assert set(days[0].inputs) == set(regime.INPUT_SIGNS)


def test_the_digest_follows_the_bands() -> None:
    other = RegimeParams(normal_at=-0.5, stressed_at=0.5, crisis_at=2.0)
    assert other.digest() != PARAMS.digest()
