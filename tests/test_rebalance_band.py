"""The no-trade rebalance band (v1 parity audit 2026-11-09, row 17).

v1 (`crucible-executor` `optimizer_shadow._compute_trade_deltas`) skipped any
trade smaller than ``rebalance_band_pct`` of the book. v2 had no band: every
delta of a share or more traded, so daily trims and adds of a few basis points
counted as churn. These tests pin the v2 port: the band edits the TARGET (so
grading and trading walk one book), never bands an exit or a cap-forced trim,
never shrinks cash below what the solve chose, and records what it held on
every solve.

**The parameter values below are FIXTURE values**; the operating band lives in
the private strategy tree.
"""

from __future__ import annotations

import numpy as np
import pytest

from crucible.portfolio import (
    PORTFOLIO_PARAM_FIELDS,
    PortfolioParams,
    PortfolioParamsError,
    _apply_rebalance_band,  # pyright: ignore[reportPrivateUsage]
    solve_target_weights,
)
from tests.test_portfolio import (
    _FIXTURE_PARAMS,  # pyright: ignore[reportPrivateUsage]
    _baseline_universe,  # pyright: ignore[reportPrivateUsage]
    _params,  # pyright: ignore[reportPrivateUsage]
)

TICKERS = ["A", "B", "C", "D", "SPY", "CASH"]
CASH = 5


def _band(weights, w_prev, *, band=0.005, caps=None):
    params = _params(rebalance_band_pct=band)
    w = np.asarray(weights, dtype=float)
    p = np.asarray(w_prev, dtype=float)
    c = np.ones(len(w)) if caps is None else np.asarray(caps, dtype=float)
    return _apply_rebalance_band(w, p, c, TICKERS, CASH, params)


class TestTheBandHoldsDrift:
    def test_a_move_inside_the_band_keeps_the_held_weight_and_cash_absorbs_it(self) -> None:
        prev = [0.20, 0.10, 0.0, 0.0, 0.67, 0.03]
        solved = [0.203, 0.10, 0.0, 0.0, 0.667, 0.03]
        out, meta = _band(solved, prev)
        assert out[0] == pytest.approx(0.20)
        # SPY's -0.003 is drift too; both are held, so cash ends where it began.
        assert out[4] == pytest.approx(0.67)
        assert out.sum() == pytest.approx(1.0)
        assert {row["ticker"] for row in meta["held"]} == {"A", "SPY"}

    def test_a_move_at_or_beyond_the_band_trades(self) -> None:
        prev = [0.20, 0.10, 0.0, 0.0, 0.67, 0.03]
        solved = [0.21, 0.10, 0.0, 0.0, 0.66, 0.03]
        out, meta = _band(solved, prev)
        np.testing.assert_allclose(out, solved)
        assert meta["held"] == []

    def test_a_banded_buy_leaves_its_cash_in_cash(self) -> None:
        prev = [0.20, 0.10, 0.0, 0.0, 0.67, 0.03]
        solved = [0.203, 0.10, 0.0, 0.0, 0.664, 0.033]
        out, _ = _band(solved, prev, band=0.004)
        assert out[0] == pytest.approx(0.20)
        assert out[CASH] == pytest.approx(0.036)

    def test_solver_dust_is_not_recorded_as_a_held_trade(self) -> None:
        prev = [0.20, 0.10, 0.0, 0.0, 0.67, 0.03]
        solved = [0.2000001, 0.10, 0.0, 0.0, 0.6699999, 0.03]
        _, meta = _band(solved, prev)
        assert meta["held"] == []


class TestAnExitIsNeverDrift:
    def test_a_move_to_zero_trades_however_small(self) -> None:
        prev = [0.003, 0.10, 0.0, 0.0, 0.867, 0.03]
        solved = [0.0, 0.10, 0.0, 0.0, 0.87, 0.03]
        out, meta = _band(solved, prev)
        assert out[0] == 0.0
        assert "A" not in {row["ticker"] for row in meta["held"]}

    def test_a_held_weight_above_its_cap_is_trimmed_however_small(self) -> None:
        prev = [0.252, 0.10, 0.0, 0.0, 0.618, 0.03]
        solved = [0.25, 0.10, 0.0, 0.0, 0.62, 0.03]
        caps = [0.25, 0.25, 0.25, 0.25, 1.0, 1.0]
        out, _ = _band(solved, prev, caps=caps)
        assert out[0] == pytest.approx(0.25)


class TestTheBandCannotPutTheBookOnMargin:
    def test_banded_sells_that_funded_a_buy_are_restored_largest_first(self) -> None:
        # A buys 0.012 (not banded); B, C, D each sell 0.004 (banded) to fund it.
        prev = [0.10, 0.10, 0.10, 0.10, 0.57, 0.03]
        solved = [0.112, 0.096, 0.096, 0.096, 0.57, 0.03]
        out, meta = _band(solved, prev)
        assert out[CASH] >= 0.03 - 1e-9
        assert out.sum() == pytest.approx(1.0)
        assert len(meta["restored_sells"]) == 3
        # The gross the book holds is never above what the solve chose.
        assert 1.0 - out[CASH] <= 1.0 - 0.03 + 1e-9

    def test_cash_is_never_below_the_solved_cash(self) -> None:
        rng = np.random.default_rng(7)
        for _ in range(200):
            prev = rng.dirichlet(np.ones(5)) * 0.97
            prev = np.append(prev, 0.03)
            step = rng.normal(0.0, 0.004, size=5)
            solved = np.clip(prev[:5] + step, 0.0, None)
            solved = np.append(solved / solved.sum() * 0.97, 0.03)
            out, _ = _band(solved, prev)
            assert out[CASH] >= solved[CASH] - 1e-6
            assert out.sum() == pytest.approx(1.0)
            assert np.sum(np.abs(out - prev)) <= np.sum(np.abs(solved - prev)) + 1e-9


class TestTheBandIsRecordedOnEverySolve:
    def test_off_is_recorded_as_off(self) -> None:
        prev = [0.20, 0.10, 0.0, 0.0, 0.67, 0.03]
        solved = [0.203, 0.10, 0.0, 0.0, 0.667, 0.03]
        out, meta = _apply_rebalance_band(
            np.asarray(solved), np.asarray(prev), np.ones(6), TICKERS, CASH, _params()
        )
        np.testing.assert_allclose(out, solved)
        assert meta == {"band": None, "held": [], "restored_sells": []}

    def test_a_new_position_the_band_held_out_is_named_as_one(self) -> None:
        prev = [0.0, 0.10, 0.0, 0.0, 0.87, 0.03]
        solved = [0.004, 0.10, 0.0, 0.0, 0.866, 0.03]
        _, meta = _band(solved, prev)
        (row,) = [r for r in meta["held"] if r["ticker"] == "A"]
        assert row["is_new_position"] is True

    def test_the_solve_carries_the_block_and_trades_less(self) -> None:
        u = _baseline_universe(n_active=4, sector_labels=["a", "b", "c", "d"])
        u["stance_caps"] = np.full(len(u["tickers"]), 1.0)
        u["alpha_hat"] = np.array([0.0020, 0.0019, 0.0021, 0.0018, 0.0, 0.0])
        u["params"] = _params(max_daily_turnover=None, max_sector_pct=1.0)
        first = solve_target_weights(**u)
        # The next session: the same book held, a fractionally different forecast.
        u["w_prev"] = np.asarray(first.weights)
        u["alpha_hat"] = np.array([0.00201, 0.00189, 0.00211, 0.00179, 0.0, 0.0])
        unbanded = solve_target_weights(**u)
        u["params"] = _params(max_daily_turnover=None, max_sector_pct=1.0, rebalance_band_pct=0.01)
        banded = solve_target_weights(**u)
        assert banded.diagnostics["rebalance_band"]["band"] == 0.01
        assert unbanded.diagnostics["rebalance_band"]["band"] is None
        assert (
            banded.diagnostics["turnover_one_way"]
            <= unbanded.diagnostics["turnover_one_way"] + 1e-9
        )
        assert float(np.sum(banded.weights)) == pytest.approx(1.0)


class TestTheParameter:
    def test_it_is_registered_and_nullable(self) -> None:
        assert PORTFOLIO_PARAM_FIELDS["rebalance_band_pct"] == "number-or-null"

    @pytest.mark.parametrize("value", [0.0, -0.01, 1.0])
    def test_out_of_range_is_refused(self, value: float) -> None:
        with pytest.raises(PortfolioParamsError):
            _params(rebalance_band_pct=value)

    def test_absent_reads_as_off_during_the_rollout(self) -> None:
        params = PortfolioParams.from_mapping(
            {k: v for k, v in _FIXTURE_PARAMS.items() if k != "rebalance_band_pct"},
            source="<pre-band file>",
        )
        assert params.rebalance_band_pct is None
