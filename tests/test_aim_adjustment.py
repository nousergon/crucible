"""The aim book and partial adjustment (`alpha-engine-config-I12024`).

Brian's ruling of 2026-11-10: v2 should run "a longer-term portfolio managed
daily with small adjustments, not rebuilt daily". `aim_adjustment_rate` (k)
makes the session's solve, without the daily turnover constraint, the AIM, and
the session trades k of the way from the held book toward it. Null (or absent,
during rollout) keeps today's behaviour exactly.

The replay classes are built from the two order books the v1→v2 optimizer
parity audit read (decision days 2026-11-06 and 2026-11-09): the fourteen
highest-alpha names the engine was handed each day, whether each was in the U
cut, and the held weight carried in. The covariance is a one-factor stand-in
(the panel is not in this repository), so the solved weights are not the
traded ones; what is asserted is the SHAPE the ruling asks for — the held book
moves part of the way, a higher-tier exit is never slowed, and the day's
budget still caps the step.
"""

from __future__ import annotations

import numpy as np
import pytest

from crucible.portfolio import (
    CostModel,
    PortfolioParams,
    PortfolioParamsError,
    solve_target_weights,
)

#: (ticker, engine alpha, in the U cut, held weight carried in)
_REPLAY: dict[str, tuple[list[tuple[str, float, bool, float]], float]] = {
    # Decision day 2026-11-06 (traded Mon 11-09): 48% idle cash and SNDK held
    # at 47.9% outside the cut, above the 25% per-name cap.
    "2026-11-06": (
        [
            ("SNDK", 0.09721, False, 0.47915),
            ("MU", 0.03583, True, 0.0),
            ("PLTR", 0.02409, True, 0.04149),
            ("CVLT", 0.0154, True, 0.0),
            ("ANF", 0.01459, True, 0.0),
            ("NTNX", 0.01313, True, 0.0),
            ("WING", 0.01053, True, 0.0),
            ("PATH", 0.01044, True, 0.0),
            ("NVDA", 0.00783, True, 0.0),
            ("FSLR", 0.00509, True, 0.0),
            ("LULU", 0.00499, True, 0.0),
            ("SN", 0.00173, True, 0.0),
            ("MANH", 0.00034, True, 0.0),
            ("EXE", -0.00011, True, 0.0),
        ],
        0.47937,
    ),
    # Decision day 2026-11-09 (traded Tue 11-10): the book the 11-06 plan
    # built, with CVLT held one day and slipping from rank 3 to rank 5.
    "2026-11-09": (
        [
            ("SNDK", 0.09435, False, 0.24645),
            ("MU", 0.03203, True, 0.24854),
            ("PLTR", 0.02266, False, 0.24872),
            ("FSLR", 0.01672, True, 0.0),
            ("ANF", 0.01495, True, 0.114),
            ("NTNX", 0.01487, True, 0.0),
            ("CVLT", 0.01183, True, 0.1092),
            ("WING", 0.01165, True, 0.0),
            ("NVDA", 0.01155, True, 0.0),
            ("PATH", 0.01075, True, 0.0),
            ("CELH", 0.00814, True, 0.0),
            ("LULU", 0.0078, True, 0.0),
            ("SN", 0.00543, True, 0.0),
            ("MANH", 0.00319, True, 0.0),
        ],
        0.03309,
    ),
}

#: The S slot's parameter shape as of 2026-11-10, with the participation cap
#: off (no ADV in a replay) and k supplied per test.
_PARAMS: dict[str, object] = {
    "risk_aversion": 5.0,
    "cash_sleeve_pct": 0.03,
    "max_sector_pct": 0.25,
    "min_position_pct": 0.005,
    "covariance_shrinkage": "ledoit_wolf",
    "sigma_horizon_days": 1,
    "ewma_lambda_decay": 0.94,
    "vol_target_annual": None,
    "alpha_uncertainty_penalty": 0.0,
    "alpha_uncertainty_min_cv": 0.01,
    "max_pct_adv": None,
    "max_daily_turnover": 0.20,
    "large_move_turnover_flag": 0.35,
    "conviction_budget_gate_enabled": True,
    "conviction_ir_floor": 0.35,
    "conviction_ir_full": 0.75,
    "conviction_budget_min_multiple": 0.05,
    "conviction_gate_min_names": 3,
    "max_position_pct": 0.25,
    "ineligible_held_exit": "discretionary",
    "book_notional_usd": 1_000_000.0,
}

#: About 3 bp per unit traded: the audit's reading of the sqrt-impact model on
#: a $1M book of liquid names.
_COST = CostModel(
    name="flat_bps_v0",
    placeholder=True,
    params={"half_spread_bps": 0.0, "commission_bps": 0.0, "slippage_bps": 3.0},
)


def _params(**overrides: object) -> PortfolioParams:
    return PortfolioParams.from_mapping({**_PARAMS, **overrides}, source="<aim test>")


def _covariance(n: int, benchmark_idx: int, cash_idx: int) -> np.ndarray:
    """One market factor (16%/yr) plus 35%/yr idiosyncratic, daily; cash riskless."""
    market = 0.16**2 / 252
    idio = 0.35**2 / 252
    cov = np.full((n, n), market)
    np.fill_diagonal(cov, market + idio)
    cov[benchmark_idx, benchmark_idx] = market
    cov[cash_idx, :] = 0.0
    cov[:, cash_idx] = 0.0
    cov[cash_idx, cash_idx] = 1e-12
    return cov


def _solve(
    day: str,
    *,
    w_prev: np.ndarray | None = None,
    alpha: np.ndarray | None = None,
    eligibility: np.ndarray | None = None,
    **overrides: object,
):
    rows, cash_prev = _REPLAY[day]
    tickers = [r[0] for r in rows] + ["SPY", "__CASH__"]
    n = len(tickers)
    bench, cash = n - 2, n - 1
    if alpha is None:
        alpha = np.array([r[1] for r in rows] + [0.0, -1e-6])
    if eligibility is None:
        eligibility = np.array([r[2] for r in rows] + [True, True])
    if w_prev is None:
        w_prev = np.array([r[3] for r in rows] + [0.0, cash_prev])
    result = solve_target_weights(
        tickers,
        alpha,
        None,
        w_prev,
        ["__unsectored__"] * (n - 2) + ["__benchmark__", "__cash__"],
        np.ones(n),
        eligibility,
        bench,
        cash,
        _params(**overrides),
        _COST,
        covariance=_covariance(n, bench, cash),
        portfolio_notional=1_000_000.0,
    )
    return tickers, w_prev, result


def _turnover(weights: np.ndarray, w_prev: np.ndarray) -> float:
    return float(np.sum(np.abs(weights - w_prev)) / 2)


class TestTheParameter:
    def test_absent_reads_as_off(self) -> None:
        params = PortfolioParams.from_mapping(_PARAMS, source="<pre-aim file>")
        assert params.aim_adjustment_rate is None
        assert params.to_dict()["aim_adjustment_rate"] is None

    @pytest.mark.parametrize("value", [0.0, -0.1, 1.5])
    def test_out_of_range_is_refused(self, value: float) -> None:
        with pytest.raises(PortfolioParamsError, match="aim_adjustment_rate"):
            _params(aim_adjustment_rate=value)

    @pytest.mark.parametrize("value", [0.25, 0.5, 1.0])
    def test_a_rate_in_range_loads(self, value: float) -> None:
        assert _params(aim_adjustment_rate=value).aim_adjustment_rate == value


class TestOffIsTodaysBehaviour:
    @pytest.mark.parametrize("day", sorted(_REPLAY))
    def test_null_and_absent_solve_identically(self, day: str) -> None:
        _t, _w, explicit = _solve(day, aim_adjustment_rate=None)
        _t, _w, absent = _solve(day)
        np.testing.assert_allclose(explicit.weights, absent.weights, atol=1e-12)
        assert explicit.diagnostics["aim_weights"] is None
        assert explicit.diagnostics["aim_adjustment"]["applied"] is False
        assert explicit.diagnostics["aim_adjustment"]["reason"] == "aim_adjustment_rate_null"


class TestReplayOfTheCvltRoundTrip:
    """11-09: CVLT, bought the session before, slips rank 3 → 5 and the
    single-period solve sells it whole to fund a swap."""

    def test_off_the_held_name_is_sold_whole(self) -> None:
        tickers, w_prev, result = _solve("2026-11-09")
        cvlt = tickers.index("CVLT")
        assert w_prev[cvlt] > 0.10
        assert result.weights[cvlt] == pytest.approx(0.0, abs=1e-6)

    def test_at_k_one_half_it_steps_halfway_and_turnover_falls(self) -> None:
        tickers, w_prev, off = _solve("2026-11-09")
        _t, _w, on = _solve("2026-11-09", aim_adjustment_rate=0.5)
        cvlt = tickers.index("CVLT")
        aim = on.diagnostics["aim_weights"]

        assert aim["CVLT"] == pytest.approx(0.0, abs=1e-6)
        assert on.weights[cvlt] == pytest.approx(w_prev[cvlt] / 2, abs=2e-3)
        off_turnover = _turnover(off.weights, w_prev)
        on_turnover = _turnover(on.weights, w_prev)
        assert on_turnover < 0.65 * off_turnover
        assert on.diagnostics["aim_adjustment"]["rate_applied"] == 0.5

    def test_every_discretionary_name_lands_between_held_and_aim(self) -> None:
        tickers, w_prev, on = _solve("2026-11-09", aim_adjustment_rate=0.5)
        aim = on.diagnostics["aim_weights"]
        for i, ticker in enumerate(tickers[:-1]):
            target = aim.get(ticker, 0.0)
            lo, hi = sorted((float(w_prev[i]), float(target)))
            assert lo - 1e-6 <= on.weights[i] <= hi + 1e-6, ticker

    def test_the_book_is_whole_and_the_sleeve_pinned(self) -> None:
        tickers, _w, on = _solve("2026-11-09", aim_adjustment_rate=0.5)
        assert on.weights.sum() == pytest.approx(1.0, abs=1e-9)
        assert on.weights[tickers.index("__CASH__")] == pytest.approx(0.03, abs=1e-6)
        assert np.all(on.weights >= -1e-12)
        assert np.all(on.weights[:-2] <= 0.25 + 1e-6)

    def test_a_second_session_closes_three_quarters_of_the_gap(self) -> None:
        tickers, w_prev, first = _solve("2026-11-09", aim_adjustment_rate=0.5)
        _t, _w, second = _solve("2026-11-09", w_prev=first.weights, aim_adjustment_rate=0.5)
        cvlt = tickers.index("CVLT")
        assert second.weights[cvlt] == pytest.approx(w_prev[cvlt] / 4, abs=3e-3)


class TestStepsNeverDelayAHigherTierExit:
    """Precedence: hard risk > eligibility > optimizer > timing."""

    def test_a_cap_trim_is_taken_in_full(self) -> None:
        """11-06: SNDK held at 47.9% against a 25% cap trims to the cap the same day."""
        tickers, w_prev, on = _solve("2026-11-06", aim_adjustment_rate=0.5)
        sndk = tickers.index("SNDK")
        assert w_prev[sndk] > 0.45
        assert on.weights[sndk] <= 0.25 + 1e-6
        assert "SNDK" in on.diagnostics["aim_adjustment"]["forced_moves"]

    def test_a_forced_eligibility_exit_is_taken_in_full(self) -> None:
        """Under `forced`, a held name the cut drops is pinned to zero — never halved."""
        tickers, w_prev, on = _solve(
            "2026-11-09", aim_adjustment_rate=0.5, ineligible_held_exit="forced"
        )
        for name in ("SNDK", "PLTR"):
            i = tickers.index(name)
            assert w_prev[i] > 0.2
            assert on.weights[i] == pytest.approx(0.0, abs=1e-9), name
            assert name in on.diagnostics["aim_adjustment"]["forced_moves"]

    def test_a_cap_zero_pin_is_taken_in_full_whatever_the_rate(self) -> None:
        """A hard-risk exit arrives as an effective cap of zero (an ineligible,
        forced name); a small k must not slow it."""
        rows, _cash = _REPLAY["2026-11-09"]
        eligibility = np.array([r[2] for r in rows] + [True, True])
        cvlt = [r[0] for r in rows].index("CVLT")
        eligibility[cvlt] = False
        tickers, w_prev, on = _solve(
            "2026-11-09",
            eligibility=eligibility,
            aim_adjustment_rate=0.1,
            ineligible_held_exit="forced",
        )
        assert w_prev[cvlt] > 0.10
        assert on.weights[cvlt] == pytest.approx(0.0, abs=1e-9)

    def test_the_cash_sleeve_pin_is_not_stepped(self) -> None:
        """11-06: 48% idle cash. The sleeve is an equality pin, so it is deployed
        in full — the engine's existing mandatory-turnover treatment."""
        tickers, _w, on = _solve("2026-11-06", aim_adjustment_rate=0.5)
        assert on.weights[tickers.index("__CASH__")] == pytest.approx(0.03, abs=1e-6)


class TestFinishTheExitAndDeferTheEntry:
    def _book(self) -> tuple[np.ndarray, np.ndarray]:
        rows, cash = _REPLAY["2026-11-09"]
        w_prev = np.array([r[3] for r in rows] + [0.0, cash])
        alpha = np.array([r[1] for r in rows] + [0.0, -1e-6])
        return w_prev, alpha

    def test_a_small_remainder_is_sold_rather_than_left_as_a_stub(self) -> None:
        w_prev, alpha = self._book()
        names = [r[0] for r in _REPLAY["2026-11-09"][0]]
        manh = names.index("MANH")
        # A 0.8% MANH holding, carried as an under-ranked stub: its aim is 0,
        # and half of 0.8% is below the 0.5% floor.
        w_prev[manh] = 0.008
        w_prev[names.index("ANF")] -= 0.008
        tickers, _w, on = _solve("2026-11-09", w_prev=w_prev, alpha=alpha, aim_adjustment_rate=0.5)
        assert on.diagnostics["aim_weights"]["MANH"] == pytest.approx(0.0, abs=1e-6)
        assert on.weights[tickers.index("MANH")] == 0.0
        assert "MANH" in on.diagnostics["aim_adjustment"]["exits_finished"]

    def test_an_entry_below_the_floor_waits(self) -> None:
        """At a slow pace a new name's first step is under the dust floor: it is
        not bought as a stub, and it is named as deferred."""
        tickers, w_prev, on = _solve("2026-11-09", aim_adjustment_rate=0.02)
        deferred = on.diagnostics["aim_adjustment"]["entries_deferred"]
        assert deferred
        for name in deferred:
            i = tickers.index(name)
            assert w_prev[i] == 0.0
            assert on.weights[i] == 0.0


class TestTheBudgetStillCapsTheDay:
    def test_a_step_larger_than_the_budget_is_slowed_to_fit(self) -> None:
        tickers, w_prev, on = _solve("2026-11-09", aim_adjustment_rate=1.0, max_daily_turnover=0.02)
        meta = on.diagnostics["aim_adjustment"]
        assert meta["reason"] == "budget_limited_step"
        assert 0.0 <= meta["rate_applied"] < 1.0
        assert meta["stepped_turnover_one_way"] <= on.diagnostics["turnover_constraint_cap"] + 0.01
        assert on.weights.sum() == pytest.approx(1.0, abs=1e-9)

    def test_k_one_with_a_slack_budget_trades_to_the_aim(self) -> None:
        _t, _w, on = _solve("2026-11-09", aim_adjustment_rate=1.0, max_daily_turnover=None)
        meta = on.diagnostics["aim_adjustment"]
        assert meta["rate_applied"] == 1.0
        assert meta["stepped_turnover_one_way"] == pytest.approx(
            meta["aim_turnover_one_way"], abs=1e-6
        )

    def test_the_infeasible_fallback_records_no_step(self) -> None:
        """An infeasible program holds the book; the aim block says why no step was taken."""
        _t, _w, on = _solve("2026-11-09", aim_adjustment_rate=0.5, vol_target_annual=0.0001)
        assert on.diagnostics["status"] == "infeasible_fallback"
        assert on.diagnostics["aim_weights"] is None
        assert on.diagnostics["aim_adjustment"]["reason"] == "infeasible_fallback"
