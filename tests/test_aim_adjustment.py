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
    held_sessions: np.ndarray | None = None,
    hard_exit: np.ndarray | None = None,
    buy_multiplier: float = 1.0,
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
        held_sessions=held_sessions,
        hard_exit=hard_exit,
        buy_multiplier=buy_multiplier,
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

    def test_the_book_is_whole_and_the_sleeve_steps_toward_its_pin(self) -> None:
        tickers, w_prev, on = _solve("2026-11-09", aim_adjustment_rate=0.5)
        cash = tickers.index("__CASH__")
        assert on.weights.sum() == pytest.approx(1.0, abs=1e-9)
        assert on.weights[cash] == pytest.approx(
            w_prev[cash] + 0.5 * (0.03 - w_prev[cash]), abs=1e-6
        )
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


class TestIdleCashIsPacedToo:
    """Brian 2026-11-10, "Pace cash too": idle cash deploys at the same pace as
    every other move toward the aim, except at inception."""

    def test_the_11_06_cash_deploys_k_of_the_gap_not_all_of_it(self) -> None:
        """11-06 carried 47.9% cash against a 3% sleeve. Off, the whole ~45
        points deploys in one session; at k = 1/2 about half of it does."""
        tickers, w_prev, off = _solve("2026-11-06")
        _t, _w, on = _solve("2026-11-06", aim_adjustment_rate=0.5)
        cash = tickers.index("__CASH__")
        gap = float(w_prev[cash]) - 0.03
        assert gap > 0.44
        assert float(w_prev[cash] - off.weights[cash]) == pytest.approx(gap, abs=1e-6)
        deployed = float(w_prev[cash] - on.weights[cash])
        assert deployed == pytest.approx(0.5 * gap, abs=1e-6)
        meta = on.diagnostics["aim_adjustment"]
        assert meta["cash_paced"] is True
        assert meta["cash_aim"] == pytest.approx(0.03, abs=1e-6)
        assert on.diagnostics["aim_weights"]["__CASH__"] == pytest.approx(0.03, abs=1e-6)
        assert on.weights.sum() == pytest.approx(1.0, abs=1e-9)

    def test_a_forced_trim_is_still_taken_in_full_while_cash_is_paced(self) -> None:
        tickers, _w, on = _solve("2026-11-06", aim_adjustment_rate=0.25)
        assert on.weights[tickers.index("SNDK")] <= 0.25 + 1e-6
        assert "SNDK" in on.diagnostics["aim_adjustment"]["forced_moves"]

    def test_a_first_build_is_never_paced(self) -> None:
        """Inception: a book holding nothing builds the whole aim in one session."""
        rows, _cash = _REPLAY["2026-11-06"]
        w_prev = np.zeros(len(rows) + 2)
        w_prev[-1] = 1.0
        tickers, _w, on = _solve("2026-11-06", w_prev=w_prev, aim_adjustment_rate=0.25)
        aim = on.diagnostics["aim_weights"]
        meta = on.diagnostics["aim_adjustment"]
        assert meta["cash_paced"] is False
        assert on.weights[tickers.index("__CASH__")] == pytest.approx(0.03, abs=1e-6)
        for i, ticker in enumerate(tickers):
            assert on.weights[i] == pytest.approx(aim.get(ticker, 0.0), abs=1e-6), ticker


class TestHardRiskIsNotPaced:
    def test_a_negative_alpha_holding_sells_to_its_aim_in_full(self) -> None:
        """Precedence: hard risk > timing. A held name whose alpha turned
        negative is not wound down at the pace."""
        rows, _cash = _REPLAY["2026-11-09"]
        alpha = np.array([r[1] for r in rows] + [0.0, -1e-6])
        cvlt = [r[0] for r in rows].index("CVLT")
        alpha[cvlt] = -0.02
        tickers, w_prev, on = _solve("2026-11-09", alpha=alpha, aim_adjustment_rate=0.25)
        assert w_prev[cvlt] > 0.10
        assert on.diagnostics["aim_weights"]["CVLT"] == pytest.approx(0.0, abs=1e-6)
        assert on.weights[cvlt] == pytest.approx(0.0, abs=1e-6)
        assert "CVLT" in on.diagnostics["aim_adjustment"]["hard_risk_exits"]


def _held_sessions(w_prev: np.ndarray, age: int) -> np.ndarray:
    return np.where(w_prev > 0.0, float(age), 0.0)


class TestComposesWithTheChurnControls:
    """crucible#499's minimum hold and exit hysteresis bind under the pace: the
    aim is solved under their floors and no step goes below them."""

    def test_a_name_inside_its_minimum_hold_is_never_stepped_down(self) -> None:
        """11-09: CVLT, held one session, would be halved by the pace alone."""
        rows, cash = _REPLAY["2026-11-09"]
        w_prev = np.array([r[3] for r in rows] + [0.0, cash])
        tickers, _w, paced = _solve("2026-11-09", aim_adjustment_rate=0.5)
        cvlt = tickers.index("CVLT")
        assert paced.weights[cvlt] < w_prev[cvlt] - 0.03

        _t, _w, on = _solve(
            "2026-11-09",
            aim_adjustment_rate=0.5,
            min_hold_days=3,
            held_sessions=_held_sessions(w_prev, 1),
        )
        churn = on.diagnostics["churn_controls"]
        floors = {row["ticker"]: row["floor"] for row in churn["protected"]}
        assert "CVLT" in floors
        assert "CVLT" in on.diagnostics["aim_adjustment"]["floor_protected"]
        for ticker, floor in floors.items():
            i = tickers.index(ticker)
            assert on.weights[i] >= floor - 1e-6, ticker
            assert on.diagnostics["aim_weights"][ticker] >= floor - 1e-6, ticker
        assert on.weights[cvlt] >= w_prev[cvlt] - 1e-6
        assert on.weights.sum() == pytest.approx(1.0, abs=1e-9)
        assert np.all(on.weights >= -1e-12)

    def test_an_aged_out_name_is_stepped_as_usual(self) -> None:
        rows, cash = _REPLAY["2026-11-09"]
        w_prev = np.array([r[3] for r in rows] + [0.0, cash])
        tickers, _w, off = _solve("2026-11-09", aim_adjustment_rate=0.5)
        _t, _w, on = _solve(
            "2026-11-09",
            aim_adjustment_rate=0.5,
            min_hold_days=3,
            held_sessions=_held_sessions(w_prev, 5),
        )
        assert on.diagnostics["churn_controls"]["protected"] == []
        np.testing.assert_allclose(on.weights, off.weights, atol=1e-9)

    def test_the_rank_buffer_floor_binds_under_the_pace(self) -> None:
        tickers, w_prev, on = _solve("2026-11-09", aim_adjustment_rate=0.5, exit_rank_buffer=6)
        cvlt = tickers.index("CVLT")
        protected = {row["ticker"] for row in on.diagnostics["churn_controls"]["protected"]}
        assert "CVLT" in protected
        assert on.weights[cvlt] >= w_prev[cvlt] - 1e-6

    def test_a_released_name_is_still_paced(self) -> None:
        """An ineligible held name is released by #499 (eligibility outranks
        timing) and, under ``discretionary``, stepped like any other."""
        rows, cash = _REPLAY["2026-11-09"]
        w_prev = np.array([r[3] for r in rows] + [0.0, cash])
        tickers, _w, on = _solve(
            "2026-11-09",
            aim_adjustment_rate=0.5,
            min_hold_days=3,
            held_sessions=_held_sessions(w_prev, 1),
        )
        released = {row["ticker"] for row in on.diagnostics["churn_controls"]["released"]}
        assert {"SNDK", "PLTR"} <= released


class TestTheBandAppliesAfterTheStep:
    """The no-trade band edits the STEPPED book: a stepped move under the band
    is held, a move to zero and a forced trim are not."""

    def test_small_stepped_moves_are_held_and_cash_absorbs_them(self) -> None:
        tickers, w_prev, unbanded = _solve("2026-11-09", aim_adjustment_rate=0.1)
        _t, _w, banded = _solve("2026-11-09", aim_adjustment_rate=0.1, rebalance_band_pct=0.01)
        cash = tickers.index("__CASH__")
        held = {row["ticker"] for row in banded.diagnostics["rebalance_band"]["held"]}
        assert held, "a slow pace must produce at least one sub-band move"
        restored = set(banded.diagnostics["rebalance_band"]["restored_sells"])
        for i, ticker in enumerate(tickers):
            if i == cash:
                continue
            step = float(unbanded.weights[i] - w_prev[i])
            if ticker in held:
                assert abs(step) < 0.01
                assert banded.weights[i] == pytest.approx(w_prev[i], abs=1e-12), ticker
            elif ticker not in restored:
                assert banded.weights[i] == pytest.approx(unbanded.weights[i], abs=1e-12), ticker
        assert banded.weights.sum() == pytest.approx(1.0, abs=1e-9)
        assert _turnover(banded.weights, w_prev) <= _turnover(unbanded.weights, w_prev) + 1e-12
        assert banded.diagnostics["aim_adjustment"]["applied"] is True

    def test_band_and_minimum_hold_and_pace_together(self) -> None:
        rows, cash = _REPLAY["2026-11-09"]
        w_prev = np.array([r[3] for r in rows] + [0.0, cash])
        tickers, _w, on = _solve(
            "2026-11-09",
            aim_adjustment_rate=0.5,
            min_hold_days=3,
            held_sessions=_held_sessions(w_prev, 1),
            rebalance_band_pct=0.01,
        )
        for row in on.diagnostics["churn_controls"]["protected"]:
            i = tickers.index(row["ticker"])
            assert on.weights[i] >= row["floor"] - 1e-6
        moved = np.abs(on.weights - w_prev)[:-1]
        assert np.all((moved < 1e-6) | (moved >= 0.01 - 1e-9) | (on.weights[:-1] == 0.0))
        assert on.weights.sum() == pytest.approx(1.0, abs=1e-9)


def _mask(tickers: list[str], names: set[str]) -> np.ndarray:
    return np.array([t in names for t in tickers])


class TestComposesWithTheHardRiskLayer:
    """crucible#496's hard-risk layer outranks the pace (hard risk > eligibility
    > optimizer > timing): a forced exit leaves in full at any ``k``, past any
    churn floor, and the drawdown throttle bounds the STEPPED book."""

    def test_a_hard_exit_leaves_in_full_under_the_pace_and_a_minimum_hold(self) -> None:
        """MU: held 24.9%, eligible, positive alpha, one session old under a
        three-session minimum hold -- every timing rule would keep it. A
        caller hard exit (loss floor, drawdown exit, lockout) sells it whole."""
        rows, cash = _REPLAY["2026-11-09"]
        tickers = [r[0] for r in rows] + ["SPY", "__CASH__"]
        w_prev = np.array([r[3] for r in rows] + [0.0, cash])
        mu = tickers.index("MU")
        for rate in (0.1, 0.5):
            _t, _w, on = _solve(
                "2026-11-09",
                aim_adjustment_rate=rate,
                min_hold_days=3,
                held_sessions=_held_sessions(w_prev, 1),
                hard_exit=_mask(tickers, {"MU"}),
            )
            assert w_prev[mu] > 0.2
            assert on.weights[mu] == pytest.approx(0.0, abs=1e-9), rate
            assert on.diagnostics["aim_weights"]["MU"] == pytest.approx(0.0, abs=1e-9)
            aim = on.diagnostics["aim_adjustment"]
            assert aim["applied"] is True
            assert "MU" in aim["forced_moves"]
            assert aim["hard_risk_forced_exits"] == ["MU"]
            assert "MU" not in aim["floor_protected"]
            released = {
                row["ticker"]: row["reason"] for row in on.diagnostics["churn_controls"]["released"]
            }
            assert released["MU"] == "hard_risk"
            assert on.diagnostics["hard_risk"]["hard_exit_held_exits"] == ["MU"]
            assert on.weights.sum() == pytest.approx(1.0, abs=1e-9)
            assert np.all(on.weights >= -1e-12)

    def test_a_budget_limited_step_still_takes_the_hard_exit_in_full(self) -> None:
        """A 2% day: the pace is bisected down to fit, the forced exit is not."""
        tickers, w_prev, on = _solve(
            "2026-11-09",
            aim_adjustment_rate=1.0,
            max_daily_turnover=0.02,
            conviction_budget_gate_enabled=False,
        )
        anf = tickers.index("ANF")
        _t, _w, forced = _solve(
            "2026-11-09",
            aim_adjustment_rate=1.0,
            max_daily_turnover=0.02,
            conviction_budget_gate_enabled=False,
            hard_exit=_mask(tickers, {"ANF"}),
        )
        aim = forced.diagnostics["aim_adjustment"]
        assert aim["reason"] == "budget_limited_step"
        assert aim["rate_applied"] < 1.0
        assert forced.weights[anf] == pytest.approx(0.0, abs=1e-9)
        assert w_prev[anf] > 0.1
        assert on.weights[anf] > 0.0

    def test_a_negative_alpha_floor_exit_leaves_in_full_under_the_pace(self) -> None:
        """``negative_alpha_exit_below``: CVLT at -3% against a -2% floor."""
        rows, _cash = _REPLAY["2026-11-09"]
        alpha = np.array([r[1] for r in rows] + [0.0, -1e-6])
        cvlt = [r[0] for r in rows].index("CVLT")
        alpha[cvlt] = -0.03
        tickers, w_prev, on = _solve(
            "2026-11-09",
            alpha=alpha,
            aim_adjustment_rate=0.25,
            negative_alpha_exit_below=-0.02,
        )
        assert w_prev[cvlt] > 0.10
        assert on.weights[cvlt] == pytest.approx(0.0, abs=1e-9)
        assert on.diagnostics["hard_risk"]["negative_alpha_held_exits"] == ["CVLT"]
        assert on.diagnostics["aim_adjustment"]["hard_risk_forced_exits"] == ["CVLT"]

    def test_at_the_breaker_the_stepped_book_grows_nothing(self) -> None:
        """11-06: 48% idle cash that the pace would deploy k of the way. At the
        circuit breaker no name may grow, so the cash stays cash."""
        tickers, w_prev, free = _solve("2026-11-06", aim_adjustment_rate=0.5)
        cash = tickers.index("__CASH__")
        assert np.any(free.weights[:cash] > w_prev[:cash] + 0.01), "the pace must buy unthrottled"
        _t, _w, on = _solve("2026-11-06", aim_adjustment_rate=0.5, buy_multiplier=0.0)
        assert np.all(on.weights[:cash] <= w_prev[:cash] + 1e-9)
        assert on.weights.sum() == pytest.approx(1.0, abs=1e-9)
        assert on.weights[cash] >= w_prev[cash] - 1e-9
        assert on.diagnostics["aim_adjustment"]["cash_absorbs_residual"] is True
        assert on.diagnostics["hard_risk"]["cash_sleeve_is_floor"] is True

    def test_at_the_breaker_a_protected_name_is_held_and_a_hard_exit_stays_cash(self) -> None:
        rows, cash_prev = _REPLAY["2026-11-09"]
        tickers = [r[0] for r in rows] + ["SPY", "__CASH__"]
        w_prev = np.array([r[3] for r in rows] + [0.0, cash_prev])
        cash = tickers.index("__CASH__")
        mu, cvlt = tickers.index("MU"), tickers.index("CVLT")
        _t, _w, on = _solve(
            "2026-11-09",
            aim_adjustment_rate=0.5,
            min_hold_days=3,
            held_sessions=_held_sessions(w_prev, 1),
            hard_exit=_mask(tickers, {"MU"}),
            buy_multiplier=0.0,
        )
        assert on.weights[mu] == pytest.approx(0.0, abs=1e-9)
        assert np.all(on.weights[:cash] <= w_prev[:cash] + 1e-9)
        # CVLT is inside its minimum hold: held, not grown.
        assert on.weights[cvlt] == pytest.approx(w_prev[cvlt], abs=1e-6)
        # MU's proceeds are not redeployed: cash holds at least what it freed.
        assert on.weights[cash] >= cash_prev + w_prev[mu] - 1e-6
        assert on.weights.sum() == pytest.approx(1.0, abs=1e-9)

    def test_at_half_size_no_stepped_name_grows_past_half_its_room(self) -> None:
        tickers, w_prev, on = _solve("2026-11-06", aim_adjustment_rate=1.0, buy_multiplier=0.5)
        cash = tickers.index("__CASH__")
        cap = 0.25  # _PARAMS["max_position_pct"]; every replay name is unsectored
        for i in range(cash):
            room = max(cap - w_prev[i], 0.0)
            assert on.weights[i] <= w_prev[i] + 0.5 * room + 1e-6, tickers[i]
        assert on.weights.sum() == pytest.approx(1.0, abs=1e-9)


def test_what_the_throttled_caps_cannot_take_stays_cash() -> None:
    """``_aim_step`` under the drawdown ladder: a deferred entry's weight that
    no cap has room for is left in cash, never lost from the book."""
    from crucible.portfolio import _aim_step

    prev = np.array([0.0, 0.0, 1.0])
    aim = np.array([0.006, 0.006, 0.988])
    caps = np.array([0.006, 0.006, 1.0])
    none = np.zeros(3, dtype=bool)
    args = (0.5, aim, prev, caps, np.zeros(3), none, none, 2, 0.005)
    w, zeroed, _mass, unplaced = _aim_step(*args, pace_cash=False, cash_absorbs=True)
    assert zeroed.tolist() == [True, True, False]
    assert unplaced == 0.0
    assert w.tolist() == pytest.approx([0.0, 0.0, 1.0])
    _w, _z, _m, lost = _aim_step(*args, pace_cash=False, cash_absorbs=False)
    assert lost > 0.0
