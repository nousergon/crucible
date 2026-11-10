"""The three churn controls Brian approved on 2026-11-10 (alpha-engine-config-I12024).

v2 bought CVLT on 11-09 and sold it on 11-10 because its alpha rank slipped
from 3 to 5, and sent four touch-up orders of under 0.2% of the book each:
"it shouldn't be just redoing the order book every day." The rules:

1. ``min_hold_days`` -- a name entered stays held that many sessions;
2. ``exit_rank_buffer`` -- after that, a held name is exited only when it
   falls out of the top N by alpha rank, not out of the optimizer's target;
3. ``rebalance_band_pct`` -- a weight change smaller than the band is skipped.

Precedence: hard risk (negative alpha, the per-name cap) > eligibility >
optimizer > timing. All three are OFF when the parameter file does not
declare them, so merging the engine changes no book.

**The parameter values below are FIXTURE values** (they are Brian's ruled
values because the tests are about his cases); the operating values live in
the private strategy tree.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from crucible.portfolio import (
    PORTFOLIO_PARAM_FIELDS,
    PortfolioParams,
    PortfolioParamsError,
    alpha_ranks,
    solve_target_weights,
)
from crucible.slots.strategy import (
    BookUniverse,
    ExitRuleSpec,
    SessionInputs,
    StrategyRecipe,
    construct_book,
)
from tests.test_portfolio import (
    _FIXTURE_PARAMS,  # pyright: ignore[reportPrivateUsage]
    _flat_model,  # pyright: ignore[reportPrivateUsage]
    _synthetic_returns,  # pyright: ignore[reportPrivateUsage]
)

N_NAMES = 12
TICKERS = [f"T{i}" for i in range(N_NAMES)] + ["SPY", "CASH"]
SPY, CASH = N_NAMES, N_NAMES + 1
CVLT = 2  # T2 plays CVLT: rank 3 on the day it is bought.
RULED = {"min_hold_days": 5, "exit_rank_buffer": 10, "rebalance_band_pct": 0.01}

#: Day 1: T0 > T1 > T2 > ... The optimizer buys the top four, T2 among them.
ALPHA_ENTRY = [0.010, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001, 5e-4, 2e-4]
#: Day 2: T3, T4 and T5 overtake T2, which is now rank 5 -- still top 10.
ALPHA_RANK_5 = [0.010, 0.009, 0.0065, 0.0085, 0.0075, 0.007, 0.004, 0.003, 0.002, 0.001, 5e-4, 2e-4]


def _params(**overrides: Any) -> PortfolioParams:
    base = {**_FIXTURE_PARAMS, "max_daily_turnover": None}
    return PortfolioParams.from_mapping({**base, **overrides}, source="<churn fixture>")


def _alpha(per_name: list[float]) -> np.ndarray:
    return np.array([*per_name, 0.0, -1e-4])


def _with_rank(rank: int) -> list[float]:
    """ALPHA_ENTRY with only T2 moved, so it ranks ``rank`` (1-based) of twelve."""
    others = sorted((a for i, a in enumerate(ALPHA_ENTRY) if i != CVLT), reverse=True)
    if rank == 1:
        value = others[0] + 1e-3
    elif rank == N_NAMES:
        value = others[-1] / 2  # still positive: last, not negative
    else:
        value = (others[rank - 2] + others[rank - 1]) / 2
    out = list(ALPHA_ENTRY)
    out[CVLT] = value
    return out


def _solve(
    alpha: list[float],
    w_prev: np.ndarray,
    params: PortfolioParams,
    *,
    held: dict[int, int] | None = None,
    eligibility: np.ndarray | None = None,
):
    ages = None
    if held is not None:
        ages = np.zeros(len(TICKERS))
        for i, sessions in held.items():
            ages[i] = sessions
    return solve_target_weights(
        TICKERS,
        _alpha(alpha),
        _synthetic_returns(len(TICKERS)),
        np.asarray(w_prev, dtype=float),
        [f"s{i}" for i in range(N_NAMES)] + ["__benchmark__", "__cash__"],
        np.full(len(TICKERS), 0.25),
        np.ones(len(TICKERS), dtype=bool) if eligibility is None else eligibility,
        SPY,
        CASH,
        params,
        _flat_model(),
        held_sessions=ages,
    )


def _all_cash() -> np.ndarray:
    w = np.zeros(len(TICKERS))
    w[CASH] = 1.0
    return w


@pytest.fixture(scope="module")
def entry_book() -> np.ndarray:
    """The book the CVLT day bought: T2 held at the cap after one session."""
    weights = _solve(ALPHA_ENTRY, _all_cash(), _params()).weights
    assert weights[CVLT] == pytest.approx(0.25, abs=1e-4)
    return weights


def _protected(result) -> dict[str, list[str]]:
    return {r["ticker"]: r["reasons"] for r in result.diagnostics["churn_controls"]["protected"]}


def _released(result) -> dict[str, str]:
    return {r["ticker"]: r["reason"] for r in result.diagnostics["churn_controls"]["released"]}


class TestTheCvltCase:
    """Bought at rank 3, rank 5 the next session: v2 sold it. It must stay held."""

    def test_held_one_session_and_slipping_to_rank_5_it_stays_held(self, entry_book) -> None:
        result = _solve(ALPHA_RANK_5, entry_book, _params(**RULED), held={0: 1, 1: 1, 2: 1, 3: 1})
        assert result.weights[CVLT] >= entry_book[CVLT] - 1e-6
        assert _protected(result)["T2"] == ["min_hold", "rank_buffer"]
        assert result.diagnostics["status"] in {"optimal", "optimal_inaccurate"}

    def test_the_minimum_hold_alone_holds_it_even_out_of_the_top_10(self, entry_book) -> None:
        result = _solve(
            _with_rank(12), entry_book, _params(min_hold_days=5), held={0: 1, 1: 1, 2: 1, 3: 1}
        )
        assert result.weights[CVLT] >= entry_book[CVLT] - 1e-6
        assert _protected(result)["T2"] == ["min_hold"]

    def test_without_the_controls_the_optimizer_sells_it_the_next_day(self, entry_book) -> None:
        """The 11-10 defect, reproduced: the reason the rules exist."""
        result = _solve(ALPHA_RANK_5, entry_book, _params())
        assert result.weights[CVLT] == 0.0


class TestExitHysteresis:
    """After the hold, the exit line is the top-10 rank, not the target set."""

    def test_inside_the_top_10_after_the_hold_it_keeps_its_position(self, entry_book) -> None:
        result = _solve(_with_rank(10), entry_book, _params(**RULED), held={0: 6, 1: 6, 2: 6, 3: 6})
        assert result.weights[CVLT] >= entry_book[CVLT] - 1e-6
        assert _protected(result)["T2"] == ["rank_buffer"]
        # Without the buffer the optimizer drops it: the buffer is what holds it.
        assert _solve(_with_rank(10), entry_book, _params()).weights[CVLT] == 0.0

    def test_out_of_the_top_10_after_the_hold_it_is_exited(self, entry_book) -> None:
        result = _solve(_with_rank(11), entry_book, _params(**RULED), held={0: 6, 1: 6, 2: 6, 3: 6})
        assert result.weights[CVLT] == 0.0
        assert "T2" not in _protected(result)

    def test_a_protected_name_may_still_be_added_to(self) -> None:
        """The floor is the held weight; the optimizer may go above it."""
        held = _all_cash()
        held[CVLT], held[SPY], held[CASH] = 0.05, 0.92, 0.03
        result = _solve(ALPHA_ENTRY, held, _params(**RULED), held={CVLT: 1})
        assert result.weights[CVLT] > 0.05 + 0.01


class TestHigherTiersStillWin:
    """Hard risk > eligibility > optimizer > timing."""

    def test_negative_alpha_overrides_the_hold(self, entry_book) -> None:
        alpha = list(ALPHA_RANK_5)
        alpha[CVLT] = -0.005
        result = _solve(alpha, entry_book, _params(**RULED), held={0: 1, 1: 1, 2: 1, 3: 1})
        assert result.weights[CVLT] == 0.0
        assert _released(result)["T2"] == "negative_alpha"

    def test_ineligibility_overrides_the_hold(self, entry_book) -> None:
        eligible = np.ones(len(TICKERS), dtype=bool)
        eligible[CVLT] = False
        result = _solve(
            ALPHA_RANK_5,
            entry_book,
            _params(**RULED),
            held={0: 1, 1: 1, 2: 1, 3: 1},
            eligibility=eligible,
        )
        assert result.weights[CVLT] == 0.0
        assert _released(result)["T2"] == "ineligible"

    def test_the_per_name_cap_bounds_the_floor(self, entry_book) -> None:
        result = _solve(
            ALPHA_RANK_5,
            entry_book,
            _params(**RULED, max_position_pct=0.10),
            held={0: 1, 1: 1, 2: 1, 3: 1},
        )
        assert result.weights[CVLT] == pytest.approx(0.10, abs=1e-5)
        assert {r["ticker"]: r["floor"] for r in result.diagnostics["churn_controls"]["protected"]}[
            "T2"
        ] == pytest.approx(0.10)


class TestTheNoTradeBand:
    def test_a_0_2_pct_drift_trade_is_skipped(self, entry_book) -> None:
        drifted = entry_book.copy()
        drifted[3] -= 0.002  # T3 drifted 0.2% under the target the solve returns.
        drifted[SPY] += 0.002
        unbanded = _solve(ALPHA_ENTRY, drifted, _params())
        assert unbanded.weights[3] - drifted[3] == pytest.approx(0.002, abs=2e-4)
        banded = _solve(ALPHA_ENTRY, drifted, _params(**RULED), held={0: 2, 1: 2, 2: 2, 3: 2})
        assert banded.weights[3] == pytest.approx(drifted[3])
        assert "T3" in {r["ticker"] for r in banded.diagnostics["rebalance_band"]["held"]}


class TestOffByDefault:
    """No key in the parameter file: no rule, and the book the engine built before."""

    def test_absent_keys_read_as_off(self) -> None:
        params = _params()
        assert params.min_hold_days is None
        assert params.exit_rank_buffer is None
        assert params.rebalance_band_pct is None

    def test_absent_keys_build_the_unprotected_book(self, entry_book) -> None:
        nulls = {"min_hold_days": None, "exit_rank_buffer": None, "rebalance_band_pct": None}
        absent = _solve(ALPHA_RANK_5, entry_book, _params())
        explicit = _solve(ALPHA_RANK_5, entry_book, _params(**nulls))
        np.testing.assert_array_equal(absent.weights, explicit.weights)
        assert absent.weights[CVLT] == 0.0
        controls = absent.diagnostics["churn_controls"]
        assert controls["protected"] == [] and controls["released"] == []
        assert absent.diagnostics["rebalance_band"]["held"] == []

    def test_off_needs_no_holding_ages(self, entry_book) -> None:
        """A caller that has never heard of the clock keeps working."""
        result = _solve(ALPHA_RANK_5, entry_book, _params(exit_rank_buffer=10))
        assert result.weights[CVLT] > 0.0


class TestTheHoldingClock:
    def test_a_held_book_without_ages_is_refused_when_the_hold_is_on(self, entry_book) -> None:
        with pytest.raises(ValueError, match="no held_sessions were given"):
            _solve(ALPHA_RANK_5, entry_book, _params(min_hold_days=5))

    @pytest.mark.parametrize("bad", [0, 1.5])
    def test_a_held_name_needs_a_whole_positive_age(self, entry_book, bad) -> None:
        with pytest.raises(ValueError, match="whole sessions >= 1"):
            _solve(ALPHA_RANK_5, entry_book, _params(min_hold_days=5), held={0: 1, 1: 1, 2: bad})

    def test_ages_must_cover_the_universe(self, entry_book) -> None:
        with pytest.raises(ValueError, match="held_sessions shape"):
            solve_target_weights(
                TICKERS,
                _alpha(ALPHA_RANK_5),
                _synthetic_returns(len(TICKERS)),
                entry_book,
                [f"s{i}" for i in range(N_NAMES)] + ["__benchmark__", "__cash__"],
                np.full(len(TICKERS), 0.25),
                np.ones(len(TICKERS), dtype=bool),
                SPY,
                CASH,
                _params(min_hold_days=5),
                _flat_model(),
                held_sessions=np.ones(3),
            )

    def test_floors_that_would_crowd_out_the_sleeve_are_scaled(self) -> None:
        held = np.zeros(len(TICKERS))
        held[:4] = 0.25  # fully invested: no cash, the sleeve pin needs 3%
        result = _solve(ALPHA_ENTRY, held, _params(**RULED), held={0: 1, 1: 1, 2: 1, 3: 1})
        assert result.diagnostics["churn_controls"]["floors_scaled"] is True
        assert result.weights[CASH] == pytest.approx(0.03, abs=1e-6)
        assert all(result.weights[i] > 0.2 for i in range(4))

    def test_floors_inside_one_sector_are_scaled_to_its_cap(self) -> None:
        held = _all_cash()
        held[0], held[1], held[SPY], held[CASH] = 0.2, 0.2, 0.57, 0.03
        result = solve_target_weights(
            TICKERS,
            _alpha(ALPHA_ENTRY),
            _synthetic_returns(len(TICKERS)),
            held,
            ["tech", "tech"] + [f"s{i}" for i in range(2, N_NAMES)] + ["__benchmark__", "__cash__"],
            np.full(len(TICKERS), 0.25),
            np.ones(len(TICKERS), dtype=bool),
            SPY,
            CASH,
            _params(exit_rank_buffer=10),
            _flat_model(),
        )
        assert result.diagnostics["churn_controls"]["floors_scaled"] is True
        assert result.weights[0] + result.weights[1] <= 0.25 + 1e-6

    def test_tied_alphas_share_the_better_rank(self) -> None:
        alpha = np.array([0.0, 0.0, 0.01, 0.0, -1e-4])
        eligible = np.array([True, True, True, True, True])
        np.testing.assert_array_equal(alpha_ranks(alpha, eligible, 3, 4), [2, 2, 1, 0, 0])


def _book_universe() -> BookUniverse:
    return BookUniverse(
        tickers=tuple(TICKERS),
        sectors=tuple([f"s{i}" for i in range(N_NAMES)] + ["__benchmark__", "__cash__"]),
        benchmark_idx=SPY,
        cash_idx=CASH,
    )


def _walk(days: list[list[float]]) -> tuple[SessionInputs, ...]:
    panel = _synthetic_returns(len(TICKERS))
    panel[:, CASH] = 0.0
    return tuple(
        SessionInputs(
            trading_day=f"2026-11-{9 + i:02d}",
            alpha_hat=_alpha(alpha),
            eligibility=np.ones(len(TICKERS), dtype=bool),
            stance_caps=np.full(len(TICKERS), 0.25),
            realized_returns=np.zeros(len(TICKERS)),
            benchmark_return=0.0,
            returns_panel=panel,
        )
        for i, alpha in enumerate(days)
    )


def _construct(params: PortfolioParams, sessions, w_initial=None, held_sessions=None):
    return construct_book(
        recipe=StrategyRecipe(
            name="churn_fixture",
            rules=(ExitRuleSpec(rule_id="profit_take", params={"profit_take_pct": 0.25}),),
            cost_model=_flat_model(),
        ),
        params=params,
        universe=_book_universe(),
        sessions=sessions,
        portfolio_notional=1_000_000.0,
        w_initial=_all_cash() if w_initial is None else w_initial,
        held_sessions=held_sessions,
    )


class TestTheWalkKeepsTheClock:
    """A walk from cash knows every age and advances it session by session."""

    def test_the_hold_expires_on_the_fifth_session(self) -> None:
        sessions = _walk([ALPHA_ENTRY] + [_with_rank(12)] * 6)
        weights = [w[CVLT] for w in _construct(_params(min_hold_days=5), sessions).weights]
        # Entered on the first decision; held into sessions 1..4; out on the 5th.
        assert weights[0] == pytest.approx(0.25, abs=1e-4)
        assert all(w > 0.2 for w in weights[1:5])
        assert weights[5] == 0.0

    def test_a_walk_from_a_held_book_takes_its_ages(self, entry_book) -> None:
        sessions = _walk([_with_rank(12)] * 2)
        built = _construct(
            _params(min_hold_days=5),
            sessions,
            w_initial=entry_book,
            held_sessions={"T0": 4, "T1": 4, "T2": 4, "T3": 4},
        )
        assert built.weights[0][CVLT] > 0.2  # four sessions held: still inside the hold
        assert built.weights[1][CVLT] == 0.0  # five: the optimizer decides, and sells

    def test_an_age_for_a_name_outside_the_universe_is_refused(self) -> None:
        with pytest.raises(ValueError, match="universe does not contain"):
            _construct(_params(), _walk([ALPHA_ENTRY]), held_sessions={"ZZZ": 3})

    def test_off_by_default_the_walk_is_unchanged(self) -> None:
        sessions = _walk([ALPHA_ENTRY, ALPHA_RANK_5, _with_rank(11)])
        nulls = {"min_hold_days": None, "exit_rank_buffer": None, "rebalance_band_pct": None}
        assert (
            _construct(_params(), sessions).weights
            == _construct(_params(**nulls), sessions).weights
        )


class TestTheParameters:
    def test_the_registry_declares_the_two_controls(self) -> None:
        assert PORTFOLIO_PARAM_FIELDS["min_hold_days"] == "integer-or-null"
        assert PORTFOLIO_PARAM_FIELDS["exit_rank_buffer"] == "integer-or-null"

    @pytest.mark.parametrize("field", ["min_hold_days", "exit_rank_buffer"])
    def test_zero_is_spelled_null(self, field) -> None:
        with pytest.raises(PortfolioParamsError, match="Zero is spelled null"):
            _params(**{field: 0})

    @pytest.mark.parametrize("field", ["min_hold_days", "exit_rank_buffer"])
    def test_a_fractional_count_is_refused(self, field) -> None:
        with pytest.raises(PortfolioParamsError, match="must be an integer"):
            _params(**{field: 2.5})

    def test_the_ruled_values_load(self) -> None:
        params = _params(**RULED)
        assert (params.min_hold_days, params.exit_rank_buffer) == (5, 10)
        assert params.to_dict()["min_hold_days"] == 5
