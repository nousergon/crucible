"""The hard-risk layer (v1 → v2 parity audit 2026-11-09, rows 3, 4, 6, 7, 8).

Forced exits must be FORCED: above the eligibility cut, above
``ineligible_held_exit: discretionary``, and never limited by the turnover
budget. Every threshold below is the v1 live value, used as a fixture only;
the repository declares none.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from crucible.hard_risk import (
    DRAWDOWN_EXIT,
    LOSS_FLOOR_EXIT,
    Holding,
    drawdown_state,
    lockout_through,
    loss_floor_breaches,
    select_drawdown_exits,
)
from crucible.portfolio import (
    _ROLLOUT_ABSENT,  # pyright: ignore[reportPrivateUsage]
    DRAWDOWN_LADDER_FIELDS,
    PortfolioParams,
    PortfolioParamsError,
    _fallback_weights,  # pyright: ignore[reportPrivateUsage]
    declared_value,
    forced_exit_mask,
    solve_target_weights,
)
from tests.test_portfolio import _FIXTURE_PARAMS, _baseline_universe, _params

#: v1's live values (`alpha-engine-config/executor/risk.yaml`,
#: `strategies/config.py`) -- a fixture, not a default.
V1 = {
    "negative_alpha_exit_below": -0.02,
    "position_loss_floor_pct": -0.15,
    "hard_risk_reentry_lockout_sessions": 5,
    "drawdown_half_size_at": -0.03,
    "drawdown_quarter_size_at": -0.05,
    "drawdown_circuit_breaker_at": -0.08,
    "drawdown_forced_exits_at_half": 1,
    "drawdown_forced_exits_at_quarter": 2,
}


def _v1(**overrides) -> PortfolioParams:
    return _params(**{**V1, **overrides})


def _universe(**param_overrides) -> dict:
    """Four names + SPY + CASH, uncapped, no sector binding."""
    u = _baseline_universe(n_active=4, sector_labels=["a", "b", "c", "d"])
    u["stance_caps"] = np.full(len(u["tickers"]), 1.0)
    u["alpha_hat"] = np.array([0.03, 0.01, 0.01, 0.01, 0.0, -0.0001])
    u["params"] = _v1(max_sector_pct=1.0, **param_overrides)
    return u


# ── Parameters ───────────────────────────────────────────────────────────


class TestParameters:
    def test_absent_hard_risk_fields_read_as_off(self) -> None:
        params = PortfolioParams.from_mapping(dict(_FIXTURE_PARAMS), source="<old file>")
        for name in V1:
            assert name in _ROLLOUT_ABSENT
            assert getattr(params, name) is None

    def test_v1_values_load_with_integers_as_integers(self) -> None:
        params = _v1()
        assert params.hard_risk_reentry_lockout_sessions == 5
        assert isinstance(params.drawdown_forced_exits_at_quarter, int)
        assert params.to_dict()["position_loss_floor_pct"] == -0.15

    def test_an_explicit_null_is_off(self) -> None:
        params = _v1(
            negative_alpha_exit_below=None,
            **dict.fromkeys(DRAWDOWN_LADDER_FIELDS),
        )
        assert params.negative_alpha_exit_below is None
        assert params.drawdown_circuit_breaker_at is None

    def test_an_integer_field_refuses_a_float(self) -> None:
        with pytest.raises(PortfolioParamsError, match="integer"):
            _v1(hard_risk_reentry_lockout_sessions=5.0)

    def test_half_a_ladder_is_refused(self) -> None:
        with pytest.raises(PortfolioParamsError, match="all five"):
            _v1(drawdown_circuit_breaker_at=None)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("negative_alpha_exit_below", 0.01),
            ("position_loss_floor_pct", 0.0),
            ("position_loss_floor_pct", -1.0),
            ("hard_risk_reentry_lockout_sessions", 0),
            ("drawdown_half_size_at", -0.06),
            ("drawdown_circuit_breaker_at", -1.0),
            ("drawdown_forced_exits_at_half", -1),
        ],
    )
    def test_out_of_range_values_are_refused(self, field, value) -> None:
        with pytest.raises(PortfolioParamsError):
            _v1(**{field: value})

    def test_declared_value_refuses_null(self) -> None:
        assert declared_value(-0.03) == -0.03
        with pytest.raises(PortfolioParamsError):
            declared_value(None)


# ── The pure rules ───────────────────────────────────────────────────────


class TestLossFloor:
    def test_a_name_16pct_below_cost_breaches_and_14pct_does_not(self) -> None:
        holdings = {
            "AAA": Holding(shares=100, avg_cost=100.0, mark=84.0),
            "BBB": Holding(shares=100, avg_cost=100.0, mark=86.0),
            "CCC": Holding(shares=100, avg_cost=100.0, mark=85.0),
        }
        breaches = loss_floor_breaches(holdings, _v1())
        assert sorted(breaches) == ["AAA", "CCC"]  # at the floor exactly is a breach
        assert breaches["AAA"] == pytest.approx(-0.16)

    def test_off_when_null(self) -> None:
        holdings = {"AAA": Holding(shares=1, avg_cost=100.0, mark=1.0)}
        assert loss_floor_breaches(holdings, _v1(position_loss_floor_pct=None)) == {}

    def test_an_unpriced_holding_raises_rather_than_being_skipped(self) -> None:
        with pytest.raises(ValueError, match="partial book"):
            loss_floor_breaches({"AAA": Holding(shares=1, avg_cost=None, mark=90.0)}, _v1())

    def test_a_flat_position_is_ignored(self) -> None:
        holdings = {"AAA": Holding(shares=0, avg_cost=None, mark=None)}
        assert loss_floor_breaches(holdings, _v1()) == {}

    @pytest.mark.parametrize(
        ("cost", "mark"), [(0.0, 10.0), (10.0, 0.0), (float("nan"), 10.0), (10.0, None)]
    )
    def test_unusable_prices_have_no_loss(self, cost, mark) -> None:
        assert Holding(shares=1, avg_cost=cost, mark=mark).loss_from_cost is None


class TestDrawdownLadder:
    @pytest.mark.parametrize(
        ("nav", "multiplier", "exits", "breaker"),
        [
            (1_000_000.0, 1.0, 0, False),
            (975_000.0, 1.0, 0, False),
            (970_000.0, 0.5, 1, False),
            (951_000.0, 0.5, 1, False),
            (950_000.0, 0.25, 2, False),
            (921_000.0, 0.25, 2, False),
            (920_000.0, 0.0, 2, True),
            (800_000.0, 0.0, 2, True),
        ],
    )
    def test_the_rungs(self, nav, multiplier, exits, breaker) -> None:
        state = drawdown_state(nav, 1_000_000.0, _v1())
        assert state is not None
        assert state.multiplier == multiplier
        assert state.forced_exit_count == exits
        assert state.circuit_breaker is breaker

    def test_a_new_high_resets_the_peak(self) -> None:
        state = drawdown_state(1_100_000.0, 1_000_000.0, _v1())
        assert state is not None
        assert state.peak_nav_usd == 1_100_000.0
        assert state.drawdown == 0.0
        assert state.to_document()["multiplier"] == 1.0

    def test_the_first_session_seeds_the_peak(self) -> None:
        state = drawdown_state(900_000.0, None, _v1())
        assert state is not None
        assert state.peak_nav_usd == 900_000.0
        assert state.multiplier == 1.0

    def test_off_when_null(self) -> None:
        assert drawdown_state(1.0, 2.0, _v1(**dict.fromkeys(DRAWDOWN_LADDER_FIELDS))) is None

    @pytest.mark.parametrize("nav", [0.0, -1.0, float("nan")])
    def test_an_unreadable_nav_raises(self, nav) -> None:
        with pytest.raises(ValueError, match="NAV"):
            drawdown_state(nav, 1.0, _v1())


class TestDrawdownExitSelection:
    HOLDINGS = {
        "AAA": Holding(shares=10, avg_cost=10.0, mark=9.0),
        "BBB": Holding(shares=10, avg_cost=10.0, mark=11.0),
        "CCC": Holding(shares=10, avg_cost=10.0, mark=8.0),
        "DDD": Holding(shares=0, avg_cost=None, mark=None),
    }

    def test_lowest_alpha_first(self) -> None:
        alpha = {"AAA": 0.02, "BBB": -0.01, "CCC": 0.05}
        assert select_drawdown_exits(self.HOLDINGS, alpha, 2) == ["BBB", "AAA"]

    def test_ties_break_on_the_worse_loss_from_cost(self) -> None:
        assert select_drawdown_exits(self.HOLDINGS, {}, 1) == ["CCC"]

    def test_names_already_exiting_count_toward_the_tier(self) -> None:
        assert select_drawdown_exits(self.HOLDINGS, {}, 2, already_exiting={"CCC"}) == ["AAA"]
        assert select_drawdown_exits(self.HOLDINGS, {}, 1, already_exiting={"CCC"}) == []

    def test_zero_count_selects_nothing(self) -> None:
        assert select_drawdown_exits(self.HOLDINGS, {}, 0) == []


class TestLockoutWindow:
    def test_five_sessions_after_a_friday_exit_skip_the_weekend(self) -> None:
        # 2026-11-06 is a Friday; five sessions later is Friday 2026-11-13.
        assert lockout_through(dt.date(2026, 11, 6), 5) == dt.date(2026, 11, 13)

    def test_a_holiday_is_not_a_session(self) -> None:
        # Thanksgiving 2026-11-26 is not a session: Wed 11-25 + 1 is Fri 11-27.
        assert lockout_through(dt.date(2026, 11, 25), 1) == dt.date(2026, 11, 27)

    def test_zero_sessions_is_refused(self) -> None:
        with pytest.raises(ValueError, match="null disables"):
            lockout_through(dt.date(2026, 11, 6), 0)

    def test_reason_vocabulary(self) -> None:
        assert LOSS_FLOOR_EXIT != DRAWDOWN_EXIT


# ── The solver: forced pins are forced ───────────────────────────────────


class TestForcedExitMask:
    def test_negative_alpha_pins_and_sentinels_are_exempt(self) -> None:
        u = _universe()
        u["alpha_hat"] = np.array([-0.03, -0.02, 0.0, 0.01, -0.5, -0.5])
        mask = forced_exit_mask(u["alpha_hat"], None, 4, 5, u["params"])
        assert mask.tolist() == [True, False, False, False, False, False]

    def test_off_when_null(self) -> None:
        params = _v1(negative_alpha_exit_below=None)
        assert not forced_exit_mask(np.full(6, -1.0), None, 4, 5, params).any()

    def test_a_hard_exit_on_a_sentinel_is_refused(self) -> None:
        mask = np.zeros(6, dtype=bool)
        mask[5] = True
        with pytest.raises(ValueError, match="sentinel"):
            forced_exit_mask(np.zeros(6), mask, 4, 5, _v1())

    def test_a_misshapen_mask_is_refused(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            forced_exit_mask(np.zeros(6), np.zeros(3, dtype=bool), 4, 5, _v1())


class TestForcedExitsBypassTheBudgetAndTheDiscretionaryPolicy:
    def _held(self, **param_overrides) -> dict:
        u = _universe(
            ineligible_held_exit="discretionary", max_daily_turnover=0.02, **param_overrides
        )
        u["w_prev"] = np.array([0.40, 0.20, 0.20, 0.17, 0.0, 0.03])
        return u

    def test_a_hard_exit_held_name_is_sold_whole_under_a_two_percent_budget(self) -> None:
        u = self._held()
        u["hard_exit"] = np.array([True, False, False, False, False, False])
        result = solve_target_weights(**u)
        assert result.weights[0] == pytest.approx(0.0, abs=1e-8)
        assert result.diagnostics["turnover_constraint_cap"] > 0.02
        assert result.diagnostics["turnover_mandatory_floor_by_cause"][
            "ineligibility_pin"
        ] == pytest.approx(0.2)
        assert result.diagnostics["hard_risk"]["hard_exit_held_exits"] == ["T0"]

    def test_a_negative_alpha_held_name_is_forced_even_when_discretionary(self) -> None:
        u = self._held()
        u["alpha_hat"] = np.array([-0.05, 0.01, 0.01, 0.01, 0.0, -0.0001])
        u["eligibility"] = np.array([False, True, True, True, True, True])
        result = solve_target_weights(**u)
        assert result.weights[0] == pytest.approx(0.0, abs=1e-8)
        assert result.diagnostics["hard_risk"]["negative_alpha_held_exits"] == ["T0"]

    def test_without_the_pin_discretionary_keeps_it(self) -> None:
        u = self._held(negative_alpha_exit_below=None)
        u["alpha_hat"] = np.array([-0.05, 0.01, 0.01, 0.01, 0.0, -0.0001])
        u["eligibility"] = np.array([False, True, True, True, True, True])
        result = solve_target_weights(**u)
        assert result.weights[0] > 0.3

    def test_diagnostics_are_emitted_when_nothing_engaged(self) -> None:
        result = solve_target_weights(**_universe())
        block = result.diagnostics["hard_risk"]
        assert block["hard_exit_pinned"] == []
        assert block["negative_alpha_pinned"] == []
        assert block["buy_multiplier"] == 1.0
        assert block["cash_sleeve_is_floor"] is False


class TestTheDrawdownThrottle:
    def _held(self) -> dict:
        u = _universe(max_daily_turnover=None)
        u["w_prev"] = np.array([0.10, 0.30, 0.30, 0.27, 0.0, 0.03])
        u["alpha_hat"] = np.array([0.10, 0.0, 0.0, 0.0, 0.0, -0.0001])
        return u

    def test_at_the_breaker_no_name_grows_and_freed_cash_stays_cash(self) -> None:
        u = self._held()
        u["hard_exit"] = np.array([False, True, False, False, False, False])
        result = solve_target_weights(**u, buy_multiplier=0.0)
        w = result.weights
        assert np.all(w[:5] <= u["w_prev"][:5] + 1e-7)
        assert w[1] == pytest.approx(0.0, abs=1e-8)
        assert w[5] >= 0.03 + 0.29
        assert result.diagnostics["hard_risk"]["cash_sleeve_is_floor"] is True

    def test_at_half_size_a_name_grows_by_at_most_half_its_room(self) -> None:
        u = self._held()
        result = solve_target_weights(**u, buy_multiplier=0.5)
        room = 1.0 - 0.10
        assert result.weights[0] <= 0.10 + 0.5 * room + 1e-6
        assert result.weights[0] > 0.10

    def test_an_out_of_range_multiplier_is_refused(self) -> None:
        with pytest.raises(ValueError, match="buy_multiplier"):
            solve_target_weights(**self._held(), buy_multiplier=1.5)


def test_the_infeasible_fallback_still_exits_a_forced_name() -> None:
    w_prev = np.array([0.4, 0.3, 0.27, 0.0, 0.03])
    forced = np.array([True, False, False, False, False])
    weights = _fallback_weights(w_prev, 4, 0.03, forced=forced)
    assert weights[0] == 0.0
    assert weights[1:3].tolist() == pytest.approx([0.3, 0.27])
    assert weights[4] == pytest.approx(0.43)
    assert weights.sum() == pytest.approx(1.0)


def test_an_infeasible_solve_with_a_forced_exit_takes_the_fallback_without_the_name() -> None:
    u = _universe(vol_target_annual=1e-9, max_daily_turnover=None)
    u["w_prev"] = np.array([0.40, 0.20, 0.20, 0.17, 0.0, 0.03])
    u["hard_exit"] = np.array([True, False, False, False, False, False])
    result = solve_target_weights(**u)
    assert result.diagnostics["status"] == "infeasible_fallback"
    assert result.weights[0] == 0.0
    assert result.diagnostics["hard_risk"]["hard_exit_pinned"] == ["T0"]


# ── Replay: a -16% held name is forced out and not rebought for 5 sessions ──


def test_replay_loss_floor_exit_then_five_session_lockout() -> None:
    """Seven sessions of the trader's hard-risk orchestration over the engine.

    T0 is held at 40% and marks 16% below its average cost on 11-06 while the
    model still ranks it first. It is forced out that session under a 2%
    discretionary budget, stays out for the five sessions after (11-09 ..
    11-13) on the same top alpha, and is bought back on 11-16.
    """
    params = _v1(max_sector_pct=1.0, max_daily_turnover=0.02, ineligible_held_exit="discretionary")
    sessions = ["2026-11-06", "2026-11-09", "2026-11-10", "2026-11-11", "2026-11-12"]
    sessions += ["2026-11-13", "2026-11-16"]
    u = _universe()
    tickers = u["tickers"]
    alpha = np.array([0.08, 0.01, 0.01, 0.01, 0.0, -0.0001])
    w_prev = np.array([0.40, 0.20, 0.20, 0.17, 0.0, 0.03])
    locked: dict[str, dt.date] = {}
    held_weight: list[float] = []
    for day_text in sessions:
        day = dt.date.fromisoformat(day_text)
        holdings = {
            t: Holding(
                shares=1 if w_prev[i] > 0 else 0, avg_cost=100.0, mark=84.0 if i == 0 else 101.0
            )
            for i, t in enumerate(tickers[:4])
        }
        for ticker in loss_floor_breaches(holdings, params):
            locked[ticker] = lockout_through(day, params.hard_risk_reentry_lockout_sessions or 0)
        locked = {t: until for t, until in locked.items() if day <= until}
        hard_exit = np.array([t in locked for t in tickers])
        result = solve_target_weights(
            tickers,
            alpha,
            u["returns_panel"],
            w_prev,
            u["sectors"],
            u["stance_caps"],
            u["eligibility"],
            u["benchmark_idx"],
            u["cash_idx"],
            params,
            u["cost_model"],
            hard_exit=hard_exit,
        )
        w_prev = result.weights
        held_weight.append(float(w_prev[0]))
    assert held_weight[0] == pytest.approx(0.0, abs=1e-8)  # cut the day it breached
    assert held_weight[1:6] == pytest.approx([0.0] * 5, abs=1e-8)  # locked five sessions
    assert held_weight[6] > 0.0  # unlocked on the sixth session after the exit


# ── construct_book carries the session's hard-risk inputs ────────────────


def _book_fixtures():
    from tests.test_portfolio_contracts import FLAT, IMPACT, _recipe, _sessions, _universe

    return FLAT, IMPACT, _recipe, _sessions, _universe


def test_construct_book_pins_a_session_hard_exit_and_records_it() -> None:
    import dataclasses

    from crucible.slots.strategy import construct_book

    flat, _, recipe, sessions, universe = _book_fixtures()
    (session,) = sessions(days=("2026-08-24",))
    session = dataclasses.replace(
        session, hard_exit=np.array([True, False, False, False]), buy_multiplier=0.5
    )
    constructed = construct_book(
        recipe=recipe(flat),
        params=_v1(),
        universe=universe(),
        sessions=(session,),
        portfolio_notional=1_000_000.0,
        w_initial=np.array([0.10, 0.10, 0.77, 0.03]),
    )
    (weights,) = constructed.weights
    assert weights[0] == pytest.approx(0.0, abs=1e-8)
    (diagnostics,) = constructed.diagnostics
    assert diagnostics["hard_risk"]["hard_exit_held_exits"] == ["AAA"]
    assert diagnostics["hard_risk"]["buy_multiplier"] == 0.5


def test_construct_book_refuses_an_unpriceable_hard_exit_under_a_participation_model() -> None:
    import dataclasses

    from crucible.portfolio import CostModelInputError
    from crucible.slots.strategy import construct_book

    _, impact, recipe, sessions, universe = _book_fixtures()
    (session,) = sessions(days=("2026-08-24",))
    session = dataclasses.replace(
        session,
        hard_exit=np.array([True, False, False, False]),
        adv_usd=np.array([np.nan, 1.5e7, 1e10, 0.0]),
    )
    with pytest.raises(CostModelInputError, match="AAA"):
        construct_book(
            recipe=recipe(impact),
            params=_v1(),
            universe=universe(),
            sessions=(session,),
            portfolio_notional=1_000_000.0,
            w_initial=np.array([0.10, 0.10, 0.77, 0.03]),
        )
