"""One S-slot input resolution for the grade and the trader (`alpha-engine-config-I10654`).

Plan §10.6 row 2: "the S-slot grade and the trader call the same function on
the same inputs." `construct_book` is the function;
`crucible.slots.inputs.resolve_strategy_sessions` — which the grade itself
calls — resolves the inputs. The load-bearing assertion is
`TestTheResolverIsTheGradesResolution`: the resolver's inputs, walked through
`construct_book`, reproduce the `portfolio_construction` evidence a REAL
`experiment.grade` run put on its manifest.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import tests.test_slot_strategy_cycle_job as cycle_job
from crucible.documents import load_store_document
from crucible.keys import arm_predictions_key, data_panel_key, session_inputs_key
from crucible.portfolio import (
    CostModel,
    CostModelInputError,
    load_portfolio_params,
    manifest_records_portfolio_engine,
)
from crucible.slots.cycle import MissingArtifactError
from crucible.slots.inputs import (
    UNSETTLED_RETURN,
    ArmPredictionsContractError,
    resolve_strategy_sessions,
)
from crucible.slots.strategy import construct_book, load_strategy_slot, registration_specs
from tests.test_slot_strategy_cycle_job import AS_OF, _next_session, _run_grade, _run_produce

#: The S cycle job's own fixture — a real panel, real champions' feeds, a real
#: strategy tree — re-exported so this module constructs over the SAME world.
world = cycle_job.world


def _arm(settings):
    return registration_specs(load_strategy_slot(strategy_dir=settings.strategy_dir))[0]


def _params(settings):
    return load_portfolio_params(settings.strategy_dir / "slots" / "s.yaml")


def _cash(universe) -> np.ndarray:
    weights = np.zeros(len(universe.tickers))
    weights[universe.cash_idx] = 1.0
    return weights


def _decide(store, arm_id: str, benchmark: str, day: str):
    return resolve_strategy_sessions(
        store,
        arm_id=arm_id,
        benchmark=benchmark,
        decision_days=[day],
        as_of=day,
        unsettled_last=True,
    )


def _extend_panel_one_session(store) -> dt.date:
    """Compile the NEXT session's panel: every row of AS_OF's plus one new close."""
    panel = pd.read_parquet(io.BytesIO(store.get_bytes(data_panel_key(AS_OF.isoformat()))))
    nxt = _next_session(AS_OF)
    last = panel[panel["trading_day"] == panel["trading_day"].max()].copy()
    last["trading_day"] = nxt
    for column in ("open_raw", "high_raw", "low_raw", "close_raw"):
        last[column] = last[column] * 1.002
    extended = pd.concat([panel, last], ignore_index=True)
    store.put_bytes(data_panel_key(nxt.isoformat()), extended.to_parquet(index=False))
    return nxt


class TestTheResolverIsTheGradesResolution:
    def test_the_settled_walk_reproduces_the_grades_manifest_evidence(self, world) -> None:
        store, settings, _root, days = world
        _run_produce(store, settings, days)
        _result, manifest, _ctx = _run_grade(store, settings)
        graded = manifest_records_portfolio_engine(manifest)
        assert graded is not None

        arm = _arm(settings)
        settled_days = [d.isoformat() for d in days[:-1]]
        inputs = resolve_strategy_sessions(
            store,
            arm_id=arm.arm_id,
            benchmark=arm.recipe.benchmark,
            decision_days=settled_days,
            as_of=AS_OF.isoformat(),
        )
        params = _params(settings)
        constructed = construct_book(
            recipe=arm.recipe,
            params=params,
            universe=inputs.universe,
            sessions=inputs.sessions,
            portfolio_notional=params.book_notional_usd,
            w_initial=_cash(inputs.universe),
        )

        assert inputs.settled is True
        assert inputs.session_inputs_keys == tuple(
            session_inputs_key(arm.arm_id, d) for d in settled_days
        )
        assert constructed.evidence == graded, (
            "the resolver's inputs walked through construct_book did not reproduce the "
            "grade's own portfolio_construction record"
        )

    def test_a_decision_session_solves_the_weights_its_settled_twin_solves(self, world) -> None:
        """The trader decides AS_OF over AS_OF's panel; the grade later walks the
        same day over the next session's panel. Same weights, same charge."""
        store, settings, _root, days = world
        _run_produce(store, settings, days)
        nxt = _extend_panel_one_session(store)
        arm = _arm(settings)
        params = _params(settings)
        day = AS_OF.isoformat()

        decision = _decide(store, arm.arm_id, arm.recipe.benchmark, day)
        settled = resolve_strategy_sessions(
            store,
            arm_id=arm.arm_id,
            benchmark=arm.recipe.benchmark,
            decision_days=[day],
            as_of=nxt.isoformat(),
        )
        assert decision.universe == settled.universe
        (d_session,) = decision.sessions
        (s_session,) = settled.sessions
        for field in ("alpha_hat", "eligibility", "stance_caps", "returns_panel", "adv_usd"):
            np.testing.assert_array_equal(getattr(d_session, field), getattr(s_session, field))
        assert decision.settled is False
        assert np.isnan(d_session.realized_returns).all()
        assert math.isnan(d_session.benchmark_return) and math.isnan(UNSETTLED_RETURN)

        def build(inputs):
            return construct_book(
                recipe=arm.recipe,
                params=params,
                universe=inputs.universe,
                sessions=inputs.sessions,
                portfolio_notional=params.book_notional_usd,
                w_initial=_cash(inputs.universe),
            )

        d_book, s_book = build(decision), build(settled)
        assert d_book.weights == s_book.weights
        assert d_book.book.cost_bps == s_book.book.cost_bps
        assert d_book.book.turnover == s_book.book.turnover


class TestTheResolverRefusesRatherThanGuesses:
    def _produced(self, world):
        store, settings, _root, days = world
        _run_produce(store, settings, days)
        return store, _arm(settings), days

    def test_an_unrecorded_session_is_refused(self, world) -> None:
        store, settings, _root, _days = world
        arm = _arm(settings)
        with pytest.raises(MissingArtifactError, match="recorded no construction inputs"):
            _decide(store, arm.arm_id, "SPY", AS_OF.isoformat())

    def test_an_unsettled_day_never_enters_a_settled_walk(self, world) -> None:
        store, arm, days = self._produced(world)
        with pytest.raises(MissingArtifactError, match="unsettled"):
            resolve_strategy_sessions(
                store,
                arm_id=arm.arm_id,
                benchmark="SPY",
                decision_days=[d.isoformat() for d in days],
                as_of=AS_OF.isoformat(),
            )

    def test_a_decision_on_a_day_its_panel_has_passed_is_refused(self, world) -> None:
        store, arm, days = self._produced(world)
        earlier = days[-2].isoformat()
        store.put_bytes(data_panel_key(earlier), store.get_bytes(data_panel_key(AS_OF.isoformat())))
        with pytest.raises(MissingArtifactError, match="must be the last session"):
            _decide(store, arm.arm_id, "SPY", earlier)

    def test_a_benchmark_the_panel_does_not_price_is_refused(self, world) -> None:
        store, arm, _days = self._produced(world)
        with pytest.raises(MissingArtifactError, match="no proxy is substituted"):
            _decide(store, arm.arm_id, "QQQ", AS_OF.isoformat())

    @pytest.mark.parametrize("decision_days", [[], ["2026-08-27", "2026-08-26"]])
    def test_an_empty_or_unordered_walk_is_refused(self, world, decision_days) -> None:
        store, arm, _days = self._produced(world)
        with pytest.raises(ValueError, match="no decision days|unique and ascending"):
            resolve_strategy_sessions(
                store,
                arm_id=arm.arm_id,
                benchmark="SPY",
                decision_days=decision_days,
                as_of=AS_OF.isoformat(),
            )

    def test_a_session_document_filed_under_another_day_is_refused(self, world) -> None:
        store, arm, days = self._produced(world)
        day, other = AS_OF.isoformat(), days[-2].isoformat()
        document = load_store_document(store, session_inputs_key(arm.arm_id, other))
        store.put_bytes(session_inputs_key(arm.arm_id, day), json.dumps(document).encode("utf-8"))
        with pytest.raises(ArmPredictionsContractError, match="filed under another day"):
            _decide(store, arm.arm_id, "SPY", day)


# ---------------------------------------------------------------------------
# alpha-engine-config-I10754: a held name leaving the M cross-section.
# ---------------------------------------------------------------------------


def _write_panel_through(store, day: dt.date) -> None:
    """The panel compiled FOR ``day``: every AS_OF row on or before it."""
    panel = pd.read_parquet(io.BytesIO(store.get_bytes(data_panel_key(AS_OF.isoformat()))))
    through = panel[pd.to_datetime(panel["trading_day"]).dt.date <= day]
    store.put_bytes(data_panel_key(day.isoformat()), through.to_parquet(index=False))


def _drop_from_m_predictions(store, day: dt.date, ticker: str) -> None:
    key = arm_predictions_key(cycle_job.M_CHAMPION, day.isoformat())
    document = load_store_document(store, key)
    del document["predicted_alpha"][ticker]
    store.put_bytes(key, json.dumps(document, indent=2, sort_keys=True).encode("utf-8"))


def _weights_vector(universe, weights: dict[str, float]) -> np.ndarray:
    return np.array([float(weights.get(t, 0.0)) for t in universe.tickers], dtype=np.float64)


def _held(universe, weights) -> list[str]:
    sentinels = {universe.tickers[universe.benchmark_idx], universe.tickers[universe.cash_idx]}
    return [
        t for t, w in zip(universe.tickers, weights, strict=True) if w != 0.0 and t not in sentinels
    ]


def _impact_recipe(recipe):
    return replace(
        recipe,
        cost_model=CostModel(
            name="sqrt_impact_v1",
            placeholder=False,
            params={
                "half_spread_bps": 2.5,
                "commission_bps": 0.5,
                "impact_coef_bps": 12.0,
                "min_cost_bps": 1.0,
            },
        ),
    )


class TestAHeldNameLeavingTheUniverseIsExitedNotRefused:
    """The trader decides two consecutive days from its own carried book; the
    grade walks the same two days from cash. On the second day the M champion
    stops pricing the name the first day's book held most of."""

    def _two_days(self, world):
        store, settings, _root, days = world
        arm = _arm(settings)
        params = _params(settings)
        d_prev = days[-2]
        _run_produce(store, settings, days[:-1])
        _write_panel_through(store, d_prev)

        day1 = _decide(store, arm.arm_id, arm.recipe.benchmark, d_prev.isoformat())
        (w1,) = construct_book(
            recipe=arm.recipe,
            params=params,
            universe=day1.universe,
            sessions=day1.sessions,
            portfolio_notional=params.book_notional_usd,
            w_initial=_cash(day1.universe),
        ).weights
        held = _held(day1.universe, w1)
        assert held, "the fixture's first book holds no name; nothing can leave the universe"
        dropped = max(held, key=lambda t: w1[day1.universe.tickers.index(t)])
        _drop_from_m_predictions(store, AS_OF, dropped)
        _run_produce(store, settings, [AS_OF])
        previous = dict(zip(day1.universe.tickers, w1, strict=True))
        return store, arm, params, d_prev, previous, held, dropped

    def test_without_the_held_names_the_dropped_name_is_not_in_the_universe(self, world) -> None:
        store, arm, _params_, _d_prev, _previous, _held_names, dropped = self._two_days(world)
        decision = _decide(store, arm.arm_id, arm.recipe.benchmark, AS_OF.isoformat())
        assert dropped not in decision.universe.tickers

    def test_the_trader_and_the_settled_grade_walk_construct_the_same_exit(self, world) -> None:
        store, arm, params, d_prev, previous, held, dropped = self._two_days(world)
        nxt = _extend_panel_one_session(store)

        trader = resolve_strategy_sessions(
            store,
            arm_id=arm.arm_id,
            benchmark=arm.recipe.benchmark,
            decision_days=[AS_OF.isoformat()],
            as_of=AS_OF.isoformat(),
            unsettled_last=True,
            held_tickers=held,
        )
        grade_walk = resolve_strategy_sessions(
            store,
            arm_id=arm.arm_id,
            benchmark=arm.recipe.benchmark,
            decision_days=[d_prev.isoformat(), AS_OF.isoformat()],
            as_of=nxt.isoformat(),
        )
        assert trader.universe == grade_walk.universe
        assert trader.held_tickers == tuple(sorted(held))
        i = trader.universe.tickers.index(dropped)
        (t_session,) = trader.sessions
        assert not t_session.eligibility[i] and t_session.alpha_hat[i] == 0.0
        for field in ("alpha_hat", "eligibility", "stance_caps", "returns_panel", "adv_usd"):
            np.testing.assert_array_equal(
                getattr(t_session, field), getattr(grade_walk.sessions[1], field)
            )

        traded = construct_book(
            recipe=arm.recipe,
            params=params,
            universe=trader.universe,
            sessions=trader.sessions,
            portfolio_notional=params.book_notional_usd,
            w_initial=_weights_vector(trader.universe, previous),
        )
        graded = construct_book(
            recipe=arm.recipe,
            params=params,
            universe=grade_walk.universe,
            sessions=grade_walk.sessions,
            portfolio_notional=params.book_notional_usd,
            w_initial=_cash(grade_walk.universe),
        )
        assert graded.weights[0] == tuple(previous[t] for t in grade_walk.universe.tickers)
        assert traded.weights[0] == graded.weights[1]
        assert traded.book.cost_bps[0] == graded.book.cost_bps[1]
        assert traded.book.turnover[0] == graded.book.turnover[1]
        assert traded.weights[0][i] < previous[dropped], "the dropped name was not exited"
        assert traded.book.cost_bps[0] > 0.0

    def test_a_participation_aware_exit_is_priced_from_the_days_adv(self, world) -> None:
        store, arm, params, _d_prev, previous, held, dropped = self._two_days(world)
        population = [t for t in previous if t not in {"SPY", "__CASH__"}]
        cycle_job._write_features(store, days=[AS_OF], tickers=population, dollar_volume_usd=4e8)
        trader = resolve_strategy_sessions(
            store,
            arm_id=arm.arm_id,
            benchmark=arm.recipe.benchmark,
            decision_days=[AS_OF.isoformat()],
            as_of=AS_OF.isoformat(),
            unsettled_last=True,
            held_tickers=held,
        )
        i = trader.universe.tickers.index(dropped)
        assert trader.sessions[0].adv_usd[i] == 4e8
        book = construct_book(
            recipe=_impact_recipe(arm.recipe),
            params=params,
            universe=trader.universe,
            sessions=trader.sessions,
            portfolio_notional=params.book_notional_usd,
            w_initial=_weights_vector(trader.universe, previous),
        )
        assert book.book.cost_bps[0] > 0.0

    def test_a_participation_aware_exit_with_no_adv_raises(self, world) -> None:
        store, arm, params, _d_prev, previous, held, dropped = self._two_days(world)
        population = [t for t in previous if t not in {"SPY", "__CASH__", dropped}]
        cycle_job._write_features(store, days=[AS_OF], tickers=population)
        trader = resolve_strategy_sessions(
            store,
            arm_id=arm.arm_id,
            benchmark=arm.recipe.benchmark,
            decision_days=[AS_OF.isoformat()],
            as_of=AS_OF.isoformat(),
            unsettled_last=True,
            held_tickers=held,
        )
        with pytest.raises(CostModelInputError, match=dropped):
            construct_book(
                recipe=_impact_recipe(arm.recipe),
                params=params,
                universe=trader.universe,
                sessions=trader.sessions,
                portfolio_notional=params.book_notional_usd,
                w_initial=_weights_vector(trader.universe, previous),
            )

    def test_a_held_name_with_no_price_at_all_raises(self, world) -> None:
        store, arm, _params_, _d_prev, _previous, held, _dropped = self._two_days(world)
        with pytest.raises(MissingArtifactError, match="no proxy is substituted"):
            resolve_strategy_sessions(
                store,
                arm_id=arm.arm_id,
                benchmark=arm.recipe.benchmark,
                decision_days=[AS_OF.isoformat()],
                as_of=AS_OF.isoformat(),
                unsettled_last=True,
                held_tickers=[*held, "NOPRICE"],
            )
