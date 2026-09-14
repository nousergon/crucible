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

import numpy as np
import pandas as pd
import pytest

import tests.test_slot_strategy_cycle_job as cycle_job
from crucible.documents import load_store_document
from crucible.keys import data_panel_key, session_inputs_key
from crucible.portfolio import load_portfolio_params, manifest_records_portfolio_engine
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
