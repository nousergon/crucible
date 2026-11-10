"""The conviction gate is fed the M champion's own σ_α̂ (`alpha-engine-config-I12024`).

`conviction_budget_gate_enabled` was on in the S slot's parameters while the
session inputs carried no predictive std, so every solve returned
``no_alpha_uncertainty_vector`` and the full discretionary budget — a gate
switched on and inert. These tests pin the wiring end to end: the M feed's
`predicted_alpha_std` (and its epistemic component) reach the recorded
`session_inputs.v1` document, the resolver hands them to `SessionInputs`, and
the solve's diagnostics show the gate evaluated.

The replay class reproduces the 2026-11-06 / 2026-11-09 cross-sections the
v1→v2 optimizer parity audit measured: a 60-name U cut whose demeaned alpha
spread is ~1.1-1.2 points against a model std of 11.1 points on every name.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

import tests.test_slot_strategy_cycle_job as cycle_job
from crucible.documents import load_store_document
from crucible.keys import arm_predictions_key, session_inputs_key
from crucible.portfolio import (
    PortfolioParams,
    compute_conviction_budget_multiplier,
    load_portfolio_params,
)
from crucible.slots.inputs import read_arm_predictions_with_uncertainty, resolve_strategy_sessions
from crucible.slots.strategy import (
    SESSION_INPUTS_SCHEMA_VERSION,
    ResolvedSession,
    construct_book,
    load_strategy_slot,
    re_resolve_session,
    registration_specs,
)
from tests.test_slot_strategy_cycle_job import M_CHAMPION, _run_grade, _run_produce

world = cycle_job.world

#: The M champion's measured total std on 2026-11-06 and 2026-11-09 (every name).
MODEL_STD = 0.1114


def _add_std(store, days, *, std: float = MODEL_STD, method: str = "bayesian_posterior_predictive"):
    """Give the fixture's M feed the std fields the live feed carries."""
    for day in days:
        key = arm_predictions_key(M_CHAMPION, day.isoformat())
        document = load_store_document(store, key)
        names = sorted(document["predicted_alpha"])
        document["predicted_alpha_std_method"] = method
        document["predicted_alpha_std_note"] = "fixture posterior predictive"
        if method != "omitted":
            document["predicted_alpha_std"] = dict.fromkeys(names, std)
            document["predicted_alpha_std_aleatoric"] = dict.fromkeys(names, std)
            document["predicted_alpha_std_epistemic"] = {
                t: 1e-4 * (i + 1) for i, t in enumerate(names)
            }
        store.put_bytes(key, json.dumps(document, indent=2, sort_keys=True).encode("utf-8"))


def _arm(settings):
    return registration_specs(load_strategy_slot(strategy_dir=settings.strategy_dir))[0]


def _params(settings) -> PortfolioParams:
    return load_portfolio_params(settings.strategy_dir / "slots" / "s.yaml")


def _decide(store, settings, day: str):
    arm = _arm(settings)
    params = _params(settings)
    inputs = resolve_strategy_sessions(
        store,
        arm_id=arm.arm_id,
        benchmark=arm.recipe.benchmark,
        decision_days=[day],
        as_of=day,
        unsettled_last=True,
    )
    w0 = np.zeros(len(inputs.universe.tickers))
    w0[inputs.universe.cash_idx] = 1.0
    constructed = construct_book(
        recipe=arm.recipe,
        params=params,
        universe=inputs.universe,
        sessions=inputs.sessions,
        portfolio_notional=params.book_notional_usd,
        w_initial=w0,
    )
    return inputs, constructed


class TestTheReaderReturnsTheFeedsStd:
    def test_alpha_and_std_come_from_the_same_document(self, world) -> None:
        store, _settings, _root, days = world
        _add_std(store, days[:1])
        alpha, uncertainty = read_arm_predictions_with_uncertainty(
            store, arm_id=M_CHAMPION, trading_day=days[0].isoformat()
        )
        assert uncertainty is not None
        assert uncertainty.method == "bayesian_posterior_predictive"
        assert uncertainty.std is not None and set(uncertainty.std) == set(alpha)
        assert uncertainty.epistemic is not None and set(uncertainty.epistemic) == set(alpha)

    def test_a_document_predating_the_field_returns_none(self, world) -> None:
        store, _settings, _root, days = world
        _alpha, uncertainty = read_arm_predictions_with_uncertainty(
            store, arm_id=M_CHAMPION, trading_day=days[0].isoformat()
        )
        assert uncertainty is None


class TestTheSessionDocumentCarriesTheStd:
    def test_produce_records_the_std_aligned_to_the_tickers(self, world) -> None:
        store, settings, _root, days = world
        _add_std(store, days)
        _run_produce(store, settings, days)
        arm = _arm(settings)
        document = load_store_document(store, session_inputs_key(arm.arm_id, days[0].isoformat()))

        assert document["schema_version"] == SESSION_INPUTS_SCHEMA_VERSION == "session_inputs.v1"
        assert len(document["alpha_uncertainty"]) == len(document["tickers"])
        assert set(document["alpha_uncertainty"]) == {MODEL_STD}
        assert len(document["alpha_uncertainty_epistemic"]) == len(document["tickers"])
        assert document["alpha_uncertainty_source"].startswith(
            arm_predictions_key(M_CHAMPION, days[0].isoformat())
        )

    def test_a_feed_without_std_records_the_absence_as_a_sentence(self, world) -> None:
        store, settings, _root, days = world
        _run_produce(store, settings, days)
        document = load_store_document(
            store, session_inputs_key(_arm(settings).arm_id, days[0].isoformat())
        )
        assert document["alpha_uncertainty"] is None
        assert document["alpha_uncertainty_source"].startswith("absent — ")

    def test_an_omitted_std_is_recorded_as_absent_with_the_feeds_reason(self, world) -> None:
        store, settings, _root, days = world
        _add_std(store, days, method="omitted")
        _run_produce(store, settings, days)
        document = load_store_document(
            store, session_inputs_key(_arm(settings).arm_id, days[0].isoformat())
        )
        assert document["alpha_uncertainty"] is None
        assert "`omitted`" in document["alpha_uncertainty_source"]

    def test_a_document_recorded_before_the_fields_still_reads(self) -> None:
        """Every document already in the store stays valid as written."""
        legacy = {
            "schema_version": "session_inputs.v1",
            "trading_day": "2026-11-06",
            "tickers": ["AAA", "BBB"],
            "alpha_hat": [0.01, 0.02],
            "eligibility": [True, True],
            "stance_caps": [1.0, 1.0],
            "alpha_source": "arm_predictions/m~x~1/2026-11-06.json",
            "eligibility_source": "universe/2026-11-06/members.json",
            "sectors_source": "absent",
            "stance_caps_source": "uniform",
        }
        session = ResolvedSession.from_dict(legacy)
        assert session.alpha_uncertainty is None
        assert session.alpha_uncertainty_epistemic is None
        assert session.alpha_uncertainty_source is None

    def test_the_document_round_trips(self, world) -> None:
        store, settings, _root, days = world
        _add_std(store, days)
        _run_produce(store, settings, days)
        document = load_store_document(
            store, session_inputs_key(_arm(settings).arm_id, days[0].isoformat())
        )
        assert ResolvedSession.from_dict(document).to_dict() == document


class TestTheSolveEvaluatesTheGate:
    def test_the_std_reaches_session_inputs_on_the_universe_index(self, world) -> None:
        store, settings, _root, days = world
        _add_std(store, days)
        _run_produce(store, settings, days)
        inputs, _constructed = _decide(store, settings, days[-1].isoformat())
        (session,) = inputs.sessions
        universe = inputs.universe
        assert session.alpha_uncertainty is not None
        assert session.alpha_uncertainty_epistemic is not None
        assert np.isnan(session.alpha_uncertainty[universe.cash_idx])
        assert np.isnan(session.alpha_uncertainty[universe.benchmark_idx])
        real = [
            i
            for i in range(len(universe.tickers))
            if i not in (universe.cash_idx, universe.benchmark_idx)
        ]
        assert np.allclose(session.alpha_uncertainty[real], MODEL_STD)

    def test_the_gate_is_evaluated_and_throttles_a_tied_cross_section(self, world) -> None:
        store, settings, _root, days = world
        _add_std(store, days)
        _run_produce(store, settings, days)
        _inputs, constructed = _decide(store, settings, days[-1].isoformat())
        (diagnostics,) = constructed.diagnostics

        # The fixture's alpha spread is ~0.4 points against an 11.1-point std.
        assert diagnostics["conviction_gate_reason"] == "alpha_spread_below_own_noise"
        assert diagnostics["conviction_gate_applied"] is True
        assert diagnostics["conviction_ir_xs"] < 0.35
        assert diagnostics["conviction_budget_multiplier"] == pytest.approx(0.05)
        assert diagnostics["turnover_budget_discretionary"] == pytest.approx(0.15 * 0.05)

    def test_without_std_the_gate_says_so_and_leaves_the_budget_open(self, world) -> None:
        store, settings, _root, days = world
        _run_produce(store, settings, days)
        _inputs, constructed = _decide(store, settings, days[-1].isoformat())
        (diagnostics,) = constructed.diagnostics
        assert diagnostics["conviction_gate_reason"] == "no_alpha_uncertainty_vector"
        assert diagnostics["conviction_budget_multiplier"] == 1.0

    def test_the_grade_attests_clean_with_the_std_recorded(self, world) -> None:
        """Both attestation passes read the std, so its arrival is not contamination."""
        store, settings, _root, days = world
        _add_std(store, days)
        _run_produce(store, settings, days)
        result, manifest, _ctx = _run_grade(store, settings)
        assert manifest["status"] == "ok", manifest.get("reason")
        for arm_id, graded in result["strategy_grades"].items():
            assert graded["attestation"] == "PASS", (arm_id, graded)
            assert graded["attestation_mean_delta"] == 0.0


class TestTheAttestationRereadsWhatWasRecorded:
    def test_a_std_recorded_is_re_read(self, world) -> None:
        store, settings, _root, days = world
        _add_std(store, days)
        _run_produce(store, settings, days)
        recorded = ResolvedSession.from_dict(
            load_store_document(
                store, session_inputs_key(_arm(settings).arm_id, days[0].isoformat())
            )
        )
        assert re_resolve_session(store, recorded=recorded) == recorded

    def test_a_legacy_session_is_re_read_without_std_even_once_the_feed_has_it(self, world) -> None:
        """A session recorded before the fields existed was decided with no std;
        re-reading it with one would measure the gate's arrival as a revision."""
        store, settings, _root, days = world
        _run_produce(store, settings, days)
        recorded = ResolvedSession.from_dict(
            load_store_document(
                store, session_inputs_key(_arm(settings).arm_id, days[0].isoformat())
            )
        )
        legacy = dataclasses.replace(
            recorded,
            alpha_uncertainty=None,
            alpha_uncertainty_epistemic=None,
            alpha_uncertainty_source=None,
        )
        _add_std(store, days)
        assert re_resolve_session(store, recorded=legacy) == legacy


class TestReplayOfTheAuditedSessions:
    """The parity audit's row 7, on the engine's own gate function.

    A 60-name eligible cut with a demeaned alpha spread of ~1.1-1.2 points and
    a model std of 11.1 points on every name: IR ≈ 0.10, below the 0.35 floor,
    so the multiplier sits at its 0.05 floor and a 20% budget becomes 1%/day.
    """

    @staticmethod
    def _params(**overrides) -> PortfolioParams:
        payload = {
            "book_notional_usd": 1_000_000.0,
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
            "max_pct_adv": 0.05,
            "max_daily_turnover": 0.20,
            "large_move_turnover_flag": 0.35,
            "conviction_budget_gate_enabled": True,
            "conviction_ir_floor": 0.35,
            "conviction_ir_full": 0.75,
            "conviction_budget_min_multiple": 0.05,
            "conviction_gate_min_names": 3,
            "max_position_pct": 0.25,
            "ineligible_held_exit": "discretionary",
        }
        payload.update(overrides)
        return PortfolioParams.from_mapping(payload, source="audit replay")

    @pytest.mark.parametrize(("day", "spread"), [("2026-11-06", 0.0117), ("2026-11-09", 0.0111)])
    def test_the_audited_cross_section_cuts_the_budget_to_one_percent(self, day, spread) -> None:
        rng = np.random.default_rng(int(day.replace("-", "")))
        n = 60
        raw = rng.normal(0.0, 1.0, n)
        alpha = np.concatenate([(raw - raw.mean()) / raw.std() * spread, [0.0, -1e-6]])
        sigma = np.concatenate([np.full(n, 0.1114), [np.nan, np.nan]])
        eligible = np.ones(n + 2, dtype=bool)
        params = self._params()

        gate = compute_conviction_budget_multiplier(alpha, sigma, eligible, n, n + 1, params)

        assert gate["conviction_ir_xs"] == pytest.approx(spread / 0.1114, rel=1e-6)
        assert 0.09 < gate["conviction_ir_xs"] < 0.11
        assert gate["conviction_budget_multiplier"] == pytest.approx(0.05)
        assert params.max_daily_turnover is not None
        budget = params.max_daily_turnover * gate["conviction_budget_multiplier"]
        assert budget == pytest.approx(0.01)
