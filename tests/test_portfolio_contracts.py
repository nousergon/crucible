"""The portfolio boundary: a named cost model, required parameters, and evidence.

Normative sources: `alpha-engine-config-I10500` (the engine is used by S-slot
grading and the call reaches a manifest), `-I10503` (the model in force is
recorded by name and parameters, and a fallback is impossible), plan §10.2.

The defect these exist to make unreachable, in one sentence: an optimizer ran
on a fallback cost model and nothing said so. Every test here is either a
refusal — the fallback cannot be reached — or a recording — what was in force
is on the artifact.

The three halves of that, and where each is tested:

* **No model can be unnamed.** :data:`COST_MODELS` is a closed registry, and a
  model is named there or the recipe does not load.
* **No named model can be partly declared.** Its parameters are its registry
  entry's, all of them and no others — so the library constructor that fills
  absent keys from institutional defaults is never reachable.
* **No run that applies costs can omit what charged it.** The evidence carries
  the model, the parameters, and whether it is a stand-in.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from crucible.portfolio import (
    COST_MODEL_KINDS,
    COST_MODELS,
    PORTFOLIO_EVIDENCE_SCHEMA_VERSION,
    PORTFOLIO_METRIC_NAME,
    PORTFOLIO_PARAM_FIELDS,
    CostModel,
    CostModelError,
    CostModelInputError,
    PortfolioParams,
    PortfolioParamsError,
    cost_model_from_mapping,
    load_portfolio_params,
    manifest_records_portfolio_engine,
    params_digest,
    portfolio_evidence,
    portfolio_metric_record,
)
from crucible.slots.strategy import (
    Book,
    BookUniverse,
    ExitRuleSpec,
    SessionInputs,
    StrategyRecipe,
    construct_book,
    grade_arm,
)

FIXTURE_PARAMS: dict = {
    "risk_aversion": 5.0,
    "cash_sleeve_pct": 0.03,
    "max_sector_pct": 0.25,
    "min_position_pct": 0.005,
    "covariance_shrinkage": "sample",
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
}

FLAT = {
    "name": "flat_bps_v0",
    "placeholder": True,
    "params": {"half_spread_bps": 2.5, "commission_bps": 0.5, "slippage_bps": 10.0},
}
IMPACT = {
    "name": "sqrt_impact_v1",
    "placeholder": False,
    "params": {
        "half_spread_bps": 2.5,
        "impact_coef_bps": 10.0,
        "commission_bps": 0.5,
        "min_cost_bps": 0.0,
    },
}


def _params(**overrides) -> PortfolioParams:
    return PortfolioParams.from_mapping({**FIXTURE_PARAMS, **overrides}, source="<fixture>")


def _recipe(cost: dict) -> StrategyRecipe:
    return StrategyRecipe(
        name="contract_fixture",
        rules=(ExitRuleSpec(rule_id="profit_take", params={"profit_take_pct": 0.25}),),
        cost_model=CostModel(**cost),
    )


def _universe() -> BookUniverse:
    return BookUniverse(
        tickers=("AAA", "BBB", "SPY", "CASH"),
        sectors=("tech", "health", "__benchmark__", "__cash__"),
        benchmark_idx=2,
        cash_idx=3,
    )


def _sessions(days=("2026-08-24", "2026-08-25"), *, with_adv: bool = True):
    rng = np.random.default_rng(11)
    panel = rng.normal(0.0, 0.01, size=(260, 4))
    panel[:, 3] = 0.0
    out = []
    for i, day in enumerate(days):
        out.append(
            SessionInputs(
                trading_day=day,
                alpha_hat=np.array([0.02, 0.01, 0.0, -1e-6]) * (1 + 0.05 * i),
                eligibility=np.ones(4, dtype=bool),
                stance_caps=np.array([0.12, 0.12, 1.0, 1.0]),
                realized_returns=np.array([0.003, -0.001, 0.001, 0.0]),
                benchmark_return=0.001,
                returns_panel=panel,
                adv_usd=np.array([4e7, 1.5e7, 1e10, 0.0]) if with_adv else None,
            )
        )
    return tuple(out)


class TestTheCostModelRegistryIsClosed:
    """A model is in `COST_MODELS` or it does not exist. There is no other door."""

    def test_an_unregistered_model_is_refused_by_name(self) -> None:
        with pytest.raises(CostModelError, match="unknown cost model 'made_up_v9'"):
            CostModel(name="made_up_v9", placeholder=False, params={"a": 1.0})

    def test_an_anonymous_model_is_refused(self) -> None:
        with pytest.raises(CostModelError, match="must be named"):
            CostModel(name="", placeholder=False, params={})

    def test_every_registered_model_declares_a_kind(self) -> None:
        """A name without a kind would leave the grader dispatching on a string
        it has no case for — which is a silent no-op, not an error."""
        assert set(COST_MODELS) == set(COST_MODEL_KINDS)
        assert set(COST_MODEL_KINDS.values()) <= {"flat", "sqrt_impact"}

    @pytest.mark.parametrize("dropped", sorted(COST_MODELS["sqrt_impact_v1"]))
    def test_a_partly_declared_model_is_refused_field_by_field(self, dropped: str) -> None:
        """Each parameter, individually, is load-bearing.

        This is the test that makes the library's defaulting constructor
        unreachable: every field it would have filled in silently must be
        present here, and the refusal names which one is missing.
        """
        params = {k: v for k, v in IMPACT["params"].items() if k != dropped}
        with pytest.raises(CostModelError, match=dropped):
            CostModel(name="sqrt_impact_v1", placeholder=False, params=params)

    def test_an_unread_parameter_is_refused(self) -> None:
        """A constant in a recipe that nothing reads is a value someone believes
        is priced and is not."""
        with pytest.raises(CostModelError, match="spread_multiplier"):
            CostModel(
                name="sqrt_impact_v1",
                placeholder=False,
                params={**IMPACT["params"], "spread_multiplier": 1.5},
            )

    def test_the_placeholder_flag_is_required_not_inferred(self) -> None:
        with pytest.raises(CostModelError, match="'placeholder'"):
            cost_model_from_mapping(
                {"name": "flat_bps_v0", "params": FLAT["params"]}, source="<fixture>"
            )


class TestTheModelRecordedIsTheModelInForce:
    """The producer/consumer contract at the cost boundary.

    Recording the name is worth nothing if the object doing the charging was
    built from something else. These assert the identity, not the resemblance.
    """

    def test_the_library_model_is_built_from_the_declared_parameters(self) -> None:
        model = CostModel(**IMPACT)
        built = model.impact_model()
        recorded = model.record()["params"]
        assert built.half_spread_bps == recorded["half_spread_bps"]
        assert built.impact_coef_bps == recorded["impact_coef_bps"]
        assert built.commission_bps == recorded["commission_bps"]
        assert built.min_cost_bps == recorded["min_cost_bps"]

    def test_the_library_default_is_never_silently_substituted(self) -> None:
        """The library's own `from_config` fills absent keys from institutional
        defaults. A model declaring a DIFFERENT value must therefore come back
        carrying its own, or the defaulting path is live somewhere.
        """
        from nousergon_lib.quant.transaction_cost import TransactionCostModel

        library_default = TransactionCostModel.from_config({})
        declared = CostModel(
            name="sqrt_impact_v1",
            placeholder=False,
            params={**IMPACT["params"], "impact_coef_bps": library_default.impact_coef_bps + 7.0},
        )
        assert declared.impact_model().impact_coef_bps != library_default.impact_coef_bps

    def test_a_flat_model_has_no_participation_rate_and_says_so(self) -> None:
        with pytest.raises(CostModelInputError, match="prices participation"):
            CostModel(**IMPACT).bps_per_unit_turnover()

    def test_a_participation_model_prices_a_thin_name_above_a_liquid_one(self) -> None:
        model = CostModel(**IMPACT)
        deltas = np.array([0.01, 0.0, 0.0, 0.0])
        liquid = model.cost_bps_for_trades(
            weight_deltas=deltas,
            adv_usd=np.array([1e9, 1e9, 1e9, 1e9]),
            portfolio_notional=4_000_000.0,
        )
        thin = model.cost_bps_for_trades(
            weight_deltas=deltas,
            adv_usd=np.array([1e6, 1e9, 1e9, 1e9]),
            portfolio_notional=4_000_000.0,
        )
        assert thin > liquid


class TestTheParametersAreRequiredAndPrivate:
    """`alpha-engine-config-I10500` deliverable 3: load loud, never default."""

    def test_an_absent_parameter_file_raises_rather_than_defaulting(self, tmp_path) -> None:
        with pytest.raises(PortfolioParamsError, match="no portfolio parameter set"):
            load_portfolio_params(tmp_path / "s.yaml")

    def test_a_file_without_a_portfolio_block_raises(self, tmp_path) -> None:
        path = tmp_path / "s.yaml"
        path.write_text("something_else: 1\n", encoding="utf-8")
        with pytest.raises(PortfolioParamsError, match="carrying a 'portfolio' block"):
            load_portfolio_params(path)

    @pytest.mark.parametrize("dropped", sorted(PORTFOLIO_PARAM_FIELDS))
    def test_every_field_is_required_individually(self, dropped: str) -> None:
        payload = {k: v for k, v in FIXTURE_PARAMS.items() if k != dropped}
        with pytest.raises(PortfolioParamsError, match=dropped):
            PortfolioParams.from_mapping(payload, source="<fixture>")

    def test_an_unknown_field_is_refused_by_name(self) -> None:
        with pytest.raises(PortfolioParamsError, match="leverage_cap"):
            PortfolioParams.from_mapping(
                {**FIXTURE_PARAMS, "leverage_cap": 1.5}, source="<fixture>"
            )

    def test_a_null_on_a_non_nullable_field_is_refused(self) -> None:
        """`null` means DISABLED, and a field with no disabled state must not
        accept it — a risk aversion of None would otherwise read as a knob
        someone deliberately turned off."""
        with pytest.raises(PortfolioParamsError, match="risk_aversion"):
            PortfolioParams.from_mapping(
                {**FIXTURE_PARAMS, "risk_aversion": None}, source="<fixture>"
            )

    def test_a_string_does_not_pass_for_a_boolean(self) -> None:
        with pytest.raises(PortfolioParamsError, match="conviction_budget_gate_enabled"):
            PortfolioParams.from_mapping(
                {**FIXTURE_PARAMS, "conviction_budget_gate_enabled": "yes"},
                source="<fixture>",
            )

    def test_an_estimator_that_is_not_carried_is_refused_by_name(self) -> None:
        """Not resolved to the default. A parameter set asking for an estimator
        it does not get is a different strategy from the one it declares."""
        with pytest.raises(PortfolioParamsError, match="Use 'ledoit_wolf'"):
            _params(covariance_shrinkage="oas")

    def test_this_repository_declares_no_value_for_any_field(self) -> None:
        """`repository-tiering-policy` test 2. The field REGISTRY is public; the
        values are strategy edge. A default on the dataclass would be a tuned
        constant committed to a public repository, which is how tier 3 leaks.
        """
        import dataclasses

        for field in dataclasses.fields(PortfolioParams):
            assert field.default is dataclasses.MISSING, (
                f"{field.name} carries a default. A parameter the harness can supply "
                "on the operator's behalf is a book traded on a number nobody chose."
            )
            assert field.default_factory is dataclasses.MISSING, field.name

    def test_the_digest_is_stable_and_the_values_travel_beside_it(self) -> None:
        """`alpha-engine-config-I10498`'s failure mode, designed against: a
        record carrying only a content address is unreadable the moment the
        addressed content moves, and nothing about that is loud. The digest is
        an equality test; the values are on the record.
        """
        first = _params()
        assert params_digest(first) == params_digest(_params())
        assert params_digest(_params(risk_aversion=6.0)) != params_digest(first)
        evidence = _evidence()
        assert evidence["params"] == first.to_dict()
        assert evidence["params_digest"] == params_digest(first)


def _evidence(cost: dict | None = None) -> dict:
    constructed = construct_book(
        recipe=_recipe(cost or IMPACT),
        params=_params(),
        universe=_universe(),
        sessions=_sessions(),
        portfolio_notional=4_000_000.0,
        w_initial=np.array([0.0, 0.0, 0.97, 0.03]),
    )
    return constructed.evidence


class TestTheGradingPathUsesTheEngineAndSaysSo:
    """`alpha-engine-config-I10500` deliverable 1. The module existing is not the
    deliverable; the grading path using it, and the call being legible in the
    evidence, is."""

    def test_construct_book_produces_evidence_naming_the_engine(self) -> None:
        evidence = _evidence()
        assert evidence["engine"] == "crucible.portfolio"
        assert evidence["schema_version"] == PORTFOLIO_EVIDENCE_SCHEMA_VERSION
        assert evidence["sessions"] == 2

    def test_all_four_named_components_are_recorded_with_a_reason(self) -> None:
        """A component absent from an artifact is indistinguishable from one
        that never ran, so all four appear whether or not they engaged."""
        components = _evidence()["components"]
        assert set(components) == {"mvo", "turnover_governor", "cost_model", "adv_cap"}
        for name, row in components.items():
            assert isinstance(row["applied"], bool), name
            assert row["reason"] is None or isinstance(row["reason"], str), name

    def test_a_disabled_adv_cap_is_recorded_as_disabled_not_omitted(self) -> None:
        constructed = construct_book(
            recipe=_recipe(IMPACT),
            params=_params(max_pct_adv=None),
            universe=_universe(),
            sessions=_sessions(),
            portfolio_notional=4_000_000.0,
            w_initial=np.array([0.0, 0.0, 0.97, 0.03]),
        )
        cap = constructed.evidence["components"]["adv_cap"]
        assert cap["applied"] is False
        assert cap["reason"] == "disabled"

    def test_the_cost_model_in_force_is_on_the_evidence(self) -> None:
        recorded = _evidence()["cost_model"]
        assert recorded["name"] == "sqrt_impact_v1"
        assert recorded["kind"] == "sqrt_impact"
        assert recorded["placeholder"] is False
        assert recorded["params"] == {k: float(v) for k, v in sorted(IMPACT["params"].items())}

    def test_a_stand_in_model_is_visibly_distinguishable_on_the_evidence(self) -> None:
        """`alpha-engine-config-I10503` closes-when 2. A placeholder is not
        inferred from a name a reader may not recognise — it is a field."""
        flat = construct_book(
            recipe=_recipe(FLAT),
            params=_params(),
            universe=_universe(),
            sessions=_sessions(with_adv=False),
            portfolio_notional=4_000_000.0,
            w_initial=np.array([0.0, 0.0, 0.97, 0.03]),
        )
        assert flat.evidence["cost_model"]["placeholder"] is True
        assert _evidence()["cost_model"]["placeholder"] is False

    def test_the_grade_is_charged_the_cost_the_engine_priced(self) -> None:
        """Not a second estimate of it, computed downstream from a summary."""
        constructed = construct_book(
            recipe=_recipe(IMPACT),
            params=_params(),
            universe=_universe(),
            sessions=_sessions(),
            portfolio_notional=4_000_000.0,
            w_initial=np.array([0.0, 0.0, 0.97, 0.03]),
        )
        recipe = _recipe(IMPACT)
        grade = grade_arm(recipe, constructed.book, as_of="2026-08-25")
        assert grade.total_cost_bps == pytest.approx(sum(constructed.book.cost_bps), rel=1e-9)
        assert grade.cost_model == recipe.cost_model.record()

    def test_a_participation_model_refuses_a_book_it_did_not_price(self) -> None:
        """The one route by which a verdict could name a participation-aware
        model and have been charged a flat rate — closed."""
        book = Book(
            dates=("2026-08-24",),
            portfolio_returns=(0.001,),
            benchmark_returns=(0.0005,),
            turnover=(0.1,),
        )
        with pytest.raises(CostModelInputError, match="prices participation"):
            grade_arm(_recipe(IMPACT), book, as_of="2026-08-24")

    def test_a_book_over_no_session_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no sessions to construct a book over"):
            construct_book(
                recipe=_recipe(IMPACT),
                params=_params(),
                universe=_universe(),
                sessions=(),
                portfolio_notional=4_000_000.0,
                w_initial=np.array([0.0, 0.0, 0.97, 0.03]),
            )


class TestTheEvidenceReachesAManifestAndIsReadable:
    """`alpha-engine-config-I10500` closes-when 1: a clause reads evidence of the
    call from a manifest, not the module merely existing."""

    def test_the_metric_row_satisfies_the_manifest_schema(self) -> None:
        schema = json.loads(
            (Path("crucible/schemas/run_manifest.v2.json")).read_text(encoding="utf-8")
        )
        from jsonschema import Draft202012Validator

        row = portfolio_metric_record(_evidence(), now_utc="2026-08-25T21:00:00Z")
        validator = Draft202012Validator(schema["$defs"]["MetricRecordRow"])
        assert sorted(validator.iter_errors(row), key=lambda e: list(e.path)) == []

    def test_the_reader_finds_the_evidence_on_a_manifest(self) -> None:
        evidence = _evidence()
        row = portfolio_metric_record(evidence, now_utc="2026-08-25T21:00:00Z")
        found = manifest_records_portfolio_engine({"metrics": [row]})
        assert found == evidence

    def test_a_manifest_with_no_such_row_reads_as_nothing_recorded(self) -> None:
        """Not `False`. "This run recorded nothing" and "the engine was not
        used" are different answers, and a clause that cannot tell them apart
        grades an unobserved run as a failed one."""
        assert manifest_records_portfolio_engine({"metrics": []}) is None

    def test_a_row_that_names_the_engine_without_carrying_evidence_is_refused(self) -> None:
        """A clause counting rows would read it as proof."""
        with pytest.raises(ValueError, match="no `portfolio_construction` payload"):
            manifest_records_portfolio_engine(
                {"metrics": [{"name": PORTFOLIO_METRIC_NAME, "module": "crucible.portfolio"}]}
            )

    def test_evidence_written_under_another_schema_version_is_refused(self) -> None:
        evidence = {**_evidence(), "schema_version": "portfolio_construction.v9"}
        row = portfolio_metric_record(_evidence(), now_utc="2026-08-25T21:00:00Z")
        row["portfolio_construction"] = evidence
        with pytest.raises(ValueError, match="this reader understands"):
            manifest_records_portfolio_engine({"metrics": [row]})

    def test_malformed_evidence_fails_at_the_producer(self) -> None:
        """Validated where it is built, so a malformed record never reaches the
        consumer that would have to guess what it meant."""
        with pytest.raises(ValueError, match="sessions"):
            portfolio_evidence(
                trading_day="2026-08-25",
                arm_id="s:contract_fixture:0123456789abcdef",
                params=_params(),
                cost_model=CostModel(**IMPACT),
                diagnostics={"status": "optimal"},
                sessions=0,
                turnover_one_way_total=0.1,
                cost_bps_total=1.0,
            )


class TestTheEvidenceSchemaIsAGuardThatFires:
    """A schema nobody has made reject something is a schema nobody knows
    constrains anything (AGENTS.md, Test discipline)."""

    def test_the_committed_schema_is_a_valid_2020_12_schema(self) -> None:
        from jsonschema import Draft202012Validator

        schema = json.loads(
            Path(f"crucible/schemas/{PORTFOLIO_EVIDENCE_SCHEMA_VERSION}.json").read_text(
                encoding="utf-8"
            )
        )
        Draft202012Validator.check_schema(schema)
        assert schema["title"] == PORTFOLIO_EVIDENCE_SCHEMA_VERSION

    @pytest.mark.parametrize(
        "dropped",
        ["cost_model", "params", "params_digest", "components", "engine", "sessions"],
    )
    def test_a_document_missing_a_required_field_is_refused(self, dropped: str) -> None:
        from jsonschema import Draft202012Validator

        schema = json.loads(
            Path(f"crucible/schemas/{PORTFOLIO_EVIDENCE_SCHEMA_VERSION}.json").read_text(
                encoding="utf-8"
            )
        )
        document = {k: v for k, v in _evidence().items() if k != dropped}
        errors = list(Draft202012Validator(schema).iter_errors(document))
        assert errors, f"a document without {dropped!r} must not validate"

    def test_a_cost_model_block_without_its_placeholder_flag_is_refused(self) -> None:
        """The field that keeps a stand-in from reading as the configured model
        is required by the SCHEMA too, not only by the producer — a consumer
        reading an older document must not find it optional."""
        from jsonschema import Draft202012Validator

        schema = json.loads(
            Path(f"crucible/schemas/{PORTFOLIO_EVIDENCE_SCHEMA_VERSION}.json").read_text(
                encoding="utf-8"
            )
        )
        document = _evidence()
        document["cost_model"] = {
            k: v for k, v in document["cost_model"].items() if k != "placeholder"
        }
        assert list(Draft202012Validator(schema).iter_errors(document))

    def test_a_document_naming_another_engine_is_refused(self) -> None:
        """`engine` is what a gate clause matches on, so it is a constant in the
        schema: a field that could say anything can say nothing."""
        from jsonschema import Draft202012Validator

        schema = json.loads(
            Path(f"crucible/schemas/{PORTFOLIO_EVIDENCE_SCHEMA_VERSION}.json").read_text(
                encoding="utf-8"
            )
        )
        document = {**_evidence(), "engine": "some.other.module"}
        assert list(Draft202012Validator(schema).iter_errors(document))
