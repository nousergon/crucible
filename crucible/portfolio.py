"""Portfolio construction: MVO, the turnover governor, the cost model, the ADV cap.

Normative sources: plan §4.4, §10.2; `champion-challenger-policy.md` §4;
`repository-tiering-policy.md` test 2.

Lifted, not imported, from `crucible-executor/executor/portfolio_optimizer.py`
— a v1 repository that retires at the phase-4 cutover. What travelled is the
ENGINE. What did NOT travel is every tuned value it carried inline: those are
strategy edge and live in the private configuration tree, loaded at runtime as
:class:`PortfolioParams` and schema-validated. This module declares the SHAPE a
parameter set must have and refuses one that is short a field; it declares no
value for any of them.

Four components, which is what the phase-3 deliverable names:

**MVO.** A constrained mean-variance program, solved as a convex problem::

    maximize   wᵀα̂  −  λ·wᵀΣ_H w  −  γ·wᵀΩw  −  C(w − w_prev)/NAV
    s.t.       Σwᵢ = 1                                    (budget)
               w[CASH] = cash_sleeve_pct                  (sleeve pin)
               0 ≤ wᵢ ≤ stance_capᵢ                       (per-name cap)
               Σ_{i∈sector S} wᵢ ≤ max_sector_pct         (sector cap)
               |wᵢ − w_prevᵢ|·NAV ≤ max_pct_adv·ADVᵢ      (participation cap)
               wᵢ = 0 where eligibility is False          (gate mask)
               ‖w − w_prev‖₁/2 ≤ turnover budget          (governor)
               wᵀΣ_H w ≤ σ²_target_H                      (vol target, optional)

The benchmark is the no-conviction fill and cash is a pinned sleeve, so
conviction expresses DEVIATION from the benchmark rather than an absolute
holding. Σ_H is the H-day covariance; under an i.i.d. log-return assumption
Σ_H = H·Σ_daily, and H is a declared parameter rather than an implicit 1.

Ω = diag(σ_ε²) is the Garlappi-Uppal-Wang (2007) estimation-uncertainty
penalty. It is built from the ESTIMATION-error std of α̂ and never from a
predictive std whose observation-noise term is a per-batch scalar: that term
double-counts risk Σ already carries and is constant across the cross-section,
so an Ω built from it is a uniform ridge that discriminates between nothing.
The three ways the term can be inoperative are each RECORDED by name —
:func:`_resolve_alpha_uncertainty` — because a reason that appears only when a
term engages is indistinguishable from a term that is quietly dead.

**Turnover governor.** The daily one-way turnover budget is a CONSTRAINT
inside the convex program, never a post-solve shrink. The objective therefore
chooses which trades fit the budget and each surviving name lands at a size it
actually wants; a uniform post-hoc shrink instead trades a little less of
everything, which can push an entire entry cohort under a downstream
rebalance band and delete it while the solve still reports `optimal`.

The budget governs DISCRETIONARY trading only. A held name pinned to zero by
the eligibility mask, and the cash sleeve's equality pin, MANDATE movement;
:func:`_mandatory_turnover_floor` bounds it and the constraint's right-hand
side is raised to it when it is larger, which makes the feasible set non-empty
by construction rather than by assumption. A forced exit is not discretionary
and is never starved of budget.

:func:`_apply_turnover_governor` runs after the solve and only MEASURES —
it raises :class:`TurnoverBudgetError` and never modifies the vector. Its
tolerance is derived from the mass the dust rule actually zeroed, not chosen,
so a real breach cannot be absorbed by a hand-picked epsilon.

**Cost model.** Named, and recorded by name and parameters on every artifact —
see :class:`CostModel`. There is no default and no fallback: a recipe names a
model in :data:`COST_MODELS` or it does not load.

**ADV cap.** ``|Δwᵢ|·NAV ≤ max_pct_adv·ADVᵢ``, the hard capacity guardrail the
participation-priced cost term complements. The cost term prices
participation; this refuses to trade a name so thin that the price is not to
be trusted.

**The convex layer is called, not re-derived.** ``cvxpy`` owns the DCP
analysis and the conic solve; ``nousergon_lib.quant.factor_risk`` owns
Ledoit-Wolf shrinkage; ``nousergon_lib.quant.transaction_cost`` owns the
square-root impact law. This module composes them and adds the constraint
geometry, exactly as `crucible.slots` configures the arena without
re-implementing it.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from jsonschema import Draft202012Validator

__all__ = [
    "COST_MODELS",
    "COST_MODEL_KINDS",
    "COVARIANCE_ESTIMATORS",
    "PORTFOLIO_EVIDENCE_SCHEMA_VERSION",
    "PORTFOLIO_METRIC_NAME",
    "PORTFOLIO_PARAM_FIELDS",
    "CostModel",
    "CostModelError",
    "CostModelInputError",
    "OptimizerResult",
    "PortfolioParams",
    "PortfolioParamsError",
    "TurnoverBudgetError",
    "compute_conviction_budget_multiplier",
    "cost_model_from_mapping",
    "load_portfolio_params",
    "make_cash_sentinel_returns",
    "manifest_records_portfolio_engine",
    "params_digest",
    "portfolio_evidence",
    "portfolio_metric_record",
    "solve_target_weights",
]


# ---------------------------------------------------------------------------
# The cost model. `alpha-engine-config-I10503`.
# ---------------------------------------------------------------------------

#: The registered transaction-cost models and the parameters each REQUIRES.
#: The implementations and this registry are framework and therefore public;
#: the calibrated VALUES are strategy edge and live in the private strategy
#: tree (`repository-tiering-policy` test 2). A recipe naming a model absent
#: from here is refused at load, by name.
#:
#: This registry is the whole reason there is no fallback to fall into. The
#: square-root model's own library constructor accepts a config block and
#: fills every absent key from an institutional default, which is how an
#: optimizer once ran on a cost model nobody had configured and no artifact
#: said so. Here a model is named or the recipe does not load, and every
#: parameter is declared or the model does not construct.
COST_MODELS: dict[str, tuple[str, ...]] = {
    # Flat round-trip basis points per unit of turnover. Liquidity-blind, and
    # its name is required to say so: a flat model cannot answer "what did
    # trading THIS name cost", only "what does trading cost on average".
    "flat_bps_v0": ("half_spread_bps", "commission_bps", "slippage_bps"),
    # The square-root market-impact law (Almgren-Chriss / Kissell), evaluated
    # per name against its average daily dollar volume. The parameters are
    # exactly `nousergon_lib.quant.transaction_cost.TransactionCostModel`'s
    # four fields, and all four are required here precisely because they are
    # optional there.
    "sqrt_impact_v1": (
        "half_spread_bps",
        "impact_coef_bps",
        "commission_bps",
        "min_cost_bps",
    ),
}

#: Which pricing law each registered model is. A KIND decides what inputs a
#: grade needs; a NAME decides what a card reports. Keeping them separate is
#: what lets a recalibrated square-root model be a new name — and therefore a
#: new arm id — without the grader learning a new code path.
COST_MODEL_KINDS: dict[str, str] = {
    "flat_bps_v0": "flat",
    "sqrt_impact_v1": "sqrt_impact",
}


class CostModelError(ValueError):
    """A cost model was named that does not exist, or declared incompletely."""


class CostModelInputError(ValueError):
    """A named cost model was given inputs it cannot price.

    Raised rather than degraded. The live trader's optimizer answers a missing
    book notional or an absent ADV vector by silently substituting a flat
    penalty, because holding a book for a session is the worse outcome there.
    Grading is offline and the caller owns every input, so the same
    substitution here would only produce a verdict whose cost model is not the
    one the recipe named — which is the defect, not the mitigation.
    """


@dataclass(frozen=True)
class CostModel:
    """The cost model a grade is net of. NAMED in the recipe, never here.

    ``placeholder`` is a required field rather than an inference from the
    name: a verdict graded against a stand-in must say so on its face, and a
    card that hid it would read as a net-of-cost result the day it was not
    one.

    Every parameter the named model declares must be present and no others:
    an extra key is a value someone believes is live, and a missing one makes
    the grade unreproducible from the recipe.
    """

    name: str
    placeholder: bool
    params: dict[str, float]

    def __post_init__(self) -> None:
        if not self.name:
            raise CostModelError("a cost model must be named; an anonymous cost is unreproducible")
        if self.name not in COST_MODELS:
            raise CostModelError(
                f"unknown cost model {self.name!r}; the registered models are "
                f"{sorted(COST_MODELS)}. A model absent from the registry has no "
                "declared parameter set, so a grade charged by it could not be "
                "reproduced from the recipe."
            )
        required = set(COST_MODELS[self.name])
        declared = set(self.params)
        missing = sorted(required - declared)
        if missing:
            raise CostModelError(
                f"cost model {self.name!r} is missing constant(s) {missing}. A grade "
                "net of a partially declared cost is not reproducible from the "
                "recipe, and the absent constants would be filled by a default "
                "nothing recorded."
            )
        extra = sorted(declared - required)
        if extra:
            raise CostModelError(
                f"cost model {self.name!r} declares {extra}, which it does not read. "
                f"Its parameters are {sorted(required)}. An unread constant in a "
                "recipe is a value someone believes is priced and is not."
            )

    @property
    def kind(self) -> str:
        """``"flat"`` or ``"sqrt_impact"`` — which pricing law is in force."""
        return COST_MODEL_KINDS[self.name]

    def bps_per_unit_turnover(self) -> float:
        """Round-trip cost in basis points per unit of turnover — FLAT only.

        A participation-aware model has no such number: its cost per unit of
        turnover depends on the trade's size against the name's volume, which
        is the whole point of it. Returning an average here would hand every
        consumer a figure that reads like a rate and is not one.
        """
        if self.kind != "flat":
            raise CostModelInputError(
                f"cost model {self.name!r} prices participation, so it has no single "
                "cost per unit of turnover. Price the realized trades with "
                "`cost_bps_for_trades`, or name a flat model in the recipe."
            )
        return (
            2.0 * float(self.params["half_spread_bps"])
            + float(self.params["commission_bps"])
            + float(self.params["slippage_bps"])
        )

    def impact_model(self) -> Any:
        """The library's :class:`TransactionCostModel`, built from DECLARED values.

        Constructed field by field rather than through the library's
        ``from_config``, which fills every absent key from an institutional
        default. Here there are no absent keys — :meth:`__post_init__` has
        already refused a model short one — so the object returned is the
        recipe's model and provably nothing else.
        """
        if self.kind != "sqrt_impact":
            raise CostModelInputError(
                f"cost model {self.name!r} is not a square-root impact model; it has "
                "no impact coefficient to build one from."
            )
        from nousergon_lib.quant.transaction_cost import TransactionCostModel

        return TransactionCostModel(
            half_spread_bps=float(self.params["half_spread_bps"]),
            impact_coef_bps=float(self.params["impact_coef_bps"]),
            commission_bps=float(self.params["commission_bps"]),
            min_cost_bps=float(self.params["min_cost_bps"]),
        )

    def cost_bps_for_trades(
        self,
        *,
        weight_deltas: np.ndarray,
        adv_usd: np.ndarray | None,
        portfolio_notional: float | None,
        name_sigma: np.ndarray | None = None,
        benchmark_idx: int | None = None,
        cash_idx: int | None = None,
    ) -> float:
        """Cost of one rebalance, in basis points OF THE BOOK.

        The number a graded day is charged. ``weight_deltas`` is the per-name
        change in weight; one rebalance is ONE side, so a buy-then-sell cycle
        is naturally two applications of this.

        A flat model prices it as ``bps_per_unit_turnover × one-way turnover``.
        A square-root model prices each name separately against its own ADV and
        raises rather than averaging when it is handed a name it cannot price.

        ``benchmark_idx`` and ``cash_idx`` are excluded from the impact term and
        from the volatility reference, exactly as the objective's cost term
        excludes them: the benchmark fill and the cash sleeve carry no market
        impact. They are passed rather than inferred, and they MATTER — the
        reference volatility is a cross-sectional median, so including a
        benchmark in it moves the scaling of every other name's impact, and the
        charge a grade subtracts would then be priced off a different reference
        from the one the solve optimised against.
        """
        dw = np.abs(np.asarray(weight_deltas, dtype=np.float64).ravel())
        if self.kind == "flat":
            return self.bps_per_unit_turnover() * float(dw.sum()) / 2.0
        if portfolio_notional is None or not (float(portfolio_notional) > 0.0):
            raise CostModelInputError(
                f"cost model {self.name!r} prices participation and needs the book "
                "notional to turn a weight delta into a trade size; none was given. "
                "A grade cannot silently fall back to a flat charge — the verdict "
                "would name this model and not have been charged by it."
            )
        if adv_usd is None:
            raise CostModelInputError(
                f"cost model {self.name!r} prices each name against its average "
                "daily dollar volume; no ADV vector was given. Grading is offline "
                "and the caller owns this input, so an absent one is a caller "
                "defect, not a market condition."
            )
        adv = np.asarray(adv_usd, dtype=np.float64).ravel()
        if adv.shape != dw.shape:
            raise CostModelInputError(
                f"adv_usd shape {adv.shape} != weight_deltas shape {dw.shape}; one "
                "ADV entry per name."
            )
        nav = float(portfolio_notional)
        model = self.impact_model()
        usable = np.isfinite(adv) & (adv > 0.0)
        for sentinel in (benchmark_idx, cash_idx):
            if sentinel is not None and 0 <= sentinel < usable.size:
                usable[sentinel] = False
        sigma_used, ref_sigma = _resolve_ref_sigma(name_sigma, usable)
        total_usd = 0.0
        for i in range(dw.size):
            notional = float(dw[i]) * nav
            if notional <= 0.0:
                continue
            name_adv = float(adv[i]) if usable[i] else None
            sigma_i: float | None = None
            if sigma_used is not None and ref_sigma:
                candidate = float(sigma_used[i])
                if math.isfinite(candidate) and candidate > 0.0:
                    sigma_i = candidate
            total_usd += model.cost_for_turnover(
                notional, name_adv, sigma=sigma_i, ref_sigma=ref_sigma
            )
        return total_usd / nav * 1e4

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "placeholder": self.placeholder, "params": dict(self.params)}

    def record(self) -> dict[str, Any]:
        """The model in force, for the artifact — name, kind, placeholder, params.

        `alpha-engine-config-I10503`. Every grading run that applies costs
        records this. ``kind`` is carried alongside ``name`` so a reader who
        has never seen the name can still tell what law priced the run, and
        ``placeholder`` so a stand-in can never be mistaken for the configured
        model on a card, a console pane or a promotion decision.
        """
        return {
            "name": self.name,
            "kind": self.kind,
            "placeholder": bool(self.placeholder),
            "params": {k: float(v) for k, v in sorted(self.params.items())},
        }


def cost_model_from_mapping(payload: Mapping[str, Any], *, source: str) -> CostModel:
    """Build a :class:`CostModel` from a recipe's ``cost_model`` block.

    ``source`` names the file in every refusal, so a malformed recipe is
    identified without grepping for the value that broke.
    """
    for field_name in ("name", "placeholder", "params"):
        if field_name not in payload:
            raise CostModelError(
                f"{source}: the cost_model block declares no {field_name!r}. All three "
                "of name, placeholder and params are required — a model missing any "
                "of them cannot be recorded in a way a later reader can act on."
            )
    params = payload["params"]
    if not isinstance(params, Mapping):
        raise CostModelError(
            f"{source}: cost_model.params must be a mapping, got {type(params).__name__}"
        )
    return CostModel(
        name=str(payload["name"]),
        placeholder=bool(payload["placeholder"]),
        params={str(k): float(v) for k, v in params.items()},
    )


# ---------------------------------------------------------------------------
# The parameter set. `alpha-engine-config-I10500`.
# ---------------------------------------------------------------------------

#: The covariance estimators this engine supports, and what each is for.
#: `ledoit_wolf` is the institutional default and is the library's own
#: pure-numpy implementation, not a second one written here.
COVARIANCE_ESTIMATORS: dict[str, str] = {
    "ledoit_wolf": "Ledoit-Wolf (2004) shrinkage toward a scaled identity",
    "sample": "raw sample covariance, unshrunk",
    "ewma": "exponentially weighted, RiskMetrics-style, decayed by ewma_lambda_decay",
}

#: An estimator the lift's source supported that this engine deliberately does
#: not, mapped to the reason. Refused BY NAME rather than quietly resolved to
#: the default: a parameter set asking for an estimator it does not get is a
#: different strategy from the one whose id was hashed.
_ESTIMATORS_NOT_CARRIED: dict[str, str] = {
    "oas": (
        "Oracle Approximating Shrinkage shrinks toward the same target family as "
        "ledoit_wolf and was never the configured estimator, but it is available "
        "only through scikit-learn — a compiled dependency this repository would "
        "carry for a second member of one estimator family. Use 'ledoit_wolf'."
    ),
}

#: Every field a portfolio parameter set must declare, with its type. Public
#: because its ENFORCER lives in another repository: the private strategy
#: tree's own validation imports this rather than restating it, the way
#: `crucible.slots`' registries are imported rather than mirrored. No value
#: for any of them appears in this repository.
PORTFOLIO_PARAM_FIELDS: dict[str, str] = {
    "risk_aversion": "number",
    "cash_sleeve_pct": "number",
    "max_sector_pct": "number",
    "min_position_pct": "number",
    "covariance_shrinkage": "string",
    "sigma_horizon_days": "integer",
    "ewma_lambda_decay": "number",
    "vol_target_annual": "number-or-null",
    "alpha_uncertainty_penalty": "number",
    "alpha_uncertainty_min_cv": "number",
    "max_pct_adv": "number-or-null",
    "max_daily_turnover": "number-or-null",
    "large_move_turnover_flag": "number-or-null",
    "conviction_budget_gate_enabled": "boolean",
    "conviction_ir_floor": "number",
    "conviction_ir_full": "number",
    "conviction_budget_min_multiple": "number",
    "conviction_gate_min_names": "integer",
}

_NULLABLE_PARAMS: frozenset[str] = frozenset(
    name for name, kind in PORTFOLIO_PARAM_FIELDS.items() if kind.endswith("-or-null")
)


class PortfolioParamsError(ValueError):
    """A portfolio parameter set is absent, short a field, or carries an unknown one."""


@dataclass(frozen=True)
class PortfolioParams:
    """The tuned inputs to portfolio construction. Every field is REQUIRED.

    No field carries a default, and the dataclass is constructed only through
    :meth:`from_mapping`, which refuses a mapping that is short a field or
    carries one this engine does not read. That is the whole design: a default
    here would be a tuned value committed to a public repository, and a
    silently-defaulted risk aversion is a book traded on a number nobody
    chose.

    A nullable field is nullable to express *disabled*, and its key is still
    required: an explicit ``null`` says "this guardrail is off and I meant it",
    an absent key says nothing at all, and the two must not be spelled the
    same.
    """

    risk_aversion: float
    cash_sleeve_pct: float
    max_sector_pct: float
    min_position_pct: float
    covariance_shrinkage: str
    sigma_horizon_days: int
    ewma_lambda_decay: float
    vol_target_annual: float | None
    alpha_uncertainty_penalty: float
    alpha_uncertainty_min_cv: float
    max_pct_adv: float | None
    max_daily_turnover: float | None
    large_move_turnover_flag: float | None
    conviction_budget_gate_enabled: bool
    conviction_ir_floor: float
    conviction_ir_full: float
    conviction_budget_min_multiple: float
    conviction_gate_min_names: int

    def __post_init__(self) -> None:
        if self.covariance_shrinkage in _ESTIMATORS_NOT_CARRIED:
            raise PortfolioParamsError(
                f"covariance_shrinkage {self.covariance_shrinkage!r} is not carried by "
                f"this engine. {_ESTIMATORS_NOT_CARRIED[self.covariance_shrinkage]}"
            )
        if self.covariance_shrinkage not in COVARIANCE_ESTIMATORS:
            raise PortfolioParamsError(
                f"unknown covariance_shrinkage {self.covariance_shrinkage!r}; the "
                f"supported estimators are {sorted(COVARIANCE_ESTIMATORS)}"
            )
        if self.sigma_horizon_days < 1:
            raise PortfolioParamsError(
                f"sigma_horizon_days must be >= 1 trading day; got {self.sigma_horizon_days}"
            )
        if not 0.5 <= self.ewma_lambda_decay <= 1.0:
            raise PortfolioParamsError(
                f"ewma_lambda_decay must be in [0.5, 1.0]; got {self.ewma_lambda_decay}"
            )
        if not 0.0 <= self.cash_sleeve_pct < 1.0:
            raise PortfolioParamsError(
                f"cash_sleeve_pct must be in [0, 1); got {self.cash_sleeve_pct}. The "
                "sleeve is an equality pin, so a value at or above 1 leaves the "
                "budget constraint with no equity to allocate."
            )
        if self.risk_aversion < 0.0:
            raise PortfolioParamsError(
                f"risk_aversion must be >= 0; got {self.risk_aversion}. A negative "
                "coefficient turns the variance term into a REWARD for risk, which "
                "is a sign error that still solves."
            )
        if self.conviction_ir_full <= self.conviction_ir_floor:
            raise PortfolioParamsError(
                "conviction_ir_full must exceed conviction_ir_floor; got "
                f"{self.conviction_ir_full} <= {self.conviction_ir_floor}. An inverted "
                "band would map every signal quality onto an arbitrary throttle."
            )
        if not 0.0 <= self.conviction_budget_min_multiple <= 1.0:
            raise PortfolioParamsError(
                "conviction_budget_min_multiple must be in [0, 1]; got "
                f"{self.conviction_budget_min_multiple}"
            )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], *, source: str) -> PortfolioParams:
        """Validate and build. Refuses a missing or unknown field BY NAME.

        ``source`` names the file in every refusal so the operator is not left
        grepping a tree for which parameter set was short a key.
        """
        if not isinstance(payload, Mapping):
            raise PortfolioParamsError(
                f"{source}: portfolio parameters must be a mapping, got {type(payload).__name__}"
            )
        missing = sorted(set(PORTFOLIO_PARAM_FIELDS) - set(payload))
        if missing:
            raise PortfolioParamsError(
                f"{source}: portfolio parameters are missing {missing}. Every field is "
                "required and none has a default — a book must never be constructed "
                "on a number this repository chose on the operator's behalf."
            )
        unknown = sorted(set(payload) - set(PORTFOLIO_PARAM_FIELDS))
        if unknown:
            raise PortfolioParamsError(
                f"{source}: portfolio parameters declare {unknown}, which this engine "
                f"does not read. The fields it reads are {sorted(PORTFOLIO_PARAM_FIELDS)}. "
                "An unread parameter is a knob someone believes is live."
            )
        values: dict[str, Any] = {}
        for name, kind in PORTFOLIO_PARAM_FIELDS.items():
            raw = payload[name]
            if raw is None:
                if name not in _NULLABLE_PARAMS:
                    raise PortfolioParamsError(
                        f"{source}: {name!r} is null, but it has no disabled state — "
                        "it is read on every solve and must carry a value."
                    )
                values[name] = None
                continue
            if kind == "string":
                values[name] = str(raw)
            elif kind == "boolean":
                if not isinstance(raw, bool):
                    raise PortfolioParamsError(
                        f"{source}: {name!r} must be a boolean; got {raw!r}. A truthy "
                        "string switches a guardrail on for a reason nobody wrote down."
                    )
                values[name] = raw
            elif kind == "integer":
                if isinstance(raw, bool) or not isinstance(raw, int):
                    raise PortfolioParamsError(
                        f"{source}: {name!r} must be an integer; got {raw!r}"
                    )
                values[name] = int(raw)
            else:
                if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                    raise PortfolioParamsError(f"{source}: {name!r} must be a number; got {raw!r}")
                values[name] = float(raw)
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        """The parameter set as written, for the artifact."""
        return {name: getattr(self, name) for name in sorted(PORTFOLIO_PARAM_FIELDS)}


def load_portfolio_params(path: Path | str) -> PortfolioParams:
    """Read a portfolio parameter set from the private strategy tree.

    `alpha-engine-config/strategy/slots/s.yaml` in production. The file is
    read, every field is validated, and an absent file RAISES: there is no
    parameter set this repository can supply, and a default that let grading
    proceed would be a tuned value chosen by the harness and recorded nowhere.

    The parameters are the SLOT's, not an arm's. Every S arm is graded through
    the same book construction for the same reason no arm chooses its own
    benchmark: an arm able to pick its own risk aversion or its own turnover
    budget could pick an easy one, and the comparison the slot exists to make
    would be confounded by the construction rather than decided by the rules.
    """
    resolved = Path(path)
    if not resolved.exists():
        raise PortfolioParamsError(
            f"no portfolio parameter set at {resolved}. Portfolio construction has no "
            "default parameters: risk aversion, the turnover budget and the "
            "participation cap are strategy edge and live in the private strategy "
            "tree. Grading refuses rather than constructing a book on numbers this "
            "repository invented."
        )
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or "portfolio" not in payload:
        raise PortfolioParamsError(
            f"{resolved}: expected a mapping carrying a 'portfolio' block; the file declares none."
        )
    return PortfolioParams.from_mapping(payload["portfolio"], source=str(resolved))


# ---------------------------------------------------------------------------
# The engine.
# ---------------------------------------------------------------------------

_CLARABEL = "CLARABEL"
_FALLBACK_SOLVERS = ("SCS", "OSQP")

#: Complementary-slackness tolerances for the turnover-budget binding test.
#: ``_DUAL_ACTIVE_TOL`` sits orders of magnitude above the numerical dust an
#: interior-point solver returns for an INACTIVE constraint and well below the
#: smallest genuine active dual. ``_PRIMAL_AT_BOUND_REL_TOL`` is wide enough to
#: admit the drift the post-solve clip-and-renormalize step introduces.
_DUAL_ACTIVE_TOL = 1e-6
_PRIMAL_AT_BOUND_REL_TOL = 5e-3

#: The fewest clean rows a covariance estimate is allowed to be built from. A
#: shorter panel produces a matrix whose eigenvalues are dominated by sampling
#: noise, and shrinkage narrows that without removing it.
_MIN_COVARIANCE_ROWS = 20


@dataclass(frozen=True)
class OptimizerResult:
    """The solved weights and everything the solve recorded about itself."""

    weights: np.ndarray
    diagnostics: dict


class TurnoverBudgetError(RuntimeError):
    """The solved weight vector exceeds the daily turnover budget.

    Raised, not silently corrected. The budget is enforced INSIDE the convex
    program, so a vector that violates it means the solver returned a point
    that does not satisfy a constraint it was given, or the post-solve
    clip-and-renormalize moved it further than that step can account for. Both
    are defects in this module, not market conditions.

    Shrinking the vector back under the budget is precisely the mechanism the
    constraint construction removed, and doing it here would restore the
    failure it fixed while now also hiding a solver defect behind a
    plausible-looking weight vector.
    """


def solve_target_weights(
    tickers: list[str],
    alpha_hat: np.ndarray,
    returns_panel: np.ndarray | None,
    w_prev: np.ndarray,
    sectors: list[str],
    stance_caps: np.ndarray,
    eligibility: np.ndarray,
    benchmark_idx: int,
    cash_idx: int,
    params: PortfolioParams,
    cost_model: CostModel,
    *,
    alpha_uncertainty: np.ndarray | None = None,
    alpha_uncertainty_epistemic: np.ndarray | None = None,
    covariance: np.ndarray | None = None,
    adv_usd: np.ndarray | None = None,
    portfolio_notional: float | None = None,
    name_sigma: np.ndarray | None = None,
) -> OptimizerResult:
    """Solve the constrained MVO and return target weights plus diagnostics.

    Args:
        tickers: the length-N universe. Must contain the benchmark fill and a
            cash sentinel.
        alpha_hat: shape (N,) predicted alpha. Convention: the benchmark entry
            is 0.0 (the benchmark is the null hypothesis) and the cash entry is
            a small negative number, so the benchmark is preferred to cash when
            the two are otherwise indifferent.
        returns_panel: shape (T, N) daily returns for the covariance estimate.
            Rows carrying a NaN are dropped before shrinkage. May be None when
            ``covariance`` is supplied.
        w_prev: shape (N,) current weights (positions / NAV).
        sectors: length-N sector labels. The benchmark and cash entries carry
            sentinel labels of the form ``__name__`` so they are not summed
            into a real sector's cap.
        stance_caps: shape (N,) per-name upper bound on weight, composed by the
            caller. The cash entry's cap is overridden by the equality pin.
        eligibility: shape (N,) bool. False pins the name to zero. The
            benchmark and cash entries must be eligible.
        benchmark_idx, cash_idx: positions in ``tickers``.
        params: the slot's parameter set. Required, and carries no defaults.
        cost_model: the NAMED model the objective's turnover term is priced by.
            Required — there is no default and no fallback.
        alpha_uncertainty: optional shape (N,) TOTAL predictive std per name.
            This is the conviction gate's input and does NOT enter the
            estimation-uncertainty penalty; see :func:`_resolve_alpha_uncertainty`.
        alpha_uncertainty_epistemic: optional shape (N,) ESTIMATION-error std
            per name — the only vector the penalty's Ω is built from.
        covariance: optional shape (N,N) DAILY covariance. When supplied the
            estimator step is skipped and this matrix is used directly, with
            the same horizon scaling applied.
        adv_usd: optional shape (N,) average daily DOLLAR volume per name.
            Drives both the participation-priced cost term and the
            participation constraint.
        portfolio_notional: book size in dollars, needed to turn weight deltas
            into trade notionals.
        name_sigma: optional shape (N,) per-name return volatility, used to
            scale the impact term by σᵢ/refσ where refσ is the cross-sectional
            median.

    Returns:
        :class:`OptimizerResult`. On infeasibility the weights are the
        current book with cash absorbing the residual and
        ``diagnostics["status"]`` is ``"infeasible_fallback"`` — a REPORTED
        outcome, never an unmarked one.
    """
    N = len(tickers)
    _validate_inputs(
        tickers,
        alpha_hat,
        returns_panel,
        w_prev,
        sectors,
        stance_caps,
        eligibility,
        benchmark_idx,
        cash_idx,
        covariance_provided=covariance is not None,
    )

    if covariance is not None:
        # Re-solve path: reuse an already-estimated DAILY Σ. The SAME horizon
        # scaling the estimator path applies is applied here; injecting an
        # already-scaled matrix would double-scale it and silently corrupt the
        # vol-target constraint and every vol diagnostic.
        sigma = params.sigma_horizon_days * _validate_covariance(covariance, N)
    else:
        sigma = _estimate_covariance(returns_panel, params)

    omega_diag, alpha_unc_used, alpha_unc_meta = _resolve_alpha_uncertainty(
        alpha_uncertainty_epistemic,
        N,
        params,
    )

    import cvxpy as cp

    sigma_psd = cp.psd_wrap(sigma)
    w = cp.Variable(N)

    tcost = _build_tcost_term(
        cp,
        w,
        w_prev,
        adv_usd,
        portfolio_notional,
        name_sigma,
        benchmark_idx,
        cash_idx,
        cost_model,
    )
    objective_terms = [
        alpha_hat @ w,
        -params.risk_aversion * cp.quad_form(w, sigma_psd),
        tcost.objective_term,
    ]
    if alpha_unc_used:
        # γ · Σᵢ (σ_εᵢ² · wᵢ²) — the diagonal-Ω estimation-uncertainty penalty.
        # `cp.square(w)` is convex, a non-negatively weighted sum of convex
        # terms is convex, and negating it inside a Maximize is concave.
        objective_terms.append(
            -float(params.alpha_uncertainty_penalty) * (omega_diag @ cp.square(w))
        )
    objective = cp.Maximize(sum(objective_terms))

    ineligible_idx = np.where(~eligibility)[0]
    effective_caps = np.where(eligibility, stance_caps, 0.0)

    constraints = [
        cp.sum(w) == 1.0,
        w >= 0,
        w <= effective_caps,
        w[cash_idx] == params.cash_sleeve_pct,
    ]
    if params.vol_target_annual is not None:
        # Σ is at horizon H. Under i.i.d. log-returns Var_ann = Var_H · (252/H),
        # so the H-day variance budget corresponding to an annual vol target is
        # target² · H/252.
        sigma_target_squared = (
            float(params.vol_target_annual) ** 2 * params.sigma_horizon_days / 252
        )
        constraints.append(cp.quad_form(w, sigma_psd) <= sigma_target_squared)
    if ineligible_idx.size > 0:
        constraints.append(w[ineligible_idx] == 0)

    for sector_label in _real_sectors(sectors):
        idx = [i for i, s in enumerate(sectors) if s == sector_label]
        constraints.append(cp.sum(w[idx]) <= params.max_sector_pct)

    adv_cap_meta = _apply_max_pct_adv_constraint(
        w,
        w_prev,
        constraints,
        adv_usd,
        portfolio_notional,
        benchmark_idx,
        cash_idx,
        params,
    )
    turnover_meta = _apply_turnover_constraint(
        cp,
        w,
        w_prev,
        constraints,
        effective_caps,
        cash_idx,
        params,
        eligibility=eligibility,
        alpha_hat=alpha_hat,
        alpha_uncertainty=alpha_uncertainty,
        benchmark_idx=benchmark_idx,
    )

    problem = cp.Problem(objective, constraints)
    weights, status, solver_notes = _solve_with_fallback(problem, w)

    if weights is None:
        weights = _fallback_weights(w_prev, cash_idx, params.cash_sleeve_pct)
        diagnostics = _build_diagnostics(
            weights,
            w_prev,
            sigma,
            alpha_hat,
            benchmark_idx,
            "infeasible_fallback",
            params,
            omega_diag=omega_diag,
            alpha_unc_used=alpha_unc_used,
            alpha_unc_meta=alpha_unc_meta,
        )
        diagnostics.update(tcost.diagnostics)
        diagnostics.update(adv_cap_meta)
        diagnostics.update(_turnover_diagnostics(weights, w_prev, turnover_meta))
        diagnostics["solver_attempts"] = solver_notes
        diagnostics["cost_model"] = cost_model.record()
        return OptimizerResult(weights=weights, diagnostics=diagnostics)

    weights, clip_mass_zeroed = _clip_and_renormalize(weights, effective_caps, cash_idx, params)
    weights, governor = _apply_turnover_governor(
        weights,
        w_prev,
        params,
        turnover_meta=turnover_meta,
        clip_mass_zeroed=clip_mass_zeroed,
    )
    diagnostics = _build_diagnostics(
        weights,
        w_prev,
        sigma,
        alpha_hat,
        benchmark_idx,
        status,
        params,
        omega_diag=omega_diag,
        alpha_unc_used=alpha_unc_used,
        alpha_unc_meta=alpha_unc_meta,
    )
    diagnostics.update(governor)
    diagnostics.update(tcost.diagnostics)
    diagnostics.update(adv_cap_meta)
    diagnostics["solver_attempts"] = solver_notes
    # `alpha-engine-config-I10503`: the model in force, on EVERY solve, on the
    # infeasible path too. A cost model recorded only when the solve succeeded
    # is a cost model absent from exactly the artifacts an investigation reads.
    diagnostics["cost_model"] = cost_model.record()
    return OptimizerResult(weights=weights, diagnostics=diagnostics)


def _resolve_alpha_uncertainty(
    alpha_uncertainty_epistemic: np.ndarray | None,
    n_names: int,
    params: PortfolioParams,
) -> tuple[np.ndarray, bool, dict]:
    """Build Ω = diag(σ_ε²) and decide whether the penalty is OPERATIVE.

    Ω is the covariance of the ESTIMATION ERROR of α̂ (Garlappi, Uppal & Wang
    2007). A Bayesian predictive std is ``1/α̂ + xᵀΣ_w x``, whose first term is
    a scalar learned once at fit time and therefore identical for every name in
    a batch. Feeding the total to Ω would both double-count observation risk Σ
    already carries and add a per-batch constant that annihilates the
    cross-section the penalty exists to exploit. So Ω is built from the
    estimation-error half and from nothing else, and it is never re-pointed at
    the total when that half is absent.

    The three ways the term can be inoperative are all RECORDED:

    ``gamma_zero``                the penalty is configured off.
    ``epistemic_field_absent``    no usable estimation-error vector was given.
    ``cross_section_below_floor`` the vector is present but cross-sectionally
                                  flat, i.e. it too is a uniform ridge. A
                                  magnitude test cannot see this: the vector of
                                  a model whose posterior never left its prior
                                  is LARGE as well as flat.

    ``meta`` is populated on EVERY path — a field that appears only when the
    penalty engages is indistinguishable from a penalty that is quietly dead.
    """
    gamma = float(params.alpha_uncertainty_penalty)
    meta: dict = {
        "alpha_uncertainty_vintage": "epistemic",
        "alpha_uncertainty_inoperative_reason": None,
        "alpha_uncertainty_epistemic_cv": None,
        "alpha_uncertainty_min_cv": float(params.alpha_uncertainty_min_cv),
        "alpha_uncertainty_n_usable": 0,
        "alpha_uncertainty_n_negative_coerced": 0,
    }
    if gamma <= 0.0:
        meta["alpha_uncertainty_inoperative_reason"] = "gamma_zero"
        return np.zeros(n_names), False, meta
    if alpha_uncertainty_epistemic is None:
        meta["alpha_uncertainty_inoperative_reason"] = "epistemic_field_absent"
        return np.zeros(n_names), False, meta

    arr = np.asarray(alpha_uncertainty_epistemic, dtype=np.float64).ravel()
    if arr.shape != (n_names,):
        raise ValueError(
            f"alpha_uncertainty_epistemic shape {arr.shape} != ({n_names},) — one entry per ticker"
        )
    # A negative entry is an upstream contract violation: an estimation std is
    # non-negative by construction. Coerced to zero (no penalty for that name)
    # and COUNTED on the artifact, which is the durable surface — a warning in
    # a log is not evidence anyone reads a week later.
    meta["alpha_uncertainty_n_negative_coerced"] = int(np.sum((arr < 0.0) & np.isfinite(arr)))
    arr = np.where(np.isfinite(arr) & (arr >= 0.0), arr, 0.0)
    usable = arr[arr > 0.0]
    meta["alpha_uncertainty_n_usable"] = int(usable.size)
    if usable.size < 2:
        meta["alpha_uncertainty_inoperative_reason"] = "epistemic_field_absent"
        return np.zeros(n_names), False, meta

    cv = float(usable.std() / usable.mean()) if usable.mean() > 0.0 else 0.0
    meta["alpha_uncertainty_epistemic_cv"] = cv
    if cv < meta["alpha_uncertainty_min_cv"]:
        meta["alpha_uncertainty_inoperative_reason"] = "cross_section_below_floor"
        return np.zeros(n_names), False, meta

    return arr**2, True, meta


@dataclass(frozen=True)
class _TCostTerm:
    """The objective's turnover-cost term and its observability."""

    objective_term: object
    diagnostics: dict


def _clean_adv(
    adv_usd: np.ndarray | None,
    n_names: int,
    benchmark_idx: int,
    cash_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize an ADV$ vector into ``(adv, usable_mask)``.

    ``adv`` has NaN / non-positive / non-finite entries coerced to 0.0 (no
    coverage). ``usable_mask`` is True only for real names with ADV > 0: the
    benchmark fill and the cash sleeve carry no market impact and are always
    excluded from both the cost term and the participation constraint.
    """
    if adv_usd is None:
        return np.zeros(n_names), np.zeros(n_names, dtype=bool)
    adv = np.asarray(adv_usd, dtype=np.float64).ravel()
    if adv.shape != (n_names,):
        raise ValueError(f"adv_usd shape {adv.shape} != ({n_names},) — one ADV$ entry per ticker")
    adv = np.where(np.isfinite(adv) & (adv > 0.0), adv, 0.0)
    usable = adv > 0.0
    usable[benchmark_idx] = False
    usable[cash_idx] = False
    return adv, usable


def _resolve_ref_sigma(
    name_sigma: np.ndarray | None,
    usable_mask: np.ndarray,
) -> tuple[np.ndarray | None, float | None]:
    """Return ``(sigma_used, ref_sigma)`` for the volatility scaling.

    ``ref_sigma`` is the cross-sectional MEDIAN σ over usable names — the
    library's self-calibrating reference, at which the median-volatility name
    reproduces the volatility-agnostic cost. ``(None, None)`` when no per-name
    σ is supplied or none is finite and positive, which collapses the scaling
    to 1.0 rather than inventing a reference.
    """
    if name_sigma is None:
        return None, None
    sig = np.asarray(name_sigma, dtype=np.float64).ravel()
    finite_pos = np.isfinite(sig) & (sig > 0.0) & np.asarray(usable_mask, dtype=bool)
    if not np.any(finite_pos):
        return None, None
    ref = float(np.median(sig[finite_pos]))
    if not (ref > 0.0):
        return None, None
    return sig, ref


def _build_tcost_term(
    cp,
    w,
    w_prev: np.ndarray,
    adv_usd: np.ndarray | None,
    portfolio_notional: float | None,
    name_sigma: np.ndarray | None,
    benchmark_idx: int,
    cash_idx: int,
    cost_model: CostModel,
) -> _TCostTerm:
    """Build the objective's turnover-cost term from the NAMED cost model.

    A flat model is an L1 penalty at its round-trip rate. A square-root model
    decomposes the per-name one-side DOLLAR cost into a convex sum cvxpy can
    analyse::

        C_i = (half_spread + commission)/1e4 · NAV · |Δwᵢ|            (linear)
            + impact_coef·(σᵢ/refσ)/1e4 · NAV^{1.5}/√ADVᵢ · |Δwᵢ|^{1.5}

    That is algebraically the library's own ``cost_for_turnover`` with
    ``notional = |Δwᵢ|·NAV`` substituted and the square root distributed; the
    coefficients come from the model the recipe named and the impact LAW comes
    from the library. Only the convex-expression plumbing is local, because
    the library's scalar form calls ``math.sqrt`` on a Python float and cannot
    sit inside a cvxpy expression graph. ``|Δwᵢ|`` and ``|Δwᵢ|^{1.5}`` are both
    convex, so ``−ΣCᵢ/NAV`` is concave and valid inside a Maximize. Dividing by
    NAV keeps the term commensurate with the weight-space alpha and risk terms.

    **There is no degradation path.** A square-root model handed no book
    notional, or no name with ADV coverage, RAISES. The live trader's optimizer
    substitutes a flat penalty there because holding a book for a session is
    the worse outcome; grading is offline, the caller owns every input, and a
    substitution would produce a verdict naming a model that did not charge it
    — which is the defect, not the mitigation.
    """
    n_names = w.shape[0]
    adv, usable = _clean_adv(adv_usd, n_names, benchmark_idx, cash_idx)
    n_usable = int(np.sum(usable))

    if cost_model.kind == "flat":
        rate = cost_model.bps_per_unit_turnover()
        return _TCostTerm(
            # One-way turnover is ‖Δw‖₁/2 and the rate is round-trip, so the
            # L1 norm and the round-trip rate compose without a further factor.
            objective_term=-(rate / 1e4) * cp.norm(w - w_prev, 1) / 2.0,
            diagnostics={
                "tcost_term_kind": "flat",
                "tcost_round_trip_bps": float(rate),
                "tcost_n_names_with_adv": n_usable,
            },
        )

    if portfolio_notional is None or not (float(portfolio_notional) > 0.0):
        raise CostModelInputError(
            f"cost model {cost_model.name!r} prices participation and needs the book "
            "notional to turn a weight delta into a trade size; none was given. The "
            "objective refuses rather than substituting a flat penalty, which would "
            "charge the solve by a model the recipe did not name."
        )
    if n_usable == 0:
        raise CostModelInputError(
            f"cost model {cost_model.name!r} prices each name against its average "
            "daily dollar volume and no name in this universe has ADV coverage. "
            "Grading is offline and the caller owns this input; an absent one is a "
            "caller defect, not a market condition."
        )

    model = cost_model.impact_model()
    nav = float(portfolio_notional)
    sig_used, ref_sigma = _resolve_ref_sigma(name_sigma, usable)

    # The linear half — the spread and commission floor — applies to every name
    # that trades, ADV-covered or not: it is not an impact term.
    linear_bps = model.half_spread_bps + model.commission_bps
    dw = w - w_prev
    linear_cost = (linear_bps / 1e4) * nav * cp.abs(dw)

    impact_terms = []
    for i in np.where(usable)[0]:
        sigma_scale = 1.0
        if sig_used is not None and ref_sigma:
            s = sig_used[i]
            if np.isfinite(s) and s > 0.0:
                sigma_scale = float(s) / ref_sigma
        k_i = model.impact_coef_bps * sigma_scale / 1e4 * (nav**1.5) / math.sqrt(adv[i])
        if k_i > 0.0:
            impact_terms.append(k_i * cp.power(cp.abs(dw[i]), 1.5))

    total_cost = cp.sum(linear_cost)
    if impact_terms:
        total_cost = total_cost + cp.sum(impact_terms)
    return _TCostTerm(
        objective_term=-total_cost / nav,
        diagnostics={
            "tcost_term_kind": "sqrt_impact",
            "tcost_n_names_with_adv": n_usable,
            "tcost_impact_coef_bps": float(model.impact_coef_bps),
            "tcost_half_spread_bps": float(model.half_spread_bps),
            "tcost_commission_bps": float(model.commission_bps),
            "tcost_sigma_scaled": bool(sig_used is not None and ref_sigma),
            "tcost_ref_sigma": None if ref_sigma is None else float(ref_sigma),
            "tcost_portfolio_notional": nav,
        },
    )


def _apply_max_pct_adv_constraint(
    w,
    w_prev: np.ndarray,
    constraints: list,
    adv_usd: np.ndarray | None,
    portfolio_notional: float | None,
    benchmark_idx: int,
    cash_idx: int,
    params: PortfolioParams,
) -> dict:
    """Append the per-name participation cap, in place.

    ``|wᵢ − w_prevᵢ|·NAV ≤ max_pct_adv·ADVᵢ`` for every real name with usable
    ADV coverage, encoded as the affine pair ``Δwᵢ ≤ bᵢ``, ``−Δwᵢ ≤ bᵢ`` so the
    program stays convex. Returns diagnostics recording whether the cap is live
    and over how many names — a constraint whose absence is invisible is a
    constraint nobody can tell is working.

    ``max_pct_adv = null`` is an EXPLICIT disable and is recorded as one. A
    name with no ADV coverage is exempt, because no participation bound can be
    formed for it; that is a reported exemption, not a silent one.
    """
    n_names = w.shape[0]
    adv, usable = _clean_adv(adv_usd, n_names, benchmark_idx, cash_idx)
    n_usable = int(np.sum(usable))
    cap = params.max_pct_adv
    if cap is None or not (float(cap) > 0.0):
        return {"max_pct_adv_applied": False, "max_pct_adv_reason": "disabled"}
    if portfolio_notional is None or not (float(portfolio_notional) > 0.0):
        return {"max_pct_adv_applied": False, "max_pct_adv_reason": "no_portfolio_notional"}
    if n_usable == 0:
        return {"max_pct_adv_applied": False, "max_pct_adv_reason": "no_adv_coverage"}

    nav = float(portfolio_notional)
    cap = float(cap)
    idx = np.where(usable)[0]
    bounds = cap * adv[idx] / nav
    dw = w[idx] - w_prev[idx]
    constraints.append(dw <= bounds)
    constraints.append(-dw <= bounds)
    return {
        "max_pct_adv_applied": True,
        "max_pct_adv": cap,
        "max_pct_adv_reason": "applied",
        "max_pct_adv_n_names_constrained": int(len(idx)),
        "max_pct_adv_min_bound_weight": float(np.min(bounds)) if len(bounds) else None,
    }


def _validate_inputs(
    tickers: list[str],
    alpha_hat: np.ndarray,
    returns_panel: np.ndarray | None,
    w_prev: np.ndarray,
    sectors: list[str],
    stance_caps: np.ndarray,
    eligibility: np.ndarray,
    benchmark_idx: int,
    cash_idx: int,
    covariance_provided: bool = False,
) -> None:
    n_names = len(tickers)
    if n_names == 0:
        raise ValueError("empty universe — there is no portfolio to construct")
    for name, arr in (
        ("alpha_hat", alpha_hat),
        ("w_prev", w_prev),
        ("stance_caps", stance_caps),
        ("eligibility", eligibility),
    ):
        if arr.shape != (n_names,):
            raise ValueError(f"{name} shape {arr.shape} != ({n_names},)")
    if not covariance_provided:
        if returns_panel is None:
            raise ValueError("returns_panel is required when covariance is not provided")
        if returns_panel.ndim != 2 or returns_panel.shape[1] != n_names:
            raise ValueError(
                f"returns_panel shape {returns_panel.shape} incompatible with N={n_names}"
            )
    if len(sectors) != n_names:
        raise ValueError(f"sectors length {len(sectors)} != N={n_names}")
    if not (0 <= benchmark_idx < n_names) or not (0 <= cash_idx < n_names):
        raise ValueError(
            f"benchmark_idx={benchmark_idx} cash_idx={cash_idx} out of range [0,{n_names})"
        )
    if benchmark_idx == cash_idx:
        raise ValueError(
            "benchmark_idx and cash_idx name the same position; the benchmark fill and "
            "the cash sleeve are different sleeves and pinning one pins the other"
        )
    if not eligibility[benchmark_idx]:
        raise ValueError("the benchmark must be eligible — it is the no-conviction fill")
    if not eligibility[cash_idx]:
        raise ValueError("cash must be eligible — the sleeve is an equality pin")


def _ewma_covariance(returns: np.ndarray, lambda_decay: float) -> np.ndarray:
    """Exponentially weighted covariance under a zero-mean assumption.

    ``Σ = (1−λ)·Σₖ λᵏ·r_{t−k}r_{t−k}ᵀ``, normalized so the weights sum to one
    over the finite window. The zero-mean simplification is standard for daily
    equity returns, where the mean is small against the volatility.

    Degenerate at ``λ = 1``: the weights become uniform and the estimator
    reduces to the sample covariance up to the ``1/T`` versus ``1/(T−1)``
    factor.
    """
    if not 0.5 <= lambda_decay <= 1.0:
        raise ValueError(f"ewma_lambda_decay must be in [0.5, 1.0]; got {lambda_decay}")
    n_rows = returns.shape[0]
    if lambda_decay >= 1.0 - 1e-12:
        return (returns.T @ returns) / n_rows
    # Newest observation first, so row 0 carries the largest weight.
    reversed_rows = returns[::-1]
    weights = (1.0 - lambda_decay) * lambda_decay ** np.arange(n_rows)
    weights /= weights.sum()
    return (reversed_rows.T * weights) @ reversed_rows


def _validate_covariance(cov: np.ndarray, n_names: int) -> np.ndarray:
    """Validate and symmetrize an injected DAILY covariance.

    A JSON round trip introduces tiny asymmetry and non-PSD perturbation.
    Symmetrize, and fail LOUD on a shape mismatch, a non-finite entry, or a
    materially negative eigenvalue: a silently mis-shaped Σ corrupts the
    vol-target constraint and every volatility diagnostic at once.
    """
    cov = np.asarray(cov, dtype=float)
    if cov.shape != (n_names, n_names):
        raise ValueError(f"covariance shape {cov.shape} != ({n_names}, {n_names})")
    if not np.all(np.isfinite(cov)):
        raise ValueError("covariance contains non-finite entries")
    cov = 0.5 * (cov + cov.T)
    min_eig = float(np.linalg.eigvalsh(cov).min())
    tol = -1e-8 * max(1.0, float(np.trace(cov)) / n_names)
    if min_eig < tol:
        raise ValueError(f"covariance is not PSD (min eigenvalue {min_eig:.3e} < tol {tol:.3e})")
    return cov


def _estimate_covariance_daily(returns_panel: np.ndarray, params: PortfolioParams) -> np.ndarray:
    """Estimate the DAILY covariance, before horizon scaling.

    ``ledoit_wolf`` calls `nousergon_lib.quant.factor_risk.ledoit_wolf_cov` —
    the fleet's one shrinkage implementation, and a pure-numpy one, so this
    repository carries no compiled estimator dependency to reach the
    institutional default.
    """
    clean = returns_panel[~np.isnan(returns_panel).any(axis=1)]
    if clean.shape[0] < _MIN_COVARIANCE_ROWS:
        raise ValueError(
            f"need >= {_MIN_COVARIANCE_ROWS} clean return rows for a covariance "
            f"estimate; got {clean.shape[0]}. Shrinkage narrows sampling noise on a "
            "short panel, it does not remove it."
        )
    estimator = params.covariance_shrinkage
    if estimator == "ledoit_wolf":
        from nousergon_lib.quant.factor_risk import ledoit_wolf_cov

        return ledoit_wolf_cov(clean, shrinkage="ledoit_wolf")
    if estimator == "sample":
        return np.cov(clean, rowvar=False)
    if estimator == "ewma":
        return _ewma_covariance(clean, params.ewma_lambda_decay)
    raise PortfolioParamsError(
        f"unknown covariance_shrinkage {estimator!r}; the supported estimators are "
        f"{sorted(COVARIANCE_ESTIMATORS)}"
    )


def _estimate_covariance(returns_panel: np.ndarray, params: PortfolioParams) -> np.ndarray:
    """The covariance at horizon ``sigma_horizon_days``.

    Under an i.i.d. log-return assumption ``Σ_H = H · Σ_daily``. H is a
    declared parameter: a horizon left implicit at 1 silently measures a
    one-day risk against a multi-day alpha.
    """
    return params.sigma_horizon_days * _estimate_covariance_daily(returns_panel, params)


def _real_sectors(sectors: list[str]) -> set[str]:
    """The sector labels a cap applies to — sentinels of the form ``__x__`` excluded."""
    return {s for s in sectors if not (s.startswith("__") and s.endswith("__"))}


def _solve_with_fallback(problem, w) -> tuple[np.ndarray | None, str, list[dict]]:
    """Solve with the primary conic solver, then the declared alternatives.

    Returns ``(weights, status, attempts)``. ``attempts`` records every solver
    tried and what it returned, so a book solved by the second choice is not
    indistinguishable from one solved by the first.
    """
    import cvxpy as cp

    attempts: list[dict] = []
    installed = cp.installed_solvers()
    for solver in (_CLARABEL, *_FALLBACK_SOLVERS):
        if solver not in installed:
            attempts.append({"solver": solver, "outcome": "not_installed"})
            continue
        try:
            problem.solve(solver=solver)
        except (cp.error.SolverError, ValueError) as exc:
            attempts.append({"solver": solver, "outcome": "raised", "detail": repr(exc)})
            continue
        attempts.append({"solver": solver, "outcome": problem.status})
        if problem.status in ("optimal", "optimal_inaccurate"):
            return np.asarray(w.value, dtype=float), problem.status, attempts
    return None, problem.status or "no_solver_available", attempts


def _fallback_weights(
    w_prev: np.ndarray,
    cash_idx: int,
    cash_sleeve_pct: float,
) -> np.ndarray:
    """The held book with cash absorbing the residual — the infeasible path."""
    weights = np.maximum(w_prev.copy(), 0.0)
    weights[cash_idx] = 0.0
    equity_sum = weights.sum()
    target_equity = 1.0 - cash_sleeve_pct
    if equity_sum > 0:
        weights *= target_equity / equity_sum
    weights[cash_idx] = cash_sleeve_pct
    return weights


def _clip_and_renormalize(
    weights: np.ndarray,
    effective_caps: np.ndarray,
    cash_idx: int,
    params: PortfolioParams,
) -> tuple[np.ndarray, float]:
    """Clip to the box, drop sub-``min_position_pct`` dust, renormalize.

    Returns ``(weights, mass_zeroed)``. ``mass_zeroed`` is not bookkeeping: it
    is the exact budget by which this post-solve step may legitimately push the
    vector past the solver's turnover constraint — zeroing mass ``m`` adds at
    most ``m`` of one-way turnover from the drop and at most ``m`` more from
    renormalizing the survivors up by ``1/(1−m)``. The governor uses it as its
    assertion tolerance, so the tolerance is DERIVED from what happened rather
    than being an epsilon chosen wide enough to absorb a real breach.
    """
    weights = np.maximum(weights, 0.0)
    weights = np.minimum(weights, effective_caps + 1e-8)
    small = (weights < params.min_position_pct) & (np.arange(len(weights)) != cash_idx)
    mass_zeroed = float(np.sum(weights[small]))
    weights = np.where(small, 0.0, weights)
    total = weights.sum()
    if total > 0:
        weights = weights / total
    return weights, mass_zeroed


def _mandatory_turnover_floor(
    w_prev: np.ndarray,
    effective_caps: np.ndarray,
    cash_idx: int,
    params: PortfolioParams,
) -> float:
    """One-way turnover the OTHER constraints force, regardless of alpha.

    ``w_prev`` has zero turnover and so always satisfies the turnover
    constraint itself — but not necessarily the rest of the program: a held
    name that went ineligible is pinned to zero, and the cash sleeve is pinned
    to its target. Those pins MANDATE movement, and if the mandated movement
    exceeds the budget the program is infeasible and the whole book falls to
    the hold path — a new failure mode, introduced by the budget, on exactly
    the day a forced exit is what must happen.

    So the budget governs DISCRETIONARY trading only. This bounds the forced
    movement; the caller raises the constraint's right-hand side to it when it
    is larger, which makes the feasible set non-empty BY CONSTRUCTION:

    * ``dᵢ`` is the distance from ``w_prevᵢ`` to its box, so projecting into
      the box costs ``Σdᵢ`` of L1;
    * the projection need not sum to one, and the residual must be absorbed by
      names with slack, costing a further ``|r|`` of L1.

    ``(Σd + |r|)/2`` is therefore an attainable one-way turnover for a point
    satisfying the box, the sleeve pin and the budget identity.
    """
    lower = np.zeros_like(w_prev)
    upper = np.array(effective_caps, dtype=float)
    sleeve = float(params.cash_sleeve_pct)
    lower[cash_idx] = sleeve
    upper[cash_idx] = sleeve
    projected = np.clip(w_prev, lower, upper)
    forced_l1 = float(np.sum(np.abs(projected - w_prev)))
    residual = abs(1.0 - float(projected.sum()))
    return (forced_l1 + residual) / 2.0


def _decompose_mandatory_turnover(
    w_prev: np.ndarray,
    effective_caps: np.ndarray,
    eligibility: np.ndarray,
    cash_idx: int,
    params: PortfolioParams,
) -> dict:
    """Which constraint forced each unit of the mandatory floor.

    "How much of today's trading was forced" has a different fix for each of
    its three causes, so the total on its own is not actionable:

    * ``cash_sleeve_pin`` — the sleeve is an equality pin and drift into or out
      of it is mandated every session. Not a defect.
    * ``ineligibility_pin`` — a held name went ineligible and is pinned to
      zero. A forced exit is the system working.
    * ``position_cap`` — a held name sits above its per-name cap. Alpha-
      independent, and the one of the three whose right response is a design
      question rather than an acknowledgement.

    Emitted on every solve, healthy included: a component emitting nothing is
    not healthy, it is unobserved.
    """
    sleeve = float(params.cash_sleeve_pct)
    lower = np.zeros_like(w_prev)
    upper = np.array(effective_caps, dtype=float)
    lower[cash_idx] = sleeve
    upper[cash_idx] = sleeve
    projected = np.clip(w_prev, lower, upper)
    per_name = np.abs(projected - w_prev)

    idx = np.arange(len(w_prev))
    is_cash = idx == cash_idx
    # An ineligible name carries an effective cap of zero, so its whole holding
    # is forced out. Attributing that to the position cap would read as a
    # sizing artifact when it is a deliberate exit, and the two have opposite
    # responses.
    is_ineligible = (~np.asarray(eligibility, dtype=bool)) & (~is_cash)

    sleeve_l1 = float(per_name[is_cash].sum())
    ineligible_l1 = float(per_name[is_ineligible].sum())
    cap_l1 = float(per_name[~is_cash & ~is_ineligible].sum())
    residual = abs(1.0 - float(projected.sum()))

    over_cap = [
        int(i) for i in idx if not is_cash[i] and not is_ineligible[i] and per_name[i] > 1e-9
    ]
    pinned_out = [int(i) for i in idx if is_ineligible[i] and per_name[i] > 1e-9]

    return {
        "cash_sleeve_pin": sleeve_l1 / 2.0,
        "ineligibility_pin": ineligible_l1 / 2.0,
        "position_cap": cap_l1 / 2.0,
        "renormalization": residual / 2.0,
        "total": (sleeve_l1 + ineligible_l1 + cap_l1 + residual) / 2.0,
        "n_names_over_cap": len(over_cap),
        "n_names_pinned_out": len(pinned_out),
    }


def compute_conviction_budget_multiplier(
    alpha_hat: np.ndarray,
    alpha_uncertainty: np.ndarray | None,
    eligibility: np.ndarray | None,
    benchmark_idx: int,
    cash_idx: int,
    params: PortfolioParams,
) -> dict:
    """A signal-quality multiplier on the DISCRETIONARY turnover budget.

    A turnover budget answers "how far may the book move today". It has never
    answered "is today's target worth moving toward at all", and that gap is
    what lets an optimizer sit at its cap for weeks while ranking names its own
    model cannot distinguish. The gate closes it by scaling the discretionary
    budget by a measured quality multiplier ``q ∈ [min_multiple, 1]``::

        IR_xs = stdev_cross_section(α̂_eligible) / median(σ_α̂_eligible)
        q     = clip((IR_xs − ir_floor) / (ir_full − ir_floor), min_multiple, 1)

    ``IR_xs`` is the cross-sectional information ratio of the alpha vector —
    how large the spread being ranked on is, relative to the error bar the
    model itself puts on each element of it. Below the floor the names are
    statistically tied and rebalancing between them is a pure transaction cost.

    The MANDATORY floor is unaffected; a hard exit is not discretionary and is
    never starved of budget by this gate.

    The block is returned on EVERY solve — gate on, gate off, gate inevaluable
    — because a field that appears only when the gate engages is
    indistinguishable from a dead gate. It never raises: every degradation
    returns ``q = 1.0``, the UNTHROTTLED budget, with a reason. Missing data
    must never produce a tighter budget than the operator configured; a gate
    that silently stops the book on an input outage is a worse failure than the
    churn it exists to stop.
    """
    out: dict = {
        "conviction_gate_applied": False,
        "conviction_ir_xs": None,
        "conviction_alpha_dispersion": None,
        "conviction_alpha_noise": None,
        "conviction_n_names": 0,
        "conviction_budget_multiplier": 1.0,
        "conviction_gate_reason": "disabled",
    }
    if not params.conviction_budget_gate_enabled:
        return out
    if alpha_uncertainty is None:
        out["conviction_gate_reason"] = "no_alpha_uncertainty_vector"
        return out

    alpha = np.asarray(alpha_hat, dtype=float)
    sigma = np.asarray(alpha_uncertainty, dtype=float)
    n_names = alpha.shape[0]
    if sigma.shape[0] != n_names:
        out["conviction_gate_reason"] = "alpha_uncertainty_length_mismatch"
        return out

    # Discretionary names only: the benchmark is the fill and cash is the
    # sleeve, neither carries a predicted alpha, and both would drag the
    # dispersion toward a number that says nothing about the ranking traded on.
    mask = np.ones(n_names, dtype=bool)
    for i in (benchmark_idx, cash_idx):
        if 0 <= i < n_names:
            mask[i] = False
    if eligibility is not None:
        elig = np.asarray(eligibility, dtype=bool)
        if elig.shape[0] == n_names:
            mask &= elig
    mask &= np.isfinite(alpha)

    min_names = max(2, int(params.conviction_gate_min_names))
    if int(mask.sum()) < min_names:
        out["conviction_n_names"] = int(mask.sum())
        out["conviction_gate_reason"] = "too_few_discretionary_names"
        return out

    sig_ok = mask & np.isfinite(sigma) & (sigma > 0)
    if int(sig_ok.sum()) < min_names:
        out["conviction_n_names"] = int(mask.sum())
        out["conviction_gate_reason"] = "no_usable_alpha_uncertainty"
        return out

    dispersion = float(np.std(alpha[mask]))
    noise = float(np.median(sigma[sig_ok]))
    out["conviction_alpha_dispersion"] = dispersion
    out["conviction_alpha_noise"] = noise
    out["conviction_n_names"] = int(mask.sum())
    if not np.isfinite(dispersion) or not np.isfinite(noise) or noise <= 0:
        out["conviction_gate_reason"] = "non_finite_statistic"
        return out

    ir = dispersion / noise
    out["conviction_ir_xs"] = float(ir)

    lo = float(params.conviction_ir_floor)
    hi = float(params.conviction_ir_full)
    q_min = float(params.conviction_budget_min_multiple)
    q = (ir - lo) / (hi - lo)
    q = float(min(max(q, q_min), 1.0))
    out["conviction_budget_multiplier"] = q
    out["conviction_gate_applied"] = bool(q < 1.0)
    out["conviction_gate_reason"] = (
        "signal_quality_ok" if q >= 1.0 else "alpha_spread_below_own_noise"
    )
    return out


def _apply_turnover_constraint(
    cp,
    w,
    w_prev: np.ndarray,
    constraints: list,
    effective_caps: np.ndarray,
    cash_idx: int,
    params: PortfolioParams,
    eligibility: np.ndarray | None = None,
    *,
    alpha_hat: np.ndarray | None = None,
    alpha_uncertainty: np.ndarray | None = None,
    benchmark_idx: int | None = None,
) -> dict:
    """Append the L1 daily-turnover budget to ``constraints``.

    Returns the metadata the diagnostics and the post-solve assertion both
    read: the configured cap, the mandatory floor, and the EFFECTIVE
    right-hand side actually imposed. ``max_daily_turnover = null`` disables
    the budget.
    """
    cap = params.max_daily_turnover
    # The conviction block is computed and emitted even when the budget is off:
    # "how good was today's signal" is a fact in its own right, and a statistic
    # that exists only on the throttled path cannot be used to judge whether
    # the throttle was right.
    conviction = (
        compute_conviction_budget_multiplier(
            alpha_hat,
            alpha_uncertainty,
            eligibility,
            benchmark_idx if benchmark_idx is not None else -1,
            cash_idx,
            params,
        )
        if alpha_hat is not None
        else {
            "conviction_gate_applied": False,
            "conviction_ir_xs": None,
            "conviction_alpha_dispersion": None,
            "conviction_alpha_noise": None,
            "conviction_n_names": 0,
            "conviction_budget_multiplier": 1.0,
            "conviction_gate_reason": "no_alpha_vector_supplied",
        }
    )
    meta: dict = {
        "turnover_constraint_applied": False,
        "turnover_constraint_cap": None,
        "turnover_budget_configured": None if cap is None else float(cap),
        "turnover_budget_discretionary": None,
        "turnover_mandatory_floor": None,
        "turnover_mandatory_floor_by_cause": None,
        "turnover_constraint": None,
        **conviction,
    }
    if cap is None or cap <= 0:
        return meta
    floor = _mandatory_turnover_floor(w_prev, effective_caps, cash_idx, params)
    by_cause = (
        _decompose_mandatory_turnover(w_prev, effective_caps, eligibility, cash_idx, params)
        if eligibility is not None
        else None
    )
    # The gate scales the DISCRETIONARY budget only. The mandatory floor is
    # applied after it, so a throttled budget can never starve a forced exit —
    # the `max` keeps the two kinds of trading ordered correctly. The small
    # slack above the floor is there because the floor bounds an ATTAINABLE
    # point, and pinning the right-hand side exactly to it leaves a feasible
    # set of measure near zero that an interior-point solver reports infeasible.
    q = float(conviction["conviction_budget_multiplier"])
    discretionary = float(cap) * q
    meta["turnover_budget_discretionary"] = discretionary
    effective_cap = max(discretionary, floor * (1.0 + 1e-6) + 1e-9)
    constraint = cp.norm(w - w_prev, 1) / 2 <= effective_cap
    constraints.append(constraint)
    meta.update(
        {
            "turnover_constraint_applied": True,
            "turnover_constraint_cap": float(effective_cap),
            "turnover_mandatory_floor": float(floor),
            "turnover_mandatory_floor_by_cause": by_cause,
            "turnover_budget_raised_above_configured": bool(effective_cap > float(cap) + 1e-9),
            "turnover_constraint": constraint,
        }
    )
    return meta


def _turnover_diagnostics(weights: np.ndarray, w_prev: np.ndarray, turnover_meta: dict) -> dict:
    """Turnover observability, emitted on EVERY solve.

    Including the infeasible-fallback path and the budget-disabled path: a
    field that appears only on the interesting path is indistinguishable from
    a dead emitter.

    ``turnover_constraint_binding`` is the instrument that says the budget
    bound the solve, and ``turnover_constraint_shadow_price`` is the
    constraint's dual — the marginal objective value of one more unit of
    budget, i.e. exactly what the restraint cost.
    """
    executed = float(np.sum(np.abs(weights - w_prev)) / 2)
    cap = turnover_meta.get("turnover_constraint_cap")
    out: dict = {
        "requested_turnover_one_way": executed,
        "turnover_constraint_applied": bool(turnover_meta.get("turnover_constraint_applied")),
        "turnover_constraint_cap": cap,
        "turnover_mandatory_floor": turnover_meta.get("turnover_mandatory_floor"),
        "turnover_mandatory_floor_by_cause": turnover_meta.get("turnover_mandatory_floor_by_cause"),
        "turnover_constraint_binding": False,
        "turnover_constraint_shadow_price": None,
        "turnover_budget_configured": turnover_meta.get("turnover_budget_configured"),
        "turnover_budget_discretionary": turnover_meta.get("turnover_budget_discretionary"),
        "conviction_gate_applied": turnover_meta.get("conviction_gate_applied", False),
        "conviction_ir_xs": turnover_meta.get("conviction_ir_xs"),
        "conviction_alpha_dispersion": turnover_meta.get("conviction_alpha_dispersion"),
        "conviction_alpha_noise": turnover_meta.get("conviction_alpha_noise"),
        "conviction_n_names": turnover_meta.get("conviction_n_names", 0),
        "conviction_budget_multiplier": turnover_meta.get("conviction_budget_multiplier", 1.0),
        "conviction_gate_reason": turnover_meta.get("conviction_gate_reason"),
    }
    if cap is not None:
        constraint = turnover_meta.get("turnover_constraint")
        dual = getattr(constraint, "dual_value", None) if constraint is not None else None
        if dual is not None:
            try:
                out["turnover_constraint_shadow_price"] = float(np.ravel(dual)[0])
            except (TypeError, ValueError, IndexError):
                # A solver that returned a dual in a shape this cannot read is
                # recorded as having produced none, and the binding test below
                # falls back to the primal and NAMES that it did. Swallowed
                # here because the failure mode is "no dual available", which
                # `turnover_binding_test` states on the artifact.
                out["turnover_constraint_shadow_price"] = None
        # COMPLEMENTARY SLACKNESS, both halves. A constraint is active iff its
        # dual is strictly positive AND its primal sits at the bound. Testing
        # one half is wrong in a different direction each way: the primal alone
        # misses a genuinely binding budget, because `weights` here is
        # post-clip-and-renormalize and its turnover is basis points off the
        # solver's own; the dual alone reports binding on numerical dust, which
        # an interior-point solver returns for an INACTIVE constraint.
        shadow = out["turnover_constraint_shadow_price"]
        at_bound = bool(executed >= float(cap) * (1.0 - _PRIMAL_AT_BOUND_REL_TOL) - 1e-6)
        if shadow is not None:
            out["turnover_constraint_binding"] = bool(shadow > _DUAL_ACTIVE_TOL and at_bound)
            out["turnover_binding_test"] = "complementary_slackness"
        else:
            out["turnover_constraint_binding"] = at_bound
            out["turnover_binding_test"] = "primal_only_no_dual"
    out["turnover_capped"] = out["turnover_constraint_binding"]
    return out


def _apply_turnover_governor(
    weights: np.ndarray,
    w_prev: np.ndarray,
    params: PortfolioParams,
    *,
    turnover_meta: dict | None = None,
    clip_mass_zeroed: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """Post-solve ASSERTION that the daily turnover budget held.

    This function MEASURES and RAISES; it never modifies ``weights``. The
    budget is a constraint inside the convex program, so the solver returns a
    vector that already satisfies it, and the only legitimate source of extra
    turnover is the clip-and-renormalize step — whose whole footprint is
    bounded by ``clip_mass_zeroed`` of one-way turnover. Anything beyond that
    plus solver slack is unexplained, and unexplained is what this raises on.
    """
    meta = turnover_meta or {}
    requested = float(np.sum(np.abs(weights - w_prev)) / 2)
    flag = params.large_move_turnover_flag
    gov = _turnover_diagnostics(weights, w_prev, meta)
    binding = bool(gov["turnover_constraint_binding"])
    above_flag = bool(flag is not None and requested > float(flag))
    gov["large_move_flagged"] = bool(above_flag or binding)
    # Why it was flagged, so a reader can act on the true thing. Under the
    # constraint construction the executed turnover can no longer exceed the
    # flag on a capped day, so a flag driven only by the raw comparison would
    # go permanently silent — a detector killed by a fix is a worse outcome
    # than the fix is good. `binding` is the honest successor signal: the
    # optimizer wanted to move more than the budget allowed.
    #
    # When the conviction gate is throttling, a binding budget is the guard
    # DOING ITS JOB rather than a large move. Flagging that would produce an
    # alert that fires every session on a healthy state. The fact is not lost:
    # the whole conviction block is in the diagnostics.
    gate_on = bool(gov.get("conviction_gate_applied"))
    if above_flag:
        gov["large_move_reason"] = "executed_turnover_above_flag"
    elif binding and gate_on:
        gov["large_move_reason"] = "conviction_throttled_budget_binding"
        gov["large_move_flagged"] = bool(above_flag)
    elif binding:
        gov["large_move_reason"] = "turnover_budget_binding"
    else:
        gov["large_move_reason"] = None

    cap = meta.get("turnover_constraint_cap")
    if cap is not None:
        tolerance = float(clip_mass_zeroed) + 1e-6
        if requested > float(cap) + tolerance:
            raise TurnoverBudgetError(
                f"solved one-way turnover {requested:.6f} exceeds the daily budget "
                f"{float(cap):.6f} by more than the post-solve clip can account for "
                f"(clip zeroed {float(clip_mass_zeroed):.6f} of weight, tolerance "
                f"{tolerance:.6f}). The budget is a constraint inside the convex "
                "program, so this is a solver or parameter defect, not a market "
                "condition."
            )
    return weights, gov


def _build_diagnostics(
    weights: np.ndarray,
    w_prev: np.ndarray,
    sigma: np.ndarray,
    alpha_hat: np.ndarray,
    benchmark_idx: int,
    status: str,
    params: PortfolioParams,
    *,
    omega_diag: np.ndarray | None = None,
    alpha_unc_used: bool = False,
    alpha_unc_meta: dict | None = None,
) -> dict:
    # sigma is at horizon H, so Var_ann = Var_H · (252/H).
    horizon = params.sigma_horizon_days
    horizon_var = max(float(weights @ sigma @ weights), 0.0)
    vol_ann = float(np.sqrt((252 / horizon) * horizon_var))
    benchmark_only = np.zeros_like(weights)
    benchmark_only[benchmark_idx] = 1.0 - params.cash_sleeve_pct
    active_share = float(np.sum(np.abs(weights - benchmark_only)) / 2)
    n_active = int(np.sum(weights > params.min_position_pct))
    turnover = float(np.sum(np.abs(weights - w_prev)) / 2)
    out = {
        "status": status,
        "portfolio_vol_ann": vol_ann,
        "active_share_vs_benchmark": active_share,
        "n_active_positions": n_active,
        "turnover_one_way": turnover,
        "expected_alpha": float(weights @ alpha_hat),
        "weight_sum": float(weights.sum()),
        "alpha_uncertainty_penalty_used": alpha_unc_used,
        **(alpha_unc_meta or {}),
    }
    if omega_diag is not None and np.any(omega_diag > 0.0):
        active_mask = weights > params.min_position_pct
        active_omega = omega_diag[active_mask]
        if active_omega.size > 0:
            out["mean_alpha_std_active"] = float(np.sqrt(active_omega.mean()))
            out["alpha_uncertainty_penalty_contribution"] = float(
                params.alpha_uncertainty_penalty * (omega_diag @ (weights**2))
            )
    return out


def make_cash_sentinel_returns(n_rows: int) -> np.ndarray:
    """The cash column of a returns panel: zero return at the sleeve.

    A caller's helper. Cash must occupy a real column of the panel so the
    covariance estimate has one entry per ticker, and its returns are zero
    rather than absent — an absent column would make the panel's width
    disagree with the universe's length, which the input validation refuses.
    """
    return np.zeros(n_rows)


# ---------------------------------------------------------------------------
# Evidence. `alpha-engine-config-I10500` deliverable 1, `-I10503` in full.
# ---------------------------------------------------------------------------

PORTFOLIO_EVIDENCE_SCHEMA_VERSION = "portfolio_construction.v1"

#: The metric row name the evidence rides on. A gate clause matches this, so
#: it is a constant here rather than a literal at the reader.
PORTFOLIO_METRIC_NAME = "portfolio_construction"

_SCHEMA_PATH = Path(__file__).parent / "schemas" / f"{PORTFOLIO_EVIDENCE_SCHEMA_VERSION}.json"


def _evidence_validator() -> Draft202012Validator:
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def params_digest(params: PortfolioParams) -> str:
    """A stable digest over a parameter set.

    Canonical JSON with sorted keys, so two processes that read the same file
    agree. The digest is an equality test and never the only copy: the values
    travel inline beside it on every artifact, because an artifact that carries
    only a content address is unreadable the moment the addressed content moves
    — and nothing about that failure is loud.
    """
    import hashlib

    canonical = json.dumps(params.to_dict(), sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def portfolio_evidence(
    *,
    trading_day: str,
    arm_id: str,
    params: PortfolioParams,
    cost_model: CostModel,
    diagnostics: Mapping[str, Any],
    sessions: int,
    turnover_one_way_total: float,
    cost_bps_total: float,
) -> dict[str, Any]:
    """Build and VALIDATE the construction evidence for one graded arm.

    The document a phase-3 clause reads to answer "did `crucible.portfolio`
    build the book this grade was taken over, and what priced it". Validated
    against its own schema before it is returned, so a malformed record fails
    at the producer rather than at whichever consumer reads it first.

    All four named components are recorded with a reason whether or not they
    engaged: a component absent from the evidence is indistinguishable from
    one that never ran.
    """
    if sessions < 1:
        raise ValueError(
            f"portfolio evidence over {sessions} sessions; a construction that spanned "
            "no session produced no evidence and must not be recorded as though it had"
        )
    adv_applied = bool(diagnostics.get("max_pct_adv_applied"))
    adv_reason = diagnostics.get("max_pct_adv_reason")
    governor_applied = bool(diagnostics.get("turnover_constraint_applied"))
    document = {
        "schema_version": PORTFOLIO_EVIDENCE_SCHEMA_VERSION,
        "trading_day": trading_day,
        "arm_id": arm_id,
        "engine": "crucible.portfolio",
        "cost_model": cost_model.record(),
        "params": params.to_dict(),
        "params_digest": params_digest(params),
        "components": {
            "mvo": {"applied": True, "reason": str(diagnostics.get("status", "unknown"))},
            "turnover_governor": {
                "applied": governor_applied,
                "reason": (
                    "budget_enforced_as_constraint"
                    if governor_applied
                    else "max_daily_turnover_disabled"
                ),
            },
            "cost_model": {
                "applied": True,
                "reason": str(diagnostics.get("tcost_term_kind", cost_model.kind)),
            },
            "adv_cap": {
                "applied": adv_applied,
                "reason": None if adv_reason is None else str(adv_reason),
            },
        },
        "sessions": int(sessions),
        "turnover_one_way_total": float(turnover_one_way_total),
        "cost_bps_total": float(cost_bps_total),
        "solver_status": str(diagnostics.get("status", "unknown")),
    }
    errors = sorted(_evidence_validator().iter_errors(document), key=lambda e: list(e.path))
    if errors:
        paths = "; ".join(
            f"{'/'.join(str(part) for part in e.path) or '<root>'}: {e.message}" for e in errors
        )
        raise ValueError(
            f"portfolio construction evidence does not satisfy "
            f"{PORTFOLIO_EVIDENCE_SCHEMA_VERSION}: {paths}"
        )
    return document


def portfolio_metric_record(evidence: Mapping[str, Any], *, now_utc: str) -> dict[str, Any]:
    """The manifest metric row carrying the construction evidence.

    `run_manifest.v2`'s metric rows are open, which is what lets a structured
    record ride on one; the required core is filled here so a job body records
    it with one call and cannot get the shape wrong.

    ``now_utc`` is passed rather than read from a clock: a timestamp a function
    invents is a timestamp no test can pin.
    """
    model = evidence["cost_model"]
    stand_in = " (a declared stand-in)" if model["placeholder"] else ""
    return {
        "name": PORTFOLIO_METRIC_NAME,
        "module": "crucible.portfolio",
        "metric_type": "construction",
        "n_floor": 1,
        "status": "OK",
        # Operator-readable and specific: the row's whole job is to say which
        # model charged this run, so the sentence a reader sees first says it.
        "status_reason": (
            f"book constructed over {evidence['sessions']} session(s) by "
            f"crucible.portfolio, charged by cost model {model['name']}{stand_in}"
        ),
        "source_path": "crucible/portfolio.py",
        "last_updated_utc": now_utc,
        "value": float(evidence["cost_bps_total"]),
        "unit": "bps",
        "horizon_trading_days": None,
        "portfolio_construction": dict(evidence),
    }


def manifest_records_portfolio_engine(manifest: Mapping[str, Any]) -> dict[str, Any] | None:
    """The construction evidence on ``manifest``, or None if it carries none.

    The predicate a phase-3 clause reads. It lives beside the PRODUCER so the
    gate calls one function rather than restating the row's shape — a shape
    restated at the reader is a contract restated twice, and one of them
    drifts. A manifest with no such row returns None, which is the honest
    answer to "was the engine used": not "no", but "this run recorded nothing".
    """
    for row in manifest.get("metrics") or ():
        if not isinstance(row, Mapping):
            continue
        if row.get("name") != PORTFOLIO_METRIC_NAME:
            continue
        evidence = row.get("portfolio_construction")
        if not isinstance(evidence, Mapping):
            raise ValueError(
                f"manifest carries a {PORTFOLIO_METRIC_NAME!r} metric row with no "
                "`portfolio_construction` payload. A row that names the engine without "
                "carrying its evidence would read as proof to any clause counting rows."
            )
        if evidence.get("schema_version") != PORTFOLIO_EVIDENCE_SCHEMA_VERSION:
            raise ValueError(
                f"portfolio construction evidence declares schema_version "
                f"{evidence.get('schema_version')!r}; this reader understands "
                f"{PORTFOLIO_EVIDENCE_SCHEMA_VERSION!r}. A document read against the "
                "wrong version is read wrong, not read approximately."
            )
        return dict(evidence)
    return None
