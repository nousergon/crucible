"""The S slot: exit and risk rules, graded market-relative and net of cost.

Normative sources: `champion-challenger-policy.md` §3.1, §4, §5.3; plan §4.4,
§9.1 (contamination attestation), §10.2 (the real cost model is phase 3).

Lifted, not imported, from `crucible-executor/executor/strategies/` (the
`ExitRule` contract and the seven `stock_registry()` rules),
`crucible-backtester/synthetic/pit_folds.py` (walk-forward folds) and
`crucible-backtester/analysis/pit_parity.py` (the contamination attestation).

**S is the one slot benchmarked against SPY.** Its output IS a market
position, so market-relative canonical alpha is the correct axis. The
population rule that protects U and R — a selection stage must beat the
population it drew from, never an index — would be the wrong benchmark here,
and over-applying it would be a second defect rather than a fix. The
`ArenaConfig` for U and R refuses SPY; S declares it deliberately.

**Three things this module refuses to do quietly:**

1. **Grade without naming its cost model.** §10.2's real transaction-cost
   model lands in phase 3. Phase 1 therefore requires each arm file to NAME a
   placeholder and carry its constants, so a grade is reproducible from the
   recipe. A cost constant living in the grader would silently re-price every
   arm's history the day it changed.
2. **Render a grade without an attestation.** Plan §9.1: "a card without
   `attestation: PASS` renders UNVERIFIED, never a grade."
   :func:`render_verdict` omits the grade key entirely rather than showing a
   number beside a caveat nobody reads.
3. **Treat missing coverage as clean.** `pit_parity` returns `UNKNOWN`, never
   `PASS`, when the walk-forward pass scored nothing. A contamination check
   that did not answer must not read as an answer of "no contamination".

**Point-in-time exit-rule EXECUTION is not in this module.** The rules'
runtime lives with the trader; what S grades is the recipe's realized book,
which arrives as a :class:`Book`. That is the plan §3 separation held at the
module boundary: the harness must complete every acceptance test with the
trader switched off.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from nousergon_lib.arena import ArmRegister, ArmSeries, derive_arm_id
from nousergon_lib.arena.engine import ServingPrecondition

from crucible.attribution import (
    AttributionFactorParams,
    attribution_metric_record,
    compute_factor_attribution,
    factor_return_series,
    load_attribution_params_from_store,
)
from crucible.features import DEFAULT_FEATURE_VERSION, feature_names, read_features
from crucible.keys import arm_predictions_key, features_key
from crucible.portfolio import (
    PORTFOLIO_PARAM_FIELDS,
    CostModel,
    CostModelInputError,
    PortfolioParams,
    cost_model_from_mapping,
    load_portfolio_params_from_store,
    make_cash_sentinel_returns,
    portfolio_evidence,
    portfolio_metric_record,
    solve_target_weights,
)
from crucible.slots.arms import SupersededArmUndeclaredError, resolve_declared_lineage
from crucible.slots.inputs import InputRefusal, SlotUnservableError
from crucible.slots.vocab import refuse_unknown_keys

__all__ = [
    "ARM_REFUSED_METRIC",
    "ATTESTATION_STATUSES",
    "EXIT_RULES",
    "SESSION_INPUTS_SCHEMA_VERSION",
    "SLOT",
    "Book",
    "BookUniverse",
    "ConstructedBook",
    "CostModel",
    "ExitRuleSpec",
    "PitParityVerdict",
    "RegisteredStrategyArm",
    "ResolvedSession",
    "SessionInputs",
    "SlotStrategies",
    "StrategyGrade",
    "StrategyRecipe",
    "SupersededArmUndeclaredError",
    "WalkForwardFold",
    "WalkForwardSpec",
    "build_walk_forward_folds",
    "construct_book",
    "grade",
    "grade_arm",
    "load_strategy_recipes",
    "load_strategy_slot",
    "parse_strategy_document",
    "pit_parity",
    "produce",
    "produce_history",
    "registration_specs",
    "render_verdict",
    "resolve_session",
]

#: The metric one refused arm files on the manifest of whatever job loaded the
#: slot. The same name `crucible.slots.cycle` uses for U/R and
#: `crucible.slots.model` uses for M, restated here rather than imported
#: because importing either would be circular through `crucible.slots`.
#: `tests/test_cycle_refuses_by_name.py` pins the names equal.
ARM_REFUSED_METRIC = "arm_refused_at_registration"


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


#: The registered exit rules, in the canonical `stock_registry()` order, with
#: the parameters each declares. The IMPLEMENTATIONS and this registry are
#: framework and therefore public; the tuned VALUES are strategy edge and live
#: in `alpha-engine-config/strategy/arms/s/` (`repository-tiering-policy`
#: test 2). A recipe naming a rule absent from here is refused at load.
#:
#: `position_loss_floor` is first and stays first: it is the hard MAE floor,
#: stance-agnostic, and a chain that can reach a profit-take before its loss
#: floor has a different risk shape whatever its parameters say.
EXIT_RULES: dict[str, tuple[str, ...]] = {
    "position_loss_floor": ("position_loss_floor_pct",),
    "catalyst_hard_exit": ("catalyst_followthrough_days",),
    "atr_trailing_stop": (
        "atr_period",
        "atr_multiplier",
        "sector_relative_outperform_threshold",
    ),
    "fallback_stop": ("fallback_stop_pct",),
    "profit_take": ("profit_take_pct",),
    "momentum_exit": ("momentum_exit_threshold", "momentum_exit_rsi"),
    "time_decay": ("time_decay_reduce_days", "time_decay_exit_days"),
}

#: The attestation vocabulary, closed. `PARTIAL` and `UNKNOWN` exist so that
#: "the check could not answer" never has to be rounded to a pass or a fail.
ATTESTATION_STATUSES: tuple[str, ...] = ("PASS", "FAIL", "PARTIAL", "UNKNOWN")

BASIS_POINT = 1e-4


# ---------------------------------------------------------------------------
# The recipe.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExitRuleSpec:
    """One rule in the chain, with its parameters. Part of the recipe hash."""

    rule_id: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rule_id not in EXIT_RULES:
            raise ValueError(
                f"unknown exit rule {self.rule_id!r}; registered rules are "
                f"{sorted(EXIT_RULES)}. A rule resolved at runtime rather than from "
                "this registry is a chain whose recipe does not describe what it runs."
            )
        expected = set(EXIT_RULES[self.rule_id])
        unknown = sorted(set(self.params) - expected)
        if unknown:
            raise ValueError(
                f"exit rule {self.rule_id!r} does not take parameter(s) {unknown}; it "
                f"takes {sorted(expected)}. A parameter nothing reads is a tuning "
                "someone believes is live."
            )

    def to_dict(self) -> dict[str, Any]:
        return {"rule_id": self.rule_id, "params": dict(self.params)}


@dataclass(frozen=True)
class WalkForwardSpec:
    """Fold geometry, in trading days (§4.12).

    Three of these five fields have a meaning a reader can get wrong, so they
    are stated here rather than left to the caller's assumption:

    * ``purge`` is the LABEL HORIZON in trading days, not a gap width. A label
      stamped on training row ``i`` spans ``i .. i + purge``, so the last row
      that may be trained on is ``test_start_idx - purge - 1`` and the fold
      carries ``purge`` fully excluded rows between train and test. Reading it
      as a gap width yields ``test_start_idx - purge``, whose label lands ON
      the first test day — a one-day overlap that looks like a purge.
    * ``min_train`` is the MINIMUM NUMBER OF TRAINING ROWS a fold must have
      after the purge, not merely the index the first test block may start at.
      A fold that cannot reach it is skipped rather than emitted short.
    * under ``train_mode="rolling"`` the training block is ``min_train`` rows
      long. The lifted source
      (`crucible-backtester/synthetic/pit_folds.py:111-112`) sizes the rolling
      window by ``test_window`` instead, which makes a spec declaring
      ``min_train=504`` train on 21 rows; that reading is not carried over,
      because it leaves ``min_train`` naming nothing the folds honour.
    """

    test_window: int = 21
    min_train: int = 504
    purge: int = 21
    embargo: int = 2
    train_mode: str = "expanding"

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_window": self.test_window,
            "min_train": self.min_train,
            "purge": self.purge,
            "embargo": self.embargo,
            "train_mode": self.train_mode,
        }


@dataclass(frozen=True)
class StrategyRecipe:
    """One S arm: an ordered exit-rule chain, a cost model, a fold geometry.

    The chain is FIRST-DECISION-WINS, so its order is semantic. Reordering it
    changes the arm's behaviour and therefore its id — which is why
    :meth:`with_rules` produces a new arm carrying `supersedes` rather than
    mutating one (policy §3.1).
    """

    name: str
    rules: tuple[ExitRuleSpec, ...]
    cost_model: CostModel
    walk_forward: WalkForwardSpec = field(default_factory=WalkForwardSpec)
    benchmark: str = "SPY"
    supersedes: str | None = None
    slot: str = "s"
    #: The start of the arm's out-of-sample clock, and a SESSION when set
    #: (§4.12). Optional on the dataclass and optional to register
    #: (`alpha-engine-config-I10634`, Brian ruling 2026-09-13, option (b)):
    #: an arm declaring none is not refused, it registers with its clock
    #: stamped from the first `experiment.run[s]` that registers it — see
    #: :func:`_resolve_registered_at`. A DECLARED value is still refused if
    #: it names the future or a date after the arm's already-recorded first
    #: register row, because a clock that can be moved is not a clock.
    #: Deliberately NOT part of :attr:`spec`, exactly as
    #: `crucible.slots.arms.ArmSpec` keeps it out of its own hash: when an
    #: arm was registered is provenance, not what it computes, and hashing
    #: it would give one recipe two ids.
    registered_at: str | None = None

    def __post_init__(self) -> None:
        if not self.rules:
            raise ValueError(f"arm {self.name!r} declares no exit rules")
        seen = [r.rule_id for r in self.rules]
        if self.registered_at is not None:
            from crucible.calendar import assert_trading_day  # noqa: PLC0415 - avoids a cycle

            # A raise rather than a resolution to the neighbouring session: a
            # recipe naming a Saturday was written by something that keyed off
            # the wall clock, and quietly moving the date to Friday would hide
            # the writer while producing a plausible arm. Same contract, same
            # reason, as `crucible.slots.arms.ArmSpec.__post_init__`.
            assert_trading_day(
                self.registered_at,
                context=f"arm {self.slot}:{self.name} `registered_at`",
            )
        duplicates = sorted({r for r in seen if seen.count(r) > 1})
        if duplicates:
            raise ValueError(
                f"arm {self.name!r} lists rule(s) {duplicates} more than once. The chain "
                "is first-decision-wins, so a second copy can never fire and is a "
                "parameter someone believes is live."
            )

    @property
    def spec(self) -> dict[str, Any]:
        return {
            "rules": [r.to_dict() for r in self.rules],
            "cost_model": self.cost_model.to_dict(),
            "walk_forward": self.walk_forward.to_dict(),
            "benchmark": self.benchmark,
        }

    @property
    def arm_id(self) -> str:
        return derive_arm_id(self.slot, self.name, self.spec)

    def with_rules(self, rules: tuple[ExitRuleSpec, ...]) -> StrategyRecipe:
        """A NEW arm with a different chain, carrying `supersedes` (§3.1)."""
        return replace(self, rules=rules, supersedes=self.arm_id)

    def with_rule_params(self, rule_id: str, params: dict[str, Any]) -> StrategyRecipe:
        """A NEW arm with one rule retuned, carrying `supersedes` (§3.1)."""
        if rule_id not in {r.rule_id for r in self.rules}:
            raise ValueError(f"arm {self.name!r} does not run rule {rule_id!r}")
        return self.with_rules(
            tuple(
                ExitRuleSpec(rule_id=r.rule_id, params={**r.params, **params})
                if r.rule_id == rule_id
                else r
                for r in self.rules
            )
        )


REQUIRED_STRATEGY_FIELDS: tuple[str, ...] = ("rules", "cost_model")

#: Every key a filed S recipe's `spec:` may declare (`alpha-engine-config-
#: I9944`) — the required fields above, plus the two optional keys this
#: loader also reads: `walk_forward` (defaults to `WalkForwardSpec()`) and
#: `benchmark` (defaults to `"SPY"`).
S_SPEC_KEYS: frozenset[str] = frozenset({*REQUIRED_STRATEGY_FIELDS, "walk_forward", "benchmark"})

#: Every top-level key a filed S recipe may declare (`alpha-engine-config-
#: I9944`). `registered_at` is here since `alpha-engine-config-I10512`: while
#: the S slot had no `produce`/`grade` it was true that "an S arm is not
#: OOS-clocked the way U/R/M arms are", because nothing read the field —
#: making the slot dispatchable is exactly what stops that being true, since
#: every ladder rung the arena decides is counted in trading weeks from it.
#: Optional in the VOCABULARY and optional to REGISTER since
#: `alpha-engine-config-I10634`: a recipe declaring none registers with its
#: clock stamped from its first registering cycle rather than being refused
#: (see :func:`_resolve_registered_at`).
S_TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {"slot", "name", "notes", "supersedes", "spec", "registered_at"}
)


def load_strategy_recipes(directory: Path | str) -> tuple[StrategyRecipe, ...]:
    """Load every `*.yaml` S recipe under ``directory``, sorted by filename.

    `alpha-engine-config/strategy/arms/s/` in production. Tuned values stay
    private; this repository holds the registry the values are checked against.
    """
    root = Path(directory)
    return tuple(
        parse_strategy_document(path.read_bytes(), origin=str(path))
        for path in sorted(root.glob("*.yaml"))
    )


def parse_strategy_document(payload: bytes, *, origin: str) -> StrategyRecipe:
    """One filed S recipe, validated, from bytes.

    Extracted from :func:`load_strategy_recipes` so the SAME parse serves both
    sources a running job reads (`alpha-engine-config-I10512`): a checkout on
    a laptop and the strategy tree synced into the store on a box. A second
    parser for the store path is how a recipe comes to mean two things.
    """
    document = yaml.safe_load(payload.decode("utf-8")) or {}
    spec = document.get("spec") or {}
    missing = [f for f in REQUIRED_STRATEGY_FIELDS if f not in spec]
    if missing:
        raise ValueError(
            f"{origin}: recipe is missing pre-registration field(s) {missing}. Plan §9.1: "
            "missing fields mean the arm does not register."
        )
    refuse_unknown_keys(
        path=origin,
        keys=set(document),
        vocabulary=S_TOP_LEVEL_KEYS,
        level="top-level recipe",
        slot_label="S",
    )
    refuse_unknown_keys(
        path=origin,
        keys=set(spec),
        vocabulary=S_SPEC_KEYS,
        level="spec",
        slot_label="S",
    )
    return StrategyRecipe(
        slot=document.get("slot", "s"),
        name=document["name"],
        rules=tuple(
            ExitRuleSpec(rule_id=r["rule_id"], params=dict(r.get("params") or {}))
            for r in spec["rules"]
        ),
        cost_model=cost_model_from_mapping(spec["cost_model"], source=origin),
        walk_forward=WalkForwardSpec(**(spec.get("walk_forward") or {})),
        benchmark=spec.get("benchmark", "SPY"),
        supersedes=document.get("supersedes"),
        registered_at=document.get("registered_at"),
    )


# ---------------------------------------------------------------------------
# Walk-forward folds.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WalkForwardFold:
    """One expanding-or-rolling fold, as index bounds and dates."""

    train_start_idx: int
    train_end_idx: int
    test_start_idx: int
    test_end_idx: int
    train_start_date: str
    train_end_date: str
    test_start_date: str
    test_end_date: str


def build_walk_forward_folds(
    dates: list[str],
    *,
    test_window: int,
    min_train: int,
    purge: int,
    embargo: int,
    train_mode: str = "expanding",
) -> list[WalkForwardFold]:
    """Purged, embargoed walk-forward folds over ``dates``.

    Lifted from `crucible-backtester/synthetic/pit_folds.py`. Two properties
    are the reason it is lifted rather than reinvented:

    * ``purge`` whole trading days are excluded between the training block and
      the test block, removing the overlapping-label leak that a naive
      expanding split carries;
    * when ``embargo > purge`` the next fold starts later, so the serial
      correlation immediately after a test block is not scored either.

    **The purge is an off-by-one trap, and this function had the bug.** With
    ``purge`` read as the label horizon, the label on training row ``i`` spans
    ``i .. i + purge``. Setting ``train_end_idx = test_start_idx - purge``
    therefore keeps a row whose own label is realized ON the first test day:
    the fold reports a purge and leaks a day. The invariant this function now
    holds, and :mod:`tests.test_slot_strategy` asserts directly, is

        ``train_end_idx + purge < test_start_idx``

    — no training row's label may extend into the test window. That is the
    same arithmetic the CPCV purge in :mod:`crucible.slots.model` already
    used (``lo = a - label_horizon_trading_days``, excluding through ``a - 1``
    and leaving ``a - h - 1`` as the last trainable row); two purge
    implementations in one repository must not differ by a day.

    ``min_train`` is enforced as a minimum training LENGTH: a candidate fold
    whose training block would be shorter is skipped, not emitted short. Under
    ``train_mode="rolling"`` the training block is exactly ``min_train`` rows.

    ``dates`` are opaque here: the caller supplies trading days, and this
    function never derives one, so there is no second calendar to drift.
    """
    if test_window <= 0 or min_train <= 0:
        raise ValueError("test_window and min_train must be positive")
    if purge < 0 or embargo < 0:
        raise ValueError("purge and embargo must be non-negative")
    if train_mode not in ("expanding", "rolling"):
        raise ValueError(
            f"unknown train_mode {train_mode!r}; expanding|rolling. A window rule this "
            "function does not implement must not be silently treated as expanding."
        )

    n = len(dates)
    folds: list[WalkForwardFold] = []
    fold_start_idx = min_train
    while fold_start_idx < n:
        remaining = n - fold_start_idx
        if remaining < test_window // 2:
            break
        test_start_idx = fold_start_idx
        test_end_idx = min(fold_start_idx + test_window - 1, n - 1)
        # `purge` is the label horizon, so the last trainable row is the one
        # whose label is fully realized BEFORE the test window opens.
        train_end_idx = fold_start_idx - purge - 1
        if train_end_idx + 1 < min_train:
            fold_start_idx += test_window
            continue
        if train_mode == "expanding":
            train_start_idx = 0
        else:
            train_start_idx = train_end_idx - min_train + 1
        if train_end_idx < train_start_idx:
            fold_start_idx += test_window
            continue
        folds.append(
            WalkForwardFold(
                train_start_idx=train_start_idx,
                train_end_idx=train_end_idx,
                test_start_idx=test_start_idx,
                test_end_idx=test_end_idx,
                train_start_date=dates[train_start_idx],
                train_end_date=dates[train_end_idx],
                test_start_date=dates[test_start_idx],
                test_end_date=dates[test_end_idx],
            )
        )
        advance = test_window
        if embargo > purge:
            advance = max(test_window, test_window + (embargo - purge))
        fold_start_idx += advance
    return folds


# ---------------------------------------------------------------------------
# The contamination attestation.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PitParityVerdict:
    """Whether the look-ahead-vs-point-in-time delta is distinguishable from 0."""

    status: str
    reason: str
    mean_delta: float | None = None
    ci: tuple[float, float] | None = None
    coverage_fraction: float | None = None

    def __post_init__(self) -> None:
        if self.status not in ATTESTATION_STATUSES:
            raise ValueError(
                f"attestation status {self.status!r} is not one of {ATTESTATION_STATUSES}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "pit_parity",
            "status": self.status,
            "reason": self.reason,
            "mean_delta": self.mean_delta,
            "ci": list(self.ci) if self.ci else None,
            "coverage_fraction": self.coverage_fraction,
        }


def pit_parity(
    *,
    contaminated: Mapping[str, float],
    point_in_time: Mapping[str, float],
    coverage: dict[str, Any],
    alpha: float = 0.05,
) -> PitParityVerdict:
    """Compare the look-ahead pass against the point-in-time pass, per date.

    Materiality is a confidence interval on the per-date delta that EXCLUDES
    zero, not a fixed threshold on a summary statistic: the fixed-threshold
    version answers "is the number big" rather than "is the difference
    distinguishable from noise", and a wide, noisy series passes it by being
    noisy.

    Coverage governs everything else, and the ordering is deliberate:

    * material delta → ``FAIL``, whatever the coverage;
    * budget exhausted with zero folds scored → ``FAIL`` (the pass ran and
      produced nothing, which is a broken check, not an absent one);
    * nothing measurable at all → ``UNKNOWN``;
    * partial or unproven coverage → ``PARTIAL``;
    * otherwise → ``PASS``.

    ``UNKNOWN`` and ``PARTIAL`` are refusals at the reader
    (:mod:`crucible.champion`), so neither can be rounded up to a grade.

    **The two passes are paired by DATE KEY, and a mismatch raises.** Both
    arguments are mappings of trading day to that day's score, never parallel
    sequences. Positional pairing over two lists — the shape this function
    carried — truncated to ``min(len(a), len(b))`` and then paired by index,
    which is wrong in the one way that matters: a single extra element on one
    side shifts every pair by a day, and a shifted pair is uncorrelated noise
    around the true delta. Measured on a series carrying +150bp/day of real
    look-ahead alpha over full proven coverage: clean pairing renders ``FAIL``
    (mean delta 0.0150, interval excluding zero) and an off-by-one pairing of
    the same data renders ``PASS`` (mean delta 0.0155, interval straddling
    zero) with no exception raised. Since
    :func:`crucible.champion._assert_attested` treats that ``PASS`` as the
    sole gate on an S champion, one stray element promoted a contaminated arm.
    A ``strict=True`` on a ``zip`` of two already-truncated sequences cannot
    see it, because the truncation happened first.

    So misalignment is an exception, not a verdict. A date present on one side
    and absent on the other means the two passes scored different windows, and
    there is no answer to render from that — the same refusal
    :class:`Book` makes when a per-date series is shorter than its dates.
    """
    fraction = coverage.get("coverage_fraction")
    budget_stopped = bool(coverage.get("budget_stopped"))
    measured = bool(coverage.get("measured"))

    for name, series in (("contaminated", contaminated), ("point_in_time", point_in_time)):
        if not isinstance(series, Mapping):
            raise TypeError(
                f"pit_parity {name}= must be a mapping of trading day to score, not "
                f"{type(series).__name__}. Positional pairing is the defect this "
                "signature exists to make unexpressible."
            )
    only_contaminated = sorted(set(contaminated) - set(point_in_time))
    only_pit = sorted(set(point_in_time) - set(contaminated))
    if only_contaminated or only_pit:
        raise ValueError(
            "the look-ahead and point-in-time passes scored different dates: "
            f"{len(only_contaminated)} only in contaminated (e.g. {only_contaminated[:3]}), "
            f"{len(only_pit)} only in point_in_time (e.g. {only_pit[:3]}). Pairing what "
            "is left over would compare a date against a different date, and the "
            "resulting delta reads as noise — which renders PASS."
        )

    days = sorted(contaminated)
    n = len(days)
    deltas = [float(contaminated[d]) - float(point_in_time[d]) for d in days]
    material: bool | None = None
    mean_delta: float | None = None
    ci: tuple[float, float] | None = None
    if n >= 2:
        mean_delta = sum(deltas) / n
        variance = sum((d - mean_delta) ** 2 for d in deltas) / (n - 1)
        stderr = math.sqrt(variance / n)
        # Normal quantile at 1 - alpha/2; 1.959964 at alpha = 0.05.
        z = 1.959963984540054 if abs(alpha - 0.05) < 1e-12 else _z(alpha)
        ci = (mean_delta - z * stderr, mean_delta + z * stderr)
        material = ci[0] > 0.0 or ci[1] < 0.0

    if material is True:
        # The materiality test is deliberately TWO-SIDED. A negative delta —
        # the point-in-time pass beating the look-ahead pass — is not
        # contamination; a look-ahead pass that loses to its own point-in-time
        # counterpart means the harness is wired wrong, and rendering that
        # PASS would be a fail-open on the one gate an S champion has. It
        # stays a FAIL, but it is told to the operator as what it is: the two
        # directions need different remediations, and a broken-harness result
        # reported as "MATERIAL contamination" sends the reader to look for a
        # leak that is not there.
        interval = (
            f"averages {mean_delta:.6g} with a {int((1 - alpha) * 100)}% interval "
            f"[{ci[0]:.6g}, {ci[1]:.6g}] that excludes zero"
        )
        if mean_delta is not None and mean_delta < 0.0:
            reason = (
                f"INVERTED attestation: the per-date look-ahead delta {interval}. The "
                "point-in-time pass BEAT the look-ahead pass, which no leak produces — "
                "the two passes are mismatched, mislabelled or scoring different books. "
                "Fix the harness; this is not a contamination finding."
            )
        else:
            reason = f"MATERIAL contamination: the per-date look-ahead delta {interval}"
        return PitParityVerdict(
            "FAIL",
            reason,
            mean_delta,
            ci,
            fraction,
        )
    if budget_stopped and fraction == 0.0:
        return PitParityVerdict(
            "FAIL",
            "the walk-forward pass exhausted its budget without scoring a single fold; "
            "a check that ran and produced nothing is broken, not absent",
            mean_delta,
            ci,
            fraction,
        )
    if material is None or fraction in (None, 0.0):
        return PitParityVerdict(
            "UNKNOWN",
            "the contamination check did not answer this cycle: "
            f"{n} paired date(s), coverage_fraction={fraction!r}. Not a pass — an "
            "unmeasured gate reported as clean is the defect the gate prevents "
            "(champion-challenger-policy.md §5.1).",
            mean_delta,
            ci,
            fraction,
        )
    if budget_stopped or (fraction is not None and fraction < 1.0) or not measured:
        return PitParityVerdict(
            "PARTIAL",
            f"no material delta over the {float(fraction) * 100:.0f}% of the window that "
            "was scored, but coverage is incomplete or unproven",
            mean_delta,
            ci,
            fraction,
        )
    return PitParityVerdict(
        "PASS",
        "the look-ahead-vs-point-in-time delta is not statistically distinguishable "
        "from zero over full, proven coverage",
        mean_delta,
        ci,
        fraction,
    )


def _z(alpha: float) -> float:
    """Two-sided normal quantile, Acklam's rational approximation.

    Closed form rather than a scipy dependency: one quantile at one or two
    levels does not justify carrying scipy into every consumer of this module.
    """
    p = 1.0 - alpha / 2.0
    a = [
        -39.69683028665376,
        220.9460984245205,
        -275.9285104469687,
        138.3577518672690,
        -30.66479806614716,
        2.506628277459239,
    ]
    b = [
        -54.47609879822406,
        161.5858368580409,
        -155.6989798598866,
        66.80131188771972,
        -13.28068155288572,
    ]
    c = [
        -0.007784894002430293,
        -0.3223964580411365,
        -2.400758277161838,
        -2.549732539343734,
        4.374664141464968,
        2.938163982698783,
    ]
    d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )


def render_verdict(
    *,
    arm_id: str,
    as_of: str,
    alpha_vs_spy: float,
    attestation: dict[str, Any] | None,
    cost_model: dict[str, Any],
) -> dict[str, Any]:
    """The S-slot verdict card. **No attestation PASS, no grade.**

    Plan §9.1: "a card without `attestation: PASS` renders UNVERIFIED, never
    a grade." The grade key is OMITTED rather than set alongside a caveat: a
    number rendered beside a warning is a number people quote.

    ``cost_model`` is REQUIRED, and it is rendered on the card whether or not
    the card carries a grade. A net-of-cost number is a claim about what was
    charged, and a card able to make that claim without stating the model is
    the shape in which an optimizer once ran on a cost model nobody configured
    and nothing said so (`alpha-engine-config-I10503`, precedent `-I6902`). A
    card whose model is a declared stand-in says `placeholder: true` on its
    face, where a reader cannot miss it.
    """
    if not cost_model or "name" not in cost_model:
        raise ValueError(
            "a verdict card must name the cost model its grade is net of; got "
            f"{cost_model!r}. Use `StrategyGrade.cost_model`."
        )
    status = (attestation or {}).get("status")
    card: dict[str, Any] = {
        "arm_id": arm_id,
        "as_of": as_of,
        "attestation": attestation,
        "cost_model": dict(cost_model),
    }
    if status != "PASS":
        card["rendered"] = "UNVERIFIED"
        card["reason"] = (
            f"contamination attestation is {status!r}, not 'PASS'; an unsigned backtest "
            "is not a grade (plan §9.1)"
        )
        return card
    card["rendered"] = "GRADED"
    card["grade"] = {"alpha_vs_spy": alpha_vs_spy, "benchmark": "SPY"}
    return card


# ---------------------------------------------------------------------------
# Grading.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Book:
    """One arm's realized daily book, already reduced to per-date returns.

    ``turnover`` is the fraction of the book traded on that date; it is what a
    flat cost model prices. It is a required field rather than an optional one
    because a cost model applied to an assumed turnover would produce a
    net-of-cost number whose cost nobody supplied.

    ``cost_bps`` is the per-date realized cost, in basis points of the book, as
    priced by the model the recipe names. It is present when the book was built
    by :func:`construct_book` and absent when the caller reduced a book by some
    other route. A PARTICIPATION-AWARE model cannot be applied to a book that
    omits it: a square-root cost depends on the size of each name's trade
    against that name's volume, and turnover alone does not carry either — so
    :func:`grade_arm` refuses rather than averaging the difference away.
    """

    dates: tuple[str, ...]
    portfolio_returns: tuple[float, ...]
    benchmark_returns: tuple[float, ...]
    turnover: tuple[float, ...]
    benchmark_symbol: str = "SPY"
    cost_bps: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        n = len(self.dates)
        series = ["portfolio_returns", "benchmark_returns", "turnover"]
        if self.cost_bps is not None:
            series.append("cost_bps")
        for name in series:
            if len(getattr(self, name)) != n:
                raise ValueError(
                    f"Book.{name} has {len(getattr(self, name))} entries for {n} dates; "
                    "a per-date series shorter than its dates is a silent truncation"
                )


@dataclass(frozen=True)
class BookUniverse:
    """The names a book is constructed over, and the two sentinel positions.

    Frozen and separate from the per-session inputs because it does not move
    between sessions: a universe that changed shape mid-walk would make every
    weight vector in the walk a different object, and the turnover between two
    of them meaningless.
    """

    tickers: tuple[str, ...]
    sectors: tuple[str, ...]
    benchmark_idx: int
    cash_idx: int

    def __post_init__(self) -> None:
        if len(self.tickers) != len(self.sectors):
            raise ValueError(
                f"{len(self.tickers)} tickers against {len(self.sectors)} sector labels; "
                "one sector per name"
            )


@dataclass(frozen=True)
class SessionInputs:
    """One session's point-in-time inputs to portfolio construction.

    Every array is as of the session's open and the realized returns are the
    session's own. Nothing here reaches forward: the walk-forward geometry and
    the contamination attestation are what police that, and a caller that
    hands this a forward-looking alpha has produced a contaminated grade that
    :func:`pit_parity` exists to catch.
    """

    trading_day: str
    alpha_hat: np.ndarray
    eligibility: np.ndarray
    stance_caps: np.ndarray
    realized_returns: np.ndarray
    benchmark_return: float
    returns_panel: np.ndarray | None = None
    covariance: np.ndarray | None = None
    adv_usd: np.ndarray | None = None
    name_sigma: np.ndarray | None = None
    alpha_uncertainty: np.ndarray | None = None
    alpha_uncertainty_epistemic: np.ndarray | None = None


@dataclass(frozen=True)
class ConstructedBook:
    """A book built by `crucible.portfolio`, with the evidence that it was.

    ``evidence`` is the `portfolio_construction.v1` document — the record a
    phase-3 clause reads off the grading run's manifest. It is produced HERE,
    beside the construction, rather than assembled later from diagnostics by a
    reporting layer: a record built by a second party is a record that can
    describe a run that did not happen.
    """

    book: Book
    evidence: dict[str, Any]
    weights: tuple[tuple[float, ...], ...]
    diagnostics: tuple[dict[str, Any], ...]


def _refuse_an_unpriceable_forced_exit(
    *,
    recipe: StrategyRecipe,
    universe: BookUniverse,
    session: SessionInputs,
    w_prev: np.ndarray,
) -> None:
    """A participation-aware model RAISES on a forced exit it has no ADV for.

    A held name that is ineligible this session is pinned to zero by the
    solver, so its whole holding WILL trade. `CostModel.cost_bps_for_trades`
    prices a traded name with no usable ADV through the library's no-coverage
    branch — a charge that does not come from the model the recipe named. For
    a trade the optimizer chose that is a pre-existing boundary; for a trade
    the harness FORCED it is a known-at-construction gap, so it is refused here
    with the name and the source it was looked for in
    (`alpha-engine-config-I10754`, `-I10503`). A flat model reads no ADV and is
    never refused for this.
    """
    if recipe.cost_model.kind == "flat":
        return
    eligible = np.asarray(session.eligibility, dtype=bool)
    adv = (
        np.full(len(universe.tickers), np.nan)
        if session.adv_usd is None
        else np.asarray(session.adv_usd, dtype=np.float64)
    )
    unpriced = [
        universe.tickers[i]
        for i in range(len(universe.tickers))
        if i not in (universe.benchmark_idx, universe.cash_idx)
        and not eligible[i]
        and w_prev[i] != 0.0
        and not (math.isfinite(float(adv[i])) and float(adv[i]) > 0.0)
    ]
    if unpriced:
        raise CostModelInputError(
            f"arm {recipe.name!r} on {session.trading_day}: held name(s) {unpriced} are "
            "ineligible this session, so the solver must exit them, and cost model "
            f"{recipe.cost_model.name!r} prices every exit against the name's "
            f"`{ADV_FEATURE_COLUMN}` on the decision day — which the compiled feature "
            "layer does not carry for them. Charging the exit through a no-coverage "
            "branch would put a number on the book that the named model did not price."
        )


def construct_book(
    *,
    recipe: StrategyRecipe,
    params: PortfolioParams,
    universe: BookUniverse,
    sessions: Sequence[SessionInputs],
    portfolio_notional: float,
    w_initial: np.ndarray,
) -> ConstructedBook:
    """Walk ``sessions``, solving the portfolio each one, and reduce to a Book.

    This is `crucible.portfolio` used by S-slot grading, and it is the only
    route by which an S arm's book is built. Per session: solve the constrained
    MVO from the previous session's weights, price the realized trades with the
    model the recipe NAMES, and take the session's book return as the solved
    weights against the session's realized returns.

    The cost is priced from the actual weight deltas rather than from the
    turnover scalar, because a participation-aware model needs to know which
    names moved and by how much. That per-date cost travels on the
    :class:`Book`, so the number the grade subtracts is the number the engine
    charged — not a second estimate of it computed downstream from a summary.

    Raises rather than degrading on every input the named cost model cannot
    price. Grading is offline and the caller owns every input.
    """
    if not sessions:
        raise ValueError(
            f"arm {recipe.name!r}: no sessions to construct a book over. An empty walk "
            "produces no evidence, and recording it as a grade of zero sessions would "
            "put a number on a comparison that never happened."
        )
    if not (float(portfolio_notional) > 0.0):
        raise ValueError(
            f"portfolio_notional must be positive; got {portfolio_notional!r}. Weight "
            "deltas become trade sizes only against a book size."
        )
    n_names = len(universe.tickers)
    w_prev = np.asarray(w_initial, dtype=np.float64).ravel()
    if w_prev.shape != (n_names,):
        raise ValueError(f"w_initial shape {w_prev.shape} != ({n_names},)")

    dates: list[str] = []
    portfolio_returns: list[float] = []
    benchmark_returns: list[float] = []
    turnover: list[float] = []
    cost_bps: list[float] = []
    all_weights: list[tuple[float, ...]] = []
    all_diagnostics: list[dict[str, Any]] = []

    for session in sessions:
        _refuse_an_unpriceable_forced_exit(
            recipe=recipe, universe=universe, session=session, w_prev=w_prev
        )
        result = solve_target_weights(
            list(universe.tickers),
            np.asarray(session.alpha_hat, dtype=np.float64),
            session.returns_panel,
            w_prev,
            list(universe.sectors),
            np.asarray(session.stance_caps, dtype=np.float64),
            np.asarray(session.eligibility, dtype=bool),
            universe.benchmark_idx,
            universe.cash_idx,
            params,
            recipe.cost_model,
            alpha_uncertainty=session.alpha_uncertainty,
            alpha_uncertainty_epistemic=session.alpha_uncertainty_epistemic,
            covariance=session.covariance,
            adv_usd=session.adv_usd,
            portfolio_notional=portfolio_notional,
            name_sigma=session.name_sigma,
        )
        weights = np.asarray(result.weights, dtype=np.float64)
        delta = weights - w_prev
        realized = np.asarray(session.realized_returns, dtype=np.float64)
        if realized.shape != (n_names,):
            raise ValueError(
                f"{session.trading_day}: realized_returns shape {realized.shape} != ({n_names},)"
            )
        dates.append(session.trading_day)
        portfolio_returns.append(float(weights @ realized))
        benchmark_returns.append(float(session.benchmark_return))
        turnover.append(float(np.sum(np.abs(delta)) / 2))
        cost_bps.append(
            recipe.cost_model.cost_bps_for_trades(
                weight_deltas=delta,
                adv_usd=session.adv_usd,
                portfolio_notional=portfolio_notional,
                name_sigma=session.name_sigma,
                benchmark_idx=universe.benchmark_idx,
                cash_idx=universe.cash_idx,
            )
        )
        all_weights.append(tuple(float(x) for x in weights))
        all_diagnostics.append(result.diagnostics)
        w_prev = weights

    book = Book(
        dates=tuple(dates),
        portfolio_returns=tuple(portfolio_returns),
        benchmark_returns=tuple(benchmark_returns),
        turnover=tuple(turnover),
        benchmark_symbol=recipe.benchmark,
        cost_bps=tuple(cost_bps),
    )
    evidence = portfolio_evidence(
        trading_day=dates[-1],
        arm_id=recipe.arm_id,
        params=params,
        cost_model=recipe.cost_model,
        diagnostics=all_diagnostics[-1],
        sessions=len(dates),
        turnover_one_way_total=float(sum(turnover)),
        cost_bps_total=float(sum(cost_bps)),
    )
    return ConstructedBook(
        book=book,
        evidence=evidence,
        weights=tuple(all_weights),
        diagnostics=tuple(all_diagnostics),
    )


def _attribute_book(
    constructed: ConstructedBook,
    *,
    universe: BookUniverse,
    returns: Any,
    params: AttributionFactorParams,
    as_of: str,
) -> dict[str, Any]:
    """The `factor_attribution.v1` evidence for ``constructed``'s book.

    Decomposes the book's realized gross return (the same panel window
    `construct_book` walked) against the factor spec's ETF proxies. Every
    input series is read from ``returns`` — the same `_close_returns(panel)`
    frame the book was constructed over — so the decomposition and the
    construction it explains are measured against the identical panel, never
    a second read.

    Fails loud (module rule 5) on any proxy or holding the panel carries no
    row for: S is graded against a market index, and a decomposition missing
    a factor is not the deliverable — nothing here is stubbed or zero-filled
    to make a thin panel pass.
    """
    from crucible.slots.cycle import MissingArtifactError  # noqa: PLC0415 - avoids a cycle

    dates = constructed.book.dates
    holding_returns: dict[str, list[float]] = {}
    for ticker in universe.tickers:
        if ticker == CASH_TICKER:
            # The book's own realized-return construction (`_build_sessions`)
            # charges cash a flat 0.0 every session — the identical sentinel,
            # not a second convention for the same position.
            holding_returns[ticker] = [0.0] * len(dates)
            continue
        if ticker not in returns.columns:
            raise MissingArtifactError(
                f"attribution over the book graded {as_of}: holding {ticker!r} has no row "
                "in the price panel this book was constructed over. A book position with "
                "no return series cannot be attributed and none is invented for it."
            )
        holding_returns[ticker] = [float(returns.loc[day, ticker]) for day in dates]

    factor_returns: dict[str, list[float]] = {}
    for name, fdef in params.factors.items():
        proxies = {fdef.proxy, *([fdef.short_proxy] if fdef.short_proxy is not None else [])}
        proxy_returns: dict[str, list[float]] = {}
        for ticker in proxies:
            if ticker not in returns.columns:
                raise MissingArtifactError(
                    f"attribution factor {name!r} proxies {ticker!r}, which the price panel "
                    f"at {as_of} carries no rows for. Factor-neutral attribution is not "
                    "optional for the S slot and no proxy is substituted for a missing one "
                    f"— compile the panel with {ticker!r} in the universe."
                )
            proxy_returns[ticker] = [float(returns.loc[day, ticker]) for day in dates]
        factor_returns[name] = factor_return_series(fdef, proxy_returns)

    weights = dict(zip(universe.tickers, constructed.weights[-1], strict=True))
    return compute_factor_attribution(
        trading_day=dates[-1],
        window_sessions=len(dates),
        holding_returns=holding_returns,
        weights=weights,
        factor_returns=factor_returns,
        params=params,
        gross_return=float(sum(constructed.book.portfolio_returns)),
        cost_bps_total=float(constructed.evidence["cost_bps_total"]),
    )


@dataclass(frozen=True)
class StrategyGrade:
    """One S arm's cycle grade: the market-relative series, net of cost.

    ``cost_model`` is the full record of the model in force — name, kind,
    placeholder flag and every parameter. It travels with the grade rather
    than being looked up beside it, because a grade and the cost that produced
    it become separable the moment they live in two places, and a net-of-cost
    number whose cost model is a lookup away is one that gets quoted without it.
    """

    series: ArmSeries
    benchmark: str
    cost_model_name: str
    cost_model_is_placeholder: bool
    total_cost_bps: float
    cost_model: dict[str, Any]


def grade_arm(
    recipe: StrategyRecipe,
    book: Book,
    *,
    as_of: str,
    apply_costs: bool = True,
) -> StrategyGrade:
    """Score ``recipe`` per trading day: portfolio return minus the benchmark, net of cost.

    ``apply_costs=False`` exists for the one comparison that needs it — the
    gross-versus-net delta on the verdict card — and is never the production
    path: a gross grade promotes an arm on turnover it never paid for.

    Where the book carries a per-date ``cost_bps``, that is what is charged:
    the engine priced the realized trades and re-deriving the number here from
    a turnover summary would produce a second, quieter estimate of it. Where it
    does not, a FLAT model is charged at its round-trip rate against the day's
    turnover, and a participation-aware model REFUSES — its cost is not a
    function of turnover alone and pretending otherwise is the substitution
    this whole path exists to make impossible.
    """
    if book.benchmark_symbol != recipe.benchmark:
        raise ValueError(
            f"arm {recipe.name!r} declares benchmark {recipe.benchmark!r} and the book "
            f"carries {book.benchmark_symbol!r}. Grading against a benchmark the recipe "
            "did not declare inverts wins and losses outright."
        )
    if apply_costs and book.cost_bps is None and recipe.cost_model.kind != "flat":
        raise CostModelInputError(
            f"arm {recipe.name!r} names cost model {recipe.cost_model.name!r}, which "
            "prices participation, but the book carries no per-date cost. Build the "
            "book with `construct_book` so the engine prices the realized trades, or "
            "name a flat model in the recipe — a participation-aware cost cannot be "
            "recovered from turnover alone, and charging a flat rate instead would "
            "grade the arm under a model its recipe does not name."
        )
    scores: dict[str, float] = {}
    total_cost = 0.0
    for index, (day, port, bench, turn) in enumerate(
        zip(
            book.dates,
            book.portfolio_returns,
            book.benchmark_returns,
            book.turnover,
            strict=True,
        )
    ):
        if day > as_of:
            continue
        if not apply_costs:
            cost = 0.0
        elif book.cost_bps is not None:
            cost = float(book.cost_bps[index]) * BASIS_POINT
        else:
            cost = recipe.cost_model.bps_per_unit_turnover() * BASIS_POINT * float(turn)
        total_cost += cost
        scores[day] = float(port) - float(bench) - cost
    return StrategyGrade(
        series=ArmSeries(arm_id=recipe.arm_id, scores=scores),
        benchmark=recipe.benchmark,
        cost_model_name=recipe.cost_model.name,
        cost_model_is_placeholder=recipe.cost_model.placeholder,
        total_cost_bps=total_cost / BASIS_POINT,
        cost_model=recipe.cost_model.record(),
    )


# ---------------------------------------------------------------------------
# The S cycle job: `produce` and `grade` (`alpha-engine-config-I10512`).
#
# `crucible-PR210` shipped `crucible.portfolio` and `construct_book` — the S
# grading path calling the engine — and deliberately stopped short of these
# two names, because `crucible.slots.dispatchable_slots()` derives the
# dispatchable set by reading `produce`/`grade` off each slot module. Adding
# them makes S dispatchable the moment this merges, which is why the PR is
# gated. What the gap COST while it stood: `crucible.portfolio` is 2,078
# lines reachable only from `tests/`, its `portfolio_construction.v1`
# evidence reached no manifest any scheduled job wrote, and the phase-3
# clause `-I10510` wired to read that evidence could only ever read
# UNMEASURABLE — a deliverable complete in every part except the one that
# makes it observable.
# ---------------------------------------------------------------------------

SLOT = "s"

#: The point-in-time construction inputs artifact, one per (arm, session).
SESSION_INPUTS_SCHEMA_VERSION = "session_inputs.v1"

#: The synthetic rows every constructed book carries beside the real names.
#: `crucible.portfolio` requires a benchmark position (the no-conviction fill)
#: and a cash sleeve, each pinned by index, and `_real_sectors` excludes a
#: `__x__` sector from the sector cap — so the two sentinels are named in that
#: shape rather than given a sector of their own that the cap would then bind.
CASH_TICKER = "__CASH__"
BENCHMARK_SECTOR = "__benchmark__"
CASH_SECTOR = "__cash__"

#: The sector label every real name carries today, and the reason it does.
#: The v2 price panel (`crucible.keys.data_panel_key`) carries
#: `trading_day, ticker, open_raw, high_raw, low_raw, close_raw, volume_raw`
#: and no sector column; the feature catalogue declares no sector producer
#: either. A sentinel rather than a guess: `max_sector_pct` then binds on no
#: group at all, which is visibly a cap that did not apply, where inventing
#: one sector per name would make it bind on singletons and inventing one
#: shared sector would make it bind on everything. Recorded on every session
#: inputs document as `sectors_source` so the absence is read, not assumed.
UNSECTORED = "__unsectored__"

#: Trading days of returns history the covariance estimate is taken over.
COVARIANCE_LOOKBACK_TRADING_DAYS = 260

#: Trading days an S decision is held for, and therefore the span its score
#: and its controls are both measured over. One: the session's construction
#: inputs are what the arm saw, and the book earns the NEXT session's return.
#: Not the canonical 21-session label horizon, which is the axis U, R and M
#: are graded on — S's output is a position held daily, not a selection
#: settled a month later, and grading it on someone else's horizon would
#: compare the slots on axes that only look alike.
S_HORIZON_TRADING_DAYS = 1

#: The ADV series a participation-aware cost model prices trades against.
#: Already in the feature catalog (`alpha-engine-config-I10669`: the earlier
#: belief that this series "does not exist" was false, and filing a second
#: USD-volume column under a new name would be the exact `avg_volume_20d`
#: units-suffix defect the fleet's feature-store rule exists to prevent) —
#: `dollar_volume_20d_raw` is the mean traded notional over 20 sessions,
#: which is average daily dollar volume by definition. Checked dynamically,
#: never hardcoded, so a catalog change is what :func:`_participation_refusal`
#: reflects, not a belief frozen at the time this line was written.
ADV_FEATURE_COLUMN = "dollar_volume_20d_raw"

#: The portfolio-params field :func:`construct_book`'s ``portfolio_notional``
#: is read from (`crucible.portfolio.PortfolioParams.book_notional_usd`) —
#: the real book size a participation-aware model needs, threaded through the
#: slot's own tuned parameter set rather than a `GRADING_NOTIONAL = 1.0`
#: module constant this repository invented (`alpha-engine-config-I10669`,
#: the `-I10503` defect — the optimizer ran on a fallback cost model and
#: nothing said so — reproduced inside v2). A FLAT cost model's charge is a
#: function of the weight deltas alone and never reads this value, so
#: threading the real book size through it does not change what it grades.
NOTIONAL_PARAM_FIELD = "book_notional_usd"


@dataclass(frozen=True)
class SlotStrategies:
    """The S slot as loaded: the arms that register, and the arms that do not.

    The same pair `crucible.slots.model.SlotRecipes` carries for M, for the
    same reason (`alpha-engine-config-I9955`): a refusal is a VALUE, so one
    unregisterable arm cannot take its siblings down with it. An exception is
    slot-wide by construction and would be the wrong blast radius.
    """

    registered: tuple[RegisteredStrategyArm, ...]
    refused: tuple[InputRefusal, ...]

    def refusal_metrics(self, *, slot: str) -> list[dict[str, Any]]:
        """One `unservable` MetricRecord per refused arm.

        Its declared home is the manifest of whatever job loaded the slot,
        and the M slot's whole tracked defect was that it had no such job —
        the rows landed on no manifest a scheduled run ever wrote. Recorded
        FIRST by both entry points here, before anything that can raise.
        """
        return [_refusal_metric(slot, refusal) for refusal in self.refused]


def _refusal_metric(slot: str, refusal: InputRefusal) -> dict[str, Any]:
    """The refused-arm row. Deliberately the same shape `crucible.slots.cycle`
    writes for U/R and `crucible.slots.model` writes for M — one name, one
    status, one source path convention, so a console reading `unservable` rows
    across the four slots is reading one thing."""
    return {
        "name": ARM_REFUSED_METRIC,
        "module": f"crucible.slots.{slot}",
        "metric_type": "count",
        "value": float(len(refusal.unresolvable)),
        "unit": "inputs",
        "n_floor": 1,
        "status": "unservable",
        "status_reason": refusal.reason,
        "source_path": f"strategy/current/arms/{slot}/{refusal.arm}.yaml",
        "last_updated_utc": _utc_now(),
    }


@dataclass(frozen=True)
class RegisteredStrategyArm:
    """A :class:`StrategyRecipe` in the shape the register and the engine read.

    An adapter, not a second recipe type. `crucible.slots.arms.register_arms`
    and `crucible.slots.cycle.run_grade` read six things off a spec —
    `arm_id`, `name`, `slot`, `spec`, `registered_at`, `params` — and an S
    recipe carries its own id (the hash of its own spec), which is what must
    register. Re-deriving an id here through an `ArmSpec` view would give one
    arm two identities, and the register, the artifacts and the series would
    each speak about a different one.
    """

    recipe: StrategyRecipe
    #: The register LINK, which is not the same fact as the recipe's declared
    #: `supersedes` — see :func:`registration_specs`. Set there, never here.
    supersedes: str | None = None
    #: Provenance the register row carries verbatim: the declared lineage,
    #: including a parent this slot refuses and therefore never registers.
    notes: str = ""
    #: An S recipe is never a control: controls are GENERATED by
    #: `crucible.slots.arms.control_specs` and never filed, precisely so a
    #: planted-edge arm cannot sit in the strategy tree where an operator
    #: could clear the flag (§10.1).
    control: bool = False
    control_kind: str | None = None
    bootstrap: bool = False
    #: The arm's RESOLVED out-of-sample clock start, when it is not simply
    #: `recipe.registered_at` (`alpha-engine-config-I10634`): set by
    #: :func:`load_strategy_slot`/:func:`_resolve_registered_at` when the
    #: recipe declares no `registered_at` and the field is stamped instead
    #: from the register's own row or from the current cycle's trading day.
    #: None here means "read `recipe.registered_at` instead" — which is what
    #: every caller that builds this class straight from a recipe already
    #: declaring the field gets, unchanged.
    registered_at_override: str | None = None

    @property
    def name(self) -> str:
        return self.recipe.name

    @property
    def slot(self) -> str:
        return self.recipe.slot

    @property
    def arm_id(self) -> str:
        return self.recipe.arm_id

    @property
    def spec(self) -> dict[str, Any]:
        return self.recipe.spec

    @property
    def registered_at(self) -> str:
        if self.registered_at_override is not None:
            return self.registered_at_override
        # Never None here in that case: an arm resolved with no override and
        # no declared `registered_at` never reaches this class — it is
        # refused at load (:func:`_resolve_registered_at`).
        assert self.recipe.registered_at is not None
        return self.recipe.registered_at

    @property
    def params(self) -> dict[str, Any]:
        """What `run_grade._control_top_n` reads, and nothing else.

        An S arm selects no top-N: its book is every eligible name at the
        weight the optimizer solved. So it declares no `top_n` and the
        controls fall back to the count the helper defaults to — which is
        correct, because the controls check the GRADER, and the S grader's
        count-matching is the benchmark sleeve, not a selection size.
        """
        return {}


def _resolve_registered_at(
    recipe: StrategyRecipe,
    *,
    origin: str,
    register: ArmRegister | None,
    today: str | None,
) -> tuple[str | None, InputRefusal | None]:
    """The arm's resolved out-of-sample clock start, or why it refuses.

    **Ruled 2026-09-13** (`alpha-engine-config-I10634`, Brian, option (b)):
    the clock starts at the first cycle that produced scorable evidence, not
    at a filing date nobody can verify from a recipe file alone. So a recipe
    declaring no `registered_at` REGISTERS rather than refuses — the field
    is stamped by the S cycle job (`experiment.run[s]`, via
    :func:`load_strategy_slot`) from the arm's own register row if it
    already has one, or from ``today`` (this cycle's own trading day) on its
    first registering cycle. Every ladder rung the arena decides on —
    `promote_min_weeks`, `grace_weeks`, the retirement grace window — is
    counted in trading weeks from whatever this function returns, exactly as
    it always was for U/R arms via `crucible.slots.arms.ArmSpec`.

    A recipe that DOES declare `registered_at` is still refused in two
    cases, because a clock that can be moved is not a clock:

    * the date is AFTER ``today`` — an out-of-sample clock cannot start in
      the future;
    * the arm already has a register row and the declared date is AFTER
      that row's `created_date` — moving the clock forward once an arm is
      already serving would silently shorten every ladder rung already
      elapsed against the original date.

    Declaring a date that is on-or-before the arm's own first register row
    (or, for an arm not yet registered, any valid trading day at or before
    ``today``) is accepted and used verbatim — unchanged from how S recipes
    that declare the field always worked, and how U/R recipes work today.

    ``register`` and ``today`` are ``None`` for a caller with no cycle
    context — `crucible.track_a`'s manual `experiment.new` path, and any
    fixture that calls :func:`load_strategy_slot` directly. Such a caller
    keeps the PRE-`alpha-engine-config-I10634` behaviour: a recipe declaring
    no `registered_at` still refuses, because without a cycle's trading day
    there is no date to stamp it with and no register to consult for one.
    """
    existing: str | None = None
    if register is not None and recipe.arm_id in register:
        existing = register.state(recipe.arm_id).record.created_date

    if recipe.registered_at is not None:
        if today is not None and recipe.registered_at > today:
            return None, InputRefusal(
                arm=recipe.name,
                unresolvable=("registered_at",),
                reason=(
                    f"arm {recipe.name!r} ({origin}) declares `registered_at: "
                    f"{recipe.registered_at}`, which is AFTER {today}, this cycle's "
                    "trading day. An out-of-sample clock cannot start in the future: "
                    "declare a date on or before the day this arm actually registers, or "
                    "omit the field and let the S cycle job stamp it from the first run "
                    "that registers the arm."
                ),
            )
        if existing is not None and recipe.registered_at > existing:
            return None, InputRefusal(
                arm=recipe.name,
                unresolvable=("registered_at",),
                reason=(
                    f"arm {recipe.name!r} ({origin}) declares `registered_at: "
                    f"{recipe.registered_at}`, which is AFTER {existing}, the date this "
                    "arm's register row already carries. Every ladder rung is counted in "
                    "trading weeks from the FIRST register row, and a clock that can be "
                    "moved forward after the fact is not a clock: the recipe refuses "
                    "rather than re-dating an arm already serving."
                ),
            )
        return recipe.registered_at, None

    if existing is not None:
        return existing, None
    if today is not None:
        return today, None
    return None, InputRefusal(
        arm=recipe.name,
        unresolvable=("registered_at",),
        reason=(
            f"arm {recipe.name!r} ({origin}) declares no `registered_at`, so it has no "
            "out-of-sample clock and no cycle trading day was supplied to stamp one "
            "from. The S cycle job (`experiment.run --slot s`) stamps this field from "
            "the first run that registers the arm; loaded outside a cycle, the arm has "
            "nothing to be stamped with and refuses."
        ),
    )


def _adv_column_declared() -> bool:
    """Whether the feature catalog can ever produce the ADV series.

    A registration-time question about the CATALOG, not about one session's
    compiled artifact — a compiled-feature gap for one day is a per-session
    absence, and `crucible.portfolio._clean_adv`/`_build_tcost_term` already
    raise (`CostModelInputError`) if a participation-aware arm is ever handed
    no usable ADV coverage for the names it is pricing. This function answers
    the structural question underneath that: does this repository's feature
    layer declare the series at all.
    """
    return ADV_FEATURE_COLUMN in feature_names()


def _notional_field_declared() -> bool:
    """Whether the portfolio-params schema carries a real book notional.

    Also structural: `PortfolioParams.__post_init__` refuses a non-positive
    value and `PortfolioParams.from_mapping` refuses a payload missing the
    field entirely, so a slot whose strategy tree fails to declare a real
    book size RAISES loudly when the parameter set loads — it does not reach
    this function as a silent gap. This checks that the field the engine
    would read even exists in the schema, which is the fact that was false
    before `alpha-engine-config-I10669` (`GRADING_NOTIONAL = 1.0` was a
    module constant, not a configurable field at all).
    """
    return NOTIONAL_PARAM_FIELD in PORTFOLIO_PARAM_FIELDS


def _participation_refusal(recipe: StrategyRecipe) -> InputRefusal | None:
    """Why a registered arm cannot be CONSTRUCTED this cycle, or None.

    A participation-aware cost model prices each name's trade against that
    name's ADV and against the book's size. Both inputs are now wired end to
    end (`alpha-engine-config-I10669`): `_build_sessions` joins the feature
    layer's `dollar_volume_20d_raw` onto `SessionInputs.adv_usd`, and
    `construct_book`'s `portfolio_notional` is read from the slot's own
    `PortfolioParams.book_notional_usd` rather than a unit placeholder. So
    this refuses ONLY when one of the two is genuinely unresolvable — the
    catalog declares no ADV series, or the params schema declares no real
    notional field — never unconditionally on the cost model's kind, which
    would keep refusing every participation-priced arm forever regardless of
    whether its inputs are actually available.

    The refusal is at registration and PER ARM, and it must not be "fall back
    to a flat charge": a book graded under a model its recipe does not name is
    `alpha-engine-config-I6902` — the optimizer ran on a fallback cost model
    and nothing said so — reproduced inside v2. The RAISE that guards the same
    substitution one layer in (`grade_arm`'s `CostModelInputError`, and
    `construct_book`'s per-session pricing) is not softened by this and is
    tested still firing: this refusal stops the arm before it is handed
    nothing structural to work with; that raise stops it if a SPECIFIC
    session's compiled data still leaves it with no usable coverage.
    """
    if recipe.cost_model.kind == "flat":
        return None
    missing: list[str] = []
    if not _adv_column_declared():
        missing.append("adv_usd")
    if not _notional_field_declared():
        missing.append("portfolio_notional")
    if not missing:
        return None
    return InputRefusal(
        arm=recipe.name,
        unresolvable=tuple(missing),
        reason=(
            f"arm {recipe.name!r} names cost model {recipe.cost_model.name!r}, whose "
            f"kind {recipe.cost_model.kind!r} prices participation: it needs "
            f"{', '.join(missing)}, and the S construction path has no route to "
            f"{'either' if len(missing) > 1 else 'it'}. Refused BY NAME at registration; "
            "the model is NOT swapped for a flat one, because a book graded under a "
            "model its recipe does not name is a number whose cost nobody supplied."
        ),
    )


def registration_specs(loaded: SlotStrategies) -> list[RegisteredStrategyArm]:
    """The slot's loaded arms with their declared lineage resolved.

    Thin per-slot adapter over `crucible.slots.arms.resolve_declared_lineage`
    (`alpha-engine-config-I10637`, `policy-shared-code`'s second-adoption
    trigger: this function and `crucible.slots.model.registration_specs`
    carried the identical lineage-resolution guard, raising the identical
    :class:`~crucible.slots.arms.SupersededArmUndeclaredError`, over a
    different recipe type — `StrategyRecipe.with_rules` is the ONLY way an S
    arm is retuned (§3.1) and it always sets `supersedes`, so the moment an
    arm is refused every descendant of it would fail to register too, exactly
    the M-slot collision the shared resolver was written for). The
    resolution logic now lives once, in `crucible.slots.arms`; what stays
    here is only the S-specific shape: `loaded.registered` is already a
    tuple of :class:`RegisteredStrategyArm`, so the declared parent is read
    off `arm.recipe.supersedes` rather than off a bare recipe.
    """
    resolved = resolve_declared_lineage(
        registered=[(arm.name, arm.arm_id, arm.recipe.supersedes) for arm in loaded.registered],
        refused=[r.arm for r in loaded.refused],
    )
    return [
        RegisteredStrategyArm(
            arm.recipe,
            supersedes=resolved[arm.arm_id][0],
            notes=resolved[arm.arm_id][1],
            # Forwarded, not re-resolved: `arm.registered_at` already
            # carries whatever :func:`load_strategy_slot` resolved (a
            # declared date, a register-row date, or today's stamp), and
            # this rebuild must not lose it (`alpha-engine-config-I10634`).
            registered_at_override=arm.registered_at,
        )
        for arm in loaded.registered
    ]


def load_strategy_slot(
    *,
    store: Any = None,
    strategy_dir: Path | str | None = None,
    register: ArmRegister | None = None,
    today: str | None = None,
) -> SlotStrategies:
    """Every filed S recipe, split into the arms that register and those that do not.

    Mirrors `crucible.slots.arms.load_arm_specs`'s resolution order, which
    already solves exactly this for U/R arms: the checkout wins when
    ``strategy_dir`` is configured — a developer editing
    `alpha-engine-config/strategy/` expects the edit to take effect — and a
    box with no checkout (the production path) reads the tree synced into the
    store under `crucible.keys.strategy_arms_prefix('s')` instead.

    :func:`load_strategy_recipes` stays the directory-only reader that
    consumers validating recipe CONTENT call; this is the slot-aware entry
    point production code calls, so a job never has to know in advance
    whether it is running on a laptop or on a spot box.

    ``register`` (the slot's persisted `ArmRegister`, for resolving an
    already-registered arm's first row) and ``today`` (this cycle's trading
    day, for stamping a NEW arm's clock and refusing a future-dated one) are
    optional — see :func:`_resolve_registered_at` for what a caller that
    omits them gets.
    """
    from crucible.keys import strategy_arms_prefix  # noqa: PLC0415 - avoids a cycle

    sources: list[tuple[str, bytes]] = []
    if strategy_dir is not None:
        root = Path(strategy_dir) / "arms" / SLOT
        sources = [(str(path), path.read_bytes()) for path in sorted(root.glob("*.yaml"))]
    elif store is not None:
        prefix = strategy_arms_prefix(SLOT)
        sources = [
            (key, store.get_bytes(key))
            for key in sorted(store.list_keys(prefix))
            if key.endswith(".yaml")
        ]
    else:
        raise ValueError("load_strategy_slot needs either a store or a strategy_dir")

    registered: list[RegisteredStrategyArm] = []
    refused: list[InputRefusal] = []
    for origin, payload in sources:
        recipe = parse_strategy_document(payload, origin=origin)
        resolved_at, refusal = _resolve_registered_at(
            recipe, origin=origin, register=register, today=today
        )
        if refusal is None:
            refusal = _participation_refusal(recipe)
        if refusal is not None:
            refused.append(refusal)
            continue
        registered.append(RegisteredStrategyArm(recipe, registered_at_override=resolved_at))
    if sources and not registered:
        # Every arm refused: the slot can serve nothing, and that PAGES
        # through the ordinary failed-manifest path (plan §7 `unservable`).
        # The rows are recorded by the caller BEFORE this is raised.
        raise SlotUnservableError(tuple(refused))
    return SlotStrategies(registered=tuple(registered), refused=tuple(refused))


# ---------------------------------------------------------------------------
# Point-in-time session inputs.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedSession:
    """One session's construction inputs, as resolved on that session.

    Separate from :class:`SessionInputs` because the two carry different
    facts. `SessionInputs` is what `construct_book` walks and includes the
    session's REALIZED return, which does not exist at the decision date;
    this is what `experiment.run --slot s` knew at the decision date and
    persists, and the realized half is joined onto it at grade time.
    """

    trading_day: str
    tickers: tuple[str, ...]
    alpha_hat: tuple[float, ...]
    eligibility: tuple[bool, ...]
    stance_caps: tuple[float, ...]
    alpha_source: str
    eligibility_source: str
    sectors_source: str
    stance_caps_source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SESSION_INPUTS_SCHEMA_VERSION,
            "trading_day": self.trading_day,
            "tickers": list(self.tickers),
            "alpha_hat": [float(a) for a in self.alpha_hat],
            "eligibility": [bool(e) for e in self.eligibility],
            "stance_caps": [float(c) for c in self.stance_caps],
            # Every provenance field is REQUIRED and is a sentence, not a
            # flag. Three of the four record an ABSENCE today; a document that
            # simply omitted them would be indistinguishable from one whose
            # producers all exist, which is the state this artifact will be in
            # after phase 5 and must be told apart from now.
            "alpha_source": self.alpha_source,
            "eligibility_source": self.eligibility_source,
            "sectors_source": self.sectors_source,
            "stance_caps_source": self.stance_caps_source,
        }

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> ResolvedSession:
        version = document.get("schema_version")
        if version != SESSION_INPUTS_SCHEMA_VERSION:
            raise ValueError(
                f"session inputs declare schema_version {version!r}; this reader "
                f"understands {SESSION_INPUTS_SCHEMA_VERSION!r}. A document read against "
                "the wrong version is read wrong, not read approximately."
            )
        return cls(
            trading_day=str(document["trading_day"]),
            tickers=tuple(str(t) for t in document["tickers"]),
            alpha_hat=tuple(float(a) for a in document["alpha_hat"]),
            eligibility=tuple(bool(e) for e in document["eligibility"]),
            stance_caps=tuple(float(c) for c in document["stance_caps"]),
            alpha_source=str(document["alpha_source"]),
            eligibility_source=str(document["eligibility_source"]),
            sectors_source=str(document["sectors_source"]),
            stance_caps_source=str(document["stance_caps_source"]),
        )


def resolve_session(store: Any, *, trading_day: str, ctx: Any = None) -> ResolvedSession:
    """The point-in-time inputs for one session, from the champions' own feeds.

    **The alpha vector is the M champion's own predictions and nothing else.**
    `crucible.portfolio` trades a return forecast against a covariance
    estimated in return units; an R arm's ranker score is a z-score or a ratio
    and putting one on that axis produces a book whose risk/return tradeoff is
    meaningless while every artifact around it looks well-formed. So with no M
    champion the S slot has no alpha, and that is a
    :class:`~crucible.slots.cycle.MissingArtifactError` naming the pointer —
    not a substitution. Read through
    `crucible.slots.inputs.read_arm_predictions`, which checks the document's
    own `trading_day` against the key's, so a document written for a later
    session is refused rather than constructed on.

    **Eligibility is the U champion's cut**, which is what that slot's
    champion feed is for. With no U champion every priced name is eligible and
    the document SAYS so — an absence recorded, which is distinct from a cut
    that happened to include everything.
    """
    from crucible.documents import load_store_document  # noqa: PLC0415 - avoids a cycle
    from crucible.keys import champion_key, shadow_key  # noqa: PLC0415 - avoids a cycle
    from crucible.slots.cycle import MissingArtifactError  # noqa: PLC0415 - avoids a cycle
    from crucible.slots.inputs import read_arm_predictions  # noqa: PLC0415 - avoids a cycle

    m_pointer = champion_key("m")
    champion = (
        load_store_document(store, m_pointer).get("arm_id") if store.exists(m_pointer) else None
    )
    if not champion:
        raise MissingArtifactError(
            f"the S slot has no alpha for {trading_day}: the M champion pointer at "
            f"{m_pointer} names no arm. `crucible.portfolio` solves a return forecast "
            "against a covariance in return units, and an R arm's ranker score is a "
            "z-score or a ratio — substituting one would produce a book whose objective "
            "is meaningless while every artifact around it looks well-formed. Promote an "
            "M arm first; the S slot's book is constructed on its predictions."
        )
    predicted = read_arm_predictions(store, arm_id=champion, trading_day=trading_day, ctx=ctx)

    u_pointer = champion_key("u")
    u_champion = (
        load_store_document(store, u_pointer).get("arm_id") if store.exists(u_pointer) else None
    )
    cut: set[str] | None = None
    eligibility_source = (
        f"absent — {u_pointer} names no arm, so every name the M champion priced is "
        "eligible. A recorded absence, not a cut that happened to include everything."
    )
    if u_champion:
        u_shadow = shadow_key(u_champion, trading_day)
        if not store.exists(u_shadow):
            raise MissingArtifactError(
                f"the U champion pointer names {u_champion!r}, which wrote no cut at "
                f"{u_shadow}. The serving path resolves the pointer — it never imports a "
                "ranking function — so a pointer to an arm that did not produce means "
                "the S slot has no eligibility mask for this session."
            )
        cut = {str(t) for t in load_store_document(store, u_shadow)["selection"]}
        eligibility_source = u_shadow

    tickers = tuple(sorted(predicted))
    eligibility = tuple(True if cut is None else t in cut for t in tickers)
    return ResolvedSession(
        trading_day=trading_day,
        tickers=tickers,
        alpha_hat=tuple(float(predicted[t]) for t in tickers),
        eligibility=eligibility,
        # One cap per name, uniform, because no stance producer exists: an
        # arm's conviction stance is phase 5's. Uniform at 1.0 leaves the
        # per-name bound to `max_sector_pct`, `min_position_pct` and the
        # optimizer, which are the constraints that DO have declared values.
        stance_caps=tuple(1.0 for _ in tickers),
        alpha_source=arm_predictions_key(champion, trading_day),
        eligibility_source=eligibility_source,
        sectors_source=(
            "absent — the v2 price panel carries no sector column and the feature "
            f"catalogue declares no sector producer, so every name carries {UNSECTORED!r} "
            "and `max_sector_pct` binds on no group. Recorded rather than guessed."
        ),
        stance_caps_source=(
            "uniform 1.0 — no conviction-stance producer exists in v2 (phase 5). A "
            "recorded absence: the binding per-name constraints this cycle are "
            "`min_position_pct`, `max_sector_pct` and the optimizer's own."
        ),
    )


def _load_slot(ctx: Any, *, settings: Any) -> SlotStrategies:
    """Load the S slot and put every refusal on THIS run's manifest, first.

    The order matters in both directions:

    * a refused arm's `unservable` row is recorded BEFORE anything that can
      raise, so it reaches the manifest on the success path and the raise path
      alike — `crucible.runner.run_job` writes the manifest in a `finally`,
      and metrics recorded before the raise are on it;
    * when NOTHING registers, :class:`SlotUnservableError` is re-raised after
      the rows are recorded rather than propagating straight out of the
      loader. Without this the whole-slot case would page with a `reason` and
      no per-arm rows — the least informative manifest of the three possible
      outcomes, on the worst of them.

    Reads the slot's register and this cycle's trading day so
    :func:`load_strategy_slot` can resolve `registered_at` for a recipe that
    declares none (`alpha-engine-config-I10634`) — the register is always
    read from ``ctx.store`` regardless of ``strategy_dir``, since the arm
    register is a store artifact, not part of the strategy checkout.
    """
    from crucible.slots.arms import read_register  # noqa: PLC0415 - avoids a cycle

    strategy_dir = getattr(settings, "strategy_dir", None)
    register = read_register(ctx.store, SLOT)
    today = ctx.trading_day.isoformat()
    try:
        loaded = load_strategy_slot(
            store=None if strategy_dir is not None else ctx.store,
            strategy_dir=strategy_dir,
            register=register,
            today=today,
        )
    except SlotUnservableError as exc:
        for metric in SlotStrategies(registered=(), refused=exc.refusals).refusal_metrics(
            slot=SLOT
        ):
            ctx.record_metric(metric)
        raise
    for metric in loaded.refusal_metrics(slot=SLOT):
        ctx.record_metric(metric)
    return loaded


def _registered_arms(
    ctx: Any, *, settings: Any
) -> tuple[SlotStrategies, list[RegisteredStrategyArm]]:
    loaded = _load_slot(ctx, settings=settings)
    return loaded, registration_specs(loaded)


def _select(
    specs: list[RegisteredStrategyArm], arm_name: str | None
) -> list[RegisteredStrategyArm]:
    if arm_name is None:
        return specs
    from crucible.slots import arm_name as name_component  # noqa: PLC0415 - avoids a cycle
    from crucible.slots.cycle import MissingArtifactError  # noqa: PLC0415 - avoids a cycle

    # A bare name or a registered `{slot}:{name}:{spec_hash}` id, resolved
    # through the one parser that knows both shapes — `experiment.new` PRINTS
    # ids, so the id is what an operator has in the terminal.
    selector = name_component(arm_name)
    chosen = [s for s in specs if s.name == selector]
    if not chosen:
        raise MissingArtifactError(
            f"no arm named {selector!r} registered in slot {SLOT!r}; the slot registers "
            f"{sorted(s.name for s in specs)}. Producing nothing and exiting 0 would be "
            "indistinguishable from an arm that ran and had nothing to do."
        )
    return chosen


def produce(ctx: Any, *, settings: Any, **kwargs: Any) -> dict[str, Any]:
    """Record what every registered S arm SAW on one trading day.

    The S half of `experiment.run`, and the same signature as
    `crucible.slots.research.produce` because `crucible.track_a` dispatches
    all four slots through one call and `crucible.slots.dispatchable_slots`
    reads this name off the module.

    U, R and M produce a SELECTION here; S produces none, and inventing one
    would be a second answer to what an S arm decides. What it writes instead
    is the session's point-in-time construction inputs — the alpha vector, the
    eligibility mask and the caps, each with the key it came from — one
    document per arm, under :func:`crucible.keys.session_inputs_key`. The
    grade walks those documents rather than re-resolving the inputs, which is
    what makes an S grade point-in-time rather than a reconstruction over
    whatever the upstream artifacts say on the day it runs.

    It does NOT write the S champion feed (`strategies/current`, the trader's
    half of the contract). S has no champion, that feed has no key builder and
    no schema in this repository, and it is tracked separately — recorded here
    rather than silently absent.
    """
    from crucible.calendar import assert_trading_day  # noqa: PLC0415 - avoids a cycle
    from crucible.keys import session_inputs_key  # noqa: PLC0415 - avoids a cycle
    from crucible.slots import get_slot  # noqa: PLC0415 - avoids a cycle
    from crucible.slots.arms import (  # noqa: PLC0415 - avoids a cycle
        control_specs,
        read_register,
        register_arms,
        write_register,
    )

    trading_day = ctx.trading_day.isoformat()
    assert_trading_day(
        ctx.trading_day, context=f"experiment.run --slot {SLOT} --date {trading_day}"
    )
    loaded, specs = _registered_arms(ctx, settings=settings)

    register = read_register(ctx.store, SLOT)
    register, _ = register_arms(
        register, specs + control_specs(get_slot(SLOT)), filed_on=trading_day
    )
    write_register(ctx.store, SLOT, register)

    # Resolved ONCE for the session, not once per arm. The alpha vector, the
    # cut and the caps are the slot's inputs, not an arm's — an arm able to
    # resolve its own would be graded on a universe it chose, and the
    # comparison the slot exists to make would be confounded by the inputs
    # rather than decided by the exit rules. A failure here is slot-wide by
    # construction and that is correct: it is a compromised shared input.
    session = resolve_session(ctx.store, trading_day=trading_day, ctx=ctx)

    produced: list[str] = []
    for spec in _select(specs, kwargs.get("arm_name")):
        payload = json.dumps(session.to_dict(), indent=2, sort_keys=True).encode("utf-8")
        ctx.record_output(
            session_inputs_key(spec.arm_id, trading_day),
            payload,
            schema_version=SESSION_INPUTS_SCHEMA_VERSION,
        )
        produced.append(spec.arm_id)

    ctx.record_rows(rows_in=len(session.tickers), rows_out=len(produced))
    ctx.record_metric(
        {
            "name": "arms_produced",
            "module": f"crucible.slots.{SLOT}",
            "metric_type": "count",
            "value": float(len(produced)),
            "unit": "arms",
            "n_floor": 1,
            "status": "OK",
            "status_reason": (
                f"slot {SLOT}: {len(produced)} registered arm(s) recorded point-in-time "
                f"construction inputs for {trading_day} over {len(session.tickers)} "
                f"priced name(s), {sum(session.eligibility)} eligible"
            ),
            # Display-only, not a key any store call reads: this metric is
            # about every arm produced this cycle rather than one
            # (alpha-engine-config-I9852).
            "source_path": f"experiments/*/{trading_day}/session_inputs.json",
            "last_updated_utc": _utc_now(),
        }
    )
    return {
        "slot": SLOT,
        "trading_day": trading_day,
        "arms": produced,
        "refused": [{"arm": r.arm, "unresolvable": list(r.unresolvable)} for r in loaded.refused],
        "eligible_names": int(sum(session.eligibility)),
        "alpha_source": session.alpha_source,
        "champion": None,
        "feed_key": None,
    }


def produce_history(ctx: Any, *, settings: Any, **kwargs: Any) -> dict[str, Any]:
    """Record what the selected S arm SAW on ONE historical session.

    `experiment.backfill`'s per-session entry point for S
    (`alpha-engine-config-I11005`), and deliberately :func:`produce` itself
    rather than a second body: S has no serving half to omit. It writes no
    champion feed on a production cycle either — the S feed
    (`strategies/current`) has no key builder and no schema in this
    repository and is tracked separately — so the produce and the history
    paths are the same path, and the equality is declared here rather than
    inherited from a silent fallback in the resolver.

    U, R and M do NOT get to do this: each of those has a serving half that a
    backfill must not enter, which is why
    :func:`crucible.slots.history_producer` raises for a module that declares
    no ``produce_history`` instead of quietly using ``produce``.
    """
    return produce(ctx, settings=settings, **kwargs)


def _close_returns(panel: Any) -> Any:
    """Simple session-over-session returns, `trading_day x ticker`.

    Simple, never log: `construct_book` multiplies weights by these to get the
    book's return, and a weighted sum of log returns is not the log of the
    book's return. The whole class of defect this repository exists around is
    a units mismatch that every surface agreed on.
    """
    pivot = panel.pivot_table(index="trading_day", columns="ticker", values="close_raw")
    pivot = pivot.sort_index()
    # The index is stringified because the panel carries `datetime.date`
    # objects and every session key on this path — the artifact keys, the
    # documents' own `trading_day`, `ArmSeries.scores` — is an ISO STRING.
    # Comparing the two raises rather than mis-sorting, which is the good
    # failure; normalising once here is what stops a caller from being the
    # place that does it, differently, per call site (§4.12).
    pivot.index = [str(day) for day in pivot.index]
    return pivot.pct_change()


def _session_dates(store: Any, arm_id: str) -> list[str]:
    """Every trading day this arm recorded construction inputs for, ascending.

    Listed from the store rather than derived from a date range, for the same
    reason `crucible.slots.cycle._shadow_dates` is: an arm registered mid-window
    has no document before its registration, and inventing the dates would
    turn its absence into a run of zero-return sessions.
    """
    from crucible.keys import experiments_prefix  # noqa: PLC0415 - avoids a cycle

    prefix = experiments_prefix(arm_id)
    return sorted(
        key[len(prefix) :].split("/", 1)[0]
        for key in store.list_keys(prefix)
        if key.endswith("/session_inputs.json")
    )


def _read_session_adv(store: Any, *, feature_version: str, trading_day: str) -> dict[str, float]:
    """This trading day's `dollar_volume_20d_raw`, one entry per ticker the
    compiled feature layer carries for it — or an empty mapping.

    Absence here is deliberately NOT the ``_participation_refusal`` question
    ("can the catalog ever produce this series") — it is answered by whatever
    consumes the resulting `adv_usd` vector. `crucible.portfolio._clean_adv`
    reports an uncovered name as unusable, and a participation-aware cost
    model left with no usable name RAISES (`CostModelInputError`) rather than
    pricing the gap silently; a flat-cost arm never reads this vector at all,
    so a compiled-feature gap that would matter to nobody does not fail its
    grade.
    """
    key = features_key(feature_version, trading_day)
    if not store.exists(key):
        return {}
    frame = read_features(store.get_bytes(key))
    if ADV_FEATURE_COLUMN not in frame.columns:
        return {}
    return {
        str(ticker): float(value)
        for ticker, value in zip(frame["ticker"], frame[ADV_FEATURE_COLUMN], strict=True)
    }


def _adv_by_day(
    store: Any,
    *,
    feature_version: str,
    days: Iterable[str],
    cache: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Populate ``cache`` with every day in ``days`` not already resolved.

    Shared across the point-in-time pass and the contamination re-resolution
    pass in :func:`_attest`, which mostly walk the same decision dates — one
    feature read per day for the whole grade cycle, not one per arm.
    """
    for day in days:
        if day not in cache:
            cache[day] = _read_session_adv(store, feature_version=feature_version, trading_day=day)
    return cache


def _build_sessions(
    resolved: Sequence[ResolvedSession],
    *,
    returns: Any,
    benchmark: str,
    universe: BookUniverse,
    next_session: Mapping[str, str],
    adv_by_day: Mapping[str, Mapping[str, float]],
) -> list[SessionInputs]:
    """Join each recorded session onto the return it actually earned.

    The decision date's inputs are held through the NEXT session, and that
    session's return is what the book earned — the same settlement direction
    every other slot uses. A session whose successor the panel does not carry
    yet is not settled and is simply absent here; it enters on the first cycle
    after it settles, never as a zero.

    ``adv_by_day`` carries each session's `dollar_volume_20d_raw`, keyed by
    ticker (:func:`_read_session_adv`), and is joined onto ``SessionInputs.
    adv_usd`` — one entry per name in ``universe.tickers`` order, benchmark
    and cash sentinels excluded (they are never in ``session.tickers`` to
    begin with, so they are left NaN here, the same value
    `crucible.portfolio._clean_adv` already treats as "no coverage" and
    additionally excludes both sentinels from by index — belt and suspenders,
    not a second mechanism).
    """
    n = len(universe.tickers)
    index = {ticker: i for i, ticker in enumerate(universe.tickers)}
    lookback = COVARIANCE_LOOKBACK_TRADING_DAYS
    out: list[SessionInputs] = []
    for session in resolved:
        settle = next_session[session.trading_day]
        alpha = np.zeros(n)
        eligible = np.zeros(n, dtype=bool)
        caps = np.zeros(n)
        adv = np.full(n, np.nan)
        day_adv = adv_by_day.get(session.trading_day, {})
        for ticker, a, e, c in zip(
            session.tickers,
            session.alpha_hat,
            session.eligibility,
            session.stance_caps,
            strict=True,
        ):
            i = index[ticker]
            alpha[i], eligible[i], caps[i] = a, e, c
        # ADV is a property of the NAME on the day, read from the day's compiled
        # feature layer — not of whether the M champion priced it. A name in the
        # universe but not in this session (held from an earlier session, or
        # carried in by the caller's book) is ineligible and its whole holding is
        # a forced exit; the participation-aware cost model prices that exit off
        # this same point-in-time `dollar_volume_20d_raw`, never off a carried-
        # forward value (`alpha-engine-config-I10754`). Sentinels stay NaN.
        for ticker, i in index.items():
            if i in (universe.benchmark_idx, universe.cash_idx):
                continue
            value = day_adv.get(ticker)
            if value is not None:
                adv[i] = value
        # The two sentinels are always eligible and always uncapped: the
        # benchmark IS the no-conviction fill and cash is an equality pin, and
        # `crucible.portfolio._validate_inputs` refuses a book where either is
        # ineligible — correctly, since pinning one would pin the other.
        eligible[universe.benchmark_idx] = True
        eligible[universe.cash_idx] = True
        caps[universe.benchmark_idx] = 1.0
        caps[universe.cash_idx] = 1.0

        window = returns.loc[: session.trading_day].tail(lookback)
        panel_matrix = np.nan_to_num(
            window.reindex(columns=list(universe.tickers)).to_numpy(dtype="float64"), nan=0.0
        )
        panel_matrix[:, universe.cash_idx] = make_cash_sentinel_returns(panel_matrix.shape[0])

        realized_row = returns.loc[settle].reindex(list(universe.tickers))
        realized = np.nan_to_num(realized_row.to_numpy(dtype="float64"), nan=0.0)
        realized[universe.cash_idx] = 0.0
        out.append(
            SessionInputs(
                trading_day=session.trading_day,
                alpha_hat=alpha,
                eligibility=eligible,
                stance_caps=caps,
                realized_returns=realized,
                benchmark_return=float(returns.loc[settle, benchmark]),
                returns_panel=panel_matrix,
                adv_usd=adv,
            )
        )
    return out


def _universe_for(
    resolved: Sequence[ResolvedSession], *, benchmark: str, held: Iterable[str] = ()
) -> BookUniverse:
    """The union of every session's priced names and the held names, plus the two sentinels.

    ``held`` is the book carried INTO the first session (`alpha-engine-config-
    I10754`). A held name the recorded sessions did not price still enters the
    universe, and `_build_sessions` gives it alpha 0 and eligibility False on
    every session that did not price it — so the solver's ineligibility pin
    exits it and the recipe's cost model charges the exit. This is exactly what
    a multi-session walk already does for a name priced on an earlier session
    and dropped on a later one; passing ``held`` makes a walk that STARTS from
    a held book resolve that case identically.

    A union rather than an intersection, with the per-session eligibility mask
    carrying which names were actually priced on each day. `BookUniverse` is
    frozen and shared across the walk because a universe that changed shape
    mid-walk would make every weight vector a different object and the
    turnover between two of them meaningless — so the shape is fixed here and
    the MOVEMENT is expressed where it belongs, in `eligibility`.
    """
    names = sorted(
        ({t for session in resolved for t in session.tickers} | {str(t) for t in held})
        - {benchmark, CASH_TICKER}
    )
    tickers = (*names, benchmark, CASH_TICKER)
    return BookUniverse(
        tickers=tickers,
        sectors=(*(UNSECTORED for _ in names), BENCHMARK_SECTOR, CASH_SECTOR),
        benchmark_idx=len(names),
        cash_idx=len(names) + 1,
    )


def _attestation_precondition(verdict: PitParityVerdict) -> ServingPrecondition:
    """§9.1 as a SERVING precondition: `PASS`, or the arm does not serve.

    `PARTIAL` and `UNKNOWN` FAIL it, and that is the point of their existing.
    Policy §5.1: an uncomputed gate is not a pass, and plan §9.1 is explicit —
    "a card without `attestation: PASS` renders UNVERIFIED, never a grade". A
    contamination check that could not answer must never be rounded up to an
    answer of "no contamination", which is the direction this check may not
    fail in.
    """
    return ServingPrecondition(
        name="pit_parity_attested",
        passed=verdict.status == "PASS",
        reason=(
            f"contamination attestation {verdict.status}: {verdict.reason} "
            f"(mean delta {verdict.mean_delta}, coverage {verdict.coverage_fraction})"
        ),
    )


def grade(ctx: Any, *, settings: Any, **kwargs: Any) -> dict[str, Any]:
    """Construct every registered S arm's book and run the slot's arena cycle.

    `crucible.slots.cycle.run_grade`, unchanged, with four S-specific facts
    supplied to it as evaluated RESULTS rather than as computations the engine
    performs:

    * the arm set (`specs=`), because an S recipe is a `StrategyRecipe` and
      `crucible.slots.arms.load_arm_specs` refuses slot `s` by name;
    * the per-arm **series** (`series=`) from :func:`grade_arm` over a book
      :func:`construct_book` built — an S score is its book's return against
      the benchmark net of the cost the ENGINE charged, not a selection's
      realized excess return, and re-deriving it inside the cycle driver would
      be a second portfolio engine;
    * the **settled decision dates** (`settled_dates=`) the §10.1 controls are
      scored on, which the shadow loop would otherwise have supplied;
    * a **contamination-attestation serving precondition** per arm (§9.1),
      from :func:`pit_parity` over the point-in-time pass against a pass whose
      inputs are re-resolved TODAY. That delta is exactly the contamination
      `session_inputs_key` exists to make measurable: if anything upstream was
      revised since the session, the two passes disagree.

    The construction evidence reaches the manifest through
    `crucible.portfolio.portfolio_metric_record`, whose ``now_utc`` is PASSED
    rather than read from a clock inside the helper — a timestamp a function
    invents is a timestamp no test can pin.

    The refusal rows are recorded here too, on this manifest, for the same
    reason they are recorded on the produce manifest: a slot that became
    unservable between the two jobs must page from whichever one ran.
    """
    from crucible.slots.cycle import (  # noqa: PLC0415 - avoids a cycle
        MissingArtifactError,
        _read_panel,
        run_grade,
    )
    from crucible.slots.inputs import resolve_strategy_sessions  # noqa: PLC0415 - avoids a cycle

    loaded, specs = _registered_arms(ctx, settings=settings)
    as_of = ctx.trading_day.isoformat()
    params = load_portfolio_params_from_store(
        SLOT, store=ctx.store, strategy_dir=getattr(settings, "strategy_dir", None)
    )
    # The factor-neutral attribution spec (beta/sector/size ETF proxies) —
    # loaded once for the whole cycle, the same way `params` is, since it is
    # the apparatus every graded S arm's book is measured through and not a
    # per-arm choice (`strategy/slots/attribution.yaml`'s own header).
    attribution_params = load_attribution_params_from_store(
        store=ctx.store, strategy_dir=getattr(settings, "strategy_dir", None)
    )
    feature_version = kwargs.get("feature_version") or DEFAULT_FEATURE_VERSION
    panel = _read_panel(ctx.store, ctx.trading_day)
    returns = _close_returns(panel)
    sessions_index = [str(d) for d in returns.index]
    # Decision date -> the session it is held through. The last session in the
    # panel has no successor and is therefore UNSETTLED, which is a state, not
    # a zero-return day.
    next_session = dict(zip(sessions_index[:-1], sessions_index[1:], strict=True))
    # One feature read per decision day for the WHOLE cycle, shared by every
    # arm's point-in-time pass and its `_attest` re-resolution — not one per
    # arm, which would re-read the same day's artifact once per registered arm.
    adv_cache: dict[str, dict[str, float]] = {}

    series: dict[str, ArmSeries] = {}
    preconditions: dict[str, list[ServingPrecondition]] = {}
    graded: dict[str, dict[str, Any]] = {}
    settled: set[str] = set()
    for spec in specs:
        recipe = spec.recipe
        dates = [d for d in _session_dates(ctx.store, spec.arm_id) if d <= as_of]
        settleable = [d for d in dates if d in next_session]
        if not settleable:
            # The warm-up, per ARM rather than slot-wide — the same
            # blast-radius rule `alpha-engine-config-I9955` made one layer in.
            # An arm registered this cycle has recorded no session it has also
            # been held through, and an arm with no book has no series; the
            # engine is still SUPPLIED an empty one below, so it cannot null
            # another arm's figure. What is NOT absorbed is a compromised
            # panel or a cost model that cannot price — those raise and fail
            # the whole slot.
            ctx.record_metric(
                _refusal_metric(
                    SLOT,
                    InputRefusal(
                        arm=recipe.name,
                        unresolvable=("session_inputs",),
                        reason=(
                            f"arm {recipe.name!r} has no settled session on or before "
                            f"{as_of}: it recorded construction inputs for {dates or []} "
                            "and none of those sessions has a successor in the price "
                            "panel yet. A decision date is held through the NEXT session, "
                            "so the most recent one is never settled. Not a miss and not "
                            "a zero — it enters the series on the first cycle after it "
                            "settles."
                        ),
                    ),
                )
            )
            series[spec.arm_id] = ArmSeries(arm_id=spec.arm_id, scores={})
            continue

        # The ONE S-slot input resolution, shared with the trader
        # (`alpha-engine-config-I10654`): the recorded documents, the universe,
        # the benchmark check, ADV and the session join all happen inside
        # `resolve_strategy_sessions`, handed this cycle's already-loaded returns
        # and shared ADV cache so nothing is re-read per arm.
        arm_inputs = resolve_strategy_sessions(
            ctx.store,
            arm_id=spec.arm_id,
            benchmark=recipe.benchmark,
            decision_days=settleable,
            as_of=as_of,
            feature_version=feature_version,
            returns=returns,
            adv_cache=adv_cache,
        )
        resolved = list(arm_inputs.resolved)
        universe = arm_inputs.universe
        built = list(arm_inputs.sessions)
        w_initial = np.zeros(len(universe.tickers))
        w_initial[universe.cash_idx] = 1.0

        constructed = construct_book(
            recipe=recipe,
            params=params,
            universe=universe,
            sessions=built,
            portfolio_notional=params.book_notional_usd,
            w_initial=w_initial,
        )
        ctx.record_metric(portfolio_metric_record(constructed.evidence, now_utc=_utc_now()))
        ctx.record_metric(
            attribution_metric_record(
                _attribute_book(
                    constructed,
                    universe=universe,
                    returns=returns,
                    params=attribution_params,
                    as_of=as_of,
                ),
                now_utc=_utc_now(),
            )
        )
        arm_grade = grade_arm(recipe, constructed.book, as_of=as_of)
        series[spec.arm_id] = arm_grade.series
        settled.update(arm_grade.series.scores)

        attestation = _attest(
            ctx,
            recipe=recipe,
            params=params,
            universe=universe,
            resolved=resolved,
            returns=returns,
            next_session=next_session,
            w_initial=w_initial,
            point_in_time=arm_grade.series.scores,
            as_of=as_of,
            feature_version=feature_version,
            adv_cache=adv_cache,
        )
        preconditions[spec.arm_id] = [_attestation_precondition(attestation)]
        graded[spec.arm_id] = {
            "sessions": int(constructed.evidence["sessions"]),
            "cost_model": constructed.evidence["cost_model"]["name"],
            "cost_bps_total": float(constructed.evidence["cost_bps_total"]),
            "total_cost_bps": float(arm_grade.total_cost_bps),
            "attestation": attestation.status,
            # The DELTA, not only the verdict. A three-session window makes a
            # confidence interval wide enough that a real revision can still
            # render PASS, and a surface carrying only the verdict would then
            # be indistinguishable from one whose check saw nothing at all.
            "attestation_mean_delta": float(attestation.mean_delta or 0.0),
            "attestation_coverage": float(attestation.coverage_fraction or 0.0),
        }

    if not settled:
        raise MissingArtifactError(
            f"slot {SLOT!r} has no settled session on or before {as_of}. There is nothing "
            "to grade — which is a state, not a verdict, so this run FAILS rather than "
            "publishing an empty cycle. Record construction inputs at least one session "
            f"back first:\n    crucible experiment.run --slot {SLOT} --date <an earlier "
            "trading day>"
        )

    result = run_grade(
        ctx,
        slot=SLOT,
        settings=settings,
        # ONE session, not the canonical 21. An S decision is held through the
        # next session and its score is that session's book return, so the
        # §10.1 controls must be drawn and scored over the same span: a
        # planted-edge control ranking on a 21-session forward return would
        # check a grader nobody is using, and `run_grade` asserts a single
        # measured horizon per cycle precisely so the two cannot diverge
        # silently. The horizon travels ON the measurement here as everywhere
        # else — `crucible.slots.grading.ForwardReturnWindow` derives it from
        # the panel index rather than echoing this argument back.
        horizon_trading_days=S_HORIZON_TRADING_DAYS,
        specs=specs,
        preconditions=preconditions,
        series=series,
        settled_dates=sorted(settled),
        **{k: v for k, v in kwargs.items() if k not in {"arm_name", "feature_version"}},
    )
    result["strategy_grades"] = graded
    result["refused"] = [
        {"arm": r.arm, "unresolvable": list(r.unresolvable)} for r in loaded.refused
    ]
    return result


def _attest(
    ctx: Any,
    *,
    recipe: StrategyRecipe,
    params: PortfolioParams,
    universe: BookUniverse,
    resolved: Sequence[ResolvedSession],
    returns: Any,
    next_session: Mapping[str, str],
    w_initial: np.ndarray,
    point_in_time: Mapping[str, float],
    as_of: str,
    feature_version: str,
    adv_cache: dict[str, dict[str, float]],
) -> PitParityVerdict:
    """The §9.1 contamination attestation, measured rather than asserted.

    The point-in-time pass is the book built from what each session RECORDED.
    The second pass re-resolves each session's inputs from the upstream
    artifacts AS THEY STAND TODAY and builds the same book from those — so the
    per-date delta between the two is exactly the effect of any upstream
    revision since the session, which is the contamination
    `crucible.keys.session_inputs_key` exists to make measurable.

    A session that cannot be re-resolved today reduces COVERAGE rather than
    being dropped from the pairing: :func:`pit_parity` refuses a pairing whose
    two sides carry different dates outright, and partial coverage renders
    `PARTIAL`, which fails the serving precondition. Both are refusals; only
    one of them is silent, and it is the one this avoids.
    """
    current: list[ResolvedSession] = []
    for session in resolved:
        try:
            current.append(resolve_session(ctx.store, trading_day=session.trading_day))
        except Exception:  # noqa: BLE001 - see below
            # The ONLY swallow on this path, and it is not a degrade. What is
            # absorbed: "one session's upstream artifacts are no longer
            # resolvable today". Where it is recorded: the coverage fraction
            # below, which drives the verdict to PARTIAL and FAILS the serving
            # precondition — so an unmeasurable attestation cannot be read as
            # a passing one. The primary deliverable survives because the
            # point-in-time grade is already complete; this pass only decides
            # whether the arm may SERVE.
            continue
    if current:
        _adv_by_day(
            ctx.store,
            feature_version=feature_version,
            days=(s.trading_day for s in current),
            cache=adv_cache,
        )
        contaminated_book = construct_book(
            recipe=recipe,
            params=params,
            universe=universe,
            sessions=_build_sessions(
                current,
                returns=returns,
                benchmark=recipe.benchmark,
                universe=universe,
                next_session=next_session,
                adv_by_day=adv_cache,
            ),
            portfolio_notional=params.book_notional_usd,
            w_initial=w_initial,
        )
        contaminated = dict(grade_arm(recipe, contaminated_book.book, as_of=as_of).series.scores)
    else:
        contaminated = {}
    paired = {day: score for day, score in point_in_time.items() if day in contaminated}
    fraction = len(paired) / len(point_in_time) if point_in_time else 0.0
    return pit_parity(
        contaminated={day: contaminated[day] for day in paired},
        point_in_time=paired,
        coverage={
            "measured": bool(paired),
            "coverage_fraction": fraction,
            "budget_stopped": False,
        },
    )
