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

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from nousergon_lib.arena import ArmSeries, derive_arm_id

from crucible.portfolio import (
    CostModel,
    CostModelInputError,
    PortfolioParams,
    cost_model_from_mapping,
    portfolio_evidence,
    solve_target_weights,
)
from crucible.slots.vocab import refuse_unknown_keys

__all__ = [
    "ATTESTATION_STATUSES",
    "EXIT_RULES",
    "Book",
    "BookUniverse",
    "ConstructedBook",
    "CostModel",
    "ExitRuleSpec",
    "PitParityVerdict",
    "SessionInputs",
    "StrategyGrade",
    "StrategyRecipe",
    "WalkForwardFold",
    "WalkForwardSpec",
    "build_walk_forward_folds",
    "construct_book",
    "grade_arm",
    "load_strategy_recipes",
    "pit_parity",
    "render_verdict",
]

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

    def __post_init__(self) -> None:
        if not self.rules:
            raise ValueError(f"arm {self.name!r} declares no exit rules")
        seen = [r.rule_id for r in self.rules]
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
#: I9944`). `StrategyRecipe` has no `registered_at` field — an S arm is not
#: OOS-clocked the way U/R/M arms are (no such key is read anywhere in this
#: module) — so it is deliberately absent from this vocabulary too.
S_TOP_LEVEL_KEYS: frozenset[str] = frozenset({"slot", "name", "notes", "supersedes", "spec"})


def load_strategy_recipes(directory: Path | str) -> tuple[StrategyRecipe, ...]:
    """Load every `*.yaml` S recipe under ``directory``, sorted by filename.

    `alpha-engine-config/strategy/arms/s/` in production. Tuned values stay
    private; this repository holds the registry the values are checked against.
    """
    root = Path(directory)
    recipes: list[StrategyRecipe] = []
    for path in sorted(root.glob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        spec = payload.get("spec") or {}
        missing = [f for f in REQUIRED_STRATEGY_FIELDS if f not in spec]
        if missing:
            raise ValueError(
                f"{path}: recipe is missing pre-registration field(s) {missing}. Plan §9.1: "
                "missing fields mean the arm does not register."
            )
        refuse_unknown_keys(
            path=path,
            keys=set(payload),
            vocabulary=S_TOP_LEVEL_KEYS,
            level="top-level recipe",
            slot_label="S",
        )
        refuse_unknown_keys(
            path=path,
            keys=set(spec),
            vocabulary=S_SPEC_KEYS,
            level="spec",
            slot_label="S",
        )
        recipes.append(
            StrategyRecipe(
                slot=payload.get("slot", "s"),
                name=payload["name"],
                rules=tuple(
                    ExitRuleSpec(rule_id=r["rule_id"], params=dict(r.get("params") or {}))
                    for r in spec["rules"]
                ),
                cost_model=cost_model_from_mapping(spec["cost_model"], source=str(path)),
                walk_forward=WalkForwardSpec(**(spec.get("walk_forward") or {})),
                benchmark=spec.get("benchmark", "SPY"),
                supersedes=payload.get("supersedes"),
            )
        )
    return tuple(recipes)


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
