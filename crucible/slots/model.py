"""The M slot: which trained recipe emits `predicted_alpha`.

Normative sources: `champion-challenger-policy.md` §3 (improper training
fails the TASK), §3.1 (the recipe is the immutable unit; a refit is not a
change), §5.3 (the behavioural veto stays SCALE-DEPENDENT; input
completeness); plan §4.4, §9.1, §10.4.

Lifted, not imported, from `crucible-predictor`: the CPCV splitter
(`training/leakfree_meta_ic.py::cpcv_meta_oos_ic`), the behavioural veto
(`training/promotion_behavioral_veto.py`) and the input-completeness gate
(`training/data_completeness.py`). Nothing in this module imports from that
repository — v2 is a separate wheel, and a runtime edge back into v1 would
make the cutover impossible to finish.

**Four things this module is careful about, each a measured defect:**

1. **`TrainingIntegrityError` fails the whole slot, never one arm.** Arms in a
   slot share a training substrate, so a defect that spoils one fit is
   evidence the cycle's inputs are compromised. The week of 2026-08-29 every
   surface said healthy while every model was fitted with seven features
   hard-zeroed. `train_arm` raises; the runner's `finally` writes a `failed`
   manifest; nothing downstream sees a miss.
2. **The behavioural veto stays scale-DEPENDENT** (policy §5.3). A
   standardized version passes the collapsed 2026-08-28 model at 0.943 and the
   2026-08-21 model at 0.973, because dividing by the spread divides the
   collapse away. The rules below compare RAW dispersion against the
   incumbent's raw dispersion and apply ABSOLUTE floors. Do not normalise them.
3. **An uncomputable check is `insufficient`, never a pass.** Policy §5.1:
   "you cannot gate on a statistic you did not measure", and an uncomputed
   gate reported as a pass is the defect the gate exists to prevent.
4. **The benchmark is the scored cross-section, never SPY.** M is graded on
   CPCV out-of-sample IC against the population it scored (`ArenaConfig`
   refuses anything else for a selection-kind slot; M declares `population`
   for the same reason).
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field, replace
from itertools import combinations
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml
from nousergon_lib.arena import ArmSeries, ServingPrecondition, derive_arm_id
from nousergon_lib.arena.engine import TrainingIntegrityError, TrainingStatus

__all__ = [
    "DISPERSION_METRICS",
    "FLOOR_VETO_METRICS",
    "MIN_DISPERSION_RATIO",
    "UNITS_SUFFIXES",
    "ZERO_VETO_METRICS",
    "CPCVResult",
    "CPCVSpec",
    "CompletenessResult",
    "EstimatorSpec",
    "FeatureLayerSource",
    "FeaturePanel",
    "Fit",
    "ModelGrade",
    "ModelRecipe",
    "TrainingWindowSpec",
    "VetoResult",
    "assert_units_suffixes",
    "cpcv_oos_ic",
    "evaluate_behavioural_veto",
    "evaluate_input_completeness",
    "grade_arm",
    "load_model_recipes",
    "train_arm",
]

# ---------------------------------------------------------------------------
# The behavioural veto — policy §5.3 precondition 1, in its measured form.
# ---------------------------------------------------------------------------

#: A candidate whose dispersion falls below this fraction of the INCUMBENT's is
#: refused. Half is the stated bar, not a tuned threshold: halving the spread
#: the executor ranks on is a different model, not a better one.
MIN_DISPERSION_RATIO = 0.5

#: Compared as a ratio against the incumbent's value for the SAME metric, in
#: the metric's own units. Not standardized: dividing by the spread is exactly
#: the transformation that made 2026-08-28's collapse read as healthy.
DISPERSION_METRICS: tuple[str, ...] = ("alpha_stdev", "stdev_p_up")

#: Vetoed on an absolute zero — no incumbent value needed. A model that names
#: nothing with high confidence produced five live sessions with zero names on
#: 2026-08-21 while every ratio said healthy.
ZERO_VETO_METRICS: tuple[str, ...] = ("n_high_confidence",)

#: Vetoed on falling below an absolute floor, in the metric's own units. This
#: is the scale-DEPENDENT part by construction: it asserts the metric is a
#: 0–1 proportion. A producer emitting a percentage would break it loudly at
#: the first cycle, which is the correct failure.
FLOOR_VETO_METRICS: dict[str, float] = {"model_hit_rate_30d": 0.50}

#: The fleet's units-suffix contract (`AGENTS.md`): `avg_volume_20d` was
#: emitted as a normalized ratio and consumed as raw shares, silently failing
#: 901 of 903 tickers for months. A recipe may not name a bare column.
UNITS_SUFFIXES: tuple[str, ...] = ("_raw", "_ratio", "_pct", "_zscore", "_log_return")


@dataclass(frozen=True)
class VetoResult:
    """The veto's verdict, with every reason it reached it."""

    status: str  # veto | pass | insufficient
    reasons: tuple[str, ...] = ()
    uncomputable: tuple[str, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)

    def as_precondition(self) -> ServingPrecondition:
        """The engine consumes an evaluated RESULT, never a computation.

        Policy §5.3: "Both are per-slot facts and are supplied to the engine
        as evaluated results; the engine does not compute them and must not
        be given a default." `insufficient` fails the precondition — an
        uncomputed gate reported as a pass is the defect it prevents.
        """
        passed = self.status == "pass"
        if self.status == "insufficient":
            reason = (
                "behavioural veto could not be computed for "
                f"{', '.join(self.uncomputable)}; an uncomputed gate is not a pass "
                "(champion-challenger-policy.md §5.1)"
            )
        else:
            reason = "; ".join(self.reasons)
        return ServingPrecondition(name="behavioural_veto", passed=passed, reason=reason)


def evaluate_behavioural_veto(
    candidate: dict[str, Any],
    incumbent: dict[str, Any],
    *,
    min_dispersion_ratio: float = MIN_DISPERSION_RATIO,
) -> VetoResult:
    """The M slot's serving veto. **Keep it scale-dependent.**

    Three rule families, applied to raw metric values:

    * ``DISPERSION_METRICS`` — ``candidate / incumbent < min_dispersion_ratio``;
    * ``ZERO_VETO_METRICS`` — ``candidate == 0``;
    * ``FLOOR_VETO_METRICS`` — ``candidate < floor``.

    A metric absent from either side, or an incumbent dispersion of zero, is
    recorded in ``uncomputable`` and makes the whole result ``insufficient``.
    Any ``standardized_*`` key present on the inputs is deliberately IGNORED:
    it is carried by producers for reporting, and reading it here would
    reintroduce the exact normalisation policy §5.3 forbids.
    """
    reasons: list[str] = []
    uncomputable: list[str] = []
    metrics: dict[str, Any] = {}

    for name in DISPERSION_METRICS:
        cand = candidate.get(name)
        inc = incumbent.get(name)
        if cand is None or inc is None:
            uncomputable.append(name)
            continue
        if float(inc) == 0.0:
            uncomputable.append(name)
            continue
        ratio = float(cand) / float(inc)
        metrics[f"{name}_ratio_vs_incumbent"] = ratio
        if ratio < min_dispersion_ratio:
            reasons.append(
                f"{name}={cand!r} is {ratio:.3f} of the incumbent's {inc!r}, below the "
                f"{min_dispersion_ratio} bar: halving the spread the executor ranks on "
                "is a different model, not a better one"
            )

    for name in ZERO_VETO_METRICS:
        cand = candidate.get(name)
        if cand is None:
            uncomputable.append(name)
            continue
        metrics[name] = cand
        if float(cand) == 0.0:
            reasons.append(
                f"{name}=0: the model names nothing at high confidence, which is not a "
                "conservative model, it is a model with no output to serve"
            )

    for name, floor in FLOOR_VETO_METRICS.items():
        cand = candidate.get(name)
        if cand is None:
            uncomputable.append(name)
            continue
        metrics[name] = cand
        if float(cand) < floor:
            reasons.append(f"{name}={cand!r} is below the absolute floor {floor}")

    if reasons:
        return VetoResult("veto", tuple(reasons), tuple(uncomputable), metrics)
    if uncomputable:
        return VetoResult("insufficient", (), tuple(uncomputable), metrics)
    return VetoResult("pass", (), (), metrics)


# ---------------------------------------------------------------------------
# Input completeness — policy §5.3 precondition 2.
# ---------------------------------------------------------------------------

#: An input whose newest observation is older than this many calendar days is
#: stale. Calendar, not trading, days: this is a data-freshness property of an
#: upstream vendor feed, which is one of §4.12's exhaustive exceptions — the
#: value it gates is never an input to a promotion or grading decision, only
#: to whether the arm may serve at all.
DEFAULT_MAX_STALENESS_DAYS = 7

#: An input with fewer than this fraction of the expected rows is incomplete.
DEFAULT_MIN_ROWS_RATIO = 0.90


@dataclass(frozen=True)
class CompletenessResult:
    """Per-input coverage grades and the precondition they imply."""

    grades: dict[str, dict[str, Any]]
    failures: tuple[str, ...]

    def as_precondition(self) -> ServingPrecondition:
        if not self.failures:
            return ServingPrecondition(
                name="input_completeness", passed=True, reason="every required input complete"
            )
        detail = "; ".join(
            f"{name}: {self.grades[name]['status']} ({self.grades[name]['detail']})"
            for name in self.failures
        )
        return ServingPrecondition(
            name="input_completeness",
            passed=False,
            reason=(
                f"required input(s) incomplete — {detail}. An arm scored on partial "
                "inputs may rank first and still be unfit to trade "
                "(champion-challenger-policy.md §5.3)."
            ),
        )


def evaluate_input_completeness(
    *,
    required: tuple[str, ...],
    observed: dict[str, dict[str, Any]],
    expected_rows: int,
    as_of: dt.date,
    min_rows_ratio: float = DEFAULT_MIN_ROWS_RATIO,
    max_staleness_days: int = DEFAULT_MAX_STALENESS_DAYS,
) -> CompletenessResult:
    """Grade every required input on rows and freshness.

    Root cause this exists in this shape: a symbol-PRESENCE coverage ratio
    read 1.0 while a 16-row VIX3M series forward-filled into a full macro
    feature column across 799 dates. Rows are counted, not symbols.
    """
    grades: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for name in required:
        row = observed.get(name)
        if row is None:
            grades[name] = {"status": "absent", "detail": "no observation reported"}
            failures.append(name)
            continue
        rows = int(row.get("rows", 0))
        ratio = rows / expected_rows if expected_rows else 0.0
        last = row.get("last_date")
        staleness = None
        if last:
            staleness = (as_of - dt.date.fromisoformat(str(last))).days
        if ratio < min_rows_ratio:
            grades[name] = {
                "status": "insufficient_rows",
                "detail": f"{rows}/{expected_rows} rows = {ratio:.3f} < {min_rows_ratio}",
                "rows_ratio": ratio,
            }
            failures.append(name)
        elif staleness is None:
            grades[name] = {"status": "unknown_dates", "detail": "no last_date reported"}
            failures.append(name)
        elif staleness > max_staleness_days:
            grades[name] = {
                "status": "stale",
                "detail": f"newest observation {last} is {staleness}d old",
                "rows_ratio": ratio,
            }
            failures.append(name)
        else:
            grades[name] = {
                "status": "ok",
                "detail": f"{rows} rows, newest {last}",
                "rows_ratio": ratio,
            }
    return CompletenessResult(grades=grades, failures=tuple(failures))


# ---------------------------------------------------------------------------
# The feature panel and the layer that produces it.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeaturePanel:
    """A (trading_day x name) block of features and their forward labels.

    Held as arrays rather than a dataframe so this module depends on numpy
    alone: parquet, the registry and the point-in-time read are track A's
    feature layer (plan §10.4), and pulling pandas in here would put it on
    every consumer's cold start.
    """

    dates: tuple[str, ...]
    names: tuple[str, ...]
    features: dict[str, np.ndarray]
    forward_returns: np.ndarray
    feature_version: str

    def __post_init__(self) -> None:
        shape = (len(self.dates), len(self.names))
        if self.forward_returns.shape != shape:
            raise ValueError(
                f"forward_returns has shape {self.forward_returns.shape}, expected {shape}"
            )
        for name, block in self.features.items():
            if block.shape != shape:
                raise ValueError(f"feature {name!r} has shape {block.shape}, expected {shape}")

    def column(self, name: str) -> np.ndarray:
        try:
            return self.features[name]
        except KeyError as exc:
            raise KeyError(
                f"feature {name!r} is not in this panel (version {self.feature_version!r}); "
                f"available: {sorted(self.features)}. A recipe naming a column the layer "
                "does not produce must fail here, not train on a silently substituted zero."
            ) from exc

    def head(self, n: int) -> FeaturePanel:
        """The first ``n`` trading days. Used to build deliberately thin panels."""
        return FeaturePanel(
            dates=self.dates[:n],
            names=self.names,
            features={k: v[:n] for k, v in self.features.items()},
            forward_returns=self.forward_returns[:n],
            feature_version=self.feature_version,
        )

    def with_zeroed(self, columns: tuple[str, ...]) -> FeaturePanel:
        """The 2026-08-28 condition: named columns hard-zeroed.

        A fixture rather than a production path — but the production path
        must fail on it, so it lives beside the panel it corrupts.
        """
        zeroed = dict(self.features)
        for name in columns:
            zeroed[name] = np.zeros_like(self.column(name))
        return replace(self, features=zeroed)


class FeatureRegistry(Protocol):
    """The registry interface track A's `crucible/features/` exposes."""

    def __iter__(self) -> Any: ...


@dataclass(frozen=True)
class FeatureLayerSource:
    """Reads a versioned feature panel through track A's registry.

    Track B commits to the CONSUMER side of plan §10.4: the M slot resolves
    its declared columns through the registry and records the layer's version
    and hash as a manifest input, so R and M provably read the same artifact.
    The producer — `features/{version}/{trading_day}.parquet`, its registry
    and its lineage — is track A's, and until it lands this raises rather
    than falling back to recomputing features locally. A local recomputation
    is the precise defect the feature layer exists to remove: R and M
    computing from different code, so "the signal degraded" cannot be
    separated from "the feature changed".
    """

    registry: Any
    version: str

    def key(self, trading_day: str) -> str:
        """The documented path shape, single-sourced here."""
        return f"features/{self.version}/{trading_day}.parquet"

    def panel(self, *, trading_day: str, columns: tuple[str, ...]) -> FeaturePanel:
        raise NotImplementedError(
            "the feature layer producer is track A's (crucible/features/, "
            "alpha-engine-config-I9757, plan §10.4). This consumer reads "
            f"{self.key(trading_day)} through the registry; it deliberately does NOT "
            "fall back to recomputing features locally, which would put R and M on "
            "different code and make a signal change indistinguishable from a "
            "feature change."
        )


def assert_units_suffixes(columns: tuple[str, ...]) -> None:
    """Refuse a bare column name.

    `AGENTS.md`: `avg_volume_20d` was emitted as a normalized ratio and
    consumed as raw shares, silently failing 901 of 903 tickers on the
    scanner liquidity gate for months. There is no grandfather list here —
    this repository carries no suppression collection at all (§11.1) — so a
    v1 column arrives in v2 renamed or not at all.
    """
    bare = sorted(c for c in columns if not c.endswith(UNITS_SUFFIXES))
    if bare:
        raise ValueError(
            f"feature column(s) {bare} carry no units suffix; every column must end in "
            f"one of {list(UNITS_SUFFIXES)}. A bare name is how `avg_volume_20d` was "
            "emitted as a ratio and consumed as raw shares."
        )


# ---------------------------------------------------------------------------
# The recipe — the immutable unit (policy §3.1).
# ---------------------------------------------------------------------------

_ESTIMATORS: tuple[str, ...] = ("ridge", "ols")


@dataclass(frozen=True)
class EstimatorSpec:
    """Which estimator, with which hyperparameters. Part of the recipe hash."""

    kind: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in _ESTIMATORS:
            raise ValueError(
                f"unknown estimator {self.kind!r}; registered kinds are {list(_ESTIMATORS)}. "
                "An estimator resolved by name at runtime rather than from this closed "
                "set is an arm whose recipe does not describe what it fits."
            )

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "params": dict(self.params)}


@dataclass(frozen=True)
class TrainingWindowSpec:
    """The training-window RULE, fixed at registration (policy §3.1)."""

    kind: str = "expanding"
    min_trading_days: int = 504

    def __post_init__(self) -> None:
        if self.kind not in ("expanding", "rolling"):
            raise ValueError(f"training_window.kind must be expanding|rolling; got {self.kind!r}")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "min_trading_days": self.min_trading_days}


@dataclass(frozen=True)
class CPCVSpec:
    """Combinatorial purged cross-validation parameters (López de Prado)."""

    n_groups: int = 6
    k_test: int = 2
    embargo_trading_days: int = 2

    def __post_init__(self) -> None:
        if self.k_test < 1 or self.k_test >= self.n_groups:
            raise ValueError(
                f"k_test must be in [1, n_groups); got k_test={self.k_test}, "
                f"n_groups={self.n_groups}"
            )
        if self.embargo_trading_days < 0:
            raise ValueError("embargo_trading_days must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_groups": self.n_groups,
            "k_test": self.k_test,
            "embargo_trading_days": self.embargo_trading_days,
        }


@dataclass(frozen=True)
class ModelRecipe:
    """One M arm. Its id is the hash of this spec (policy §3.1).

    Editing any field below produces a different id, so an edited recipe
    cannot inherit a track record. A refit on `refit_cadence_trading_days`
    changes nothing here — same id, continuous series.
    """

    name: str
    features: tuple[str, ...]
    estimator: EstimatorSpec
    label_horizon_trading_days: int
    refit_cadence_trading_days: int
    training_window: TrainingWindowSpec
    cpcv: CPCVSpec
    feature_version: str
    supersedes: str | None = None
    slot: str = "m"

    def __post_init__(self) -> None:
        if not self.features:
            raise ValueError(f"arm {self.name!r} declares no features")
        assert_units_suffixes(self.features)
        if self.label_horizon_trading_days < 1:
            raise ValueError("label_horizon_trading_days must be >= 1 (trading days, §4.12)")
        if self.refit_cadence_trading_days < 1:
            raise ValueError("refit_cadence_trading_days must be >= 1 (trading days, §4.12)")

    @property
    def spec(self) -> dict[str, Any]:
        """The canonical spec the arm id hashes. Order-independent by key."""
        return {
            "features": list(self.features),
            "estimator": self.estimator.to_dict(),
            "label_horizon_trading_days": self.label_horizon_trading_days,
            "refit_cadence_trading_days": self.refit_cadence_trading_days,
            "training_window": self.training_window.to_dict(),
            "cpcv": self.cpcv.to_dict(),
            "feature_version": self.feature_version,
        }

    @property
    def arm_id(self) -> str:
        return derive_arm_id(self.slot, self.name, self.spec)


#: Every field a recipe must declare before its first score (plan §9.1
#: pre-registration). Missing any of them and the arm does not register.
REQUIRED_RECIPE_FIELDS: tuple[str, ...] = (
    "features",
    "estimator",
    "label_horizon_trading_days",
    "refit_cadence_trading_days",
    "training_window",
    "cpcv",
    "feature_version",
)


def load_model_recipes(directory: Path | str) -> tuple[ModelRecipe, ...]:
    """Load every `*.yaml` M recipe under ``directory``, sorted by name.

    The directory is `alpha-engine-config/strategy/arms/m/` in production —
    recipes are strategy content and live in the private repository
    (`repository-tiering-policy` test 2). This repository holds the shape.
    """
    root = Path(directory)
    recipes: list[ModelRecipe] = []
    for path in sorted(root.glob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        spec = payload.get("spec") or {}
        missing = [f for f in REQUIRED_RECIPE_FIELDS if f not in spec]
        if missing:
            raise ValueError(
                f"{path}: recipe is missing pre-registration field(s) {missing}. Plan §9.1: "
                "an arm declares its slot, recipe and lineage BEFORE its first score, and "
                "missing fields mean the arm does not register — a half-declared arm's "
                "verdicts cannot be interpreted later."
            )
        recipes.append(
            ModelRecipe(
                slot=payload.get("slot", "m"),
                name=payload["name"],
                features=tuple(spec["features"]),
                estimator=EstimatorSpec(
                    kind=spec["estimator"]["kind"],
                    params={k: v for k, v in spec["estimator"].items() if k != "kind"},
                ),
                label_horizon_trading_days=int(spec["label_horizon_trading_days"]),
                refit_cadence_trading_days=int(spec["refit_cadence_trading_days"]),
                training_window=TrainingWindowSpec(**spec["training_window"]),
                cpcv=CPCVSpec(**spec["cpcv"]),
                feature_version=str(spec["feature_version"]),
                supersedes=payload.get("supersedes"),
            )
        )
    return tuple(recipes)


# ---------------------------------------------------------------------------
# Fitting.
# ---------------------------------------------------------------------------

#: A design column whose standard deviation is below this is degenerate. It is
#: an absolute floor on a z-scored/ratio feature, deliberately: the 2026-08-28
#: condition was columns hard-ZEROED, and a relative test against the column's
#: own scale cannot see a column that has no scale.
DEGENERATE_STD = 1e-9


@dataclass(frozen=True)
class Fit:
    """One fitted arm. The weights are the refit; the recipe is the arm."""

    arm_id: str
    recipe: ModelRecipe
    coefficients: np.ndarray
    intercept: float
    fitted_at: str
    n_rows: int
    training_status: TrainingStatus


def _design(recipe: ModelRecipe, panel: FeaturePanel, rows: np.ndarray) -> np.ndarray:
    return np.column_stack([panel.column(f).reshape(-1)[rows] for f in recipe.features])


def _assert_trainable(recipe: ModelRecipe, matrix: np.ndarray, labels: np.ndarray) -> None:
    """Raise TrainingIntegrityError on any condition that voids the fit.

    A hard raise, never a status field: policy §3 is explicit that a status a
    caller can read and ignore is how a complete-looking artifact gets written
    over void inputs.
    """
    if matrix.size == 0 or matrix.shape[0] < matrix.shape[1] + 2:
        raise TrainingIntegrityError(
            f"arm {recipe.name}: {matrix.shape[0]} usable row(s) for "
            f"{matrix.shape[1]} feature(s) — not enough to fit. This fails the whole "
            "slot's cycle rather than recording a miss: a miss means the arm "
            "legitimately had nothing to say, and broken inputs are not that "
            "(champion-challenger-policy.md §3)."
        )
    if not np.isfinite(matrix).all() or not np.isfinite(labels).all():
        raise TrainingIntegrityError(
            f"arm {recipe.name}: non-finite values in the training block. The cycle does "
            "not run on inputs nobody vouched for (Brian ruling 2026-08-29)."
        )
    stds = matrix.std(axis=0)
    dead = [recipe.features[i] for i, s in enumerate(stds) if float(s) < DEGENERATE_STD]
    if dead:
        raise TrainingIntegrityError(
            f"arm {recipe.name}: feature(s) {dead} are constant across the training "
            "block — the 2026-08-28 seven-hard-zeroed-features condition. Every "
            "surface said healthy that week; this raises instead, which fails the "
            "whole slot (champion-challenger-policy.md §3)."
        )


def train_arm(recipe: ModelRecipe, panel: FeaturePanel, *, as_of: str) -> Fit:
    """Fit ``recipe`` on ``panel`` up to ``as_of``. Raises on an unsound fit.

    The refit is the arm executing the cadence its own recipe declares: the
    id and the score series are unchanged across it (policy §3.1), which is
    why `fitted_at` is on the :class:`Fit` and not in the arm id.
    """
    usable = [i for i, d in enumerate(panel.dates) if d <= as_of]
    if len(usable) < recipe.training_window.min_trading_days:
        raise TrainingIntegrityError(
            f"arm {recipe.name}: {len(usable)} trading day(s) available up to {as_of}, "
            f"below the recipe's declared min_trading_days="
            f"{recipe.training_window.min_trading_days}. The window rule is part of the "
            "recipe, so a fit on a shorter one is a different arm's fit."
        )
    if recipe.training_window.kind == "rolling":
        usable = usable[-recipe.training_window.min_trading_days :]

    rows = np.array([d * len(panel.names) + n for d in usable for n in range(len(panel.names))])
    matrix = _design(recipe, panel, rows)
    labels = panel.forward_returns.reshape(-1)[rows]
    _assert_trainable(recipe, matrix, labels)

    coefficients, intercept = _fit_linear(recipe.estimator, matrix, labels)
    return Fit(
        arm_id=recipe.arm_id,
        recipe=recipe,
        coefficients=coefficients,
        intercept=intercept,
        fitted_at=as_of,
        n_rows=int(matrix.shape[0]),
        training_status=TrainingStatus(arm_id=recipe.arm_id, ok=True),
    )


def _fit_linear(
    estimator: EstimatorSpec, matrix: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, float]:
    """Ridge / OLS by normal equations, centred.

    Closed form rather than an iterative solver: at a handful of features the
    normal equations are exact, deterministic and seed-free, and determinism
    is a plan §9.1 requirement (a rerun of a replayed date must reproduce the
    verdict byte for byte).
    """
    centre = matrix.mean(axis=0)
    centred = matrix - centre
    label_mean = float(labels.mean())
    gram = centred.T @ centred
    if estimator.kind == "ridge":
        alpha = float(estimator.params.get("alpha", 1.0))
        if alpha < 0:
            raise ValueError(f"ridge alpha must be >= 0; got {alpha}")
        gram = gram + alpha * np.eye(gram.shape[0])
    coefficients = np.linalg.solve(gram, centred.T @ (labels - label_mean))
    intercept = label_mean - float(centre @ coefficients)
    return coefficients, intercept


def _predict(fit: Fit, panel: FeaturePanel, rows: np.ndarray) -> np.ndarray:
    return _design(fit.recipe, panel, rows) @ fit.coefficients + fit.intercept


# ---------------------------------------------------------------------------
# CPCV — combinatorial purged cross-validation with an embargo.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CPCVFold:
    """One combination's train / test / purged date sets."""

    test_groups: tuple[int, ...]
    train_dates: tuple[str, ...]
    test_dates: tuple[str, ...]
    purged_dates: tuple[str, ...]


@dataclass(frozen=True)
class CPCVResult:
    """The out-of-sample IC distribution across every group combination."""

    status: str  # ok | unmeasurable
    reason: str
    ics: tuple[float, ...]
    folds: tuple[CPCVFold, ...]
    n_backtest_paths: int

    @property
    def mean_ic(self) -> float:
        if not self.ics:
            raise ValueError(
                f"mean_ic on an unmeasurable CPCV result: {self.reason}. "
                "n_dates == 0 across all folds is a defect, not a result "
                "(champion-challenger-policy.md §7.2)."
            )
        return float(np.mean(self.ics))


def _contiguous_groups(n: int, n_groups: int) -> list[tuple[int, int]]:
    """``n_groups`` contiguous [start, end] index blocks over ``n`` dates.

    Contiguous, not interleaved: the labels are overlapping forward returns,
    so an interleaved split leaks the test block's label into training
    through the overlap the purge is sized to remove.
    """
    size = n / n_groups
    bounds: list[tuple[int, int]] = []
    for g in range(n_groups):
        start = int(round(g * size))
        end = int(round((g + 1) * size)) - 1
        bounds.append((start, end))
    return bounds


def cpcv_oos_ic(
    panel: FeaturePanel,
    *,
    recipe: ModelRecipe,
    cpcv: CPCVSpec,
    label_horizon_trading_days: int,
) -> CPCVResult:
    """Out-of-sample rank IC over every ``C(n_groups, k_test)`` combination.

    Purge and embargo, verbatim in behaviour from the lifted implementation:
    every date within ``label_horizon_trading_days`` BEFORE a test block, and
    ``embargo_trading_days`` AFTER it, is removed from training. The first
    window is the overlapping-label purge; the second is the serial-correlation
    embargo. Removing only one of them is the leak this whole construction
    exists to close.

    An empty or unfittable fold makes the whole result ``unmeasurable`` with a
    reason. It is never a zero IC: `n_dates == 0` is a defect, not a result.
    """
    n = len(panel.dates)
    if n < cpcv.n_groups:
        return CPCVResult(
            "unmeasurable",
            f"{n} trading day(s) cannot be split into {cpcv.n_groups} groups",
            (),
            (),
            0,
        )
    groups = _contiguous_groups(n, cpcv.n_groups)
    folds: list[CPCVFold] = []
    ics: list[float] = []
    n_paths = int(cpcv.k_test * math.comb(cpcv.n_groups, cpcv.k_test) / cpcv.n_groups)

    for combo in combinations(range(cpcv.n_groups), cpcv.k_test):
        is_test = np.zeros(n, dtype=bool)
        excluded = np.zeros(n, dtype=bool)
        for gi in combo:
            a, b = groups[gi]
            is_test[a : b + 1] = True
            lo = max(0, a - label_horizon_trading_days)
            hi = min(n - 1, b + cpcv.embargo_trading_days)
            excluded[lo : hi + 1] = True
        keep = (~is_test) & (~excluded)
        purged = excluded & (~is_test)

        fold = CPCVFold(
            test_groups=combo,
            train_dates=tuple(d for d, k in zip(panel.dates, keep, strict=True) if k),
            test_dates=tuple(d for d, k in zip(panel.dates, is_test, strict=True) if k),
            purged_dates=tuple(d for d, k in zip(panel.dates, purged, strict=True) if k),
        )
        folds.append(fold)

        train_idx = np.flatnonzero(keep)
        test_idx = np.flatnonzero(is_test)
        if train_idx.size == 0 or test_idx.size == 0:
            return CPCVResult(
                "unmeasurable",
                (
                    f"combination {combo} leaves {train_idx.size} training and "
                    f"{test_idx.size} test day(s) after a "
                    f"{label_horizon_trading_days}d purge and a "
                    f"{cpcv.embargo_trading_days}d embargo"
                ),
                (),
                tuple(folds),
                n_paths,
            )

        try:
            ic = _fold_ic(panel, recipe, train_idx, test_idx)
        except (TrainingIntegrityError, np.linalg.LinAlgError) as exc:
            return CPCVResult(
                "unmeasurable",
                f"combination {combo} could not be fitted: {exc}",
                (),
                tuple(folds),
                n_paths,
            )
        ics.append(ic)

    return CPCVResult("ok", "", tuple(ics), tuple(folds), n_paths)


def _flatten(day_idx: np.ndarray, n_names: int) -> np.ndarray:
    return np.concatenate([d * n_names + np.arange(n_names) for d in day_idx])


def _fold_ic(
    panel: FeaturePanel, recipe: ModelRecipe, train_idx: np.ndarray, test_idx: np.ndarray
) -> float:
    n_names = len(panel.names)
    train_rows = _flatten(train_idx, n_names)
    test_rows = _flatten(test_idx, n_names)
    matrix = _design(recipe, panel, train_rows)
    labels = panel.forward_returns.reshape(-1)[train_rows]
    _assert_trainable(recipe, matrix, labels)
    coefficients, intercept = _fit_linear(recipe.estimator, matrix, labels)
    predicted = _design(recipe, panel, test_rows) @ coefficients + intercept
    actual = panel.forward_returns.reshape(-1)[test_rows]
    return _rank_ic(predicted, actual)


def _rank_ic(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Spearman rank IC. Zero when either side has no dispersion to rank."""
    if predicted.size < 2:
        return 0.0
    pr = _ranks(predicted)
    ar = _ranks(actual)
    pr = pr - pr.mean()
    ar = ar - ar.mean()
    denom = float(np.sqrt((pr * pr).sum() * (ar * ar).sum()))
    if denom == 0.0:
        return 0.0
    return float((pr * ar).sum() / denom)


def _ranks(values: np.ndarray) -> np.ndarray:
    order = values.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(values.size, dtype=float)
    return ranks


# ---------------------------------------------------------------------------
# Grading — the per-date series the arena pairs on.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelGrade:
    """One arm's cycle grade: the series, the CPCV battery, the fit."""

    series: ArmSeries
    cpcv: CPCVResult
    fit: Fit
    benchmark: str = "population"


def grade_arm(recipe: ModelRecipe, panel: FeaturePanel, *, as_of: str) -> ModelGrade:
    """Score ``recipe`` per trading day against the population it scored.

    The per-date score is the cross-sectional rank IC of that day's
    predictions against that day's realized forward return — the scored
    cross-section IS the benchmark, which is why M declares `population`
    rather than SPY. The engine then pairs these series across arms; nothing
    here computes a comparison.
    """
    fit = train_arm(recipe, panel, as_of=as_of)
    scores: dict[str, float] = {}
    n_names = len(panel.names)
    for i, day in enumerate(panel.dates):
        if day > as_of:
            continue
        rows = i * n_names + np.arange(n_names)
        predicted = _predict(fit, panel, rows)
        actual = panel.forward_returns.reshape(-1)[rows]
        scores[day] = _rank_ic(predicted, actual)
    cpcv = cpcv_oos_ic(
        panel,
        recipe=recipe,
        cpcv=recipe.cpcv,
        label_horizon_trading_days=recipe.label_horizon_trading_days,
    )
    return ModelGrade(series=ArmSeries(arm_id=fit.arm_id, scores=scores), cpcv=cpcv, fit=fit)
