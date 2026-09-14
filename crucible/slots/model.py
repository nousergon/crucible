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

**Seven things this module is careful about, each a measured defect:**

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
   out-of-sample rank IC against the population it scored (`ArenaConfig`
   refuses anything else for a selection-kind slot; M declares `population`
   for the same reason).
5. **The series the pointer is decided on is OUT-OF-SAMPLE** (plan §9.1, the
   out-of-sample clock). `grade_arm` used to fit once on every date up to
   `as_of` and then score those same dates with that fit, so the number the
   arena paired on was in-sample and the only OOS figure — `ModelGrade.cpcv`
   — was consumed by nothing on the promotion path. Measured on 40 pure-noise
   features over a pure-noise label: the in-sample series read +0.0728 mean IC
   while CPCV OOS read -0.0214, and a 40-feature arm beat a 3-feature arm
   0.0728 vs 0.0118 on data containing no signal. Overfit was the thing being
   ranked. :func:`grade_arm` now walks forward — every scored day is predicted
   by a fit that never saw that day's label — and the window opens no earlier
   than the recipe's `registered_at`.
6. **`_rank_ic` averages tied ranks.** The lifted implementation
   (`training/leakfree_meta_ic.py`) called `scipy.stats.spearmanr`, which
   assigns mid-ranks to ties; the lift replaced it with an argsort, which
   assigns ties distinct index-order ranks. Measured: two flat vectors scored
   +1.0 (the docstring claimed 0.0), a collapsed model emitting one constant
   `predicted_alpha` scored +1.0 against a flat day, and 45 of 50 names tied
   scored -0.4558 where the mid-rank Spearman is -0.1043. A collapsed model is
   the exact condition the behavioural veto exists to catch; the grader was
   handing it a perfect score. Zero dispersion on either side is now
   UNMEASURABLE — a miss on the series, never a number.
7. **A fit dated `as_of` may not consume a label realized after it.** A
   `label_horizon_trading_days=21` forward return on `as_of` settles on
   `as_of + 21` sessions, so :func:`train_arm` purges the final horizon from
   its training rows. Without it every replayed Saturday in the plan's §6.1
   gate used post-date information, and a panel that leaves the unsettled rows
   NULL (which `FeatureLayerSource.panel` does) failed the finiteness check
   instead of naming the real condition.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from itertools import combinations
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml
from nousergon_lib.arena import ArmSeries, ServingPrecondition, derive_arm_id
from nousergon_lib.arena.engine import TrainingIntegrityError, TrainingStatus

from crucible.features.registry import UNIT_SUFFIXES
from crucible.keys import (
    arena_cycle_key,
    arm_predictions_key,
    cross_section_key,
    features_key,
    features_prefix,
    shadow_key,
    strategy_arms_prefix,
)
from crucible.slots.arms import SupersededArmUndeclaredError, resolve_declared_lineage
from crucible.slots.inputs import (
    BasePredictionsUnavailableError,
    InputRef,
    InputRefusal,
    SlotUnservableError,
    UnresolvedInputError,
    parse_input_ref,
    partition_producible,
    resolve_declared_inputs,
    write_arm_predictions,
)
from crucible.slots.vocab import refuse_unknown_keys

__all__ = [
    "CPCV_OOS_IC_METRIC",
    "DEAD_SLOT_METRIC",
    "DISPERSION_METRICS",
    "FLOOR_VETO_METRICS",
    "HIGH_CONFIDENCE_P_UP",
    "MIN_DISPERSION_RATIO",
    "SERVING_VETO_WINDOW_METRIC",
    "SETTLED_WINDOW_DECISION_DATES",
    "UNPRODUCED_VETO_METRICS",
    "M_SELECTION_TOP_N",
    "OOS_METHOD",
    "SLOT",
    "PROPORTION_METRICS",
    "UNITS_SUFFIXES",
    "ZERO_VETO_METRICS",
    "CPCVResult",
    "CPCVSpec",
    "CompletenessResult",
    "EstimatorSpec",
    "FeatureLayerSource",
    "FeaturePanel",
    "Fit",
    "MetricScaleError",
    "ModelGrade",
    "InputRef",
    "InputRefusal",
    "ModelRecipe",
    "RegisteredModelArm",
    "SettledCrossSections",
    "SupersededArmUndeclaredError",
    "SlotRecipes",
    "SlotUnservableError",
    "UnresolvedInputError",
    "UpProbabilityCalibration",
    "calibrate_up_probability",
    "design_panel",
    "predict_cross_section",
    "score_cross_section",
    "produce_arm_predictions",
    "TrainingWindowSpec",
    "VetoResult",
    "assert_units_suffixes",
    "cpcv_oos_ic",
    "evaluate_behavioural_veto",
    "evaluate_input_completeness",
    "grade",
    "grade_arm",
    "load_model_recipes",
    "produce",
    "realized_hit_rate",
    "registration_specs",
    "serving_metrics",
    "settled_training_days",
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
#: is the scale-DEPENDENT part by construction, and the scale is ASSERTED
#: rather than assumed — see :data:`PROPORTION_METRICS`.
FLOOR_VETO_METRICS: dict[str, float] = {"model_hit_rate_30d": 0.50}

#: The veto inputs NOTHING IN THIS HARNESS PRODUCES, and the producer each
#: one waits on (`alpha-engine-config-I9759`, from `crucible-PR149`'s note).
#:
#: **EMPTY since `alpha-engine-config-I10680`, and that is the whole point of
#: it still existing.** It carried `stdev_p_up`, `n_high_confidence` and
#: `model_hit_rate_30d` from 2026-09-13 until their producers landed: for as
#: long as a veto input has no producer, the veto can only ever read
#: `insufficient`, the serving precondition can only ever FAIL, and the M
#: pointer can never move — not this cycle, but permanently. §5.1's rule is
#: "an uncomputed gate is not a pass"; it is not "a gate that can never
#: compute is a healthy slot", and the difference is a slot that renders
#: identically to one that merely held its pointer this week.
#:
#: The three producers are now :func:`serving_metrics` and its helpers:
#: `model_hit_rate_30d` from the arm's own settled walk-forward cross
#: sections, and `stdev_p_up`/`n_high_confidence` through the per-arm
#: up-probability calibration fitted on that same settled block. None of them
#: is a stand-in — each is read off a real measurement, which is the only
#: form §5.1 permits.
#:
#: Two contract tests hold this to the truth in BOTH directions
#: (`tests/test_slot_model.py::TestTheDeadSlotIsObserved`): every name here
#: must be one a rule family actually reads, and every veto input
#: :func:`serving_metrics` does not emit must be declared here. So a metric
#: added to a rule family with no producer re-populates this map and
#: re-arms :data:`DEAD_SLOT_METRIC` rather than killing the slot in silence.
UNPRODUCED_VETO_METRICS: dict[str, str] = {}

#: The metric name :data:`UNPRODUCED_VETO_METRICS` is reported under.
DEAD_SLOT_METRIC = "serving_veto_has_no_producer"

#: The metric name the cycle reports the veto's per-cycle reading under when
#: every producer exists and the window is simply not long enough yet
#: (`alpha-engine-config-I10680`). Distinct from :data:`DEAD_SLOT_METRIC` by
#: design: "no arm can EVER serve" and "no arm can serve YET" have different
#: owners and different remedies, and collapsing them into one row is how a
#: temporary state gets read as a permanent one and a permanent one gets
#: waited out.
SERVING_VETO_WINDOW_METRIC = "serving_veto_window"

#: Settled out-of-sample DECISION DATES an arm needs before its realized
#: veto inputs exist. The metric's own name declares the window —
#: `model_hit_rate_30d` — and a hit rate computed over eight dates reported
#: under that name is a different statistic wearing it, which is the same
#: class of defect as the unit-scale error :data:`PROPORTION_METRICS` exists
#: against. Not a tuned bar and not an evidence floor in policy §5.0's sense
#: (those are removed, and `promote_min_weeks` is the only age rule): it is
#: the well-formedness condition of the statistic itself. The up-probability
#: calibration is fitted on the SAME block, so one number states the whole
#: minimum.
#:
#: What it costs, in sessions: a date's label settles `label_horizon` sessions
#: after it, and the walk-forward window opens at `registered_at`, so an arm
#: needs `SETTLED_WINDOW_DECISION_DATES + label_horizon_trading_days`
#: sessions of panel beyond its registration — 51 at the default 21-session
#: horizon, a little over ten trading weeks.
SETTLED_WINDOW_DECISION_DATES = 30

#: The calibrated up-probability above which a name counts toward
#: `n_high_confidence`. **The coin flip, and deliberately nothing else.**
#:
#: A tuned confidence bar would be a strategy parameter and would not belong
#: in this repository at all. It is not needed: the zero-veto rule this feeds
#: asks whether the model *names anything it can serve*, and the answer is
#: read off the selection the slot actually hands downstream — the top
#: :data:`M_SELECTION_TOP_N` names — counting those whose calibrated
#: probability of beating the cross-section is better than a coin flip. Zero
#: then means exactly what the 2026-08-21 model did: of the ten names it
#: ranked highest, not one was calibrated to beat the population, so there
#: was nothing in its output to trade. Both inputs (the selection width and
#: the coin flip) are already declared in this module; the bar introduces no
#: new number.
HIGH_CONFIDENCE_P_UP = 0.5

#: Metrics this module declares to be 0–1 proportions, range-checked on BOTH
#: sides before any floor is applied.
#:
#: The comment that used to sit on :data:`FLOOR_VETO_METRICS` claimed "a
#: producer emitting a percentage would break it loudly at the first cycle,
#: which is the correct failure". It did the opposite. Measured: the same
#: model at two scales — `model_hit_rate_30d=0.4` vetoes on "below the
#: absolute floor 0.5", and `model_hit_rate_30d=40.0` PASSES, because 40.0 is
#: comfortably above 0.5. A unit-scale error made the veto fail OPEN, which is
#: the one direction a veto may never fail. An out-of-range value now raises
#: :class:`MetricScaleError`: it is not a veto (the model may be fine) and not
#: a pass (nothing was measured) — it is a producer contract violation, and
#: the fleet default on those is RAISE.
#:
#: Only metrics with a DECLARED unit appear here. `alpha_stdev` and
#: `stdev_p_up` are compared as ratios against the incumbent's value for the
#: same metric, which is unit-free by construction, so a scale error on those
#: cancels rather than misreads and there is nothing to assert.
PROPORTION_METRICS: tuple[str, ...] = ("model_hit_rate_30d",)


class MetricScaleError(ValueError):
    """A metric declared a 0–1 proportion arrived outside [0, 1].

    Raised rather than vetoed or recorded: a producer emitting a percentage
    where the contract says proportion has broken the contract, and every
    downstream comparison against an absolute floor is meaningless in a
    direction that fails OPEN. `champion-challenger-policy.md` §5.1 — you
    cannot gate on a statistic you did not measure — and the fleet default on
    a producer contract violation is to raise, not to degrade.
    """


#: The fleet's units-suffix contract (`AGENTS.md`): `avg_volume_20d` was
#: emitted as a normalized ratio and consumed as raw shares, silently failing
#: 901 of 903 tickers for months. A recipe may not name a bare column.
#:
#: **Re-exported, not re-declared** (`alpha-engine-config-I9772`). This module
#: and `crucible.features.registry` carried two declarations of one contract,
#: one character apart in their names, which is how the duplication survived
#: review. The producer's is canonical — it is enforced at `FeatureSpec`
#: construction — and the consumer's name is kept as an alias so a recipe and
#: the column it names can never be checked against different rules.
UNITS_SUFFIXES: tuple[str, ...] = UNIT_SUFFIXES


@dataclass(frozen=True)
class VetoResult:
    """The veto's verdict, with every reason it reached it.

    ``uncomputable`` and ``inapplicable`` are deliberately two fields, not
    one (`alpha-engine-config-I10680`). A metric that has no producer, or
    whose window is too short, was NOT MEASURED and makes the result
    ``insufficient`` — §5.1. A *relative* rule on a slot with no incumbent
    has no comparand to measure against at all, which is a different fact
    with a different remedy, and folding it into ``uncomputable`` would make
    a cold slot permanently unservable for a reason no producer could ever
    fix. See :func:`evaluate_behavioural_veto`.
    """

    status: str  # veto | pass | insufficient
    reasons: tuple[str, ...] = ()
    uncomputable: tuple[str, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)
    inapplicable: tuple[str, ...] = ()

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
        if self.inapplicable:
            # Recorded on the PASS too, and on the precondition rather than
            # only in `metrics`: "passed every rule" and "passed every rule
            # that had a comparand" are different claims about a first
            # champion, and the artifact must not render them identically.
            note = (
                f"{', '.join(self.inapplicable)} had no incumbent to compare against "
                "(the slot has no champion); the absolute rules were applied in full"
            )
            reason = f"{reason}; {note}" if reason else note
        return ServingPrecondition(name="behavioural_veto", passed=passed, reason=reason)


def evaluate_behavioural_veto(
    candidate: dict[str, Any],
    incumbent: dict[str, Any],
    *,
    min_dispersion_ratio: float = MIN_DISPERSION_RATIO,
    has_incumbent: bool = True,
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

    Every metric in :data:`PROPORTION_METRICS` is range-asserted on BOTH sides
    first and raises :class:`MetricScaleError` when it is outside [0, 1]. A
    percentage-scaled hit rate used to sail past its own absolute floor.

    **``has_incumbent=False`` — the cold-start reading**
    (`alpha-engine-config-I10680`). Every dispersion rule is a ratio against
    what is BEING SERVED. On a slot whose `champions/m/current.json` does not
    exist, nothing is being served, so the rule has no comparand — and this
    is a *measured* property of the M slot, not a convenience: §10.1's null
    control, which `crucible-PR257` made the baseline for a first champion,
    is a SELECTION-shaped harness control (`control_null_m` ranks names on
    noise). It publishes no predicted-alpha cross-section, so it has no
    `alpha_stdev` and no `stdev_p_up` in the candidate's units; scoring one
    off its noise draw would compare a 1e-3 alpha spread against a unit
    normal and veto every real arm on arithmetic.

    So a cold slot records the dispersion family in ``inapplicable`` rather
    than ``uncomputable``, and the verdict rests on the two ABSOLUTE families
    — which are fully measured. This is not §5.1's forbidden move: nothing is
    reported as a pass for a statistic nobody measured, because the statistic
    is not a statistic about this slot until something is serving. It applies
    at most once in a slot's life, and from the second cycle after a first
    champion the ratios are back.

    What the cold slot keeps instead is an ABSOLUTE non-degeneracy check on
    the same two metrics: a dispersion of zero is vetoed outright. That is
    strictly narrower than the ratio rule — it catches a model that collapsed
    to a constant, not one that merely halved — and it is the part of the
    2026-08-28 lesson that survives having nothing to halve against.
    """
    _assert_proportions(candidate, "candidate")
    _assert_proportions(incumbent, "incumbent")

    reasons: list[str] = []
    uncomputable: list[str] = []
    inapplicable: list[str] = []
    metrics: dict[str, Any] = {}

    for name in DISPERSION_METRICS:
        cand = candidate.get(name)
        if cand is None:
            uncomputable.append(name)
            continue
        if not has_incumbent:
            # No comparand exists. The absolute floor of the ratio rule — a
            # collapsed constant — still applies and still vetoes.
            inapplicable.append(name)
            metrics[name] = cand
            if float(cand) <= 0.0:
                reasons.append(
                    f"{name}={cand!r} with no incumbent to compare against: the slot has no "
                    "champion, so the dispersion RATIO has no comparand, but a cross-section "
                    "with zero spread is a collapsed model whichever way it is measured and "
                    "is not the arm a cold slot wins its first champion with"
                )
            continue
        inc = incumbent.get(name)
        if inc is None:
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
        return VetoResult("veto", tuple(reasons), tuple(uncomputable), metrics, tuple(inapplicable))
    if uncomputable:
        return VetoResult("insufficient", (), tuple(uncomputable), metrics, tuple(inapplicable))
    return VetoResult("pass", (), (), metrics, tuple(inapplicable))


def _assert_proportions(metrics: dict[str, Any], side: str) -> None:
    """Raise on any :data:`PROPORTION_METRICS` value outside [0, 1]."""
    for name in PROPORTION_METRICS:
        value = metrics.get(name)
        if value is None:
            # Absence is already handled by the veto itself, which records it
            # in `uncomputable` and returns `insufficient` — an absent metric
            # is never a pass. Nothing is swallowed here.
            continue
        numeric = float(value)
        if not (0.0 <= numeric <= 1.0):
            raise MetricScaleError(
                f"{side} {name}={value!r} is outside [0, 1], and this module declares it a "
                "0–1 proportion. A producer emitting a percentage makes the absolute floor "
                f"{FLOOR_VETO_METRICS.get(name)} meaningless in the failing-OPEN direction: "
                "measured, hit rate 0.4 vetoes and the same model at 40.0 passes. Fix the "
                "producer's units; there is no scale this gate infers."
            )


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
    #: The `spec.inputs` references materialised onto this panel, in the
    #: text form the recipe declared them (`predictions[base]`). Empty on a
    #: panel straight from :meth:`FeatureLayerSource.panel`, which produces
    #: feature columns and nothing else. :func:`_design` reads it to tell a
    #: skipped seam from a missing parquet column — two defects whose
    #: remedies have nothing in common and which used to raise the same
    #: `KeyError` (`alpha-engine-config-I9777`).
    resolved_inputs: tuple[str, ...] = ()

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
            resolved_inputs=self.resolved_inputs,
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
    """The registry interface track A's `crucible/features/` exposes.

    A tuple of :class:`crucible.features.FeatureSpec`, iterated for its specs
    and read for their names. It is deliberately NOT a tuple of strings: a
    registry of names is enough for a units check and nothing else, and the
    lineage each spec carries is what `explain` answers "which data produced
    this column" from (`alpha-engine-config-I9772`).
    """

    def __iter__(self) -> Any: ...


@dataclass(frozen=True)
class FeatureLayerSource:
    """Reads a versioned feature panel through track A's registry.

    Plan §10.4: the M slot resolves its declared columns through the registry
    and records the layer's version and hash as a manifest input, so R and M
    provably read the same artifact. **It never recomputes a feature.** A
    local recomputation is the precise defect the feature layer exists to
    remove: R and M computing from different code, so "the signal degraded"
    cannot be separated from "the feature changed". Every value here comes
    from `features/{version}/{trading_day}.parquet` as track A wrote it.

    **The version is resolved, never named by a consumer**
    (`alpha-engine-config-I9772`). `version=None` resolves
    `crucible.features.DEFAULT_FEATURE_VERSION`, which is derived by hashing
    the catalogue — so an edited recipe writes to a new prefix and cannot
    overwrite the layer an earlier verdict was computed from. A consumer
    hard-coding `"v1"` would defeat that mechanism silently.
    """

    store: Any
    registry: Any = None
    version: str | None = None

    def __post_init__(self) -> None:
        from crucible.features import CATALOG, DEFAULT_FEATURE_VERSION  # noqa: PLC0415

        if self.registry is None:
            object.__setattr__(self, "registry", CATALOG)
        if self.version is None:
            object.__setattr__(self, "version", DEFAULT_FEATURE_VERSION)

    @property
    def columns(self) -> tuple[str, ...]:
        """Every column this layer version produces, from the registry."""
        return tuple(spec.name for spec in self.registry)

    def key(self, trading_day: str) -> str:
        """The documented path shape, single-sourced in `crucible.keys`."""
        return features_key(str(self.version), trading_day)

    def _read(self, trading_day: str, ctx: Any = None) -> Any:
        """One day's cross-section, recorded as a manifest input when asked.

        The recorded entry is byte-for-byte what `crucible.slots.cycle`
        records for the R slot — same key, same sha256, same schema version —
        which is what makes the §10.4 identity an assertion rather than a
        convention.
        """
        from crucible.features import read_features  # noqa: PLC0415
        from crucible.slots.cycle import MissingArtifactError  # noqa: PLC0415

        key = self.key(trading_day)
        if not self.store.exists(key):
            raise MissingArtifactError(
                f"the feature layer is absent at {key}. The M slot reads this artifact "
                "and nothing else (§10.4) — it does NOT recompute features locally, "
                "which would put R and M on different code. Compile it with:\n"
                f"    crucible data.daily --date {trading_day}"
            )
        payload = self.store.get_bytes(key)
        if ctx is not None:
            ctx.record_input(key, payload, schema_version="features.v1")
        return read_features(payload)

    def _sessions(self, trading_day: str, lookback_trading_days: int) -> list[str]:
        """The days to read, listed from the STORE rather than from a calendar.

        A date range derived from the calendar would name days the layer was
        never compiled for and turn a producer gap into a run of nulls. The
        store knows which days exist; a gap therefore raises by name below.
        """
        prefix = features_prefix(self.version)
        available = sorted(
            key[len(prefix) : -len(".parquet")]
            for key in self.store.list_keys(prefix)
            if key.endswith(".parquet")
        )
        earlier = [d for d in available if d <= trading_day]
        if trading_day not in earlier:
            earlier.append(trading_day)
        wanted = earlier[-(lookback_trading_days + 1) :]
        return wanted

    def panel(
        self,
        *,
        trading_day: str,
        columns: tuple[str, ...],
        lookback_trading_days: int = 0,
        label_horizon_trading_days: int | None = None,
        ctx: Any = None,
    ) -> FeaturePanel:
        """The declared ``columns`` for ``trading_day``, read from the layer.

        ``lookback_trading_days`` extends the panel backwards over the days
        the layer has actually been compiled for; ``label_horizon_trading_days``
        fills `forward_returns` from the layer's own ``close_raw`` over that
        many SESSIONS of the panel's own date axis (§4.12), leaving a row
        whose horizon has not settled NULL rather than zero — an unsettled
        return is not a flat one, and training raises on the null instead of
        banking it.
        """
        import numpy as np  # noqa: PLC0415

        requested = tuple(columns)
        if not requested:
            raise ValueError(
                "panel() needs at least one column; a panel of no features would train "
                "an arm on an empty design matrix"
            )
        unknown = sorted(set(requested) - set(self.columns))
        if unknown:
            raise KeyError(
                f"column(s) {unknown} are not produced by feature layer version "
                f"{self.version!r}; the registry produces {sorted(self.columns)}. A "
                "recipe naming a column the layer does not produce fails HERE, at "
                "load, and is never trained on a silently substituted zero "
                "(the 2026-08-28 seven-hard-zeroed-features condition)."
            )

        dates = tuple(self._sessions(trading_day, lookback_trading_days))
        frames = {day: self._read(day, ctx=ctx) for day in dates}

        anchor = frames[trading_day]
        names = tuple(str(t) for t in sorted(anchor["ticker"]))
        needed = sorted(set(requested) | {"close_raw"})

        blocks: dict[str, np.ndarray] = {}
        closes = np.full((len(dates), len(names)), np.nan, dtype="float64")
        for column in needed:
            block = np.full((len(dates), len(names)), np.nan, dtype="float64")
            for row, day in enumerate(dates):
                frame = frames[day].set_index("ticker")
                if column not in frame.columns:
                    raise KeyError(
                        f"feature {column!r} is absent from {self.key(day)}, which the "
                        f"registry for version {self.version!r} says it produces. The "
                        "artifact and the registry disagree; nothing is substituted."
                    )
                series = frame[column].reindex(list(names))
                block[row, :] = series.to_numpy(dtype="float64")
            if column == "close_raw":
                closes = block
            if column in requested:
                blocks[column] = block

        forward = np.full((len(dates), len(names)), np.nan, dtype="float64")
        if label_horizon_trading_days is not None:
            if label_horizon_trading_days < 1:
                raise ValueError(
                    "label_horizon_trading_days is a count of SESSIONS and is at least "
                    "one (plan §4.12)"
                )
            settled = len(dates) - label_horizon_trading_days
            for row in range(max(settled, 0)):
                start = closes[row, :]
                end = closes[row + label_horizon_trading_days, :]
                with np.errstate(divide="ignore", invalid="ignore"):
                    forward[row, :] = np.where(start > 0, end / start - 1.0, np.nan)

        return FeaturePanel(
            dates=dates,
            names=names,
            features=blocks,
            forward_returns=forward,
            feature_version=str(self.version),
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
    registered_at: str
    #: Typed input references (`alpha-engine-config-I9777`). Empty for every
    #: arm whose whole design matrix comes from the feature layer, which is
    #: why it is absent from :attr:`spec` when empty — see that property.
    inputs: tuple[InputRef, ...] = ()
    #: The share of the training block's ticker-rows this arm tolerates losing
    #: to an incomplete feature vector before the fit refuses. `None` means
    #: the arm declares none and is held to
    #: :data:`DEFAULT_MAX_INCOMPLETE_ROW_RATIO`.
    #:
    #: A TUNED number and therefore a private-tree one: how much of a
    #: cross-section an arm may lose and still be the arm that was
    #: pre-registered is a belief about that arm's hypothesis, not a property
    #: of the harness. It is read here, defaulted here, and set in
    #: `alpha-engine-config/strategy/`.
    max_incomplete_row_ratio: float | None = None
    supersedes: str | None = None
    slot: str = "m"
    #: Where these bytes were read from — a checkout path or a store key.
    #: Provenance, exactly as `crucible.slots.arms.ArmSpec.source_key` is, and
    #: deliberately outside :attr:`spec`: which tree a recipe was read from is
    #: not what it computes, and hashing it would give one recipe two ids.
    source_key: str = ""

    def __post_init__(self) -> None:
        # alpha-engine-config-I9821: this guard predates `inputs` (I9777) — a
        # PURE meta-learner (every design column a `predictions[...]` input,
        # `features == ()`) is a real, intended arm shape (the canonical
        # stacking ensemble), and `not self.features` refused it outright.
        # The refusal itself is still correct and still required: an arm
        # with NEITHER features nor inputs has an empty design matrix and
        # cannot be fit. Testing `design_columns` (the union both
        # `assert_units_suffixes` calls below and the duplicate check already
        # use) is the union this guard should have been checking all along.
        if not self.design_columns:
            raise ValueError(f"arm {self.name!r} declares no design columns")
        assert_units_suffixes(self.features)
        assert_units_suffixes(tuple(r.column for r in self.inputs))
        duplicates = sorted({c for c in self.design_columns if self.design_columns.count(c) > 1})
        if duplicates:
            raise ValueError(
                f"arm {self.name!r} declares column(s) {duplicates} more than once across "
                "`features` and `inputs`. A design matrix with a column twice is exactly "
                "collinear, and the normal equations answer it with an arbitrary split of "
                "one coefficient across two."
            )
        if self.label_horizon_trading_days < 1:
            raise ValueError("label_horizon_trading_days must be >= 1 (trading days, §4.12)")
        if self.refit_cadence_trading_days < 1:
            raise ValueError("refit_cadence_trading_days must be >= 1 (trading days, §4.12)")
        if self.max_incomplete_row_ratio is not None and not (
            0.0 <= self.max_incomplete_row_ratio < 1.0
        ):
            raise ValueError(
                f"arm {self.name!r}: max_incomplete_row_ratio="
                f"{self.max_incomplete_row_ratio!r} is outside [0, 1). It is the SHARE of "
                "the training block that may be dropped for an incomplete feature vector; "
                "a value of 1 or above is a ceiling nothing can breach, which is the same "
                "as having none."
            )
        try:
            dt.date.fromisoformat(self.registered_at)
        except ValueError as exc:
            raise ValueError(
                f"arm {self.name!r}: registered_at={self.registered_at!r} is not an ISO date. "
                "It is the arm's out-of-sample clock (plan §9.1) — the grader refuses to "
                "score a date before it — so an unparseable one is not a cosmetic defect."
            ) from exc

    @property
    def design_columns(self) -> tuple[str, ...]:
        """Every column of this arm's design matrix, in fit order.

        `spec.features` first, then one column per `predictions[...]` input.
        One accessor, so the fitter, the trainability check and the panel
        builder cannot disagree about what the arm actually reads.
        """
        return tuple(self.features) + tuple(r.column for r in self.inputs)

    @property
    def spec(self) -> dict[str, Any]:
        """The canonical spec the arm id hashes. Order-independent by key.

        ``registered_at`` is deliberately NOT hashed, exactly as
        `crucible.slots.arms.ArmSpec.spec` omits it: it says when the arm
        started accumulating evidence, not what it computes. Hashing it would
        make re-registering an arm produce a new id and orphan its series.

        **There is no ``feature_version`` key here** (`alpha-engine-config-
        I9801`). The recipe's inputs are already pinned by `features` and
        `inputs`, which ARE hashed; which feature-layer artifact a run
        actually reads is resolved at produce time, not declared by the
        recipe, and is recorded as the run's lineage — `crucible.slots.
        cycle.run_produce` (the production path for every slot, M included)
        calls ``ctx.record_input(features_key(feature_version, day), ...)``,
        so the layer's resolved version is embedded in the recorded key
        (``features/<version>/<day>.parquet``) inside ``inputs[]`` on the
        run's own manifest (``runs/{job}/{day}/run.json``). R and M are
        asserted to record the SAME entry, byte-for-byte, by
        `tests/test_feature_layer_input_identity.py`. (`FeatureLayerSource`
        /`FeaturePanel.feature_version` resolve and carry the same version
        through :func:`design_panel` and :func:`grade_arm` for training and
        grading; :func:`produce_arm_predictions` also threads it into the
        `arm_predictions.v1` artifact, but that function has no production
        caller today — `cycle.run_produce`'s `ctx.record_input` is the
        lineage that is actually live.) Hashing a catalogue version into this
        spec instead would mean an unrelated feature added to the catalogue
        re-ids every M arm and orphans its score series — worse than the
        stale-and-unread field it would replace.
        """
        payload: dict[str, Any] = {
            "features": list(self.features),
            "estimator": self.estimator.to_dict(),
            "label_horizon_trading_days": self.label_horizon_trading_days,
            "refit_cadence_trading_days": self.refit_cadence_trading_days,
            "training_window": self.training_window.to_dict(),
            "cpcv": self.cpcv.to_dict(),
        }
        if self.inputs:
            # Emitted ONLY when declared. A key that always appeared would
            # re-hash every arm already registered and orphan its score
            # series the moment this field shipped (policy §3.1).
            payload["inputs"] = [r.text for r in self.inputs]
        if self.max_incomplete_row_ratio is not None:
            # Same rule, same reason. It IS hashed once declared — how much of
            # a cross-section an arm may lose and still be itself is part of
            # what the arm is, so changing it is a different arm — but an arm
            # that declares none keeps the id it registered under.
            payload["max_incomplete_row_ratio"] = float(self.max_incomplete_row_ratio)
        return payload

    @property
    def resolved_max_incomplete_row_ratio(self) -> float:
        """The ceiling this arm is actually held to — declared or default."""
        if self.max_incomplete_row_ratio is None:
            return DEFAULT_MAX_INCOMPLETE_ROW_RATIO
        return float(self.max_incomplete_row_ratio)

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
)

#: Every key a filed M recipe's `spec:` may declare (`alpha-engine-config-
#: I9944`) — the required fields above, plus `inputs`, the one optional
#: stacked-input declaration `load_model_recipes` also reads
#: (`alpha-engine-config-I9777`). A key outside this set is accepted-and-
#: ignored nowhere: it is a guarantee the loader cannot honour, and the
#: loader refuses it by name.
M_SPEC_KEYS: frozenset[str] = frozenset(
    {*REQUIRED_RECIPE_FIELDS, "inputs", "max_incomplete_row_ratio"}
)

#: Every top-level key a filed M recipe may declare (`alpha-engine-config-
#: I9944`) — named explicitly in the issue rather than derived, because
#: `supersedes_v1` is provenance-only and read by `crucible migrate.history`,
#: not by this loader, and a derivation off this module's own reads would
#: miss it.
M_TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {"slot", "name", "registered_at", "supersedes", "supersedes_v1", "notes", "spec"}
)


#: The metric a refused arm emits, one row per arm, on the manifest of
#: whatever job loaded the slot. Named rather than spelled at the call site so
#: a console adapter and a test read the same literal.
ARM_REFUSED_METRIC = "arm_refused_at_registration"


@dataclass(frozen=True)
class SlotRecipes:
    """What a slot directory REGISTERS and what it REFUSES, side by side.

    `alpha-engine-config-I9955`. Until this type existed the loader returned
    a tuple and threw on the first unbuildable arm, so a directory carrying
    one arm that will not be producible until phase 5 registered nothing at
    all — and Brian's `alpha-engine-config-I9808` ruling (b), whose entire
    justification is that the M slot accumulates evidence BEFORE phase 5,
    accumulated none.

    Returning both halves is what makes the refusal impossible to drop by
    accident: a caller cannot iterate this object as if it were the recipe
    list, so every call site is made to say what it does with :attr:`refused`
    rather than inheriting silence from a tuple that no longer mentions it.
    """

    registered: tuple[ModelRecipe, ...]
    refused: tuple[InputRefusal, ...] = ()

    def __post_init__(self) -> None:
        overlap = {r.name for r in self.registered} & {r.arm for r in self.refused}
        if overlap:
            raise ValueError(
                f"arm(s) {sorted(overlap)} appear as both registered and refused. An arm is "
                "one or the other; a set that says both renders as healthy on whichever "
                "surface reads the registered half first."
            )

    @property
    def unservable(self) -> bool:
        """True iff NOTHING registered while something was refused (plan §5.3).

        Deliberately not `not self.registered`: an EMPTY directory registers
        nothing and refuses nothing, and calling that `unservable` would page
        for a slot nobody has written an arm for yet — an absence condition,
        which `crucible.alerts` already owns.
        """
        return not self.registered and bool(self.refused)

    def refusal_metrics(
        self, *, slot: str = "m", now: dt.datetime | None = None
    ) -> list[dict[str, Any]]:
        """One MetricRecord per refused arm, naming the arm and its input.

        This is deliverable 2 of `alpha-engine-config-I9955` and it is what
        keeps a per-arm refusal from being quieter than the slot-wide
        exception it replaces: the arm and the exact unresolvable input reach
        the manifest — and through it the console and the board — with the
        same precision the exception carried.

        **Reached by no scheduled job yet** (`alpha-engine-config-I9957`): the
        rows are well-formed and schema-validated, and the only caller of
        :func:`load_model_recipes` is `tests/`. See that function's docstring
        for the measurement and the phase-3 blocker.

        ``status`` is `unservable`, the arena's own vocabulary (plan §7:
        "`unmeasurable` and `unservable` are first-class statuses"), forwarded
        verbatim rather than re-encoded into a second word that would drift
        from the first.
        """
        stamp = (now or dt.datetime.now(dt.UTC)).astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        return [
            {
                "name": ARM_REFUSED_METRIC,
                "module": f"crucible.slots.{slot}",
                "metric_type": "count",
                "value": float(len(refusal.unresolvable)),
                "unit": "inputs",
                "n_floor": 1,
                "status": "unservable",
                "status_reason": (
                    f"slot {slot}: arm {refusal.arm!r} is refused at registration and cannot "
                    f"be graded; unresolvable input(s) {list(refusal.unresolvable)}. "
                    f"{refusal.reason}"
                ),
                "source_path": f"strategy/arms/{slot}/{refusal.arm}.yaml",
                "last_updated_utc": stamp,
            }
            for refusal in self.refused
        ]


def _parse_model_recipe(payload: bytes, origin: str) -> ModelRecipe:
    """One recipe document, from wherever its bytes came from.

    Split out of :func:`load_model_recipes` so a checkout and the strategy
    tree synced into the store are read by ONE parser: two parsers is how a
    recipe that loads on a laptop refuses on the box, and the box is where
    the scheduled M cycle runs.
    """
    document = yaml.safe_load(payload.decode("utf-8")) or {}
    spec = document.get("spec") or {}
    missing = [f for f in REQUIRED_RECIPE_FIELDS if f not in spec]
    if "registered_at" not in document:
        # Top-level, beside `name` and `slot` and outside the hashed
        # `spec`, mirroring `crucible.slots.arms`. One shape for one fact.
        missing = [*missing, "registered_at"]
    if missing:
        raise ValueError(
            f"{origin}: recipe is missing pre-registration field(s) {missing}. Plan §9.1: "
            "an arm declares its slot, recipe and lineage BEFORE its first score, and "
            "missing fields mean the arm does not register — a half-declared arm's "
            "verdicts cannot be interpreted later."
        )
    refuse_unknown_keys(
        path=origin,
        keys=set(document),
        vocabulary=M_TOP_LEVEL_KEYS,
        level="top-level recipe",
        slot_label="M",
    )
    refuse_unknown_keys(
        path=origin,
        keys=set(spec),
        vocabulary=M_SPEC_KEYS,
        level="spec",
        slot_label="M",
    )
    return ModelRecipe(
        slot=document.get("slot", "m"),
        name=document["name"],
        features=tuple(spec["features"]),
        estimator=EstimatorSpec(
            kind=spec["estimator"]["kind"],
            params={k: v for k, v in spec["estimator"].items() if k != "kind"},
        ),
        label_horizon_trading_days=int(spec["label_horizon_trading_days"]),
        refit_cadence_trading_days=int(spec["refit_cadence_trading_days"]),
        training_window=TrainingWindowSpec(**spec["training_window"]),
        cpcv=CPCVSpec(**spec["cpcv"]),
        registered_at=str(document["registered_at"]),
        inputs=tuple(parse_input_ref(t) for t in (spec.get("inputs") or ())),
        max_incomplete_row_ratio=(
            None
            if spec.get("max_incomplete_row_ratio") is None
            else float(spec["max_incomplete_row_ratio"])
        ),
        supersedes=document.get("supersedes"),
        source_key=origin,
    )


def load_model_recipes(
    directory: Path | str | None = None,
    *,
    store: Any = None,
    feature_columns: tuple[str, ...] | None = None,
) -> SlotRecipes:
    """Load every `*.yaml` M recipe under ``directory``, sorted by name.

    **Either a checkout or the store, exactly as `load_arm_specs` resolves
    U and R** (`alpha-engine-config-I9957`, blocker 3 of the 2026-09-04
    re-scope). Until this parameter existed the loader accepted a filesystem
    `Path` and nothing else, so the M cycle job — which runs on a spot box
    with no `alpha-engine-config` checkout — could not have read its own
    recipes from where they actually are. ``store`` reads
    `strategy/current/arms/m/*.yaml` (`crucible.keys.strategy_arms_prefix`),
    the same synced tree `load_arm_specs` reads for U and R.

    The directory is `alpha-engine-config/strategy/arms/m/` in production —
    recipes are strategy content and live in the private repository
    (`repository-tiering-policy` test 2). This repository holds the shape.

    **Loading is registration, and registration validates PRODUCIBILITY**
    (`alpha-engine-config-I9777`). Until this check existed the loader
    validated shape alone, so `sota_directional_combine` — which declared two
    columns that are another model's output and have no feature-layer
    producer — registered cleanly and then failed deep inside
    `FeatureLayerSource.panel()` at grading time. An arm nobody can grade is
    indistinguishable, on every surface, from an arm nobody has graded yet;
    that is the failure mode this refusal makes structurally impossible.

    **The refusal is PER ARM** (`alpha-engine-config-I9955`). It used to end
    the load, so one arm declaring an input nothing can produce refused the
    whole directory — three arms in `alpha-engine-config/strategy/arms/m/`,
    one unbuildable until phase 5, and none of the three loaded. The refusal
    itself is unchanged and is not softened: a refused arm does not register,
    it is returned as an :class:`~crucible.slots.inputs.InputRefusal` naming
    the exact unresolvable input, and :meth:`SlotRecipes.refusal_metrics`
    renders it as a MetricRecord for the loading job's manifest. What changed
    is the blast radius.

    **This function has NO production caller** (`alpha-engine-config-I9957`,
    measured 2026-09-04 on `main` 32933bf; the whole of this module's
    ``__all__`` is in the same position). `crucible.track_a._slot_module`
    refuses slots `m` and `s` by design until phase 3, so `experiment.run
    --slot m` and `experiment.grade --slot m` — both declared weekly-arc
    stages — exit non-zero, and the refusal metrics above therefore reach no
    manifest a scheduled run ever writes. `tests/test_slot_inputs_wiring.py::
    TestTheModelPathIsNotDeclaredAndLeftUnwired` pins that as a measured gap
    and goes red the day it closes. Read every "…on the manifest" sentence in
    this module against that fact until the M cycle job exists.

    When NOTHING registers the slot is `unservable` and
    :class:`~crucible.slots.inputs.SlotUnservableError` is raised — which,
    once a job does load the slot, reaches `crucible.runner.run_job`'s
    `try/finally`, writes a `status: failed` manifest, and pages through the
    existing failure condition. A slot with one good arm and one refused arm
    SERVES and reports the refusal, which is the honest reading and the one that lets
    Brian's `alpha-engine-config-I9808` ruling (b) accumulate the evidence it
    was chosen for.

    A dependency CYCLE still refuses the whole slot: a cycle is a property of
    the graph, not of one member, and so is two files sharing a name.

    ``feature_columns`` defaults to the live catalogue, exactly as
    :class:`FeatureLayerSource` resolves its version rather than taking one
    from a consumer. It is a parameter only so a test can state a small
    catalogue explicitly — never so a caller can opt out, which is why there
    is no value of it that disables the check.
    """
    if (directory is None) == (store is None):
        raise ValueError(
            "load_model_recipes reads EITHER a checkout (`directory`) or the strategy "
            "tree synced into the store (`store`), and needs exactly one of them. "
            "Neither is a caller that resolved no source and would register nothing; "
            "both is two trees that can disagree about what the slot declares."
        )
    recipes: list[ModelRecipe] = []
    if directory is not None:
        for path in sorted(Path(directory).glob("*.yaml")):
            recipes.append(_parse_model_recipe(path.read_bytes(), str(path)))
    else:
        prefix = strategy_arms_prefix("m")
        for key in sorted(store.list_keys(prefix)):
            if key.endswith(".yaml"):
                recipes.append(_parse_model_recipe(store.get_bytes(key), key))

    if feature_columns is None:
        from crucible.features import CATALOG  # noqa: PLC0415

        feature_columns = tuple(f.name for f in CATALOG)
    registered, refused = partition_producible(recipes, feature_columns=feature_columns)
    result = SlotRecipes(registered=tuple(registered), refused=refused)
    if result.unservable:
        raise SlotUnservableError(refused)
    return result


# ---------------------------------------------------------------------------
# Fitting.
# ---------------------------------------------------------------------------

#: A design column whose standard deviation is below this is degenerate. It is
#: an absolute floor on a z-scored/ratio feature, deliberately: the 2026-08-28
#: condition was columns hard-ZEROED, and a relative test against the column's
#: own scale cannot see a column that has no scale.
DEGENERATE_STD = 1e-9


def design_panel(
    recipe: ModelRecipe,
    *,
    source: FeatureLayerSource,
    trading_day: str,
    recipes: Sequence[ModelRecipe] = (),
    store: Any = None,
    lookback_trading_days: int = 0,
    ctx: Any = None,
) -> FeaturePanel:
    """The ONE way to obtain a panel an M arm can be fitted on.

    `alpha-engine-config-I9777` was filed because a stacked arm registered
    and could never be graded. The first fix moved the refusal from
    `FeatureLayerSource.panel()` to `FeaturePanel.column()` and declared a
    producer and a stacker that no production code called — the same defect,
    one frame later. This function is the seam that was missing: it reads the
    feature layer for the arm's `spec.features` and then materialises every
    `spec.inputs` reference through
    :func:`crucible.slots.inputs.resolve_declared_inputs`, so a panel handed
    to :func:`train_arm` or :func:`grade_arm` carries the arm's WHOLE design
    matrix or the call already failed.

    **Base ids are resolved here, from the loaded recipe set.** The caller
    never supplies them, so the mapping cannot name an arm other than the one
    the recipe declared; :func:`crucible.slots.inputs.stack_prediction_columns`
    asserts the same property again on the value it receives, because a check
    that only holds when one caller behaves is not a property of the code.

    ``recipes`` is the slot's loaded set — what
    :func:`load_model_recipes` returned. It is required only when the arm
    declares prediction inputs; an arm whose design matrix is entirely the
    feature layer needs no graph and passes none.
    """
    columns = tuple(recipe.features)
    panel = source.panel(
        trading_day=trading_day,
        columns=columns,
        lookback_trading_days=lookback_trading_days,
        label_horizon_trading_days=recipe.label_horizon_trading_days,
        ctx=ctx,
    )
    if not recipe.inputs:
        return panel
    if store is None:
        store = source.store
    return resolve_declared_inputs(
        panel,
        store=store,
        recipe=recipe,
        base_arm_ids=_base_arm_ids(recipe, recipes),
        ctx=ctx,
    )


def _base_arm_ids(recipe: ModelRecipe, recipes: Sequence[ModelRecipe]) -> dict[str, str]:
    """`{base name: base arm id}` for this recipe, from the loaded slot set.

    Registration already refused an input naming an arm the slot does not
    declare (`assert_inputs_producible`), so reaching this raise means the
    panel is being built from a different recipe set than the one that
    registered — which is a caller defect and not a data condition.
    """
    by_name = {r.name: r for r in recipes}
    resolved: dict[str, str] = {}
    for ref in recipe.inputs:
        if ref.kind != "predictions":
            continue
        base = by_name.get(ref.ref)
        if base is None:
            raise UnresolvedInputError(
                f"arm {recipe.name!r} declares input {ref.text!r}, but the recipe set "
                f"passed to `design_panel` declares {sorted(by_name)}. Pass the set "
                "`load_model_recipes` returned: the base arm's ID is derived from its "
                "recipe and is never supplied by a caller, so an absent recipe means the "
                "id cannot be derived rather than that it is unknown."
            )
        resolved[ref.ref] = base.arm_id
    return resolved


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
    #: Which ticker-rows of the training block the design SELECTED, and what
    #: it dropped. Carried on the fit rather than recorded inside the fitter
    #: so the record reaches the manifest through the job that owns the
    #: manifest — `_fit_rows` is also called by the grader and by `_fold_ic`,
    #: neither of which holds a `ctx`, and a recorder threaded into a pure
    #: function would have to be optional and would then be absent exactly
    #: where it mattered.
    completeness: FeatureCompleteness | None = None


def _design(recipe: ModelRecipe, panel: FeaturePanel, rows: np.ndarray) -> np.ndarray:
    _assert_inputs_resolved(recipe, panel)
    return np.column_stack([panel.column(f).reshape(-1)[rows] for f in recipe.design_columns])


def _assert_inputs_resolved(recipe: ModelRecipe, panel: FeaturePanel) -> None:
    """Refuse a panel that never had this arm's declared inputs materialised.

    The `alpha-engine-config-I9777` defect in its second form. A stacked arm
    registers, and then `panel.column()` raises `KeyError: feature
    'predicted_alpha_base_raw' is not in this panel` — a message about a
    missing parquet column, for a column the parquet layer was never asked
    to produce and never could. The remedy is not "compile the feature
    layer"; it is "build the panel through :func:`design_panel`", and a
    refusal that does not say so sends an operator to the wrong file.

    Reads the panel's own provenance rather than re-deriving anything, so it
    is total over :data:`crucible.slots.inputs.INPUT_KINDS` — a fourth kind
    needs no line here.
    """
    if not recipe.inputs:
        return
    resolved = set(panel.resolved_inputs)
    unresolved = [r for r in recipe.inputs if r.text not in resolved]
    if not unresolved:
        return
    # Historical citation, not a phase pointer: alpha-engine-config-I9777 is where
    # this wiring gap was first found. Kept in this comment rather than the raised
    # message per alpha-engine-config-I9839.
    raise UnresolvedInputError(
        f"arm {recipe.name!r} declares input(s) {[r.text for r in unresolved]}, which "
        f"contribute design column(s) {[r.column for r in unresolved]}; this panel "
        f"resolved {sorted(resolved)} and carries columns {sorted(panel.features)}. A "
        "panel for an arm with declared inputs is built by "
        "`crucible.slots.model.design_panel`, which resolves every input through "
        "`crucible.slots.inputs.INPUT_RESOLVERS`; a panel straight from "
        "`FeatureLayerSource.panel()` carries feature columns and nothing else. This "
        "refusal replaces the `KeyError` about a missing parquet column that made the "
        "wiring gap read as a feature-layer gap."
    )


#: The share of a training block's ticker-rows that may be dropped for an
#: incomplete feature vector before the fit is refused outright.
#:
#: A GENERIC order-of-magnitude guard, not a tuned value: an arm that has to
#: discard more than one row in ten is no longer being fitted on the
#: cross-section it was pre-registered against, and the difference between
#: "a handful of new listings lack a 252-session window" and "the feature
#: layer stopped producing a column" is exactly the difference this number
#: has to keep loud. The tuned per-arm value belongs in the arm's own recipe
#: (`spec.max_incomplete_row_ratio`), in the private strategy tree; this is
#: what an arm that declares none is held to.
#:
#: Root cause it exists (`alpha-engine-config-I10688`): with no ceiling at
#: all, ANY row-selection rule silently accepts a total outage. Measured that
#: day, `residual_momentum_252d_skip21d_zscore` was null for 903 of 903
#: tickers on every compiled session; selecting complete rows without a
#: ceiling would have fitted the arm on nothing and returned `ok`.
DEFAULT_MAX_INCOMPLETE_ROW_RATIO = 0.10

#: How many excluded ticker names a completeness record names outright. The
#: full count is always carried; the sample is for a reader who wants somewhere
#: to start, and is bounded because a manifest is loaded whole by every
#: consumer of it.
_SAMPLE_SIZE = 20

#: The metric name the completeness record is filed under on the run manifest.
#: Named rather than spelled at the call site so a console adapter and a test
#: read the same literal.
FEATURE_COMPLETENESS_METRIC = "feature_completeness_excluded_ratio"


@dataclass(frozen=True)
class FeatureCompleteness:
    """Which ticker-rows of a training block carry a COMPLETE feature vector.

    The producer declares per-row completeness and the training design selects
    on it; this record is the second half of that contract — what was dropped,
    which column was missing, and which names it cost — written onto the run
    manifest so an excluded row is a recorded exclusion rather than an
    invisible one.

    The distinction this type exists to keep is between two conditions that
    produce byte-identical nulls:

    * a ticker listed inside the arm's lookback (an IPO, a symbol change) and
      genuinely has no 252-session window yet — correct, expected, and the
      feature layer's declared behaviour ("the layer never fills");
    * a column the producer stopped computing — a vendor outage, a panel
      shallower than the catalogue needs — which must FAIL the cycle.

    Row selection alone cannot tell them apart, which is why
    :attr:`excluded_ratio` is checked against a declared ceiling and the fit
    refuses above it. Dropping rows without a ceiling is the suppression
    collection `crucible/AGENTS.md` rule 4 forbids, wearing a statistician's
    hat.
    """

    arm_name: str
    rows_total: int
    rows_complete: int
    excluded_ratio: float
    ceiling: float
    nan_rows_by_column: dict[str, int]
    excluded_names: tuple[str, ...]
    excluded_dates: tuple[str, ...]

    @property
    def rows_excluded(self) -> int:
        return self.rows_total - self.rows_complete

    @property
    def breached(self) -> bool:
        return self.excluded_ratio > self.ceiling

    def to_dict(self) -> dict[str, Any]:
        """The manifest form: every COUNT in full, the identifier lists BOUNDED.

        A 504-session training block over a ~900-name universe can exclude at
        least one row of nearly every name on nearly every session, and the
        unbounded lists put ~900 tickers and ~400 dates into one metric row —
        measured at 200KB on the first real replay, on a document every
        console reader and every `explain` walk loads whole. The counts are
        what a reader acts on; the samples are what they open a parquet with.
        """
        return {
            "arm_name": self.arm_name,
            "rows_total": self.rows_total,
            "rows_complete": self.rows_complete,
            "rows_excluded": self.rows_excluded,
            "excluded_ratio": self.excluded_ratio,
            "ceiling": self.ceiling,
            "nan_rows_by_column": dict(self.nan_rows_by_column),
            "excluded_name_count": len(self.excluded_names),
            "excluded_names_sample": list(self.excluded_names[:_SAMPLE_SIZE]),
            "excluded_session_count": len(self.excluded_dates),
            "excluded_first_session": self.excluded_dates[0] if self.excluded_dates else None,
            "excluded_last_session": self.excluded_dates[-1] if self.excluded_dates else None,
        }

    @property
    def detail(self) -> str:
        """The one-line reason a rejection or a refusal carries.

        Bounded at the schema's 200-character rejection cap by construction —
        the column list is truncated, never the counts — because
        `RunContext.record_rejected` RAISES on an over-long reason and costs
        the run every other field it had recorded
        (`alpha-engine-config-I10484`).
        """
        worst = sorted(self.nan_rows_by_column.items(), key=lambda kv: (-kv[1], kv[0]))
        named = ", ".join(f"{name}={count}" for name, count in worst[:3] if count)
        return (
            f"incomplete feature vector: {self.rows_excluded}/{self.rows_total} rows "
            f"({self.excluded_ratio:.4f}); {named or 'no column null'}"
        )[:200]

    def as_metric(
        self, *, slot: str, phase: str = "training", now: dt.datetime | None = None
    ) -> dict[str, Any]:
        """The manifest row. `feature_completeness` rides as an extra field.

        ``phase`` separates the TRAINING block's reading from the SERVING
        session's: the same arm files both on one manifest, they answer
        different questions (what the fit was computed over, and which names
        the cross-section could rank), and one row overwriting the other in a
        reader's mind is how a clean fit on a session nobody could be scored
        for reads as healthy.

        `run_manifest.v2.json`'s `MetricRecordRow` is `additionalProperties:
        true` on purpose, so the whole record reaches the manifest beside the
        number rather than being flattened into prose nobody can query.
        """
        stamp = (now or dt.datetime.now(dt.UTC)).astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        return {
            "name": FEATURE_COMPLETENESS_METRIC,
            "module": f"crucible.slots.{slot}",
            "metric_type": "coverage",
            "value": float(self.excluded_ratio),
            "unit": "ratio",
            "n_floor": 1,
            "status": "BREACH" if self.breached else "OK",
            "phase": phase,
            "status_reason": (
                f"arm {self.arm_name!r} ({phase}): {self.rows_excluded} of "
                f"{self.rows_total} ticker-row(s) dropped for an incomplete feature vector "
                f"({self.excluded_ratio:.4f} against a ceiling of {self.ceiling}); "
                f"{len(self.excluded_names)} name(s) affected"
            ),
            "source_path": f"strategy/arms/{slot}/{self.arm_name}.yaml",
            "last_updated_utc": stamp,
            "baseline": float(self.ceiling),
            "feature_completeness": {**self.to_dict(), "phase": phase},
        }


def feature_completeness(
    recipe: ModelRecipe,
    panel: FeaturePanel,
    rows: np.ndarray,
    matrix: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, FeatureCompleteness]:
    """The complete-row mask over a training block, and the record of it.

    A row is COMPLETE when every design column AND the forward label on it is
    finite. Nothing is imputed and nothing is filled: an incomplete row is
    removed from the fit and counted, which is the only treatment that leaves
    the surviving rows measurements rather than a mixture of measurements and
    inventions.
    """
    complete = np.isfinite(matrix).all(axis=1) & np.isfinite(labels)
    n_names = len(panel.names)
    nan_rows_by_column: dict[str, int] = {}
    for index, column in enumerate(recipe.design_columns):
        nan_rows_by_column[column] = int((~np.isfinite(matrix[:, index])).sum())
    nan_rows_by_column["forward_return"] = int((~np.isfinite(labels)).sum())

    dropped = rows[~complete]
    excluded_names = tuple(sorted({panel.names[int(r) % n_names] for r in dropped}))
    excluded_dates = tuple(sorted({panel.dates[int(r) // n_names] for r in dropped}))
    total = int(matrix.shape[0])
    kept = int(complete.sum())
    record = FeatureCompleteness(
        arm_name=recipe.name,
        rows_total=total,
        rows_complete=kept,
        excluded_ratio=(total - kept) / total if total else 1.0,
        ceiling=recipe.resolved_max_incomplete_row_ratio,
        nan_rows_by_column=nan_rows_by_column,
        excluded_names=excluded_names,
        excluded_dates=excluded_dates,
    )
    return complete, record


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
    dead = [recipe.design_columns[i] for i, s in enumerate(stds) if float(s) < DEGENERATE_STD]
    if dead:
        raise TrainingIntegrityError(
            f"arm {recipe.name}: feature(s) {dead} are constant across the training "
            "block — the 2026-08-28 seven-hard-zeroed-features condition. Every "
            "surface said healthy that week; this raises instead, which fails the "
            "whole slot (champion-challenger-policy.md §3)."
        )


def settled_training_days(panel: FeaturePanel, *, as_of: str, label_horizon: int) -> list[int]:
    """Panel row indices whose forward label is REALIZED on or before ``as_of``.

    The purge that was missing (plan §4.12; the §6.1 replay gate). A
    ``label_horizon`` forward return observed on day ``t`` is not known until
    ``t + label_horizon`` SESSIONS have passed, so a fit dated ``as_of`` that
    trains on every date ``<= as_of`` consumes returns from after ``as_of``.
    Every replayed Saturday would then use post-date information, and the
    replay would reproduce a verdict that could not have been reached on the
    day — which is the one property a replay gate exists to establish.

    Measured before the fix: a 160-day panel with a 21-day horizon fitted on
    all 160 dates. It now fits on 139, and the 21 dates whose labels have not
    settled are excluded rather than banked.

    Sessions are counted on the PANEL's own date axis, never on a calendar:
    the axis is exactly the sessions the feature layer was compiled for, and
    a calendar-derived offset would name days the panel does not carry.
    """
    if label_horizon < 1:
        raise ValueError("label_horizon is a count of SESSIONS and is at least one (§4.12)")
    dates = panel.dates
    settled: list[int] = []
    for i, day in enumerate(dates):
        if day > as_of:
            continue
        realized = i + label_horizon
        if realized < len(dates) and dates[realized] <= as_of:
            settled.append(i)
    return settled


def _fit_rows(
    recipe: ModelRecipe, panel: FeaturePanel, day_indices: list[int]
) -> tuple[np.ndarray, float, int, FeatureCompleteness]:
    """Fit the recipe on the named panel days. Raises on an unsound fit.

    One implementation, three callers — :func:`train_arm`, :func:`_fold_ic`
    and the walk-forward grader — because a grading path that fits by
    slightly different code from the serving path is how "the model changed"
    and "the measurement changed" become indistinguishable.
    """
    n_names = len(panel.names)
    rows = np.array([d * n_names + n for d in day_indices for n in range(n_names)], dtype=int)
    matrix = _design(recipe, panel, rows)
    labels = panel.forward_returns.reshape(-1)[rows]

    # SELECT, then assert. The order is the whole design: `_assert_trainable`
    # is unchanged and still refuses any non-finite cell (Brian ruling
    # 2026-08-29) — it is now asked about a block the design VOUCHED for,
    # rather than about every row the panel happened to carry. What the
    # selection may not do is hide an outage, so the ratio it drops is
    # checked against the arm's declared ceiling FIRST and the fit refuses
    # above it.
    complete, record = feature_completeness(recipe, panel, rows, matrix, labels)
    if record.breached:
        worst = sorted(record.nan_rows_by_column.items(), key=lambda kv: (-kv[1], kv[0]))
        raise TrainingIntegrityError(
            f"arm {recipe.name}: {record.rows_excluded} of {record.rows_total} "
            f"ticker-row(s) carry an incomplete feature vector "
            f"({record.excluded_ratio:.4f}), above the declared ceiling "
            f"{record.ceiling}. Null rows by column: "
            f"{[f'{n}={c}' for n, c in worst if c]}; {len(record.excluded_names)} "
            f"name(s) and {len(record.excluded_dates)} session(s) affected. A handful "
            "of new listings without a full lookback is the layer working as declared; "
            "a share this large is a producer gap, and fitting on what survived would "
            "grade the arm on a cross-section nobody pre-registered it against."
        )
    matrix = matrix[complete]
    labels = labels[complete]

    _assert_trainable(recipe, matrix, labels)
    coefficients, intercept = _fit_linear(recipe.estimator, matrix, labels)
    return coefficients, intercept, int(matrix.shape[0]), record


def train_arm(recipe: ModelRecipe, panel: FeaturePanel, *, as_of: str) -> Fit:
    """Fit ``recipe`` on ``panel`` up to ``as_of``. Raises on an unsound fit.

    The refit is the arm executing the cadence its own recipe declares: the
    id and the score series are unchanged across it (policy §3.1), which is
    why `fitted_at` is on the :class:`Fit` and not in the arm id.

    The training rows are the SETTLED ones — see
    :func:`settled_training_days`. `min_trading_days` is measured against the
    settled count, not the raw one: a window rule satisfied only by rows whose
    labels do not exist yet is not satisfied.
    """
    usable = settled_training_days(
        panel, as_of=as_of, label_horizon=recipe.label_horizon_trading_days
    )
    if len(usable) < recipe.training_window.min_trading_days:
        raise TrainingIntegrityError(
            f"arm {recipe.name}: {len(usable)} trading day(s) with a SETTLED "
            f"{recipe.label_horizon_trading_days}-session label up to {as_of}, below the "
            f"recipe's declared min_trading_days="
            f"{recipe.training_window.min_trading_days}. The window rule is part of the "
            "recipe, so a fit on a shorter one is a different arm's fit — and a fit that "
            f"reached the count by including dates whose labels settle after {as_of} would "
            "be trained on post-date information."
        )
    if recipe.training_window.kind == "rolling":
        usable = usable[-recipe.training_window.min_trading_days :]

    coefficients, intercept, n_rows, completeness = _fit_rows(recipe, panel, usable)
    return Fit(
        arm_id=recipe.arm_id,
        recipe=recipe,
        coefficients=coefficients,
        intercept=intercept,
        fitted_at=as_of,
        n_rows=n_rows,
        training_status=TrainingStatus(arm_id=recipe.arm_id, ok=True),
        completeness=completeness,
    )


def score_cross_section(
    fit: Fit, panel: FeaturePanel, *, trading_day: str
) -> tuple[dict[str, float], FeatureCompleteness]:
    """``fit``'s predicted alpha per name for ONE session of ``panel``.

    The same `_design` the fit was trained through, so a stacked arm's
    serving row is assembled by the code that assembled its training rows —
    a serving path that built its design matrix separately is how "the model
    changed" and "the input changed" stop being distinguishable.
    """
    try:
        row = panel.dates.index(trading_day)
    except ValueError as exc:
        raise KeyError(
            f"{trading_day} is not on this panel's session axis "
            f"({panel.dates[0]}..{panel.dates[-1]}, {len(panel.dates)} session(s)). A "
            "prediction for a session the panel does not carry would be computed from "
            "some other session's features."
        ) from exc
    rows = row * len(panel.names) + np.arange(len(panel.names))
    matrix = _design(fit.recipe, panel, rows)

    # The SERVING half of the completeness contract. A name whose feature
    # vector is incomplete on this session is a name this arm cannot score,
    # and there are exactly two honest things to do with it: leave it out of
    # the cross-section, or refuse the session. Writing a NaN into the
    # predictions artifact is neither — it is a hole a stacked arm would train
    # on — which is what the refusal below used to be the only defence
    # against, at the cost of one new listing failing the whole slot.
    #
    # The SAME ceiling the training block is held to, deliberately: if this
    # many names cannot be scored, the layer is broken and a ranking over what
    # survived is a ranking of a different universe.
    scorable = np.isfinite(matrix).all(axis=1)
    labels = np.full(matrix.shape[0], 0.0)
    _, record = feature_completeness(fit.recipe, panel, rows, matrix, labels)
    if record.breached:
        raise TrainingIntegrityError(
            f"arm {fit.recipe.name}: {record.rows_excluded} of {record.rows_total} "
            f"name(s) carry an incomplete feature vector on {trading_day} "
            f"({record.excluded_ratio:.4f}), above the declared ceiling "
            f"{record.ceiling}; first five {list(record.excluded_names[:5])}. Ranking "
            "the remainder would rank a different universe from the one this arm is "
            "graded against."
        )

    values = matrix @ fit.coefficients + fit.intercept
    unscored = [
        n
        for n, v, ok in zip(panel.names, values, scorable, strict=True)
        if ok and not np.isfinite(float(v))
    ]
    if unscored:
        raise TrainingIntegrityError(
            f"arm {fit.recipe.name}: {len(unscored)} name(s) produced a non-finite "
            f"prediction on {trading_day} from a COMPLETE feature vector, first five "
            f"{unscored[:5]}. A finite design matrix that fits to a NaN is a defect in "
            "the fit, not a gap in the inputs, and it fails here rather than writing a "
            "hole a downstream stacked arm would train on."
        )
    predicted = {n: float(v) for n, v, ok in zip(panel.names, values, scorable, strict=True) if ok}
    return predicted, record


def predict_cross_section(fit: Fit, panel: FeaturePanel, *, trading_day: str) -> dict[str, float]:
    """``fit``'s predicted alpha per SCORABLE name for one session.

    The scores alone, for every caller that has no manifest to record the
    completeness half onto. :func:`score_cross_section` is the same
    computation and returns both; there is no second derivation.
    """
    predicted, _ = score_cross_section(fit, panel, trading_day=trading_day)
    return predicted


def produce_arm_predictions(ctx: Any, *, fit: Fit, panel: FeaturePanel, trading_day: str) -> str:
    """Write ONE arm's cross-section for ONE session. The M produce path.

    The producer half of the `arm_predictions.v1` contract, called from the
    module that owns fitting rather than from the module that defines the
    schema — so the artifact a stacked arm consumes is written by the same
    code that trains the arm which consumes it, and neither half of the
    contract is reachable only from a test.
    """
    return write_arm_predictions(
        ctx,
        arm_id=fit.arm_id,
        trading_day=trading_day,
        feature_version=panel.feature_version,
        predicted_alpha=predict_cross_section(fit, panel, trading_day=trading_day),
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
        if ic is None:
            return CPCVResult(
                "unmeasurable",
                (
                    f"combination {combo} produced no dispersion on one side of the rank "
                    "correlation — a collapsed prediction or a flat label block cannot be "
                    "ranked, and an unrankable fold is not a zero IC"
                ),
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
) -> float | None:
    """One fold's out-of-sample rank IC, or ``None`` when it cannot be ranked."""
    n_names = len(panel.names)
    coefficients, intercept, _, _ = _fit_rows(recipe, panel, [int(i) for i in train_idx])
    test_rows = _flatten(test_idx, n_names)
    predicted = _design(recipe, panel, test_rows) @ coefficients + intercept
    actual = panel.forward_returns.reshape(-1)[test_rows]
    return _rank_ic(predicted, actual)


def _rank_ic(predicted: np.ndarray, actual: np.ndarray) -> float | None:
    """Spearman rank IC over AVERAGE ranks, or ``None`` when unmeasurable.

    ``None`` is an explicit verdict, not a swallowed failure: it means one
    side of the comparison has no dispersion, so there is no ordering to
    correlate. Every call site handles it by name — :func:`cpcv_oos_ic` turns
    it into an ``unmeasurable`` result and :func:`grade_arm` records the date
    as a MISS on the arm's series. Neither substitutes a number, because "the
    cross-section could not be ranked" and "the model ranked it at zero skill"
    are different facts and only one of them is a grade.

    **Ties are averaged** (mid-ranks), which is what `scipy.stats.spearmanr`
    does and what the lifted `training/leakfree_meta_ic.py` therefore did.
    The lift replaced it with an argsort, which hands tied values distinct
    ranks in index order — an ordering invented by the sort, not present in
    the data. Measured on the argsort form:

    * two flat vectors scored ``+1.0``, against a docstring claiming ``0.0``;
    * a collapsed model emitting one constant ``predicted_alpha`` scored
      ``+1.0`` on a flat day — the grader awarding a perfect score to exactly
      the condition the behavioural veto exists to refuse;
    * 45 of 50 names tied scored ``-0.4558`` where the mid-rank Spearman is
      ``-0.1043``, a sign-preserving but four-fold error driven entirely by
      ticker order.

    Implemented in numpy rather than by adding `scipy`. The fleet rule is to
    mirror the SOTA pattern that already exists rather than invent a parallel
    one, and mid-ranking IS that pattern — but the pattern is the statistic,
    not the wheel. `scipy` is a large compiled dependency this repository does
    not otherwise need, on the cold start of every job, for one function of
    twelve lines whose result is asserted here against hand-computed
    mid-rank values. Adding it would also make the M slot's grade depend on a
    transitive numpy ABI pin that `pyproject.toml` deliberately floors.
    """
    if predicted.size != actual.size:
        raise ValueError(
            f"rank IC needs paired vectors; got {predicted.size} predicted and "
            f"{actual.size} actual. A length mismatch here is a panel-indexing defect, "
            "not something to align away."
        )
    if predicted.size < 2:
        return None
    pr = _ranks(predicted)
    ar = _ranks(actual)
    pr = pr - pr.mean()
    ar = ar - ar.mean()
    denom = float(np.sqrt((pr * pr).sum() * (ar * ar).sum()))
    if denom == 0.0:
        return None
    return float((pr * ar).sum() / denom)


def _ranks(values: np.ndarray) -> np.ndarray:
    """Average (mid-)ranks: every member of a tied block gets the block mean.

    A group of ``k`` equal values occupying sorted positions ``i .. i+k-1``
    all receive ``(i + i+k-1) / 2``. With no ties this is identical to the
    argsort form it replaces, so the untied paths of the grader are
    unchanged; with ties it is the only form under which a constant vector
    has zero dispersion and is therefore correctly unrankable.
    """
    order = np.argsort(values, kind="stable")
    ordered = np.asarray(values, dtype="float64")[order]
    ranks = np.empty(ordered.size, dtype="float64")
    start = 0
    for end in range(1, ordered.size + 1):
        if end == ordered.size or ordered[end] != ordered[start]:
            ranks[order[start:end]] = 0.5 * (start + end - 1)
            start = end
    return ranks


# ---------------------------------------------------------------------------
# Grading — the per-date series the arena pairs on.
# ---------------------------------------------------------------------------


#: A per-date score is only produced from a fit that never saw that date's
#: label. This names the mechanism in the artifact so a reader of a verdict
#: does not have to infer it from the code that produced it.
OOS_METHOD = "walk_forward_purged"


@dataclass(frozen=True)
class SettledCrossSections:
    """An arm's out-of-sample predictions beside the labels that REALIZED.

    `alpha-engine-config-I10680`. The walk-forward grader already builds a
    predicted and a realized vector for every date it WALKS, and used to
    keep only the rank IC of the pair. Three of the §5.3 veto's four inputs
    are statistics of exactly this block — a realized directional hit rate,
    and the up-probability calibration that `stdev_p_up` and
    `n_high_confidence` are read through — so the block is now carried out
    of the grader rather than recomputed by a second walk somewhere else,
    which is the only way the veto's inputs and the arm's score can be
    guaranteed to describe the same fit on the same dates.

    **Settled only.** A date is here when its `label_horizon` forward return
    is realized on or before the cycle's `as_of` (:func:`settled_training_
    days`). A date whose label has not settled has a prediction and no
    outcome, and counting it would put an unresolved bet in a hit rate.

    **Not gated by `registered_at`** (`alpha-engine-config-I10709`). Every
    date here is still walked forward and still predicted by a fit trained
    only on days that had settled before it — it is the arm's measurable
    BEHAVIOUR, not its track record. The track record is `ModelGrade.series`
    and that one opens at registration; nothing read off this block reaches
    a ladder rung, a paired window or `promote_min_weeks`.

    **Cross-sectional excess, not raw return.** Both sides are demeaned per
    date. M declares `population` as its benchmark — the scored cross-section
    IS the benchmark (plan §9.1) — so "the model was directionally right"
    means right about beating the population, and the 0.50 floor
    :data:`FLOOR_VETO_METRICS` applies is then a genuine coin flip rather
    than a bar that market drift alone clears. Demeaning is a shift, not a
    division: nothing about §5.3's scale-dependence is touched by it.
    """

    dates: tuple[str, ...] = ()
    #: Shape ``(len(dates), n_names)``, cross-sectionally demeaned.
    predicted: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype="float64"))
    #: Shape ``(len(dates), n_names)``, cross-sectionally demeaned.
    realized: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype="float64"))

    def __post_init__(self) -> None:
        if self.predicted.shape != self.realized.shape:
            raise ValueError(
                f"predicted {self.predicted.shape} and realized {self.realized.shape} must "
                "describe the same block; a mismatch would pair a prediction with another "
                "name's outcome"
            )
        if self.predicted.shape[0] != len(self.dates):
            raise ValueError(
                f"{len(self.dates)} date(s) but {self.predicted.shape[0]} row(s) of predictions"
            )

    @property
    def n_dates(self) -> int:
        return len(self.dates)

    def tail(self, n: int) -> SettledCrossSections:
        """The most recent ``n`` settled decision dates."""
        return SettledCrossSections(
            dates=self.dates[-n:], predicted=self.predicted[-n:], realized=self.realized[-n:]
        )


@dataclass(frozen=True)
class ModelGrade:
    """One arm's cycle grade: the OOS series, the CPCV battery, the fit.

    ``series`` is the number the arena pairs on and the pointer is therefore
    decided by, and every date in it is OUT-OF-SAMPLE (see :func:`grade_arm`).
    ``status`` is ``ok`` or ``unmeasurable``; an arm with too little
    out-of-sample history carries ``unmeasurable``, an empty series and a
    ``reason``, and is never handed a backfilled one.

    **There is deliberately no pooled figure on this object** (plan §9.1: "the
    grader refuses to report a single pooled figure"). ``oos_n`` and
    ``in_sample_n`` are reported separately, and a caller that wants a summary
    states which population it is summarising. A single mean across an
    in-sample warm-up and an out-of-sample window is a number no decision can
    legitimately be taken on, and the way it gets taken anyway is by existing.
    """

    series: ArmSeries
    cpcv: CPCVResult
    fit: Fit
    status: str
    reason: str
    oos_start: str
    oos_n: int
    in_sample_n: int
    unrankable_dates: tuple[str, ...] = ()
    oos_method: str = OOS_METHOD
    benchmark: str = "population"
    #: The WALKED dates whose labels have REALIZED, with both sides of each
    #: cross-section. The §5.3 veto's realized inputs are read off this and
    #: nothing else — see :class:`SettledCrossSections`. Deliberately a wider
    #: window than ``series``, which opens at ``registered_at``: see
    #: :func:`grade_arm`'s "two windows" paragraph
    #: (`alpha-engine-config-I10709`).
    settled: SettledCrossSections = field(default_factory=SettledCrossSections)

    def __post_init__(self) -> None:
        if self.status not in ("ok", "unmeasurable"):
            raise ValueError(f"ModelGrade.status must be ok|unmeasurable; got {self.status!r}")
        if self.status == "unmeasurable" and not self.reason:
            raise ValueError(
                "an unmeasurable grade must carry the reason it could not be measured; "
                "an unexplained absence is indistinguishable from a producer that never ran"
            )


def grade_arm(recipe: ModelRecipe, panel: FeaturePanel, *, as_of: str) -> ModelGrade:
    """Score ``recipe`` per trading day, OUT-OF-SAMPLE, against the population.

    The per-date score is the cross-sectional rank IC of that day's
    predictions against that day's realized forward return — the scored
    cross-section IS the benchmark, which is why M declares `population`
    rather than SPY. The engine then pairs these series across arms; nothing
    here computes a comparison.

    **What changed, and why it had to** (plan §9.1, the out-of-sample clock).
    This function used to fit ONCE on every date ``<= as_of`` and then score
    those same dates with that fit. The series the arena paired on — and so
    the series the champion pointer was decided by — was therefore in-sample,
    and the only out-of-sample number on this object, ``cpcv``, was consumed
    by nothing on the promotion path. Measured on 40 pure-noise features over
    a pure-noise label: the in-sample series read ``+0.0728`` mean IC while
    CPCV OOS read ``-0.0214``, and a 40-feature arm beat a 3-feature arm
    ``0.0728`` vs ``0.0118`` on data containing no signal at all. The slot was
    ranking capacity to overfit and calling it edge. Exposing ``cpcv``
    alongside the in-sample series would not have fixed it: the number the
    pointer reads is the one that has to change.

    **The construction.** Walking forward over the panel's own session axis:

    * the window opens at ``max(recipe.registered_at, first date with a full
      settled training window)`` — an arm accumulates evidence from
      registration, never before it, so a three-day-old arm cannot be handed a
      120-date track record;
    * a day is scored by a fit trained only on days whose labels had SETTLED
      by that day (:func:`settled_training_days`), which is the purge — the
      day's own label, and the ``label_horizon`` days before it whose labels
      overlap it, are outside the fit;
    * refits happen on the recipe's declared ``refit_cadence_trading_days``,
      which is the cadence the arm actually runs at; a fit carried forward
      between refits is older than the day it scores and so still strictly
      out-of-sample. The cadence is anchored at the END OF THE WARM-UP — the
      first day the arm has a fit that could have existed — not at
      ``registered_at``, so one arm has one walk-forward whatever date it was
      filed on.

    **Two windows, and only one of them is the registration clock**
    (`alpha-engine-config-I10709`). ``series`` — the track record the arena
    pairs, and so everything ``promote_min_weeks`` measures — opens at
    ``registered_at`` and nothing widens it. ``settled`` — the block the §5.3
    behavioural veto's realized inputs are read off (:func:`serving_metrics`)
    — is every walked date whose label has realized, registration or no
    registration. The veto is a SERVING PRECONDITION about how the arm behaves
    (dispersion, a directional hit rate), not evidence of edge: it can refuse
    an arm and can never promote one, and each of its dates is predicted by a
    fit trained only on days settled before it, so widening it backfills no
    track record and clears no age bar.

    Measured 2026-09-14, which is why this distinction is written down: with
    one window for both, ``residual_momentum`` — registered 2026-09-01, 68
    backfilled sessions behind it, 47 paired dates in the arena — carried 0
    settled dates, the veto read "could not be computed" for three of its four
    inputs, and the arm was INELIGIBLE for a reason no amount of accumulated
    history could resolve inside :func:`_grade_lookback`'s window. A gate dark
    for a new arm's first ~50 sessions is dark exactly where it is needed.

    **Nothing is backfilled.** Too little OOS history is ``status ==
    "unmeasurable"`` with an empty series and a reason. A day whose
    cross-section cannot be ranked — a collapsed constant prediction, a flat
    label block — is a MISS on the series (`ArmSeries.misses`), which the
    engine excludes from every window, because "unrankable" is not "zero
    skill".

    **The series carries its feature-layer lineage** (`alpha-engine-config-
    I9903`, wired by `-I9963`). `ArmSeries.lineage` is an opaque
    `dimension -> distinct values` map the arena engine passes through
    unread and `ScoreLadder` emits per ladder entry, so a verdict artifact
    can answer "did this champion's score series rest on one feature-layer
    version or several" without walking every constituent run manifest. The
    M slot declares one dimension, `feature_version`, and its value is the
    panel's — not a version supplied by a caller.

    **No scheduled job calls this function** (`alpha-engine-config-I9957`,
    still true: the whole of this module is reachable only from `tests/`).
    The lineage above is therefore a CONTRACT exercised by tests, not a fact
    observed on a live M cycle, and it reaches no `arena_cycle` document any
    scheduled run writes until phase 3 builds the M cycle job.
    """
    fit = train_arm(recipe, panel, as_of=as_of)
    horizon = recipe.label_horizon_trading_days
    cadence = recipe.refit_cadence_trading_days
    minimum = recipe.training_window.min_trading_days
    n_names = len(panel.names)

    scorable = [i for i, day in enumerate(panel.dates) if day <= as_of]
    # The scored dates whose labels have realized by `as_of`. Computed once,
    # from the same function the purge uses, so "settled" means one thing on
    # the training side and the measurement side (`alpha-engine-config-I10680`).
    realized_by_as_of = set(settled_training_days(panel, as_of=as_of, label_horizon=horizon))
    settled_dates: list[str] = []
    settled_predicted: list[np.ndarray] = []
    settled_realized: list[np.ndarray] = []
    in_sample_n = 0
    scores: dict[str, float] = {}
    unrankable: list[str] = []
    coefficients: np.ndarray | None = None
    intercept = 0.0
    last_refit: int | None = None

    for i in scorable:
        day = panel.dates[i]
        train_days = settled_training_days(panel, as_of=day, label_horizon=horizon)
        if recipe.training_window.kind == "rolling":
            train_days = train_days[-minimum:]
        if len(train_days) < minimum:
            # Warm-up: the arm has no fit that could have existed on this day.
            # Counted and reported as `in_sample_n`, never scored — the count
            # is what makes "OOS N = 4" legible beside "panel N = 160".
            in_sample_n += 1
            continue
        if coefficients is None or last_refit is None or (i - last_refit) >= cadence:
            coefficients, intercept, _, _ = _fit_rows(recipe, panel, train_days)
            last_refit = i
        rows = i * n_names + np.arange(n_names)
        predicted = _design(recipe, panel, rows) @ coefficients + intercept
        actual = panel.forward_returns.reshape(-1)[rows]
        if i in realized_by_as_of:
            # Demeaned per date: M's benchmark is the cross-section it scored
            # (plan §9.1), so the quantity with a settled outcome is the
            # excess, not the raw return. An unrankable date is still a
            # settled observation — it is a miss on the SERIES because "no
            # ordering" is not "zero skill", which says nothing about whether
            # the day's directions were right.
            #
            # Collected BEFORE the registration gate below, and that placement
            # is the whole of `alpha-engine-config-I10709`. See the
            # "two windows" paragraph in this function's docstring.
            #
            # Demeaned over the names carrying BOTH sides, and only them
            # (`alpha-engine-config-I10709`, the second half). A plain `.mean()`
            # over a cross-section holding one non-finite name returns NaN and
            # writes NaN into every one of that date's 900-odd entries, so a
            # single unpriced ticker silently voided the whole date — and, on
            # the real universe, all 30 of them: measured 2026-09-14, 0 of
            # 27,240 name-date pairs finite, `model_hit_rate_30d` absent and
            # the up-probability calibration refusing a block it read as
            # constant. Both downstream readers mask non-finite pairs already
            # (:func:`realized_hit_rate`, :func:`calibrate_up_probability`), so
            # the mask belongs here, where the population being demeaned is
            # decided: the excess is measured against the names actually
            # scored, which is the same population both sides describe.
            usable = np.isfinite(predicted) & np.isfinite(actual)
            if usable.any():
                settled_dates.append(day)
                settled_predicted.append(
                    np.where(usable, predicted - predicted[usable].mean(), np.nan)
                )
                settled_realized.append(np.where(usable, actual - actual[usable].mean(), np.nan))
        if day < recipe.registered_at:
            # Before registration the arm did not exist to be SCORED. Plan
            # §9.1: the track record BEGINS at registration, so these dates
            # reach no series, no ladder rung and no paired window.
            in_sample_n += 1
            continue
        ic = _rank_ic(predicted, actual)
        if ic is None:
            unrankable.append(day)
            continue
        scores[day] = ic

    cpcv = cpcv_oos_ic(
        panel,
        recipe=recipe,
        cpcv=recipe.cpcv,
        label_horizon_trading_days=recipe.label_horizon_trading_days,
    )
    oos_start = min(scores) if scores else recipe.registered_at
    if scores:
        status, reason = "ok", ""
    else:
        status = "unmeasurable"
        reason = (
            f"arm {recipe.name!r} has no out-of-sample date up to {as_of}: its window opens "
            f"at registered_at={recipe.registered_at} and needs "
            f"{minimum} settled training day(s) at a {horizon}-session label horizon before "
            f"its first score. {in_sample_n} panel date(s) fell in the warm-up and "
            f"{len(unrankable)} could not be ranked. An arm with no out-of-sample history "
            "gets no series — a backfilled one is what let a three-day-old arm carry a "
            "120-date track record past `promote_min_weeks` (plan §9.1)."
        )
    return ModelGrade(
        series=ArmSeries(
            arm_id=fit.arm_id,
            scores=scores,
            misses=frozenset(unrankable),
            # The M slot's own provenance, carried to the verdict surface
            # (`alpha-engine-config-I9903`/`-I9963`). Every date scored above
            # comes from THIS panel, so the panel's resolved feature-layer
            # version is the lineage of the whole series and no manifest walk
            # is needed to derive it.
            #
            # Declared only when there IS a scored date. The contract is "the
            # distinct values this dimension took across the dates in
            # `scores`", and an unmeasurable arm has no dates — declaring the
            # panel's version anyway would record a version the series never
            # rested on. `{}` is the honest reading, and the engine's own
            # normaliser refuses a dimension with nothing to report rather
            # than emitting an empty claim.
            lineage={"feature_version": (panel.feature_version,)} if scores else {},
        ),
        cpcv=cpcv,
        fit=fit,
        status=status,
        reason=reason,
        oos_start=oos_start,
        oos_n=len(scores),
        in_sample_n=in_sample_n,
        unrankable_dates=tuple(unrankable),
        settled=SettledCrossSections(
            dates=tuple(settled_dates),
            predicted=np.array(settled_predicted, dtype="float64").reshape(
                len(settled_dates), n_names
            ),
            realized=np.array(settled_realized, dtype="float64").reshape(
                len(settled_dates), n_names
            ),
        ),
    )


# ---------------------------------------------------------------------------
# The M cycle job — `experiment.run --slot m` and `experiment.grade --slot m`.
#
# `alpha-engine-config-I9957` deliverables 1 and 2. Everything above this line
# was reachable only from `tests/` until this section existed: the loader had
# no production caller, so `SlotRecipes.refusal_metrics` — one `unservable`
# row per refused arm — landed on no manifest a scheduled run ever wrote, and
# a standing refusal was a fact nobody was shown.
#
# The shape is the one every other slot already has (plan §4.4, "four slots,
# one engine"): `produce` and `grade` with `crucible.slots.research`'s
# signature, read off the module by `crucible.slots.dispatchable_slots`.
# M differs from U and R in exactly one place — how its cross-section is
# produced. A recipe is FITTED (`design_panel` -> `train_arm` ->
# `predict_cross_section`) rather than ranked by a pure function, and the
# fitted cross-section is written twice: once as the `arm_predictions.v1`
# artifact a stacked arm consumes, and once as the `ShadowSelection` +
# `ScoredCrossSection` pair `crucible.slots.cycle.run_grade` scores. The
# grading half is `run_grade` UNCHANGED — a second copy of the grader for one
# slot is the four-drifting-implementations shape v2 exists to remove.
# ---------------------------------------------------------------------------

#: This module's slot key, read by `crucible.slots.dispatchable_slots` through
#: :func:`produce` / :func:`grade` rather than from this constant.
SLOT = "m"

#: How many names an M arm's shadow SELECTS out of the cross-section it
#: scored. The M recipe declares no `top_n` — it predicts an alpha for every
#: name and expresses no view about portfolio size, which is the S slot's
#: decision — so the count is the harness's, and it is the same default
#: `crucible.slots.grading._top_n` applies to a U/R recipe that omits one.
#: Identical across every M arm on purpose: policy §4 grades a slot's arms on
#: one axis, and two arms selecting different counts have different dispersion
#: before either has any skill.
M_SELECTION_TOP_N = 10

#: The metric one arm's out-of-sample CPCV rank IC files on the grading job's
#: manifest. Named rather than spelled at the call site so a console adapter
#: and a test read the same literal.
CPCV_OOS_IC_METRIC = "model_cpcv_oos_ic"


@dataclass(frozen=True)
class RegisteredModelArm:
    """A loaded :class:`ModelRecipe` in the shape the register and grader read.

    `crucible.slots.arms.register_arms` and `crucible.slots.cycle.run_grade`
    are written against `ArmSpec` — `arm_id`, `spec`, `registered_at`,
    `supersedes`, `control`, `params`. An M recipe carries the same facts
    under a different document schema (it has no `ranker` and no `params`),
    which is exactly why `load_arm_specs` refuses slot `m` by name
    (`crucible.slots.arms.FOREIGN_RECIPE_LOADERS`).

    **The id is the RECIPE's, never re-derived here.** Wrapping a
    `ModelRecipe` in a real `ArmSpec` would hash `ranker`/`params` into
    `derive_arm_id` and produce a second id for one arm — the register, the
    shadows, the predictions artifact and the series would then each speak
    about a different arm, and every surface would render it as healthy.
    This adapter forwards :attr:`ModelRecipe.arm_id` and
    :attr:`ModelRecipe.spec` untouched, so there is one identity.

    ``params`` exists for one reader: `run_grade`'s `_control_top_n`, which
    count-matches the slot's controls to the arms they are checking.
    """

    recipe: ModelRecipe
    top_n: int = M_SELECTION_TOP_N

    def __post_init__(self) -> None:
        """§4.12: the OOS clock starts on a SESSION, or it starts nowhere.

        The same assertion `ArmSpec.__post_init__` makes, made here for the
        same reason: `registered_at` is what every ladder rung, eligibility
        rung and grace rung is counted from, so a recipe naming a Saturday
        starts its whole eligibility clock on a day the market never traded.
        :class:`ModelRecipe` validates the field is an ISO date and stops
        there — a date and a session are two different claims.
        """
        from crucible.calendar import assert_trading_day  # noqa: PLC0415 - avoids a cycle

        assert_trading_day(
            self.recipe.registered_at,
            context=(
                f"M arm {self.recipe.slot}:{self.recipe.name} `registered_at` "
                f"(source: {self.recipe.source_key or 'constructed in code'})"
            ),
        )

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
        return self.recipe.registered_at

    @property
    def params(self) -> dict[str, Any]:
        return {"top_n": self.top_n}

    #: The register LINK, which is not the same fact as the recipe's declared
    #: `supersedes` — see :func:`registration_specs`. Set there, never here.
    supersedes: str | None = None
    #: Provenance the register row carries verbatim: the declared lineage,
    #: including a parent this slot refuses and therefore never registers.
    notes: str = ""
    #: An M recipe is never a control: controls are GENERATED by
    #: `crucible.slots.arms.control_specs` and never filed, precisely so a
    #: planted-edge arm cannot sit in the strategy tree where an operator
    #: could clear the flag (§10.1).
    control: bool = False
    control_kind: str | None = None
    bootstrap: bool = False


def registration_specs(loaded: SlotRecipes) -> list[RegisteredModelArm]:
    """The slot's registered recipes, with their declared lineage resolved.

    Thin per-slot adapter over `crucible.slots.arms.resolve_declared_lineage`
    (`alpha-engine-config-I10637`, `policy-shared-code`'s second-adoption
    trigger: this function and `crucible.slots.strategy.registration_specs`
    carried the identical lineage-resolution guard, raising the identical
    :class:`~crucible.slots.arms.SupersededArmUndeclaredError`, over a
    different recipe type). The resolution logic — what makes an M arm that
    supersedes a declared-but-refused sibling register anyway
    (`alpha-engine-config-I9957`, `-I9808` ruling (b)) — now lives once, in
    `crucible.slots.arms`; what stays here is only the M-specific shape:
    `loaded.registered` is a tuple of raw `ModelRecipe`s (M has no separate
    "registered arm" wrapper until this function builds one), and the output
    wraps each in :class:`RegisteredModelArm`.
    """
    resolved = resolve_declared_lineage(
        registered=[
            (recipe.name, recipe.arm_id, recipe.supersedes) for recipe in loaded.registered
        ],
        refused=[r.arm for r in loaded.refused],
    )
    specs: list[RegisteredModelArm] = []
    for recipe in loaded.registered:
        link, notes = resolved[recipe.arm_id]
        specs.append(RegisteredModelArm(recipe, supersedes=link, notes=notes))
    return specs


def _in_dependency_order(specs: Sequence[RegisteredModelArm]) -> list[RegisteredModelArm]:
    """Base arms before the arms that stack on them, name-ordered within a rank.

    A stacked arm's design matrix reads `predictions/{base}/{day}.json`, which
    the base arm's own produce writes THIS cycle. Producing in file-name order
    would therefore refuse the stack on the one session it could have been
    satisfied — the base's artifact for today does not exist until the base
    has run. The cycle is already impossible: `partition_producible` runs
    `_assert_acyclic` at load and refuses the whole slot on one, so a rank
    always exists here.
    """
    by_name = {s.name: s for s in specs}
    rank: dict[str, int] = {}

    def _rank(name: str, seen: frozenset[str]) -> int:
        if name in rank:
            return rank[name]
        spec = by_name[name]
        bases = [r.ref for r in spec.recipe.inputs if r.kind == "predictions" and r.ref in by_name]
        rank[name] = 1 + max((_rank(b, seen | {name}) for b in bases), default=-1)
        return rank[name]

    for spec in specs:
        _rank(spec.name, frozenset())
    return sorted(specs, key=lambda s: (rank[s.name], s.name))


def _load_slot(ctx: Any, *, settings: Any) -> SlotRecipes:
    """Load the M slot and put every refusal on THIS run's manifest, first.

    Deliverables 1 and 2 of `alpha-engine-config-I9957` in one place, and the
    order matters in both directions:

    * a refused arm's `unservable` row is recorded BEFORE any fitting work,
      so it reaches the manifest whether the rest of the run succeeds or
      raises — `crucible.runner.run_job` writes the manifest in a `finally`,
      and metrics recorded before the raise are on it;
    * when NOTHING registers, :class:`SlotUnservableError` is re-raised after
      the rows are recorded rather than propagating straight out of the
      loader. Without this the whole-slot case would page with a `reason` and
      no per-arm rows — the least informative manifest of the three possible
      outcomes, on the worst of them.

    The source is the checkout when one is configured and the store otherwise,
    the same resolution order `crucible.slots.arms.load_arm_specs` uses: a
    developer editing `alpha-engine-config/strategy/` expects the edit to take
    effect, and a spot box has no checkout at all.
    """
    strategy_dir = getattr(settings, "strategy_dir", None)
    directory = Path(strategy_dir) / "arms" / SLOT if strategy_dir is not None else None
    try:
        loaded = load_model_recipes(directory, store=None if directory is not None else ctx.store)
    except SlotUnservableError as exc:
        for metric in SlotRecipes(registered=(), refused=exc.refusals).refusal_metrics(slot=SLOT):
            ctx.record_metric(metric)
        raise
    for metric in loaded.refusal_metrics(slot=SLOT):
        ctx.record_metric(metric)
    return loaded


def _registered_arms(ctx: Any, *, settings: Any) -> tuple[SlotRecipes, list[RegisteredModelArm]]:
    loaded = _load_slot(ctx, settings=settings)
    return loaded, registration_specs(loaded)


def produce(ctx: Any, *, settings: Any, **kwargs: Any) -> dict[str, Any]:
    """Fit every registered M arm for one trading day and write its cross-section.

    The M half of `experiment.run`. Same signature as
    `crucible.slots.research.produce`, because `crucible.track_a` dispatches
    both through one call and `crucible.slots.dispatchable_slots` reads this
    name off the module.

    Per arm, in base-before-stack order: :func:`design_panel` (the ONE seam
    that materialises a declared `predictions[...]` input),
    :func:`train_arm`, :func:`predict_cross_section`, then two writes of the
    same numbers —

    * :func:`produce_arm_predictions`, the `arm_predictions.v1` artifact a
      stacked arm consumes and `crucible explain` walks;
    * a :class:`~crucible.slots.grading.ShadowSelection` and
      :class:`~crucible.slots.grading.ScoredCrossSection` through
      `write_shadow` / `write_cross_section`, exactly as
      `crucible.slots.cycle.run_produce` writes them for R — which is what
      lets `run_grade` score M with no M-specific grading code at all.

    Both are derived from ONE :func:`predict_cross_section` call, so the
    artifact a stacked arm trains on and the shadow the arena scores can
    never be two different rankings of one session.

    **A `TrainingIntegrityError` fails the whole slot run** (plan §4.4, Brian
    ruling 2026-08-29). It is not caught here and must not be: arms in a slot
    share a training substrate, so a defect that spoils one fit is evidence
    the cycle's inputs are compromised, and "this arm had nothing to say" and
    "this arm's inputs were broken" must never render alike.
    """
    from crucible.calendar import assert_trading_day  # noqa: PLC0415 - avoids a cycle
    from crucible.serving import publish_predictions_feed  # noqa: PLC0415 - avoids a cycle
    from crucible.slots import get_slot  # noqa: PLC0415 - avoids a cycle
    from crucible.slots.arms import (  # noqa: PLC0415 - avoids a cycle
        control_specs,
        read_register,
        register_arms,
        write_register,
    )
    from crucible.slots.cycle import MissingArtifactError  # noqa: PLC0415 - avoids a cycle
    from crucible.slots.grading import (  # noqa: PLC0415 - avoids a cycle
        ScoredCrossSection,
        ShadowSelection,
        write_cross_section,
        write_shadow,
    )

    arm_name: str | None = kwargs.get("arm_name")
    feature_version: str | None = kwargs.get("feature_version")
    slot_spec = get_slot(SLOT)
    trading_day = ctx.trading_day.isoformat()
    assert_trading_day(
        ctx.trading_day, context=f"experiment.run --slot {SLOT} --date {trading_day}"
    )

    loaded, specs = _registered_arms(ctx, settings=settings)
    if arm_name is not None:
        specs = [s for s in specs if s.name == arm_name]
        if not specs:
            raise MissingArtifactError(
                f"no arm named {arm_name!r} registered in slot {SLOT!r}; the slot "
                f"registered {[r.name for r in loaded.registered]} and refused "
                f"{[r.arm for r in loaded.refused]}. Producing nothing and exiting 0 "
                "would be indistinguishable from an arm that ran and predicted nothing."
            )

    register = read_register(ctx.store, SLOT)
    register, _ = register_arms(register, [*specs, *control_specs(slot_spec)])
    write_register(ctx.store, SLOT, register)

    source = FeatureLayerSource(store=ctx.store, version=feature_version)
    # The WHOLE registered set, never the `--arm` filtered one: `design_panel`
    # derives a stacked arm's base id from the recipe set it is given, so a
    # filtered set makes `experiment.run --slot m --arm <a stack>` refuse its
    # own base as an arm the slot does not declare. Which arms PRODUCE is a
    # different question from which arms the slot DECLARES.
    recipes = list(loaded.registered)
    produced: list[str] = []
    warming: list[InputRefusal] = []
    excluded_rows = 0
    for spec in _in_dependency_order(specs):
        recipe = spec.recipe
        try:
            panel = design_panel(
                recipe,
                source=source,
                trading_day=trading_day,
                recipes=recipes,
                store=ctx.store,
                # The whole declared training window, plus the label horizon
                # whose final rows `settled_training_days` purges. Asking for
                # fewer sessions than the recipe's own `min_trading_days` would
                # make `train_arm` refuse a window the store could have
                # supplied.
                lookback_trading_days=(
                    recipe.training_window.min_trading_days + recipe.label_horizon_trading_days
                ),
                ctx=ctx,
            )
        except BasePredictionsUnavailableError as exc:
            # A DELIBERATE per-arm refusal, and the only swallow in this loop.
            #
            # Failure mode absorbed: a stacked arm's base has not produced a
            # prediction for every session of the stack's training window yet.
            # That is a WARM-UP, not a broken input — the base is registered,
            # producible, and filling its own history one cycle at a time —
            # and it is unsatisfiable by construction on the base arm's first
            # `min_trading_days` cycles. Left slot-wide it makes the whole M
            # slot unproducible for as long as any stacked arm is filed, which
            # is exactly the blast radius `alpha-engine-config-I9955` removed
            # one layer in, at registration; the arm Brian's I9808 ruling (b)
            # added is a stack, so the condition is live from the first cycle.
            #
            # Why the deliverable survives: the base arm and every unstacked
            # sibling produce normally, so the slot serves and accumulates
            # evidence. Recording surface: `ARM_REFUSED_METRIC` on THIS run's
            # manifest with `status: unservable`, the same row and the same
            # vocabulary a registration refusal files, naming the arm, the
            # input and the missing sessions.
            #
            # What is NOT absorbed: a defective feature layer, a non-finite
            # prediction, or a training window the layer cannot supply. Those
            # are `TrainingIntegrityError` and still fail the whole slot.
            warming.append(
                InputRefusal(
                    arm=recipe.name,
                    unresolvable=tuple(r.text for r in recipe.inputs if r.kind == "predictions"),
                    reason=str(exc),
                )
            )
            continue
        fit = train_arm(recipe, panel, as_of=trading_day)
        if fit.completeness is not None:
            # §9.2 class 4 and class 5, both: the count with its reason, and
            # the ratio with the ceiling it was measured against. An excluded
            # ticker-row that reaches no manifest is a suppression collection
            # with no file (`crucible/AGENTS.md` rule 4).
            ctx.record_metric(fit.completeness.as_metric(slot=SLOT, phase="training"))
            if fit.completeness.rows_excluded:
                ctx.record_rejected(fit.completeness.detail, fit.completeness.rows_excluded)
                excluded_rows += fit.completeness.rows_excluded
        predicted, serving = score_cross_section(fit, panel, trading_day=trading_day)
        ctx.record_metric(serving.as_metric(slot=SLOT, phase="serving"))
        if serving.rows_excluded:
            ctx.record_rejected(serving.detail, serving.rows_excluded)
            excluded_rows += serving.rows_excluded
        produce_arm_predictions(ctx, fit=fit, panel=panel, trading_day=trading_day)
        ranked = sorted(predicted.items(), key=lambda item: (-item[1], item[0]))
        shadow = ShadowSelection(
            arm_id=fit.arm_id,
            trading_day=trading_day,
            selection=tuple(ticker for ticker, _ in ranked[: spec.top_n]),
            population=tuple(sorted(predicted)),
            # Provenance, not a lookup: the M slot resolves no `crucible.slots.
            # rankers` callable, and naming one here would claim a ranking
            # function this arm never ran. The estimator IS what ranked it.
            ranker=f"model:{recipe.estimator.kind}",
            params=dict(spec.params),
            feature_version=panel.feature_version,
        )
        write_shadow(ctx.store, shadow)
        ctx.record_output(
            shadow_key(shadow.arm_id, shadow.trading_day),
            json.dumps(shadow.to_dict(), indent=2, sort_keys=True).encode("utf-8"),
            schema_version="shadow.v1",
        )
        cross_section = ScoredCrossSection(
            arm_id=fit.arm_id,
            trading_day=trading_day,
            ranks=tuple(
                (ticker, score, position)
                for position, (ticker, score) in enumerate(ranked, start=1)
            ),
        )
        write_cross_section(ctx.store, cross_section)
        ctx.record_output(
            cross_section_key(cross_section.arm_id, cross_section.trading_day),
            json.dumps(cross_section.to_dict(), indent=2, sort_keys=True).encode("utf-8"),
            schema_version="cross_section.v2",
        )
        produced.append(fit.arm_id)

    for refusal in warming:
        ctx.record_metric(
            SlotRecipes(registered=(), refused=(refusal,)).refusal_metrics(slot=SLOT)[0]
        )
    if warming and not produced:
        # Every arm that registered is waiting on a base that has not run
        # enough sessions: the slot can serve nothing this cycle, which is the
        # same `unservable` reading an all-refused registration produces and
        # pages through the same failed manifest. It is NOT `ok` — a slot that
        # produced no cross-section did not have nothing to do.
        raise SlotUnservableError(tuple(warming))

    # The SERVING half of this job, and the second half of the trader contract
    # (`crucible/AGENTS.md`: the trader reads `champions/{slot}/current.json`
    # PLUS `predictions/{trading_day}.json`). A republication of exactly the
    # `arm_predictions.v1` document the champion pointer resolves to — never a
    # fourth derivation of the same numbers — so the trader's cross-section and
    # the one `crucible explain` walks are the same bytes. Returns None and
    # writes nothing while the slot has no champion, which is every cycle
    # before the M slot's first promotion; refuses outright for a pointer whose
    # producing run was not `ok`, and raises when the pointer names an arm that
    # produced nothing this cycle. `alpha-engine-config-I10129`.
    feed_written = publish_predictions_feed(ctx.store, trading_day=trading_day, ctx=ctx)

    ctx.record_rows(rows_in=len(specs), rows_out=len(produced))
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
                f"slot {SLOT}: {len(produced)} registered arm(s) fitted and predicted a "
                f"cross-section for {trading_day}; {len(loaded.refused)} arm(s) refused "
                "at registration"
            ),
            "source_path": f"predictions/*/{trading_day}.json",
            "last_updated_utc": _utc_now(),
        }
    )
    return {
        "slot": SLOT,
        "trading_day": trading_day,
        "arms": produced,
        "refused": [{"arm": r.arm, "unresolvable": list(r.unresolvable)} for r in loaded.refused],
        "feature_version": str(source.version),
        # The count, not the record: the per-column detail is on the manifest
        # metric, and a payload that restated it would be a second copy free
        # to disagree with the first.
        "training_rows_excluded": excluded_rows,
        # `None` when the slot has no champion, which is a true statement and
        # deliberately a different one from a key that was written.
        "champion_feed": feed_written,
    }


#: IRLS iterations the up-probability calibration is allowed. A logistic fit
#: on two parameters converges in a handful; not converging inside this many
#: is a degenerate block (perfect separation, a constant predictor), and the
#: calibration REFUSES rather than returning the last iterate. A half-fitted
#: map would put a number on `stdev_p_up` that no measurement stands behind.
_CALIBRATION_MAX_ITERATIONS = 50

#: Convergence tolerance on the coefficient step, in units of the internally
#: standardised predictor.
_CALIBRATION_TOLERANCE = 1e-8


@dataclass(frozen=True)
class UpProbabilityCalibration:
    """An arm's fitted map from predicted excess alpha to P(beats the population).

    `alpha-engine-config-I10680`. Platt scaling — a two-parameter logistic
    of the realized direction on the predicted excess — fitted on the arm's
    OWN settled walk-forward block (:class:`SettledCrossSections`), which is
    the only history that is out-of-sample by construction.

    **Fitted on HISTORY, applied to TODAY, and that is what keeps §5.3's
    scale-dependence intact.** The map is a fixed affine function of the
    predicted alpha, so a cross-section whose spread collapses to half
    produces an up-probability spread that collapses with it — which is the
    property `stdev_p_up` is in :data:`DISPERSION_METRICS` for. Fitting the
    map on today's cross-section instead (or standardising the predictor at
    serving time) would divide the collapse away exactly as the standardized
    ratio did on 2026-08-28, and is the one thing this class must never do.
    The internal standardisation below is of the TRAINING block only and is
    folded back out of the returned coefficients, so the map it yields is in
    the arm's own predicted-alpha units.
    """

    slope: float
    intercept: float
    n_pairs: int
    n_dates: int

    def p_up(self, predicted_excess: np.ndarray) -> np.ndarray:
        """P(this name beats the cross-section) for each predicted excess."""
        return 1.0 / (1.0 + np.exp(-(self.slope * np.asarray(predicted_excess) + self.intercept)))


def calibrate_up_probability(settled: SettledCrossSections) -> UpProbabilityCalibration | None:
    """Fit :class:`UpProbabilityCalibration` on ``settled``, or ``None``.

    ``None`` — never a default map — when the block cannot support a fit:
    no rows, one realized direction only (nothing to discriminate), a
    constant predictor, a non-finite value, or IRLS not converging inside
    :data:`_CALIBRATION_MAX_ITERATIONS`. The caller records the absence, the
    veto reads `insufficient`, and the arm does not serve. An identity or
    a 0.5-everywhere fallback would be a stand-in for a statistic nobody
    measured (`champion-challenger-policy.md` §5.1).
    """
    x = np.asarray(settled.predicted, dtype="float64").reshape(-1)
    outcomes = np.asarray(settled.realized, dtype="float64").reshape(-1)
    if x.size == 0:
        return None
    finite = np.isfinite(x) & np.isfinite(outcomes)
    x, outcomes = x[finite], outcomes[finite]
    if x.size == 0:
        return None
    y = (outcomes > 0.0).astype("float64")
    if float(y.sum()) in (0.0, float(y.size)):
        # One class only: a logistic fit would run to infinity, and a map
        # that says "every name goes up" is not a calibration.
        return None
    scale = float(x.std())
    if scale < DEGENERATE_STD:
        return None
    z = x / scale

    beta = np.zeros(2, dtype="float64")
    design = np.column_stack((np.ones_like(z), z))
    converged = False
    for _ in range(_CALIBRATION_MAX_ITERATIONS):
        probability = 1.0 / (1.0 + np.exp(-(design @ beta)))
        weights = probability * (1.0 - probability)
        hessian = design.T @ (design * weights[:, None])
        # A ridge on the order of the float epsilon, not a regulariser: it
        # keeps a near-singular Hessian solvable without moving the fit.
        hessian[np.diag_indices(2)] += 1e-10
        gradient = design.T @ (y - probability)
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            return None
        if not np.isfinite(step).all():
            return None
        beta = beta + step
        if float(np.max(np.abs(step))) < _CALIBRATION_TOLERANCE:
            converged = True
            break
    if not converged or not np.isfinite(beta).all():
        return None
    return UpProbabilityCalibration(
        slope=float(beta[1] / scale),
        intercept=float(beta[0]),
        n_pairs=int(x.size),
        n_dates=settled.n_dates,
    )


def realized_hit_rate(settled: SettledCrossSections) -> float | None:
    """Sign agreement between predicted and realized excess, or ``None``.

    The proportion of (settled date, name) pairs whose predicted excess and
    realized excess share a sign. Both sides are already cross-sectionally
    demeaned (:class:`SettledCrossSections`), so this is "how often was the
    model right about beating the population" — against which
    :data:`FLOOR_VETO_METRICS`' 0.50 is a coin flip rather than a bar that a
    rising market clears on its own.

    Pairs where either side is exactly zero or non-finite are excluded: a
    name the model expressed no view on is not a directional call, and
    scoring it either way would move the statistic without any prediction
    behind it. ``None`` when nothing is left, which the caller records as
    uncomputable rather than as a rate.
    """
    predicted = np.asarray(settled.predicted, dtype="float64").reshape(-1)
    realized = np.asarray(settled.realized, dtype="float64").reshape(-1)
    usable = np.isfinite(predicted) & np.isfinite(realized) & (predicted != 0.0) & (realized != 0.0)
    if not usable.any():
        return None
    agree = np.sign(predicted[usable]) == np.sign(realized[usable])
    return float(agree.sum()) / float(usable.sum())


def _serving_metrics(predicted: dict[str, float]) -> dict[str, Any]:
    """The §5.3 veto input one predicted cross-section supplies on its own.

    `alpha_stdev`, and only it: the other three are statistics of the arm's
    settled HISTORY as well as of today's cross-section, and they are
    produced by :func:`serving_metrics`, which takes both.

    Raw, never standardized (policy §5.3): dividing by the spread is the
    transformation that made 2026-08-28's collapse read as healthy.
    """
    values = np.array(sorted(predicted.values()), dtype="float64")
    return {"alpha_stdev": float(values.std())}


def serving_metrics(
    predicted: dict[str, float],
    *,
    settled: SettledCrossSections,
    window: int = SETTLED_WINDOW_DECISION_DATES,
    top_n: int = M_SELECTION_TOP_N,
) -> tuple[dict[str, Any], str]:
    """Every §5.3 veto input this arm can supply, and why any is missing.

    `alpha-engine-config-I10680` — the producers the veto waited on.

    * ``alpha_stdev`` — today's predicted cross-section's spread, raw.
    * ``model_hit_rate_30d`` — :func:`realized_hit_rate` over the trailing
      :data:`SETTLED_WINDOW_DECISION_DATES` settled decision dates.
    * ``stdev_p_up`` — the spread of the calibrated up-probability across
      today's cross-section, through the arm's own
      :class:`UpProbabilityCalibration`.
    * ``n_high_confidence`` — how many of the ``top_n`` names the slot would
      actually serve carry a calibrated probability above
      :data:`HIGH_CONFIDENCE_P_UP` of beating the population.

    Returns the metrics and a REASON, empty when all four are present. The
    reason is non-empty exactly when the settled window is too short (or a
    fit on it is not supportable), which is the only legitimate route to
    `insufficient` once the producers exist — and it is stated rather than
    left for a reader to infer from an absent key.

    Nothing is defaulted. A metric that cannot be computed is ABSENT, which
    :func:`evaluate_behavioural_veto` reads as `insufficient` and never as a
    pass (`champion-challenger-policy.md` §5.1).
    """
    metrics = _serving_metrics(predicted)
    if settled.n_dates < window:
        return metrics, (
            f"{settled.n_dates} settled out-of-sample decision date(s), and the realized "
            f"veto inputs need {window}: `model_hit_rate_30d` names its own window, and a "
            "hit rate computed over fewer dates reported under that name is a different "
            "statistic wearing it. The up-probability calibration is fitted on the same "
            "block. This is the window being short, not a producer being absent"
        )

    block = settled.tail(window)
    hit_rate = realized_hit_rate(block)
    if hit_rate is not None:
        metrics["model_hit_rate_30d"] = hit_rate

    calibration = calibrate_up_probability(block)
    if calibration is None:
        return metrics, (
            f"the up-probability calibration could not be fitted on {block.n_dates} settled "
            f"date(s) ({block.predicted.size} name-date pair(s)): the block carries one "
            "realized direction only, a constant predictor, or a fit that did not "
            "converge. `stdev_p_up` and `n_high_confidence` are read through that map and "
            "are absent rather than assumed"
        )

    names = sorted(predicted)
    values = np.array([predicted[n] for n in names], dtype="float64")
    # Demeaned to match the block the map was fitted on, which is the
    # cross-sectional excess. A shift, not a division — the collapse the
    # dispersion rule looks for survives it.
    probabilities = calibration.p_up(values - values.mean())
    metrics["stdev_p_up"] = float(probabilities.std())

    served = np.argsort(-values, kind="stable")[: min(top_n, values.size)]
    metrics["n_high_confidence"] = int((probabilities[served] > HIGH_CONFIDENCE_P_UP).sum())
    if hit_rate is None:
        return metrics, (
            f"no (date, name) pair over {block.n_dates} settled date(s) carried a non-zero "
            "predicted AND realized excess, so there is no directional call to score. "
            "`model_hit_rate_30d` is absent rather than 0.5"
        )
    return metrics, ""


def _grade_lookback(recipe: ModelRecipe) -> int:
    """Sessions of panel :func:`grade` reads, so the settled window can fill.

    `alpha-engine-config-I10680`. The produce path needs exactly one fit, so
    it reads ``min_trading_days + label_horizon`` sessions. The grade path
    additionally measures the §5.3 veto's realized inputs over
    :data:`SETTLED_WINDOW_DECISION_DATES` settled out-of-sample decision
    dates, and each of those costs a session twice over: once to be scorable
    at all, once for its label to settle. With the old lookback the
    walk-forward had room for a handful of scorable dates and none of them
    settled, so the realized inputs could not have existed however long the
    slot ran — a producer bounded by the window it was given rather than by
    the data.

    Reading MORE panel does not weaken the purge: every scored day is still
    predicted by a fit trained only on days whose labels had settled by that
    day (:func:`grade_arm`), and the window still opens no earlier than the
    arm's ``registered_at``. It reads more parquet, which is the whole cost.
    """
    return (
        recipe.training_window.min_trading_days
        + 2 * recipe.label_horizon_trading_days
        + SETTLED_WINDOW_DECISION_DATES
    )


def _grade_panel(
    recipe: ModelRecipe, *, source: Any, as_of: str, loaded: SlotRecipes, ctx: Any
) -> FeaturePanel:
    """The panel :func:`grade` walks, at :func:`_grade_lookback` where it can be.

    Two attempts, and the second is not a degradation to hide
    (`alpha-engine-config-I10680`). A STACKED arm reads `predictions[base]` on
    every row of its panel, so it can only be measured over a window its base
    has actually published — and the extended window reaches further back than
    any base has run. Falling back to the produce-sized window keeps the
    stacked arm's CPCV reading and its veto exactly as they were: its settled
    window is then short, the veto reads `insufficient`, and
    :func:`_record_serving_veto_window` says so with the arm named. The
    alternative was the extended read raising and the arm being refused
    outright, which would have taken away a reading it already had.
    """

    def read(lookback: int) -> FeaturePanel:
        return design_panel(
            recipe,
            source=source,
            trading_day=as_of,
            recipes=list(loaded.registered),
            store=ctx.store,
            lookback_trading_days=lookback,
            ctx=ctx,
        )

    try:
        return read(_grade_lookback(recipe))
    except BasePredictionsUnavailableError:
        # Not swallowed: the produce-sized read below either succeeds — and
        # the arm grades exactly as it did before this change — or raises the
        # same exception straight into `grade`'s refusal row.
        return read(_produce_lookback(recipe))


def _produce_lookback(recipe: ModelRecipe) -> int:
    """Sessions of panel one fit needs: the declared window plus the purge."""
    return recipe.training_window.min_trading_days + recipe.label_horizon_trading_days


def grade(ctx: Any, *, settings: Any, **kwargs: Any) -> dict[str, Any]:
    """Score every settled M shadow and run the slot's arena cycle.

    `crucible.slots.cycle.run_grade`, unchanged, with two M-specific facts
    supplied to it as evaluated RESULTS rather than as computations the
    engine performs (policy §5.3):

    * the per-arm **CPCV out-of-sample rank IC** from :func:`grade_arm`,
      recorded as a metric on this run's manifest. It is the battery §3
      requires, and it is a reading, not a second verdict path — the series
      the pointer is decided on is still the one the arena pairs;
    * a **behavioural-veto serving precondition** per arm, from
      :func:`evaluate_behavioural_veto` over the arm's own predicted
      cross-section against the incumbent's. All four of the metrics the veto
      reads are produced here (:func:`serving_metrics`,
      `alpha-engine-config-I10680`), so the reading is `pass` or `veto` on any
      arm carrying :data:`SETTLED_WINDOW_DECISION_DATES` settled
      out-of-sample decision dates, and `insufficient` only while that window
      is short. The veto is evaluated in a second pass over the graded arms,
      because the incumbent's inputs must come off the same producer and the
      same window as every candidate's — see :func:`_baseline_serving_metrics`.

    The refusal rows are recorded here too, on this manifest, for the same
    reason they are recorded on the produce manifest: a slot that became
    unservable between the two jobs must page from whichever one ran.
    """
    from crucible.slots.cycle import run_grade  # noqa: PLC0415 - avoids a cycle

    loaded, specs = _registered_arms(ctx, settings=settings)
    as_of = ctx.trading_day.isoformat()
    source = FeatureLayerSource(store=ctx.store, version=kwargs.get("feature_version"))
    champion = _champion_arm(ctx.store)

    preconditions: dict[str, list[ServingPrecondition]] = {}
    grades: dict[str, dict[str, Any]] = {}
    candidates: dict[str, dict[str, Any]] = {}
    windows: dict[str, str] = {}
    for spec in _in_dependency_order(specs):
        recipe = spec.recipe
        try:
            panel = _grade_panel(recipe, source=source, as_of=as_of, loaded=loaded, ctx=ctx)
        except BasePredictionsUnavailableError as exc:
            # The produce-side warm-up, seen again here. Same failure mode,
            # same reason it is per-arm rather than slot-wide (see `produce`),
            # and the same recording surface: an `unservable` row on this
            # run's manifest naming the arm and its unresolvable input. An
            # arm with no panel has no CPCV reading and no serving
            # precondition to evaluate; `run_grade` below still supplies its
            # score series, which is empty, so it cannot null another arm's
            # figure. What is NOT absorbed is anything about the feature layer
            # or the fit — those still fail the whole slot.
            ctx.record_metric(
                SlotRecipes(
                    registered=(),
                    refused=(
                        InputRefusal(
                            arm=recipe.name,
                            unresolvable=tuple(
                                r.text for r in recipe.inputs if r.kind == "predictions"
                            ),
                            reason=str(exc),
                        ),
                    ),
                ).refusal_metrics(slot=SLOT)[0]
            )
            continue
        model_grade = grade_arm(recipe, panel, as_of=as_of)
        measurable = bool(model_grade.cpcv.ics)
        row: dict[str, Any] = {
            "name": CPCV_OOS_IC_METRIC,
            "module": f"crucible.slots.{SLOT}",
            "metric_type": "gauge",
            "n_floor": 1,
            "status": "OK" if measurable else "unmeasurable",
            "status_reason": (
                f"arm {recipe.name!r}: combinatorial purged CV over "
                f"{len(model_grade.cpcv.folds)} fold(s) at a "
                f"{recipe.label_horizon_trading_days}-session label horizon; the "
                f"walk-forward series carries {model_grade.oos_n} out-of-sample "
                f"date(s). {model_grade.cpcv.reason} {model_grade.reason}"
            ).strip(),
            "source_path": arena_cycle_key(SLOT, as_of),
            "last_updated_utc": _utc_now(),
            "horizon_trading_days": recipe.label_horizon_trading_days,
        }
        if measurable:
            row["value"] = float(model_grade.cpcv.mean_ic)
            row["unit"] = "rank_ic"
        # An unmeasurable battery carries NO value and NO unit. Not a zero and
        # not a carried-forward figure: `unmeasurable` is a first-class status
        # (plan §7) and a number beside it would be read as a measurement
        # nobody made. `CPCVResult.mean_ic` refuses to be read at all in that
        # state, which is what surfaced this.
        ctx.record_metric(row)
        candidates[spec.arm_id], windows[spec.arm_id] = serving_metrics(
            predict_cross_section(model_grade.fit, panel, trading_day=as_of),
            settled=model_grade.settled,
        )
        grades[spec.arm_id] = {
            "cpcv_mean_ic": model_grade.cpcv.mean_ic if measurable else None,
            "oos_n": model_grade.oos_n,
            "status": model_grade.status,
            # The window the veto's realized inputs were read over, on the
            # artifact. A gate whose window a reader has to re-derive from the
            # code that produced it is a gate nobody can check.
            "settled_n": model_grade.settled.n_dates,
            "settled_first": model_grade.settled.dates[0] if model_grade.settled.dates else None,
            "settled_last": model_grade.settled.dates[-1] if model_grade.settled.dates else None,
        }

    # The veto runs in a SECOND pass, because the incumbent's own veto inputs
    # are produced by the same code over the same window as every candidate's
    # (`alpha-engine-config-I10680`). Policy §4: hold everything constant
    # except the thing under test — a ratio whose numerator came from this
    # cycle's walk-forward and whose denominator came from a stored artifact
    # would be comparing two constructions, not two models.
    incumbent, has_incumbent = _baseline_serving_metrics(
        ctx.store, champion=champion, candidates=candidates, as_of=as_of
    )
    for arm_id, candidate in candidates.items():
        veto = evaluate_behavioural_veto(candidate, incumbent, has_incumbent=has_incumbent)
        preconditions[arm_id] = [veto.as_precondition()]
        grades[arm_id]["veto"] = veto.status
        grades[arm_id]["veto_metrics"] = veto.metrics
        if windows[arm_id]:
            grades[arm_id]["veto_window_reason"] = windows[arm_id]

    _record_dead_slot_finding(ctx, grades, as_of=as_of)
    _record_serving_veto_window(ctx, grades, windows, as_of=as_of)

    result = run_grade(
        ctx,
        slot=SLOT,
        settings=settings,
        specs=specs,
        preconditions=preconditions,
        **{k: v for k, v in kwargs.items() if k != "feature_version"},
    )
    result["model_grades"] = grades
    result["refused"] = [
        {"arm": r.arm, "unresolvable": list(r.unresolvable)} for r in loaded.refused
    ]
    return result


def _record_dead_slot_finding(ctx: Any, grades: dict[str, dict[str, Any]], *, as_of: str) -> None:
    """Emit :data:`DEAD_SLOT_METRIC` when no M arm can ever take the pointer.

    `alpha-engine-config-I9759`. The condition is "every arm this cycle
    evaluated reads `insufficient`", which is what
    :data:`UNPRODUCED_VETO_METRICS` guarantees for as long as those three
    producers do not exist: the veto cannot pass, so the slot cannot promote,
    so the M pointer will never move on evidence no matter how many cycles
    run. Without this row the only trace is a precondition reason buried per
    arm in the cycle artifact, and the slot renders as one that held its
    pointer — which is a different fact with a different owner.

    `status: FAIL`, not `unmeasurable`: this IS a measurement, of a permanent
    condition, and it is the reading that should keep the M slot off any
    surface claiming the arena is complete. It does not fail the run — the
    cycle's verdicts, ladders and ranking are all sound and the slot is being
    graded correctly; what it cannot do is serve.

    Silent when at least one arm's veto passed or vetoed on real values: the
    slot is then alive and this row would be a false permanent finding.
    """
    if not UNPRODUCED_VETO_METRICS:
        # Every veto input has a producer (`alpha-engine-config-I10680`), so
        # an all-`insufficient` cycle is a short window, not a dead slot, and
        # `_record_serving_veto_window` is the row that says so. Emitting this
        # one anyway would name three producers that exist.
        return
    statuses = {row.get("veto") for row in grades.values()}
    if not statuses or statuses != {"insufficient"}:
        return
    missing = ", ".join(f"{name} (needs {why})" for name, why in UNPRODUCED_VETO_METRICS.items())
    ctx.record_metric(
        {
            "name": DEAD_SLOT_METRIC,
            "module": f"crucible.slots.{SLOT}",
            "metric_type": "count",
            "value": float(len(UNPRODUCED_VETO_METRICS)),
            "unit": "count",
            "n_floor": 1,
            "status": "FAIL",
            "status_reason": (
                f"every one of {len(grades)} graded arm(s) read `insufficient` on the "
                f"§5.3 behavioural veto, because nothing in this harness produces: "
                f"{missing}. An uncomputed gate is not a pass "
                "(champion-challenger-policy.md §5.1), so no M arm can take the pointer "
                "— not this cycle and not any cycle until those producers exist. The "
                "slot is graded correctly and cannot serve; `promote` will file a "
                "verdict-backed non-promotion every week until then."
            ),
            "source_path": arena_cycle_key(SLOT, as_of),
            "last_updated_utc": _utc_now(),
        }
    )


def _champion_arm(store: Any) -> str | None:
    """The M slot's champion arm id, or ``None`` when the slot has none."""
    from crucible.documents import load_store_document  # noqa: PLC0415 - avoids a cycle
    from crucible.keys import champion_key  # noqa: PLC0415 - one call site

    key = champion_key(SLOT)
    if not store.exists(key):
        return None
    champion = load_store_document(store, key).get("arm_id")
    return str(champion) if champion else None


def _baseline_serving_metrics(
    store: Any,
    *,
    champion: str | None,
    candidates: dict[str, dict[str, Any]],
    as_of: str,
) -> tuple[dict[str, Any], bool]:
    """The incumbent's veto inputs, and whether there IS an incumbent.

    Three states, deliberately distinguished (`alpha-engine-config-I10680`):

    1. **No champion pointer.** ``({}, False)``. Every dispersion rule is a
       ratio against what is being served, and nothing is; the veto records
       the family `inapplicable` and decides on the absolute rules alone. §10.1's
       null control, which `crucible-PR257` made the baseline for a first
       champion on the SCORE series, cannot stand in here: `control_null_m` is
       a selection-shaped harness control that ranks names on a noise draw and
       publishes no predicted-alpha cross-section, so it has no `alpha_stdev`
       in the candidates' units at all. Measured, not assumed — see
       `crucible.slots._controls`.
    2. **Champion graded this cycle.** Its own row out of ``candidates`` —
       same producer, same window, same fit vintage as every candidate's
       (policy §4). This is the ordinary case: the champion arm is registered
       and graded every cycle.
    3. **Champion NOT graded this cycle** (retired from the register,
       refused on inputs). ``(alpha_stdev only, True)`` off the stored
       cross-section it published for ``as_of``, or ``({}, True)`` when even
       that is absent. Either way `stdev_p_up` is uncomputable and the veto
       reads `insufficient` — which is correct and is emphatically not state 1:
       a champion we cannot measure must not read as a slot with nothing to
       compare against, because that direction lets an unmeasured arm keep
       serving while a candidate walks past a gate on absolute rules alone.
    """
    from crucible.slots.inputs import read_arm_predictions  # noqa: PLC0415 - avoids a cycle

    if champion is None:
        return {}, False
    if champion in candidates:
        return candidates[champion], True
    if not store.exists(arm_predictions_key(champion, as_of)):
        return {}, True
    return _serving_metrics(read_arm_predictions(store, arm_id=champion, trading_day=as_of)), True


def _record_serving_veto_window(
    ctx: Any,
    grades: dict[str, dict[str, Any]],
    windows: dict[str, str],
    *,
    as_of: str,
) -> None:
    """Emit :data:`SERVING_VETO_WINDOW_METRIC` — the veto's per-cycle reading.

    `alpha-engine-config-I10680`. The M slot spent every cycle of its life
    reading `insufficient`, and the reason was permanent (no producer). Now
    that the producers exist the same reading has a temporary cause — an arm
    whose settled out-of-sample window is shorter than
    :data:`SETTLED_WINDOW_DECISION_DATES` — and a cycle must say which it is
    or the fix looks like the gap.

    ``status``:

    * ``OK`` — at least one arm reached a real verdict (`pass` or `veto`). The
      value is how many did, and the slot can serve.
    * ``unmeasurable`` — every arm reads `insufficient` and every one of them
      named a short window. Nothing is broken and nothing is owed; the arms
      are accruing settled dates. No VALUE is carried, for the same reason an
      unmeasurable CPCV battery carries none.
    * ``FAIL`` — every arm reads `insufficient` and at least one did NOT name
      a short window, so something other than the window is missing. That is
      an unexplained dead slot and it pages.

    Silent on a cycle that graded no arm: an empty slot is not an unservable
    one.
    """
    if not grades:
        return
    verdicts = [row.get("veto") for row in grades.values()]
    live = [v for v in verdicts if v in ("pass", "veto")]
    row: dict[str, Any] = {
        "name": SERVING_VETO_WINDOW_METRIC,
        "module": f"crucible.slots.{SLOT}",
        "metric_type": "count",
        "n_floor": 1,
        "source_path": arena_cycle_key(SLOT, as_of),
        "last_updated_utc": _utc_now(),
    }
    if live:
        row["status"] = "OK"
        row["value"] = float(len(live))
        row["unit"] = "arms"
        row["status_reason"] = (
            f"{len(live)} of {len(grades)} graded arm(s) reached a real §5.3 behavioural-veto "
            f"verdict on {as_of} against {SETTLED_WINDOW_DECISION_DATES} settled "
            "out-of-sample decision date(s); the slot can serve a champion"
        )
    elif all(windows.get(arm) for arm in grades):
        row["status"] = "unmeasurable"
        row["status_reason"] = (
            f"every one of {len(grades)} graded arm(s) reads `insufficient` on {as_of} "
            f"because its settled out-of-sample window is shorter than "
            f"{SETTLED_WINDOW_DECISION_DATES} decision date(s): "
            + " | ".join(f"{arm}: {windows[arm]}" for arm in sorted(grades))
            + ". Every producer exists; the arms are accruing dates and no action is owed"
        )
    else:
        row["status"] = "FAIL"
        row["value"] = float(len(grades))
        row["unit"] = "arms"
        unexplained = sorted(arm for arm in grades if not windows.get(arm))
        row["status_reason"] = (
            f"every one of {len(grades)} graded arm(s) reads `insufficient` on {as_of} and "
            f"arm(s) {unexplained} did NOT name a short settled window, so a veto input is "
            "missing for a reason other than the window. An uncomputed gate is not a pass "
            "(champion-challenger-policy.md §5.1), so the M pointer cannot move until this "
            "is diagnosed"
        )
    ctx.record_metric(row)


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
