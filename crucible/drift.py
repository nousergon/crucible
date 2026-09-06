"""Three drift metrics, with control bands, as rendered rows — never pages.

Normative source: plan §10 component 5.

    feature_psi_max_ratio        features now vs the training window
    prediction_psi_ratio         the prediction distribution now vs then
    ic_decay_ratio               IC by horizon, now vs its own baseline

**Why this exists.** An arm can be perfectly healthy and its inputs silently
shifted underneath it. The audit's "the system stopped thinking on 07-19 while
every detector was green" is exactly an input-drift blind spot: nothing was
failing, so nothing fired, and the thing that had changed was what the model
was being shown.

**Rows, not pages.** §4.6 has exactly two page conditions and drift is neither.
A drifting feature is not an outage; it is a number that has moved and wants
looking at on the console. Making it a third page condition is how a fleet
gets to 185 alert rules and misses three Saturdays inside them.

**Bands, not thresholds.** Each metric carries a control band, and the row's
status is where the value sits in it. `_ratio` suffixes everywhere: PSI is
dimensionless and the fleet's feature-store contract requires an explicit
units suffix on every column name, because `avg_volume_20d` was emitted as a
normalized ratio and consumed as raw shares for months.

**No numpy.** Three metrics over a few thousand rows do not need an array
library, and a pure-Python implementation is one less thing installed on a
Lambda and one less version to pin (principle 6).
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "BANDS",
    "Band",
    "DEFAULT_BINS",
    "PSI_EPSILON",
    "Band",
    "drift_metrics",
    "feature_psi",
    "ic_by_horizon",
    "ic_decay",
    "prediction_drift",
    "psi",
]

#: Ten equal-mass bins cut on the REFERENCE sample. Equal-mass rather than
#: equal-width because a feature with a long tail puts every observation in
#: one equal-width bin and reports a PSI of zero forever — a metric that
#: cannot move is worse than no metric.
DEFAULT_BINS = 10

#: Floor applied to an empty bin's proportion. Without it a single bin that
#: emptied sends PSI to infinity and the row renders as a catastrophe every
#: time a rare value stops appearing. The industry convention is 1e-4 or
#: 1/(10*N); the fixed floor is chosen so the number is comparable across
#: cycles of different N.
PSI_EPSILON = 1e-4


@dataclass(frozen=True)
class Band:
    """A control band: below ``watch`` is OK, above ``breach`` is BREACH.

    Two edges rather than one threshold, because "it moved a bit" and "it is
    a different distribution" are different operator actions and collapsing
    them produces a row that is either always green or always red.
    """

    watch: float
    breach: float
    rationale: str

    def status(self, value: float | None) -> str:
        """`OK` / `WATCH` / `BREACH`, or `UNREPORTED` for no value.

        `UNREPORTED` and never a silent green: a component emitting nothing
        is unobserved, not healthy (principle 7), and this is the one place a
        drift row could quietly claim health it has no evidence for.
        """
        if value is None or math.isnan(value):
            return "UNREPORTED"
        if value >= self.breach:
            return "BREACH"
        if value >= self.watch:
            return "WATCH"
        return "OK"


#: The declared bands. PSI's 0.10 / 0.25 pair is the long-standing credit-risk
#: convention and is used here unchanged rather than tuned to whatever the
#: first cycle happened to produce — a band fitted to current data is a band
#: that can never be breached by current data. IC decay's band is expressed as
#: a RATIO of current IC to baseline IC, so a halving reads as 0.5 whatever
#: the absolute IC was.
BANDS: dict[str, Band] = {
    "feature_psi_max_ratio": Band(
        watch=0.10,
        breach=0.25,
        rationale=(
            "The conventional PSI reading: <0.10 no material shift, 0.10-0.25 a shift "
            "worth looking at, >0.25 a different population. Adopted unchanged rather "
            "than fitted to observed data — a band fitted to current data can never be "
            "breached by current data."
        ),
    ),
    "prediction_psi_ratio": Band(
        watch=0.10,
        breach=0.25,
        rationale=(
            "Same scale as the feature band on purpose. A prediction distribution that "
            "has moved while the features have not is a model-side change; the two rows "
            "are only comparable if their bands are."
        ),
    ),
    "ic_decay_ratio": Band(
        watch=0.50,
        breach=0.75,
        rationale=(
            "Fraction of baseline IC LOST. 0.50 means half the measured edge is gone at "
            "that horizon. Expressed as a loss ratio so the band does not depend on the "
            "absolute IC, which differs by an order of magnitude between slots."
        ),
    ),
}


def _equal_mass_edges(reference: Sequence[float], bins: int) -> list[float]:
    """Bin edges at equal quantiles of the reference sample."""
    ordered = sorted(reference)
    n = len(ordered)
    edges: list[float] = []
    for i in range(1, bins):
        idx = min(n - 1, max(0, int(round(i * n / bins)) - 1))
        edges.append(ordered[idx])
    # Duplicate edges collapse silently into empty bins, which then hit the
    # epsilon floor and inflate PSI. De-duplicating keeps the bin count honest
    # for a low-cardinality feature rather than manufacturing drift from it.
    deduped: list[float] = []
    for e in edges:
        if not deduped or e > deduped[-1]:
            deduped.append(e)
    return deduped


def _proportions(sample: Sequence[float], edges: Sequence[float]) -> list[float]:
    counts = [0] * (len(edges) + 1)
    for value in sample:
        placed = len(edges)
        for i, edge in enumerate(edges):
            if value <= edge:
                placed = i
                break
        counts[placed] += 1
    total = len(sample)
    return [max(c / total, PSI_EPSILON) for c in counts]


def psi(
    reference: Sequence[float],
    current: Sequence[float],
    *,
    bins: int = DEFAULT_BINS,
) -> float:
    """Population Stability Index of ``current`` against ``reference``.

    Raises on an empty sample rather than returning 0.0. A PSI of zero for a
    feature that produced nothing is the exact shape of a monitor reporting
    health from the absence of data, and it is the reason this whole module
    exists.
    """
    if not reference:
        raise ValueError("PSI needs a reference sample; an empty one measures nothing")
    if not current:
        raise ValueError(
            "PSI needs a current sample. A feature that produced no rows this cycle is "
            "a lineage failure to be reported as one, not a stable distribution."
        )
    edges = _equal_mass_edges(reference, bins)
    ref_p = _proportions(reference, edges)
    cur_p = _proportions(current, edges)
    return sum((c - r) * math.log(c / r) for r, c in zip(ref_p, cur_p, strict=True))


def feature_psi(
    training: dict[str, Sequence[float]],
    live: dict[str, Sequence[float]],
    *,
    bins: int = DEFAULT_BINS,
) -> dict[str, float]:
    """PSI per feature, training window vs live.

    A feature present in training and absent live is a **KeyError**, not a
    skipped row. Silently dropping it would let a feature disappear from the
    pipeline entirely while the drift page stayed green — the same class as
    the four detectors that were dark rather than red.
    """
    out: dict[str, float] = {}
    for name in sorted(training):
        if name not in live:
            raise KeyError(
                f"feature {name!r} is in the training window and absent from the live "
                "sample. A feature that vanished is the largest drift there is, and "
                "skipping it would render the row green."
            )
        out[name] = psi(training[name], live[name], bins=bins)
    return out


def prediction_drift(
    reference: Sequence[float],
    current: Sequence[float],
    *,
    bins: int = DEFAULT_BINS,
) -> float:
    """PSI of the prediction distribution. Same scale as the feature row."""
    return psi(reference, current, bins=bins)


def ic_by_horizon(
    predictions: Sequence[float],
    forward_returns: dict[int, Sequence[float]],
) -> dict[int, float]:
    """Spearman rank IC per forward horizon, keyed by TRADING days.

    Rank rather than Pearson: an IC dominated by two outliers is the number
    that makes a dead signal look alive. The keys are integer trading-day
    counts — 21 / 63 / 126 / 252 — because §4.12 forbids a calendar horizon
    anywhere a decision reads it.
    """
    out: dict[int, float] = {}
    for horizon in sorted(forward_returns):
        if horizon <= 0:
            raise ValueError(f"horizon {horizon} is not a positive trading-day count")
        returns = forward_returns[horizon]
        if len(returns) != len(predictions):
            raise ValueError(
                f"horizon {horizon}: {len(returns)} forward returns against "
                f"{len(predictions)} predictions. A truncated pairing silently drops "
                "the tail of the cross-section and biases the IC."
            )
        out[horizon] = _spearman(predictions, returns)
    return out


def _rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def _spearman(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) < 2:
        raise ValueError("a rank correlation over fewer than two observations is not one")
    ra, rb = _rank(a), _rank(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb, strict=True))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    if da == 0 or db == 0:
        # A constant series has no rank correlation with anything. Returning
        # 0.0 would read as "no edge"; this is "the question is malformed",
        # and the two want different operator actions.
        raise ValueError(
            "one of the series is constant, so its ranks have zero variance and no "
            "correlation exists. A constant prediction vector is a producer defect."
        )
    return num / (da * db)


def ic_decay(current: dict[int, float], baseline: dict[int, float]) -> dict[int, float]:
    """Fraction of baseline IC lost, per horizon.

    ``1 - current/baseline``, clamped at 0 below — an IC that IMPROVED is not
    decay, and reporting a negative decay would put an improvement in the same
    column as a degradation with only its sign to tell them apart. A sign flip
    (current and baseline of opposite sign) reports a loss greater than 1.0,
    which is correct: the signal is not weaker, it is backwards.
    """
    out: dict[int, float] = {}
    for horizon in sorted(baseline):
        if horizon not in current:
            raise KeyError(
                f"horizon {horizon} has a baseline IC and no current one. A horizon "
                "that stopped being measured renders green if it is skipped."
            )
        base = baseline[horizon]
        if base == 0:
            raise ValueError(
                f"horizon {horizon} has a baseline IC of exactly zero, so decay from it "
                "is undefined. A baseline of no edge is not a baseline."
            )
        out[horizon] = max(0.0, 1.0 - (current[horizon] / base))
    return out


def _record(
    name: str,
    value: float | None,
    *,
    unit: str,
    now: dt.datetime,
    source_path: str,
    detail: str,
    horizon: int | None = None,
) -> dict[str, Any]:
    band = BANDS[name]
    status = band.status(value)
    record: dict[str, Any] = {
        "name": name,
        "module": "crucible.drift",
        "metric_type": "drift",
        "value": None if value is None else float(value),
        "unit": unit,
        "n_floor": 0,
        "status": status,
        "status_reason": (
            f"{detail} Band: OK < {band.watch:g} <= WATCH < {band.breach:g} <= BREACH. "
            f"{band.rationale}"
        ),
        "source_path": source_path,
        "last_updated_utc": now.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "baseline": {"watch": band.watch, "breach": band.breach},
    }
    if horizon is not None:
        record["horizon_trading_days"] = horizon
    return record


def drift_metrics(
    *,
    trading_day: dt.date,
    feature_psi_by_name: dict[str, float],
    prediction_psi: float | None,
    ic_decay_by_horizon: dict[int, float],
    now: dt.datetime | None = None,
    unmeasured_reasons: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """The three MetricRecords, one per cycle. §10 component 5.

    ``unmeasured_reasons`` — per input (`features` / `predictions` / `ic`),
    the producer's own statement of WHY a value is absent, carried onto the
    `UNREPORTED` row's `status_reason` so the console shows "no settled
    horizon yet" rather than the generic "nothing was measured".

    Three and exactly three: the plan names them, and a drift module that
    grows a row per feature is the 185-rule fleet again with a different
    noun. The per-feature numbers are carried in the row's `detail`, so the
    worst offender is named without every feature getting its own tile.

    ``prediction_psi`` is ``float | None`` rather than a bare ``float``: a
    caller whose predictions input carries a null `psi` (structurally
    present, nothing measured) is a legitimate absence to represent, not a
    type error to mask.

    Raises :class:`ValueError` when every one of the three records would
    render `UNREPORTED` — see the check at the end of this function.
    """
    moment = now or dt.datetime.now(dt.UTC)
    day = trading_day.isoformat()
    reasons = unmeasured_reasons or {}

    worst_feature, worst_value = (
        max(feature_psi_by_name.items(), key=lambda kv: kv[1])
        if feature_psi_by_name
        else (None, None)
    )
    worst_horizon, worst_decay = (
        max(ic_decay_by_horizon.items(), key=lambda kv: kv[1])
        if ic_decay_by_horizon
        else (None, None)
    )

    records = [
        _record(
            "feature_psi_max_ratio",
            worst_value,
            unit="psi",
            now=moment,
            # Human-readable diagnostic text, not a store key — no store call
            # reads this string (alpha-engine-config-I9852). Not calling
            # `crucible.keys.drift_input_key` here: that function's actual
            # write shape is `drift/{day}/input_{name}.json`
            # (`input_features.json`), one segment longer than the friendlier
            # `features.json` this MetricRecord shows a human on the console
            # (alpha-engine-config-I9875).
            source_path=f"drift/{day}/features.json",
            detail=(
                f"Worst of {len(feature_psi_by_name)} feature(s): "
                f"{worst_feature} at {worst_value:.4f}."
                if worst_feature is not None
                else reasons.get("features", "No features were compared, so nothing was measured.")
            ),
        ),
        _record(
            "prediction_psi_ratio",
            prediction_psi,
            unit="psi",
            now=moment,
            # Same reasoning as feature_psi_max_ratio's source_path above:
            # display text, not a store key; drift_input_key's real shape is
            # `input_predictions.json` (alpha-engine-config-I9875).
            source_path=f"drift/{day}/predictions.json",
            detail=(
                f"Prediction distribution against the training window: {prediction_psi:.4f}."
                if prediction_psi is not None and not math.isnan(prediction_psi)
                else reasons.get("predictions", "The prediction distribution was not measured.")
            ),
        ),
        _record(
            "ic_decay_ratio",
            worst_decay,
            unit="ratio",
            now=moment,
            # Same reasoning as feature_psi_max_ratio's source_path above:
            # display text, not a store key; drift_input_key's real shape is
            # `input_ic.json` (alpha-engine-config-I9875).
            source_path=f"drift/{day}/ic.json",
            detail=(
                f"Worst horizon: {worst_horizon} trading days, "
                f"{worst_decay:.4f} of baseline IC lost."
                if worst_horizon is not None
                else reasons.get("ic", "No horizons were compared, so nothing was measured.")
            ),
            horizon=worst_horizon,
        ),
    ]

    # alpha-engine-config-I9757 (C5): inputs can be STRUCTURALLY present and
    # semantically empty — a features file with `psi_by_feature: {}`, an IC
    # file with `decay_by_horizon: {}`, a predictions file whose `psi` is
    # null. None of those trip the caller's file-existence check, so a run
    # over them would otherwise return three `UNREPORTED` rows and let the
    # caller write `status: ok`. That is the exact blind spot the audit
    # named — "the system stopped thinking on 07-19 while every detector
    # was green" — reproduced inside the one module built to close it.
    # Raising here, at the point every record's status is already known,
    # makes the run fail the same way the missing-artifact case already
    # does, rather than requiring every future caller to re-derive this
    # check for itself (principle 6: one place carries the rule).
    if all(r["status"] == "UNREPORTED" for r in records):
        raise ValueError(
            f"drift {day}: all three metrics carry no value — zero features were "
            "compared, zero horizons were compared, and the prediction distribution "
            "was not measured. Inputs were structurally present but empty, which is "
            "the same failure as absent inputs wearing a different shape, and this "
            "run must not exit `ok` over it."
        )
    return records
