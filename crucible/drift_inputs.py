"""`drift`'s three inputs, COMPUTED from the store rather than read from
documents nothing writes.

Normative source: plan §10 component 5 (three MetricRecords per cycle);
`crucible.drift` holds the arithmetic, this module holds the reads.

**The defect this closes (measured 2026-09-05, weekly@2026-08-07 on the
first x86 replay box).** `track_c.drift_handler` read
`drift/{day}/input_{features,predictions,ic}.json` — "from the feature and
prediction artifacts track A and track B write" — and raised when they were
absent. No job in this repository has ever written them: the only reference
to `drift_input_key` outside `keys.py` was the reader. So the first arc that
reached `drift` died at it, one stage after both slots had produced and
graded. A monitor whose inputs have no producer is the exact blind spot §10
component 5 exists to remove, wearing the monitor's own name.

**What each input is, in phase 1, and what it becomes.**

* ``features`` — current day against the trailing :data:`REFERENCE_SESSIONS`
  compiled days, per catalogue column. **Two comparisons, chosen per column
  from `FeatureSpec.market_wide`** (never inferred from the live data's
  variance): a **cross-sectional** column (varies across tickers on a day)
  keeps the PSI reading — today's whole cross-section against the pooled
  trailing cross-sections, as before. A **market-wide** column (one value
  repeated across every ticker on a day, by construction —
  ``market_return_1d_log_return`` is the phase-1 example) is compared
  **along time** instead, as a z-score against the trailing window's daily
  values (one observation per session), scaled onto the same numeric range
  `psi()` reports (`_along_time_ratio`). A market-wide column fed the
  cross-sectional PSI reads a point mass (today) against 1-20 pooled point
  masses (the reference), which is BREACH by construction whatever the
  market did (`alpha-engine-config-I10071`); PSI ALSO breaks the other
  direction — a length-1 "current" sample concentrates all of its mass in
  one equal-mass bin against the reference's ~10, which reads a large PSI
  (measured ~8.3, matching the original defect's own reported value)
  regardless of where that value actually sits, so `_along_time_ratio`'s
  z-score is not a style choice, it is the only one of the two that is
  actually meaningful for a single observation. A market-wide column is
  only SCORED once its reference reaches
  :data:`ALONG_TIME_MIN_REFERENCE_SESSIONS` sessions — below that, a sample
  standard deviation is undefined at n=1 (any differing value would read an
  infinite z-score, i.e. BREACH by construction, the same defect this issue
  closes) and is mostly noise at n=2-4; a column below the floor is
  reported unscored in ``columns_awaiting_reference`` rather than scored at
  a fabricated value. ``method_by_feature`` on the returned document names
  which comparison each SCORED column used. The trailing window is the
  phase-1 proxy for "the training window"; when the M slot ships (phase 3)
  its declared training window replaces the proxy and this module's
  ``reference`` selection is the one place that changes.
* ``predictions`` — PSI of each produced arm's scored cross-section
  (`cross_section.v2`, the whole ranked population) against that same arm's
  earlier cross-sections. Reported as the WORST arm, with every arm named in
  the document. An arm with no earlier cross-section has no reference yet,
  and the document says so per arm rather than inventing one.
* ``ic`` — per horizon, the fraction of baseline rank IC lost: the latest
  settled date's IC against the mean of the earlier settled dates', per arm,
  worst arm reported. Settlement needs ``horizon_trading_days`` sessions to
  pass, so on a fresh store this is honestly `UNREPORTED` with the reason on
  the row — never a zero and never a green.

**Honest absence, not silence.** An input this store cannot yet support
produces a document whose measured fields are empty and whose
``unmeasured_reason`` says why; `crucible.drift.drift_metrics` renders that
as `UNREPORTED` and refuses a cycle in which all three are. The one raise is
an absent feature layer for the day itself: with nothing to measure, an `ok`
manifest would be the green-over-nothing this repository exists to refuse.

**Files the documents it computed.** The three `drift/{day}/input_*.json`
keys are still written, as OUTPUTS of `drift` rather than inputs it found —
so `crucible explain` walks a drift row back to the exact frames and
cross-sections it read, and a reader can re-derive the number from the
document rather than from this code.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from crucible.documents import load_store_document
from crucible.drift import BANDS, DEFAULT_BINS, feature_psi, ic_decay, psi
from crucible.features import DEFAULT_FEATURE_VERSION, FeatureSpec, read_features
from crucible.keys import experiments_prefix, features_key, features_prefix
from crucible.slots.arms import read_register
from crucible.slots.grading import RankICSkip, spearman_ic

if TYPE_CHECKING:
    from crucible.store import Store

__all__ = [
    "DRIFT_INPUT_SCHEMA_VERSION",
    "DRIFT_SLOTS",
    "REFERENCE_SESSIONS",
    "DriftInputs",
    "compute_drift_inputs",
    "feature_days",
]

DRIFT_INPUT_SCHEMA_VERSION = "drift_input.v1"

#: Trailing compiled feature days used as the phase-1 reference window. Twenty
#: sessions is one trading month — the shortest window the fleet's horizons
#: use (§4.12: 21 / 63 / 126 / 252). Not a floor: a store with fewer compiled
#: days measures against what it has and SAYS how many, because the reference
#: sample is names x sessions (~900 x N), which is a real distribution at N=1
#: and `psi` refuses an empty one outright.
REFERENCE_SESSIONS = 20

#: The slots whose arms' cross-sections and settled IC feed the prediction and
#: IC rows. Both U and R rank a cross-section; M and S join when they can be
#: dispatched (phase 3).
DRIFT_SLOTS: tuple[str, ...] = ("u", "r")

_CROSS_SECTION = "/cross_section.json"
_SETTLED = "/cross_section_settled.json"


@dataclass(frozen=True)
class DriftInputs:
    """The three documents `drift_handler` files and then reads, plus the
    store keys they were derived from (for the manifest's lineage)."""

    features: dict[str, Any]
    predictions: dict[str, Any]
    ic: dict[str, Any]
    sources: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, dict[str, Any]]:
        return {"features": self.features, "predictions": self.predictions, "ic": self.ic}


def feature_days(store: Store, version: str) -> list[dt.date]:
    """Every trading day ``version`` has a compiled feature layer for, ascending."""
    prefix = features_prefix(version)
    days: list[dt.date] = []
    for key in store.list_keys(prefix):
        leaf = key[len(prefix) :]
        if not leaf.endswith(".parquet") or "/" in leaf:
            continue
        try:
            days.append(dt.date.fromisoformat(leaf[: -len(".parquet")]))
        except ValueError:
            # Not a dated frame (the registry sits beside them as
            # `registry.json`, filtered above; anything else parquet-shaped
            # and undated is a writer defect, surfaced by name).
            raise ValueError(
                f"{key} sits under the feature layer prefix and is not a dated frame; "
                "every compiled frame is `{trading_day}.parquet`"
            ) from None
    return sorted(days)


def _finite(values: Iterable[Any]) -> list[float]:
    out: list[float] = []
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return out


def _numeric_columns(frame: Any, catalog_columns: Sequence[str]) -> list[str]:
    return [c for c in catalog_columns if c in frame.columns]


#: `psi_by_feature[name]`'s comparison, carried verbatim onto
#: `method_by_feature[name]` and into `drift.drift_metrics`'s detail string
#: for the worst feature, so a reader of the row (not just the input
#: document) can tell a genuine cross-sectional shift from an along-time one
#: (`alpha-engine-config-I10071`).
_METHOD_CROSS_SECTIONAL = (
    f"cross-sectional PSI ({DEFAULT_BINS} bins): today's whole cross-section against the "
    "pooled trailing cross-sections"
)
_METHOD_ALONG_TIME = (
    "along-time z-score ratio: today's single value against the mean and standard "
    "deviation of the trailing window's daily values, one observation per session — "
    "the market-wide comparison"
)

#: Standard deviations from the trailing window's mean that reads BREACH on
#: the along-time ratio — the conventional statistical-process-control
#: threshold (Western Electric rule 1: three sigma from the center line is
#: an out-of-control signal). NOT `psi()`: equal-mass histogram PSI needs a
#: multi-observation CURRENT sample to bin against the reference, and a
#: market-wide column's current sample is exactly one value (today's). Fed
#: to `psi()` anyway, a single observation concentrates 100% of its mass in
#: one bin against a reference spread over `DEFAULT_BINS`, which reads a
#: large PSI (measured ~8.3, matching the ORIGINAL cross-sectional defect's
#: reported value) REGARDLESS of where that value sits in the reference —
#: the same "point mass reads BREACH by construction" failure this issue
#: exists to close, reproduced one axis over. A z-score has no such failure
#: mode: it is defined for exactly this shape (one observation against a
#: sample's mean and spread).
ALONG_TIME_BREACH_SIGMA = 3.0

#: Minimum trailing daily observations before a market-wide column's
#: along-time comparison is SCORED at all. At `n=1` the sample variance is
#: zero by definition (one point has no spread), so `_along_time_ratio`
#: would read ANY differing current value as an infinite z-score and
#: BREACH — on the first replay day after a fresh feature layer, or the day
#: after any gap, whatever the market did. That is the exact
#: desensitisation `alpha-engine-config-I10071` exists to close, reproduced
#: on the reference-size axis instead of the ticker-count axis. At `n=2..4`
#: a sample standard deviation is itself mostly sampling noise, not a
#: measurement of the column's real spread. Five is one trading week
#: (`AGENTS.md` §3: "a trading week is 5 trading days") — the smallest
#: window this repo already treats as a real sample rather than a handful
#: of points, so a market-wide column below it is reported unscored with a
#: reason, never scored at a fabricated 0.0 or a manufactured breach.
ALONG_TIME_MIN_REFERENCE_SESSIONS = 5


def _daily_representative(frame: Any, column: str) -> float | None:
    """The single value a market-wide column carries on this trading day.

    A market-wide column is identical across the whole cross-section by
    DECLARED construction (`FeatureSpec.market_wide`), so the mean of its
    finite values on the day is that shared value — and is robust to the
    rare row whose other columns are null while this one still carries the
    day's value. `None` when the column has no finite value at all that day.
    """
    values = _finite(frame[column].tolist())
    if not values:
        return None
    return sum(values) / len(values)


def _along_time_ratio(reference: Sequence[float], current: float) -> float:
    """A market-wide column's drift, on the SAME numeric scale `psi()`
    reports for a cross-sectional column, so both can share one row's
    "worst of N" without unit confusion.

    Built from a z-score of ``current`` against ``reference``'s mean and
    sample standard deviation, scaled so exactly
    :data:`ALONG_TIME_BREACH_SIGMA` reads the existing
    ``feature_psi_max_ratio`` band's `breach` value — a market-wide column
    at the three-sigma control limit and a cross-sectional column at a
    "different population" PSI both read the same number, by declared
    convention, not by claiming the two arithmetics are the same thing.

    Raises rather than scoring below :data:`ALONG_TIME_MIN_REFERENCE_SESSIONS`:
    the caller is the one place that decides whether a column is scored at
    all (`_features_input` routes a too-short reference to
    ``columns_awaiting_reference`` instead of calling this), so reaching
    this function with too few points is a caller defect, not a value this
    function should paper over with an invented number.
    """
    n = len(reference)
    if n < ALONG_TIME_MIN_REFERENCE_SESSIONS:
        raise ValueError(
            f"{n} reference session(s), fewer than "
            f"ALONG_TIME_MIN_REFERENCE_SESSIONS={ALONG_TIME_MIN_REFERENCE_SESSIONS}; the "
            "caller must route this column to columns_awaiting_reference instead of "
            "scoring it — a sample of this size has no real standard deviation to "
            "measure a z-score against"
        )
    mean = sum(reference) / n
    # n >= ALONG_TIME_MIN_REFERENCE_SESSIONS (checked above) is always > 1,
    # so the sample-variance denominator never divides by zero.
    variance = sum((v - mean) ** 2 for v in reference) / (n - 1)
    std = math.sqrt(variance)
    breach = BANDS["feature_psi_max_ratio"].breach
    if std == 0.0:
        # A trailing window with zero variance (every prior day carried the
        # identical value) has no scale to measure against. A current value
        # equal to that constant is DEFINED as no drift; the market-wide
        # column's whole point is that its value CAN be flat across a quiet
        # stretch, so this is a legitimate reading, not a producer defect.
        return 0.0 if current == mean else breach
    z = abs(current - mean) / std
    # Not capped at 1.0: `psi()` is unbounded too (a total population
    # replacement reads well past 1.0), and capping here would make an
    # order-of-magnitude regime jump read identically to a marginal breach.
    return z / ALONG_TIME_BREACH_SIGMA * breach


def _features_input(
    store: Store,
    trading_day: dt.date,
    *,
    version: str,
    reference_sessions: int,
    catalog: Sequence[FeatureSpec],
    sources: list[str],
) -> dict[str, Any]:
    catalog_columns = tuple(spec.name for spec in catalog)
    market_wide_names = frozenset(spec.name for spec in catalog if spec.market_wide)
    current_key = features_key(version, trading_day.isoformat())
    if not store.exists(current_key):
        raise FileNotFoundError(
            f"the feature layer for {trading_day.isoformat()} is absent at {current_key}; "
            "drift has nothing to measure. A drift row over no feature layer is the "
            "green-over-nothing this job exists to refuse — compile the day first "
            "(`crucible data.daily --date ...`)."
        )
    current = read_features(store.get_bytes(current_key))
    sources.append(current_key)
    columns = _numeric_columns(current, catalog_columns)
    prior = [d for d in feature_days(store, version) if d < trading_day][-reference_sessions:]
    document: dict[str, Any] = {
        "schema_version": DRIFT_INPUT_SCHEMA_VERSION,
        "trading_day": trading_day.isoformat(),
        "current_session": trading_day.isoformat(),
        "reference_sessions": [d.isoformat() for d in prior],
        "reference_sessions_requested": reference_sessions,
        "feature_version": version,
        "method": (
            "per catalogue column, one of two comparisons chosen from the column's "
            "DECLARED `FeatureSpec.market_wide` (never inferred from variance): a "
            f"cross-sectional column reads '{_METHOD_CROSS_SECTIONAL}'; a market-wide "
            f"column (identical across every ticker on a day) reads '{_METHOD_ALONG_TIME}', "
            f"once its trailing reference reaches {ALONG_TIME_MIN_REFERENCE_SESSIONS} "
            "sessions — see columns_awaiting_reference for one that has not. See "
            "method_by_feature for the per-column choice actually used. The trailing "
            "window is the phase-1 proxy for the training window; the M slot's declared "
            "training window replaces it in phase 3."
        ),
        "psi_by_feature": {},
        "method_by_feature": {},
    }
    if not prior:
        document["unmeasured_reason"] = (
            f"no compiled feature day precedes {trading_day.isoformat()} under "
            f"{features_prefix(version)}; a first day has nothing to drift from"
        )
        return document
    cross_sectional_columns = [c for c in columns if c not in market_wide_names]
    market_wide_columns = [c for c in columns if c in market_wide_names]
    cs_training: dict[str, list[float]] = {c: [] for c in cross_sectional_columns}
    mw_training: dict[str, list[float]] = {c: [] for c in market_wide_columns}
    for day in prior:
        key = features_key(version, day.isoformat())
        frame = read_features(store.get_bytes(key))
        sources.append(key)
        for c in cross_sectional_columns:
            if c in frame.columns:
                cs_training[c].extend(_finite(frame[c].tolist()))
        for c in market_wide_columns:
            if c in frame.columns:
                value = _daily_representative(frame, c)
                if value is not None:
                    mw_training[c].append(value)
    # A column with no finite reference value anywhere in the window is a
    # NEW feature, not a vanished one: it is reported as not comparable
    # rather than fed to `psi`, which would (correctly) refuse the empty
    # reference and take the whole row down over a column that only just
    # appeared.
    cs_comparable = {c: v for c, v in cs_training.items() if v}
    mw_comparable = {c: v for c, v in mw_training.items() if v}
    not_comparable = sorted(
        (set(cs_training) - set(cs_comparable)) | (set(mw_training) - set(mw_comparable))
    )
    # A market-wide column with SOME reference but fewer than
    # ALONG_TIME_MIN_REFERENCE_SESSIONS sessions of it is not "not
    # comparable" (that means zero reference, a brand-new column) — it has
    # a reference, just too short a one to measure a real standard
    # deviation against. Reported with its own honest-absence reason
    # (mirroring how the predictions input reports an arm with no earlier
    # cross-section) rather than scored at n=1's fabricated zero-variance
    # breach.
    mw_scoreable = {
        c: v for c, v in mw_comparable.items() if len(v) >= ALONG_TIME_MIN_REFERENCE_SESSIONS
    }
    columns_awaiting_reference = {
        c: f"no reference yet: {len(v)} of {ALONG_TIME_MIN_REFERENCE_SESSIONS} sessions"
        for c, v in mw_comparable.items()
        if c not in mw_scoreable
    }
    cs_live = {c: _finite(current[c].tolist()) for c in cs_comparable}
    mw_live: dict[str, list[float]] = {}
    for c in mw_scoreable:
        value = _daily_representative(current, c)
        mw_live[c] = [] if value is None else [value]
    empty_live = sorted(
        [c for c, v in cs_live.items() if not v] + [c for c, v in mw_live.items() if not v]
    )
    if empty_live:
        raise ValueError(
            f"feature(s) {empty_live} have no finite value on {trading_day.isoformat()} "
            "while the reference window carries them: a column that emptied is a "
            "lineage failure, reported as one rather than as a stable distribution"
        )
    psi_by_feature = feature_psi(cs_comparable, cs_live)
    method_by_feature = {c: _METHOD_CROSS_SECTIONAL for c in psi_by_feature}
    for c in sorted(mw_scoreable):
        psi_by_feature[c] = _along_time_ratio(mw_scoreable[c], mw_live[c][0])
        method_by_feature[c] = _METHOD_ALONG_TIME
    document["psi_by_feature"] = psi_by_feature
    document["method_by_feature"] = method_by_feature
    document["columns_not_comparable"] = not_comparable
    document["columns_awaiting_reference"] = columns_awaiting_reference
    document["reference_rows_by_feature"] = {
        **{c: len(v) for c, v in cs_comparable.items()},
        **{c: len(v) for c, v in mw_scoreable.items()},
    }
    if not psi_by_feature:
        # Every catalogue column present today is either brand new
        # (`columns_not_comparable`) or a market-wide column whose reference
        # has not reached ALONG_TIME_MIN_REFERENCE_SESSIONS yet
        # (`columns_awaiting_reference`) — the same honest-absence shape as
        # an empty `prior` above, reached one step later.
        document["unmeasured_reason"] = (
            "no catalogue column could be scored this cycle: "
            f"not comparable (no reference at all): {not_comparable or 'none'}; "
            f"awaiting a longer reference: {columns_awaiting_reference or 'none'}"
        )
    return document


def _arms(store: Store, slots: Sequence[str]) -> list[str]:
    arms: list[str] = []
    for slot in slots:
        arms.extend(sorted(read_register(store, slot).all_arms()))
    return arms


def _dated_documents(store: Store, arm_id: str, suffix: str) -> dict[dt.date, str]:
    prefix = experiments_prefix(arm_id)
    out: dict[dt.date, str] = {}
    for key in store.list_keys(prefix):
        if not key.endswith(suffix):
            continue
        day = key[len(prefix) :].split("/", 1)[0]
        out[dt.date.fromisoformat(day)] = key
    return out


def _scores(document: dict[str, Any]) -> list[float]:
    return _finite(row["score"] for row in document["ranks"])


def _predictions_input(
    store: Store,
    trading_day: dt.date,
    *,
    slots: Sequence[str],
    reference_sessions: int,
    sources: list[str],
) -> dict[str, Any]:
    by_arm: dict[str, float] = {}
    reference_dates: dict[str, list[str]] = {}
    without_current: list[str] = []
    without_reference: list[str] = []
    for arm_id in _arms(store, slots):
        dated = _dated_documents(store, arm_id, _CROSS_SECTION)
        if trading_day not in dated:
            without_current.append(arm_id)
            continue
        prior = sorted(d for d in dated if d < trading_day)[-reference_sessions:]
        if not prior:
            without_reference.append(arm_id)
            continue
        current_doc = load_store_document(store, dated[trading_day])
        sources.append(dated[trading_day])
        reference: list[float] = []
        for day in prior:
            reference.extend(_scores(load_store_document(store, dated[day])))
            sources.append(dated[day])
        by_arm[arm_id] = psi(reference, _scores(current_doc))
        reference_dates[arm_id] = [d.isoformat() for d in prior]
    worst_arm, worst = max(by_arm.items(), key=lambda kv: kv[1]) if by_arm else (None, None)
    document: dict[str, Any] = {
        "schema_version": DRIFT_INPUT_SCHEMA_VERSION,
        "trading_day": trading_day.isoformat(),
        "method": (
            "equal-mass PSI of each arm's scored cross-section (cross_section.v2, the whole "
            "ranked population) against the same arm's pooled earlier cross-sections; the "
            "row reports the worst arm"
        ),
        "psi": worst,
        "worst_arm": worst_arm,
        "psi_by_arm": by_arm,
        "reference_dates_by_arm": reference_dates,
        "arms_without_a_cross_section_today": without_current,
        "arms_without_an_earlier_cross_section": without_reference,
    }
    if worst is None:
        document["unmeasured_reason"] = (
            "no arm has both a cross-section on "
            f"{trading_day.isoformat()} and an earlier one to compare it with"
            + (f"; arms with only today's: {without_reference}" if without_reference else "")
        )
    return document


def _ic_series(
    store: Store, arm_id: str, sources: list[str]
) -> dict[int, list[tuple[dt.date, float]]]:
    """Per horizon, the (settled date, rank IC) pairs for ``arm_id``, ascending."""
    series: dict[int, list[tuple[dt.date, float]]] = {}
    for day, key in sorted(_dated_documents(store, arm_id, _SETTLED).items()):
        document = load_store_document(store, key)
        score_by_ticker = {row["ticker"]: row["score"] for row in document["ranks"]}
        return_by_ticker = {
            row["ticker"]: row["realized_forward_return_ratio"]
            for row in document["ranks"]
            if row["realized_forward_return_ratio"] is not None
        }
        result = spearman_ic(score_by_ticker, return_by_ticker)
        if isinstance(result, RankICSkip):
            continue
        ic, _n = result
        sources.append(key)
        series.setdefault(int(document["horizon_trading_days"]), []).append((day, ic))
    return series


def _ic_input(
    store: Store,
    trading_day: dt.date,
    *,
    slots: Sequence[str],
    sources: list[str],
) -> dict[str, Any]:
    decay_by_arm: dict[str, dict[int, float]] = {}
    current_by_arm: dict[str, dict[int, float]] = {}
    baseline_by_arm: dict[str, dict[int, float]] = {}
    thin: dict[str, dict[int, int]] = {}
    for arm_id in _arms(store, slots):
        series = _ic_series(store, arm_id, sources)
        current: dict[int, float] = {}
        baseline: dict[int, float] = {}
        for horizon, points in series.items():
            settled = [(d, ic) for d, ic in points if d <= trading_day]
            if len(settled) < 2:
                thin.setdefault(arm_id, {})[horizon] = len(settled)
                continue
            *earlier, latest = settled
            base = sum(ic for _d, ic in earlier) / len(earlier)
            if base == 0:
                # `ic_decay` refuses a zero baseline; a baseline of no edge
                # is not a baseline and the horizon is reported as thin
                # rather than fed to arithmetic that would raise mid-row.
                thin.setdefault(arm_id, {})[horizon] = len(settled)
                continue
            current[horizon] = latest[1]
            baseline[horizon] = base
        if current:
            decay_by_arm[arm_id] = ic_decay(current, baseline)
            current_by_arm[arm_id] = current
            baseline_by_arm[arm_id] = baseline
    worst: dict[int, float] = {}
    worst_arm_by_horizon: dict[int, str] = {}
    for arm_id, decay in decay_by_arm.items():
        for horizon, value in decay.items():
            if horizon not in worst or value > worst[horizon]:
                worst[horizon] = value
                worst_arm_by_horizon[horizon] = arm_id
    document: dict[str, Any] = {
        "schema_version": DRIFT_INPUT_SCHEMA_VERSION,
        "trading_day": trading_day.isoformat(),
        "method": (
            "per arm and horizon, fraction of baseline rank IC lost: the latest settled "
            "date's Spearman IC against the mean of the earlier settled dates'; the row "
            "reports the worst arm per horizon. Settlement needs horizon_trading_days "
            "sessions to pass."
        ),
        "decay_by_horizon": {str(h): v for h, v in sorted(worst.items())},
        "worst_arm_by_horizon": {str(h): a for h, a in sorted(worst_arm_by_horizon.items())},
        "decay_by_arm": {a: {str(h): v for h, v in d.items()} for a, d in decay_by_arm.items()},
        "current_ic_by_arm": {
            a: {str(h): v for h, v in d.items()} for a, d in current_by_arm.items()
        },
        "baseline_ic_by_arm": {
            a: {str(h): v for h, v in d.items()} for a, d in baseline_by_arm.items()
        },
        "settled_dates_too_few_by_arm": {
            a: {str(h): n for h, n in d.items()} for a, d in thin.items()
        },
    }
    if not worst:
        document["unmeasured_reason"] = (
            "no arm has two or more settled cross-sections at any horizon on or before "
            f"{trading_day.isoformat()}; IC decay needs a baseline and a current reading, "
            "and settlement needs horizon_trading_days sessions to pass"
        )
    return document


def compute_drift_inputs(
    store: Store,
    trading_day: dt.date,
    *,
    feature_version: str = DEFAULT_FEATURE_VERSION,
    slots: Sequence[str] = DRIFT_SLOTS,
    reference_sessions: int = REFERENCE_SESSIONS,
    catalog: Sequence[FeatureSpec] | None = None,
) -> DriftInputs:
    """The three drift input documents for ``trading_day``, from the store.

    ``catalog`` carries each column's DECLARED `market_wide` property
    (`alpha-engine-config-I10071`), which is why this takes the
    `FeatureSpec` sequence rather than bare column names as it did before —
    a name alone cannot say which of the two drift comparisons a column
    wants.
    """
    if catalog is None:
        from crucible.features import CATALOG  # noqa: PLC0415 - one call site

        catalog = CATALOG
    sources: list[str] = []
    features = _features_input(
        store,
        trading_day,
        version=feature_version,
        reference_sessions=reference_sessions,
        catalog=catalog,
        sources=sources,
    )
    predictions = _predictions_input(
        store, trading_day, slots=slots, reference_sessions=reference_sessions, sources=sources
    )
    ic = _ic_input(store, trading_day, slots=slots, sources=sources)
    return DriftInputs(
        features=features, predictions=predictions, ic=ic, sources=tuple(dict.fromkeys(sources))
    )
