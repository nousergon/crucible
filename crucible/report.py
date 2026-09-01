"""`crucible report` — the five-row attribution table, and nothing else.

Normative source: plan §2 row 5 ("pinpoint underperformance"), §4.5.

    report/{trading_day}/attribution.json

Five rows, always five, one per layer the verdict passes through: data
freshness/coverage, the R slot's signal, the M slot's prediction, the S slot's
portfolio alpha, and execution shortfall. Each is a `MetricRecord`
(`krepis.metrics.MetricRecord`) carrying ``value``, a confidence interval,
``n_samples`` against a declared floor, a ``baseline`` and a ``status``. That
table IS the report card — the 32-stage evaluator is not carried.

**A row is never absent and never zero-by-default.** Where the producing layer
has not run, the row carries one of `MetricRecord`'s declared not-measured
states — ``N/A-NOT-RUN``, ``N/A-MISSING-INPUT``, ``N/A-LOW-N``,
``N/A-NOT-IMPL`` — with a ``status_reason`` naming the artifact that is
missing. The vocabulary is `krepis.metrics`'s own; nothing is invented here.
Rendering an unmeasured layer as a number would be the failure principle 7
names: *no data* is never rendered as green, and a five-row table with three
rows in it is a report card that has quietly stopped grading three layers.

**The reducer computes nothing a producer already computes.** The coverage row
is reduced from the ``universe_coverage_ratio`` MetricRecords that `data.daily`
already emits; the slot rows are reduced from the verdict artifacts the arena
already writes. A second implementation of a number is a second answer to the
same question.

**Units are declared, and they are not the plan's shorthand.** §4.5 calls rows
two and three "signal IC" and "prediction IC". What the durable artifacts
identify is the champion's realized excess return against the population it
drew from — `shadow.v1` records the selected names, not the ranked
cross-section, so a rank correlation is not reconstructible from what is
stored. The rows are therefore named and united as what they are, and each
carries ``plan_row`` naming the §4.5 row it answers, so the mapping is
machine-checkable rather than a matter of reading two documents side by side.
Emitting an excess return under the name of a correlation is the
``avg_volume_20d`` defect with a report card around it.
"""

from __future__ import annotations

import datetime as dt
import json
import random
from dataclasses import dataclass
from typing import Any

from krepis.metrics import MetricRecord, derive_status

from crucible.calendar import previous_trading_day
from crucible.data.daily import COVERAGE_FLOOR_RATIO
from crucible.keys import arm_key_segment, champion_key
from crucible.manifest import manifest_key
from crucible.store import Store

__all__ = [
    "ATTRIBUTION_SCHEMA_VERSION",
    "ROWS",
    "REPORT_WINDOW_TRADING_DAYS",
    "RowSpec",
    "attribution_key",
    "build_attribution",
    "rows_complete_metric",
]

ATTRIBUTION_SCHEMA_VERSION = "attribution.v1"

#: The freshness window, in TRADING days (§4.12). One trading week: the
#: attribution table is written weekly, so a window shorter than a week would
#: leave sessions no report card ever looked at, and a longer one would let a
#: dead day age out of view before the next report noticed it.
REPORT_WINDOW_TRADING_DAYS = 5

#: Settled decision dates a slot row needs before its value is read as a
#: measurement rather than as noise. Below half of it the row is
#: ``N/A-LOW-N``; between half and the floor `derive_status` returns WATCH,
#: which is the honest reading of "a number with a CI wider than the effect".
SLOT_N_FLOOR = 6

#: Bootstrap resamples behind every confidence interval here, and the seed
#: that makes them reproducible. A CI that moved between two runs over the
#: same artifacts would make every replay diff unattributable.
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260901


@dataclass(frozen=True)
class RowSpec:
    """One row of the §4.5 table: what it answers, and where it reads."""

    #: The §4.5 wording, carried into the artifact so the mapping from this
    #: table to the plan is a field rather than an act of interpretation.
    plan_row: str
    name: str
    module: str
    unit: str
    metric_type: str
    #: The arena slot whose champion this row grades, or None when the row is
    #: reduced from the data layer or from the trader.
    slot: str | None


#: The five rows. Frozen and ordered: the table's shape is the clause, so a
#: sixth row or a missing one is a change to this tuple, in the one place the
#: reason for each row is written down.
ROWS: tuple[RowSpec, ...] = (
    RowSpec(
        plan_row="data freshness/coverage",
        name="data_coverage_ratio",
        module="crucible.data",
        unit="ratio",
        metric_type="ratio",
        slot=None,
    ),
    RowSpec(
        plan_row="signal IC (R)",
        name="signal_excess_return_r_ratio",
        module="crucible.slots.research",
        unit="ratio",
        metric_type="ratio",
        slot="r",
    ),
    RowSpec(
        plan_row="prediction IC (M)",
        name="prediction_excess_return_m_ratio",
        module="crucible.slots.model",
        unit="ratio",
        metric_type="ratio",
        slot="m",
    ),
    RowSpec(
        plan_row="portfolio alpha (S)",
        name="portfolio_excess_return_s_ratio",
        module="crucible.slots.strategy",
        unit="ratio",
        metric_type="ratio",
        slot="s",
    ),
    RowSpec(
        plan_row="execution shortfall",
        name="execution_shortfall_bps",
        module="crucible.executor",
        unit="bps",
        metric_type="ratio",
        slot=None,
    ),
)


def attribution_key(trading_day: str) -> str:
    """The week's report card. Keyed by trading day like everything else."""
    return f"report/{trading_day}/attribution.json"


def build_attribution(
    store: Store,
    *,
    trading_day: dt.date,
    now: dt.datetime,
    run_id: str,
) -> tuple[dict[str, Any], list[str]]:
    """Reduce the store into the five-row table. Returns (document, sources).

    ``sources`` is every artifact key the reduction actually read, so the
    caller records them as manifest inputs and `explain` can walk from a
    report card back to the runs behind each row.
    """
    sessions = _window(trading_day)
    sources: list[str] = []
    rows: list[dict[str, Any]] = []

    for spec in ROWS:
        if spec.slot is None and spec.name == "data_coverage_ratio":
            row = _coverage_row(store, spec, sessions=sessions, now=now, sources=sources)
        elif spec.slot is not None:
            row = _slot_row(store, spec, trading_day=trading_day, now=now, sources=sources)
        else:
            row = _execution_row(spec, now=now)
        rows.append(row)

    if len(rows) != len(ROWS):
        raise AssertionError(
            f"the attribution table produced {len(rows)} rows against {len(ROWS)} declared. "
            "A report card that grades fewer layers than it declares is the shape this "
            "table exists to make impossible."
        )

    return (
        {
            "schema_version": ATTRIBUTION_SCHEMA_VERSION,
            "trading_day": trading_day.isoformat(),
            "generated_utc": _utc(now),
            "run_id": run_id,
            "window_trading_days": REPORT_WINDOW_TRADING_DAYS,
            "window_sessions": [d.isoformat() for d in sessions],
            "rows": rows,
        },
        sorted(set(sources)),
    )


def rows_complete_metric(
    document: dict[str, Any], *, now: dt.datetime, source_path: str
) -> dict[str, Any]:
    """`components.yaml` declares this as `report`'s outcome signal.

    It counts the rows PRESENT, not the rows measured: a row that is honestly
    ``N/A-NOT-RUN`` is a complete report card telling the truth, while a
    missing row is the table having quietly stopped grading a layer. The
    number of measured rows rides along in the reason, where it is read by a
    person rather than compared against a threshold.
    """
    rows = document["rows"]
    measured = [r for r in rows if not str(r["status"]).startswith("N/A")]
    return {
        "name": "attribution_rows_complete",
        "module": "crucible.report",
        "metric_type": "count",
        "value": float(len(rows)),
        "unit": "rows",
        "n_floor": len(ROWS),
        "status": "OK" if len(rows) == len(ROWS) else "FAIL",
        "status_reason": (
            f"{len(rows)} of {len(ROWS)} declared attribution rows written; "
            f"{len(measured)} carry a measured value and "
            f"{len(rows) - len(measured)} declare a not-measured state naming the "
            "artifact they are waiting on"
        ),
        "source_path": source_path,
        "last_updated_utc": _utc(now),
        "baseline": float(len(ROWS)),
    }


# -- the rows ---------------------------------------------------------------


def _coverage_row(
    store: Store,
    spec: RowSpec,
    *,
    sessions: list[dt.date],
    now: dt.datetime,
    sources: list[str],
) -> dict[str, Any]:
    """Freshness and coverage in one number, and it is one number on purpose.

    A session whose `data.daily` run is absent or failed contributes 0.0: it
    covered nothing. Reporting the mean over the days that DID run would be
    the "coverage of 100% computed over whatever arrived" defect — the number
    would read best exactly when the most days were missing.

    When NO session in the window has a data.daily manifest at all, the row is
    ``N/A-NOT-RUN`` rather than 0.0. A layer that never ran and a layer that
    ran and covered nothing are different facts, and only the second is a
    measurement.
    """
    observed: list[float] = []
    ran: list[dt.date] = []
    missing: list[dt.date] = []
    for day in sessions:
        key = manifest_key("data.daily", day.isoformat())
        manifest = _read_json(store, key)
        if manifest is None:
            missing.append(day)
            continue
        sources.append(key)
        ran.append(day)
        if manifest["status"] != "ok":
            observed.append(0.0)
            continue
        ratio = _metric_value(manifest, "universe_coverage_ratio")
        observed.append(0.0 if ratio is None else float(ratio))
    for _ in missing:
        observed.append(0.0)

    if not ran:
        return _row(
            spec,
            value=None,
            ci=(None, None),
            n_samples=0,
            baseline=COVERAGE_FLOOR_RATIO,
            now=now,
            source_path=manifest_key("data.daily", sessions[-1].isoformat()),
            ran=False,
            reason=(
                f"no data.daily manifest exists for any of the {len(sessions)} sessions "
                f"{sessions[0]}..{sessions[-1]}; the data layer has not run, which is an "
                "absence rather than a coverage of zero"
            ),
        )

    mean = sum(observed) / len(observed)
    low, high = _bootstrap_ci(observed)
    detail = (
        f"mean universe coverage {mean:.3f} over {len(sessions)} session(s) against a "
        f"floor of {COVERAGE_FLOOR_RATIO:.2f}"
    )
    if missing:
        detail += (
            f"; {len(missing)} session(s) have no data.daily manifest and count as 0.0 "
            f"({', '.join(d.isoformat() for d in missing)})"
        )
    return _row(
        spec,
        value=mean,
        ci=(low, high),
        n_samples=len(observed),
        baseline=COVERAGE_FLOOR_RATIO,
        now=now,
        source_path=manifest_key("data.daily", ran[-1].isoformat()),
        reason=detail,
        n_floor=len(sessions),
        target=COVERAGE_FLOOR_RATIO,
        red_line=COVERAGE_FLOOR_RATIO * 0.5,
    )


def _slot_row(
    store: Store,
    spec: RowSpec,
    *,
    trading_day: dt.date,
    now: dt.datetime,
    sources: list[str],
) -> dict[str, Any]:
    """One slot's champion, graded on its own settled verdicts.

    The population benchmark makes the baseline 0.0 without an argument: a
    verdict's ``score_ratio`` is the selection's realized return minus the
    equal-weight return of the population it drew from, so an arm with no edge
    scores zero in expectation and the row's question is whether the champion
    is distinguishable from a coin flip over the same names.
    """
    pointer_key = champion_key(spec.slot or "")
    pointer = _read_json(store, pointer_key)
    if pointer is None:
        return _row(
            spec,
            value=None,
            ci=(None, None),
            n_samples=0,
            baseline=0.0,
            now=now,
            source_path=pointer_key,
            ran=False,
            reason=(
                f"slot {spec.slot!r} has no champion pointer at {pointer_key}; the slot has "
                "not run an arena cycle, so there is no champion whose alpha this row could "
                "report"
            ),
        )
    sources.append(pointer_key)
    arm_id = pointer["arm_id"]

    prefix = f"experiments/{arm_key_segment(arm_id)}/"
    scores: list[float] = []
    horizons: set[int] = set()
    for key in sorted(store.list_keys(prefix)):
        if not key.endswith("/verdict.json"):
            continue
        verdict = json.loads(store.get_bytes(key).decode("utf-8"))
        if dt.date.fromisoformat(verdict["trading_day"]) > trading_day:
            continue
        sources.append(key)
        scores.append(float(verdict["score_ratio"]))
        horizons.add(int(verdict["horizon_trading_days"]))

    if not scores:
        return _row(
            spec,
            value=None,
            ci=(None, None),
            n_samples=0,
            baseline=0.0,
            now=now,
            source_path=prefix,
            input_present=False,
            reason=(
                f"champion {arm_id} holds no settled verdict on or before {trading_day} under "
                f"{prefix}; a decision date's verdict does not exist until its horizon "
                "settles, so this is a horizon that has not passed, not an arm with no edge"
            ),
        )
    if len(horizons) != 1:
        raise ValueError(
            f"champion {arm_id} carries verdicts at {sorted(horizons)} horizons. Averaging "
            "scores measured over different horizons produces a number that is not a "
            "measurement of anything (§4.12)."
        )

    mean = sum(scores) / len(scores)
    low, high = _bootstrap_ci(scores)
    return _row(
        spec,
        value=mean,
        ci=(low, high),
        n_samples=len(scores),
        baseline=0.0,
        now=now,
        source_path=prefix,
        horizon_trading_days=next(iter(horizons)),
        reason=(
            f"champion {arm_id}: mean realized excess return {mean:+.5f} against its own "
            f"population over {len(scores)} settled decision date(s), "
            f"{_ci_phrase(low, high)}; the baseline is 0.0 because the benchmark is the "
            "equal-weight population the selection drew from"
        ),
        n_floor=SLOT_N_FLOOR,
        target=0.0,
        red_line=0.0,
    )


def _execution_row(spec: RowSpec, *, now: dt.datetime) -> dict[str, Any]:
    """Execution shortfall — the trader's row, and the trader is not v2 phase 1.

    ``N/A-NOT-IMPL`` rather than a zero or a silence: §4.5 lists this row as
    "execution shortfall (trader, when present)", the harness must pass every
    acceptance test with the trader switched off, and a row that vanished when
    its producer was absent would make the table's shape depend on which
    systems happened to be running.
    """
    return _row(
        spec,
        value=None,
        ci=(None, None),
        n_samples=0,
        baseline=0.0,
        now=now,
        source_path="fills/{trading_day}/fills.json",
        implemented=False,
        reason=(
            "the trader is a separate system from this harness and publishes no fills "
            "artifact into the v2 store; §4.5 lists this row as 'when present', and it is "
            "declared here rather than omitted so the table's shape does not change with "
            "which systems happen to be running"
        ),
    )


# -- shared shape -----------------------------------------------------------


def _row(
    spec: RowSpec,
    *,
    value: float | None,
    ci: tuple[float | None, float | None],
    n_samples: int,
    baseline: float,
    now: dt.datetime,
    source_path: str,
    reason: str,
    n_floor: int = SLOT_N_FLOOR,
    target: float | None = None,
    red_line: float | None = None,
    horizon_trading_days: int | None = None,
    implemented: bool = True,
    ran: bool = True,
    input_present: bool = True,
) -> dict[str, Any]:
    """Assemble one row, and REFUSE to emit one that is not a MetricRecord.

    Built as a `krepis.metrics.MetricRecord` and dumped back to a dict, so the
    row's status vocabulary, its unit-when-value rule and its required fields
    are enforced by the shared contract rather than by this module agreeing
    with it. A row that is not a MetricRecord raises here rather than reaching
    the artifact, where the console would render whatever it was given.
    """
    low, high = ci
    status = derive_status(
        value=value,
        n_samples=n_samples,
        n_floor=n_floor,
        target=target,
        red_line=red_line,
        ci_low=low,
        ci_high=high,
        implemented=implemented,
        ran=ran,
        input_present=input_present,
    )
    record = MetricRecord(
        name=spec.name,
        module=spec.module,
        metric_type=spec.metric_type,
        value=value,
        unit=spec.unit if value is not None else None,
        ci_low=low,
        ci_high=high,
        ci_method=None if low is None else f"bootstrap-percentile-{BOOTSTRAP_RESAMPLES}",
        n_samples=n_samples,
        n_floor=n_floor,
        target=target,
        red_line=red_line,
        status=status,
        status_reason=reason,
        criticality="critical",
        source_path=source_path,
        last_updated_utc=now.astimezone(dt.UTC),
    )
    row = json.loads(record.model_dump_json())
    row["plan_row"] = spec.plan_row
    row["baseline"] = baseline
    row["last_updated_utc"] = _utc(now)
    if horizon_trading_days is not None:
        row["horizon_trading_days"] = horizon_trading_days
    return row


def _window(trading_day: dt.date) -> list[dt.date]:
    days = [trading_day]
    for _ in range(REPORT_WINDOW_TRADING_DAYS - 1):
        days.append(previous_trading_day(days[-1]))
    return sorted(days)


def _read_json(store: Store, key: str) -> dict[str, Any] | None:
    """One artifact, or None when it is absent. A corrupt one RAISES.

    Absence and corruption are different facts: returning None for both would
    let a truncated manifest render as a session that simply did not run.
    """
    if not store.exists(key):
        return None
    return json.loads(store.get_bytes(key).decode("utf-8"))


def _metric_value(manifest: dict[str, Any], name: str) -> float | None:
    for metric in manifest.get("metrics", []):
        if metric.get("name") == name:
            return metric.get("value")
    return None


def _ci_phrase(low: float | None, high: float | None) -> str:
    """The interval in words, or why there is none.

    A single observation has no interval, and saying so is not the same as
    omitting the phrase: a reason that silently drops the CI reads as a
    measurement whose interval nobody bothered to print.
    """
    if low is None or high is None:
        return (
            "no confidence interval — a single observation has none, and a zero-width "
            "one would read as certainty"
        )
    return f"95% bootstrap CI [{low:+.5f}, {high:+.5f}]"


def _bootstrap_ci(values: list[float]) -> tuple[float | None, float | None]:
    """A seeded 95% percentile bootstrap interval over ``values``.

    Seeded and fixed-size, so the same artifacts produce the same interval on
    every replay. A single observation has no interval — reported as None
    rather than as a zero-width one, which would read as certainty.
    """
    if len(values) < 2:
        return (None, None)
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(values)
    means = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return (means[int(0.025 * BOOTSTRAP_RESAMPLES)], means[int(0.975 * BOOTSTRAP_RESAMPLES) - 1])


def _utc(now: dt.datetime) -> str:
    return now.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
