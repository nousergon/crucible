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

**Every row declares the span it was reduced over.** The document header
carries the data-freshness window (row 1's, and the anchor the LLM cap is
paced across); each row additionally carries ``window_trading_days``,
``window_start`` and ``window_end``, because the layers settle at different
rates and a single header window would have to be a lie about four rows to be
true about one. A slot verdict settles on a 21-trading-day horizon at a weekly
decision cadence, so the slot rows are graded over twelve trading weeks
(:data:`SLOT_WINDOW_TRADING_DAYS`) — bounded, so no regime stays in the score
forever, and declared, so the denominator travels with the number.

**The reducer computes nothing a producer already computes.** The coverage row
is reduced from the ``universe_coverage_ratio`` MetricRecords that `data.daily`
already emits; the slot rows are reduced from the verdict artifacts the arena
already writes. A second implementation of a number is a second answer to the
same question.

**Rows two and three are the true rank IC §4.5 asks for.** They used to be
named as realized excess return, carrying a ``plan_row`` field naming the
§4.5 wording as an admission that the two disagreed: `shadow.v1` recorded
only the selected names, not the ranked cross-section, so a rank correlation
was not reconstructible from what was stored, and emitting an excess return
under the name of a correlation would have been the ``avg_volume_20d``
defect with a report card around it. `shadow.v2`
(`crucible.slots.grading.ScoredCrossSection`, `cross_section.json` +
`cross_section_settled.json`) persists the whole scored cross-section, so
the R and M rows now reduce a per-date Spearman rank IC
(`crucible.slots.grading.spearman_ic`) — the champion's score against the
realized forward return, paired by ticker, over every settled decision date
in the window — and ``plan_row`` is gone: the row's own name is the §4.5
wording (`alpha-engine-config-I9778`). The S row is unchanged — portfolio
alpha is a claim about realized return, not a ranking, so it stays
`_slot_row`'s realized-excess-return reduction.
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
from crucible.keys import arm_key_segment, attribution_key, champion_key
from crucible.manifest import manifest_key
from crucible.slots.grading import CROSS_SECTION_MIN_NAMES, RankICSkip, spearman_ic
from crucible.store import Store

__all__ = [
    "ATTRIBUTION_SCHEMA_VERSION",
    "ROWS",
    "REPORT_WINDOW_TRADING_DAYS",
    "SLOT_WINDOW_TRADING_DAYS",
    "RowSpec",
    "attribution_key",
    "build_attribution",
    "rows_complete_metric",
]

ATTRIBUTION_SCHEMA_VERSION = "attribution.v1"

#: The DATA-FRESHNESS window, in TRADING days (§4.12). One trading week: the
#: coverage row is written weekly, so a window shorter than a week would
#: leave sessions no report card ever looked at, and a longer one would let a
#: dead day age out of view before the next report noticed it.
#:
#: **It is row 1's window, not the document's.** See :data:`SLOT_WINDOW_
#: TRADING_DAYS` and ``window_scope`` below for why every row declares its
#: own and what the alternative measured.
REPORT_WINDOW_TRADING_DAYS = 5

#: The window the SLOT rows are graded over, in trading days: twelve trading
#: weeks.
#:
#: **Why the rows do not share row 1's five sessions.** A slot verdict is a
#: settled decision date, and the arena decides weekly on a 21-trading-day
#: horizon — so the five most recent sessions contain at most one decision
#: date and none of them have settled. Clamping the slot rows to five
#: sessions would make every slot row permanently ``N/A-LOW-N`` against a
#: floor of :data:`SLOT_N_FLOOR`, which is a report card that has stopped
#: grading three of its five layers.
#:
#: **Why it is not the champion's whole history either.** That is what this
#: constant replaces, and it was measured on 2026-09-01: a document declaring
#: ``window_trading_days: 5`` and sessions ``2026-08-24..2026-08-28`` carried
#: an R row of ``n_samples = 8`` averaging verdicts back to January. A weekly
#: report card whose slot rows are lifetime means under a five-session header
#: is a number with the wrong denominator, and an unbounded window means no
#: regime ever ages out of the champion's score.
#:
#: Sized off the floor rather than picked: :data:`SLOT_N_FLOOR` settled
#: decision dates at a weekly cadence is 30 trading days, doubled so that a
#: missed cycle does not drop the row below its floor. It is a quarter, which
#: is also the shortest span over which a regime claim is worth making.
SLOT_WINDOW_TRADING_DAYS = 60

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

#: Settled decision DATES a rank-IC row needs before its value is read as a
#: measurement rather than as noise. Same number as :data:`SLOT_N_FLOOR` and
#: for the same reason — both floor a count of settled dates in the same
#: :data:`SLOT_WINDOW_TRADING_DAYS` window — kept as a separate name because
#: the two rows floor DIFFERENT things (one arm-dates of excess return, one
#: dates that cleared :data:`~crucible.slots.grading.CROSS_SECTION_MIN_NAMES`
#: names each) and a shared constant would make that an accident rather than
#: a fact.
RANK_IC_N_FLOOR = SLOT_N_FLOOR


@dataclass(frozen=True)
class RowSpec:
    """One row of the §4.5 table: what it answers, and where it reads."""

    name: str
    module: str
    unit: str
    metric_type: str
    #: The arena slot whose champion this row grades, or None when the row is
    #: reduced from the data layer or from the trader.
    slot: str | None
    #: True for R and M: the row is a Spearman rank IC reduced from
    #: `shadow.v2`'s settled cross-sections, not a realized excess return.
    #: S stays excess return — portfolio alpha is a claim about what was
    #: actually earned, not a ranking, so there is no rank to correlate.
    rank_ic: bool = False


#: The five rows. Frozen and ordered: the table's shape is the clause, so a
#: sixth row or a missing one is a change to this tuple, in the one place the
#: reason for each row is written down.
ROWS: tuple[RowSpec, ...] = (
    RowSpec(
        name="data_coverage_ratio",
        module="crucible.data",
        unit="ratio",
        metric_type="ratio",
        slot=None,
    ),
    RowSpec(
        name="signal_rank_ic_r",
        module="crucible.slots.research",
        unit="ic",
        metric_type="ic",
        slot="r",
        rank_ic=True,
    ),
    RowSpec(
        name="prediction_rank_ic_m",
        module="crucible.slots.model",
        unit="ic",
        metric_type="ic",
        slot="m",
        rank_ic=True,
    ),
    RowSpec(
        name="portfolio_excess_return_s_ratio",
        module="crucible.slots.strategy",
        unit="ratio",
        metric_type="ratio",
        slot="s",
    ),
    RowSpec(
        name="execution_shortfall_bps",
        module="crucible.executor",
        unit="bps",
        metric_type="ratio",
        slot=None,
    ),
)


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
        elif spec.rank_ic:
            row = _rank_ic_row(store, spec, trading_day=trading_day, now=now, sources=sources)
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
            # Row 1's window, and the anchor `crucible.track_e` paces the LLM
            # cap across. NOT the document's window: the rows are graded over
            # different spans because the layers settle at different rates,
            # and each row carries its own `window_*` fields saying which.
            "window_scope": "per-row",
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

    **An absent session is absent, not an observation of zero.** The shape
    here used to append 0.0 for every session with no `data.daily` manifest
    and then report ``n_samples = len(sessions)`` against
    ``n_floor = len(sessions)`` — so one real observation out of five read as
    five samples, cleared its own floor, and carried a
    ``bootstrap-percentile-2000`` interval computed over four numbers nobody
    measured (reproduced 2026-09-01). ``n`` and a CI are claims about data,
    and a zero-fill on an absent input is the exact defect §4.3 names.

    So the mean is taken over the sessions that RAN, and the sessions that
    did not are what collapses ``n_samples`` against a floor of the full
    window. That is not the "coverage of 100% computed over whatever arrived"
    escape it looks like at first glance: with four of five sessions missing
    the value may read 1.0, but ``n_samples = 1`` against ``n_floor = 5`` is
    ``N/A-LOW-N``, the reason names every absent session, and the row can
    never reach GREEN until the window is whole. Missing days degrade the
    row's *evidence*, which is what they are, rather than its *value*, which
    they are not.

    A session whose `data.daily` run exists and FAILED still contributes 0.0:
    that layer ran and covered nothing, which is a measurement.

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
            window=sessions,
            reason=(
                f"no data.daily manifest exists for any of the {len(sessions)} sessions "
                f"{sessions[0]}..{sessions[-1]}; the data layer has not run, which is an "
                "absence rather than a coverage of zero"
            ),
        )

    mean = sum(observed) / len(observed)
    low, high = _bootstrap_ci(observed)
    detail = (
        f"mean universe coverage {mean:.3f} over the {len(ran)} session(s) that ran, of "
        f"{len(sessions)} in {sessions[0]}..{sessions[-1]}, against a floor of "
        f"{COVERAGE_FLOOR_RATIO:.2f}"
    )
    if missing:
        detail += (
            f"; {len(missing)} session(s) have no data.daily manifest and are ABSENT rather "
            f"than zero ({', '.join(d.isoformat() for d in missing)}) — they are what holds "
            f"n_samples at {len(observed)} against a floor of {len(sessions)}, so this row "
            "cannot read GREEN until the window is whole"
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
        window=sessions,
    )


def _slot_row(
    store: Store,
    spec: RowSpec,
    *,
    trading_day: dt.date,
    now: dt.datetime,
    sources: list[str],
) -> dict[str, Any]:
    """One slot's champion, graded on its own settled verdicts IN A WINDOW.

    The population benchmark makes the baseline 0.0 without an argument: a
    verdict's ``score_ratio`` is the selection's realized return minus the
    equal-weight return of the population it drew from, so an arm with no edge
    scores zero in expectation and the row's question is whether the champion
    is distinguishable from a coin flip over the same names.

    **The window is declared and bounded** (:data:`SLOT_WINDOW_TRADING_DAYS`).
    This used to read every verdict the champion had ever produced on or
    before the trading day; the row then carried a lifetime mean under a
    header declaring a five-session week. The window is twelve trading weeks
    rather than the header's five sessions because a decision date settles on
    a 21-trading-day horizon and the arena decides weekly — the five most
    recent sessions hold no settled verdict at all — and each row now carries
    ``window_trading_days``/``window_start``/``window_end``, so the span a
    number was reduced over travels with the number instead of with the
    document.

    **Status calibration is two-state, deliberately, and says so here.**
    ``target`` and ``red_line`` are both 0.0, so `krepis.metrics.derive_status`
    returns RED for any champion whose bootstrap CI reaches zero and GREEN
    only for one whose whole interval clears it. There is no amber band
    between them, and there is no honest way to add one from this module: an
    amber band needs a *minimum detectable effect* — a per-slot excess return
    below which the champion is uninteresting rather than harmful — and the
    plan declares none. SOTA is a pre-registered MDE per slot in the arena's
    `ArenaConfig`; the delta is that inventing one here would be a threshold
    with no provenance sitting between a report card and a promotion. The
    conservative reading (not distinguishable from the population ⇒ RED) is
    the one that cannot render *no evidence* as green.
    """
    pointer_key = champion_key(spec.slot or "")
    window = _window(trading_day, SLOT_WINDOW_TRADING_DAYS)
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
            window=window,
            reason=(
                f"slot {spec.slot!r} has no champion pointer at {pointer_key}; the slot has "
                "not run an arena cycle, so there is no champion whose alpha this row could "
                "report"
            ),
        )
    sources.append(pointer_key)
    arm_id = pointer["arm_id"]

    first, last = window[0], window[-1]
    prefix = f"experiments/{arm_key_segment(arm_id)}/"
    scores: list[float] = []
    horizons: set[int] = set()
    outside = 0
    for key in sorted(store.list_keys(prefix)):
        if not key.endswith("/verdict.json"):
            continue
        verdict = json.loads(store.get_bytes(key).decode("utf-8"))
        day = dt.date.fromisoformat(verdict["trading_day"])
        if not first <= day <= last:
            outside += 1
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
            window=window,
            reason=(
                f"champion {arm_id} holds no settled verdict in the {len(window)}-session "
                f"window {first}..{last} under {prefix} ({outside} verdict(s) exist outside "
                "it); a decision date's verdict does not exist until its horizon settles, "
                "so this is a horizon that has not passed, not an arm with no edge"
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
        window=window,
        reason=(
            f"champion {arm_id}: mean realized excess return {mean:+.5f} against its own "
            f"population over {len(scores)} settled decision date(s) in the "
            f"{len(window)}-session window {first}..{last}, {_ci_phrase(low, high)}; the "
            "baseline is 0.0 because the benchmark is the equal-weight population the "
            "selection drew from"
        ),
        n_floor=SLOT_N_FLOOR,
        target=0.0,
        red_line=0.0,
    )


def _rank_ic_row(
    store: Store,
    spec: RowSpec,
    *,
    trading_day: dt.date,
    now: dt.datetime,
    sources: list[str],
) -> dict[str, Any]:
    """R and M: the champion's TRUE rank IC, reduced from `shadow.v2`.

    Replaces the realized-excess-return stand-in this row used to carry
    under a ``plan_row`` of "signal IC (R)" / "prediction IC (M)"
    (`alpha-engine-config-I9778`). `shadow.v1` recorded only the top-N
    selection, so a rank correlation was not reconstructible from what was
    stored; `crucible.slots.grading.ScoredCrossSection` now persists the
    whole scored cross-section, settled once each decision date's horizon
    passes, at ``experiments/{arm}/{day}/cross_section_settled.json``.

    **One IC per settled date, then aggregated across the window** — the
    same two-stage shape :func:`_slot_row` uses for excess return, for the
    same reason: a rank IC is a per-date measurement (score vs. realized
    return, paired by ticker, within one cross-section) and averaging raw
    ticker pairs across dates would let a date with an unusually large
    population dominate the mean. ``n_samples`` is the count of DATES that
    produced an IC, against :data:`RANK_IC_N_FLOOR` — a separate axis from
    :data:`~crucible.slots.grading.CROSS_SECTION_MIN_NAMES`, the floor on
    names WITHIN one date's cross-section below which that date contributes
    no IC at all. `crucible.slots.grading.spearman_ic` enforces two distinct
    skip conditions (:class:`~crucible.slots.grading.RankICSkip`) rather than
    one: too few paired names, or a paired-but-DEGENERATE cross-section — a
    constant score or a constant realized return, whose correlation is
    undefined rather than zero. Neither skip contributes a value OR
    increments ``n_samples``: a date that carries no information does not
    count as an observation, even when it clears
    :data:`~crucible.slots.grading.CROSS_SECTION_MIN_NAMES`
    (`alpha-engine-config-I9778` review, finding 1 — a constant-ranker date
    was previously scored as a real ``ic=0.0`` and could clear
    :data:`RANK_IC_N_FLOOR` on its own, turning a `WATCH` row `GREEN`). This
    function counts and reports both skip reasons separately in
    ``status_reason``, so a low n is legible rather than merely low.

    **No backfill was performed** (plan §4.5's explicit instruction on this
    clause): `cross_section_settled.json` exists only for cycles run after
    `shadow.v2` shipped, so the first weeks after rollout show climbing n
    against the floor rather than a false GREEN or a fabricated history —
    that IS the honest first true-IC week, not a defect in it.
    """
    pointer_key = champion_key(spec.slot or "")
    window = _window(trading_day, SLOT_WINDOW_TRADING_DAYS)
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
            window=window,
            reason=(
                f"slot {spec.slot!r} has no champion pointer at {pointer_key}; the slot has "
                "not run an arena cycle, so there is no champion whose rank IC this row "
                "could report"
            ),
        )
    sources.append(pointer_key)
    arm_id = pointer["arm_id"]

    first, last = window[0], window[-1]
    prefix = f"experiments/{arm_key_segment(arm_id)}/"
    ics: list[float] = []
    horizons: set[int] = set()
    outside = 0
    skipped_low_names = 0
    skipped_degenerate = 0
    for key in sorted(store.list_keys(prefix)):
        if not key.endswith("/cross_section_settled.json"):
            continue
        document = json.loads(store.get_bytes(key).decode("utf-8"))
        day = dt.date.fromisoformat(document["trading_day"])
        if not first <= day <= last:
            outside += 1
            continue
        score_by_ticker = {row["ticker"]: row["score"] for row in document["ranks"]}
        return_by_ticker = {
            row["ticker"]: row["realized_forward_return_ratio"]
            for row in document["ranks"]
            if row["realized_forward_return_ratio"] is not None
        }
        result = spearman_ic(score_by_ticker, return_by_ticker)
        if result is RankICSkip.TOO_FEW_NAMES:
            skipped_low_names += 1
            continue
        if result is RankICSkip.DEGENERATE:
            # Undefined correlation (constant score or constant realized
            # return): this date carries NO information and must not be
            # counted toward n_samples, even though it cleared
            # CROSS_SECTION_MIN_NAMES. Reported separately from
            # skipped_low_names so a status_reason never conflates "too
            # little data to pair" with "paired, but the pairing measured
            # nothing" (alpha-engine-config-I9778 review, finding 1).
            skipped_degenerate += 1
            continue
        sources.append(key)
        ic, _n_names = result
        ics.append(ic)
        horizons.add(int(document["horizon_trading_days"]))

    if not ics:
        return _row(
            spec,
            value=None,
            ci=(None, None),
            n_samples=0,
            baseline=0.0,
            now=now,
            source_path=prefix,
            input_present=False,
            window=window,
            n_floor=RANK_IC_N_FLOOR,
            reason=(
                f"champion {arm_id} holds no settled cross-section with at least "
                f"{CROSS_SECTION_MIN_NAMES} paired names in the {len(window)}-session window "
                f"{first}..{last} under {prefix} ({outside} settled cross-section(s) exist "
                f"outside it, {skipped_low_names} inside it were skipped for too few paired "
                f"names, {skipped_degenerate} inside it were skipped for an undefined "
                "correlation — a constant score or a constant realized return). No backfill "
                "was performed for shadow.v2, so this is expected in the weeks immediately "
                "after rollout, not an arm with no edge"
            ),
        )
    if len(horizons) != 1:
        raise ValueError(
            f"champion {arm_id} carries settled cross-sections at {sorted(horizons)} "
            "horizons. Averaging rank ICs measured over different horizons produces a "
            "number that is not a measurement of anything (§4.12)."
        )

    mean = sum(ics) / len(ics)
    low, high = _bootstrap_ci(ics)
    return _row(
        spec,
        value=mean,
        ci=(low, high),
        n_samples=len(ics),
        baseline=0.0,
        now=now,
        source_path=prefix,
        horizon_trading_days=next(iter(horizons)),
        window=window,
        n_floor=RANK_IC_N_FLOOR,
        reason=(
            f"champion {arm_id}: mean Spearman rank IC {mean:+.5f} of score against realized "
            f"forward return over {len(ics)} settled decision date(s) (each clearing "
            f"{CROSS_SECTION_MIN_NAMES}+ paired names) in the {len(window)}-session window "
            f"{first}..{last}, {_ci_phrase(low, high)}; {skipped_low_names} date(s) inside the "
            f"window were skipped for too few paired names, {skipped_degenerate} were skipped "
            "for an undefined correlation (a constant score or a constant realized return — "
            f"not counted as an observation), and {outside} settled cross-section(s) exist "
            "outside it; the baseline is 0.0 because that is the expectation of a random "
            "ranking's correlation with the realized return"
        ),
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
    window: list[dt.date] | None = None,
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
    row["baseline"] = baseline
    # Every row declares the span it was reduced over. A row whose window is
    # only stated in the document header is a row that can silently be read
    # over a different span than the header claims — measured 2026-09-01, the
    # three slot rows were lifetime means under a five-session header.
    row["window_trading_days"] = None if window is None else len(window)
    row["window_start"] = None if window is None else window[0].isoformat()
    row["window_end"] = None if window is None else window[-1].isoformat()
    row["last_updated_utc"] = _utc(now)
    if horizon_trading_days is not None:
        row["horizon_trading_days"] = horizon_trading_days
    return row


def _window(trading_day: dt.date, length: int = REPORT_WINDOW_TRADING_DAYS) -> list[dt.date]:
    """The ``length`` trading sessions ending at ``trading_day``, inclusive.

    Trading days (§4.12), never calendar days, so a holiday week is four
    sessions of five rather than five days of which one can never arrive.
    """
    if length < 1:
        raise ValueError(f"a window is a count of sessions and is at least 1; got {length}")
    days = [trading_day]
    for _ in range(length - 1):
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
            "no confidence interval — a single observation has none, identical "
            "observations give a bootstrap nothing to resample, and a zero-width interval "
            "would read as certainty"
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
    low = means[int(0.025 * BOOTSTRAP_RESAMPLES)]
    high = means[int(0.975 * BOOTSTRAP_RESAMPLES) - 1]
    if low == high:
        # Every resample produced the same mean, which happens when every
        # observation is identical. The interval is zero-width, and a
        # zero-width interval is not certainty — it is a bootstrap with
        # nothing to resample. Emitting it would put GREEN on the row off an
        # interval the method could not have produced, and would stamp
        # `ci_method: bootstrap-percentile-2000` on a claim the bootstrap did
        # not make. None, and `_ci_phrase` says why.
        return (None, None)
    return (low, high)


def _utc(now: dt.datetime) -> str:
    return now.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
