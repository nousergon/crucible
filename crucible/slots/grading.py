"""Producing shadows, scoring them once the horizon settles, and running the cycle.

Normative source: plan §4.4, §10.1; binding source
`champion-challenger-policy.md` §3, §3.1, §4, §5.0, §7.

**Produce and grade are separate acts separated by the horizon.** An arm
SELECTS on trading day *d* and writes `experiments/{arm}/{d}/shadow.json`
before any outcome exists. It is SCORED only once *d + horizon* sessions
have closed, into `experiments/{arm}/{d}/verdict.json`. Collapsing the two
is how a look-ahead gets written into a production artifact and read back as
a result.

**There is no cross-arm intersection.** `I9745`: v1 recomputed every arm's
figure over the dates on which EVERY registered arm had scored, so a single
arm with no shadow at all nulled every other arm's paired figure. Here each
comparison consumes
`nousergon_lib.arena.window.pair_on_common_window(a, b)` — the dates on
which *those two* produced — so the engine cannot see un-paired data even by
accident. `tests/test_grading.py` reproduces the I9745 shape and asserts the
surviving arm keeps its figure.

**The grader has two halves, and each needs its own control** (§10.1).
Half one CONSTRUCTS the label — :func:`forward_returns` turns a price panel
into a realized forward return per ticker. Half two SCORES a selection
against that label — :func:`score_selection`. A cycle is only as trustworthy
as both.

*The scoring half* is checked by the planted/null pair. The planted control
ranks on the realized forward return plus calibrated noise, so it carries a
KNOWN edge; the null control ranks on noise alone. If the grader does not
rank planted above null, the GRADER is broken and the cycle's verdicts are
void — the cycle fails rather than publishing them. Both controls are
produced at GRADE time, not at produce time, because a planted edge is by
construction a look-ahead: it is legal only in a harness device that can
never be promoted, and generating it in the produce path would put a
look-ahead artifact on the real-time write path.

*The label half was, until this module carried a second control, invisible
to the first one.* Both controls are generated from and scored against the
SAME `returns` mapping the real arms are scored against, so any defect in
:func:`forward_returns` moves planted and null identically and the
planted-over-null margin survives it. Measured: forcing the horizon to 5
sessions published a clean cycle at `control margin=0.025838, n_paired=6`,
while every `verdict.json` in the run claimed `horizon_trading_days: 21`
because that number was a separately-passed literal rather than a property
of what was measured. Two changes close that:

* :func:`forward_returns` returns a :class:`ForwardReturnWindow` whose
  `horizon_trading_days` is DERIVED from the panel's own session index — the
  count of sessions between the anchor and the settle date it actually
  used — and :func:`write_verdict` takes the window rather than an integer.
  A verdict can no longer claim a horizon nobody measured.
* :func:`assert_label_control` recomputes the same labels through
  :func:`reference_forward_returns`, a second implementation that shares no
  code with the first, and voids the cycle when the two disagree or when the
  span measured is not the span declared.

:func:`assert_label_control`'s reach and its blind spot are written out on
the function itself. Read them before trusting a green cycle: a control
whose blind spot is undocumented is worse than one whose blind spot is
written down.

**Benchmark: the population the arm drew from, count-matched.** For an
equal-weight selection of *k* names, the equal-weight mean of the population
IS the expectation of a random *k*-name draw from it, so the count match is
exact without sampling. Never SPY for a selection slot — the library's
`ArenaConfig` refuses that outright, and 2026-08-17 is why.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from nousergon_lib.arena.engine import (
    ArenaCycle,
    ServingPrecondition,
    TrainingIntegrityError,
    TrainingStatus,
    run_cycle,
)
from nousergon_lib.arena.window import ArmSeries, pair_on_common_window

from crucible.keys import shadow_key, verdict_key
from crucible.slots import SlotSpec
from crucible.slots.arms import ArmSpec
from crucible.slots.rankers import MissingFeatureError, rank_with

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from crucible.store import Store

__all__ = [
    "CONTROL_PLANTED_IC",
    "DEFAULT_HORIZON_TRADING_DAYS",
    "LABEL_CONTROL_REL_TOL",
    "ForwardReturnWindow",
    "GraderControlError",
    "PopulationIntegrityError",
    "SelectionMissError",
    "ShadowSelection",
    "assert_label_control",
    "forward_returns",
    "grade_slot",
    "produce_shadow",
    "reference_forward_returns",
    "score_selection",
    "series_from_verdicts",
]

#: §4.12: a horizon is a count of SESSIONS. 21 is the canonical one-month
#: forward window the fleet's labels already use; expressing it as "1 month"
#: is unrepresentable in the manifest schema, which is the point.
DEFAULT_HORIZON_TRADING_DAYS = 21

#: The planted control's information coefficient — the correlation its
#: ranking signal is CONSTRUCTED to have with the realized forward return.
#: A known quantity, which is the entire value of a positive control: if the
#: grader cannot see an edge this large, it cannot see a real one.
CONTROL_PLANTED_IC = 0.35

#: How far the label control lets the two independent label implementations
#: drift before it voids the cycle. Relative, and tight: the two paths read
#: the same two closes and divide them, so anything beyond float
#: representation error is a real disagreement about WHICH closes. A loose
#: tolerance here would let a wrong-but-nearby settle date through, which is
#: exactly the class the control exists for.
LABEL_CONTROL_REL_TOL = 1e-9


class GraderControlError(RuntimeError):
    """The controls did not rank as constructed. The cycle's verdicts are void.

    Not a warning and not a metric with a red status. The audit's central
    finding was a grading loop that ran for months while measuring nothing;
    a control that can produce a negative result is the only evidence the
    harness itself works, so a failed control fails the run.
    """


class SelectionMissError(ValueError):
    """This arm had nothing scoreable to say on this date. A MISS, not a failure.

    Plan §4.4 and policy §3: *"a cycle in which an arm legitimately selects
    nothing is a miss"* — and a miss is data. Every name an arm picked was
    delisted, halted or otherwise carried no settled close at both ends of
    the horizon, while the population it drew from is intact and every other
    arm scores normally. That is a fact about this arm on this date, and
    recording it is the whole point: policy §3 requires that silent absence
    and a genuine zero never render identically.

    Deliberately a `ValueError` subclass: the raise it replaces was a bare
    `ValueError`, and a caller that still catches the general class keeps
    working rather than losing the refusal.

    The defect this type exists to end (`I9757`): the raise had NO handler at
    the `run_grade` call site, so one arm whose picks had all delisted took
    the whole slot down — no arena cycle and no verdict for any healthy arm.
    That is the `ChallengerShadowGapError` shape v2 claims to have retired,
    reappearing one step later.
    """


class PopulationIntegrityError(ValueError):
    """The BENCHMARK could not be formed. Compromised inputs, not a miss.

    The other half of the distinction above, and the half that must still
    fail the slot. A population with fewer than two settled forward returns
    is not a population; every arm in the slot is scored against it, so the
    defect is in the cycle's shared input substrate rather than in one arm's
    picks (policy §3, Brian's 2026-08-29 ruling). The call site turns this
    into a :class:`~nousergon_lib.arena.engine.TrainingIntegrityError` and
    the run FAILS.
    """


@dataclass(frozen=True)
class ForwardReturnWindow:
    """The labels for one anchor date, WITH the span they were measured over.

    The horizon is a field of this object rather than an argument travelling
    beside it, because that is the defect it closes. A verdict used to state
    `horizon_trading_days` from a literal passed independently of the returns
    it described, so a run whose labels spanned 5 sessions published verdicts
    claiming 21 and nothing anywhere disagreed.

    ``horizon_trading_days`` is DERIVED — the number of sessions the panel's
    own index carries between ``start`` and ``end`` — so it cannot be
    asserted, only measured.
    """

    start: str
    end: str
    horizon_trading_days: int
    returns: dict[str, float]

    def __post_init__(self) -> None:
        if self.horizon_trading_days < 1:
            raise ValueError(
                f"a forward-return window spans {self.horizon_trading_days} sessions "
                f"({self.start} to {self.end}). A horizon of zero or fewer sessions is a "
                "same-day return wearing a forward return's name."
            )


@dataclass(frozen=True)
class ShadowSelection:
    """What one arm chose on one trading day, before any outcome existed."""

    arm_id: str
    trading_day: str
    selection: tuple[str, ...]
    population: tuple[str, ...]
    ranker: str
    params: dict[str, Any]
    look_ahead: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "shadow.v1",
            "arm_id": self.arm_id,
            "trading_day": self.trading_day,
            "selection": list(self.selection),
            "population": list(self.population),
            "ranker": self.ranker,
            "params": dict(self.params),
            "look_ahead": self.look_ahead,
        }


def _top_n(params: dict[str, Any], default: int = 10) -> int:
    value = int(params.get("top_n", default))
    if value < 1:
        raise ValueError(f"top_n must be >= 1; got {value}")
    return value


def produce_shadow(
    spec: ArmSpec,
    features: pd.DataFrame,
    trading_day: dt.date,
) -> ShadowSelection:
    """Run one arm's recipe against the feature cross-section.

    A ranker whose declared input is absent raises
    :class:`~crucible.slots.rankers.MissingFeatureError`, and the caller
    turns that into a failed run. It is deliberately NOT recorded as a miss:
    a miss means "this arm legitimately had nothing to say", and a missing
    input means the cycle's inputs were compromised (plan §4.4).
    """
    population = tuple(sorted(str(t) for t in features["ticker"].unique()))
    scores = rank_with(spec.ranker, features, spec.params)
    top_n = _top_n(spec.params)
    selection = tuple(str(t) for t in scores.head(top_n).index)
    if not selection:
        raise MissingFeatureError(
            f"arm {spec.name!r} ranked zero names on {trading_day} from a population of "
            f"{len(population)}. Every feature its ranker reads was null across the "
            "whole cross-section, which is a broken feature layer, not an arm with "
            "nothing to say."
        )
    return ShadowSelection(
        arm_id=spec.arm_id,
        trading_day=trading_day.isoformat(),
        selection=selection,
        population=population,
        ranker=spec.ranker,
        params=dict(spec.params),
    )


def _sessions(panel: pd.DataFrame) -> list[Any]:
    """The panel's own session index, ascending.

    The set of days the market actually traded, taken from the data rather
    than from a calendar offset: a calendar offset lands on a holiday and
    silently shortens or lengthens the horizon per ticker.
    """
    return sorted({d for d in panel["trading_day"].unique()})


def _settle_session(sessions: list[Any], start: dt.date, horizon_trading_days: int) -> Any:
    """The session ``horizon_trading_days`` after ``start``, or a named refusal."""
    if start not in sessions:
        raise ValueError(
            f"{start} is not a session in the supplied panel (which spans "
            f"{sessions[0]}..{sessions[-1]}); a forward return anchored to a day the "
            "panel does not carry would silently anchor to a neighbour"
        )
    index = sessions.index(start)
    end_index = index + horizon_trading_days
    if end_index >= len(sessions):
        raise ValueError(
            f"the horizon from {start} needs {horizon_trading_days} further sessions and "
            f"the panel carries {len(sessions) - index - 1}. The horizon has not settled; "
            "a return computed over a short window is a different measurement wearing "
            "the same name (policy §7: horizon-vs-retention, asserted not assumed)."
        )
    return sessions[end_index]


def forward_returns(
    panel: pd.DataFrame,
    *,
    start: dt.date,
    horizon_trading_days: int = DEFAULT_HORIZON_TRADING_DAYS,
) -> ForwardReturnWindow:
    """Simple forward return per ticker over ``horizon_trading_days`` SESSIONS.

    Sessions are counted from the panel's own date index, which is the set of
    days the market actually traded — not a calendar offset, which would land
    on a holiday and silently shorten or lengthen the horizon per ticker.

    A ticker without a settled close at both ends is ABSENT from the result.
    It is not zero: a delisting is not a flat return, and the arm that picked
    it records the exclusion rather than banking a 0%.

    Returns a :class:`ForwardReturnWindow`, not a bare mapping. The window
    carries the anchor, the settle session, and the span between them as
    counted in the panel — so every downstream artifact states the horizon it
    was measured over rather than one supplied alongside it.
    """
    sessions = _sessions(panel)
    end = _settle_session(sessions, start, horizon_trading_days)

    frame = panel[panel["trading_day"].isin({start, end})]
    pivot = frame.pivot_table(index="ticker", columns="trading_day", values="close_raw")
    if start not in pivot.columns or end not in pivot.columns:
        raise ValueError(f"panel carries no closes for {start} or {end}")
    both = pivot[[start, end]].dropna()
    both = both[both[start] > 0]
    ratio = (both[end] / both[start]) - 1.0
    return ForwardReturnWindow(
        start=str(start),
        end=str(end),
        # DERIVED from the index actually walked, never echoed back from the
        # argument: if this function ever measured a different span than it
        # was asked for, this is the number that would say so.
        horizon_trading_days=sessions.index(end) - sessions.index(start),
        returns={str(t): float(v) for t, v in ratio.items()},
    )


def reference_forward_returns(
    panel: pd.DataFrame,
    *,
    start: dt.date,
    horizon_trading_days: int,
) -> dict[str, float]:
    """The same labels, computed a second time by a different construction.

    The input to the label control (:func:`assert_label_control`). It shares
    no code with :func:`forward_returns`: where that one reshapes the panel
    into a `ticker x trading_day` pivot and divides two columns, this one
    walks the rows into a plain `{ticker: {session: close}}` mapping and
    divides two scalars per name. Nothing is imported from the other path,
    so a defect in the pivot — an aggregation over duplicate rows, a column
    that resolved to the wrong session, a `dropna` that silently kept a
    misaligned pair — is not shared.

    ``horizon_trading_days`` is the span the CALLER declared, which is the
    load-bearing asymmetry: the cycle passes the same declared horizon to
    both paths, so a function that measured something else than it was asked
    for disagrees with this one rather than agreeing with itself.
    """
    sessions = _sessions(panel)
    end = _settle_session(sessions, start, horizon_trading_days)

    closes: dict[str, dict[Any, float]] = {}
    for row in panel.itertuples(index=False):
        day = row.trading_day
        if day != start and day != end:
            continue
        close = float(row.close_raw)
        if close != close:  # NaN: an unsettled close is absent, never zero
            continue
        closes.setdefault(str(row.ticker), {})[day] = close

    out: dict[str, float] = {}
    for ticker, by_day in closes.items():
        first = by_day.get(start)
        last = by_day.get(end)
        if first is None or last is None or first <= 0:
            continue
        out[ticker] = (last / first) - 1.0
    return out


def assert_label_control(
    window: ForwardReturnWindow,
    reference: dict[str, float],
    *,
    slot: str,
    declared_horizon_trading_days: int,
) -> dict[str, Any]:
    """§10.1's control over the LABEL half of the grader. Voids the cycle when it fails.

    The planted/null pair checks that the grader can SEE an edge. It cannot
    check that the edge it saw was measured over the right window, because
    both controls are generated from and scored against the very mapping
    under test — a wrong label moves planted and null identically and the
    margin survives it. This is the control for that half, and like the
    other one its whole value is that it can produce a negative result.

    **What it catches** — every defect that makes the labels under test
    differ from a correct measurement of the declared horizon:

    * a horizon that is not the one declared, wherever it came from — an
      argument ignored, a constant edited, an off-by-one on the settle index.
      This is the reproduced `I9757` defect: forcing a 5-session horizon
      published a clean cycle at `control margin=0.025838`, and here it
      raises instead.
    * an anchor or settle session resolved to a neighbouring day.
    * a per-ticker misalignment: a close paired with the wrong session, a
      pivot aggregating duplicate rows, a `dropna` that kept a broken pair.
    * a name present in one construction and absent from the other — a
      silently dropped or silently invented ticker.

    **What it CANNOT catch**, because both paths read the same panel through
    the same contract, and saying so is the point (a control whose blind spot
    is undocumented is worse than one whose blind spot is written down):

    * a defect in the PANEL itself. If `close_raw` is adjusted wrongly, stale,
      survivorship-biased, or carries a look-ahead from a restatement, both
      constructions reproduce it exactly and this control passes. Panel
      integrity is the data layer's contract, not this one's.
    * a WRONGLY DECLARED horizon. If the cycle declares 5 sessions, both
      paths measure 5 and agree; the control passes and every verdict
      truthfully states 5. That is not a silent failure — the derived
      `horizon_trading_days` on every artifact is what makes it loud — but it
      is a configuration question this control does not answer.
    * whether the panel's session index is the right one. Both paths take the
      trading-day axis from the data; a panel missing a session shortens the
      real-world span of a 21-session horizon in both, identically. The
      §4.12 trading-day contract test owns that.
    * the SCORING half. Sign, benchmark, count-matching and ranking are the
      planted/null pair's job, and a grader that scores a correct label
      backwards passes this control and fails that one.
    """
    if window.horizon_trading_days != declared_horizon_trading_days:
        raise GraderControlError(
            f"slot {slot!r}: the labels anchored at {window.start} span "
            f"{window.horizon_trading_days} session(s) to {window.end}, and the cycle "
            f"declared {declared_horizon_trading_days}. The horizon a verdict claims is "
            "derived from what was measured, so this is a real disagreement about the "
            "measurement and not a labelling slip: the cycle's verdicts are void (§10.1)."
        )

    measured = window.returns
    only_measured = sorted(set(measured) - set(reference))
    only_reference = sorted(set(reference) - set(measured))
    if only_measured or only_reference:
        raise GraderControlError(
            f"slot {slot!r}: the two label constructions disagree about WHICH names "
            f"settled between {window.start} and {window.end}. Present only in the "
            f"measured labels: {only_measured[:10]}; only in the reference: "
            f"{only_reference[:10]}. A name that exists in one construction and not the "
            "other is a selection scored against a benchmark drawn from a different "
            "cross-section; the cycle's verdicts are void (§10.1)."
        )

    worst_ticker, worst = "", 0.0
    for ticker, value in measured.items():
        other = reference[ticker]
        delta = abs(value - other)
        scale = max(abs(value), abs(other), 1.0)
        if delta / scale > worst:
            worst_ticker, worst = ticker, delta / scale
    if worst > LABEL_CONTROL_REL_TOL:
        raise GraderControlError(
            f"slot {slot!r}: the two label constructions disagree by {worst:.3e} "
            f"(relative) on {worst_ticker!r} over {window.start}..{window.end}, which is "
            f"beyond the {LABEL_CONTROL_REL_TOL:.0e} float tolerance. They read the same "
            "two closes and divide them, so a disagreement this size is a disagreement "
            "about WHICH closes. The cycle's verdicts are void (§10.1)."
        )

    return {
        "anchor": window.start,
        "settled_on": window.end,
        "horizon_trading_days": window.horizon_trading_days,
        "declared_horizon_trading_days": declared_horizon_trading_days,
        "n_names": len(measured),
        "max_relative_disagreement": worst,
        "tolerance": LABEL_CONTROL_REL_TOL,
    }


def score_selection(
    selection: tuple[str, ...],
    population: tuple[str, ...],
    returns: dict[str, float],
) -> tuple[float, dict[str, Any]]:
    """Realized alpha of the selection against the population it drew from.

    Count-matched by construction: the equal-weight mean of the population is
    the expectation of a random draw of any size from it, so the selection's
    equal-weight mean is compared against exactly what a coin-flip selector
    of the same size would have earned in expectation. No sampling, no seed,
    no Monte-Carlo variance in the benchmark.

    **Two unscoreable cases, and they are not the same event** (plan §4.4).
    Both used to raise a bare `ValueError`, so the only handler either could
    ever get was one that treated them alike:

    * every PICK unscoreable while the population is intact —
      :class:`SelectionMissError`. This arm legitimately has nothing to say
      on this date; it is a miss, a miss is data, and the other arms in the
      slot grade normally.
    * the POPULATION unscoreable — :class:`PopulationIntegrityError`. Every
      arm in the slot is benchmarked against it, so the cycle's shared inputs
      are compromised and the whole slot's run fails.

    The population is checked FIRST, deliberately. A cycle whose population
    has collapsed also has no settled picks, and reporting that as a miss
    would file a broken input under "this arm had nothing to say" — the
    precise confusion policy §3 forbids.
    """
    picked = [returns[t] for t in selection if t in returns]
    pool = [returns[t] for t in population if t in returns]
    if len(pool) < 2:
        raise PopulationIntegrityError(
            f"the population has {len(pool)} settled forward return(s) out of "
            f"{len(population)}; a benchmark drawn from fewer than two names is not a "
            "benchmark. Every arm in this slot is scored against it, so this is a "
            "compromised input for the whole cycle, not one arm's miss."
        )
    if not picked:
        raise SelectionMissError(
            f"none of the {len(selection)} selected names has a settled forward return, "
            f"while {len(pool)} of {len(population)} population names do. The selection "
            "cannot be scored, and scoring it as zero would credit a delisting as a flat "
            "month. Recorded as a MISS for this arm on this date (plan §4.4): the arm "
            "legitimately has nothing to say, and the slot's other arms grade normally."
        )
    selection_mean = sum(picked) / len(picked)
    population_mean = sum(pool) / len(pool)
    detail = {
        "selection_mean_ratio": selection_mean,
        "population_mean_ratio": population_mean,
        "n_selected_settled": len(picked),
        "n_selected": len(selection),
        "n_population_settled": len(pool),
        "n_population": len(population),
    }
    return selection_mean - population_mean, detail


def control_selection(
    kind: str,
    returns: dict[str, float],
    *,
    top_n: int,
    seed: int,
) -> tuple[str, ...]:
    """The planted and null controls' picks for one settled date.

    The planted arm ranks on ``IC * z(forward return) + sqrt(1 - IC^2) *
    noise``, so its ranking signal has a Pearson correlation with the
    realized return of :data:`CONTROL_PLANTED_IC` by construction rather than
    by hope. The null arm ranks on noise alone.

    Seeded from the trading day, so a replay of the same date reproduces the
    same controls — a control that moved between replays would make every
    replay diff unattributable.
    """
    names = sorted(returns)
    if len(names) < top_n:
        raise ValueError(
            f"cannot draw a control selection of {top_n} from {len(names)} settled names"
        )
    rng = random.Random(seed)
    if kind == "null":
        scored = [(rng.gauss(0.0, 1.0), name) for name in names]
    elif kind == "planted":
        values = [returns[n] for n in names]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = math.sqrt(variance)
        if std <= 0:
            raise GraderControlError(
                "every settled forward return is identical, so a planted edge cannot be "
                "planted. That is not a market condition — it is a broken panel."
            )
        weight = math.sqrt(max(0.0, 1.0 - CONTROL_PLANTED_IC**2))
        scored = [
            (
                CONTROL_PLANTED_IC * (returns[name] - mean) / std + weight * rng.gauss(0.0, 1.0),
                name,
            )
            for name in names
        ]
    else:
        raise ValueError(f"unknown control kind {kind!r}; the controls are planted and null")
    scored.sort(reverse=True)
    return tuple(name for _, name in scored[:top_n])


def series_from_verdicts(verdicts: dict[str, dict[str, float]]) -> dict[str, ArmSeries]:
    """`{arm_id: {trading_day: score}}` to the engine's series objects.

    Nothing is intersected here. Every arm keeps every date it scored, and
    the pairing happens per comparison inside the engine — which is the
    whole of the I9745 fix.
    """
    return {
        arm_id: ArmSeries(arm_id=arm_id, scores=dict(scores))
        for arm_id, scores in sorted(verdicts.items())
    }


def assert_controls_ordered(
    control_ids: dict[str, str],
    series_by_arm: dict[str, ArmSeries],
    *,
    slot: str,
) -> dict[str, Any]:
    """§10.1: planted must beat null on their own paired window.

    ``control_ids`` maps the control KIND to its REGISTERED arm id. Passing
    the registered ids rather than the `SlotSpec.control_arms` names is not a
    detail: an arm's identity is `{slot}:{name}:{spec_hash}` everywhere else
    in the system, and addressing a control by its bare name here would look
    up a series that no register row backs — which the engine refuses, and
    rightly.

    The paired window, not the raw means: the two controls are scored on the
    same dates, and comparing their unpaired averages would let a date one
    of them missed decide the comparison.

    The falsifiable claim is the SIGN, and the observed margin is reported
    beside it. A fixed absolute margin would be a threshold tuned to one
    replay window's dispersion — it would fire on a quiet month and pass on
    a violent one, which is a gate that measures volatility.
    """
    planted = control_ids.get("planted")
    null = control_ids.get("null")
    if planted is None or null is None:
        raise GraderControlError(
            f"slot {slot!r} does not declare both controls; a cycle without a "
            "positive control cannot establish that the grader measures anything"
        )
    missing = [a for a in (planted, null) if a not in series_by_arm]
    if missing:
        raise GraderControlError(
            f"control arm(s) {missing} were not scored this cycle. Controls are scored "
            "EVERY cycle (§10.1) — an unscored control is an unverified grader, and the "
            "cycle's verdicts have nothing standing behind them."
        )

    window = pair_on_common_window(series_by_arm[planted], series_by_arm[null], min_dates=1)
    if not window.measurable:
        raise GraderControlError(
            f"the controls share no scored dates ({window.unmeasurable_reason}). The "
            "grader cannot be checked this cycle, so the cycle's verdicts are void."
        )
    margin = window.mean_diff
    detail = {
        "planted_arm": planted,
        "null_arm": null,
        "n_paired_dates": window.n_dates,
        "margin_ratio": margin,
        "planted_mean_ratio": sum(window.scores_a) / len(window.scores_a),
        "null_mean_ratio": sum(window.scores_b) / len(window.scores_b),
        "planted_ic": CONTROL_PLANTED_IC,
    }
    if margin <= 0.0:
        raise GraderControlError(
            f"the planted control ({planted}) did not outrank the null control ({null}) "
            f"over their {window.n_dates} shared date(s): margin {margin:.6f}. The "
            f"planted arm's ranking signal is constructed with an IC of "
            f"{CONTROL_PLANTED_IC} against the realized return, so a grader that cannot "
            "see it cannot see a real edge either. The GRADER is broken and this "
            "cycle's verdicts are void (§10.1)."
        )
    return detail


def grade_slot(
    slot_spec: SlotSpec,
    *,
    as_of: dt.date,
    register: Any,
    control_ids: dict[str, str],
    series_by_arm: dict[str, ArmSeries],
    incumbent: str | None,
    preconditions: dict[str, list[ServingPrecondition]] | None = None,
    training: dict[str, TrainingStatus] | None = None,
) -> tuple[ArenaCycle, dict[str, Any]]:
    """Check the controls, then run the library cycle. Nothing else decides.

    The ladder, the paired windows, the confidence sequence, the Condorcet
    ranking, the pointer and the cap-with-grace retirement all live in
    `nousergon_lib.arena` and are CALLED. A slot re-implementing policy
    §§3-6 is a defect (policy §10), so this function's whole job is to
    assemble inputs, verify the harness, and hand over.
    """
    control_detail = assert_controls_ordered(control_ids, series_by_arm, slot=slot_spec.slot)

    cycle = run_cycle(
        config=slot_spec.arena,
        as_of=as_of.isoformat(),
        register=register,
        series_by_arm=series_by_arm,
        incumbent=incumbent,
        preconditions=preconditions,
        training=training,
    )

    # §10.1: controls never touch the pointer. The library decides the
    # pointer from the arms it was given, so the exclusion is asserted here
    # rather than assumed — a control that reached the pointer would be a
    # look-ahead arm serving production.
    if cycle.decision.champion is not None and cycle.decision.champion in set(control_ids.values()):
        raise GraderControlError(
            f"the pointer landed on {cycle.decision.champion!r}, which is a control arm. "
            "The planted control reads the realized forward return; serving it would "
            "be a look-ahead in production (§10.1)."
        )
    return cycle, control_detail


def write_shadow(store: Store, shadow: ShadowSelection) -> bytes:
    payload = json.dumps(shadow.to_dict(), indent=2, sort_keys=True).encode("utf-8")
    store.put_bytes(shadow_key(shadow.arm_id, shadow.trading_day), payload)
    return payload


def write_verdict(
    store: Store,
    *,
    arm_id: str,
    trading_day: str,
    slot: str,
    score: float,
    window: ForwardReturnWindow,
    benchmark: str,
    detail: dict[str, Any],
    control: bool,
) -> bytes:
    """One arm's settled score for one decision date.

    Takes the :class:`ForwardReturnWindow` the score was computed from, not
    a horizon integer. The horizon and the settle session on the artifact are
    then properties OF THE MEASUREMENT rather than values a caller supplied
    beside it — which is the whole of the `I9757` fix for this defect: a run
    whose labels spanned 5 sessions used to write `horizon_trading_days: 21`
    into every verdict, and no artifact anywhere disagreed.

    ``settled_on`` is additive on `verdict.v1`: the anchor was always on the
    record as `trading_day`, and the session the return settled at never was,
    so the artifact could not previously be checked against the panel at all.
    """
    payload = json.dumps(
        {
            "schema_version": "verdict.v1",
            "arm_id": arm_id,
            "slot": slot,
            "trading_day": trading_day,
            "score_ratio": score,
            "horizon_trading_days": window.horizon_trading_days,
            "settled_on": window.end,
            "benchmark": benchmark,
            "control": control,
            "detail": detail,
        },
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    store.put_bytes(verdict_key(arm_id, trading_day), payload)
    return payload


def training_ok(arm_ids: list[str]) -> dict[str, TrainingStatus]:
    """A sound-fit status for every arm in a slot whose arms are not fitted.

    U and R rank deterministically on the feature layer; there is no fit to
    vouch for. The statuses are supplied rather than passing ``training=None``
    so that the day an arm in these slots DOES carry a fit, the shape it must
    report is already the one the cycle reads.
    """
    return {
        arm: TrainingStatus(arm_id=arm, ok=True, reason="deterministic ranker; no fitted state")
        for arm in arm_ids
    }


def raise_training_integrity(arm_id: str, cause: BaseException) -> None:
    """A compromised input fails the whole slot's run (plan §4.4)."""
    raise TrainingIntegrityError(
        f"arm {arm_id!r} could not be produced from this cycle's inputs: "
        f"{type(cause).__name__}: {cause}. Arms in a slot share an input substrate, so a "
        "defect that spoils one arm's inputs is evidence the cycle's inputs are "
        "compromised. This is a TASK FAILURE, not a miss and not a degraded run "
        "(Brian ruling 2026-08-29)."
    ) from cause
