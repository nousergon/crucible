"""Declared SOURCE BOUNDARIES: where a catalogue column's history legitimately
ends, so a dead band before one is a BOUNDARY and a dead band after one is a
DEFECT.

Normative source: `alpha-engine-config-I10721`, and the point-in-time knowledge
rules this file restates from `crucible.data.point_in_time`.

Why this exists
===============

`check_feature_layer_completeness` graded every dead band the same way, and a
detector whose RED means two different things gets ignored. Measured
2026-09-17 against the live layer `features/v553618c991dd/` (1,180 sessions):
the reading was RED and its `detail` named session **2022-01-03** — the first
session of history, where no fundamentals snapshot can be admissible because
the earliest EDGAR label IS 2022-01-03 and the rule is `label < session`. That
band is unfixable by construction. It is also, structurally, the band the
reading will always report: the worst-session tie-break picks the session with
the most dead columns, and a head-of-history boundary has every
source-derived column dead at once, which no downstream defect can exceed.

So the RED an operator saw was a permanent, correct, unactionable statement
about the start of history — and sitting inside the same RED, invisible, were
92 sessions (2025-07-09..2025-11-14) whose fundamentals were dead because an
EDGAR re-heal chunk died four minutes in. The hole survived because the row it
would have lit was already lit.

A boundary is DECLARED, never inferred from the data
====================================================

The tempting fix is to infer "this band is expected" from the artifacts —
every session in both bands records `fundamental: predates_source` in its
`data/{date}/coverage.json`, so an inference over the coverage records calls
BOTH bands by-design and closes the detector's eyes completely. That is the
exact trap: the 92-session hole reports `predates_source` because it was
compiled against the SUPERSEDED v1 snapshot source
(`point_in_time.snapshot_id == "v1-snapshots:..."`, whose earliest label is
2026-03-27) while every session around it was compiled against
`edgar-filing-date:...`, whose history starts 2022-01-03. A producer's own
account of why it measured nothing cannot distinguish "there was nothing to
measure" from "I was pointed at the wrong source".

So each boundary here is a written claim with its derivation and its evidence,
and `tests/test_feature_source_boundaries.py` holds the claim to the rules in
`crucible.data.point_in_time`. A boundary that moves — a deeper EDGAR
backfill, a dated GICS source, a filing-date 13F feed — moves HERE, in a diff,
with the reason. That is the difference between a declared boundary and a
suppression list (`crucible/AGENTS.md` rule 4): a boundary names a date before
which a column cannot exist and says why; it never excuses a column on a
session at or after it.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
from dataclasses import dataclass

from crucible.features.compute import PILLAR_COMPONENTS

__all__ = [
    "DECLARED_SOURCE_BOUNDARIES",
    "SourceBoundary",
    "boundary_for_column",
    "explain_dead_columns",
]


@dataclass(frozen=True)
class SourceBoundary:
    """One input group's first measurable session, and the columns it bounds.

    ``first_measurable_session`` is the EARLIEST NYSE session on which this
    group can carry a value. A catalogue column in ``columns`` that is dead on
    a session STRICTLY BEFORE it is explained by this boundary. The same
    column dead on that session or any later one is a defect, and nothing in
    this module softens that.
    """

    group: str
    first_measurable_session: dt.date
    columns: frozenset[str]
    reason: str
    evidence: str

    def explains(self, session: dt.date, column: str) -> bool:
        return column in self.columns and session < self.first_measurable_session

    def to_dict(self) -> dict[str, object]:
        return {
            "group": self.group,
            "first_measurable_session": self.first_measurable_session.isoformat(),
            "columns": sorted(self.columns),
            "reason": self.reason,
            "evidence": self.evidence,
        }


#: The eleven vendor fields plus the one column derived from them on the
#: session row alone (`sustainable_growth_rate_ratio` = roe x (1 - payout)).
#: Kept as a literal rather than imported from
#: `crucible.data.point_in_time.FUNDAMENTAL_FIELD_COLUMNS` so that a field
#: ADDED there does not silently inherit an excuse for the whole pre-boundary
#: band; `tests/test_feature_source_boundaries.py` asserts the two agree.
_FUNDAMENTAL_COLUMNS: frozenset[str] = frozenset(
    {
        "roe_ratio",
        "debt_to_equity_div2_ratio",
        "gross_margin_ratio",
        "current_ratio_div3_ratio",
        "pe_div30_ratio",
        "pb_div5_ratio",
        "fcf_yield_ratio",
        "revenue_growth_3y_ratio",
        "eps_growth_3y_ratio",
        "capex_growth_5y_ratio",
        "payout_ratio",
        "sustainable_growth_rate_ratio",
    }
)

_INSTITUTIONAL_COLUMNS: frozenset[str] = frozenset(
    {
        "n_funds_increasing_raw",
        "n_funds_decreasing_raw",
        "institutional_accumulation_raw",
    }
)

#: The first session each group can be measured for. Both dates are DERIVED
#: from the point-in-time knowledge rules and then MEASURED against the live
#: layer; the derivation is in `reason`, the measurement in `evidence`.
_GROUP_BOUNDARIES: tuple[SourceBoundary, ...] = (
    SourceBoundary(
        group="fundamental",
        first_measurable_session=dt.date(2022, 1, 4),
        columns=_FUNDAMENTAL_COLUMNS,
        reason=(
            "`FilingDatePointInTimeSource` admits the EDGAR session file labelled L for "
            "session S only when L < S, and the producer's backfill starts at label "
            "2022-01-03 — so 2022-01-03 itself has no admissible snapshot and 2022-01-04 "
            "is the first session that does. A one-session head-of-history boundary, not "
            "a gap: deepening it needs an EDGAR backfill before 2022-01-03, which is a "
            "producer change, never a heal."
        ),
        evidence=(
            "measured 2026-09-17: the data bucket's "
            "`fundamentals_pit/edgar/v1/sessions/` prefix holds 1,180 objects, earliest "
            "`2022-01-03.parquet` (`crucible.keys.edgar_fundamentals_session_key`); "
            "crucible/data/2022-01-03/coverage.json reads `fundamental: predates_source` "
            "and every session from 2022-01-04 on reads `measured`"
        ),
    ),
    SourceBoundary(
        group="institutional",
        first_measurable_session=dt.date(2022, 11, 15),
        columns=_INSTITUTIONAL_COLUMNS,
        reason=(
            "13F accumulation is public only from the quarter end plus the 45-day Form "
            "13F filing deadline (`THIRTEEN_F_FILING_LAG_DAYS`, 17 CFR 240.13f-1), and "
            "the earliest quarter table in the store is 2022Q3: 2022-09-30 + 45 days = "
            "2022-11-14, admissible for sessions strictly after it. Deepening it needs "
            "earlier quarterly 13F tables, never a heal."
        ),
        evidence=(
            "measured 2026-09-17 over all 1,180 sessions of features/v553618c991dd/: "
            "`institutional_accumulation_raw` is >= 50% non-null from 2022-11-15 and "
            "null on every session before it; the regulatory date is the declared "
            "convention in `crucible.data.point_in_time.quarter_knowledge_date`"
        ),
    ),
)


def _pillar_boundaries() -> tuple[SourceBoundary, ...]:
    """Each pillar inherits the LATEST boundary among its components.

    Derived from `PILLAR_COMPONENTS` rather than listed, so a pillar whose
    recipe changes cannot keep an excuse its components no longer justify. A
    pillar needs every component (`crucible.features.compute` composes them
    over the full set), so one bounded component bounds the pillar, and the
    binding one is the latest.
    """
    by_group: dict[str, set[str]] = {}
    for pillar, components in PILLAR_COMPONENTS.items():
        binding: SourceBoundary | None = None
        for column, _weight, _invert in components:
            for boundary in _GROUP_BOUNDARIES:
                if column not in boundary.columns:
                    continue
                if (
                    binding is None
                    or boundary.first_measurable_session > binding.first_measurable_session
                ):
                    binding = boundary
        if binding is not None:
            by_group.setdefault(binding.group, set()).add(pillar)

    return tuple(
        SourceBoundary(
            group=boundary.group,
            first_measurable_session=boundary.first_measurable_session,
            columns=boundary.columns | frozenset(by_group.get(boundary.group, set())),
            reason=boundary.reason,
            evidence=boundary.evidence,
        )
        for boundary in _GROUP_BOUNDARIES
    )


#: The declared boundaries, group columns plus every pillar they bind.
DECLARED_SOURCE_BOUNDARIES: tuple[SourceBoundary, ...] = _pillar_boundaries()


def boundary_for_column(column: str) -> SourceBoundary | None:
    """The boundary binding ``column``, or ``None`` when it has none.

    A column bound by more than one group carries the LATEST, which is the one
    that actually gates it — `stewardship_pillar_pct` reads `payout_ratio`
    (fundamental, 2022-01-04) and `institutional_accumulation_raw`
    (institutional, 2022-11-15), and it is measurable only from the later.
    """
    binding: SourceBoundary | None = None
    for boundary in DECLARED_SOURCE_BOUNDARIES:
        if column not in boundary.columns:
            continue
        if binding is None or boundary.first_measurable_session > binding.first_measurable_session:
            binding = boundary
    return binding


def explain_dead_columns(
    session: str, dead_columns: Iterable[str]
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[SourceBoundary, ...]]:
    """Split ``dead_columns`` on ``session`` into (unexplained, explained, boundaries).

    ``unexplained`` is the defect: catalogue columns dead on a session at or
    after every boundary that could have covered them, or dead with no
    declared boundary at all. It is the ONLY thing that may turn the
    completeness reading RED.

    ``session`` is an ISO date. A session string this cannot parse is not
    silently treated as explained — it raises, because a session key the
    detector cannot place on the calendar is a store defect and reading it as
    "no boundary applies" would hide exactly the band this module exists to
    expose.
    """
    day = dt.date.fromisoformat(session)
    unexplained: list[str] = []
    explained: list[str] = []
    applied: dict[str, SourceBoundary] = {}
    for column in dead_columns:
        boundary = boundary_for_column(column)
        if boundary is not None and boundary.explains(day, column):
            explained.append(column)
            applied[boundary.group] = boundary
        else:
            unexplained.append(column)
    return (
        tuple(sorted(unexplained)),
        tuple(sorted(explained)),
        tuple(sorted(applied.values(), key=lambda b: b.group)),
    )
