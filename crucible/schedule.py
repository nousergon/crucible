"""Plan §6.1's expedited-cutover milestone table, as data.

Normative source: plan §6.1 ("Expedited schedule — full cutover in three
weeks"), `alpha-engine-config/private-docs/crucible_v2_rebuild_plan_260901.md`,
copied literally on 2026-09-03 for `alpha-engine-config-I9914`. `crucible` is
a public repo (`repository-tiering-policy`) and the plan is a private-repo
doc, so this module does NOT read it at runtime — the five milestones below
are literals, and this docstring names the exact section and date they were
copied from so drift is checkable by a human. `alpha-engine-config`'s own
docs-PR carries the private-side pin against the plan's literal table
(`tests/test_schedule.py` pins this module's copy against a second, embedded
transcription of the same table, so the two cannot silently diverge without a
failing test in THIS repo).

**Informational, never a gate input.** Plan §6.1's own words: "Weeks are
sequencing, not commitments. The gates are." A milestone tells a reader WHEN
the plan expected a phase's gate to read a certain way; it is never a page
condition (plan §4.6 admits exactly two, absence and failure, and a schedule
row is neither) and nothing in `crucible.gate` reads it. `crucible.board`
renders each milestone as a REFLECTION of the phase gate reading it names —
never a second, competing measurement of the same phase.

Every milestone names the `crucible.gate.PHASES` id whose ladder row it
reflects. A milestone whose phase carries no registered gate yet (a `PHASES`
entry with `gate=None`, true of phases 2-5 until their clause lists land)
reflects the ladder's own honest `UNMEASURED` — the milestone invents nothing.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from crucible.calendar import resolve_trading_day

__all__ = ["MILESTONES", "Milestone"]


@dataclass(frozen=True)
class Milestone:
    """One row of the plan §6.1 table.

    ``plan_date`` is the CALENDAR date the plan names. Several of the five are
    not trading days — day 5 is a Sunday, every live Saturday is a Saturday —
    and `crucible/AGENTS.md` rule 3 binds every reading to the last COMPLETED
    trading session, never to a non-trading calendar day: "a Saturday weekly
    run is keyed to Friday's close". :attr:`trading_day` is that anchor.

    ``phase_id`` is the `crucible.gate.PHASES` id whose ladder row this
    milestone quotes — never a second, independent gate evaluation of its own.
    """

    id: str
    plan_date: dt.date
    what: str
    phase_id: str

    @property
    def trading_day(self) -> dt.date:
        """``plan_date`` anchored to the last completed trading session.

        `crucible.calendar.resolve_trading_day` is idempotent, so a
        ``plan_date`` that is already a trading day resolves to itself and a
        weekend or holiday date walks back to the session it binds to. Anchored
        at 23:59 on ``plan_date`` — the latest possible moment of that
        calendar day — so a `plan_date` that IS a trading day never walks back
        an extra session past itself.
        """
        return resolve_trading_day(dt.datetime.combine(self.plan_date, dt.time(23, 59)))


#: Plan §6.1's table, copied literally 2026-09-03. Day 0 (the ruling day) is
#: 2026-09-01 (a Tuesday); every other row below is that plan's own day count
#: measured from it, converted to the calendar date it lands on.
MILESTONES: tuple[Milestone, ...] = (
    Milestone(
        id="day5_replays",
        plan_date=dt.date(2026, 9, 6),
        what=(
            "day 5: 5 replay Saturdays ok locally, all R/M/S arms graded; deploy "
            "pointer flips on a real smoke"
        ),
        phase_id="phase1",
    ),
    Milestone(
        id="days6_7_scheduler",
        plan_date=dt.date(2026, 9, 8),
        what=("days 6-7: 5 unattended scheduler runs ok, 0 human mutating calls, page count <= 1"),
        phase_id="phase2",
    ),
    Milestone(
        id="live1",
        plan_date=dt.date(2026, 9, 12),
        what="live Saturday #1 on v2 — first-attempt ok",
        phase_id="phase2",
    ),
    Milestone(
        id="live2_cutover",
        plan_date=dt.date(2026, 9, 19),
        what="live Saturday #2 — first-attempt ok, then cutover the same day",
        phase_id="phase2",
    ),
    Milestone(
        id="live3",
        plan_date=dt.date(2026, 9, 26),
        what=("live Saturday #3, confirmation — AWS month-to-date on track for <= $70"),
        phase_id="phase2",
    ),
)


def _check_milestones(milestones: tuple[Milestone, ...]) -> None:
    """Refuse a milestone table with a duplicate id or out of plan-date order.

    A function rather than two import-time `if`s so `tests/test_schedule.py`
    can show each refusal firing; the module still calls it at import, so a
    bad table fails the process at start-up exactly as before.
    """
    if len({m.id for m in milestones}) != len(milestones):
        raise ValueError("two §6.1 milestones share an id — the board would render one row twice")
    if list(milestones) != sorted(milestones, key=lambda m: m.plan_date):
        raise ValueError(
            "MILESTONES is not in plan_date order — the board and the morning report "
            "both render this table in its declared order, and an out-of-order table "
            "would render the schedule out of calendar order with no sign why"
        )


_check_milestones(MILESTONES)
