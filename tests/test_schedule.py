"""The plan §6.1 milestone table, as data — `alpha-engine-config-I9914`.

Normative source: plan §6.1 ("Expedited schedule — full cutover in three
weeks"). `crucible/schedule.py` carries the table as a literal because
`crucible` is public and the plan is a private-repo doc it cannot read at
runtime (repository-tiering-policy); this file pins that literal against a
SECOND, independently-typed transcription of the plan's own table below, so
an edit to one without the other fails here rather than silently drifting.

Fixed date literals throughout (`crucible/AGENTS.md`, "a test whose subject
moves with the clock stops testing the same thing").
"""

from __future__ import annotations

import datetime as dt

import pytest

from crucible.gate import PHASES
from crucible.schedule import MILESTONES, Milestone

#: The plan §6.1 table's dates, copied independently of `schedule.py` — the
#: literal source of truth this test checks `MILESTONES` against. Day 0 (the
#: ruling day) is 2026-09-01, a Tuesday.
_PLAN_TABLE: dict[str, dt.date] = {
    "day5_replays": dt.date(2026, 9, 6),  # day 5
    "days6_7_scheduler": dt.date(2026, 9, 8),  # days 6-7, the later of the two
    "live1": dt.date(2026, 9, 12),  # Sat 09-12
    "live2_cutover": dt.date(2026, 9, 19),  # Sat 09-19
    "live3": dt.date(2026, 9, 26),  # Sat 09-26
}


class TestTheTablePinsThePlanLiterally:
    def test_every_milestone_id_is_in_the_plan_table(self) -> None:
        assert {m.id for m in MILESTONES} == set(_PLAN_TABLE), (
            "MILESTONES and this test's independent transcription of plan §6.1 "
            "name different milestones — one of the two was edited without the other"
        )

    @pytest.mark.parametrize("milestone", MILESTONES, ids=lambda m: m.id)
    def test_the_plan_date_matches_the_independently_typed_literal(
        self, milestone: Milestone
    ) -> None:
        assert milestone.plan_date == _PLAN_TABLE[milestone.id], (
            f"{milestone.id}: schedule.py says {milestone.plan_date}, plan §6.1 (as "
            f"independently copied here) says {_PLAN_TABLE[milestone.id]}"
        )

    def test_five_milestones_not_four_not_six(self) -> None:
        """Plan §6.1's table has exactly five rows. A sixth or a missing one
        is a silent edit to the schedule this board reports."""
        assert len(MILESTONES) == 5


class TestEveryMilestoneNamesARealPhase:
    @pytest.mark.parametrize("milestone", MILESTONES, ids=lambda m: m.id)
    def test_phase_id_is_registered_in_gate_phases(self, milestone: Milestone) -> None:
        assert milestone.phase_id in {p.id for p in PHASES}, (
            f"{milestone.id} names phase {milestone.phase_id!r}, which "
            "crucible.gate.PHASES does not register — a milestone naming a phase "
            "that does not exist could never read MET"
        )

    def test_day5_reflects_phase1(self) -> None:
        """Day 5's own gate is phase 1's — 5 replay Saturdays, all R/M/S
        arms graded (plan §6.1's own words for this row)."""
        (day5,) = [m for m in MILESTONES if m.id == "day5_replays"]
        assert day5.phase_id == "phase1"


class TestOrderingAndUniqueness:
    def test_milestones_are_in_plan_date_order(self) -> None:
        dates = [m.plan_date for m in MILESTONES]
        assert dates == sorted(dates)

    def test_no_two_milestones_share_an_id(self) -> None:
        ids = [m.id for m in MILESTONES]
        assert len(ids) == len(set(ids))


class TestTradingDayAnchoring:
    """AGENTS.md rule 3: every key binds to the last COMPLETED trading
    session, never to a non-trading calendar day — "a Saturday weekly run is
    keyed to Friday's close."
    """

    def test_a_sunday_plan_date_anchors_to_the_preceding_friday(self) -> None:
        """Day 5 (2026-09-06) is a Sunday; 2026-09-04 is the Friday before
        it and is a real NYSE session (no holiday that week)."""
        (day5,) = [m for m in MILESTONES if m.id == "day5_replays"]
        assert day5.trading_day == dt.date(2026, 9, 4)

    def test_a_saturday_plan_date_anchors_to_the_preceding_friday(self) -> None:
        (live1,) = [m for m in MILESTONES if m.id == "live1"]
        assert live1.trading_day == dt.date(2026, 9, 11)

    def test_a_weekday_plan_date_that_is_already_a_trading_day_is_unchanged(self) -> None:
        """days6_7's own date, 2026-09-08, is a Tuesday and a real session —
        anchoring must be idempotent, not walk back an extra day."""
        (days67,) = [m for m in MILESTONES if m.id == "days6_7_scheduler"]
        assert days67.trading_day == dt.date(2026, 9, 8)

    def test_anchoring_is_idempotent(self) -> None:
        """Resolving an already-resolved trading day returns it unchanged —
        `crucible.calendar.resolve_trading_day`'s own documented contract,
        exercised here rather than re-asserted."""
        for milestone in MILESTONES:
            anchored = milestone.trading_day
            twice = Milestone(
                id=milestone.id,
                plan_date=anchored,
                what=milestone.what,
                phase_id=milestone.phase_id,
            ).trading_day
            assert twice == anchored


class TestTheTableGuardsAreShownFiring:
    """`_check_milestones` runs at import and used to be two bare `if`s that
    no test could reach (they carried `pragma: no cover`). A guard nobody has
    made fail is a guard nobody knows works; each refusal is exercised here."""

    def test_a_duplicate_id_is_refused(self) -> None:
        from crucible.schedule import _check_milestones

        a = Milestone(id="same", plan_date=dt.date(2026, 9, 6), what="a", phase_id="phase1")
        b = Milestone(id="same", plan_date=dt.date(2026, 9, 12), what="b", phase_id="phase2")
        with pytest.raises(ValueError, match="share an id"):
            _check_milestones((a, b))

    def test_an_out_of_order_table_is_refused(self) -> None:
        from crucible.schedule import _check_milestones

        later = Milestone(id="l", plan_date=dt.date(2026, 9, 12), what="l", phase_id="phase2")
        earlier = Milestone(id="e", plan_date=dt.date(2026, 9, 6), what="e", phase_id="phase1")
        with pytest.raises(ValueError, match="plan_date order"):
            _check_milestones((later, earlier))

    def test_the_committed_table_passes_its_own_guard(self) -> None:
        from crucible.schedule import _check_milestones

        _check_milestones(MILESTONES)
