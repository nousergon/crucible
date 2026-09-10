"""A deadline says WHEN a manifest is due; the cadence says WHICH DAYS it is
due on. Before the cadence existed, only the first question had an answer.

**The measured defect (2026-09-08).** `crucible.alerts.evaluate_absence`
graded every scheduled registry row against every trading day in its
catch-up window. Eight rows declare `schedule: weekly, Saturday` — prose,
read by nothing — so on each of the four weekdays a weekly job was never
going to run, each of them was reported absent. The ABSENCE page for trading
day 2026-09-08 named twenty-two members; seven of them (`console`, `drift`,
`experiment.grade`, `experiment.run`, `report`, `weekly`, `data.weekly`)
were that, and would have recurred every week for as long as the system ran.

A page that is mostly noise on most days is how the one true page stops
being read, which is exactly what §4.6's two-conditions design exists to
prevent — so this is a defect in the alerter, never a case for a mute.
"""

from __future__ import annotations

import datetime as dt

import pytest

from crucible.alerts import evaluate_absence
from crucible.calendar import NonTradingDayKeyError, is_week_final_trading_day
from crucible.components import CADENCES, Deadline, load_registry
from crucible.store import LocalStore

#: A Tuesday. Mid-week: a weekly row binds to Friday and is not due here.
MIDWEEK = dt.date(2026, 9, 8)
#: The Friday that closes MIDWEEK's week.
WEEK_FINAL = dt.date(2026, 9, 11)

WEEKLY_ROWS = frozenset(
    {"console", "data.weekly", "drift", "experiment.grade", "experiment.run", "report", "weekly"}
)


class TestTheWeekFinalSessionIsReadFromTheCalendar:
    def test_a_friday_closes_its_week(self) -> None:
        assert is_week_final_trading_day(dt.date(2026, 9, 11))

    def test_a_tuesday_does_not(self) -> None:
        assert not is_week_final_trading_day(MIDWEEK)

    def test_a_short_holiday_week_moves_the_final_session_off_friday(self) -> None:
        """2026-07-03 is the observed Independence Day holiday, so that week's
        last session is Thursday the 2nd. A `weekday() == 4` predicate would
        skip the week entirely — which is the whole reason this is asked of
        the NYSE calendar rather than of the date."""
        assert is_week_final_trading_day(dt.date(2026, 7, 2))
        assert not is_week_final_trading_day(dt.date(2026, 7, 1))

    def test_a_non_session_is_a_raise_not_a_false(self) -> None:
        """A cadence question about a day the market never opened is a caller
        bug. Answering `False` would let it pass as 'not due today'."""
        with pytest.raises(NonTradingDayKeyError):
            is_week_final_trading_day(dt.date(2026, 9, 12))  # a Saturday


class TestEveryDeadlineDeclaresItsCadence:
    def test_the_registry_leaves_none_undeclared(self) -> None:
        for name, component in load_registry().items():
            if component.deadline is None:
                continue
            assert component.deadline.cadence in CADENCES, name

    def test_the_prose_schedule_and_the_machine_cadence_agree(self) -> None:
        """`schedule` is prose and `cadence` is wiring — the same split
        `schedule` and `dispatch` already carry. They may not disagree: a row
        reading 'weekly, Saturday' while graded daily is precisely the state
        that produced the 2026-09-08 page."""
        for name, component in load_registry().items():
            if component.deadline is None:
                continue
            assert component.schedule is not None
            says_weekly = component.schedule.lower().startswith("weekly")
            assert says_weekly == (component.deadline.cadence == "weekly"), (
                f"{name}: schedule reads {component.schedule!r} but its deadline cadence "
                f"is {component.deadline.cadence!r}. One of the two is wrong, and the "
                "prose is not the one anything reads."
            )

    def test_a_deadline_row_without_a_cadence_is_refused(self) -> None:
        """Not defaulted. A default would let an omission read as a
        declaration, which is the same rule every signal class follows."""
        with pytest.raises(ValueError, match="declares no `cadence`"):
            Deadline.from_yaml({"anchor": "close_plus", "offset_hours": 3})

    def test_a_cadence_outside_the_closed_set_is_refused(self) -> None:
        with pytest.raises(ValueError, match="is not a deadline cadence"):
            Deadline(anchor="close_plus", cadence="fortnightly", offset_hours=3)


class TestAWeeklyRowIsNotPagedMidWeek:
    def _absent_jobs(self, tmp_path, trading_day: dt.date) -> set[str]:
        """Every job the absence condition names for ``trading_day``, against
        an EMPTY store — so nothing is present and the only thing deciding
        which rows appear is whether they were due at all."""
        store = LocalStore(tmp_path)
        # `days_to_evaluate` reads the sweep's own manifests to bound the
        # window; with none, it evaluates exactly today's trading day, which
        # is what this test wants to pin.
        pages = evaluate_absence(store, now=self._close_of(trading_day))
        return {page.job for page in pages}

    @staticmethod
    def _close_of(trading_day: dt.date) -> dt.datetime:
        """A moment on the calendar day AFTER ``trading_day``, late enough
        that every deadline in the registry for it has passed, while still
        resolving to ``trading_day`` as the current session."""
        return dt.datetime.combine(
            trading_day + dt.timedelta(days=1), dt.time(23, 30), tzinfo=dt.UTC
        )

    def test_no_weekly_row_appears_on_a_midweek_trading_day(self, tmp_path) -> None:
        assert not (self._absent_jobs(tmp_path, MIDWEEK) & WEEKLY_ROWS)

    def test_every_weekly_row_appears_on_the_week_final_session(self, tmp_path) -> None:
        """The other half of the same assertion. A cadence that only ever
        suppressed would be a suppression collection (§11.1) wearing a field
        name; this is what makes it a schedule."""
        assert WEEKLY_ROWS <= self._absent_jobs(tmp_path, WEEK_FINAL)

    def test_the_daily_rows_are_paged_on_both(self, tmp_path) -> None:
        """`data.daily` is due every trading day and the cadence must not
        have quietly narrowed anything else."""
        assert "data.daily" in self._absent_jobs(tmp_path, MIDWEEK)
        assert "data.daily" in self._absent_jobs(tmp_path, WEEK_FINAL)
