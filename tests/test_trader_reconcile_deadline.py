"""`trader.reconcile` has an ABSENCE deadline (alpha-engine-config-I12353).

**The measured defect.** The 2026-10-14 and 10-15 broker reconciliations never
wrote a manifest and nothing paged. The FAILURE condition reads a manifest, so
it had nothing to read. The ABSENCE condition reads `deadline`, and the row's
was `null`, held there by crucible-PR412 until the reconcile timing settled.

**Why `next_open_plus 0`.** The reconcile for trading day d runs the same
evening, started by the 16:45 ET box timer or by the post-close SF drain.
Manifests landed 19:05-20:45 ET through 10-21 and at 16:02 ET from 10-22 on. A
same-evening `close_plus` deadline has to fall before the 21:00 ET
`alerts.sweep`, which is within 15 minutes of the timer-path worst case. Due at
the next session's open, a miss pages at the next sweep, and a weekend or a
holiday moves the deadline with the calendar.
"""

from __future__ import annotations

import datetime as dt

import pytest

from crucible.alerts import evaluate_absence
from crucible.components import NYSE_TZ, Component, Deadline, load_registry
from crucible.keys import manifest_key
from crucible.release_history import UNDECLARED, Declaration, row_has_a_starter
from crucible.store import LocalStore

JOB = "trader.reconcile"

#: Friday 2026-10-16; the next session is Monday 10-19.
FRIDAY = dt.date(2026, 10, 16)
MONDAY = dt.date(2026, 10, 19)


def _et(day: dt.date, hh: int, mm: int) -> dt.datetime:
    return dt.datetime.combine(day, dt.time(hh, mm), tzinfo=NYSE_TZ).astimezone(dt.UTC)


def _row() -> Component:
    return load_registry()[JOB]


def _absent(store: LocalStore, now: dt.datetime, **kwargs) -> list[str]:
    return [
        page.trading_day.isoformat()
        for page in evaluate_absence(store, now=now, **kwargs)
        if page.job == JOB
    ]


class TestTheRowIsScheduled:
    def test_it_names_the_box_timer_that_starts_it(self) -> None:
        row = _row()
        assert row.lifecycle == "ACTIVE"
        assert row.scheduled
        assert row.dispatch == "box-timer"
        assert row.dispatch_unit == "alpha-engine-trader-reconcile.timer"
        assert row.absence_watched_by == "alerts.sweep"

    def test_its_deadline_is_the_next_open(self) -> None:
        assert _row().deadline == Deadline(anchor="next_open_plus", cadence="daily", offset_hours=0)

    def test_a_release_declaring_it_counts_as_a_starter(self) -> None:
        assert row_has_a_starter(
            {
                "lifecycle": "ACTIVE",
                "dispatch": "box-timer",
                "deadline": {"anchor": "next_open_plus", "cadence": "daily", "offset_hours": 0},
            }
        )


class TestTheDeadlineReadsTheTradingCalendar:
    @pytest.mark.parametrize(
        ("trading_day", "session"),
        [
            (dt.date(2026, 10, 15), dt.date(2026, 10, 16)),
            (FRIDAY, MONDAY),
            # Thanksgiving Thursday is closed; Wednesday's is due Friday's open.
            (dt.date(2026, 11, 25), dt.date(2026, 11, 27)),
        ],
    )
    def test_it_is_due_at_0930_et_on_the_next_session(
        self, trading_day: dt.date, session: dt.date
    ) -> None:
        assert _row().deadline.due_at(trading_day) == _et(session, 9, 30)

    def test_it_follows_new_york_time_across_the_dst_change(self) -> None:
        """US DST ends 2026-11-01: Friday 10-30's deadline is Monday 11-02
        09:30 EST = 14:30Z, where Friday 10-16's was 13:30Z under EDT."""
        assert _row().deadline.due_at(dt.date(2026, 10, 30)) == dt.datetime(
            2026, 11, 2, 14, 30, tzinfo=dt.UTC
        )
        assert _row().deadline.due_at(FRIDAY) == dt.datetime(2026, 10, 19, 13, 30, tzinfo=dt.UTC)


class TestAMissingReconcilePagesOnlyOnceItIsLate:
    def test_the_same_evening_raises_no_page(self, tmp_path) -> None:
        assert _absent(LocalStore(tmp_path), _et(FRIDAY, 21, 0)) == []

    def test_the_weekend_raises_no_page(self, tmp_path) -> None:
        assert _absent(LocalStore(tmp_path), _et(dt.date(2026, 10, 18), 21, 0)) == []

    def test_the_next_session_after_the_open_pages(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        assert _absent(store, _et(MONDAY, 9, 29)) == []
        assert _absent(store, _et(MONDAY, 9, 31)) == [FRIDAY.isoformat()]

    def test_a_written_manifest_clears_the_day(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        store.put_bytes(manifest_key(JOB, FRIDAY.isoformat()), b"{}")
        assert _absent(store, _et(MONDAY, 9, 31)) == []

    def test_a_day_before_the_release_declared_it_is_not_paged_after_the_fact(
        self, tmp_path
    ) -> None:
        """The already-known 10-14/10-15 misses stay quiet: no release in force
        at their deadlines declared a starter for this row."""

        class _NothingDeclared:
            def declaration(self, name: str, instant: dt.datetime) -> Declaration:
                return Declaration(UNDECLARED, f"no release declared {name}")

        assert (
            _absent(LocalStore(tmp_path), _et(MONDAY, 9, 31), release_history=_NothingDeclared())
            == []
        )
