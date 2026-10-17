"""`trader.session` has an ABSENCE deadline, and it reads the trading calendar
(alpha-engine-config-I12353).

**The measured defect.** From 2026-10-14 to 10-15, four v2 trader unit runs on
the executor box ended `ActiveState=failed Result=exit-code` and nothing
paged. The 10-14 session died in its venv build (crucible-trader#39), before
it wrote any manifest. The FAILURE condition reads a manifest, so it had
nothing to read. The ABSENCE condition reads `deadline`, and the row's was
`null`. A person found the lost paper day by reading S3 the next night.

**Why a third anchor.** The session binds to the LAST CLOSED session, so the
manifest for trading day d is written on the NEXT session's morning: Monday
10:45 ET for a Friday d, and Tuesday when that Monday is a holiday. Measured
2026-10-02..15, every `ok` manifest landed at 14:45-14:47Z on the next
session. `close_plus` counts calendar hours from d's close, so no single
offset fits both a Thursday d and a Friday d. `next_calendar_day_at` would
fall on Saturday. `next_open_plus` resolves the next session from the
calendar, so a weekend or a holiday moves the deadline, and nothing has to be
muted.
"""

from __future__ import annotations

import datetime as dt

import pytest

from crucible.alerts import evaluate_absence
from crucible.components import DISPATCHES, NYSE_TZ, Component, Deadline, load_registry
from crucible.keys import manifest_key
from crucible.release_history import UNDECLARED, Declaration, row_has_a_starter
from crucible.store import LocalStore

JOB = "trader.session"

#: Friday 2026-10-16. Its session runs on Monday 10-19.
FRIDAY = dt.date(2026, 10, 16)
MONDAY = dt.date(2026, 10, 19)


def _et(day: dt.date, hh: int, mm: int) -> dt.datetime:
    return dt.datetime.combine(day, dt.time(hh, mm), tzinfo=NYSE_TZ).astimezone(dt.UTC)


def _row() -> Component:
    return load_registry()[JOB]


def _trader_absent(store: LocalStore, now: dt.datetime, **kwargs) -> list[str]:
    """The trader.session ABSENCE pages at ``now``, each named by trading day."""
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
        assert row.dispatch_unit == "alpha-engine-trader-session.timer"
        assert row.absence_watched_by == "alerts.sweep"

    def test_its_deadline_is_two_hours_after_the_next_open_every_session(self) -> None:
        assert _row().deadline == Deadline(anchor="next_open_plus", cadence="daily", offset_hours=2)

    def test_a_release_declaring_it_counts_as_a_starter(self) -> None:
        """`row_has_a_starter` decides whether the release in force at a
        deadline declared the row. Before `box-timer` was in its set, every
        release would have read as UNDECLARED for this row. Combined with the
        release-history gate in `evaluate_absence`, the deadline would then
        never page."""
        assert row_has_a_starter(
            {
                "lifecycle": "ACTIVE",
                "dispatch": "box-timer",
                "deadline": {"anchor": "next_open_plus", "cadence": "daily", "offset_hours": 2},
            }
        )


class TestTheDeadlineReadsTheTradingCalendar:
    @pytest.mark.parametrize(
        ("trading_day", "session"),
        [
            # A Thursday: the next morning.
            (dt.date(2026, 10, 15), dt.date(2026, 10, 16)),
            # A Friday: Monday, never the weekend.
            (FRIDAY, MONDAY),
            # 2026-07-03 is the observed Independence Day holiday, so the
            # session after Thursday 07-02 is Monday 07-06.
            (dt.date(2026, 7, 2), dt.date(2026, 7, 6)),
            # Thanksgiving Thursday is closed, so Wednesday's session is
            # served on Friday, which is a half day. The OPEN does not move on
            # a half day.
            (dt.date(2026, 11, 25), dt.date(2026, 11, 27)),
            # New Year's Day 2027 is a Friday, so 12-31's session is served on
            # Monday 2027-01-04.
            (dt.date(2026, 12, 31), dt.date(2027, 1, 4)),
        ],
    )
    def test_it_is_due_at_1130_et_on_the_next_session(
        self, trading_day: dt.date, session: dt.date
    ) -> None:
        assert _row().deadline.due_at(trading_day) == _et(session, 11, 30)

    def test_it_follows_new_york_time_across_the_dst_change(self) -> None:
        """US DST ends 2026-11-01. Friday 10-30's deadline is Monday 11-02 at
        11:30 EST, which is 16:30Z, not the 15:30Z it would be in EDT."""
        due = _row().deadline.due_at(dt.date(2026, 10, 30))
        assert due == dt.datetime(2026, 11, 2, 16, 30, tzinfo=dt.UTC)
        assert _row().deadline.due_at(FRIDAY) == dt.datetime(2026, 10, 19, 15, 30, tzinfo=dt.UTC)

    def test_the_sentence_is_rendered_from_the_structure(self) -> None:
        assert _row().deadline.describe(FRIDAY) == (
            "2h after the open of the first session after trading day 2026-10-16 "
            "(every trading day)"
        )

    def test_a_next_open_plus_deadline_needs_an_offset(self) -> None:
        with pytest.raises(ValueError, match="next_open_plus deadline needs offset_hours"):
            Deadline(anchor="next_open_plus", cadence="daily")


class TestAMissingSessionPagesOnlyOnceItIsLate:
    """Against an EMPTY store, so the only thing that decides whether the
    session appears is whether its deadline has passed. With no sweep
    manifests, `days_to_evaluate` grades exactly the last closed session."""

    def test_the_weekend_after_a_friday_raises_no_page(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        assert _trader_absent(store, _et(dt.date(2026, 10, 17), 12, 0)) == []
        assert _trader_absent(store, _et(dt.date(2026, 10, 18), 21, 0)) == []

    def test_monday_morning_before_the_deadline_raises_no_page(self, tmp_path) -> None:
        assert _trader_absent(LocalStore(tmp_path), _et(MONDAY, 11, 29)) == []

    def test_monday_after_the_deadline_pages_for_friday(self, tmp_path) -> None:
        assert _trader_absent(LocalStore(tmp_path), _et(MONDAY, 11, 31)) == [FRIDAY.isoformat()]

    def test_the_page_says_what_was_due_and_when(self, tmp_path) -> None:
        (page,) = [
            p
            for p in evaluate_absence(LocalStore(tmp_path), now=_et(MONDAY, 11, 31))
            if p.job == JOB
        ]
        assert page.condition == "absence"
        assert "runs/trader.session/2026-10-16/" in page.reason
        assert "2026-10-19T15:30:00Z" in page.reason

    def test_a_holiday_monday_moves_the_page_to_tuesday(self, tmp_path) -> None:
        """2026-09-07 is Labor Day. Friday 09-04's session runs Tuesday 09-08,
        so nothing is due on the holiday itself."""
        store = LocalStore(tmp_path)
        assert _trader_absent(store, _et(dt.date(2026, 9, 7), 21, 0)) == []
        assert _trader_absent(store, _et(dt.date(2026, 9, 8), 11, 29)) == []
        assert _trader_absent(store, _et(dt.date(2026, 9, 8), 11, 31)) == ["2026-09-04"]

    def test_a_written_manifest_clears_the_day(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        store.put_bytes(manifest_key(JOB, FRIDAY.isoformat()), b"{}")
        assert _trader_absent(store, _et(MONDAY, 11, 31)) == []

    def test_a_day_before_the_release_declared_it_is_not_paged_after_the_fact(
        self, tmp_path
    ) -> None:
        """The release-history gate applies to `box-timer` rows. Without it,
        the first sweep after this ships would page the catch-up window's
        already-known misses (10-08, 10-14). No release in force at those
        deadlines declared a starter for this row."""

        class _NothingDeclared:
            def declaration(self, name: str, instant: dt.datetime) -> Declaration:
                return Declaration(UNDECLARED, f"no release declared {name}")

        assert (
            _trader_absent(
                LocalStore(tmp_path),
                _et(MONDAY, 11, 31),
                release_history=_NothingDeclared(),
            )
            == []
        )


class TestTheBoxTimerStarterIsNamed:
    def _component(self, **overrides) -> Component:
        base = {
            "name": "trader.example",
            "description": "x",
            "lifecycle": "ACTIVE",
            "signals": {},
            "log_location": "cloudwatch:/crucible/trader.example",
            "log_retention_days": 90,
            "alert_channel": "telegram:crucible-pages",
            "console_surface": "crucible/runs",
            "artifact_retention": "forever",
            "schedule": "weekdays",
            "deadline": Deadline(anchor="next_open_plus", cadence="daily", offset_hours=2),
            "dispatch": "box-timer",
            "dispatch_unit": "alpha-engine-trader-example.timer",
        }
        return Component(**{**base, **overrides})

    def test_box_timer_is_a_declared_starter(self) -> None:
        assert "box-timer" in DISPATCHES

    def test_a_valid_row_is_accepted(self) -> None:
        assert self._component().dispatch_unit == "alpha-engine-trader-example.timer"

    def test_a_box_timer_row_naming_no_timer_is_refused(self) -> None:
        with pytest.raises(ValueError, match="dispatch_unit is required on a `box-timer` row"):
            self._component(dispatch_unit=None)

    def test_a_timer_named_on_another_starter_is_refused(self) -> None:
        with pytest.raises(ValueError, match="dispatch_unit is required on a `box-timer` row"):
            self._component(dispatch="scheduler")

    def test_the_unit_must_be_the_timer_not_its_service(self) -> None:
        with pytest.raises(ValueError, match="is not a systemd timer basename"):
            self._component(dispatch_unit="alpha-engine-trader-example.service")
