"""The trading-day contract (plan §4.12), tested rather than requested.

Brian's ruling, 2026-09-01: *"each data point should bind to a trading day,
not a calendar day. We should make all decisions based on trading days."*

The plan's own words are that this is **enforced, not requested**: "A
contract test walks every artifact key, manifest field and arena_cycle window
and fails on any date that is not a trading day."

This module is the scaffold of that walk. It exercises the walker against a
store today, so a track filling in a real S3 backend inherits a test that
already fails on the defect rather than one written after the fact.
"""

from __future__ import annotations

import datetime as dt

import pytest

from crucible.calendar import (
    NonTradingDayKeyError,
    assert_trading_day,
    is_trading_day,
    resolve_trading_day,
    trading_weeks_between,
)
from crucible.store import LocalStore

# 2026-08-28 is a Friday and a full NYSE session. 2026-08-29/30 are the
# weekend that follows it. Fixed literals, not `today` arithmetic: a test
# whose subject moves with the clock stops testing the same thing.
FRIDAY = dt.date(2026, 8, 28)
SATURDAY = dt.date(2026, 8, 29)
SUNDAY = dt.date(2026, 8, 30)

# Independence Day 2026 falls on a Saturday, so the NYSE observes it on
# Friday 2026-07-03 — a weekday that is NOT a trading day, which a
# `weekday() < 5` implementation gets wrong and a calendar gets right.
OBSERVED_HOLIDAY = dt.date(2026, 7, 3)
DAY_BEFORE_OBSERVED_HOLIDAY = dt.date(2026, 7, 2)


class TestResolution:
    def test_a_saturday_resolves_to_friday(self) -> None:
        """The 2026-05-08 ruling, restated as v2's single axis: a Saturday
        weekly run is keyed to Friday's close, not to the wall-clock date."""
        saturday_10am = dt.datetime(2026, 8, 29, 10, 0)
        assert resolve_trading_day(saturday_10am) == FRIDAY

    def test_a_sunday_also_resolves_to_friday(self) -> None:
        assert resolve_trading_day(dt.datetime(2026, 8, 30, 10, 0)) == FRIDAY

    def test_a_weekday_holiday_resolves_backward_past_it(self) -> None:
        """Observed-Independence-Day 2026-07-03 is a Friday and closed. A run
        on it binds to Thursday 07-02, and a `weekday() < 5` check would
        silently key an artifact to a day the market never traded."""
        assert not is_trading_day(OBSERVED_HOLIDAY)
        assert resolve_trading_day(dt.datetime(2026, 7, 3, 10, 0)) == DAY_BEFORE_OBSERVED_HOLIDAY

    def test_before_the_close_resolves_to_the_prior_session(self) -> None:
        """Backward-looking, always: 09:00 on a trading day binds to the
        previous session, because today's has not closed. A morning job that
        keyed to `today` would publish an artifact for a day with no data."""
        assert resolve_trading_day(dt.datetime(2026, 8, 28, 9, 0)) == dt.date(2026, 8, 27)

    def test_after_the_close_resolves_to_the_same_session(self) -> None:
        assert resolve_trading_day(dt.datetime(2026, 8, 28, 16, 30)) == FRIDAY

    def test_resolution_is_idempotent(self) -> None:
        """Resolving a resolution changes nothing. A pipeline that resolves at
        several stages must not walk backwards one session per stage."""
        once = resolve_trading_day(dt.datetime(2026, 8, 29, 10, 0))
        twice = resolve_trading_day(dt.datetime.combine(once, dt.time(16, 30)))
        assert once == twice


class TestTradingWeeks:
    def test_a_trading_week_is_five_trading_days(self) -> None:
        """§4.12: `promote_min_weeks = 4` is 20 paired trading days."""
        assert trading_weeks_between(dt.date(2026, 8, 21), FRIDAY) == 1.0

    def test_a_holiday_week_is_still_one_rung(self) -> None:
        """§4.12 verbatim: "a holiday week is still one rung". The week of
        2026-07-03 has four sessions, not five; the ladder must not silently
        under-count it into 0.8 of a rung."""
        assert trading_weeks_between(dt.date(2026, 6, 26), dt.date(2026, 7, 10), rungs=True) == 2

    def test_a_backwards_interval_is_zero_not_negative(self) -> None:
        assert trading_weeks_between(FRIDAY, dt.date(2026, 8, 21)) == 0.0


class TestKeyEnforcement:
    def test_assert_trading_day_accepts_a_session(self) -> None:
        assert_trading_day(FRIDAY, context="signals/2026-08-28/signals.json")

    @pytest.mark.parametrize("bad", [SATURDAY, SUNDAY, OBSERVED_HOLIDAY])
    def test_assert_trading_day_refuses_a_non_session(self, bad: dt.date) -> None:
        with pytest.raises(NonTradingDayKeyError) as exc:
            assert_trading_day(bad, context=f"signals/{bad}/signals.json")
        # The error names the key, or an operator cannot find the writer.
        assert str(bad) in str(exc.value)
        assert "signals/" in str(exc.value)


class TestStoreWalk:
    """The §4.12 walk itself, over the local backend.

    The S3 backend is a track's to fill in; the walk is written against the
    `Store` interface, so it inherits it rather than being rewritten for it.
    """

    def test_a_clean_store_passes_the_walk(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        store.put_bytes("runs/data.daily/2026-08-27/run.json", b"{}")
        store.put_bytes("runs/data.daily/2026-08-28/run.json", b"{}")
        store.put_bytes("signals/2026-08-28/signals.json", b"{}")
        store.assert_keys_bind_to_trading_days()

    def test_the_walk_rejects_a_saturday_key(self, tmp_path) -> None:
        """The defect this exists to catch: a job that keyed by wall-clock
        date instead of resolving. It writes a Saturday, every downstream
        pairing silently loses that date, and nothing else complains."""
        store = LocalStore(tmp_path)
        store.put_bytes("runs/data.weekly/2026-08-28/run.json", b"{}")
        store.put_bytes("runs/data.weekly/2026-08-29/run.json", b"{}")
        with pytest.raises(NonTradingDayKeyError) as exc:
            store.assert_keys_bind_to_trading_days()
        assert "2026-08-29" in str(exc.value)

    def test_the_walk_rejects_a_weekday_holiday_key(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        store.put_bytes("runs/data.daily/2026-07-03/run.json", b"{}")
        with pytest.raises(NonTradingDayKeyError):
            store.assert_keys_bind_to_trading_days()

    def test_the_walk_reports_every_offender_not_only_the_first(self, tmp_path) -> None:
        """An operator fixing one bad key per run of the test is how a
        multi-day backfill defect takes a week to clear."""
        store = LocalStore(tmp_path)
        for bad in ("2026-08-29", "2026-08-30", "2026-07-03"):
            store.put_bytes(f"runs/data.daily/{bad}/run.json", b"{}")
        with pytest.raises(NonTradingDayKeyError) as exc:
            store.assert_keys_bind_to_trading_days()
        for bad in ("2026-08-29", "2026-08-30", "2026-07-03"):
            assert bad in str(exc.value)

    def test_a_key_with_no_date_component_is_not_silently_passed_over(self, tmp_path) -> None:
        """A key carrying no ISO date is legal (`releases/current`,
        `champions/r/current.json`). It must be *recognised* as dateless, not
        merely fail to match a regex — otherwise a malformed date like
        `2026-8-29` reads as dateless and escapes the walk entirely."""
        store = LocalStore(tmp_path)
        store.put_bytes("champions/r/current.json", b"{}")
        store.put_bytes("releases/current", b"{}")
        store.assert_keys_bind_to_trading_days()

        store.put_bytes("runs/data.daily/2026-8-29/run.json", b"{}")
        with pytest.raises(NonTradingDayKeyError) as exc:
            store.assert_keys_bind_to_trading_days()
        assert "2026-8-29" in str(exc.value)
