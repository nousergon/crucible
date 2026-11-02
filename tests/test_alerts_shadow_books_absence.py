"""`alpha-engine-config-I12353` gap 2: an absent shadow book pages ABSENCE.

`crucible_trader.shadow_books_daily` is not a runner job, so it writes no
manifest and has no `components.yaml` row, and `evaluate_absence` never saw a
missing `trader/shadow_books/{d}.json`. The 2026-10-15 shadow-books unit died
in the venv build (crucible-trader#39) and paged nobody: the store has no book
for 2026-10-13 and none for 2026-10-07 (read-only, 2026-11-02).

The producer binds to the last closed session and advances the one before
it, at 11:00 ET, so the book for ``d`` lands on the SECOND session after
``d``. Every date below is a real NYSE session, holiday or weekend.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
from collections.abc import Iterator

import pytest

from crucible.alerts import (
    SHADOW_BOOKS_JOB,
    SWEEP_JOB,
    TRADER_SESSION_JOB,
    StoreAccessError,
    evaluate_shadow_books_absence,
    shadow_books_due_at,
    sweep,
)
from crucible.components import load_registry
from crucible.keys import TRADER_SHADOW_BOOKS_PREFIX, shadow_books_key
from crucible.manifest import manifest_key
from crucible.store import LocalStore

#: Every NYSE session from 10-01 to 10-23, as the sweep ran on them.
SESSIONS = [
    dt.date(2026, 10, d) for d in (1, 2, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 19, 20, 21, 22, 23)
]

#: The books the live store held by the evening of 2026-10-15: every session
#: from 10-01 to 10-12 except 10-07. 10-13's would have landed 10-15 11:00 ET.
FILED_BY_OCT_15 = [d for d in SESSIONS if d <= dt.date(2026, 10, 12) and d.day != 7]

#: The 21:00 ET sweep on Thursday 2026-10-15 (01:00Z on the 16th, EDT).
OCT_15_SWEEP = dt.datetime(2026, 10, 16, 1, 0, tzinfo=dt.UTC)


def _store(tmp_path, books: list[dt.date]) -> LocalStore:
    store = LocalStore(tmp_path)
    for day in SESSIONS:
        key = manifest_key(SWEEP_JOB, day.isoformat(), discriminator=day.isoformat())
        store.put_bytes(key, b'{"status": "ok"}')
    for day in books:
        store.put_bytes(shadow_books_key(day.isoformat()), b'{"schema_version": "shadow_books.v1"}')
    return store


def _paged_days(store: LocalStore, at: dt.datetime, **kwargs) -> list[dt.date]:
    return [p.trading_day for p in evaluate_shadow_books_absence(store, now=at, **kwargs)]


class TestTheDeadline:
    def test_a_book_is_due_at_noon_et_on_the_second_session_after_its_day(self) -> None:
        # Tue 10-13 -> filed Thu 10-15; 12:00 EDT is 16:00Z.
        assert shadow_books_due_at(dt.date(2026, 10, 13)) == dt.datetime(
            2026, 10, 15, 16, 0, tzinfo=dt.UTC
        )

    def test_a_weekend_moves_the_deadline_to_the_next_sessions(self) -> None:
        # Thu 10-29 -> Fri 10-30 -> Mon 11-02; Fri 10-30 -> Tue 11-03. EST after
        # 11-01, so 12:00 ET is 17:00Z.
        assert shadow_books_due_at(dt.date(2026, 10, 29)) == dt.datetime(
            2026, 11, 2, 17, 0, tzinfo=dt.UTC
        )
        assert shadow_books_due_at(dt.date(2026, 10, 30)) == dt.datetime(
            2026, 11, 3, 17, 0, tzinfo=dt.UTC
        )

    def test_a_holiday_moves_the_deadline_past_it(self) -> None:
        # Thanksgiving Thu 11-26 is closed: Tue 11-24 -> Wed 11-25 -> Fri 11-27.
        assert shadow_books_due_at(dt.date(2026, 11, 24)) == dt.datetime(
            2026, 11, 27, 17, 0, tzinfo=dt.UTC
        )

    def test_a_non_session_owes_no_book(self) -> None:
        with pytest.raises(ValueError):
            shadow_books_due_at(dt.date(2026, 10, 17))  # a Saturday


class TestEvaluateShadowBooksAbsence:
    def test_the_missing_october_13_book_pages_on_the_evening_it_was_due(self, tmp_path) -> None:
        store = _store(tmp_path, FILED_BY_OCT_15)
        pages = evaluate_shadow_books_absence(store, now=OCT_15_SWEEP)
        assert [p.trading_day for p in pages] == [dt.date(2026, 10, 13)]
        (page,) = pages
        assert page.condition == "absence"
        assert page.job == SHADOW_BOOKS_JOB
        assert page.run_id is None
        assert shadow_books_key("2026-10-13") in page.reason
        assert "2026-10-15" in page.reason
        assert "Auto-fix:" in page.reason
        assert "Resolution:" in page.reason

    def test_nothing_pages_before_the_deadline(self, tmp_path) -> None:
        store = _store(tmp_path, FILED_BY_OCT_15)
        # 11:59 EDT on 10-15: the window now reaches back to 10-07, whose book
        # is really missing and long due; 10-13's is due at noon.
        just_before = dt.datetime(2026, 10, 15, 15, 59, tzinfo=dt.UTC)
        assert _paged_days(store, just_before) == [dt.date(2026, 10, 7)]
        assert _paged_days(store, just_before + dt.timedelta(minutes=1)) == [
            dt.date(2026, 10, 7),
            dt.date(2026, 10, 13),
        ]

    def test_a_filed_book_clears_its_day(self, tmp_path) -> None:
        store = _store(tmp_path, [*FILED_BY_OCT_15, dt.date(2026, 10, 13)])
        assert _paged_days(store, OCT_15_SWEEP) == []

    def test_the_weekend_after_a_friday_pages_nothing(self, tmp_path) -> None:
        """Sat 10-17 21:00 ET: Thu 10-15's book is due Mon 10-19 and Fri 10-16's
        Tue 10-20, so no weekend evening grades either as late."""
        books = [*FILED_BY_OCT_15, dt.date(2026, 10, 13), dt.date(2026, 10, 14)]
        store = _store(tmp_path, books)
        saturday = dt.datetime(2026, 10, 18, 1, 0, tzinfo=dt.UTC)
        assert _paged_days(store, saturday) == []

    def test_every_missing_day_in_the_catch_up_window_pages(self, tmp_path) -> None:
        store = _store(tmp_path, [dt.date(2026, 10, 1)])
        # Wed 10-21 21:00 ET: today 10-21, window 10-14..10-20; due are the
        # days whose second session has reached noon, 10-14 through 10-19.
        at = dt.datetime(2026, 10, 22, 1, 0, tzinfo=dt.UTC)
        assert _paged_days(store, at) == [
            dt.date(2026, 10, 14),
            dt.date(2026, 10, 15),
            dt.date(2026, 10, 16),
            dt.date(2026, 10, 19),
        ]

    def test_a_store_with_no_book_ever_filed_pages_nothing(self, tmp_path) -> None:
        """A producer that never started is the phase-4 clause's finding, not a
        page per day."""
        assert _paged_days(_store(tmp_path, []), OCT_15_SWEEP) == []

    def test_days_before_the_first_book_page_nothing(self, tmp_path) -> None:
        store = _store(tmp_path, [dt.date(2026, 10, 12)])
        # Window 10-08..10-15: 10-08 and 10-09 predate the first book; 10-13 is
        # due and missing.
        assert _paged_days(store, OCT_15_SWEEP) == [dt.date(2026, 10, 13)]

    def test_a_non_book_key_under_the_prefix_is_ignored(self, tmp_path) -> None:
        store = _store(tmp_path, FILED_BY_OCT_15)
        store.put_bytes(f"{TRADER_SHADOW_BOOKS_PREFIX}README.txt", b"not a book")
        assert _paged_days(store, OCT_15_SWEEP) == [dt.date(2026, 10, 13)]

    def test_a_retired_trader_owes_no_book(self, tmp_path) -> None:
        store = _store(tmp_path, FILED_BY_OCT_15)
        registry = load_registry()
        retired = {
            **registry,
            TRADER_SESSION_JOB: dataclasses.replace(
                registry[TRADER_SESSION_JOB], lifecycle="RETIRED"
            ),
        }
        assert _paged_days(store, OCT_15_SWEEP, registry=retired) == []

    def test_an_unlistable_prefix_is_an_access_fault_not_silence(self, tmp_path) -> None:
        class _Denied(LocalStore):
            def list_keys(self, prefix: str = "") -> Iterator[str]:
                if prefix.startswith(TRADER_SHADOW_BOOKS_PREFIX):
                    raise PermissionError("AccessDenied")
                return super().list_keys(prefix)

        store = _store(tmp_path, FILED_BY_OCT_15)
        denied = _Denied(tmp_path)
        assert store.exists(shadow_books_key("2026-10-12"))
        with pytest.raises(StoreAccessError):
            evaluate_shadow_books_absence(denied, now=OCT_15_SWEEP)
        faults: list[str] = []
        assert evaluate_shadow_books_absence(denied, now=OCT_15_SWEEP, access_faults=faults) == []
        assert faults and TRADER_SHADOW_BOOKS_PREFIX in faults[0]


class TestTheSweepPagesAMissingBook:
    def test_the_sweep_sends_an_absence_page_naming_the_book(self, tmp_path, transport) -> None:
        store = _store(tmp_path, FILED_BY_OCT_15)
        result = sweep(store, now=OCT_15_SWEEP, transport=transport, sweep_run_id="Z" * 26)
        rows = [json.loads(store.get_bytes(k)) for k in result["bus_keys"]]
        books = [r for r in rows if any(m["job"] == SHADOW_BOOKS_JOB for m in r["members"])]
        assert len(books) == 1
        assert books[0]["condition"] == "absence"
        assert books[0]["sent"] is True
        (member,) = [m for m in books[0]["members"] if m["job"] == SHADOW_BOOKS_JOB]
        assert shadow_books_key("2026-10-13") in member["reason"]

    def test_a_filed_book_sends_no_page(self, tmp_path, transport) -> None:
        store = _store(tmp_path, [*FILED_BY_OCT_15, dt.date(2026, 10, 13)])
        result = sweep(store, now=OCT_15_SWEEP, transport=transport, sweep_run_id="Z" * 26)
        rows = [json.loads(store.get_bytes(k)) for k in result["bus_keys"]]
        assert not [r for r in rows if any(m["job"] == SHADOW_BOOKS_JOB for m in r["members"])]
