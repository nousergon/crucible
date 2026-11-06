"""`alpha-engine-config-I13129`: a back-office break pages.

An overdue queue item, or a failed ledger or month-rebuild check.

`trader.reconcile` step 5e scores the reviewer's queue every trading day and
counts the items older than 3 trading days. That count used to make a metric
row WATCH and nothing else: the manifest exists (no ABSENCE) and its status
is about the broker, not the queue (no FAILURE), so a break nobody decided
aged in silence, and the ledger's and month rebuild's FAIL rows paged nobody
either. `crucible.alerts.evaluate_back_office_breaks` reuses the FAILURE
condition, once per failing streak.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import pathlib
from collections.abc import Iterator, Mapping
from typing import cast

import pytest
from conftest import CapturingTransport

from crucible.alerts import (
    BACK_OFFICE_JOB,
    EXCEPTION_OVERDUE_METRIC,
    EXCEPTION_SCORE_METRIC,
    OVERDUE_LOOKBACK_TRADING_DAYS,
    TRADER_RECONCILE_JOB,
    StoreAccessError,
    evaluate_back_office_breaks,
    sweep,
)
from crucible.calendar import previous_trading_day
from crucible.components import load_registry
from crucible.manifest import manifest_key
from crucible.store import LocalStore

TUESDAY, WEDNESDAY, THURSDAY, FRIDAY = (dt.date(2026, 8, d) for d in (25, 26, 27, 28))
SATURDAY_NIGHT = dt.datetime(2026, 8, 29, 23, 30, tzinfo=dt.UTC)
OVERDUE_REASON = (
    "1 back-office queue items on 2026-08-28 are older than 3 trading days with no "
    "decision, oldest first: prop-1 (2026-08-21, 5 trading days)"
)


def _row(name: str, status: str, reason: str) -> dict[str, object]:
    return {
        "name": name,
        "module": "crucible_trader.exception_decision",
        "metric_type": "operational",
        "value": 1.0,
        "unit": "count",
        "n_floor": 0,
        "status": status,
        "status_reason": reason,
        "source_path": "trader/exception_scores/x.json",
        "last_updated_utc": "2026-08-28T21:00:00Z",
    }


def _overdue(status: str = "FAIL", reason: str = OVERDUE_REASON) -> list[object]:
    return [
        _row(EXCEPTION_SCORE_METRIC, "WATCH" if status == "FAIL" else "OK", "queue"),
        _row(EXCEPTION_OVERDUE_METRIC, status, reason),
    ]


def _write(
    store: LocalStore,
    day: dt.date,
    metrics: list[object],
    *,
    status: str = "ok",
    run_id: str = "01JG0000000000000000000RC1",
    discriminator: str | None = None,
) -> None:
    payload: dict[str, object] = {
        "schema_version": "run_manifest.v2",
        "run_id": run_id,
        "job": TRADER_RECONCILE_JOB,
        "trading_day": day.isoformat(),
        "calendar_date": day.isoformat(),
        "status": status,
        "reason": "",
        "cost_usd": 0.0,
        "metrics": metrics,
        "attempts": [{"n": 1, "reason": "initial"}],
    }
    if discriminator is not None:
        payload["discriminator"] = discriminator
    store.put_bytes(
        manifest_key(TRADER_RECONCILE_JOB, day.isoformat(), discriminator=discriminator),
        json.dumps(payload).encode(),
    )


class TestEvaluateOverdueBreaks:
    def test_an_overdue_queue_pages_with_the_traders_reason(self, tmp_path: pathlib.Path) -> None:
        store = LocalStore(tmp_path)
        _write(store, FRIDAY, _overdue())
        (page,) = evaluate_back_office_breaks(store, now=SATURDAY_NIGHT)
        assert page.condition == "failure"
        assert page.job == BACK_OFFICE_JOB
        assert page.trading_day == FRIDAY
        assert page.run_id == "01JG0000000000000000000RC1"
        assert "prop-1 (2026-08-21, 5 trading days)" in page.reason
        assert "Auto-fix:" in page.reason and "Resolution:" in page.reason

    def test_a_queue_with_nothing_overdue_pages_nothing(self, tmp_path: pathlib.Path) -> None:
        store = LocalStore(tmp_path)
        _write(store, FRIDAY, _overdue("OK", "nothing older than 3"))
        assert evaluate_back_office_breaks(store, now=SATURDAY_NIGHT) == []

    def test_a_reconcile_that_failed_on_the_broker_still_reports_its_queue(
        self, tmp_path: pathlib.Path
    ) -> None:
        """The broker failure pages on its own; the queue it read is a second incident."""
        store = LocalStore(tmp_path)
        _write(store, FRIDAY, _overdue(), status="failed")
        (page,) = evaluate_back_office_breaks(store, now=SATURDAY_NIGHT)
        assert page.job == BACK_OFFICE_JOB

    def test_a_trader_pinned_before_the_dedicated_row_pages_on_its_watch(
        self, tmp_path: pathlib.Path
    ) -> None:
        store = LocalStore(tmp_path)
        reason = "back-office queue for 2026-08-28: 1 open, 1 overdue, agreement None"
        _write(store, FRIDAY, [_row(EXCEPTION_SCORE_METRIC, "WATCH", reason)])
        (page,) = evaluate_back_office_breaks(store, now=SATURDAY_NIGHT)
        assert "1 overdue" in page.reason
        _write(store, FRIDAY, [_row(EXCEPTION_SCORE_METRIC, "OK", "0 overdue")])
        assert evaluate_back_office_breaks(store, now=SATURDAY_NIGHT) == []

    def test_an_unfiled_score_is_not_an_overdue_reading(self, tmp_path: pathlib.Path) -> None:
        store = LocalStore(tmp_path)
        _write(store, FRIDAY, [_row(EXCEPTION_SCORE_METRIC, "FAIL", "NOT filed"), "junk"])
        _write(store, THURSDAY, [], run_id="01JG0000000000000000000RC0")
        assert evaluate_back_office_breaks(store, now=SATURDAY_NIGHT) == []

    def test_a_manifest_with_no_metric_list_is_skipped(self, tmp_path: pathlib.Path) -> None:
        store = LocalStore(tmp_path)
        _write(store, FRIDAY, _overdue())
        payload = json.loads(store.get_bytes(manifest_key(TRADER_RECONCILE_JOB, "2026-08-28")))
        payload["metrics"] = "not a list"
        store.put_bytes(
            manifest_key(TRADER_RECONCILE_JOB, "2026-08-28"), json.dumps(payload).encode()
        )
        assert evaluate_back_office_breaks(store, now=SATURDAY_NIGHT) == []

    def test_a_failed_ledger_check_pages(self, tmp_path: pathlib.Path) -> None:
        store = LocalStore(tmp_path)
        reason = "ledger checks failed on 2026-08-28: period_lock 2026-07 closed month changed"
        _write(
            store, FRIDAY, [*_overdue("OK", "none"), _row("ledger_book_of_record", "FAIL", reason)]
        )
        (page,) = evaluate_back_office_breaks(store, now=SATURDAY_NIGHT)
        assert "ledger_book_of_record: " + reason in page.reason
        assert EXCEPTION_OVERDUE_METRIC not in page.reason.split("Resolution")[0]

    def test_a_month_that_does_not_rebuild_pages_beside_an_overdue_queue(
        self, tmp_path: pathlib.Path
    ) -> None:
        store = LocalStore(tmp_path)
        _write(store, FRIDAY, [*_overdue(), _row("month_rebuild", "FAIL", "fee bill differs")])
        (page,) = evaluate_back_office_breaks(store, now=SATURDAY_NIGHT)
        head = page.reason.split("\n")[0]
        assert "prop-1" in head and "month_rebuild: fee bill differs" in head

    def test_a_gap_booked_to_suspense_is_left_to_the_queue(self, tmp_path: pathlib.Path) -> None:
        """Ledger WATCH: the gap is a queue item and ages there, not here."""
        store = LocalStore(tmp_path)
        _write(store, FRIDAY, [_row("ledger_book_of_record", "WATCH", "1 break to suspense")])
        assert evaluate_back_office_breaks(store, now=SATURDAY_NIGHT) == []

    def test_a_clean_ledger_reading_ends_a_streak(self, tmp_path: pathlib.Path) -> None:
        store = LocalStore(tmp_path)
        _write(store, WEDNESDAY, [_row("ledger_book_of_record", "FAIL", "roll_forward")])
        _write(store, THURSDAY, [_row("ledger_book_of_record", "OK", "ties")])
        _write(store, FRIDAY, [_row("ledger_book_of_record", "FAIL", "roll_forward")])
        (page,) = evaluate_back_office_breaks(store, now=SATURDAY_NIGHT)
        assert page.trading_day == FRIDAY

    def test_the_latest_run_of_the_day_decides(self, tmp_path: pathlib.Path) -> None:
        """A re-run after the reviewer decided clears the day."""
        store = LocalStore(tmp_path)
        _write(
            store,
            FRIDAY,
            _overdue("OK", "cleared"),
            run_id="01JG...B".ljust(26, "B"),
            discriminator="b",
        )
        _write(store, FRIDAY, _overdue(), run_id="01JG...A".ljust(26, "A"), discriminator="a")
        assert evaluate_back_office_breaks(store, now=SATURDAY_NIGHT) == []

    def test_a_standing_backlog_is_one_incident_anchored_where_it_began(
        self, tmp_path: pathlib.Path
    ) -> None:
        store = LocalStore(tmp_path)
        _write(store, TUESDAY, _overdue("OK", "not yet"))
        _write(store, WEDNESDAY, _overdue())
        _write(store, FRIDAY, _overdue())  # Thursday has no reading: skipped, not a break
        (page,) = evaluate_back_office_breaks(store, now=SATURDAY_NIGHT)
        assert page.trading_day == WEDNESDAY
        assert "latest reading 2026-08-28" in page.reason

    def test_a_streak_longer_than_the_lookback_keeps_its_first_day(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Codex finding 1f3925eff99ee3cb: the anchor of a streak longer than
        the lookback never slides, so the incident keeps one identity."""
        store = LocalStore(tmp_path)
        day = FRIDAY
        days = []
        for _ in range(OVERDUE_LOOKBACK_TRADING_DAYS + 5):
            days.append(day)
            day = previous_trading_day(day)
        for d in days:
            _write(store, d, _overdue())
        (page,) = evaluate_back_office_breaks(store, now=SATURDAY_NIGHT)
        assert page.trading_day == days[-1]

    def test_a_reading_older_than_the_lookback_is_not_read(self, tmp_path: pathlib.Path) -> None:
        store = LocalStore(tmp_path)
        _write(store, dt.date(2026, 7, 1), _overdue())
        assert OVERDUE_LOOKBACK_TRADING_DAYS == 20
        assert evaluate_back_office_breaks(store, now=SATURDAY_NIGHT) == []

    def test_a_retired_reconcile_row_pages_nothing(self, tmp_path: pathlib.Path) -> None:
        store = LocalStore(tmp_path)
        _write(store, FRIDAY, _overdue())
        registry = load_registry()
        retired = {
            **registry,
            TRADER_RECONCILE_JOB: dataclasses.replace(
                registry[TRADER_RECONCILE_JOB], lifecycle="RETIRED"
            ),
        }
        assert evaluate_back_office_breaks(store, now=SATURDAY_NIGHT, registry=retired) == []

    def test_an_unlistable_prefix_is_an_access_fault_not_silence(
        self, tmp_path: pathlib.Path
    ) -> None:
        class _Denied(LocalStore):
            def list_keys(self, prefix: str = "") -> Iterator[str]:
                if prefix.startswith(f"runs/{TRADER_RECONCILE_JOB}/"):
                    raise PermissionError("AccessDenied")
                return super().list_keys(prefix)

        with pytest.raises(StoreAccessError):
            evaluate_back_office_breaks(_Denied(tmp_path), now=SATURDAY_NIGHT)
        faults: list[str] = []
        assert (
            evaluate_back_office_breaks(_Denied(tmp_path), now=SATURDAY_NIGHT, access_faults=faults)
            == []
        )
        assert len(faults) == 1  # one fault, however many days could not be listed


class TestTheSweepPagesAnOverdueQueue:
    def _rows(self, store: LocalStore, result: Mapping[str, object]) -> list[dict[str, object]]:
        keys = cast("list[str]", result["bus_keys"])
        rows = [cast("dict[str, object]", json.loads(store.get_bytes(k))) for k in keys]
        return [
            r
            for r in rows
            if any(
                m["job"] == BACK_OFFICE_JOB for m in cast("list[dict[str, object]]", r["members"])
            )
        ]

    def test_one_page_then_re_observed_not_re_sent(
        self, tmp_path: pathlib.Path, transport: CapturingTransport
    ) -> None:
        store = LocalStore(tmp_path)
        _write(store, THURSDAY, _overdue())
        _write(store, FRIDAY, _overdue())
        first = sweep(store, now=SATURDAY_NIGHT, transport=transport, sweep_run_id="Z" * 26)
        (row,) = self._rows(store, first)
        assert row["condition"] == "failure" and row["sent"] is True
        assert row["trading_day"] == THURSDAY.isoformat()
        later = SATURDAY_NIGHT + dt.timedelta(days=1)
        second = sweep(store, now=later, transport=transport, sweep_run_id="Y" * 26)
        assert len(self._rows(store, second)) == 1
        sent = [c for c in transport.calls if BACK_OFFICE_JOB in c.message]
        assert len(sent) == 1  # re-observed by the second sweep, never re-sent

    def test_an_unreadable_streak_start_still_pages_the_latest_day(
        self, tmp_path: pathlib.Path
    ) -> None:
        class _Denied(LocalStore):
            def list_keys(self, prefix: str = "") -> Iterator[str]:
                if prefix == f"runs/{TRADER_RECONCILE_JOB}/{THURSDAY.isoformat()}/":
                    raise PermissionError("AccessDenied")
                return super().list_keys(prefix)

        store = _Denied(tmp_path)
        _write(store, FRIDAY, _overdue())
        faults: list[str] = []
        (page,) = evaluate_back_office_breaks(store, now=SATURDAY_NIGHT, access_faults=faults)
        assert page.trading_day == FRIDAY and len(faults) == 1
