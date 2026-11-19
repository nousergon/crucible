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
    Page,
    StoreAccessError,
    emit,
    evaluate_back_office_breaks,
    group_pages,
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

    @pytest.mark.parametrize("with_overdue_row", [False, True])
    def test_an_unfiled_score_pages(
        self, tmp_path: pathlib.Path, *, with_overdue_row: bool
    ) -> None:
        store = LocalStore(tmp_path)
        metrics: list[object] = [_row(EXCEPTION_SCORE_METRIC, "FAIL", "HeadObject 403")]
        if with_overdue_row:
            metrics.extend(_overdue("OK", "nothing overdue"))
        _write(store, FRIDAY, metrics)
        _write(store, THURSDAY, [], run_id="01JG0000000000000000000RC0")
        (page,) = evaluate_back_office_breaks(store, now=SATURDAY_NIGHT)
        assert "exception_score_filed: HeadObject 403" in page.reason

    def test_any_failed_back_office_step_pages_with_its_reason(
        self, tmp_path: pathlib.Path
    ) -> None:
        store = LocalStore(tmp_path)
        _write(store, FRIDAY, [_row("performance_statement", "FAIL", "statement NOT filed")])
        (page,) = evaluate_back_office_breaks(store, now=SATURDAY_NIGHT)
        assert "performance_statement: statement NOT filed" in page.reason

    @pytest.mark.parametrize(
        "step",
        [
            "exception_records_filed",
            "exception_proposals_filed",
            "exception_rule_clears_filed",
            "ledger_book_of_record",
            "benchmark_closes_capture",
            "performance_statement",
            "month_end_close",
            "month_rebuild",
            "books_worm_retention",
        ],
    )
    def test_each_back_office_manifest_step_failure_is_visible(
        self, tmp_path: pathlib.Path, step: str
    ) -> None:
        store = LocalStore(tmp_path)
        _write(store, FRIDAY, [_row(step, "FAIL", "record could not be filed")])
        (page,) = evaluate_back_office_breaks(store, now=SATURDAY_NIGHT)
        assert page.condition == "failure"
        assert f"{step}: record could not be filed" in page.reason

    def test_a_broker_row_failing_is_not_a_back_office_break(self, tmp_path: pathlib.Path) -> None:
        """The broker's breaks fail the reconcile run and page under its own
        job; the back-office label must not page them a second time."""
        store = LocalStore(tmp_path)
        _write(store, FRIDAY, [_row("order_intent_reconciliation_breaks", "FAIL", "2 breaks")])
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
        days: list[dt.date] = []
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

    @staticmethod
    def _queue(day: dt.date, items: list[tuple[str, str, int]]) -> list[object]:
        """The producer's wording: the day and every age move each trading day."""
        named = "; ".join(f"{pid} ({opened}, {age} trading days)" for pid, opened, age in items)
        return _overdue(
            reason=(
                f"{len(items)} back-office queue items on {day.isoformat()} are older than "
                f"3 trading days with no decision, oldest first: {named}"
            )
        )

    def test_a_new_member_of_a_standing_backlog_is_sent_once(
        self, tmp_path: pathlib.Path, transport: CapturingTransport
    ) -> None:
        store = LocalStore(tmp_path)
        _write(store, THURSDAY, self._queue(THURSDAY, [("prop-1", "2026-08-21", 4)]))
        sweep(store, now=SATURDAY_NIGHT, transport=transport, sweep_run_id="Z" * 26)

        both = [("prop-1", "2026-08-21", 5), ("prop-2", "2026-08-24", 4)]
        _write(store, FRIDAY, self._queue(FRIDAY, both))
        later = SATURDAY_NIGHT + dt.timedelta(days=1)
        sweep(store, now=later, transport=transport, sweep_run_id="Y" * 26)
        last = sweep(store, now=later, transport=transport, sweep_run_id="X" * 26)

        sent = [call for call in transport.calls if BACK_OFFICE_JOB in call.message]
        assert len(sent) == 2
        assert "prop-2 (2026-08-24, 4 trading days)" in sent[-1].message
        (row,) = self._rows(store, last)
        assert row["observations"] == 3  # one incident, one row, updated in place

    def test_an_unchanged_backlog_ageing_a_day_is_not_re_sent(
        self, tmp_path: pathlib.Path, transport: CapturingTransport
    ) -> None:
        store = LocalStore(tmp_path)
        _write(store, THURSDAY, self._queue(THURSDAY, [("prop-1", "2026-08-21", 4)]))
        sweep(store, now=SATURDAY_NIGHT, transport=transport, sweep_run_id="Z" * 26)
        _write(store, FRIDAY, self._queue(FRIDAY, [("prop-1", "2026-08-21", 5)]))
        later = SATURDAY_NIGHT + dt.timedelta(days=1)
        sweep(store, now=later, transport=transport, sweep_run_id="Y" * 26)

        sent = [call for call in transport.calls if BACK_OFFICE_JOB in call.message]
        assert len(sent) == 1

    def test_a_newly_failing_step_in_a_standing_streak_is_sent(
        self, tmp_path: pathlib.Path, transport: CapturingTransport
    ) -> None:
        store = LocalStore(tmp_path)
        _write(store, THURSDAY, self._queue(THURSDAY, [("prop-1", "2026-08-21", 4)]))
        sweep(store, now=SATURDAY_NIGHT, transport=transport, sweep_run_id="Z" * 26)
        friday = self._queue(FRIDAY, [("prop-1", "2026-08-21", 5)])
        friday.append(_row("ledger_book_of_record", "FAIL", "roll-forward broke"))
        _write(store, FRIDAY, friday)
        sweep(
            store,
            now=SATURDAY_NIGHT + dt.timedelta(days=1),
            transport=transport,
            sweep_run_id="Y" * 26,
        )

        sent = [call for call in transport.calls if BACK_OFFICE_JOB in call.message]
        assert len(sent) == 2
        assert "ledger_book_of_record: roll-forward broke" in sent[-1].message

    def test_failed_rule_clear_filing_updates_a_standing_incident_once(
        self, tmp_path: pathlib.Path, transport: CapturingTransport
    ) -> None:
        store = LocalStore(tmp_path)
        _write(store, THURSDAY, self._queue(THURSDAY, [("prop-1", "2026-08-21", 4)]))
        first = sweep(store, now=SATURDAY_NIGHT, transport=transport, sweep_run_id="Z" * 26)
        (original,) = self._rows(store, first)
        friday = self._queue(FRIDAY, [("prop-1", "2026-08-21", 5)])
        friday.append(_row("exception_rule_clears_filed", "FAIL", "rule evidence unreadable"))
        _write(store, FRIDAY, friday)
        later = SATURDAY_NIGHT + dt.timedelta(days=1)
        sweep(store, now=later, transport=transport, sweep_run_id="Y" * 26)
        last = sweep(store, now=later, transport=transport, sweep_run_id="X" * 26)

        sent = [call for call in transport.calls if BACK_OFFICE_JOB in call.message]
        assert len(sent) == 2
        assert "exception_rule_clears_filed: rule evidence unreadable" in sent[-1].message
        (row,) = self._rows(store, last)
        assert row["trading_day"] == original["trading_day"] == THURSDAY.isoformat()
        assert row["observations"] == 3
        assert "exception_rule_clears_filed: rule evidence unreadable" in str(row["rendered"])
        assert "Current observation (not a delivery receipt)" in str(row["rendered"])
        assert row["rendered_initial"] == original["rendered"]
        assert any(
            "exception_rule_clears_filed: rule evidence unreadable" in str(member["reason"])
            for member in cast("list[dict[str, object]]", row["members_now"])
        )

    @staticmethod
    def _published_queue(store: LocalStore, day: dt.date, ids: list[str]) -> None:
        items = [(pid, "2026-08-21", 5) for pid in ids]
        metrics = TestTheSweepPagesAnOverdueQueue._queue(day, items[:5])
        row = cast("dict[str, object]", metrics[-1])
        row["status_reason"] = (
            str(row["status_reason"]).replace("5 back-office", f"{len(ids)} back-office")
            + f"; and {len(ids) - 5} more"
        )
        row["value"] = float(len(ids))
        key = f"trader/exception_scores/{day.isoformat()}.json"
        row["source_path"] = key
        store.put_bytes(
            key,
            json.dumps(
                {
                    "schema_version": "exception_score.v1",
                    "trading_day": day.isoformat(),
                    "generated_at": row["last_updated_utc"],
                    "totals": {"overdue": len(ids)},
                    "queue": [
                        {
                            "proposal_id": pid,
                            "trading_day": opened,
                            "age_trading_days": age,
                            "overdue": True,
                        }
                        for pid, opened, age in items
                    ],
                }
            ).encode(),
        )
        _write(store, day, metrics)

    def test_members_beyond_the_five_name_summary_are_named_and_deduplicated(
        self, tmp_path: pathlib.Path, transport: CapturingTransport
    ) -> None:
        store = LocalStore(tmp_path)
        first_five = [f"prop-{i}" for i in range(5)]
        self._published_queue(store, THURSDAY, first_five)
        sweep(store, now=SATURDAY_NIGHT, transport=transport, sweep_run_id="Z" * 26)
        self._published_queue(store, FRIDAY, [*first_five, "sixth"])
        sweep(store, now=SATURDAY_NIGHT, transport=transport, sweep_run_id="Y" * 26)
        sent = [call for call in transport.calls if BACK_OFFICE_JOB in call.message]
        assert len(sent) == 2
        assert "sixth (2026-08-21, 5 trading days)" in sent[-1].message
        self._published_queue(store, FRIDAY, [*first_five, "replacement"])
        sweep(store, now=SATURDAY_NIGHT, transport=transport, sweep_run_id="X" * 26)
        last = sweep(store, now=SATURDAY_NIGHT, transport=transport, sweep_run_id="W" * 26)
        sent = [call for call in transport.calls if BACK_OFFICE_JOB in call.message]
        assert len(sent) == 3
        assert "replacement (2026-08-21, 5 trading days)" in sent[-1].message
        (row,) = self._rows(store, last)
        assert row["trading_day"] == THURSDAY.isoformat()
        assert row["observations"] == 4

    @pytest.mark.parametrize(
        "fault", ["absent", "corrupt", "stale", "count", "identity", "invalid-day"]
    )
    def test_unusable_full_queue_evidence_never_hides_the_break(
        self, tmp_path: pathlib.Path, fault: str
    ) -> None:
        store = LocalStore(tmp_path)
        self._published_queue(store, FRIDAY, [f"prop-{i}" for i in range(6)])
        key = "trader/exception_scores/2026-08-28.json"
        if fault == "absent":
            (tmp_path / key).unlink()
        elif fault == "corrupt":
            store.put_bytes(key, b"not json")
        else:
            document = json.loads(store.get_bytes(key))
            if fault == "stale":
                document["generated_at"] = "2026-08-27T21:00:00Z"
            elif fault == "count":
                document["queue"].pop()
            elif fault == "invalid-day":
                document["queue"][-1]["trading_day"] = "2026-99-99"
            else:
                document["queue"][0]["proposal_id"] = "different"
            store.put_bytes(key, json.dumps(document).encode())
        (page,) = evaluate_back_office_breaks(store, now=SATURDAY_NIGHT)
        assert "6 back-office queue items" in page.reason
        assert "Queue evidence unavailable:" in page.reason
        assert "prop-5 (" not in page.reason

    @pytest.mark.parametrize("corrupt", [False, True])
    @pytest.mark.parametrize("initial_six", [False, True])
    def test_recovering_identical_queue_evidence_does_not_notify_again(
        self,
        tmp_path: pathlib.Path,
        transport: CapturingTransport,
        *,
        corrupt: bool,
        initial_six: bool,
    ) -> None:
        store = LocalStore(tmp_path)
        five = [f"prop-{i}" for i in range(5)]
        self._published_queue(store, THURSDAY, [*five, "sixth"] if initial_six else five)
        sweep(store, now=SATURDAY_NIGHT, transport=transport, sweep_run_id="Z" * 26)
        self._published_queue(store, FRIDAY, [*five, "sixth"])
        sweep(store, now=SATURDAY_NIGHT, transport=transport, sweep_run_id="Y" * 26)
        key = "trader/exception_scores/2026-08-28.json"
        saved = store.get_bytes(key)
        if corrupt:
            store.put_bytes(key, b"bad json")
        else:
            (tmp_path / key).unlink()
        missing = sweep(store, now=SATURDAY_NIGHT, transport=transport, sweep_run_id="X" * 26)
        (row,) = self._rows(store, missing)
        assert "Queue evidence unavailable" in str(row["members_now"])
        assert "Queue evidence unavailable" in str(row["rendered"])
        store.put_bytes(key, saved)
        last = sweep(store, now=SATURDAY_NIGHT, transport=transport, sweep_run_id="W" * 26)
        sent = [call for call in transport.calls if BACK_OFFICE_JOB in call.message]
        assert len(sent) == (1 if initial_six else 2)
        (row,) = self._rows(store, last)
        assert row["observations"] == 4

    def test_a_long_queue_gain_remains_visible_in_the_console_message(
        self, tmp_path: pathlib.Path, transport: CapturingTransport
    ) -> None:
        store = LocalStore(tmp_path)
        old = [f"proposal-{i:032d}" for i in range(100)]
        self._published_queue(store, THURSDAY, old)
        first = sweep(store, now=SATURDAY_NIGHT, transport=transport, sweep_run_id="Z" * 26)
        (original,) = self._rows(store, first)
        self._published_queue(store, FRIDAY, [*old, "new-overdue"])
        last = sweep(store, now=SATURDAY_NIGHT, transport=transport, sweep_run_id="Y" * 26)
        (row,) = self._rows(store, last)
        assert "new-overdue (2026-08-21, 5 trading days)" in str(row["rendered"])
        assert row["rendered_initial"] == original["rendered"]
        assert row["members"] == original["members"]
        assert row["observations"] == 2

    @pytest.mark.parametrize("reobserve_first", [False, True])
    def test_console_observation_drops_a_resolved_back_office_member(
        self, tmp_path: pathlib.Path, transport: CapturingTransport, *, reobserve_first: bool
    ) -> None:
        store = LocalStore(tmp_path)
        office = Page("failure", BACK_OFFICE_JOB, FRIDAY, "s3_throttling", "R" * 26)
        other = Page("failure", "data.daily", FRIDAY, "s3_throttling", "S" * 26)
        shared = group_pages([office, other])
        (key,) = emit(store, shared, now=SATURDAY_NIGHT, transport=transport, sweep_run_id=None)
        original = json.loads(store.get_bytes(key))
        if reobserve_first:
            emit(store, shared, now=SATURDAY_NIGHT, transport=transport, sweep_run_id=None)
        (last_key,) = emit(
            store, group_pages([other]), now=SATURDAY_NIGHT, transport=transport, sweep_run_id=None
        )
        assert last_key == key
        row = json.loads(store.get_bytes(key))
        assert BACK_OFFICE_JOB not in row["rendered"]
        assert "data.daily" in row["rendered"]
        assert row["rendered_initial"] == original["rendered"]
        assert row["members"] == original["members"]

    def test_secondary_evidence_access_fault_does_not_rekey_the_incident(
        self, tmp_path: pathlib.Path, transport: CapturingTransport
    ) -> None:
        class EvidenceFault(LocalStore):
            deny_evidence = False

            def get_bytes(self, key: str) -> bytes:
                if self.deny_evidence and key.startswith("trader/exception_scores/"):
                    raise PermissionError("s3_throttling while reading queue evidence")
                return super().get_bytes(key)

        store = EvidenceFault(tmp_path)
        self._published_queue(store, FRIDAY, [f"prop-{i}" for i in range(6)])
        first = sweep(store, now=SATURDAY_NIGHT, transport=transport, sweep_run_id="Z" * 26)
        (original,) = self._rows(store, first)
        store.deny_evidence = True
        last = sweep(store, now=SATURDAY_NIGHT, transport=transport, sweep_run_id="Y" * 26)
        (row,) = self._rows(store, last)
        assert row["cause_key"] == original["cause_key"]
        assert row["observations"] == 2
        assert "s3_throttling while reading queue evidence" in str(row["rendered"])
        assert len([c for c in transport.calls if BACK_OFFICE_JOB in c.message]) == 1

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
