"""`alpha-engine-config-I13334`: a served session whose order book was not kept pages.

`crucible_trader.order_book.record_order_book` never raises into the session,
so a failed write is an `ok` manifest with `trader_order_book_written` at FAIL.
On 2026-11-04 that row read `AccessDenied` and paged nobody: the manifest
existed (no ABSENCE) and was `ok` (no FAILURE). `evaluate_order_book` reuses
the FAILURE condition under the `trader.order_book` label.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
from collections.abc import Iterator

import pytest

from crucible.alerts import (
    ORDER_BOOK_JOB,
    ORDER_BOOK_METRIC,
    ORDER_BOOK_REQUIRED_FROM,
    SESSION_SERVED_METRIC,
    TRADER_SESSION_JOB,
    StoreAccessError,
    evaluate_order_book,
    sweep,
)
from crucible.components import load_registry
from crucible.manifest import manifest_key
from crucible.runmode import RUN_MODE_REPLAY
from crucible.store import LocalStore

FRIDAY = dt.date(2026, 11, 6)
SATURDAY_NIGHT = dt.datetime(2026, 11, 7, 23, 30, tzinfo=dt.UTC)
#: The shape of the 2026-11-04 row (measured read-only), with the resource
#: names replaced: this repo is public.
DENIED = (
    "order book not written: AccessDenied: An error occurred (AccessDenied) when calling "
    "the PutObject operation: <principal> is not authorized to perform: s3:PutObject"
)


def _row(name: str, status: str, value: float, reason: str) -> dict[str, object]:
    return {
        "name": name,
        "module": "crucible_trader.order_book",
        "metric_type": "operational",
        "value": value,
        "unit": "count",
        "n_floor": 0,
        "status": status,
        "status_reason": reason,
        "source_path": "trader/order_books/",
        "last_updated_utc": "2026-11-09T15:46:00Z",
    }


def _served(value: float = 1.0) -> dict[str, object]:
    return _row(SESSION_SERVED_METRIC, "OK", value, "live session on m:x")


def _book(status: str = "OK", reason: str = "41 names written") -> dict[str, object]:
    return _row(ORDER_BOOK_METRIC, status, 1.0 if status == "OK" else 0.0, reason)


def _write_session(
    store: LocalStore,
    metrics: list[dict[str, object]],
    *,
    day: dt.date = FRIDAY,
    status: str = "ok",
    run_id: str = "01JG0000000000000000000OB1",
    discriminator: str | None = None,
    run_mode: str | None = None,
) -> None:
    payload: dict[str, object] = {
        "schema_version": "run_manifest.v2",
        "run_id": run_id,
        "job": TRADER_SESSION_JOB,
        "trading_day": day.isoformat(),
        "calendar_date": "2026-11-09",
        "status": status,
        "reason": "" if status == "ok" else "boom",
        "cost_usd": 0.0,
        "metrics": metrics,
        "attempts": [{"n": 1, "reason": "initial"}],
    }
    if discriminator is not None:
        payload["discriminator"] = discriminator
    if run_mode is not None:
        payload["run_mode"] = run_mode
    store.put_bytes(
        manifest_key(TRADER_SESSION_JOB, day.isoformat(), discriminator=discriminator),
        json.dumps(payload).encode(),
    )


class TestEvaluateOrderBook:
    def test_a_failed_write_pages_with_the_trader_reason(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _write_session(store, [_served(), _book("FAIL", DENIED)])
        (page,) = evaluate_order_book(store, now=SATURDAY_NIGHT)
        assert page.condition == "failure"
        assert page.job == ORDER_BOOK_JOB
        assert page.trading_day == FRIDAY
        assert page.run_id == "01JG0000000000000000000OB1"
        assert "AccessDenied" in page.reason
        assert "Auto-fix:" in page.reason
        assert "Resolution:" in page.reason

    def test_a_written_book_pages_nothing(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _write_session(store, [_served(), _book()])
        assert evaluate_order_book(store, now=SATURDAY_NIGHT) == []

    def test_a_served_session_with_no_book_row_pages(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _write_session(store, [_served()])
        (page,) = evaluate_order_book(store, now=SATURDAY_NIGHT)
        assert ORDER_BOOK_METRIC in page.reason

    def test_a_held_session_is_left_to_the_held_rule(self, tmp_path) -> None:
        """A held session builds no book; `evaluate_held_session` pages it."""
        store = LocalStore(tmp_path)
        _write_session(store, [_served(0.0)])
        assert evaluate_order_book(store, now=SATURDAY_NIGHT) == []

    def test_a_failed_session_is_left_to_the_failure_condition(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _write_session(store, [_served(), _book("FAIL", DENIED)], status="failed")
        assert evaluate_order_book(store, now=SATURDAY_NIGHT) == []

    def test_no_manifest_is_left_to_the_absence_condition(self, tmp_path) -> None:
        assert evaluate_order_book(LocalStore(tmp_path), now=SATURDAY_NIGHT) == []

    def test_a_day_before_the_book_existed_is_not_graded(self, tmp_path) -> None:
        day = ORDER_BOOK_REQUIRED_FROM - dt.timedelta(days=1)
        store = LocalStore(tmp_path)
        _write_session(store, [_served()], day=day)
        evening = dt.datetime.combine(day + dt.timedelta(days=1), dt.time(2, 30), tzinfo=dt.UTC)
        assert evaluate_order_book(store, now=evening) == []

    def test_a_later_run_that_wrote_the_book_clears_the_day(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _write_session(store, [_served(), _book("FAIL", DENIED)], discriminator="a")
        _write_session(store, [_served(), _book()], discriminator="b", run_id="0" * 25 + "B")
        assert evaluate_order_book(store, now=SATURDAY_NIGHT) == []

    def test_a_replay_that_wrote_the_book_does_not_clear_production(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _write_session(store, [_served(), _book("FAIL", DENIED)], discriminator="a")
        _write_session(
            store,
            [_served(), _book()],
            discriminator="b",
            run_id="0" * 25 + "B",
            run_mode=RUN_MODE_REPLAY,
        )
        (page,) = evaluate_order_book(store, now=SATURDAY_NIGHT)
        assert page.synthetic is None

    def test_a_retired_trader_session_row_pages_nothing(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _write_session(store, [_served(), _book("FAIL", DENIED)])
        registry = load_registry()
        retired = {
            **registry,
            TRADER_SESSION_JOB: dataclasses.replace(
                registry[TRADER_SESSION_JOB], lifecycle="RETIRED"
            ),
        }
        assert evaluate_order_book(store, now=SATURDAY_NIGHT, registry=retired) == []

    def test_an_unlistable_prefix_is_an_access_fault_not_silence(self, tmp_path) -> None:
        class _Denied(LocalStore):
            def list_keys(self, prefix: str = "") -> Iterator[str]:
                if prefix.startswith(f"runs/{TRADER_SESSION_JOB}/"):
                    raise PermissionError("AccessDenied")
                return super().list_keys(prefix)

        with pytest.raises(StoreAccessError):
            evaluate_order_book(_Denied(tmp_path), now=SATURDAY_NIGHT)
        faults: list[str] = []
        assert (
            evaluate_order_book(_Denied(tmp_path), now=SATURDAY_NIGHT, access_faults=faults) == []
        )


class TestTheRegistryDeclaresTheBook:
    def test_trader_session_declares_the_row_and_the_prefix(self) -> None:
        signals = load_registry()[TRADER_SESSION_JOB].signals
        assert ORDER_BOOK_METRIC in str(signals["outcome"])
        assert "trader/order_books/*" in str(signals["lineage"])


class TestTheSweepPagesAMissingBook:
    def test_the_sweep_sends_one_page_for_a_failed_book(self, tmp_path, transport) -> None:
        store = LocalStore(tmp_path)
        _write_session(store, [_served(), _book("FAIL", DENIED)])
        result = sweep(store, now=SATURDAY_NIGHT, transport=transport, sweep_run_id="Z" * 26)
        rows = [json.loads(store.get_bytes(k)) for k in result["bus_keys"]]
        book = [r for r in rows if any(m["job"] == ORDER_BOOK_JOB for m in r["members"])]
        assert len(book) == 1
        assert book[0]["condition"] == "failure"
        assert book[0]["sent"] is True

    def test_a_written_book_sends_no_page(self, tmp_path, transport) -> None:
        store = LocalStore(tmp_path)
        _write_session(store, [_served(), _book()])
        result = sweep(store, now=SATURDAY_NIGHT, transport=transport, sweep_run_id="Z" * 26)
        rows = [json.loads(store.get_bytes(k)) for k in result["bus_keys"]]
        assert not [r for r in rows if any(m["job"] == ORDER_BOOK_JOB for m in r["members"])]
