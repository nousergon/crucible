"""The daily v2 pipelines notice (alpha-engine-config-I13334).

v1's daily "Pipelines" message counts v1 state machines only; on 2026-11-06 it
read "all 2 succeeded" the morning the v2 trader session never ran. These
tests pin the clean day, that day, the 2026-11-04 order-book failure, and the
sweep wiring: once per trading day, never on a weekend firing, a dry run or a
`--now` replay, and a Telegram outage never fails the sweep.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import dataclass
from typing import Any

import pytest

from crucible import daily_notice, track_c
from crucible.daily_chain import DAILY_CHAIN_SERVE_ORDER
from crucible.daily_notice import (
    NOTICE_SOURCE,
    ORDER_BOOK_METRIC,
    RunReading,
    daily_runs,
    order_book_line,
    read_runs,
    render_notice,
    send_notice,
)
from crucible.execution import EXECUTION_METRIC_NAME
from crucible.keys import manifest_key
from crucible.runner import RunContext
from crucible.store import LocalStore

DAY = "2026-11-05"
DENIED = (
    "order book not written: AccessDenied: An error occurred (AccessDenied) when calling "
    "the PutObject operation: <principal> is not authorized to perform: s3:PutObject"
)


@dataclass
class Result:
    any_ok: bool = True
    telegram_destination: str = "operator_chat"


class Transport:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, message: str, **kwargs: Any) -> Result:
        self.calls.append((message, kwargs))
        return Result()


def _metric(name: str, status: str = "OK", **extra: object) -> dict[str, object]:
    return {"name": name, "status": status, "status_reason": "", "value": 1.0, **extra}


def _write(
    store: LocalStore,
    job: str,
    slot: str | None = None,
    *,
    status: str = "ok",
    metrics: list[dict[str, object]] | None = None,
    day: str = DAY,
) -> None:
    doc = {"job": job, "trading_day": day, "status": status, "metrics": metrics or []}
    store.put_bytes(manifest_key(job, day, discriminator=slot), json.dumps(doc).encode())


def _clean_day(store: LocalStore, *, skip: tuple[str, ...] = ()) -> None:
    for job, slot in daily_runs():
        if job in skip:
            continue
        metrics = [_metric(ORDER_BOOK_METRIC)] if job == "trader.session" else []
        _write(store, job, slot, metrics=metrics)


class TestRunsOwed:
    def test_every_served_slot_is_owed(self) -> None:
        serve = [slot for job, slot in daily_runs() if job == "serve.daily"]
        assert tuple(serve) == DAILY_CHAIN_SERVE_ORDER
        assert [job for job, _ in daily_runs()][-2:] == ["trader.session", "trader.reconcile"]


class TestReadRuns:
    def test_missing_failed_and_unreadable_are_told_apart(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _write(store, "data.daily", status="failed")
        store.put_bytes(manifest_key("trader.reconcile", DAY), b"{not json")
        states = {r.label: r.state for r in read_runs(store, DAY)}
        assert states["data.daily"] == "failed"
        assert states["serve.daily m"] == "missing"
        assert states["trader.reconcile"] == "unreadable"

    def test_a_skip_if_ok_noop_is_not_read_as_the_run(self, tmp_path) -> None:
        """`--skip-if-ok` files under its own discriminator; only the
        canonical key is the day's run."""
        store = LocalStore(tmp_path)
        _write(store, "data.daily", f"{DAY}-already-ok")
        states = {r.label: r.state for r in read_runs(store, DAY)}
        assert states["data.daily"] == "missing"


class TestOrderBookLine:
    def _session(self, *metrics: dict[str, object]) -> RunReading:
        return RunReading("trader.session", None, "ok", {"metrics": list(metrics)})

    def test_written_with_fills(self) -> None:
        reading = self._session(
            _metric(ORDER_BOOK_METRIC), _metric(EXECUTION_METRIC_NAME, n_samples=12)
        )
        assert order_book_line(reading) == "Order book: written, 12 order(s) filled."

    def test_written_shadow_day_has_no_count(self) -> None:
        assert order_book_line(self._session(_metric(ORDER_BOOK_METRIC))) == "Order book: written."

    def test_failed_names_the_cause_not_the_resource(self) -> None:
        line = order_book_line(
            self._session(_metric(ORDER_BOOK_METRIC, "FAIL", status_reason=DENIED))
        )
        assert line == "Order book: FAILED (AccessDenied)."

    def test_no_row_is_missing(self) -> None:
        assert "missing" in order_book_line(self._session())

    def test_no_session_is_missing(self) -> None:
        line = order_book_line(RunReading("trader.session", None, "missing"))
        assert line == "Order book: missing (trader.session missing)."


class TestRender:
    def test_a_clean_day(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _clean_day(store)
        message = render_notice(DAY, read_runs(store, DAY))
        assert message.splitlines() == [
            f"<b>v2 pipelines {DAY}: 6 of 6 succeeded</b>",
            "Order book: written.",
            "No action needed.",
        ]

    def test_the_day_the_session_did_not_run(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _clean_day(store, skip=("trader.session",))
        message = render_notice(DAY, read_runs(store, DAY))
        assert message.splitlines() == [
            f"<b>v2 pipelines {DAY}: 5 of 6 succeeded — missing: trader.session</b>",
            "Order book: missing (trader.session missing).",
            f"Look at runs/trader.session/{DAY}/ first; failures and absences also page.",
        ]

    def test_a_failed_book_on_an_otherwise_clean_day_is_not_clean(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _clean_day(store)
        _write(
            store,
            "trader.session",
            metrics=[_metric(ORDER_BOOK_METRIC, "FAIL", status_reason=DENIED)],
        )
        lines = render_notice(DAY, read_runs(store, DAY)).splitlines()
        assert lines[0] == f"<b>v2 pipelines {DAY}: 6 of 6 succeeded</b>"
        assert lines[1] == "Order book: FAILED (AccessDenied)."
        assert lines[2].startswith(f"Look at runs/trader.session/{DAY}/")

    def test_failed_and_missing_are_grouped(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _clean_day(store, skip=("trader.session",))
        _write(store, "serve.daily", "s", status="failed")
        head = render_notice(DAY, read_runs(store, DAY)).splitlines()[0]
        assert head == (
            f"<b>v2 pipelines {DAY}: 4 of 6 succeeded — failed: serve.daily s; "
            "missing: trader.session</b>"
        )

    def test_reason_text_is_escaped(self) -> None:
        reading = RunReading(
            "trader.session",
            None,
            "ok",
            {"metrics": [_metric(ORDER_BOOK_METRIC, "FAIL", status_reason="x: <b>")]},
        )
        assert "&lt;b&gt;" in render_notice(DAY, [reading])


class TestSend:
    def test_delivers_as_info_on_the_operator_chat(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _clean_day(store)
        transport = Transport()
        assert send_notice(store, DAY, console_artifact="x", transport=transport) == "operator_chat"
        ((message, kwargs),) = transport.calls
        assert "6 of 6 succeeded" in message
        assert kwargs["severity"] == "info"
        assert kwargs["source"] == NOTICE_SOURCE
        assert kwargs["sns"] is False
        assert kwargs["dedup_key"] is None


def _ctx(started: dt.datetime, trading_day: dt.date) -> RunContext:
    return RunContext(
        job="alerts.sweep",
        run_id="Z" * 26,
        trading_day=trading_day,
        calendar_date=started.date(),
        started=started,
        seed=0,
        store=None,  # pyright: ignore[reportArgumentType]
    )


class TestSweepWiring:
    @pytest.fixture
    def sent(self, monkeypatch) -> list[tuple[str, str]]:
        calls: list[tuple[str, str]] = []

        def fake(_store: object, day: str, *, console_artifact: str) -> str:
            calls.append((day, console_artifact))
            return "operator_chat"

        monkeypatch.setattr(daily_notice, "send_notice", fake)
        return calls

    def test_a_trading_evening_reports_the_previous_trading_day(self, tmp_path, sent) -> None:
        # Monday 2026-11-09 21:00 ET is 02:00Z on the 10th.
        ctx = _ctx(dt.datetime(2026, 11, 10, 2, 0, tzinfo=dt.UTC), dt.date(2026, 11, 9))
        track_c._send_daily_notice(LocalStore(tmp_path), ctx)
        assert sent == [("2026-11-06", "runs/alerts.sweep/2026-11-09/2026-11-10/run.json")]

    def test_a_weekend_firing_sends_nothing(self, tmp_path, sent) -> None:
        # Saturday 2026-11-07 21:00 ET resolves to Friday's trading day.
        ctx = _ctx(dt.datetime(2026, 11, 8, 2, 0, tzinfo=dt.UTC), dt.date(2026, 11, 6))
        track_c._send_daily_notice(LocalStore(tmp_path), ctx)
        assert sent == []

    def test_a_delivery_failure_never_fails_the_sweep(self, tmp_path, monkeypatch, capsys):
        def boom(*_a: object, **_k: object) -> str:
            raise RuntimeError("telegram down")

        monkeypatch.setattr(daily_notice, "send_notice", boom)
        ctx = _ctx(dt.datetime(2026, 11, 10, 2, 0, tzinfo=dt.UTC), dt.date(2026, 11, 9))
        track_c._send_daily_notice(LocalStore(tmp_path), ctx)
        assert "daily notice NOT delivered: RuntimeError: telegram down" in capsys.readouterr().err

    def test_the_handler_sends_after_the_run(self, tmp_path, monkeypatch) -> None:
        sent: list[RunContext] = []
        monkeypatch.setattr(track_c, "_send_daily_notice", lambda _s, ctx: sent.append(ctx))
        monkeypatch.setattr(track_c.alerts, "sweep", _fake_sweep)
        args = argparse.Namespace(
            store=str(tmp_path), dry_run=False, run_mode="live", trading_day=None, now=None
        )
        track_c.sweep_handler(args)
        assert len(sent) == 1
        assert sent[0].job == "alerts.sweep"

    def test_a_dry_run_or_a_replay_sends_nothing(self, tmp_path, monkeypatch) -> None:
        sent: list[RunContext] = []
        monkeypatch.setattr(track_c, "_send_daily_notice", lambda _s, ctx: sent.append(ctx))
        monkeypatch.setattr(track_c.alerts, "sweep", _fake_sweep)
        args = argparse.Namespace(
            store=str(tmp_path), dry_run=True, run_mode="live", trading_day=None, now=None
        )
        track_c.sweep_handler(args)
        assert sent == []


def _fake_sweep(_store: object, **_kwargs: object) -> dict[str, Any]:
    return {
        "bus_keys": (),
        "pages_emitted": 0,
        "members": 0,
        "metric": {
            "name": "pages_per_month",
            "module": "crucible.alerts",
            "metric_type": "operational",
            "value": 0.0,
            "unit": "pages",
            "n_floor": 0,
            "status": "OK",
            "status_reason": "test",
            "source_path": "alerts/",
            "last_updated_utc": "2026-11-10T02:00:00Z",
        },
    }
