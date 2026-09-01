"""The four faults, each end to end through the real runner and alerter.

Nothing here is mocked except the two things that would otherwise reach the
outside world: the alert transport (captured, so "exactly one page" is
observable) and the data source (a stub that raises, which is the fault).
"""

from __future__ import annotations

import datetime as dt
import json
import os
import signal

import pytest

from crucible.alerts import sweep
from crucible.manifest import manifest_key, validate
from crucible.release import POINTER_KEY, publish_release, resolve_release, wheel_key
from crucible.runner import RunContext, SpotInterruptionError, run_job, spot_interruption_guard
from crucible.store import ETAG_ABSENT, LocalStore

FRIDAY = dt.date(2026, 8, 28)
NOW = dt.datetime(2026, 8, 28, 21, 0, tzinfo=dt.UTC)
AFTER_EVERY_DEADLINE = dt.datetime(2026, 8, 29, 23, 30, tzinfo=dt.UTC)
SHA = "a" * 40


def _run_and_capture(store, job, body, *, retry=True):
    """Run one job, expect it to raise, and hand back its manifest."""
    with pytest.raises(BaseException):  # noqa: B017 - the fault is the point
        run_job(job, body, store=store, trading_day=FRIDAY, now=NOW, transient_retry=retry)
    return json.loads(store.get_bytes(manifest_key(job, FRIDAY.isoformat())))


def _assert_full_telemetry(manifest: dict) -> None:
    """All five §9.2 signal classes, under one correlation id, and valid.

    Asserted through the schema rather than field by field: a hand-written
    list of fields to check is written from the fields someone remembered.
    """
    validate(manifest)
    assert manifest["run_id"]
    assert manifest["resource"]["instance_type"]  # class 3
    assert manifest["cost_usd"] >= 0  # class 2
    assert "inputs" in manifest and "outputs" in manifest  # class 4
    assert isinstance(manifest["metrics"], list)  # class 5


def _failure_pages(store, transport) -> int:
    """How many FAILURE pages the alerter emits for the state a fault left.

    Failure pages only. Every one of these tests runs one job in an otherwise
    empty store, so the absence condition legitimately fires for the fifteen
    jobs the test never ran — counting those would make "exactly one page"
    unassertable, and loosening the absence condition to make the assertion
    convenient would be the gate bending to fit the test.
    """
    sweep(store, now=AFTER_EVERY_DEADLINE, transport=transport, sweep_run_id="0" * 26)
    return sum(1 for call in transport.calls if call.kwargs.get("source") == "crucible-v2/failure")


class TestFaultOneSpotTerminatedMidJob:
    """The most common real failure in the fleet, and the one that used to
    leave no manifest at all — an ABSENCE page with the cause discarded."""

    def test_it_fails_with_the_right_reason_full_telemetry_and_one_page(
        self, tmp_path, transport
    ) -> None:
        store = LocalStore(tmp_path)

        def reclaimed(ctx: RunContext) -> None:
            ctx.record_rows(rows_in=100, rows_out=0)
            os.kill(os.getpid(), signal.SIGTERM)

        with spot_interruption_guard():
            manifest = _run_and_capture(store, "data.daily", reclaimed, retry=False)

        assert manifest["status"] == "failed"
        assert "spot_interruption" in manifest["reason"]
        _assert_full_telemetry(manifest)
        assert _failure_pages(store, transport) == 1

    def test_the_declared_class_retries_it_once_before_paging(self, tmp_path, transport) -> None:
        """The whole point of the class: a 3am page for a reclamation that
        would have succeeded on a fresh instance is minimal-touch, not
        zero-touch."""
        store = LocalStore(tmp_path)
        calls = []

        def flaky(ctx: RunContext) -> None:
            calls.append(1)
            if len(calls) == 1:
                raise SpotInterruptionError("spot_interruption: reclaimed")

        run_job("data.daily", flaky, store=store, trading_day=FRIDAY, now=NOW)
        manifest = json.loads(store.get_bytes(manifest_key("data.daily", FRIDAY.isoformat())))
        assert manifest["status"] == "ok"
        assert [a["reason"] for a in manifest["attempts"]] == ["initial", "spot_interruption"]
        assert _failure_pages(store, transport) == 0


class TestFaultTwoDataSourceWithheld:
    def test_it_fails_with_the_right_reason_full_telemetry_and_one_page(
        self, tmp_path, transport
    ) -> None:
        store = LocalStore(tmp_path)

        def withheld(ctx: RunContext) -> None:
            ctx.record_rejected("data source withheld", 903)
            raise RuntimeError("data source yfinance returned no rows for 903 tickers")

        manifest = _run_and_capture(store, "data.daily", withheld)
        assert manifest["status"] == "failed"
        assert "data source" in manifest["reason"]
        assert manifest["rows_rejected"] == [{"reason": "data source withheld", "count": 903}]
        _assert_full_telemetry(manifest)
        assert _failure_pages(store, transport) == 1

    def test_a_withheld_source_is_not_in_the_transient_class(self, tmp_path) -> None:
        """It pages immediately. A vendor that returned nothing is not a
        thing a fresh instance fixes, and retrying it would delay the page by
        the length of a second run."""
        from crucible.runner import classify_transient

        assert classify_transient(RuntimeError("data source yfinance returned no rows")) is None

    def test_five_jobs_hitting_one_outage_produce_one_page(self, tmp_path, transport) -> None:
        """§9.3's requirement, induced rather than asserted about."""
        store = LocalStore(tmp_path)
        for job in ("data.daily", "experiment.run", "experiment.grade", "report", "drift"):

            def withheld(ctx: RunContext) -> None:
                raise RuntimeError("data source yfinance is unreachable")

            _run_and_capture(store, job, withheld)
        assert _failure_pages(store, transport) == 1


class TestFaultThreeRouterReturns500:
    def test_it_fails_with_the_right_reason_full_telemetry_and_one_page(
        self, tmp_path, transport
    ) -> None:
        store = LocalStore(tmp_path)

        def router_500(ctx: RunContext) -> None:
            ctx.record_llm_call(
                {
                    "callsite_id": "crucible.faults.router_probe",
                    "model_requested": "research-class",
                    "model_served": "none",
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "cache_read": 0,
                    "cache_write": 0,
                    "usd": 0.0,
                }
            )
            raise RuntimeError("provider_5xx: router returned 503 service unavailable")

        # Retry disabled so the FIRST failure is what is asserted; the retry
        # path has its own test above.
        manifest = _run_and_capture(store, "experiment.run", router_500, retry=False)
        assert manifest["status"] == "failed"
        assert "provider_5xx" in manifest["reason"]
        assert manifest["llm_calls"][0]["model_served"] == "none"
        _assert_full_telemetry(manifest)
        assert _failure_pages(store, transport) == 1

    def test_a_router_500_that_survives_the_retry_still_pages_exactly_once(
        self, tmp_path, transport
    ) -> None:
        store = LocalStore(tmp_path)

        def router_500(ctx: RunContext) -> None:
            raise RuntimeError("provider_5xx: 503")

        manifest = _run_and_capture(store, "experiment.run", router_500, retry=True)
        assert len(manifest["attempts"]) == 2
        assert _failure_pages(store, transport) == 1


class TestFaultFourStaleReleasePointer:
    def test_a_pointer_to_a_deleted_release_fails_the_job_rather_than_running_stale_code(
        self, tmp_path, transport
    ) -> None:
        """A job that installed nothing and carried on would run whatever was
        already on the box — the silent version of every deploy bug."""
        store = LocalStore(tmp_path)
        publish_release(
            store,
            sha=SHA,
            wheel=b"wheel",
            lockfile=b"lock",
            test_summary="ok",
            workflow_run_url="",
            now=NOW,
        )
        store.compare_and_swap(
            POINTER_KEY, ETAG_ABSENT, json.dumps({"sha": SHA, "target": "current"}).encode()
        )
        (tmp_path / wheel_key(SHA)).unlink()

        def boot(ctx: RunContext) -> None:
            resolve_release(store)

        manifest = _run_and_capture(store, "data.daily", boot)
        assert manifest["status"] == "failed"
        assert "StaleReleasePointerError" in manifest["reason"]
        _assert_full_telemetry(manifest)
        assert _failure_pages(store, transport) == 1

    def test_an_unset_pointer_refuses_rather_than_choosing_a_release(self, tmp_path) -> None:
        from crucible.release import StaleReleasePointerError

        with pytest.raises(StaleReleasePointerError, match="unset"):
            resolve_release(LocalStore(tmp_path))


class TestEveryFaultLeavesADurableRecord:
    def test_the_alerts_bus_carries_a_row_for_each_page(self, tmp_path, transport) -> None:
        """Principle 1: someone reconstructs why from durable artifacts
        alone, without asking Brian."""
        store = LocalStore(tmp_path)

        def withheld(ctx: RunContext) -> None:
            raise RuntimeError("data source yfinance is unreachable")

        _run_and_capture(store, "data.daily", withheld)
        result = sweep(store, now=AFTER_EVERY_DEADLINE, transport=transport, sweep_run_id="0" * 26)
        rows = [json.loads(store.get_bytes(k)) for k in result["bus_keys"]]
        assert any(r["condition"] == "failure" for r in rows)
        assert all(r["rendered"] for r in rows)
