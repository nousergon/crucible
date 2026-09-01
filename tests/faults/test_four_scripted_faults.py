"""The four faults, each end to end through the real runner and alerter.

Nothing here is mocked except the two things that would otherwise reach the
outside world: the alert transport (captured, so "exactly one page" is
observable), the data source (`FramePriceSource` withholding every requested
ticker, which is the fault fault 2 induces) and the LLM transport
(`krepis.llm.LLMClient` itself, replaced with a stub whose `complete()`
raises the fault fault 3 induces — the same seam `tests/test_llm_cap.py`
patches to exercise the cap without a provider).

Faults 2 and 3 used to `raise RuntimeError(...)` directly in the job body
and then assert the manifest carried the string the test itself wrote —
alpha-engine-config-I9780. Neither could fail from a defect in the data
layer or the LLM path; both now run the real production seam
(`crucible.data.daily.run_daily` / `crucible.llm.call`) and let IT raise.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import signal

import pytest

from crucible.alerts import sweep
from crucible.data.daily import run_daily
from crucible.data.sources import FramePriceSource
from crucible.llm import CallSite, SpendCap, call as llm_call
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
        """Induced at the seam: `FramePriceSource` withholds every ticker
        `run_daily` asks for — the same shape a live ArcticDB outage takes,
        since `ArcticPriceSource.load_panel` raises `MissingSourceError` the
        identical way when a requested symbol is absent from what it read.
        `run_daily` never gets to compute a coverage ratio; the source
        refuses before the panel exists. A change that made either source
        zero-fill an absent ticker instead of raising turns this test red —
        the 901-of-903 bug class the coverage floor exists to catch, moved
        one layer up to where it would never reach the floor at all."""
        store = LocalStore(tmp_path)

        def withheld(ctx: RunContext) -> None:
            run_daily(
                ctx,
                source=FramePriceSource({}),
                expected_symbols=["AAA", "BBB", "CCC"],
            )

        manifest = _run_and_capture(store, "data.daily", withheld)
        assert manifest["status"] == "failed"
        assert "MissingSourceError" in manifest["reason"]
        assert "AAA" in manifest["reason"]
        _assert_full_telemetry(manifest)
        assert _failure_pages(store, transport) == 1

    def test_a_withheld_source_is_not_in_the_transient_class(self, tmp_path) -> None:
        """It pages immediately. A vendor that returned nothing is not a
        thing a fresh instance fixes, and retrying it would delay the page by
        the length of a second run."""
        from crucible.data.sources import MissingSourceError
        from crucible.runner import classify_transient

        assert classify_transient(MissingSourceError("source frame(s) are empty")) is None

    def test_five_jobs_hitting_one_outage_produce_one_page(self, tmp_path, transport) -> None:
        """§9.3's requirement, induced rather than asserted about — the same
        outage hit by five different jobs collapses to one page.

        This asserts `crucible.alerts.group_pages`, not the fault: what
        makes five reasons "the same outage" is `CAUSE_MATCHERS` in
        `crucible/alerts.py` keying on a substring in `reason` (`"data
        source"` / `"yfinance"` / `"fred"`), and no production exception in
        `crucible.data.sources` carries that literal phrase — the file this
        test does not own. The fault itself (`FramePriceSource` withholding
        a ticker through `run_daily`) is induced in full, once, above; this
        test is entitled to script the reason text because the grouping
        behaviour it exercises lives in a different module than the one the
        fault touches.
        """
        store = LocalStore(tmp_path)
        for job in ("data.daily", "experiment.run", "experiment.grade", "report", "drift"):

            def withheld(ctx: RunContext) -> None:
                raise RuntimeError("data source yfinance is unreachable")

            _run_and_capture(store, job, withheld)
        assert _failure_pages(store, transport) == 1


_ROUTER_500_SITE = CallSite(
    callsite_id="faults.router_probe",
    purpose="induce a provider 5xx through crucible.llm.call",
    capability_class="high",
    max_usd_per_call=1.0,
    owner="tests.faults",
)


class _Provider5xxClient:
    """The transport `crucible.llm.call` reaches. `.complete()` is the one
    call surface `LLMClient.complete` invokes on it (`messages.create` /
    `chat.completions.create` on a real transport); raising here is a
    provider 5xx arriving on the wire, upstream of `LLMClient` itself."""

    def complete(self, **_kwargs: object) -> None:
        raise RuntimeError("provider_5xx: router returned 503 service unavailable")


class TestFaultThreeRouterReturns500:
    def test_it_fails_with_the_right_reason_full_telemetry_and_one_page(
        self, tmp_path, transport, monkeypatch
    ) -> None:
        """Induced at the seam: `krepis.llm.LLMClient` — what `crucible.llm.call`
        constructs and calls `.complete()` on — is replaced with a stub that
        raises a 503. The cap reservation, the registry lookup and the
        ceiling check in `crucible.llm.call` all run unmocked; only the
        provider transport is faked. A change that swallowed the provider
        error (a bare `except` around `client.complete(...)`) turns this test
        red — no manifest would carry `status: failed` at all."""
        store = LocalStore(tmp_path)
        # `resolve_model_spec` would otherwise reach SSM/env for a real
        # deployment spec — irrelevant to what this fault exercises, which is
        # what `crucible.llm.call` does once the transport fails. Same seam
        # `tests/test_llm_cap.py::test_a_bill_above_the_admitted_ceiling_fails_the_run_and_is_recorded`
        # patches.
        monkeypatch.setattr("krepis.llm_config.resolve_model_spec", lambda **_kw: object())
        monkeypatch.setattr("krepis.llm.LLMClient", lambda *a, **k: _Provider5xxClient())

        def router_500(ctx: RunContext) -> None:
            llm_call(
                ctx,
                callsite_id=_ROUTER_500_SITE.callsite_id,
                capability_class=_ROUTER_500_SITE.capability_class,
                messages=[{"role": "user", "content": "probe"}],
                cap=SpendCap(cap_usd=5.0),
                estimate_usd=0.01,
                client_factory=lambda *a, **k: _Provider5xxClient(),
                registry={_ROUTER_500_SITE.callsite_id: _ROUTER_500_SITE},
            )

        # Retry disabled so the FIRST failure is what is asserted; the retry
        # path has its own test above.
        manifest = _run_and_capture(store, "experiment.run", router_500, retry=False)
        assert manifest["status"] == "failed"
        assert "provider_5xx" in manifest["reason"]
        assert manifest["llm_calls"] == [], (
            "the call raised before `crucible.llm.call` could record it — nothing "
            "was billed for a call that never completed"
        )
        _assert_full_telemetry(manifest)
        assert _failure_pages(store, transport) == 1

    def test_a_router_500_that_survives_the_retry_still_pages_exactly_once(
        self, tmp_path, transport, monkeypatch
    ) -> None:
        store = LocalStore(tmp_path)
        # `resolve_model_spec` would otherwise reach SSM/env for a real
        # deployment spec — irrelevant to what this fault exercises, which is
        # what `crucible.llm.call` does once the transport fails. Same seam
        # `tests/test_llm_cap.py::test_a_bill_above_the_admitted_ceiling_fails_the_run_and_is_recorded`
        # patches.
        monkeypatch.setattr("krepis.llm_config.resolve_model_spec", lambda **_kw: object())
        monkeypatch.setattr("krepis.llm.LLMClient", lambda *a, **k: _Provider5xxClient())

        def router_500(ctx: RunContext) -> None:
            llm_call(
                ctx,
                callsite_id=_ROUTER_500_SITE.callsite_id,
                capability_class=_ROUTER_500_SITE.capability_class,
                messages=[{"role": "user", "content": "probe"}],
                cap=SpendCap(cap_usd=5.0),
                estimate_usd=0.01,
                client_factory=lambda *a, **k: _Provider5xxClient(),
                registry={_ROUTER_500_SITE.callsite_id: _ROUTER_500_SITE},
            )

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
