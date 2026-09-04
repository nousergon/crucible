"""The five LLM failure classes `test_four_scripted_faults.py` never induced.

`TestFaultThreeRouterReturns500` (in the sibling module) covers exactly one
`krepis` LLM failure class: an unclassified transport error that
`crucible.runner.classify_transient` recognises as `provider_5xx`. `krepis`
distinguishes five more, each requiring different behaviour from a crucible
run, and none of them had an induced test — alpha-engine-config-I9973.

Same discipline as the sibling module: nothing here is mocked except the LLM
transport (`krepis.llm.LLMClient`), replaced with a stub whose `complete()`
raises (or returns) exactly what the real transport would for the induced
class. The cap reservation, the registry lookup, the ceiling check and the
retry/paging path in `crucible.llm.call` / `crucible.runner.run_job` all run
unmocked.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from crucible.alerts import sweep
from crucible.llm import CallSite, SpendCap
from crucible.llm import call as llm_call
from crucible.manifest import manifest_key, validate
from crucible.runner import RunContext, classify_transient, run_job
from crucible.store import LocalStore

FRIDAY = dt.date(2026, 8, 28)
NOW = dt.datetime(2026, 8, 28, 21, 0, tzinfo=dt.UTC)
AFTER_EVERY_DEADLINE = dt.datetime(2026, 8, 29, 23, 30, tzinfo=dt.UTC)


def _run_and_capture(store, job, body, *, retry=True):
    """Run one job, expect it to raise, and hand back its manifest."""
    with pytest.raises(BaseException):  # noqa: B017 - the fault is the point
        run_job(job, body, store=store, trading_day=FRIDAY, now=NOW, transient_retry=retry)
    return json.loads(store.get_bytes(manifest_key(job, FRIDAY.isoformat())))


def _assert_full_telemetry(manifest: dict) -> None:
    """All five §9.2 signal classes, under one correlation id, and valid."""
    validate(manifest)
    assert manifest["run_id"]
    assert manifest["resource"]["instance_type"]
    assert manifest["cost_usd"] >= 0
    assert "inputs" in manifest and "outputs" in manifest
    assert isinstance(manifest["metrics"], list)


def _failure_pages(store, transport) -> int:
    """How many FAILURE pages the alerter emits for the state a fault left.

    Same convention as `test_four_scripted_faults.py::_failure_pages`: each
    test here runs one or two jobs in an otherwise empty store, so counting
    only `source == "crucible-v2/failure"` keeps "exactly one page" (or
    "exactly N pages" for the multi-job cases below) assertable.
    """
    sweep(store, now=AFTER_EVERY_DEADLINE, transport=transport, sweep_run_id="0" * 26)
    return sum(1 for call in transport.calls if call.kwargs.get("source") == "crucible-v2/failure")


def _stub_router(monkeypatch) -> None:
    """The registry/router edge stubbed the same way as `test_llm_router_route.py`
    and `TestFaultThreeRouterReturns500`: irrelevant to every fault below, so
    it is faked identically rather than re-derived per test."""
    monkeypatch.setenv("KREPIS_EXEC_CONTEXT", "ci")
    monkeypatch.setattr("krepis.router.resolve_group_spec", lambda *a, **k: (object(), {}))
    monkeypatch.setattr("krepis.router.route_is_degraded", lambda _route: False)


# --------------------------------------------------------------------------
# Deliverable 1: 429 / rate limit stays in the AVAILABILITY class.
# --------------------------------------------------------------------------

_RATE_LIMIT_SITE = CallSite(
    callsite_id="faults.rate_limit_probe",
    purpose="induce a 429 through crucible.llm.call",
    capability_class="high",
    max_usd_per_call=1.0,
    owner="tests.faults",
)


class _RateLimitError(RuntimeError):
    """Text and `status_code` match what `litellm.RateLimitError` carries on
    a real 429 — a router-level fallback chain already exhausted, so what
    reaches `crucible.llm.call` is what is left after `krepis`'s own
    failover gave up."""

    def __init__(self) -> None:
        super().__init__(
            "litellm.RateLimitError: RateLimitError: rate limit reached for "
            "model group low-deepseek-v4-flash-low"
        )
        self.status_code = 429


class _RateLimit429Client:
    def complete(self, **_kwargs: object) -> None:
        raise _RateLimitError()


class TestFaultRateLimit429StaysAvailability:
    def test_429_is_classified_as_availability_not_permanent(self) -> None:
        from krepis.llm_errors import is_permanent_contract_error

        assert is_permanent_contract_error(_RateLimitError()) is False, (
            "krepis.llm_errors.TRANSIENT_4XX_STATUSES keeps 429 in the availability "
            "class deliberately, so ordinary rate-limit failover keeps working"
        )

    def test_it_fails_with_the_right_reason_full_telemetry_and_one_page(
        self, tmp_path, transport, monkeypatch
    ) -> None:
        store = LocalStore(tmp_path)
        _stub_router(monkeypatch)
        monkeypatch.setattr("krepis.llm.LLMClient", lambda *a, **k: _RateLimit429Client())

        def rate_limited(ctx: RunContext) -> None:
            llm_call(
                ctx,
                callsite_id=_RATE_LIMIT_SITE.callsite_id,
                capability_class=_RATE_LIMIT_SITE.capability_class,
                messages=[{"role": "user", "content": "probe"}],
                cap=SpendCap(cap_usd=5.0),
                estimate_usd=0.01,
                client_factory=lambda *a, **k: _RateLimit429Client(),
                registry={_RATE_LIMIT_SITE.callsite_id: _RATE_LIMIT_SITE},
            )

        manifest = _run_and_capture(store, "experiment.run", rate_limited, retry=False)
        assert manifest["status"] == "failed"
        assert "rate limit" in manifest["reason"].lower()
        assert manifest["llm_calls"] == [], (
            "the call raised before crucible.llm.call could record it — nothing "
            "was billed for a call that never completed"
        )
        _assert_full_telemetry(manifest)
        assert _failure_pages(store, transport) == 1


# --------------------------------------------------------------------------
# Deliverable 2: permanent contract 4xx — never retried, champion's cause
# surfaces, the fallback's own noise survives only as a labelled aside.
# --------------------------------------------------------------------------

_PERMANENT_4XX_SITE = CallSite(
    callsite_id="faults.permanent_contract_probe",
    purpose="induce a permanent contract 4xx through crucible.llm.call",
    capability_class="high",
    max_usd_per_call=1.0,
    owner="tests.faults",
)

#: The exact shape alpha-engine-config-I7904 measured: a 400 that litellm
#: decorates with the fallback chain it went on to try.
_CHAMPION_CAUSE = "OpenAIException - Thinking mode does not support this tool_choice."
_FALLBACK_NOISE = (
    "Error doing the fallback: litellm.RateLimitError - Rate limit reached\n"
    "No fallback model group found for original model_group=glm-4.7-flash."
)


def _sample_permanent_contract_error():
    from krepis.llm_errors import PermanentContractError

    return PermanentContractError(
        _CHAMPION_CAUSE,
        status_code=400,
        deployment="low-deepseek-v4-flash-low",
        model_group="low-deepseek-v4-flash-low",
        suppressed_fallback_error=_FALLBACK_NOISE,
    )


class _PermanentContract4xxClient:
    def complete(self, **_kwargs: object) -> None:
        raise _sample_permanent_contract_error()


class TestFaultPermanentContract4xx:
    def test_a_400_is_classified_permanent_not_availability(self) -> None:
        from krepis.llm_errors import is_permanent_contract_error

        assert is_permanent_contract_error(_sample_permanent_contract_error()) is True

    def test_it_is_not_retried_and_surfaces_the_champions_cause(
        self, tmp_path, transport, monkeypatch
    ) -> None:
        store = LocalStore(tmp_path)
        _stub_router(monkeypatch)
        monkeypatch.setattr("krepis.llm.LLMClient", lambda *a, **k: _PermanentContract4xxClient())

        def permanent_4xx(ctx: RunContext) -> None:
            llm_call(
                ctx,
                callsite_id=_PERMANENT_4XX_SITE.callsite_id,
                capability_class=_PERMANENT_4XX_SITE.capability_class,
                messages=[{"role": "user", "content": "probe"}],
                cap=SpendCap(cap_usd=5.0),
                estimate_usd=0.01,
                client_factory=lambda *a, **k: _PermanentContract4xxClient(),
                registry={_PERMANENT_4XX_SITE.callsite_id: _PERMANENT_4XX_SITE},
            )

        # `retry=True`: transient_retry is ON, so the assertion that only one
        # attempt happened is about classify_transient refusing this
        # exception, not about the retry path being disabled at the call.
        manifest = _run_and_capture(store, "experiment.run", permanent_4xx, retry=True)

        assert manifest["status"] == "failed"
        assert _CHAMPION_CAUSE in manifest["reason"], (
            "the champion's own message must lead the surfaced reason"
        )
        assert "NOT THE CAUSE" in manifest["reason"], (
            "the fallback's half survives only as a labelled aside"
        )
        assert manifest["reason"].index(_CHAMPION_CAUSE) < manifest["reason"].index(
            "NOT THE CAUSE"
        ), "the champion's cause leads; the fallback's noise trails, labelled"
        assert len(manifest["attempts"]) == 1, (
            "a permanent contract error must not consume the retry budget"
        )
        assert classify_transient(_sample_permanent_contract_error()) is None
        assert manifest["llm_calls"] == []
        _assert_full_telemetry(manifest)
        assert _failure_pages(store, transport) == 1


# --------------------------------------------------------------------------
# Deliverable 3: stream idle vs stream total timeout — distinguishable.
# --------------------------------------------------------------------------

_STREAM_IDLE_SITE = CallSite(
    callsite_id="faults.stream_idle_probe",
    purpose="induce a stream idle timeout through crucible.llm.call",
    capability_class="high",
    max_usd_per_call=1.0,
    owner="tests.faults",
)
_STREAM_TOTAL_SITE = CallSite(
    callsite_id="faults.stream_total_probe",
    purpose="induce a stream total timeout through crucible.llm.call",
    capability_class="high",
    max_usd_per_call=1.0,
    owner="tests.faults",
)


class _StreamIdleTimeoutClient:
    def complete(self, **_kwargs: object) -> None:
        from krepis.llm import StreamIdleTimeoutError

        raise StreamIdleTimeoutError(
            "stream_idle_timeout: 1,842 characters arrived across 96 chunks, then nothing for 90s",
            chunks=96,
            idle_timeout=90.0,
            elapsed=90.0,
        )


class _StreamTotalTimeoutClient:
    def complete(self, **_kwargs: object) -> None:
        from krepis.llm import StreamTotalTimeoutError

        raise StreamTotalTimeoutError(
            "stream_total_timeout: the stream kept producing chunks for 610s "
            "against a 600s total budget",
            chunks=340,
            total_timeout=600.0,
            elapsed=610.0,
        )


class TestFaultStreamTimeoutsAreDistinguishable:
    def _run(self, tmp_path, monkeypatch, client_cls, site, job, subdir):
        store = LocalStore(tmp_path / subdir)
        _stub_router(monkeypatch)
        monkeypatch.setattr("krepis.llm.LLMClient", lambda *a, **k: client_cls())

        def body(ctx: RunContext) -> None:
            llm_call(
                ctx,
                callsite_id=site.callsite_id,
                capability_class=site.capability_class,
                messages=[{"role": "user", "content": "probe"}],
                cap=SpendCap(cap_usd=5.0),
                estimate_usd=0.01,
                client_factory=lambda *a, **k: client_cls(),
                registry={site.callsite_id: site},
            )

        manifest = _run_and_capture(store, job, body, retry=False)
        return store, manifest

    def test_idle_and_total_timeouts_produce_different_manifest_reasons(
        self, tmp_path, transport, monkeypatch
    ) -> None:
        idle_store, idle_manifest = self._run(
            tmp_path,
            monkeypatch,
            _StreamIdleTimeoutClient,
            _STREAM_IDLE_SITE,
            "experiment.run",
            "idle",
        )
        total_store, total_manifest = self._run(
            tmp_path,
            monkeypatch,
            _StreamTotalTimeoutClient,
            _STREAM_TOTAL_SITE,
            "experiment.grade",
            "total",
        )

        assert idle_manifest["status"] == "failed"
        assert total_manifest["status"] == "failed"
        idle_kind = idle_manifest["reason"].split(":", 1)[0]
        total_kind = total_manifest["reason"].split(":", 1)[0]
        assert idle_kind == "StreamIdleTimeoutError"
        assert total_kind == "StreamTotalTimeoutError"
        assert idle_manifest["reason"] != total_manifest["reason"], (
            "a future collapse of the two exception types into one turns this red"
        )
        _assert_full_telemetry(idle_manifest)
        _assert_full_telemetry(total_manifest)
        # One `transport` shared across both sweeps (each store its own —
        # the failure-page filter's absence condition needs an otherwise
        # empty store per §"_failure_pages"'s own docstring), so the second
        # reading is CUMULATIVE: one page for the idle job's failure, plus
        # one more, distinct, for the total job's.
        assert _failure_pages(idle_store, transport) == 1
        assert _failure_pages(total_store, transport) == 2


# --------------------------------------------------------------------------
# Deliverable 4: budget exhausted — fails, and books no spend for a call
# that produced nothing.
# --------------------------------------------------------------------------

_BUDGET_SITE = CallSite(
    callsite_id="faults.budget_exhausted_probe",
    purpose="induce a budget-exhausted failure through crucible.llm.call",
    capability_class="high",
    max_usd_per_call=1.0,
    owner="tests.faults",
)


class _BudgetExhaustedClient:
    def complete(self, **_kwargs: object) -> None:
        from krepis.llm import BudgetExhaustedError

        raise BudgetExhaustedError(
            "provider=stub model=stub-reasoning: the completion budget was "
            "exhausted before any content was produced — max_tokens=256, "
            "finish_reason='length', reasoning_tokens=256."
        )


class TestFaultBudgetExhausted:
    def test_it_fails_without_booking_spend_for_a_call_that_produced_nothing(
        self, tmp_path, transport, monkeypatch
    ) -> None:
        store = LocalStore(tmp_path)
        _stub_router(monkeypatch)
        monkeypatch.setattr("krepis.llm.LLMClient", lambda *a, **k: _BudgetExhaustedClient())
        cap = SpendCap(cap_usd=5.0)

        def budget_exhausted(ctx: RunContext) -> None:
            llm_call(
                ctx,
                callsite_id=_BUDGET_SITE.callsite_id,
                capability_class=_BUDGET_SITE.capability_class,
                messages=[{"role": "user", "content": "probe"}],
                cap=cap,
                estimate_usd=0.01,
                client_factory=lambda *a, **k: _BudgetExhaustedClient(),
                registry={_BUDGET_SITE.callsite_id: _BUDGET_SITE},
            )

        manifest = _run_and_capture(store, "experiment.run", budget_exhausted, retry=False)
        assert manifest["status"] == "failed"
        assert "budget" in manifest["reason"].lower()
        assert manifest["llm_calls"] == [], (
            "the call raised before crucible.llm.call could record a row for it"
        )
        assert cap.spent_usd == 0.0, (
            "a call that produced nothing must not book spend against the weekly cap "
            "— SpendCap.record() is only reached after client.complete() returns"
        )
        _assert_full_telemetry(manifest)
        assert _failure_pages(store, transport) == 1


# --------------------------------------------------------------------------
# Deliverable 5: degraded route — a fallback-served call is marked degraded
# on the manifest's `llm_calls` row, distinguishable from a primary-served
# one. Depends on alpha-engine-config-I9969 (fixed, crucible-PR100) and
# alpha-engine-config-I10006 / crucible-PR108 (`fallback_used` and
# `served_deployment` stamped from `krepis.llm.LLMResult`).
# --------------------------------------------------------------------------

_DEGRADED_SITE = CallSite(
    callsite_id="faults.degraded_route_probe",
    purpose="prove a fallback-served call is distinguishable in the manifest",
    capability_class="high",
    max_usd_per_call=1.0,
    owner="tests.faults",
)


class _Usage:
    input_tokens = 100
    output_tokens = 20
    cache_read_tokens = 0
    cache_create_tokens = 0
    provider_cost_usd = 0.02


class _DegradedResult:
    """Stands in for `krepis.llm.LLMResult` — same shape
    `test_llm_router_route.py::_Result` stands in for."""

    def __init__(self, *, fallback_used: bool, served_deployment: str | None) -> None:
        self.model = "some-vendor-model-the-router-picked"
        self.usage = _Usage()
        self.fallback_used = fallback_used
        self.served_deployment = served_deployment


class _DegradedClient:
    def __init__(self, *, fallback_used: bool, served_deployment: str | None) -> None:
        self._result = _DegradedResult(
            fallback_used=fallback_used, served_deployment=served_deployment
        )

    def complete(self, **_kwargs: object) -> _DegradedResult:
        return self._result


class TestFaultDegradedRouteIsNotIndistinguishable:
    def _run(self, tmp_path, monkeypatch, *, fallback_used, served_deployment, job, subdir):
        store = LocalStore(tmp_path / subdir)
        _stub_router(monkeypatch)
        monkeypatch.setattr(
            "krepis.llm.LLMClient",
            lambda *a, **k: _DegradedClient(
                fallback_used=fallback_used, served_deployment=served_deployment
            ),
        )

        def body(ctx: RunContext) -> None:
            llm_call(
                ctx,
                callsite_id=_DEGRADED_SITE.callsite_id,
                capability_class=_DEGRADED_SITE.capability_class,
                messages=[{"role": "user", "content": "probe"}],
                cap=SpendCap(cap_usd=5.0),
                estimate_usd=0.01,
                client_factory=lambda *a, **k: _DegradedClient(
                    fallback_used=fallback_used, served_deployment=served_deployment
                ),
                registry={_DEGRADED_SITE.callsite_id: _DEGRADED_SITE},
            )

        run_job(job, body, store=store, trading_day=FRIDAY, now=NOW, transient_retry=False)
        return json.loads(store.get_bytes(manifest_key(job, FRIDAY.isoformat())))

    def test_a_fallback_served_call_is_marked_degraded_and_differs_from_primary(
        self, tmp_path, monkeypatch
    ) -> None:
        primary_manifest = self._run(
            tmp_path,
            monkeypatch,
            fallback_used=False,
            served_deployment="high-1",
            job="experiment.run",
            subdir="primary",
        )
        degraded_manifest = self._run(
            tmp_path,
            monkeypatch,
            fallback_used=True,
            served_deployment="high-2",
            job="experiment.grade",
            subdir="degraded",
        )

        assert primary_manifest["status"] == "ok"
        assert degraded_manifest["status"] == "ok"
        primary_row = primary_manifest["llm_calls"][0]
        degraded_row = degraded_manifest["llm_calls"][0]

        assert primary_row["fallback_used"] is False
        assert degraded_row["fallback_used"] is True
        assert primary_row["served_deployment"] != degraded_row["served_deployment"]
        assert primary_row != degraded_row, (
            "a fallback-served call must not be indistinguishable from a "
            "primary-served one in the durable record"
        )
        _assert_full_telemetry(primary_manifest)
        _assert_full_telemetry(degraded_manifest)
