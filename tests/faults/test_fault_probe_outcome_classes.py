"""The three ways `crucible fault.probe`'s one router call can end, and the
only one that is evidence of plan §10.7 fault 3
(`alpha-engine-config-I10367` deliverable 3).

Fault 3 is "the LLM router returns 500". `crucible.runner.TRANSIENT_CLASSIFIERS`
keys the retry class the fault exists to exercise on `provider_5xx` /
`provider_timeout`, so a record filed `induced` against anything else is
evidence about something the fault does not name.

Three cases, measured rather than imagined. Every payload below is the
LITERAL text a real transport produced, captured 2026-09-10:

* the **routing/reachability refusal** — nothing upstream was contacted. Two
  live shapes: `crucible.llm` refusing the capability class before a client
  exists, and the router edge answering 401 before `litellm` sees the request
  (`nous-ergon-ops/alpha-engine-dashboard/live/infrastructure/nginx/conf.d/
  litellm-router.conf`, `location /`).
* the **upstream transport failure** — an upstream answered 5xx or timed out.
  The only honest `induced` evidence.
* the **upstream refusal** — an upstream answered and refused with a 4xx.
  This is what `chaos_probe` produced on 2026-09-10 through the live edge, and
  filing it as fault 3 would have been an attestation about the provider's
  request validator.

The fourth payload is the trap the classifier has to see through: LiteLLM
maps a failure to CONNECT to its upstream onto an HTTP **500** carrying
`OpenAIException - Connection error.`. That is a routing refusal wearing a
5xx, and a classifier keyed on the status code alone reads it as fault 3.
Measured against `litellm` 1.80.4 with the generated `chaos_probe` group
pointed at a dead port.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from crucible.fault_probe import (
    FAULT_PROBE_JOB,
    PROBE_OUTCOME_MARKER,
    PROBE_OUTCOME_ROUTING_REFUSAL,
    PROBE_OUTCOME_UPSTREAM_REFUSAL,
    PROBE_OUTCOME_UPSTREAM_TRANSPORT_FAILURE,
    PROBE_OUTCOMES,
    FaultProbeFailure,
    classify_probe_failure,
    probe_body,
    probe_outcome_from_reason,
)
from crucible.faults import FaultRecordRefusedError, record_fault
from crucible.manifest import manifest_key
from crucible.runner import SpotInterruptionError, classify_transient, run_job
from crucible.store import LocalStore

FRIDAY = dt.date(2026, 8, 28)
BUS_KEY = "alerts/2026-08-28/failure.fault.probe.json"


class _StatusError(RuntimeError):
    """What an `openai.APIStatusError` presents to `crucible`: a `status_code`
    attribute and the upstream body in `str(exc)`.

    Not the openai class itself — `crucible` does not depend on the SDK
    krepis's transport happens to use, and a classifier that needed the type
    would be untestable without it. The two attributes below are the whole
    interface, which is the point.
    """

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


# ---------------------------------------------------------------------------
# MEASURED payloads, 2026-09-10.
# ---------------------------------------------------------------------------

#: The live edge, `POST /v1/chat/completions {"model":"chaos_probe"}` — both
#: chain members walked, both refused by the upstream at request validation.
MEASURED_UPSTREAM_400 = _StatusError(
    400,
    "Error code: 400 - {'error': {'message': \"litellm.BadRequestError: "
    "OpenAIException - Model Not Exist: you passed "
    "chaos-probe-decommissioned-i10126. Received Model Group=chaos_probe\\n"
    "Available Model Group Fallbacks=['chaos-probe-zhipu']\\nError doing the "
    "fallback: Unknown Model\", 'code': '400'}}",
)

#: The same group with its members pointed at an always-503 listener.
MEASURED_UPSTREAM_503 = _StatusError(
    503,
    "Error code: 503 - {'error': {'message': \"litellm.ServiceUnavailableError: "
    "ServiceUnavailableError: OpenAIException - chaos probe: this endpoint "
    "always fails. Received Model Group=chaos_probe\\nAvailable Model Group "
    "Fallbacks=['chaos-probe-zhipu']\\nError doing the fallback: "
    "litellm.ServiceUnavailableError: ServiceUnavailableError: OpenAIException "
    "- chaos probe: this endpoint always fails\", 'code': '503'}}",
)

#: The same group with its members pointed at a dead port. A 500 that names no
#: upstream answer at all.
MEASURED_CONNECT_FAILURE_500 = _StatusError(
    500,
    "Error code: 500 - {'error': {'message': \"litellm.InternalServerError: "
    "InternalServerError: OpenAIException - Connection error.. Received Model "
    "Group=chaos_probe\\nAvailable Model Group Fallbacks=['chaos-probe-zhipu']"
    "\\nError doing the fallback: litellm.InternalServerError: "
    "InternalServerError: OpenAIException - Connection error.\", 'code': '500'}}",
)

#: The router edge before `litellm` is consulted at all.
EDGE_401 = _StatusError(401, "Error code: 401 - Unauthorized")


class TestTheThreeCases:
    """The classifier, in both directions, on every measured shape."""

    def test_an_upstream_5xx_is_the_only_transport_failure(self) -> None:
        assert classify_probe_failure(MEASURED_UPSTREAM_503) == (
            PROBE_OUTCOME_UPSTREAM_TRANSPORT_FAILURE
        )

    def test_the_measured_5xx_is_in_the_declared_transient_class(self) -> None:
        # The whole reason this outcome is the honest one: the retry class
        # fault 3 exists to exercise is keyed on exactly this reading.
        assert classify_transient(MEASURED_UPSTREAM_503) == "provider_5xx"

    def test_an_upstream_4xx_is_a_refusal_not_a_transport_failure(self) -> None:
        assert classify_probe_failure(MEASURED_UPSTREAM_400) == PROBE_OUTCOME_UPSTREAM_REFUSAL

    def test_a_connect_failure_dressed_as_a_500_is_a_routing_refusal(self) -> None:
        # The trap: `classify_transient` reads this as `provider_5xx` because
        # the STRING carries " 500". Nothing upstream answered.
        assert classify_transient(MEASURED_CONNECT_FAILURE_500) == "provider_5xx"
        assert classify_probe_failure(MEASURED_CONNECT_FAILURE_500) == (
            PROBE_OUTCOME_ROUTING_REFUSAL
        )

    def test_the_edge_401_is_a_routing_refusal(self) -> None:
        assert classify_probe_failure(EDGE_401) == PROBE_OUTCOME_ROUTING_REFUSAL

    def test_a_failure_with_no_status_at_all_is_a_routing_refusal(self) -> None:
        # `crucible.llm` refusing the capability class, `krepis.router`
        # refusing the group: no request was ever built.
        assert (
            classify_probe_failure(ValueError("capability class 'chaos_probe' is not routed"))
            == PROBE_OUTCOME_ROUTING_REFUSAL
        )

    def test_a_read_timeout_is_a_transport_failure(self) -> None:
        class ReadTimeout(RuntimeError):
            pass

        assert classify_probe_failure(ReadTimeout("read timeout")) == (
            PROBE_OUTCOME_UPSTREAM_TRANSPORT_FAILURE
        )

    def test_a_connect_timeout_is_a_routing_refusal(self) -> None:
        class ConnectTimeout(RuntimeError):
            pass

        # `classify_transient` puts both timeouts in one class; the probe may
        # not, because one of them never reached an upstream.
        assert classify_transient(ConnectTimeout("timed out")) == "provider_timeout"
        assert classify_probe_failure(ConnectTimeout("timed out")) == (
            PROBE_OUTCOME_ROUTING_REFUSAL
        )

    def test_every_outcome_is_declared(self) -> None:
        assert set(PROBE_OUTCOMES) == {
            PROBE_OUTCOME_ROUTING_REFUSAL,
            PROBE_OUTCOME_UPSTREAM_TRANSPORT_FAILURE,
            PROBE_OUTCOME_UPSTREAM_REFUSAL,
        }


class _RaisingClient:
    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def complete(self, **_kwargs: object) -> None:
        raise self._exc


def _stub_router(monkeypatch) -> None:
    monkeypatch.setenv("KREPIS_EXEC_CONTEXT", "ci")
    monkeypatch.setattr("krepis.router.resolve_group_spec", lambda *a, **k: (object(), {}))
    monkeypatch.setattr("krepis.router.route_is_degraded", lambda _route: False)


def _probe_manifest(store, monkeypatch, exc: BaseException) -> dict:
    """One real `fault.probe` run through the real runner against a transport
    that raises `exc`. Nothing is mocked but the transport."""
    _stub_router(monkeypatch)
    monkeypatch.setattr("krepis.llm.LLMClient", lambda *a, **k: _RaisingClient(exc))
    with pytest.raises(BaseException):  # noqa: B017 - the failure is the point
        run_job(
            FAULT_PROBE_JOB,
            probe_body,
            store=store,
            trading_day=FRIDAY,
            transient_retry=False,
            fault_capability_class="chaos_probe",
        )
    return json.loads(store.get_bytes(manifest_key(FAULT_PROBE_JOB, FRIDAY.isoformat())))


class TestTheManifestCarriesTheOutcome:
    def test_the_reason_names_the_outcome(self, tmp_path, monkeypatch) -> None:
        manifest = _probe_manifest(LocalStore(tmp_path), monkeypatch, MEASURED_UPSTREAM_503)
        assert manifest["status"] == "failed"
        assert PROBE_OUTCOME_MARKER + PROBE_OUTCOME_UPSTREAM_TRANSPORT_FAILURE in manifest["reason"]
        assert probe_outcome_from_reason(manifest["reason"]) == (
            PROBE_OUTCOME_UPSTREAM_TRANSPORT_FAILURE
        )

    def test_a_4xx_run_records_the_refusal_outcome(self, tmp_path, monkeypatch) -> None:
        manifest = _probe_manifest(LocalStore(tmp_path), monkeypatch, MEASURED_UPSTREAM_400)
        assert probe_outcome_from_reason(manifest["reason"]) == PROBE_OUTCOME_UPSTREAM_REFUSAL

    def test_the_underlying_failure_is_still_readable(self, tmp_path, monkeypatch) -> None:
        # Wrapping must not discard the cause: the manifest is the only
        # durable record of what the router actually said.
        manifest = _probe_manifest(LocalStore(tmp_path), monkeypatch, MEASURED_UPSTREAM_400)
        assert "chaos-probe-decommissioned-i10126" in manifest["reason"]

    def test_a_spot_reclamation_is_not_swallowed_into_an_outcome(
        self, tmp_path, monkeypatch
    ) -> None:
        # `SpotInterruptionError` derives from BaseException on purpose. A
        # probe that caught it would file fault 1 as fault 3 — the exact
        # cross-contamination `crucible.faults`' refusals exist to prevent,
        # arriving from the induction side instead.
        store = LocalStore(tmp_path)
        manifest = _probe_manifest(
            store,
            monkeypatch,
            SpotInterruptionError(
                "spot_interruption: received signal 15; the instance is being reclaimed"
            ),
        )
        assert PROBE_OUTCOME_MARKER not in manifest["reason"]
        assert probe_outcome_from_reason(manifest["reason"]) is None


class TestTheRecordRefusesEverythingButATransportFailure:
    """`alpha-engine-config-I10367` deliverable 3's table, both directions."""

    def _record(self, store, manifest) -> None:
        """File the record the way the CLI does — through `run_job`, so a
        refusal is a telemetered `status: failed` run of `fault.record`
        rather than a bare traceback."""

        def job(ctx) -> None:
            record_fault(
                ctx,
                store,
                fault_id="router_returns_500",
                outcome="induced",
                target_job=FAULT_PROBE_JOB,
                trading_day=FRIDAY.isoformat(),
                run_id=manifest["run_id"],
                bus_key=BUS_KEY,
            )

        run_job("fault.record", job, store=store, trading_day=FRIDAY, transient_retry=False)

    def test_a_transport_failure_is_accepted(self, tmp_path, monkeypatch) -> None:
        store = LocalStore(tmp_path)
        manifest = _probe_manifest(store, monkeypatch, MEASURED_UPSTREAM_503)
        store.put_bytes(BUS_KEY, b"{}")
        self._record(store, manifest)

    def test_an_upstream_4xx_is_refused(self, tmp_path, monkeypatch) -> None:
        store = LocalStore(tmp_path)
        manifest = _probe_manifest(store, monkeypatch, MEASURED_UPSTREAM_400)
        store.put_bytes(BUS_KEY, b"{}")
        with pytest.raises(FaultRecordRefusedError, match="upstream_refusal"):
            self._record(store, manifest)

    def test_a_routing_refusal_is_refused(self, tmp_path, monkeypatch) -> None:
        store = LocalStore(tmp_path)
        manifest = _probe_manifest(store, monkeypatch, MEASURED_CONNECT_FAILURE_500)
        store.put_bytes(BUS_KEY, b"{}")
        with pytest.raises(FaultRecordRefusedError, match="routing_refusal"):
            self._record(store, manifest)

    def test_a_probe_manifest_carrying_no_outcome_is_refused(self, tmp_path) -> None:
        # The `--fault-capability-class` omission, and every other way the job
        # can fail before it reaches the router: no outcome was classified, so
        # there is no evidence of fault 3 either way.
        store = LocalStore(tmp_path)

        def body(ctx) -> None:
            raise ValueError("no fault-injection capability class")

        with pytest.raises(ValueError):
            run_job(FAULT_PROBE_JOB, body, store=store, trading_day=FRIDAY, transient_retry=False)
        manifest = json.loads(store.get_bytes(manifest_key(FAULT_PROBE_JOB, FRIDAY.isoformat())))
        store.put_bytes(BUS_KEY, b"{}")
        with pytest.raises(FaultRecordRefusedError, match="no fault-probe outcome"):
            self._record(store, manifest)


class TestTheWrapperItself:
    def test_the_failure_carries_its_outcome_and_its_cause(self) -> None:
        failure = FaultProbeFailure(PROBE_OUTCOME_UPSTREAM_REFUSAL, MEASURED_UPSTREAM_400)
        assert failure.outcome == PROBE_OUTCOME_UPSTREAM_REFUSAL
        assert PROBE_OUTCOME_MARKER + PROBE_OUTCOME_UPSTREAM_REFUSAL in str(failure)

    def test_an_undeclared_outcome_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not a declared fault-probe outcome"):
            FaultProbeFailure("something_else", MEASURED_UPSTREAM_400)

    def test_a_reason_with_no_marker_reads_none(self) -> None:
        assert probe_outcome_from_reason("MissingSourceError: no data") is None

    def test_a_reason_with_an_undeclared_marker_reads_none(self) -> None:
        assert probe_outcome_from_reason(PROBE_OUTCOME_MARKER + "invented") is None
