"""The door declares every call site's trace capture (alpha-engine-config-I12312).

`crucible.llm.call` is the only way this package reaches a model, so it is
where an `llm_trace.v1` capture level is decided — once, for every call site,
by construction. Each `llm_callsites.yaml` row declares `trace_capture`
(`hash_only` or `off`, never `full`), and the door hands it to
`krepis.llm.LLMClient(trace_capture=)`.

What these tests hold, and why each one exists:

* **The declaration wins over the environment.** krepis traces at `full` the
  moment `KREPIS_TRACE_SINK_*` appears unless the call site says otherwise,
  and it obeys `KREPIS_TRACE_CAPTURE=full`. A v2 prompt carries vendor market
  data, so a `hash_only` row must reach the real client as `hash_only` even
  with `KREPIS_TRACE_CAPTURE=full` set.
* **No sink environment means `off`, not a raise.** No v2 box sets the trace
  sink today; passing `hash_only` with no sink would make krepis raise at
  construction, which on a fault-injection run reads as plan §10.7 fault 3.
* **A half-configured sink still raises, before the provider.** One variable
  without the other is a deploy defect, not an absence.
* **`full` is not admitted by the schema**, and the field has no default.
"""

from __future__ import annotations

import datetime as dt

import pytest
from pydantic import ValidationError

from crucible.llm import CallSite, SpendCap, call, load_registry, trace_capture_for
from crucible.models import LlmCallSiteRow
from crucible.runner import RunContext

BUCKET = "KREPIS_TRACE_SINK_BUCKET"
PREFIX = "KREPIS_TRACE_SINK_PREFIX"

SITE = CallSite(
    callsite_id="tests.trace_capture",
    purpose="exercise the door's trace_capture declaration",
    capability_class="high",
    max_usd_per_call=1.0,
    owner="tests",
    trace_capture="hash_only",
)

ROW = {
    "purpose": "p",
    "capability_class": "high",
    "max_usd_per_call": 0.01,
    "owner": "o",
    "trace_capture": "hash_only",
    "trace_capture_reason": "r",
}


@pytest.fixture(autouse=True)
def _no_ambient_sink(monkeypatch):
    for name in (BUCKET, PREFIX, "KREPIS_TRACE_CAPTURE"):
        monkeypatch.delenv(name, raising=False)
    from krepis.trace_sink import reset_default_trace_sink_for_tests

    reset_default_trace_sink_for_tests()
    yield
    reset_default_trace_sink_for_tests()


class TestTheRegistryDeclaresIt:
    def test_every_production_row_declares_a_level_that_is_not_full(self) -> None:
        registry = load_registry()
        assert registry, "the registry has at least the fault probe row"
        for callsite_id, site in registry.items():
            assert site.trace_capture in {"hash_only", "off"}, callsite_id

    def test_the_fault_probe_row_is_hash_only(self) -> None:
        assert load_registry()["faults.router_probe"].trace_capture == "hash_only"

    def test_full_is_refused_by_the_schema(self) -> None:
        with pytest.raises(ValidationError):
            LlmCallSiteRow.model_validate({**ROW, "trace_capture": "full"})

    def test_the_field_has_no_default(self) -> None:
        row = {k: v for k, v in ROW.items() if k != "trace_capture"}
        with pytest.raises(ValidationError):
            LlmCallSiteRow.model_validate(row)

    def test_a_reason_is_required(self) -> None:
        row = {k: v for k, v in ROW.items() if k != "trace_capture_reason"}
        with pytest.raises(ValidationError):
            LlmCallSiteRow.model_validate(row)


class TestTraceCaptureFor:
    def test_off_stays_off_whatever_the_environment(self) -> None:
        off = CallSite("x", "p", "high", 1.0, "o", trace_capture="off")
        assert trace_capture_for(off, {BUCKET: "b", PREFIX: "p"}) == "off"

    def test_no_sink_environment_is_off(self) -> None:
        assert trace_capture_for(SITE, {}) == "off"
        assert trace_capture_for(SITE, {BUCKET: " ", PREFIX: ""}) == "off"

    def test_a_configured_sink_gets_the_declaration(self) -> None:
        env = {BUCKET: "b", PREFIX: "llm_traces/v1", "KREPIS_TRACE_CAPTURE": "full"}
        assert trace_capture_for(SITE, env) == "hash_only"

    def test_a_half_configured_sink_gets_the_declaration_so_krepis_can_raise(self) -> None:
        assert trace_capture_for(SITE, {BUCKET: "b"}) == "hash_only"
        assert trace_capture_for(SITE, {PREFIX: "p"}) == "hash_only"

    def test_a_hand_built_registry_defaults_to_off(self) -> None:
        assert CallSite("x", "p", "high", 1.0, "o").trace_capture == "off"


class _Result:
    model = "served"
    fallback_used = False
    served_deployment = None

    class usage:  # noqa: N801 - stands in for the attribute
        input_tokens = 1
        output_tokens = 1
        cache_read_tokens = 0
        cache_create_tokens = 0
        provider_cost_usd = 0.0


def _through_the_door(monkeypatch, tmp_path) -> list:
    """Call the door with the REAL `krepis.llm.LLMClient` constructor.

    Only the router resolution and `complete` are stubbed, so the arguments
    the door passes are judged by the installed krepis, which raises on a
    bad level or a declared capture with no sink.
    """
    import krepis.llm as krepis_llm

    real = krepis_llm.LLMClient
    built: list = []

    def construct(*args, **kwargs):
        client = real(*args, **kwargs)
        client.complete = lambda **_kw: _Result()
        built.append((kwargs, client))
        return client

    monkeypatch.setenv("KREPIS_EXEC_CONTEXT", "ci")
    from krepis.llm_config import ModelSpec

    spec = ModelSpec(provider="openai", model="test-trace-capture")
    monkeypatch.setattr("krepis.router.resolve_group_spec", lambda *a, **k: (spec, {}))
    monkeypatch.setattr("krepis.router.route_is_degraded", lambda _route: False)
    monkeypatch.setattr("krepis.llm.LLMClient", construct)
    ctx = RunContext(
        run_id="0" * 26,
        job="experiment.run",
        trading_day=dt.date(2026, 10, 16),
        calendar_date=dt.date(2026, 10, 17),
        store=None,
        seed=1,
        started=dt.datetime(2026, 10, 17, 12, tzinfo=dt.UTC),
    )
    call(
        ctx,
        callsite_id=SITE.callsite_id,
        capability_class=SITE.capability_class,
        messages=[{"role": "user", "content": "probe"}],
        cap=SpendCap(cap_usd=5.0),
        estimate_usd=0.01,
        registry={SITE.callsite_id: SITE},
    )
    return built


class TestTheDoorPassesIt:
    def test_no_sink_builds_an_untraced_client(self, monkeypatch, tmp_path) -> None:
        built = _through_the_door(monkeypatch, tmp_path)
        (kwargs, client) = built[0]
        assert kwargs["trace_capture"] == "off"
        assert client._trace_sink is None

    def test_a_configured_sink_is_hash_only_even_under_capture_full(
        self, monkeypatch, tmp_path
    ) -> None:
        from krepis.trace_sink import InMemoryTraceSink

        sink = InMemoryTraceSink()
        monkeypatch.setenv(BUCKET, "test-bucket")
        monkeypatch.setenv(PREFIX, "llm_traces/v1")
        monkeypatch.setenv("KREPIS_TRACE_CAPTURE", "full")
        monkeypatch.setattr("krepis.trace_sink.default_trace_sink_from_env", lambda: sink)
        built = _through_the_door(monkeypatch, tmp_path)
        (kwargs, client) = built[0]
        assert kwargs["trace_capture"] == "hash_only"
        assert client._trace_capture == "hash_only"
        assert client._trace_sink is sink

    def test_a_half_configured_sink_raises_before_the_provider(self, monkeypatch, tmp_path) -> None:
        from krepis.trace_sink import TraceSinkConfigError

        monkeypatch.setenv(BUCKET, "test-bucket")
        with pytest.raises(TraceSinkConfigError):
            _through_the_door(monkeypatch, tmp_path)
