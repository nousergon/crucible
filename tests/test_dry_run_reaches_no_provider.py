"""A `--dry-run` reaches no provider and spends nothing.

`alpha-engine-config-I11012`, the EGRESS half.

`crucible.store.capturing` seals a rehearsal's WRITES by type. A capturing
store sitting above an LLM client that really calls out is a half-sealed
boundary, and the half that leaks is the expensive, externally visible one:

1. Brian's standing instruction is that no pipeline makes provider calls it
   was not commissioned to make. A rehearsal that reaches a model violates it
   on a path nobody would think to check.
2. A rehearsal with side effects outside the store is the same defect class
   as one that writes, one axis over.

So `crucible.llm.call` — the only door in the package, which
`tests/test_llm_callsite_audit.py` enforces statically — resolves a client
with no transport when the run's store is capturing, and that client RECORDS
the request and then REFUSES. It does not fabricate a completion: a
completion's TEXT drives everything downstream, and a synthetic one sends the
rehearsal to a verdict about words nobody generated.
"""

from __future__ import annotations

import datetime as dt

import pytest

from crucible.llm import (
    CallSite,
    DryRunProviderCallRefused,
    ProviderCaptureLedger,
    active_provider_ledger,
    begin_provider_capture,
    call,
    end_provider_capture,
    provider_capture_of,
    provider_methods,
)
from crucible.runner import run_job
from crucible.store import LocalStore, capturing

TRADING_DAY = dt.date(2026, 9, 11)

#: A call site that does not exist in production, injected through `call`'s
#: own `registry` parameter — the injection point its docstring declares for
#: exactly this. Nothing here names a vendor, a model id or a base URL: the
#: capability class is what a call site states, and the router maps it.
CALLSITE = "test.rehearsal_probe"
#: A router GROUP, which is what a call site is allowed to state. Not a model
#: id, not a provider, not a base URL — `_require_capability_class` refuses
#: all three, and this test would be asserting the wrong thing if it named one.
CAPABILITY_CLASS = "low"


@pytest.fixture
def registry() -> dict[str, CallSite]:
    from crucible.llm import load_registry

    known = load_registry()
    template = next(iter(known.values()))
    return {
        CALLSITE: CallSite(
            **{
                **{f: getattr(template, f) for f in template.__dataclass_fields__},
                "callsite_id": CALLSITE,
                "max_usd_per_call": 0.25,
            }
        )
    }


def _cap():
    from crucible.llm import SpendCap

    return SpendCap(cap_usd=100.0, spent_usd=0.0)


def _exploding_client_factory(*args, **kwargs):
    """A client factory that fails the test if it is ever reached.

    The assertion of record: `call` must not construct a real client at all
    under capture, so this must never run. A test that only checked "no
    network happened" would pass on a machine with no credentials for
    reasons that have nothing to do with the guard.
    """
    raise AssertionError(
        "a real provider client was constructed during a dry run — the capture is not sealed"
    )


class TestTheDoorRefusesUnderCapture:
    def test_a_captured_call_records_the_request_and_raises(self, registry, tmp_path) -> None:
        store = capturing(LocalStore(tmp_path / "store"))
        ledger = ProviderCaptureLedger()

        def body(ctx):
            ctx.provider_capture = ledger
            call(
                ctx,
                callsite_id=CALLSITE,
                capability_class=CAPABILITY_CLASS,
                messages=[
                    {"role": "system", "content": "you are a rehearsal"},
                    {"role": "user", "content": "twelve chars"},
                ],
                cap=_cap(),
                estimate_usd=0.01,
                registry=registry,
                client_factory=_exploding_client_factory,
            )

        with pytest.raises(DryRunProviderCallRefused) as excinfo:
            run_job(
                "fault.probe",
                body,
                store=store,
                trading_day=TRADING_DAY,
                run_mode="replay",
                transient_retry=False,
            )

        assert CALLSITE in str(excinfo.value)
        (record,) = ledger.calls
        assert record.callsite_id == CALLSITE
        assert record.capability_class == CAPABILITY_CLASS
        assert record.capability_group  # the ROUTER group, not a provider name
        assert record.method == "complete"
        assert record.messages == 2
        assert record.prompt_chars == len("you are a rehearsal") + len("twelve chars")
        assert record.admitted_usd == pytest.approx(0.25)
        assert record.estimate_usd == pytest.approx(0.01)

    def test_the_record_names_no_vendor_no_model_id_and_no_url(self, registry) -> None:
        """Principle 8, asserted rather than asserted-about. A rehearsal
        report that named a provider would be a provider name in a new place,
        which is the thing the capability-class addressing exists to avoid."""
        ledger = ProviderCaptureLedger()
        ctx = _bare_ctx(ledger)
        with pytest.raises(DryRunProviderCallRefused):
            call(
                ctx,
                callsite_id=CALLSITE,
                capability_class=CAPABILITY_CLASS,
                messages=[{"role": "user", "content": "hi"}],
                cap=_cap(),
                estimate_usd=0.01,
                registry=registry,
                client_factory=_exploding_client_factory,
            )
        rendered = ledger.render()
        for forbidden in ("anthropic", "openai", "claude-", "gpt-", "https://", "http://"):
            assert forbidden not in rendered.lower()

    def test_no_real_client_is_constructed_and_no_route_is_resolved(
        self, registry, monkeypatch
    ) -> None:
        """Sealed by TYPE: `resolve_group_spec` is not reached either.
        Resolving a route for a call that will not be made is itself an
        authenticated round trip, and a guard that stopped one hop later
        would still have left the process talking to the router."""
        import krepis.router as router_module

        def _forbidden(*args, **kwargs):
            raise AssertionError("resolve_group_spec was reached during a dry run")

        monkeypatch.setattr(router_module, "resolve_group_spec", _forbidden)

        ledger = ProviderCaptureLedger()
        ctx = _bare_ctx(ledger)
        with pytest.raises(DryRunProviderCallRefused):
            call(
                ctx,
                callsite_id=CALLSITE,
                capability_class=CAPABILITY_CLASS,
                messages=[{"role": "user", "content": "hi"}],
                cap=_cap(),
                estimate_usd=0.01,
                registry=registry,
                client_factory=_exploding_client_factory,
            )

    def test_nothing_is_recorded_on_the_manifest_as_an_llm_call(self, registry, tmp_path) -> None:
        """A captured call spent nothing, so `llm_calls` and `cost_usd` must
        stay empty. Recording a row for a call that never happened is the
        same fabrication as returning a synthetic completion, one field
        over — and it would feed the weekly spend roll-up."""
        ledger = ProviderCaptureLedger()
        ctx = _bare_ctx(ledger)
        with pytest.raises(DryRunProviderCallRefused):
            call(
                ctx,
                callsite_id=CALLSITE,
                capability_class=CAPABILITY_CLASS,
                messages=[{"role": "user", "content": "hi"}],
                cap=_cap(),
                estimate_usd=0.01,
                registry=registry,
                client_factory=_exploding_client_factory,
            )
        assert ctx.llm_calls == []
        assert ctx.cost_usd == 0.0


class TestTheRefusalIsNeverReclassified:
    def test_a_fault_injected_run_still_reports_the_refusal_as_a_refusal(
        self, registry, tmp_path
    ) -> None:
        """`call` wraps a provider failure into `FaultProbeFailure` when the
        run carries a fault-injection override. "We declined to call" is not
        "the router failed": folding it in would write a manifest claiming
        plan §10.7 fault 3 was observed on a run that never left the
        process."""
        from crucible.llm import FAULT_INJECTION_CAPABILITY_CLASSES

        ledger = ProviderCaptureLedger()
        ctx = _bare_ctx(ledger)
        ctx.fault_capability_class = next(iter(FAULT_INJECTION_CAPABILITY_CLASSES))

        with pytest.raises(DryRunProviderCallRefused):
            call(
                ctx,
                callsite_id=CALLSITE,
                capability_class=CAPABILITY_CLASS,
                messages=[{"role": "user", "content": "hi"}],
                cap=_cap(),
                estimate_usd=0.01,
                registry=registry,
                client_factory=_exploding_client_factory,
            )

    def test_a_backfill_session_loop_does_not_swallow_it(self) -> None:
        """Fail loud — no silent skip of the offending sessions. A refusal in
        `crucible.backfill.SESSION_REFUSALS` would be recorded per session
        and stepped over, and the rehearsal would report a completed range
        having called nothing and checked nothing."""
        from crucible.backfill import SESSION_REFUSALS

        assert DryRunProviderCallRefused not in SESSION_REFUSALS
        assert not any(
            issubclass(DryRunProviderCallRefused, refusal) for refusal in SESSION_REFUSALS
        )


class TestTheRefusalSetIsDerivedNotListed:
    def test_every_public_krepis_client_method_is_refused(self) -> None:
        """The `Store.MUTATORS` shape, one package over. `krepis.llm.LLMClient`
        exposes `complete`, `complete_grounded` and `structured` (measured
        2026-09-17); this package calls only `complete`, so a capturing client
        overriding that one alone would leak the moment a call site reached
        for either of the others — and would leak silently the day krepis
        adds a fourth.
        """
        from crucible.llm import _build_capturing_client_class

        derived = provider_methods()
        assert "complete" in derived, "the derivation has gone blind"
        assert len(derived) >= 3, derived

        cls = _build_capturing_client_class()
        for name in derived:
            assert name in cls.__dict__, (
                f"krepis.llm.LLMClient.{name} can reach a provider and the capturing "
                "client does not override it"
            )

    def test_the_capturing_client_is_not_a_krepis_client(self) -> None:
        """No base class, so no inherited transport: there is no code path
        from this object to a provider, which is what makes the guarantee
        structural rather than a flag nobody re-reads."""
        from krepis.llm import LLMClient

        from crucible.llm import _build_capturing_client_class

        cls = _build_capturing_client_class()
        assert not issubclass(cls, LLMClient)
        assert cls.__bases__ == (object,)

    def test_every_refusal_raises_and_records(self, registry) -> None:
        """Each derived method, not only `complete` — a guard nobody has made
        fire on every one of its arms is a guard that works on one."""
        from crucible.llm import _capturing_client

        for name in provider_methods():
            ledger = ProviderCaptureLedger()
            client = _capturing_client(
                ledger,
                callsite_id=CALLSITE,
                capability_class=CAPABILITY_CLASS,
                capability_group="group",
                messages=1,
                admitted_usd=0.25,
                estimate_usd=0.01,
            )
            with pytest.raises(DryRunProviderCallRefused):
                getattr(client, name)(system="s", user_content="u")
            assert [record.method for record in ledger.calls] == [name]

    def test_a_blind_derivation_is_refused(self, monkeypatch) -> None:
        """An empty refusal set is a blind guard, not a client that cannot
        call out — refused where it can first be seen."""
        from crucible import llm as llm_module

        monkeypatch.setattr(llm_module, "provider_methods", lambda: ())
        with pytest.raises(RuntimeError, match="blind guard"):
            llm_module._build_capturing_client_class()


class TestTheDoorIsTheOnlyDoor:
    def test_the_static_audit_is_what_makes_a_new_call_site_covered(self) -> None:
        """This capture sits at `crucible.llm.call`, so it covers a new call
        site only if a new call site must go through that door.

        It must, and that is enforced independently and statically by
        `crucible.llm.audit_call_sites`, which walks the tree for any module
        reaching a provider package directly. Asserted here as a LINKAGE
        rather than re-implemented: if this ever finds a finding, the egress
        capture has a hole that no amount of testing at the door would show.
        """
        from pathlib import Path

        from crucible.llm import audit_call_sites

        findings = audit_call_sites(Path(__file__).resolve().parents[1] / "crucible")
        assert not findings, [f.describe() for f in findings]


class TestTheInvocationLedger:
    def test_begin_is_re_entrant_and_end_clears(self) -> None:
        try:
            first = begin_provider_capture()
            assert begin_provider_capture() is first
            assert active_provider_ledger() is first
        finally:
            end_provider_capture()
        assert active_provider_ledger() is None

    def test_an_empty_ledger_says_no_provider_was_reached(self) -> None:
        assert "would reach no model" in ProviderCaptureLedger().render()

    def test_run_job_attaches_a_ledger_when_the_store_captures(self, tmp_path) -> None:
        seen = {}
        try:
            begin_provider_capture()
            run_job(
                "report",
                lambda ctx: seen.update(ledger=provider_capture_of(ctx)),
                store=capturing(LocalStore(tmp_path / "store")),
                trading_day=TRADING_DAY,
                run_mode="replay",
            )
        finally:
            end_provider_capture()
        assert isinstance(seen["ledger"], ProviderCaptureLedger)

    def test_run_job_attaches_none_on_a_real_run(self, tmp_path) -> None:
        """The other direction, which is the one that matters: a real run
        must reach a real provider, and a ledger left attached would silently
        turn production into a no-op that refuses every call."""
        seen = {}
        run_job(
            "report",
            lambda ctx: seen.update(ledger=provider_capture_of(ctx)),
            store=LocalStore(tmp_path / "store"),
            trading_day=TRADING_DAY,
            run_mode="replay",
        )
        assert seen["ledger"] is None


def _bare_ctx(ledger: ProviderCaptureLedger):
    """A `RunContext` carrying ``ledger``, without going through `run_job`.

    Used where the assertion is about the DOOR rather than about the runner;
    `run_job`'s own wiring is asserted separately above.
    """
    from crucible.runner import RunContext

    ctx = RunContext(
        run_id="0" * 26,
        job="fault.probe",
        trading_day=TRADING_DAY,
        calendar_date=TRADING_DAY,
        store=None,
        seed=1,
        started=dt.datetime(2026, 9, 11, 12, tzinfo=dt.UTC),
        run_mode="replay",
    )
    ctx.provider_capture = ledger
    return ctx
