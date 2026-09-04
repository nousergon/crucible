"""The per-weekly-run LLM spend cap: declared, enforced, and never overspent.

Plan §2 row 3. The clause is not "a cap exists" — it is that *a run which
would exceed it FAILS rather than overspending*, so the tests that matter here
assert what did NOT happen: no provider was constructed, no cost was recorded,
and the manifest says `failed` with the cap in its reason.

Three ceilings are exercised separately because they failed separately. The
per-call ceiling used to LOWER the reservation rather than refuse the call —
`reserve(min(estimate, site.max_usd_per_call))` admitted a $100 call against
$0.05 of headroom — and the actual cost was booked with no re-check at all, so
a run finished `ok` having spent $9.85 against a declared $5.00 cap. The test
that was supposed to cover the first computed the same `min` in its own body
(`cap.reserve(min(1.0, cheap.max_usd_per_call), ...)`), which asserts the
implementation against itself and passes for any implementation of it. It is
replaced below by tests that go through `call()` — the only door — and that
fail against the shape they describe.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from crucible.config import DEFAULT_LLM_CAP_USD, settings
from crucible.llm import (
    PACING_PERIOD,
    CallSite,
    LlmCallCeilingExceeded,
    LlmSpendCapExceeded,
    LlmSpendOverrun,
    SpendCap,
    call,
    cap_metric,
    spend_pace,
    week_to_date_llm_spend,
)
from crucible.manifest import manifest_key
from crucible.runner import run_job
from crucible.store import LocalStore

DAY = dt.date(2026, 8, 28)
NOW = dt.datetime(2026, 8, 29, 12, 0, tzinfo=dt.UTC)

SITE = CallSite(
    callsite_id="test.cap_probe",
    purpose="exercise the cap without a provider",
    capability_class="high",
    max_usd_per_call=10.0,
    owner="tests.test_llm_cap",
)


class TestDeclaration:
    def test_the_cap_is_declared_in_config_with_its_provenance(self) -> None:
        resolved = settings()
        assert resolved.llm_cap_usd == DEFAULT_LLM_CAP_USD
        assert resolved.origins["llm_cap_usd"] == "default"
        assert resolved.to_dict()["llm_cap_usd"] == DEFAULT_LLM_CAP_USD

    def test_an_operator_can_move_it_and_the_origin_says_so(self, monkeypatch) -> None:
        monkeypatch.setenv("CRUCIBLE_LLM_CAP_USD", "1.25")
        resolved = settings()
        assert resolved.llm_cap_usd == 1.25
        assert resolved.origins["llm_cap_usd"] == "environ:CRUCIBLE_LLM_CAP_USD"

    def test_the_untouched_default_is_machine_checkably_unmeasured(self) -> None:
        """alpha-engine-config-I9778: the docstring beside `DEFAULT_LLM_CAP_USD`
        already said this in prose ("A declared ceiling, not a measurement");
        this is the field a manifest or a report can be compared against
        instead of trusting the comment."""
        from crucible.llm import DEFAULT_LLM_CAP_USD_MEASURED

        assert DEFAULT_LLM_CAP_USD_MEASURED is False, (
            "no phase-5 LLM arm has run a full cycle yet — flipping this without one "
            "would be exactly the unmeasured-number-presented-as-measured defect this "
            "flag exists to make impossible"
        )
        resolved = settings()
        assert resolved.llm_cap_usd_measured is False
        assert resolved.to_dict()["llm_cap_usd_measured"] is False

    def test_an_operator_override_does_not_read_as_measured(self, monkeypatch) -> None:
        """`alpha-engine-config-I9823` review: an operator ASSERTING a number
        is not the same fact as the number having been MEASURED from a
        phase-5 cost-sink cycle, and conflating them let an operator flip
        `llm_cap_usd_measured` at will — including by re-declaring the
        identical $5.00 default through the env var. `llm_cap_usd_measured`
        now tracks `DEFAULT_LLM_CAP_USD_MEASURED` only; the override's own
        provenance is still recorded, verbatim, in `origins["llm_cap_usd"]`."""
        monkeypatch.setenv("CRUCIBLE_LLM_CAP_USD", "1.25")
        resolved = settings()
        assert resolved.llm_cap_usd_measured is False
        assert resolved.origins["llm_cap_usd"] == "environ:CRUCIBLE_LLM_CAP_USD"

    def test_declaring_the_identical_default_through_the_env_var_stays_unmeasured(
        self, monkeypatch
    ) -> None:
        """The exact scenario the review demonstrated: re-asserting the same
        $5.00 number via `CRUCIBLE_LLM_CAP_USD` must not make the flag say
        `True` — an identical number is still not a measurement."""
        from crucible.llm import DEFAULT_LLM_CAP_USD

        monkeypatch.setenv("CRUCIBLE_LLM_CAP_USD", f"{DEFAULT_LLM_CAP_USD:.2f}")
        resolved = settings()
        assert resolved.llm_cap_usd == DEFAULT_LLM_CAP_USD
        assert resolved.llm_cap_usd_measured is False

    @pytest.mark.parametrize("bad", ["", "0", "-3", "five dollars"])
    def test_a_malformed_cap_raises_rather_than_falling_back(self, monkeypatch, bad) -> None:
        """A ceiling silently replaced by the default is a ceiling nobody is
        running under, and the environment variable is where the typo lands."""
        monkeypatch.setenv("CRUCIBLE_LLM_CAP_USD", bad)
        if bad == "":
            assert settings().llm_cap_usd == DEFAULT_LLM_CAP_USD
            return
        with pytest.raises(ValueError, match="cap"):
            settings()


class TestEnforcement:
    def test_a_call_that_would_cross_the_cap_is_refused_by_name(self) -> None:
        cap = SpendCap(cap_usd=1.0, spent_usd=0.9)
        with pytest.raises(LlmSpendCapExceeded) as excinfo:
            cap.reserve(0.5, callsite_id="test.cap_probe")
        message = str(excinfo.value)
        assert "test.cap_probe" in message
        assert "$1.00" in message and "0.9000" in message
        assert cap.spent_usd == 0.9, "a refused call books nothing"

    def test_a_call_within_the_cap_is_admitted(self) -> None:
        cap = SpendCap(cap_usd=1.0, spent_usd=0.4)
        reserved = cap.reserve(0.5, callsite_id="test.cap_probe")
        assert reserved == 0.5
        cap.record(0.45, reserved_usd=reserved, callsite_id="test.cap_probe")
        assert cap.headroom_usd() == pytest.approx(0.15)

    def test_record_refuses_to_book_an_overrun_in_silence(self) -> None:
        """The cost IS booked — the money is gone — and then it raises."""
        cap = SpendCap(cap_usd=5.0, spent_usd=0.0)
        reserved = cap.reserve(0.01, callsite_id="test.cap_probe")
        with pytest.raises(LlmSpendOverrun, match="0.0100"):
            cap.record(4.90, reserved_usd=reserved, callsite_id="test.cap_probe")
        assert cap.spent_usd == pytest.approx(4.90), (
            "the ledger tells the truth about what was spent even when the run fails"
        )

    def test_record_raises_when_the_booked_total_crosses_the_cap(self) -> None:
        cap = SpendCap(cap_usd=1.0, spent_usd=0.95)
        with pytest.raises(LlmSpendOverrun, match=r"\$1\.00"):
            cap.record(0.10, reserved_usd=0.50, callsite_id="test.cap_probe")
        assert cap.spent_usd == pytest.approx(1.05)

    def test_the_run_fails_and_the_provider_is_never_reached(self, tmp_path) -> None:
        """The whole clause in one assertion set: `failed`, the cap named in
        the reason, zero cost, no llm_calls row, and a client factory that
        would have raised if the refusal had come one line too late."""
        store = LocalStore(tmp_path)
        reached: list[str] = []

        def never(*args, **kwargs):
            reached.append("provider")
            raise AssertionError("the provider was constructed after the cap refused the call")

        def body(ctx):
            cap = SpendCap(cap_usd=0.10, spent_usd=0.09)
            call(
                ctx,
                callsite_id="test.cap_probe",
                capability_class="high",
                messages=[{"role": "user", "content": "hello"}],
                cap=cap,
                estimate_usd=0.50,
                client_factory=never,
                registry={SITE.callsite_id: SITE},
            )

        with pytest.raises(LlmSpendCapExceeded):
            run_job("report", body, store=store, trading_day=DAY, now=NOW, transient_retry=False)

        manifest = json.loads(store.get_bytes(manifest_key("report", DAY.isoformat())).decode())
        assert manifest["status"] == "failed"
        assert "cap" in manifest["reason"] and "0.10" in manifest["reason"]
        assert manifest["cost_usd"] == 0.0
        assert manifest["llm_calls"] == []
        assert reached == []

    def test_a_call_site_absent_from_the_registry_cannot_spend(self, tmp_path) -> None:
        store = LocalStore(tmp_path)

        def body(ctx):
            call(
                ctx,
                callsite_id="not.registered",
                capability_class="high",
                messages=[],
                cap=SpendCap(cap_usd=100.0),
                estimate_usd=0.0,
            )

        with pytest.raises(KeyError, match="not.registered"):
            run_job("report", body, store=store, trading_day=DAY, now=NOW, transient_retry=False)

    @pytest.mark.parametrize(
        "asked",
        [
            "claude-opus-4",
            "gpt-5",
            # The five the nine-entry substring DENYLIST let straight through.
            # Each of these reached a provider under the old shape; under an
            # allowlist of the router's declared groups they are refused by
            # membership, and so is the next vendor nobody has heard of yet.
            "grok-4",
            "qwen3-max",
            "command-r-plus",
            "nova-pro",
            "kimi-k2",
            # Not model ids at all — a base url and a provider name, neither
            # of which a list of model-name fragments can see.
            "https://api.example.com/v1",
            "openrouter",
        ],
    )
    def test_only_a_declared_capability_class_is_askable(self, tmp_path, asked) -> None:
        """Principle 8, by ALLOWLIST: a call site names a class the router
        declares; everything else is refused."""
        store = LocalStore(tmp_path)

        def body(ctx):
            call(
                ctx,
                callsite_id="test.cap_probe",
                capability_class=asked,
                messages=[],
                cap=SpendCap(cap_usd=100.0),
                estimate_usd=0.0,
                registry={SITE.callsite_id: SITE},
            )

        with pytest.raises(ValueError, match="not a router capability class"):
            run_job("report", body, store=store, trading_day=DAY, now=NOW, transient_retry=False)

    def test_the_allowlist_reads_the_router_rather_than_restating_it(self) -> None:
        """A second copy of the router's groups is the copy that drifts, so
        the router's tier groups are READ; the registry file declares only the
        classes this deployment serves beyond them."""
        from krepis.router import TIER_GROUPS

        from crucible.llm import _capability_classes, load_capability_classes

        allowed = _capability_classes()
        assert frozenset(TIER_GROUPS) <= allowed
        assert frozenset(TIER_GROUPS.values()) <= allowed
        assert frozenset(load_capability_classes()) <= allowed
        assert allowed == (
            frozenset(TIER_GROUPS)
            | frozenset(TIER_GROUPS.values())
            | frozenset(load_capability_classes())
        ), "nothing is askable that neither the router nor the registry declares"

    def test_a_registry_row_naming_an_undeclared_class_is_refused_at_load(self, tmp_path) -> None:
        """The allowlist binds the REGISTRY too, not only the call.

        A row could otherwise declare `capability_class: gpt-5`, and the
        refusal would only fire at the call — after the row had passed review
        as a registered, attributable call site.
        """
        import crucible.llm as llm

        registry = tmp_path / "llm_callsites.yaml"
        registry.write_text(
            "schema_version: llm_callsite_registry.v1\n"
            "capability_classes: []\n"
            "callsites:\n"
            "  bad.site:\n"
            "    purpose: p\n"
            "    capability_class: gpt-5\n"
            "    max_usd_per_call: 0.01\n"
            "    owner: t\n",
            encoding="utf-8",
        )
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(llm, "CALLSITE_REGISTRY_PATH", registry)
            llm.load_registry.cache_clear()
            llm.load_capability_classes.cache_clear()
            llm._capability_classes.cache_clear()
            try:
                with pytest.raises(ValueError, match="not a router capability class"):
                    llm.load_registry()
            finally:
                llm.load_registry.cache_clear()
                llm.load_capability_classes.cache_clear()
                llm._capability_classes.cache_clear()

    def test_an_estimate_above_the_site_ceiling_is_REFUSED_not_shrunk(self, tmp_path) -> None:
        """The E1 reproduction, as a test.

        Measured against the shape this replaces: a $100 estimate at a site
        with a $0.01 ceiling and $0.05 of headroom was ADMITTED, because
        `reserve(min(estimate, ceiling))` shrank the reservation to something
        that fit; the provider was constructed, the call billed $4.90, and the
        run ended `ok` with the week at $9.85 against a $5.00 cap. Every
        assertion below is false under that shape.
        """
        store = LocalStore(tmp_path)
        cheap = CallSite(
            callsite_id="test.cap_probe",
            purpose="a site whose own ceiling is tighter than the ask",
            capability_class="high",
            max_usd_per_call=0.01,
            owner="tests.test_llm_cap",
        )
        cap = SpendCap(cap_usd=5.00, spent_usd=4.95)
        reached: list[str] = []

        def never(*args, **kwargs):
            reached.append("provider")
            raise AssertionError("the provider was constructed after the ceiling refused")

        def body(ctx):
            call(
                ctx,
                callsite_id="test.cap_probe",
                capability_class="high",
                messages=[{"role": "user", "content": "hello"}],
                cap=cap,
                estimate_usd=100.0,
                client_factory=never,
                registry={cheap.callsite_id: cheap},
            )

        with pytest.raises(LlmCallCeilingExceeded) as excinfo:
            run_job("report", body, store=store, trading_day=DAY, now=NOW, transient_retry=False)
        assert "0.0100" in str(excinfo.value) and "REFUSED" in str(excinfo.value)
        assert reached == [], "a refused call never reaches a provider"
        assert cap.spent_usd == pytest.approx(4.95), "a refused call books nothing"
        manifest = json.loads(store.get_bytes(manifest_key("report", DAY.isoformat())).decode())
        assert manifest["status"] == "failed"
        assert manifest["cost_usd"] == 0.0
        assert manifest["llm_calls"] == []

    def test_admission_is_on_the_site_ceiling_not_on_the_estimate(self) -> None:
        """A cheap estimate does not buy a call whose worst case does not fit.

        The ceiling is the largest bill the call may produce, so the cap is
        checked against it. Admitting on an optimistic estimate is a cap
        crossed every time an estimate is low.
        """
        site = CallSite("x", "p", "high", max_usd_per_call=0.40, owner="t")
        cap = SpendCap(cap_usd=1.0, spent_usd=0.80)
        with pytest.raises(LlmSpendCapExceeded):
            cap.reserve(site.max_usd_per_call, callsite_id="x")
        assert cap.reserve(0.05, callsite_id="x") == 0.05, (
            "a site whose ceiling fits is still admitted"
        )

    def test_a_bill_above_the_admitted_ceiling_fails_the_run_and_is_recorded(
        self, tmp_path, monkeypatch
    ) -> None:
        """The second half of E1: the actual cost was never re-checked.

        A provider that bills past the site ceiling used to be booked into
        `spent_usd` and returned; the run finished `ok`. Now the call lands in
        the manifest — the money is spent and the artifact says so — and the
        run fails.
        """
        store = LocalStore(tmp_path)
        site = CallSite("test.cap_probe", "p", "high", max_usd_per_call=0.10, owner="t")
        cap = SpendCap(cap_usd=5.00)

        class _Usage:
            input_tokens = 1000
            output_tokens = 1000
            cache_read_tokens = 0
            cache_create_tokens = 0
            provider_cost_usd = 4.90

        class _Result:
            model = "router:high:primary"
            usage = _Usage()
            # The call-time facts `krepis.llm.LLMResult` declares
            # (`alpha-engine-config-I10006`). Carried on the stub with the
            # real dataclass's own defaults — the adapter reads them as
            # attributes rather than with a `getattr` default, so a stub
            # missing them raises here instead of recording a plausible
            # `false` on every call.
            fallback_used = False
            served_deployment = "router:high:primary"

        class _Client:
            def complete(self, **_kw):
                return _Result()

        def factory(*args, **kwargs):
            return _Client()

        # The router edge is stubbed at the adapter's own two imports rather
        # than at a network boundary: this test is about what the cap does
        # with a bill, and reaching a real router would make it a test of
        # registry permissions instead.
        monkeypatch.setenv("KREPIS_EXEC_CONTEXT", "ci")
        monkeypatch.setattr("krepis.router.resolve_group_spec", lambda *a, **k: (object(), {}))
        monkeypatch.setattr("krepis.router.route_is_degraded", lambda _route: False)
        monkeypatch.setattr("krepis.llm.LLMClient", lambda *a, **k: factory())

        def body(ctx):
            call(
                ctx,
                callsite_id="test.cap_probe",
                capability_class="high",
                messages=[{"role": "user", "content": "hello"}],
                cap=cap,
                estimate_usd=0.05,
                client_factory=factory,
                registry={site.callsite_id: site},
            )

        with pytest.raises(LlmSpendOverrun, match="4.9000"):
            run_job("report", body, store=store, trading_day=DAY, now=NOW, transient_retry=False)
        assert cap.spent_usd == pytest.approx(4.90)
        manifest = json.loads(store.get_bytes(manifest_key("report", DAY.isoformat())).decode())
        assert manifest["status"] == "failed"
        assert manifest["cost_usd"] == pytest.approx(4.90), (
            "the overrun is in the durable record, not only in the provider's invoice"
        )
        assert [c["usd"] for c in manifest["llm_calls"]] == [pytest.approx(4.90)]


class TestWindow:
    def test_spend_is_summed_from_the_weeks_manifests_not_this_process(self, tmp_path) -> None:
        store = LocalStore(tmp_path)

        def body(ctx):
            ctx.record_llm_call(
                {
                    "callsite_id": "test.cap_probe",
                    "model_requested": "high",
                    "model_served": "router:high:primary",
                    "route_degraded": False,
                    "fallback_used": False,
                    "served_deployment": "high-1",
                    "tokens_in": 10,
                    "tokens_out": 5,
                    "cache_read": 0,
                    "cache_write": 0,
                    "usd": 0.75,
                }
            )

        run_job("report", body, store=store, trading_day=DAY, now=NOW)
        run_job("drift", body, store=store, trading_day=dt.date(2026, 8, 27), now=NOW)
        run_job("drift", body, store=store, trading_day=dt.date(2026, 6, 1), now=NOW)

        total = week_to_date_llm_spend(store, window_start=dt.date(2026, 8, 24), trading_day=DAY)
        assert total == pytest.approx(1.5), "the June run is outside the window"

    def test_pacing_uses_the_weekly_window_and_catches_a_front_loaded_burst(self) -> None:
        anchor = dt.datetime(2026, 8, 24, tzinfo=dt.UTC)
        assert PACING_PERIOD == dt.timedelta(days=7)
        tuesday = anchor + dt.timedelta(days=2)
        ahead = spend_pace(4.0, cap_usd=5.0, now=tuesday, anchor=anchor)
        assert ahead.exceeded and ahead.overrun > 0
        under = spend_pace(0.5, cap_usd=5.0, now=tuesday, anchor=anchor)
        assert not under.exceeded

    def test_a_zero_cap_is_refused_rather_than_dividing(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            spend_pace(0.0, cap_usd=0.0, now=NOW, anchor=NOW)

    def test_the_cap_metric_breaches_loudly(self) -> None:
        over = cap_metric(SpendCap(cap_usd=1.0, spent_usd=1.4), now=NOW, source_path="runs/")
        assert over["status"] == "BREACH"
        assert over["unit"] == "usd" and over["baseline"] == 1.0
        ok = cap_metric(SpendCap(cap_usd=1.0), now=NOW, source_path="runs/")
        assert ok["status"] == "OK" and ok["value"] == 0.0

    def test_the_cap_metric_publishes_whether_the_cap_was_measured(self) -> None:
        """`alpha-engine-config-I9823`: the flag reaches an artifact now —
        the run's own `llm_spend_usd` metric — rather than only
        `Settings.to_dict()`, which nothing in this package read."""
        declared = cap_metric(SpendCap(cap_usd=5.0), now=NOW, source_path="runs/")
        assert declared["cap_usd_measured"] is False
        assert "declared, not measured" in declared["status_reason"]

        measured = cap_metric(
            SpendCap(cap_usd=5.0, cap_usd_measured=True), now=NOW, source_path="runs/"
        )
        assert measured["cap_usd_measured"] is True
        assert "declared, not measured" not in measured["status_reason"]
