"""The per-weekly-run LLM spend cap: declared, enforced, and never overspent.

Plan §2 row 3. The clause is not "a cap exists" — it is that *a run which
would exceed it FAILS rather than overspending*, so the tests that matter here
assert what did NOT happen: no provider was constructed, no cost was recorded,
and the manifest says `failed` with the cap in its reason.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from crucible.config import DEFAULT_LLM_CAP_USD, settings
from crucible.llm import (
    PACING_PERIOD,
    CallSite,
    LlmSpendCapExceeded,
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
    capability_class="reasoning_high",
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
        cap.reserve(0.5, callsite_id="test.cap_probe")
        cap.record(0.55)
        assert cap.headroom_usd() == pytest.approx(0.05)

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
                capability_class="reasoning_high",
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
                capability_class="reasoning_high",
                messages=[],
                cap=SpendCap(cap_usd=100.0),
                estimate_usd=0.0,
            )

        with pytest.raises(KeyError, match="not.registered"):
            run_job("report", body, store=store, trading_day=DAY, now=NOW, transient_retry=False)

    def test_a_provider_model_id_is_refused_at_the_call_site(self, tmp_path) -> None:
        """Principle 8: a call site names a capability class, never a model."""
        store = LocalStore(tmp_path)

        def body(ctx):
            call(
                ctx,
                callsite_id="test.cap_probe",
                capability_class="claude-opus-4",
                messages=[],
                cap=SpendCap(cap_usd=100.0),
                estimate_usd=0.0,
                registry={SITE.callsite_id: SITE},
            )

        with pytest.raises(ValueError, match="model id"):
            run_job("report", body, store=store, trading_day=DAY, now=NOW, transient_retry=False)

    def test_the_site_ceiling_binds_even_when_the_estimate_is_smaller(self) -> None:
        cap = SpendCap(cap_usd=0.5)
        cheap = CallSite("x", "p", "reasoning_high", max_usd_per_call=0.25, owner="t")
        cap.reserve(min(1.0, cheap.max_usd_per_call), callsite_id="x")
        with pytest.raises(LlmSpendCapExceeded):
            SpendCap(cap_usd=0.2).reserve(cheap.max_usd_per_call, callsite_id="x")


class TestWindow:
    def test_spend_is_summed_from_the_weeks_manifests_not_this_process(self, tmp_path) -> None:
        store = LocalStore(tmp_path)

        def body(ctx):
            ctx.record_llm_call(
                {
                    "callsite_id": "test.cap_probe",
                    "model_requested": "reasoning_high",
                    "model_served": "router:reasoning_high:primary",
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
