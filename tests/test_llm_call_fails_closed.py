"""`crucible.llm.call` must FAIL when its preconditions are unmet — never
complete a degraded, unrecorded call.

Normative source: `alpha-engine-config-I9986` deliverable 3 (`I9972`
deliverable 4, `#9972`'s own measured facts). Two independent fail-closed
behaviors live inside the installed `krepis` (0.59.50) and this package
reaches both of them through `crucible.llm.call` — the one door — using the
REAL `krepis.llm.LLMClient`, not the stub every other test in this package
uses, because the whole point is to observe krepis's OWN precondition
checks firing through this adapter rather than a stub that would swallow
them by construction.

1. **DLP scanning.** `krepis.llm.LLMClient._dlp_scan_request` runs before
   every forwarded payload and fails closed
   (`krepis.session_dlp.dlp_enabled()` is `True` by default) when the
   gitleaks config chain does not resolve. `session_dlp.GITLEAKS_DIR` /
   `GITLEAKS_CONFIG` are resolved ONCE, at import time, from
   `KREPIS_GITLEAKS_DIR` and a short list of standard paths
   (`krepis/session_dlp.py:108`) — so a test cannot make this deterministic
   by setting an environment variable after import; it patches the two
   already-resolved module globals directly, which is also what makes this
   test pass identically on a laptop that DOES have a gitleaks config
   provisioned (this one does, under `.llm-routing/`) and on a CI runner
   that has none (`alpha-engine-config-I7913`, `I7660`).

2. **The cost sink.** `krepis.cost_sink.default_sink_from_env` raises
   `CostSinkConfigError` when exactly ONE of `KREPIS_COST_SINK_BUCKET` /
   `KREPIS_COST_SINK_PREFIX` is set — deliberately, per `I5206` — and
   returns `None` (no raise) when NEITHER is set, which is the "this
   consumer deliberately wants no cost telemetry" case krepis treats as
   legitimate. The gotcha named in this issue: a test that unsets ONE
   variable to simulate "unconfigured" is testing the RAISE branch, not the
   `None` branch; both are exercised below, separately.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from crucible.llm import CallSite, SpendCap, call
from crucible.manifest import manifest_key
from crucible.runner import run_job
from crucible.store import LocalStore

DAY = dt.date(2026, 8, 28)
NOW = dt.datetime(2026, 8, 29, 12, 0, tzinfo=dt.UTC)

SITE = CallSite(
    callsite_id="tests.fail_closed_probe",
    purpose="prove the door fails closed with no DLP config and no cost sink",
    capability_class="high",
    max_usd_per_call=1.0,
    owner="tests.test_llm_call_fails_closed",
)


def _never_reached(*_a, **_k):
    pytest.fail(
        "a transport client was constructed — the precondition check should have "
        "raised before any client method was reached"
    )


def _body():
    def body(ctx):
        call(
            ctx,
            callsite_id=SITE.callsite_id,
            capability_class="high",
            messages=[{"role": "user", "content": "probe"}],
            cap=SpendCap(cap_usd=5.0),
            estimate_usd=0.01,
            client_factory=_never_reached,
            registry={SITE.callsite_id: SITE},
        )

    return body


class TestNoGitleaksConfigAndNoCostSinkVariables:
    """Both preconditions absent at once — the exact shape `I9972` measured
    on a fresh crucible-v2 spot box: neither the DLP config nor the cost
    sink is provisioned there."""

    def test_the_call_raises_rather_than_completing(self, tmp_path, monkeypatch) -> None:
        import krepis.session_dlp as session_dlp
        from krepis.llm import LLMError
        from krepis.llm_config import ModelSpec

        # Neither cost-sink variable set: krepis reads this as "no telemetry
        # wanted" and returns a None sink WITHOUT raising
        # (`default_sink_from_env`) — legitimate on its own. The missing
        # gitleaks config is what must fail THIS call closed.
        monkeypatch.delenv("KREPIS_COST_SINK_BUCKET", raising=False)
        monkeypatch.delenv("KREPIS_COST_SINK_PREFIX", raising=False)
        monkeypatch.delenv("KREPIS_DLP_DISABLED", raising=False)
        monkeypatch.setenv("KREPIS_EXEC_CONTEXT", "ci")

        missing_dir = str(tmp_path / "no-such-gitleaks-config")
        monkeypatch.setattr(session_dlp, "GITLEAKS_DIR", missing_dir)
        monkeypatch.setattr(session_dlp, "GITLEAKS_CONFIG", f"{missing_dir}/gitleaks-egress.toml")

        spec = ModelSpec(provider="openai", model="test-dlp-probe")
        monkeypatch.setattr(
            "krepis.router.resolve_group_spec", lambda *a, **k: (spec, {"degraded": False})
        )

        store = LocalStore(tmp_path)
        with pytest.raises(LLMError, match="DLP scan"):
            run_job("report", _body(), store=store, trading_day=DAY, now=NOW, transient_retry=False)

        manifest = json.loads(store.get_bytes(manifest_key("report", DAY.isoformat())).decode())
        assert manifest["status"] == "failed"
        assert "DLP scan" in manifest["reason"]
        assert manifest["llm_calls"] == [], "a call that never completed billed nothing"
        assert manifest["cost_usd"] == 0.0


class TestHalfConfiguredCostSinkAlsoFailsClosed:
    """The gotcha this issue names explicitly: unsetting ONE cost-sink
    variable to simulate 'unconfigured' tests the RAISE
    (`CostSinkConfigError`), not the legitimate `None` path both variables
    absent produce. DLP is administratively disabled here so this test
    isolates the cost-sink precondition rather than re-proving the one
    above."""

    def test_exactly_one_cost_sink_variable_set_raises(self, tmp_path, monkeypatch) -> None:
        from krepis.cost_sink import CostSinkConfigError, reset_default_sink_for_tests
        from krepis.llm_config import ModelSpec

        monkeypatch.setenv("KREPIS_DLP_DISABLED", "1")
        monkeypatch.setenv("KREPIS_COST_SINK_BUCKET", "some-bucket")
        monkeypatch.delenv("KREPIS_COST_SINK_PREFIX", raising=False)
        monkeypatch.setenv("KREPIS_EXEC_CONTEXT", "ci")
        reset_default_sink_for_tests()

        spec = ModelSpec(provider="openai", model="test-cost-sink-probe")
        monkeypatch.setattr(
            "krepis.router.resolve_group_spec", lambda *a, **k: (spec, {"degraded": False})
        )

        store = LocalStore(tmp_path)
        try:
            with pytest.raises(CostSinkConfigError, match="KREPIS_COST_SINK_PREFIX"):
                run_job(
                    "report",
                    _body(),
                    store=store,
                    trading_day=DAY,
                    now=NOW,
                    transient_retry=False,
                )
        finally:
            reset_default_sink_for_tests()
