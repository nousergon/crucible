"""The one door reaches the ROUTER — not the pre-router flip surface.

Normative source: `alpha-engine-config-I9969`; `model-router-policy` §2
layer 5, R12, R20, R28/R29; principle 8.

**The defect these tests were written against.** `crucible.llm.call`
validated its `capability_class` against router groups and then discarded it,
resolving a spec through `krepis.llm_config.resolve_model_spec` out of
`/crucible/llm/<class>` in SSM. That is a second routing plane with a single
rung: the spec named one vendor endpoint directly, so there was no
cross-provider fallback chain, no per-consumer attribution at the
authenticated edge, and nothing the egress proxy could scan — verbatim the
three objections `krepis.llm_config.ModelSpec.__post_init__` writes down as
its reason for refusing `provider="litellm"` at construction. The parameter
did not exist and the runtime could not have read it if it had (the
crucible-v2 role grants `parameter/crucible-v2/*`, a different prefix), so
the first phase-5 call raised `LLMConfigError` before reaching a model at
all.

**What is asserted here, and why each half exists.**

The behavioural tests show the call REACHING `krepis.router.resolve_group_spec`
with the group, the declared execution context and the wire format it can
speak, and show the pre-router surface never being touched — patched to raise,
so "it was not called" is a failure the test can see rather than an absence it
assumes.

The source-level test is the one that survives the CLASS rather than the
instance: a behavioural stub can be satisfied by a call path that ALSO reads
SSM, and a future edit could reintroduce `resolve_model_spec` beside the
router call and keep every behavioural assertion green. `PROVIDER_MODULES`
already refuses `krepis.llm_config` from every other module in the package —
but this module is the exempt adapter, by path identity, so the refusal has
to be written here or it does not cover the one file that had the defect.

**Seen red against the pre-I9969 adapter**, measured rather than asserted:
with `origin/main`'s `crucible/llm.py` swapped back in, this module does not
import at all — `CAPABILITY_CLASS_GROUPS`, `CapabilityClassNotRouted`,
`capability_group` and `EXEC_CONTEXT_ENV` do not exist on that shape, and no
`route_degraded` reaches a manifest row because nothing writes one. Each
assertion is on a property the old code could not have satisfied, not on a
stub arranged to look satisfied.
"""

from __future__ import annotations

import ast
import datetime as dt
import json
import pathlib

import pytest

from crucible.llm import (
    EXEC_CONTEXT_ENV,
    CallSite,
    CapabilityClassNotRouted,
    SpendCap,
    call,
    capability_group,
    load_registry,
)
from crucible.manifest import manifest_key
from crucible.runner import run_job
from crucible.store import LocalStore

ADAPTER = pathlib.Path(__file__).resolve().parents[1] / "crucible" / "llm.py"

DAY = dt.date(2026, 8, 28)
NOW = dt.datetime(2026, 8, 29, 12, 0, tzinfo=dt.UTC)

SITE = CallSite(
    callsite_id="tests.router_route",
    purpose="prove the door resolves a group through the router",
    capability_class="high",
    max_usd_per_call=1.0,
    owner="tests.test_llm_router_route",
)


class _Usage:
    input_tokens = 100
    output_tokens = 20
    cache_read_tokens = 0
    cache_create_tokens = 0
    provider_cost_usd = 0.02


class _Result:
    """Stands in for `krepis.llm.LLMResult`.

    The call-time fields carry the SAME defaults the real dataclass declares
    (`fallback_used: bool = False`, `served_deployment: Optional[str] = None`)
    so a stub that silently diverged from the contract would be caught by
    `TestTheCallTimeFactsComeFromTheResult::
    test_the_result_contract_this_stub_stands_in_for_is_real`, which asserts
    the fields against the installed krepis rather than against this class.
    """

    model = "some-vendor-model-the-router-picked"
    usage = _Usage()
    fallback_used = False
    served_deployment: str | None = None

    def __init__(self, *, fallback_used: bool = False, served_deployment: str | None = None):
        self.fallback_used = fallback_used
        self.served_deployment = served_deployment


class _Client:
    def __init__(self, *, fallback_used: bool = False, served_deployment: str | None = None):
        self._result = _Result(fallback_used=fallback_used, served_deployment=served_deployment)

    def complete(self, **_kw):
        return self._result


def _asked(
    monkeypatch,
    *,
    degraded: bool = False,
    fallback_used: bool = False,
    served_deployment: str | None = None,
) -> dict:
    """Stub the router edge and return the dict it records the ask into.

    The stub stands in for the REGISTRY, not for the router's contract: it
    returns the same `(spec, route)` pair `resolve_group_spec` returns, so
    what is exercised is the shape of the ask crucible makes.
    """
    recorded: dict = {}

    def _resolve(group, **kwargs):
        recorded["group"] = group
        recorded.update(kwargs)
        return object(), {"registry_id": "stub", "degraded": degraded}

    monkeypatch.setenv(EXEC_CONTEXT_ENV, "ci")
    monkeypatch.setattr("krepis.router.resolve_group_spec", _resolve)
    monkeypatch.setattr("krepis.router.route_is_degraded", lambda route: route["degraded"])
    monkeypatch.setattr(
        "krepis.llm.LLMClient",
        lambda *a, **k: _Client(fallback_used=fallback_used, served_deployment=served_deployment),
    )
    return recorded


def _body(capability_class: str = "high"):
    def body(ctx) -> None:
        call(
            ctx,
            callsite_id=SITE.callsite_id,
            capability_class=capability_class,
            messages=[{"role": "user", "content": "probe"}],
            cap=SpendCap(cap_usd=5.0),
            estimate_usd=0.01,
            registry={SITE.callsite_id: SITE},
        )

    return body


def _manifest(store: LocalStore, job: str = "report") -> dict:
    return json.loads(store.get_bytes(manifest_key(job, DAY.isoformat())).decode())


class TestTheDoorReachesTheRouter:
    def test_it_asks_the_router_for_a_GROUP_from_a_declared_context(
        self, tmp_path, monkeypatch
    ) -> None:
        """The capability class survives as a routing DECISION, not as a
        path segment in an SSM parameter name."""
        recorded = _asked(monkeypatch)
        run_job("report", _body(), store=LocalStore(tmp_path), trading_day=DAY, now=NOW)

        assert recorded["group"] == "high", (
            "the class the call site declared must reach the router as the group it "
            "addresses; the pre-I9969 shape discarded it after validating it"
        )
        assert recorded["exec_context"] == "ci", (
            "R29: where the caller runs is DECLARED, and it is declared per call"
        )
        assert recorded["wire"] == "openai", (
            "this door builds an LLMClient on the openai transport; asking for the "
            "anthropic wire would let a fallback hand it a URL it cannot speak"
        )

    def test_the_pre_router_flip_surface_is_never_reached(self, tmp_path, monkeypatch) -> None:
        """`resolve_model_spec` is patched to RAISE, so its non-use is
        something this test observes rather than something it assumes."""

        def _refuse(**_kw):
            raise AssertionError(
                "crucible.llm reached krepis.llm_config.resolve_model_spec — the "
                "pre-router flip surface I9969 removed"
            )

        _asked(monkeypatch)
        monkeypatch.setattr("krepis.llm_config.resolve_model_spec", _refuse)
        run_job("report", _body(), store=LocalStore(tmp_path), trading_day=DAY, now=NOW)

    def test_the_ADAPTER_SOURCE_cannot_carry_the_resolve_model_spec_shape(self) -> None:
        """The assertion that survives the class rather than the instance.

        A behavioural stub is satisfied by a call path that also reads SSM.
        This one reads the adapter's own AST: `krepis.llm_config` is in
        `PROVIDER_MODULES`, but `crucible/llm.py` is the module that
        collection exempts by path identity, so without this the one file
        that had the defect is the one file nothing checks.
        """
        tree = ast.parse(ADAPTER.read_text(encoding="utf-8"))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
            elif isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
        # The tracker is cited in this module's docstring, not in the message:
        # a tracker literal in a failure string is the stale-pointer class
        # `tests/test_no_stale_tracker_literals.py` refuses package-wide.
        assert "krepis.llm_config" not in imported, (
            "crucible/llm.py imports the pre-router flip surface again"
        )
        assert "krepis.router" in imported, "the adapter must reach the router"

        names = {
            node.attr if isinstance(node, ast.Attribute) else node.id
            for node in ast.walk(tree)
            if isinstance(node, (ast.Attribute, ast.Name))
        }
        aliases = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        assert "resolve_model_spec" not in names | aliases
        assert "resolve_group_spec" in names | aliases

    def test_no_module_in_the_package_still_reads_the_retired_ssm_path(self) -> None:
        """Deliverable 4: the parameter path and the env-var convention are
        DELETED, not left beside the router call as a second door."""
        package = ADAPTER.parent
        offenders = [
            str(path.relative_to(package.parent))
            for path in package.rglob("*.py")
            if "/crucible/llm/" in path.read_text(encoding="utf-8")
        ]
        assert offenders == [], (
            f"{offenders} still name the retired `/crucible/llm/<class>` SSM path"
        )


class TestDegradationIsRecorded:
    """R12: serving from a fallback is an ALERT, not a log line — so it has
    to be in the artifact a verdict is reconstructed from."""

    @pytest.mark.parametrize("degraded", [True, False])
    def test_the_manifest_row_says_whether_resolution_fell_past_the_primary(
        self, tmp_path, monkeypatch, degraded: bool
    ) -> None:
        _asked(monkeypatch, degraded=degraded)
        store = LocalStore(tmp_path)
        run_job("report", _body(), store=store, trading_day=DAY, now=NOW)

        row = _manifest(store)["llm_calls"][0]
        assert row["route_degraded"] is degraded
        assert row["model_requested"] == "high"
        assert row["model_served"] == _Result.model, (
            "the two halves are recorded together: `route_degraded` is the "
            "resolve-time answer and `model_served` the call-time one, and on the "
            "router-edge route only the second can say which entry served"
        )


class TestTheCallTimeFactsComeFromTheResult:
    """`alpha-engine-config-I10006`: the manifest records what happened to
    THIS CALL, not what the route object declared about itself.

    `route_is_degraded` answers a resolve-time question about the route. On
    the router edge the fallback chain is walked by the proxy AFTER
    resolution, and on `litellm_proxy` — the route krepis prefers whenever
    the health probe answers, so the route v2 actually takes — it returned
    `False` unconditionally. A manifest carrying only that predicate does not
    leave the fact missing; it states it FALSE beside a call the primary
    never answered, which is a run grading a model nobody selected.

    The discriminating case is the FIRST test below: route not degraded,
    call served by a fallback. Stamping either field from
    `route_is_degraded` makes it red; nothing else in this module would.
    """

    def test_a_fallback_served_call_says_so_even_when_the_route_reads_healthy(
        self, tmp_path, monkeypatch
    ) -> None:
        _asked(
            monkeypatch,
            degraded=False,
            fallback_used=True,
            served_deployment="high-2",
        )
        store = LocalStore(tmp_path)
        run_job("report", _body(), store=store, trading_day=DAY, now=NOW)

        row = _manifest(store)["llm_calls"][0]
        assert row["fallback_used"] is True, (
            "the call was served by a fallback; a manifest saying otherwise grades "
            "a model nobody selected"
        )
        assert row["served_deployment"] == "high-2"
        assert row["route_degraded"] is False, (
            "the resolve-time answer is recorded UNCHANGED beside the call-time one — "
            "this row is exactly the disagreement the two fields exist to expose, and "
            "collapsing either into the other would erase it"
        )

    def test_a_primary_served_call_is_distinguishable_from_a_fallback_served_one(
        self, tmp_path, monkeypatch
    ) -> None:
        _asked(monkeypatch, degraded=False, fallback_used=False, served_deployment="high-1")
        store = LocalStore(tmp_path)
        run_job("report", _body(), store=store, trading_day=DAY, now=NOW)

        row = _manifest(store)["llm_calls"][0]
        assert row["fallback_used"] is False
        assert row["served_deployment"] == "high-1"

    def test_a_route_reporting_no_deployment_records_null_not_a_missing_field(
        self, tmp_path, monkeypatch
    ) -> None:
        """`None` is the router's own answer, and it is not an absence.

        The schema requires the key and admits `null`, so a reader can tell
        "the router reported no deployment" from "this producer does not
        record the field" — the distinction `Store.get_bytes` raising on a
        missing key makes everywhere else in this package.
        """
        _asked(monkeypatch, degraded=False, fallback_used=False, served_deployment=None)
        store = LocalStore(tmp_path)
        run_job("report", _body(), store=store, trading_day=DAY, now=NOW)

        row = _manifest(store)["llm_calls"][0]
        assert "served_deployment" in row
        assert row["served_deployment"] is None

    def test_a_result_without_the_call_time_fields_RAISES(self, tmp_path, monkeypatch) -> None:
        """Rule 5. A krepis that withdrew the field must not read `false`.

        `getattr(result, "fallback_used", False)` would record a plausible
        answer forever — the `dropped_params` failure mode (I7232) the field
        was itself added to end, one layer out. The attribute is read
        directly so its absence is loud.
        """

        class _Bare:
            model = "m"
            usage = _Usage()

        class _BareClient:
            def complete(self, **_kw):
                return _Bare()

        _asked(monkeypatch)
        monkeypatch.setattr("krepis.llm.LLMClient", lambda *a, **k: _BareClient())
        store = LocalStore(tmp_path)
        with pytest.raises(AttributeError, match="fallback_used"):
            run_job("report", _body(), store=store, trading_day=DAY, now=NOW)

    def test_the_result_contract_this_stub_stands_in_for_is_real(self) -> None:
        """The stub above is only evidence if the real class carries the fields.

        Asserted against the INSTALLED krepis, so the pin in `pyproject.toml`
        is what this test grades — a downgrade past `LLMResult.fallback_used`
        turns it red here rather than at the first phase-5 call.
        """
        from krepis.llm import LLMResult

        fields = LLMResult.__dataclass_fields__
        assert "fallback_used" in fields
        assert "served_deployment" in fields


class TestAnUnroutedCapabilityClass:
    """The group a class addresses is a MAPPING, and an unmade one refuses.

    `CAPABILITY_CLASS_GROUPS` carries no unrouted entry today
    (`alpha-engine-config-I9970`, 2026-09-08 — `reasoning_high` is retired
    and both `high` and `ultra` resolve by identity). These tests exercise
    the REFUSAL MECHANISM itself, not a specific production mapping, so they
    monkeypatch a synthetic unrouted class onto the module rather than
    reading a real one that no longer exists — the mechanism is what must
    keep working the next time some class needs a ruling, not any one
    instance of it.
    """

    UNROUTED = "synthetic_unrouted_class"

    def test_a_class_with_no_ruled_group_refuses_and_says_what_is_missing(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr("crucible.llm.CAPABILITY_CLASS_GROUPS", {self.UNROUTED: None})
        with pytest.raises(CapabilityClassNotRouted) as excinfo:
            capability_group(self.UNROUTED)
        message = str(excinfo.value)
        assert self.UNROUTED in message
        assert "RULING" in message
        assert "CAPABILITY_CLASS_GROUPS" in message, (
            "the refusal names where the mapping is written down"
        )

    def test_a_class_that_IS_a_group_name_passes_through_unchanged(self) -> None:
        assert capability_group("high") == "high"
        assert capability_group("ultra") == "ultra", (
            "identity needs no entry: whether the registry declares the name is the "
            "ROUTER's question, and it answers it by naming the groups it has"
        )

    def test_the_door_refuses_it_BEFORE_the_router_or_a_provider_is_reached(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr("crucible.llm.CAPABILITY_CLASS_GROUPS", {self.UNROUTED: None})
        # The allowlist check in `call()` runs BEFORE the group mapping, and
        # the synthetic class is declared nowhere real — mirror the shape a
        # real unrouted class would have had (declared in `llm_callsites.yaml`
        # so it clears the allowlist, then refused by the mapping) by
        # widening the allowlist directly rather than writing a throwaway
        # registry file this test does not otherwise need.
        monkeypatch.setattr("crucible.llm._capability_classes", lambda: frozenset({self.UNROUTED}))
        monkeypatch.setenv(EXEC_CONTEXT_ENV, "ci")
        monkeypatch.setattr(
            "krepis.router.resolve_group_spec",
            lambda *a, **k: pytest.fail("the router was asked for an unruled group"),
        )
        monkeypatch.setattr(
            "krepis.llm.LLMClient",
            lambda *a, **k: pytest.fail("a client was constructed for an unruled group"),
        )
        store = LocalStore(tmp_path)
        with pytest.raises(CapabilityClassNotRouted):
            run_job(
                "report",
                _body(self.UNROUTED),
                store=store,
                trading_day=DAY,
                now=NOW,
                transient_retry=False,
            )
        manifest = _manifest(store)
        assert manifest["status"] == "failed"
        assert manifest["llm_calls"] == [], "nothing was billed for a call never made"

    def test_a_REGISTERED_row_declaring_it_fails_when_the_registry_LOADS(
        self, tmp_path, monkeypatch
    ) -> None:
        """The earliest point the mapping hole is knowable.

        A phase-5 arm registering a call site against an unruled class is
        refused at import, with the mapping named — not on the first call of
        the first weekly run, after the job has already been scheduled.
        """
        monkeypatch.setattr("crucible.llm.CAPABILITY_CLASS_GROUPS", {self.UNROUTED: None})
        registry_file = tmp_path / "llm_callsites.yaml"
        registry_file.write_text(
            "schema_version: llm_callsite_registry.v1\n"
            f"capability_classes:\n  - {self.UNROUTED}\n"
            "callsites:\n"
            "  phase5.arm:\n"
            "    purpose: draft one arm's thesis\n"
            f"    capability_class: {self.UNROUTED}\n"
            "    max_usd_per_call: 0.25\n"
            "    owner: crucible.slots.research\n",
            encoding="utf-8",
        )
        monkeypatch.setattr("crucible.llm.CALLSITE_REGISTRY_PATH", registry_file)
        import crucible.llm as llm

        load_registry.cache_clear()
        llm.load_capability_classes.cache_clear()
        llm._capability_classes.cache_clear()
        try:
            with pytest.raises(CapabilityClassNotRouted):
                load_registry()
        finally:
            load_registry.cache_clear()
            llm.load_capability_classes.cache_clear()
            llm._capability_classes.cache_clear()


class TestAnUnregisteredGroupFailsByName:
    def test_the_router_refuses_it_and_names_the_groups_the_registry_declares(
        self, tmp_path, monkeypatch
    ) -> None:
        """Not a stub: the REAL `krepis.router` against a fixture registry.

        The point of routing through the router is that membership is the
        registry's answer, not a list this package keeps. So the assertion is
        that an unknown group produces a refusal naming the real declared
        set — which is also what makes `alpha-engine-config-I9971` (the
        router exposes no enumerator, so consumers approximate the set)
        visible rather than theoretical.
        """
        registry = tmp_path / "LLM_MODEL_REGISTRY.yaml"
        registry.write_text(
            "model_groups:\n  high:\n    - stub-high\n  ultra:\n    - stub-ultra\n"
            "models:\n  - id: stub-high\n  - id: stub-ultra\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("LLM_MODEL_REGISTRY_PATH", str(registry))
        monkeypatch.setenv(EXEC_CONTEXT_ENV, "ci")
        monkeypatch.setattr(
            "krepis.llm.LLMClient",
            lambda *a, **k: pytest.fail("a client was constructed for an unknown group"),
        )

        store = LocalStore(tmp_path)
        with pytest.raises(ValueError) as excinfo:
            run_job(
                "report",
                _body("med"),
                store=store,
                trading_day=DAY,
                now=NOW,
                transient_retry=False,
            )
        message = str(excinfo.value)
        assert "med" in message
        assert "high" in message and "ultra" in message, (
            "the refusal must name the groups that DO exist; a bare 'not found' "
            "leaves a reader guessing at the vocabulary"
        )


class TestTheExecutionContextIsDeclared:
    def test_an_undeclared_context_is_refused_rather_than_defaulted(
        self, tmp_path, monkeypatch
    ) -> None:
        """R28/R29. krepis' own resolution still defaults an undeclared
        context to `laptop` with a warning (a staged migration toward
        raising); this package refuses, so a spot box is never handed an
        endpoint only a laptop can reach on the strength of a log line
        nobody read."""
        monkeypatch.delenv(EXEC_CONTEXT_ENV, raising=False)
        monkeypatch.setattr(
            "krepis.router.resolve_group_spec",
            lambda *a, **k: pytest.fail("the router was asked from an undeclared context"),
        )
        with pytest.raises(ValueError) as excinfo:
            run_job(
                "report",
                _body(),
                store=LocalStore(tmp_path),
                trading_day=DAY,
                now=NOW,
                transient_retry=False,
            )
        assert EXEC_CONTEXT_ENV in str(excinfo.value)
