"""`chaos_probe` is reachable on purpose, and reachable ONLY on purpose.

`alpha-engine-config-I10343`. Plan §10.7 fault 3 ("the LLM router returns
500") needs a router group contracted never to serve. The fleet's private
model registry declares one; until this change nothing in `crucible` could ask
for it, because `crucible.llm._require_capability_class` admits only the
router's tier groups plus `llm_callsites.yaml`'s own allowlist, and the
allowlist was empty. The seam existed on one side of a boundary with no
consumer able to reach it.

Making it reachable is the easy half. The half this module holds is that it
is reachable from **exactly one place** — a deliberate per-run operator
request, recorded on the manifest — and from nowhere else: not from an arm
recipe, not from a real router group's fallback chain, not from a job that
was handed the flag and quietly ignored it, and not by an operator who meant
to pick a cheaper model. A fault-injection target a production arm could
select is worse than no seam at all, so every one of those closures gets a
test that shows it FIRING rather than a test that shows the happy path.

**What is asserted here and what is asserted elsewhere.** The "nothing falls
back INTO the group" property is enforced where fallback chains are declared —
`alpha-engine-config/scripts/validate_llm_model_registry.py` invariant 21, a
blocking error on every registry PR, checked per group and in both directions
(no `chaos_probe` member may join an ordinary chain; no ordinary model may
join a chaos group). This package cannot read that registry and must not learn
to. What it CAN assert, and does below, is the crucible-side half: no
`CAPABILITY_CLASS_GROUPS` row redirects any class into a fault-injection
group, and no fault-injection class is a bare router tier — so the only reason
one is askable at all is the deliberate, reasoned edit in
`llm_callsites.yaml`. The runtime backstop for the registry half is
`crucible.fault_probe.FaultProbeServedError`: if the group ever serves, the
probe fails loudly rather than recording an `ok`.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from crucible import fault_probe, llm
from crucible.alerts import sweep
from crucible.cli import FAULT_CAPABILITY_CLASS_JOBS, build_parser
from crucible.manifest import manifest_key, validate
from crucible.runner import RunContext, run_job
from crucible.slots.arms import load_arm_specs
from crucible.store import LocalStore

FRIDAY = dt.date(2026, 8, 28)
NOW = dt.datetime(2026, 8, 28, 21, 0, tzinfo=dt.UTC)
AFTER_EVERY_DEADLINE = dt.datetime(2026, 8, 29, 23, 30, tzinfo=dt.UTC)
PROBE_CLASS = "chaos_probe"


class _Refusing:
    """The upstream host answering the fault-injection group.

    `.complete()` is the surface `krepis.llm.LLMClient` calls, so raising here
    is the provider refusing on the wire — which is what a `chaos_probe`
    member's unservable `model` string really produces. Everything above it in
    `crucible.llm.call` (the registry lookup, the ceiling, the cap admission,
    the group resolution) runs unfaked.
    """

    def complete(self, **_kwargs: object) -> None:
        raise RuntimeError("provider_5xx: model not found for the requested deployment")


class _Serving:
    """The upstream host answering, which must never happen."""

    model = "some-real-model"
    fallback_used = False
    served_deployment = None

    class usage:  # noqa: N801 - mirrors krepis' own attribute shape
        input_tokens = 1
        output_tokens = 1
        cache_read_tokens = 0
        cache_create_tokens = 0
        provider_cost_usd = 0.0

    def complete(self, **_kwargs: object) -> _Serving:
        return self


@pytest.fixture
def routed(monkeypatch):
    """Resolve the router without the private registry, and record the ASK.

    Returns the list the resolved group name is appended to, which is how the
    tests below tell "the router was asked for the fault-injection group" from
    "the call site said it was".
    """
    asked: list[str] = []

    def _resolve(group, **_kwargs):
        asked.append(group)
        return (object(), {})

    monkeypatch.setenv("KREPIS_EXEC_CONTEXT", "ci")
    monkeypatch.setattr("krepis.router.resolve_group_spec", _resolve)
    monkeypatch.setattr("krepis.router.route_is_degraded", lambda _route: False)
    return asked


class TestTheClassIsAskableAndOnlyDeliberatelySo:
    def test_every_fault_injection_class_is_askable(self) -> None:
        """The finding itself: `chaos_probe` was declared as a model group and
        `_require_capability_class` raised on it, so no call site could ask.
        Fails with `llm_callsites.yaml`'s `capability_classes` back to `[]`."""
        allowed = llm._capability_classes()
        for name in sorted(llm.FAULT_INJECTION_CAPABILITY_CLASSES):
            assert name in allowed, (
                f"{name!r} is declared a fault-injection target but is not askable, so "
                "fault 3 cannot be induced against any job"
            )

    def test_no_fault_injection_class_is_a_bare_router_tier(self) -> None:
        """What makes the allowlist edit the SOURCE of reachability rather
        than decoration. If one of these ever became a `krepis.router` tier
        group it would be askable with no deliberate edit and no reason
        written down anywhere, which is the property this repository's
        allowlist exists to prevent."""
        from krepis.router import TIER_GROUPS

        tiers = frozenset(TIER_GROUPS) | frozenset(TIER_GROUPS.values())
        assert not (llm.FAULT_INJECTION_CAPABILITY_CLASSES & tiers), (
            "a fault-injection class became a router TIER group; it is now askable "
            "without the deliberate allowlist edit that carries its reason"
        )

    def test_the_allowlist_and_the_constant_agree(self) -> None:
        """The file carries the REASON for each name, the constant is what the
        code refuses on. Two places, so both are checkable; equal, so neither
        can drift into being the only one that is true."""
        assert set(llm.load_capability_classes()) == llm.FAULT_INJECTION_CAPABILITY_CLASSES

    def test_nothing_is_redirected_INTO_a_fault_injection_group(self) -> None:
        """The crucible-side half of "nothing falls back into it".

        A `CAPABILITY_CLASS_GROUPS` row mapping some ordinary class onto a
        fault-injection group would silently point a real call site at a group
        contracted never to serve — the same defect as a poisoned fallback
        chain, reached through this package instead of the registry. The
        registry-side half is invariant 21 in
        `validate_llm_model_registry.py`.
        """
        for declared, group in llm.CAPABILITY_CLASS_GROUPS.items():
            assert group not in llm.FAULT_INJECTION_CAPABILITY_CLASSES, (
                f"capability class {declared!r} is mapped onto fault-injection group "
                f"{group!r}; every call site asking for {declared!r} would be routed into "
                "a group that never serves"
            )

    def test_a_fault_injection_class_resolves_by_identity(self) -> None:
        """`-I9970`'s shape: a class whose name IS a registry group needs no
        row, and a row would be a second place for the mapping to drift."""
        for name in sorted(llm.FAULT_INJECTION_CAPABILITY_CLASSES):
            assert llm.capability_group(name) == name


class TestTheRequestMechanism:
    def test_a_fault_injection_class_is_accepted(self) -> None:
        assert llm.parse_fault_capability_class(PROBE_CLASS) == PROBE_CLASS

    @pytest.mark.parametrize("value", ["high", "low", "med", "mid", "ultra"])
    def test_a_real_router_group_is_REFUSED(self, value: str) -> None:
        """The refusal that keeps this from becoming a second routing plane.

        A per-invocation flag that could select `high` would let an operator
        decide which model serves a graded run — the layer
        `model-router-policy` §2 keeps above this consumer. That a name is a
        perfectly legitimate capability class everywhere else is exactly why
        it has to be refused HERE.
        """
        with pytest.raises(llm.FaultCapabilityClassRefused):
            llm.parse_fault_capability_class(value)

    @pytest.mark.parametrize("value", ["", "  ", "gpt-5", "CHAOS_PROBE", "chaos", "chaos_probe2"])
    def test_a_non_class_is_refused(self, value: str) -> None:
        with pytest.raises(llm.FaultCapabilityClassRefused):
            llm.parse_fault_capability_class(value)

    def test_a_run_with_no_override_asks_for_the_declared_class(self) -> None:
        """The default path, and the reason the getattr default in
        `effective_capability_class` is a reading rather than a swallow:
        "this run declared no override" is the true state of every natural
        run, including on a context that has no such attribute at all."""

        class _Bare:
            pass

        assert llm.effective_capability_class(_Bare(), "high") == "high"

    def test_an_override_planted_outside_the_CLI_is_still_refused(self) -> None:
        """The flag validates, but the flag is not the only way to set the
        attribute — a job body, a fixture or a future launcher could. The door
        re-validates, so the CLI's `choices` is a convenience and not the
        enforcement."""

        class _Sneaky:
            fault_capability_class = "high"

        with pytest.raises(llm.FaultCapabilityClassRefused):
            llm.effective_capability_class(_Sneaky(), "low")

    def test_the_router_is_asked_for_the_OVERRIDE_not_the_declared_class(
        self, tmp_path, routed, monkeypatch
    ) -> None:
        """The acting half: a call site that declares `high` and knows nothing
        about fault injection is redirected, which is what makes faulting a
        real arm run possible in phase 5.

        Fails if `crucible.llm.call` resolves the group from the declared
        class — the state before this change, in which the override could be
        recorded on the manifest while changing no routing at all, which is
        the worst of the available shapes: a manifest claiming an induced
        fault over a run that reached a real model.
        """
        monkeypatch.setattr("krepis.llm.LLMClient", lambda *a, **k: _Refusing())
        site = llm.CallSite(
            callsite_id="research.thesis",
            purpose="a real, serving call site",
            capability_class="high",
            max_usd_per_call=0.25,
            owner="tests.test_chaos_probe_containment",
        )
        store = LocalStore(tmp_path)

        def body(ctx: RunContext) -> None:
            llm.call(
                ctx,
                callsite_id="research.thesis",
                capability_class="high",
                messages=[{"role": "user", "content": "probe"}],
                cap=llm.SpendCap(cap_usd=5.0),
                estimate_usd=0.01,
                registry={"research.thesis": site},
                client_factory=lambda *a, **k: _Refusing(),
            )

        with pytest.raises(RuntimeError, match="provider_5xx"):
            run_job(
                "experiment.run",
                body,
                store=store,
                trading_day=FRIDAY,
                now=NOW,
                run_mode="replay",
                transient_retry=False,
                fault_capability_class=PROBE_CLASS,
            )

        assert routed == [PROBE_CLASS], (
            "the router was asked for the call site's declared class, so the override "
            "changed nothing but the manifest"
        )

    def test_the_manifest_records_the_override(self, tmp_path, routed, monkeypatch) -> None:
        """`-I10125`'s `now_override_utc` shape. A fault record filed as
        `induced` names a FAILED manifest, so a reader must be able to tell an
        arranged transport failure from an observed one."""
        monkeypatch.setattr("krepis.llm.LLMClient", lambda *a, **k: _Refusing())
        store = LocalStore(tmp_path)

        with pytest.raises(RuntimeError, match="provider_5xx"):
            run_job(
                fault_probe.FAULT_PROBE_JOB,
                fault_probe.probe_body,
                store=store,
                trading_day=FRIDAY,
                now=NOW,
                run_mode="replay",
                transient_retry=False,
                fault_capability_class=PROBE_CLASS,
            )

        document = json.loads(
            store.get_bytes(manifest_key(fault_probe.FAULT_PROBE_JOB, FRIDAY.isoformat()))
        )
        validate(document)
        assert document["fault_capability_class"] == PROBE_CLASS
        assert document["status"] == "failed"

    def test_a_natural_run_carries_no_such_field_at_all(self, tmp_path) -> None:
        """Omitted, not written as null: a manifest from a run that never
        overrode anything stays byte-identical to one from before the field
        existed."""
        store = LocalStore(tmp_path)
        ctx = run_job(
            "heartbeat",
            lambda _ctx: None,
            store=store,
            trading_day=FRIDAY,
            now=NOW,
            run_mode="replay",
        )
        document = json.loads(store.get_bytes(manifest_key("heartbeat", FRIDAY.isoformat())))
        assert "fault_capability_class" not in document
        assert ctx.fault_capability_class is None


class TestTheFlagSurface:
    def test_the_flag_is_required_where_it_is_offered(self) -> None:
        """`fault.probe` exists for no other purpose, so there is no default:
        a probe that picked its own class could induce a fault nobody asked
        for."""
        with pytest.raises(SystemExit):
            build_parser().parse_args([fault_probe.FAULT_PROBE_JOB])

    def test_the_flag_refuses_a_real_router_group_at_the_parser(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(
                [fault_probe.FAULT_PROBE_JOB, "--fault-capability-class", "high"]
            )

    def test_a_job_that_cannot_reach_a_model_refuses_the_flag(self) -> None:
        """Rule 5, in the shape argparse can enforce it: the flag only does
        anything for a job that reaches the door, so offering it everywhere
        would let an operator "induce" a fault on `board` and get a manifest
        recording an override that had no call site to fire in."""
        assert "board" not in FAULT_CAPABILITY_CLASS_JOBS
        with pytest.raises(SystemExit):
            build_parser().parse_args(["board", "--fault-capability-class", PROBE_CLASS])

    def test_every_job_offered_the_flag_can_actually_reach_a_model(self) -> None:
        """The other direction. Today the set is exactly the probe; phase 5
        adds `experiment.run` in the same change that gives it an LLM arm, and
        this is the assertion that has to be edited deliberately when it
        does."""
        assert FAULT_CAPABILITY_CLASS_JOBS == frozenset({fault_probe.FAULT_PROBE_JOB})


class TestNoArmCanSelectIt:
    """The containment that matters most: an arm is what gets GRADED."""

    @staticmethod
    def _recipe(directory, *, callsite: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "thesis.yaml").write_text(
            "name: thesis\n"
            "slot: u\n"
            "ranker: momentum_sleeve\n"
            "registered_at: '2026-06-01'\n"
            "params:\n"
            "  top_n: 8\n"
            f"  llm_callsite: {callsite}\n",
            encoding="utf-8",
        )

    def test_an_arm_naming_the_fault_injection_call_site_is_refused_at_load(self, tmp_path) -> None:
        """Refused at PARSE time, before an id exists. An arm id is the hash
        of its spec, so a recipe that reached a register row would have a
        durable id nobody can retract — and the arena would grade a model
        that never answered.

        Fails without the refusal in `crucible.slots.arms._require_registered
        _call_site`: `params.llm_callsite` accepts any key of the registry,
        and registering the probe is exactly what put a fault-injection target
        into that keyspace.
        """
        self._recipe(tmp_path / "arms" / "u", callsite=fault_probe.FAULT_PROBE_CALLSITE_ID)
        with pytest.raises(ValueError, match="FAULT-INJECTION target"):
            load_arm_specs("u", strategy_dir=tmp_path)

    def test_an_arm_naming_an_ordinary_call_site_still_loads(self, tmp_path, monkeypatch) -> None:
        """The guard is not a blanket refusal of LLM arms — which is what a
        test asserting only the refusal above would leave undetectable."""
        registry = dict(llm.LLM_CALLSITE_REGISTRY)
        registry["research.thesis"] = llm.CallSite(
            callsite_id="research.thesis",
            purpose="a real, serving call site",
            capability_class="high",
            max_usd_per_call=0.25,
            owner="tests.test_chaos_probe_containment",
        )
        monkeypatch.setattr(llm, "LLM_CALLSITE_REGISTRY", registry)
        self._recipe(tmp_path / "arms" / "u", callsite="research.thesis")
        (spec,) = load_arm_specs("u", strategy_dir=tmp_path)
        assert spec.spec["params"]["llm_callsite"] == "research.thesis"

    def test_no_registered_call_site_but_the_probe_declares_one(self) -> None:
        """The registry's own shape: a fault-injection class belongs to a call
        site that exists in order to fail, and to nothing else. A serving call
        site declaring one would be an arm's whole route into the group even
        with the loader refusal above, since the arm would look ordinary.
        """
        faulty = {
            callsite_id
            for callsite_id, row in llm.LLM_CALLSITE_REGISTRY.items()
            if row.capability_class in llm.FAULT_INJECTION_CAPABILITY_CLASSES
        }
        assert faulty == {fault_probe.FAULT_PROBE_CALLSITE_ID}


class TestTheProbeJob:
    def test_it_fails_with_the_transport_cause_full_telemetry_and_one_page(
        self, tmp_path, routed, transport, monkeypatch
    ) -> None:
        """§10.7's actual requirement for fault 3, against the real runner and
        the real alerter: `status: failed` naming the transport failure, all
        five §9.2 signal classes, exactly one page."""
        monkeypatch.setattr("krepis.llm.LLMClient", lambda *a, **k: _Refusing())
        store = LocalStore(tmp_path)

        with pytest.raises(RuntimeError, match="provider_5xx"):
            run_job(
                fault_probe.FAULT_PROBE_JOB,
                fault_probe.probe_body,
                store=store,
                trading_day=FRIDAY,
                now=NOW,
                run_mode="replay",
                transient_retry=False,
                fault_capability_class=PROBE_CLASS,
            )

        document = json.loads(
            store.get_bytes(manifest_key(fault_probe.FAULT_PROBE_JOB, FRIDAY.isoformat()))
        )
        validate(document)
        assert document["status"] == "failed"
        assert "provider_5xx" in document["reason"]
        assert document["llm_calls"] == [], (
            "the call raised before it could be recorded — nothing was billed for a call "
            "that never completed"
        )
        assert document["cost_usd"] == 0.0
        assert document["resource"]["instance_type"]
        assert isinstance(document["metrics"], list)

        sweep(store, now=AFTER_EVERY_DEADLINE, transport=transport, sweep_run_id="0" * 26)
        failures = sum(
            # `crucible-v2/synthetic/failure`, not `crucible-v2/failure`: this
            # probe declared `--fault-capability-class`, so `crucible.synthetic`
            # marks its page as the deliberate exercise it is. Asserting the
            # SYNTHETIC source rather than widening the count to both is the
            # point — a probe whose page were indistinguishable from a real
            # data outage is the defect measured on 2026-09-09, and this
            # assertion is what would catch its return.
            1
            for call in transport.calls
            if call.kwargs.get("source") == "crucible-v2/synthetic/failure"
        )
        assert failures == 1

    def test_a_probe_that_IS_SERVED_fails_rather_than_reporting_ok(
        self, tmp_path, routed, monkeypatch
    ) -> None:
        """The runtime backstop for the registry-side purity invariant this
        package cannot read. If the group ever serves, every `induced` record
        written against this path would be evidence about a fault that no
        longer exists — so the probe fails loudly instead of recording an `ok`
        run that quietly proves nothing."""
        monkeypatch.setattr("krepis.llm.LLMClient", lambda *a, **k: _Serving())
        store = LocalStore(tmp_path)

        with pytest.raises(fault_probe.FaultProbeServedError):
            run_job(
                fault_probe.FAULT_PROBE_JOB,
                fault_probe.probe_body,
                store=store,
                trading_day=FRIDAY,
                now=NOW,
                run_mode="replay",
                transient_retry=False,
                fault_capability_class=PROBE_CLASS,
                # The served path RECORDS a call, so the ceiling and the cap
                # run for real here — a served probe that also breached the cap
                # must still report the seam breach, which is the louder fact.
            )

        document = json.loads(
            store.get_bytes(manifest_key(fault_probe.FAULT_PROBE_JOB, FRIDAY.isoformat()))
        )
        assert document["status"] == "failed"
        assert "never to serve" in document["reason"]

    def test_the_body_refuses_a_context_that_never_requested_a_class(self, tmp_path) -> None:
        """There is no natural invocation of this job. A context reaching the
        body with no request did not come from the CLI, and defaulting to a
        fault-injection class would be this module choosing to induce a fault
        nobody asked for."""
        store = LocalStore(tmp_path)
        with pytest.raises(ValueError, match="no fault-injection capability class"):
            run_job(
                fault_probe.FAULT_PROBE_JOB,
                fault_probe.probe_body,
                store=store,
                trading_day=FRIDAY,
                now=NOW,
                run_mode="replay",
                transient_retry=False,
            )

    def test_a_dry_run_probe_still_fails_and_writes_no_manifest(
        self, tmp_path, routed, monkeypatch
    ) -> None:
        """The `--dry-run` row `tests/test_cli_and_alerts.py` excludes this
        job from, asserted here instead: the probe reaches a real provider on
        every invocation by design, so there is no clean dry-run print — only
        the transport failure, and no manifest."""
        monkeypatch.setattr("krepis.llm.LLMClient", lambda *a, **k: _Refusing())
        store = LocalStore(tmp_path)

        with pytest.raises(RuntimeError, match="provider_5xx"):
            run_job(
                fault_probe.FAULT_PROBE_JOB,
                fault_probe.probe_body,
                store=store,
                trading_day=FRIDAY,
                now=NOW,
                run_mode="replay",
                transient_retry=False,
                dry_run=True,
                fault_capability_class=PROBE_CLASS,
            )

        assert sorted(store.list_keys()) == []
