"""`crucible.llm.registry_preflight` — the model registry reaches this process.

`alpha-engine-config-I10346`. `crucible fault.probe` dispatched to a real v2
spot box on release `7a526e5b` failed with

    FileNotFoundError: LLM_MODEL_REGISTRY.yaml not found — set
    LLM_MODEL_REGISTRY_PATH or run from within a repo whose private-docs/
    directory contains the file.   (krepis router.py:1691)

The fault-capability-class override had carried correctly and the manifest
recorded it; the call died before reaching a transport, so the failure could
not be recorded as plan §10.7's fault 3 without putting a false attestation on
the record. Every capability class was affected identically: a v2 box checks
out no repository, so `krepis.model_registry.find_registry`'s walk-up cannot
succeed there and nothing exported `LLM_MODEL_REGISTRY_PATH`.

**What this module grades, and what it deliberately does not.** It grades the
predicate the box's bootstrap runs BEFORE the job, in the phase whose failures
page as bootstrap failures. It does not grade any LLM call: `registry_preflight`
makes none, and `test_no_socket_is_opened` asserts that by making every socket
constructor raise and requiring the happy path to still pass. Whether a
declared, live, reachable model actually answers is a different question with a
different cost, and this file makes no claim about it.

The consuming half — the S3 fetch, the `LLM_MODEL_REGISTRY_PATH` export and the
call into this function — lives in `nous-ergon-ops`
`infrastructure/cloudformation/crucible-v2.yaml` and is graded there against
the RENDERED bootstrap.
"""

from __future__ import annotations

import socket

import pytest
import yaml

from crucible.llm import (
    EXEC_CONTEXT_ENV,
    FAULT_INJECTION_CAPABILITY_CLASSES,
    RegistryPreflightFailed,
    capability_group,
    registry_preflight,
)

REGISTRY_ENV = "LLM_MODEL_REGISTRY_PATH"


def _entry(model_id: str, **overrides) -> dict:
    """A minimally routable registry row.

    Only the fields the preflight's own predicates read are set: `id`,
    `status` and `reachable_from`. Everything a real row carries — route,
    provider, endpoints, pricing, capabilities — is deliberately absent, so a
    test here cannot start passing because of a field this function never
    looks at.
    """
    row = {"id": model_id, "reachable_from": ["ec2", "laptop"]}
    row.update(overrides)
    return row


def _write_registry(tmp_path, *, groups: dict, models: list[dict]):
    path = tmp_path / "LLM_MODEL_REGISTRY.yaml"
    path.write_text(yaml.safe_dump({"models": models, "model_groups": groups}), encoding="utf-8")
    return path


@pytest.fixture
def routable(tmp_path, monkeypatch):
    """A registry in which every class this package can address resolves.

    The group set is DERIVED from `capability_group` over the same classes the
    function under test will iterate, so this fixture cannot fall behind a
    class the package gains: a new class simply gets a member here too.
    """
    from crucible.llm import _capability_classes

    classes = sorted(_capability_classes() - FAULT_INJECTION_CAPABILITY_CLASSES)
    groups = {}
    models = []
    for capability_class in classes:
        group = capability_group(capability_class)
        if group in groups:
            continue
        model_id = f"model-for-{group}"
        groups[group] = [model_id]
        models.append(_entry(model_id))
    path = _write_registry(tmp_path, groups=groups, models=models)
    monkeypatch.setenv(REGISTRY_ENV, str(path))
    monkeypatch.setenv(EXEC_CONTEXT_ENV, "ec2")
    return path


class TestTheKrepisSeamsExist:
    """The krepis entry points this function reads, asserted by name.

    `registry_preflight` calls `krepis.model_registry.entry_reachable_from` —
    PUBLIC since `alpha-engine-config-I10349` — krepis' single implementation
    of model-router-policy R28, including the rule that an entry declaring no
    `reachable_from` is reachable from NOWHERE. A second implementation of
    that predicate in this package would be the copy deciding whether a box
    may route, so this module imports it rather than re-deriving it. Prior to
    -I10349 this asserted the (then-private) router symbol by name so a
    rename would fail HERE, in CI, rather than on a spot box at the first
    preflight; the public name is a normal import now, asserted below by the
    AST scan in `test_no_underscore_prefixed_krepis_symbol_is_imported`.
    """

    def test_the_reachability_predicate_is_importable_and_honours_absence(self) -> None:
        from krepis.model_registry import entry_reachable_from

        assert entry_reachable_from({"reachable_from": ["ec2"]}, "ec2") is True
        assert entry_reachable_from({"reachable_from": ["laptop"]}, "ec2") is False
        # The load-bearing half. An entry with no `reachable_from` used to be
        # reachable from everywhere, which is how the Director Lambda resolved
        # a model at a provider it reached unscanned.
        assert entry_reachable_from({}, "ec2") is False

    def test_the_registry_loader_is_the_one_krepis_router_uses(self) -> None:
        from krepis import model_registry

        assert hasattr(model_registry, "load_registry")
        assert hasattr(model_registry.Registry, "live_group_ids")
        assert hasattr(model_registry.Registry, "capability_rejections")

    def test_no_underscore_prefixed_krepis_symbol_is_imported(self) -> None:
        """`alpha-engine-config-I10349` closes-when: an AST scan, not a grep,
        so a multi-line or aliased import cannot slip past it."""
        import ast
        import inspect

        import crucible.llm as llm_module

        tree = ast.parse(inspect.getsource(llm_module))
        offenders = [
            f"{node.module}.{alias.name}"
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module and "krepis" in node.module
            for alias in node.names
            if alias.name.startswith("_")
        ]
        assert not offenders, f"underscore-prefixed krepis import(s): {offenders}"


class TestTheTierMappingIsReadNotCopied:
    """`alpha-engine-config-I10346`, second finding.

    `_capability_classes` reads `krepis.router.TIER_GROUPS` to decide which
    tier names a call site may ASK for, and `capability_group` did not read it
    to decide what they RESOLVE to. `mid` was therefore askable and addressed
    a group named `mid` — which the registry has never declared, and never
    will: `TIER_GROUPS` maps `mid` to `med`. A call site declaring the tier
    krepis itself names would have passed the allowlist and failed at the
    router. Found by running this preflight, which is the preflight working.
    """

    def test_a_tier_resolves_to_the_group_krepis_names(self) -> None:
        from krepis.router import TIER_GROUPS

        for tier, group in TIER_GROUPS.items():
            assert capability_group(tier) == group

    def test_the_mapping_is_not_restated_in_this_package(self) -> None:
        """A copy of `TIER_GROUPS` in `CAPABILITY_CLASS_GROUPS` would be a
        second mapping, and the copy would be the one deciding what a box
        routes to."""
        from krepis.router import TIER_GROUPS

        from crucible.llm import CAPABILITY_CLASS_GROUPS

        assert not (set(CAPABILITY_CLASS_GROUPS) & set(TIER_GROUPS))

    def test_a_group_name_still_resolves_by_identity(self) -> None:
        assert capability_group("ultra") == "ultra"
        assert capability_group("chaos_probe") == "chaos_probe"


class TestTheHappyPathSaysWhatItProved:
    def test_it_names_the_registry_the_context_and_every_group(self, routable, monkeypatch) -> None:
        from crucible.llm import _capability_classes

        summary = registry_preflight()
        assert str(routable) in summary
        assert "'ec2'" in summary
        for capability_class in _capability_classes() - FAULT_INJECTION_CAPABILITY_CLASSES:
            assert f"{capability_class}->" in summary, capability_class

    def test_it_says_no_llm_call_was_made_because_none_was(self, routable) -> None:
        assert "no LLM call was made" in registry_preflight()

    def test_no_socket_is_opened(self, routable, monkeypatch) -> None:
        """The claim in the return string, asserted rather than asserted-by-
        docstring. A preflight that quietly probed a router edge would make a
        transient blip shut a box down at boot, and would make the bootstrap
        comment that says no end-to-end scan is claimed a false statement."""

        def _refuse(*args, **kwargs):
            raise AssertionError("registry_preflight opened a socket")

        monkeypatch.setattr(socket, "socket", _refuse)
        monkeypatch.setattr(socket, "create_connection", _refuse)
        assert registry_preflight()

    def test_a_fault_injection_class_is_never_asserted(
        self, routable, tmp_path, monkeypatch
    ) -> None:
        """`chaos_probe` is CONTRACTED never to serve. It would pass today —
        its members are real hosts carrying models they will never serve — so
        asserting it proves nothing about a serving path, and a legitimate
        future edit of that group would turn every box red at boot."""
        doc = yaml.safe_load(routable.read_text(encoding="utf-8"))
        for name in FAULT_INJECTION_CAPABILITY_CLASSES:
            doc["model_groups"][name] = ["a-model-that-is-not-declared"]
        routable.write_text(yaml.safe_dump(doc), encoding="utf-8")
        assert registry_preflight()


class TestTheMeasuredFailureIsCaughtBeforeTheJob:
    def test_an_absent_registry_names_the_variable_that_fixes_it(
        self, tmp_path, monkeypatch
    ) -> None:
        """The exact condition measured on the box, and the message a reader
        of a bootstrap page gets. `router.py:1691`'s own text names
        `LLM_MODEL_REGISTRY_PATH` and a repo checkout; on a v2 box the second
        half is unreachable advice, so this adds the half that is actionable
        there."""
        monkeypatch.setenv(EXEC_CONTEXT_ENV, "ec2")
        monkeypatch.delenv(REGISTRY_ENV, raising=False)
        monkeypatch.chdir(tmp_path)
        with pytest.raises(RegistryPreflightFailed) as excinfo:
            registry_preflight()
        message = str(excinfo.value)
        assert REGISTRY_ENV in message
        assert "checks out no repository" in message

    def test_a_registry_path_that_does_not_exist_is_the_same_finding(
        self, tmp_path, monkeypatch
    ) -> None:
        """`find_registry` raises `RegistryNotFoundError` — a
        `FileNotFoundError` subclass — for a set-but-missing path, and that is
        the shape a mistyped bootstrap export produces."""
        monkeypatch.setenv(EXEC_CONTEXT_ENV, "ec2")
        monkeypatch.setenv(REGISTRY_ENV, str(tmp_path / "nope.yaml"))
        with pytest.raises(RegistryPreflightFailed):
            registry_preflight()

    def test_an_undeclared_exec_context_keeps_its_own_message(self, routable, monkeypatch) -> None:
        """Not wrapped. An undeclared context is a launcher defect whose own
        message names the variable and the legal values; re-raising it as a
        registry failure would point the reader at the wrong file."""
        monkeypatch.delenv(EXEC_CONTEXT_ENV, raising=False)
        with pytest.raises(ValueError) as excinfo:
            registry_preflight()
        assert EXEC_CONTEXT_ENV in str(excinfo.value)
        assert not isinstance(excinfo.value, RegistryPreflightFailed)


class TestAStaleRegistryIsCaughtNotJustAnAbsentOne:
    """A presence check would pass against every case below.

    `alpha-engine-config-I6183` is the measured precedent: the Director
    Lambda's published copy predated the `reachable_from` migration and the
    `ultra` chain resolved a model at a provider the Lambda reached unscanned,
    while every surface read healthy. The published copy a v2 box routes on
    can fall behind the repository the same way.
    """

    @staticmethod
    def _break_one_group(path, member: dict):
        """Replace the `low` group's single member with *member*, leaving every
        other group routable.

        The preflight iterates classes in sorted order and raises on the FIRST
        unroutable one, so a fixture that declared only the group under test
        would fail on an alphabetically earlier group and the assertion would
        be about the wrong finding.
        """
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        group = capability_group("low")
        doc["model_groups"][group] = [member["id"]]
        doc["models"] = [m for m in doc["models"] if m["id"] != f"model-for-{group}"]
        doc["models"].append(member)
        path.write_text(yaml.safe_dump(doc), encoding="utf-8")
        return group

    def test_a_group_with_no_member_reachable_from_here_fails(self, routable) -> None:
        group = self._break_one_group(routable, _entry("laptop-only", reachable_from=["laptop"]))
        with pytest.raises(RegistryPreflightFailed) as excinfo:
            registry_preflight()
        message = str(excinfo.value)
        assert repr(group) in message
        assert "'ec2'" in message
        assert "laptop-only" in message
        assert "reachable_from=['laptop']" in message

    def test_a_group_whose_members_are_all_excluded_by_status_fails(self, routable) -> None:
        """Status and reachability are two different rejections, and the
        message must say which one happened: a member excluded for status may
        come back, a member unreachable from here never will without a
        registry edit."""
        self._break_one_group(routable, _entry("retired-model", status="deprecated"))
        with pytest.raises(RegistryPreflightFailed) as excinfo:
            registry_preflight()
        assert "deprecated" in str(excinfo.value)

    def test_a_group_the_registry_never_declares_fails(self, tmp_path, monkeypatch) -> None:
        """The `mid` case before `capability_group` read `TIER_GROUPS`, and the
        general case of a registry that dropped a group this package still
        addresses."""
        path = _write_registry(tmp_path, groups={}, models=[])
        monkeypatch.setenv(REGISTRY_ENV, str(path))
        monkeypatch.setenv(EXEC_CONTEXT_ENV, "ec2")
        with pytest.raises(RegistryPreflightFailed) as excinfo:
            registry_preflight()
        assert "no live member" in str(excinfo.value)


class TestThePreflightRefusesToPassVacuously:
    def test_no_assertable_class_is_a_failure_not_a_pass(self, routable, monkeypatch) -> None:
        """Today `llm_callsites.yaml` declares exactly one call site and its
        class is `chaos_probe`, so "every class the registry declares" derived
        from the CALL-SITE registry alone would be the empty set — a check
        reporting coverage of nothing. The set is derived from
        `_capability_classes`, which unions the router's own tiers, so it
        cannot empty out; this asserts what happens if it ever does."""
        import crucible.llm as llm

        monkeypatch.setattr(llm, "_capability_classes", lambda: FAULT_INJECTION_CAPABILITY_CLASSES)
        with pytest.raises(RegistryPreflightFailed) as excinfo:
            registry_preflight()
        assert "vacuously" in str(excinfo.value)
