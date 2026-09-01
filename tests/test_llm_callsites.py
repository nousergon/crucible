"""`LLM_CALLSITE_REGISTRY` coverage, measured by walking the code.

Plan §2 row 7 / §4.8. The clause names the method as well as the result:
coverage is "measured by a test that enumerates call sites from the code
rather than from a list". A registry compared against a hand-maintained list
of call sites is two lists agreeing with each other, which is what the audit
found under every green check it examined.

Phase 1 carries no LLM arms, so the registry is empty. That makes the second
half of this module the load-bearing half: the enumerator is exercised against
fixture trees that DO contain call sites, and it must report each of them.
An enumerator that finds nothing in an empty package and would also find
nothing in a full one is the dark-detector failure, not a passing gate.
"""

from __future__ import annotations

import pathlib

import pytest

from crucible.llm import (
    LLM_CALLSITE_REGISTRY,
    PROVIDER_MODULES,
    CallSite,
    audit_call_sites,
    load_registry,
)

PACKAGE = pathlib.Path(__file__).resolve().parents[1] / "crucible"

REGISTERED = {
    "research.thesis": CallSite(
        callsite_id="research.thesis",
        purpose="draft one arm's thesis from the day's features",
        capability_class="reasoning_high",
        max_usd_per_call=0.25,
        owner="crucible.slots.research",
    )
}


def _write(root: pathlib.Path, name: str, source: str) -> pathlib.Path:
    path = root / name
    path.write_text(source, encoding="utf-8")
    return path


class TestTheRegistry:
    def test_it_loads_and_is_the_module_level_mapping(self) -> None:
        assert load_registry() == LLM_CALLSITE_REGISTRY

    def test_every_row_declares_an_owner_a_class_and_a_ceiling(self) -> None:
        for site in LLM_CALLSITE_REGISTRY.values():
            assert site.owner and site.purpose and site.capability_class
            assert site.max_usd_per_call > 0

    def test_no_row_names_a_provider_model(self) -> None:
        """Principle 8: the registry addresses capability classes. A model id
        here would put the lock-in one indirection away rather than remove it."""
        for site in LLM_CALLSITE_REGISTRY.values():
            lowered = site.capability_class.lower()
            assert not any(m in lowered for m in ("gpt-", "claude-", "gemini-", "glm-"))


class TestCoverageOfThePackage:
    def test_every_llm_call_site_in_the_package_is_registered(self) -> None:
        findings = audit_call_sites(PACKAGE)
        assert findings == [], "\n".join(f.describe() for f in findings)

    def test_the_walk_actually_reads_the_package(self) -> None:
        """A walk that reads nothing reports clean. Asserting a floor on what
        it parsed means a broken walk fails here instead of blessing the tree."""
        modules = [p for p in PACKAGE.rglob("*.py") if "__pycache__" not in p.parts]
        assert len(modules) >= 20


class TestTheEnumeratorCatchesWhatItMustCatch:
    """Proof by fixture: each tree below contains a real call site, and the
    audit must report it. These are what make the empty result above evidence
    rather than a coincidence."""

    def test_an_unregistered_callsite_id_is_reported(self, tmp_path) -> None:
        _write(
            tmp_path,
            "arm.py",
            "from crucible.llm import call\n"
            "def go(ctx, cap):\n"
            "    return call(ctx, callsite_id='research.unregistered', "
            "capability_class='reasoning_high', messages=[], cap=cap, estimate_usd=0.1)\n",
        )
        findings = audit_call_sites(tmp_path, registry=REGISTERED)
        assert [f.kind for f in findings] == ["unregistered_callsite"]
        assert "research.unregistered" in findings[0].detail

    def test_a_registered_callsite_id_passes(self, tmp_path) -> None:
        _write(
            tmp_path,
            "arm.py",
            "from crucible.llm import call\n"
            "def go(ctx, cap):\n"
            "    return call(ctx, callsite_id='research.thesis', "
            "capability_class='reasoning_high', messages=[], cap=cap, estimate_usd=0.1)\n",
        )
        assert audit_call_sites(tmp_path, registry=REGISTERED) == []

    def test_a_computed_callsite_id_is_reported(self, tmp_path) -> None:
        """An id a static reader cannot resolve is an id the spend join cannot
        resolve either — the 18x cost undercount wore exactly this shape."""
        _write(
            tmp_path,
            "arm.py",
            "from crucible.llm import call\n"
            "def go(ctx, cap, slot):\n"
            "    return call(ctx, callsite_id='research.' + slot, "
            "capability_class='reasoning_high', messages=[], cap=cap, estimate_usd=0.1)\n",
        )
        findings = audit_call_sites(tmp_path, registry=REGISTERED)
        assert [f.kind for f in findings] == ["unregistered_callsite"]
        assert "literal" in findings[0].detail

    def test_a_call_through_the_module_alias_is_still_seen(self, tmp_path) -> None:
        _write(
            tmp_path,
            "arm.py",
            "from crucible import llm\n"
            "def go(ctx, cap):\n"
            "    return llm.call(ctx, callsite_id='nope', capability_class='x', "
            "messages=[], cap=cap, estimate_usd=0.1)\n",
        )
        assert [f.kind for f in audit_call_sites(tmp_path, registry=REGISTERED)] == [
            "unregistered_callsite"
        ]

    @pytest.mark.parametrize("module", sorted(PROVIDER_MODULES))
    def test_reaching_a_provider_around_the_adapter_is_reported(self, tmp_path, module) -> None:
        """The half that makes coverage mean anything: a call site that never
        touches `crucible.llm.call` would otherwise be invisible to a registry
        check, and it is exactly how spend escapes attribution."""
        _write(tmp_path, "arm.py", f"import {module}\n")
        findings = audit_call_sites(tmp_path, registry=REGISTERED)
        assert [f.kind for f in findings] == ["adapter_bypass"]
        assert module in findings[0].detail

    def test_a_from_import_of_a_provider_is_reported(self, tmp_path) -> None:
        _write(tmp_path, "arm.py", "from krepis.llm import LLMClient\n")
        assert [f.kind for f in audit_call_sites(tmp_path, registry=REGISTERED)] == [
            "adapter_bypass"
        ]

    def test_a_file_that_cannot_be_parsed_is_a_finding_not_a_pass(self, tmp_path) -> None:
        """Skipping an unparseable file would make a syntax error the way to
        become invisible to the audit."""
        _write(tmp_path, "broken.py", "def go(:\n")
        assert [f.kind for f in audit_call_sites(tmp_path, registry=REGISTERED)] == ["unparseable"]
