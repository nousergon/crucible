"""Enforces `crucible/keys.py`'s own rule: every store key shape, in one place.

Normative source: `crucible/keys.py`'s module docstring — "a key format
restated at each call site is a contract restated fifty times, and one of
them has already drifted" — and alpha-engine-config-I9807, which moved
`arm_predictions_key` out of `crucible/slots/inputs.py` after exactly that
drift was found (§ its sibling `champion_key`/`arena_cycle_key` had already
happened, silently, before this test existed).

**What this checks, precisely, and why not more.** A predicate over string
CONTENT (grepping every f-string in the package for something that looks
like `"data/{x}/y.json"`) would be noisy: plenty of f-strings that are not
store keys share that shape by coincidence (a log message, a glob pattern
for a status line). Instead this walks the AST for a much narrower, exact
signal: a **function definition** (top-level or nested, `def` or `async
def`) outside `crucible/keys.py` whose **name** matches the naming
convention every real key function in this package already uses —
`..._key` or `..._prefix` (`arm_predictions_key`, `cross_section_key`,
`manifest_prefix`, `gate_key`, `bus_key`, ...). That convention is not
incidental: it is what a reader, and every existing key function, already
commits to. Still not caught: an assignment-defined callable
(`foo_key = _make_key_fn(...)`) — `ast.Name` targets are not function
definitions, and matching them by name would mean matching every assignment
in the package, the exact string-content noise this test is built to avoid.
No real key function in this package is written that way today.

A hit is not automatically a defect: some `*_key` functions are not store
keys at all (`crucible.alerts.dedup_key` is a dedup identity string), and a
few are store keys that are *legitimately* not in `crucible.keys` because
they are built from a private validator reused elsewhere in their own
module (moving just the key function would either duplicate that validator
in `crucible.keys` or make the generic key module import domain logic).
Those go in `_KNOWN_ARCHITECTURAL_EXCEPTIONS` below. A hit that is a real
key function simply not moved YET — because the file it lives in is out of
scope for the PR that would move it — goes in `_KNOWN_TRACKED_DEBT`
instead, naming the issue that clears it. **Both registries are named
`_KNOWN_*` on purpose**, so `tests/test_no_suppressions.py`'s own scan sees
and sanctions them explicitly (AGENTS.md rule 4: no suppression collection
the scanner is blind to — see each collection's docstring for the full
argument). An unlisted hit fails this test until it is moved into
`crucible.keys`, or added to one of the two registries with a reason.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*_(key|prefix)$")

_CRUCIBLE_ROOT = Path(__file__).resolve().parent.parent / "crucible"

#: module (dotted, relative to `crucible/`) -> {function name: reason it
#: CORRECTLY lives outside `crucible.keys`, permanently}. This is an
#: architectural registry, not a debt list: every entry here is expected to
#: still be here in a year, because the function genuinely does not belong
#: in the generic key module (it is not a store key at all, or it is built
#: from a private validator this module owns and reuses elsewhere). A case
#: that is merely UNFIXED YET belongs in `_KNOWN_TRACKED_DEBT` below, not
#: here — see that collection's own docstring for why the distinction
#: matters and is enforced, not just asserted.
#:
#: Named `_KNOWN_*` ON PURPOSE, deliberately matching
#: `tests/test_no_suppressions.py::FORBIDDEN`'s `_KNOWN_` pattern, so that
#: scanner sees this collection and sanctions it explicitly (a named,
#: reviewed exception it knows about) rather than being blind to it because
#: the name never matched anything it checks for — which is exactly the
#: defect this collection would otherwise BE: a suppression list evading the
#: suppression scanner (AGENTS.md rule 4; caught on review of
#: alpha-engine-config-I9807, 2026-09-02).
_KNOWN_ARCHITECTURAL_EXCEPTIONS: dict[str, dict[str, str]] = {
    "config": {
        "cloudtrail_bucket_prefix": (
            "a Settings METHOD (bound to self, found only once the AST walk was "
            "widened to ast.walk for I9807's review) that splits a CloudTrail archive "
            "URI into (bucket, prefix) — returns a tuple, not a string, and is not "
            "the crucible store key grammar at all; unrelated AWS domain."
        ),
    },
    "alerts": {
        "dedup_key": "a dedup IDENTITY string (condition, subject, day), not a store key.",
        "cause_key": "a dedup IDENTITY string derived from a Page, not a store key.",
        "incident_key": "a dedup IDENTITY string derived from a PageGroup, not a store key.",
        "bus_key": (
            "a real store key, but built from this module's own incident_id(group) — "
            "moving it alone would either duplicate that derivation in crucible.keys or "
            "make the generic key module import alert-domain logic (I9807 sweep)."
        ),
    },
    "release": {
        "release_prefix": (
            "wraps _assert_sha, a release-domain validator reused directly by other "
            "callers in this module (not just by the key functions below) — moving it "
            "would duplicate sha validation in crucible.keys (I9807 sweep)."
        ),
        "wheel_key": "built on release_prefix; same reason as release_prefix.",
        "release_json_key": "built on release_prefix; same reason as release_prefix.",
        "provenance_key": "built on release_prefix; same reason as release_prefix.",
    },
}

#: module -> {function name: reason it has NOT moved yet, and what clears
#: the entry}. Unlike `_KNOWN_ARCHITECTURAL_EXCEPTIONS`, every entry here is
#: expected to be REMOVED — this collection's target size is zero, not a
#: fact about where the function belongs. Also named `_KNOWN_*` on purpose,
#: for the same reason as above: a debt list the suppression scanner cannot
#: see is worse than no list, because it looks like every entry was reviewed
#: when only its EXISTENCE was.
_KNOWN_TRACKED_DEBT: dict[str, dict[str, str]] = {
    "gate": {
        "gate_key": (
            "gate.py was being edited concurrently by another session when I9807 "
            "landed, the same reason arm_predictions_key was originally placed "
            "outside crucible.keys — not touched here for the same reason. Moving "
            "it is alpha-engine-config-I9852's own tracked deliverable, which also "
            "removes this entry as part of closing that issue."
        ),
    },
}


def _module_level_key_functions() -> list[tuple[str, str]]:
    """(module, function name) for every `*_key`/`*_prefix` def under
    `crucible/`, excluding `crucible/keys.py` itself.

    `ast.walk`, not `tree.body`: catches a nested `def foo_key` (inside a
    class, an `if`, or another function) and `async def foo_key`, not only a
    plain top-level `def`. An assignment-defined callable
    (`foo_key = _make_key_fn(...)`) is still not caught — `ast.Name` targets
    are not function definitions, and matching them would mean matching
    every assignment in the package by name, which is a real risk of a false
    positive this test's own docstring warns against elsewhere.
    """
    hits: list[tuple[str, str]] = []
    for path in sorted(_CRUCIBLE_ROOT.rglob("*.py")):
        if path == _CRUCIBLE_ROOT / "keys.py":
            continue
        module = path.relative_to(_CRUCIBLE_ROOT).with_suffix("").as_posix().replace("/", ".")
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and _NAME_RE.match(
                node.name
            ):
                hits.append((module, node.name))
    return hits


def _accounted_for(module: str, name: str) -> bool:
    return name in _KNOWN_ARCHITECTURAL_EXCEPTIONS.get(
        module, {}
    ) or name in _KNOWN_TRACKED_DEBT.get(module, {})


class TestEveryKeyShapedFunctionIsAccountedFor:
    def test_every_hit_outside_keys_py_is_accounted_for_with_a_reason(self) -> None:
        hits = _module_level_key_functions()
        assert hits, (
            "the AST walk found zero *_key/*_prefix functions outside crucible/keys.py, "
            "which is suspicious given known ones (crucible.gate.gate_key at least) — the "
            "walk itself is probably broken, not the codebase."
        )
        unaccounted = [
            f"{module}::{name}" for module, name in hits if not _accounted_for(module, name)
        ]
        assert not unaccounted, (
            "a *_key/*_prefix function was added outside crucible/keys.py with no "
            "registry entry: " + ", ".join(unaccounted) + ". Either move it into "
            "crucible/keys.py (crucible/keys.py's own rule: every store key shape, in "
            "one place), add it to _KNOWN_ARCHITECTURAL_EXCEPTIONS if it genuinely "
            "belongs outside crucible.keys permanently, or to _KNOWN_TRACKED_DEBT with "
            "the issue that will clear it if it does not."
        )

    @pytest.mark.parametrize(
        ("registry_name", "module", "name"),
        [
            ("_KNOWN_ARCHITECTURAL_EXCEPTIONS", m, n)
            for m, entries in _KNOWN_ARCHITECTURAL_EXCEPTIONS.items()
            for n in entries
        ]
        + [
            ("_KNOWN_TRACKED_DEBT", m, n)
            for m, entries in _KNOWN_TRACKED_DEBT.items()
            for n in entries
        ],
    )
    def test_no_stale_registry_entry(self, registry_name: str, module: str, name: str) -> None:
        """An entry naming a function that moved, was renamed, or was deleted is a
        stale exemption — silently permissive rather than caught."""
        matching = {n for m, n in _module_level_key_functions() if m == module}
        assert name in matching, (
            f"{registry_name} names {module}::{name}, but no such function exists in "
            "crucible/ any more. Remove the stale entry."
        )
