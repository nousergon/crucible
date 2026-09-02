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
signal: a **module-level function definition** outside `crucible/keys.py`
whose **name** matches the naming convention every real key function in this
package already uses — `..._key` or `..._prefix` (`arm_predictions_key`,
`cross_section_key`, `manifest_prefix`, `gate_key`, `bus_key`, ...). That
convention is not incidental: it is what a reader, and every existing key
function, already commits to.

A hit is not automatically a defect: some `*_key` functions are not store
keys at all (`crucible.alerts.dedup_key` is a dedup identity string), and a
few are store keys that are *legitimately* not in `crucible.keys` because
they are built from a private validator reused elsewhere in their own module
(moving just the key function would either duplicate that validator in
`crucible.keys` or make the generic key module import domain logic — see
each entry's reason below). Every hit is therefore required to be named,
with a reason, in `_ALLOWED`. An unlisted module fails the test: a new
`*_key`/`*_prefix` function found by the walk below fails until it is
either moved into `crucible.keys` or added here with a reason.

**What the walk does NOT see — narrower than "fails closed" would imply.**
`_module_level_key_functions` matches only `ast.FunctionDef` nodes in a
module's top-level `tree.body`. It does not see: `async def foo_key(...)`,
a `def foo_key` nested inside a class, an `if`, or another function, or an
assignment-defined callable (`foo_key = _make_key_fn(...)`). Given every
real key function in this package today is a plain top-level `def`, this is
low practical risk, not zero — a future key function written in one of
those shapes would not be caught here. Narrow this comment before trusting
it as a hard guarantee; widening the walk (`ast.walk` plus `AsyncFunctionDef`
and an assignment check) is the fix if that gap is ever exercised.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*_(key|prefix)$")

_CRUCIBLE_ROOT = Path(__file__).resolve().parent.parent / "crucible"

#: module (dotted, relative to `crucible/`) -> {function name: reason it is
#: not in `crucible.keys`}. Every entry needs a reason; there is no bare
#: module-level exemption.
_ALLOWED: dict[str, dict[str, str]] = {
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
    "gate": {
        "gate_key": (
            "TRACKED DEBT, not an architectural exception: gate.py was being edited "
            "concurrently by another session when I9807 landed, the same reason "
            "arm_predictions_key was originally placed outside crucible.keys. Moving it "
            "is alpha-engine-config-I9807's own follow-up, filed at the same time."
        ),
    },
}


def _module_level_key_functions() -> list[tuple[str, str]]:
    """(module, function name) for every module-level `*_key`/`*_prefix` def
    under `crucible/`, excluding `crucible/keys.py` itself."""
    hits: list[tuple[str, str]] = []
    for path in sorted(_CRUCIBLE_ROOT.rglob("*.py")):
        if path == _CRUCIBLE_ROOT / "keys.py":
            continue
        module = path.relative_to(_CRUCIBLE_ROOT).with_suffix("").as_posix().replace("/", ".")
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and _NAME_RE.match(node.name):
                hits.append((module, node.name))
    return hits


class TestEveryKeyShapedFunctionIsAccountedFor:
    def test_every_hit_outside_keys_py_is_in_the_allowlist_with_a_reason(self) -> None:
        hits = _module_level_key_functions()
        assert hits, (
            "the AST walk found zero *_key/*_prefix functions outside crucible/keys.py, "
            "which is suspicious given known ones (crucible.gate.gate_key at least) — the "
            "walk itself is probably broken, not the codebase."
        )
        unaccounted = [
            f"{module}::{name}" for module, name in hits if name not in _ALLOWED.get(module, {})
        ]
        assert not unaccounted, (
            "a *_key/*_prefix function was added outside crucible/keys.py with no "
            "allowlist entry: " + ", ".join(unaccounted) + ". Either move it into "
            "crucible/keys.py (crucible/keys.py's own rule: every store key shape, in "
            "one place), or add it to _ALLOWED in this test with a stated reason."
        )

    @pytest.mark.parametrize(
        ("module", "name"),
        [(m, n) for m, entries in _ALLOWED.items() for n in entries],
    )
    def test_no_stale_allowlist_entry(self, module: str, name: str) -> None:
        """An entry naming a function that moved, was renamed, or was deleted is a
        stale exemption — silently permissive rather than caught."""
        matching = {n for m, n in _module_level_key_functions() if m == module}
        assert name in matching, (
            f"_ALLOWED names {module}::{name}, but no such module-level function exists "
            "any more. Remove the stale entry."
        )
