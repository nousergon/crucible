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
Those, and only those, go in `_KNOWN_ARCHITECTURAL_EXCEPTIONS` below —
there is deliberately no companion "not moved yet" debt registry: a
suppression list is a rule change (`AGENTS.md` rule 4, "no suppression
collections, at all," names exactly two exemptions and neither is a file
list) and shipping one inside this test would make that call by default
instead of putting it to Brian. So a hit that is a real key function
simply not moved yet is not accounted for here — it fails this test, and
the fix is to move it into `crucible.keys` in the same change (see
`crucible.gate.gate_key`, which alpha-engine-config-I9807's review moved
this way rather than leaving as debt).

`_KNOWN_ARCHITECTURAL_EXCEPTIONS` is named `_KNOWN_*` **on purpose**,
deliberately matching `tests/test_no_suppressions.py::FORBIDDEN`'s
`_KNOWN_` pattern, so that scanner sees this one collection and sanctions
it explicitly by NAME (not by file, and not the whole `_KNOWN_` pattern in
this file — see that test's own exemption and its self-test) rather than
being blind to it because its name never matched anything checked for,
which is exactly the defect this collection would otherwise be: a
suppression list evading the suppression scanner.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*_(key|prefix)$")

_CRUCIBLE_ROOT = Path(__file__).resolve().parent.parent / "crucible"

#: module (dotted, relative to `crucible/`) -> {function name: reason it
#: CORRECTLY lives outside `crucible.keys` today}. This is an architectural
#: registry, not a debt list — deliberately the ONLY registry in this file
#: (see the module docstring: a companion "not moved yet" debt collection
#: would be a suppression list, and shipping one is a rule change that
#: belongs to Brian, not a review call). Every entry names a real reason a
#: function is built from something local to its own module, not merely
#: that nobody has moved it yet; each is expected to hold, though a future
#: refactor could still dissolve one (a `crucible.keys` that took a
#: validator callback would fold the `release.py` cluster in, for
#: instance) — "correct today" is checked, not "permanent by decree".
#:
#: This identifier — `_KNOWN_ARCHITECTURAL_EXCEPTIONS` — is spelled with
#: that specific leading prefix ON PURPOSE, deliberately matching
#: `tests/test_no_suppressions.py::FORBIDDEN`'s suppression-collection
#: pattern, so that scanner sees this collection and sanctions it
#: explicitly, BY THIS EXACT NAME (not by file — see that test's own
#: exemption and its self-test) rather than being blind to it because its
#: name never matched anything it checks for, which is exactly the defect
#: this collection would otherwise BE: a suppression list evading the
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
        "published_wheel_key": (
            "not a key SHAPE at all — it takes a Store, READS that release's "
            "release.json and returns the shape wheel_key_for (already registered "
            "here) builds from the recorded wheel_filename. crucible.keys is pure "
            "string construction with no I/O and no imports from the release domain; "
            "moving this there would make the key module depend on Store and on "
            "parse_release_record, which is the inversion this whole test exists to "
            "prevent. Permanent, not 'not moved yet'."
        ),
        # Added by the alpha-engine-config-I9917 residual fix, which made
        # resolve_release/pin read the wheel filename from release.json
        # instead of deriving it from the sha. Cited in a comment, not in the
        # reason string: test_no_stale_tracker_literals.py permits a tracker
        # number in prose only.
        "release_json_key": "built on release_prefix; same reason as release_prefix.",
        "provenance_key": "built on release_prefix; same reason as release_prefix.",
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
    return name in _KNOWN_ARCHITECTURAL_EXCEPTIONS.get(module, {})


class TestEveryKeyShapedFunctionIsAccountedFor:
    def test_every_hit_outside_keys_py_is_accounted_for_with_a_reason(self) -> None:
        hits = _module_level_key_functions()
        assert hits, (
            "the AST walk found zero *_key/*_prefix functions outside crucible/keys.py, "
            "which is suspicious given known ones (crucible.release.wheel_key at least) — "
            "the walk itself is probably broken, not the codebase."
        )
        unaccounted = [
            f"{module}::{name}" for module, name in hits if not _accounted_for(module, name)
        ]
        assert not unaccounted, (
            "a *_key/*_prefix function was added outside crucible/keys.py with no "
            "registry entry: " + ", ".join(unaccounted) + ". Move it into "
            "crucible/keys.py (crucible/keys.py's own rule: every store key shape, in "
            "one place) in the same change, or add it to _KNOWN_ARCHITECTURAL_EXCEPTIONS "
            "if it genuinely, permanently belongs outside crucible.keys — there is no "
            "'not moved yet' registry; that would be a suppression list (AGENTS.md rule 4) "
            "and shipping one is Brian's call, not this test's."
        )

    @pytest.mark.parametrize(
        ("registry_name", "module", "name"),
        [
            ("_KNOWN_ARCHITECTURAL_EXCEPTIONS", m, n)
            for m, entries in _KNOWN_ARCHITECTURAL_EXCEPTIONS.items()
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
