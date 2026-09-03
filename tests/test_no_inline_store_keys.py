"""No literal store-key string reaches a store or manifest call site.

Normative source: `crucible/keys.py`'s module docstring ("every store key
shape, in one place") and `alpha-engine-config-I9852`, which fixed a long
list of INSTANCES of one class — an inline literal restating a key or prefix
shape at the call site instead of calling the `crucible.keys` function (or
constant) that owns it.

`tests/test_key_construction_placement.py` guards the class from one side:
no `*_key`/`*_prefix` FUNCTION lives outside `crucible/keys.py`. It says so
in its own docstring: a predicate over call-site string CONTENT would be
noisy, so it deliberately checks function definitions only, never call
sites. That leaves the other side of the class wide open — nothing stops a
caller from hardcoding `f"drift/{day}/input_features.json"` inline instead
of calling `crucible.keys.drift_input_key`, and I9852 fixed nine call sites
that were doing exactly that with zero detector catching any of them.

**What this checks, precisely.** A `Store.get_bytes`/`put_bytes`/`list_keys`/
`exists` or `RunContext.record_output`/`record_input` call whose FIRST
positional argument is a string constant or an f-string (`ast.Constant` or
`ast.JoinedStr`) is a key or prefix built inline at the call site rather than
by a `crucible.keys` function or constant. A call passing a `Name`,
`Attribute` or another `Call` as that argument — `drift_input_key(...)`,
`RUNS_ROOT`, a local variable already built from one of those — passes,
because the shape is owned somewhere else and this call is only using it.

**Scope, stated rather than suppressed.** This walks every `.py` file under
`crucible/` EXCEPT `crucible/keys.py` itself (the producer, not a call site)
and `crucible/gate.py`. `gate.py` is excluded by IDENTITY, not by a growing
allowlist: it is a single, named, structural boundary tied to a live,
concurrent PR under `alpha-engine-config-I9852` that owns moving `gate_key`
into `crucible.keys` (this PR's own part of I9852 is scoped away from
`gate.py` for the same reason — two PRs editing the same file). `gate.py`
carries two known hits today — `gate.py:470` (`store.list_keys("runs/smoke/")`,
which `crucible.keys.runs_prefix("smoke")` already covers) and `gate.py:1124`
(`store.list_keys(f"gates/{gate}/")`, which wants a `gate_prefix(gate)`
function alongside `gate_key`'s move) — both reported in this PR's body
rather than fixed here, and both will still be reachable by anyone who reads
this module's own exclusion list: `self_test_gate_py_hits_are_real` proves
they exist, so an accidental widening of the exclusion (or a stale one, once
`gate.py`'s PR lands) is caught rather than silently made permanent.
"""

from __future__ import annotations

import ast
from pathlib import Path

_CRUCIBLE_ROOT = Path(__file__).resolve().parent.parent / "crucible"

#: The store/context methods this test polices. Exhaustive: these five are
#: every method through which a key or prefix reaches the store layer.
_STORE_METHODS = frozenset({"get_bytes", "put_bytes", "list_keys", "exists"})
_CONTEXT_METHODS = frozenset({"record_output", "record_input"})

#: `crucible/keys.py` is the producer, not a call site — excluded by path
#: identity, exactly like `test_key_construction_placement.py` excludes it.
#: `crucible/gate.py` is excluded for the reason the module docstring gives:
#: a live, concurrent, same-issue PR owns that file today. ONE path, by
#: identity, not a pattern and not a name that could silently match a second
#: file later.
_SELF_EXCLUDED = {_CRUCIBLE_ROOT / "keys.py"}
_GATE_PY = _CRUCIBLE_ROOT / "gate.py"


def _hits_in(path: Path) -> list[tuple[int, str, str]]:
    """(lineno, method, arg-shape) for every inline-literal hit in ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        method = node.func.attr
        if method not in _STORE_METHODS and method not in _CONTEXT_METHODS:
            continue
        if not node.args:
            continue
        arg0 = node.args[0]
        if isinstance(arg0, ast.JoinedStr):
            found.append((node.lineno, method, "f-string"))
        elif isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
            found.append((node.lineno, method, f"literal {arg0.value!r}"))
    return found


def _walk(*, exclude_gate: bool) -> dict[Path, list[tuple[int, str, str]]]:
    results: dict[Path, list[tuple[int, str, str]]] = {}
    for path in sorted(_CRUCIBLE_ROOT.rglob("*.py")):
        if path in _SELF_EXCLUDED:
            continue
        if exclude_gate and path == _GATE_PY:
            continue
        hits = _hits_in(path)
        if hits:
            results[path] = hits
    return results


class TestNoInlineStoreKeyLiterals:
    def test_no_call_site_outside_gate_py_hardcodes_a_key_or_prefix(self) -> None:
        results = _walk(exclude_gate=True)
        assert not results, (
            "a store or manifest call passed a literal string or f-string as its key "
            "instead of calling the crucible.keys function/constant that owns that "
            "shape: "
            + "; ".join(
                f"{path.relative_to(_CRUCIBLE_ROOT.parent)}:"
                + ",".join(f"{lineno}({method}:{shape})" for lineno, method, shape in hits)
                for path, hits in results.items()
            )
            + ". Move the literal into crucible/keys.py (a *_key/*_prefix function or a "
            "ROOT constant) and call it from here — or, if it is genuinely not a store "
            "key (a dedup identity, human-readable message text, unresolved template "
            "prose), it should not be arg0 to one of these five methods at all."
        )

    def test_the_detector_actually_fires_the_self_test_that_shows_it_working(self) -> None:
        """§ Test discipline: "give every guard a self-test that shows it
        firing." A tiny synthetic AST, not a real file, so this cannot be
        made to pass by fixing the codebase out from under it."""
        source = (
            "def f(store):\n"
            "    day = '2026-08-28'\n"
            "    store.get_bytes(f'drift/{day}/input_features.json')\n"
            "    store.exists('champions/current.json')\n"
            "    store.list_keys(RUNS_ROOT)\n"  # a Name — must NOT fire
        )
        tree = ast.parse(source)
        hits = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in _STORE_METHODS or not node.args:
                continue
            arg0 = node.args[0]
            if isinstance(arg0, ast.JoinedStr) or (
                isinstance(arg0, ast.Constant) and isinstance(arg0.value, str)
            ):
                hits.append(node.func.attr)
        assert hits == ["get_bytes", "exists"], (
            f"expected the f-string and the literal to fire and the Name (RUNS_ROOT) not "
            f"to, got {hits}"
        )

    def test_gate_py_hits_are_real_not_a_stale_exclusion(self) -> None:
        """`gate.py` is excluded above by identity, with a stated reason:
        a live, concurrent PR under the same issue owns that file. This
        proves the exclusion is still covering REAL hits rather than having
        gone stale once that PR lands — if `gate.py` stops carrying these,
        remove it from `_SELF_EXCLUDED`'s sibling check above instead of
        leaving a boundary that no longer excludes anything real."""
        hits = _hits_in(_GATE_PY)
        assert hits, (
            "crucible/gate.py carries no inline store-key literals any more — the "
            "concurrent PR that owns it must have landed. Remove the gate.py exclusion "
            "from this test's _walk() and let the main test cover it."
        )
