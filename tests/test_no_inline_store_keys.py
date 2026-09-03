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

**What this checks, precisely.** A call to one of `crucible.store.Store`'s
declared mutating or reading methods, or one of `RunContext`'s key-writing
methods, whose FIRST positional argument is a string constant or an f-string
(`ast.Constant` or `ast.JoinedStr`) is a key or prefix built inline at the
call site rather than by a `crucible.keys` function or constant. A call
passing a `Name`, `Attribute` or another `Call` as that argument —
`drift_input_key(...)`, `RUNS_ROOT`, a local variable already built from one
of those — passes, because the shape is owned somewhere else and this call
is only using it.

**The method set is DERIVED, not a hand-kept literal.** An earlier version
of this file hardcoded `{"get_bytes", "put_bytes", "list_keys", "exists"}`
and called it "exhaustive" — false the moment it shipped:
`crucible.store.Store.compare_and_swap` (the champion pointer's own write
path), `etag` and `assert_keys_bind_to_trading_days` were all invisible to
it, and so was `RunContext.record_output_cas`. `crucible/store.py` already
solves exactly this problem for exactly this reason — `Store.MUTATORS` /
`Store.READERS` are declared there, with the comment: "a guard can be
derived from it rather than from a literal list kept in step by hand — a
third mutator added without a line here would leave that guard silently
blind to it." This module reuses that declaration instead of re-inventing a
second hand-kept list beside it. `RunContext` carries no equivalent declared
constant, so `_CONTEXT_METHODS` below is read directly off the class by a
human rather than derived — named as a limitation, not claimed as
exhaustive.

**Scope, stated rather than suppressed.** This walks every `.py` file under
`crucible/` EXCEPT `crucible/keys.py` itself (the producer, not a call site)
and `crucible/gate.py`. `gate.py` is excluded by IDENTITY, not by a growing
allowlist: it is a single, named, structural boundary tied to a live,
concurrent PR under `alpha-engine-config-I9875` that owns adding
`gate_prefix(gate)` for the one remaining hit inside it —
`gate.py`'s OTHER hit (`store.list_keys("runs/smoke/")`) was fixed directly
in this PR (`runs_prefix("smoke")` already existed), leaving exactly one:
`store.list_keys(f"gates/{gate}/")`. `self_test_gate_py_hits_are_real`
proves that hit still exists, so an accidental widening of the exclusion (or
a stale one, once I9875 lands) is caught rather than silently made
permanent.
"""

from __future__ import annotations

import ast
from pathlib import Path

from crucible.store import Store

_CRUCIBLE_ROOT = Path(__file__).resolve().parent.parent / "crucible"

#: Derived from `Store`'s own declared partition (`crucible/store.py`'s
#: `MUTATORS`/`READERS`), not a second hand-kept list beside it — see the
#: module docstring for why a hand-kept version already missed
#: `compare_and_swap`, `etag` and `assert_keys_bind_to_trading_days`.
_STORE_METHODS = frozenset(Store.MUTATORS) | frozenset(Store.READERS)

#: `RunContext` (`crucible/runner.py`) has no `Store`-style declared
#: partition to derive this from, so these three are read directly off the
#: class: every method whose first parameter after `self` is a store key —
#: `record_input`, `record_output`, `record_output_cas`. NOT claimed
#: exhaustive the way `_STORE_METHODS` now is; a fourth key-writing method
#: added to `RunContext` without a line here is exactly the blind spot this
#: file exists to avoid reintroducing, and there is no declared constant on
#: `RunContext` yet to derive it from instead.
_CONTEXT_METHODS = frozenset({"record_input", "record_output", "record_output_cas"})

#: `crucible/keys.py` is the producer, not a call site — excluded by path
#: identity, exactly like `test_key_construction_placement.py` excludes it.
#: `crucible/gate.py` is excluded for the reason the module docstring gives:
#: a live, concurrent PR under `alpha-engine-config-I9875` owns its one
#: remaining hit. ONE path, by identity, not a pattern and not a name that
#: could silently match a second file later.
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
            "prose), it should not be arg0 to one of these methods at all."
        )

    def test_the_detector_fires_on_an_fstring_and_a_literal_but_not_a_name(
        self, tmp_path: Path
    ) -> None:
        """Calls the REAL `_hits_in` against a real file on disk — not a
        second, hand-copied reimplementation of its walk — so a future edit
        that breaks `_hits_in`, or removes the `ast.JoinedStr` arm (the exact
        shape I9852 fixed nine instances of), fails THIS test directly.
        Measured: a prior version of this test re-implemented the walk
        inline, and neutering `_hits_in` to `return []`, or deleting its
        `ast.JoinedStr` branch outright, left it — and the other two tests in
        this file — green.
        """
        source = (
            "def f(store):\n"
            "    day = '2026-08-28'\n"
            "    store.get_bytes(f'drift/{day}/input_features.json')\n"
            "    store.exists('champions/current.json')\n"
            "    store.list_keys(RUNS_ROOT)\n"
        )
        path = tmp_path / "synthetic_module.py"
        path.write_text(source, encoding="utf-8")

        hits = _hits_in(path)

        methods = [method for _, method, _ in hits]
        assert methods == ["get_bytes", "exists"], (
            "expected the f-string arg to get_bytes and the literal arg to exists to "
            f"fire, and the Name RUNS_ROOT passed to list_keys not to; got {methods}"
        )
        shapes = [shape for _, _, shape in hits]
        assert shapes == ["f-string", "literal 'champions/current.json'"]

    def test_gate_py_hits_are_real_not_a_stale_exclusion(self) -> None:
        """`gate.py` is excluded above by identity, with a stated reason:
        a live, concurrent PR (`alpha-engine-config-I9875`) owns its one
        remaining hit. This proves the exclusion is still covering a REAL
        hit rather than having gone stale once that PR lands — if `gate.py`
        stops carrying it, remove the exclusion from `_walk()` above instead
        of leaving a boundary that no longer excludes anything real."""
        hits = _hits_in(_GATE_PY)
        assert hits, (
            "crucible/gate.py carries no inline store-key literals any more — the "
            "concurrent PR named in this test's docstring must have landed. Remove the "
            "gate.py exclusion from this test's _walk() and let the main test cover it."
        )
