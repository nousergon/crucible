"""No store or manifest call site builds its own key. Only `crucible.keys` may.

Normative source: `crucible/keys.py`'s module docstring ("every store key
shape, in one place") and `alpha-engine-config-I9852`, which fixed a long
list of INSTANCES of one class — an inline literal restating a key or prefix
shape at the call site instead of calling the `crucible.keys` function (or
constant) that owns it.

`tests/test_key_construction_placement.py` guards the class from one side:
no `*_key`/`*_prefix` FUNCTION lives outside `crucible/keys.py` (or is
registered there, by name and with a reason, as an architectural exception).
It says so in its own docstring: a predicate over call-site string CONTENT
would be noisy, so it deliberately checks function definitions only, never
call sites. That leaves the other side of the class open — nothing stops a
caller from hardcoding `f"drift/{day}/input_features.json"` inline instead of
calling `crucible.keys.drift_input_key`, and I9852 fixed nine call sites that
were doing exactly that with zero detector catching any of them.

**What this checks, precisely.** A call to one of `crucible.store.Store`'s
declared mutating or reading methods, or one of `RunContext`'s key-writing
methods, whose FIRST positional argument is anything other than:

- a `Name` (a local already built by one of the shapes below, or a constant
  such as `RUNS_ROOT`),
- an `Attribute` (`keys.BOARD_CURRENT_KEY`, `self.key`, `published.wheel_key`),
- a `Call` whose callee is NAMED like a key function — `..._key` or
  `..._prefix`, the convention `test_key_construction_placement.py` already
  enforces, so every callee this accepts either lives in `crucible/keys.py`
  or is registered there by name with a reason.

Everything else is a hit: a string `Constant`, an f-string (`JoinedStr`), a
`BinOp` (`"gates/" + gate + "/"`), a `.format(...)` call, `os.path.join(...)`,
`%` formatting, `str(...)` wrapping, a `Subscript`, a conditional expression.
An ALLOWLIST OF SHAPES, not a denylist of syntaxes: the earlier version of
this file flagged `Constant` and `JoinedStr` only, and
`alpha-engine-config-I9899` measured that a `BinOp`, a `.format` and an
`os.path.join` at a store call site each returned ZERO hits — the exact
shapes the guard exists to stop, one syntax away from the two it knew. A
denylist is beaten by the next syntax; an allowlist of the three shapes a
correctly-written call site actually uses is not.

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
`crucible/` EXCEPT `crucible/keys.py` itself (the producer, not a call site).
`crucible/gate.py` was excluded here through `alpha-engine-config-I9875`
landing: it carried one remaining hit, `store.list_keys(f"gates/{gate}/")`,
held back only because a concurrent session owned the file while this
detector was written. `I9875` added `gate_prefix(gate)` to `crucible/keys.py`
and switched `gate.py:last_read` to call it, so `gate.py` is walked like
every other file below — no exclusion, and no allowlist to go stale later.
"""

from __future__ import annotations

import ast
import re
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
#: This is the only exclusion left; `crucible/gate.py`'s was removed once
#: `alpha-engine-config-I9875` fixed its one remaining hit.
_SELF_EXCLUDED = {_CRUCIBLE_ROOT / "keys.py"}

#: The naming convention every key-building function in this package commits
#: to, and the one `tests/test_key_construction_placement.py` enforces lives
#: in `crucible/keys.py` or is registered there by name. Kept textually
#: identical to that file's `_NAME_RE` on purpose: the two guards are two
#: sides of one class, and a callee this one accepts must be one that one
#: accounts for.
_KEY_CALLEE_RE = re.compile(r"^[a-z][a-z0-9_]*_(key|prefix)$")


def _callee_name(func: ast.expr) -> str | None:
    """`foo(...)` -> `foo`; `keys.foo(...)` / `release.foo(...)` -> `foo`;
    anything else (a call on a call, a subscript) -> None."""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _shape(arg: ast.expr) -> str | None:
    """`None` when ``arg`` is an accepted key shape; otherwise a short label
    naming the rejected shape, for the failure message."""
    if isinstance(arg, ast.Name | ast.Attribute):
        return None
    if isinstance(arg, ast.Call):
        callee = _callee_name(arg.func)
        if callee is not None and _KEY_CALLEE_RE.match(callee):
            return None
        return f"call to {callee or '<expr>'}(...)"
    if isinstance(arg, ast.JoinedStr):
        return "f-string"
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return f"literal {arg.value!r}"
    return type(arg).__name__


def _hits_in(path: Path) -> list[tuple[int, str, str]]:
    """(lineno, method, arg-shape) for every non-key-shaped key argument in ``path``."""
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
        shape = _shape(node.args[0])
        if shape is not None:
            found.append((node.lineno, method, shape))
    return found


def _walk() -> dict[Path, list[tuple[int, str, str]]]:
    results: dict[Path, list[tuple[int, str, str]]] = {}
    for path in sorted(_CRUCIBLE_ROOT.rglob("*.py")):
        if path in _SELF_EXCLUDED:
            continue
        hits = _hits_in(path)
        if hits:
            results[path] = hits
    return results


def _hits_for_source(tmp_path: Path, source: str) -> list[tuple[int, str, str]]:
    """Run the REAL `_hits_in` over ``source`` written to disk — never a
    second, hand-copied reimplementation of its walk."""
    path = tmp_path / "synthetic_module.py"
    path.write_text(source, encoding="utf-8")
    return _hits_in(path)


class TestNoInlineStoreKeyLiterals:
    def test_no_call_site_hardcodes_a_key_or_prefix(self) -> None:
        results = _walk()
        assert not results, (
            "a store or manifest call built its key at the call site instead of "
            "calling the crucible.keys function/constant that owns that shape: "
            + "; ".join(
                f"{path.relative_to(_CRUCIBLE_ROOT.parent)}:"
                + ",".join(f"{lineno}({method}:{shape})" for lineno, method, shape in hits)
                for path, hits in results.items()
            )
            + ". Move the shape into crucible/keys.py (a *_key/*_prefix function or a "
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
        hits = _hits_for_source(tmp_path, source)

        methods = [method for _, method, _ in hits]
        assert methods == ["get_bytes", "exists"], (
            "expected the f-string arg to get_bytes and the literal arg to exists to "
            f"fire, and the Name RUNS_ROOT passed to list_keys not to; got {methods}"
        )
        shapes = [shape for _, _, shape in hits]
        assert shapes == ["f-string", "literal 'champions/current.json'"]


class TestTheAllowlistIsOfShapesNotOfSyntaxes:
    """alpha-engine-config-I9899: each of these three synthetic sites returned
    ZERO hits under the `Constant`/`JoinedStr` denylist. Each must now produce
    exactly one."""

    def test_string_concatenation_is_a_hit(self, tmp_path: Path) -> None:
        hits = _hits_for_source(
            tmp_path, "def f(store, gate):\n    store.list_keys('gates/' + gate + '/')\n"
        )
        assert [(m, s) for _, m, s in hits] == [("list_keys", "BinOp")]

    def test_str_format_is_a_hit(self, tmp_path: Path) -> None:
        hits = _hits_for_source(
            tmp_path, "def f(store, gate):\n    store.list_keys('gates/{}/'.format(gate))\n"
        )
        assert [(m, s) for _, m, s in hits] == [("list_keys", "call to format(...)")]

    def test_os_path_join_is_a_hit(self, tmp_path: Path) -> None:
        hits = _hits_for_source(
            tmp_path,
            "import os\ndef f(store, gate):\n    store.list_keys(os.path.join('gates', gate))\n",
        )
        assert [(m, s) for _, m, s in hits] == [("list_keys", "call to join(...)")]

    def test_percent_formatting_and_str_wrapping_are_hits(self, tmp_path: Path) -> None:
        hits = _hits_for_source(
            tmp_path,
            "def f(store, gate, p):\n"
            "    store.list_keys('gates/%s/' % gate)\n"
            "    store.get_bytes(str(p))\n",
        )
        assert [(m, s) for _, m, s in hits] == [
            ("list_keys", "BinOp"),
            ("get_bytes", "call to str(...)"),
        ]

    def test_a_key_function_call_a_name_and_an_attribute_are_not_hits(
        self, tmp_path: Path
    ) -> None:
        """The three shapes a correctly-written call site uses. `release.
        wheel_key(...)` is accepted by NAME — `test_key_construction_placement.py`
        is what guarantees that every `*_key` callee is either in
        `crucible/keys.py` or registered there with a reason."""
        hits = _hits_for_source(
            tmp_path,
            "def f(store, ctx, gate, day, published):\n"
            "    store.list_keys(gate_prefix(gate))\n"
            "    store.get_bytes(keys.gate_key(gate, day))\n"
            "    store.exists(release.wheel_key(sha))\n"
            "    ctx.record_output(published.wheel_key, b'')\n"
            "    ctx.record_input(BOARD_CURRENT_KEY, b'')\n",
        )
        assert hits == []

    def test_a_call_to_a_non_key_function_is_a_hit_even_when_it_returns_a_key(
        self, tmp_path: Path
    ) -> None:
        """`build_it(gate)` may well return a correct key — but the shape is
        then owned by a function this package's naming convention does not
        mark as a key builder, which is the placement guard's blind spot
        restated at the call site."""
        hits = _hits_for_source(
            tmp_path, "def f(store, gate):\n    store.list_keys(build_it(gate))\n"
        )
        assert [(m, s) for _, m, s in hits] == [("list_keys", "call to build_it(...)")]

    def test_the_callee_convention_matches_the_placement_guards(self) -> None:
        """Two guards, one class: the callee shapes this file accepts must be
        exactly the definition shapes `test_key_construction_placement.py`
        accounts for, or a function could satisfy one guard and evade the
        other."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "test_key_construction_placement",
            Path(__file__).with_name("test_key_construction_placement.py"),
        )
        assert spec is not None and spec.loader is not None
        placement = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(placement)
        assert _KEY_CALLEE_RE.pattern == placement._NAME_RE.pattern
