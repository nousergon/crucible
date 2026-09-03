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
methods, whose KEY ARGUMENT — the first positional argument, or the `key=` /
`prefix=` keyword when there is no positional — resolves to anything other
than:

- an `Attribute` (`keys.BOARD_CURRENT_KEY`, `self.key`, `published.wheel_key`),
- a `Call` whose callee is NAMED like a key function — `..._key` or
  `..._prefix`, the convention `test_key_construction_placement.py` already
  enforces, so every callee this accepts either lives in `crucible/keys.py`
  or is registered there by name with a reason,
- a `Name` that RESOLVES to one of those: a function parameter, a name bound
  more than once in its scope, or a name never bound in the function or the
  module (an import, a loop variable, a `with ... as`). A name bound exactly
  ONCE in the enclosing function — or, failing that, once at module level —
  is followed to the right-hand side of that binding and graded by the same
  rules, recursively. `k = f"drift/{d}"` one line above `store.get_bytes(k)`
  is therefore a hit, reported as `k = f-string`.

Everything else is a hit: a string `Constant`, an f-string (`JoinedStr`), a
`BinOp` (`"gates/" + gate + "/"`), a `.format(...)` call, `os.path.join(...)`,
`%` formatting, `str(...)` wrapping, a `Subscript`, a conditional expression.
An ALLOWLIST OF SHAPES, not a denylist of syntaxes: the earlier version of
this file flagged `Constant` and `JoinedStr` only, and
`alpha-engine-config-I9899` measured that a `BinOp`, a `.format` and an
`os.path.join` at a store call site each returned ZERO hits — the exact
shapes the guard exists to stop, one syntax away from the two it knew. A
denylist is beaten by the next syntax; an allowlist of the shapes a
correctly-written call site actually uses is not.

**Residuals, named rather than claimed away** (round-1 review of I9899
found the first version accepted every bare `Name` unconditionally and
called that complete):

- A name bound MORE THAN ONCE in its scope (`k = a; if x: k = b`) is
  accepted: which binding reaches the call is control flow this scanner
  does not model. Both bindings are still visible to a reader.
- A name bound through a tuple target, `for`, `with ... as`, `+=` or a
  walrus is treated as unknowable and accepted.
- A key built in ONE function and passed as an ARGUMENT to another that
  calls the store is accepted at the callee (a parameter) and NOT graded at
  the caller (the caller's call is not a store method). That is the
  placement guard's side of the class: a function that builds a key and
  hands it on is a key function, and belongs in `crucible/keys.py`.
- Class attributes (`self.key`) and module-level names bound more than
  once are accepted as `Attribute` / multi-bound.

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
from dataclasses import dataclass, field
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

#: The keyword names under which those methods take their key when it is not
#: passed positionally. `store.get_bytes(key="...")` scored zero hits before
#: round-1 review of alpha-engine-config-I9899.
_KEY_KEYWORDS = frozenset({"key", "prefix"})

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
#: accounts for. A bare `key` / `prefix` — a METHOD such as
#: `FeatureLayerSource.key(trading_day)` — is the class-attribute form of the
#: same convention and is covered by the same registry (round 2).
_KEY_CALLEE_RE = re.compile(r"^(?:[a-z][a-z0-9_]*_)?(key|prefix)$")

#: How many single-assignment hops a `Name` is followed through before the
#: chain is accepted as too deep to grade. Three is the deepest chain this
#: package carries; eight leaves room without letting a cycle spin.
_MAX_RESOLUTION_DEPTH = 8

_SCOPE_BOUNDARIES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


@dataclass
class _Scope:
    """One function's (or the module's) bindings, collected WITHOUT
    descending into nested functions, lambdas or classes — those are their
    own scopes and get their own pass."""

    params: frozenset[str]
    #: name -> every binding seen; `None` marks a binding whose value cannot
    #: be graded (tuple target, `for`, `with ... as`, `+=`, walrus).
    bindings: dict[str, list[ast.expr | None]] = field(default_factory=dict)
    calls: list[ast.Call] = field(default_factory=list)
    parent: _Scope | None = None

    def _bind(self, name: str, value: ast.expr | None) -> None:
        self.bindings.setdefault(name, []).append(value)

    def _bind_target(self, target: ast.expr, value: ast.expr | None) -> None:
        if isinstance(target, ast.Name):
            self._bind(target.id, value)
        elif isinstance(target, ast.Tuple | ast.List):
            for elt in target.elts:
                self._bind_target(elt, None)
        elif isinstance(target, ast.Starred):
            self._bind_target(target.value, None)


def _params_of(node: ast.AST) -> frozenset[str]:
    if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
        return frozenset()
    a = node.args
    names = [p.arg for p in (*a.posonlyargs, *a.args, *a.kwonlyargs)]
    if a.vararg:
        names.append(a.vararg.arg)
    if a.kwarg:
        names.append(a.kwarg.arg)
    return frozenset(names)


def _collect_scope(owner: ast.AST, parent: _Scope | None) -> _Scope:
    scope = _Scope(params=_params_of(owner), parent=parent)
    if isinstance(owner, ast.Lambda):
        roots: list[ast.AST] = [owner.body]
    elif isinstance(owner, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        roots = list(owner.body)
    else:
        roots = []
    stack = list(roots)
    while stack:
        node = stack.pop()
        if isinstance(node, _SCOPE_BOUNDARIES):
            continue
        if isinstance(node, ast.Assign):
            for target in node.targets:
                scope._bind_target(target, node.value)
        elif isinstance(node, ast.AnnAssign):
            scope._bind_target(node.target, node.value)
        elif isinstance(node, ast.AugAssign):
            scope._bind_target(node.target, None)
        elif isinstance(node, ast.NamedExpr):
            scope._bind_target(node.target, node.value)
        elif isinstance(node, ast.For | ast.AsyncFor | ast.comprehension):
            scope._bind_target(node.target, None)
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            scope._bind_target(node.optional_vars, None)
        elif isinstance(node, ast.Call):
            scope.calls.append(node)
        stack.extend(ast.iter_child_nodes(node))
    return scope


def _callee_name(func: ast.expr) -> str | None:
    """`foo(...)` -> `foo`; `keys.foo(...)` / `release.foo(...)` -> `foo`;
    anything else (a call on a call, a subscript) -> None."""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _lookup(scope: _Scope, name: str) -> tuple[str, ast.expr | None]:
    """How ``name`` reaches this scope: ('param', None), ('single', value),
    ('multi', None), ('unknowable', None) or ('unbound', None)."""
    current: _Scope | None = scope
    while current is not None:
        if name in current.params:
            return "param", None
        if name in current.bindings:
            values = current.bindings[name]
            if len(values) != 1:
                return "multi", None
            return ("single", values[0]) if values[0] is not None else ("unknowable", None)
        current = current.parent
    return "unbound", None


def _shape(arg: ast.expr, scope: _Scope, depth: int = 0) -> str | None:
    """`None` when ``arg`` is (or resolves to) an accepted key shape;
    otherwise a short label naming the rejected shape, for the failure
    message."""
    if isinstance(arg, ast.Attribute):
        return None
    if isinstance(arg, ast.Name):
        if depth >= _MAX_RESOLUTION_DEPTH:
            return None
        how, value = _lookup(scope, arg.id)
        if how != "single" or value is None:
            return None
        resolved = _shape(value, scope, depth + 1)
        return None if resolved is None else f"{arg.id} = {resolved}"
    if isinstance(arg, ast.Call):
        callee = _callee_name(arg.func)
        if callee is not None and _KEY_CALLEE_RE.match(callee):
            return None
        return f"call to {callee or '<expr>'}(...)"
    if isinstance(arg, ast.IfExp):
        # A choice between two keys is graded on BOTH arms: `POINTER_KEY if
        # target == "current" else TRADER_PIN_KEY` is two accepted shapes and
        # is accepted; `'a/b' if x else 'c/d'` is two literals and is a hit,
        # labelled with the first arm that failed.
        for arm in (arg.body, arg.orelse):
            resolved = _shape(arm, scope, depth + 1)
            if resolved is not None:
                return f"IfExp[{resolved}]"
        return None
    if isinstance(arg, ast.BoolOp):
        # `a or b` as a key: same rule, every operand must be an accepted shape.
        for operand in arg.values:
            resolved = _shape(operand, scope, depth + 1)
            if resolved is not None:
                return f"BoolOp[{resolved}]"
        return None
    if isinstance(arg, ast.JoinedStr):
        return "f-string"
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return f"literal {arg.value!r}"
    return type(arg).__name__


def _key_argument(node: ast.Call) -> ast.expr | None:
    if node.args:
        return node.args[0]
    for keyword in node.keywords:
        if keyword.arg in _KEY_KEYWORDS:
            return keyword.value
    return None


def _hits_in(path: Path) -> list[tuple[int, str, str]]:
    """(lineno, method, arg-shape) for every non-key-shaped key argument in ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[tuple[int, str, str]] = []
    # Every scope in the file, each with its lexical parent: the module, then
    # every def / lambda / class at any depth.
    pending: list[tuple[ast.AST, _Scope | None]] = [(tree, None)]
    while pending:
        owner, parent = pending.pop()
        scope = _collect_scope(owner, parent)
        for node in scope.calls:
            if not isinstance(node.func, ast.Attribute):
                continue
            method = node.func.attr
            if method not in _STORE_METHODS and method not in _CONTEXT_METHODS:
                continue
            key = _key_argument(node)
            if key is None:
                continue
            shape = _shape(key, scope)
            if shape is not None:
                found.append((node.lineno, method, shape))
        # Nested scopes, found by a walk that stops at scope boundaries so
        # each is visited exactly once, with THIS scope as its parent.
        stack: list[ast.AST] = list(ast.iter_child_nodes(owner))
        while stack:
            child = stack.pop()
            if isinstance(child, _SCOPE_BOUNDARIES):
                pending.append((child, scope))
                continue
            stack.extend(ast.iter_child_nodes(child))
    return sorted(found)


def _walk() -> dict[Path, list[tuple[int, str, str]]]:
    results: dict[Path, list[tuple[int, str, str]]] = {}
    for path in sorted(_CRUCIBLE_ROOT.rglob("*.py")):
        if path in _SELF_EXCLUDED:
            continue
        hits = _hits_in(path)
        if hits:
            results[path] = hits
    return results


def _hits_for_source(tmp_path: Path, source: str) -> list[tuple[str, str]]:
    """Run the REAL `_hits_in` over ``source`` written to disk — never a
    second, hand-copied reimplementation of its walk — and return the
    (method, shape) pairs in source order."""
    path = tmp_path / "synthetic_module.py"
    path.write_text(source, encoding="utf-8")
    return [(method, shape) for _, method, shape in _hits_in(path)]


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
            "prose), it should not be the key argument to one of these methods at all."
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
        assert _hits_for_source(tmp_path, source) == [
            ("get_bytes", "f-string"),
            ("exists", "literal 'champions/current.json'"),
        ]


class TestTheAllowlistIsOfShapesNotOfSyntaxes:
    """alpha-engine-config-I9899: each of these three synthetic sites returned
    ZERO hits under the `Constant`/`JoinedStr` denylist. Each must now produce
    exactly one."""

    def test_string_concatenation_is_a_hit(self, tmp_path: Path) -> None:
        source = "def f(store, gate):\n    store.list_keys('gates/' + gate + '/')\n"
        assert _hits_for_source(tmp_path, source) == [("list_keys", "BinOp")]

    def test_str_format_is_a_hit(self, tmp_path: Path) -> None:
        source = "def f(store, gate):\n    store.list_keys('gates/{}/'.format(gate))\n"
        assert _hits_for_source(tmp_path, source) == [("list_keys", "call to format(...)")]

    def test_os_path_join_is_a_hit(self, tmp_path: Path) -> None:
        source = (
            "import os\ndef f(store, gate):\n    store.list_keys(os.path.join('gates', gate))\n"
        )
        assert _hits_for_source(tmp_path, source) == [("list_keys", "call to join(...)")]

    def test_percent_formatting_and_str_wrapping_are_hits(self, tmp_path: Path) -> None:
        source = (
            "def f(store, gate, p):\n"
            "    store.list_keys('gates/%s/' % gate)\n"
            "    store.get_bytes(str(p))\n"
        )
        assert _hits_for_source(tmp_path, source) == [
            ("list_keys", "BinOp"),
            ("get_bytes", "call to str(...)"),
        ]

    def test_a_key_function_call_a_name_and_an_attribute_are_not_hits(self, tmp_path: Path) -> None:
        """The shapes a correctly-written call site uses. `release.
        wheel_key(...)` is accepted by NAME — `test_key_construction_placement.py`
        is what guarantees that every `*_key` callee is either in
        `crucible/keys.py` or registered there with a reason."""
        source = (
            "def f(store, ctx, gate, day, published):\n"
            "    store.list_keys(gate_prefix(gate))\n"
            "    store.get_bytes(keys.gate_key(gate, day))\n"
            "    store.exists(release.wheel_key(sha))\n"
            "    ctx.record_output(published.wheel_key, b'')\n"
            "    ctx.record_input(BOARD_CURRENT_KEY, b'')\n"
        )
        assert _hits_for_source(tmp_path, source) == []

    def test_a_call_to_a_non_key_function_is_a_hit_even_when_it_returns_a_key(
        self, tmp_path: Path
    ) -> None:
        """`build_it(gate)` may well return a correct key — but the shape is
        then owned by a function this package's naming convention does not
        mark as a key builder, which is the placement guard's blind spot
        restated at the call site."""
        source = "def f(store, gate):\n    store.list_keys(build_it(gate))\n"
        assert _hits_for_source(tmp_path, source) == [("list_keys", "call to build_it(...)")]

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


class TestANameIsResolvedNotTrusted:
    """Round-1 review of alpha-engine-config-I9899: accepting every bare
    `Name` was a total launder — `k = f'drift/{d}'` one line above
    `store.get_bytes(k)` scored zero — and a keyword-passed key was never
    inspected at all."""

    def test_a_keyword_passed_literal_is_a_hit(self, tmp_path: Path) -> None:
        source = (
            "def f(store):\n"
            "    store.get_bytes(key='champions/current.json')\n"
            "    store.list_keys(prefix=f'gates/{g}/')\n"
        )
        assert _hits_for_source(tmp_path, source) == [
            ("get_bytes", "literal 'champions/current.json'"),
            ("list_keys", "f-string"),
        ]

    def test_a_local_bound_once_to_an_fstring_is_a_hit_through_the_name(
        self, tmp_path: Path
    ) -> None:
        source = "def f(store, d):\n    k = f'drift/{d}/x.json'\n    store.get_bytes(k)\n"
        assert _hits_for_source(tmp_path, source) == [("get_bytes", "k = f-string")]

    def test_a_chain_of_single_bindings_is_followed(self, tmp_path: Path) -> None:
        source = (
            "def f(store, gate):\n"
            "    a = 'gates/' + gate + '/'\n"
            "    b = a\n"
            "    store.list_keys(b)\n"
        )
        assert _hits_for_source(tmp_path, source) == [("list_keys", "b = a = BinOp")]

    def test_a_conditional_expression_is_a_hit(self, tmp_path: Path) -> None:
        source = "def f(store, x):\n    k = 'a/b' if x else 'c/d'\n    store.exists(k)\n"
        assert _hits_for_source(tmp_path, source) == [("exists", "k = IfExp[literal 'a/b']")]

    def test_a_conditional_between_two_accepted_shapes_is_accepted(self, tmp_path: Path) -> None:
        """`POINTER_KEY if target == "current" else TRADER_PIN_KEY`
        (`crucible/release.py::pin`): both arms are names this scope never
        binds, so both are accepted and so is the choice."""
        source = (
            "def f(store, target):\n"
            "    key = POINTER_KEY if target == 'current' else TRADER_PIN_KEY\n"
            "    store.etag(key)\n"
            "    store.exists(keys.a_key(target) or keys.b_key(target))\n"
        )
        assert _hits_for_source(tmp_path, source) == []

    def test_a_local_bound_once_to_a_key_function_is_accepted(self, tmp_path: Path) -> None:
        source = "def f(store, gate):\n    k = gate_prefix(gate)\n    store.list_keys(k)\n"
        assert _hits_for_source(tmp_path, source) == []

    def test_a_parameter_is_accepted(self, tmp_path: Path) -> None:
        source = "def f(store, key):\n    store.get_bytes(key)\n"
        assert _hits_for_source(tmp_path, source) == []

    def test_a_name_bound_twice_is_accepted_and_that_is_the_named_residual(
        self, tmp_path: Path
    ) -> None:
        source = (
            "def f(store, x):\n"
            "    k = 'a/b'\n"
            "    if x:\n"
            "        k = gate_prefix(x)\n"
            "    store.exists(k)\n"
        )
        assert _hits_for_source(tmp_path, source) == []

    def test_a_loop_variable_and_a_with_target_are_accepted(self, tmp_path: Path) -> None:
        source = (
            "def f(store, keys, opener):\n"
            "    for k in keys:\n"
            "        store.exists(k)\n"
            "    with opener() as w:\n"
            "        store.exists(w)\n"
        )
        assert _hits_for_source(tmp_path, source) == []

    def test_a_module_level_literal_used_inside_a_function_is_a_hit(self, tmp_path: Path) -> None:
        """Module scope is the fallback for a name the function never binds:
        `_KEY = "champions/x.json"` at the top of the file is the inline
        literal moved up forty lines, not a key owned by `crucible.keys`."""
        source = "_KEY = 'champions/x.json'\ndef f(store):\n    store.get_bytes(_KEY)\n"
        assert _hits_for_source(tmp_path, source) == [
            ("get_bytes", "_KEY = literal 'champions/x.json'")
        ]

    def test_a_nested_function_has_its_own_scope(self, tmp_path: Path) -> None:
        """The inner `k` shadows the outer: the inner call resolves to the
        inner binding (a hit), and the outer call to the outer (accepted)."""
        source = (
            "def outer(store, g):\n"
            "    k = gate_prefix(g)\n"
            "    def inner():\n"
            "        k = 'gates/x/'\n"
            "        store.list_keys(k)\n"
            "    store.list_keys(k)\n"
        )
        assert _hits_for_source(tmp_path, source) == [("list_keys", "k = literal 'gates/x/'")]
