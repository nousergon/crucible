"""No acceptance clause hardcodes a tracker issue number in its failure message.

Normative source: alpha-engine-config-I9839. Every clause in this directory
used to fail with a literal `alpha-engine-config-I9757` baked into `_unmet`,
regardless of which phase the clause actually belonged to — so a phase-2
clause and a phase-3 clause both pointed a reader at phase 1, and stayed
pointed at it after phase 1 closed (2026-09-02T01:31Z). `crucible.gate.PHASES`
is the single source of truth for which tracker issue owns which phase; a
clause message is fixed by deriving the pointer from it, not by hand-editing
the string at each of the (currently four) call sites — that is the contract
restated once per call site, which is how this drifted the first time.

This test is the backstop: it fails if a hardcoded `I<digits>` tracker
literal is ever reintroduced into a `pytest.fail(...)` message anywhere under
`tests/acceptance/`, so the class recurring is a red test rather than a thing
a reviewer happens to notice.

**Not a text grep.** A grep for `I9757` would go clean the moment the literal
is replaced by a *different* wrong number (a phase-3 clause hardcoding
phase 2's issue, say) — the exact shape of this defect, one number over. This
walks the AST of every function whose body reaches a `pytest.fail(...)` call
and refuses ANY hardcoded `I<digits>` string literal in that function's own
body (its docstring excluded, since a clause has to be able to name its own
history in prose) — the only thing a fixed clause may do is derive the
pointer through `crucible.gate.PHASES`.

**Fail closed.** A file this scan cannot parse is a finding, not a skip: a
scanner that goes quiet on the one file it cannot read is indistinguishable
from a clean tree.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: This file is exempt for the same reason `tests/test_no_suppressions.py`
#: exempts itself: it must contain the pattern it searches for, in its own
#: docstring and comments, in order to explain what it forbids.
SELF = Path(__file__).resolve()

#: A tracker issue number: `I` followed by four-or-more digits, as it appears
#: in `alpha-engine-config-I9757` etc. Four digits, not `\d+`, so this does
#: not fire on an unrelated `I9` or a two-digit identifier elsewhere in the
#: tree.
_ISSUE_LITERAL = re.compile(r"\bI\d{4,}\b")


_REPO_ROOT = HERE.parent.parent


def _display(path: Path) -> str:
    """A path relative to the repo root, or absolute for one outside it
    (a scan target constructed in a test's own `tmp_path`)."""
    try:
        return str(path.relative_to(_REPO_ROOT))
    except ValueError:
        return str(path)


def _acceptance_py_files() -> list[Path]:
    return sorted(p for p in HERE.rglob("*.py") if p.resolve() != SELF)


def test_the_scan_actually_reads_files() -> None:
    """A guard that walks nothing reports clean — assert it walked something."""
    files = _acceptance_py_files()
    assert len(files) >= 2, (
        f"the tracker-literal scan walked only {len(files)} files under "
        f"{HERE} — it is not reading the acceptance tree, so a clean result "
        "means nothing."
    )


def _is_pytest_fail_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "fail"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "pytest"
    )


def _own_scope_nodes(func: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.AST]:
    """Every node in `func`'s own body, never descending into a nested scope.

    A nested `def`/`class`/`lambda` is checked separately when the outer walk
    reaches it as its own top-level `FunctionDef` — descending into it here
    would attribute its literals (and its own `pytest.fail` calls) to the
    wrong enclosing function.
    """
    found: list[ast.AST] = []
    stack = list(ast.iter_child_nodes(func))
    while stack:
        node = stack.pop()
        found.append(node)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda):
            continue
        stack.extend(ast.iter_child_nodes(node))
    return found


def _docstring_constant(func: ast.FunctionDef | ast.AsyncFunctionDef) -> ast.Constant | None:
    if not func.body:
        return None
    first = func.body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        return first.value
    return None


def _findings_in_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        # Fail closed: a file this scan cannot parse is a finding, never a
        # quiet skip — an unreadable file reporting clean is the same defect
        # as a scanner that walked nothing.
        return [f"{_display(path)}: does not parse ({exc})"]

    findings: list[str] = []
    for func in ast.walk(tree):
        if not isinstance(func, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        own = _own_scope_nodes(func)
        if not any(_is_pytest_fail_call(node) for node in own):
            continue
        docstring = _docstring_constant(func)
        for node in own:
            if node is docstring:
                continue
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                hits = _ISSUE_LITERAL.findall(node.value)
                if hits:
                    findings.append(
                        f"{_display(path)}:{node.lineno}: "
                        f"{func.name} reaches pytest.fail(...) and embeds hardcoded "
                        f"tracker literal(s) {hits} — derive the pointer from "
                        "crucible.gate.PHASES instead (alpha-engine-config-I9839)"
                    )
    return findings


def test_no_hardcoded_tracker_literal_reaches_a_failure_message() -> None:
    findings: list[str] = []
    for path in _acceptance_py_files():
        findings.extend(_findings_in_file(path))
    assert not findings, (
        "an acceptance clause's failure message names a hardcoded tracker issue "
        "number instead of deriving it from crucible.gate.PHASES "
        f"(alpha-engine-config-I9839). {len(findings)} finding(s):\n"
        + "\n".join(f"  - {f}" for f in findings)
    )


def test_the_scan_can_actually_find_something(tmp_path: Path) -> None:
    """A guard nobody has made fail is a guard nobody knows works."""
    sample = tmp_path / "test_sample.py"
    sample.write_text(
        "import pytest\n\n"
        "def test_x() -> None:\n"
        '    """A docstring may name alpha-engine-config-I9757 freely."""\n'
        "    pytest.fail(\n"
        '        "UNMET — x\\n  Status: not yet satisfied (alpha-engine-config-I9757)."\n'
        "    )\n",
        encoding="utf-8",
    )
    findings = _findings_in_file(sample)
    assert findings, "the detector did not fire on a hardcoded literal it constructed itself"
    assert "I9757" in findings[0]


def test_the_scan_permits_a_derived_pointer(tmp_path: Path) -> None:
    """The companion positive case: a message built from an attribute access
    (`{phase.tracker}`) carries no literal `Constant` string with an issue
    number in it, and must not be flagged."""
    sample = tmp_path / "test_sample_ok.py"
    sample.write_text(
        "import pytest\n\n"
        "class _P:\n"
        "    tracker = 'alpha-engine-config-I9758'\n\n"
        "def test_x() -> None:\n"
        "    phase = _P()\n"
        '    pytest.fail(f"UNMET — x. Status: not yet satisfied ({phase.tracker}).")\n',
        encoding="utf-8",
    )
    findings = _findings_in_file(sample)
    assert not findings, f"a derived pointer was wrongly flagged: {findings}"
