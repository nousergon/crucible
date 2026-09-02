"""No acceptance clause hardcodes a tracker issue number, anywhere.

Normative source: alpha-engine-config-I9839. Every clause under
`tests/acceptance/` used to fail with a literal `alpha-engine-config-I9757`
baked into `_unmet`, regardless of which phase the clause actually belonged
to — so a phase-2 clause and a phase-3 clause both pointed a reader at
phase 1, and stayed pointed at it after phase 1 closed (2026-09-02T01:31Z).
`crucible.gate.PHASES` is the single source of truth for which tracker issue
owns which phase; the fix derives the pointer from it, at the two places
(`_unmet` in `test_plan_section_2_objectives.py` and
`test_feature_layer_binding.py`) that build an UNMET status line.

**Not a text grep, and not gated on "does this call `pytest.fail` directly".**
The first version of this guard was gated that way and missed the actual
attack: `TestAutonomy.test_the_ruled_live_gate_...` never calls `pytest.fail`
itself — it calls `_unmet(clause, requirement, phase="phase2")` — so a
literal appended to `requirement` in THAT function's own body reaches the
failure text without the guard's scanned function ever containing a
`pytest.fail` call. Tracing the real call graph (`_unmet`/`_attempt` call
`pytest.fail`; test methods call `_unmet`/`_attempt`; either could be
reached through an alias, `assert`, string concatenation, an f-string over a
local, or a module-level constant) is the harder-to-keep-honest half of that
problem. So this scan does not try to trace it: it treats every function in
every file under `tests/acceptance/` as reachable, and every non-docstring
string literal in the whole tree as a potential failure-message fragment —
the same posture `tests/test_no_suppressions.py` already takes for its own,
differently-shaped scan. A hardcoded `alpha-engine-config-I<N>` reference has
exactly one legitimate home in this tree: prose (a docstring, or a comment,
neither of which reaches a test failure — see "What's exempt" below).
Anywhere else, it is written to be replaced by a derived lookup.

**What's exempt, and why it is only these two things.** A docstring, because
a clause has to be able to narrate its own history in prose — this test
excludes exactly the same node shape `test_no_suppressions.py` does: the
first statement of a module, class, or function body when it is a bare
string expression. A `#` comment, because `ast.parse` never sees one — it is
not a carve-out this file grants, it is a fact about what the AST contains.
Both classes of text style are unreachable from a pytest failure line, so a
citation living there costs nothing and gains nothing by being derived.

**Fail closed.** A file this scan cannot parse is a finding, not a skip: a
scanner that goes quiet on the one file it cannot read is indistinguishable
from a clean tree.

**Why this lives in `tests/`, not `tests/acceptance/`.** `ci.yml`'s
`acceptance` job — the one that actually executes `tests/acceptance` — is
`if: github.event_name != 'pull_request'`; the PR path only ever runs
`pytest tests/acceptance --collect-only`. A guard placed inside
`tests/acceptance/` could therefore never block a PR, only mail a failure
after something had already merged to `main` — and a second-order defect:
`check_reading.py`/`ratchet.json` compare the acceptance suite's collected
`Class::method` ids against a committed set, and a plain function here would
collect as an ID neither file describes, red-ing the `acceptance` job on
every `push: [main]` for a reason unrelated to any plan clause. Living beside
`tests/test_no_suppressions.py` makes this a blocking PR check and keeps it
out of the acceptance ratchet's id set entirely.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ACCEPTANCE_DIR = REPO_ROOT / "tests" / "acceptance"

#: An issue number as it appears in `alpha-engine-config-I9757`: `I` followed
#: by four-or-more digits. Four, not `\d+`, so this does not fire on an
#: unrelated single- or double-digit `I` token elsewhere in the tree.
_ISSUE_LITERAL = re.compile(r"\bI\d{4,}\b")


def _acceptance_py_files() -> list[Path]:
    return sorted(ACCEPTANCE_DIR.rglob("*.py"))


def test_the_scan_actually_reads_files() -> None:
    """A guard that walks nothing reports clean — assert it walked something."""
    files = _acceptance_py_files()
    assert len(files) >= 2, (
        f"the tracker-literal scan walked only {len(files)} files under "
        f"{ACCEPTANCE_DIR} — it is not reading the acceptance tree, so a "
        "clean result means nothing."
    )


def _docstring_node_ids(tree: ast.AST) -> set[int]:
    """`id()` of every AST node that IS a docstring — the exemption.

    Same shape as `tests/test_no_suppressions.py::_docstring_lines`: the
    first statement of a module, class, function or async-function body,
    when it is itself a bare string expression.
    """
    ids: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        body = getattr(node, "body", [])
        if not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
            if isinstance(first.value.value, str):
                ids.add(id(first.value))
    return ids


def _findings_in_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        # Fail closed: a file this scan cannot parse is a finding, never a
        # quiet skip — an unreadable file reporting clean is the same defect
        # as a scanner that walked nothing.
        return [f"{_display(path)}: does not parse ({exc})"]

    exempt = _docstring_node_ids(tree)
    findings: list[str] = []
    # Every Constant string in the WHOLE module — not gated on which function
    # it sits in, or on what that function calls. A `pytest.fail(...)`
    # literal, a `MSG = "..."` module constant, an `assert x, "..."` message,
    # implicit string concatenation, and a literal chunk of an f-string are
    # all just `ast.Constant` nodes at this level, so all five are caught by
    # the same check.
    for node in ast.walk(tree):
        if id(node) in exempt:
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            hits = _ISSUE_LITERAL.findall(node.value)
            if hits:
                findings.append(
                    f"{_display(path)}:{node.lineno}: hardcoded tracker "
                    f"literal(s) {hits} — derive from crucible.gate.PHASES, "
                    "or move the reference into a docstring/comment if it "
                    "cites a different (non-phase) issue "
                    "(alpha-engine-config-I9839)"
                )
    return findings


def _display(path: Path) -> str:
    """A path relative to the repo root, or absolute for one outside it
    (a scan target constructed in a test's own `tmp_path`)."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def test_no_hardcoded_tracker_literal_anywhere_in_the_acceptance_tree() -> None:
    findings: list[str] = []
    for path in _acceptance_py_files():
        findings.extend(_findings_in_file(path))
    assert not findings, (
        "an acceptance-tree file names a hardcoded tracker issue number instead "
        "of deriving it from crucible.gate.PHASES, or citing it only in prose "
        f"(alpha-engine-config-I9839). {len(findings)} finding(s):\n"
        + "\n".join(f"  - {f}" for f in findings)
    )


# ---------------------------------------------------------------------------
# The detector is shown firing — and shown firing on every shape that was
# proposed against the first version of this guard and found it blind:
# a call site building `requirement`/message text with no `pytest.fail` in
# its own body, an aliased `from pytest import fail`, an `assert` message, a
# module-level constant, a nested `def`, and implicit string concatenation.
# A guard nobody has made fail is a guard nobody knows works.
# ---------------------------------------------------------------------------


def test_the_scan_fires_on_a_literal_in_a_function_with_no_direct_pytest_fail_call(
    tmp_path: Path,
) -> None:
    """The exact miss reported against the previous version of this guard:
    appending a literal to `requirement` in the CALLING test method, which
    itself never calls `pytest.fail` — only `_unmet` does, in another
    function entirely."""
    sample = tmp_path / "test_sample.py"
    sample.write_text(
        "import pytest\n\n"
        "def _unmet(clause, requirement):\n"
        '    pytest.fail(f"UNMET {clause}: {requirement}")\n\n'
        "def test_x() -> None:\n"
        '    requirement = "the real requirement"\n'
        '    requirement = requirement + " (tracked: alpha-engine-config-I9757)"\n'
        '    _unmet("clause", requirement)\n',
        encoding="utf-8",
    )
    findings = _findings_in_file(sample)
    assert findings, "did not catch a literal appended in the CALLING function, not _unmet itself"
    assert "I9757" in findings[0]


def test_the_scan_fires_on_an_aliased_fail_import(tmp_path: Path) -> None:
    sample = tmp_path / "test_sample.py"
    sample.write_text(
        "from pytest import fail\n\n"
        "def test_x() -> None:\n"
        '    fail("UNMET (alpha-engine-config-I9757)")\n',
        encoding="utf-8",
    )
    findings = _findings_in_file(sample)
    assert findings, "did not catch a literal reaching an aliased `from pytest import fail`"


def test_the_scan_fires_on_an_assert_message(tmp_path: Path) -> None:
    sample = tmp_path / "test_sample.py"
    sample.write_text(
        'def test_x() -> None:\n    assert False, "UNMET (alpha-engine-config-I9757)"\n',
        encoding="utf-8",
    )
    findings = _findings_in_file(sample)
    assert findings, 'did not catch a literal in an `assert ..., "..."` message'


def test_the_scan_fires_on_implicit_string_concatenation(tmp_path: Path) -> None:
    sample = tmp_path / "test_sample.py"
    sample.write_text(
        "import pytest\n\n"
        "def test_x() -> None:\n"
        '    pytest.fail("UNMET tracked: " "alpha-engine-config-" "I9757")\n',
        encoding="utf-8",
    )
    findings = _findings_in_file(sample)
    assert findings, "did not catch a literal split across implicitly concatenated strings"


def test_the_scan_fires_on_a_module_level_constant(tmp_path: Path) -> None:
    sample = tmp_path / "test_sample.py"
    sample.write_text(
        "import pytest\n\n"
        'MSG = "UNMET (alpha-engine-config-I9757)"\n\n'
        "def test_x() -> None:\n"
        "    pytest.fail(MSG)\n",
        encoding="utf-8",
    )
    findings = _findings_in_file(sample)
    assert findings, "did not catch a literal defined as a module-level constant"


def test_the_scan_fires_inside_a_nested_def(tmp_path: Path) -> None:
    sample = tmp_path / "test_sample.py"
    sample.write_text(
        "import pytest\n\n"
        "def test_x() -> None:\n"
        "    def _inner():\n"
        '        pytest.fail("UNMET (alpha-engine-config-I9757)")\n'
        "    _inner()\n",
        encoding="utf-8",
    )
    findings = _findings_in_file(sample)
    assert findings, "did not catch a literal inside a nested function"


def test_the_scan_permits_a_derived_pointer(tmp_path: Path) -> None:
    """The positive case, in the same shape `crucible.gate.Phase.tracker`
    actually uses: the issue NUMBER is an int, and the `alpha-engine-config-I`
    prefix plus that int are joined in an f-string at read time — so no
    `ast.Constant` string anywhere carries the literal `I9758`, and this must
    not be flagged. (A class attribute holding the finished string directly,
    e.g. `tracker = 'alpha-engine-config-I9758'`, is NOT this case — that
    would be a hardcoded literal one level removed, and the scan is right to
    catch it.)"""
    sample = tmp_path / "test_sample_ok.py"
    sample.write_text(
        "import pytest\n\n"
        "class _Phase:\n"
        "    def __init__(self, issue: int) -> None:\n"
        "        self.issue = issue\n\n"
        "    @property\n"
        "    def tracker(self) -> str:\n"
        "        return f'alpha-engine-config-I{self.issue}'\n\n"
        "def test_x() -> None:\n"
        "    phase = _Phase(9758)\n"
        '    pytest.fail(f"UNMET — x. Status: not yet satisfied ({phase.tracker}).")\n',
        encoding="utf-8",
    )
    findings = _findings_in_file(sample)
    assert not findings, f"a derived pointer was wrongly flagged: {findings}"


def test_the_scan_permits_a_docstring_citation(tmp_path: Path) -> None:
    """The other positive case: prose. A docstring narrating history is not
    reachable from a test failure and costs nothing left as-is."""
    sample = tmp_path / "test_sample_ok2.py"
    sample.write_text(
        "def test_x() -> None:\n"
        '    """Landed once alpha-engine-config-I9772 reconciled two tracks."""\n'
        "    assert True\n",
        encoding="utf-8",
    )
    findings = _findings_in_file(sample)
    assert not findings, f"a docstring citation was wrongly flagged: {findings}"
