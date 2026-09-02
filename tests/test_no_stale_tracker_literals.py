"""No hardcoded tracker issue number, anywhere in the package.

Normative source: alpha-engine-config-I9839. Every acceptance clause under
`tests/acceptance/` used to fail with a literal `alpha-engine-config-I9757`
baked into `_unmet`, regardless of which phase the clause actually belonged
to — so a phase-2 clause and a phase-3 clause both pointed a reader at
phase 1, and stayed pointed at it after phase 1 closed (2026-09-02T01:31Z).
The same shape turned out to live in PRODUCTION code too, not just tests:
`crucible/cli.py`'s `_todo` stub and `crucible/track_a.py`'s `_slot_module`
both handed a live user the same closed `I9757` (round 2 of this PR's own
review). `crucible.gate.PHASES` is the single source of truth for which
tracker issue owns which phase; every one of those sites now derives its
pointer from it instead of restating a literal.

**Scope is the whole package, not one directory.** The class is "a hardcoded
tracker literal that can go stale", and nothing about that class is specific
to `tests/acceptance/` — it happened there first only because that is where
this issue started. This scan walks `crucible/` and `tests/` both.

**Not a text grep, and not gated on "does this call `pytest.fail` directly".**
An earlier version of this guard was gated that way and missed the actual
attack: a test method that never calls `pytest.fail` itself — it calls
`_unmet(clause, requirement, phase="phase2")` — can still leak a literal
into the eventual failure text by building `requirement` with one appended
locally. Tracing the real call graph (`_unmet`/`_attempt` call `pytest.fail`;
callers reach either through an alias, `assert`, string concatenation, an
f-string over a local, or a module-level constant) is the harder-to-keep-
honest half of that problem. So this scan does not try to trace it: it
treats every function in every scanned file as reachable, and every
non-docstring string literal in the whole tree as a potential
message fragment — the same posture `tests/test_no_suppressions.py` already
takes for its own, differently-shaped scan. A hardcoded
`alpha-engine-config-I<N>` reference has exactly one legitimate home:
prose (a docstring, or a comment, neither of which reaches a raised message
or a test failure — see "What's exempt" below). Anywhere else, it is written
to be replaced by a derived lookup, or — when it cites a historical, non-phase
issue that can never be derived from `PHASES` (the `I9772`/`I9777`/`I9778`/
`I9780`/`I9786`/`I9787`/`I9816`/`I9745` shape) — moved into a comment.

**What's exempt, and why it is only these two things.** A docstring, because
code has to be able to narrate its own history in prose — this scan excludes
exactly the same node shape `test_no_suppressions.py` does: the first
statement of a module, class, function or async-function body when it is a
bare string expression. A `#` comment, because `ast.parse` never sees one —
it is not a carve-out this file grants, it is a fact about what the AST
contains. Both classes of text are unreachable from a raised message or a
test failure, so a citation living there costs nothing and gains nothing by
being derived.

**A test asserting a derivation's output is not exempt from this — the
oracle's expectation is built from the test's own INPUT instead.**
`tests/test_phase_ladder.py` asserts `crucible.gate`'s ladder output names
phase 0's tracker; the naive fix (comparing against `PHASES[0].tracker`)
would make the assertion compare `PHASES[0].tracker` against itself and
prove nothing. The actual fix is smaller: the test already supplies phase
0's issue NUMBER as its own input (`_PHASE0_ISSUE = 9756`, a plain `int`,
independently cross-checked against `PHASES[0].issue` by
`test_every_plan_phase_has_a_rung`), and builds the expected string from
that input with an f-string (`f"alpha-engine-config-I{_PHASE0_ISSUE}"`) —
non-circular, and no `ast.Constant` string anywhere carries the literal.

**This scan does not carry a per-file exemption list, and does not need
one.** `crucible/console/render.py` and `crucible/slots/inputs.py` (plus
its paired test) are, as of this PR, still red — both are owned by
concurrent sessions (`i9837-red-board` and `crucible-PR43` respectively)
this PR is not authorized to edit. That is a KNOWN, reported (
`alpha-engine-config-I9868`), currently-red state, not a bug in the scan:
merging past it would be exactly the "a red check the author decided
doesn't count" anti-pattern `crucible-PR38` existed to close (Brian's
2026-08-10 ruling). It is expected to clear on its own, with nothing
further to do here, once those two branches land and this PR is rebased
onto `main`.

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
`Class::method` ids against a committed set, and a plain function there
would collect as an id neither file describes, red-ing the `acceptance` job
on every `push: [main]` for a reason unrelated to any plan clause. Living
beside `tests/test_no_suppressions.py` makes this a blocking PR check and
keeps it out of the acceptance ratchet's id set entirely.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The whole package: production code plus every test. `tests/acceptance/`
#: is included (it is already clean) rather than special-cased back out.
SCAN_ROOTS = (REPO_ROOT / "crucible", REPO_ROOT / "tests")

#: This file, and nothing else — it must contain the pattern it searches for
#: in order to document and self-test it. Same exemption shape as
#: `tests/test_no_suppressions.py::SELF`: a single path compared by equality,
#: never a collection, because an exemption LIST is itself the bug class this
#: repository forbids.
SELF = Path(__file__).resolve()

_IGNORED_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache"}

#: An issue number as it appears in `alpha-engine-config-I9757`: `I` followed
#: by four-or-more digits. Four, not `\d+`, so this does not fire on an
#: unrelated single- or double-digit `I` token elsewhere in the tree.
_ISSUE_LITERAL = re.compile(r"\bI\d{4,}\b")


def _scanned_py_files() -> list[Path]:
    files: list[Path] = []
    for root in SCAN_ROOTS:
        for path in root.rglob("*.py"):
            if any(part in _IGNORED_DIRS for part in path.parts):
                continue
            if path.resolve() == SELF:
                continue
            files.append(path)
    return sorted(set(files))


def test_the_scan_actually_reads_files() -> None:
    """A guard that walks nothing reports clean — assert it walked something."""
    files = _scanned_py_files()
    assert len(files) >= 50, (
        f"the tracker-literal scan walked only {len(files)} files under "
        f"{SCAN_ROOTS} — it is not reading the package, so a clean result "
        "means nothing."
    )
    assert any(f.parent.name == "crucible" for f in files), (
        "the scan did not reach the package source under crucible/"
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


def _display(path: Path) -> str:
    """A path relative to the repo root, or absolute for one outside it
    (a scan target constructed in a test's own `tmp_path`)."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


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
    # it sits in, or on what that function calls. A `pytest.fail(...)`/
    # `raise ...(...)` literal, a `MSG = "..."` module constant, an
    # `assert x, "..."` message, implicit string concatenation, and a
    # literal chunk of an f-string are all just `ast.Constant` nodes at this
    # level, so all five are caught by the same check.
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


def test_no_hardcoded_tracker_literal_anywhere_in_the_package() -> None:
    findings: list[str] = []
    for path in _scanned_py_files():
        findings.extend(_findings_in_file(path))
    assert not findings, (
        "the package names a hardcoded tracker issue number instead of deriving "
        "it from crucible.gate.PHASES, or citing it only in prose "
        f"(alpha-engine-config-I9839). {len(findings)} finding(s):\n"
        + "\n".join(f"  - {f}" for f in findings)
    )


# ---------------------------------------------------------------------------
# The detector is shown firing — and shown firing on every shape that was
# proposed against earlier versions of this guard and found them blind:
# a call site building message text with no `pytest.fail`/`raise` in its own
# body, an aliased `from pytest import fail`, an `assert` message, a
# module-level constant, a nested `def`, and implicit string concatenation.
# A guard nobody has made fail is a guard nobody knows works.
# ---------------------------------------------------------------------------


def test_the_scan_fires_on_a_literal_in_a_function_with_no_direct_pytest_fail_call(
    tmp_path: Path,
) -> None:
    """The exact miss reported against an earlier version of this guard:
    appending a literal to `requirement` in the CALLING function, which
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


def test_the_scan_fires_on_a_hardcoded_literal_in_production_code(tmp_path: Path) -> None:
    """The round-2 miss: a hardcoded tracker literal reaching a live user
    through a raised exception in production code, not a test failure."""
    sample = tmp_path / "cli.py"
    sample.write_text(
        "def handler() -> None:\n"
        '    raise NotImplementedError("not implemented yet (alpha-engine-config-I9757)")\n',
        encoding="utf-8",
    )
    findings = _findings_in_file(sample)
    assert findings, "did not catch a hardcoded literal in a production `raise`, no test involved"


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


def test_the_scan_fails_closed_on_an_unparseable_file(tmp_path: Path) -> None:
    sample = tmp_path / "test_broken.py"
    sample.write_text("def test_x(:\n    this is not python\n", encoding="utf-8")
    findings = _findings_in_file(sample)
    assert findings, "an unparseable file must be a finding, never a quiet skip"
    assert "does not parse" in findings[0]


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
    reachable from a raised message or a test failure and costs nothing left
    as-is."""
    sample = tmp_path / "test_sample_ok2.py"
    sample.write_text(
        "def test_x() -> None:\n"
        '    """Landed once alpha-engine-config-I9772 reconciled two tracks."""\n'
        "    assert True\n",
        encoding="utf-8",
    )
    findings = _findings_in_file(sample)
    assert not findings, f"a docstring citation was wrongly flagged: {findings}"


def test_the_scan_permits_a_comment_citation(tmp_path: Path) -> None:
    """A `#` comment is not part of the AST at all — this is a fact about
    the parser, not an exemption this file grants, and it is worth proving:
    a comment sitting on the same line pattern that would be flagged as a
    string must not be."""
    sample = tmp_path / "test_sample_ok3.py"
    sample.write_text(
        "def test_x() -> None:\n"
        "    # Historical citation: alpha-engine-config-I9772.\n"
        "    assert True\n",
        encoding="utf-8",
    )
    findings = _findings_in_file(sample)
    assert not findings, f"a comment citation was wrongly flagged: {findings}"
