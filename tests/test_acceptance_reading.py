"""The acceptance arm signals by grading its READING, so the grader is tested.

`tests/acceptance/check_reading.py` is the only thing standing between "the
suite is red by design" and "the suite is red because someone broke it". Every
outcome it can produce is asserted here, including the three that look like
success and are not: an absent report, an empty one, and a regression offset by
a gain.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "tests" / "acceptance" / "check_reading.py"
RATCHET = ROOT / "tests" / "acceptance" / "ratchet.json"
OBJECTIVES_SOURCE = ROOT / "tests" / "acceptance" / "test_plan_section_2_objectives.py"


def _report(
    path: Path,
    *,
    met: list[str],
    unmet: list[str],
    unmeasurable: dict[str, str] | list[str] | None = None,
) -> Path:
    """Build a JUnit report. `unmeasurable` maps clause id -> `blocked_on_class`
    (a bare list defaults every entry to `"NoCredentialsError"`) and is
    written as `<properties>` — `record_property`'s real wire shape — never
    as message text, matching what `_unmeasurable` actually emits (round 2)."""
    if unmeasurable is None:
        unmeasurable = {}
    elif isinstance(unmeasurable, list):
        unmeasurable = dict.fromkeys(unmeasurable, "NoCredentialsError")

    cases = "".join(
        f'<testcase classname="tests.acceptance.mod.{name.split("::")[0]}" '
        f'name="{name.split("::")[1]}"/>'
        for name in met
    )
    cases += "".join(
        f'<testcase classname="tests.acceptance.mod.{name.split("::")[0]}" '
        f'name="{name.split("::")[1]}"><failure message="Failed: UNMET — {name}">'
        f"UNMET</failure></testcase>"
        for name in unmet
    )
    cases += "".join(
        f'<testcase classname="tests.acceptance.mod.{name.split("::")[0]}" '
        f'name="{name.split("::")[1]}"><properties>'
        f'<property name="outcome" value="unmeasurable" />'
        f'<property name="blocked_on_class" value="{blocked_on_class}" />'
        f'</properties><failure message="Failed: UNMEASURABLE — {name}">'
        f"UNMEASURABLE</failure></testcase>"
        for name, blocked_on_class in unmeasurable.items()
    )
    path.write_text(f'<?xml version="1.0"?><testsuites><testsuite>{cases}</testsuite></testsuites>')
    return path


def _run(report: Path, ratchet: Path) -> subprocess.CompletedProcess[str]:
    # The checker reads ratchet.json from its own directory, so a test ratchet
    # means a checker copy beside it.
    checker = ratchet.parent / "check_reading.py"
    checker.write_text(CHECKER.read_text())
    return subprocess.run(
        [sys.executable, str(checker), str(report)],
        capture_output=True,
        text=True,
        check=False,
    )


def _run_with_args(
    report: Path, ratchet: Path, *extra_args: str
) -> subprocess.CompletedProcess[str]:
    """Same shape as `_run`, with extra CLI args inserted before the report
    path — `--write-json <path>` and `--commit <sha>`, for
    `alpha-engine-config-I9902`'s publisher."""
    checker = ratchet.parent / "check_reading.py"
    checker.write_text(CHECKER.read_text())
    return subprocess.run(
        [sys.executable, str(checker), *extra_args, str(report)],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def ratchet(tmp_path: Path) -> Path:
    path = tmp_path / "ratchet.json"
    path.write_text(json.dumps({"unmet": {"T::a": "phase 2"}, "met": ["T::b", "T::c"]}))
    return path


@pytest.fixture
def ratchet_with_unmeasurable(tmp_path: Path) -> Path:
    """`unmeasurable` is a SUBSET of `unmet`'s keys: T::d is listed in both,
    since `crucible/gate.py`'s phase-0 clause reads `met | unmet` as the full
    clause set and does not know about the finer `unmeasurable` bucket. Each
    `unmeasurable` VALUE is an object — `reason`, `blocked_on_class`,
    `last_moved` (round 2, review finding 1/3) — matching `_report`'s default
    `blocked_on_class` of `"NoCredentialsError"` for a bare id list."""
    path = tmp_path / "ratchet.json"
    path.write_text(
        json.dumps(
            {
                "unmet": {"T::a": "phase 2", "T::d": "no AWS credentials in this environment"},
                "unmeasurable": {
                    "T::d": {
                        "reason": "no AWS credentials in this environment",
                        "blocked_on_class": "NoCredentialsError",
                        "last_moved": "2026-09-02",
                    }
                },
                "met": ["T::b", "T::c"],
            }
        )
    )
    return path


def _checker():
    """Import `check_reading` as a module, for the model-level assertions.

    Registered into `sys.modules` under its own name before `exec_module` —
    `Ratchet`'s `unmeasurable: dict[str, UnmeasurableEntry]` needs pydantic to
    resolve the `UnmeasurableEntry` forward reference via the module's own
    globals (`from __future__ import annotations` makes every annotation a
    string), which pydantic looks up through `sys.modules[cls.__module__]` —
    unregistered, that lookup fails with `PydanticUserError: not fully
    defined` even though the class is defined earlier in the same file.
    """
    import sys
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("check_reading", CHECKER)
    assert spec and spec.loader
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _objectives():
    """Import `test_plan_section_2_objectives` as a module, for round-2 tests
    that need `TestCost`, `_unmeasurable` and its allowlists directly.
    Registered into `sys.modules` for the same reason `_checker()` is (see
    its docstring); the name is distinct from pytest's own collected copy of
    this file, so this never shadows or is shadowed by it. Loading it
    executes no test body (only class/function definitions), the same
    guarantee pytest's own collection of this file already relies on."""
    import sys
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("_round2_objectives_reflection", OBJECTIVES_SOURCE)
    assert spec and spec.loader
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_committed_ratchet_is_wellformed() -> None:
    """The file CI grades against must parse and carry both fields."""
    # The model IS the validation — id shape, disjointness, duplicate ids and
    # empty reasons are all field-level rules on `Ratchet`, so loading it is
    # the assertion. Hand-rolled type checks here would be the contract
    # restated in a second place, which is how a contract drifts.
    ratchet = _checker().load_ratchet(RATCHET)
    assert ratchet.clauses, "the committed ratchet describes no clauses"


def _collected_clause_ids() -> list[str]:
    """The `Class::method` id of every acceptance clause, by collection.

    `-o addopts=` clears `pyproject.toml`'s `-q`, which would otherwise combine
    with the `-q` below into `-qq` and print per-file counts instead of ids.
    Collection imports the modules and executes no test body, so this touches
    no live AWS and belongs on the PR path.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/acceptance",
            "--collect-only",
            "-q",
            "-o",
            "addopts=",
            "-p",
            "no:cacheprovider",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    ids = [
        "::".join(line.strip().split("::")[-2:])
        for line in result.stdout.splitlines()
        if line.startswith("tests/acceptance/") and line.count("::") >= 2
    ]
    # A parser that silently yields nothing would make every assertion below
    # vacuously true — the dark-gate shape, one level up.
    assert ids, f"collected no clause ids from:\n{result.stdout}"
    return ids


def test_the_committed_ratchet_matches_the_real_suite() -> None:
    """The ratchet is verified by ID SET on the PR path, not by count.

    Counting was the round-two defect one level up: `check_reading.py`
    compares a set, so verifying the committed file by count alone let a
    RENAMED clause pass every PR check and then fail the first `push: [main]`
    run with the actively misleading "clauses now pass that ratchet.json still
    lists as unmet". Deleting a clause and editing `collected` to match had
    the same shape — the ratchet grading itself.
    """
    ids = _collected_clause_ids()
    ratchet = json.loads(RATCHET.read_text())

    assert len(ids) == len(set(ids)), (
        f"duplicate clause ids collected: {sorted({i for i in ids if ids.count(i) > 1})}. "
        "Two same-named classes in different modules collapse to one id, and one "
        "can then regress invisibly."
    )
    want = set(ratchet["met"]) | set(ratchet["unmet"]) | set(ratchet.get("unmeasurable", {}))
    vanished = sorted(want - set(ids))
    appeared = sorted(set(ids) - want)
    assert not vanished and not appeared, (
        "tests/acceptance no longer collects what ratchet.json describes."
        + (f" Gone: {vanished}." if vanished else "")
        + (f" New: {appeared}." if appeared else "")
        + " Every clause is compared by ID, met ones included: comparing counts "
        "let any of the passing plan §2 objectives be renamed or deleted with a "
        "one-digit edit while every check read green. Update the ratchet in the "
        "PR that changes the suite."
    )


def test_an_unchanged_reading_is_green(tmp_path: Path, ratchet: Path) -> None:
    report = _report(tmp_path / "r.xml", met=["T::b", "T::c"], unmet=["T::a"])
    result = _run(report, ratchet)
    assert result.returncode == 0, result.stderr
    assert "unchanged" in result.stdout


def test_a_regression_is_red(tmp_path: Path, ratchet: Path) -> None:
    report = _report(tmp_path / "r.xml", met=["T::c"], unmet=["T::a", "T::b"])
    result = _run(report, ratchet)
    assert result.returncode == 1
    assert "REGRESSION" in result.stderr
    assert "T::b" in result.stderr


def test_a_regression_offset_by_a_gain_is_red(tmp_path: Path, ratchet: Path) -> None:
    """The count is unchanged and something is broken.

    This is why the comparison is set-shaped. An earlier version of the grader
    compared only `met` and returned "the reading is unchanged" for exactly
    this input.
    """
    report = _report(tmp_path / "r.xml", met=["T::a", "T::c"], unmet=["T::b"])
    result = _run(report, ratchet)
    assert result.returncode == 1
    assert "REGRESSION" in result.stderr
    assert "T::b" in result.stderr
    assert "T::a" in result.stderr  # names the gain too, so the delta is legible


def test_unrecorded_progress_is_red(tmp_path: Path, ratchet: Path) -> None:
    """A clause going green without the ratchet moving is not a pass.

    Otherwise the number nobody updates stops meaning anything, and the next
    regression is measured against a floor that was never true.
    """
    report = _report(tmp_path / "r.xml", met=["T::a", "T::b", "T::c"], unmet=[])
    result = _run(report, ratchet)
    assert result.returncode == 1
    assert "not recorded" in result.stderr
    assert "T::a" in result.stderr


def test_a_removed_clause_is_red(tmp_path: Path, ratchet: Path) -> None:
    """A clause added, removed, renamed, or failing to import."""
    report = _report(tmp_path / "r.xml", met=["T::b"], unmet=["T::a"])
    result = _run(report, ratchet)
    assert result.returncode == 1
    assert "Gone: T::c" in result.stderr


def test_a_renamed_met_clause_is_red(tmp_path: Path, ratchet: Path) -> None:
    """The count is unchanged and a plan §2 objective has silently moved.

    Comparing counts made every MET clause deletable with a one-digit edit —
    the 21 passing objectives were the unprotected majority.
    """
    report = _report(tmp_path / "r.xml", met=["T::b", "T::renamed"], unmet=["T::a"])
    result = _run(report, ratchet)
    assert result.returncode == 1
    assert "Gone: T::c" in result.stderr
    assert "New: T::renamed" in result.stderr


@pytest.mark.parametrize(
    ("document", "expected"),
    [
        ({"unmet": {"T::a": "phase 2"}}, "met"),
        ({"met": ["T::b"]}, "unmet"),
        ({"unmet": {"T::a": "why"}, "met": ["T::a"]}, "both met and unmet"),
        ({"unmet": {"T::a": "why"}, "met": ["T::b", "T::b"]}, "same clause twice"),
        ({"unmet": {"not-an-id": "why"}, "met": []}, "clause id"),
        ({"unmet": {"T::a": "   "}, "met": []}, "no reason"),
        ({"unmet": {}, "met": [], "surprise": 1}, "surprise"),
    ],
)
def test_a_malformed_ratchet_is_red_and_names_the_field(
    tmp_path: Path, document: dict, expected: str
) -> None:
    """Every malformed shape fails at the boundary, naming the field.

    Before the model, a missing key surfaced as a KeyError traceback three
    functions in — unreadable in an Actions log and indistinguishable from the
    grader itself being broken. `extra="forbid"` is in the list because a typo
    in a key name is otherwise a silently ignored edit.
    """
    ratchet = tmp_path / "ratchet.json"
    ratchet.write_text(json.dumps(document))
    report = _report(tmp_path / "r.xml", met=["T::b"], unmet=["T::a"])
    result = _run(report, ratchet)
    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    assert expected in result.stderr


def test_a_missing_report_is_red_not_green(tmp_path: Path, ratchet: Path) -> None:
    """Absence is never a pass (principle 7)."""
    result = _run(tmp_path / "does-not-exist.xml", ratchet)
    assert result.returncode == 1
    assert "dark, not green" in result.stderr


def test_an_empty_report_is_red_not_green(tmp_path: Path, ratchet: Path) -> None:
    report = tmp_path / "r.xml"
    report.write_text('<?xml version="1.0"?><testsuites><testsuite></testsuite></testsuites>')
    result = _run(report, ratchet)
    assert result.returncode == 1
    assert "collected no tests" in result.stderr


def test_an_unparseable_report_is_red_not_green(tmp_path: Path, ratchet: Path) -> None:
    report = tmp_path / "r.xml"
    report.write_text("this is not xml <<<")
    result = _run(report, ratchet)
    assert result.returncode == 1
    assert "does not parse" in result.stderr


def test_a_collection_error_is_red(tmp_path: Path, ratchet: Path) -> None:
    """pytest records a collection error as an `error` case, not `no tests ran`.

    This is the shape the old `grep -q "no tests ran"` guard could not see.
    """
    report = tmp_path / "r.xml"
    report.write_text(
        textwrap.dedent(
            """\
            <?xml version="1.0"?><testsuites><testsuite>
            <testcase classname="tests.acceptance.test_x" name="">
            <error message="ImportError">cannot import name 'gone'</error>
            </testcase>
            </testsuite></testsuites>
            """
        )
    )
    result = _run(report, ratchet)
    assert result.returncode == 1
    assert "Gone:" in result.stderr or "New:" in result.stderr


def test_a_credential_failure_reads_unmeasurable_never_unmet(tmp_path: Path) -> None:
    """alpha-engine-config-I9828: the whole point of the taxonomy.

    A clause whose `<properties>` carry `outcome=unmeasurable` (round 2 —
    what `_unmeasurable` in `test_plan_section_2_objectives.py` writes via
    `record_property`, never message text) is classified into
    `Reading.unmeasurable`, a SUBSET of `Reading.unmet` (so
    `crucible/gate.py`'s coarse `met | unmet` view still sees it as
    not-met) — and it is not counted in `Reading.met`, nor in
    `Reading.plain_unmet` (the genuine, code-fixable gaps), either.
    """
    checker = _checker()
    report = _report(tmp_path / "r.xml", met=["T::b"], unmet=["T::a"], unmeasurable=["T::d"])
    reading = checker.read_report(report)
    assert reading.unmeasurable == {"T::d"}
    assert reading.unmet == {"T::a", "T::d"}
    assert reading.plain_unmet == {"T::a"}
    assert reading.met == {"T::b"}, "neither an unmet nor an unmeasurable clause is a pass"


def test_an_unmet_reading_renders_unmet_not_unmeasurable(tmp_path: Path) -> None:
    """A plain UNMET failure — no marker — stays in `unmet`, unchanged from
    before this clause-taxonomy existed."""
    checker = _checker()
    report = _report(tmp_path / "r.xml", met=["T::b"], unmet=["T::a"])
    reading = checker.read_report(report)
    assert reading.unmet == {"T::a"}
    assert reading.unmeasurable == set()


def test_a_matching_unmeasurable_reading_is_green(
    tmp_path: Path, ratchet_with_unmeasurable: Path
) -> None:
    report = _report(
        tmp_path / "r.xml", met=["T::b", "T::c"], unmet=["T::a"], unmeasurable=["T::d"]
    )
    result = _run(report, ratchet_with_unmeasurable)
    assert result.returncode == 0, result.stderr
    assert "unmeasurable" in result.stdout


def test_an_undeclared_unmeasurable_clause_is_red(
    tmp_path: Path, ratchet_with_unmeasurable: Path
) -> None:
    """A clause reading UNMEASURABLE that the ratchet does not list at all is
    a vanished/appeared collection drift, exactly like an unmet clause would
    be — it must not be silently absorbed as "extra credit"."""
    report = _report(
        tmp_path / "r.xml",
        met=["T::b", "T::c"],
        unmet=["T::a"],
        unmeasurable=["T::d", "T::e"],
    )
    result = _run(report, ratchet_with_unmeasurable)
    assert result.returncode == 1
    assert "New: T::e" in result.stderr


def test_a_clause_that_stops_being_readable_is_red_not_silently_absorbed(
    tmp_path: Path, ratchet_with_unmeasurable: Path
) -> None:
    """A clause the ratchet says is MET that now fails to read at all is not
    a regression (no code broke it) but it is still a drifted reading that
    must fail the run and name the ratchet as needing an update."""
    report = _report(
        tmp_path / "r.xml", met=["T::c"], unmet=["T::a"], unmeasurable=["T::b", "T::d"]
    )
    result = _run(report, ratchet_with_unmeasurable)
    assert result.returncode == 1
    assert "no longer be READ" in result.stderr
    assert "T::b" in result.stderr


def test_a_clause_reclassified_between_unmet_and_unmeasurable_is_red(
    tmp_path: Path, ratchet_with_unmeasurable: Path
) -> None:
    """T::a is `unmet` in the ratchet; reading it as `unmeasurable` instead
    is neither a regression nor progress, but the ratchet no longer
    describes the actual reading and must fail."""
    report = _report(
        tmp_path / "r.xml", met=["T::b", "T::c"], unmet=[], unmeasurable=["T::a", "T::d"]
    )
    result = _run(report, ratchet_with_unmeasurable)
    assert result.returncode == 1
    assert "failure reason changed" in result.stderr
    assert "T::a" in result.stderr


def test_progress_out_of_unmeasurable_is_recorded_like_progress_out_of_unmet(
    tmp_path: Path, ratchet_with_unmeasurable: Path
) -> None:
    report = _report(
        tmp_path / "r.xml", met=["T::b", "T::c", "T::d"], unmet=["T::a"], unmeasurable=[]
    )
    result = _run(report, ratchet_with_unmeasurable)
    assert result.returncode == 1
    assert "not recorded" in result.stderr
    assert "T::d" in result.stderr


def test_a_malformed_unmeasurable_bucket_is_red(tmp_path: Path) -> None:
    """`unmeasurable` gets the same field-level rules as `unmet`: well-formed
    ids, a non-empty reason, and disjointness from the other two buckets."""
    ratchet = tmp_path / "ratchet.json"
    ratchet.write_text(json.dumps({"unmet": {}, "unmeasurable": {"T::a": "   "}, "met": ["T::a"]}))
    report = _report(tmp_path / "r.xml", met=["T::a"], unmet=[])
    result = _run(report, ratchet)
    assert result.returncode == 1
    assert "Traceback" not in result.stderr


def test_the_module_path_is_not_part_of_a_clause_id() -> None:
    """Renaming a file is not a moved reading."""
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("check_reading", CHECKER)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.clause_id("tests.acceptance.old_name.TestX", "test_y") == "TestX::test_y"
    assert module.clause_id("tests.acceptance.new_name.TestX", "test_y") == "TestX::test_y"


def test_ratchets_unmet_phase_agrees_with_the_clauses_own_phase_kwarg() -> None:
    """`ratchet.json`'s `unmet` reasons and `_unmet`'s `phase=` kwarg are two
    restatements of the same fact — the ratchet says "phase 2 — ..." in
    prose, the code passes `phase="phase2"` — and nothing compared them
    (alpha-engine-config-I9839, round-2 review note 2). This closes that:
    for every `unmet` clause in the committed ratchet, the phase number its
    reason names must match the phase number `crucible.gate.PHASES` resolves
    the clause's own `phase=` kwarg to, so the two can no longer drift apart
    silently.

    AST-based, not an import-and-run: two of these clauses read live AWS or
    require production stores, and this file's own docstring is explicit
    that collection — never execution — is what belongs on the PR path.
    """
    import ast
    import re

    from crucible.gate import PHASES

    phase_number = {phase.id: phase.number for phase in PHASES}

    ratchet = json.loads(RATCHET.read_text())
    unmet: dict[str, str] = ratchet["unmet"]
    assert unmet, "the committed ratchet names no unmet clauses — nothing to check"

    source_path = ROOT / "tests" / "acceptance" / "test_plan_section_2_objectives.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))

    # method name -> {phase kwargs it passes to _unmet/_attempt/_unmeasurable}
    phases_by_method: dict[str, set[str]] = {}
    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef):
            continue
        for func in cls.body:
            if not isinstance(func, ast.FunctionDef):
                continue
            found: set[str] = set()
            for node in ast.walk(func):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id in {"_unmet", "_attempt", "_unmeasurable"}
                ):
                    for kw in node.keywords:
                        if kw.arg == "phase" and isinstance(kw.value, ast.Constant):
                            found.add(kw.value.value)
            if found:
                phases_by_method[f"{cls.name}::{func.name}"] = found

    mismatches: list[str] = []
    for clause_id, reason in unmet.items():
        match = re.match(r"phase (\d+) —", reason)
        assert match, f"{clause_id!r}'s ratchet reason does not start 'phase N — ': {reason!r}"
        ratchet_phase_number = int(match.group(1))

        code_phases = phases_by_method.get(clause_id)
        assert code_phases, (
            f"{clause_id!r} is unmet in the ratchet but calls no "
            "`_unmet`/`_attempt`/`_unmeasurable` with a `phase=` kwarg in "
            "test_plan_section_2_objectives.py — either the clause moved, or it "
            "no longer names its phase"
        )
        code_numbers = {phase_number[p] for p in code_phases}
        if code_numbers != {ratchet_phase_number}:
            mismatches.append(
                f"{clause_id}: ratchet says phase {ratchet_phase_number}, code passes "
                f"phase={sorted(code_phases)} (phase {sorted(code_numbers)})"
            )
    assert not mismatches, (
        "ratchet.json's unmet reason and the clause's own phase= kwarg disagree — "
        "one of the two restatements drifted:\n" + "\n".join(f"  - {m}" for m in mismatches)
    )


# ============================================================================
# Round 2 — independent adversarial review of alpha-engine-config-I9828's
# first PR (crucible-PR55) found two BLOCKING gaps and one SHOULD-FIX in the
# taxonomy itself. Each test below reproduces the finding before asserting
# the fix; see each docstring for what failed under the round-1 code.
# ============================================================================


def _unmeasurable_handler_types(source: str) -> list[frozenset[str] | None]:
    """For every call to `_unmeasurable(...)` in `source`, the exception type
    name(s) of its immediately-enclosing `except` handler.

    `None` means the call sits outside any `except` handler, OR the handler
    is bare (`except:`) — both are violations, never distinguished from each
    other because neither is allowed to reach `_unmeasurable` at all.

    Mirrors `crucible/gate.py::_acceptance_source_scan`'s shape (parse the
    committed source with `ast`, never import-and-run) for the same reason
    that module gives: this file's own docstring is explicit that
    collection, never execution, belongs on the PR path, and some of these
    clauses read live AWS.
    """

    def _names(node: ast.expr | None) -> frozenset[str] | None:
        if node is None:
            return None
        if isinstance(node, ast.Name):
            return frozenset({node.id})
        if isinstance(node, ast.Attribute):
            return frozenset({node.attr})
        if isinstance(node, ast.Tuple):
            names: set[str] = set()
            for elt in node.elts:
                resolved = _names(elt)
                if resolved is None:
                    return None
                names |= resolved
            return frozenset(names)
        return None

    tree = ast.parse(source)
    results: list[frozenset[str] | None] = []

    class _Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.stack: list[ast.ExceptHandler] = []

        def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
            self.stack.append(node)
            self.generic_visit(node)
            self.stack.pop()

        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Name) and node.func.id == "_unmeasurable":
                handler = self.stack[-1] if self.stack else None
                results.append(_names(handler.type) if handler is not None else None)
            self.generic_visit(node)

    _Visitor().visit(tree)
    return results


def test_the_except_handler_scanner_fires_on_a_bare_except_and_a_disallowed_type() -> None:
    """Self-test: the scanner must show its own detection firing, on synthetic
    input, before it is trusted against the real file (repo convention — "a
    detector nobody has made fail is a detector nobody knows works")."""
    outside_any_handler = textwrap.dedent(
        """
        def f():
            _unmeasurable("c", "r", exc, phase="phase0")
        """
    )
    bare_except = textwrap.dedent(
        """
        def f():
            try:
                x()
            except:
                _unmeasurable("c", "r", exc, phase="phase0")
        """
    )
    disallowed_type = textwrap.dedent(
        """
        def f():
            try:
                x()
            except Exception as exc:
                _unmeasurable("c", "r", exc, phase="phase0")
        """
    )
    allowed_type = textwrap.dedent(
        """
        def f():
            try:
                x()
            except NoCredentialsError as exc:
                _unmeasurable("c", "r", exc, phase="phase0")
        """
    )
    assert _unmeasurable_handler_types(outside_any_handler) == [None]
    assert _unmeasurable_handler_types(bare_except) == [None]
    assert _unmeasurable_handler_types(disallowed_type) == [frozenset({"Exception"})]
    assert _unmeasurable_handler_types(allowed_type) == [frozenset({"NoCredentialsError"})]


def _exception_provenance(source: str, allowed: frozenset[str]) -> dict[str, str]:
    """Allowed exception class name -> the module the objectives source
    imports it FROM. Derived from the source's own `from X import Y`
    statements (module- or function-level), never a restated list: the
    allowlist names classes, and the only fact about where each is raised
    that this file can honestly hold is where the source got it."""
    imported: dict[str, str] = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                imported[alias.asname or alias.name] = node.module
    missing = sorted(name for name in allowed if name not in imported)
    assert not missing, (
        f"allowed exception type(s) never imported by the objectives module: {missing} — "
        "the provenance of an exception nobody imports cannot be derived"
    )
    return {name: imported[name] for name in allowed}


def _packages_that_raise(module: str) -> frozenset[str]:
    """The import bindings a `try` body must call INTO for an exception
    defined in ``module`` to be reachable from it.

    The module itself and its top-level package always qualify. For
    `botocore.exceptions` the SDK front door `boto3` qualifies too — not as
    a restated fact but as one asserted here: a `boto3.client(...)` IS a
    `botocore.client.BaseClient`, and that class is what raises
    `botocore.exceptions.*` from `_make_api_call`. If that ever stops being
    true, this function stops accepting `boto3` and the guard goes red on
    the real handler, which is the correct direction to fail in.
    """
    top = module.split(".")[0]
    accepted = {module, top}
    if top == "botocore":
        import boto3
        import botocore.client

        client = boto3.client(
            "sts", region_name="us-east-1", aws_access_key_id="x", aws_secret_access_key="x"
        )
        assert isinstance(client, botocore.client.BaseClient), (
            "boto3.client no longer returns a botocore client; the derivation that lets a "
            "boto3 call count as a botocore-exception source no longer holds"
        )
        accepted.add("boto3")
    return frozenset(accepted)


def _unmeasurable_unreachable_handlers(source: str, provenance: dict[str, str]) -> list[str]:
    """One violation string per `except <allowed type>` handler that reaches
    `_unmeasurable` from a `try` body that calls NOTHING which could raise
    that type — the launder alpha-engine-config-I9910 demonstrated: an
    allowlisted `except NoCredentialsError:` around a pure assertion, which
    `_unmeasurable_handler_types` accepts because it grades the handler's
    DECLARED type and never looks at the body.

    "Could raise" is import-binding based, not name based: a call whose root
    name was bound by `from M import f` reaches `M`; one bound by `import P`
    (or `import P.Q`) reaches `P`. The body must reach one of
    `_packages_that_raise(module)` for the module the caught type is
    imported from. Disallowed types are ignored here — they are the other
    guard's violation, and reporting them twice would let a fix for one
    read as a fix for both.
    """
    tree = ast.parse(source)
    bindings: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bindings[(alias.asname or alias.name).split(".")[0]] = alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                bindings[alias.asname or alias.name] = node.module

    def _root(func: ast.expr) -> str | None:
        while isinstance(func, ast.Attribute):
            func = func.value
        return func.id if isinstance(func, ast.Name) else None

    def _reached(body: list[ast.stmt]) -> set[str]:
        reached: set[str] = set()
        for stmt in body:
            for node in ast.walk(stmt):
                if isinstance(node, ast.Call):
                    root = _root(node.func)
                    if root in bindings:
                        reached.add(bindings[root])
        return reached

    def _type_names(node: ast.expr | None) -> frozenset[str]:
        if isinstance(node, ast.Name):
            return frozenset({node.id})
        if isinstance(node, ast.Attribute):
            return frozenset({node.attr})
        if isinstance(node, ast.Tuple):
            names: set[str] = set()
            for elt in node.elts:
                names |= _type_names(elt)
            return frozenset(names)
        return frozenset()

    def _calls_unmeasurable(handler: ast.ExceptHandler) -> bool:
        return any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "_unmeasurable"
            for n in ast.walk(handler)
        )

    violations: list[str] = []

    class _Visitor(ast.NodeVisitor):
        def visit_Try(self, node: ast.Try) -> None:
            reached = _reached(node.body)
            for handler in node.handlers:
                if not _calls_unmeasurable(handler):
                    continue
                for type_name in sorted(_type_names(handler.type)):
                    module = provenance.get(type_name)
                    if module is None:
                        continue
                    if not (reached & _packages_that_raise(module)):
                        violations.append(
                            f"line {handler.lineno}: `except {type_name}` reaches _unmeasurable, "
                            f"but its try body calls nothing imported from {module} (it reaches "
                            f"{sorted(reached) or 'no import at all'}) — an outcome laundered "
                            "from unmet to unmeasurable, not a read that failed"
                        )
            self.generic_visit(node)

    _Visitor().visit(tree)
    return violations


_LAUNDER = textwrap.dedent(
    """
    from botocore.exceptions import NoCredentialsError
    from crucible.tags import StackNotAppliedError

    def f(record_property):
        try:
            assert 1 == 1
        except NoCredentialsError as exc:
            _unmeasurable("c", "r", exc, phase="phase0", record_property=record_property)
        try:
            assert 2 == 2
        except StackNotAppliedError as exc:
            _unmeasurable("c", "r", exc, phase="phase0", record_property=record_property)
    """
)

_REACHABLE = textwrap.dedent(
    """
    from botocore.exceptions import ClientError, NoCredentialsError
    from crucible.tags import StackNotAppliedError, audit_stack_tags

    def f(record_property):
        try:
            import boto3
            audit = audit_stack_tags(stack="s", cfn=boto3.client("cloudformation"))
        except StackNotAppliedError as exc:
            _unmeasurable("c", "r", exc, phase="phase0", record_property=record_property)
        except (NoCredentialsError, ClientError) as exc:
            _unmeasurable("c", "r", exc, phase="phase0", record_property=record_property)
    """
)


def test_the_reachability_scanner_fires_on_the_launder_the_declared_type_guard_accepts() -> None:
    """alpha-engine-config-I9910, reproduced exactly: an allowlisted `except`
    around a pure assertion. The declared-type guard ACCEPTS it (that is the
    gap), the reachability guard names it — once per handler, naming the
    module the body never called into."""
    module = _objectives()
    allowed = module._UNMEASURABLE_ALLOWED_EXCEPTIONS
    # The gap, stated: every handler's declared type is allowlisted, so the
    # existing guard sees nothing wrong with the launder.
    assert all(h is not None and h.issubset(allowed) for h in _unmeasurable_handler_types(_LAUNDER))
    provenance = _exception_provenance(
        _LAUNDER, frozenset({"NoCredentialsError", "StackNotAppliedError"})
    )
    violations = _unmeasurable_unreachable_handlers(_LAUNDER, provenance)
    assert len(violations) == 2, violations
    assert "NoCredentialsError" in violations[0] and "botocore.exceptions" in violations[0]
    assert "StackNotAppliedError" in violations[1] and "crucible.tags" in violations[1]


def test_the_reachability_scanner_accepts_a_body_that_calls_into_the_raising_module() -> None:
    """The real handler's shape: `audit_stack_tags` (imported from
    `crucible.tags`) covers `StackNotAppliedError`; `boto3.client(...)`
    covers the botocore types, through the boto3-is-a-botocore-client
    derivation in `_packages_that_raise`."""
    provenance = _exception_provenance(
        _REACHABLE, frozenset({"NoCredentialsError", "ClientError", "StackNotAppliedError"})
    )
    assert _unmeasurable_unreachable_handlers(_REACHABLE, provenance) == []


def test_the_provenance_is_derived_from_the_imports_not_restated() -> None:
    module = _objectives()
    provenance = _exception_provenance(
        OBJECTIVES_SOURCE.read_text(encoding="utf-8"), module._UNMEASURABLE_ALLOWED_EXCEPTIONS
    )
    # Every allowed type resolves to SOME module, and the modules are the
    # ones the source actually imports from — asserted as a property of the
    # mapping rather than as a second copy of it.
    assert set(provenance) == set(module._UNMEASURABLE_ALLOWED_EXCEPTIONS)
    assert all(module_name for module_name in provenance.values())
    with pytest.raises(AssertionError, match="never imported"):
        _exception_provenance("x = 1\n", frozenset({"NoCredentialsError"}))


def test_every_allowlisted_handler_in_the_objectives_is_reachable_from_its_try_body() -> None:
    """alpha-engine-config-I9910: the declared-type guard below makes an
    `unmeasurable` outcome come from an ALLOWLISTED handler; this one makes
    it come from a handler whose `try` body could actually have raised what
    it catches. Together: a clause can be moved from unmet to unmeasurable
    only by a read that failed, never by wrapping an assertion in an
    allowlisted `except`."""
    module = _objectives()
    source = OBJECTIVES_SOURCE.read_text(encoding="utf-8")
    provenance = _exception_provenance(source, module._UNMEASURABLE_ALLOWED_EXCEPTIONS)
    violations = _unmeasurable_unreachable_handlers(source, provenance)
    assert not violations, "\n".join(violations)


def test_unmeasurable_is_only_called_from_an_allowed_except_handler() -> None:
    """BLOCKING, round-2 review finding 1: without this, an author can turn
    ANY code bug into "unmeasurable" by wrapping `_unmeasurable(...)` in
    `except Exception` (or calling it unconditionally) — `check_reading.py`
    has no way to tell that apart from a genuine read failure, because both
    of its inputs (the JUnit reading and the ratchet) would agree if the
    author also edited `ratchet.json` in the same PR. This is the structural
    half of the fix: `_unmeasurable` can only be reached from a handler whose
    caught type(s) are in the declared allowlist, checked here on every PR
    (this file is in the FOUNDATION suite, which — unlike the acceptance job,
    per review finding 1's `ci.yml:159` note — DOES run on `pull_request`).
    """
    module = _objectives()
    allowed = module._UNMEASURABLE_ALLOWED_EXCEPTIONS
    handler_types = _unmeasurable_handler_types(OBJECTIVES_SOURCE.read_text(encoding="utf-8"))
    assert handler_types, "no _unmeasurable calls found in the source — nothing to check"
    violations = [h for h in handler_types if h is None or not h.issubset(allowed)]
    assert not violations, (
        f"{len(violations)} call(s) to _unmeasurable are not reachable only from an "
        f"allowed except handler (allowed: {sorted(allowed)}); found: {violations}. "
        "Either the call sits outside any except handler / a bare except, or it "
        "catches a type not in _UNMEASURABLE_ALLOWED_EXCEPTIONS."
    )


def test_unmeasurable_raises_on_an_exception_type_outside_the_allowlist() -> None:
    """Redundant runtime guard inside `_unmeasurable` itself (belt, not the
    braces — the AST test above is the real enforcement, since a call site
    is trusted to pass the exception it actually caught). Shows the guard
    firing: calling with a `TypeError` — not in the allowlist — raises
    `TypeError`, not `pytest.fail`."""
    module = _objectives()
    calls: list[tuple[str, object]] = []
    with pytest.raises(TypeError, match="not in the declared allowlist"):
        module._unmeasurable(
            "clause", "requirement", TypeError("boom"), phase="phase0", record_property=calls.append
        )
    assert calls == [], "no property should be recorded for a rejected exception type"


class _Recorder:
    """Collects `record_property(name, value)` calls, pytest's own fixture
    shape, without needing a live pytest run."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def __call__(self, name: str, value: object) -> None:
        self.calls.append((name, value))

    def get(self, name: str) -> object | None:
        for recorded_name, value in self.calls:
            if recorded_name == name:
                return value
        return None


def _run_test_cost(module, *, audit_stack_tags, recorder: _Recorder) -> None:
    """Invoke the real `TestCost.test_every_v2_resource_is_tagged_for_cost_attribution`
    with `crucible.tags.audit_stack_tags` replaced, so the method's own
    except-clause structure is exercised directly rather than re-implemented
    here as a second copy of the contract. Any exception the method raises
    (or fails to catch) propagates to the caller — this does not swallow.

    `boto3.client` is ALSO stubbed. Reproduced live (2026-09-03): on a
    runner with no AWS region configured at all — every GitHub Actions
    runner, unlike this laptop's default profile — the real
    `boto3.client("cloudformation")` call inside the method's `try` block
    raises `NoRegionError` before `audit_stack_tags` (the thing actually
    being exercised) is ever reached, so every case except the one
    expecting `NoRegionError` itself failed in CI with `blocked_on_class ==
    "NoRegionError"` regardless of what was injected. The client
    construction is not what these tests are about; stubbing it removes
    the environment as a variable entirely, matching what the mocked
    `audit_stack_tags` already does for the call it wraps.
    """
    import boto3

    import crucible.tags as tags_module

    original_audit = tags_module.audit_stack_tags
    original_client = boto3.client
    tags_module.audit_stack_tags = audit_stack_tags
    boto3.client = lambda *args, **kwargs: object()
    try:
        module.TestCost().test_every_v2_resource_is_tagged_for_cost_attribution(recorder)
    finally:
        tags_module.audit_stack_tags = original_audit
        boto3.client = original_client


def test_a_typeerror_from_the_audit_is_not_caught_and_is_not_unmeasurable() -> None:
    """BLOCKING, round-2 review finding 2, reproduced exactly as named: an
    injected `TypeError` from `audit_stack_tags` used to be caught by the
    round-1 bare `except Exception` and read as UNMEASURABLE. It must now
    propagate uncaught — a pytest ERROR, not a classified outcome — and
    record no property.
    """
    module = _objectives()
    recorder = _Recorder()

    def _raise(**kwargs):
        raise TypeError("audit_stack_tags: unexpected keyword")

    with pytest.raises(TypeError, match="unexpected keyword"):
        _run_test_cost(module, audit_stack_tags=_raise, recorder=recorder)
    assert recorder.calls == [], "a code bug must record no outcome property"


def test_a_clienterror_with_a_disallowed_code_is_not_swallowed() -> None:
    """A `ClientError` whose code is NOT in the access/auth allowlist (e.g. a
    throttle, a malformed request) is a real API problem, not an environment
    one — it must re-raise, exactly like the `TypeError` case."""
    from botocore.exceptions import ClientError

    module = _objectives()
    recorder = _Recorder()

    def _raise(**kwargs):
        raise ClientError(
            {"Error": {"Code": "ThrottlingException", "Message": "slow down"}}, "ListStackResources"
        )

    with pytest.raises(ClientError):
        _run_test_cost(module, audit_stack_tags=_raise, recorder=recorder)
    assert recorder.calls == [], "a non-auth ClientError code must record no outcome property"


@pytest.mark.parametrize(
    ("code", "expected_class"),
    [
        ("AccessDenied", "ClientError"),
        ("AccessDeniedException", "ClientError"),
        ("UnauthorizedOperation", "ClientError"),
        ("ExpiredToken", "ClientError"),
        ("InvalidClientTokenId", "ClientError"),
    ],
)
def test_a_clienterror_with_an_access_denied_code_is_unmeasurable(
    code: str, expected_class: str
) -> None:
    """The laptop's real failure mode (measured: `AccessDenied` on
    `cloudformation:ListStackResources`) and its four siblings all classify
    as UNMEASURABLE, with `blocked_on_class` recording the exception TYPE
    (`ClientError`), not the AWS error code — matching what
    `type(exc).__name__` actually is for every one of these."""
    from botocore.exceptions import ClientError

    module = _objectives()
    recorder = _Recorder()

    def _raise(**kwargs):
        raise ClientError({"Error": {"Code": code, "Message": "denied"}}, "ListStackResources")

    with pytest.raises(pytest.fail.Exception, match="UNMEASURABLE"):
        _run_test_cost(module, audit_stack_tags=_raise, recorder=recorder)
    assert recorder.get("outcome") == "unmeasurable"
    assert recorder.get("blocked_on_class") == expected_class


@pytest.mark.parametrize(
    ("build_exc", "expected_class"),
    [
        (
            lambda: __import__(
                "crucible.tags", fromlist=["StackNotAppliedError"]
            ).StackNotAppliedError("no stack"),
            "StackNotAppliedError",
        ),
        (
            lambda: __import__(
                "botocore.exceptions", fromlist=["NoCredentialsError"]
            ).NoCredentialsError(),
            "NoCredentialsError",
        ),
        (
            lambda: __import__("botocore.exceptions", fromlist=["NoRegionError"]).NoRegionError(),
            "NoRegionError",
        ),
        (
            lambda: __import__(
                "botocore.exceptions", fromlist=["EndpointConnectionError"]
            ).EndpointConnectionError(
                endpoint_url="https://cloudformation.us-east-1.amazonaws.com"
            ),
            "EndpointConnectionError",
        ),
    ],
)
def test_every_allowed_exception_type_classifies_unmeasurable_with_its_own_class_name(
    build_exc, expected_class: str
) -> None:
    """Every type in `_UNMEASURABLE_ALLOWED_EXCEPTIONS` actually reaches
    UNMEASURABLE through the real method, each recording ITS OWN type name —
    not a shared constant — so a future family drift (finding 3) has
    something true to compare against."""
    module = _objectives()
    recorder = _Recorder()

    def _raise(**kwargs):
        raise build_exc()

    with pytest.raises(pytest.fail.Exception, match="UNMEASURABLE"):
        _run_test_cost(module, audit_stack_tags=_raise, recorder=recorder)
    assert recorder.get("outcome") == "unmeasurable"
    assert recorder.get("blocked_on_class") == expected_class


def test_message_text_alone_no_longer_classifies_as_unmeasurable(
    tmp_path: Path, ratchet: Path
) -> None:
    """SHOULD-FIX, round-2 review finding 3, both reproductions in one test:
    an `AssertionError` that happens to quote the marker word, and the
    marker surviving inside an unrelated `<failure>` body. Round 1 classified
    both as unmeasurable via substring search; round 2 reads `<properties>`
    only, so neither has any effect without the property being present.
    """
    report = tmp_path / "r.xml"
    report.write_text(
        textwrap.dedent(
            """\
            <?xml version="1.0"?><testsuites><testsuite>
            <testcase classname="tests.acceptance.mod.T" name="a">
            <failure message="AssertionError: expected UNMEASURABLE — clause, got MET">
            Traceback shows an old UNMEASURABLE — clause failure body quoted for context
            </failure>
            </testcase>
            </testsuite></testsuites>
            """
        )
    )
    reading = _checker().read_report(report)
    assert reading.unmeasurable == set(), (
        "message text alone must not classify anything as unmeasurable"
    )
    assert reading.unmet == {"T::a"}


def test_a_blocked_on_class_family_drift_is_red(
    tmp_path: Path, ratchet_with_unmeasurable: Path
) -> None:
    """SHOULD-FIX, round-2 review finding 3's second half: T::d stays
    unmeasurable in both ratchet and reading, but the OBSERVED exception type
    changed from what the ratchet commits (`NoCredentialsError` ->
    `ClientError`, e.g. credentials went from entirely absent to merely
    insufficient) — a real change in the failure story that the earlier
    reclassification checks cannot see, because the clause never left
    `unmeasurable`."""
    report = _report(
        tmp_path / "r.xml",
        met=["T::b", "T::c"],
        unmet=["T::a"],
        unmeasurable={"T::d": "ClientError"},
    )
    result = _run(report, ratchet_with_unmeasurable)
    assert result.returncode == 1
    assert "blocked_on_class" in result.stderr
    assert "T::d" in result.stderr
    assert "NoCredentialsError" in result.stderr
    assert "ClientError" in result.stderr


def test_a_matching_blocked_on_class_is_still_green(
    tmp_path: Path, ratchet_with_unmeasurable: Path
) -> None:
    """The family-drift check must not false-positive when nothing changed —
    T::d's `blocked_on_class` in the reading matches the ratchet exactly."""
    report = _report(
        tmp_path / "r.xml",
        met=["T::b", "T::c"],
        unmet=["T::a"],
        unmeasurable={"T::d": "NoCredentialsError"},
    )
    result = _run(report, ratchet_with_unmeasurable)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "bad_entry",
    [
        {"reason": "   ", "blocked_on_class": "ClientError", "last_moved": "2026-09-02"},
        {"reason": "why", "blocked_on_class": "   ", "last_moved": "2026-09-02"},
        {"reason": "why", "blocked_on_class": "ClientError", "last_moved": "09-02-2026"},
        {"reason": "why", "blocked_on_class": "ClientError", "last_moved": ""},
        {"reason": "why", "blocked_on_class": "ClientError"},
    ],
)
def test_a_malformed_unmeasurable_entry_is_red_and_names_the_field(
    tmp_path: Path, bad_entry: dict
) -> None:
    """Round 2: `unmeasurable` values are objects now (`reason`,
    `blocked_on_class`, `last_moved`) — each field gets the same
    fail-loud-with-the-field-named treatment as every other malformed shape
    in this grader (principle 7: a malformed input is an absence)."""
    ratchet = tmp_path / "ratchet.json"
    ratchet.write_text(
        json.dumps({"unmet": {"T::a": "phase 2"}, "unmeasurable": {"T::a": bad_entry}, "met": []})
    )
    report = _report(
        tmp_path / "r.xml", met=[], unmet=["T::a"], unmeasurable={"T::a": "ClientError"}
    )
    result = _run(report, ratchet)
    assert result.returncode == 1
    assert "Traceback" not in result.stderr


# ── `--write-json` — the producer contract, alpha-engine-config-I9902 ───────
#
# `crucible.keys.acceptance_reading_key`'s docstring declares the shape
# `crucible.morning._acceptance_line` parses:
#   {"met": int, "unmet": int, "unmeasurable": int, "commit": str, "measured_at": str}
# These tests assert the checker actually emits exactly that shape, on both a
# clean reading and a moved one — the moved case is the one that matters,
# since plan §12 rule 3 makes a RED count the progress figure, and a
# publisher that only wrote on success would be silent on every run that
# moved.


def test_write_json_emits_the_producer_contract(tmp_path: Path, ratchet: Path) -> None:
    report = _report(tmp_path / "r.xml", met=["T::b", "T::c"], unmet=["T::a"])
    out = tmp_path / "reading.json"
    result = _run_with_args(report, ratchet, "--write-json", str(out), "--commit", "deadbeef")
    assert result.returncode == 0, result.stderr
    document = json.loads(out.read_text())
    assert set(document) == {"met", "unmet", "unmeasurable", "commit", "measured_at"}
    assert document["met"] == 2
    assert document["unmet"] == 1
    assert document["unmeasurable"] == 0
    assert document["commit"] == "deadbeef"
    # ISO-8601, parseable — the exact format is not the contract, but a
    # non-parseable timestamp would be.
    from datetime import datetime as _dt

    _dt.fromisoformat(document["measured_at"])


def test_write_json_counts_unmeasurable_separately_from_unmet(
    tmp_path: Path, ratchet_with_unmeasurable: Path
) -> None:
    report = _report(
        tmp_path / "r.xml",
        met=["T::b", "T::c"],
        unmet=["T::a"],
        unmeasurable={"T::d": "NoCredentialsError"},
    )
    out = tmp_path / "reading.json"
    result = _run_with_args(
        report, ratchet_with_unmeasurable, "--write-json", str(out), "--commit", "abc123"
    )
    assert result.returncode == 0, result.stderr
    document = json.loads(out.read_text())
    # plain_unmet (T::a) is 1; unmeasurable (T::d) is its own count, never
    # folded into unmet — the same distinction `crucible.tags`'s own
    # docstring and this grader's module docstring both make.
    assert document["unmet"] == 1
    assert document["unmeasurable"] == 1


def test_write_json_still_writes_on_a_moved_reading(tmp_path: Path, ratchet: Path) -> None:
    """The publish must NOT be skipped when the reading is red — a regressed
    clause is exactly the run whose reading matters most to publish."""
    report = _report(tmp_path / "r.xml", met=["T::c"], unmet=["T::a", "T::b"])
    out = tmp_path / "reading.json"
    result = _run_with_args(report, ratchet, "--write-json", str(out), "--commit", "cafef00d")
    assert result.returncode == 1, "a regression must still fail the run"
    assert out.exists(), "the reading must be published even when it moved"
    document = json.loads(out.read_text())
    assert document["met"] == 1
    assert document["unmet"] == 2
    assert document["commit"] == "cafef00d"


def test_write_json_falls_back_to_github_sha_variable_when_commit_flag_absent(
    tmp_path: Path, ratchet: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The workflow passes `--commit "${GITHUB_SHA}"` explicitly; the fallback
    exists for a bare local invocation run from inside a checkout, where the
    variable is already set and re-typing the sha is friction, not safety."""
    import os as _os

    report = _report(tmp_path / "r.xml", met=["T::b", "T::c"], unmet=["T::a"])
    out = tmp_path / "reading.json"
    checker = ratchet.parent / "check_reading.py"
    checker.write_text(CHECKER.read_text())
    variables = dict(_os.environ)
    variables["GITHUB_SHA"] = "envsha"
    result = subprocess.run(
        [sys.executable, str(checker), "--write-json", str(out), str(report)],
        capture_output=True,
        text=True,
        check=False,
        env=variables,
    )
    assert result.returncode == 0, result.stderr
    document = json.loads(out.read_text())
    assert document["commit"] == "envsha"


def test_write_json_with_no_path_argument_at_all_fails_with_usage(
    tmp_path: Path, ratchet: Path
) -> None:
    """`--write-json` with nothing after it at all (not even a swallowed
    report path) is the genuine "requires a path argument" case."""
    checker = ratchet.parent / "check_reading.py"
    checker.write_text(CHECKER.read_text())
    result = subprocess.run(
        [sys.executable, str(checker), "--write-json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "requires a path argument" in result.stderr


def test_write_json_swallowing_the_report_path_fails_with_usage(
    tmp_path: Path, ratchet: Path
) -> None:
    """`--write-json` as the LAST argument consumes the report path as its
    own value, leaving no positional report argument — the usage message,
    not a crash, is the correct failure here."""
    report = _report(tmp_path / "r.xml", met=["T::b", "T::c"], unmet=["T::a"])
    result = _run_with_args(report, ratchet, "--write-json")
    assert result.returncode == 1
    assert "usage:" in result.stderr
