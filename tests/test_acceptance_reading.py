"""The acceptance arm signals by grading its READING, so the grader is tested.

`tests/acceptance/check_reading.py` is the only thing standing between "the
suite is red by design" and "the suite is red because someone broke it". Every
outcome it can produce is asserted here, including the three that look like
success and are not: an absent report, an empty one, and a regression offset by
a gain.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "tests" / "acceptance" / "check_reading.py"
RATCHET = ROOT / "tests" / "acceptance" / "ratchet.json"


def _report(
    path: Path, *, met: list[str], unmet: list[str], unmeasurable: list[str] | None = None
) -> Path:
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
        f'name="{name.split("::")[1]}"><failure message="Failed: UNMEASURABLE — {name}">'
        f"UNMEASURABLE</failure></testcase>"
        for name in (unmeasurable or [])
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


@pytest.fixture
def ratchet(tmp_path: Path) -> Path:
    path = tmp_path / "ratchet.json"
    path.write_text(json.dumps({"unmet": {"T::a": "phase 2"}, "met": ["T::b", "T::c"]}))
    return path


@pytest.fixture
def ratchet_with_unmeasurable(tmp_path: Path) -> Path:
    """`unmeasurable` is a SUBSET of `unmet`'s keys: T::d is listed in both,
    since `crucible/gate.py`'s phase-0 clause reads `met | unmet` as the full
    clause set and does not know about the finer `unmeasurable` bucket."""
    path = tmp_path / "ratchet.json"
    path.write_text(
        json.dumps(
            {
                "unmet": {"T::a": "phase 2", "T::d": "no AWS credentials in this environment"},
                "unmeasurable": {"T::d": "no AWS credentials in this environment"},
                "met": ["T::b", "T::c"],
            }
        )
    )
    return path


def _checker():
    """Import `check_reading` as a module, for the model-level assertions."""
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("check_reading", CHECKER)
    assert spec and spec.loader
    module = module_from_spec(spec)
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
    """alpha-engine-config-I9828: the whole point of the marker.

    A clause whose message carries the `UNMEASURABLE — ` marker (what
    `_unmeasurable` in `test_plan_section_2_objectives.py` writes) is
    classified into `Reading.unmeasurable`, a SUBSET of `Reading.unmet` (so
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
