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


def _report(path: Path, *, met: list[str], unmet: list[str]) -> Path:
    cases = "".join(
        f'<testcase classname="tests.acceptance.mod.{name.split("::")[0]}" '
        f'name="{name.split("::")[1]}"/>'
        for name in met
    )
    cases += "".join(
        f'<testcase classname="tests.acceptance.mod.{name.split("::")[0]}" '
        f'name="{name.split("::")[1]}"><failure>UNMET</failure></testcase>'
        for name in unmet
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
    want = set(ratchet["met"]) | set(ratchet["unmet"])
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


def test_the_module_path_is_not_part_of_a_clause_id() -> None:
    """Renaming a file is not a moved reading."""
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("check_reading", CHECKER)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.clause_id("tests.acceptance.old_name.TestX", "test_y") == "TestX::test_y"
    assert module.clause_id("tests.acceptance.new_name.TestX", "test_y") == "TestX::test_y"
