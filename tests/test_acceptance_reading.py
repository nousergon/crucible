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
    path.write_text(json.dumps({"collected": 3, "unmet": {"T::a": "phase 2"}}))
    return path


def test_the_committed_ratchet_is_wellformed() -> None:
    """The file CI grades against must parse and carry both fields."""
    data = json.loads(RATCHET.read_text())
    assert isinstance(data["collected"], int)
    assert isinstance(data["unmet"], dict)
    assert len(data["unmet"]) <= data["collected"]
    for clause, reason in data["unmet"].items():
        assert "::" in clause, f"{clause!r} is not a `Class::method` clause id"
        assert reason.strip(), f"{clause} carries no reason"


def test_the_committed_ratchet_matches_the_real_suite() -> None:
    """`collected` is checkable on the PR path, so check it.

    Otherwise deleting a clause and editing `collected` to match is green on
    the PR and green on main — the ratchet grading itself. Collection imports
    the modules and executes no test body, so this touches no live AWS.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/acceptance",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    collected = sum(
        int(line.rsplit(":", 1)[1])
        for line in result.stdout.splitlines()
        if line.startswith("tests/acceptance/") and line.rsplit(":", 1)[-1].strip().isdigit()
    )
    want = json.loads(RATCHET.read_text())["collected"]
    assert collected == want, (
        f"tests/acceptance collects {collected} clauses; ratchet.json says {want}. "
        "Update the ratchet in the PR that changes the suite."
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


def test_a_changed_collected_count_is_red(tmp_path: Path, ratchet: Path) -> None:
    """A clause added, removed, or failing to import."""
    report = _report(tmp_path / "r.xml", met=["T::b"], unmet=["T::a"])
    result = _run(report, ratchet)
    assert result.returncode == 1
    assert "collected" in result.stderr


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
    assert "collected" in result.stderr


def test_the_module_path_is_not_part_of_a_clause_id() -> None:
    """Renaming a file is not a moved reading."""
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("check_reading", CHECKER)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.clause_id("tests.acceptance.old_name.TestX", "test_y") == "TestX::test_y"
    assert module.clause_id("tests.acceptance.new_name.TestX", "test_y") == "TestX::test_y"
