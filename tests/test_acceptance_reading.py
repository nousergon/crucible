"""The acceptance arm signals by grading its READING, so the grader is tested.

`tests/acceptance/check_reading.py` is the only thing standing between "the
suite is red by design" and "the suite is red because someone broke it". Every
outcome it can produce is asserted here, including the two that look like
success and are not: an absent report and an empty one.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

CHECKER = Path(__file__).resolve().parents[1] / "tests" / "acceptance" / "check_reading.py"
RATCHET = Path(__file__).resolve().parents[1] / "tests" / "acceptance" / "ratchet.json"


def _report(path: Path, *, passed: int, failed: int) -> Path:
    cases = "".join(f'<testcase classname="T" name="ok{i}"/>' for i in range(passed))
    cases += "".join(
        f'<testcase classname="T" name="bad{i}"><failure>UNMET</failure></testcase>'
        for i in range(failed)
    )
    path.write_text(f'<?xml version="1.0"?><testsuites><testsuite>{cases}</testsuite></testsuites>')
    return path


def _run(report: Path, *, ratchet: Path | None = None) -> subprocess.CompletedProcess[str]:
    checker = CHECKER
    if ratchet is not None:
        # The checker reads ratchet.json from its own directory, so a test
        # ratchet means a checker copy beside it.
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
    path.write_text(json.dumps({"met": 21, "collected": 24}))
    return path


def test_the_committed_ratchet_is_wellformed() -> None:
    """The file CI grades against must parse and carry both numbers."""
    data = json.loads(RATCHET.read_text())
    assert isinstance(data["met"], int)
    assert isinstance(data["collected"], int)
    assert 0 <= data["met"] <= data["collected"]


def test_an_unchanged_reading_is_green(tmp_path: Path, ratchet: Path) -> None:
    report = _report(tmp_path / "r.xml", passed=21, failed=3)
    result = _run(report, ratchet=ratchet)
    assert result.returncode == 0, result.stderr
    assert "unchanged" in result.stdout


def test_a_regression_is_red(tmp_path: Path, ratchet: Path) -> None:
    report = _report(tmp_path / "r.xml", passed=20, failed=4)
    result = _run(report, ratchet=ratchet)
    assert result.returncode == 1
    assert "REGRESSION" in result.stderr


def test_unrecorded_progress_is_red(tmp_path: Path, ratchet: Path) -> None:
    """A clause going green without the ratchet moving is not a pass.

    Otherwise the number nobody updates stops meaning anything, and the next
    regression is measured against a floor that was never true.
    """
    report = _report(tmp_path / "r.xml", passed=22, failed=2)
    result = _run(report, ratchet=ratchet)
    assert result.returncode == 1
    assert "not recorded" in result.stderr


def test_a_changed_collected_count_is_red(tmp_path: Path, ratchet: Path) -> None:
    """A clause added, removed, or failing to import."""
    report = _report(tmp_path / "r.xml", passed=21, failed=2)
    result = _run(report, ratchet=ratchet)
    assert result.returncode == 1
    assert "collected" in result.stderr


def test_a_missing_report_is_red_not_green(tmp_path: Path, ratchet: Path) -> None:
    """Absence is never a pass (principle 7)."""
    result = _run(tmp_path / "does-not-exist.xml", ratchet=ratchet)
    assert result.returncode == 1
    assert "dark, not green" in result.stderr


def test_an_empty_report_is_red_not_green(tmp_path: Path, ratchet: Path) -> None:
    report = tmp_path / "r.xml"
    report.write_text('<?xml version="1.0"?><testsuites><testsuite></testsuite></testsuites>')
    result = _run(report, ratchet=ratchet)
    assert result.returncode == 1
    assert "collected no tests" in result.stderr


def test_an_unparseable_report_is_red_not_green(tmp_path: Path, ratchet: Path) -> None:
    report = tmp_path / "r.xml"
    report.write_text("this is not xml <<<")
    result = _run(report, ratchet=ratchet)
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
    result = _run(report, ratchet=ratchet)
    assert result.returncode == 1
    assert "collected" in result.stderr
