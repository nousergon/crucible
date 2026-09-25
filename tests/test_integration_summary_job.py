"""`crucible test.integration` — the integration tier's own registered CLI job.

Normative source: `alpha-engine-config-I10459`. `.github/workflows/
integration-nightly.yml` used to write a plain `integration_summary.v1`
JSON object via a raw `store.put_bytes` call in the workflow itself — never
a `run_manifest.v2` document, so it was not manifest-or-it-didn't-happen and
carried no `components.yaml` row. This job replaces that hand-rolled write:
a thin handler that shells out to `pytest tests/integration` and reports
pass/fail through `crucible.runner.run_job`, writing a real
`runs/test.integration/{trading_day}/run.json` like every other job.

Tests here fake the subprocess call (never actually re-running
`tests/integration` from inside the unit suite — that tier needs real S3 and
real ArcticDB and is not collected here) and assert the MANIFEST's shape.
"""

from __future__ import annotations

import argparse
import datetime as dt

import pytest

import crucible.integration_summary as integration_summary_module
from crucible.integration_summary import INTEGRATION_TEST_JOB, integration_test_handler
from crucible.runner import CODE_SHA_ENV
from crucible.store import LocalStore
from tests.support.manifests import only_manifest

DAY = dt.date(2026, 8, 28)

#: `subprocess` is a process-wide singleton module object, so patching it
#: through ANY reference (including `integration_summary_module.subprocess`)
#: patches it for every caller in the process — including
#: `crucible.runner.resolve_code_sha`'s own real `git rev-parse HEAD` call,
#: made later in the SAME `run_job` invocation. Setting `$CRUCIBLE_CODE_SHA`
#: short-circuits that call entirely (`resolve_code_sha` reads it first and
#: never reaches `subprocess.run` when it is set), which is the only clean
#: way to fake one subprocess call site without faking the other.
_FAKE_SHA = "a" * 40


@pytest.fixture(autouse=True)
def _fixed_code_sha(monkeypatch):
    monkeypatch.setenv(CODE_SHA_ENV, _FAKE_SHA)


class _FakeCompletedProcess:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_a_passing_pytest_run_writes_an_ok_manifest(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        integration_summary_module.subprocess,
        "run",
        lambda *a, **k: _FakeCompletedProcess(0, stdout="8 passed in 3.21s"),
    )
    store = LocalStore(tmp_path / "store")
    integration_test_handler(
        argparse.Namespace(
            trading_day=DAY, store=str(tmp_path / "store"), dry_run=False, run_mode=None
        )
    )
    manifest = only_manifest(store, INTEGRATION_TEST_JOB, DAY.isoformat())[1]
    assert manifest["status"] == "ok", manifest.get("reason")
    assert manifest["job"] == INTEGRATION_TEST_JOB


def test_a_failing_pytest_run_writes_a_failed_manifest_and_raises(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        integration_summary_module.subprocess,
        "run",
        lambda *a, **k: _FakeCompletedProcess(1, stdout="1 failed, 7 passed in 4.02s"),
    )
    store = LocalStore(tmp_path / "store")
    with pytest.raises(RuntimeError, match="tests/integration failed"):
        integration_test_handler(
            argparse.Namespace(
                trading_day=DAY, store=str(tmp_path / "store"), dry_run=False, run_mode=None
            )
        )
    manifest = only_manifest(store, INTEGRATION_TEST_JOB, DAY.isoformat())[1]
    assert manifest["status"] == "failed"
    assert "1 failed" in manifest["reason"]


def test_the_pytest_invocation_targets_tests_integration(tmp_path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _FakeCompletedProcess(0, stdout="0 passed")

    monkeypatch.setattr(integration_summary_module.subprocess, "run", _fake_run)
    integration_test_handler(
        argparse.Namespace(
            trading_day=DAY, store=str(tmp_path / "store"), dry_run=False, run_mode=None
        )
    )
    cmd = captured["cmd"]
    assert "tests/integration" in cmd
    assert "pytest" in "".join(str(part) for part in cmd)
