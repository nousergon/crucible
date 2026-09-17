"""`crucible review.record` — the review verdict's own registered CLI job.

Normative source: `alpha-engine-config-I10968`. The verdict
`crucible.gate._clause_independently_reviewed` grades was written by a raw
`store.put_bytes` reached from `crucible/review.py`'s own `__main__`, so the
producer of phase-exit evidence filed no run manifest on either path — and a
REFUSED review filed nothing anywhere, making "a review was attempted and
refused" indistinguishable from "no review was attempted".

`tests/test_review.py` still owns the document's shape and the independence
comparison. This module owns the JOB: the manifest, the refusal's exit
convention, and the dry run.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from crucible.cli import main as cli_main
from crucible.keys import manifest_key
from crucible.review import REVIEW_RECORD_JOB
from crucible.runner import CODE_SHA_ENV
from crucible.store import LocalStore

DAY = dt.date(2026, 8, 28)
AUTHOR = "session_01AuthorAAAAAAAA"
REVIEWER = "session_01ReviewerBBBBB"
HEAD_SHA = "b" * 40
_FAKE_SHA = "a" * 40


@pytest.fixture(autouse=True)
def _fixed_code_sha(monkeypatch):
    monkeypatch.setenv(CODE_SHA_ENV, _FAKE_SHA)


def _commits(tmp_path) -> str:
    path = tmp_path / "commits.json"
    path.write_text(
        json.dumps(
            [
                {
                    "sha": "c" * 40,
                    "commit": {
                        "message": f"feat: a change\n\nClaude-Session: {AUTHOR}",
                        "author": {"email": "brian@nousergon.ai"},
                        "committer": {"email": "brian@nousergon.ai"},
                    },
                    "author": {"login": "cipher813"},
                    "committer": {"login": "cipher813"},
                }
            ]
        ),
        encoding="utf-8",
    )
    return str(path)


def _argv(tmp_path, *, reviewer: str = REVIEWER, extra: tuple[str, ...] = ()) -> list[str]:
    return [
        REVIEW_RECORD_JOB,
        "--date",
        DAY.isoformat(),
        "--store",
        str(tmp_path / "store"),
        "--commits",
        _commits(tmp_path),
        "--reviewer",
        reviewer,
        "--phase",
        "phase1",
        "--verdict",
        "pass",
        "--head-sha",
        HEAD_SHA,
        "--pr-number",
        "48",
        "--summary",
        "no findings",
        *extra,
    ]


def _manifest(tmp_path) -> dict:
    store = LocalStore(tmp_path / "store")
    return json.loads(store.get_bytes(manifest_key(REVIEW_RECORD_JOB, DAY.isoformat())))


class TestTheManifest:
    def test_a_recorded_verdict_names_its_artifact_in_its_own_manifest(self, tmp_path) -> None:
        assert cli_main(_argv(tmp_path)) == 0
        manifest = _manifest(tmp_path)
        assert manifest["status"] == "ok", manifest.get("reason")
        outputs = [row["key"] for row in manifest["outputs"]]
        assert len(outputs) == 1
        assert outputs[0].endswith(f"/{REVIEWER.lower()}/pass.json")

    def test_a_refused_review_is_a_failed_manifest_not_an_absence(self, tmp_path) -> None:
        """The gap this job closes. A self-review used to write nothing at all,
        so nothing anywhere distinguished a refusal from a review that was
        never dispatched."""
        assert cli_main(_argv(tmp_path, reviewer=AUTHOR)) == 2
        manifest = _manifest(tmp_path)
        assert manifest["status"] == "failed"
        assert "independent of the author" in manifest["reason"]
        assert manifest["outputs"] == []

    def test_the_refusal_keeps_the_workflows_exit_convention(self, tmp_path, capsys) -> None:
        """Exit 2 and one `::error::` line, not a traceback: the recording
        workflow's step reads both. The manifest carries the full record."""
        assert cli_main(_argv(tmp_path, reviewer=AUTHOR)) == 2
        assert "::error::" in capsys.readouterr().err


class TestDryRun:
    """The row `tests/test_cli_and_alerts.py::TestDryRunNeverWrites` excludes
    this job from, asserted directly."""

    def test_a_dry_run_builds_the_document_and_writes_nothing(self, tmp_path) -> None:
        assert cli_main(_argv(tmp_path, extra=("--dry-run",))) == 0
        store_dir = tmp_path / "store"
        assert not store_dir.exists() or not list(LocalStore(store_dir).list_keys(""))

    def test_a_dry_run_still_refuses_a_self_review(self, tmp_path) -> None:
        """The rehearsal fails the way the run fails
        (`alpha-engine-config-I11012`): the independence comparison is the job
        body, and a dry run executes it."""
        assert cli_main(_argv(tmp_path, reviewer=AUTHOR, extra=("--dry-run",))) == 2
