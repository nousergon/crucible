"""`crucible acceptance.publish` — the acceptance reading's own registered job.

Normative source: `alpha-engine-config-I10968`. `.github/workflows/ci.yml`
shipped the plan §12 rule-3 reading with a raw `aws s3 cp` — never through
`crucible.store.Store`, so the ONE producer of the figure the plan names as
progress filed no run manifest at all, while `crucible report.morning`,
`crucible board` and phase 0's gate clause read the artifact as evidence.

Every test here was seen failing before the module existed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json

import pytest

from crucible.acceptance_publish import (
    ACCEPTANCE_PUBLISH_JOB,
    AcceptanceReadingUnpublishable,
    acceptance_publish_handler,
)
from crucible.keys import acceptance_reading_key, manifest_key
from crucible.runner import CODE_SHA_ENV
from crucible.store import LocalStore

DAY = dt.date(2026, 8, 28)
_FAKE_SHA = "a" * 40

READING = {
    "met": 21,
    "unmet": 5,
    "unmeasurable": 2,
    "commit": "b" * 40,
    "measured_at": "2026-08-28T12:00:00Z",
    "met_clauses": ["TestOne::test_a"],
    "unmet_clauses": ["TestOne::test_b"],
    "unmeasurable_clauses": ["TestOne::test_c"],
}


@pytest.fixture(autouse=True)
def _fixed_code_sha(monkeypatch):
    monkeypatch.setenv(CODE_SHA_ENV, _FAKE_SHA)


def _args(tmp_path, *, reading: str, dry_run: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        trading_day=DAY,
        store=str(tmp_path / "store"),
        dry_run=dry_run,
        run_mode=None,
        reading=reading,
    )


def _write_reading(tmp_path, payload) -> str:
    path = tmp_path / "acceptance-reading.json"
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload), "utf-8")
    return str(path)


class TestThePublishedDocument:
    def test_the_reading_lands_at_the_key_the_consumers_read(self, tmp_path) -> None:
        acceptance_publish_handler(_args(tmp_path, reading=_write_reading(tmp_path, READING)))
        store = LocalStore(tmp_path / "store")
        assert json.loads(store.get_bytes(acceptance_reading_key(DAY.isoformat()))) == READING

    def test_the_bytes_are_the_file_the_grader_wrote_not_a_re_serialisation(self, tmp_path) -> None:
        """One measurement, published. Re-serialising would make the published
        document a second rendering of a file the grader already produced, and
        the `commit` field is what ties the reading to the tree it graded."""
        path = tmp_path / "acceptance-reading.json"
        path.write_text(json.dumps(READING, indent=4), encoding="utf-8")
        acceptance_publish_handler(_args(tmp_path, reading=str(path)))
        store = LocalStore(tmp_path / "store")
        assert store.get_bytes(acceptance_reading_key(DAY.isoformat())) == path.read_bytes()

    def test_the_write_is_recorded_as_an_output_of_its_own_run(self, tmp_path) -> None:
        """Rule 1, which is the whole point: the artifact a phase gate reads is
        named in the manifest of the run that wrote it."""
        acceptance_publish_handler(_args(tmp_path, reading=_write_reading(tmp_path, READING)))
        store = LocalStore(tmp_path / "store")
        manifest = json.loads(
            store.get_bytes(manifest_key(ACCEPTANCE_PUBLISH_JOB, DAY.isoformat()))
        )
        assert manifest["status"] == "ok", manifest.get("reason")
        assert [row["key"] for row in manifest["outputs"]] == [
            acceptance_reading_key(DAY.isoformat())
        ]

    def test_the_clause_count_is_on_the_manifest_as_its_outcome_signal(self, tmp_path) -> None:
        acceptance_publish_handler(_args(tmp_path, reading=_write_reading(tmp_path, READING)))
        store = LocalStore(tmp_path / "store")
        manifest = json.loads(
            store.get_bytes(manifest_key(ACCEPTANCE_PUBLISH_JOB, DAY.isoformat()))
        )
        metric = next(m for m in manifest["metrics"] if m["name"] == "acceptance_clauses_met")
        assert metric["value"] == 21.0
        # A red count is plan §12 rule 3's whole point and must never page: the
        # §2 suite is red BY DESIGN for months, and a metric that FAILED on it
        # would train the operator to ignore the one progress figure there is.
        assert metric["status"] == "OK"


class TestRefusals:
    """A detector nobody has made fail is a detector nobody knows works."""

    def test_an_absent_reading_is_refused_and_the_failure_is_recorded(self, tmp_path) -> None:
        with pytest.raises(AcceptanceReadingUnpublishable, match="does not exist"):
            acceptance_publish_handler(_args(tmp_path, reading=str(tmp_path / "nope.json")))
        store = LocalStore(tmp_path / "store")
        manifest = json.loads(
            store.get_bytes(manifest_key(ACCEPTANCE_PUBLISH_JOB, DAY.isoformat()))
        )
        assert manifest["status"] == "failed"
        assert not list(store.list_keys("report/acceptance/"))

    def test_a_file_that_is_not_json_is_refused(self, tmp_path) -> None:
        with pytest.raises(AcceptanceReadingUnpublishable, match="is not JSON"):
            acceptance_publish_handler(_args(tmp_path, reading=_write_reading(tmp_path, "{")))

    def test_a_document_the_consumers_parser_rejects_is_refused_here(self, tmp_path) -> None:
        """The producer refuses what `crucible.keys.parse_acceptance_reading`
        would refuse, naming the file — rather than publishing it and having
        two surfaces render 'no acceptance reading' with nothing saying why."""
        incomplete = {k: v for k, v in READING.items() if k != "commit"}
        with pytest.raises(AcceptanceReadingUnpublishable, match="not an acceptance reading"):
            acceptance_publish_handler(
                _args(tmp_path, reading=_write_reading(tmp_path, incomplete))
            )
        assert not list(LocalStore(tmp_path / "store").list_keys("report/acceptance/"))


class TestDryRun:
    """The row `tests/test_cli_and_alerts.py::TestDryRunNeverWrites` excludes
    this job from, asserted directly: the reading is read and validated, the
    real body runs, and the store gains no key at all — not even a manifest."""

    def test_a_dry_run_validates_and_writes_nothing(self, tmp_path) -> None:
        store_dir = tmp_path / "store"
        acceptance_publish_handler(
            _args(tmp_path, reading=_write_reading(tmp_path, READING), dry_run=True)
        )
        assert not store_dir.exists() or not list(LocalStore(store_dir).list_keys(""))

    def test_a_dry_run_still_refuses_an_unpublishable_reading(self, tmp_path) -> None:
        """The rehearsal fails the way the run fails
        (`alpha-engine-config-I11012`): the body executes for real against a
        write-capturing store, so a dry run over a malformed file raises here
        exactly as the live invocation would."""
        with pytest.raises(AcceptanceReadingUnpublishable):
            acceptance_publish_handler(
                _args(tmp_path, reading=_write_reading(tmp_path, "{"), dry_run=True)
            )
