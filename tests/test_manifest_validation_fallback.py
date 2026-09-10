"""Manifest-or-it-didn't-happen, including when the manifest will not validate.

**The measured defect (`alpha-engine-config-I10410`).** `_write_manifest`
validated the assembled manifest and let `ManifestValidationError` propagate,
from inside `run_job`'s own `finally`. Nothing was written. The job exited
non-zero, the store held nothing, and `crucible.alerts` reported the run as an
**ABSENCE** — the condition reserved for "nothing was even attempted" — with
the cause discarded.

`gate.close` for trading day 2026-09-08 (GitHub Actions run 34297633089,
2026-09-09T01:05:20Z) filed two phase closing records and then died on
`outputs/0/sha256: String should match pattern '^[0-9a-f]{64}$'`.
`runs/gate.close/2026-09-08/` is empty, and it was still one of the two true
members of the ABSENCE page a day later. The only surviving account of a run
that really did file two closing records is a GitHub Actions log on a 90-day
clock.

`crucible/AGENTS.md` rule 1 is the invariant; the validator is the one thing
in the system that was allowed to break it.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from crucible.manifest import manifest_key, validate
from crucible.runner import run_job
from crucible.store import LocalStore

TRADING_DAY = dt.date(2026, 9, 8)
STARTED = dt.datetime(2026, 9, 9, 1, 5, tzinfo=dt.UTC)


def _rejected_output(ctx) -> None:
    """A job that records an output whose `sha256` is not a content digest —
    the exact shape `gate.close` produced (a store version token in the
    digest field)."""
    ctx.outputs.append(
        {"key": "gates/phase0/closing.json", "sha256": "a-version-token", "schema_version": "v1"}
    )


def _read(store: LocalStore, job: str) -> dict:
    return json.loads(store.get_bytes(manifest_key(job, TRADING_DAY.isoformat())).decode())


class TestARunWhoseManifestIsRejectedStillLeavesOne:
    def test_a_manifest_exists_at_the_expected_key(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        run_job("gate.close", _rejected_output, store=store, trading_day=TRADING_DAY, now=STARTED)
        assert _read(store, "gate.close")["run_id"]

    def test_it_is_a_failure_not_a_success(self, tmp_path) -> None:
        """The run thought it succeeded. A run whose own account of itself is
        malformed has not succeeded, and recording `ok` on a document we had
        to rebuild would be a graceful-degrade."""
        store = LocalStore(tmp_path)
        run_job("gate.close", _rejected_output, store=store, trading_day=TRADING_DAY, now=STARTED)
        assert _read(store, "gate.close")["status"] == "failed"

    def test_the_reason_names_the_validator_complaint(self, tmp_path) -> None:
        """What an operator needs in order to fix the producer. Before this,
        it existed only in a GitHub Actions log."""
        store = LocalStore(tmp_path)
        run_job("gate.close", _rejected_output, store=store, trading_day=TRADING_DAY, now=STARTED)
        reason = _read(store, "gate.close")["reason"]
        assert "outputs/0/sha256" in reason
        assert "does not validate" in reason

    def test_the_reason_carries_the_runs_own_status(self, tmp_path) -> None:
        """Both facts, not one: the job's own failure is what the operator
        came for, and the validator's complaint is what they need to fix."""
        store = LocalStore(tmp_path)

        def failing(ctx) -> None:
            _rejected_output(ctx)
            raise RuntimeError("the job also failed on its own")

        with pytest.raises(RuntimeError):
            run_job(
                "gate.close",
                failing,
                store=store,
                trading_day=TRADING_DAY,
                now=STARTED,
                transient_retry=False,
            )
        reason = _read(store, "gate.close")["reason"]
        assert "the job also failed on its own" in reason
        assert "outputs/0/sha256" in reason

    def test_the_fallback_itself_conforms(self, tmp_path) -> None:
        """A stand-in that does not validate would be the same defect one
        layer down."""
        store = LocalStore(tmp_path)
        run_job("gate.close", _rejected_output, store=store, trading_day=TRADING_DAY, now=STARTED)
        validate(_read(store, "gate.close"))

    def test_the_rejected_field_is_dropped_not_carried_forward(self, tmp_path) -> None:
        """Carrying the job's own fields forward would reproduce the failure
        the fallback exists to survive."""
        store = LocalStore(tmp_path)
        run_job("gate.close", _rejected_output, store=store, trading_day=TRADING_DAY, now=STARTED)
        assert _read(store, "gate.close")["outputs"] == []

    def test_a_conformant_manifest_is_untouched(self, tmp_path) -> None:
        """The fallback is reached only by the failing path. A job whose
        manifest validates writes exactly what it assembled."""
        store = LocalStore(tmp_path)

        def clean(ctx) -> None:
            ctx.record_output("gates/phase0/closing.json", b"{}", schema_version="v1")

        run_job("gate.close", clean, store=store, trading_day=TRADING_DAY, now=STARTED)
        manifest = _read(store, "gate.close")
        assert manifest["status"] == "ok"
        assert len(manifest["outputs"]) == 1
