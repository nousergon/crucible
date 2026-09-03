"""`crucible.track_f.gate_handler` — the gate job's own lineage recording.

`alpha-engine-config-I9869` round 3, finding 1 (BLOCKING). Round 2 gave
`crucible.gate` a fully guarded reading path: every phase-1 clause turns a
store access failure into an UNMEASURABLE clause rather than raising.
`gate_handler` then re-read those SAME evidence keys with a bare
`store.exists`/`store.get_bytes` pair to record job lineage
(`ctx.record_input`) — so on an access failure, `evaluate()` correctly
returned an UNMEASURABLE clause, and the job body then tracebacked one call
later reading the exact key the clause had already flagged. Round 2's own
guard, undone by its own lineage recorder.
"""

from __future__ import annotations

from unittest import mock

from crucible.cli import main
from crucible.gate import GATES, Clause
from crucible.store import LocalStore


class TestGateHandlerLineageReadIsGuarded:
    def test_a_chmod_000_evidence_file_does_not_crash_the_gate_job(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        evidence_key = "runs/report/2026-08-28/run.json"
        store.put_bytes(evidence_key, b'{"status": "ok", "reason": ""}')
        evidence_path = tmp_path / evidence_key
        original_mode = evidence_path.stat().st_mode
        evidence_path.chmod(0o000)

        def fake_clauses(*_args, **_kwargs) -> list[Clause]:
            # The reading itself is already correct after round 2: an
            # unreadable input is a red clause, not an exception. This test
            # is about what `gate_handler` does NEXT, re-reading the same
            # evidence key for job lineage.
            return [Clause("c", "requirement", False, "unmet", (evidence_key,))]

        try:
            with mock.patch.dict(GATES, {"phase1": (5, fake_clauses)}):
                exit_code = main(
                    [
                        "gate",
                        "--gate",
                        "phase1",
                        "--store",
                        str(tmp_path),
                        "--date",
                        "2026-08-28",
                        "--dry-run",
                    ]
                )
        finally:
            # Restored even on failure: a permission-denied fixture left
            # behind would break every later test that walks this tmp_path,
            # and pytest does not always clean up a chmod-000 file itself.
            evidence_path.chmod(original_mode)

        assert isinstance(exit_code, int)
        # The ladder was rendered and published — the job did not die before
        # reaching its own output write.
        ladder_bytes = store.get_bytes("gates/ladder.json")
        assert ladder_bytes
