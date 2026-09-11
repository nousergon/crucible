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

import json
from unittest import mock

from crucible.cli import main
from crucible.gate import GATE_DELIVERABLES, GATES, Clause
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
            with (
                mock.patch.dict(GATES, {"phase1": (5, fake_clauses)}),
                mock.patch.dict(GATE_DELIVERABLES, {"phase1": ()}),
            ):
                # No `--dry-run`: this test asserts the ladder was WRITTEN
                # for real (below) — `--dry-run` here was vestigial and, as
                # of alpha-engine-config-I9922 N1, would now correctly refuse
                # that write rather than silently being a no-op. `--publish`
                # (`alpha-engine-config-I10492`): the write this test reads
                # back no longer happens without it.
                exit_code = main(
                    [
                        "gate",
                        "--gate",
                        "phase1",
                        "--store",
                        str(tmp_path),
                        "--date",
                        "2026-08-28",
                        "--publish",
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


class TestGateHandlerOutcomeMetricNamesTheUnmeasurableCause:
    """`alpha-engine-config-I9869` round 4, finding 2 (BLOCKING). Round 3
    widened `GateResult.met_ratio` to `None` for ANY unmeasurable clause, not
    only for zero registered clauses, but left `gate_handler`'s branch text
    unchanged: it always said "no clauses registered, nothing measured" —
    false when clauses ARE registered and one merely could not be read. That
    false statement lands on `gate_clauses_met_ratio`, the gate job's
    declared outcome metric (`crucible/components.yaml:498`), and points an
    operator at the wrong remedy ("register the clauses" instead of "fix our
    credentials")."""

    def test_status_reason_names_the_unmeasurable_clause_not_no_clauses_registered(
        self, tmp_path
    ) -> None:
        store = LocalStore(tmp_path)

        def fake_clauses(*_args, **_kwargs) -> list[Clause]:
            # Six clauses registered, five measured and met, one
            # UNMEASURABLE — the exact shape round 4's reproduction used
            # (a chmod-000 arc manifest): registered-but-unmeasurable, not
            # "nothing registered".
            return [
                Clause("arc_runs_ok", "req", False, "1 could not be read", (), unmeasurable=True),
                Clause("arms_all_scored", "req", True, "ok", ()),
                Clause("attribution_renders", "req", True, "ok", ()),
                Clause("explain_walks_a_verdict", "req", True, "ok", ()),
                Clause("pointer_flipped_on_smoke", "req", True, "ok", ()),
                Clause("independently_reviewed", "req", True, "ok", ()),
            ]

        with mock.patch.dict(GATES, {"phase1": (6, fake_clauses)}):
            # No `--dry-run`: this test reads the real run manifest below —
            # see the sibling test's comment above for why the flag was
            # dropped rather than kept as a (now-refusing) no-op.
            main(
                [
                    "gate",
                    "--gate",
                    "phase1",
                    "--store",
                    str(tmp_path),
                    "--date",
                    "2026-08-28",
                ]
            )

        run_document = json.loads(store.get_bytes("runs/gate/2026-08-28/run.json"))
        metric = next(m for m in run_document["metrics"] if m["name"] == "gate_clauses_met_ratio")
        assert metric["value"] is None
        assert metric["n_samples"] == 6
        assert "arc_runs_ok" in metric["status_reason"]
        assert "no clauses registered" not in metric["status_reason"]
        assert "1 of 6" in metric["status_reason"]
        # The honest status for "registered but could not be read" — not the
        # zero-clause N/A-LOW-N branch, and not a status invented outside the
        # existing `krepis.metrics.derive_status` vocabulary.
        assert metric["status"] == "N/A-MISSING-INPUT"
