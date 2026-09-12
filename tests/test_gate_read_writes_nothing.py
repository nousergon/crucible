"""`crucible gate` without `--publish` reaches the store for reads only.

`alpha-engine-config-I10576`. `--publish` already gated the two SHARED
artifacts — the dated reading `gates/{gate}/{day}/gate.json` and
`gates/ladder.json` (`alpha-engine-config-I10492`) — but the job's OWN
manifest at `runs/gate/{day}/run.json` was still written on every
non-`--dry-run` invocation, on the stated ground that rule 1 is
unconditional.

Measured 2026-09-12T20:12Z: a laptop `crucible gate` read as the operator
profile wrote `runs/gate/2026-09-11/run.json`, and `crucible.autonomy` counts
that PutObject as a human-originated mutating call inside phase 2's own
`zero_human_mutating_calls` window. The read recipe `crucible/AGENTS.md`
documents was therefore a phase-2 exit hazard every time anyone followed it —
a surface asserting a control ("reads and prints") that did not exist.

Three cases, and they are the whole contract: the default writes NOTHING,
`--publish` writes all three artifacts, and `--dry-run` still writes nothing.
Asserted against a store that RECORDS every mutating call rather than only
against the key listing afterwards: a write that was attempted and refused,
and a write that landed and was cleaned up, both leave an empty listing.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pytest

from crucible.cli import main
from crucible.gate import GATE_DELIVERABLES, GATES, LADDER_KEY, Clause
from crucible.keys import gate_key
from crucible.store import LocalStore

#: A real NYSE session, fixed. Never `today` arithmetic — a test whose subject
#: moves with the clock stops testing the same thing.
DAY = dt.date(2026, 8, 28)

GATE = "phase2"


class RecordingStore(LocalStore):
    """A `LocalStore` that appends every mutating call to `self.mutations`.

    A subclass rather than a wrapper, for the same reason
    `crucible.store.read_only` builds one: `isinstance(store, S3Store)` checks
    elsewhere in the tree are real, and the read-only wrapper this test's
    subject applies is itself a dynamic subclass of whatever it is handed.
    """

    def __init__(self, root: Any) -> None:
        super().__init__(root)
        self.mutations: list[str] = []

    def put_bytes(self, key: str, payload: bytes, **kwargs: Any) -> Any:
        self.mutations.append(key)
        return super().put_bytes(key, payload, **kwargs)

    def compare_and_swap(self, key: str, expected: str, payload: bytes, **kwargs: Any) -> Any:
        self.mutations.append(key)
        return super().compare_and_swap(key, expected, payload, **kwargs)


@pytest.fixture
def recorder(tmp_path, monkeypatch) -> RecordingStore:
    """One `RecordingStore` over `tmp_path`, returned by `open_store` for
    every resolution inside the run under test."""
    store = RecordingStore(tmp_path)

    def fake_open_store(uri: str | None, *, dry_run: bool = False) -> Any:
        from crucible.store import read_only

        return read_only(store) if dry_run else store

    monkeypatch.setattr("crucible.track_f.open_store", fake_open_store)
    return store


@pytest.fixture(autouse=True)
def _a_readable_phase2(monkeypatch) -> None:
    """Two unmet clauses, and the two environment variables `crucible gate`
    refuses to start without (`alpha-engine-config-I10492`/`I10438`).

    The clauses are faked so this test measures the WRITE behaviour of the
    command and not the live readability of phase 2's real clauses, several of
    which reach CloudTrail, Cost Explorer and SNS.
    """
    monkeypatch.setenv("CRUCIBLE_CLOUDTRAIL_ARCHIVE", "s3://example/trail")
    monkeypatch.setenv("CRUCIBLE_MUTED_TOPIC", "example-muted")
    monkeypatch.setenv("CRUCIBLE_RUN_MODE", "live")
    monkeypatch.setitem(
        GATES,
        GATE,
        (
            2,
            lambda *_a, **_k: [
                Clause("c", "req", False, "unmet", ()),
                Clause("d", "req", True, "met", ()),
            ],
        ),
    )
    monkeypatch.setitem(GATE_DELIVERABLES, GATE, ())


def _argv(*extra: str) -> list[str]:
    return [
        "gate",
        "--gate",
        GATE,
        "--store",
        "ignored-by-the-fake",
        "--date",
        DAY.isoformat(),
        *extra,
    ]


class TestTheDefaultInvocationWritesNothing:
    def test_no_publish_records_zero_mutating_calls(self, recorder, capsys) -> None:
        exit_code = main(_argv())

        # The reading still happened, was printed, and the unmet gate still
        # exits non-zero — this is a READ that reports, not a suppressed run.
        assert exit_code == 1
        printed = capsys.readouterr().out
        assert GATE in printed

        assert recorder.mutations == []
        assert list(recorder.list_keys()) == []

    def test_the_printed_line_does_not_call_a_live_read_a_dry_run(self, recorder, capsys) -> None:
        """The reason `run_job` gained `write_manifest` instead of reusing
        `dry_run`: this run read live state, and a line telling the operator
        they performed a "dry_run" would be the same class of false surface
        claim the issue is about."""
        main(_argv())

        printed = capsys.readouterr().out
        assert "read-only: gate" in printed
        assert "dry_run: gate" not in printed


class TestPublishWritesAllThree:
    def test_publish_writes_the_manifest_the_reading_and_the_ladder(self, recorder) -> None:
        exit_code = main(_argv("--publish"))

        assert exit_code == 1  # still unmet; publishing is not a verdict
        assert set(recorder.mutations) == {
            f"runs/gate/{DAY.isoformat()}/run.json",
            gate_key(GATE, DAY.isoformat()),
            LADDER_KEY,
        }


class TestDryRunStillWritesNothing:
    def test_dry_run_records_zero_mutating_calls(self, recorder, capsys) -> None:
        exit_code = main(_argv("--dry-run"))

        assert exit_code == 1
        assert recorder.mutations == []
        assert list(recorder.list_keys()) == []
        # And the dry-run wording is still the dry-run wording: the second
        # parameter added for the read path must not have collapsed the two.
        assert "dry_run: gate" in capsys.readouterr().out
