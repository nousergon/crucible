"""The trial ledger under concurrent writers (alpha-engine-config-I12562, audit CR-01).

`append_trials` is a read-modify-write of one object. Until this test existed
it read the ledger twice and then wrote with an UNCONDITIONAL `put_bytes`, so
two writers that both passed the second read before either wrote both
returned success and the ledger kept only the later one's row — a trial that
was graded, reported as recorded, and then silently absent from the DSR
denominator it exists to feed.

These tests interleave the writers deterministically rather than by luck: a
barrier holds each writer's FIRST store write until every writer has reached
it, which is after both of their final reads. That is the exact interleaving
the audit reproduced (`nous-ergon-ops/audits/2026-10-22-core-ne`,
`reproduce_ledger_race.py.txt`). Both backends are exercised — `S3Store`
through a fake that applies `IfMatch`/`IfNoneMatch` atomically, as the
service does, and `LocalStore` on a real directory.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from crucible.ledger import LedgerAppendError, append_trials, n_trials, read_trials
from crucible.store import LocalStore, PointerConflictError, S3Store
from tests.conftest import FakeS3

_JOIN_TIMEOUT_S = 15.0


def _row(slot: str, arm: str, as_of: str = "2026-10-22") -> dict[str, Any]:
    return {
        "schema_version": "trial.v1",
        "slot": slot,
        "arm_id": arm,
        "as_of": as_of,
        "control": False,
        "active": True,
        "benchmark": "population",
        "n_dates_scored": 4,
        "first_date": "2026-10-16",
        "last_date": "2026-10-21",
        "mean_score_ratio": 0.01,
        "run_id": "01JG0000000000000000000000",
        "arena_cycle_key": f"arena/{slot}/{as_of}/arena_cycle.json",
        "written_at_utc": "2026-10-22T21:00:00Z",
    }


class _FirstWriteGate:
    """Holds each thread's first write until ``parties`` threads have reached one.

    Only the FIRST write per thread waits: a writer that loses a conditional
    write and retries must not block on a barrier nobody else will reach.
    """

    def __init__(self, parties: int) -> None:
        self.barrier = threading.Barrier(parties, timeout=10.0)
        self._passed: set[int] = set()
        self._lock = threading.Lock()

    def exempt_current_thread(self) -> None:
        """Let this thread's writes through ungated: the test's own seeding."""
        with self._lock:
            self._passed.add(threading.get_ident())

    def wait_if_first(self) -> None:
        ident = threading.get_ident()
        with self._lock:
            first = ident not in self._passed
            self._passed.add(ident)
        if first:
            self.barrier.wait()


class _GatedFakeS3(FakeS3):
    """`FakeS3` whose `put_object` is atomic, behind a first-write gate.

    The condition check and the store of the body happen under one lock,
    which is what S3 guarantees for a conditional PUT; without it the fake
    itself would have a check-then-act window and the test would be testing
    the fake.
    """

    def __init__(self, gate: _FirstWriteGate) -> None:
        super().__init__()
        self.gate = gate
        self._atomic = threading.Lock()

    def put_object(self, **kw: Any) -> dict[str, Any]:
        self.gate.wait_if_first()
        with self._atomic:
            return super().put_object(**kw)


def _gated_local_store(root, gate: _FirstWriteGate) -> LocalStore:
    class _GatedLocalStore(LocalStore):
        def put_bytes(self, key: str, payload: bytes, **kw: Any) -> str:
            gate.wait_if_first()
            return super().put_bytes(key, payload, **kw)

        def compare_and_swap(self, key: str, expected: str, payload: bytes) -> str:
            gate.wait_if_first()
            return super().compare_and_swap(key, expected, payload)

    return _GatedLocalStore(root)


def _store(backend: str, tmp_path, gate: _FirstWriteGate):
    if backend == "s3":
        return S3Store("bucket", client=_GatedFakeS3(gate))
    return _gated_local_store(tmp_path / "store", gate)


def _race(store, batches: list[list[dict[str, Any]]]) -> tuple[list[int], list[str]]:
    counts: list[int] = []
    errors: list[str] = []
    lock = threading.Lock()

    def append(rows: list[dict[str, Any]]) -> None:
        try:
            landed = append_trials(store, rows)
        except Exception as exc:  # recorded and asserted on below, never swallowed
            with lock:
                errors.append(repr(exc))
            return
        with lock:
            counts.append(landed)

    workers = [threading.Thread(target=append, args=(rows,)) for rows in batches]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=_JOIN_TIMEOUT_S)
    assert not any(worker.is_alive() for worker in workers), "a writer never finished"
    return sorted(counts), errors


@pytest.mark.parametrize("backend", ["s3", "local"])
@pytest.mark.parametrize("seeded", [False, True], ids=["create", "replace"])
class TestTwoWritersAfterTheirFinalRead:
    def test_both_distinct_trials_survive(self, backend, seeded, tmp_path) -> None:
        """The CR-01 interleaving: both writers have read, neither has written.

        ``create`` races two writers onto an absent ledger (the `IfNoneMatch`
        / create-if-absent half); ``replace`` races them onto an existing one
        (the `IfMatch` half). Either way both trials must be in the ledger,
        each exactly once, and both calls must report the one row they added.
        """
        gate = _FirstWriteGate(parties=2)
        store = _store(backend, tmp_path, gate)
        if seeded:
            # The seed is the only write this thread makes; let it through.
            gate.exempt_current_thread()
            assert append_trials(store, [_row("m", "seed")]) == 1

        counts, errors = _race(store, [[_row("m", "trial-A")], [_row("m", "trial-B")]])

        assert errors == []
        assert counts == [1, 1]
        arms = [row["arm_id"] for row in read_trials(store)]
        expected = (["seed"] if seeded else []) + ["trial-A", "trial-B"]
        assert sorted(arms) == sorted(expected), (
            "a writer that reported success had its trial dropped by the other "
            f"writer's rewrite: ledger holds {arms}"
        )
        assert len(arms) == len(set(arms)), "a trial identity was written twice"
        assert n_trials(store, slot="m") == len(expected), (
            "the DSR denominator must count every trial that was recorded"
        )

    def test_a_concurrent_replay_lands_exactly_once(self, backend, seeded, tmp_path) -> None:
        """Two writers appending the SAME identity: one adds it, one adds nothing."""
        gate = _FirstWriteGate(parties=2)
        store = _store(backend, tmp_path, gate)
        if seeded:
            gate.exempt_current_thread()
            append_trials(store, [_row("m", "seed")])

        counts, errors = _race(store, [[_row("m", "same")], [_row("m", "same")]])

        assert errors == []
        assert counts == [0, 1]
        arms = [row["arm_id"] for row in read_trials(store)]
        assert arms.count("same") == 1


@pytest.mark.parametrize("backend", ["s3", "local"])
def test_many_writers_leave_a_complete_monotone_ledger(backend, tmp_path) -> None:
    """Six writers, each with its own trial plus one replayed identity, all at once.

    Under free-running contention a writer may also meet the I9757 re-read
    guard — another writer landed between its two reads — which refuses
    with an explicit :class:`LedgerAppendError` rather than writing. That is
    the issue's "or raise an explicit conflict" outcome, and it is the ONLY
    failure allowed: every writer that returned a count has its rows in the
    ledger, exactly once, behind an unmoved prefix.
    """
    writers = 6
    gate = _FirstWriteGate(parties=writers)
    store = _store(backend, tmp_path, gate)
    gate.exempt_current_thread()
    append_trials(store, [_row("m", "seed")])
    before = [r["arm_id"] for r in read_trials(store)]

    outcomes: dict[int, int | Exception] = {}
    lock = threading.Lock()

    def append(i: int) -> None:
        try:
            result: int | Exception = append_trials(
                store, [_row("m", f"trial-{i}"), _row("m", "replayed")]
            )
        except Exception as exc:  # classified and asserted on below, never swallowed
            result = exc
        with lock:
            outcomes[i] = result

    threads = [threading.Thread(target=append, args=(i,)) for i in range(writers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=_JOIN_TIMEOUT_S)
    assert not any(t.is_alive() for t in threads), "a writer never finished"

    refused = {i: o for i, o in outcomes.items() if isinstance(o, Exception)}
    for exc in refused.values():
        assert isinstance(exc, LedgerAppendError), f"an unexplained failure: {exc!r}"
    landed = [i for i, o in outcomes.items() if not isinstance(o, Exception)]
    assert landed, "at least the first writer to the version token must land"

    arms = [r["arm_id"] for r in read_trials(store)]
    assert arms[: len(before)] == before, "the ledger is append-only: its prefix never moves"
    assert len(arms) == len(set(arms)), "a trial identity was written twice"
    for i in landed:
        assert f"trial-{i}" in arms, f"writer {i} reported success but its trial is absent"
    for i in refused:
        assert f"trial-{i}" not in arms, f"writer {i} raised but its trial was recorded"
    assert sum(o for o in outcomes.values() if isinstance(o, int)) == len(arms) - len(before)


def test_a_writer_that_never_wins_raises_an_explicit_conflict(tmp_path) -> None:
    """Bounded retry: a ledger under permanent contention is a loud failure."""

    class _AlwaysLoses(LocalStore):
        def compare_and_swap(self, key: str, expected: str, payload: bytes) -> str:
            raise PointerConflictError("another writer won")

    store = _AlwaysLoses(tmp_path / "store")
    with pytest.raises(LedgerAppendError, match="conditional write"):
        append_trials(store, [_row("m", "trial-A")])
    assert read_trials(store) == []
