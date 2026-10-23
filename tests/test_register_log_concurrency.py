"""The other append-only logs under concurrent writers (CR-01 same-class sweep).

`alpha-engine-config-I12562` (audit CR-01) found the trial ledger's
read-modify-write losing a concurrent writer's row, and asked for "arm-register
and mutable-history writers in a bounded same-class sweep" with explicit
dispositions. `tests/test_ledger_concurrency.py` holds the ledger; this file
holds the rest, with the same deterministic interleaving — every writer has
done its final read before any writer's first store write:

* ``crucible.slots.arms.write_register`` (the arm register, rewritten from an
  in-memory fold by every grade, `experiment.new` and migration): a writer
  whose register never saw the other's event must RAISE, never erase it.
* ``crucible.promote._apply_retirements_to_register`` (retirement verdicts
  appended to that same register): both retirements land, a replayed one
  lands once.
* ``crucible.promote._append_events`` (the retirement log and the
  EXPERIMENTS event log): both rows land, a replayed row lands once.

Both backends are exercised through the gates in `tests/test_ledger_concurrency.py`.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

import pytest

from crucible.keys import arm_register_key
from crucible.promote import _append_events, _apply_retirements_to_register
from crucible.slots.arms import (
    ArmRegister,
    ArmRegisterConflictError,
    load_arm_specs,
    read_register,
    register_arms,
    write_register,
)
from crucible.store import (
    ConditionalAppendError,
    LocalStore,
    PointerConflictError,
    append_conditionally,
)
from tests.test_ledger_concurrency import _FirstWriteGate, _store

_JOIN_TIMEOUT_S = 15.0
BACKENDS = ["s3", "local"]


def _run_together(*jobs) -> list[Any]:
    """Run ``jobs`` on their own threads; each result is a return value or an exception."""
    results: list[Any] = [None] * len(jobs)

    def run(i: int) -> None:
        try:
            results[i] = jobs[i]()
        except Exception as exc:  # classified and asserted on by the caller, never swallowed
            results[i] = exc

    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(jobs))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=_JOIN_TIMEOUT_S)
    assert not any(t.is_alive() for t in threads), "a writer never finished"
    return results


def _lines(store, key: str) -> list[str]:
    return [line for line in store.get_bytes(key).decode().splitlines() if line.strip()]


@pytest.mark.parametrize("backend", BACKENDS)
class TestArmRegister:
    def test_a_writer_that_never_saw_the_other_event_raises_instead_of_erasing_it(
        self, backend, strategy_dir, tmp_path
    ) -> None:
        specs = load_arm_specs("u", strategy_dir=strategy_dir)
        gate = _FirstWriteGate(parties=2)
        store = _store(backend, tmp_path, gate)
        gate.exempt_current_thread()
        write_register(
            store, "u", register_arms(ArmRegister(), specs[:1], filed_on="2026-08-28")[0]
        )

        # Both writers read the same register, then each folds a different arm.
        base = read_register(store, "u")
        first, _ = register_arms(base, [specs[1]], filed_on="2026-08-28")
        second, _ = register_arms(base, [specs[2]], filed_on="2026-08-28")

        results = _run_together(
            lambda: write_register(store, "u", first),
            lambda: write_register(store, "u", second),
        )

        wrote = [r for r in results if isinstance(r, bytes)]
        refused = [r for r in results if isinstance(r, Exception)]
        assert len(wrote) == 1, results
        assert len(refused) == 1, results
        assert isinstance(refused[0], ArmRegisterConflictError), refused[0]
        # The winner's event is in the log and nothing it held was erased.
        registered = set(read_register(store, "u").all_arms())
        assert wrote[0] == store.get_bytes(arm_register_key("u"))
        winner = first if isinstance(results[0], bytes) else second
        assert set(winner.all_arms()) == registered

    def test_rewriting_the_same_register_concurrently_adds_nothing(
        self, backend, strategy_dir, tmp_path
    ) -> None:
        specs = load_arm_specs("u", strategy_dir=strategy_dir)
        gate = _FirstWriteGate(parties=2)
        store = _store(backend, tmp_path, gate)
        gate.exempt_current_thread()
        register, _ = register_arms(ArmRegister(), specs, filed_on="2026-08-28")
        write_register(store, "u", register)
        before = store.get_bytes(arm_register_key("u"))

        results = _run_together(
            lambda: write_register(store, "u", register),
            lambda: write_register(store, "u", register),
        )

        assert all(isinstance(r, bytes) for r in results), results
        assert store.get_bytes(arm_register_key("u")) == before


@pytest.mark.parametrize("backend", BACKENDS)
class TestRetirementsIntoTheRegister:
    def _seeded(self, backend, strategy_dir, tmp_path, parties: int):
        specs = load_arm_specs("u", strategy_dir=strategy_dir)
        gate = _FirstWriteGate(parties=parties)
        store = _store(backend, tmp_path, gate)
        gate.exempt_current_thread()
        register, _ = register_arms(ArmRegister(), specs, filed_on="2026-08-28")
        write_register(store, "u", register)
        return store, read_register(store, "u"), sorted(register.all_arms())

    @staticmethod
    def _cycle(*arm_ids: str) -> SimpleNamespace:
        return SimpleNamespace(
            retirements=[
                SimpleNamespace(retire=True, arm_id=a, reason="trailing the champion")
                for a in arm_ids
            ]
        )

    def test_two_retirements_both_land(self, backend, strategy_dir, tmp_path) -> None:
        store, register, arms = self._seeded(backend, strategy_dir, tmp_path, 2)
        key = arm_register_key("u")
        results = _run_together(
            lambda: _apply_retirements_to_register(
                store, "u", "2026-10-16", self._cycle(arms[0]), register
            ),
            lambda: _apply_retirements_to_register(
                store, "u", "2026-10-16", self._cycle(arms[1]), register
            ),
        )
        assert results == [key, key]
        folded = read_register(store, "u")
        assert folded.state(arms[0]).retired_date is not None
        assert folded.state(arms[1]).retired_date is not None
        assert folded.state(arms[2]).retired_date is None

    def test_a_concurrent_replay_retires_once(self, backend, strategy_dir, tmp_path) -> None:
        store, register, arms = self._seeded(backend, strategy_dir, tmp_path, 2)
        results = _run_together(
            lambda: _apply_retirements_to_register(
                store, "u", "2026-10-16", self._cycle(arms[0]), register
            ),
            lambda: _apply_retirements_to_register(
                store, "u", "2026-10-16", self._cycle(arms[0]), register
            ),
        )
        assert sorted(results, key=str) == sorted([arm_register_key("u"), None], key=str)
        retired = [line for line in _lines(store, arm_register_key("u")) if '"retired"' in line]
        assert len(retired) == 1
        # The log still folds: a doubled `retired` event would make it unloadable.
        assert read_register(store, "u").state(arms[0]).retired_date is not None


@pytest.mark.parametrize("backend", BACKENDS)
class TestEventLogs:
    KEY = "experiments/2026-10-16/events.jsonl"

    def test_two_writers_both_rows_land(self, backend, tmp_path) -> None:
        gate = _FirstWriteGate(parties=2)
        store = _store(backend, tmp_path, gate)
        gate.exempt_current_thread()
        _append_events(store, self.KEY, [{"event": "seed"}])
        _run_together(
            lambda: _append_events(store, self.KEY, [{"event": "a"}]),
            lambda: _append_events(store, self.KEY, [{"event": "b"}]),
        )
        events = [line for line in _lines(store, self.KEY)]
        assert len(events) == 3
        assert events[0].startswith('{"event": "seed"')
        assert any('"event": "a"' in e for e in events)
        assert any('"event": "b"' in e for e in events)

    def test_a_concurrent_replay_lands_once(self, backend, tmp_path) -> None:
        gate = _FirstWriteGate(parties=2)
        store = _store(backend, tmp_path, gate)
        _run_together(
            lambda: _append_events(store, self.KEY, [{"event": "same"}]),
            lambda: _append_events(store, self.KEY, [{"event": "same"}]),
        )
        assert len(_lines(store, self.KEY)) == 1


class TestAppendConditionally:
    def test_a_body_that_rewrites_history_is_refused(self, tmp_path) -> None:
        store = LocalStore(tmp_path / "store")
        store.put_bytes("log.jsonl", b"a\n")
        with pytest.raises(ConditionalAppendError, match="does not begin with"):
            append_conditionally(store, "log.jsonl", lambda current: b"b\n")
        assert store.get_bytes("log.jsonl") == b"a\n"

    def test_a_writer_that_never_wins_raises(self, tmp_path) -> None:
        class _AlwaysLoses(LocalStore):
            def compare_and_swap(self, key: str, expected: str, payload: bytes) -> str:
                raise PointerConflictError("another writer won")

        store = _AlwaysLoses(tmp_path / "store")
        with pytest.raises(ConditionalAppendError, match="conditional write"):
            append_conditionally(store, "log.jsonl", lambda current: current + b"x\n")
        assert not store.exists("log.jsonl")

    def test_nothing_to_add_writes_nothing(self, tmp_path) -> None:
        store = LocalStore(tmp_path / "store")
        assert append_conditionally(store, "log.jsonl", lambda current: None) is False
        assert not store.exists("log.jsonl")
