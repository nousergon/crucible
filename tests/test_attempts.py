"""Every attempted configuration is registered (alpha-engine-config-I12563, audit CR-02).

The trial ledger counts arm-cycles that GRADED. A fit that raised, a grade
that failed, an early stop and a killed worker never reached its append, so
the family a selection correction saw was the family that survived. These
tests hold the closes-when of CR-02 against :mod:`crucible.attempts`: each of
those four outcomes, and a success, stays discoverable with its family and
config identity, provenance, allocation, terminal outcome and cost; a replay
moves the denominator once; and the selection family is complete.
"""

from __future__ import annotations

import json
import threading

import pytest
from conftest import seed_data_daily, sessions_ending

from crucible import attempts as at
from crucible.config import Settings
from crucible.keys import attempt_event_key, ledger_key
from crucible.ledger import n_trials
from crucible.runner import run_job
from crucible.slots import universe
from crucible.store import LocalStore, S3Store
from tests.conftest import FakeS3

AS_OF = "2026-10-16"


@pytest.fixture(params=["local", "s3"])
def any_store(request, tmp_path):
    if request.param == "s3":
        return S3Store("bucket", client=FakeS3())
    return LocalStore(tmp_path / "store")


def _dispatch(store, config_id: str, *, family: str = "u", as_of: str = AS_OF, **kw) -> str:
    return at.dispatch_attempt(
        store, family=family, config_id=config_id, as_of=as_of, run_id="run-1", **kw
    )


class TestLifecycle:
    def test_every_outcome_stays_discoverable(self, any_store) -> None:
        ok = _dispatch(any_store, "u:ok:1", provenance={"code_sha": "abc"})
        fit = _dispatch(any_store, "u:fit:1")
        early = _dispatch(any_store, "u:early:1")
        killed = _dispatch(any_store, "u:killed:1")
        at.terminate_attempt(
            any_store, ok, outcome=at.OUTCOME_SUCCEEDED, reason="graded", run_id="r"
        )
        at.terminate_attempt(
            any_store, fit, outcome=at.OUTCOME_FAILED, reason="ValueError: singular", run_id="r"
        )
        at.terminate_attempt(
            any_store, early, outcome=at.OUTCOME_EARLY_STOPPED, reason="budget", run_id="r"
        )
        # `killed` never reports an end: the worker died.

        by_config = {a.config_id: a for a in at.read_attempts(any_store)}
        assert {c: a.state for c, a in by_config.items()} == {
            "u:ok:1": at.OUTCOME_SUCCEEDED,
            "u:fit:1": at.OUTCOME_FAILED,
            "u:early:1": at.OUTCOME_EARLY_STOPPED,
            "u:killed:1": at.STATE_UNTERMINATED,
        }
        assert by_config["u:killed:1"].attempt_id == killed
        assert by_config["u:ok:1"].dispatched["provenance"] == {"code_sha": "abc"}
        # Unknown is named, never zero.
        assert by_config["u:fit:1"].dispatched["committed_cost"]["status"] == "unavailable"
        assert by_config["u:fit:1"].terminal["actual_cost"]["status"] == "unavailable"
        assert at.n_attempts(any_store, family="u") == 4

    def test_a_replay_moves_the_denominator_once(self, any_store) -> None:
        first = _dispatch(any_store, "u:a:1")
        again = _dispatch(any_store, "u:a:1")
        assert first == again
        at.terminate_attempt(any_store, first, outcome=at.OUTCOME_FAILED, reason="x", run_id="r")
        assert not at.terminate_attempt(
            any_store, first, outcome=at.OUTCOME_FAILED, reason="x again", run_id="r2"
        )
        # The same config on another date is the same hypothesis tried again.
        _dispatch(any_store, "u:a:1", as_of="2026-10-23")
        assert len(at.read_attempts(any_store)) == 2
        assert at.n_attempts(any_store) == 1

    def test_a_different_terminal_outcome_is_a_conflict(self, any_store) -> None:
        aid = _dispatch(any_store, "u:a:1")
        at.terminate_attempt(any_store, aid, outcome=at.OUTCOME_FAILED, reason="x", run_id="r")
        with pytest.raises(at.AttemptConflictError, match="immutable"):
            at.terminate_attempt(
                any_store, aid, outcome=at.OUTCOME_SUCCEEDED, reason="y", run_id="r2"
            )
        assert at.read_attempts(any_store)[0].state == at.OUTCOME_FAILED

    def test_a_terminal_event_needs_a_dispatch(self, any_store) -> None:
        aid = at.attempt_id(family="u", config_id="u:a:1", as_of=AS_OF)
        with pytest.raises(at.AttemptConflictError, match="no dispatched event"):
            at.terminate_attempt(any_store, aid, outcome=at.OUTCOME_FAILED, reason="x", run_id="r")
        assert not any_store.exists(attempt_event_key(aid, "terminal"))

    def test_the_family_is_complete_and_controls_are_separate(self, any_store) -> None:
        a = _dispatch(any_store, "u:a:1")
        _dispatch(any_store, "u:b:1")
        _dispatch(any_store, "u:null:1", control=True)
        _dispatch(any_store, "r:a:1", family="r")
        at.terminate_attempt(any_store, a, outcome=at.OUTCOME_FAILED, reason="x", run_id="r")
        assert [x.config_id for x in at.family_attempts(any_store, "u")] == sorted(
            ["u:a:1", "u:b:1"], key=lambda c: at.attempt_id(family="u", config_id=c, as_of=AS_OF)
        )
        assert at.n_attempts(any_store, family="u") == 2
        assert at.n_attempts(any_store, family="u", include_controls=True) == 3
        assert at.n_attempts(any_store) == 3

    def test_concurrent_dispatches_both_land(self, any_store) -> None:
        barrier = threading.Barrier(8, timeout=10.0)
        errors: list[BaseException] = []

        def worker(i: int) -> None:
            try:
                barrier.wait()
                _dispatch(any_store, f"u:arm{i}:1")
            except BaseException as exc:  # noqa: BLE001 - collected and asserted below
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(15.0)
        assert errors == []
        assert at.n_attempts(any_store) == 8


class TestAttemptLifecycleHelper:
    def test_end_open_marks_every_unended_attempt(self, any_store) -> None:
        life = at.AttemptLifecycle(any_store, family="u", as_of=AS_OF, run_id="r")
        life.dispatch(["u:a:1", "u:b:1"], control_ids=["u:b:1"])
        life.end_open(at.OUTCOME_KILLED, "KeyboardInterrupt: ")
        states = {a.config_id: (a.state, a.control) for a in at.read_attempts(any_store)}
        assert states == {"u:a:1": ("killed", False), "u:b:1": ("killed", True)}
        assert life.open_attempts == {}
        cost = at.read_attempts(any_store)[0].terminal["actual_cost"]
        assert cost["status"] == "unavailable"
        assert "run_id" in cost["reason"]


HORIZON = 21
DECISION_DATES = 6


def _produce(store, strategy_dir, cycle_date, tmp_path, source) -> Settings:
    settings = Settings(
        store_uri=str(tmp_path / "store"),
        arctic_bucket="unused-in-this-test",
        strategy_dir=strategy_dir,
        origins={"store_uri": "test", "strategy_dir": "test"},
    )
    sessions = sessions_ending(cycle_date, HORIZON + DECISION_DATES + 1)
    decision_days = sessions[:DECISION_DATES]
    seed_data_daily(store, source, [*decision_days, cycle_date])
    for day in decision_days:
        run_job(
            "experiment.run",
            lambda c: universe.produce(c, settings=settings),
            store=store,
            trading_day=day,
        )
    return settings


class TestRunGradeRegistersAttempts:
    def test_a_graded_cycle_ends_every_attempt_succeeded(
        self, store, source, strategy_dir, cycle_date, tmp_path
    ) -> None:
        settings = _produce(store, strategy_dir, cycle_date, tmp_path, source)
        run_job(
            "experiment.grade",
            lambda c: universe.grade(c, settings=settings),
            store=store,
            trading_day=cycle_date,
        )
        found = at.read_attempts(store)
        assert found, "a graded cycle registered no attempts"
        assert {a.state for a in found} == {at.OUTCOME_SUCCEEDED}
        for a in found:
            assert a.family == "u"
            assert a.dispatched["as_of"] == cycle_date.isoformat()
            assert a.dispatched["provenance"]["data_panel_key"]
            assert a.terminal["links"]["trial_ledger_key"] == ledger_key()
        # Every graded, non-control trial is an attempt; controls are flagged.
        assert at.n_attempts(store, family="u") == n_trials(store, slot="u")
        assert any(a.control for a in found)

    def test_a_failed_grade_leaves_failed_attempts_and_no_trials(
        self, store, source, strategy_dir, cycle_date, tmp_path, monkeypatch
    ) -> None:
        settings = _produce(store, strategy_dir, cycle_date, tmp_path, source)

        def boom(*_a, **_k):
            raise RuntimeError("grader fell over")

        monkeypatch.setattr("crucible.slots.cycle.grade_slot", boom)
        with pytest.raises(RuntimeError, match="grader fell over"):
            run_job(
                "experiment.grade",
                lambda c: universe.grade(c, settings=settings),
                store=store,
                trading_day=cycle_date,
            )
        found = at.read_attempts(store)
        assert found, "a failed grade left no trace of what it attempted"
        assert {a.state for a in found} == {at.OUTCOME_FAILED}
        assert all("grader fell over" in a.terminal["reason"] for a in found)
        # The survivors-only ledger saw nothing; the attempt registry saw it all.
        assert not store.exists(ledger_key())
        assert at.n_attempts(store, family="u") > 0
        dispatched = json.loads(
            store.get_bytes(attempt_event_key(found[0].attempt_id, "dispatched"))
        )
        assert dispatched["phase"] == "dispatched"
