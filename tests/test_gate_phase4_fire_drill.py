"""The phase-4 kill-switch fire drill clause and its register entry (`alpha-engine-config-I10650`).

Mutation-tested against each way a drill record can read as evidence while not
being any: a `passed` flag the evidence contradicts, an order let through after
the fire, a bound overrun, a document with no run behind it, a document edited
after its run, a replay, only announced drills, only one drill — and the two
readings that must never collapse (absence is UNMET; an unreadable store is
UNMEASURABLE).
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Any

import pytest
from test_gate_phase4_trader_clauses import _manifest, _put

from crucible.faults import ARTIFACT_GRADED_FAULTS, FaultRecordRefusedError, record_fault
from crucible.gate import (
    FIRE_DRILL_FAULT,
    FIRE_DRILL_SCHEMA_VERSION,
    PHASE4_DELIVERABLES,
    RECONCILIATION_FAULT,
    SCRIPTED_FAULTS,
    TRADER_FIRE_DRILL_JOB,
    _clause_kill_switch_fire_drill_passed,
    _phase4,
)
from crucible.keys import TRADER_FIRE_DRILLS_PREFIX, manifest_key, trader_fire_drill_key
from crucible.models import JOB_VALUES, TRADER_JOB_VALUES, FireDrillDocument
from crucible.store import LocalStore, sha256_hex

WINDOW = [dt.date(2026, 9, 1), dt.date(2026, 9, 11)]
FRIDAY = WINDOW[-1]


@pytest.fixture
def store(tmp_path) -> LocalStore:
    return LocalStore(tmp_path / "store")


def _run_id(n: int) -> str:
    return f"01JG{n:022d}"


def _drill(day: str, run_id: str, **overrides: Any) -> dict:
    """Exactly the fields `crucible_trader.fire_drill.drill_document` writes."""
    document = {
        "schema_version": FIRE_DRILL_SCHEMA_VERSION,
        "kind": "kill_switch_freeze",
        "announced": True,
        "run_id": run_id,
        "trading_day": day,
        "account": "DU0000000",
        "mode": "freeze",
        "fired_at": f"{day}T15:00:00.000000Z",
        "settled_at": f"{day}T15:00:04.000000Z",
        "settle_seconds": 4.0,
        "bound_seconds": 60.0,
        "state_before": {"positions": {"AAA": 10}},
        "state_after": {"positions": {"AAA": 10}},
        "orders_accepted_after_fire": [],
        "hold_decision": None,
        "passed": True,
    }
    document.update(overrides)
    return document


def _drill_run(
    store: LocalStore,
    day: str,
    n: int,
    *,
    record_output: bool = True,
    status: str = "ok",
    run_mode: str = "live",
    **drill_overrides: Any,
) -> str:
    run_id = _run_id(n)
    key = trader_fire_drill_key(day, run_id)
    payload = _put(store, key, _drill(day, run_id, **drill_overrides))
    outputs = (
        [{"key": key, "sha256": sha256_hex(payload), "schema_version": FIRE_DRILL_SCHEMA_VERSION}]
        if record_output
        else []
    )
    manifest = _manifest(day, outputs=outputs, status=status, run_mode=run_mode)
    manifest |= {"job": TRADER_FIRE_DRILL_JOB, "run_id": run_id}
    _put(store, manifest_key(TRADER_FIRE_DRILL_JOB, day, discriminator=f"drill-{n}"), manifest)
    return key


def _clause(store: LocalStore):
    return _clause_kill_switch_fire_drill_passed(store, WINDOW)


def _two_good(store: LocalStore) -> None:
    _drill_run(store, "2026-09-08", 1)
    _drill_run(store, "2026-09-10", 2, announced=False)


class TestMet:
    def test_two_passed_one_unannounced_is_met(self, store) -> None:
        _two_good(store)
        clause = _clause(store)
        assert clause.met and not clause.unmeasurable, clause.detail
        assert "2 passed drill(s)" in clause.detail

    def test_a_failed_drill_beside_two_passes_is_named_not_voiding(self, store) -> None:
        _two_good(store)
        _drill_run(store, "2026-09-09", 3, status="failed", settled_at=None, settle_seconds=None)
        clause = _clause(store)
        assert clause.met
        assert "failed:" in clause.detail and _run_id(3) in clause.detail


class TestUnmet:
    def test_an_empty_store_is_unmet_never_unmeasurable(self, store) -> None:
        clause = _clause(store)
        assert not clause.met and not clause.unmeasurable
        assert "undrilled" in clause.detail

    def test_one_drill_is_not_two(self, store) -> None:
        _drill_run(store, "2026-09-08", 1, announced=False)
        assert not _clause(store).met

    def test_only_announced_drills_are_unmet(self, store) -> None:
        _drill_run(store, "2026-09-08", 1)
        _drill_run(store, "2026-09-10", 2)
        clause = _clause(store)
        assert not clause.met and "0 unannounced" in clause.detail

    def test_drills_outside_the_window_do_not_count(self, store) -> None:
        _drill_run(store, "2026-08-27", 1)
        _drill_run(store, "2026-08-28", 2, announced=False)
        assert not _clause(store).met

    @pytest.mark.parametrize(
        "mutation",
        [
            pytest.param({"settled_at": None, "settle_seconds": None}, id="never-settled"),
            pytest.param({"settle_seconds": 61.0}, id="over-its-bound"),
            pytest.param(
                {
                    "orders_accepted_after_fire": [
                        {
                            "order_id": 7,
                            "symbol": "AAA",
                            "action": "BUY",
                            "quantity": 1.0,
                            "order_ref": "",
                            "accepted_at": "2026-09-10T15:00:01.000000Z",
                        }
                    ]
                },
                id="order-accepted-after-the-fire",
            ),
        ],
    )
    def test_a_passed_flag_the_evidence_contradicts_does_not_count(
        self, store, mutation: dict
    ) -> None:
        """The flag says passed; the evidence does not. The evidence wins."""
        _drill_run(store, "2026-09-08", 1)
        _drill_run(store, "2026-09-10", 2, announced=False, **mutation)
        clause = _clause(store)
        assert not clause.met and not clause.unmeasurable
        assert _run_id(2) in clause.detail

    def test_a_document_with_no_run_behind_it_does_not_count(self, store) -> None:
        _drill_run(store, "2026-09-08", 1)
        run_id = _run_id(2)
        _put(
            store,
            trader_fire_drill_key("2026-09-10", run_id),
            _drill("2026-09-10", run_id, announced=False),
        )
        clause = _clause(store)
        assert not clause.met and "no run behind it" in clause.detail

    def test_a_document_edited_after_its_run_does_not_count(self, store) -> None:
        _drill_run(store, "2026-09-08", 1)
        key = _drill_run(store, "2026-09-10", 2, announced=True)
        edited = json.loads(store.get_bytes(key)) | {"announced": False}
        _put(store, key, edited)
        clause = _clause(store)
        assert not clause.met and "changed after the run" in clause.detail

    def test_a_document_the_run_did_not_record_does_not_count(self, store) -> None:
        _drill_run(store, "2026-09-08", 1)
        _drill_run(store, "2026-09-10", 2, announced=False, record_output=False)
        clause = _clause(store)
        assert not clause.met and "0 output(s)" in clause.detail

    def test_a_replayed_drill_does_not_count(self, store) -> None:
        _drill_run(store, "2026-09-08", 1)
        _drill_run(store, "2026-09-10", 2, announced=False, run_mode="replay")
        clause = _clause(store)
        assert not clause.met and "replayed drill" in clause.detail

    def test_passed_against_a_failed_manifest_does_not_count(self, store) -> None:
        _drill_run(store, "2026-09-08", 1)
        _drill_run(store, "2026-09-10", 2, announced=False, status="failed")
        clause = _clause(store)
        assert not clause.met and "must agree" in clause.detail

    def test_a_non_conforming_document_is_unmet_naming_the_key(self, store) -> None:
        _drill_run(store, "2026-09-08", 1)
        key = trader_fire_drill_key("2026-09-10", _run_id(2))
        _put(store, key, {"schema_version": "fire_drill.v0"})
        clause = _clause(store)
        assert not clause.met and not clause.unmeasurable and key in clause.detail

    def test_a_document_filed_under_another_run_id_does_not_count(self, store) -> None:
        _drill_run(store, "2026-09-08", 1)
        run_id = _run_id(2)
        payload = _put(
            store,
            trader_fire_drill_key("2026-09-10", _run_id(9)),
            _drill("2026-09-10", run_id, announced=False),
        )
        del payload
        clause = _clause(store)
        assert not clause.met and "which belong at" in clause.detail


class TestUnmeasurable:
    def test_an_unlistable_drill_prefix_is_unmeasurable(self, store, monkeypatch) -> None:
        def denied(prefix: str):
            raise PermissionError(f"AccessDenied listing {prefix}")

        monkeypatch.setattr(store, "list_keys", denied)
        clause = _clause(store)
        assert clause.unmeasurable and not clause.met

    def test_a_denied_read_is_unmeasurable(self, store, monkeypatch) -> None:
        _two_good(store)
        real = store.get_bytes

        def denied(key: str) -> bytes:
            if key.startswith(TRADER_FIRE_DRILLS_PREFIX):
                raise PermissionError("AccessDenied")
            return real(key)

        monkeypatch.setattr(store, "get_bytes", denied)
        clause = _clause(store)
        assert clause.unmeasurable and not clause.met


class TestRegistration:
    def test_both_trader_jobs_are_admitted(self) -> None:
        for job in ("trader.kill_switch", TRADER_FIRE_DRILL_JOB):
            assert job in TRADER_JOB_VALUES and job in JOB_VALUES

    def test_the_deliverable_names_the_clause_phase_4_evaluates(self, store) -> None:
        (row,) = [d for d in PHASE4_DELIVERABLES if d.id == "kill_switch_fire_drill"]
        assert row.graded_by == "kill_switch_fire_drill_passed"
        readings = {c.name: c for c in _phase4(store, WINDOW, {}, trading_day=FRIDAY)}
        assert not readings["kill_switch_fire_drill_passed"].met
        assert not readings["kill_switch_fire_drill_passed"].unmeasurable

    def test_the_fixture_is_the_model(self) -> None:
        """The test fixture above mirrors crucible-trader's writer field for field."""
        FireDrillDocument.model_validate(_drill("2026-09-08", _run_id(1)))


class TestRegisterEntry:
    def test_the_drill_is_fault_fives_sibling(self) -> None:
        assert SCRIPTED_FAULTS[-2:] == (RECONCILIATION_FAULT, FIRE_DRILL_FAULT)

    def test_the_entry_and_the_clause_read_the_same_artifact(self) -> None:
        entry = ARTIFACT_GRADED_FAULTS[FIRE_DRILL_FAULT]
        assert entry.producer_job == TRADER_FIRE_DRILL_JOB
        assert entry.artifact_prefix == TRADER_FIRE_DRILLS_PREFIX
        assert entry.schema_version == FIRE_DRILL_SCHEMA_VERSION
        assert entry.graded_by == _clause_kill_switch_fire_drill_passed.__name__.removeprefix(
            "_clause_"
        )
        assert trader_fire_drill_key("2026-09-08", "r").startswith(entry.artifact_prefix)

    def test_a_fault_record_for_the_drill_is_refused(self, store) -> None:
        with pytest.raises(FaultRecordRefusedError, match="graded from its producer's own"):
            record_fault(
                None,  # type: ignore[arg-type] - refused before the context is touched
                store,
                fault_id=FIRE_DRILL_FAULT,
                outcome="induced",
                target_job=TRADER_FIRE_DRILL_JOB,
                trading_day="2026-09-08",
                run_id=_run_id(1),
                bus_key="alerts/2026-09-08/x.json",
            )
