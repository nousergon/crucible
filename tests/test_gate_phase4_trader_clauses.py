"""The three phase-4 trader clauses (`alpha-engine-config-I10651`, `-I10746`).

Each reads an artifact the trader AGREED to write — the harness never reaches
into the trader (plan §3) — and each is tested against the defect it exists
to catch, not only against a clean week: a clause that trusted `status: ok`
would pass a reconciler whose control arm was blind, a clause that read only
the row's value would pass a week whose shortfall was never computed, and a
clause that dropped a failed shadow book would pass a register it cannot see.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Any

import pytest
from test_execution_contract import (
    ARM,
    CHALLENGER,
    _book,
    _failed,
    _no_orders_doc,
    _order,
    _shadow_doc,
    _shortfall_doc,
)

import crucible.gate as gate_module
from crucible.gate import (
    PHASE2_SCRIPTED_FAULTS,
    PHASE4_DELIVERABLES,
    RECONCILIATION_FAULT,
    SCRIPTED_FAULTS,
    TRADER_RECONCILE_JOB,
    _clause_broker_reconciliation_control_arm_passed,
    _clause_execution_shortfall_row_graded,
    _clause_shadow_books_cover_every_active_arm,
    _phase4,
)
from crucible.keys import (
    TRADER_EVIDENCE_KEY,
    execution_shortfall_key,
    shadow_books_key,
    trader_reconciliation_key,
)
from crucible.manifest import RUN_MANIFEST_SCHEMA_VERSION, manifest_key
from crucible.models import TRADER_JOB_VALUES
from crucible.store import LocalStore, sha256_hex

FRIDAY = dt.date(2026, 9, 11)
WINDOW = [dt.date(2026, 9, 4), FRIDAY]
#: The five sessions ending 2026-09-11; 2026-09-07 is Labor Day.
SESSIONS = ["2026-09-04", "2026-09-08", "2026-09-09", "2026-09-10", "2026-09-11"]


@pytest.fixture
def store(tmp_path) -> LocalStore:
    return LocalStore(tmp_path / "store")


def _put(store: LocalStore, key: str, doc: Any) -> bytes:
    payload = json.dumps(doc, sort_keys=True).encode("utf-8")
    store.put_bytes(key, payload)
    return payload


# ── broker reconciliation ───────────────────────────────────────────────────


def _reconciliation(day: str, *, passed: bool = True, void: Any = None, fault: str = "") -> dict:
    return {
        "schema_version": "broker_reconciliation.v1",
        "trading_day": day,
        "discrepancies": [],
        "control_arm": {
            "fault": fault or RECONCILIATION_FAULT,
            "passed": passed,
            "n_planted": 3,
            "n_detected": 3 if passed else 2,
            "missed": [] if passed else ["cash_delta"],
            "outcomes": [],
        },
        "void": (not passed) if void is None else void,
    }


def _manifest(day: str, *, outputs: list[dict], status: str = "ok", run_mode: str = "live") -> dict:
    return {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": "01JG" + day.replace("-", "") + "00000000000000",
        "job": TRADER_RECONCILE_JOB,
        "run_mode": run_mode,
        "trading_day": day,
        "calendar_date": day,
        "status": status,
        "reason": "" if status == "ok" else "ReconciliationVoidError: missed cash_delta",
        "started": f"{day}T21:00:00Z",
        "finished": f"{day}T21:01:00Z",
        "code_sha": "a" * 40,
        "release_sha": "b" * 40,
        "seed": 7,
        "inputs": [],
        "outputs": outputs,
        "rows_in": 0,
        "rows_out": 0,
        "rows_rejected": [],
        "cost_usd": 0.0,
        "llm_calls": [],
        "resource": {
            "instance_type": "local",
            "spot": False,
            "escalated_to_on_demand": False,
            "interruptions": 0,
            "mem_peak_mb": 1.0,
            "disk_free_mb": 1.0,
        },
        "metrics": [],
        "attempts": [{"n": 1, "reason": "initial"}],
    }


def _reconcile_run(
    store: LocalStore,
    day: str,
    *,
    document: dict | None = None,
    record_output: bool = True,
    **manifest_overrides: Any,
) -> None:
    result_key = trader_reconciliation_key(day)
    payload = _put(store, result_key, document or _reconciliation(day))
    outputs = (
        [
            {
                "key": result_key,
                "sha256": sha256_hex(payload),
                "schema_version": "broker_reconciliation.v1",
            }
        ]
        if record_output
        else []
    )
    _put(
        store,
        manifest_key(TRADER_RECONCILE_JOB, day),
        _manifest(day, outputs=outputs, **manifest_overrides),
    )


def _week(store: LocalStore, **overrides_for_last: Any) -> None:
    for day in SESSIONS[:-1]:
        _reconcile_run(store, day)
    _reconcile_run(store, SESSIONS[-1], **overrides_for_last)


def _reconciliation_clause(store: LocalStore):
    return _clause_broker_reconciliation_control_arm_passed(store, WINDOW)


class TestBrokerReconciliationClause:
    def test_five_live_sessions_with_the_control_passed_is_met(self, store) -> None:
        _week(store)
        clause = _reconciliation_clause(store)
        assert clause.met and not clause.unmeasurable, clause.detail
        assert manifest_key(TRADER_RECONCILE_JOB, SESSIONS[0]) in clause.evidence
        assert trader_reconciliation_key(SESSIONS[-1]) in clause.evidence

    def test_the_job_name_is_a_declared_trader_job(self) -> None:
        assert TRADER_RECONCILE_JOB in TRADER_JOB_VALUES

    def test_no_manifest_at_all_is_unmet_not_unmeasurable(self, store) -> None:
        clause = _reconciliation_clause(store)
        assert not clause.met and not clause.unmeasurable
        assert "not running" in clause.detail

    def test_one_missing_session_is_unmet_and_named(self, store) -> None:
        for day in SESSIONS[1:]:
            _reconcile_run(store, day)
        clause = _reconciliation_clause(store)
        assert not clause.met and not clause.unmeasurable
        assert SESSIONS[0] in clause.detail

    def test_a_failed_run_is_unmet(self, store) -> None:
        _week(store, status="failed", document=_reconciliation(SESSIONS[-1], passed=False))
        clause = _reconciliation_clause(store)
        assert not clause.met
        assert "status `failed`" in clause.detail

    def test_a_replay_is_not_a_reconciliation_of_todays_book(self, store) -> None:
        _week(store, run_mode="replay")
        clause = _reconciliation_clause(store)
        assert not clause.met
        assert "replay" in clause.detail

    def test_an_ok_run_whose_control_arm_missed_a_plant_is_unmet(self, store) -> None:
        """The mutation this clause exists for: a producer that stopped raising
        on a void cycle writes `status: ok`, and a clause trusting the status
        would pass a reconciler that cannot see a planted discrepancy."""
        _week(store, document=_reconciliation(SESSIONS[-1], passed=False, void=False))
        clause = _reconciliation_clause(store)
        assert not clause.met
        assert "control_arm.passed is False" in clause.detail
        assert "cash_delta" in clause.detail

    def test_void_disagreeing_with_a_passed_control_is_unmet(self, store) -> None:
        _week(store, document=_reconciliation(SESSIONS[-1], passed=True, void=True))
        clause = _reconciliation_clause(store)
        assert not clause.met
        assert "void is True" in clause.detail

    def test_a_control_arm_for_another_fault_is_unmet(self, store) -> None:
        _week(store, document=_reconciliation(SESSIONS[-1], fault="some_other_control"))
        clause = _reconciliation_clause(store)
        assert not clause.met
        assert "control_arm.fault" in clause.detail

    def test_a_result_with_no_control_arm_is_unmet(self, store) -> None:
        document = _reconciliation(SESSIONS[-1])
        del document["control_arm"]
        _week(store, document=document)
        clause = _reconciliation_clause(store)
        assert not clause.met
        assert "control never ran" in clause.detail

    def test_a_result_edited_after_the_run_is_unmet(self, store) -> None:
        """The stored bytes must still hash to the digest the run recorded."""
        _week(store, document=_reconciliation(SESSIONS[-1], passed=False, void=False))
        _put(store, trader_reconciliation_key(SESSIONS[-1]), _reconciliation(SESSIONS[-1]))
        clause = _reconciliation_clause(store)
        assert not clause.met
        assert "no longer hashes" in clause.detail

    def test_a_manifest_that_does_not_record_the_result_is_unmet(self, store) -> None:
        _week(store, record_output=False)
        clause = _reconciliation_clause(store)
        assert not clause.met
        assert "0 output(s)" in clause.detail

    def test_a_manifest_claiming_an_absent_result_is_unmet(self, store) -> None:
        _week(store)
        (store.root / trader_reconciliation_key(SESSIONS[-1])).unlink()
        clause = _reconciliation_clause(store)
        assert not clause.met
        assert "does not hold it" in clause.detail

    def test_a_nonconforming_manifest_is_unmet(self, store) -> None:
        _week(store)
        key = manifest_key(TRADER_RECONCILE_JOB, SESSIONS[-1])
        manifest = json.loads(store.get_bytes(key))
        del manifest["run_mode"]
        _put(store, key, manifest)
        clause = _reconciliation_clause(store)
        assert not clause.met and not clause.unmeasurable
        assert "does not conform" in clause.detail

    def test_a_denied_read_is_unmeasurable(self, store, monkeypatch) -> None:
        _week(store)

        def denied(key: str) -> bytes:
            raise PermissionError(f"AccessDenied: {key}")

        monkeypatch.setattr(store, "get_bytes", denied)
        clause = _reconciliation_clause(store)
        assert clause.unmeasurable and not clause.met


# ── execution shortfall ─────────────────────────────────────────────────────


def _computed(day: str, *, fill: float = 100.05) -> dict:
    orders = [_order(f"{day}-o{n}", fill=fill) for n in range(4)]
    for order in orders:
        order["decided_at_utc"] = f"{day}T19:55:00Z"
        order["submitted_at_utc"] = f"{day}T19:55:02Z"
    return _shortfall_doc(day, orders=orders)


def _shortfall_clause(store: LocalStore):
    return _clause_execution_shortfall_row_graded(store, WINDOW)


class TestExecutionShortfallClause:
    def test_a_graded_week_inside_the_band_is_met(self, store) -> None:
        for day in SESSIONS:
            _put(store, execution_shortfall_key(day), _computed(day))
        clause = _shortfall_clause(store)
        assert clause.met and not clause.unmeasurable, clause.detail
        assert execution_shortfall_key(SESSIONS[0]) in clause.evidence

    def test_nothing_filed_is_unmet_not_unmeasurable(self, store) -> None:
        clause = _shortfall_clause(store)
        assert not clause.met and not clause.unmeasurable
        assert "filed no execution-shortfall artifact" in clause.detail

    def test_a_not_computed_session_is_unmet_even_beside_good_sessions(self, store) -> None:
        for day in SESSIONS[:-1]:
            _put(store, execution_shortfall_key(day), _computed(day))
        _put(
            store,
            execution_shortfall_key(SESSIONS[-1]),
            _shortfall_doc(
                SESSIONS[-1],
                outcome="not_computed",
                outcome_reason="broker fill feed down",
                orders=[],
                summary=None,
                metrics=[],
            ),
        )
        clause = _shortfall_clause(store)
        assert not clause.met
        assert "NOT computed" in clause.detail
        assert SESSIONS[-1] in clause.detail

    def test_no_orders_all_window_is_unmet_with_its_own_reason(self, store) -> None:
        for day in SESSIONS:
            _put(store, execution_shortfall_key(day), _no_orders_doc(day))
        clause = _shortfall_clause(store)
        assert not clause.met and not clause.unmeasurable
        assert "no_orders" in clause.detail
        assert "not a producer failure" in clause.detail

    def test_a_band_breach_is_unmet(self, store) -> None:
        for day in SESSIONS:
            _put(store, execution_shortfall_key(day), _computed(day, fill=101.0))
        clause = _shortfall_clause(store)
        assert not clause.met
        assert "RED" in clause.detail

    def test_a_corrupt_artifact_is_unmet_naming_the_key_not_unmeasurable(self, store) -> None:
        for day in SESSIONS[:-1]:
            _put(store, execution_shortfall_key(day), _computed(day))
        store.put_bytes(execution_shortfall_key(SESSIONS[-1]), b'{"schema_version": ')
        clause = _shortfall_clause(store)
        assert not clause.met and not clause.unmeasurable
        assert execution_shortfall_key(SESSIONS[-1]) in clause.detail


# ── shadow books ────────────────────────────────────────────────────────────


def _evidence(days: list[str]) -> dict:
    return {
        "schema_version": "trader_evidence.v1",
        "slot": "m",
        "champion": "m:ridge_21d:0123456789ab",
        "trading_days": len(days),
        "days_served": days,
        "calendar_date": days[-1],
    }


def _shadow_clause(store: LocalStore):
    return _clause_shadow_books_cover_every_active_arm(store, WINDOW)


class TestShadowBooksClause:
    """`test_execution_contract`'s books advance on 2026-09-08..09-11."""

    SERVED = ["2026-09-08", "2026-09-09", "2026-09-10", "2026-09-11"]

    def test_every_active_arm_advanced_on_every_served_day_is_met(self, store) -> None:
        _put(store, TRADER_EVIDENCE_KEY, _evidence(self.SERVED))
        _put(store, shadow_books_key("2026-09-11"), _shadow_doc())
        clause = _shadow_clause(store)
        assert clause.met and not clause.unmeasurable, clause.detail
        assert set(clause.evidence) == {shadow_books_key("2026-09-11"), TRADER_EVIDENCE_KEY}

    def test_an_absent_evidence_document_is_unmet_not_unmeasurable(self, store) -> None:
        _put(store, shadow_books_key("2026-09-11"), _shadow_doc())
        clause = _shadow_clause(store)
        assert not clause.met and not clause.unmeasurable
        assert TRADER_EVIDENCE_KEY in clause.detail

    def test_a_nonconforming_evidence_document_is_unmet(self, store) -> None:
        bad = _evidence(self.SERVED)
        bad["trading_days"] = 99
        _put(store, TRADER_EVIDENCE_KEY, bad)
        _put(store, shadow_books_key("2026-09-11"), _shadow_doc())
        clause = _shadow_clause(store)
        assert not clause.met and not clause.unmeasurable
        assert "trader_evidence.v1" in clause.detail

    def test_an_absent_shadow_book_artifact_is_unmet(self, store) -> None:
        _put(store, TRADER_EVIDENCE_KEY, _evidence(self.SERVED))
        clause = _shadow_clause(store)
        assert not clause.met
        assert "absent" in clause.detail

    def test_a_failed_book_is_unmet_and_named(self, store) -> None:
        _put(store, TRADER_EVIDENCE_KEY, _evidence(self.SERVED))
        _put(store, shadow_books_key("2026-09-11"), _shadow_doc([_book(ARM), _failed(CHALLENGER)]))
        clause = _shadow_clause(store)
        assert not clause.met
        assert CHALLENGER in clause.detail

    def test_a_served_day_a_book_did_not_advance_is_unmet(self, store) -> None:
        _put(store, TRADER_EVIDENCE_KEY, _evidence(self.SERVED))
        skipped = _book(CHALLENGER, days=["2026-09-08", "2026-09-10", "2026-09-11"])
        _put(store, shadow_books_key("2026-09-11"), _shadow_doc([_book(ARM), skipped]))
        clause = _shadow_clause(store)
        assert not clause.met
        assert "2026-09-09" in clause.detail

    def test_days_served_after_the_session_are_not_held_against_it(self, store) -> None:
        _put(store, TRADER_EVIDENCE_KEY, _evidence([*self.SERVED, "2026-09-14"]))
        _put(store, shadow_books_key("2026-09-11"), _shadow_doc())
        clause = _shadow_clause(store)
        assert clause.met, clause.detail

    def test_a_corrupt_shadow_book_artifact_is_unmet_naming_the_key(self, store) -> None:
        _put(store, TRADER_EVIDENCE_KEY, _evidence(self.SERVED))
        _put(store, shadow_books_key("2026-09-11"), _shadow_doc(active=[ARM]))
        clause = _shadow_clause(store)
        assert not clause.met and not clause.unmeasurable
        assert shadow_books_key("2026-09-11") in clause.detail


# ── registration and fault 5 ────────────────────────────────────────────────


class TestRegistration:
    @pytest.mark.parametrize(
        ("deliverable", "clause"),
        [
            ("broker_reconciliation", "broker_reconciliation_control_arm_passed"),
            ("execution_shortfall_attribution_row", "execution_shortfall_row_graded"),
            ("shadow_books_per_challenger", "shadow_books_cover_every_active_arm"),
        ],
    )
    def test_each_deliverable_names_a_clause_phase_4_evaluates(
        self, store, deliverable: str, clause: str
    ) -> None:
        (row,) = [d for d in PHASE4_DELIVERABLES if d.id == deliverable]
        assert row.graded_by == clause
        names = [c.name for c in _phase4(store, WINDOW, {}, trading_day=FRIDAY)]
        assert clause in names

    def test_an_empty_store_reads_each_clause_unmet_never_met(self, store) -> None:
        readings = {c.name: c for c in _phase4(store, WINDOW, {}, trading_day=FRIDAY)}
        for name in (
            "broker_reconciliation_control_arm_passed",
            "execution_shortfall_row_graded",
            "shadow_books_cover_every_active_arm",
        ):
            assert not readings[name].met, name
            assert not readings[name].unmeasurable, readings[name].detail


class TestFaultFive:
    def test_fault_five_is_scripted_and_recordable(self) -> None:
        assert SCRIPTED_FAULTS[-1] == RECONCILIATION_FAULT
        assert len(SCRIPTED_FAULTS) == 5

    def test_phase_two_still_grades_its_own_four(self, store) -> None:
        """Fault 5 is a phase-4 gate. Phase 2's reading is in flight, and a
        fifth member would move its denominator onto a deliverable it never
        owned."""
        assert RECONCILIATION_FAULT not in PHASE2_SCRIPTED_FAULTS
        assert len(PHASE2_SCRIPTED_FAULTS) == 4
        clause = gate_module._clause_fault_injection_against_scheduled_path(store)
        assert "each of the 4 phase-2 scripted faults" in clause.requirement
        assert RECONCILIATION_FAULT not in clause.detail
