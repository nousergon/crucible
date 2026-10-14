"""A phase that has exited STAYS exited: the closing record latches it.

The defect these tests exist for: phase 1 closed 6/6 on 2026-09-08 and its
closing record was filed at `gates/phase1/closing.json`, yet the ladder kept
re-grading it live over a rolling window. One-off exit evidence aged out of
the window and a single failed weekly arc un-met it, so phase 1 flipped
MET/UNMET some eight times — and every later phase rendered OUT_OF_ORDER each
time it did, so one failure dimmed the whole ladder.

The latch keeps both halves of the truth: the phase reads MET (it exited,
and the record says so), and every clause is still read every render, so a
clause that fails now is a REGRESSION rendered red on its own board row. Only
an explicit reopen record un-latches a phase.

Every test below was seen failing before the code that makes it pass.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from crucible.board import RED_STATES, _phase_rows
from crucible.gate import (
    GATE_DELIVERABLES,
    GATES,
    PHASES,
    Clause,
    build_ladder,
    closing_reading,
    evaluate,
    gate_key,
    ladder_payload,
)
from crucible.keys import closing_record_key, reopen_record_key
from crucible.store import LocalStore

FRIDAY = dt.date(2026, 8, 28)
NOW = dt.datetime(2026, 8, 29, 12, 0, tzinfo=dt.UTC)
COMMIT = "c06e4efb90c31b6b84e5e4d4925688ee2ea99054"
CLOSED_UTC = "2026-08-28T21:00:00Z"


@pytest.fixture
def store(tmp_path) -> LocalStore:
    return LocalStore(tmp_path)


def _clauses(*verdicts: tuple[str, bool]):
    def clause_fn(*_a, **_k):
        return [
            Clause(name, f"req {name}", met, "met" if met else "not met", ())
            for name, met in verdicts
        ]

    return clause_fn


def _set_gate(monkeypatch: pytest.MonkeyPatch, gate: str, *verdicts: tuple[str, bool]) -> None:
    monkeypatch.setitem(GATES, gate, (5, _clauses(*verdicts)))
    monkeypatch.setitem(GATE_DELIVERABLES, gate, ())


def _file_reading(store: LocalStore, gate: str, day: dt.date = FRIDAY) -> None:
    store.put_bytes(
        gate_key(gate, day.isoformat()),
        json.dumps({"gate": gate, "trading_day": day.isoformat()}).encode("utf-8"),
    )


def _close(
    store: LocalStore,
    monkeypatch: pytest.MonkeyPatch,
    gate: str,
    names: tuple[str, ...],
    **overrides,
) -> dict:
    """File a real closing record: the gate reads MET on FRIDAY and the
    record is the `closing_reading` transcript of that reading."""
    _set_gate(monkeypatch, gate, *((n, True) for n in names))
    reading = evaluate(store, gate=gate, trading_day=FRIDAY)
    document = closing_reading(reading, store_uri="local://test", commit=COMMIT)
    document["generated_utc"] = CLOSED_UTC
    document.update(overrides)
    store.put_bytes(closing_record_key(gate), json.dumps(document).encode("utf-8"))
    return document


def _reopen(store: LocalStore, phase: str, *, filed_utc: str, **overrides) -> None:
    document = {
        "phase": phase,
        "reason": "the exit evidence was found to be wrong",
        "ruled_by": "Brian",
        "filed_utc": filed_utc,
    }
    document.update(overrides)
    store.put_bytes(reopen_record_key(phase), json.dumps(document).encode("utf-8"))


def _rows(store: LocalStore) -> dict[str, dict]:
    return {
        r["phase"]: r for r in build_ladder(store, trading_day=FRIDAY, now=NOW).to_dict()["phases"]
    }


class TestALatchedPhaseStaysMet:
    def test_a_failing_clause_does_not_un_met_a_latched_phase(
        self, store: LocalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _close(store, monkeypatch, "phase0", ("a", "b"))
        _set_gate(monkeypatch, "phase0", ("a", True), ("b", False))
        _file_reading(store, "phase0")

        row = _rows(store)["phase0"]
        assert row["state"] == "MET"
        # The live measurement is not erased: gate_state is still what the
        # gate read today.
        assert row["gate_state"] == "UNMET"
        assert row["latched_by"] == {
            "key": "gates/phase0/closing.json",
            "trading_day": FRIDAY.isoformat(),
            "commit": COMMIT,
            "clauses_met": 2,
            "clauses_total": 2,
        }
        assert row["regressions"] == ["b"]
        assert "latched by gates/phase0/closing.json (2/2 on 2026-08-28 at c06e4ef" in row["detail"]
        assert "regressed: b" in row["detail"]

    def test_a_latched_phase_with_every_clause_met_has_no_regressions(
        self, store: LocalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _close(store, monkeypatch, "phase0", ("a", "b"))
        row = _rows(store)["phase0"]
        assert row["state"] == "MET"
        assert row["regressions"] == []
        assert "regressed" not in row["detail"]

    def test_a_later_phase_is_not_out_of_order_behind_a_latched_one(
        self, store: LocalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _close(store, monkeypatch, "phase0", ("a", "b"))
        _set_gate(monkeypatch, "phase0", ("a", True), ("b", False))
        _set_gate(monkeypatch, "phase1", ("x", False))
        _file_reading(store, "phase0")
        _file_reading(store, "phase1")

        ladder = build_ladder(store, trading_day=FRIDAY, now=NOW)
        rows = {r["phase"]: r for r in ladder.to_dict()["phases"]}
        assert rows["phase1"]["state"] == "UNMET"
        assert rows["phase1"]["blocked_by"] is None
        assert "phase1" not in ladder.out_of_order
        assert ladder.current_phase == "phase1"
        assert ladder.to_dict()["phases_met"] >= 1

    def test_the_latched_ladder_validates_against_the_schema(
        self, store: LocalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _close(store, monkeypatch, "phase0", ("a", "b"))
        _set_gate(monkeypatch, "phase0", ("a", True), ("b", False))
        ladder_payload(build_ladder(store, trading_day=FRIDAY, now=NOW))


class TestARegressionIsLoudOnTheBoard:
    def test_the_regression_row_is_red_and_names_the_clause(
        self, store: LocalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _close(store, monkeypatch, "phase0", ("arc_runs_ok", "attribution_renders"))
        _set_gate(monkeypatch, "phase0", ("arc_runs_ok", False), ("attribution_renders", False))
        ladder = build_ladder(store, trading_day=FRIDAY, now=NOW)
        rows = {r.id: r for r in _phase_rows(ladder, FRIDAY.isoformat())}

        assert rows["phase:phase0"].state == "MET"
        regression = rows["phase:phase0:regression"]
        assert regression.state in RED_STATES
        assert regression.state == "UNMET"
        assert "arc_runs_ok" in regression.detail
        assert "attribution_renders" in regression.detail
        assert regression.artifact == "gates/phase0/closing.json"

    def test_a_clean_latched_phase_has_a_green_regression_row(
        self, store: LocalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Declared whenever the phase is latched, so the row's absence is
        never what "no regression" looks like."""
        _close(store, monkeypatch, "phase0", ("a",))
        ladder = build_ladder(store, trading_day=FRIDAY, now=NOW)
        rows = {r.id: r for r in _phase_rows(ladder, FRIDAY.isoformat())}
        assert rows["phase:phase0:regression"].state == "MET"

    def test_an_unmeasurable_clause_on_a_latched_phase_is_red(
        self, store: LocalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _close(store, monkeypatch, "phase0", ("a", "b"))
        monkeypatch.setitem(
            GATES,
            "phase0",
            (
                5,
                lambda *_a, **_k: [
                    Clause("a", "req a", True, "met", ()),
                    Clause("b", "req b", False, "could not be read", (), unmeasurable=True),
                ],
            ),
        )
        ladder = build_ladder(store, trading_day=FRIDAY, now=NOW)
        row = {r["phase"]: r for r in ladder.to_dict()["phases"]}["phase0"]
        assert row["state"] == "MET"
        assert "unmeasurable: b" in row["detail"]
        rows = {r.id: r for r in _phase_rows(ladder, FRIDAY.isoformat())}
        assert rows["phase:phase0:regression"].state == "UNMEASURABLE"

    def test_an_unlatched_phase_has_no_regression_row(self, store: LocalStore) -> None:
        ladder = build_ladder(store, trading_day=FRIDAY, now=NOW)
        ids = {r.id for r in _phase_rows(ladder, FRIDAY.isoformat())}
        assert not any(i.endswith(":regression") for i in ids)
        assert {f"phase:{p.id}" for p in PHASES} <= ids


class TestReopeningIsExplicit:
    def test_a_newer_reopen_record_un_latches_the_phase(
        self, store: LocalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _close(store, monkeypatch, "phase0", ("a", "b"))
        _set_gate(monkeypatch, "phase0", ("a", True), ("b", False))
        _reopen(store, "phase0", filed_utc="2026-08-29T09:00:00Z")

        row = _rows(store)["phase0"]
        assert row["state"] == "UNMET"
        assert row["latched_by"] is None
        assert "reopened" in row["detail"]
        assert "Brian" in row["detail"]

    def test_a_reopen_record_older_than_the_closing_record_is_ignored(
        self, store: LocalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _close(store, monkeypatch, "phase0", ("a", "b"))
        _set_gate(monkeypatch, "phase0", ("a", True), ("b", False))
        _reopen(store, "phase0", filed_utc="2026-08-01T09:00:00Z")
        assert _rows(store)["phase0"]["state"] == "MET"

    def test_a_malformed_reopen_record_falls_back_to_live_grading_and_says_why(
        self, store: LocalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _close(store, monkeypatch, "phase0", ("a", "b"))
        _set_gate(monkeypatch, "phase0", ("a", True), ("b", False))
        _reopen(store, "phase0", filed_utc="2026-08-29T09:00:00Z", ruled_by="")
        row = _rows(store)["phase0"]
        assert row["state"] == "UNMET"
        assert "reopen record" in row["detail"]
        assert "ruled_by" in row["detail"]


class TestOnlyAValidClosingRecordLatches:
    def test_a_refused_closing_record_does_not_latch(
        self, store: LocalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _close(store, monkeypatch, "phase0", ("a", "b"), clauses_met=1)
        _set_gate(monkeypatch, "phase0", ("a", True), ("b", False))
        row = _rows(store)["phase0"]
        assert row["state"] == "UNMET"
        assert row["latched_by"] is None
        assert "not latched" in row["detail"]
        assert "1 of 2 clauses met" in row["detail"]

    def test_a_record_for_another_phase_does_not_latch(
        self, store: LocalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        document = _close(store, monkeypatch, "phase0", ("a",))
        document["phase"] = "phase1"
        store.put_bytes(closing_record_key("phase0"), json.dumps(document).encode("utf-8"))
        _set_gate(monkeypatch, "phase0", ("a", False))
        assert _rows(store)["phase0"]["state"] == "UNMET"

    def test_an_unreadable_closing_record_does_not_latch(
        self, store: LocalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store.put_bytes(closing_record_key("phase0"), b"{not json")
        _set_gate(monkeypatch, "phase0", ("a", False))
        row = _rows(store)["phase0"]
        assert row["state"] == "UNMET"
        assert "not latched" in row["detail"]

    def test_no_closing_record_is_todays_behaviour(
        self, store: LocalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_gate(monkeypatch, "phase0", ("a", False))
        _set_gate(monkeypatch, "phase1", ("x", False))
        _file_reading(store, "phase1")
        rows = _rows(store)
        assert rows["phase0"]["state"] == "UNMET"
        assert rows["phase0"]["latched_by"] is None
        assert rows["phase0"]["regressions"] == []
        assert "latched" not in rows["phase0"]["detail"]
        assert rows["phase1"]["state"] == "OUT_OF_ORDER"
        assert rows["phase1"]["blocked_by"] == "phase0"
