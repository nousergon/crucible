"""`data_collection_reliability`, the phase-4 clause on the v1 deletion
(`alpha-engine-config-I11304`).

Brian's 2026-09-21 ruling keeps exactly one time gate in the programme: the
irreversible deletion of the v1 pipelines waits on component 1's published
reliability streaks. `nousergon-data-PR1850` publishes them. This clause is
the half that reads them, and each test below is one of the four readings the
issue names, plus the scoping it asks for.

Every document here has the shape of the live 2026-09-24 reading at
`data_collection/gates/data-collection-reliability/2026-09-24/gate.json`.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Any

import pytest

from crucible import gate as gate_module
from crucible.gate import (
    CRUCIBLE_DATA_COLLECTION_STORE_VAR,
    DATA_COLLECTION_RELIABILITY_MAX_AGE_TRADING_DAYS,
    PHASE4_DELIVERABLES,
    _clause_data_collection_reliability,
    _phase4,
)
from crucible.keys import gate_key
from crucible.store import LocalStore

GATE = "data-collection-reliability"
#: A Friday, and the day the phase-4 gate reads for.
FRIDAY = dt.date(2026, 9, 25)
WINDOW = [dt.date(2026, 9, 21), FRIDAY]


def _streak(unit: str, streak: int, target: int) -> dict[str, Any]:
    return {
        "name": f"data.standing.{unit}_reliability_streak",
        "met": streak >= target,
        "unmeasurable": False,
        "phase": GATE,
        "detail": (
            f"{streak} consecutive complete cycle(s) of nousergon-data-collection/"
            f"data-collection-{unit} ending at the latest due fire, against the {target} "
            "the plan's exit names, over 20 cycle(s) examined and 15 verified unit(s)"
        ),
    }


def _document(*, eod: int, morning: int, weekly: int) -> dict[str, Any]:
    clauses = [_streak("eod", eod, 5), _streak("morning", morning, 5), _streak("weekly", weekly, 2)]
    n_met = sum(1 for c in clauses if c["met"])
    return {
        "gate": GATE,
        "met": n_met == len(clauses),
        "clauses_met": n_met,
        "clauses_total": len(clauses),
        "clauses_unmeasurable": 0,
        "clauses": clauses,
        "code_sha": "bd4ef1096405908ee6c350cbd61dcecaad22a32c",
    }


@pytest.fixture
def data_store(tmp_path, monkeypatch) -> LocalStore:
    """Component 1's store: a DIFFERENT store from the harness's, located by
    `CRUCIBLE_DATA_COLLECTION_STORE`, exactly as in production."""
    root = tmp_path / "data-collection"
    root.mkdir()
    monkeypatch.setenv(CRUCIBLE_DATA_COLLECTION_STORE_VAR, str(root))
    return LocalStore(root)


@pytest.fixture
def harness_store(tmp_path) -> LocalStore:
    return LocalStore(tmp_path / "harness")


def _publish(store: LocalStore, day: str, document: dict[str, Any]) -> str:
    key = gate_key(GATE, day)
    store.put_bytes(key, json.dumps(document).encode("utf-8"))
    return key


def _read(harness_store: LocalStore):
    return _clause_data_collection_reliability(harness_store, WINDOW)


class TestTheFourReadings:
    def test_a_met_document_reads_met(self, data_store, harness_store) -> None:
        key = _publish(data_store, "2026-09-25", _document(eod=6, morning=7, weekly=2))
        clause = _read(harness_store)
        assert clause.met and not clause.unmeasurable, clause.detail
        assert clause.evidence == (key,)

    def test_met_false_reads_unmet_naming_all_three_streak_counts(
        self, data_store, harness_store
    ) -> None:
        """The live 2026-09-24 reading: EOD 0 of 5, morning 7 of 5, weekly 0
        of 2. An operator reading UNMET learns WHICH streak is short, and by
        how much, without opening the artifact."""
        _publish(data_store, "2026-09-25", _document(eod=0, morning=7, weekly=0))
        clause = _read(harness_store)
        assert not clause.met and not clause.unmeasurable
        assert "1/3 streak(s) held" in clause.detail
        assert "data.standing.eod_reliability_streak: streak 0/5 (met=False)" in clause.detail
        assert "data.standing.morning_reliability_streak: streak 7/5 (met=True)" in clause.detail
        assert "data.standing.weekly_reliability_streak: streak 0/2 (met=False)" in clause.detail

    def test_an_absent_document_is_unmeasurable_never_met(self, data_store, harness_store) -> None:
        clause = _read(harness_store)
        assert clause.unmeasurable and not clause.met
        assert "no reading" in clause.detail

    def test_a_stale_document_is_unmeasurable_never_met(self, data_store, harness_store) -> None:
        """A MET streak from last week says nothing about the collector now,
        and the deletion it would gate cannot be undone."""
        _publish(data_store, "2026-09-22", _document(eod=6, morning=7, weekly=2))
        clause = _read(harness_store)
        assert clause.unmeasurable and not clause.met
        assert "2026-09-22" in clause.detail and "older than 2026-09-24" in clause.detail


class TestTheReadingIsTheDocumentsOwnVerdict:
    def test_one_trading_day_of_lag_is_tolerated(self, data_store, harness_store) -> None:
        """Component 1 publishes daily, and its publish may land after this
        gate runs."""
        assert DATA_COLLECTION_RELIABILITY_MAX_AGE_TRADING_DAYS == 1
        _publish(data_store, "2026-09-24", _document(eod=6, morning=7, weekly=2))
        assert _read(harness_store).met

    def test_the_newest_reading_is_the_one_graded(self, data_store, harness_store) -> None:
        _publish(data_store, "2026-09-24", _document(eod=6, morning=7, weekly=2))
        _publish(data_store, "2026-09-25", _document(eod=0, morning=7, weekly=2))
        clause = _read(harness_store)
        assert not clause.met
        assert "2026-09-25" in clause.detail

    def test_a_truthy_met_that_is_not_true_is_not_met(self, data_store, harness_store) -> None:
        document = _document(eod=6, morning=7, weekly=2)
        document["met"] = "true"
        _publish(data_store, "2026-09-25", document)
        assert not _read(harness_store).met

    def test_a_streak_detail_in_another_shape_is_unstated_not_guessed(
        self, data_store, harness_store
    ) -> None:
        document = _document(eod=0, morning=7, weekly=0)
        document["clauses"][0]["detail"] = "reshaped by a later data_gate"
        _publish(data_store, "2026-09-25", document)
        clause = _read(harness_store)
        assert "data.standing.eod_reliability_streak: streak unstated (met=False)" in clause.detail

    def test_no_configured_store_is_unmeasurable(self, harness_store, monkeypatch) -> None:
        monkeypatch.delenv(CRUCIBLE_DATA_COLLECTION_STORE_VAR, raising=False)
        clause = _read(harness_store)
        assert clause.unmeasurable and not clause.met
        assert CRUCIBLE_DATA_COLLECTION_STORE_VAR in clause.detail


class TestItIsScopedToTheDecommission:
    def test_a_decommission_deliverable_names_it_and_no_trader_deliverable_does(self) -> None:
        graded = [d for d in PHASE4_DELIVERABLES if d.graded_by == "data_collection_reliability"]
        assert [d.id for d in graded] == ["v1_deletion_gated_on_data_collection_reliability"]
        decommission_start = [d.id for d in PHASE4_DELIVERABLES].index("old_sfs_disabled")
        assert [d.id for d in PHASE4_DELIVERABLES].index(graded[0].id) > decommission_start

    def test_phase_4_assembles_it(self, data_store, harness_store) -> None:
        _publish(data_store, "2026-09-25", _document(eod=0, morning=7, weekly=0))
        readings = {c.name: c for c in _phase4(harness_store, WINDOW, {}, trading_day=FRIDAY)}
        assert "data_collection_reliability" in readings
        assert not readings["data_collection_reliability"].met

    def test_the_cutover_clause_still_reads_through_the_shared_reader(
        self, data_store, harness_store
    ) -> None:
        """Lifting the reader out of `data_cutover_ready` changed nothing about
        that clause: an absent reading is still UNMEASURABLE naming its gate."""
        clause = gate_module._clause_data_cutover_ready(harness_store)
        assert clause.unmeasurable and not clause.met
        assert "data-cutover-ready" in clause.detail
