"""The Metron consumer register and its phase-4 guard — `alpha-engine-config-I10739`.

Every test here was seen failing before the code that makes it pass, and most
of them show the guard REFUSING something: a detector nobody has made fail is
a detector nobody knows works (`AGENTS.md`, "Test discipline").
"""

from __future__ import annotations

import datetime as dt

import pytest
from pydantic import ValidationError

from crucible.gate import (
    PHASE4_DELIVERABLES,
    _clause_metron_reads_have_surviving_producer,
)
from crucible.metron_consumers import (
    METRON_CONSUMER_REGISTER_PATH,
    load_register,
    orphaned_by_phase4,
    paused_rows,
)
from crucible.models import (
    PHASE4_RETIRED_SCHEDULE_OWNERS,
    MetronConsumerRegisterDocument,
    MetronConsumerRow,
)

_CLAUSE = "metron_read_artifacts_have_surviving_producer"


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "object_key": "market_data/example/latest.json",
        "reader": "api/services/example.py:1",
        "producer": "nousergon-data:collectors/example.py",
        "schedule_owner": "data_collection_stack",
        "phase4_disposition": "rehomed",
    }
    row.update(overrides)
    return row


def _document(*rows: dict[str, object]) -> MetronConsumerRegisterDocument:
    return MetronConsumerRegisterDocument.model_validate(
        {
            "schema_version": "metron_consumer_register.v1",
            "reviewed_on": "2026-09-17",
            "metron_commit": "0123456789ab",
            "inventory_doc": "alpha-engine-config/private-docs/example.md",
            "reads": list(rows),
        }
    )


# ---------------------------------------------------------------------------
# the shipped register
# ---------------------------------------------------------------------------


class TestShippedRegister:
    def test_it_parses(self) -> None:
        register = load_register()
        assert register.schema_version == "metron_consumer_register.v1"
        assert register.reads

    def test_it_names_the_inventory_it_is_the_machine_readable_half_of(self) -> None:
        # Without this the register is a table with no analysis behind it, and
        # a later reader has no way back to WHY a row says what it says.
        assert load_register().inventory_doc.endswith(".md")

    def test_every_row_cites_a_metron_reader(self) -> None:
        # The provenance that keeps a declared register from being prose.
        for row in load_register().reads:
            assert ":" in row.reader, row.object_key

    def test_no_row_names_a_bucket(self) -> None:
        # This repository is public at phase-1 exit and forbids infrastructure
        # identifiers (`AGENTS.md`). A key is relative; a bucket is not.
        text = METRON_CONSUMER_REGISTER_PATH.read_text(encoding="utf-8")
        assert "s3://" not in text
        assert "arn:aws" not in text

    def test_every_unresolved_or_retired_row_explains_itself(self) -> None:
        # A bare `unresolved` row tells the next reader nothing about what
        # would resolve it, and a bare retirement is unverifiable.
        for row in load_register().reads:
            if row.phase4_disposition == "unresolved":
                assert row.note.strip(), row.object_key
            if row.phase4_disposition == "retired_by_ruling":
                assert row.ruling.strip(), row.object_key

    def test_the_intraday_reads_are_the_ones_that_survive_untouched(self) -> None:
        # Measured 2026-09-14 and re-swept 2026-09-17: the two intraday files
        # are produced by a host timer outside every stack. If a later edit
        # moves them under a stack this pins the change to a deliberate diff.
        survivors = {
            row.object_key
            for row in load_register().reads
            if row.phase4_disposition == "survives" and row.schedule_owner == "host_timer"
        }
        assert survivors == {
            "market_data/intraday/latest.json",
            "market_data/intraday/technical_ratings.json",
        }


# ---------------------------------------------------------------------------
# the model refuses
# ---------------------------------------------------------------------------


class TestModelRefusals:
    def test_a_retirement_must_name_its_ruling(self) -> None:
        # `architecture.d/146` rule 4 reserves retiring a Metron feature to
        # Brian. This is the check that stops an agent clearing the gate by
        # deciding a product feature is expendable.
        with pytest.raises(ValidationError, match="retired_by_ruling"):
            _document(
                _row(schedule_owner="v1_orchestration", phase4_disposition="retired_by_ruling")
            )

    def test_a_rehomed_row_may_not_still_live_on_a_retired_owner(self) -> None:
        with pytest.raises(ValidationError, match="rehomed"):
            _document(_row(schedule_owner="v1_orchestration", phase4_disposition="rehomed"))

    def test_a_rehomed_row_may_not_claim_no_owner_at_all(self) -> None:
        with pytest.raises(ValidationError, match="rehomed"):
            _document(_row(schedule_owner="none", phase4_disposition="rehomed"))

    def test_a_surviving_row_may_not_live_on_a_retired_owner(self) -> None:
        with pytest.raises(ValidationError, match="survives"):
            _document(_row(schedule_owner="v1_orchestration", phase4_disposition="survives"))

    def test_an_empty_register_is_refused(self) -> None:
        # The vacuous-truth trap, refused at the document rather than left for
        # the clause: a property over an empty set is true, and this guard
        # exists precisely for the case where the set was not empty.
        with pytest.raises(ValidationError, match="empty"):
            _document()

    def test_a_duplicate_object_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="duplicate"):
            _document(_row(), _row())

    def test_an_unknown_disposition_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            _document(_row(phase4_disposition="probably_fine"))

    def test_an_unknown_field_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            _document(_row(looks_ok="yes"))


# ---------------------------------------------------------------------------
# what counts as orphaned
# ---------------------------------------------------------------------------


class TestOrphanDetection:
    def test_a_producer_on_a_retired_schedule_is_orphaned(self) -> None:
        register = _document(
            _row(schedule_owner="v1_orchestration", phase4_disposition="unresolved")
        )
        assert [r.object_key for r in orphaned_by_phase4(register)] == [
            "market_data/example/latest.json"
        ]

    def test_unresolved_is_orphaned_whatever_the_owner(self) -> None:
        # The clause that survives the class: a Metron read added later lands
        # here by DEFAULT, without an edit to the detector.
        register = _document(
            _row(schedule_owner="data_collection_stack", phase4_disposition="unresolved")
        )
        assert orphaned_by_phase4(register)

    def test_a_ruling_is_what_lets_a_retired_schedule_pass(self) -> None:
        register = _document(
            _row(
                schedule_owner="v1_orchestration",
                phase4_disposition="retired_by_ruling",
                ruling="Brian 2026-09-14, R4 (c)",
            )
        )
        assert orphaned_by_phase4(register) == []

    def test_a_paused_row_is_reported_but_is_not_this_phase_s_failure(self) -> None:
        register = _document(
            _row(schedule_owner="paused_outside_v1", phase4_disposition="survives")
        )
        assert orphaned_by_phase4(register) == []
        assert [r.object_key for r in paused_rows(register)] == ["market_data/example/latest.json"]

    def test_the_retired_owner_set_is_exactly_the_v1_orchestration_stack(self) -> None:
        # Phase 4 disables the three v1 pipelines and nothing else. If that
        # ever widens, this constant is the one edit — and this test is the
        # thing that makes widening it deliberate.
        assert PHASE4_RETIRED_SCHEDULE_OWNERS == frozenset({"v1_orchestration"})


# ---------------------------------------------------------------------------
# the clause
# ---------------------------------------------------------------------------


class TestClause:
    def test_it_fails_naming_every_orphan(self, monkeypatch: pytest.MonkeyPatch) -> None:
        register = _document(
            _row(
                object_key="metron/example_rate.json",
                schedule_owner="v1_orchestration",
                phase4_disposition="unresolved",
                note="ruled, not built",
            ),
            _row(object_key="market_data/fine/latest.json"),
        )
        monkeypatch.setattr("crucible.gate.load_metron_consumer_register", lambda: register)
        clause = _clause_metron_reads_have_surviving_producer()
        assert clause.name == _CLAUSE
        assert clause.met is False
        assert "metron/example_rate.json" in clause.detail
        assert "market_data/fine/latest.json" not in clause.detail

    def test_it_passes_when_every_row_is_resolved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        register = _document(_row())
        monkeypatch.setattr("crucible.gate.load_metron_consumer_register", lambda: register)
        clause = _clause_metron_reads_have_surviving_producer()
        assert clause.met is True
        # A green reading still carries the age of what it read: a register
        # swept a year ago is not the same evidence as one swept today, and
        # the two must not render identically.
        assert "2026-09-17" in clause.detail
        assert "0123456789ab" in clause.detail

    def test_a_paused_row_is_named_even_on_a_green_reading(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Principle 7: "no data" is never rendered as the same thing as fine.
        register = _document(
            _row(
                object_key="crypto/example.json",
                schedule_owner="paused_outside_v1",
                phase4_disposition="survives",
                note="disabled since 2026-08-12",
            )
        )
        monkeypatch.setattr("crucible.gate.load_metron_consumer_register", lambda: register)
        clause = _clause_metron_reads_have_surviving_producer()
        assert clause.met is True
        assert "crypto/example.json" in clause.detail
        assert "paused" in clause.detail

    def test_an_unreadable_register_is_unmeasurable_never_met(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom() -> MetronConsumerRegisterDocument:
            raise ValueError("the register is malformed")

        monkeypatch.setattr("crucible.gate.load_metron_consumer_register", boom)
        clause = _clause_metron_reads_have_surviving_producer()
        assert clause.met is False
        assert clause.unmeasurable is True
        assert "malformed" in clause.detail

    def test_the_live_register_reads_through_the_clause(self) -> None:
        # Not an assertion about the CURRENT verdict — that verdict is
        # supposed to change when the trader rehomes its write. What is pinned
        # is that the shipped file produces a real reading rather than an
        # UNMEASURABLE one, which is how a register that stopped parsing would
        # otherwise hide.
        clause = _clause_metron_reads_have_surviving_producer()
        assert clause.unmeasurable is False
        assert str(len(load_register().reads)) in clause.detail


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_phase4_declares_the_deliverable_and_names_this_clause(self) -> None:
        (row,) = [d for d in PHASE4_DELIVERABLES if d.id == "metron_reads_survive_the_decommission"]
        assert row.graded_by == _CLAUSE

    def test_the_clause_is_actually_assembled_into_phase_4(self) -> None:
        # A clause defined and never assembled grades nothing while looking
        # exactly like one that does.
        from crucible import gate

        source = gate._phase4.__code__.co_names
        assert "_clause_metron_reads_have_surviving_producer" in source

    def test_a_row_of_the_register_is_a_typed_row(self) -> None:
        assert all(isinstance(r, MetronConsumerRow) for r in load_register().reads)

    def test_a_date_is_a_date_not_a_string(self) -> None:
        assert isinstance(load_register().reviewed_on, dt.date)
