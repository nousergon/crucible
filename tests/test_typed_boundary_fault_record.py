"""`fault_record.v1`'s typed boundary (`alpha-engine-config-I10320`/`-I10322`,
M0 contract discipline: a new cross-module artifact gets a versioned JSON
Schema plus a producer/consumer contract test at birth).

Mirrors `tests/test_typed_boundary_declared_universe.py`'s shape exactly:
the schema is GENERATED from `crucible.models.FaultRecordDocument`, never
hand-edited, and this test is the drift guard between the committed
`.json` file and the model that generates it.
"""

from __future__ import annotations

import json
import pathlib

import pytest
from pydantic import ValidationError

from crucible.gate import FAULT_RECORD_BUS_FIELD, FAULT_RECORD_MANIFEST_FIELD, SCRIPTED_FAULTS
from crucible.models import FaultRecordDocument

SCHEMA_PATH = (
    pathlib.Path(__file__).resolve().parents[1] / "crucible" / "schemas" / "fault_record.v1.json"
)


def _valid_document(**overrides: object) -> dict:
    document = {
        "schema_version": "fault_record.v1",
        "fault_id": "data_source_withheld",
        "trading_day": "2026-08-07",
        "run_id": "01M23BQV1C6EPDR8DEA5WS59M4",
        "manifest_key": "runs/data.weekly/2026-08-07/run.json",
        "bus_key": None,
        "recorded_at_utc": "2026-09-09T12:00:00Z",
    }
    document.update(overrides)
    return document


class TestTheCommittedSchemaIsGeneratedFromTheModel:
    def test_the_committed_schema_is_byte_identical_to_the_generated_one(self) -> None:
        generated = (
            json.dumps(FaultRecordDocument.model_json_schema(), indent=2, sort_keys=True) + "\n"
        )
        committed = SCHEMA_PATH.read_text(encoding="utf-8")
        assert committed == generated, (
            f"{SCHEMA_PATH.name} has drifted from FaultRecordDocument. The schema is "
            "GENERATED, never hand-edited: regenerate it in the same commit as the "
            "model change."
        )

    def test_a_real_record_validates_against_the_committed_schema(self) -> None:
        FaultRecordDocument.model_validate(_valid_document())


class TestTheFieldNamesMatchTheReaderTheGateAlreadyShipped:
    """`crucible.gate._clause_fault_injection_against_scheduled_path` landed
    FIRST and declared the contract this producer conforms to
    (`alpha-engine-config-I10320`); this pins the two field names so a future
    edit to either side is caught rather than silently drifting."""

    def test_the_manifest_field_name_matches(self) -> None:
        assert "manifest_key" == FAULT_RECORD_MANIFEST_FIELD

    def test_the_bus_field_name_matches(self) -> None:
        assert "bus_key" == FAULT_RECORD_BUS_FIELD

    def test_a_document_missing_the_manifest_field_is_refused(self) -> None:
        document = _valid_document()
        del document[FAULT_RECORD_MANIFEST_FIELD]
        with pytest.raises(ValidationError, match=FAULT_RECORD_MANIFEST_FIELD):
            FaultRecordDocument.model_validate(document)


class TestRunIdIsTheMatchKeyNeverTradingDayAlone:
    """`alpha-engine-config-I10322`'s whole design constraint, pinned at the
    schema level: `run_id` is required and ULID-shaped."""

    def test_run_id_is_required(self) -> None:
        document = _valid_document()
        del document["run_id"]
        with pytest.raises(ValidationError, match="run_id"):
            FaultRecordDocument.model_validate(document)

    def test_a_malformed_run_id_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="run_id"):
            FaultRecordDocument.model_validate(_valid_document(run_id="not-a-ulid"))


class TestBusKeyIsOptional:
    """The live-sweep half of §10.7 is structurally unreachable for some
    induced days (`alpha-engine-config-I10125`) — a record may legitimately
    exist with no bus row yet."""

    def test_bus_key_defaults_to_null(self) -> None:
        document = _valid_document()
        del document["bus_key"]
        FaultRecordDocument.model_validate(document)

    def test_an_empty_bus_key_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="bus_key"):
            FaultRecordDocument.model_validate(_valid_document(bus_key=""))


class TestUnknownFieldsAreRefused:
    def test_an_unknown_top_level_key_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="bogus_field"):
            FaultRecordDocument.model_validate(_valid_document(bogus_field="nope"))


class TestFaultIdVocabulary:
    """The model itself keeps `fault_id` a pattern-constrained string rather
    than a `Literal` over `crucible.gate.SCRIPTED_FAULTS` — importing gate
    into models would be circular (gate already imports models). Membership
    is enforced at WRITE TIME by `crucible.faults.record_fault`
    (`tests/faults/test_fault_record_producer.py`); this only pins that the
    four scripted fault ids are all schema-legal strings, so the write-time
    check is the only enforcement, not a schema gap masquerading as one."""

    def test_every_scripted_fault_id_is_schema_legal(self) -> None:
        for fault_id in SCRIPTED_FAULTS:
            FaultRecordDocument.model_validate(_valid_document(fault_id=fault_id))
