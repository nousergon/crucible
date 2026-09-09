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

from crucible.gate import (
    FAULT_OUTCOME_INDUCED,
    FAULT_RECORD_BUS_FIELD,
    FAULT_RECORD_MANIFEST_FIELD,
    FAULT_RECORD_OUTCOME_FIELD,
    SCRIPTED_FAULTS,
)
from crucible.models import FAULT_OUTCOME_VALUES, FaultRecordDocument

SCHEMA_PATH = (
    pathlib.Path(__file__).resolve().parents[1] / "crucible" / "schemas" / "fault_record.v1.json"
)


#: One conforming record per outcome kind. Written out per kind rather than
#: derived from one template by deletion: the whole point of
#: `alpha-engine-config-I10327` is that the three shapes are DIFFERENT, and a
#: fixture that built them by removing fields from `induced` would encode
#: exactly the one-shape assumption the change removes.
_BY_OUTCOME: dict[str, dict] = {
    "induced": {
        "schema_version": "fault_record.v1",
        "fault_id": "data_source_withheld",
        "outcome": "induced",
        "trading_day": "2026-08-07",
        "run_id": "01M23BQV1C6EPDR8DEA5WS59M4",
        "manifest_key": "runs/data.weekly/2026-08-07/run.json",
        "bus_key": "alerts/2026-08-07/failure.data.weekly.json",
        "attempt": None,
        "closed_paths": None,
        "recorded_at_utc": "2026-09-09T12:00:00Z",
    },
    "absorbed": {
        "schema_version": "fault_record.v1",
        "fault_id": "spot_terminated_mid_job",
        "outcome": "absorbed",
        "trading_day": "2026-08-07",
        "run_id": "01M23BQV1C6EPDR8DEA5WS59M4",
        "manifest_key": "runs/data.weekly/2026-08-07/run.json",
        "bus_key": None,
        "attempt": {"n": 2, "reason": "spot_interruption"},
        "closed_paths": None,
        "recorded_at_utc": "2026-09-09T12:00:00Z",
    },
    "unreachable": {
        "schema_version": "fault_record.v1",
        "fault_id": "stale_release_pointer",
        "outcome": "unreachable",
        "trading_day": "2026-08-07",
        "run_id": None,
        "manifest_key": None,
        "bus_key": None,
        "attempt": None,
        "closed_paths": [
            {
                "path": "the pointer comes to name an unpublished sha",
                "mechanism": "crucible.release.pin",
                "probe": "pin_refuses_an_unpublished_sha",
                "expected": "StaleReleasePointerError, raised before any write",
                "observed": "StaleReleasePointerError: was never published",
                "checked_at_utc": "2026-09-09T12:00:00Z",
            }
        ],
        "recorded_at_utc": "2026-09-09T12:00:00Z",
    },
}


def _valid_document(outcome: str = "induced", **overrides: object) -> dict:
    document = dict(_BY_OUTCOME[outcome])
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

    def test_an_induced_document_with_a_null_manifest_field_is_refused(self) -> None:
        with pytest.raises(ValidationError, match=f"requires `{FAULT_RECORD_MANIFEST_FIELD}`"):
            FaultRecordDocument.model_validate(
                _valid_document(**{FAULT_RECORD_MANIFEST_FIELD: None})
            )


class TestRunIdIsTheMatchKeyNeverTradingDayAlone:
    """`alpha-engine-config-I10322`'s whole design constraint, pinned at the
    schema level: `run_id` is ULID-shaped, and required for exactly the two
    outcomes that describe a run."""

    def test_run_id_is_required_for_induced(self) -> None:
        with pytest.raises(ValidationError, match="run_id"):
            FaultRecordDocument.model_validate(_valid_document(run_id=None))

    def test_run_id_is_required_for_absorbed(self) -> None:
        with pytest.raises(ValidationError, match="run_id"):
            FaultRecordDocument.model_validate(_valid_document("absorbed", run_id=None))

    def test_a_malformed_run_id_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="run_id"):
            FaultRecordDocument.model_validate(_valid_document(run_id="not-a-ulid"))

    def test_an_unreachable_record_carrying_a_run_id_is_refused(self) -> None:
        """`alpha-engine-config-I10327`: an `unreachable` record must be
        STRUCTURALLY incapable of excusing a manifest, and the field the
        exclusion is keyed on is the one it must not carry."""
        with pytest.raises(ValidationError, match="must not carry `run_id`"):
            FaultRecordDocument.model_validate(
                _valid_document("unreachable", run_id="01M23BQV1C6EPDR8DEA5WS59M4")
            )

    def test_a_field_omitted_entirely_is_refused_not_defaulted(self) -> None:
        """Every evidence field is required AND nullable, never optional with a
        default: an omitted field is indistinguishable from a forgotten one, so
        a record must DECLARE the evidence it does not carry."""
        document = _valid_document()
        del document["closed_paths"]
        with pytest.raises(ValidationError, match="closed_paths"):
            FaultRecordDocument.model_validate(document)


class TestBusKeyIsRequiredForInducedAndForbiddenForAbsorbed:
    """`alpha-engine-config-I10327`. `bus_key` was nullable for any record, and
    `-I10317`'s agent proposed satisfying §10.7 by naming an existing bus row —
    the rubber stamp the `run_id` refusal exists to prevent, arriving through
    the other field. Never optional, and never borrowable."""

    def test_an_induced_record_without_a_bus_key_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="requires `bus_key`"):
            FaultRecordDocument.model_validate(_valid_document(bus_key=None))

    def test_an_absorbed_record_carrying_a_bus_key_is_refused(self) -> None:
        """The refusal `-I10327` names explicitly: a page here would mean the
        retry did not work, so the row's ABSENCE is part of the claim."""
        with pytest.raises(ValidationError, match="must not carry `bus_key`"):
            FaultRecordDocument.model_validate(
                _valid_document("absorbed", bus_key="alerts/2026-08-07/x.json")
            )

    def test_an_unreachable_record_carrying_a_bus_key_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="must not carry `bus_key`"):
            FaultRecordDocument.model_validate(
                _valid_document("unreachable", bus_key="alerts/2026-08-07/x.json")
            )

    def test_an_empty_bus_key_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="bus_key"):
            FaultRecordDocument.model_validate(_valid_document(bus_key=""))


class TestTheOutcomeMatrixIsEnforcedInEveryCell:
    """The table in `FaultRecordDocument`'s docstring, walked rather than
    sampled: for each outcome, every field it requires is refused when null and
    every field it forbids is refused when present. A matrix asserted in the
    three cells somebody remembered is a matrix with cells nobody checks."""

    def test_every_outcome_has_a_conforming_fixture(self) -> None:
        assert set(_BY_OUTCOME) == set(FAULT_OUTCOME_VALUES)
        for outcome in FAULT_OUTCOME_VALUES:
            assert FaultRecordDocument.model_validate(_valid_document(outcome)).outcome == outcome

    def test_every_required_field_is_refused_when_null(self) -> None:
        for outcome in FAULT_OUTCOME_VALUES:
            required = FaultRecordDocument._REQUIRED_BY_OUTCOME[outcome]
            assert required, f"{outcome} requires nothing, which cannot be evidence"
            for field in required:
                with pytest.raises(ValidationError, match=f"requires `{field}`"):
                    FaultRecordDocument.model_validate(_valid_document(outcome, **{field: None}))

    def test_every_forbidden_field_is_refused_when_present(self) -> None:
        populated = {
            "run_id": "01M23BQV1C6EPDR8DEA5WS59M4",
            "manifest_key": "runs/data.weekly/2026-08-07/run.json",
            "bus_key": "alerts/2026-08-07/failure.data.weekly.json",
            "attempt": {"n": 2, "reason": "spot_interruption"},
            "closed_paths": _BY_OUTCOME["unreachable"]["closed_paths"],
        }
        for outcome in FAULT_OUTCOME_VALUES:
            required = set(FaultRecordDocument._REQUIRED_BY_OUTCOME[outcome])
            for field in FaultRecordDocument._EVIDENCE_FIELDS:
                if field in required:
                    continue
                with pytest.raises(ValidationError, match=f"must not carry `{field}`"):
                    FaultRecordDocument.model_validate(
                        _valid_document(outcome, **{field: populated[field]})
                    )

    def test_an_unknown_outcome_is_refused(self) -> None:
        document = _valid_document()
        document["outcome"] = "mostly_fine"
        with pytest.raises(ValidationError, match="outcome"):
            FaultRecordDocument.model_validate(document)

    def test_the_outcome_field_name_matches_the_reader(self) -> None:
        assert "outcome" == FAULT_RECORD_OUTCOME_FIELD
        assert FAULT_OUTCOME_INDUCED in FAULT_OUTCOME_VALUES


class TestAnAbsorbedRecordsAttemptIsTheRetryNotTheFirstTry:
    """An `ok` manifest's FIRST attempt says nothing about a fault being
    handled — every clean run has one."""

    def test_the_initial_attempt_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="must be the RETRY"):
            FaultRecordDocument.model_validate(
                _valid_document("absorbed", attempt={"n": 1, "reason": "initial"})
            )

    def test_an_attempt_numbered_one_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="at least 2"):
            FaultRecordDocument.model_validate(
                _valid_document("absorbed", attempt={"n": 1, "reason": "spot_interruption"})
            )

    def test_a_reason_outside_the_declared_transient_class_is_refused(self) -> None:
        """`AttemptRow.reason` is the manifest schema's own enum, so a retry
        reason the runner could never have recorded cannot be recorded here."""
        with pytest.raises(ValidationError, match="reason"):
            FaultRecordDocument.model_validate(
                _valid_document("absorbed", attempt={"n": 2, "reason": "felt_like_it"})
            )


class TestAnUnreachableRecordsEvidenceCannotBeEmpty:
    def test_an_empty_closed_paths_list_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="EMPTY `closed_paths`"):
            FaultRecordDocument.model_validate(_valid_document("unreachable", closed_paths=[]))

    def test_a_closed_path_with_no_observation_is_refused(self) -> None:
        """`expected` and `observed` are separate fields on purpose: a probe
        whose observation is missing is not evidence of anything."""
        row = dict(_BY_OUTCOME["unreachable"]["closed_paths"][0])
        row["observed"] = ""
        with pytest.raises(ValidationError, match="observed"):
            FaultRecordDocument.model_validate(_valid_document("unreachable", closed_paths=[row]))

    def test_a_closed_path_missing_a_field_is_refused(self) -> None:
        row = dict(_BY_OUTCOME["unreachable"]["closed_paths"][0])
        del row["probe"]
        with pytest.raises(ValidationError, match="probe"):
            FaultRecordDocument.model_validate(_valid_document("unreachable", closed_paths=[row]))


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
