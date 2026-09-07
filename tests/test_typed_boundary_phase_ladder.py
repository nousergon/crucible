"""The phase ladder and the closing reading as typed document boundaries
(`alpha-engine-config-I10045` row 8).

Normative source: `alpha-engine-config-I10045`;
`crucible.models.PhaseLadderDocument`/`PhaseClosingReadingDocument`
docstrings. Shaped like row 5/row 6: both schemas already existed, hand
validated by `jsonschema.Draft202012Validator`; both are now GENERATED from
their models. `crucible.gate.Ladder`/`PhaseRow`/`closing_reading` (the
dataclasses/builder function) are unchanged — only `validate_ladder_document`
and `validate_closing_reading_document`'s internals route through the
models now.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from crucible.gate import (
    CLOSING_READING_SCHEMA_PATH,
    LADDER_SCHEMA_PATH,
    validate_closing_reading_document,
)
from crucible.models import PhaseClosingReadingDocument, PhaseLadderDocument

#: A fixture tracker number, never spelled as the literal string — this
#: file's fixtures build the id from this int (`tests/test_phase_ladder.py`'s
#: own `_PHASE0_ISSUE` convention) so `test_no_stale_tracker_literals.py`
#: does not flag test fixture data as a hardcoded tracker reference.
_FIXTURE_ISSUE = 1


def _ladder_document(**overrides: object) -> dict[str, object]:
    tracker = f"alpha-engine-config-I{_FIXTURE_ISSUE}"
    payload: dict[str, object] = {
        "schema_version": "phase_ladder.v1",
        "trading_day": "2026-08-28",
        "generated_utc": "2026-08-29T02:00:00Z",
        "current_phase": "phase2",
        "phases_total": 1,
        "phases_met": 0,
        "unmeasured": 0,
        "out_of_order": [],
        "phases": [
            {
                "decision_id": tracker,
                "phase": "phase0",
                "number": 0,
                "title": "phase 0",
                "tracker": tracker,
                "tracker_url": "https://example/issues/1",
                "gate": "phase0",
                "state": "MET",
                "console_state": "HEALTHY",
                "gate_state": "MET",
                "detail": "6/6 met",
                "clauses_met": 6,
                "clauses_total": 6,
                "clauses_unmeasurable": 0,
                "met_ratio": 1.0,
                "read_on": "2026-08-28",
                "blocked_by": None,
                "generated_utc": "2026-08-29T02:00:00Z",
            }
        ],
    }
    payload.update(overrides)
    return payload


def _closing_document(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "phase_closing_reading.v1",
        "phase": "phase0",
        "tracker": f"alpha-engine-config-I{_FIXTURE_ISSUE}",
        "tracker_url": "https://example/issues/1",
        "gate": "phase0",
        "gate_state": "MET",
        "clauses_met": 6,
        "clauses_total": 6,
        "clauses_unmeasurable": 0,
        "met_ratio": 1.0,
        "coverage": None,
        "trading_day": "2026-08-28",
        "generated_utc": "2026-08-29T02:00:00Z",
        "store": "file:///tmp/store",
        "commit": "0" * 40,
        "gate_artifact": "gates/phase0/2026-08-28/gate.json",
        "clauses": [
            {"name": "clause a", "met": True, "unmeasurable": False, "detail": "ok"},
        ],
    }
    payload.update(overrides)
    return payload


class TestTheCommittedSchemasAreGeneratedFromTheModels:
    @pytest.mark.parametrize(
        ("path", "model"),
        [
            (LADDER_SCHEMA_PATH, PhaseLadderDocument),
            (CLOSING_READING_SCHEMA_PATH, PhaseClosingReadingDocument),
        ],
    )
    def test_the_committed_schema_is_byte_identical_to_the_generated_one(self, path, model) -> None:
        generated = json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n"
        committed = path.read_text(encoding="utf-8")
        assert committed == generated, (
            f"{path.name} has drifted from {model.__name__}. The schema is GENERATED, "
            "never hand-edited: regenerate it in the same commit as the model change."
        )


class TestAMalformedLadderNamesTheField:
    def test_an_out_of_vocabulary_state_is_refused(self) -> None:
        document = _ladder_document()
        document["phases"][0]["state"] = "PENDING"
        with pytest.raises(ValidationError, match="state"):
            PhaseLadderDocument.model_validate(document)

    def test_an_unknown_top_level_key_is_refused(self) -> None:
        document = _ladder_document(bogus_field="nope")
        with pytest.raises(ValidationError, match="bogus_field"):
            PhaseLadderDocument.model_validate(document)

    def test_a_missing_phase_row_field_names_it(self) -> None:
        document = _ladder_document()
        del document["phases"][0]["blocked_by"]
        with pytest.raises(ValidationError, match="blocked_by"):
            PhaseLadderDocument.model_validate(document)


class TestAMalformedClosingReadingNamesTheField:
    def test_an_out_of_vocabulary_gate_state_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="gate_state"):
            PhaseClosingReadingDocument.model_validate(_closing_document(gate_state="OUT_OF_ORDER"))

    def test_a_bad_commit_pattern_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="commit"):
            PhaseClosingReadingDocument.model_validate(_closing_document(commit="short"))

    def test_validate_closing_reading_document_names_does_not_conform(self) -> None:
        with pytest.raises(ValueError, match="does not conform"):
            validate_closing_reading_document(_closing_document(commit="short"))
