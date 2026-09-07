"""The release record and its provenance as typed document boundaries
(`alpha-engine-config-I10045` row 5).

Normative source: `alpha-engine-config-I10045`; `crucible.models.
ReleaseRecordDocument` / `ReleaseProvenanceDocument` docstrings.

Unlike rows 2-4, this row already had a hand-written published schema
(`release.v3.json`, `release_provenance.v1.json`) validated by a hand-rolled
`jsonschema.Draft202012Validator` — the row-1 (`RunManifestV2`) shape, not
the row-2 (`ArmRecipeDocument`, no schema today) shape. This file proves the
same two things row 1's `tests/test_manifest_schema.py` proves: the
committed schema is byte-identical to what the model generates, and the
existing dataclasses (`crucible.release.ReleaseRecord`/`ReleaseProvenance`,
UNCHANGED by this PR — see their construction-time validation in
`crucible.release._validate_release_artifact`) keep validating a real
payload the same way they did before this PR, with the same "does not
conform" message text.
"""

from __future__ import annotations

import json
import pathlib

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from crucible.models import ReleaseProvenanceDocument, ReleaseRecordDocument
from crucible.release import (
    ReleaseProvenance,
    ReleaseRecord,
    _validate_release_artifact,
    wheel_filename_for,
)

SCHEMAS = pathlib.Path(__file__).resolve().parents[1] / "crucible" / "schemas"

SHA = "a" * 40


def _record_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "release.v3",
        "sha": SHA,
        "lockfile_sha256": "0" * 64,
        "wheel_sha256": "1" * 64,
        "wheel_filename": wheel_filename_for(SHA),
        "python_requires": ">=3.12,<3.13",
        "extra": {},
    }
    payload.update(overrides)
    return payload


def _provenance_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "release_provenance.v1",
        "sha": SHA,
        "run_id": "1",
        "run_attempt": "1",
        "built_at": "2026-06-01T00:00:00Z",
        "workflow_run_url": "https://example/runs/1",
        "test_summary": "1 passed",
    }
    payload.update(overrides)
    return payload


class TestTheCommittedSchemasAreGeneratedFromTheModels:
    """One source of truth, not two that must agree — same rule as row 1."""

    @pytest.mark.parametrize(
        ("filename", "model"),
        [
            ("release.v3.json", ReleaseRecordDocument),
            ("release_provenance.v1.json", ReleaseProvenanceDocument),
        ],
    )
    def test_the_committed_schema_is_byte_identical_to_the_generated_one(
        self, filename, model
    ) -> None:
        generated = json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n"
        committed = (SCHEMAS / filename).read_text(encoding="utf-8")
        assert committed == generated, (
            f"{filename} has drifted from {model.__name__}. The schema is GENERATED, "
            "never hand-edited: regenerate it in the same commit as the model change."
        )

    def test_a_real_release_record_validates_against_the_committed_schema(self) -> None:
        record = ReleaseRecord(
            schema_version="release.v3",
            sha=SHA,
            lockfile_sha256="0" * 64,
            wheel_sha256="1" * 64,
            wheel_filename=wheel_filename_for(SHA),
        )
        Draft202012Validator(
            json.loads((SCHEMAS / "release.v3.json").read_text(encoding="utf-8"))
        ).validate(json.loads(record.to_json()))

    def test_a_real_provenance_record_validates_against_the_committed_schema(self) -> None:
        provenance = ReleaseProvenance(
            schema_version="release_provenance.v1",
            sha=SHA,
            run_id="1",
            run_attempt="1",
            built_at="2026-06-01T00:00:00Z",
            workflow_run_url="https://example/runs/1",
            test_summary="1 passed",
        )
        Draft202012Validator(
            json.loads((SCHEMAS / "release_provenance.v1.json").read_text(encoding="utf-8"))
        ).validate(json.loads(provenance.to_json()))


class TestAMalformedDocumentNamesTheFieldAtTheModelLevel:
    def test_a_bad_sha_pattern_is_refused_by_name(self) -> None:
        with pytest.raises(ValidationError, match="sha"):
            ReleaseRecordDocument.model_validate(_record_payload(sha="not-a-sha"))

    def test_an_unknown_top_level_key_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="bogus_field"):
            ReleaseRecordDocument.model_validate(_record_payload(bogus_field="nope"))

    def test_a_wrong_schema_version_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="schema_version"):
            ReleaseRecordDocument.model_validate(_record_payload(schema_version="release.v2"))

    def test_provenance_missing_run_id_names_it(self) -> None:
        payload = _provenance_payload()
        del payload["run_id"]
        with pytest.raises(ValidationError, match="run_id"):
            ReleaseProvenanceDocument.model_validate(payload)


class TestTheExistingDataclassesStillValidateTheSameWay:
    """`ReleaseRecord`/`ReleaseProvenance` are unchanged dataclasses;
    `_validate_release_artifact` now routes through the models above instead
    of a hand-rolled `Draft202012Validator`, but its public signature and
    message text ("does not conform") are unchanged — both are asserted
    directly, matching `tests/test_release.py`'s existing coverage."""

    def test_a_good_payload_does_not_raise(self) -> None:
        _validate_release_artifact("release.v3.json", _record_payload())
        _validate_release_artifact("release_provenance.v1.json", _provenance_payload())

    def test_a_bad_payload_names_does_not_conform(self) -> None:
        with pytest.raises(ValueError, match="does not conform"):
            _validate_release_artifact("release.v3.json", _record_payload(sha="bad"))

    def test_an_unknown_schema_filename_raises_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError, match="release.v99.json"):
            _validate_release_artifact("release.v99.json", {})
