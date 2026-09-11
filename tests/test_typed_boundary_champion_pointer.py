"""The champion pointer as a typed document boundary
(`alpha-engine-config-I10045` row 6).

Normative source: `alpha-engine-config-I10045`;
`crucible.models.ChampionPointerDocument`'s docstring. Shaped like row 5
(`ReleaseRecordDocument`): a hand-written schema already existed
(`champion_pointer.v1.json`), validated by a hand-rolled
`jsonschema.Draft202012Validator`; it is now GENERATED from the model.
`crucible.champion.ChampionPointer` (the frozen dataclass with `.to_dict()`)
is unchanged — only its `from_dict` and `write_champion`'s validation step
now route through the model.
"""

from __future__ import annotations

import json
import pathlib

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from crucible.champion import CHAMPION_SCHEMA_VERSION, ChampionPointer, ChampionUnusableError
from crucible.models import ChampionPointerDocument

SCHEMA_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "crucible"
    / "schemas"
    / "champion_pointer.v1.json"
)

DAY = "2026-08-28"


def _payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": CHAMPION_SCHEMA_VERSION,
        "slot": "m",
        "arm_id": "m:champ:0123456789abcdef",
        "as_of": DAY,
        "decided_at": "2026-08-29T02:00:00Z",
        "run_id": "01JG0000000000000000000000",
        "code_sha": "a" * 40,
        "promotion_source": "evidence",
        "manifest_key": f"runs/promote/{DAY}/run.json",
        "evidence": {"status": "decided", "moved": True, "paired_dates": 40},
    }
    payload.update(overrides)
    return payload


class TestTheCommittedSchemaIsGeneratedFromTheModel:
    def test_the_committed_schema_is_byte_identical_to_the_generated_one(self) -> None:
        generated = (
            json.dumps(ChampionPointerDocument.model_json_schema(), indent=2, sort_keys=True) + "\n"
        )
        committed = SCHEMA_PATH.read_text(encoding="utf-8")
        assert committed == generated, (
            f"{SCHEMA_PATH.name} has drifted from ChampionPointerDocument. The schema "
            "is GENERATED, never hand-edited: regenerate it in the same commit as the "
            "model change."
        )

    def test_a_real_pointer_validates_against_the_committed_schema(self) -> None:
        pointer = ChampionPointer(**_payload())
        Draft202012Validator(json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))).validate(
            pointer.to_dict()
        )

    def test_an_operator_revert_pointer_with_open_evidence_still_validates(self) -> None:
        """The exact shape `ChampionEvidence`'s `extra="allow"` exists for:
        an operator revert names none of the evidence-promotion fields."""
        pointer = ChampionPointer(
            **_payload(
                promotion_source="operator_bootstrap",
                evidence={"operator": "brian", "reason": "manual revert"},
            )
        )
        Draft202012Validator(json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))).validate(
            pointer.to_dict()
        )


class TestAMalformedPointerNamesTheField:
    def test_an_unknown_slot_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="slot"):
            ChampionPointerDocument.model_validate(_payload(slot="x"))

    def test_a_bad_run_id_pattern_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="run_id"):
            ChampionPointerDocument.model_validate(_payload(run_id="not-a-ulid"))

    def test_an_unknown_top_level_key_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="bogus_field"):
            ChampionPointerDocument.model_validate(_payload(bogus_field="nope"))

    def test_an_attestation_missing_status_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="status"):
            ChampionPointerDocument.model_validate(_payload(attestation={"kind": "pit_parity"}))

    def test_an_attestation_with_an_unknown_kind_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="kind"):
            ChampionPointerDocument.model_validate(
                _payload(attestation={"kind": "something_else", "status": "PASS"})
            )

    def test_from_dict_raises_champion_unusable_error_naming_the_field(self) -> None:
        with pytest.raises(ChampionUnusableError, match="run_id"):
            ChampionPointer.from_dict(_payload(run_id="not-a-ulid"))

    def test_the_all_zero_placeholder_code_sha_is_refused(self) -> None:
        """`alpha-engine-config-I10506`: same defect, same shape as
        `RunManifestV2._code_sha_is_not_the_placeholder`
        (`alpha-engine-config-I10454`) — the all-zero sha validates
        `code_sha`'s own `pattern` and answers nothing."""
        with pytest.raises(ValidationError, match="placeholder"):
            ChampionPointerDocument.model_validate(_payload(code_sha="0" * 40))

    def test_from_dict_refuses_the_placeholder_code_sha_too(self) -> None:
        with pytest.raises(ChampionUnusableError, match="placeholder"):
            ChampionPointer.from_dict(_payload(code_sha="0" * 40))


class TestCommittedSchemaRefusesThePlaceholderCodeSha:
    """The published `not` clause `_champion_pointer_json_schema_extra`
    mirrors into `champion_pointer.v1.json`, exercised with the plain
    `jsonschema` validator so a consumer with no Python import is proven to
    get the same refusal a `ChampionPointerDocument.model_validate` caller
    gets."""

    def test_the_committed_schema_rejects_the_placeholder_via_plain_jsonschema(self) -> None:
        from jsonschema import ValidationError as JsonSchemaValidationError

        with pytest.raises(JsonSchemaValidationError):
            Draft202012Validator(json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))).validate(
                _payload(code_sha="0" * 40)
            )


class TestEvidenceRoundTripsExactlyNotPaddedWithNulls:
    """`from_dict` must not materialize a `None` for every `ChampionEvidence`
    field the original document never mentioned — that would change
    `pointer.evidence`'s shape and break `ChampionPointer.__eq__`."""

    def test_a_sparse_evidence_dict_round_trips_unpadded(self) -> None:
        payload = _payload(evidence={"operator": "brian", "reason": "manual revert"})
        pointer = ChampionPointer.from_dict(payload)
        assert pointer.evidence == {"operator": "brian", "reason": "manual revert"}
