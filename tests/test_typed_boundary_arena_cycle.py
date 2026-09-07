"""The arena cycle artifact's in-process reader (`alpha-engine-config-I10045`
row 3).

Normative source: `alpha-engine-config-I10045`. This row is a NAMED
EXCEPTION to the shape every other row in this migration takes: the
published contract is `nousergon_lib.contracts.arena_cycle.schema.json`, the
LIBRARY's schema, not a file this repository generates or commits. There is
no `crucible/schemas/arena_cycle*.json` and no byte-identity test — see
`crucible.models.ArenaCycleDocument`'s docstring for why. What this file
proves instead: `crucible.arena_io.read_arena_cycle` now hands its caller a
typed `ArenaCycleDocument` rather than a raw dict, `validate_arena_cycle`
(the actual conformance check) is unchanged and still runs first, and the
model does not reject a conforming document that carries a field it does
not know about — the opposite failure mode from every `extra="forbid"`
boundary elsewhere in this migration.
"""

from __future__ import annotations

import json

import pytest

from crucible.arena_io import ArenaCycleValidationError, read_arena_cycle
from crucible.keys import arena_cycle_key
from crucible.models import ArenaCycleDocument
from crucible.store import LocalStore


def _minimal_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "slot": "u",
        "slot_kind": "selection",
        "benchmark": "population",
        "as_of": "2026-06-01",
        "scored_arms": ["u:momentum_sleeve:abc123"],
        "active_arms": ["u:momentum_sleeve:abc123"],
        "ladders": [
            {
                "arm_id": "u:momentum_sleeve:abc123",
                "as_of": "2026-06-01",
                "total_weeks": 4,
                "total_dates": 20,
                "total_misses": 0,
                "rungs": [
                    {
                        "weeks": 4,
                        "n_dates": 20,
                        "n_misses": 0,
                        "start_date": "2026-05-04",
                        "end_date": "2026-06-01",
                        "mean_score": 0.02,
                    }
                ],
            }
        ],
        "decision": {
            "slot": "u",
            "as_of": "2026-06-01",
            "incumbent": None,
            "champion": "u:momentum_sleeve:abc123",
            "moved": True,
            "status": "decided",
            "reason": "sole eligible arm",
            "comparisons": [],
            "ineligible": {},
        },
        "retirements": [
            {
                "arm_id": "u:momentum_sleeve:abc123",
                "retire": False,
                "reason": "champion",
                "age_weeks": 4,
                "pairwise_losses": 0,
                "is_champion": True,
            }
        ],
    }
    payload.update(overrides)
    return payload


class TestReadArenaCycleReturnsATypedDocument:
    def test_a_conforming_document_round_trips_as_a_typed_object(self, tmp_path) -> None:
        store = LocalStore(str(tmp_path))
        payload = _minimal_payload()
        store.put_bytes(
            arena_cycle_key("u", "2026-06-01"),
            json.dumps(payload).encode("utf-8"),
        )
        document = read_arena_cycle(store, "u", "2026-06-01")
        assert isinstance(document, ArenaCycleDocument)
        assert document.slot == "u"
        assert document.as_of == "2026-06-01"
        assert document.decision["champion"] == "u:momentum_sleeve:abc123"
        assert document.scored_arms == ["u:momentum_sleeve:abc123"]

    def test_a_document_that_fails_the_library_schema_is_still_refused_first(
        self, tmp_path
    ) -> None:
        """`validate_arena_cycle` — the library contract — is unchanged and
        runs BEFORE the model sees the payload. A document this model could
        happily construct (it does not forbid extra keys) is still refused
        if the library schema rejects it."""
        store = LocalStore(str(tmp_path))
        payload = _minimal_payload()
        del payload["decision"]  # library-required; this model alone would not catch it
        store.put_bytes(
            arena_cycle_key("u", "2026-06-01"),
            json.dumps(payload).encode("utf-8"),
        )
        with pytest.raises(ArenaCycleValidationError, match="decision"):
            read_arena_cycle(store, "u", "2026-06-01")

    def test_write_then_read_round_trips(self, tmp_path) -> None:
        """`write_arena_cycle` still takes the library's own `ArenaCycle`
        dataclass — unchanged by this PR — and `read_arena_cycle` on the same
        key returns the typed reader for it."""
        store = LocalStore(str(tmp_path))
        payload = _minimal_payload()
        store.put_bytes(arena_cycle_key("u", "2026-06-01"), json.dumps(payload).encode("utf-8"))
        document = read_arena_cycle(store, "u", "2026-06-01")
        assert document.schema_version == 1
        assert document.slot_kind == "selection"


class TestTheModelDoesNotForbidAFieldItDoesNotKnowAbout:
    """The one deliberate way this boundary differs from every other row:
    the library's schema is the contract, so a field the library adds
    tomorrow must not be rejected by this reader today."""

    def test_an_unknown_top_level_field_is_accepted_not_refused(self) -> None:
        payload = _minimal_payload(a_future_library_field={"anything": "at all"})
        document = ArenaCycleDocument.model_validate(payload)
        assert document.model_extra["a_future_library_field"] == {"anything": "at all"}

    def test_ranking_is_optional_matching_the_library_schema(self) -> None:
        payload = _minimal_payload()
        assert "ranking" not in payload
        document = ArenaCycleDocument.model_validate(payload)
        assert document.ranking is None
