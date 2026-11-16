"""Unit tests for the arena-cycle persistence boundary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from nousergon_lib.arena import ArenaCycle

import crucible.arena_io as arena_io
from crucible.arena_io import ArenaCycleValidationError
from crucible.documents import UnreadableDocumentError
from crucible.models import ArenaCycleDocument
from crucible.store import LocalStore


def _minimal_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "slot": "u",
        "slot_kind": "selection",
        "benchmark": "population",
        "as_of": "2026-06-01",
        "scored_arms": ["u:example:abc123"],
        "active_arms": ["u:example:abc123"],
        "ladders": [
            {
                "arm_id": "u:example:abc123",
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
            "champion": "u:example:abc123",
            "moved": True,
            "status": "decided",
            "reason": "sole eligible arm",
            "comparisons": [],
            "ineligible": {},
        },
        "retirements": [
            {
                "arm_id": "u:example:abc123",
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


class _Cycle:
    slot = "u"
    as_of = "2026-06-01"

    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def to_dict(self) -> dict[str, object]:
        return self._payload


class _SchemaFile:
    def __init__(self, text: str) -> None:
        self._text = text

    def joinpath(self, _name: str) -> _SchemaFile:
        return self

    def read_text(self, *, encoding: str) -> str:
        assert encoding == "utf-8"
        return self._text


def test_validation_error_is_a_value_error() -> None:
    error = ArenaCycleValidationError("invalid arena cycle")
    assert str(error) == "invalid arena cycle"
    assert isinstance(error, ValueError)


def test_arena_cycle_key_accepts_a_session_and_rejects_invalid_inputs() -> None:
    assert arena_io.arena_cycle_key("u", "2026-06-01") == ("arena/u/2026-06-01/arena_cycle.json")
    with pytest.raises(ValueError, match="slot must be non-empty"):
        arena_io.arena_cycle_key("", "2026-06-01")
    with pytest.raises(ValueError, match="not an NYSE trading day"):
        arena_io.arena_cycle_key("u", "2026-05-31")


def test_load_arena_cycle_schema_reads_the_library_schema() -> None:
    schema = arena_io.load_arena_cycle_schema()
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["title"] == "Arena cycle — one champion/challenger evaluation of one slot"
    assert schema["properties"]["schema_version"]["const"] == 1


def test_load_arena_cycle_schema_is_cached_and_rejects_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def invalid_schema_file(_package: str) -> _SchemaFile:
        return _SchemaFile("not json")

    arena_io.load_arena_cycle_schema.cache_clear()
    monkeypatch.setattr(arena_io.resources, "files", invalid_schema_file)
    with pytest.raises(json.JSONDecodeError):
        arena_io.load_arena_cycle_schema()
    arena_io.load_arena_cycle_schema.cache_clear()


def test_validate_arena_cycle_accepts_a_minimal_document() -> None:
    arena_io.validate_arena_cycle(_minimal_payload())


def test_validate_arena_cycle_reports_every_invalid_field() -> None:
    payload = _minimal_payload(slot="", decision={})
    with pytest.raises(ArenaCycleValidationError) as caught:
        arena_io.validate_arena_cycle(payload)
    message = str(caught.value)
    # Mutation guard: returning after the first schema error loses one of these paths.
    assert "decision" in message
    assert "slot" in message


def test_write_arena_cycle_serializes_validated_json(tmp_path: Path) -> None:
    store = LocalStore(str(tmp_path))
    key = arena_io.write_arena_cycle(store, cast(ArenaCycle, _Cycle(_minimal_payload())))
    assert key == "arena/u/2026-06-01/arena_cycle.json"
    written = cast(dict[str, object], json.loads(store.get_bytes(key)))
    decision = cast(dict[str, object], written["decision"])
    assert decision["champion"] == "u:example:abc123"


def test_write_arena_cycle_refuses_invalid_data_before_writing(
    tmp_path: Path,
) -> None:
    store = LocalStore(str(tmp_path))
    with pytest.raises(ArenaCycleValidationError, match="decision"):
        arena_io.write_arena_cycle(store, cast(ArenaCycle, _Cycle(_minimal_payload(decision={}))))
    assert list(tmp_path.rglob("*")) == []


def test_read_arena_cycle_returns_a_typed_document_with_extra_fields(
    tmp_path: Path,
) -> None:
    store = LocalStore(str(tmp_path))
    payload = _minimal_payload(extension={"enabled": True})
    key = "arena/u/2026-06-01/arena_cycle.json"
    store.put_bytes(key, json.dumps(payload).encode("utf-8"))
    document = arena_io.read_arena_cycle(store, "u", "2026-06-01")
    assert isinstance(document, ArenaCycleDocument)
    assert document.slot == "u"
    assert document.model_extra == {"extension": {"enabled": True}}


def test_read_arena_cycle_rejects_missing_and_malformed_documents(
    tmp_path: Path,
) -> None:
    store = LocalStore(str(tmp_path))
    with pytest.raises(KeyError, match="arena/u/2026-06-01/arena_cycle.json"):
        arena_io.read_arena_cycle(store, "u", "2026-06-01")
    store.put_bytes("arena/u/2026-06-01/arena_cycle.json", b"not json")
    with pytest.raises(UnreadableDocumentError, match="is present but is not readable JSON"):
        arena_io.read_arena_cycle(store, "u", "2026-06-01")
