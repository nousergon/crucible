"""The declared universe's WRITE-side document as a typed boundary
(`alpha-engine-config-I10045` row 10).

Normative source: `alpha-engine-config-I10045`;
`crucible.models.DeclaredUniverseDocument`'s docstring.

Scope: this row types `DeclaredUniverse.record()`'s OUTPUT document (an
artifact this repo owns, `extra="forbid"`) — the boundary the parent issue
cites at `crucible/data/universe.py:232`
(`crucible.data.universe._parse`), which is the READ side for a
THIRD-PARTY document (the fleet's live constituents artifact, written by
the v1 trading path). That read side stays untyped by this PR — the same
carve-out row 12 makes explicit for the CloudTrail partition payload
("model what we read, not what [someone else] writes") — because forbidding
extras on a document another system writes would refuse a field that
system adds tomorrow.
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib

import pytest
from pydantic import ValidationError

from crucible.data.universe import DECLARED_UNIVERSE_SCHEMA_VERSION, DeclaredUniverse
from crucible.models import DeclaredUniverseDocument

SCHEMA_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "crucible"
    / "schemas"
    / "declared_universe.v1.json"
)


class _FakeContext:
    """Duck-typed stand-in for `RunContext`: `record()` only calls
    `.trading_day.isoformat()` and `.record_output(...)`."""

    def __init__(self) -> None:
        self.trading_day = dt.date(2026, 8, 28)
        self.written: list[tuple[str, bytes, str]] = []

    def record_output(self, key: str, payload: bytes, schema_version: str = "v1") -> None:
        self.written.append((key, payload, schema_version))


def _universe(**overrides: object) -> DeclaredUniverse:
    defaults = dict(
        symbols=("AAPL", "MSFT"),
        source_uri="s3://bucket/constituents.json",
        source_sha256="0" * 64,
        origin="argument",
    )
    defaults.update(overrides)
    return DeclaredUniverse(**defaults)


class TestTheCommittedSchemaIsGeneratedFromTheModel:
    def test_the_committed_schema_is_byte_identical_to_the_generated_one(self) -> None:
        generated = (
            json.dumps(DeclaredUniverseDocument.model_json_schema(), indent=2, sort_keys=True)
            + "\n"
        )
        committed = SCHEMA_PATH.read_text(encoding="utf-8")
        assert committed == generated, (
            f"{SCHEMA_PATH.name} has drifted from DeclaredUniverseDocument. The schema "
            "is GENERATED, never hand-edited: regenerate it in the same commit as the "
            "model change."
        )

    def test_a_real_record_validates_against_the_committed_schema(self) -> None:
        ctx = _FakeContext()
        _universe().record(ctx)
        [(_, payload, schema_version)] = ctx.written
        assert schema_version == DECLARED_UNIVERSE_SCHEMA_VERSION
        DeclaredUniverseDocument.model_validate(json.loads(payload))


class TestTheCrossFieldGuardCatchesACountSymbolsMismatch:
    """The measured-incident cross-field rule this row adds: a declared
    universe whose own count disagrees with its own list is not a
    denominator anyone can trust."""

    def test_a_mismatched_count_is_refused(self) -> None:
        document = {
            "schema_version": "declared_universe.v1",
            "trading_day": "2026-08-28",
            "source_uri": "s3://bucket/constituents.json",
            "source_sha256": "0" * 64,
            "origin": "argument",
            "count": 5,
            "symbols": ["AAPL", "MSFT"],
        }
        with pytest.raises(ValidationError, match="does not match"):
            DeclaredUniverseDocument.model_validate(document)

    def test_an_unknown_top_level_key_is_refused(self) -> None:
        document = {
            "schema_version": "declared_universe.v1",
            "trading_day": "2026-08-28",
            "source_uri": "s3://bucket/constituents.json",
            "source_sha256": "0" * 64,
            "origin": "argument",
            "count": 2,
            "symbols": ["AAPL", "MSFT"],
            "bogus_field": "nope",
        }
        with pytest.raises(ValidationError, match="bogus_field"):
            DeclaredUniverseDocument.model_validate(document)


class TestRecordRefusesANonConformantDocument:
    def test_record_raises_before_writing_anything(self) -> None:
        """`DeclaredUniverse.record` validates BEFORE calling
        `ctx.record_output` — a producer that could emit a bad document
        would defeat the whole point of a declared denominator."""
        ctx = _FakeContext()
        bad = _universe(source_sha256="short")
        with pytest.raises(ValidationError):
            bad.record(ctx)
        assert ctx.written == [], "a refused document must not reach record_output"
