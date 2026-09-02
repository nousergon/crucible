"""Contract tests for `shadow.v2` (alpha-engine-config-I9778): the two
schemas `crucible/schemas/cross_section.v2.json` and
`cross_section_settled.v2.json`, and the producer/consumer round trip
between `crucible.slots.grading`'s writers and `jsonschema`'s reader.

M0 discipline: a versioned schema plus a producer/consumer contract test at
birth. The producer side is `produce_cross_section`/`write_cross_section`
and `settle_cross_section`/`write_cross_section_settled`; the consumer side
is `crucible.report`'s rank-IC row. This file is the contract between them.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

from crucible.slots.grading import (
    CROSS_SECTION_SCHEMA_VERSION,
    CROSS_SECTION_SETTLED_SCHEMA_VERSION,
    ScoredCrossSection,
    settle_cross_section,
)

SCHEMA_DIR = Path(__file__).parent.parent / "crucible" / "schemas"


def _load(name: str) -> dict:
    return json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def produce_validator() -> Draft202012Validator:
    schema = _load("cross_section.v2.json")
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


@pytest.fixture(scope="module")
def settled_validator() -> Draft202012Validator:
    schema = _load("cross_section_settled.v2.json")
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


class TestProduceTimeSchema:
    def test_a_real_producer_document_validates(self, produce_validator) -> None:
        """The PRODUCER side of the contract: `ScoredCrossSection.to_dict()`
        is not hand-built here — it is what `produce_cross_section` would
        write, exercised directly."""
        cross_section = ScoredCrossSection(
            arm_id="r:momentum_sleeve:ab12cd",
            trading_day="2026-08-28",
            ranks=(("AAA", 3.0, 1), ("BBB", 2.0, 2), ("CCC", 1.0, 3)),
        )
        produce_validator.validate(cross_section.to_dict())

    def test_schema_version_is_closed(self, produce_validator) -> None:
        cross_section = ScoredCrossSection(
            arm_id="r:x:1", trading_day="2026-08-28", ranks=(("AAA", 1.0, 1),) * 1
        )
        doc = cross_section.to_dict()
        doc["schema_version"] = "cross_section.v1"
        with pytest.raises(ValidationError):
            produce_validator.validate(doc)

    def test_schema_is_closed_to_extra_fields(self, produce_validator) -> None:
        cross_section = ScoredCrossSection(
            arm_id="r:x:1", trading_day="2026-08-28", ranks=(("AAA", 1.0, 1),)
        )
        doc = cross_section.to_dict()
        doc["look_ahead"] = False
        with pytest.raises(ValidationError):
            produce_validator.validate(doc)

    @pytest.mark.parametrize("field", ["schema_version", "arm_id", "trading_day", "ranks"])
    def test_every_required_field_is_required(self, produce_validator, field) -> None:
        cross_section = ScoredCrossSection(
            arm_id="r:x:1", trading_day="2026-08-28", ranks=(("AAA", 1.0, 1),)
        )
        doc = cross_section.to_dict()
        del doc[field]
        with pytest.raises(ValidationError):
            produce_validator.validate(doc)

    def test_a_realized_return_field_is_refused_at_produce_time(self, produce_validator) -> None:
        """§10.1's produce/grade separation, enforced by the schema: a
        produce-time document that already knows an outcome is a look-ahead
        written into a production artifact, and the schema makes it
        unrepresentable rather than trusting the caller not to write one."""
        cross_section = ScoredCrossSection(
            arm_id="r:x:1", trading_day="2026-08-28", ranks=(("AAA", 1.0, 1),)
        )
        doc = cross_section.to_dict()
        doc["ranks"][0]["realized_forward_return_ratio"] = 0.02
        with pytest.raises(ValidationError):
            produce_validator.validate(doc)


class TestSettledSchema:
    def _document(self) -> dict:
        cross_section = ScoredCrossSection(
            arm_id="r:momentum_sleeve:ab12cd",
            trading_day="2026-08-28",
            ranks=(("AAA", 3.0, 1), ("BBB", 2.0, 2), ("CCC", 1.0, 3)),
        )
        return settle_cross_section(
            cross_section.to_dict(),
            returns={"AAA": 0.03, "BBB": 0.02},
            horizon_trading_days=21,
            settled_on="2026-09-28",
        )

    def test_a_real_settlement_document_validates(self, settled_validator) -> None:
        """The full round trip: produce -> settle -> validate. `CCC` never
        settled (delisted, say) and carries a null return rather than being
        dropped or zero-filled."""
        document = self._document()
        assert document["n_settled"] == 2
        assert document["ranks"][2]["realized_forward_return_ratio"] is None
        settled_validator.validate(document)

    def test_schema_version_is_closed(self, settled_validator) -> None:
        document = self._document()
        document["schema_version"] = CROSS_SECTION_SCHEMA_VERSION
        with pytest.raises(ValidationError):
            settled_validator.validate(document)

    def test_a_null_realized_return_is_representable(self, settled_validator) -> None:
        """Absent is absent, never zero — the same rule
        `crucible.slots.grading.forward_returns` holds for the label itself."""
        document = self._document()
        assert document["schema_version"] == CROSS_SECTION_SETTLED_SCHEMA_VERSION
        settled_validator.validate(document)

    @pytest.mark.parametrize(
        "field",
        [
            "schema_version",
            "arm_id",
            "trading_day",
            "horizon_trading_days",
            "settled_on",
            "population_size",
            "n_settled",
            "ranks",
        ],
    )
    def test_every_required_field_is_required(self, settled_validator, field) -> None:
        document = self._document()
        del document[field]
        with pytest.raises(ValidationError):
            settled_validator.validate(document)

    def test_a_calendar_unit_horizon_fails_validation(self, settled_validator) -> None:
        """§4.12: a horizon is a count of TRADING days, never a calendar unit."""
        document = self._document()
        document["horizon_trading_days"] = "1 month"
        with pytest.raises(ValidationError):
            settled_validator.validate(document)

    def test_the_document_is_not_mutated_between_calls(self) -> None:
        a = self._document()
        b = self._document()
        a["ranks"][0]["score"] = 999.0
        assert b["ranks"][0]["score"] == pytest.approx(3.0)
        assert b == copy.deepcopy(self._document())
