"""`board.yaml`'s row body as a typed document boundary
(`alpha-engine-config-I10045` row 11).

Normative source: `alpha-engine-config-I10045`;
`crucible.models.BoardDeclarationRow`'s docstring. No published schema, like
row 4 (the LLM call-site registry): `board.yaml` is a package-internal
config file with no cross-repo reader, so there is nothing to generate a
schema from and no byte-identity test.

`crucible.board._declaration` used to build `Declaration` by hand off a raw
dict, with a `known`/`missing` set-difference check for shape and
`crucible.board.Declaration.__post_init__` (unchanged by this PR) for every
cross-field rule. This file proves the shape check now routes through
`crucible.models.BoardDeclarationRow` while preserving the exact "unknown
field"/"missing required field" message text
`tests/test_board.py::test_an_unknown_yaml_field_is_refused` already
depends on.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from crucible.board import _declaration
from crucible.models import BoardDeclarationRow


def _body(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "statement": "s",
        "surface": "crucible/board",
        "reader": "attribution_table_is_complete",
        "artifact": "report/{trading_day}/attribution.json",
        "means_when_red": "r",
    }
    payload.update(overrides)
    return payload


class TestBoardDeclarationRowShape:
    def test_a_well_formed_body_validates(self) -> None:
        row = BoardDeclarationRow.model_validate(_body())
        assert row.statement == "s"
        assert row.planned_because == ""

    def test_an_unknown_field_is_refused_at_the_model_level(self) -> None:
        with pytest.raises(ValidationError, match="bogus_field"):
            BoardDeclarationRow.model_validate(_body(bogus_field="nope"))

    def test_a_missing_required_field_is_refused(self) -> None:
        body = _body()
        del body["artifact"]
        with pytest.raises(ValidationError, match="artifact"):
            BoardDeclarationRow.model_validate(body)


class TestDeclarationPreservesTheExactMessages:
    """`_declaration` translates pydantic's error kinds back into the exact
    "unknown field"/"missing required field" text
    `tests/test_board.py::test_an_unknown_yaml_field_is_refused` matches on,
    row-id prefixed."""

    def test_an_unknown_field_names_it_with_the_row_id(self) -> None:
        with pytest.raises(ValueError, match=r"row-x: unknown field\(s\) \['bogus_field'\]"):
            _declaration("row-x", "objective", _body(bogus_field="nope"))

    def test_a_missing_field_names_it_with_the_row_id(self) -> None:
        body = _body()
        del body["means_when_red"]
        with pytest.raises(
            ValueError, match=r"row-y: missing required field\(s\) \['means_when_red'\]"
        ):
            _declaration("row-y", "cutover", body)

    def test_a_well_formed_body_still_builds_a_declaration(self) -> None:
        declaration = _declaration("row-z", "objective", _body())
        assert declaration.id == "row-z"
        assert declaration.title == "s"
        assert declaration.artifact == "report/{trading_day}/attribution.json"


def _declaration_pre_migration(row_id: str, source: str, body: dict) -> object:
    """`main`'s `_declaration`, verbatim (seen-failing comparison, policy
    §7.4) — not a live code path, so it can be exercised without editing
    `board.py` in place. `str(body["statement"])` silently coerces a
    wrongly typed field instead of refusing it, the exact defect class this
    migration exists to close."""
    from crucible.board import Declaration

    known = {
        "section",
        "statement",
        "clause_class",
        "surface",
        "reader",
        "artifact",
        "planned_because",
        "means_when_red",
    }
    unknown = set(body) - known
    if unknown:
        raise ValueError(f"{row_id}: unknown field(s) {sorted(unknown)}")
    missing = {"statement", "surface", "artifact", "means_when_red"} - set(body)
    if missing:
        raise ValueError(f"{row_id}: missing required field(s) {sorted(missing)}")
    return Declaration(
        id=row_id,
        source=source,
        title=str(body["statement"]).strip(),
        surface=str(body["surface"]),
        reader=body.get("reader"),
        artifact=str(body["artifact"]).strip(),
        means_when_red=str(body.get("means_when_red", "")),
        section=str(body.get("section", "")),
        clause_class=str(body.get("clause_class", "")),
        planned_because=str(body.get("planned_because", "")),
    )


class TestAWronglyTypedFieldWasSilentlyCoercedBeforeThisPR:
    def test_the_pre_migration_reader_silently_coerced_a_non_string_statement(self) -> None:
        declaration = _declaration_pre_migration("row-w", "objective", _body(statement=5))
        assert declaration.title == "5", (
            "documents the DEFECT this PR fixes: main's `_declaration` accepted an "
            "int `statement` and silently stringified it"
        )

    def test_the_new_reader_refuses_the_same_document(self) -> None:
        with pytest.raises(ValueError):
            _declaration("row-w", "objective", _body(statement=5))
