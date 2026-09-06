"""The arm recipe as a typed document boundary (`alpha-engine-config-I10045` row 2).

Normative source: `alpha-engine-config-I10045`; `crucible/models.py`'s module
docstring and `crucible.models.ArmRecipeDocument`'s own docstring carry the
design note for this specific boundary. `tests/test_typed_boundaries.py`
(row 1, `ComponentsDocument`) is the worked example this file copies the
shape of.

Same two halves as row 1: a malformed recipe used to raise `KeyError` on
whichever required field the hand-written reader happened to check for next,
never the field that was actually missing or wrong, and an unknown top-level
key registered cleanly and did nothing (`alpha-engine-config-I9944`, the
exact same silent-typo failure `components.yaml` had). The schema half is new
here — `arm_recipe.v1.json` did not exist before this PR — so the contract
test is the ONLY thing keeping the committed file and the model from
drifting the moment either one is edited alone.
"""

from __future__ import annotations

import json
import pathlib

import pytest
import yaml
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from crucible.models import ARM_RECIPE_REQUIRED_FIELDS, ArmRecipeDocument
from crucible.slots.arms import REQUIRED_ARM_FIELDS, load_arm_specs

SCHEMA_PATH = (
    pathlib.Path(__file__).resolve().parents[1] / "crucible" / "schemas" / "arm_recipe.v1.json"
)


def _write(directory: pathlib.Path, name: str, **fields: object) -> None:
    body: dict[str, object] = {
        "name": name,
        "slot": "u",
        "ranker": "momentum_sleeve",
        "params": {"top_n": 8},
        "registered_at": "2026-06-01",
    }
    body.update(fields)
    (directory / f"{name}.yaml").write_text(yaml.safe_dump(body), encoding="utf-8")


@pytest.fixture
def arms_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    directory = tmp_path / "arms" / "u"
    directory.mkdir(parents=True)
    return directory


class TestTwoTuplesOfRequiredFieldsCannotDriftApart:
    """`ARM_RECIPE_REQUIRED_FIELDS` is restated on the model, deliberately not
    imported from `crucible.slots.arms`, so the model carries no import-time
    dependency on the reader it types. Restating is only safe if the two
    tuples are pinned equal."""

    def test_the_model_and_the_reader_agree_on_the_required_set(self) -> None:
        assert ARM_RECIPE_REQUIRED_FIELDS == REQUIRED_ARM_FIELDS


class TestTheCommittedSchemaIsGeneratedFromTheModel:
    """One source of truth, not two that must agree."""

    def test_the_committed_schema_is_byte_identical_to_the_generated_one(self) -> None:
        generated = (
            json.dumps(ArmRecipeDocument.model_json_schema(), indent=2, sort_keys=True) + "\n"
        )
        committed = SCHEMA_PATH.read_text(encoding="utf-8")
        assert committed == generated, (
            f"{SCHEMA_PATH.name} has drifted from `crucible.models.ArmRecipeDocument`. "
            "The schema is GENERATED, never hand-edited: regenerate it in the same "
            "commit as the model change."
        )

    def test_a_real_filed_recipe_shape_validates_against_the_committed_schema(
        self, arms_dir: pathlib.Path
    ) -> None:
        """A schema nobody has validated a real document against is a schema
        nobody knows is right."""
        _write(arms_dir, "momentum_sleeve")
        raw = yaml.safe_load((arms_dir / "momentum_sleeve.yaml").read_text(encoding="utf-8"))
        validated = ArmRecipeDocument.model_validate(raw)
        Draft202012Validator(json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))).validate(
            json.loads(validated.model_dump_json())
        )


class TestAMalformedRecipeNamesTheFieldNotJustKeyError:
    """`I10045` row 2 deliverable: a named field error at the boundary, not a
    `KeyError` several functions in on whichever field the hand-written
    reader happened to check next."""

    def test_a_missing_required_field_names_it(self, arms_dir: pathlib.Path) -> None:
        (arms_dir / "half.yaml").write_text(
            "name: half\nslot: u\nparams:\n  top_n: 5\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match="missing required field") as excinfo:
            load_arm_specs("u", strategy_dir=arms_dir.parent.parent)
        assert "ranker" in str(excinfo.value)
        assert "registered_at" in str(excinfo.value)

    def test_empty_params_is_refused_same_as_absent_params(self, arms_dir: pathlib.Path) -> None:
        """Preserves the reader's pre-existing falsy-field behaviour exactly:
        `params: {}` and an absent `params` key were always treated alike."""
        _write(arms_dir, "empty_params", params={})
        with pytest.raises(ValueError, match="missing required field"):
            load_arm_specs("u", strategy_dir=arms_dir.parent.parent)

    def test_a_wrongly_typed_field_names_the_field(self, arms_dir: pathlib.Path) -> None:
        _write(arms_dir, "bad_type", registered_at=20260601)
        with pytest.raises(ValueError, match="registered_at"):
            load_arm_specs("u", strategy_dir=arms_dir.parent.parent)

    def test_an_unknown_top_level_key_is_refused_by_name(self, arms_dir: pathlib.Path) -> None:
        """`alpha-engine-config-I9944`'s exact failure, restated for this
        boundary: an untyped read did not merely report a typo badly, it did
        not report one at all."""
        _write(arms_dir, "extra", bogus_field="nope")
        with pytest.raises(ValueError, match="bogus_field"):
            load_arm_specs("u", strategy_dir=arms_dir.parent.parent)

    def test_a_non_date_registered_at_is_refused_by_the_schema(
        self, arms_dir: pathlib.Path
    ) -> None:
        _write(arms_dir, "not_a_date", registered_at="last Tuesday")
        with pytest.raises(ValueError, match="registered_at"):
            load_arm_specs("u", strategy_dir=arms_dir.parent.parent)


class TestARecipeThatIsWellFormedStillLoadsTheSameWay:
    """The boundary is typed; the values are unchanged."""

    def test_a_valid_recipe_still_registers_with_every_field_intact(
        self, arms_dir: pathlib.Path
    ) -> None:
        _write(arms_dir, "momentum_sleeve", notes="a clarifying comment")
        [spec] = load_arm_specs("u", strategy_dir=arms_dir.parent.parent)
        assert spec.name == "momentum_sleeve"
        assert spec.slot == "u"
        assert spec.ranker == "momentum_sleeve"
        assert spec.params == {"top_n": 8}
        assert spec.registered_at == "2026-06-01"
        assert spec.notes == "a clarifying comment"
        assert spec.control is False
        assert spec.source_key.endswith("momentum_sleeve.yaml")


class TestDirectModelValidation:
    """`ArmRecipeDocument` on its own — no store, no reader — matching the
    model-level assertions row 1 makes for `ComponentsDocument`."""

    def test_an_unregistered_llm_callsite_is_still_a_reader_check_not_a_model_one(self) -> None:
        """`params.llm_callsite` legality depends on a live, changing
        registry (`crucible.llm.LLM_CALLSITE_REGISTRY`), so the model accepts
        any string there — the READER (`_require_registered_callsite`, still
        exercised via `load_arm_specs`) is what refuses an unknown one."""
        document = ArmRecipeDocument.model_validate(
            {
                "name": "x",
                "slot": "u",
                "ranker": "momentum_sleeve",
                "params": {"llm_callsite": "not_a_real_callsite"},
                "registered_at": "2026-06-01",
            }
        )
        assert document.params["llm_callsite"] == "not_a_real_callsite"

    def test_schema_version_defaults_for_a_recipe_filed_before_this_boundary(self) -> None:
        document = ArmRecipeDocument.model_validate(
            {
                "name": "x",
                "slot": "u",
                "ranker": "momentum_sleeve",
                "params": {"top_n": 1},
                "registered_at": "2026-06-01",
            }
        )
        assert document.schema_version == "arm_recipe.v1"

    def test_extra_top_level_key_is_forbidden_at_the_model_level(self) -> None:
        with pytest.raises(ValidationError, match="bogus"):
            ArmRecipeDocument.model_validate(
                {
                    "name": "x",
                    "slot": "u",
                    "ranker": "momentum_sleeve",
                    "params": {"top_n": 1},
                    "registered_at": "2026-06-01",
                    "bogus": "nope",
                }
            )
