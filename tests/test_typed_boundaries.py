"""The typed document boundary, and the one-source-of-truth rule for schemas.

Normative source: `alpha-engine-config-I9847`. `crucible/models.py`'s
docstring is the design note; this module is what holds it to its claims.

**What is asserted, and why each half exists.**

The *reader* tests are about where a malformed document surfaces. Before this
boundary, `load_registry` did `raw["components"].items()` and then indexed
each row by hand, so a missing `log_location` raised a bare `KeyError` from
inside the loop — a traceback that reads, in a log, exactly like the reader
being broken rather than the file being wrong.

The *extra-key* test is the sharper half and is the reason `extra="forbid"`
is on every model: an untyped read does not merely report a typo badly, it
does not report one at all. `consol_surface:` in a row was a field somebody
wrote and nothing performed, in the one file whose entire purpose is
declaring what is observed.

The *schema* test is `I9847` deliverable 3. `crucible/schemas/` is the
cross-repo contract surface — `nous-ergon-ops/tests/crossrepo/
test_crucible_dispatch_lockstep.py` reads `components.yaml` across the repo
boundary, which is what makes it a versioned artifact under the M0 contract
discipline rather than a private config file. Pydantic is the in-process
reader and does NOT replace the published schema; the schema is GENERATED
from the model so there is one source of truth instead of two that must
agree. Two files that must agree is the shape that has already drifted
elsewhere in this fleet.
"""

from __future__ import annotations

import copy
import json
import pathlib

import pytest
import yaml
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from crucible.components import REGISTRY_PATH, load_registry
from crucible.models import ComponentsDocument

SCHEMA_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "crucible"
    / "schemas"
    / "components_registry.v1.json"
)


def _document() -> dict:
    return yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))


def _write(tmp_path: pathlib.Path, document: dict) -> str:
    target = tmp_path / "components.yaml"
    target.write_text(yaml.safe_dump(document), encoding="utf-8")
    return str(target)


class TestTheCommittedSchemaIsGeneratedFromTheModel:
    """One source of truth, not two that must agree."""

    def test_the_committed_schema_is_byte_identical_to_the_generated_one(self) -> None:
        generated = (
            json.dumps(ComponentsDocument.model_json_schema(), indent=2, sort_keys=True) + "\n"
        )
        committed = SCHEMA_PATH.read_text(encoding="utf-8")
        assert committed == generated, (
            f"{SCHEMA_PATH.name} has drifted from `crucible.models.ComponentsDocument`. "
            "The schema is GENERATED, never hand-edited: regenerate it in the same "
            "commit as the model change, so a cross-repo reader and this package "
            "cannot disagree about the shape of the same file."
        )

    def test_the_real_registry_validates_against_the_committed_schema(self) -> None:
        """The generated schema is only worth anything if the live file passes
        it — a schema nobody has validated a real document against is a
        schema nobody knows is right."""
        Draft202012Validator(json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))).validate(
            json.loads(ComponentsDocument.model_validate(_document()).model_dump_json())
        )


class TestAMalformedRowNamesTheRowAndTheField:
    """`I9847` deliverable 4: a named field error at the boundary, not a
    `KeyError` several functions in."""

    def test_a_missing_required_field_names_the_component_and_the_field(self, tmp_path) -> None:
        document = copy.deepcopy(_document())
        del document["components"]["data.daily"]["log_location"]

        with pytest.raises(ValidationError) as excinfo:
            load_registry(_write(tmp_path, document))

        message = str(excinfo.value)
        assert "data.daily" in message, "the failure names the ROW that is wrong"
        assert "log_location" in message, "the failure names the FIELD that is missing"

    def test_a_wrongly_typed_field_names_the_field_rather_than_failing_later(
        self, tmp_path
    ) -> None:
        document = copy.deepcopy(_document())
        document["components"]["data.daily"]["log_retention_days"] = "ninety"

        with pytest.raises(ValidationError) as excinfo:
            load_registry(_write(tmp_path, document))

        assert "log_retention_days" in str(excinfo.value)

    def test_an_UNKNOWN_key_is_REFUSED_rather_than_silently_ignored(self, tmp_path) -> None:
        """The half a careful call site cannot substitute for.

        An untyped reader indexes the keys it knows and never looks at the
        rest, so a mistyped key is an edit that does nothing — in the file
        that decides what is observed at all.
        """
        document = copy.deepcopy(_document())
        document["components"]["data.daily"]["consol_surface"] = "crucible/runs"

        with pytest.raises(ValidationError) as excinfo:
            load_registry(_write(tmp_path, document))

        message = str(excinfo.value)
        assert "consol_surface" in message
        assert "data.daily" in message

    def test_an_unknown_TOP_LEVEL_key_is_refused_too(self, tmp_path) -> None:
        document = copy.deepcopy(_document())
        document["defualts"] = {"page_channel": "x", "quiet_channel": "y"}

        with pytest.raises(ValidationError) as excinfo:
            load_registry(_write(tmp_path, document))

        assert "defualts" in str(excinfo.value)

    def test_an_omitted_signal_class_is_refused_not_defaulted_to_null(self, tmp_path) -> None:
        """§9.2: all five classes, always. An omitted class is
        indistinguishable from a forgotten one, so `null` must be written."""
        document = copy.deepcopy(_document())
        del document["components"]["data.daily"]["signals"]["outcome"]

        with pytest.raises(ValidationError) as excinfo:
            load_registry(_write(tmp_path, document))

        assert "outcome" in str(excinfo.value)


class TestTheCrossFieldRulesSurvivedTheMove:
    """The dispatch and deadline rules moved from `_assert_dispatch_declared`
    onto the model. They moved; they were not diluted."""

    def test_a_scheduled_row_with_a_null_dispatch_is_refused(self, tmp_path) -> None:
        document = copy.deepcopy(_document())
        document["components"]["data.daily"]["dispatch"] = None

        with pytest.raises(ValidationError) as excinfo:
            load_registry(_write(tmp_path, document))

        assert "scheduled but its dispatch is null" in str(excinfo.value)

    def test_an_on_demand_row_naming_a_starter_is_refused(self, tmp_path) -> None:
        document = copy.deepcopy(_document())
        row = document["components"]["data.daily"]
        row["schedule"] = None
        row["deadline"] = None
        row["dispatch"] = "scheduler"

        with pytest.raises(ValidationError) as excinfo:
            load_registry(_write(tmp_path, document))

        assert "on-demand but declares dispatch" in str(excinfo.value)

    def test_an_on_demand_row_with_a_deadline_is_refused(self, tmp_path) -> None:
        document = copy.deepcopy(_document())
        row = document["components"]["data.daily"]
        row["schedule"] = None
        row["dispatch"] = None

        with pytest.raises(ValidationError) as excinfo:
            load_registry(_write(tmp_path, document))

        assert "declares a deadline" in str(excinfo.value)

    def test_a_close_plus_deadline_without_an_offset_is_refused(self, tmp_path) -> None:
        document = copy.deepcopy(_document())
        document["components"]["data.daily"]["deadline"] = {
            "anchor": "close_plus",
            "cadence": "daily",
        }

        with pytest.raises(ValidationError) as excinfo:
            load_registry(_write(tmp_path, document))

        assert "close_plus deadline needs offset_hours" in str(excinfo.value)

    def test_a_third_anchor_is_refused_by_name(self, tmp_path) -> None:
        document = copy.deepcopy(_document())
        document["components"]["data.daily"]["deadline"] = {
            "anchor": "whenever",
            "offset_hours": 3,
        }

        with pytest.raises(ValidationError) as excinfo:
            load_registry(_write(tmp_path, document))

        assert "anchor" in str(excinfo.value)


class TestTheLiveRegistryStillReadsTheSameWay:
    """The boundary is typed; the values are unchanged."""

    def test_the_real_file_loads_and_a_known_row_is_intact(self) -> None:
        registry = load_registry()
        row = registry["data.daily"]
        assert row.dispatch == "scheduler"
        assert row.log_retention_days == 90
        assert row.deadline is not None
        assert row.deadline.anchor == "close_plus"
        assert row.deadline.offset_hours == 3
        assert set(row.signals) == {"execution", "cost", "resource", "lineage", "outcome"}
