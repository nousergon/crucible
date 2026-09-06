"""The LLM call-site registry as a typed document boundary
(`alpha-engine-config-I10045` row 4).

Normative source: `alpha-engine-config-I10045`; `crucible.llm`'s module
docstring; `crucible.models.LlmCallsiteRegistryDocument`'s own docstring.
`llm_callsite_registry.v1` is declared in `crucible/llm_callsites.yaml`
already (`schema_version: llm_callsite_registry.v1`) but had no schema file
before this PR — per the parent issue, that document class's row does not
publish `crucible/schemas/llm_callsite_registry.v1.json` because the
registry is a package-internal file with no cross-repo reader, unlike
`components.yaml`; this row types the READER
(`crucible.llm.load_registry`/`load_capability_classes`), matching row 4's
"no schema today" instruction at the model level without inventing a
cross-repo contract nothing consumes.

Same defect shape as every other row: `load_registry` used to build
`CallSite` by hand off a raw dict, so a wrongly typed `max_usd_per_call`
(a string, say) fell through to `float(row["max_usd_per_call"])` — either a
silent coercion or a bare exception naming neither the call site nor the
field.
"""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from crucible.models import LlmCallsiteRegistryDocument


def _document(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "llm_callsite_registry.v1",
        "capability_classes": ["reasoning_high"],
        "callsites": {
            "research.thesis": {
                "purpose": "draft one arm's thesis from the day's features",
                "capability_class": "reasoning_high",
                "max_usd_per_call": 0.25,
                "owner": "crucible.slots.research",
            }
        },
    }
    payload.update(overrides)
    return payload


class TestTheRealFileValidates:
    def test_the_shipped_registry_validates_as_is(self) -> None:
        from crucible.llm import CALLSITE_REGISTRY_PATH

        raw = yaml.safe_load(CALLSITE_REGISTRY_PATH.read_text(encoding="utf-8"))
        document = LlmCallsiteRegistryDocument.model_validate(raw)
        assert document.schema_version == "llm_callsite_registry.v1"
        assert document.callsites == {}
        assert "reasoning_high" in document.capability_classes


class TestALoadedRegistryStillWorksTheSameWay:
    def test_load_registry_matches_the_module_level_mapping(self) -> None:
        from crucible.llm import LLM_CALLSITE_REGISTRY, load_registry

        assert load_registry() == LLM_CALLSITE_REGISTRY

    def test_load_capability_classes_is_non_empty_and_declared(self) -> None:
        from crucible.llm import load_capability_classes

        assert load_capability_classes()


class TestAMalformedRegistryNamesTheRowAndTheField:
    def test_a_row_missing_a_required_field_names_it(self) -> None:
        document = _document()
        del document["callsites"]["research.thesis"]["owner"]
        with pytest.raises(ValidationError, match="owner"):
            LlmCallsiteRegistryDocument.model_validate(document)

    def test_a_wrongly_typed_ceiling_names_the_field_rather_than_failing_later(self) -> None:
        document = _document()
        document["callsites"]["research.thesis"]["max_usd_per_call"] = "a lot"
        with pytest.raises(ValidationError, match="max_usd_per_call"):
            LlmCallsiteRegistryDocument.model_validate(document)

    def test_a_non_positive_ceiling_is_refused(self) -> None:
        document = _document()
        document["callsites"]["research.thesis"]["max_usd_per_call"] = 0
        with pytest.raises(ValidationError, match="max_usd_per_call"):
            LlmCallsiteRegistryDocument.model_validate(document)

    def test_an_unknown_row_key_is_refused_not_silently_ignored(self) -> None:
        document = _document()
        document["callsites"]["research.thesis"]["model_id"] = "claude-x"
        with pytest.raises(ValidationError, match="model_id"):
            LlmCallsiteRegistryDocument.model_validate(document)

    def test_an_unknown_top_level_key_is_refused(self) -> None:
        document = _document(bogus_field="nope")
        with pytest.raises(ValidationError, match="bogus_field"):
            LlmCallsiteRegistryDocument.model_validate(document)

    def test_a_missing_capability_classes_key_is_refused(self) -> None:
        document = _document()
        del document["capability_classes"]
        with pytest.raises(ValidationError, match="capability_classes"):
            LlmCallsiteRegistryDocument.model_validate(document)

    def test_an_empty_string_capability_class_is_refused(self) -> None:
        document = _document(capability_classes=["reasoning_high", ""])
        with pytest.raises(ValidationError, match="capability_classes"):
            LlmCallsiteRegistryDocument.model_validate(document)

    def test_a_wrong_schema_version_is_refused_by_name(self) -> None:
        document = _document(schema_version="llm_callsite_registry.v2")
        with pytest.raises(ValidationError, match="schema_version"):
            LlmCallsiteRegistryDocument.model_validate(document)


class TestAWronglyTypedFieldSurfacesAtTheLiveReader:
    """The end-to-end version of the same guard: a broken file on disk fails
    at `load_registry`/`load_capability_classes`, not several calls later."""

    def test_a_broken_registry_file_fails_at_load_naming_the_field(self, tmp_path) -> None:
        import crucible.llm as llm

        broken = tmp_path / "llm_callsites.yaml"
        broken.write_text(
            yaml.safe_dump(
                {
                    "schema_version": "llm_callsite_registry.v1",
                    "capability_classes": [],
                    # "high" is a router TIER GROUP (`krepis.router.TIER_GROUPS`),
                    # not the open-ruling `reasoning_high`
                    # (`crucible.llm.CAPABILITY_CLASS_GROUPS`) — chosen so this
                    # fixture exercises the field-typing defect and nothing else.
                    "callsites": {
                        "research.thesis": {
                            "purpose": "x",
                            "capability_class": "high",
                            "max_usd_per_call": "not a number",
                            "owner": "crucible.slots.research",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(llm, "CALLSITE_REGISTRY_PATH", broken)
            llm.load_registry.cache_clear()
            try:
                with pytest.raises(ValueError, match="max_usd_per_call"):
                    llm.load_registry()
            finally:
                llm.load_registry.cache_clear()
