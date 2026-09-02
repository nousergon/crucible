"""The feature layer's producer/consumer contract, and the schema that is it.

Normative source: plan §10 component 4; `alpha-engine-config-I9772` and
`-I9765`; the M0 contract discipline in `AGENTS.md` ("every new cross-repo
artifact gets a versioned schema + producer/consumer contract test at
birth").

**Why this file exists at all.** Track A built the producer
(`crucible/features/`) and track B the consumer
(`crucible.slots.model.FeatureLayerSource`) against the same interface, and
they disagreed in three places — the element type of `CATALOG`, whether a
consumer names the layer version, and a column name. Neither reading was
wrong; the interface was simply never *declared*, so there was nothing for
either to be wrong against. `crucible/schemas/feature_registry.v1.json` is
that declaration, and this file is the contract test on both sides of it:

* the **producer** may not emit a registry document the schema refuses, and
  `registry_payload()` validates on the way out so no writer has to remember;
* the **consumer** resolves exactly the names the document lists, refuses a
  column it does not list, and never names a version of its own.

Every assertion below that tests the schema tests it REFUSING something. A
schema shown only to accept valid documents has not been shown to constrain
anything.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from crucible.features import (
    CATALOG,
    DEFAULT_FEATURE_VERSION,
    FEATURE_REGISTRY_SCHEMA_VERSION,
    FeatureRegistryValidationError,
    feature_names,
    load_registry_schema,
    registry_payload,
    validate_registry_payload,
)
from crucible.features.docgen import DOC_PATH, render_document
from crucible.slots.model import FeatureLayerSource


def _valid() -> dict[str, Any]:
    """The real producer's document. The contract is about THIS payload, not
    a fixture that resembles it — a schema validated only against a hand-made
    example proves nothing about what the layer actually writes."""
    return copy.deepcopy(registry_payload())


def _errors(payload: dict[str, Any]) -> list[str]:
    return [e.message for e in Draft202012Validator(load_registry_schema()).iter_errors(payload)]


class TestTheSchemaIsWellFormed:
    def test_the_schema_itself_validates_against_its_metaschema(self) -> None:
        Draft202012Validator.check_schema(load_registry_schema())

    def test_the_schema_is_the_version_the_producer_stamps(self) -> None:
        schema = load_registry_schema()
        assert schema["properties"]["schema_version"]["const"] == FEATURE_REGISTRY_SCHEMA_VERSION
        assert _valid()["schema_version"] == FEATURE_REGISTRY_SCHEMA_VERSION


class TestTheProducerConforms:
    def test_the_real_catalogue_validates(self) -> None:
        assert not _errors(_valid())

    def test_registry_payload_validates_on_the_way_out(self, monkeypatch) -> None:
        """The producer cannot emit a non-conforming document even if a caller
        forgets to check, because the check is inside the constructor of the
        document rather than beside every call site of it."""
        import crucible.features.registry as registry_module

        monkeypatch.setattr(registry_module, "feature_version", lambda catalog=CATALOG: "v1")
        with pytest.raises(FeatureRegistryValidationError, match="feature_registry.v1"):
            registry_module.registry_payload()

    def test_the_document_names_every_catalogue_column_in_order(self) -> None:
        assert tuple(f["name"] for f in _valid()["features"]) == feature_names(CATALOG)

    def test_it_is_json_serialisable_as_written(self) -> None:
        """`crucible/data/daily.py` writes it with `json.dumps(..., sort_keys=True)`."""
        assert json.loads(json.dumps(_valid(), sort_keys=True)) == _valid()


class TestTheSchemaRefuses:
    def test_a_hand_written_feature_version(self) -> None:
        """Disagreement 2 of `alpha-engine-config-I9772`, made a validation
        failure. A declared `"v1"` lets an edited recipe overwrite the layer
        an earlier verdict was computed from, and nothing would show it."""
        payload = _valid()
        payload["feature_version"] = "v1"
        assert _errors(payload)

    def test_a_bare_column_name(self) -> None:
        """`avg_volume_20d`: emitted as a normalized ratio, consumed as raw
        shares, 901 of 903 tickers silently failing the liquidity gate."""
        payload = _valid()
        payload["features"][0]["name"] = "avg_volume_20d"
        assert _errors(payload)

    def test_a_ratio_column_declaring_some_other_unit(self) -> None:
        payload = _valid()
        entry = next(f for f in payload["features"] if f["name"].endswith("_ratio"))
        entry["unit"] = "USD"
        assert _errors(payload)

    def test_a_raw_column_claiming_a_normalized_unit(self) -> None:
        """The same defect in the other direction: the name promises the
        consumer an unnormalized value and the declared unit says otherwise."""
        payload = _valid()
        entry = next(f for f in payload["features"] if f["name"].endswith("_raw"))
        entry["unit"] = "zscore"
        assert _errors(payload)

    @pytest.mark.parametrize("variant", ["Ratio", "RATIO", "ratio "])
    def test_a_raw_column_claiming_a_case_or_whitespace_variant_of_a_normalized_unit(
        self, variant: str
    ) -> None:
        """`alpha-engine-config-I9815`: the `_raw`/`not`/`enum` branch used to
        compare against the exact strings `["ratio", "pct", "zscore",
        "log_return"]` while `unit` itself was a free string, so any case or
        whitespace variant of a normalized word walked straight through —
        `avg_volume_20d_raw` declaring `unit: "Ratio"` validated. `unit` is
        now a closed enum over the concrete vocabulary the catalogue uses, so
        a variant spelling is not merely refused by the allOf check, it is
        not a legal `unit` value at all."""
        payload = _valid()
        entry = next(f for f in payload["features"] if f["name"].endswith("_raw"))
        entry["unit"] = variant
        assert _errors(payload)

    def test_a_column_with_no_lineage(self) -> None:
        payload = _valid()
        payload["features"][0]["inputs"] = []
        assert _errors(payload)

    def test_a_window_of_zero_sessions(self) -> None:
        payload = _valid()
        payload["features"][0]["window_trading_days"] = 0
        assert _errors(payload)

    def test_an_undeclared_field_on_a_feature(self) -> None:
        """`additionalProperties: false` on both levels: a field the consumer
        does not read, carried in the artifact, is a second interface."""
        payload = _valid()
        payload["features"][0]["source_of_truth"] = "somewhere else"
        assert _errors(payload)

    def test_a_document_with_no_features(self) -> None:
        payload = _valid()
        payload["features"] = []
        assert _errors(payload)

    def test_a_document_that_does_not_say_which_schema_it_is(self) -> None:
        payload = _valid()
        del payload["schema_version"]
        assert _errors(payload)

    def test_validate_registry_payload_names_the_offending_path(self) -> None:
        payload = _valid()
        payload["features"][0]["inputs"] = []
        with pytest.raises(FeatureRegistryValidationError, match=r"features/0/inputs"):
            validate_registry_payload(payload)


class TestTheConsumerBindsToTheSameDocument:
    """`FeatureLayerSource` is the consumer side. It reads the registry object
    and never a path, a literal column list, or a version string of its own."""

    def test_the_consumer_resolves_exactly_the_documented_columns(self) -> None:
        layer = FeatureLayerSource(store=object(), registry=CATALOG)
        assert layer.columns == tuple(f["name"] for f in _valid()["features"])

    def test_no_consumer_names_a_version(self) -> None:
        """Disagreement 2, resolved in the producer's favour: `version=None`
        resolves the DERIVED version, so an edited catalogue writes to a new
        prefix instead of overwriting an earlier verdict's layer."""
        layer = FeatureLayerSource(store=object(), registry=CATALOG)
        assert layer.version == DEFAULT_FEATURE_VERSION
        assert layer.version == _valid()["feature_version"]
        assert layer.key("2026-08-28") == f"features/{DEFAULT_FEATURE_VERSION}/2026-08-28.parquet"

    def test_a_column_the_document_does_not_list_is_refused_by_name(self) -> None:
        """The 2026-08-28 condition. It must fail at load, naming the column,
        and never arrive as a silently substituted zero. `alpha-engine-config
        -I9777` depends on this raising: it is how a model-output column that
        the price panel cannot produce announces itself."""
        layer = FeatureLayerSource(store=object(), registry=CATALOG)
        with pytest.raises(KeyError, match="not produced by feature layer version") as exc:
            layer.panel(
                trading_day="2026-08-28",
                columns=("gbm_directional_score_zscore",),
            )
        assert "gbm_directional_score_zscore" in str(exc.value)

    def test_a_panel_of_no_columns_is_refused(self) -> None:
        layer = FeatureLayerSource(store=object(), registry=CATALOG)
        with pytest.raises(ValueError, match="at least one column"):
            layer.panel(trading_day="2026-08-28", columns=())

    def test_the_consumers_registry_element_type_is_the_producers(self) -> None:
        """Disagreement 1, resolved in the producer's favour: `CATALOG` is a
        tuple of specs, not of names. A registry of names is enough for a
        units check and nothing else; the lineage each spec carries is what
        `explain` answers "which data produced this column" from. The
        consumer reads names through the accessor the producer exposes."""
        layer = FeatureLayerSource(store=object(), registry=CATALOG)
        assert layer.columns == feature_names(CATALOG)
        assert all(hasattr(spec, "inputs") and spec.inputs for spec in CATALOG)


class TestTheDocumentationIsTheCatalogue:
    def test_the_documentation_table_is_the_catalogue(self) -> None:
        """The fleet rule asks for a documentation row per column. Asserting
        the rendered document rather than a name set means a column added
        without regenerating the table is red HERE, and the fix is a command
        the failure message names."""
        current = DOC_PATH.read_text(encoding="utf-8")
        assert current == render_document(current), (
            f"{DOC_PATH} is stale. Regenerate it with `uv run python -m crucible.features.docgen`."
        )

    def test_every_column_has_a_row(self) -> None:
        current = DOC_PATH.read_text(encoding="utf-8")
        for name in feature_names(CATALOG):
            assert f"`{name}`" in current, f"{name} has no row in {DOC_PATH}"
