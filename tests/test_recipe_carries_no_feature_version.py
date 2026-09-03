"""alpha-engine-config-I9801: the M recipe declares no `feature_version`.

The field used to be REQUIRED, hashed into the arm id, declared as the literal
`v1` by every recipe, and read by nothing — the run bound to
`crucible.features.DEFAULT_FEATURE_VERSION` regardless. crucible-PR52 chose
resolution (a): the layer version is a property of the RUN and is recorded as
run lineage (`features/<version>/<day>.parquet` in the manifest's `inputs[]`),
never declared by the recipe. This module is the guard that keeps it that way:
a recipe field that reads as a guarantee while enforcing nothing must not come
back under either name.
"""

from __future__ import annotations

import dataclasses

from crucible.features import DEFAULT_FEATURE_VERSION
from crucible.slots.model import REQUIRED_RECIPE_FIELDS, FeatureLayerSource, ModelRecipe


def test_the_recipe_contract_carries_no_feature_version() -> None:
    assert "feature_version" not in REQUIRED_RECIPE_FIELDS
    assert "feature_version" not in {f.name for f in dataclasses.fields(ModelRecipe)}


def test_the_layer_version_a_run_reads_is_derived_not_declared() -> None:
    """The value the run binds to is the catalogue hash, resolved by the
    source — the recipe has no say, so there is nothing for it to diverge
    from. A hand-written `v1` cannot re-enter through this path."""
    source = FeatureLayerSource(store=None)  # type: ignore[arg-type]
    assert str(source.version) == DEFAULT_FEATURE_VERSION
    assert DEFAULT_FEATURE_VERSION.startswith("v") and DEFAULT_FEATURE_VERSION != "v1"
