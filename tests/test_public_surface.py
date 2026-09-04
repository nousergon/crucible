"""The consumer contract: `crucible.slots` is a stable place to import from.

`alpha-engine-config` validates `strategy/arms/{u,r,m,s}/` with THIS
package's real loaders (`alpha-engine-config-I9766`) instead of mirroring
the registries in a second file. That only works if the registries a recipe
is checked against — which exit rules exist, which estimator kinds exist,
which fields a model recipe must pre-register, and the units-suffix
contract — are reachable from a public name rather than a module-private
one. This test is the producer-side half of that contract: it fails if any
of those five names stops being importable from `crucible.slots`, or if the
package-level re-export drifts from the module that actually defines it.

Normative source: `alpha-engine-config-I9766` deliverable 1; plan §4.11,
§9.1.
"""

from __future__ import annotations

import crucible.slots as slots
from crucible.slots import arms as arms_module
from crucible.slots import model as model_module
from crucible.slots import strategy as strategy_module


def test_exit_rules_is_exported_and_is_the_strategy_modules_own_registry() -> None:
    assert slots.EXIT_RULES is strategy_module.EXIT_RULES
    assert isinstance(slots.EXIT_RULES, dict)
    assert "position_loss_floor" in slots.EXIT_RULES


def test_estimator_kinds_is_exported_and_matches_the_models_closed_set() -> None:
    # The model module keeps its estimator-kind tuple private (`_ESTIMATORS`):
    # this asserts the package-level alias has not drifted from it, so the
    # re-export cannot go stale the way `test_strategy_arms.py`'s mirrored
    # copy in `alpha-engine-config` already had.
    assert tuple(slots.ESTIMATOR_KINDS) == tuple(model_module._ESTIMATORS)
    assert "ridge" in slots.ESTIMATOR_KINDS
    assert "ols" in slots.ESTIMATOR_KINDS


def test_required_recipe_fields_is_exported_and_is_the_models_own_tuple() -> None:
    assert slots.REQUIRED_RECIPE_FIELDS is model_module.REQUIRED_RECIPE_FIELDS
    assert "features" in slots.REQUIRED_RECIPE_FIELDS
    assert "estimator" in slots.REQUIRED_RECIPE_FIELDS


def test_units_suffixes_is_exported_and_is_the_models_own_tuple() -> None:
    assert slots.UNITS_SUFFIXES is model_module.UNITS_SUFFIXES
    assert slots.UNITS_SUFFIXES == ("_raw", "_ratio", "_pct", "_zscore", "_log_return")


def test_required_arm_fields_is_exported_and_is_the_arms_modules_own_tuple() -> None:
    assert slots.REQUIRED_ARM_FIELDS is arms_module.REQUIRED_ARM_FIELDS


def test_estimator_spec_refuses_a_kind_outside_the_exported_set() -> None:
    """A guard nobody has made fail is a guard nobody knows works (AGENTS.md
    Test discipline). This shows the exported set is the one actually
    enforced, not a second list that merely resembles it."""
    import pytest

    with pytest.raises(ValueError, match="unknown estimator"):
        slots.EstimatorSpec(kind="not_a_real_estimator")


def test_the_loaders_are_exported_for_a_consumer_that_only_has_a_directory() -> None:
    """`alpha-engine-config` has no `Store`; it validates a checked-out
    strategy tree by path, so the directory-based loaders must be reachable
    from the same public surface as the registries they check against."""
    assert slots.load_model_recipes is model_module.load_model_recipes
    assert slots.load_strategy_recipes is strategy_module.load_strategy_recipes
    assert slots.load_arm_specs is arms_module.load_arm_specs
