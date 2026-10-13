"""A real-shaped `arm_predictions.v1` document, for tests that move the M pointer.

Moving slot ``m``'s pointer now requires the arm being seated to have a
servable cross-section for the decision's session — `crucible.promote`
refuses the write otherwise, and republishes `predictions/{as_of}.json` for
the new champion when it succeeds (2026-10-05: the 2026-10-02 promotion
seated an arm that had produced nothing that day). A test that promotes on M
with a store therefore writes the document `experiment.run[m]` would have.
"""

from __future__ import annotations

import json

from crucible.keys import arm_predictions_key, strategy_arm_key
from crucible.slots.inputs import ARM_PREDICTIONS_SCHEMA_VERSION
from crucible.slots.model import ModelRecipe, _parse_model_recipe
from crucible.store import Store


def seed_arm_predictions(store: Store, arm_id: str, trading_day: str) -> str:
    """Write ``arm_id``'s `arm_predictions.v1` for ``trading_day``; return the key.

    When ``arm_id`` is the id :func:`m_arm_id` derives for its own name, the
    recipe is filed too (:func:`seed_m_recipe`): an M arm that produced a
    cross-section had a recipe to produce it from, and seating it now reads
    that recipe's target (`alpha-engine-config-I12121`).
    """
    _seed_recipe_if_derived(store, arm_id)
    key = arm_predictions_key(arm_id, trading_day)
    store.put_bytes(
        key,
        json.dumps(
            {
                "schema_version": ARM_PREDICTIONS_SCHEMA_VERSION,
                "arm_id": arm_id,
                "trading_day": trading_day,
                "feature_version": "features.v3",
                "predicted_alpha": {"AAA": 0.031, "BBB": -0.012},
            },
            indent=2,
            sort_keys=True,
        ).encode("utf-8"),
    )
    return key


def m_recipe_bytes(name: str, *, target: str | None = None) -> bytes:
    """A minimal, valid M recipe named ``name``, optionally declaring ``target``.

    `alpha-engine-config-I12121`: seating an M arm (by promotion or by
    operator revert) and publishing its feed now read the arm's recipe out of
    the strategy tree, and refuse an arm whose target is not a signed forward
    return — or one no recipe declares. A test that does either therefore
    files the recipe `experiment.run[m]` would have read, and uses the arm id
    that recipe derives (:func:`m_arm_id`).
    """
    lines = [
        "slot: m",
        f"name: {name}",
        "spec:",
        "  features: [momentum_20d_zscore]",
        "  estimator: {kind: ridge, alpha: 1.0}",
        "  label_horizon_trading_days: 2",
        "  refit_cadence_trading_days: 5",
        "  training_window: {kind: expanding, min_trading_days: 10}",
        "  cpcv: {n_groups: 4, k_test: 1, embargo_trading_days: 1}",
    ]
    if target is not None:
        lines.append(f"  target: {target}")
    lines.append("registered_at: '2026-06-01'")
    return ("\n".join(lines) + "\n").encode("utf-8")


def m_recipe(name: str, *, target: str | None = None) -> ModelRecipe:
    """The parsed recipe :func:`m_recipe_bytes` files."""
    return _parse_model_recipe(m_recipe_bytes(name, target=target), f"<test recipe {name}>")


def m_arm_id(name: str, *, target: str | None = None) -> str:
    """The arm id the recipe :func:`m_recipe_bytes` files derives."""
    return m_recipe(name, target=target).arm_id


def seed_m_recipe(store: Store, name: str, *, target: str | None = None) -> str:
    """File :func:`m_recipe_bytes` in the store's synced strategy tree; return its arm id."""
    store.put_bytes(strategy_arm_key("m", name), m_recipe_bytes(name, target=target))
    return m_arm_id(name, target=target)


def _seed_recipe_if_derived(store: Store, arm_id: str) -> None:
    slot, _, rest = arm_id.partition(":")
    name = rest.rpartition(":")[0]
    if slot == "m" and name and m_arm_id(name) == arm_id:
        seed_m_recipe(store, name)
