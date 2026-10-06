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

from crucible.keys import arm_predictions_key
from crucible.slots.inputs import ARM_PREDICTIONS_SCHEMA_VERSION
from crucible.store import Store


def seed_arm_predictions(store: Store, arm_id: str, trading_day: str) -> str:
    """Write ``arm_id``'s `arm_predictions.v1` for ``trading_day``; return the key."""
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
