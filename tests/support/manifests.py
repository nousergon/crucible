"""A minimal, real-shaped `experiment.grade` run manifest, for tests that
seed the graded `arena_cycle` artifact directly rather than through the real
`experiment.grade` job.

`crucible.promote.read_graded_cycle` (`alpha-engine-config-I10679`) refuses
to act on an `arena_cycle` document unless the grade job's OWN manifest
claims it as an output with `status: ok` — a document existing at the
expected key is not proof that run wrote it (rule 1, `crucible/AGENTS.md`:
"manifest or it did not happen"). A fixture that seeds only the cycle
document, as every promote fixture did before this issue, therefore no
longer exercises the real path; this helper writes the manifest alongside it.
"""

from __future__ import annotations

import json

from crucible.keys import arena_cycle_key, manifest_key
from crucible.store import Store


def write_grade_manifest(
    store: Store,
    slot: str,
    as_of: str,
    *,
    run_id: str = "0" * 26,
    status: str = "ok",
    reason: str = "",
) -> str:
    """Write `runs/experiment.grade/{as_of}/{slot}/run.json`, claiming the
    slot's `arena_cycle` key as an output. Returns the manifest key."""
    key = manifest_key("experiment.grade", as_of, discriminator=slot)
    store.put_bytes(
        key,
        json.dumps(
            {
                "schema_version": 2,
                "run_id": run_id,
                "job": "experiment.grade",
                "trading_day": as_of,
                "status": status,
                "reason": reason,
                "outputs": [{"key": arena_cycle_key(slot, as_of), "sha256": "0" * 64}],
            }
        ).encode("utf-8"),
    )
    return key
