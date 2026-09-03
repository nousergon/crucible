"""Arm ids are pinned to literals, so "no existing arm re-ids" is a fact.

Policy §3.1: an arm's id is the hash of its spec, and an id that moves
orphans the arm's score series. Any change to what `ArmSpec.spec` hashes —
a key added, a default changed, a dict re-shaped — moves every literal below
at once, and this module turns that from a silent re-id into a red test that
names it. A deliberate spec-shape change updates these literals in the same
PR, with the re-id stated in the body.

The literals are the ids on `main` at c681458 (2026-09-03), before
`params.llm_callsite` existed (alpha-engine-config-I9920): that change had to
leave them untouched, and this is the proof. Controls are pinned because they
are the only arms whose recipe this public repo may carry (the harness
generates them; a filed recipe's params are strategy edge). The one filed
shape is synthetic — the fixture recipe `tests/test_slots_arms.py` has used
since the loader existed.

Note: `m:control_planted_m:7e8059f49558` — the literal quoted in
`crucible.slots.is_control_arm`'s docstring from I9757 (2026-09-02) — is NOT
today's id; the control spec shape moved between then and c681458, before any
register existed in the production store (verified: `arms/{slot}/register.jsonl`
is absent for every slot), so no series was orphaned. From here on it would be.
"""

from __future__ import annotations

import pytest

from crucible.slots import get_slot
from crucible.slots.arms import ArmSpec, control_specs

#: `{slot: {control name: arm id}}`, every control the harness generates.
CONTROL_IDS = {
    "u": {
        "control_planted_u": "u:control_planted_u:e74080d23f21",
        "control_null_u": "u:control_null_u:082d1c6a2c86",
    },
    "r": {
        "control_planted_r": "r:control_planted_r:21dffe423263",
        "control_null_r": "r:control_null_r:ba4ea5e29c56",
    },
    "m": {
        "control_planted_m": "m:control_planted_m:82a31b07d640",
        "control_null_m": "m:control_null_m:69da7cdc01ac",
    },
    "s": {
        "control_planted_s": "s:control_planted_s:ab39cc7bb03c",
        "control_null_s": "s:control_null_s:661bb954c2f5",
    },
}

#: The synthetic filed recipe every arm-loader test builds.
FILED_ID = "u:momentum_sleeve:0b4e82321e3f"


@pytest.mark.parametrize("slot", sorted(CONTROL_IDS))
def test_every_control_arm_keeps_its_id(slot: str) -> None:
    ids = {spec.name: spec.arm_id for spec in control_specs(get_slot(slot))}
    assert ids == CONTROL_IDS[slot], (
        f"slot {slot!r} control ids moved. Either a hashed field on ArmSpec.spec "
        "changed shape (a re-id of every registered arm — state it in the PR and "
        "update these literals) or a control's definition changed (a new arm; the old "
        "one keeps its record)."
    )


def test_a_filed_recipe_without_a_call_site_keeps_its_id() -> None:
    """The I9920 guarantee stated as a literal: an arm that never declared
    `params.llm_callsite` hashes exactly as it did before the key existed."""
    spec = ArmSpec(
        name="momentum_sleeve",
        slot="u",
        ranker="momentum_sleeve",
        params={"top_n": 8},
        registered_at="2026-06-01",
    )
    assert spec.arm_id == FILED_ID
    assert "llm_callsite" not in spec.spec["params"]


def test_the_pins_cover_every_control_the_slots_declare() -> None:
    """A slot that gains a control without a pin here is a slot whose new
    arm can re-id silently; the table above is complete or this fails."""
    for slot in ("u", "r", "m", "s"):
        declared = {c.arm_id for c in get_slot(slot).control_arms}
        assert declared == set(CONTROL_IDS[slot]), (slot, declared)
