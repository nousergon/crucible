"""`alpha-engine-config-I10636`: `min_active_arms` is read by nothing, and
`ArenaCycle.active_arms` counts the two control arms — so a one-real-arm slot
reads as three against a floor of three.

`crucible.slots.cycle.min_active_arms_finding` is the fix: it compares the
floor against the caller's own `crucible.slots.promotable_arms(...)` result
(controls already excluded), never against `ArenaCycle.active_arms`, which
stays the library's own field and deliberately still counts controls.
"""

from __future__ import annotations

from nousergon_lib.arena.arms import derive_arm_id

from crucible.slots import get_slot, promotable_arms
from crucible.slots.cycle import MIN_ACTIVE_ARMS_FINDING_METRIC, min_active_arms_finding


def _registered_id(slot: str, name: str) -> str:
    """The id this arm carries once it is in the register — see `test_slots.py`."""
    return derive_arm_id(slot, name, {"name": name})


class TestMinActiveArmsFinding:
    def test_fires_for_a_one_real_arm_slot_with_controls_present(self) -> None:
        """The S slot's actual shape today: one real arm, two controls. The
        raw `active_arms` count would be 3 against a floor of 3 and read as
        healthy; the promotable count is 1 and must fire."""
        spec = get_slot("s")
        controls = [_registered_id("s", c.arm_id) for c in spec.control_arms]
        real = [_registered_id("s", "stock_registry")]
        active_arms = controls + real

        promotable = promotable_arms(spec, active_arms)
        finding = min_active_arms_finding(spec, promotable)

        assert len(active_arms) == spec.min_active_arms, (
            "the raw active_arms count must equal the floor for this test to prove "
            "the defect — it is 3-active-arms-including-controls reading as healthy "
            "that the fix must not reproduce"
        )
        assert finding["status"] == "BELOW_FLOOR"
        assert finding["promotable_arm_count"] == 1
        assert finding["min_active_arms"] == spec.min_active_arms
        assert finding["promotable_arms"] == real
        assert "floor" in finding["reason"] or "1" in finding["reason"]

    def test_does_not_fire_for_a_three_real_arm_slot_with_controls_present(self) -> None:
        """The same slot, healthy: three real arms plus the same two controls.
        The exclusion must not manufacture a false floor breach."""
        spec = get_slot("s")
        controls = [_registered_id("s", c.arm_id) for c in spec.control_arms]
        real = [_registered_id("s", f"arm_real_{i}") for i in range(spec.min_active_arms)]
        active_arms = controls + real

        promotable = promotable_arms(spec, active_arms)
        finding = min_active_arms_finding(spec, promotable)

        assert finding["status"] == "OK"
        assert finding["promotable_arm_count"] == spec.min_active_arms
        assert set(finding["promotable_arms"]) == set(real)

    def test_finding_metric_name_is_exported(self) -> None:
        assert MIN_ACTIVE_ARMS_FINDING_METRIC == "min_active_arms_finding"

    def test_finding_renders_explicitly_never_silently(self) -> None:
        """Principle 7: no data must never render as healthy. Both branches
        carry an explicit status and a human-readable reason — there is no
        third, absent state."""
        spec = get_slot("s")
        below = min_active_arms_finding(spec, [])
        ok = min_active_arms_finding(spec, [f"s:arm_{i}:hash" for i in range(spec.min_active_arms)])

        assert below["status"] == "BELOW_FLOOR"
        assert below["reason"]
        assert ok["status"] == "OK"
        assert ok["reason"]
