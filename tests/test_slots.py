"""The slot registry: four slots, one engine, one config each.

Normative source: plan §4.4 and `champion-challenger-policy.md`.

The policy's rule (§10) is that a slot re-implementing the decision machinery
instead of calling `nousergon_lib.arena.engine.run_cycle` is a defect. These
tests hold the registry to that: the config it hands out IS the library's
`ArenaConfig`, not a parallel dataclass that happens to have the same field
names and will drift from it.

Written before `crucible/slots/__init__.py` and seen failing.
"""

from __future__ import annotations

import pytest
from nousergon_lib.arena.arms import derive_arm_id
from nousergon_lib.arena.engine import ArenaConfig, ArenaConfigError

from crucible.slots import (
    SLOTS,
    SlotSpec,
    arena_config_for,
    get_slot,
    is_control_arm,
    promotable_arms,
)


def _registered_id(slot: str, name: str) -> str:
    """The id this arm carries once it is in the register.

    `nousergon_lib.arena.derive_arm_id` is called rather than a hand-written
    string, so a change to the id format cannot leave these tests asserting
    against a shape the fleet stopped producing — which is exactly how the
    control-arm exclusion came to be tested only on inputs it never saw.
    """
    return derive_arm_id(slot, name, {"name": name})


class TestRegistry:
    def test_the_four_slots_are_registered(self) -> None:
        """FOUR, not three. The first draft folded the universe cut into
        'data'; the policy is explicit that U is its own slot with its own
        scoring path, and conflating it with the selection producer yields a
        meaningless number."""
        assert set(SLOTS) == {"u", "r", "m", "s"}

    def test_an_unknown_slot_raises_rather_than_returning_none(self) -> None:
        with pytest.raises(KeyError, match="unknown slot"):
            get_slot("x")

    @pytest.mark.parametrize("slot", ["u", "r", "m", "s"])
    def test_every_slot_yields_a_library_arena_config(self, slot: str) -> None:
        """The type assertion is the point. A crucible-local look-alike would
        pass every value test above and silently diverge the first time the
        library adds a field."""
        assert isinstance(arena_config_for(slot), ArenaConfig)


class TestConfiguredValues:
    @pytest.mark.parametrize("slot", ["u", "r", "m", "s"])
    def test_the_ruled_parameters_are_carried(self, slot: str) -> None:
        cfg = arena_config_for(slot)
        assert cfg.cap == 5
        assert cfg.grace_weeks == 4
        assert cfg.min_active_arms == 3
        assert cfg.retired_trailing_cycles == 8

    @pytest.mark.parametrize("slot", ["u", "r", "m", "s"])
    def test_promote_min_weeks_is_four(self, slot: str) -> None:
        """Brian's ruling, 2026-09-01: a new arm is promotable only after 4
        paired weeks against the incumbent — 20 paired TRADING days.

        It lives on the crucible SlotSpec rather than on ArenaConfig because
        the installed nousergon-lib ArenaConfig has no such field; the
        eligibility age is a v2 addition awaiting the policy amendment named
        in I9751."""
        assert get_slot(slot).promote_min_weeks == 4

    @pytest.mark.parametrize("slot", ["u", "r"])
    def test_a_selection_slot_is_never_benchmarked_against_spy(self, slot: str) -> None:
        """policy §4: a selection stage is graded against the POPULATION it
        drew from. Grading it against SPY inverted wins and losses on
        2026-08-17, when SPY trailed the drawn-from population by 140bp at
        21d. The library refuses the misconfiguration; this asserts crucible
        does not route around the refusal."""
        assert arena_config_for(slot).benchmark == "population"

    def test_the_library_refuses_a_selection_slot_benchmarked_on_spy(self) -> None:
        with pytest.raises(ArenaConfigError, match="population|SPY"):
            ArenaConfig(slot="r", slot_kind="selection_producer", benchmark="SPY")

    def test_s_is_benchmarked_against_spy_and_that_is_correct(self) -> None:
        """S is a strategy slot, not a selection stage: market-relative
        canonical alpha vs SPY is the right axis there. Asserted so the
        population rule above cannot be over-applied into a second defect."""
        assert arena_config_for("s").benchmark == "SPY"


class TestControlArms:
    """§10.1: a planted-edge arm and a pure-noise arm are graded beside the
    real arms every cycle. If the grader does not rank planted > real-or-null
    > null, the GRADER is broken and the cycle's verdicts are void."""

    @pytest.mark.parametrize("slot", ["u", "r", "m", "s"])
    def test_every_slot_declares_a_planted_and_a_null_control(self, slot: str) -> None:
        controls = get_slot(slot).control_arms
        assert len(controls) == 2
        assert {c.kind for c in controls} == {"planted", "null"}
        assert all(c.control for c in controls)

    def test_control_arms_are_excluded_from_the_pointer(self) -> None:
        """A control arm scored beside the real arms must never be promoted
        to serve. The planted arm looks at next-period returns; promoting it
        would be a look-ahead in production.

        **Written against the id form the register actually holds** (plan §11
        risk 1). This test previously fed BARE names — `control_planted_r`,
        `arm_real_a` — into `promotable_arms`, and passed for years against a
        filter that could not match a single registered arm: a registered id
        is `derive_arm_id`'s `{slot}:{name}:{spec_hash}`, and the exclusion
        compared it against the bare literal. A test that can only pass on a
        shape the production path never produces is the shape of a test that
        cannot fail (`alpha-engine-config-I9757`, F4).
        """
        spec = get_slot("r")
        controls = [_registered_id("r", c.arm_id) for c in spec.control_arms]
        real = [_registered_id("r", n) for n in ("arm_real_a", "arm_real_b")]
        assert all(":" in arm and len(arm.split(":")) == 3 for arm in controls + real)

        assert promotable_arms(spec, controls + real) == real

    def test_the_exclusion_matches_the_real_registered_id_not_the_literal(self) -> None:
        """The half of F4 that made the filter inert wherever it was called.

        `ControlArm.arm_id` carries the bare NAME, because that is what
        `crucible.slots.arms.control_specs` hands to `derive_arm_id` as the
        recipe's name. Everything downstream — the register, the score
        series, the pointer — speaks the hashed id. The exclusion therefore
        has to bind on the name COMPONENT.
        """
        spec = get_slot("m")
        literal = spec.control_arms[0].arm_id
        registered = _registered_id("m", literal)

        assert registered != literal
        assert registered.split(":")[1] == literal
        assert is_control_arm(spec, registered)
        assert is_control_arm(spec, literal)
        assert promotable_arms(spec, [registered]) == []

    def test_an_unparseable_arm_id_is_refused_not_reported_as_a_non_control(self) -> None:
        """Fail loud: silently answering "not a control" for an id shape the
        filter does not understand is F4 inverted — the look-ahead arm
        reaches the pointer through the case nobody handled."""
        spec = get_slot("m")
        for bad in ("m:control_planted_m", "m:control_planted_m:hash:extra", "m::hash"):
            with pytest.raises(ValueError, match="arm_id"):
                is_control_arm(spec, bad)

    def test_control_arms_do_not_consume_the_cap(self) -> None:
        """The cap of 5 is a RETIREMENT criterion over competing arms. If the
        two controls counted against it, every slot would start two arms into
        its own retirement pressure and the cap would mean 3, not 5."""
        spec = get_slot("r")
        controls = [_registered_id("r", c.arm_id) for c in spec.control_arms]
        real = [_registered_id("r", f"arm_{i}") for i in range(5)]
        assert len(promotable_arms(spec, controls + real)) == spec.arena.cap

    def test_a_control_arm_cannot_be_registered_without_the_flag(self) -> None:
        """`control: true` is what excludes it. A control arm whose flag was
        forgotten is a look-ahead arm in the promotion pool."""
        from crucible.slots import ControlArm

        with pytest.raises(ValueError, match="control"):
            ControlArm(arm_id="c", kind="planted", control=False)

    def test_an_unknown_control_kind_is_refused(self) -> None:
        from crucible.slots import ControlArm

        with pytest.raises(ValueError, match="planted|null"):
            ControlArm(arm_id="c", kind="mostly_planted")


class TestSpecIntegrity:
    def test_a_slot_spec_is_frozen(self) -> None:
        """Config is read in many places and written in one. A mutable spec
        is a per-slot parameter that can differ between two readers in the
        same process."""
        spec = get_slot("r")
        with pytest.raises(Exception):  # noqa: B017 - dataclass raises FrozenInstanceError
            spec.promote_min_weeks = 1  # type: ignore[misc]

    def test_promote_min_weeks_below_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="promote_min_weeks"):
            SlotSpec(
                slot="r",
                slot_kind="selection_producer",
                benchmark="population",
                promote_min_weeks=0,
            )
