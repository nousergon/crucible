"""Arm loading, identity, the vacuity guard and the append-only register."""

from __future__ import annotations

import pytest
from nousergon_lib.arena.arms import ArmRegister

from crucible.slots import get_slot
from crucible.slots.arms import (
    ArmSpec,
    InapplicableArmError,
    control_specs,
    load_arm_specs,
    read_register,
    register_arms,
    write_register,
)


def _write(directory, name, ranker, **fields):
    lines = [f"name: {name}", "slot: u", f"ranker: {ranker}", "registered_at: '2026-06-01'"]
    for key, value in fields.items():
        lines.append(f"{key}: {value}")
    lines.append("params:")
    lines.append("  top_n: 8")
    (directory / f"{name}.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestIdentity:
    def test_the_arm_id_is_the_hash_of_the_spec(self) -> None:
        base = ArmSpec(
            name="momentum_sleeve",
            slot="u",
            ranker="momentum_sleeve",
            params={"top_n": 8},
            registered_at="2026-06-01",
        )
        same = ArmSpec(**{**base.__dict__, "notes": "a clarifying comment"})
        edited = ArmSpec(**{**base.__dict__, "params": {"top_n": 12}})
        assert base.arm_id == same.arm_id, (
            "provenance is not part of the recipe; hashing a note would orphan the "
            "arm's score series every time someone explained it"
        )
        assert base.arm_id != edited.arm_id, (
            "§3.1: an edited recipe is a NEW arm and cannot inherit the old record"
        )
        assert base.arm_id.startswith("u:momentum_sleeve:")

    def test_a_recipe_missing_a_required_field_does_not_register(self, tmp_path) -> None:
        arms = tmp_path / "arms" / "u"
        arms.mkdir(parents=True)
        (arms / "half.yaml").write_text("name: half\nslot: u\nparams:\n  top_n: 5\n")
        with pytest.raises(ValueError, match="missing required field"):
            load_arm_specs("u", strategy_dir=tmp_path)

    def test_an_unknown_ranker_is_refused_by_name(self, tmp_path) -> None:
        arms = tmp_path / "arms" / "u"
        arms.mkdir(parents=True)
        _write(arms, "ghost", "no_such_ranker")
        with pytest.raises(KeyError, match="unknown ranker"):
            load_arm_specs("u", strategy_dir=tmp_path)

    def test_a_slot_with_no_arms_is_a_refusal_not_an_empty_list(self, tmp_path) -> None:
        (tmp_path / "arms" / "u").mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="no arm recipes"):
            load_arm_specs("u", strategy_dir=tmp_path)

    def test_an_unknown_top_level_key_is_refused_by_name(self, tmp_path) -> None:
        """`alpha-engine-config-I9944`: a top-level key `_parse` does not
        read used to register cleanly and bind nothing. `params` stays open
        — it is the ranker's own argument mapping, hashed as-is — but every
        other top-level key is a closed vocabulary."""
        arms = tmp_path / "arms" / "u"
        arms.mkdir(parents=True)
        _write(arms, "extra", "momentum_sleeve", bogus_field="nope")
        with pytest.raises(ValueError, match="bogus_field"):
            load_arm_specs("u", strategy_dir=tmp_path)


def _write_llm_arm(directory, name, callsite):
    (directory / f"{name}.yaml").write_text(
        f"name: {name}\nslot: u\nranker: momentum_sleeve\nregistered_at: '2026-06-01'\n"
        f"params:\n  top_n: 8\n  llm_callsite: {callsite}\n",
        encoding="utf-8",
    )


class TestAnArmDeclaresItsLlmCallSite:
    """alpha-engine-config-I9920: "which arms are LLM arms" is a question the
    register answers by itself, never a heuristic over ranker names."""

    def test_an_unregistered_call_site_is_refused_at_load(self, tmp_path) -> None:
        """The registry is empty until phase 5, so ANY declared site is
        unregistered today — and the arm must not register-and-be-ignored."""
        arms = tmp_path / "arms" / "u"
        arms.mkdir(parents=True)
        _write_llm_arm(arms, "thesis", "research.thesis")
        with pytest.raises(ValueError, match="not a key of LLM_CALLSITE_REGISTRY"):
            load_arm_specs("u", strategy_dir=tmp_path)

    def test_a_non_string_call_site_is_refused(self, tmp_path) -> None:
        arms = tmp_path / "arms" / "u"
        arms.mkdir(parents=True)
        _write_llm_arm(arms, "thesis", "[a, b]")
        with pytest.raises(ValueError, match="non-empty string"):
            load_arm_specs("u", strategy_dir=tmp_path)

    def test_a_registered_call_site_loads_and_is_in_the_hashed_spec(
        self, tmp_path, monkeypatch
    ) -> None:
        """The contract test: the LLM-arm set is derivable from the register's
        `spec.params` alone, with no producer state — and it is part of the
        arm's identity, so two arms differing only in the model they call are
        two arms."""
        import crucible.llm as llm

        monkeypatch.setattr(llm, "LLM_CALLSITE_REGISTRY", {"research.thesis": object()})
        arms = tmp_path / "arms" / "u"
        arms.mkdir(parents=True)
        _write_llm_arm(arms, "thesis", "research.thesis")
        (spec,) = load_arm_specs("u", strategy_dir=tmp_path)
        assert spec.spec["params"]["llm_callsite"] == "research.thesis"

        # The register carries the spec HASH, never the spec — so the join
        # from "active arm id" back to "this recipe declares that site" is the
        # id itself, which the gate reproduces by loading the synced recipes.
        register, _ = register_arms(ArmRegister(), [spec])
        (event,) = register.to_dicts()
        assert event["arm_id"] == spec.arm_id
        assert event["record"]["spec_hash"] == spec.arm_id.rsplit(":", 1)[-1]
        assert "spec" not in event

        plain = ArmSpec(**{**spec.__dict__, "params": {"top_n": 8}})
        assert plain.arm_id != spec.arm_id, (
            "the call site changes what the arm IS; it must change the id"
        )
        assert "llm_callsite" not in plain.spec["params"], (
            "an arm that never declared the key is hashed exactly as before — no re-id"
        )


class TestVacuityGuard:
    def test_two_arms_sharing_a_ranking_callable_are_refused_at_load(self, tmp_path) -> None:
        arms = tmp_path / "arms" / "u"
        arms.mkdir(parents=True)
        _write(arms, "momentum_sleeve", "momentum_sleeve")
        _write(arms, "momentum_sleeve_copy", "momentum_sleeve")
        with pytest.raises(InapplicableArmError, match="share a ranking callable"):
            load_arm_specs("u", strategy_dir=tmp_path)


class TestRegister:
    def test_registration_is_the_gate_and_is_idempotent(self, store, strategy_dir) -> None:
        specs = load_arm_specs("u", strategy_dir=strategy_dir)
        register, _ = register_arms(ArmRegister(), specs)
        write_register(store, "u", register)
        first = store.get_bytes("arms/u/register.jsonl")

        again, _ = register_arms(read_register(store, "u"), specs)
        write_register(store, "u", again)
        second = store.get_bytes("arms/u/register.jsonl")
        assert first == second, (
            "re-registering an arm already in the register must append nothing; a second "
            "`registered` event for one id breaks the fold"
        )

    def test_a_rewrite_is_a_strict_prefix_extension(self, store, strategy_dir) -> None:
        specs = load_arm_specs("u", strategy_dir=strategy_dir)
        register, _ = register_arms(ArmRegister(), specs[:2])
        write_register(store, "u", register)
        before = store.get_bytes("arms/u/register.jsonl").decode().splitlines()

        register, _ = register_arms(read_register(store, "u"), specs)
        write_register(store, "u", register)
        after = store.get_bytes("arms/u/register.jsonl").decode().splitlines()

        assert after[: len(before)] == before, "the event log may only be extended"
        assert len(after) > len(before)

    def test_supersedes_pointing_at_nothing_is_refused(self, store, strategy_dir) -> None:
        specs = load_arm_specs("u", strategy_dir=strategy_dir)
        orphan = ArmSpec(**{**specs[0].__dict__, "supersedes": "u:ghost:deadbeef"})
        with pytest.raises(ValueError, match="not in the register"):
            register_arms(ArmRegister(), [orphan])


class TestControls:
    def test_both_controls_are_generated_and_carry_the_flag(self) -> None:
        specs = control_specs(get_slot("u"))
        assert {s.control_kind for s in specs} == {"planted", "null"}
        assert all(s.control for s in specs)

    def test_controls_are_generated_not_filed_in_the_strategy_tree(self, strategy_dir) -> None:
        """A planted-edge arm in an editable tree is one edit from the pool."""
        loaded = load_arm_specs("u", strategy_dir=strategy_dir)
        assert not any(s.control for s in loaded)

    def test_a_control_registered_before_any_cycle_it_is_scored_in(self) -> None:
        from crucible.slots.arms import CONTROL_REGISTERED_AT

        for spec in control_specs(get_slot("u")):
            assert spec.registered_at == CONTROL_REGISTERED_AT


class TestServingPath:
    def test_no_module_outside_the_registry_imports_a_ranking_function(self) -> None:
        """Policy §4: the serving path resolves the pointer, never an import."""
        import pathlib

        package = pathlib.Path(__file__).resolve().parents[1] / "crucible"
        offenders = []
        for path in package.rglob("*.py"):
            if path.name == "rankers.py":
                continue
            text = path.read_text(encoding="utf-8")
            from crucible.slots import rankers as rankers_module

            private_names = sorted(
                name
                for name in vars(rankers_module)
                if name.startswith("_")
                and callable(vars(rankers_module)[name])
                and not name.startswith("__")
            )
            for private in private_names:
                if private in text:
                    offenders.append(f"{path.name}: {private}")
        assert not offenders, (
            "a module importing a ranking function directly bypasses the champion "
            f"pointer: {offenders}"
        )
