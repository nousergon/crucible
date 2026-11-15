"""Unit tests for `crucible.registration`: the RecipeLoad value and the U/R partition.

`load_registrable_recipes` dispatch to real recipe trees is covered by
`tests/test_registered_arm_produces.py` and `tests/test_experiment_register.py`;
these tests cover what they do not: `RecipeLoad`'s derived properties and the
catalogue partition's three outcomes, with the loader and predicates stubbed.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from crucible import registration
from crucible.registration import RecipeLoad, load_registrable_recipes
from crucible.slots import producibility
from crucible.slots.inputs import InputRefusal, SlotUnservableError


def _refusal(arm: str) -> InputRefusal:
    return InputRefusal(arm=arm, unresolvable=("col_raw",), reason="no producer")


def _spec(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name)


class TestRecipeLoad:
    def test_derived_properties(self) -> None:
        load = RecipeLoad(
            slot="u",
            specs=(_spec("a"), _spec("b")),
            refusals=(_refusal("c"),),
            refusal_metrics=(),
        )
        assert load.names == ("a", "b")
        assert load.refused_names == ("c",)
        assert load.n_read == 3
        assert load.waiting == ()

    def test_empty_load_reads_zero(self) -> None:
        load = RecipeLoad(slot="r", specs=(), refusals=(), refusal_metrics=())
        assert (load.names, load.refused_names, load.n_read) == ((), (), 0)

    def test_is_frozen(self) -> None:
        load = RecipeLoad(slot="u", specs=(), refusals=(), refusal_metrics=())
        with pytest.raises(dataclasses.FrozenInstanceError):
            load.slot = "r"  # pyright: ignore[reportAttributeAccessIssue]


class TestRankedPartition:
    def _stub(
        self,
        monkeypatch: pytest.MonkeyPatch,
        specs: list[SimpleNamespace],
        refused: dict[str, InputRefusal],
        waiting: dict[str, str],
    ) -> None:
        def fake_load(slot: str, **_: object) -> list[SimpleNamespace]:
            return specs

        def fake_refusal(spec: SimpleNamespace) -> InputRefusal | None:
            return refused.get(str(spec.name))

        def fake_dependency(refusal: InputRefusal) -> str | None:
            return waiting.get(refusal.arm)

        monkeypatch.setattr(registration, "load_arm_specs", fake_load)
        monkeypatch.setattr(producibility, "catalog_input_refusal", fake_refusal)
        monkeypatch.setattr(producibility, "pending_dependency", fake_dependency)

    def test_all_producible_registers_everything(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._stub(monkeypatch, [_spec("a"), _spec("b")], {}, {})
        load = load_registrable_recipes("u")
        assert load.slot == "u"
        assert load.names == ("a", "b")
        assert load.refusals == ()
        assert load.refusal_metrics == ()
        assert load.waiting == ()

    def test_a_pending_dependency_registers_and_is_listed_as_waiting(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._stub(
            monkeypatch,
            [_spec("a"), _spec("w")],
            {"w": _refusal("w")},
            {"w": "dependency-marker"},
        )
        load = load_registrable_recipes("r")
        assert load.names == ("a", "w")
        assert load.refusals == ()
        assert load.waiting == ("dependency-marker",)

    # Mutation guarded: letting a refusal with no pending dependency register
    # (names would then contain "bad", refusals would be empty).
    def test_an_unproducible_arm_is_refused_not_registered(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._stub(monkeypatch, [_spec("a"), _spec("bad")], {"bad": _refusal("bad")}, {})
        load = load_registrable_recipes("u")
        assert load.names == ("a",)
        assert load.refused_names == ("bad",)
        assert len(load.refusal_metrics) == 1
        assert load.n_read == 2

    # Mutation guarded: returning an empty RecipeLoad instead of raising.
    def test_nothing_registrable_raises_instead_of_returning_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._stub(monkeypatch, [_spec("bad")], {"bad": _refusal("bad")}, {})
        with pytest.raises(SlotUnservableError):
            load_registrable_recipes("u")

    def test_a_loader_failure_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(slot: str, **_: object) -> list[SimpleNamespace]:
            raise FileNotFoundError(slot)

        monkeypatch.setattr(registration, "load_arm_specs", boom)
        with pytest.raises(FileNotFoundError):
            load_registrable_recipes("u")
