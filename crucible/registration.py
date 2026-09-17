"""The registrable recipe set for one slot, derived once.

`experiment.new --arm <name>` registers ONE named recipe, and that flag is
required on purpose (`alpha-engine-config-I10696`): "registering whichever
arms happen to be on disk is not a deliberate act". Nothing about that
changes here.

What changes is WHERE the deliberate act sits. `experiment.register` does not
register whatever a laptop's working tree happens to contain — it registers
exactly what **the release in force declares**, read from the strategy tree
synced into the store under `crucible.keys.strategy_arms_prefix`. Pinning a
release is the deliberate act; the registrable set is then DERIVED from it,
the same way `crucible.slots.dispatchable_slots` derives the dispatchable
slots from the modules rather than from a list somebody keeps current. A
hand-written list of arms or slots is the defect both derivations exist to
prevent: it is written from what someone remembered, and it goes stale in the
direction of registering fewer of them.

Measured 2026-09-16 (`alpha-engine-config-I10927`): ten recipes merged into
`alpha-engine-config@main:strategy/arms/` had never been registered — four U
rankers, three M heads, both S recipes — because `experiment.new` is not an
arc stage and is scheduled nowhere, so the path from "a recipe is merged" to
"an arm is scored" ran only when an operator typed it and nothing reported
that nobody had.

**One loader for three recipe schemas.** U and R recipes are `ArmSpec`s, an M
recipe is a `ModelRecipe` and an S recipe is a `StrategyRecipe`; each slot
carries its own loader and its own wrapper whose id is the hash of its own
spec. This module is the one place that dispatch lives, so `experiment.new`,
`experiment.register` and the gate clause that grades the gap cannot disagree
about what "a registrable recipe" means.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from crucible.slots.arms import load_arm_specs
from crucible.slots.inputs import InputRefusal

__all__ = ["RecipeLoad", "load_registrable_recipes"]


@dataclass(frozen=True)
class RecipeLoad:
    """What a slot's recipe tree resolves to: what registers, what does not.

    A refusal is carried as a VALUE, never thrown
    (`crucible.slots.inputs.InputRefusal`): an exception is slot-wide by
    construction, so one unbuildable arm would take every sibling in the
    directory down with it. `sota_directional_combine` is the live instance —
    refused at registration by design (`alpha-engine-config-I10695`), which is
    a ruled outcome rather than an error, and the three producible M heads
    beside it must still register.
    """

    slot: str
    #: The wrappers `crucible.slots.arms.register_arms` folds into a register.
    #: Each carries its own `arm_id` — the hash of its own spec.
    specs: tuple[Any, ...]
    refusals: tuple[InputRefusal, ...]
    #: `MetricRecord`-shaped rows, one per refusal, for the manifest of
    #: whatever job loaded the slot. Empty for U and R, which have no
    #: registration-time refusal concept.
    refusal_metrics: tuple[dict[str, Any], ...]

    @property
    def names(self) -> tuple[str, ...]:
        """Recipe names that register, in load order."""
        return tuple(spec.name for spec in self.specs)

    @property
    def refused_names(self) -> tuple[str, ...]:
        return tuple(refusal.arm for refusal in self.refusals)

    @property
    def n_read(self) -> int:
        """Every recipe the tree yielded, registrable or refused."""
        return len(self.specs) + len(self.refusals)


def load_registrable_recipes(
    slot: str,
    *,
    strategy_dir: Path | str | None = None,
    store: Any = None,
) -> RecipeLoad:
    """Every recipe filed for ``slot``, split into what registers and what does not.

    Reads a checkout when ``strategy_dir`` is configured and the synced store
    tree otherwise — the same two sources every other recipe reader uses, and
    the one a spot instance has.

    Raises rather than returning an empty load: a slot whose recipe tree is
    absent, unreadable or entirely unservable is a broken release, and an
    empty result would be indistinguishable from a slot whose arms were all
    already registered (AGENTS.md rule 5).
    """
    directory = Path(strategy_dir) / "arms" / slot if strategy_dir else None
    if slot == "m":
        from crucible.slots.model import (  # noqa: PLC0415 - heavy import, one call site
            load_model_recipes,
            registration_specs,
        )

        loaded = load_model_recipes(directory, store=None if directory is not None else store)
    elif slot == "s":
        from crucible.slots.strategy import (  # noqa: PLC0415 - heavy import, one call site
            load_strategy_slot,
            registration_specs,
        )

        loaded = load_strategy_slot(
            store=None if strategy_dir else store, strategy_dir=strategy_dir
        )
    else:
        # `load_arm_specs` takes the strategy ROOT and appends `arms/{slot}`
        # itself; `load_model_recipes` takes the leaf directory. The two are
        # resolved from one value here so a caller never has to know which.
        specs = load_arm_specs(
            slot, store=store, strategy_dir=Path(strategy_dir) if strategy_dir else None
        )
        return RecipeLoad(slot=slot, specs=tuple(specs), refusals=(), refusal_metrics=())
    return RecipeLoad(
        slot=slot,
        specs=tuple(registration_specs(loaded)),
        refusals=tuple(loaded.refused),
        refusal_metrics=tuple(loaded.refusal_metrics(slot=slot)),
    )
