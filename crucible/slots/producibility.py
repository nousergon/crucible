"""Can the arm a champion pointer names actually produce? Asked BEFORE it serves.

`alpha-engine-config-I11085`. A champion pointer is a serving decision: the
slot's serving path resolves it and nothing else
(`crucible.slots.cycle._serve_champion_feed`), and a pointer naming an arm
that produced nothing this cycle raises there — correctly, because production
with no feed must fail loud. What was missing is every check UPSTREAM of that
raise. Measured 2026-09-19: `crucible migrate.history` had imported v1's R
champion onto `r:scanner_predictor_direct`, whose own recipe says it "refuses
BY NAME" until the M slot materialises `predicted_alpha_ratio` (track B). The
import succeeded on 2026-09-14; the first arc to resolve the pointer died at
`experiment.run[r]` five days later, and every stage after it died too.

**The question is structural, and it is answered by the release.** An arm can
produce when the recipe tree in force declares a recipe that (a) hashes to the
exact id the pointer names and (b) is not refused at registration — for U and
R by :func:`crucible.slots.cycle.partition_by_catalog` (a ranker column the
feature catalogue declares no producer for), for M and S by the slot's own
registration refusals (:func:`crucible.registration.load_registrable_recipes`).
A control arm never produces a feed at all. Those are the same predicates the
produce loop itself applies, read through the same functions, so this module
cannot disagree with `experiment.run` about which arms it will run.

What this does NOT answer: whether an arm whose recipe is registrable will
emit on a given day (an M arm whose inputs refuse at run time, a data outage).
That is a runtime fact and stays with the produce loop and
:func:`crucible.migrate.admission_refusal`. This module catches the class that
is knowable from the tree before anything runs — the class that took the
2026-09-19 arc down.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from crucible.features import PendingColumn
    from crucible.slots.inputs import InputRefusal
    from crucible.store import Store

__all__ = [
    "PendingDependency",
    "ReleaseArms",
    "UnproducibleChampionError",
    "assert_champions_producible",
    "catalog_input_refusal",
    "catalog_refusal",
    "champion_refusal",
    "pending_dependency",
    "release_arms",
    "waiting_arms",
]


class UnproducibleChampionError(RuntimeError):
    """A champion pointer names an arm the release in force cannot produce.

    Raised before anything serves, naming each slot, the arm, and why — so the
    failure is seconds into the arc with the cause in the message, rather than
    at the first stage that resolves the pointer with every later stage lost.
    """


@dataclass(frozen=True)
class ReleaseArms:
    """What a slot's recipe tree resolves to, split by producibility."""

    slot: str
    #: recipe name -> the arm id that recipe produces under.
    producible: dict[str, str]
    #: recipe name -> why it cannot produce: refused at registration, or
    #: registered and waiting on a declared pending producer
    #: (`alpha-engine-config-I11030`), which no champion can be either.
    refused: dict[str, str]


def _catalog_columns() -> list[str]:
    from crucible.features import CATALOG  # noqa: PLC0415 - heavy import, few call sites

    return [feature.name for feature in CATALOG]


def catalog_input_refusal(recipe: Any) -> InputRefusal | None:
    """The catalogue refusal of one U/R ``recipe``, or ``None`` when it can run.

    The single-recipe face of :func:`crucible.slots.cycle.partition_by_catalog`
    — the exact predicate `experiment.run` applies before producing — so a
    caller holding one recipe (`crucible.migrate`, which is handed recipes
    rather than a tree) asks the same question the produce loop will.
    ``recipe`` needs only ``name`` and ``ranker``.
    """
    from crucible.slots.cycle import partition_by_catalog  # noqa: PLC0415 - avoids a cycle

    _, refused = partition_by_catalog([recipe], catalog_columns=_catalog_columns())
    return refused[0] if refused else None


def catalog_refusal(recipe: Any) -> str | None:
    """Why a U/R ``recipe`` (an `ArmSpec`) cannot produce, or ``None``."""
    refusal = catalog_input_refusal(recipe)
    return refusal.reason if refusal else None


@dataclass(frozen=True)
class PendingDependency:
    """An arm that produces nothing BY DESIGN, and the producer it waits on.

    `alpha-engine-config-I11030`. Read off a catalogue refusal, never
    declared by the arm: every column the refusal names must be absent from
    the feature catalogue AND declared in
    :data:`crucible.features.PENDING_COLUMNS`.
    """

    arm: str
    columns: tuple[PendingColumn, ...]

    def waits_on(self) -> str:
        return "; ".join(column.describe() for column in self.columns)

    def describe(self) -> str:
        return f"{self.arm} waits on {self.waits_on()}"


def pending_dependency(
    refusal: InputRefusal | None, *, catalog_columns: Iterable[str] | None = None
) -> PendingDependency | None:
    """The declared producer a catalogue ``refusal`` waits on, or ``None``.

    ``None`` unless EVERY unresolvable column is (a) not produced by the
    catalogue and (b) declared in :data:`crucible.features.PENDING_COLUMNS`.
    One undeclared column makes the whole recipe a defect: part of what it
    reads will never exist. (a) is what makes the wait lapse. Once the
    catalogue produces the column, the refusal is gone or no longer names it,
    and the arm is held to production like any other.
    """
    from crucible.features import PENDING_COLUMNS  # noqa: PLC0415 - heavy import, few call sites

    if refusal is None:
        return None
    produced = set(_catalog_columns() if catalog_columns is None else catalog_columns)
    columns = []
    for column in refusal.unresolvable:
        if column in produced or column not in PENDING_COLUMNS:
            return None
        columns.append(PENDING_COLUMNS[column])
    return PendingDependency(arm=refusal.arm, columns=tuple(columns))


def waiting_arms(
    slot: str, *, store: Store, strategy_dir: Path | str | None = None
) -> dict[str, PendingDependency]:
    """Every arm id the release declares for ``slot`` that waits BY DESIGN.

    Keyed by the arm id the recipe hashes to, so a registered arm is waiting
    only if a recipe in force hashes to its exact id and every column that
    recipe lacks has a declared pending producer
    (:func:`pending_dependency`). An arm whose recipe was edited, removed, or
    reads a column nothing will produce is absent from the result, and is
    held to production like any other. M and S recipes are not ranked on
    catalogue columns, so their slots wait on nothing here.

    Raises when the recipe tree cannot be read. The caller classifies that.
    """
    from crucible.slots.arms import (  # noqa: PLC0415 - avoids a cycle
        FOREIGN_RECIPE_LOADERS,
        load_arm_specs,
    )

    if slot in FOREIGN_RECIPE_LOADERS:
        return {}
    waiting: dict[str, PendingDependency] = {}
    for recipe in load_arm_specs(
        slot, store=store, strategy_dir=Path(strategy_dir) if strategy_dir else None
    ):
        dependency = pending_dependency(catalog_input_refusal(recipe))
        if dependency is not None:
            waiting[recipe.arm_id] = dependency
    return waiting


def release_arms(
    slot: str,
    *,
    store: Store,
    strategy_dir: Path | str | None = None,
    trading_day: str | None = None,
) -> ReleaseArms:
    """Every recipe the release declares for ``slot``, producible or refused.

    Read through :func:`crucible.registration.load_registrable_recipes` — the
    one place the per-slot recipe dispatch lives — plus, for the `ArmSpec`
    slots (U and R), the catalogue partition `experiment.run` applies after
    loading. A slot whose every arm is refused raises
    `crucible.slots.inputs.SlotUnservableError` from its loader; that is
    caught here and rendered as "nothing producible", because for THIS
    question it is an answer rather than an error.
    """
    from crucible.registration import load_registrable_recipes  # noqa: PLC0415 - avoids a cycle
    from crucible.slots.arms import FOREIGN_RECIPE_LOADERS  # noqa: PLC0415 - avoids a cycle
    from crucible.slots.cycle import partition_by_catalog  # noqa: PLC0415 - avoids a cycle
    from crucible.slots.inputs import SlotUnservableError  # noqa: PLC0415 - avoids a cycle

    try:
        load = load_registrable_recipes(
            slot, strategy_dir=strategy_dir, store=store, today=trading_day
        )
    except SlotUnservableError as exc:
        return ReleaseArms(
            slot=slot,
            producible={},
            refused={refusal.arm: refusal.reason for refusal in exc.refusals},
        )
    specs = list(load.specs)
    refusals = list(load.refusals)
    if slot not in FOREIGN_RECIPE_LOADERS:
        specs, catalog_refused = partition_by_catalog(specs, catalog_columns=_catalog_columns())
        refusals += catalog_refused
    return ReleaseArms(
        slot=slot,
        producible={spec.name: spec.arm_id for spec in specs},
        refused={refusal.arm: refusal.reason for refusal in refusals},
    )


def champion_refusal(arm_id: str, *, release: ReleaseArms, register: Any = None) -> str | None:
    """Why the pointer's ``arm_id`` cannot produce under ``release``, or ``None``."""
    from crucible.slots import arm_name, get_slot, is_control_arm  # noqa: PLC0415 - avoids a cycle

    if is_control_arm(get_slot(release.slot), arm_id, register):
        return (
            "it is a control arm. Controls are scored every cycle and never produce a "
            "served selection (plan §10.1), so a pointer at one has no feed on any day"
        )
    name = arm_name(arm_id)
    if release.producible.get(name) == arm_id:
        return None
    if name in release.refused:
        refused_because = release.refused[name]
        return f"the release's own recipe {name!r} cannot produce: {refused_because}"
    if name in release.producible:
        return (
            f"the release's recipe {name!r} now produces under {release.producible[name]!r}, "
            f"not {arm_id!r}: the pointer names a spec no recipe in force hashes to, so "
            "nothing will ever write a shadow under it"
        )
    return (
        f"the release declares no recipe named {name!r} for slot {release.slot!r}, so "
        "nothing will ever write a shadow under it"
    )


def assert_champions_producible(
    store: Store,
    slots: Iterable[str],
    *,
    strategy_dir: Path | str | None = None,
    trading_day: str | None = None,
    context: str,
) -> None:
    """Raise :class:`UnproducibleChampionError` if any slot's pointer names an
    arm the release cannot produce. A slot with no pointer passes: a slot that
    has never promoted serves nothing and says so
    (`crucible.slots.cycle._serve_champion_feed`).

    Every failing slot is collected before raising, so one run names all of
    them rather than one per attempt; each line carries the arms that DO
    produce, which is the operator's next command.
    """
    from crucible.documents import load_store_document  # noqa: PLC0415 - avoids a cycle
    from crucible.keys import champion_key  # noqa: PLC0415 - avoids a cycle
    from crucible.slots.arms import read_register  # noqa: PLC0415 - avoids a cycle

    problems: list[str] = []
    for slot in slots:
        key = champion_key(slot)
        if not store.exists(key):
            continue
        arm_id = load_store_document(store, key).get("arm_id")
        if not arm_id:
            problems.append(f"slot {slot!r}: {key} names no arm_id")
            continue
        release = release_arms(
            slot, store=store, strategy_dir=strategy_dir, trading_day=trading_day
        )
        why = champion_refusal(str(arm_id), release=release, register=read_register(store, slot))
        if why is None:
            continue
        producible = sorted(
            candidate
            for candidate in release.producible.values()
            if champion_refusal(candidate, release=release) is None
        )
        problems.append(
            f"slot {slot!r}: {key} names {arm_id!r}, which cannot produce — {why.rstrip('.')}. "
            f"Arms this release can produce for {slot!r}: {producible or 'none'}. "
            f"Re-point with `crucible promote --slot {slot} --revert-to <arm> --reason <why>`."
        )
    if problems:
        raise UnproducibleChampionError(
            f"{context}: {len(problems)} champion pointer(s) name an arm that cannot "
            "produce, so the stage that serves the slot would fail — and every stage after "
            "it with it. Refusing before anything runs.\n"
            + "\n".join(f"  - {line}" for line in problems)
        )
