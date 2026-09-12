"""The slot registry: four slots, one engine.

Normative source: plan §4.4; binding source `champion-challenger-policy.md`.

| Slot | Decision | Champion feeds |
|---|---|---|
| **U** | which names reach the predictor | `universe/{trading_day}/members.json` |
| **R** | how names are scored into signals | `signals/{trading_day}/signals.json` |
| **M** | which trained recipe emits `predicted_alpha` | `predictions/{trading_day}.json` |
| **S** | exit / risk rules | `strategies/current` |

**This module configures the arena; it never re-implements it.** The ladder,
the paired windows, the anytime-valid confidence sequence, the Condorcet
ranking, the pointer decision and the cap-with-grace retirement rule all live
in `nousergon_lib.arena` and are CALLED. A slot that re-implements policy
§§3–6 is a defect (policy §10), and `arena_config_for` returning the
library's own `ArenaConfig` — rather than a look-alike — is what keeps that
honest: a crucible-local copy would pass every value assertion and diverge
silently the first time the library gained a field.

**The promotion bar is the library's, configured here per slot**
(`alpha-engine-config-I9763`, `-I10504`, `-I10547`). Two parameters say when a
challenger may take the pointer, and both live on `ArenaConfig`:
`promote_min_weeks` (Brian's ruling 2026-09-01: 4 paired weeks by default) and
`promote_evidence` (the anytime-valid sequence by default). Crucible declares
their per-slot VALUES the same way it declares `cap`, `grace_weeks` and
`alpha` — as fields fed into :attr:`SlotSpec.arena` — and never re-implements
the rule that reads them. That is the whole of `-I10504`'s invariant: what it
forbade was a SECOND place holding the value while the library used its own
default, which is what made the two able to drift. A field passed straight
into the config cannot: `spec.promote_min_weeks is spec.arena.promote_min_weeks`
by construction, and `tests/test_slots.py` asserts it for every slot.

**U serves on a point-estimate lead at 2 paired weeks** — Brian's ruling
2026-09-12 (`alpha-engine-config-I10546`): every scanner challenger is
promotable, and `universe_cut` promotes the point-estimate leader after 2
paired weeks. R, M and S keep the 4-week anytime-valid bar. The asymmetry is
deliberate: a universe cut is re-decided weekly and is cheap to reverse, and
requiring anytime-valid support of a cut's edge promoted nothing for months.

**Strategy content is not here.** An arm is an immutable recipe living in the
private config repository, loaded at runtime; its id is the hash of its spec.
This module holds the slot *shape* only, which is why it is publishable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import ModuleType
from typing import Literal

from nousergon_lib.arena import ArmRegister
from nousergon_lib.arena.engine import (
    EVIDENCE_ANYTIME_VALID,
    EVIDENCE_POINT,
    ArenaConfig,
)

__all__ = [
    "COST_MODELS",
    "EVIDENCE_ANYTIME_VALID",
    "EVIDENCE_POINT",
    "ESTIMATOR_KINDS",
    "EXIT_RULES",
    "PORTFOLIO_PARAM_FIELDS",
    "REQUIRED_ARM_FIELDS",
    "REQUIRED_RECIPE_FIELDS",
    "SLOTS",
    "UNITS_SUFFIXES",
    "ControlArm",
    "EstimatorSpec",
    "SlotRecipes",
    "SlotSpec",
    "arena_config_for",
    "arm_name",
    "dispatchable_slots",
    "get_slot",
    "is_control_arm",
    "load_arm_specs",
    "load_portfolio_params",
    "load_model_recipes",
    "load_strategy_recipes",
    "promotable_arms",
]

#: §10.1. Two controls per slot, every cycle: one with a KNOWN injected edge
#: and one that is pure noise. If the grader does not rank planted >
#: real-or-null > null with the expected margin, the grader is broken and the
#: cycle's verdicts are void. This is the only way to know the harness itself
#: works — the audit's central finding was a grading loop that ran for months
#: while measuring nothing.
ControlKind = Literal["planted", "null"]
_CONTROL_KINDS: tuple[str, ...] = ("planted", "null")


@dataclass(frozen=True)
class ControlArm:
    """A synthetic arm scored beside the real ones and never served.

    ``control`` is a field rather than an implied property so it appears
    explicitly in the register row, where the exclusion rules read it. A
    control arm whose flag was forgotten is a look-ahead arm sitting in the
    promotion pool, so the constructor refuses ``control=False`` outright.
    """

    arm_id: str
    kind: ControlKind
    control: bool = True

    def __post_init__(self) -> None:
        if self.kind not in _CONTROL_KINDS:
            raise ValueError(
                f"control kind must be one of {_CONTROL_KINDS}; got {self.kind!r}. "
                "A control that is neither a planted edge nor pure noise cannot "
                "establish that the grader ranks them in the expected order."
            )
        if not self.control:
            raise ValueError(
                f"control arm {self.arm_id!r} must carry control=True. The flag is "
                "what excludes it from the pointer and from the cap; a planted-edge "
                "arm without it is a look-ahead arm eligible for promotion."
            )


@dataclass(frozen=True)
class SlotSpec:
    """One slot's shape and its arena parameters.

    Frozen: this is read in many places and written in one, and a mutable
    spec is a per-slot parameter that can differ between two readers inside
    the same process.
    """

    slot: str
    slot_kind: str
    benchmark: str
    #: The `crucible.slots.<module>` that carries this slot's `produce` and
    #: `grade` entry points once it has them. A declared fact about where
    #: the code lives, read by :func:`dispatchable_slots`; it says nothing
    #: about whether the entry points EXIST yet -- that is read off the
    #: module itself, never asserted here.
    module: str
    #: §4.4: cap 5 (a RETIREMENT criterion, never an admission gate), grace
    #: 4 weeks, floor 3 active arms, retired arms scored 8 trailing cycles.
    cap: int = 5
    grace_weeks: int = 4
    min_active_arms: int = 3
    retired_trailing_cycles: int = 8
    diff_clip: float = 0.05
    alpha: float = 0.05
    #: §5.0 / Brian ruling 2026-09-01: paired WEEKS a challenger must share
    #: with the incumbent before it may take the pointer. Declared here and
    #: enforced by the library — `crucible.promote` carries no age rule of
    #: its own (`alpha-engine-config-I10547`).
    promote_min_weeks: int = 4
    #: The evidence a lead must clear to SERVE: the anytime-valid confidence
    #: sequence, or the point estimate. Brian ruling 2026-09-12
    #: (`alpha-engine-config-I10546`) puts the U slot on `point`.
    promote_evidence: str = EVIDENCE_ANYTIME_VALID
    control_arms: tuple[ControlArm, ...] = field(default_factory=tuple)

    @property
    def arena(self) -> ArenaConfig:
        """The library config for this slot.

        Constructed on demand rather than stored, so the library's own
        validation (`ArenaConfigError` on a selection slot benchmarked
        against SPY, on `min_active_arms > cap`, and so on) runs against
        these values every time they are read.
        """
        return ArenaConfig(
            slot=self.slot,
            slot_kind=self.slot_kind,
            benchmark=self.benchmark,
            alpha=self.alpha,
            diff_clip=self.diff_clip,
            cap=self.cap,
            grace_weeks=self.grace_weeks,
            min_active_arms=self.min_active_arms,
            retired_trailing_cycles=self.retired_trailing_cycles,
            promote_min_weeks=self.promote_min_weeks,
            promote_evidence=self.promote_evidence,
        )


def _controls(slot: str) -> tuple[ControlArm, ...]:
    return (
        ControlArm(arm_id=f"control_planted_{slot}", kind="planted"),
        ControlArm(arm_id=f"control_null_{slot}", kind="null"),
    )


SLOTS: dict[str, SlotSpec] = {
    "u": SlotSpec(
        slot="u",
        slot_kind="universe_cut",
        module="universe",
        # A selection stage is graded against the population it drew from,
        # count-matched. Never SPY.
        benchmark="population",
        # Brian ruling 2026-09-12, `alpha-engine-config-I10546`/`-I10547`:
        # the universe cut promotes the point-estimate leader after 2 paired
        # weeks. The other three slots keep the 4-week anytime-valid bar.
        promote_min_weeks=2,
        promote_evidence=EVIDENCE_POINT,
        control_arms=_controls("u"),
    ),
    "r": SlotSpec(
        slot="r",
        slot_kind="selection_producer",
        module="research",
        benchmark="population",
        control_arms=_controls("r"),
    ),
    "m": SlotSpec(
        slot="m",
        slot_kind="model",
        module="model",
        # CPCV OOS IC on canonical 21 trading-day labels; the population is
        # the scored cross-section, not an index.
        benchmark="population",
        control_arms=_controls("m"),
    ),
    "s": SlotSpec(
        slot="s",
        slot_kind="strategy",
        module="strategy",
        # S is not a selection stage: market-relative canonical alpha net of
        # the cost model, against SPY, is the correct axis here. The
        # population rule above must not be over-applied into a second defect.
        benchmark="SPY",
        control_arms=_controls("s"),
    ),
}


def dispatchable_slots() -> dict[str, ModuleType]:
    """The slots the CLI can actually run, in `SLOTS` order.

    A slot is dispatchable when the module its spec names exposes both
    ``produce`` and ``grade`` -- the entry points `experiment.run` and
    `experiment.grade` call. Read off the modules, never listed: the plan
    brings M and S onto the CLI at phase 3 (§6 row 3), and until their
    entry points exist a weekly arc that expanded over all four slots failed
    by construction at ``experiment.run[m]`` while the phase-1 gate, deriving
    the same set, could never read MET. Measured 2026-09-04 with
    ``crucible weekly --date 2026-08-28 --run-mode replay --dry-run``.

    So the arc grows the moment a slot's entry points land, with no list to
    update and no gate clause to re-derive -- both read this.
    """
    import importlib  # noqa: PLC0415 - lazy: the submodules import from this package

    found: dict[str, ModuleType] = {}
    for slot, spec in SLOTS.items():
        module = importlib.import_module(f"crucible.slots.{spec.module}")
        if callable(getattr(module, "produce", None)) and callable(getattr(module, "grade", None)):
            found[slot] = module
    if not found:
        raise RuntimeError(
            "no slot module exposes both `produce` and `grade`; the weekly arc would "
            "run no experiment at all and report `ok`"
        )
    return found


def get_slot(slot: str) -> SlotSpec:
    """The spec for ``slot``. Raises on an unknown slot — never returns None."""
    try:
        return SLOTS[slot]
    except KeyError as exc:
        raise KeyError(
            f"unknown slot {slot!r}; the four slots are {sorted(SLOTS)}. A slot that "
            "is not registered has no benchmark and no arena config, so a cycle run "
            "for it would produce an unlabelled number."
        ) from exc


def arena_config_for(slot: str) -> ArenaConfig:
    """The `nousergon_lib.arena` config for ``slot``."""
    return get_slot(slot).arena


def arm_name(arm_id: str) -> str:
    """The NAME component of a registered arm id, or a bare name unchanged.

    `nousergon_lib.arena.derive_arm_id` returns ``{slot}:{name}:{spec_hash}``
    and forbids ``:`` inside either the slot or the name, so a registered id
    has exactly three colon-separated parts and the middle one is the name.
    A string with no colon at all is a bare name — what a recipe file and a
    :class:`ControlArm` carry before registration — and is returned as it is.

    **Anything else raises.** This function is what
    :func:`is_control_arm` and :func:`promotable_arms` bind on, so an id
    shape neither branch understands must not be quietly reported as "not a
    control": that is the F4 failure mode inverted, and a look-ahead arm
    would reach the pointer through it.
    """
    if not arm_id:
        raise ValueError("arm_id must be non-empty")
    if ":" not in arm_id:
        return arm_id
    parts = arm_id.split(":")
    if len(parts) != 3 or not all(parts):
        raise ValueError(
            f"arm_id {arm_id!r} is neither a bare name nor a "
            "'{slot}:{name}:{spec_hash}' id from nousergon_lib.arena.derive_arm_id. "
            "Refusing rather than guessing: the control-arm exclusion reads the NAME "
            "component, and an id this function cannot parse would be reported as a "
            "non-control and become eligible to serve."
        )
    return parts[1]


def is_control_arm(spec: SlotSpec, arm_id: str, register: ArmRegister | None = None) -> bool:
    """Whether ``arm_id`` is one of ``spec``'s control arms.

    **Register-backed when a register is given and carries the arm**
    (`alpha-engine-config-I9943`). `crucible.slots.arms.register_arms` now
    forwards `ArmSpec.control` onto the registered `ArmRecord.control`
    (`crucible-PR102`), so once an arm is registered the FLAG is the fact —
    read from `register.state(arm_id).record.control` — and the name match
    below is only the FALLBACK for a bare (unregistered) id or for a row
    written before the flag existed. The flag is authoritative even when it
    disagrees with the name: a filed, non-control recipe whose generated
    name happens to collide with a control's is not excluded once it is
    registered with `control=False`, and a control is excluded even if
    `spec.control_arms` is later emptied, because its own record still says
    `control=True`.

    **Matched on the NAME component when falling back, never on the whole
    string.** A :class:`ControlArm` carries the bare name
    (``control_planted_m``) because that is what
    `crucible.slots.arms.control_specs` passes to `derive_arm_id` as the
    recipe's name; the arm the register, the series and the pointer all
    speak about is ``m:control_planted_m:7e8059f49558``. An equality test
    against the literal therefore matched nothing that had ever been
    registered — the exclusion existed and could not fire
    (`alpha-engine-config-I9757`, F4).
    """
    if register is not None and arm_id in register:
        return register.state(arm_id).record.control
    return arm_name(arm_id) in {c.arm_id for c in spec.control_arms}


def promotable_arms(
    spec: SlotSpec, arm_ids: list[str], register: ArmRegister | None = None
) -> list[str]:
    """``arm_ids`` minus the slot's control arms, order preserved.

    §10.1: controls are scored every cycle and are excluded from the pointer
    and from the cap. Excluded from the pointer because the planted arm reads
    next-period returns and promoting it would be a look-ahead in production;
    excluded from the cap because the cap is a retirement criterion over
    competing arms, and two controls counting against it would mean every
    slot starts two arms into its own retirement pressure — a cap of 5 that
    behaves as 3.

    Accepts registered ids and bare names alike (:func:`arm_name`), because
    both shapes exist: a recipe is loaded by name and scored by id, and a
    filter that only understood one of them is a filter that never fired on
    the path that matters.

    ``register``, when given, makes the exclusion register-backed rather than
    name-matched — see :func:`is_control_arm` (`alpha-engine-config-I9943`).
    """
    return [a for a in arm_ids if not is_control_arm(spec, a, register)]


# ---------------------------------------------------------------------------
# The public surface for consumers who validate strategy content, never
# re-implement it (`alpha-engine-config-I9766`; plan §4.11, §9.1). Imported
# down here, after the classes above, because `crucible.slots.arms` imports
# `ControlArm` and `SlotSpec` back from this package — a top-of-file import
# would be circular.
# ---------------------------------------------------------------------------

from crucible.portfolio import (  # noqa: E402
    COST_MODELS,
    PORTFOLIO_PARAM_FIELDS,
    load_portfolio_params,
)
from crucible.slots.arms import REQUIRED_ARM_FIELDS, load_arm_specs  # noqa: E402
from crucible.slots.model import _ESTIMATORS as _MODEL_ESTIMATOR_KINDS  # noqa: E402
from crucible.slots.model import (  # noqa: E402
    REQUIRED_RECIPE_FIELDS,
    UNITS_SUFFIXES,
    EstimatorSpec,
    SlotRecipes,
    load_model_recipes,
)
from crucible.slots.strategy import EXIT_RULES, load_strategy_recipes  # noqa: E402

#: The estimator kinds `EstimatorSpec` accepts (`crucible.slots.model._ESTIMATORS`).
#: Re-exported here, rather than left private, so a consumer validating a
#: recipe file (`alpha-engine-config-I9766`) has a stable name to import
#: instead of reaching into a module-private tuple. `test_public_surface.py`
#: pins this alias to the module's own tuple by identity of *value*, so the
#: two cannot silently drift apart.
ESTIMATOR_KINDS: tuple[str, ...] = _MODEL_ESTIMATOR_KINDS
