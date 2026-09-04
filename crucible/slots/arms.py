"""Loading arm recipes, and persisting the append-only register.

Normative source: plan §4.4; binding source `champion-challenger-policy.md`
§3, §3.1, §4.

**An arm is a file, and its id is the hash of its spec.** The recipe lives in
`alpha-engine-config/strategy/arms/{slot}/{name}.yaml` — private, because
tuned values are strategy edge — and reaches a running job either through a
local checkout (`CRUCIBLE_STRATEGY_DIR`, the laptop path) or through the
strategy tree synced into the store under `strategy/current/` (the spot-box
path). Either way the bytes are the same, and
`nousergon_lib.arena.derive_arm_id` turns them into an id: an edited recipe
is a NEW arm carrying `supersedes`, and cannot inherit the old arm's record.

**Registration is the gate.** Policy §3: an arm writing shadow output
without a register row is a defect. :func:`register_arms` folds every loaded
recipe into the `ArmRegister` at `arms/{slot}/register.jsonl` and returns the
register the cycle runs against, so there is no path from a recipe to a
score that skips it.

**The vacuity guard runs at load.** Two arms resolving to the same ranking
callable are refused here, before either produces anything — policy §4's
`inapplicable`. Two arms whose *parameters* also match would already have
the same spec hash and therefore the same id, so the file-level duplicate is
caught by the id collision.

**Pre-registration is enforced, not requested** (§9.1). A recipe missing
`ranker`, `params`, `registered_at` or `notes` does not register; the metric,
horizon and benchmark are the SLOT's and are deliberately not settable
per-arm, because policy §4 requires every arm to be scored on the same axis.

**`registered_at` is a SESSION, asserted on construction** (§4.12). It is the
start of the arm's out-of-sample clock, and every ladder rung,
`promote_min_weeks` rung and `grace_weeks` rung is counted from it in
trading weeks — so a date that is not a session makes the arm's whole
eligibility clock start on a day the market never traded, and the error
compounds silently for the arm's entire life. Measured (`I9757`): three
recipes carrying `registered_at: '2026-08-29'`, a Saturday, loaded and
registered without complaint, because the §4.12 contract test walks store
KEYS only while the plan requires it to walk "every artifact key, manifest
field and `arena_cycle` window" — and this field is a manifest field, not a
key. The assertion lives on :class:`ArmSpec` rather than in the YAML parser
so it binds to every construction path, the migration importer's included:
a spec built in code carries the same contract as one read off disk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from nousergon_lib.arena.arms import ArmEvent, ArmRegister, derive_arm_id

from crucible.calendar import assert_trading_day
from crucible.keys import arm_register_key, strategy_arms_prefix
from crucible.slots import ControlArm, SlotSpec
from crucible.slots.rankers import get_ranker, ranker_identity
from crucible.store import Store

__all__ = [
    "CONTROL_REGISTERED_AT",
    "LLM_CALLSITE_PARAM",
    "REQUIRED_ARM_FIELDS",
    "ArmSpec",
    "InapplicableArmError",
    "control_specs",
    "load_arm_specs",
    "read_register",
    "register_arms",
    "write_register",
]

#: The controls' registration date. Fixed and early, for the reason given at
#: the `registered_at` field below. 2026-01-02 is the first NYSE session of
#: the year, so it is a legal trading-day key as well as an early one.
CONTROL_REGISTERED_AT = "2026-01-02"

#: §9.1 pre-registration. Every field is required because each one is a
#: question a verdict must be able to answer, and a recipe that leaves one
#: blank produces a verdict that cannot.
REQUIRED_ARM_FIELDS: tuple[str, ...] = ("name", "slot", "ranker", "params", "registered_at")

#: The `params` key through which an arm declares WHICH registered LLM call
#: site it reaches a model through (`alpha-engine-config-I9920`, plan §6 row 5).
#:
#: It lives inside `params` on purpose: `params` is already part of the
#: id-hashed :attr:`ArmSpec.spec`, so two arms differing only in the model they
#: call are two arms — without adding a second hashing rule — and every arm
#: registered before this key existed keeps its id, because a key that is
#: absent from `params` was never in the hash. Its value must be a key of
#: `crucible.llm.LLM_CALLSITE_REGISTRY`; :func:`_parse` REFUSES an unregistered
#: one rather than accepting and ignoring it, so "which arms are LLM arms" is
#: answerable from the register alone and is never a string heuristic over
#: ranker names. `crucible.gate.LLM_ARM_CALLSITE_FIELD` names the same key.
LLM_CALLSITE_PARAM = "llm_callsite"


class InapplicableArmError(ValueError):
    """Two arms that are not two arms. Policy §4's vacuity refusal."""


@dataclass(frozen=True)
class ArmSpec:
    """One recipe, as loaded. Immutable; its hash is its identity."""

    name: str
    slot: str
    ranker: str
    params: dict[str, Any]
    registered_at: str
    supersedes: str | None = None
    control: bool = False
    control_kind: str | None = None
    bootstrap: bool = False
    promotion_source: str = ""
    notes: str = ""
    source_key: str = ""

    def __post_init__(self) -> None:
        """§4.12: the OOS clock starts on a SESSION, or it starts nowhere.

        Raises :class:`~crucible.calendar.NonTradingDayKeyError`. A raise
        rather than a resolution to the neighbouring session: a recipe naming
        a Saturday was written by something that keyed off the wall clock,
        and quietly moving the date to Friday would hide the writer while
        producing a plausible arm.
        """
        assert_trading_day(
            self.registered_at,
            context=(
                f"arm {self.slot}:{self.name} `registered_at` "
                f"(source: {self.source_key or 'constructed in code'})"
            ),
        )

    @property
    def spec(self) -> dict[str, Any]:
        """The hashed recipe. Provenance is deliberately NOT in it.

        `notes`, `source_key` and `promotion_source` describe where a recipe
        came from, not what it computes. Hashing them would make a
        clarifying comment produce a new arm and orphan its score series —
        the opposite of what §3.1 asks for.
        """
        return {
            "slot": self.slot,
            "name": self.name,
            "ranker": self.ranker,
            "params": dict(sorted(self.params.items())),
            "control": self.control,
            "control_kind": self.control_kind,
        }

    @property
    def arm_id(self) -> str:
        return derive_arm_id(self.slot, self.name, self.spec)

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "name": self.name,
            "slot": self.slot,
            "ranker": self.ranker,
            "params": dict(self.params),
            "registered_at": self.registered_at,
            "supersedes": self.supersedes,
            "control": self.control,
            "control_kind": self.control_kind,
            "bootstrap": self.bootstrap,
            "promotion_source": self.promotion_source,
            "notes": self.notes,
            "source_key": self.source_key,
        }


def _parse(payload: bytes, origin: str) -> ArmSpec:
    document = yaml.safe_load(payload.decode("utf-8"))
    if not isinstance(document, dict):
        raise ValueError(
            f"{origin}: an arm recipe is a YAML mapping; got {type(document).__name__}"
        )
    missing = [f for f in REQUIRED_ARM_FIELDS if not document.get(f)]
    if missing:
        raise ValueError(
            f"{origin}: arm recipe is missing required field(s) {missing}. §9.1 "
            "pre-registration: an arm declares its slot, recipe and registration date "
            "before its first score, and a recipe that leaves one blank produces a "
            "verdict that cannot answer for itself. Note that `metric`, `horizon` and "
            "`benchmark` are deliberately NOT arm fields — they are the SLOT's, so "
            "every arm is scored on the same axis (policy §4)."
        )
    params = document.get("params") or {}
    if not isinstance(params, dict):
        raise ValueError(f"{origin}: `params` must be a mapping; got {type(params).__name__}")
    get_ranker(str(document["ranker"]))  # raises by name on an unknown ranker
    if LLM_CALLSITE_PARAM in params:
        _require_registered_callsite(params[LLM_CALLSITE_PARAM], origin=origin)
    return ArmSpec(
        name=str(document["name"]),
        slot=str(document["slot"]),
        ranker=str(document["ranker"]),
        params=params,
        registered_at=str(document["registered_at"]),
        supersedes=document.get("supersedes"),
        control=bool(document.get("control", False)),
        control_kind=document.get("control_kind"),
        bootstrap=bool(document.get("bootstrap", False)),
        promotion_source=str(document.get("promotion_source", "")),
        notes=str(document.get("notes", "")),
        source_key=origin,
    )


def _require_registered_callsite(value: Any, *, origin: str) -> None:
    """Refuse an arm that names a call site the registry does not carry.

    Accepting-and-ignoring is the defect: the arm would register, the gate
    would count it as a non-LLM arm, and phase 5's "every LLM arm has a
    verdict" would quantify over a set missing exactly the arm that most
    needed grading. The registry is imported here, lazily, because it is a
    heavy module with one consumer in this file.
    """
    from crucible.llm import (  # noqa: PLC0415 - one call site, heavy import
        CALLSITE_REGISTRY_PATH,
        LLM_CALLSITE_REGISTRY,
    )

    if not isinstance(value, str) or not value:
        raise ValueError(
            f"{origin}: `params.{LLM_CALLSITE_PARAM}` must be the id of a registered LLM "
            f"call site (a non-empty string); got {value!r}"
        )
    if value not in LLM_CALLSITE_REGISTRY:
        registered = ", ".join(sorted(LLM_CALLSITE_REGISTRY)) or "(none registered)"
        raise ValueError(
            f"{origin}: `params.{LLM_CALLSITE_PARAM}` names {value!r}, which is not a key "
            f"of LLM_CALLSITE_REGISTRY ({CALLSITE_REGISTRY_PATH.name}: {registered}). An "
            "arm reaches a model only through a registered call site — register the "
            "site first, in the same change as the code that calls it."
        )


def load_arm_specs(
    slot: str,
    *,
    store: Store | None = None,
    strategy_dir: Path | None = None,
) -> list[ArmSpec]:
    """Every recipe for ``slot``, from a checkout or from the synced tree.

    The checkout wins when it is configured, because a developer editing
    `alpha-engine-config/strategy/` expects the edit to take effect; a spot
    instance has no checkout and reads the store. Which one was used is
    recorded on each spec's `source_key`, so `explain` reports it.
    """
    specs: list[ArmSpec] = []
    if strategy_dir is not None:
        directory = Path(strategy_dir) / "arms" / slot
        if not directory.is_dir():
            raise FileNotFoundError(
                f"no arm directory at {directory}. The strategy tree is "
                "`alpha-engine-config/strategy/`; point CRUCIBLE_STRATEGY_DIR at it, or "
                "unset it to read the tree synced into the store."
            )
        for path in sorted(directory.glob("*.yaml")):
            specs.append(_parse(path.read_bytes(), str(path)))
    else:
        if store is None:
            raise ValueError("load_arm_specs needs either a store or a strategy_dir")
        prefix = strategy_arms_prefix(slot)
        for key in sorted(store.list_keys(prefix)):
            if key.endswith(".yaml"):
                specs.append(_parse(store.get_bytes(key), key))

    if not specs:
        raise FileNotFoundError(
            f"no arm recipes found for slot {slot!r}. A slot with no arms produces zero "
            "comparisons — the `no_promotable_challenger` defect of 2026-08-21 and "
            "2026-08-28 — so this is a refusal, not an empty result."
        )
    _assert_applicable(specs)
    return specs


def _assert_applicable(specs: list[ArmSpec]) -> None:
    """Policy §4: no two live arms share a ranking callable."""
    by_callable: dict[int, list[str]] = {}
    for spec in specs:
        by_callable.setdefault(ranker_identity(spec.ranker), []).append(spec.name)
    collisions = {ident: names for ident, names in by_callable.items() if len(names) > 1}
    if collisions:
        detail = "; ".join(sorted(", ".join(sorted(n)) for n in collisions.values()))
        raise InapplicableArmError(
            f"arms share a ranking callable and are therefore not two arms: {detail}. "
            "Policy §4 marks this `inapplicable` and refuses it at load: a comparison "
            "of a rule with itself always reads as a tie, and it consumes a slot in a "
            "pool with a cap of five."
        )
    ids: dict[str, str] = {}
    for spec in specs:
        if spec.arm_id in ids:
            raise InapplicableArmError(
                f"arms {ids[spec.arm_id]!r} and {spec.name!r} derive the same id "
                f"{spec.arm_id}: identical recipes under two file names. One of them is "
                "a copy, and the register cannot hold two rows for one recipe."
            )
        ids[spec.arm_id] = spec.name


def control_specs(slot_spec: SlotSpec) -> list[ArmSpec]:
    """The slot's two control arms, as recipes (§10 component 1).

    Generated rather than filed: a control is defined by the harness, not by
    strategy, and a planted-edge arm sitting in the private strategy tree
    where an operator could clear its `control` flag is a look-ahead arm one
    edit away from the promotion pool.
    """
    out: list[ArmSpec] = []
    for control in slot_spec.control_arms:
        assert isinstance(control, ControlArm)
        out.append(
            ArmSpec(
                name=control.arm_id,
                slot=slot_spec.slot,
                ranker="momentum_sleeve",  # unused: controls are scored directly
                params={},
                # Earlier than any cycle a control can be scored in. The
                # library derives an arm's age as elapsed weeks from
                # `created_date` and RAISES when that date is in the future,
                # so a control registered "today" would fail every replay of
                # a past Saturday — the exact runs a control exists to check.
                registered_at=CONTROL_REGISTERED_AT,
                control=True,
                control_kind=control.kind,
                notes=(
                    "Control arm, generated by the harness. Scored every cycle, "
                    "excluded from the pointer and from the cap (§10.1)."
                ),
                source_key="crucible.slots:control",
            )
        )
    return out


def read_register(store: Store, slot: str) -> ArmRegister:
    """The folded register for ``slot``. An absent log is an empty register."""
    key = arm_register_key(slot)
    if not store.exists(key):
        return ArmRegister()
    lines = store.get_bytes(key).decode("utf-8").splitlines()
    return ArmRegister.from_dicts([json.loads(line) for line in lines if line.strip()])


def write_register(store: Store, slot: str, register: ArmRegister) -> bytes:
    """Serialize the event log. Append-only in content, rewritten as one object.

    One object rather than an S3 append because S3 has no append. The log's
    append-only property is a property of the EVENTS — `ArmRegister` never
    mutates or drops one — and `tests/test_slots_arms.py` asserts that a
    rewrite is a strict prefix extension of what was there.
    """
    payload = (
        "\n".join(json.dumps(event, sort_keys=True) for event in register.to_dicts()) + "\n"
    ).encode("utf-8")
    store.put_bytes(arm_register_key(slot), payload)
    return payload


def register_arms(
    register: ArmRegister,
    specs: list[ArmSpec],
) -> tuple[ArmRegister, dict[str, ArmSpec]]:
    """Fold every recipe into ``register``, appending only what is new.

    Returns the register and a lookup from arm id to recipe. An arm already
    registered is not re-registered — that would append a second `registered`
    event for one id and break the fold — and its recipe is still returned,
    because the cycle needs the recipe of every arm it scores.
    """
    by_id: dict[str, ArmSpec] = {}
    known = set(register.all_arms())
    for spec in specs:
        by_id[spec.arm_id] = spec
        if spec.arm_id in known:
            continue
        supersedes = spec.supersedes if spec.supersedes in known else None
        if spec.supersedes and supersedes is None:
            raise ValueError(
                f"arm {spec.name!r} declares supersedes={spec.supersedes!r}, which is not "
                "in the register. A lineage pointer to nothing is worse than none: it "
                "reads as history that was checked."
            )
        register, _ = register.register(
            slot=spec.slot,
            name=spec.name,
            spec=spec.spec,
            created_date=spec.registered_at,
            supersedes=supersedes,
            bootstrap=spec.bootstrap,
            notes=spec.notes,
        )
        known.add(spec.arm_id)
    return register, by_id


def register_events(register: ArmRegister) -> tuple[ArmEvent, ...]:
    """The raw event tuple, for tests that assert append-only behaviour."""
    return register.events
