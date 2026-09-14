"""v1 carry-over completeness: does every arm v1 serves have a v2 disposition?

Tracker: `alpha-engine-config-I10716` (the class fix, G5, of `-I10714`).

**The defect this closes.** Nothing compared v2's arm registers to v1's live
arm set. Phase 1's gate counted verdicts for every REGISTERED arm, so an arm
that was never registered could not appear on any surface — by construction.
That is how v1's serving M model, a whole U slot (`scanner_cut`) and an R
challenger failed to cross into v2 without anything turning red.

**What is read, and why from v1's live artifacts rather than a document.**
A list of "v1's arms" typed into a file is the same blindness one layer
along: it is complete exactly as long as whoever typed it was. So the v1 side
is read from the artifacts v1's own pipelines write on every cycle — the zoo
leaderboard, the M and R arenas, the two scanner champion pointers — and the
only hand-maintained document is the LEDGER, which records a decision per arm
and is itself graded against those artifacts.

The ledger (`strategy/current/v1_carryover.yaml`) is authored in the private
strategy tree and published into the v2 store; which v1 arms were excluded
and why is strategy content (`repository-tiering-policy` test 2), so this
public module carries only its schema and the grading rule.

**The S slot is the one exception, and it is stated rather than hidden.** v1
publishes no artifact naming its strategy arms: its S "slot" is one exit-rule
chain, `executor.strategies.contract.stock_registry` in the public
`crucible-executor`, and the entry-point group that would discover a second
(`load_entry_point_rules`) is not called from v1's live path. So the S arm set
is :data:`V1_S_DECLARED_ARMS`, and the live read for S is the serving
parameter document that proves the chain is running — absent or unreadable,
the reading is UNMEASURABLE like every other v1 source.

Three readings (`crucible.gate.Clause`):

* **UNMEASURABLE** — any v1 source cannot be read (unset location, absent key,
  access denied, a document of the wrong shape). v1's arm set is then
  unknown, and an unknown set can never grade MET.
* **UNMET** — a live v1 arm has no ledger row, any row is `pending`, or a
  `carried` row names an arm absent from its v2 slot's register.
* **MET** — none of the above.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import yaml

from crucible.documents import read_store_document
from crucible.keys import v1_carryover_key
from crucible.store import Store

__all__ = [
    "DISPOSITIONS",
    "LEDGER_SCHEMA_VERSION",
    "REFERENCE_PATTERN",
    "V1_EVIDENCE_PREFIX",
    "V1_SLOT_TO_V2_SLOT",
    "V1_STORE_URI_VAR",
    "V1_S_DECLARED_ARMS",
    "CarryoverFindings",
    "LedgerError",
    "LedgerRow",
    "V1ArmSet",
    "V1Reading",
    "grade_carryover",
    "parse_ledger",
    "read_v1_arm_sets",
    "v1_carryover_key",
]

#: Where v1's artifact store is. An environment variable with NO default: a
#: bucket name is an infrastructure identifier this public repo may not carry
#: (AGENTS.md, "Visibility"), and a silent default would grade whatever store
#: it happened to point at. Unset reads UNMEASURABLE, naming this variable.
V1_STORE_URI_VAR = "CRUCIBLE_V1_STORE_URI"

LEDGER_SCHEMA_VERSION = "v1_carryover.v1"

DISPOSITIONS = ("carried", "excluded", "pending")

#: v1's slots, each with the v2 slot whose register a `carried` row is checked
#: against. v1 ran TWO universe slots; both map to v2's single U slot.
V1_SLOT_TO_V2_SLOT: dict[str, str] = {
    "model_zoo": "m",
    "producer": "r",
    "scanner_spec": "u",
    "scanner_cut": "u",
    "strategy": "s",
}

#: A ruling or issue reference, the tracker's own spelling.
REFERENCE_PATTERN = re.compile(r"^alpha-engine-config-I[1-9][0-9]*$")

#: Evidence keys for v1 artifacts carry this prefix. The gate records a
#: clause's evidence keys as lineage by reading them from the V2 store; a bare
#: v1 key could name a different object there, so the prefix keeps the two
#: namespaces from ever colliding (an unreadable evidence key is skipped).
V1_EVIDENCE_PREFIX = "v1:"

V1_ZOO_LEADERBOARD_KEY = "predictor/model_zoo/leaderboard/latest.json"
V1_MODEL_ARENA_KEY = "arena/model/latest.json"
V1_PRODUCER_ARENA_KEY = "arena/producer/latest.json"
V1_PRODUCER_CHAMPION_KEY = "config/producer_champion.json"
V1_SCANNER_SPEC_KEY = "config/scanner_spec_champion.json"
V1_SCANNER_CUT_KEY = "config/scanner_cut_champion.json"
V1_S_SERVING_KEY = "config/executor_params.json"

#: v1's S arm set. See the module docstring: v1 writes no artifact naming it.
V1_S_DECLARED_ARMS: tuple[str, ...] = ("stock_registry",)

_ROW_FIELDS = frozenset(
    {"v1_slot", "v1_arm", "disposition", "v2_arm", "reason", "reference", "note"}
)


class LedgerError(ValueError):
    """The ledger is present and does not satisfy its schema."""


@dataclass(frozen=True)
class LedgerRow:
    v1_slot: str
    v1_arm: str
    disposition: str
    v2_arm: str | None = None
    reason: str | None = None
    reference: str | None = None

    @property
    def label(self) -> str:
        return f"{self.v1_slot}/{self.v1_arm}"


def _text(row: Mapping[str, Any], name: str, where: str) -> str:
    value = row.get(name)
    if not isinstance(value, str) or not value.strip():
        raise LedgerError(f"{where}: `{name}` must be a non-empty string, got {value!r}")
    return value.strip()


def parse_ledger(raw: bytes, *, source: str) -> tuple[LedgerRow, ...]:
    """Every row of the ledger, or :class:`LedgerError` naming the first
    violation. Strict: an unknown field, a duplicate `(v1_slot, v1_arm)`, or a
    disposition missing the field it requires is refused, never defaulted."""
    try:
        document = yaml.safe_load(raw.decode("utf-8"))
    except Exception as exc:
        raise LedgerError(f"{source} is not readable YAML: {type(exc).__name__}: {exc}") from exc
    if not isinstance(document, dict):
        raise LedgerError(f"{source} parsed to {type(document).__name__}, not a mapping")
    if document.get("schema_version") != LEDGER_SCHEMA_VERSION:
        raise LedgerError(
            f"{source}: schema_version is {document.get('schema_version')!r}, "
            f"expected {LEDGER_SCHEMA_VERSION!r}"
        )
    arms = document.get("arms")
    if not isinstance(arms, list) or not arms:
        raise LedgerError(f"{source}: `arms` must be a non-empty list")
    rows: list[LedgerRow] = []
    seen: set[tuple[str, str]] = set()
    for index, entry in enumerate(arms):
        where = f"{source} arms[{index}]"
        if not isinstance(entry, dict):
            raise LedgerError(f"{where} is {type(entry).__name__}, not a mapping")
        unknown = sorted(set(entry) - _ROW_FIELDS)
        if unknown:
            raise LedgerError(f"{where}: unknown field(s) {unknown}")
        v1_slot = _text(entry, "v1_slot", where)
        if v1_slot not in V1_SLOT_TO_V2_SLOT:
            raise LedgerError(
                f"{where}: v1_slot {v1_slot!r} is not one of {sorted(V1_SLOT_TO_V2_SLOT)}"
            )
        v1_arm = _text(entry, "v1_arm", where)
        disposition = _text(entry, "disposition", where)
        if disposition not in DISPOSITIONS:
            raise LedgerError(
                f"{where}: disposition {disposition!r} is not one of {list(DISPOSITIONS)}"
            )
        if (v1_slot, v1_arm) in seen:
            raise LedgerError(f"{where}: {v1_slot}/{v1_arm} has more than one row")
        seen.add((v1_slot, v1_arm))
        reference = entry.get("reference")
        if reference is not None:
            reference = _text(entry, "reference", where)
            if not REFERENCE_PATTERN.match(reference):
                raise LedgerError(
                    f"{where}: reference {reference!r} does not match {REFERENCE_PATTERN.pattern}"
                )
        v2_arm = reason = None
        if disposition == "carried":
            v2_arm = _text(entry, "v2_arm", where)
        elif disposition == "excluded":
            reason = _text(entry, "reason", where)
            if reference is None:
                raise LedgerError(
                    f"{where}: an excluded row must name its ruling or issue in `reference`"
                )
        else:
            if reference is None:
                raise LedgerError(
                    f"{where}: a pending row must name the issue carrying it in `reference`"
                )
        if disposition != "carried" and entry.get("v2_arm") is not None:
            raise LedgerError(f"{where}: `v2_arm` is only meaningful on a carried row")
        rows.append(LedgerRow(v1_slot, v1_arm, disposition, v2_arm, reason, reference))
    return tuple(rows)


@dataclass(frozen=True)
class V1ArmSet:
    v1_slot: str
    arms: frozenset[str]


@dataclass(frozen=True)
class V1Reading:
    """v1's live arm set per slot, or the reasons it could not be read.
    ``problems`` non-empty means the set is UNKNOWN, not partial."""

    arm_sets: tuple[V1ArmSet, ...]
    problems: tuple[str, ...]
    evidence: tuple[str, ...]


class _ShapeError(ValueError):
    pass


def _arm_id_names(key: str, values: Any, path: str) -> set[str]:
    from crucible.gate import arm_name  # noqa: PLC0415 - crucible.gate imports this module

    if not isinstance(values, list):
        raise _ShapeError(f"{key}: `{path}` is {type(values).__name__}, not a list")
    names: set[str] = set()
    for value in values:
        if not isinstance(value, str) or value.count(":") != 2:
            raise _ShapeError(f"{key}: `{path}` holds {value!r}, not a `slot:name:hash` arm id")
        names.add(arm_name(value))
    return names


def _spec_ids(key: str, values: Any, path: str) -> set[str]:
    if not isinstance(values, list) or not values:
        raise _ShapeError(f"{key}: `{path}` must be a non-empty list")
    names: set[str] = set()
    for value in values:
        spec_id = value.get("spec_id") if isinstance(value, dict) else None
        if not isinstance(spec_id, str) or not spec_id:
            raise _ShapeError(f"{key}: `{path}` holds an entry with no string `spec_id`: {value!r}")
        names.add(spec_id)
    return names


def _champion(key: str, document: dict[str, Any]) -> str:
    champion = document.get("champion")
    if not isinstance(champion, str) or not champion:
        raise _ShapeError(f"{key}: `champion` is {champion!r}, not an arm name")
    return champion


def _arm_map_names(key: str, document: dict[str, Any]) -> set[str]:
    arms = document.get("arms")
    if not isinstance(arms, dict) or not arms:
        raise _ShapeError(f"{key}: `arms` must be a non-empty mapping of arm name to block")
    return set(arms)


def _zoo_leaderboard(key: str, document: dict[str, Any]) -> set[str]:
    names = _spec_ids(key, document.get("candidates"), "candidates")
    register = document.get("slot_register")
    if not isinstance(register, dict):
        raise _ShapeError(f"{key}: `slot_register` is {type(register).__name__}, not a mapping")
    names |= _spec_ids(key, register.get("arms"), "slot_register.arms")
    if "champion_arch" not in document:
        raise _ShapeError(f"{key}: `champion_arch` is missing")
    if document["champion_arch"] is not None:
        # The serving base model is an arm whether or not this cycle listed it
        # as a candidate — it is the one v1 actually serves.
        names.add("champion-arch")
    return names


def _model_arena(key: str, document: dict[str, Any]) -> set[str]:
    return _arm_id_names(key, document.get("active_arms"), "active_arms")


def _producer_arena(key: str, document: dict[str, Any]) -> set[str]:
    return _arm_id_names(key, document.get("active_arms"), "active_arms")


def _producer_champion(key: str, document: dict[str, Any]) -> set[str]:
    return {_champion(key, document)}


def _scanner_spec(key: str, document: dict[str, Any]) -> set[str]:
    return _arm_map_names(key, document) | {_champion(key, document)}


def _scanner_cut(key: str, document: dict[str, Any]) -> set[str]:
    names = _arm_map_names(key, document) | {_champion(key, document)}
    arena = document.get("arena")
    if not isinstance(arena, dict):
        raise _ShapeError(f"{key}: `arena` is {type(arena).__name__}, not a mapping")
    return names | _arm_id_names(key, arena.get("active_arms"), "arena.active_arms")


def _s_serving(key: str, document: dict[str, Any]) -> set[str]:
    del key, document  # presence and readability are the reading; see the module docstring
    return set(V1_S_DECLARED_ARMS)


#: Every v1 source, in slot order. Each extractor raises `_ShapeError` rather
#: than returning a partial set.
_SOURCES: tuple[tuple[str, str, Any], ...] = (
    ("model_zoo", V1_ZOO_LEADERBOARD_KEY, _zoo_leaderboard),
    ("model_zoo", V1_MODEL_ARENA_KEY, _model_arena),
    ("producer", V1_PRODUCER_ARENA_KEY, _producer_arena),
    ("producer", V1_PRODUCER_CHAMPION_KEY, _producer_champion),
    ("scanner_spec", V1_SCANNER_SPEC_KEY, _scanner_spec),
    ("scanner_cut", V1_SCANNER_CUT_KEY, _scanner_cut),
    ("strategy", V1_S_SERVING_KEY, _s_serving),
)


def read_v1_arm_sets(v1_store: Store) -> V1Reading:
    """v1's live arm set for every slot in :data:`V1_SLOT_TO_V2_SLOT`.

    Never raises for a read or shape failure — each becomes a named problem,
    and any problem makes the whole set unknown (UNMEASURABLE upstream).
    """
    collected: dict[str, set[str]] = {slot: set() for slot in V1_SLOT_TO_V2_SLOT}
    problems: list[str] = []
    evidence: list[str] = []
    for v1_slot, key, extract in _SOURCES:
        evidence.append(f"{V1_EVIDENCE_PREFIX}{key}")
        try:
            read = read_store_document(v1_store, key)
        except (
            Exception
        ) as exc:  # the documented reader should not raise; if it does, it is a failed read
            problems.append(f"v1 {key} could not be read: {type(exc).__name__}: {exc}")
            continue
        if read.problem is not None:
            problems.append(f"v1 {read.problem}")
            continue
        if read.absent or read.document is None:
            problems.append(
                f"v1 {key} is absent, so v1's {v1_slot} arm set is unknown — an unknown set "
                "cannot be compared against the ledger"
            )
            continue
        try:
            collected[v1_slot] |= extract(key, read.document)
        except _ShapeError as exc:
            problems.append(f"v1 {exc}")
    arm_sets = tuple(V1ArmSet(slot, frozenset(names)) for slot, names in collected.items())
    return V1Reading(arm_sets, tuple(problems), tuple(evidence))


@dataclass(frozen=True)
class CarryoverFindings:
    unlisted: tuple[str, ...] = ()
    pending: tuple[LedgerRow, ...] = ()
    #: `(row, reason)` for every carried row whose v2 arm is not registered.
    carried_unregistered: tuple[tuple[LedgerRow, str], ...] = ()
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def met(self) -> bool:
        return not (self.unlisted or self.pending or self.carried_unregistered)


def grade_carryover(
    arm_sets: Iterable[V1ArmSet],
    rows: Iterable[LedgerRow],
    registers: Mapping[str, frozenset[str] | None],
    register_keys: Mapping[str, str],
) -> CarryoverFindings:
    """Compare v1's live arm set to the ledger and the v2 registers.

    ``registers`` maps a v2 slot to its ACTIVE registered arm NAMES, or None
    when that slot's register does not exist in the store (every carried row
    in the slot is then unregistered — an absent register holds no arm).
    """
    rows = tuple(rows)
    by_pair = {(r.v1_slot, r.v1_arm): r for r in rows}
    unlisted = sorted(
        f"{s.v1_slot}/{arm}" for s in arm_sets for arm in s.arms if (s.v1_slot, arm) not in by_pair
    )
    pending = tuple(sorted((r for r in rows if r.disposition == "pending"), key=lambda r: r.label))
    carried_unregistered: list[tuple[LedgerRow, str]] = []
    for row in sorted((r for r in rows if r.disposition == "carried"), key=lambda r: r.label):
        v2_slot = V1_SLOT_TO_V2_SLOT[row.v1_slot]
        names = registers.get(v2_slot)
        key = register_keys.get(v2_slot, f"arms/{v2_slot}/register.jsonl")
        if names is None:
            carried_unregistered.append((row, f"{key} does not exist"))
        elif row.v2_arm not in names:
            carried_unregistered.append((row, f"not an active arm in {key}"))
    counts = {d: sum(1 for r in rows if r.disposition == d) for d in DISPOSITIONS}
    return CarryoverFindings(tuple(unlisted), pending, tuple(carried_unregistered), counts)
