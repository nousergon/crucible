"""v1 carry-over completeness: does every thing v1 SERVES have a v2 disposition?

Tracker: `alpha-engine-config-I10716` (the class fix, G5, of `-I10714`), then
`-I10960` / `-I10961` / `-I10962` / `-I10963` / `-I10964`, which are that same
defect one dimension along each.

**The defect this closes.** Completeness was defined over the register being
BUILT, never over the system being REPLACED. Every surface asked "does each
thing we have a record of have a verdict?" and none asked "does each thing v1
actually serves have a record?". Phase 1's gate counted verdicts for every
REGISTERED arm, so an arm that was never registered could not appear on any
surface — by construction. That is how v1's serving M model, a whole U slot
(`scanner_cut`) and an R challenger failed to cross into v2 without anything
turning red.

**Why this module declares DIMENSIONS rather than carrying one arm check.**
The first fix inverted the direction of the question for ARMS only. v1 also
serves CHAMPIONS (a pointer per slot) and TUNED PARAMETERS (documents its
weekly optimizer rewrites), and a carried arm can be registered and still
never emit. Four hand-written clauses over four hand-written source lists
would be the same blindness a fourth time: the fifth dimension would be
invisible for exactly the reason the fourth was. So :data:`DIMENSIONS`
declares each dimension ONCE — its ledger section, its row identity, its
disposition vocabulary — and :data:`_SOURCES` declares the live v1 artifacts
each is enumerated from. Parsing, enumeration, and the "is there a row for
this" comparison then run uniformly over that declaration. Adding a dimension
is a row in those two tables plus its v2-side evidence, never a new clause.

**What is read, and why from v1's live artifacts rather than a document.**
A list of "v1's arms" typed into a file is the same blindness one layer
along: it is complete exactly as long as whoever typed it was. So the v1 side
is read from the artifacts v1's own pipelines write on every cycle — the zoo
leaderboard, the M and R arenas, the two scanner champion pointers, the three
tuned parameter documents — and the only hand-maintained document is the
LEDGER, which records a decision per item and is itself graded against those
artifacts.

The ledger (`strategy/current/v1_carryover.yaml`) is authored in the private
strategy tree and published into the v2 store; which v1 arms were excluded and
why, and what v1's tuned values ARE, is strategy content
(`repository-tiering-policy` test 2/3), so this public module carries only the
schema and the grading rule, and never a number.

**No allowlist, anywhere.** A tuned parameter document is flattened to its
LEAF PATHS and every one of them needs a row — including the provenance fields
v1's optimizer stamps on (`updated_at`, `assembled_by`, the diagnostics).
Those get `excluded` rows in the ledger, where the judgement is auditable. A
code-side "ignore these field names" set would be a suppression collection:
the next tuned value v1's optimizer adds would land in it by shape and never
be seen (`-I10963`; `principles.md` §2.7).

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
  access denied, a document of the wrong shape). What v1 serves is then
  unknown, and an unknown set can never grade MET.
* **UNMET** — a live v1 item has no ledger row, any row records no decision, a
  row claiming a v2 home is not confirmed by the v2 side, a `carried` arm is
  registered and has produced nothing past its slot's settle window, a
  `deferred` row's deferral condition has cleared, a `carried` parameter's
  value has been re-tuned in v1 since it was carried, or the ledger is absent
  or malformed.
* **MET** — none of the above.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import yaml

from crucible.documents import read_store_document
from crucible.keys import v1_carryover_key
from crucible.store import Store

__all__ = [
    "DIMENSIONS",
    "DISPOSITIONS",
    "LEDGER_SCHEMA_VERSION",
    "LEDGER_SCHEMA_VERSIONS_READ",
    "REFERENCE_PATTERN",
    "V1_BUCKET_VAR",
    "V1_EVIDENCE_PREFIX",
    "V1_EXECUTOR_PARAMS_KEY",
    "V1_FACTOR_WEIGHTS_KEY",
    "V1_MODEL_ARENA_KEY",
    "V1_PARAMETER_KEYS",
    "V1_PRODUCER_ARENA_KEY",
    "V1_PRODUCER_CHAMPION_KEY",
    "V1_SCANNER_CUT_KEY",
    "V1_SCANNER_PARAMS_KEY",
    "V1_SCANNER_SPEC_KEY",
    "V1_SLOT_TO_V2_SLOT",
    "V1_S_DECLARED_ARMS",
    "V1_S_SERVING_KEY",
    "V1_ZOO_CHAMPION_NAME",
    "V1_ZOO_LEADERBOARD_KEY",
    "CarryoverFindings",
    "ChampionPointerReading",
    "Dimension",
    "DimensionFindings",
    "LedgerError",
    "LedgerRow",
    "ProductionReading",
    "V1Item",
    "V1Reading",
    "V2Evidence",
    "dimension",
    "grade_carryover",
    "read_production",
    "parse_ledger",
    "read_v1_live",
    "v1_carryover_key",
]

#: Where v1's artifact store is: the data bucket setting every other v1 read
#: already resolves (`crucible.config.Settings.arctic_bucket`, which the box
#: exports from the stack's `DataBucketName` and `migrate.history` reads v1's
#: pointers from). One name for one bucket — a second variable naming the same
#: store is two settings that can disagree. NO default: a bucket name is an
#: infrastructure identifier this public repo may not carry, so unset reads
#: UNMEASURABLE, naming this variable.
V1_BUCKET_VAR = "CRUCIBLE_ARCTIC_BUCKET"

#: The version this module grades and the private tree authors.
LEDGER_SCHEMA_VERSION = "v1_carryover.v2"

#: Versions this module will READ. `v1_carryover.v1` stays readable because
#: the store holds a published v1 document (`-I10960` deliverable 1): a reader
#: that refused it would turn the clause into a parse error on the day the
#: schema bumped and say nothing about the carry-over at all. A v1 document
#: declares no champion and no parameter rows, so every live v1 champion and
#: every live v1 tuned parameter reads as UNLISTED — which is the true state
#: until the private tree republishes.
LEDGER_SCHEMA_VERSIONS_READ: tuple[str, ...] = ("v1_carryover.v1", "v1_carryover.v2")

#: v1's slots, each with the v2 slot whose register/pointer a row is checked
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

#: v1's live tuned parameter documents (`-I10963`). `config/executor_params.json`
#: is the same object the S slot's serving proof reads: one key, two readings,
#: fetched once. `config/predictor_params/assembled/latest.json` and
#: `config/scoring_weights/assembled/latest.json` are deliberately NOT here —
#: measured 2026-09-17, both read `assembled_params: {}` / `status: all_skip`,
#: so v1 holds no live tuned value in either, and a source that contributes
#: nothing but can still fail only buys UNMEASURABLE readings.
V1_SCANNER_PARAMS_KEY = "config/scanner_params.json"
V1_EXECUTOR_PARAMS_KEY = "config/executor_params.json"
V1_FACTOR_WEIGHTS_KEY = "config/factor_attractiveness_weights.json"
V1_PARAMETER_KEYS: tuple[str, ...] = (
    V1_EXECUTOR_PARAMS_KEY,
    V1_FACTOR_WEIGHTS_KEY,
    V1_SCANNER_PARAMS_KEY,
)

#: v1's S arm set. See the module docstring: v1 writes no artifact naming it.
V1_S_DECLARED_ARMS: tuple[str, ...] = ("stock_registry",)

#: v1's serving M model is the base every zoo spec overlays. It is not a
#: `spec_id` and never appears in the candidate list — which is exactly how
#: the M inventory came to exclude the only model v1 actually served.
V1_ZOO_CHAMPION_NAME = "champion-arch"


# ---------------------------------------------------------------------------
# the dimension table — declared once, graded uniformly
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Dimension:
    """One axis of "what v1 serves", and the shape its ledger rows take.

    ``target_dispositions`` are the dispositions that claim something on the
    v2 side and must therefore NAME it in ``target_field``. The v2-side check
    itself is per-dimension — a register row, a champion pointer, a declared
    home — because the evidence genuinely differs; what is shared is the
    declaration, the parse, and the completeness comparison.
    """

    name: str
    section: str
    group_field: str
    item_field: str
    dispositions: tuple[str, ...]
    target_field: str
    target_dispositions: tuple[str, ...]
    #: Dispositions that are a decision NOT to carry, and so must carry a
    #: reason and the ruling or issue that made it.
    excluding_dispositions: tuple[str, ...]
    #: Dispositions recording no decision yet. Always UNMET.
    undecided_dispositions: tuple[str, ...]
    #: Dispositions that record a decision to wait, and must name both what
    #: they are waiting on (`reason`, `reference`) and the v2 arm the wait is
    #: about, so the wait can be graded as a PREDICATE rather than trusted.
    deferring_dispositions: tuple[str, ...]
    #: The groups the dimension is enumerated over (v1 slots, or v1 keys).
    groups: tuple[str, ...]
    #: Extra row fields this dimension allows, beyond the common ones.
    extra_fields: frozenset[str] = frozenset()
    #: Row fields this dimension requires on EVERY row.
    required_fields: frozenset[str] = frozenset()

    @property
    def fields(self) -> frozenset[str]:
        return (
            frozenset(
                {
                    self.group_field,
                    self.item_field,
                    self.target_field,
                    "disposition",
                    "reason",
                    "reference",
                    "note",
                }
            )
            | self.extra_fields
        )


DIMENSIONS: tuple[Dimension, ...] = (
    Dimension(
        name="arms",
        section="arms",
        group_field="v1_slot",
        item_field="v1_arm",
        dispositions=("carried", "excluded", "pending"),
        target_field="v2_arm",
        target_dispositions=("carried",),
        excluding_dispositions=("excluded",),
        undecided_dispositions=("pending",),
        deferring_dispositions=(),
        groups=tuple(V1_SLOT_TO_V2_SLOT),
    ),
    Dimension(
        name="champions",
        section="champions",
        group_field="v1_slot",
        item_field="v1_champion",
        # `imported`   the v2 pointer holds this v1 champion's port
        # `superseded` the v2 pointer holds a DIFFERENT arm by a recorded
        #              decision, and the row names it — a demotion WITH a
        #              reason, which is what `-I10962` found missing
        # `excluded`   v2 does not serve this v1 decision at all
        # `deferred`   the import has not run yet; the row names the arm it
        #              will point at and what clears the wait
        dispositions=("imported", "superseded", "excluded", "deferred"),
        target_field="v2_arm",
        target_dispositions=("imported", "superseded", "deferred"),
        excluding_dispositions=("excluded",),
        undecided_dispositions=(),
        deferring_dispositions=("deferred", "superseded"),
        groups=tuple(V1_SLOT_TO_V2_SLOT),
        extra_fields=frozenset({"v2_slot"}),
    ),
    Dimension(
        name="parameters",
        section="parameters",
        group_field="v1_key",
        item_field="v1_param",
        dispositions=("carried", "excluded", "deferred"),
        target_field="v2_home",
        target_dispositions=("carried",),
        excluding_dispositions=("excluded",),
        undecided_dispositions=(),
        deferring_dispositions=("deferred",),
        groups=V1_PARAMETER_KEYS,
        extra_fields=frozenset({"v1_value"}),
        required_fields=frozenset({"v1_value"}),
    ),
)


def dimension(name: str) -> Dimension:
    for declared in DIMENSIONS:
        if declared.name == name:
            return declared
    raise KeyError(
        f"unknown carry-over dimension {name!r}; declared: {[d.name for d in DIMENSIONS]}"
    )


#: Every disposition any dimension declares, for a reader that wants the
#: vocabulary without walking the table.
DISPOSITIONS: tuple[str, ...] = tuple(sorted({d for dim in DIMENSIONS for d in dim.dispositions}))


class LedgerError(ValueError):
    """The ledger is present and does not satisfy its schema."""


@dataclass(frozen=True)
class LedgerRow:
    """One recorded decision about one thing v1 serves."""

    dimension: str
    group: str
    item: str
    disposition: str
    target: str | None = None
    v1_value: str | None = None
    reason: str | None = None
    reference: str | None = None

    @property
    def label(self) -> str:
        return f"{self.group}/{self.item}"

    # -- the arm dimension's original names, so a reader of the arm half does
    #    not have to learn a second vocabulary for the same row.
    @property
    def v1_slot(self) -> str:
        return self.group

    @property
    def v1_arm(self) -> str:
        return self.item

    @property
    def v2_arm(self) -> str | None:
        return self.target


def _text(row: Mapping[str, Any], name: str, where: str) -> str:
    value = row.get(name)
    if not isinstance(value, str) or not value.strip():
        raise LedgerError(f"{where}: `{name}` must be a non-empty string, got {value!r}")
    return value.strip()


def _parse_section(
    dim: Dimension, entries: Any, *, source: str, required: bool
) -> tuple[LedgerRow, ...]:
    if entries is None and not required:
        return ()
    if not isinstance(entries, list) or not entries:
        raise LedgerError(f"{source}: `{dim.section}` must be a non-empty list")
    rows: list[LedgerRow] = []
    seen: set[tuple[str, str]] = set()
    for index, entry in enumerate(entries):
        where = f"{source} {dim.section}[{index}]"
        if not isinstance(entry, dict):
            raise LedgerError(f"{where} is {type(entry).__name__}, not a mapping")
        unknown = sorted(set(entry) - dim.fields)
        if unknown:
            raise LedgerError(f"{where}: unknown field(s) {unknown}")
        group = _text(entry, dim.group_field, where)
        if group not in dim.groups:
            raise LedgerError(
                f"{where}: {dim.group_field} {group!r} is not one of {sorted(dim.groups)}"
            )
        item = _text(entry, dim.item_field, where)
        disposition = _text(entry, "disposition", where)
        if disposition not in dim.dispositions:
            raise LedgerError(
                f"{where}: disposition {disposition!r} is not one of {list(dim.dispositions)}"
            )
        if (group, item) in seen:
            raise LedgerError(f"{where}: {group}/{item} has more than one row")
        seen.add((group, item))
        for name in sorted(dim.required_fields):
            _text(entry, name, where)
        reference = entry.get("reference")
        if reference is not None:
            reference = _text(entry, "reference", where)
            if not REFERENCE_PATTERN.match(reference):
                raise LedgerError(
                    f"{where}: reference {reference!r} does not match {REFERENCE_PATTERN.pattern}"
                )
        target = reason = None
        if disposition in dim.target_dispositions:
            target = _text(entry, dim.target_field, where)
        elif entry.get(dim.target_field) is not None:
            raise LedgerError(
                f"{where}: `{dim.target_field}` is only meaningful on a row whose disposition "
                f"claims a v2 home ({list(dim.target_dispositions)})"
            )
        needs_reason = disposition in dim.excluding_dispositions or (
            disposition in dim.deferring_dispositions
        )
        if needs_reason:
            reason = _text(entry, "reason", where)
        if needs_reason or disposition in dim.undecided_dispositions:
            if reference is None:
                raise LedgerError(
                    f"{where}: a {disposition} row must name its ruling or issue in "
                    "`reference` — a decision to wait, or not to carry, with no owner is an "
                    "untracked gap wearing a verdict"
                )
        if dim.name == "champions" and entry.get("v2_slot") is not None:
            expected = V1_SLOT_TO_V2_SLOT[group]
            declared = _text(entry, "v2_slot", where)
            if declared != expected:
                raise LedgerError(
                    f"{where}: v2_slot {declared!r} is not {expected!r}, the v2 slot {group!r} "
                    "maps to. It is DERIVED; the field is readability, never a second "
                    "declaration that can disagree"
                )
        rows.append(
            LedgerRow(
                dimension=dim.name,
                group=group,
                item=item,
                disposition=disposition,
                target=target,
                v1_value=entry.get("v1_value"),
                reason=reason,
                reference=reference,
            )
        )
    return tuple(rows)


def parse_ledger(raw: bytes, *, source: str) -> tuple[LedgerRow, ...]:
    """Every row of every declared section, or :class:`LedgerError` naming the
    first violation. Strict: an unknown field, a duplicate identity, or a
    disposition missing the field it requires is refused, never defaulted.

    A `v1_carryover.v1` document carries the `arms` section only; the sections
    added at v2 parse as empty, which reads as "no champion and no parameter
    has a disposition yet" rather than as a parse failure.
    """
    try:
        document = yaml.safe_load(raw.decode("utf-8"))
    except Exception as exc:
        raise LedgerError(f"{source} is not readable YAML: {type(exc).__name__}: {exc}") from exc
    if not isinstance(document, dict):
        raise LedgerError(f"{source} parsed to {type(document).__name__}, not a mapping")
    version = document.get("schema_version")
    if version not in LEDGER_SCHEMA_VERSIONS_READ:
        raise LedgerError(
            f"{source}: schema_version is {version!r}, expected one of "
            f"{list(LEDGER_SCHEMA_VERSIONS_READ)}"
        )
    unknown_sections = sorted(
        set(document) - {"schema_version"} - {dim.section for dim in DIMENSIONS}
    )
    if unknown_sections:
        raise LedgerError(f"{source}: unknown top-level section(s) {unknown_sections}")
    rows: list[LedgerRow] = []
    for dim in DIMENSIONS:
        rows.extend(
            _parse_section(
                dim,
                document.get(dim.section),
                source=source,
                # `arms` is required in every version; the sections added at
                # v2 are required only of a v2 document.
                required=dim.section == "arms" or version == LEDGER_SCHEMA_VERSION,
            )
        )
    return tuple(rows)


# ---------------------------------------------------------------------------
# the v1 side — read live, never from the ledger
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class V1Item:
    """One thing v1 serves: an arm, a champion, or a tuned parameter."""

    dimension: str
    group: str
    name: str
    #: The live value, for a dimension that has one (parameters). Rendered
    #: canonically so a ledger row can be compared with it as text.
    value: str | None = None

    @property
    def label(self) -> str:
        return f"{self.group}/{self.name}"


@dataclass(frozen=True)
class V1Reading:
    """What v1 serves, per dimension, or the reasons it could not be read.
    ``problems`` non-empty means the reading is UNKNOWN, not partial."""

    items: tuple[V1Item, ...]
    problems: tuple[str, ...]
    evidence: tuple[str, ...]

    def for_dimension(self, name: str) -> tuple[V1Item, ...]:
        return tuple(item for item in self.items if item.dimension == name)


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
        names.add(V1_ZOO_CHAMPION_NAME)
    return names


def _zoo_champion(key: str, document: dict[str, Any]) -> set[str]:
    """v1's M CHAMPION, read from the zoo leaderboard.

    M's champion is not a standalone pointer document like R's and U's: it is
    a FIELD of the leaderboard, and the leaderboard's own `champion` key is a
    metrics block, not a name. `champion_arch` is the serving architecture and
    `serving_champion.served_version` the exact model it resolves to; the arm
    NAME v2 disposes of is the architecture, and the version is lineage the
    import carries (`crucible.migrate`). Both fields are required to be
    present: a leaderboard missing either is a document of the wrong shape,
    not a system with no champion.
    """
    if "champion_arch" not in document or "serving_champion" not in document:
        raise _ShapeError(
            f"{key}: `champion_arch` and `serving_champion` must both be present; a "
            "leaderboard missing either is the wrong shape, not a slot with no champion"
        )
    if document["champion_arch"] is None:
        return set()
    serving = document["serving_champion"]
    if not isinstance(serving, dict) or not isinstance(serving.get("served_version"), str):
        raise _ShapeError(
            f"{key}: `serving_champion.served_version` is not a string, so the model v1 "
            "actually serves cannot be named"
        )
    return {V1_ZOO_CHAMPION_NAME}


def _champion_name(key: str, document: dict[str, Any]) -> set[str]:
    return {_champion(key, document)}


def _model_arena(key: str, document: dict[str, Any]) -> set[str]:
    return _arm_id_names(key, document.get("active_arms"), "active_arms")


def _producer_arena(key: str, document: dict[str, Any]) -> set[str]:
    return _arm_id_names(key, document.get("active_arms"), "active_arms")


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


def _leaf_params(key: str, document: dict[str, Any]) -> dict[str, str]:
    """Every LEAF PATH of a tuned parameter document, with its live value.

    Flattened over nested mappings with `.`, so
    `factor_attractiveness_weights.json`'s `weights.momentum` is one parameter
    rather than one opaque blob. NOTHING is filtered — see the module
    docstring: a code-side ignore list is where the next tuned value would
    land unseen.
    """
    flat: dict[str, str] = {}

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict) and value:
            for name, child in value.items():
                if not isinstance(name, str):
                    raise _ShapeError(f"{key}: `{prefix}` has a non-string field name {name!r}")
                walk(f"{prefix}.{name}" if prefix else name, child)
            return
        flat[prefix] = json.dumps(value, sort_keys=True)

    if not document:
        raise _ShapeError(f"{key}: the document is empty, so v1's tuned values are unknown")
    walk("", document)
    return flat


@dataclass(frozen=True)
class _LiveSource:
    """One v1 artifact, the dimension it feeds, and the group it feeds it as."""

    dimension: str
    group: str
    key: str
    extract: Callable[[str, dict[str, Any]], Any]


#: Every v1 source, declared once per (dimension, group). One key may appear
#: more than once: `config/executor_params.json` is BOTH the S slot's serving
#: proof and a tuned parameter document, and is fetched once per reading.
_SOURCES: tuple[_LiveSource, ...] = (
    # arms
    _LiveSource("arms", "model_zoo", V1_ZOO_LEADERBOARD_KEY, _zoo_leaderboard),
    _LiveSource("arms", "model_zoo", V1_MODEL_ARENA_KEY, _model_arena),
    _LiveSource("arms", "producer", V1_PRODUCER_ARENA_KEY, _producer_arena),
    _LiveSource("arms", "producer", V1_PRODUCER_CHAMPION_KEY, _champion_name),
    _LiveSource("arms", "scanner_spec", V1_SCANNER_SPEC_KEY, _scanner_spec),
    _LiveSource("arms", "scanner_cut", V1_SCANNER_CUT_KEY, _scanner_cut),
    _LiveSource("arms", "strategy", V1_S_SERVING_KEY, _s_serving),
    # champions — one per v1 slot, read from the pointer v1 actually serves
    _LiveSource("champions", "model_zoo", V1_ZOO_LEADERBOARD_KEY, _zoo_champion),
    _LiveSource("champions", "producer", V1_PRODUCER_CHAMPION_KEY, _champion_name),
    _LiveSource("champions", "scanner_spec", V1_SCANNER_SPEC_KEY, _champion_name),
    _LiveSource("champions", "scanner_cut", V1_SCANNER_CUT_KEY, _champion_name),
    _LiveSource("champions", "strategy", V1_S_SERVING_KEY, _s_serving),
    # tuned parameters — the group IS the v1 key
    _LiveSource("parameters", V1_EXECUTOR_PARAMS_KEY, V1_EXECUTOR_PARAMS_KEY, _leaf_params),
    _LiveSource("parameters", V1_FACTOR_WEIGHTS_KEY, V1_FACTOR_WEIGHTS_KEY, _leaf_params),
    _LiveSource("parameters", V1_SCANNER_PARAMS_KEY, V1_SCANNER_PARAMS_KEY, _leaf_params),
)


def _read_v1_document(v1_store: Store, key: str, problems: list[str]) -> dict[str, Any] | None:
    try:
        read = read_store_document(v1_store, key)
    except Exception as exc:  # the documented reader should not raise; if it does, it failed
        problems.append(f"v1 {key} could not be read: {type(exc).__name__}: {exc}")
        return None
    if read.problem is not None:
        problems.append(f"v1 {read.problem}")
        return None
    if read.absent or read.document is None:
        problems.append(
            f"v1 {key} is absent, so what v1 serves from it is unknown — an unknown set "
            "cannot be compared against the ledger"
        )
        return None
    return read.document


def read_v1_live(v1_store: Store) -> V1Reading:
    """What v1 serves, on every dimension :data:`DIMENSIONS` declares.

    Never raises for a read or shape failure — each becomes a named problem,
    and any problem makes the whole reading unknown (UNMEASURABLE upstream).
    Each distinct key is fetched once and handed to every source declared
    against it.
    """
    problems: list[str] = []
    evidence: list[str] = []
    documents: dict[str, dict[str, Any] | None] = {}
    items: list[V1Item] = []
    for source in _SOURCES:
        if source.key not in documents:
            evidence.append(f"{V1_EVIDENCE_PREFIX}{source.key}")
            documents[source.key] = _read_v1_document(v1_store, source.key, problems)
        document = documents[source.key]
        if document is None:
            continue
        try:
            extracted = source.extract(source.key, document)
        except _ShapeError as exc:
            problems.append(f"v1 {exc}")
            continue
        if isinstance(extracted, dict):
            items.extend(
                V1Item(source.dimension, source.group, name, value)
                for name, value in extracted.items()
            )
        else:
            items.extend(V1Item(source.dimension, source.group, name) for name in extracted)
    # Two sources can name the same arm for the same slot; the identity is
    # (dimension, group, name), and the first reading of it wins.
    unique: dict[tuple[str, str, str], V1Item] = {}
    for item in items:
        unique.setdefault((item.dimension, item.group, item.name), item)
    return V1Reading(tuple(unique.values()), tuple(problems), tuple(evidence))


# ---------------------------------------------------------------------------
# the v2 side — supplied by the caller, which owns the store reads
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProductionReading:
    """Whether one v2 arm has ever PRODUCED, and when it was registered.

    ``produced is None`` means the production source could not be read: the
    arm's output is then UNKNOWN, never absent (`-I10964`). ``registered_on``
    is the register's own date for the arm, which is what the settle window is
    measured from — an arm registered inside it has not failed to produce, it
    has not been asked.
    """

    produced: bool | None
    registered_on: str | None
    key: str
    problem: str | None = None


def read_production(
    store: Store, arm_id: str, registered_on: str | None = None
) -> ProductionReading:
    """Has ``arm_id`` ever PRODUCED anything, and when was it registered?

    Read from the two prefixes an arm's own output lands under — the dated
    `experiments/{arm}/` tree every slot's `experiment.run` writes
    (`crucible.keys.experiments_prefix`) and `arm_predictions/{arm}/`, where an
    M arm's cross-section goes. Read from the arm's ARTIFACTS, never from the
    register: the register is what said "registered" in the first place, and
    grading production against it is the `-I10714` defect one more time
    (`-I10964`).

    The one production predicate in the codebase, because two callers ask it —
    the phase-3 clause, and the migration's admission rule, which must not
    point a slot at an arm that emits nothing (`-I10961`). A listing that
    cannot be run makes production UNKNOWN, never absent.
    """
    import itertools  # noqa: PLC0415 - one call site

    from crucible.keys import (  # noqa: PLC0415 - cycle: keys is light, this stays local
        ARM_PREDICTIONS_PREFIX,
        arm_key_segment,
        experiments_prefix,
    )

    prefixes = (experiments_prefix(arm_id), f"{ARM_PREDICTIONS_PREFIX}{arm_key_segment(arm_id)}/")
    where = " or ".join(prefixes)
    for prefix in prefixes:
        try:
            found = list(itertools.islice(store.list_keys(prefix), 1))
        except Exception as exc:
            return ProductionReading(
                None,
                registered_on,
                where,
                f"{prefix} could not be listed: {type(exc).__name__}: {exc}. That is a "
                "statement about our access, not about the system being measured",
            )
        if found:
            return ProductionReading(True, registered_on, where)
    return ProductionReading(False, registered_on, where)


@dataclass(frozen=True)
class ChampionPointerReading:
    """What `champions/{slot}/current.json` holds: the arm NAME, or None when
    the pointer is absent. ``problem`` non-None means it could not be read."""

    arm_name: str | None
    key: str
    problem: str | None = None


@dataclass(frozen=True)
class V2Evidence:
    """Everything the v2 store says, read by the caller and passed in whole.

    One object rather than five mappings because the three dimensions are
    graded in one pass, and a caller that supplied four of five would be
    grading a dimension against evidence it never read.
    """

    #: v2 slot -> ACTIVE registered arm NAMES, or None when the register does
    #: not exist (an absent register holds no arm).
    registers: Mapping[str, frozenset[str] | None]
    register_keys: Mapping[str, str]
    champions: Mapping[str, ChampionPointerReading]
    #: (v2 slot, arm name) -> what production says about it.
    production: Mapping[tuple[str, str], ProductionReading]
    #: The trading day the reading is taken on, and the settle window in
    #: trading sessions (the slot's own horizon constant).
    trading_day: dt.date
    settle_window_sessions: int


# ---------------------------------------------------------------------------
# grading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DimensionFindings:
    """Findings for ONE dimension. Every list is a reason the clause is UNMET;
    a dimension with no findings is met."""

    dimension: str
    #: Live v1 items with no ledger row at all.
    unlisted: tuple[str, ...] = ()
    #: Rows recording no decision yet.
    undecided: tuple[LedgerRow, ...] = ()
    #: `(row, why)` for a row whose claim the v2 side does not confirm.
    unsatisfied: tuple[tuple[LedgerRow, str], ...] = ()
    #: `(row, why)` for a carried arm that is registered and has produced
    #: nothing past its slot's settle window.
    unproduced: tuple[tuple[LedgerRow, str], ...] = ()
    #: `(row, why)` for a row whose recorded v1 value is no longer v1's live one.
    drifted: tuple[tuple[LedgerRow, str], ...] = ()
    counts: dict[str, int] = field(default_factory=dict)
    n_live: int = 0

    @property
    def met(self) -> bool:
        return not (
            self.unlisted or self.undecided or self.unsatisfied or self.unproduced or self.drifted
        )

    @property
    def conditions(self) -> tuple[str, ...]:
        """Which of the five conditions fired, named. An operator reading
        UNMET must not have to diff the store to learn whether an item is
        missing, mute, or merely undecided (`-I10964` deliverable 3)."""
        fired = []
        if self.unlisted:
            fired.append("unlisted")
        if self.undecided:
            fired.append("undecided")
        if self.unsatisfied:
            fired.append("unsatisfied")
        if self.unproduced:
            fired.append("carried-unproduced")
        if self.drifted:
            fired.append("value-drifted")
        return tuple(fired)


@dataclass(frozen=True)
class CarryoverFindings:
    dimensions: tuple[DimensionFindings, ...] = ()

    def for_dimension(self, name: str) -> DimensionFindings:
        for found in self.dimensions:
            if found.dimension == name:
                return found
        raise KeyError(name)

    @property
    def met(self) -> bool:
        return all(found.met for found in self.dimensions)

    @property
    def n_live(self) -> int:
        return sum(found.n_live for found in self.dimensions)


def _by_label(row: LedgerRow) -> str:
    return row.label


def _counts(dim: Dimension, rows: Iterable[LedgerRow]) -> dict[str, int]:
    rows = tuple(rows)
    return {d: sum(1 for r in rows if r.disposition == d) for d in dim.dispositions}


def _unlisted(live: tuple[V1Item, ...], rows: tuple[LedgerRow, ...]) -> tuple[str, ...]:
    listed = {(r.group, r.item) for r in rows}
    return tuple(sorted(item.label for item in live if (item.group, item.name) not in listed))


def _register_satisfies(row: LedgerRow, v2: V2Evidence) -> str | None:
    v2_slot = V1_SLOT_TO_V2_SLOT[row.group]
    names = v2.registers.get(v2_slot)
    key = v2.register_keys.get(v2_slot, f"arms/{v2_slot}/register.jsonl")
    if names is None:
        return f"{key} does not exist"
    if row.target not in names:
        return f"not an active arm in {key}"
    return None


def _production_of(row: LedgerRow, v2: V2Evidence) -> tuple[ProductionReading | None, str]:
    v2_slot = V1_SLOT_TO_V2_SLOT[row.group]
    return v2.production.get((v2_slot, row.target or "")), v2_slot


def _past_settle_window(reading: ProductionReading, v2: V2Evidence) -> tuple[bool, str]:
    """Has ``reading``'s arm been registered long enough to have been asked?"""
    from crucible.calendar import count_trading_days  # noqa: PLC0415 - light, one call site

    if reading.registered_on is None:
        return False, "the register carries no date for it, so its settle window is unknown"
    try:
        registered = dt.date.fromisoformat(reading.registered_on)
    except ValueError:
        return False, f"the register's date {reading.registered_on!r} is not a date"
    sessions = count_trading_days(registered, v2.trading_day)
    if sessions < v2.settle_window_sessions:
        return (
            False,
            f"registered {reading.registered_on}: {sessions} of the slot's "
            f"{v2.settle_window_sessions}-session settle window have passed, so it has not "
            "failed to produce — it has not been asked",
        )
    return True, f"registered {reading.registered_on}, {sessions} sessions ago"


def _grade_arms(
    dim: Dimension, live: tuple[V1Item, ...], rows: tuple[LedgerRow, ...], v2: V2Evidence
) -> DimensionFindings:
    undecided = tuple(
        sorted((r for r in rows if r.disposition in dim.undecided_dispositions), key=_by_label)
    )
    unsatisfied: list[tuple[LedgerRow, str]] = []
    unproduced: list[tuple[LedgerRow, str]] = []
    for row in sorted((r for r in rows if r.disposition == "carried"), key=_by_label):
        why = _register_satisfies(row, v2)
        if why is not None:
            # Registration is the FIRST bar. An arm that never registered is
            # reported once, here, and never a second time as mute (`-I10964`).
            unsatisfied.append((row, why))
            continue
        reading, v2_slot = _production_of(row, v2)
        if reading is None or reading.produced is None:
            detail = reading.problem if reading is not None else "no production reading was taken"
            unsatisfied.append(
                (
                    row,
                    f"whether {v2_slot}:{row.target} has produced could not be read "
                    f"({detail}) — unknown output is never a pass",
                )
            )
            continue
        if reading.produced:
            continue
        past, why_window = _past_settle_window(reading, v2)
        if past:
            unproduced.append(
                (
                    row,
                    f"{v2_slot}:{row.target} is registered and has produced nothing under "
                    f"{reading.key} ({why_window}). REGISTERED IS NOT PRODUCING: a carried "
                    "arm that can never emit carries nothing",
                )
            )
    return DimensionFindings(
        dimension=dim.name,
        unlisted=_unlisted(live, rows),
        undecided=undecided,
        unsatisfied=tuple(unsatisfied),
        unproduced=tuple(unproduced),
        counts=_counts(dim, rows),
        n_live=len(live),
    )


def _grade_champions(
    dim: Dimension, live: tuple[V1Item, ...], rows: tuple[LedgerRow, ...], v2: V2Evidence
) -> DimensionFindings:
    unsatisfied: list[tuple[LedgerRow, str]] = []
    for row in sorted(rows, key=_by_label):
        if row.disposition in dim.excluding_dispositions:
            continue
        v2_slot = V1_SLOT_TO_V2_SLOT[row.group]
        pointer = v2.champions.get(v2_slot)
        if pointer is None or pointer.problem is not None:
            detail = pointer.problem if pointer is not None else "no pointer reading was taken"
            unsatisfied.append((row, f"champions/{v2_slot}/current.json: {detail}"))
            continue
        if row.disposition == "deferred":
            # A deferral is a PREDICATE, not a default argument: it holds only
            # while the arm it names cannot take the seat. The moment that arm
            # is registered AND producing, the import is outstanding work and
            # this goes red (`-I10961`). Blocked on producing, never merely on
            # registering — otherwise the seat is filled by an arm that emits
            # nothing.
            if _register_satisfies(row, v2) is not None:
                # The slot's arms have not arrived: an arm with no register row
                # has no arm id, so no artifact can exist under one. That is a
                # definite not-produced, not an unknown, and the deferral holds
                # for exactly the reason it was written.
                continue
            reading, _ = _production_of(row, v2)
            if reading is None or reading.produced is None:
                detail = (
                    reading.problem if reading is not None else "no production reading was taken"
                )
                unsatisfied.append(
                    (
                        row,
                        f"the deferral names {v2_slot}:{row.target}, and whether it has "
                        f"produced could not be read ({detail}) — unknown is never a pass",
                    )
                )
            elif reading.produced and pointer.arm_name is None:
                unsatisfied.append(
                    (
                        row,
                        f"the deferral has CLEARED: {v2_slot}:{row.target} is registered and "
                        f"producing, and {pointer.key} is still absent. v1's champion lineage "
                        f"is lost the moment slot {v2_slot} promotes on its own evidence — "
                        f"run `crucible migrate.history` ({row.reference})",
                    )
                )
            continue
        # imported / superseded: the pointer must exist and name the row's arm.
        if pointer.arm_name is None:
            unsatisfied.append((row, f"{pointer.key} is absent, so no v2 pointer holds it"))
        elif pointer.arm_name != row.target:
            unsatisfied.append(
                (
                    row,
                    f"{pointer.key} points at {v2_slot}:{pointer.arm_name}, not "
                    f"{v2_slot}:{row.target}",
                )
            )
    return DimensionFindings(
        dimension=dim.name,
        unlisted=_unlisted(live, rows),
        unsatisfied=tuple(unsatisfied),
        counts=_counts(dim, rows),
        n_live=len(live),
    )


def _grade_parameters(
    dim: Dimension, live: tuple[V1Item, ...], rows: tuple[LedgerRow, ...], v2: V2Evidence
) -> DimensionFindings:
    del v2  # a tuned parameter's v2 home is a declaration, not a store object
    live_values = {(item.group, item.name): item.value for item in live}
    drifted: list[tuple[LedgerRow, str]] = []
    for row in sorted((r for r in rows if r.disposition == "carried"), key=_by_label):
        actual = live_values.get((row.group, row.item))
        if actual is None:
            # The row names a parameter v1 no longer serves. That is not a
            # failed carry, and completeness runs the other way round anyway.
            continue
        if row.v1_value != actual:
            drifted.append(
                (
                    row,
                    f"the ledger carries {row.v1_value!r} to {row.target!r}; v1 now serves "
                    f"{actual!r}. A carried value v1 has since re-tuned is a carry that no "
                    "longer holds",
                )
            )
    return DimensionFindings(
        dimension=dim.name,
        unlisted=_unlisted(live, rows),
        drifted=tuple(drifted),
        counts=_counts(dim, rows),
        n_live=len(live),
    )


_GRADERS: dict[str, Callable[..., DimensionFindings]] = {
    "arms": _grade_arms,
    "champions": _grade_champions,
    "parameters": _grade_parameters,
}


def grade_carryover(
    reading: V1Reading, rows: Iterable[LedgerRow], v2: V2Evidence
) -> CarryoverFindings:
    """Compare what v1 serves, on every declared dimension, with the ledger
    and with what the v2 store shows."""
    rows = tuple(rows)
    return CarryoverFindings(
        tuple(
            _GRADERS[dim.name](
                dim,
                reading.for_dimension(dim.name),
                tuple(r for r in rows if r.dimension == dim.name),
                v2,
            )
            for dim in DIMENSIONS
        )
    )
