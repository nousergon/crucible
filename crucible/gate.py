"""Phase exit gates, read from the artifacts the phase produced.

Normative source: plan §6 (each phase has a numeric exit gate), §11.1, and
the `closes-when` of `alpha-engine-config-I9757`.

**A gate reads; it never runs.** Every clause below is evaluated against
manifests, cycle artifacts and pointers already in the store. That is the
whole design: a gate that ran the thing it grades could not tell a run made
under gate conditions from a run made in production, and the phase-1 gate was
closed once already — on 2026-09-01, the moment its build PRs merged — while
not one replay had been run and nine acceptance clauses were failing. A gate
that is a *measurement* cannot be satisfied by a merge.

**A clause is met, unmet, or unmeasurable, and unmeasurable is never met.**
`no data` is not a pass (principle 7). An absent artifact makes a clause unmet
with the missing key named, so the operator's next action is in the output.

**The gate's own job succeeds when the measurement succeeds.** An unmet gate
is a real result, not a broken run: `crucible gate` writes its artifact and
its manifest reads `ok`, while the PROCESS exits non-zero so a caller — CI, a
person, a sweep — cannot mistake "not there yet" for "done". Conflating the
two would make a store outage look like a failed phase.
"""

from __future__ import annotations

import ast
import datetime as dt
import io
import json
import os
import re
import shlex
from calendar import monthrange
from collections.abc import Callable, Iterable
from contextlib import redirect_stderr
from dataclasses import dataclass, field, replace
from functools import lru_cache, wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jsonschema import Draft202012Validator
from pydantic import ValidationError

from crucible.aggregation import MemberRow, member_dicts
from crucible.alerts import (
    MUTED_TOPIC_VAR,
    NON_OPERATOR_DESTINATIONS,
    PAGE_CONDITIONS,
    pages_in_range,
)
from crucible.calendar import TRADING_DAYS_PER_WEEK, is_trading_day, resolve_trading_day
from crucible.components import Component, load_registry
from crucible.config import CLOUDTRAIL_ARCHIVE_VAR
from crucible.documents import DocumentRead, read_manifests_under
from crucible.documents import read_path_document as _read_path_document
from crucible.documents import read_store_document as _read_store_document
from crucible.keys import (
    ALERTS_ROOT,
    FAULT_INJECTION_ROOT,
    acceptance_reading_key,
    arena_cycle_key,
    arm_register_key,
    champion_key,
    gate_key,
    gate_prefix,
    is_manifest_key,
    legacy_dead_lambdas_key,
    legacy_weekly_executions_key,
    manifest_prefix,
    parse_acceptance_reading,
    parse_bus_key,
    parse_fault_injection_key,
    review_key,
    review_prefix,
    runs_prefix,
    strategy_arms_prefix,
    verdict_key,
)  # noqa: F401 - re-exported
from crucible.manifest import load_schema, manifest_key
from crucible.models import (
    FaultRecordDocument,
    PhaseClosingReadingDocument,
    PhaseLadderDocument,
)
from crucible.release import POINTER_KEY
from crucible.report import attribution_key
from crucible.runner import TRANSIENT_CLASSIFIERS
from crucible.slots import SLOTS, dispatchable_slots, is_control_arm
from crucible.store import S3Store, Store
from crucible.synthetic import synthetic_routing_active
from crucible.tags import (
    TAG_KEY,
    TAG_VALUE,
    CostAllocationTagUnreadableError,
    cost_allocation_tag_status,
)
from crucible.weekly import arc_stages

if TYPE_CHECKING:
    # Annotation only: the arena package is imported lazily at the one call
    # site that folds a register (`_register_arms`), so `crucible gate
    # --help` and every unit test that imports this module stay off the
    # heavy import path.
    from nousergon_lib.arena import ArmRegister

__all__ = [
    "ACCEPTANCE_RATCHET_PATH",
    "CLAUSE_MEMBER_RANK",
    "GATE_SCHEMA_VERSION",
    "GATES",
    "LEGACY_DEAD_LAMBDAS_SCHEMA_VERSION",
    "LEGACY_DEAD_LAMBDA_NAMES",
    "LEGACY_WEEKLY_EXECUTIONS_SCHEMA_VERSION",
    "LEGACY_WEEKLY_MAX_STARTS_PER_WEEK",
    "LEGACY_WEEKLY_MIN_RUNS_PER_WEEK",
    "LEGACY_WEEKLY_RERUN_NAME_PREFIX",
    "LEGACY_WEEKLY_TOPIC_FIELD",
    "muted_alerts_topic_name",
    "V2_STORE_VERSIONING_ENABLED",
    "V2_TAG_ACCEPTANCE_CLAUSE_ID",
    "WEEKLY_RUN_DAY_GATE_SKIP_MAX_SECONDS",
    "LADDER_CONSOLE_STATE",
    "LADDER_KEY",
    "LADDER_SCHEMA_VERSION",
    "LADDER_STATES",
    "ClauseMisconfiguredError",
    "GATE_DELIVERABLES",
    "GATE_REQUIRED_ENV",
    "required_env_for_run",
    "missing_required_env",
    "LLM_ARM_CALLSITE_FIELD",
    "LLM_ARM_RECIPE_SLOTS",
    "MANIFEST_RUN_MODE_FIELD",
    "MANIFEST_RUN_MODE_LIVE",
    "PHASE0_DELIVERABLES",
    "PHASE1_DELIVERABLES",
    "PHASE2_DELIVERABLES",
    "REPOSITORY_GRADED_CLAUSES",
    "PHASE3_DELIVERABLES",
    "PHASE4_DELIVERABLES",
    "PHASE5_DELIVERABLES",
    "PHASE2_AUTONOMY_MIN_DAILY_CYCLES",
    "autonomy_daily_cycles_in_span",
    "autonomy_earliest_satisfiable_render_day",
    "PHASE2_LIVE_SATURDAYS",
    "PHASE2_WINDOW_WEEKS",
    "PHASE2_MAX_PAGES",
    "PHASE2_MAX_TAGGED_USD",
    "PHASE2_REPLAY_SATURDAYS",
    "COST_LEADING_DAYS",
    "COST_TRAILING_DAYS",
    "WEEKLY_ANCHORED_GATES",
    "weekly_window",
    "PHASE4_MAX_TOTAL_USD",
    "PHASES",
    "TRADER_EVIDENCE_KEY",
    "REVIEW_SCHEMA_VERSION",
    "SOURCE_SCAN_SCOPE",
    "Clause",
    "Deliverable",
    "DocumentRead",
    "SourceScan",
    "GateResult",
    "Ladder",
    "Phase",
    "PhaseRow",
    "arm_name",
    "build_ladder",
    "coverage_note",
    "evaluate",
    "gate_key",
    "gate_prefix",
    "ladder_payload",
    "expected_legacy_weekly_window",
    "legacy_dead_lambdas_key",
    "legacy_weekly_executions_key",
    "review_key",
    "review_prefix",
    "weekly_anchor",
    "last_read",
    "ladder_schema",
    "validate_ladder_document",
    # The closing reading (`alpha-engine-config-I9967`). Public because its
    # ENFORCER lives in another repository: `alpha-engine-config`'s
    # phase-tracker consistency sweep imports these names rather than
    # restating the contract, the way `crucible.slots`' registries are
    # imported rather than mirrored (`alpha-engine-config-I9766`).
    "CLOSING_READING_SCHEMA_VERSION",
    "CLOSING_READING_FENCE",
    "CLOSING_READING_COMMIT_MIN",
    "closing_reading",
    "closing_reading_refusals",
    "closing_reading_schema",
    "gate_state_for",
    "parse_closing_comment",
    "phase_for_gate",
    "render_closing_comment",
    "validate_closing_reading_document",
]

GATE_SCHEMA_VERSION = "gate.v1"


@dataclass(frozen=True)
class Clause:
    """One gate condition and what the store said about it.

    A clause is met, unmet, or UNMEASURABLE, and unmeasurable is never met
    (module docstring; `alpha-engine-config-I9869` round 2). ``unmeasurable``
    is a fact about OUR READING, never about the system being graded, and it
    renders distinctly rather than being folded into "unmet" so an operator
    does not go fix a producer when the fault is on our side of the boundary.
    Two things make a reading unmeasurable:

    * a store read that raised — a permission denial, a transient
      AccessDenied: we could not obtain the artifact;
    * an artifact whose schema PREDATES the question the clause now asks —
      we obtained it and it cannot answer (`_clause_old_weekly_within_cadence`
      after the 2026-09-04 ruling, `alpha-engine-config-I9962`). Grading it
      against the retired metric under the new clause's name would publish a
      number nobody asked for; grading it unmet would report a finding against
      a system behaving exactly as ruled.

    Neither is "the system failed", and neither is ever met.
    """

    name: str
    requirement: str
    met: bool
    detail: str
    evidence: tuple[str, ...] = ()
    unmeasurable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "requirement": self.requirement,
            "met": self.met,
            "unmeasurable": self.unmeasurable,
            "detail": self.detail,
            # Ordered, not de-duplicated. Each arc stage now reads its own
            # discriminated manifest key (alpha-engine-config-I9781), so a
            # repeated key in this list is evidence of a real collision
            # rather than an artifact of a bare, undiscriminated read that
            # de-duplication would otherwise mask.
            "evidence": sorted(self.evidence),
        }


#: `alpha-engine-config-I10417`: the rank a gate clause's member status is
#: reduced under, HIGHER IS WORSE. `UNMEASURABLE` ranks worse than `UNMET`
#: for the same reason `crucible.report.GRADE_RANK` ranks every `N/A-*`
#: status worse than `RED` — "we could not read this" must never render
#: better than "we read it and it said no".
CLAUSE_MEMBER_RANK: dict[str, int] = {"MET": 0, "UNMET": 1, "UNMEASURABLE": 2}


def _clause_member_status(clause: Clause) -> str:
    if clause.unmeasurable:
        return "UNMEASURABLE"
    return "MET" if clause.met else "UNMET"


@dataclass
class GateResult:
    """Every clause of one gate, and whether the phase may exit."""

    gate: str
    trading_day: dt.date
    window: list[dt.date]
    clauses: list[Clause] = field(default_factory=list)
    #: How much of the phase issue this clause list grades. Every registered
    #: gate now carries a declared deliverable table (`alpha-engine-config-
    #: I10309`), so `coverage_note` always returns a line rather than None;
    #: the field stays `str | None` for a `GateResult` constructed without
    #: going through `evaluate`. Not a clause: it is true by construction, so
    #: counting it in `met_ratio` would inflate the one figure plan §6 rule 3
    #: exists to keep honest. Set by `evaluate`.
    coverage: str | None = None

    @property
    def met(self) -> bool:
        return bool(self.clauses) and all(c.met for c in self.clauses)

    @property
    def met_ratio(self) -> float | None:
        """Met clauses over total, or `None` when nothing was measured.

        Zero clauses is `None`, never `0.0` and never `1.0` — an empty gate
        measured nothing, and `0.0` there is a false measurement: it reads as
        "measured, and none passed" when the truth is "never measured"
        (principle 7, `alpha-engine-config-I9824`). This is the ONE place the
        ratio is computed; `build_ladder` and `PhaseRow` both read it from
        here rather than re-deriving it, so the durable gate artifact and the
        overwritten ladder cannot disagree about the same reading.

        `None` too when ANY clause is `unmeasurable` — round 3 of
        `alpha-engine-config-I9869`: a store access failure on one clause,
        with the rest genuinely measured, was rendering as e.g. `0.5`, a
        specific number that reads as "half the requirement was checked and
        failed" when the truth is "one of two checks could not even run".
        `None` over "exclude the unmeasurable clause from the denominator"
        because the second option still publishes a number computed from a
        PARTIAL read, and plan §6 rule 1 is "no data is never a pass" — a
        partial read is not "no data", but averaging over what happened to
        succeed is the same shape of overclaim in miniature. `None` says
        plainly that this reading cannot be trusted as a ratio; the clause
        list itself still names exactly which one is unmeasurable.
        """
        if not self.clauses or any(c.unmeasurable for c in self.clauses):
            return None
        return sum(1 for c in self.clauses if c.met) / len(self.clauses)

    @property
    def members(self) -> list[MemberRow]:
        """`alpha-engine-config-I10417`: this gate's clauses, as members.

        `id` is the clause name, `value` is `met` (the clause's own boolean),
        `status` is MET/UNMET/UNMEASURABLE (:data:`CLAUSE_MEMBER_RANK`). `met`
        and `met_ratio` above are already pure functions of `self.clauses` —
        this is a VIEW onto the same list, not a second reduction, so the two
        cannot drift apart.
        """
        return [
            MemberRow(id=c.name, value=c.met, status=_clause_member_status(c)) for c in self.clauses
        ]

    def to_dict(self) -> dict[str, Any]:
        ratio = self.met_ratio
        return {
            "schema_version": GATE_SCHEMA_VERSION,
            "gate": self.gate,
            "trading_day": self.trading_day.isoformat(),
            "window": [d.isoformat() for d in self.window],
            "met": self.met,
            # `null`, never 0.0, when nothing was measured — see `met_ratio`.
            "met_ratio": None if ratio is None else round(ratio, 6),
            # What this reading does NOT grade, on the durable artifact that
            # gets pasted into the phase issue (plan §6 rule 2) — not only in
            # a clause detail somebody has to read to the end.
            "coverage": self.coverage,
            "clauses": [c.to_dict() for c in self.clauses],
            # `alpha-engine-config-I10417`: the SAME clauses, as members[],
            # reconstructible via `worst_member(self.members, rank=
            # CLAUSE_MEMBER_RANK)` — never `MET` when any clause is `UNMET`
            # or `UNMEASURABLE`, by construction of `met`/`met_ratio` above.
            "members": member_dicts(self.members),
        }

    def render(self) -> str:
        lines = [
            f"gate {self.gate}: {'MET' if self.met else 'NOT MET'} "
            f"({sum(1 for c in self.clauses if c.met)}/{len(self.clauses)} clauses)",
            f"window: {self.window[0].isoformat()}..{self.window[-1].isoformat()}"
            if self.window
            else "window: (empty)",
        ]
        if self.coverage:
            lines.append(f"coverage: {self.coverage}")
        for clause in self.clauses:
            marker = "x" if clause.met else ("?" if clause.unmeasurable else " ")
            lines.append(f"  [{marker}] {clause.name}: {clause.detail}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# reading helpers. Each returns None on absence rather than raising, because
# an absent artifact is a clause's ANSWER, not an error in the gate.
# ---------------------------------------------------------------------------


# `DocumentRead` and its two readers used to be DEFINED here. They moved to
# `crucible.documents` under `alpha-engine-config-I9900`, unchanged, because
# the console needed exactly this guard and a second copy of it would be the
# third reader in the tree — the defect `crucible.keys`' own docstring names
# ("a contract restated at each call site is a contract restated fifty times,
# and one of them has already drifted"). Imported under the private names the
# clauses below already call, so this file's call sites are untouched and the
# move is provably behaviour-preserving.


@dataclass(frozen=True)
class LinesRead:
    """One newline-delimited JSON event log, or the reason it could not be
    read.

    :class:`DocumentRead`'s shape does not fit an arm register: it is a JSONL
    event log, not one JSON object, and reporting the first bad line as "the
    whole key is unreadable" would lose which line. Same three outcomes —
    present-and-readable, absent, unreadable — and the same access-vs-content
    distinction as :class:`DocumentRead`, for the same reason.
    """

    lines: list[dict[str, Any]] | None
    absent: bool
    problem: str | None
    access_problem: bool = False


def _read_store_lines(store: Store, key: str) -> LinesRead:
    """Read one JSONL event log, guarded the same way :func:`_read_document`
    guards a single document.

    `_register_arms` used to do a bare `json.loads` per line with no guard at
    all: one malformed line (`arms/{slot}/register.jsonl` = `{not json`)
    raised `JSONDecodeError` out of `_clause_arms_all_scored`, out of
    `_phase1`, out of `evaluate` — no ladder, no artifact
    (`alpha-engine-config-I9869` round 2).
    """
    try:
        present = store.exists(key)
    except Exception as exc:
        return LinesRead(
            None,
            False,
            f"{key} could not be read: {type(exc).__name__}: {exc}. That is a statement "
            "about our access, not about the system being measured",
            access_problem=True,
        )
    if not present:
        return LinesRead(None, True, None)
    try:
        raw = store.get_bytes(key)
    except Exception as exc:
        return LinesRead(
            None,
            False,
            f"{key} could not be read: {type(exc).__name__}: {exc}. That is a statement "
            "about our access, not about the system being measured",
            access_problem=True,
        )
    try:
        text = raw.decode("utf-8")
    except Exception as exc:
        return LinesRead(None, False, f"{key} is not readable UTF-8: {type(exc).__name__}: {exc}")
    lines: list[dict[str, Any]] = []
    for lineno, entry in enumerate(text.splitlines(), start=1):
        if not entry.strip():
            continue
        try:
            parsed = json.loads(entry)
        except Exception as exc:
            return LinesRead(
                None, False, f"{key}:{lineno} is not readable JSON: {type(exc).__name__}: {exc}"
            )
        if not isinstance(parsed, dict):
            return LinesRead(
                None,
                False,
                f"{key}:{lineno} parsed to {type(parsed).__name__}, not an object with fields",
            )
        lines.append(parsed)
    return LinesRead(lines, False, None)


@dataclass(frozen=True)
class KeysRead:
    """Every key under one store prefix, or the reason the listing could not
    be read.

    Round 2 of `alpha-engine-config-I9869` guarded every single-key read
    (`DocumentRead`, `LinesRead`) and left every LISTING unguarded:
    `store.list_keys` can raise a permission denial mid-listing exactly the
    way `get_bytes` can, and a bare call at three sites
    (`_clause_pointer_flipped_on_smoke`, `_clause_independently_reviewed`,
    `last_read`) raised straight out of `evaluate`/`build_ladder` — round 3,
    finding 2. No `absent` outcome: an empty listing under a prefix that
    exists is a legitimate, ordinary reading (zero artifacts filed yet), not
    an error — unlike a single key, a prefix has no not-there state to
    distinguish from empty.
    """

    keys: list[str] | None
    problem: str | None
    access_problem: bool = False


def _list_store_keys(store: Store, prefix: str) -> KeysRead:
    try:
        keys = list(store.list_keys(prefix))
    except Exception as exc:
        return KeysRead(
            None,
            f"listing {prefix!r} could not be read: {type(exc).__name__}: {exc}. That is a "
            "statement about our access, not about the system being measured",
            access_problem=True,
        )
    return KeysRead(keys, None)


@dataclass(frozen=True)
class BytesRead:
    """One key's raw bytes, guarded the same way :class:`DocumentRead` is,
    for a caller that wants the bytes themselves rather than a parsed
    document.

    `track_f.gate_handler` records each clause's evidence keys as job
    lineage (`ctx.record_input`), which hashes the exact bytes the store
    holds — re-serializing a `DocumentRead.document` back to JSON would not
    reliably round-trip to those same bytes (key order, whitespace), so
    lineage recording needs the raw bytes, guarded, not a parsed-and-
    re-emitted copy.
    """

    raw: bytes | None
    absent: bool
    problem: str | None
    access_problem: bool = False


def _read_store_bytes(store: Store, key: str) -> BytesRead:
    try:
        present = store.exists(key)
    except Exception as exc:
        return BytesRead(
            None,
            False,
            f"{key} could not be read: {type(exc).__name__}: {exc}. That is a statement "
            "about our access, not about the system being measured",
            access_problem=True,
        )
    if not present:
        return BytesRead(None, True, None)
    try:
        raw = store.get_bytes(key)
    except Exception as exc:
        return BytesRead(
            None,
            False,
            f"{key} could not be read: {type(exc).__name__}: {exc}. That is a statement "
            "about our access, not about the system being measured",
            access_problem=True,
        )
    return BytesRead(raw, False, None)


def arm_name(arm_id: str) -> str:
    """The NAME component of a registered arm id.

    `nousergon_lib.arena.derive_arm_id` returns `{slot}:{name}:{spec_hash}`,
    while `SlotSpec.control_arms` carries the bare name. Comparing the two
    forms directly is a filter that matches nothing — the defect two
    independent reviews found in `promotable_arms` on 2026-09-01, where the
    §10.1 control exclusion could never fire because it compared
    `control_planted_m` against `m:control_planted_m:7e8059f49558`. This
    function is the one translation, so there is exactly one place for that to
    be right.
    """
    parts = arm_id.split(":")
    return parts[1] if len(parts) == 3 else arm_id


def _register_arms(
    store: Store, slot: str
) -> tuple[set[str], str, str | None, bool, ArmRegister | None]:
    """The slot's ACTIVE registered arm ids, the key, the reason if the
    register could not be read (``problem``, ``access_problem``), and the
    :class:`~nousergon_lib.arena.ArmRegister` the ids were folded from.

    Guarded through :func:`_read_store_lines` rather than the bare
    `json.loads` per line this used to run: an unreadable register is a red
    `arms_all_scored` reading naming the key, never an exception out of
    `evaluate` (`alpha-engine-config-I9869` round 2).

    The register is returned alongside the folded id set (rather than
    discarded once `active_arms()` is taken) so a caller needing register-
    backed control classification (`crucible.slots.is_control_arm`,
    `alpha-engine-config-I10044`) threads the register this function already
    loaded instead of re-reading the same key a second time.
    """
    key = arm_register_key(slot)
    read = _read_store_lines(store, key)
    if read.problem is not None:
        return set(), key, read.problem, read.access_problem, None
    if read.absent:
        return set(), key, None, False, None
    from nousergon_lib.arena import ArmRegister  # noqa: PLC0415 - heavy import, one call site

    register = ArmRegister.from_dicts(read.lines or [])
    return set(register.active_arms()), key, None, False, register


# ---------------------------------------------------------------------------
# phase 1 clauses
# ---------------------------------------------------------------------------


def _field(key: str, document: dict[str, Any], field_name: str, expected_type: type) -> str | None:
    """``None`` when ``field_name`` is present on ``document`` AND of
    ``expected_type``, else the malformed detail naming the key, the field,
    and what is wrong.

    A missing REQUIRED field and a present-but-wrong-typed one are both
    malformed, not absent: the document exists and is readable JSON, it just
    does not carry the shape the schema promises. Reporting either as
    "absent" would send the operator looking for a producer that already
    ran. Round 1 of `alpha-engine-config-I9869` guarded presence only —
    `{"rows": 5}`, `{"scored_arms": [{"a": 1}]}` and `{"sha": 12345}` each
    still crashed a downstream index (`len()`, `set()`, a slice) on a
    present-but-wrong-typed value; round 2 closes that.

    ``bool`` is a `Python` subclass of `int`, so a caller checking `int`
    would otherwise accept `True`/`False` — no phase-1 field is typed `int`
    today, so this is not yet exercised, but the exclusion is written down
    rather than left to be found the way `alpha-engine-config-I9870`'s
    `executions_started` check found it.
    """
    if field_name not in document:
        return f"{key}: missing required field `{field_name}`"
    value = document[field_name]
    if isinstance(value, bool) and expected_type is not bool:
        return f"{key}: `{field_name}` is {value!r}, not {expected_type.__name__}"
    if not isinstance(value, expected_type):
        return f"{key}: `{field_name}` is {value!r}, not {expected_type.__name__}"
    return None


@lru_cache(maxsize=1)
def _manifest_status_values() -> frozenset[str]:
    """The exhaustive run-manifest `status` vocabulary, derived from the
    CURRENT schema every manifest is validated against at write time — never a
    restated literal, so a schema change is picked up here without a second
    edit that could drift from it. `alpha-engine-config-I9869` round 2: a
    manifest carrying `status: "degraded"` or `status: 3` was re-rendered as
    `failed` by the `!= "ok"` comparison every phase-1 clause used, silently
    accepting any value the schema itself forbids.
    """
    return frozenset(load_schema()["properties"]["status"]["enum"])


def _status(key: str, document: dict[str, Any]) -> tuple[str | None, str | None]:
    """The document's validated `status`, or the malformed detail naming why
    it is not one.

    Three checks, in order: `status` is present and a string; it is one of
    the schema's exhaustive values (never `!= "ok"`, which lets anything
    through); and `status: "ok"` implies `reason == ""` — the same
    implication the run manifest's own conditional schema enforces at write
    time, so a manifest that satisfied it when written and was since
    hand-edited is exactly the malformed input this clause exists to catch.
    """
    problem = _field(key, document, "status", str)
    if problem is not None:
        return None, problem
    status = document["status"]
    if status not in _manifest_status_values():
        return None, (
            f"{key}: `status` is {status!r}, not one of {sorted(_manifest_status_values())}"
        )
    problem = _field(key, document, "reason", str)
    if problem is not None:
        return None, problem
    reason = document["reason"]
    if status == "ok" and reason != "":
        return None, f"{key}: status `ok` but `reason` is {reason!r}, not empty"
    return status, None


def _fault_excused_run_ids(store: Store) -> tuple[frozenset[str] | None, str | None]:
    """Every `run_id` a fault-injection record (`faults/{day}/{fault}.json`,
    `alpha-engine-config-I10320`/`-I10322`) claims to excuse, across every
    trading day.

    Matched on `run_id` alone, never on the trading day: a failed manifest is
    excused from `arc_runs_ok`/`replays_ok` only when SOME fault record names
    its exact `run_id`, so a genuine failure on a day a fault was once
    induced still fails those clauses — `-I10322`'s whole design constraint.

    **And only an `induced` record excuses anything** (`-I10327`). The record
    grew two more outcome kinds, and neither may reach this set: `absorbed`
    names a manifest reading `ok`, which these clauses were never going to
    count against anyone, and `unreachable` names no run at all. Keying the
    exclusion on `run_id` AND narrowing it by `outcome` is what keeps the
    two new kinds from being a second route to the excusal the `run_id`
    refusal guards — a record filed for a fault the system survived must not
    be able to excuse a failure it did not cause.

    Returns ``(None, problem)`` when the listing itself could not be read —
    an access failure, not "no faults filed" — so the caller folds it into
    `unmeasurable` rather than silently grading with zero exclusions. A
    record that fails to parse, or names no `run_id`, is skipped rather than
    raised: a malformed or hand-broken record excuses nothing, which is the
    only safe default for a mechanism whose entire point is that it must
    never be able to turn an arbitrary red clause green.
    """
    listed = _list_store_keys(store, FAULT_INJECTION_ROOT)
    if listed.problem is not None:
        return None, listed.problem
    run_ids: set[str] = set()
    for key in listed.keys or []:
        if parse_fault_injection_key(key) is None:
            continue
        read = _read_store_document(store, key)
        if read.problem is not None or read.absent:
            continue
        document = read.document or {}
        if document.get("outcome") != FAULT_OUTCOME_INDUCED:
            continue
        run_id = str(document.get("run_id") or "").strip()
        if run_id:
            run_ids.add(run_id)
    return frozenset(run_ids), None


def _clause_arc_runs_ok(
    store: Store, window: list[dt.date], registry: dict[str, Component]
) -> Clause:
    # A failure excused by a matching fault-injection record's run_id is not
    # counted against this clause (alpha-engine-config-I10322) -- see
    # `_fault_excused_run_ids` below.
    requirement = (
        "every stage of the weekly arc wrote a manifest with status `ok` for each "
        "trading day in the window, or its failure is excused by a fault-injection "
        "record naming that exact run_id"
    )
    missing: list[str] = []
    malformed: list[str] = []
    unmeasurable: list[str] = []
    failed: list[str] = []
    excused: list[str] = []
    evidence: list[str] = []
    excused_run_ids, excused_problem = _fault_excused_run_ids(store)
    if excused_problem is not None:
        unmeasurable.append(excused_problem)
    for day in window:
        for stage in arc_stages(day, registry):
            key = manifest_key(stage.job, day.isoformat(), discriminator=stage.slot)
            evidence.append(key)
            read = _read_store_document(store, key)
            if read.problem is not None:
                (unmeasurable if read.access_problem else malformed).append(read.problem)
                continue
            if read.absent:
                missing.append(f"{stage.label}@{day.isoformat()}")
                continue
            document = read.document or {}
            status, problem = _status(key, document)
            if problem is not None:
                malformed.append(problem)
                continue
            if status != "ok":
                run_id = str(document.get("run_id") or "")
                if run_id and excused_run_ids is not None and run_id in excused_run_ids:
                    excused.append(f"{stage.label}@{day.isoformat()} (run_id {run_id})")
                    continue
                failed.append(f"{stage.label}@{day.isoformat()}: {document['reason']}")
    if missing or malformed or unmeasurable or failed:
        parts = []
        if unmeasurable:
            parts.append(f"{len(unmeasurable)} could not be read: {'; '.join(unmeasurable[:2])}")
        if missing:
            parts.append(f"{len(missing)} never ran ({', '.join(missing[:4])}...)")
        if malformed:
            parts.append(f"{len(malformed)} malformed: {'; '.join(malformed[:4])}")
        if failed:
            parts.append(f"{len(failed)} failed ({'; '.join(failed[:2])})")
        if excused:
            parts.append(f"{len(excused)} excused by a fault record: {'; '.join(excused[:2])}")
        content_gap = bool(missing or malformed or failed)
        return Clause(
            "arc_runs_ok",
            requirement,
            False,
            "; ".join(parts),
            tuple(evidence),
            # A content gap (missing/malformed/failed) is a real reading —
            # never masked as "could not measure" just because an unrelated
            # access failure also occurred (round 3, finding 6).
            unmeasurable=bool(unmeasurable) and not content_gap,
        )
    detail = f"{len(evidence)} stage manifests over {len(window)} trading days, all ok"
    if excused:
        detail += f" ({len(excused)} excused by a matching fault record: {'; '.join(excused[:2])})"
    return Clause("arc_runs_ok", requirement, True, detail, tuple(evidence))


def _clause_arms_all_scored(store: Store, window: list[dt.date]) -> Clause:
    requirement = (
        "each slot's arena cycle scored every ACTIVE registered arm and both control "
        "arms, on every trading day in the window"
    )
    gaps: list[str] = []
    unmeasurable: list[str] = []
    evidence: list[str] = []
    # Each slot's register is read ONCE, before the day loop. `_register_arms`
    # reads a single key per SLOT, not per day — re-reading it inside the day
    # x slot loop (round 2's shape) meant one malformed register filed the
    # SAME problem up to `len(window)` times, filling `gaps[:4]` with
    # duplicates of it and hiding a genuinely missing arena cycle for an
    # unrelated slot/day (`alpha-engine-config-I9869` round 3, finding 5).
    registers: dict[str, tuple[set[str], str | None, str | None, bool]] = {}
    # The slots the CLI can RUN, not every slot the harness declares: M and
    # S have no `produce`/`grade` until phase 3, and a phase-1 clause that
    # demanded their arena cycles could never read MET before phase 3 shipped
    # (measured 2026-09-04). Same derivation `arc_stages` uses, so the two
    # phase-1 clauses agree about which slots a week must have scored.
    slots = {slot: SLOTS[slot] for slot in SLOTS if slot in dispatchable_slots()}
    for slot in slots:
        registered, register_key, register_problem, register_access, _register_unused = (
            _register_arms(store, slot)
        )
        registers[slot] = (registered, register_key, register_problem, register_access)
        if register_key is not None:
            evidence.append(register_key)
        if register_problem is not None:
            (unmeasurable if register_access else gaps).append(register_problem)
    for day in window:
        for slot, spec in slots.items():
            key = arena_cycle_key(slot, day.isoformat())
            evidence.append(key)
            read = _read_store_document(store, key)
            if read.problem is not None:
                (unmeasurable if read.access_problem else gaps).append(read.problem)
                continue
            if read.absent:
                gaps.append(f"{slot}@{day.isoformat()}: no arena_cycle artifact")
                continue
            cycle = read.document or {}
            problem = _field(key, cycle, "scored_arms", list)
            if problem is not None:
                gaps.append(problem)
                continue
            non_str = [a for a in cycle["scored_arms"] if not isinstance(a, str)]
            if non_str:
                gaps.append(f"{key}: `scored_arms` contains non-string element(s): {non_str[:2]!r}")
                continue
            scored = set(cycle["scored_arms"])
            registered, _register_key, register_problem, _register_access = registers[slot]
            if register_problem is not None:
                # Already recorded once, above, when the register was read.
                # This day's contribution to the SAME problem is not a
                # second independent finding.
                continue
            unscored = registered - scored
            if unscored:
                gaps.append(f"{slot}@{day.isoformat()}: {sorted(unscored)} registered but unscored")
            controls = {c.arm_id for c in spec.control_arms}
            if not controls & {arm_name(a) for a in scored}:
                gaps.append(
                    f"{slot}@{day.isoformat()}: no control arm was scored — an unscored "
                    "control is an unverified grader (§10.1)"
                )
    # De-duplicated before truncating: a defensive backstop over the hoist
    # above, not a substitute for it — the hoist removes the duplication at
    # its source, this just refuses to let any future duplicate source push
    # a distinct finding out of the `[:4]` window.
    gaps = list(dict.fromkeys(gaps))
    unmeasurable = list(dict.fromkeys(unmeasurable))
    if gaps or unmeasurable:
        parts = []
        if unmeasurable:
            parts.append(f"{len(unmeasurable)} could not be read: {'; '.join(unmeasurable[:2])}")
        if gaps:
            parts.append("; ".join(gaps[:4]))
        return Clause(
            "arms_all_scored",
            requirement,
            False,
            "; ".join(parts),
            tuple(evidence),
            # A content gap is a real reading — never masked as "could not
            # measure" just because an unrelated access failure also
            # occurred (round 3, finding 6).
            unmeasurable=bool(unmeasurable) and not gaps,
        )
    return Clause(
        "arms_all_scored",
        requirement,
        True,
        f"{len(slots)} slots x {len(window)} days, every registered arm and both controls scored",
        tuple(evidence),
    )


def _clause_attribution_renders(store: Store, window: list[dt.date]) -> Clause:
    requirement = (
        "report/{trading_day}/attribution.json exists for the final day of the window "
        "with five rows, each carrying a value or an explicit N/A status and a reason"
    )
    day = window[-1]
    key = attribution_key(day.isoformat())
    read = _read_store_document(store, key)
    if read.problem is not None:
        return Clause(
            "attribution_renders",
            requirement,
            False,
            read.problem,
            (key,),
            unmeasurable=read.access_problem,
        )
    if read.absent:
        return Clause("attribution_renders", requirement, False, f"{key} is absent", (key,))
    document = read.document or {}
    rows_problem = _field(key, document, "rows", list)
    if rows_problem is not None:
        return Clause("attribution_renders", requirement, False, rows_problem, (key,))
    rows = document["rows"]
    if len(rows) != 5:
        return Clause(
            "attribution_renders", requirement, False, f"{len(rows)} rows, expected 5", (key,)
        )
    silent: list[str] = []
    for idx, r in enumerate(rows):
        row_key = f"{key}[{idx}]"
        if not isinstance(r, dict):
            return Clause(
                "attribution_renders",
                requirement,
                False,
                f"{row_key}: row is {type(r).__name__}, not an object with fields",
                (key,),
            )
        name_problem = _field(row_key, r, "name", str)
        if name_problem is not None:
            return Clause("attribution_renders", requirement, False, name_problem, (key,))
        if (
            r.get("value") is None
            and not str(r.get("status", "")).startswith("N/A")
            or not r.get("status_reason")
        ):
            silent.append(r["name"])
    if silent:
        return Clause(
            "attribution_renders",
            requirement,
            False,
            f"rows {silent} carry neither a value nor an explained N/A status",
            (key,),
        )
    return Clause(
        "attribution_renders", requirement, True, f"5 rows, all explained, at {key}", (key,)
    )


def _clause_explain_walks_a_verdict(store: Store, window: list[dt.date]) -> Clause:
    requirement = (
        "an `explain` run in the window read a verdict artifact — the lineage walk "
        "was exercised on real output, not only in tests"
    )
    evidence = [manifest_key("explain", d.isoformat()) for d in window]
    malformed: list[str] = []
    unmeasurable: list[str] = []
    for key in evidence:
        read = _read_store_document(store, key)
        if read.problem is not None:
            (unmeasurable if read.access_problem else malformed).append(read.problem)
            continue
        if read.absent:
            continue
        document = read.document or {}
        status, problem = _status(key, document)
        if problem is not None:
            malformed.append(problem)
            continue
        if status != "ok":
            continue
        inputs_problem = _field(key, document, "inputs", list)
        if inputs_problem is not None:
            malformed.append(inputs_problem)
            continue
        inputs = document["inputs"]
        if any(isinstance(i, dict) and "verdict.json" in str(i.get("key", "")) for i in inputs):
            return Clause("explain_walks_a_verdict", requirement, True, f"walked at {key}", (key,))
    if unmeasurable or malformed:
        detail = ""
        if unmeasurable:
            detail = f"{len(unmeasurable)} could not be read: {'; '.join(unmeasurable[:2])}"
        if malformed:
            detail = f"{detail}; " if detail else detail
            detail += "; ".join(malformed)
        return Clause(
            "explain_walks_a_verdict",
            requirement,
            False,
            detail,
            tuple(evidence),
            # A content gap (malformed) is a real reading — never masked as
            # "could not measure" just because an unrelated access failure
            # also occurred (round 3, finding 6).
            unmeasurable=bool(unmeasurable) and not malformed,
        )
    return Clause(
        "explain_walks_a_verdict",
        requirement,
        False,
        "no ok `explain` manifest in the window records a verdict as an input",
        tuple(evidence),
    )


def _clause_pointer_flipped_on_smoke(store: Store, window: list[dt.date]) -> Clause:
    requirement = (
        "releases/current names a sha, and an `ok` smoke manifest carries that same "
        "release_sha — the pointer flipped on a real smoke, not by hand"
    )
    pointer_read = _read_store_document(store, POINTER_KEY)
    if pointer_read.problem is not None:
        return Clause(
            "pointer_flipped_on_smoke",
            requirement,
            False,
            pointer_read.problem,
            (POINTER_KEY,),
            unmeasurable=pointer_read.access_problem,
        )
    if pointer_read.absent:
        return Clause(
            "pointer_flipped_on_smoke",
            requirement,
            False,
            f"{POINTER_KEY} is absent; no release has ever been published",
            (POINTER_KEY,),
        )
    pointer = pointer_read.document or {}
    sha_problem = _field(POINTER_KEY, pointer, "sha", str)
    if sha_problem is not None:
        return Clause(
            "pointer_flipped_on_smoke",
            requirement,
            False,
            sha_problem,
            (POINTER_KEY,),
        )
    sha = pointer["sha"]
    evidence = [POINTER_KEY]
    malformed: list[str] = []
    unmeasurable: list[str] = []
    smoke_keys = _list_store_keys(store, runs_prefix("smoke"))
    if smoke_keys.problem is not None:
        return Clause(
            "pointer_flipped_on_smoke",
            requirement,
            False,
            smoke_keys.problem,
            tuple(evidence),
            unmeasurable=smoke_keys.access_problem,
        )
    for key in smoke_keys.keys or []:
        if not key.endswith("run.json"):
            continue
        evidence.append(key)
        read = _read_store_document(store, key)
        if read.problem is not None:
            (unmeasurable if read.access_problem else malformed).append(read.problem)
            continue
        if read.absent:
            continue
        document = read.document or {}
        status, problem = _status(key, document)
        if problem is not None:
            malformed.append(problem)
            continue
        if status == "ok" and document.get("release_sha") == sha:
            return Clause(
                "pointer_flipped_on_smoke",
                requirement,
                True,
                f"{POINTER_KEY} -> {sha[:12]}, smoked at {key}",
                tuple(evidence),
            )
    if unmeasurable or malformed:
        detail = ""
        if unmeasurable:
            detail = f"{len(unmeasurable)} could not be read: {'; '.join(unmeasurable[:2])}"
        if malformed:
            detail = f"{detail}; " if detail else detail
            detail += "; ".join(malformed)
        return Clause(
            "pointer_flipped_on_smoke",
            requirement,
            False,
            detail,
            tuple(evidence),
            # A content gap (malformed) is a real reading — never masked as
            # "could not measure" just because an unrelated access failure
            # also occurred in the same window (round 3, finding 6).
            unmeasurable=bool(unmeasurable) and not malformed,
        )
    return Clause(
        "pointer_flipped_on_smoke",
        requirement,
        False,
        f"{POINTER_KEY} names {sha[:12] or '(no sha)'} but no ok smoke manifest carries it",
        tuple(evidence),
    )


#: The review document's schema token. Written by `crucible.review.record`,
#: driven by `.github/workflows/adversarial-review-record.yml`, and read by
#: :func:`_clause_independently_reviewed`. A document without it is a shape
#: this clause has never agreed to read, and is refused rather than guessed at.
REVIEW_SCHEMA_VERSION = "review.v1"

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class _Review:
    """One review artifact, with the facts its key carries kept separate.

    ``day``, ``reviewer`` and ``verdict`` come from the KEY; the rest from the
    body. Held apart on purpose: the key is what the store actually did, the
    body is what a producer claimed, and :func:`_review_problem` refuses a
    document whose body disagrees with its own key. Reading either alone would
    let a `fail` be filed under a `pass` key — the exact overwrite the key
    shape exists to prevent, wearing a different hat.
    """

    key: str
    day: str
    verdict: str
    reviewer: str
    authors: frozenset[str]
    head_sha: str
    summary: str

    @property
    def is_self_review(self) -> bool:
        return self.reviewer in self.authors


def _clause_independently_reviewed(store: Store, phase: str, window: list[dt.date]) -> Clause:
    """Plan §11 risk 1, as a MEASUREMENT over the store.

    "Every track's exit is reviewed by an independent adversarial agent
    against §2's acceptance tests, not by its author." Until this clause
    existed that lived only as a GitHub check-run, so `crucible gate` — the
    thing that decides whether a phase may exit — could not see it at all, and
    a phase could exit with no review having happened (plan §11 risk 1: "v2
    passes its gates because its tests were written to pass").

    It is now the ONLY place the review is enforced. `adversarial-review-gate`
    is never re-armed as a required status check (Brian ruling 2026-09-03): a
    `skipped` check-run under a required name counts as success, so the graded
    party could green it by pressing a dispatch button. A phase gate cannot be
    pressed.

    **The independence comparison is between a reviewer and a DERIVED author
    set.** ``authors`` is not the recorder's opinion of who wrote the code:
    `crucible.review.author_identities` reads it out of the commits under
    review. A gate whose input is supplied by the thing it grades measures
    nothing, and that is what the first version of this control did — reviewer
    and author were free-text `workflow_dispatch` inputs typed by the session
    asking to be passed (`alpha-engine-config-I9873`). It is better-evidenced,
    not proof; `crucible.review`'s docstring states the residual exactly, and
    this clause does not claim more than that module does.

    **An adverse verdict is superseded, never erased.** The verdict is a key
    segment, so a `pass` cannot overwrite a `fail`. A fail in the window is
    cleared only by an independent `pass` that names a DIFFERENT head sha and
    was filed no earlier than the fail — findings are answered by changing the
    code, and changing the code changes the sha. A second reviewer passing the
    same sha is not a rebuttal, and neither is the first reviewer changing its
    own mind on it.

    **The review's date, reviewer and verdict come from the KEY**, never from
    the body: those are what the store actually holds, and
    `Store.assert_keys_bind_to_trading_days` already holds the date to a
    session. A review from outside the window reviewed a superseded state of
    the code and does not satisfy a later exit.
    """
    requirement = (
        "an independent reviewer recorded `pass` in the window against the commits "
        "under review, and no adverse verdict in the window is still outstanding"
    )
    prefix = review_prefix(phase)
    days = {day.isoformat() for day in window}
    listed = _list_store_keys(store, prefix)
    if listed.problem is not None:
        # A listing that could not even be attempted is not "no review has
        # ever been filed" — it is "we could not ask", which is a fact about
        # our access, not about whether an independent review exists
        # (`alpha-engine-config-I9869` round 3, finding 2).
        return Clause(
            "independently_reviewed",
            requirement,
            False,
            listed.problem,
            (prefix,),
            unmeasurable=listed.access_problem,
        )
    keys = sorted(key for key in listed.keys or [] if key.endswith(".json"))
    if not keys:
        return Clause(
            "independently_reviewed",
            requirement,
            False,
            f"no review document under `{prefix}`. No independent adversarial review of "
            f"{phase} has ever been filed, so this gate has nothing to read (plan §11 "
            "risk 1)",
            (prefix,),
        )

    evidence = tuple(keys)
    problems: list[str] = []
    access_problems: list[str] = []
    reviews: list[_Review] = []
    for key in keys:
        read = _read_store_document(store, key)
        if read.problem is not None:
            (access_problems if read.access_problem else problems).append(read.problem)
            continue
        if read.absent:
            problems.append(f"{key} was listed and then could not be read")
            continue
        problem, review = _review_problem(key, prefix, read.document or {})
        if problem is not None:
            problems.append(problem)
            continue
        if review is not None:
            reviews.append(review)
    if problems or access_problems:
        parts = []
        if access_problems:
            parts.append(
                f"{len(access_problems)} could not be read: {'; '.join(access_problems[:2])}"
            )
        if problems:
            parts.append("; ".join(problems))
        return Clause(
            "independently_reviewed",
            requirement,
            False,
            "; ".join(parts),
            evidence,
            # A content problem is a real reading — never masked as "could
            # not measure" just because an unrelated access failure also
            # occurred (round 3, finding 6, same class as finding 3).
            unmeasurable=bool(access_problems) and not problems,
        )

    self_reviews = [r for r in reviews if r.is_self_review]
    independent = [r for r in reviews if not r.is_self_review]
    in_window = [r for r in independent if r.day in days]
    stale = [r for r in independent if r.day not in days]

    passes = [r for r in in_window if r.verdict == "pass"]
    outstanding = [
        r
        for r in in_window
        if r.verdict == "fail"
        and not any(p.head_sha != r.head_sha and p.day >= r.day for p in passes)
    ]
    if outstanding:
        return Clause(
            "independently_reviewed",
            requirement,
            False,
            "; ".join(
                f"{r.key}: `{r.reviewer}` recorded `fail` on {r.head_sha[:12]} — "
                f"{r.summary or '(no summary)'}, and no later independent pass names a "
                "different head sha. Findings are answered by changing the code, which "
                "changes the sha — not by re-recording a pass on this one"
                for r in sorted(outstanding, key=lambda r: r.key)
            ),
            evidence,
        )
    if passes:
        return Clause(
            "independently_reviewed",
            requirement,
            True,
            "; ".join(
                f"{r.key}: `pass` from {r.reviewer} on {r.head_sha[:12]}"
                for r in sorted(passes, key=lambda r: r.key)
            ),
            evidence,
        )

    detail = [
        f"{r.key}: reviewer `{r.reviewer}` is one of the authors read out of the commits "
        "reviewed — a self-review is not a review"
        for r in sorted(self_reviews, key=lambda r: r.key)
    ] + [
        f"{r.key}: filed on {r.day}, outside this gate's window"
        for r in sorted(stale, key=lambda r: r.key)
    ]
    return Clause(
        "independently_reviewed",
        requirement,
        False,
        "; ".join(detail)
        or f"{len(keys)} review documents under `{prefix}`, none of them an independent "
        "`pass` in the window",
        evidence,
    )


def _review_problem(
    key: str, prefix: str, document: dict[str, Any]
) -> tuple[str | None, _Review | None]:
    """Why ``document`` is not a review this clause will read, or the review.

    Every branch is a RED clause, never an exception and never a skip: a review
    document we cannot read is `no data`, and `no data` is never a pass
    (principle 7). It is also never silently ignored — a malformed document
    sitting in the prefix would otherwise let the gate report "no review
    exists" when what happened is "the producer wrote something we do not
    understand", and those two name different remedies.
    """
    segments = key[len(prefix) :].removesuffix(".json").split("/")
    if len(segments) != 3:
        return (
            f"{key}: not a `{{trading_day}}/{{reviewer}}/{{verdict}}.json` key. A review "
            "filed under a shape this clause cannot parse is unreadable, not absent",
            None,
        )
    day, reviewer, verdict = segments
    if document.get("schema_version") != REVIEW_SCHEMA_VERSION:
        return (
            f"{key}: schema_version is {document.get('schema_version')!r}, not "
            f"{REVIEW_SCHEMA_VERSION!r}",
            None,
        )
    # The body must agree with the key. A `fail` body filed under a `pass` key
    # would be counted as a pass by the key and as a fail by the body, and the
    # verdict-in-the-key durability would mean nothing.
    for name, from_key in (("verdict", verdict), ("reviewer", reviewer)):
        claimed = document.get(name)
        if not isinstance(claimed, str) or claimed.lower() != from_key:
            return (
                f"{key}: body says {name}={claimed!r} while its own key says "
                f"{from_key!r}. A document that disagrees with the key it was filed "
                "under is refused, not reconciled",
                None,
            )
    authors = document.get("authors")
    if not isinstance(authors, list) or not authors or not all(isinstance(a, str) for a in authors):
        return (
            f"{key}: authors is {authors!r}, not a non-empty list of identities read out "
            "of the commits reviewed. An empty author set would make every reviewer "
            "independent by construction",
            None,
        )
    head_sha = document.get("head_sha")
    if not isinstance(head_sha, str) or not _SHA_RE.match(head_sha):
        return (
            f"{key}: head_sha is {head_sha!r}, not a full 40-hex commit sha. A review "
            "that does not name what it reviewed grades nothing",
            None,
        )
    return None, _Review(
        key=key,
        day=day,
        verdict=verdict,
        reviewer=reviewer,
        authors=frozenset(a.lower() for a in authors),
        head_sha=head_sha,
        summary=str(document.get("summary") or ""),
    )


def _phase1(
    store: Store,
    window: list[dt.date],
    registry: dict[str, Component],
    *,
    trading_day: dt.date,
) -> list[Clause]:
    # ``window`` is the anchored weekly window (`WEEKLY_ANCHORED_GATES`): the
    # arc, arena and attribution artifacts are filed at those closes. The
    # review and the explain walk are filed on the day they happened, so they
    # read the whole span up to the render day — see `_session_span`.
    span = _session_span(window, trading_day)
    return [
        _clause_arc_runs_ok(store, window, registry),
        _clause_arms_all_scored(store, window),
        _clause_attribution_renders(store, window),
        _clause_explain_walks_a_verdict(store, span),
        _clause_pointer_flipped_on_smoke(store, window),
        _clause_independently_reviewed(store, "phase1", span),
    ]


# ---------------------------------------------------------------------------
# phase 0 clauses
#
# Plan §6 names phase 0's exit gate in two numbers — "old-weekly executions/week
# <= 1, acceptance suite present and failing honestly" — and names, in the same
# table, the figures that are NOT the gate: the acceptance PASS count and the
# deleted-Lambda count. So this clause list grades two of the five deliverables
# `alpha-engine-config-I9756` carries.
#
# The other three are not silently absent: :data:`GATE_DELIVERABLES` records why
# no artifact answers them, :func:`coverage_note` renders that as a line on the
# reading itself, and `build_ladder` carries it into the ladder row's detail —
# so a phase that one day reads MET says, on the same surface, which of its
# deliverables were never measured. That disclosure is deliberately NOT a
# clause: it is true by construction, no store state can change it, and a
# tautology counted in `met_ratio` inflates the headline figure that plan §6
# rule 3 exists to keep honest.
# ---------------------------------------------------------------------------

#: The most executions of the v1 weekly state machine that phase 0 tolerates in
#: one week (plan §6; `alpha-engine-config-I5489`'s punitive 2.5x/day cadence is
#: what it ends). Since Brian's 2026-09-04 ruling on
#: `alpha-engine-config-I9756` this counts executions that PASSED
#: `WeeklyRunDayGate`, not raw starts — see
#: `_clause_old_weekly_within_cadence`.
LEGACY_WEEKLY_MAX_STARTS_PER_WEEK = 1

#: The FEWEST gate-passing executions phase 0 accepts in one week.
#:
#: The ruled predicate reads "at most one execution that passes
#: `WeeklyRunDayGate` per calendar week". Taken literally, a week with ZERO
#: gate-passing executions satisfies it — while meaning the weekly pipeline
#: never ran, which `sf-pipeline-policy.md` §5 line 213 calls worse than a
#: duplicate ("missing a weekly run is worse than a duplicate; the mutex
#: handles duplicates"). The THU-SAT fail-open margin exists to prevent
#: exactly that, so a clause certifying a silent week as "clean" would grade
#: the absence of the thing the margin protects. With phase 0's window
#: narrowed to ONE week on 2026-09-04, that failure mode goes from unlikely to
#: one bad Saturday, so phase 0 grades EXACTLY one run a week, not at most one
#: (`alpha-engine-config-I9962`).
#:
#: This is a bound, not a ceiling variant: phase 4's correct lower bound is
#: ZERO, and it says so at its own call site.
LEGACY_WEEKLY_MIN_RUNS_PER_WEEK = 1

#: The schema the filed weekly document must declare for the ruled metric to be
#: answerable. `v1` carried one integer, `executions_started`, and nothing else;
#: a `v1` week is UNMEASURABLE under the ruled question rather than a pass, so
#: the version is checked rather than the fields being probed for.
#: Producer: `nous-ergon-ops/scripts/legacy_weekly_executions_producer.py`.
LEGACY_WEEKLY_EXECUTIONS_SCHEMA_VERSION = "legacy-weekly-executions.v2"

#: Below this wall-clock duration a SUCCEEDED execution of the v1 weekly state
#: machine is a `WeeklyRunDayGate` Succeed-skip, not a run.
#:
#: **Measured 2026-09-04**, the pair the ruling names: Thursday 2026-09-03
#: started 02:00:49.259 PT and stopped 02:00:52.275 PT SUCCEEDED — **3.0
#: seconds**, the gate skipping a non-run-day; Saturday 2026-08-29 started
#: 02:00:49.233 PT and stopped 07:02:40.833 PT — **5h 01m**, the real cycle.
#: Three orders of magnitude separate them, so 10s sits in an empty band rather
#: than near either population; it is not a tuned figure and nothing is close
#: enough to it for a slower gate evaluation to flip a reading.
#:
#: **Why a duration proxy and not the gate's own state.** The precise answer is
#: in each execution's history (`WeeklyRunDayGate` -> `Succeed` versus the run
#: branch), which needs `states:GetExecutionHistory`. The producer's OIDC role
#: (`crucible-v2-legacy-weekly-producer`, defined in
#: `nous-ergon-ops/infrastructure/cloudformation/crucible-v2.yaml`) holds
#: `states:ListExecutions` and nothing else, and IAM in that template is an
#: operator-applied security boundary — widening it puts the reading behind
#: that apply (`pull-request-policy` §4.2 form 3), which is a real cost even
#: though it is a legitimate form. (Corrected 2026-09-04, `I9964`: an earlier
#: revision of this comment called a widening "undeployable by the merge
#: button alone", which overstated it — that template's whole apply path is
#: form 3, and `alpha-engine-config-I9964` does widen this role, for the
#: routing fact `ListExecutions` cannot answer at all. The proxy still stands:
#: it costs no extra call for the fact it answers.) `ListExecutions` already
#: returns
#: `startDate`, `stopDate`, `status` and `name` for every execution, so the
#: proxy costs no extra call and no extra grant. If the grant is ever widened,
#: the producer should file the gate state and this constant becomes a
#: fallback, not a deletion: an execution still RUNNING has no duration at all.
WEEKLY_RUN_DAY_GATE_SKIP_MAX_SECONDS = 10.0

#: An execution whose name starts with this was issued by the sf-watch rerun
#: path — the operator cost phase 0 deliverable 1 exists to remove (thirteen
#: `watch-rerun-2026-08-28-*` starts on 2026-08-30, twelve of them FAILED). It
#: FAILS the clause regardless of duration: the name identifies the issuer, and
#: a rerun that happened to be short is still a rerun.
LEGACY_WEEKLY_RERUN_NAME_PREFIX = "watch-rerun-"

#: The week a `watch-rerun-*` execution names, as the sf-watch rerun path
#: spelled it: `watch-rerun-YYYY-MM-DD-N`, where the date is the week being
#: retried and `N` is the attempt number.
LEGACY_WEEKLY_RERUN_NAME_RE = re.compile(
    rf"^{re.escape(LEGACY_WEEKLY_RERUN_NAME_PREFIX)}(\d{{4}}-\d{{2}}-\d{{2}})-\d+$"
)


def rerun_names_another_week(name: object, anchor: dt.date) -> bool:
    """True when `name` is a `watch-rerun-*` naming a week other than `anchor`.

    ONE implementation, called by both `_clause_old_weekly_within_cadence`
    (through `_LegacyWeeklyExecution`) and `_clause_old_alerts_muted` (on the
    raw filed entry, which it reads as dicts). Two clauses that disagreed
    about which executions belong to a week would grade two different weeks
    under one anchor, and this repository has already found one reader
    drifting from its twin inside the change that introduced it.

    Anything that is not a parseable rerun name returns False, so the
    execution stays attributed to the week it was FILED under. That is the
    deliberate direction: an unparseable name must fail somewhere, and the
    week it was filed under is the only week guaranteed to read it.
    """
    if not isinstance(name, str) or not name.startswith(LEGACY_WEEKLY_RERUN_NAME_PREFIX):
        return False
    match = LEGACY_WEEKLY_RERUN_NAME_RE.match(name)
    if match is None:
        return False
    try:
        return dt.date.fromisoformat(match.group(1)) != anchor
    except ValueError:  # a name-shaped string that is not a real date
        return False


#: The schema the dead-Lambda probe must declare. Same refusal rule as the
#: weekly executions document: an unrecognised version is UNMEASURABLE, never
#: a pass. Producer:
#: `nous-ergon-ops/scripts/legacy_dead_lambdas_producer.py`.
LEGACY_DEAD_LAMBDAS_SCHEMA_VERSION = "legacy-dead-lambdas.v1"

#: The six zero-invocation v1 Lambda functions `alpha-engine-config-I9756`'s
#: second deliverable deletes, by EXACT name.
#:
#: **The prefix trap, stated because it is one edit away.**
#: `alpha-engine-research-eval-judge-process`, `-poll`, `-submit` and
#: `-spot-dispatcher` all EXIST and are live, different functions (measured
#: 2026-09-04). Every one of them starts with the fifth name here. A clause
#: matching on prefix, substring, or a `startswith` over a live function list
#: would find four survivors and report this deliverable UNMET forever against
#: a system that satisfies it — the mirror image of the defect this gate keeps
#: finding, a detector that can never go green rather than one that can never
#: go red. So the contract is per exact name on both sides: the producer files
#: one entry per name, and :func:`_clause_dead_lambdas_deleted` refuses a
#: document that does not cover this exact set.
LEGACY_DEAD_LAMBDA_NAMES: tuple[str, ...] = (
    "alpha-engine-ci-watch-liveness-probe",
    "alpha-engine-ec2-lifecycle",
    "alpha-engine-research-eval-judge",
    "alpha-engine-research-perturbation-battery",
    "alpha-engine-research-thinktank",
    "alpha-engine-sf-watch-reclaim-sweep-handler",
)


#: The SNS topic the v1 weekly pipeline's alerts must land in for
#: `alpha-engine-config-I9756`'s third deliverable to hold. Matched on the
#: ARN's last segment, never on the whole ARN: the account id is an
#: infrastructure identifier this repo forbids (`tests/test_no_infra_literals.py`),
#: and the topic NAME is the part that carries the meaning. The same literal
#: already lives in `crucible/alerts.py`.
def muted_alerts_topic_name() -> str:
    """The v1 muted topic's NAME, read from the environment.

    Was a literal until Brian's 2026-09-07 ruling (`alpha-engine-config-I10156`);
    delegates to `crucible.alerts` so the two cannot drift, which is what the
    old comment here promised and a second literal could not deliver.
    """
    from crucible.alerts import muted_topic

    return muted_topic()


#: The per-execution field carrying the topic that execution's INPUT named.
#: Presence of the FIELD, not the document's `schema_version`, is what
#: :func:`_clause_old_alerts_muted` keys on — see the field's own description
#: in `crucible/schemas/legacy_weekly_executions.v2.json` for why it was added
#: without a version bump.
LEGACY_WEEKLY_TOPIC_FIELD = "sns_topic_arn"

#: The ONE §2 acceptance clause `alpha-engine-config-I9756`'s fourth
#: deliverable declares as its reading surface for the cost-attribution tag.
#: Named here rather than re-derived: the deliverable table has said since it
#: was written that this clause is what grades that half, and
#: :func:`_clause_v2_resources_tagged_and_versioned` reads exactly it.
V2_TAG_ACCEPTANCE_CLAUSE_ID = "TestCost::test_every_v2_resource_is_tagged_for_cost_attribution"

#: The S3 versioning `Status` the store must report for the other half of that
#: deliverable. `Suspended`, and a bucket that was never versioned (which
#: reports no status at all), are both UNMET; an absent reading is
#: UNMEASURABLE.
V2_STORE_VERSIONING_ENABLED = "Enabled"


def weekly_anchor(day: dt.date) -> dt.date:
    """The session keying the most recent week whose count can already be filed.

    **A weekly artifact may not be keyed on the day somebody rendered the
    gate.** `_window` steps back in raw calendar weeks from whatever day the
    caller passed, and the ladder is rendered DAILY, so a raw-day key sends
    Monday's read and Tuesday's read to two different objects. A producer
    filing one document per week would then make phase 0 read MET on one
    weekday and UNMET on the other four, forever — the contract would be
    unsatisfiable rather than merely unmet.

    So every window day collapses onto the **Friday close STRICTLY BEFORE it**,
    resolved through the calendar so a holiday Friday walks back to a real
    session (§4.12: every key is a trading day). Strictly before, not
    on-or-before: the week ending this Friday is not counted until its close
    has passed and the weekly producer has run, so anchoring to it would make
    the newest week absent for a day and flap once a week instead of four times
    a week.
    """
    days_back = ((day.weekday() - 4) % 7) or 7
    friday = day - dt.timedelta(days=days_back)
    return resolve_trading_day(dt.datetime.combine(friday, dt.time(23, 59)))


#: The optional per-document `window` object's key names
#: (`crucible/schemas/legacy_weekly_executions.v2.json`), named once so a
#: reader and a producer can never spell the shape differently.
LEGACY_WEEKLY_WINDOW_FIELD = "window"


def expected_legacy_weekly_window(anchor: dt.date) -> tuple[str, str]:
    """The one true span a `legacy/weekly/{anchor}/executions.json` document
    should declare it collected over — the Sunday of the anchor's week
    through the Saturday that measures it.

    `nous-ergon-ops-PR1034` set the producer's window to exactly this span
    after `alpha-engine-config-I9983` (a wrong window, `anchor-6..anchor`,
    collecting every off-day Succeed-skip of the anchor's week and none of
    its runs). The `+1` is load-bearing, not a rounding choice:
    `WeeklyRunDayGate` passes the day AFTER a week's last trading session, so
    the run measuring the week ending `anchor` starts on `anchor + 1` — the
    Saturday `alpha-engine-saturday`'s `cron(0 9 ? * THU-SAT *)` targets.
    `anchor - 5` is the Sunday that opens that same calendar week (`anchor`
    is always a Friday — see `weekly_anchor`).

    Derived here, in ONE place, so `_clause_old_weekly_within_cadence` and
    `_clause_old_alerts_muted` — the two readers of this document — cannot
    restate the arithmetic and drift from each other or from the producer.
    Calendar dates, not trading-day keys: this is bookkeeping about WHEN the
    producer queried, not an input to a promotion, retirement, freshness or
    grading decision (§3's exhaustive exceptions already cover a collection
    window).
    """
    start = anchor - dt.timedelta(days=5)
    end = anchor + dt.timedelta(days=1)
    return start.isoformat(), end.isoformat()


def _legacy_weekly_window_mismatch(document: dict[str, Any], anchor: dt.date) -> str | None:
    """`None` when `window` is absent (nothing to check) or agrees with
    `anchor`; otherwise a message naming both spans.

    **Presence-keyed, like `sns_topic_arn`, never version-keyed.** A document
    with no `window` at all is unaffected — it is read exactly as it was
    before this field existed, per the schema's own description of why this
    was not a `schema_version` bump. A document that DOES declare a window
    is checked before a single count in it is trusted: a wrong window is a
    broken reading, not a finding about the pipeline
    (`alpha-engine-config-I9983` was found by reading the producer, not by
    any detector — this is that detector).
    """
    window = document.get(LEGACY_WEEKLY_WINDOW_FIELD)
    if window is None:
        return None
    expected_start, expected_end = expected_legacy_weekly_window(anchor)
    if not isinstance(window, dict):
        return f"`window` is {window!r}, not an object with start/end"
    declared_start, declared_end = window.get("start"), window.get("end")
    if (declared_start, declared_end) == (expected_start, expected_end):
        return None
    return (
        f"`window` declares {declared_start!r}..{declared_end!r}, expected "
        f"{expected_start!r}..{expected_end!r} for anchor {anchor.isoformat()} — a wrong "
        "window is a broken reading, not a finding about the pipeline"
    )


#: The committed reading of the plan §2 acceptance suite. Not a store artifact:
#: the suite's existence is a property of the REPOSITORY, and git is the
#: durable record of it. `tests/acceptance/ratchet.json` commits the exact id
#: sets, `tests/test_acceptance_reading.py` fails when they drift from what the
#: suite collects, and `ci.yml`'s acceptance job fails a push to `main` on any
#: movement. Read from a checkout; absent (a wheel install, where `tests/` does
#: not ship) the clause is UNMET with the path named, never a pass.
ACCEPTANCE_RATCHET_PATH = (
    Path(__file__).resolve().parent.parent / "tests" / "acceptance" / "ratchet.json"
)

#: The suite the ratchet claims to describe. Read as SOURCE, not as a committed
#: summary of source: a ratchet beside a deleted suite parses perfectly and
#: names 24 clauses that no longer exist anywhere.
ACCEPTANCE_SUITE_DIR = ACCEPTANCE_RATCHET_PATH.parent


@dataclass(frozen=True)
class SourceScan:
    """What an `ast` walk of the acceptance suite could see.

    ``problems`` is non-empty when a module could not be parsed at all — a
    `SyntaxError` there must be a red clause naming the file, not an
    exception out of `evaluate`.
    """

    ids: set[str]
    module_level_tests: list[str]
    problems: list[str]


#: Exactly what the source scan can see, published in the clause's own
#: requirement so a future clause author is never surprised by a red gate.
#: Written as a sentence rather than left implicit: an id-set comparison is
#: only as good as the collection model behind it, and pytest's is wider than
#: any static parse.
SOURCE_SCAN_SCOPE = (
    "the scan reads `Test*` classes and the `test_*` methods they define or "
    "inherit from a base class in the same suite, plus module-level `test_*` "
    "functions; methods attached at runtime (setattr, a metaclass, a generated "
    "class) are invisible to any static parse and must not be relied on here"
)


def _acceptance_source_scan(directory: Path) -> SourceScan:
    """Every `Class::method` acceptance clause DEFINED under ``directory``.

    Parsed with `ast`, never imported and never collected: importing the suite
    would execute module-level code that reaches live AWS, and a gate does not
    run the thing it grades. Parametrised ids (`...[case]`) are compared on
    their base, which is all a source parse can see.

    **Inheritance is resolved.** `class TestChild(TestBase)` collects
    `TestChild::test_inherited` under pytest, and a scan that reported only
    `TestBase::test_inherited` would flip this clause to UNMET against a
    perfectly correct ratchet — measured on a probe file during review. Bases
    are resolved transitively across every module in the directory; a base is
    matched by its bare name, which is what a suite that keeps its bases beside
    its cases actually writes.

    Module-level `test_*` functions are returned separately: pytest collects
    them, but the ratchet's id grammar is `Class::method`, so they cannot be
    recorded at all. That is a finding, not an invisible gap.
    """
    own: dict[str, set[str]] = {}
    bases: dict[str, list[str]] = {}
    module_level: list[str] = []
    problems: list[str] = []
    for path in sorted(directory.glob("test_*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except Exception as exc:
            # The failure mode: a module of the graded suite cannot be parsed.
            # Recorded on the clause returned below, never raised, because an
            # exception here would take the whole ladder render with it.
            problems.append(f"{path.name} could not be parsed: {type(exc).__name__}: {exc}")
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name.startswith(
                "test_"
            ):
                module_level.append(f"{path.name}::{node.name}")
                continue
            if not isinstance(node, ast.ClassDef):
                continue
            own.setdefault(node.name, set()).update(
                item.name
                for item in node.body
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef)
                and item.name.startswith("test_")
            )
            bases.setdefault(node.name, []).extend(
                base.id for base in node.bases if isinstance(base, ast.Name)
            )

    def methods(name: str, seen: frozenset[str]) -> set[str]:
        if name in seen or name not in own:
            return set()
        collected = set(own[name])
        for base in bases.get(name, []):
            collected |= methods(base, seen | {name})
        return collected

    ids = {
        f"{cls}::{method}"
        for cls in own
        if cls.startswith("Test")
        for method in methods(cls, frozenset())
    }
    return SourceScan(ids, module_level, problems)


@dataclass(frozen=True)
class _LegacyWeeklyExecution:
    """One v1 weekly execution as the producer filed it.

    Classification lives HERE, not in the producer: the producer states facts
    (`states:ListExecutions` gave it a name, a status and two timestamps) and
    this repository owns what those facts MEAN for the gate. A producer that
    filed `gate_passing: true` would move the ruled metric into a repository
    the gate cannot test, and the threshold would then be un-reviewable from
    the clause that grades against it.
    """

    name: str
    status: str
    duration_seconds: float | None

    @property
    def is_rerun(self) -> bool:
        return self.name.startswith(LEGACY_WEEKLY_RERUN_NAME_PREFIX)

    @property
    def rerun_target_week(self) -> dt.date | None:
        """The week this rerun RETRIES, from its own name — or `None`.

        `None` covers both "not a rerun" and "a rerun whose name this reader
        cannot parse". The two are distinguished by `is_rerun`, and the
        callers treat an unparseable rerun as belonging to the week it was
        filed under, which is the reading that fails loudly rather than the
        one that lets an execution escape every clause.
        """
        if not self.is_rerun:
            return None
        match = LEGACY_WEEKLY_RERUN_NAME_RE.match(self.name)
        if match is None:
            return None
        try:
            return dt.date.fromisoformat(match.group(1))
        except ValueError:  # a name-shaped string that is not a real date
            return None

    def reruns_a_week_other_than(self, anchor: dt.date) -> bool:
        """True when this is a rerun that names some OTHER week than `anchor`.

        Delegates to `rerun_names_another_week` so the routing clause, which
        reads raw filed entries, cannot drift from this one. See
        `_clause_old_weekly_within_cadence` for why the distinction is
        load-bearing and what it deliberately stops detecting.
        """
        return rerun_names_another_week(self.name, anchor)

    @property
    def is_gate_skip(self) -> bool:
        """A `WeeklyRunDayGate` Succeed-skip: the fail-open margin, not a run.

        A rerun is never a skip however short it ran, and an execution that is
        still RUNNING (no `stopDate`, so no duration) is never a skip either —
        `None` is not "fast", and treating an unfinished execution as a skip
        would let a run in flight be excluded from the count that grades it.
        """
        return (
            not self.is_rerun
            and self.status == "SUCCEEDED"
            and self.duration_seconds is not None
            and self.duration_seconds < WEEKLY_RUN_DAY_GATE_SKIP_MAX_SECONDS
        )


def _legacy_weekly_executions(
    key: str, document: dict[str, Any]
) -> tuple[str | None, list[_LegacyWeeklyExecution] | None]:
    """Parse one filed week, or say why it cannot be parsed.

    Every field is checked before it is used. This document crosses a repo
    boundary — `nous-ergon-ops` writes it, `crucible` grades it — so a shape
    this reader does not understand is a red reading, never an exception out
    of `evaluate` that would take the whole ladder and the board render down.
    """
    entries = document.get("executions")
    if not isinstance(entries, list):
        return (f"{key}: `executions` is {entries!r}, not a list of executions", None)
    out: list[_LegacyWeeklyExecution] = []
    for index, entry in enumerate(entries):
        where = f"{key}: executions[{index}]"
        if not isinstance(entry, dict):
            return (f"{where} is {entry!r}, not an object", None)
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            return (f"{where}: `name` is {name!r}, not an execution name", None)
        status = entry.get("status")
        if not isinstance(status, str) or not status:
            return (f"{where}: `status` is {status!r}, not a status", None)
        duration = entry.get("duration_seconds")
        if duration is not None and (
            isinstance(duration, bool) or not isinstance(duration, int | float) or duration < 0
        ):
            return (f"{where}: `duration_seconds` is {duration!r}, not a duration or null", None)
        out.append(
            _LegacyWeeklyExecution(
                name=name,
                status=status,
                duration_seconds=None if duration is None else float(duration),
            )
        )
    # The v1 field stays in the v2 document for backward reading, so the two
    # must agree. A document whose count and whose list disagree is answering
    # the same question twice with two answers, and picking one silently is
    # how a producer bug survives to grade a phase exit.
    started = document.get("executions_started")
    if isinstance(started, bool) or not isinstance(started, int) or started < 0:
        return (f"{key}: `executions_started` is {started!r}, not a count", None)
    if started != len(out):
        return (
            f"{key}: `executions_started` is {started} while `executions` lists "
            f"{len(out)}; the document disagrees with itself",
            None,
        )
    return (None, out)


def _clause_old_weekly_within_cadence(
    store: Store,
    window: list[dt.date],
    *,
    name: str = "old_weekly_within_cadence",
    maximum: int | None = LEGACY_WEEKLY_MAX_STARTS_PER_WEEK,
    minimum: int = LEGACY_WEEKLY_MIN_RUNS_PER_WEEK,
    skips_count_as_runs: bool = False,
    minimum_succeeded: int = 0,
    reruns_fail: bool = True,
) -> Clause:
    """The v1 weekly cycle count, read from a filed document, against a ceiling.

    **Phase 0's exit is a SUCCESSFUL run, and reruns are how it gets one**
    (Brian's ruling, 2026-09-04 evening, on `alpha-engine-config-I9756`:
    *"assume the weekly sf tomorrow will fail. we will fix and rerun until it
    is successful, and that needs to be the trigger to complete phase 0."*).
    The last three Saturday canonical runs — 08-15, 08-22, 08-29 — all
    FAILED, each for a different reason, so a clause that graded EXACTLY one
    gate-passing execution would have exited phase 0 on a week whose only
    run failed, and would have FAILED the week in which the pipeline was
    repaired and re-run to success. Two parameters carry the ruling, both
    passed explicitly at phase 0's call site and left at their defaults by
    phase 4:

    * ``minimum_succeeded`` — how many of the week's gate-passing executions
      must have ``status == "SUCCEEDED"``. Phase 0 passes 1: the trigger is
      a cycle that WORKED, not one that started. A week of failures reads
      UNMET naming every failed run, never MET on the count alone.
    * ``reruns_fail`` — whether a `watch-rerun-*` execution naming THIS week
      fails the clause. Phase 0 passes ``False``: the automated rerun issuer
      is graded by `_clause_dead_lambdas_deleted` (absent by exact name), and
      a rerun in the graded week is now Brian repairing the pipeline, which
      the ruling asks for. Reruns are still NAMED in the detail on every
      reading. Phase 4 keeps ``True``: a decommissioned pipeline emits
      nothing, reruns included.

    ``maximum=None`` removes the ceiling, which is what "rerun until it is
    successful" means; phase 4 keeps ``0``.

    **What is counted changed on 2026-09-04** (Brian's ruling on
    `alpha-engine-config-I9756`; implemented under
    `alpha-engine-config-I9962`). The clause used to count raw
    `StartExecution`, under which a 3.0-second `WeeklyRunDayGate` Succeed-skip
    and a five-hour real cycle are the same event. `alpha-engine-saturday`'s
    `cron(0 9 ? * THU-SAT *)` fires three times a week BY DESIGN — the
    self-select is the holiday fail-open margin `sf-pipeline-policy.md` §5
    line 213 states outright ("missing a weekly run is worse than a duplicate;
    the mutex handles duplicates") — so a raw-start ceiling of 1 could only be
    satisfied by deleting a safety property. That cron is now the INTENDED
    configuration, not drift.

    What is graded instead:

    * an execution that PASSED `WeeklyRunDayGate` is a run and counts;
    * a SUCCEEDED execution shorter than
      `WEEKLY_RUN_DAY_GATE_SKIP_MAX_SECONDS` is a Succeed-skip and does not
      (see that constant for the measurement behind the threshold, and for why
      duration is the proxy rather than the gate's own state);
    * any `watch-rerun-*` execution FAILS the clause regardless of duration —
      that name identifies a rerun ISSUER, which is what deliverable 1
      disables, and a short rerun is still a rerun.

    **A `v1` document is UNMEASURABLE, never a pass.** A week filed before
    this schema carries one integer and cannot answer the ruled question at
    all. Reading it as met would grade the retired metric under the new
    clause's name; reading it as unmet would report a finding against a system
    behaving exactly as ruled. `[?]` with the reason named is the only honest
    third answer (module docstring; principle 7).

    ``maximum`` is a parameter because plan §6 asks the SAME question at two
    ceilings: phase 0 tolerates one cycle a week while the v1 system is being
    quieted, and phase 4 requires ZERO once it is decommissioned. A second
    function for the second ceiling would be the same reader restated, and
    this repository has already found one of those drifting inside the change
    that introduced it (`_review_problem`). ``name`` travels with it so the
    two readings do not both land on the ladder under one clause name.

    ``minimum`` is the LOWER bound, and it is the half that makes phase 0's
    one-week window safe. The ruled predicate says "at most one execution that
    passes `WeeklyRunDayGate` per calendar week"; read literally, a week with
    ZERO gate-passing executions satisfies it — while meaning the weekly
    pipeline never ran, the failure `sf-pipeline-policy.md` §5 line 213 calls
    worse than a duplicate. A window of one week makes that a single bad
    Saturday away, so phase 0 grades EXACTLY one run a week and a silent week
    reads UNMET with the missing run named. Not MET; and not UNMEASURABLE
    either — "we read the document and it lists no gate-passing execution" is
    a real reading, distinct from "we could not obtain the document", which
    this clause already reports separately (`alpha-engine-config-I9962`).

    Phase 4's correct lower bound is ZERO: it grades a DECOMMISSIONED
    pipeline, where no executions at all is the pass. So ``minimum`` is
    explicitly passed at BOTH call sites and is never inferred from
    ``maximum == 0`` — for the same reason ``skips_count_as_runs`` is not: the
    two phases must differ where they are CALLED, in each phase's own clause
    list, and a reader that branches on another parameter's value hides one
    phase's semantics inside the other's.

    **A rerun is graded against the week it RETRIES, not the week it ran in**
    (`alpha-engine-config-I9756`, 2026-09-04). The sf-watch rerun path names
    its executions `watch-rerun-YYYY-MM-DD-N`, where the date is the week
    being retried, so the execution declares its own subject. Thirteen
    `watch-rerun-2026-08-28-*` executions ran on Sunday 2026-08-30 — retries
    of the 08-28 cycle's failure — and by start date they land in the week
    ending Friday 2026-09-04. Failing THAT week on them reports a finding
    about a week in which nothing went wrong; the reruns are evidence about
    2026-08-28.

    No 7-day window that tiles the calendar and is complete when it is filed
    can separate them, which is why the fix is here and not in the producer's
    window: the fact that distinguishes them is the name, and names are what
    this repository classifies (see `_LegacyWeeklyExecution`).

    **What this deliberately stops detecting, stated so it can be reversed.**
    A rerun of an EARLIER week now fails no week's clause: the week it names
    was already graded and its document, keyed by start date, does not contain
    it. So a rerun issuer that only ever retried old weeks would be invisible
    HERE. It is not invisible: `_clause_dead_lambdas_deleted` grades the
    issuer itself — `alpha-engine-sf-watch-reclaim-sweep-handler` must be
    absent by exact name — which is the direct measurement this clause only
    ever proxied. Every such execution is also named in this clause's detail
    on both a MET and an UNMET reading, so it appears on the ladder either
    way. A rerun naming THIS week still fails it, and an unparseable rerun
    name is attributed to the week it was filed under, which fails loudly
    rather than letting an execution escape every clause.

    ``skips_count_as_runs`` keeps phase 4's meaning intact across this change.
    Phase 4 asks whether the v1 pipeline is DECOMMISSIONED, and a
    decommissioned state machine emits no executions at all — a surviving
    Succeed-skip means the trigger is still live and still firing. Excluding
    skips there would have silently weakened phase 4 while fixing phase 0, so
    phase 4 passes ``skips_count_as_runs=True`` and keeps counting every
    start. It is an explicit argument rather than a branch on ``maximum == 0``
    so the two readings differ where they are CALLED, in the phase's own
    clause list, rather than inside a reader neither phase names.
    """
    if maximum is not None and minimum > maximum:
        # A bound pair that can never be satisfied would read UNMET on every
        # week forever with a reason that looks like a finding about the
        # pipeline. Refuse at the call site instead.
        raise ClauseMisconfiguredError(
            f"minimum {minimum} exceeds maximum {maximum}; no week can satisfy this clause"
        )
    if minimum_succeeded > minimum:
        raise ValueError(
            f"minimum_succeeded {minimum_succeeded} exceeds minimum {minimum}; a succeeded "
            "execution is a gate-passing execution, so the floor on successes cannot "
            "exceed the floor on runs"
        )
    counted = "execution of any kind" if skips_count_as_runs else "execution that PASSED"
    requirement = (
        "the v1 weekly state machine started "
        + (f"at most {maximum} " if maximum is not None else "any number of ")
        + counted
        + ("" if skips_count_as_runs else " `WeeklyRunDayGate`")
        + (f" — and at least {minimum} — " if minimum else " ")
        + "in EACH week of the window"
        + (
            f", of which at least {minimum_succeeded} SUCCEEDED (a fix-and-rerun cycle "
            "that ended in success is the pass; Brian ruling 2026-09-04)"
            if minimum_succeeded
            else ""
        )
        + (
            ", and no `watch-rerun-*` execution at all"
            if reruns_fail
            else " (reruns permitted and named)"
        )
        + ", read from a filed per-execution record keyed on the week, not on the day "
        "the gate was read"
    )
    anchors = list(dict.fromkeys(weekly_anchor(day) for day in window))
    evidence = [legacy_weekly_executions_key(a.isoformat()) for a in anchors]
    if len(anchors) != len(window):
        return Clause(
            name,
            requirement,
            False,
            f"{len(window)} window weeks collapsed onto {len(anchors)} week anchors "
            f"({', '.join(a.isoformat() for a in anchors)}); one document would be "
            "graded twice",
            tuple(evidence),
        )
    missing: list[str] = []
    malformed: list[str] = []
    stale: list[str] = []
    over: list[str] = []
    under: list[str] = []
    unsucceeded: list[str] = []
    reruns: list[str] = []
    rerun_notes: list[str] = []
    elsewhere: list[str] = []
    skipped_total = 0
    for anchor, key in zip(anchors, evidence, strict=True):
        # Read through the guarded reader, never `json.loads` + indexing. This
        # document is written by a producer outside this repository, and an
        # exception here does not fail one clause: it propagates out of
        # `evaluate` and takes `crucible gate`, `build_ladder` AND the board
        # render down together. An unreadable input is a red reading on the
        # surface, never an absence from it.
        read = _read_store_document(store, key)
        if read.problem is not None:
            malformed.append(read.problem)
            continue
        if read.absent:
            missing.append(key)
            continue
        document = read.document or {}
        version = document.get("schema_version")
        if version != LEGACY_WEEKLY_EXECUTIONS_SCHEMA_VERSION:
            stale.append(
                f"{key}: schema_version is {version!r}, not "
                f"{LEGACY_WEEKLY_EXECUTIONS_SCHEMA_VERSION!r} — it carries only a raw "
                "start count, which cannot tell a WeeklyRunDayGate Succeed-skip from a "
                "run. This week cannot answer the metric ruled on 2026-09-04; it is "
                "unmeasurable, not a pass"
            )
            continue
        window_mismatch = _legacy_weekly_window_mismatch(document, anchor)
        if window_mismatch is not None:
            stale.append(f"{key}: {window_mismatch}")
            continue
        problem, executions = _legacy_weekly_executions(key, document)
        if problem is not None or executions is None:
            malformed.append(problem or f"{key}: unreadable")
            continue
        # A rerun is attributed to the week its NAME retries, not to the week
        # its StartDate fell in. `alpha-engine-config-I9756`, 2026-09-04.
        foreign = [e for e in executions if e.reruns_a_week_other_than(anchor)]
        if foreign:
            elsewhere.append(
                f"{key}: {len(foreign)} watch-rerun execution(s) naming other week(s) "
                f"({', '.join(sorted({str(e.rerun_target_week) for e in foreign}))}) — "
                "attributed there, not graded against this week"
            )
        executions = [e for e in executions if not e.reruns_a_week_other_than(anchor)]
        week_reruns = [e for e in executions if e.is_rerun]
        if week_reruns and reruns_fail:
            reruns.append(
                f"{key}: {len(week_reruns)} watch-rerun execution(s) "
                f"({', '.join(e.name for e in week_reruns[:3])}) — a rerun issuer fails "
                "regardless of duration"
            )
        elif week_reruns:
            rerun_notes.append(
                f"{key}: {len(week_reruns)} watch-rerun execution(s) "
                f"({', '.join(e.name for e in week_reruns[:3])}) — permitted under the "
                "2026-09-04 fix-and-rerun ruling, named here"
            )
        skips = [e for e in executions if e.is_gate_skip]
        skipped_total += len(skips)
        runs = executions if skips_count_as_runs else [e for e in executions if not e.is_gate_skip]
        if maximum is not None and len(runs) > maximum:
            over.append(
                f"{key}: {len(runs)} {'start' if skips_count_as_runs else 'gate-passing execution'}"
                f"{'s' if len(runs) != 1 else ''}, ceiling {maximum} "
                f"({len(skips)} sub-{WEEKLY_RUN_DAY_GATE_SKIP_MAX_SECONDS:g}s Succeed-skip"
                f"{'s' if len(skips) != 1 else ''} excluded)"
            )
        if len(runs) < minimum:
            # The document was read and it lists no run. That is a READING,
            # not an inability to measure, so it is UNMET with the missing
            # run named — never `[?]`, and never MET on the ruled predicate's
            # literal upper bound.
            under.append(
                f"{key}: {len(runs)} "
                f"{'start' if skips_count_as_runs else 'gate-passing execution'}"
                f"{'s' if len(runs) != 1 else ''}, floor {minimum} — the weekly pipeline "
                f"did not run this week ({len(skips)} sub-"
                f"{WEEKLY_RUN_DAY_GATE_SKIP_MAX_SECONDS:g}s Succeed-skip"
                f"{'s' if len(skips) != 1 else ''} present, which is the gate declining "
                "the day, not a cycle). A missing weekly run is worse than a duplicate "
                "(`sf-pipeline-policy.md` §5)"
            )
        if minimum_succeeded:
            succeeded = [e for e in runs if e.status == "SUCCEEDED"]
            if len(runs) >= minimum and len(succeeded) < minimum_succeeded:
                # Every run this week FAILED (or was still running when the
                # producer filed). The ruling's trigger is a cycle that
                # worked: fix the pipeline and re-run it; the next filing of
                # this week's document carries the successful execution.
                failed = [e for e in runs if e.status != "SUCCEEDED"]
                unsucceeded.append(
                    f"{key}: {len(runs)} gate-passing execution(s), "
                    f"{len(succeeded)} SUCCEEDED (floor {minimum_succeeded}) — "
                    + ", ".join(f"{e.name}: {e.status}" for e in failed[:4])
                    + ". Phase 0 exits on a SUCCESSFUL run: fix and rerun "
                    "(Brian ruling 2026-09-04)"
                )
    if missing or malformed or stale or over or under or unsucceeded or reruns:
        parts: list[str] = []
        if missing:
            parts.append(
                f"{len(missing)} of {len(evidence)} weeks have no filed count: "
                f"{', '.join(missing)}. Nothing writes this key yet; a gate may not "
                "call `states:ListExecutions` to find out"
            )
        if malformed:
            parts.append(f"{len(malformed)} malformed: {'; '.join(malformed)}")
        if stale:
            parts.append(f"{len(stale)} unmeasurable: {'; '.join(stale)}")
        if reruns:
            parts.append("; ".join(reruns))
        if over:
            parts.append("; ".join(over))
        if under:
            parts.append("; ".join(under))
        if unsucceeded:
            parts.append("; ".join(unsucceeded))
        if rerun_notes:
            parts.append("; ".join(rerun_notes))
        if elsewhere:
            # Named even on a FAILING reading: an execution the clause chose
            # not to grade here must never be invisible, whichever way the
            # clause lands.
            parts.append("; ".join(elsewhere))
        return Clause(
            name,
            requirement,
            False,
            "; ".join(parts),
            tuple(evidence),
            # UNMEASURABLE only when nothing else is wrong. A window with one
            # stale week and one week genuinely over the ceiling has a finding
            # in it, and `[?]` would hide the finding behind the stale read.
            unmeasurable=bool(stale)
            and not (missing or malformed or over or under or unsucceeded or reruns),
        )
    counted_noun = "start" if skips_count_as_runs else "gate-passing execution"
    bound = (
        (f"{minimum}-{maximum}" if minimum else f"<= {maximum}")
        if maximum is not None
        else f">= {minimum}"
    )
    return Clause(
        name,
        requirement,
        True,
        f"{len(evidence)} consecutive weeks at {bound} {counted_noun} each"
        + (f", at least {minimum_succeeded} SUCCEEDED" if minimum_succeeded else "")
        + (", no watch-rerun executions" if reruns_fail else "")
        + f" ({skipped_total} sub-"
        f"{WEEKLY_RUN_DAY_GATE_SKIP_MAX_SECONDS:g}s Succeed-skip"
        f"{'s' if skipped_total != 1 else ''} "
        f"{'counted' if skips_count_as_runs else 'excluded'})"
        + ("; " + "; ".join(rerun_notes) if rerun_notes else "")
        + ("; " + "; ".join(elsewhere) if elsewhere else ""),
        tuple(evidence),
    )


def _weekly_anchors(window: list[dt.date]) -> list[dt.date]:
    """``window``'s days collapsed onto their weekly closes, order preserved.

    The same collapse `_clause_old_weekly_within_cadence` does inline, lifted
    so phase 0's three v1 clauses read ONE key set. Two clauses stepping back
    by two different rules would grade two different weeks and report them
    under one phase.
    """
    return list(dict.fromkeys(weekly_anchor(day) for day in window))


def _clause_dead_lambdas_deleted(store: Store, window: list[dt.date]) -> Clause:
    """The six zero-invocation v1 functions are gone, read from a filed probe.

    `alpha-engine-config-I9756`'s second deliverable. Until
    `alpha-engine-config-I9964` this deliverable was declared *not
    gate-readable* on the argument that "the only surface that answers it is a
    live AWS read, which a gate may not make" — true about the READ, and a
    non-sequitur about the CLAUSE. A gate may not call AWS; a producer holding
    the identity may, and the gate reads what it filed. That is exactly the
    shape `_clause_old_weekly_within_cadence` already had, so the deliverable
    was ungraded for want of a producer, not for want of a clause. Phase 0 was
    therefore on course to exit on gate evidence covering two of its five
    deliverables, with the other three resting on somebody having checked by
    hand and said so.

    **`present: true` is the only UNMET.** Everything else that can go wrong
    here is a statement about our reading, not about the system:

    * the document is absent, or the store refused the read — UNMEASURABLE;
    * it declares a schema version this clause does not recognise, or its
      shape is unreadable — UNMEASURABLE, because a document this clause
      cannot parse has not told it anything about the six functions;
    * it does not cover exactly :data:`LEGACY_DEAD_LAMBDA_NAMES` —
      UNMEASURABLE naming the difference. A probe of five of the six answers a
      narrower question than the deliverable asks, and grading it MET would
      publish the narrower answer under the wider name.

    A producer that could not decide presence for a name does not file
    `present: false`: it raises, and the document's ABSENCE is the reading
    (see the schema's `present` description). That is what keeps an
    AccessDenied from arriving here as a met deliverable.
    """
    requirement = (
        f"none of the {len(LEGACY_DEAD_LAMBDA_NAMES)} zero-invocation v1 Lambda functions "
        f"named by `{phase_tracker('phase0')}` still exists, read per EXACT name from a "
        "filed probe — never by prefix, which four live sibling functions would match"
    )
    anchors = _weekly_anchors(window)
    evidence = [legacy_dead_lambdas_key(a.isoformat()) for a in anchors]
    required = set(LEGACY_DEAD_LAMBDA_NAMES)
    unreadable: list[str] = []
    surviving: list[str] = []
    probed_total = 0
    for key in evidence:
        read = _read_store_document(store, key)
        if read.problem is not None:
            unreadable.append(read.problem)
            continue
        if read.absent:
            unreadable.append(
                f"{key}: no filed probe. Nothing writes this key yet, and a gate may not "
                "call `lambda:GetFunctionConfiguration` to find out"
            )
            continue
        document = read.document or {}
        version = document.get("schema_version")
        if version != LEGACY_DEAD_LAMBDAS_SCHEMA_VERSION:
            unreadable.append(
                f"{key}: schema_version is {version!r}, not {LEGACY_DEAD_LAMBDAS_SCHEMA_VERSION!r}"
            )
            continue
        entries = document.get("functions")
        if not isinstance(entries, list) or not entries:
            unreadable.append(f"{key}: `functions` is {type(entries).__name__}, not a list")
            continue
        probed: dict[str, Any] = {}
        malformed = False
        for entry in entries:
            if not isinstance(entry, dict):
                unreadable.append(f"{key}: an entry in `functions` is not an object")
                malformed = True
                break
            name = entry.get("name")
            present = entry.get("present")
            if not isinstance(name, str) or not name:
                unreadable.append(f"{key}: an entry in `functions` carries no `name`")
                malformed = True
                break
            if not isinstance(present, bool):
                unreadable.append(
                    f"{key}: {name} carries `present` as {type(present).__name__}, not a "
                    "boolean — a probe that did not answer is not an absent function"
                )
                malformed = True
                break
            probed[name] = present
        if malformed:
            continue
        missing = sorted(required - set(probed))
        if missing:
            unreadable.append(
                f"{key}: the probe does not cover {len(missing)} of the "
                f"{len(LEGACY_DEAD_LAMBDA_NAMES)} names this deliverable asks about "
                f"({', '.join(missing)}) — it answers a narrower question"
            )
            continue
        probed_total += 1
        alive = sorted(name for name in LEGACY_DEAD_LAMBDA_NAMES if probed[name])
        if alive:
            surviving.append(f"{key}: {len(alive)} still present: {', '.join(alive)}")
    if surviving:
        # A real reading about the system, and it takes precedence: a window
        # with one unreadable week and one surviving function has a FINDING in
        # it, and `[?]` would hide the finding behind the unreadable week.
        detail = "; ".join(surviving)
        if unreadable:
            detail += f"; {len(unreadable)} week(s) also unreadable: {'; '.join(unreadable)}"
        return Clause("dead_lambdas_deleted", requirement, False, detail, tuple(evidence))
    if unreadable:
        return Clause(
            "dead_lambdas_deleted",
            requirement,
            False,
            "; ".join(unreadable),
            tuple(evidence),
            unmeasurable=True,
        )
    return Clause(
        "dead_lambdas_deleted",
        requirement,
        True,
        f"{probed_total} filed probe(s) report all {len(LEGACY_DEAD_LAMBDA_NAMES)} named "
        "functions absent, matched per exact name",
        tuple(evidence),
    )


def _clause_old_alerts_muted(store: Store, window: list[dt.date]) -> Clause:
    """Every v1 weekly execution in the window was routed to the muted topic.

    `alpha-engine-config-I9756`'s third deliverable, and the fact Brian
    measured it on: the 2026-09-03 execution's INPUT carries
    ``"sns_topic_arn": "...:<the muted topic>"``, so all 28 of the
    weekly state machine's `sns:publish` states land in a topic with no
    subscribers. The deliverable's prior "not gate-readable" reason called the
    evidence "an observation about a notification channel, which leaves no
    artifact in this store" — but the routing is declared in each execution's
    input, which is a durable, per-execution fact, and
    `nous-ergon-ops/scripts/legacy_weekly_executions_producer.py` now files it
    beside the cadence facts it already files.

    **Scope, stated so a MET reading is not read as wider than it is.** This
    grades the state machine's OWN publish path — the `sns_topic_arn` its
    input names. The second, independent paging path on the same pipeline is
    the native CloudWatch alarm `ExecutionsFailed`, repointed separately
    (`nous-ergon-ops-PR988`); it is not an execution input and this clause
    does not see it. The requirement string says so.

    **A week with no executions is UNMEASURABLE, not MET.** Routing cannot be
    read from an absence: "no execution named a paging topic" and "no
    execution ran" are the same reading, and only one of them is the
    deliverable. The cadence clause is the one that grades a silent week, and
    it reads UNMET there with the missing run named — the two clauses split
    the two questions rather than both half-answering each.

    **An execution entry without the field is UNMEASURABLE too.** Documents
    filed between `alpha-engine-config-I9962` and `I9964` carry every other
    `legacy-weekly-executions.v2` field and answer the cadence question
    correctly; they simply predate the routing fact. Keying on the FIELD
    rather than on a bumped `schema_version` is what lets one week be
    measurable for cadence and unmeasurable for routing, which is what those
    weeks actually are.
    """
    requirement = (
        "every execution of the v1 weekly state machine in the window declared "
        f"`{LEGACY_WEEKLY_TOPIC_FIELD}` = `{muted_alerts_topic_name()}` in its input, so the "
        "state machine's own publish states page nobody. The pipeline's native "
        "CloudWatch alarm is a separate path and is not read here"
    )
    anchors = _weekly_anchors(window)
    evidence = [legacy_weekly_executions_key(a.isoformat()) for a in anchors]
    unreadable: list[str] = []
    paging: list[str] = []
    elsewhere: list[str] = []
    routed_total = 0
    for anchor, key in zip(anchors, evidence, strict=True):
        read = _read_store_document(store, key)
        if read.problem is not None:
            unreadable.append(read.problem)
            continue
        if read.absent:
            unreadable.append(
                f"{key}: no filed execution record. Nothing writes this key yet, and a "
                "gate may not call `states:DescribeExecution` to find out"
            )
            continue
        document = read.document or {}
        version = document.get("schema_version")
        if version != LEGACY_WEEKLY_EXECUTIONS_SCHEMA_VERSION:
            unreadable.append(
                f"{key}: schema_version is {version!r}, not "
                f"{LEGACY_WEEKLY_EXECUTIONS_SCHEMA_VERSION!r} — it records no per-execution "
                "input at all"
            )
            continue
        window_mismatch = _legacy_weekly_window_mismatch(document, anchor)
        if window_mismatch is not None:
            unreadable.append(f"{key}: {window_mismatch}")
            continue
        entries = document.get("executions")
        if not isinstance(entries, list):
            unreadable.append(f"{key}: `executions` is {type(entries).__name__}, not a list")
            continue
        if not entries:
            unreadable.append(
                f"{key}: the week lists no execution, so no input named a topic. Routing "
                "cannot be read from an absence — the cadence clause is what grades a "
                "week in which the pipeline did not run"
            )
            continue
        malformed = False
        unrouted: list[str] = []
        foreign = 0
        for entry in entries:
            if not isinstance(entry, dict):
                unreadable.append(f"{key}: an entry in `executions` is not an object")
                malformed = True
                break
            name = entry.get("name")
            # Attributed to the week its name retries, exactly as the cadence
            # clause does — the two clauses must agree about which executions
            # belong to a week or one anchor grades two different weeks.
            if rerun_names_another_week(name, anchor):
                foreign += 1
                continue
            if LEGACY_WEEKLY_TOPIC_FIELD not in entry:
                unreadable.append(
                    f"{key}: execution {name!r} carries no `{LEGACY_WEEKLY_TOPIC_FIELD}` — "
                    "filed by a producer that predates the routing fact; this week is "
                    "unmeasurable for routing and still measurable for cadence"
                )
                malformed = True
                break
            arn = entry.get(LEGACY_WEEKLY_TOPIC_FIELD)
            if arn is None:
                unrouted.append(f"{name}: input declared no topic")
                continue
            if not isinstance(arn, str) or not arn:
                unreadable.append(
                    f"{key}: execution {name!r} carries `{LEGACY_WEEKLY_TOPIC_FIELD}` as "
                    f"{type(arn).__name__}, which is neither an ARN nor a declared absence"
                )
                malformed = True
                break
            if arn.rsplit(":", 1)[-1] != muted_alerts_topic_name():
                unrouted.append(f"{name}: {arn.rsplit(':', 1)[-1]}")
        if malformed:
            continue
        if foreign:
            elsewhere.append(
                f"{key}: {foreign} watch-rerun execution(s) naming other week(s), "
                "attributed there and not graded for routing here"
            )
        if unrouted:
            paging.append(
                f"{key}: {len(unrouted)} execution(s) not routed to "
                f"`{muted_alerts_topic_name()}`: {'; '.join(unrouted)}"
            )
            continue
        if foreign == len(entries):
            # Every execution the week filed was another week's rerun, so this
            # week declared no routing of its own. That is an absence, and the
            # clause already refuses to read routing from one.
            unreadable.append(
                f"{key}: every filed execution is a watch-rerun of another week, so this "
                "week's own publish path named no topic. Routing cannot be read from an "
                "absence"
            )
            continue
        routed_total += len(entries) - foreign
    if paging:
        detail = "; ".join(paging)
        if unreadable:
            detail += f"; {len(unreadable)} week(s) also unreadable: {'; '.join(unreadable)}"
        if elsewhere:
            detail += "; " + "; ".join(elsewhere)
        return Clause("old_alerts_muted", requirement, False, detail, tuple(evidence))
    if unreadable:
        return Clause(
            "old_alerts_muted",
            requirement,
            False,
            "; ".join(unreadable) + ("; " + "; ".join(elsewhere) if elsewhere else ""),
            tuple(evidence),
            unmeasurable=True,
        )
    return Clause(
        "old_alerts_muted",
        requirement,
        True,
        f"{routed_total} execution(s) across {len(evidence)} week(s), every input naming "
        f"`{muted_alerts_topic_name()}`" + ("; " + "; ".join(elsewhere) if elsewhere else ""),
        tuple(evidence),
    )


def _clause_v2_resources_tagged_and_versioned(
    store: Store, window: list[dt.date], trading_day: dt.date
) -> Clause:
    """The v2 store is versioned and every v2 resource carries the cost tag.

    `alpha-engine-config-I9756`'s fourth deliverable, and the one whose prior
    "not gate-readable" reason already named the condition that would close
    it: *"it becomes a gate clause the day that audit files its result to the
    store."* That day is now — the acceptance identity runs the §2
    suite on every push to `main` and files
    :func:`crucible.keys.acceptance_reading_key`. What the filed document
    lacked was per-clause detail: `{"met": 22, "unmet": 2}` cannot say whether
    :data:`V2_TAG_ACCEPTANCE_CLAUSE_ID` is one of the 22.

    So the producer files `met_clauses` and, from the same read, the store
    bucket's S3 versioning `Status` — the deliverable's other half, which no
    §2 clause grades. Both halves must hold, and each is read from a NAMED
    field: `met_clauses` absent is UNMEASURABLE, never "the clause must have
    passed since it is not in `unmet_clauses`", and `store_versioning` absent
    is UNMEASURABLE, never `Suspended`.

    **Which day's reading.** The document is keyed to the trading day CI ran,
    and CI runs on merges, not on a schedule — so reading only
    ``trading_day`` would make this clause flap with the merge calendar. It
    scans the sessions the window covers, newest first, and grades the most
    recent reading it finds, naming that day and its commit. Never past
    ``trading_day``: a reading filed tomorrow did not exist when the gate was
    read, and letting it satisfy today's exit is how a phase closes on
    evidence that post-dates it.
    """
    requirement = (
        f"the most recent §2 acceptance reading in the window reports "
        f"`{V2_TAG_ACCEPTANCE_CLAUSE_ID}` MET by name, reports the store bucket's S3 "
        f"versioning status as `{V2_STORE_VERSIONING_ENABLED}`, and Billing has `{TAG_KEY}` "
        "Active as a cost-allocation tag (a tagged estate under an inactive key has a $0.00 "
        "denominator)"
    )
    sessions = _session_span(_weekly_anchors(window), trading_day)
    if not sessions:
        # `weekly_anchor` is strictly before its argument, so the span always
        # holds at least the anchor itself. Refuse rather than index into an
        # empty list: a clause that reads no key at all would report the
        # deliverable on no evidence.
        return Clause(
            "v2_resources_tagged_and_versioned",
            requirement,
            False,
            f"the window's anchor is not on or before {trading_day.isoformat()}, so there "
            "is no session to read a reading from",
            (),
            unmeasurable=True,
        )
    evidence = [acceptance_reading_key(day.isoformat()) for day in sessions]
    problems: list[str] = []
    for day in reversed(sessions):
        key = acceptance_reading_key(day.isoformat())
        read = _read_store_document(store, key)
        if read.problem is not None:
            # An access failure is about us, and it is the FIRST thing the
            # scan hits going backwards — stop rather than silently grading an
            # older day, which would report a reading taken before whatever
            # the denial is hiding.
            return Clause(
                "v2_resources_tagged_and_versioned",
                requirement,
                False,
                read.problem,
                tuple(evidence),
                unmeasurable=True,
            )
        if read.absent:
            continue
        reading = parse_acceptance_reading(read.document)
        if reading is None:
            problems.append(f"{key} is not an acceptance reading")
            continue
        if reading.met_clauses is None:
            problems.append(
                f"{key} (commit {reading.commit}) names no `met_clauses`, so it cannot say "
                f"whether `{V2_TAG_ACCEPTANCE_CLAUSE_ID}` is one of its {reading.met} met "
                "clauses. An absent list is 'not filed', never 'there are none'"
            )
            continue
        if reading.store_versioning is None:
            problems.append(
                f"{key} (commit {reading.commit}) names no `store_versioning`; the "
                "producer could not read the bucket's versioning status, which is a "
                "statement about its access, not about the bucket"
            )
            continue
        failures: list[str] = []
        if V2_TAG_ACCEPTANCE_CLAUSE_ID not in reading.met_clauses:
            where = (
                "unmet"
                if reading.unmet_clauses and V2_TAG_ACCEPTANCE_CLAUSE_ID in reading.unmet_clauses
                else "not named at all"
            )
            failures.append(f"`{V2_TAG_ACCEPTANCE_CLAUSE_ID}` is {where}, not met")
        if reading.store_versioning != V2_STORE_VERSIONING_ENABLED:
            failures.append(
                f"store versioning is {reading.store_versioning!r}, not "
                f"{V2_STORE_VERSIONING_ENABLED!r}"
            )
        if failures:
            return Clause(
                "v2_resources_tagged_and_versioned",
                requirement,
                False,
                f"{key} (commit {reading.commit}): " + "; ".join(failures),
                tuple(evidence),
            )
        # The third half (`alpha-engine-config-I10076` deliverable 4): a fully
        # tagged, versioned estate whose tag KEY Billing has not activated as
        # a cost-allocation tag has a denominator of exactly $0.00 for every
        # dollar clause above it -- measured 2026-09-06, when `system` read
        # `Inactive` while every resource carried it and phase 2's cost row
        # read UNMEASURABLE for a week. Read live from Cost Explorer, the same
        # identity and the same `_ce_client` the dollar clauses use, so a
        # later deactivation reads RED here rather than as a $0.00 elsewhere.
        try:
            activation = cost_allocation_tag_status(_ce_client())
        except Exception as exc:  # noqa: BLE001 - a reading; the cause is the detail
            # `CostAllocationTagUnreadableError` from the read itself, or
            # whatever constructing the client raised (no credentials, no
            # region) -- both are statements about our side of the boundary
            # and both render UNMEASURABLE with the cause, never `Inactive`.
            return Clause(
                "v2_resources_tagged_and_versioned",
                requirement,
                False,
                f"{key} (commit {reading.commit}): `{V2_TAG_ACCEPTANCE_CLAUSE_ID}` met and "
                f"store versioning {reading.store_versioning}; whether `{TAG_KEY}` is "
                f"activated as a cost-allocation tag could not be read "
                f"(ce:ListCostAllocationTags): {type(exc).__name__}: {exc}",
                (*evidence, "ce:ListCostAllocationTags"),
                unmeasurable=True,
            )
        if not activation.active:
            return Clause(
                "v2_resources_tagged_and_versioned",
                requirement,
                False,
                f"{key} (commit {reading.commit}): `{V2_TAG_ACCEPTANCE_CLAUSE_ID}` met and "
                f"store versioning {reading.store_versioning}, but `{TAG_KEY}` is "
                f"{activation.status} as a cost-allocation tag in Billing -- every resource "
                "carries the tag and Cost Explorer indexes none of it. Activate it with "
                "`aws ce update-cost-allocation-tags-status --cost-allocation-tags-status "
                f"TagKey={TAG_KEY},Status=Active`",
                (*evidence, "ce:ListCostAllocationTags"),
            )
        return Clause(
            "v2_resources_tagged_and_versioned",
            requirement,
            True,
            f"{key} (commit {reading.commit}): `{V2_TAG_ACCEPTANCE_CLAUSE_ID}` met, "
            f"store versioning {reading.store_versioning}, `{TAG_KEY}` Active as a "
            f"cost-allocation tag since {activation.last_updated_date or 'an unknown date'}",
            (*evidence, "ce:ListCostAllocationTags"),
        )
    detail = (
        f"no §2 acceptance reading in {sessions[0].isoformat()}..{trading_day.isoformat()} "
        f"({len(sessions)} session(s)) answers this deliverable"
    )
    if problems:
        detail += f": {'; '.join(problems)}"
    else:
        detail += "; nothing is filed under this prefix for those days"
    return Clause(
        "v2_resources_tagged_and_versioned",
        requirement,
        False,
        detail,
        tuple(evidence),
        unmeasurable=True,
    )


def _clause_acceptance_suite_committed() -> Clause:
    """The §2 suite exists in source, and the committed reading matches it.

    Deliberately NOT "the suite passes", and deliberately not "the suite
    fails" either. Phase 0 asks for the clauses to be WRITTEN before the code
    that satisfies them; a clause keyed on the suite being red would go unmet
    on the day the system finally satisfied it, which is a gate that inverts.

    What is graded is that the clause set is non-empty, that every unmet clause
    carries a written reason, and that the ids in the ratchet are the ids the
    suite actually DEFINES — because a ratchet beside a deleted suite parses
    perfectly and reports 24 clauses that exist nowhere.

    Every read here is guarded. The ratchet and the suite are checked-in files
    that a person edits by hand, so a malformed one is a routine event, and an
    exception raised here would take the ladder and the board down rather than
    turn one clause red.
    """
    requirement = (
        "the plan §2 acceptance suite exists in source and its committed reading "
        f"names exactly the clauses it defines, each unmet one with a reason. "
        f"Scan scope: {SOURCE_SCAN_SCOPE}"
    )
    path = str(ACCEPTANCE_RATCHET_PATH)
    evidence = (path, str(ACCEPTANCE_SUITE_DIR))

    def unmet_clause(detail: str) -> Clause:
        return Clause("acceptance_suite_committed", requirement, False, detail, evidence)

    read = _read_path_document(ACCEPTANCE_RATCHET_PATH)
    if read.absent:
        return unmet_clause(f"{path} is absent; the §2 clause set has no committed reading here")
    if read.problem is not None:
        return unmet_clause(read.problem)
    document = read.document or {}
    met = document.get("met")
    unmet = document.get("unmet")
    if not isinstance(met, list) or not all(isinstance(cid, str) for cid in met):
        return unmet_clause(f"{path}: `met` is {type(met).__name__}, not a list of clause ids")
    if not isinstance(unmet, dict):
        return unmet_clause(
            f"{path}: `unmet` is {type(unmet).__name__}, not a mapping of clause id to reason"
        )
    clauses = set(met) | set(unmet)
    if not clauses:
        return unmet_clause(
            f"{path} names no clauses at all — a suite that collects nothing is dark, not green"
        )
    silent = sorted(cid for cid, reason in unmet.items() if not str(reason).strip())
    if silent:
        return unmet_clause(f"unmet clauses with no stated reason: {silent}")
    if not ACCEPTANCE_SUITE_DIR.is_dir():
        return unmet_clause(
            f"{ACCEPTANCE_SUITE_DIR} is absent; the ratchet describes a suite that is not here"
        )
    scan = _acceptance_source_scan(ACCEPTANCE_SUITE_DIR)
    if scan.problems:
        return unmet_clause(
            f"the suite could not be read as source: {'; '.join(scan.problems)}. An "
            "unparseable module is an unknown clause set, never an empty one"
        )
    if scan.module_level_tests:
        return unmet_clause(
            f"module-level tests pytest collects but the `Class::method` ratchet grammar "
            f"cannot record: {scan.module_level_tests}"
        )
    committed = {cid.split("[", 1)[0] for cid in clauses}
    vanished = sorted(committed - scan.ids)
    unrecorded = sorted(scan.ids - committed)
    if vanished or unrecorded:
        parts = []
        if vanished:
            parts.append(f"{len(vanished)} committed clauses are defined nowhere: {vanished[:4]}")
        if unrecorded:
            parts.append(
                f"{len(unrecorded)} defined clauses are not in the ratchet: {unrecorded[:4]}"
            )
        return unmet_clause("; ".join(parts))
    return Clause(
        "acceptance_suite_committed",
        requirement,
        True,
        f"{len(clauses)} §2 clauses committed and defined in source, {len(unmet)} unmet, "
        "each with a reason",
        evidence,
    )


@dataclass(frozen=True)
class Deliverable:
    """One of a phase issue's deliverables, and what grades it — or why nothing does."""

    id: str
    summary: str
    #: The clause that grades it, or None when no durable artifact records it.
    graded_by: str | None
    #: Why it is not gate-readable. Required when ``graded_by`` is None.
    reason: str = ""


#: `alpha-engine-config-I9756`'s five deliverables, mapped onto this gate.
#:
#: **All five are graded** since `alpha-engine-config-I9964`. Three were
#: ungraded until then, each on the same argument: the fact lives in AWS and a
#: gate does not call AWS. That is true about the READ and a non-sequitur
#: about the CLAUSE — `old_weekly_within_cadence` reads a live AWS fact too,
#: filed by a producer that holds the identity, and the other three needed the
#: same shape rather than a different rule. Brian ruled option (a) on
#: 2026-09-04: phase 0 exits on evidence covering all five, not two.
#:
#: :func:`coverage_note` still publishes the subset onto every surface the
#: reading reaches, and still RAISES on a table that disagrees with the clause
#: list — an ungraded deliverable simply missing from the clause list is how
#: "the gate is met" comes to mean less than a reader assumes, and that guard
#: is what keeps this table honest if a clause is later renamed or dropped.
PHASE0_DELIVERABLES: tuple[Deliverable, ...] = (
    Deliverable(
        "old_weekly_once_per_week",
        "the v1 weekly pipeline runs at most once a week, with no rerun issuers",
        "old_weekly_within_cadence",
    ),
    Deliverable(
        "dead_lambdas_deleted",
        "the six zero-invocation v1 functions are deleted via their owning IaC",
        "dead_lambdas_deleted",
    ),
    Deliverable(
        "old_alerts_muted",
        "v1 weekly/rehearsal alert emitters route to the muted topic, trading alerts unchanged",
        "old_alerts_muted",
    ),
    Deliverable(
        "v2_resources_tagged_and_versioned",
        "S3 versioning on the v2 prefix and `system=crucible-v2` on every v2 resource",
        "v2_resources_tagged_and_versioned",
    ),
    Deliverable(
        "acceptance_tests_written_as_failing_pytest",
        "the plan §2 clauses exist as pytest, written before the code that satisfies them",
        "acceptance_suite_committed",
    ),
)

#: `alpha-engine-config-I9758`'s (phase 2) deliverables, split at the
#: semicolons of that issue's own single, unbulleted "### Deliverables"
#: paragraph — the same granularity `alpha-engine-config-I10309` itself uses
#: when it names these six items.
#:
#: **All six are graded** since `alpha-engine-config-I10314`. Four were
#: ungraded until then, each on a variant of one argument: the fact lives
#: somewhere a gate does not read — a transport field, the runner's source,
#: a one-time exercise, prose in git. That argument is what
#: `alpha-engine-config-I9964` already retired for phase 0: it is a statement
#: about where the READ would have to go, not about whether a clause can be
#: written, and under Brian's 2026-09-09 ruling ("all of phase 2 should be
#: fully validated by the next weekly SF") a deliverable asserted by a human
#: having looked is not validated at all.
#:
#: Three of the four now read a durable artifact on the store and the fourth
#: reads the repository, the way `acceptance_suite_committed` does. None of
#: them was made green to arrive: two of the four read UNMEASURABLE today,
#: naming the artifact that does not exist yet, which is the honest rendering
#: and the one that puts the gap on the ladder instead of only in an issue.
#:
#: `zero_human_mutating_calls`, `live_saturdays_first_attempt_ok`,
#: `pages_within_ceiling` and `aws_cost_within_ceiling` grade plan §6's
#: *closes-when* row, not this issue's declared deliverables, and are
#: correctly absent from every ``graded_by`` here.
PHASE2_DELIVERABLES: tuple[Deliverable, ...] = (
    Deliverable(
        "scheduler_live_over_five_replay_dates",
        "the scheduler runs live over the 5 replay dates at accelerated cadence (one per 4h)",
        "replays_ok",
    ),
    Deliverable(
        "two_page_conditions_on_real_channel",
        "the two page conditions (absence, failure) route to the real paging channel",
        "two_page_conditions_on_real_channel",
    ),
    Deliverable(
        "transient_retry_class_in_runner",
        "the transient-retry class (plan §11.2) is implemented in `crucible.runner`",
        "transient_retry_class_in_runner",
    ),
    Deliverable(
        "fault_injection_against_scheduled_path",
        "fault injection (plan §10.7) was run against the live scheduled path",
        "fault_injection_against_scheduled_path",
    ),
    Deliverable(
        "both_page_conditions_commissioned",
        "both page conditions were induced for real, delivered, and stood down",
        "pages_commissioned",
    ),
    Deliverable(
        "runbook_in_readme",
        "the runbook (rerun, replay, roll back, heal, unseal) is written into README",
        "runbook_in_readme",
    ),
)

#: `alpha-engine-config-I9757`'s (phase 1) deliverables, split at the
#: semicolons inside each of that issue's three bulleted Track A/B/C lines —
#: the same per-clause granularity used for phase 2 above, applied to a
#: Deliverables section that happens to be structured as three tracks rather
#: than one paragraph. Phase 1 already exited MET 6/6 on 2026-09-09 before
#: this table existed (`alpha-engine-config-I10309`); nothing here changes
#: that reading, it only names — for the first time — what of the 20 declared
#: items those six clauses actually cover (8 of 20).
PHASE1_DELIVERABLES: tuple[Deliverable, ...] = (
    # --- Track A ---
    Deliverable(
        "cli_commands",
        "the `crucible` CLI: data.daily, data.weekly, data.heal, experiment.new/run/"
        "grade, report, explain, migrate.history",
        "arc_runs_ok",
    ),
    Deliverable(
        "run_manifest_schema",
        "run-manifest schema: `status` in {ok, failed} only, typed `trading_day`/"
        "`calendar_date`, resource class",
        "arc_runs_ok",
    ),
    Deliverable(
        "data_layer_lifted",
        "data layer lifted from `nousergon-data` + `nousergon_lib.arcticdb`",
        "arc_runs_ok",
    ),
    Deliverable(
        "feature_layer",
        "feature layer (plan §10.4)",
        "arc_runs_ok",
    ),
    Deliverable(
        "u_and_r_slots_on_run_cycle",
        "U and R slots on `nousergon_lib.arena.engine.run_cycle`",
        "arms_all_scored",
    ),
    Deliverable(
        "control_arms",
        "control arms (plan §10.1)",
        "arms_all_scored",
    ),
    Deliverable(
        "trial_ledger",
        "trial ledger",
        None,
        "no clause reads a trial-ledger artifact; none of the six phase-1 clauses name one",
    ),
    Deliverable(
        "trading_day_contract_test",
        "trading-day contract test (plan §4.12)",
        None,
        "enforced by the blocking pytest suite on every push, not re-read by a live "
        "gate clause over the store",
    ),
    # --- Track B ---
    Deliverable(
        "m_slot_lifted",
        "M slot: `crucible-predictor/training/model_zoo.py` lifted, behavioural veto "
        "scale-dependent, `TrainingIntegrityError` fails the run",
        None,
        "M is excluded from `dispatchable_slots()` until phase 3 (`_clause_"
        "arms_all_scored`'s own comment: 'M and S have no produce/grade until phase "
        "3'), so no phase-1 clause reads its arena cycle",
    ),
    Deliverable(
        "s_slot_lifted",
        "S slot: `strategy_arena.py` + walk-forward + `pit_parity` lifted",
        None,
        "S is excluded from `dispatchable_slots()` until phase 3, the same as M above",
    ),
    Deliverable(
        "promote_and_condorcet_retirement",
        "`promote` with `promote_min_weeks=4` and Condorcet retirement via the lib engine",
        None,
        "promotion/retirement is graded by phase 3's `{slot}_promotion_or_verdict_"
        "backed_non_promotion` clauses, which read the champion pointer and `promote` "
        "manifests; no phase-1 clause reads either",
    ),
    Deliverable(
        "arena_cycle_artifacts_per_slot",
        "`arena_cycle` artifacts per slot",
        "arms_all_scored",
    ),
    # --- Track C ---
    Deliverable(
        "ci_workflow",
        "`ci.yml`: uv, ruff, pytest, pip-audit on lockfile, path-filtered, <=5 min",
        None,
        "CI's own correctness is enforced by every push running it green, not by a "
        "gate clause reading the workflow file",
    ),
    Deliverable(
        "deploy_workflow",
        "`deploy.yml`: wheel to `releases/{sha}/`, smoke, conditional-PUT pointer flip",
        "pointer_flipped_on_smoke",
    ),
    Deliverable(
        "cfn_template",
        "CFN template in `nous-ergon-ops` (bucket, scheduler, dispatcher Lambda, spot "
        "launch template, deploy + runtime roles, default tag)",
        None,
        "infrastructure-as-code in a different, private repository; this public repo's "
        "gate clauses read the store, never another repo's IaC, and `tests/test_no_"
        "infra_literals.py` forbids naming its resources here",
    ),
    Deliverable(
        "alerting_two_page_conditions",
        "alerting: two page conditions + weekly heartbeat + `alerts/` bus rows + causal grouping",
        None,
        "page-condition commissioning is graded by phase 2's `pages_commissioned` "
        "clause; phase 1 registers no clause over the alerts bus",
    ),
    Deliverable(
        "drift_metrics",
        "drift metrics (plan §10.5)",
        None,
        "no clause reads a drift-metric artifact",
    ),
    Deliverable(
        "fault_injection_scripts",
        "fault-injection scripts (plan §10.7)",
        None,
        "the scripts' existence is not store-observable; exercising them against the "
        "scheduled path is phase 2's own (also ungraded) deliverable",
    ),
    Deliverable(
        "console_page_from_manifests",
        "console page rendered from manifests",
        None,
        "a rendering surface; no gate clause asserts the console page exists or reads "
        "correctly from manifests",
    ),
    Deliverable(
        "components_yaml",
        "`components.yaml`",
        None,
        "enforced by its own coverage pytest (AGENTS.md rule 1), not by a live gate clause",
    ),
)

#: `alpha-engine-config-I9759`'s (phase 3) deliverables, split at the
#: semicolons of that issue's single Deliverables paragraph. Phase 3's four
#: registered clauses (one per slot) read only the champion pointer and
#: `promote` manifests — none of them reaches the portfolio engine, the
#: attribution table, the sealed holdout, the cost model, or the per-slot
#: benchmark this issue actually declares, so the honest reading is 0 of 5.
#: That is the gap this mechanism exists to expose, not a mapping to smooth
#: over.
PHASE3_DELIVERABLES: tuple[Deliverable, ...] = (
    Deliverable(
        "portfolio_engine_used_by_s_slot",
        "`crucible.portfolio` (MVO + turnover governor + cost model + ADV cap) used "
        "by S-slot grading",
        None,
        "no phase-3 clause reads `crucible.portfolio` or any evidence it was used in "
        "grading; the four promotion/non-promotion clauses read only the champion "
        "pointer and `promote` manifests",
    ),
    Deliverable(
        "factor_neutral_attribution",
        "factor-neutral attribution (beta/sector/size/residual, OLS on ArcticDB ETF series)",
        None,
        "no phase-3 clause reads an attribution artifact for residual alpha or gross/net returns",
    ),
    Deliverable(
        "sealed_holdout",
        "sealed holdout `strategy/holdout.json` with `--unseal` requiring a ruling reference",
        None,
        "no phase-3 clause reads `strategy/holdout.json` or an unseal audit trail",
    ),
    Deliverable(
        "named_transaction_cost_model",
        "named transaction-cost model",
        None,
        "no phase-3 clause reads a transaction-cost-model artifact",
    ),
    Deliverable(
        "benchmark_per_slot",
        "benchmark per slot in `ArenaConfig`",
        None,
        "a config-shape fact for unit tests over `crucible.slots.arena_config_for`, "
        "not a live gate reading",
    ),
)

#: `alpha-engine-config-I9760`'s (phase 4) deliverables, split at the
#: semicolons of the issue's Deliverables paragraph plus its separate
#: "Decommission:" paragraph under the same heading — 7 trader items, 5
#: decommission items. Only two of twelve are gate-readable today:
#: `old_sf_execution_count_zero` grades "old SFs disabled" exactly, and
#: `trader_one_week_on_v2_champion` is the one consumer-evidence artifact the
#: trader contract declares at all.
PHASE4_DELIVERABLES: tuple[Deliverable, ...] = (
    Deliverable(
        "trader_reads_champion_contract",
        "trader reads `champions/{slot}/current.json` + `predictions/{date}.json`, "
        "refuses a champion whose manifest is not ok/attested",
        "trader_one_week_on_v2_champion",
    ),
    Deliverable(
        "trader_release_pin_ib_paper_smoke",
        "trader release pin (`crucible release pin --target trader <sha>`) with IB-paper smoke",
        None,
        "no clause reads a release-pin or paper-smoke artifact for the trader",
    ),
    Deliverable(
        "kill_switch_fire_drill",
        "kill-switch fire drill on paper",
        None,
        "no clause reads any fire-drill artifact",
    ),
    Deliverable(
        "broker_reconciliation",
        "broker reconciliation emitting `run.json`",
        None,
        "`trader_one_week_on_v2_champion` reads only the `trading_days` field on the "
        "trader's consumer-evidence artifact, not a broker-reconciliation manifest",
    ),
    Deliverable(
        "execution_shortfall_attribution_row",
        "execution-shortfall attribution row",
        None,
        "no clause reads an execution-shortfall artifact",
    ),
    Deliverable(
        "shadow_books_per_challenger",
        "shadow books per challenger (plan §10.6)",
        None,
        "no clause reads a shadow-book artifact",
    ),
    Deliverable(
        "portfolio_adopted_by_trader",
        "`crucible.portfolio` adopted by the trader",
        None,
        "no clause reads evidence that the trader adopted `crucible.portfolio`",
    ),
    Deliverable(
        "old_sfs_disabled",
        "old SFs disabled",
        "old_sf_execution_count_zero",
    ),
    Deliverable(
        "lambdas_and_alarms_removed",
        "66 Lambdas + 153 alarms removed via IaC",
        None,
        "no clause reads a Lambda/alarm inventory; `old_sf_execution_count_zero` "
        "counts Step Functions executions only",
    ),
    Deliverable(
        "artifact_registry_tombstoned",
        "`ARTIFACT_REGISTRY` rows tombstoned RETIRED",
        None,
        "no clause reads `ARTIFACT_REGISTRY`",
    ),
    Deliverable(
        "codebuild_consumers_retired",
        "CodeBuild consumers retired",
        None,
        "no clause reads CodeBuild consumer state",
    ),
    Deliverable(
        "system_optimized_doc_rewritten",
        "`SYSTEM_OPTIMIZED.md` rewritten to v2 as target",
        None,
        "documentation content, not store-observable",
    ),
)

#: `alpha-engine-config-I9761`'s (phase 5) deliverables, split at the
#: semicolons of the issue's Deliverables paragraph. Phase 5 registers exactly
#: one clause, `every_llm_arm_has_a_verdict_within_one_cycle`, which reads
#: whether each ACTIVE arm declaring an LLM call site has a verdict — the
#: closest artifact to "arms re-enter... and are graded", and silent on the
#: other three items.
PHASE5_DELIVERABLES: tuple[Deliverable, ...] = (
    Deliverable(
        "llm_arms_re_enter_as_r_challengers",
        "LLM analyst arms re-enter one at a time as R challengers via `krepis.llm`/"
        "LiteLLM router, per-run cost cap via `krepis.usage_pacing`, gated on LIVE "
        "cycles only with look-ahead disclosure",
        "every_llm_arm_has_a_verdict_within_one_cycle",
    ),
    Deliverable(
        "judge_calibration",
        "judge model IDs pinned, calibration set >=30 traces, swap-order pairwise",
        None,
        "no clause reads a judge-model pin, calibration-set size, or swap-order result",
    ),
    Deliverable(
        "pit_constituent_lists",
        "PIT constituent lists",
        None,
        "no clause reads a PIT-constituent-list artifact",
    ),
    Deliverable(
        "leave_one_out_ablation_replay",
        "leave-one-out ablation replay",
        None,
        "no clause reads an ablation-replay artifact",
    ),
)

#: Which phase issue's deliverables each gate is answerable for. Every gate
#: `PHASES` registers now carries an entry — `coverage_note` RAISES for a gate
#: absent from this table (`alpha-engine-config-I10309`, deliverable 3), so a
#: seventh phase cannot be added without declaring what it owes before its
#: gate can be evaluated at all. Before this table had five more rows, a
#: reader comparing phase 0's coverage line to phase 1's `null` closing-record
#: field had nothing telling them the difference was *undeclared coverage*
#: rather than *equivalent coverage* — phase 1 had already exited on it.
GATE_DELIVERABLES: dict[str, tuple[Deliverable, ...]] = {
    "phase0": PHASE0_DELIVERABLES,
    "phase1": PHASE1_DELIVERABLES,
    "phase2": PHASE2_DELIVERABLES,
    "phase3": PHASE3_DELIVERABLES,
    "phase4": PHASE4_DELIVERABLES,
    "phase5": PHASE5_DELIVERABLES,
}


def coverage_note(gate: str, clause_names: Iterable[str]) -> str:
    """How much of ``gate``'s phase issue this clause list actually grades.

    A single line naming the graded subset and every deliverable nothing
    measures — carried on :class:`GateResult`, printed by `render`, and
    appended to the ladder row's detail, so a phase cannot render MET on a
    surface that gives no sign some of its deliverables were never looked at.

    **Raises** when ``gate`` carries no entry in :data:`GATE_DELIVERABLES` at
    all (`alpha-engine-config-I10309`, deliverable 3) — a gate with no
    declared table is not silence, it is the same failure `alpha-engine-
    config-I9757` already shipped: a phase exiting MET with nobody having
    enumerated what it owed. Every gate `PHASES` registers must carry an
    entry before it can be evaluated.

    **Also raises** when the table and the clause list disagree: a
    deliverable naming a clause this reading does not contain, or an
    ungraded deliverable with no written reason. That is a defect in this
    file, not a condition of the store, and rule 5's default is `raise` — a
    reading published from an inconsistent table would quietly grade less
    than it claims.
    """
    if gate not in GATE_DELIVERABLES:
        raise ValueError(
            f"gate {gate!r} has no entry in GATE_DELIVERABLES. Every gate registered "
            "in PHASES must declare what its phase issue owes before it can be "
            "evaluated — an undeclared table is exactly how a phase can exit MET "
            "with a `coverage: null` closing record. Add a Deliverable table for "
            "this gate, built from its own tracker issue's Deliverables line."
        )
    deliverables = GATE_DELIVERABLES[gate]
    names = set(clause_names)
    orphaned = sorted(
        d.id for d in deliverables if d.graded_by is not None and d.graded_by not in names
    )
    unexplained = sorted(d.id for d in deliverables if d.graded_by is None and not d.reason.strip())
    if orphaned or unexplained:
        raise ValueError(
            f"gate {gate!r}'s deliverable table does not match its clause list: "
            f"{orphaned} name a clause the reading does not contain; {unexplained} are "
            "ungraded with no stated reason. Every deliverable is graded by a clause "
            "in the reading or carries a written reason why no artifact records it."
        )
    issue = next((p.issue for p in PHASES if p.gate == gate), None)
    tracker = f"alpha-engine-config-I{issue}" if issue else gate
    ungraded = [d for d in deliverables if d.graded_by is None]
    if not ungraded:
        # `all N of N`, not a bare `all N`: the incomplete branch below reads
        # `grades 2 of 5`, and a reader comparing two renderings of this line
        # across a change should be able to read the ratio in both without
        # knowing the denominator from somewhere else.
        return f"grades all {len(deliverables)} of {len(deliverables)} {tracker} deliverables"
    return (
        f"grades {len(deliverables) - len(ungraded)} of {len(deliverables)} {tracker} "
        f"deliverables; not gate-readable: {', '.join(d.id for d in ungraded)}"
    )


def _phase0(
    store: Store,
    window: list[dt.date],
    registry: dict[str, Component],
    *,
    trading_day: dt.date,
) -> list[Clause]:
    """Phase 0's exit gate.

    ``registry`` is unused: phase 0 predates the weekly arc entirely — it is
    about the v1 system being quieted and the §2 clauses being written — so
    there is no `components.yaml` row to read. The parameter is kept because
    `GATES` holds one callable shape, and a second signature would be a
    per-gate special case in `evaluate`.

    ``trading_day`` IS used since `alpha-engine-config-I9964`: the §2
    acceptance reading is filed on the day CI ran, not at a weekly close, so
    the deliverable-4 clause needs the render day to know which readings
    already existed when the gate was read.

    Five clauses, one per `alpha-engine-config-I9756` deliverable — see
    :data:`PHASE0_DELIVERABLES`.
    """
    _unused((registry,))
    return [
        _clause_old_weekly_within_cadence(
            store,
            window,
            # Brian ruling 2026-09-04 (evening): "assume the weekly sf
            # tomorrow will fail. we will fix and rerun until it is
            # successful, and that needs to be the trigger to complete phase
            # 0." So: no ceiling, at least one gate-passing run, at least one
            # of them SUCCEEDED, reruns permitted and named. Every value is
            # passed explicitly beside phase 4's, so the two phases' bounds
            # are visible in the clause lists that hold them, not inferred
            # inside the shared reader.
            maximum=None,
            minimum=LEGACY_WEEKLY_MIN_RUNS_PER_WEEK,
            minimum_succeeded=1,
            reruns_fail=False,
        ),
        _clause_dead_lambdas_deleted(store, window),
        _clause_old_alerts_muted(store, window),
        _clause_v2_resources_tagged_and_versioned(store, window, trading_day),
        _clause_acceptance_suite_committed(),
    ]


# ---------------------------------------------------------------------------
# phases 2-5 clauses
#
# `alpha-engine-config-I9913`. Every plan §6 phase is registered, and a phase
# with nothing to read is UNMEASURABLE by clause — never blank. Blank and
# "no data yet" render identically, so a ladder row with no clause list gives
# a reader no way to tell whether the instrument for phase 3 EXISTS or merely
# has nothing to read. Brian, 2026-09-03: "set up the eval mechanism
# completely red and evaluate our progress by gradually hooking up the eval".
# Red is a reading; blank is not.
#
# The comment this replaces argued a phase-2 list "would be a gate that could
# go green on replays". That risk is closed twice over: the live clause reads
# the manifest's own LIVE/REPLAY field rather than inferring it from the date,
# and `Clause.unmeasurable` (`alpha-engine-config-I9869`) means a clause whose
# input does not exist reads UNMEASURABLE with the reason, never MET.
# ---------------------------------------------------------------------------


def _unmeasurable(name: str, requirement: str, detail: str, evidence: Iterable[str] = ()) -> Clause:
    """One clause that could not be read, with the reason.

    `met=False` and `unmeasurable=True` together, always: `met` is what the
    ladder and `met_ratio` count, and an unmeasurable clause that set `met`
    True would be *no data* painted green — the single failure mode plan §6
    rule 1 exists to forbid.
    """
    return Clause(name, requirement, False, detail, tuple(evidence), unmeasurable=True)


#: The prefix every clause function in this module carries. The containment
#: wrapper below is applied by walking this module's globals for it, so a clause
#: added tomorrow is contained without anybody remembering to decorate it — the
#: difference between a rule and a habit.
CLAUSE_FUNCTION_PREFIX = "_clause_"


class ClauseMisconfiguredError(ValueError):
    """A clause was CALLED wrongly — the arguments cannot describe any system.

    The one thing :func:`_contained` re-raises, and the distinction is between
    a fact about the environment and a bug in this package. Every other
    exception a clause can raise is a failed READING: a client that would not
    build, a bucket that would not list, a document that would not parse. Those
    are UNMEASURABLE, because the system was not observed.

    This one is different in kind. `_clause_old_weekly_within_cadence` with
    `minimum > maximum` is a clause list that asks an unanswerable question, and
    the only place such a call can come from is this module's own phase
    assemblers — never from an operator, a box, or a credential. Rendering it as
    an UNMEASURABLE row would put a bug in our clause list behind a message that
    reads like an AWS problem, and it would render that way every week forever.

    Not a suppression list (AGENTS.md rule 4): one declared type, raised only by
    argument validation, with the reason stated here. Adding a second type to
    escape containment would be exactly the collection that rule forbids.
    """


def _contained(fn: Callable[..., Clause]) -> Callable[..., Clause]:
    """``fn``, with any escaped exception turned into an UNMEASURABLE clause.

    **No gate clause may raise into its caller** (`alpha-engine-config-I10328`,
    and `-I9869`'s class recurring). A clause is one reading among dozens; a
    reading that could not be taken is that clause's UNMEASURABLE, never the
    death of the whole ladder. Measured 2026-09-09: `_clause_zero_human_
    mutating_calls` gained a `DescribeStacks` call whose CLIENT CONSTRUCTION
    raised `NoRegionError` on a box that exports no region — outside the try
    block the read itself had — and every `weekly` arc failed at the `console`
    stage that renders `gates/ladder.json`, which then regressed phase 1's
    `arc_runs_ok`. One unguarded line in one clause darkened every phase gate
    in the system and turned a gate READ into a system FAILURE.

    Guarding each clause individually is what failed: `-I9869` fixed exactly
    this for manifest reads, clause by clause, and the next clause to reach a
    new AWS service reintroduced it. So the containment lives HERE, applied to
    every `_clause_*` function by :func:`_contain_clause_exceptions`, and a
    clause author cannot forget it.

    **This is a deliberate swallow** (AGENTS.md rule 5), so, explicitly: the
    failure mode swallowed is "a clause raised instead of returning a reading";
    the primary deliverable — a gate result carrying every other clause —
    survives; and the recording surface is the returned clause's own
    UNMEASURABLE detail, which names the exception type and message and is
    rendered on the ladder, the board and `crucible gate`'s output. `met=False`
    always, via :func:`_unmeasurable`, so an uncontainable clause can never be
    counted as passing. `Exception`, not `BaseException`: a KeyboardInterrupt or
    a spot reclamation must still stop the process.
    """

    @wraps(fn)
    def _guarded(*args: Any, **kwargs: Any) -> Clause:
        try:
            return fn(*args, **kwargs)
        except ClauseMisconfiguredError:
            # Re-raised, not contained: a clause called with arguments that
            # cannot describe any system is a bug in this module's own clause
            # list, and containing it would render our defect as an
            # environment reading, every week, forever. See the class.
            raise
        except Exception as exc:
            name = fn.__name__.removeprefix(CLAUSE_FUNCTION_PREFIX)
            return _unmeasurable(
                name,
                f"{name} could be evaluated at all",
                f"the clause raised {type(exc).__name__}: {exc}. A clause that raises has "
                "learned nothing about the system, so this is UNMEASURABLE — it is not a "
                "finding about the system, and it does not darken the other clauses",
            )

    _guarded._contained = True  # type: ignore[attr-defined]
    return _guarded


def _contain_clause_exceptions() -> None:
    """Wrap every `_clause_*` function in this module with :func:`_contained`.

    Called once at import, below every clause definition and above
    :data:`GATES`. Rebinding the module global is what makes it reach the
    `_phaseN` assemblers too: they look their clauses up by name at call time,
    so they get the contained version without being edited.

    Raises if it wraps nothing — a containment pass that silently matched no
    clause is the shape of a guard that grades an empty set.
    """
    wrapped = 0
    for name, value in list(globals().items()):
        if not name.startswith(CLAUSE_FUNCTION_PREFIX) or not callable(value):
            continue
        if getattr(value, "_contained", False):
            continue
        globals()[name] = _contained(value)
        wrapped += 1
    if not wrapped:
        raise RuntimeError(
            "the clause containment pass wrapped 0 functions. Either the "
            f"{CLAUSE_FUNCTION_PREFIX!r} convention changed or this ran before the "
            "clause definitions; both leave every clause able to raise into the ladder "
            "again."
        )


@lru_cache(maxsize=1)
def _manifest_property_names() -> frozenset[str]:
    """Every field the CURRENT run manifest schema declares, read from it.

    Derived, never restated: the phase-2 live clause asks whether the manifest
    contract can distinguish a live Saturday from a replay at all, and that is
    a question about the schema. Restating the field list here would make the
    clause answer it from a copy that drifts.
    """
    return frozenset(load_schema().get("properties", {}))


#: The run-manifest field that says whether a weekly run was a LIVE Saturday
#: or a REPLAY of a historical one, and the value meaning live.
#:
#: **The field landed in `run_manifest.v2`** (`alpha-engine-config-I9918`):
#: required, closed vocabulary, no default, set from the invocation by
#: `crucible.runmode.resolve_run_mode`. v1 declared no such field and set
#: `additionalProperties: false`, so no producer could write one and this
#: clause read UNMEASURABLE. The schema check below stays regardless — it is
#: what makes the clause answer from the CONTRACT rather than from this
#: constant, so a build whose schema does not declare the field reads
#: unmeasurable again instead of grading every manifest as malformed.
#:
#: `calendar_date` is NOT this field and must never be used as it: the schema
#: says it is "recorded for provenance ONLY. Never used as a key, never an
#: input to a promotion, retirement, freshness or grading decision."
MANIFEST_RUN_MODE_FIELD = "run_mode"
MANIFEST_RUN_MODE_LIVE = "live"

#: The tracker issues owning the two manifest/recipe CONTRACT questions a
#: clause below reads: which issue introduced the live-or-replay field
#: (`alpha-engine-config-I9918`, landed in `run_manifest.v2`), and which owns
#: the still-open gap where an arm recipe cannot name an LLM call site
#: (`alpha-engine-config-I9920`).
#:
#: Held as plain `int`s and rendered into the reading with an f-string at the
#: one call site, never as a full literal:
#: `tests/test_no_stale_tracker_literals.py` (`alpha-engine-config-I9839`)
#: forbids the whole `alpha-engine-config-I<N>` shape in any string a reader
#: can reach, because such a literal goes stale silently. Neither of these is
#: a PHASE issue, so neither can be derived from `PHASES` the way
#: `phase_tracker` derives phase 4's.
MANIFEST_RUN_MODE_GAP_ISSUE = 9918
LLM_ARM_CALLSITE_GAP_ISSUE = 9920

#: How many first-attempt `ok` LIVE Saturdays phase 2 requires, and NOTHING
#: else. Read by `_clause_live_saturdays_first_attempt_ok` alone.
#:
#: **One, per Brian's ruling 2026-09-09** (`alpha-engine-config-I10324`),
#: narrowed from §6.1's "2 consecutive first-attempt `ok` Saturdays, not 4" so
#: phase 2 is completable by the 2026-09-12 weekly Step Function. A lowered bar
#: carries its rationale or it is drift, so:
#:
#: * the evidence loss is bounded, not waived — nine other phase-2 clauses
#:   carry the unattended claim (`zero_human_mutating_calls`, `replays_ok` over
#:   five replay Saturdays, the two page clauses, the retry class, the fault
#:   injection, the runbook, the commissioned-pages clause and the cost
#:   ceiling), so one live Saturday is the only piece of evidence that shrinks;
#: * phase 4's `trader_one_week_on_v2_champion` collects a further week of live
#:   operation regardless, on the same infrastructure, before the old system is
#:   gone — the soak is deferred, not deleted;
#: * §6.1 already priced the trade this extends ("replay in place of two of the
#:   four soak weeks, on a path whose inputs are point-in-time addressable"),
#:   and named its residual as a live-only failure mode surfacing on Saturday
#:   #3 or #4 — unchanged by this narrowing, because 09-19 and 09-26 still run.
#:
#: **This was ONE constant doing two unrelated jobs until
#: `alpha-engine-config-I10324`.** It was `GATES["phase2"]`'s window width in
#: weeks as well as the Saturday count, so narrowing the Saturday count would
#: silently have narrowed the window `zero_human_mutating_calls`,
#: `pages_within_ceiling` and `replays_ok` read — a ceiling of two pages over
#: one week asserting almost nothing, with nothing in the edit saying so. The
#: window width is now :data:`PHASE2_WINDOW_WEEKS`, and
#: `tests/test_gate_phases_2_5.py` asserts neither name is reachable from the
#: other's call site.
PHASE2_LIVE_SATURDAYS = 1

#: How many weeks wide phase 2's gate WINDOW is, and nothing else. Read by
#: `GATES["phase2"]` alone, whence `evaluate` builds the window every phase-2
#: clause is handed.
#:
#: TWO, which is what the pre-split constant happened to hold — so this split
#: changes no window, deliberately: the Saturday count moved and the window did
#: not, which is the whole point of there being two names. Two weeks is what
#: makes `pages_within_ceiling`'s ceiling of :data:`PHASE2_MAX_PAGES` a
#: statement about a period rather than about a day, and it is the shortest
#: window that can contain a full weekly cycle plus the cycle it is compared
#: against.
#:
#: NOT the same question as :data:`PHASE2_LIVE_SATURDAYS` and not derivable
#: from it: how many live Saturdays must have gone perfectly is a claim about
#: evidence, how much history a ceiling is counted over is a claim about a
#: denominator, and the two moved together only by accident of one assignment.
PHASE2_WINDOW_WEEKS = 2

#: How many complete DAILY cycles — trading days, strictly after the system's
#: last change and no later than the render day — must fall inside the autonomy
#: window, alongside the one complete weekly cycle.
#:
#: One. The daily half exists because the weekly half alone is a claim about a
#: single artifact: a window can contain a weekly close and still not have
#: carried the daily path — the preopen/postclose axis — past the change. One is
#: the floor at which "the daily cycles falling in that span" is a non-empty
#: statement; a larger number would be a second, undeclared waiting period, and
#: the waiting period is the weekly cycle's job.
PHASE2_AUTONOMY_MIN_DAILY_CYCLES = 1

#: The furthest ahead :func:`autonomy_earliest_satisfiable_render_day` will
#: search before refusing. Twenty-one calendar days is three weeks: a span no
#: sequence of NYSE holidays can fill without a weekly close, so exhausting it
#: means the calendar itself is unreadable and the honest answer is to raise
#: rather than return a date nothing verified.
_AUTONOMY_SEARCH_HORIZON_DAYS = 21


def autonomy_earliest_satisfiable_render_day(change: dt.date) -> dt.date:
    """The first render day on which `zero_human_mutating_calls` can read MET
    for a system last changed on ``change``.

    **The minimum span is DERIVED here, not declared** (`alpha-engine-config-
    I10327`). `alpha-engine-config-I10324` shipped this guard as a
    `PHASE2_AUTONOMY_MIN_SPAN = 7 days` constant AND a "the weekly cycle must
    close after the change" requirement, and for the attack the floor was
    written against — an operator apply an hour before the read leaving a
    one-hour window containing nothing — the two are redundant: one hour cannot
    contain a complete weekly cycle, so the cycle requirement already refuses
    it. What the independent constant added was a second, arithmetically
    separate waiting period, and it put the earliest satisfiable render day
    four days past the weekly Step Function that phase 2 must exit on.

    **Brian's 2026-09-04 phase-0 ruling is the precedent and the same trade:**
    *"we can't wait a week on phase 0, it should clear after this week's
    weekly"* — where the second week's protection was replaced by a daily
    guard rather than deleted. Same here: the calendar floor is replaced by
    the two cycle requirements it was standing in for, so the bar is now
    stated in the unit the claim is denominated in (cycles) rather than in an
    unrelated one (calendar days), and it moves with the calendar instead of
    being wrong on a holiday week.

    Both requirements, and both are load-bearing:

    * `weekly_anchor(day) > change` — a COMPLETE weekly cycle whose close falls
      after the change. This is what makes the anti-gaming property survive: a
      change one hour before the read still yields an unsatisfiable window,
      because `weekly_anchor` resolves to the Friday STRICTLY BEFORE the render
      day and a change on or after that Friday cannot have a completed weekly
      cycle behind it.
    * at least :data:`PHASE2_AUTONOMY_MIN_DAILY_CYCLES` complete daily cycles
      in `(change, day]`. The weekly close is one artifact; the daily path is
      the one that runs four more times a week, and a window that graded a
      weekly cycle without one daily cycle in it would be asserting unattended
      operation from a single Saturday.

    Raises :class:`LastChangeUnreadableError` if no day inside
    :data:`_AUTONOMY_SEARCH_HORIZON_DAYS` satisfies both — fail loud, never a
    silently-clamped date the clause would then grade against.
    """
    day = change + dt.timedelta(days=1)
    for _ in range(_AUTONOMY_SEARCH_HORIZON_DAYS):
        if (
            weekly_anchor(day) > change
            and autonomy_daily_cycles_in_span(change, day) >= PHASE2_AUTONOMY_MIN_DAILY_CYCLES
        ):
            return day
        day += dt.timedelta(days=1)
    raise LastChangeUnreadableError(
        f"no render day within {_AUTONOMY_SEARCH_HORIZON_DAYS} days of {change.isoformat()} "
        "contains both a complete weekly cycle closing after it and "
        f"{PHASE2_AUTONOMY_MIN_DAILY_CYCLES} complete daily cycle(s). Three weeks of "
        "calendar with no weekly close is not a holiday pattern, it is an unreadable "
        "calendar, and a clamped date would be graded as though it had been verified."
    )


def autonomy_daily_cycles_in_span(change: dt.date, render_day: dt.date) -> int:
    """How many complete daily cycles fall in ``(change, render_day]``.

    Trading days, counted through `crucible.calendar` — a holiday week
    legitimately contributes fewer, which is the whole reason this is counted
    rather than derived from a calendar-day span. Strictly after the change: a
    daily cycle that ran on the change's own day may have run BEFORE it, and
    `_clause_zero_human_mutating_calls` already filters that day's operator
    actions by `eventTime` for the same reason.
    """
    if render_day <= change:
        return 0
    days = (render_day - change).days
    return sum(1 for n in range(1, days + 1) if is_trading_day(change + dt.timedelta(days=n)))


#: How many replay Saturdays phase 2 re-grades through the phase-1 predicate.
PHASE2_REPLAY_SATURDAYS = 5

#: Plan §6 row 2: at most two pages over the phase-2 window.
PHASE2_MAX_PAGES = 2

#: Plan §6 row 2 and row 4, in USD. Row 2 is the spend carrying
#: `system=crucible-v2`; row 4 is the whole account.
PHASE2_MAX_TAGGED_USD = 40.0
PHASE4_MAX_TOTAL_USD = 70.0

#: The store key carrying the trader's evidence that it ran a week on the v2
#: champion — `None` because the trader contract declares no such artifact
#: today. Phase 4's trader clause reads UNMEASURABLE naming that missing
#: declaration; the gap is `alpha-engine-config-I9760`'s own scope (plan §3:
#: the trader is a separate system, and the harness may not reach into it).
#:
#: A declared constant rather than an inline `None` check so the day the
#: contract names its artifact is a one-line edit here, and so the MET and
#: UNMET branches below are reachable and tested today.
TRADER_EVIDENCE_KEY: str | None = None

#: The `spec.params` key naming which registered LLM call site an arm reaches
#: a model through — `crucible.slots.arms.LLM_CALLSITE_PARAM`, restated here
#: by value rather than imported so the gate never imports the slot machinery
#: to read a register; `tests/test_gate_phases_2_5.py` asserts the two agree.
#: `_parse` refuses a recipe naming an unregistered site, so membership of
#: this value in `LLM_CALLSITE_REGISTRY` is what makes an arm an LLM arm, and
#: the set is derivable from the register alone (`alpha-engine-config-I9920`).
#: Typed `str | None` so the "no such field" branch below stays reachable and
#: tested — it is the reading a build would give if the field were removed.
LLM_ARM_CALLSITE_FIELD: str | None = "llm_callsite"

#: The clauses whose artifact is the REPOSITORY rather than the store.
#:
#: Both grade something git is the durable record of — the §2 acceptance
#: suite and the README runbook — so both read correctly against a checkout
#: and both are UNMET (never a pass) from a wheel install where the file does
#: not ship. Named as a set because a reader of an empty-store reading has to
#: know which clauses an empty store says nothing about:
#: `tests/test_gate_phases_2_5.py` asserts BOTH directions of it — every
#: clause outside this set is unmet against an empty store, and every clause
#: inside it is met against this checkout — so it cannot become a place to
#: park a clause that merely happens to be green.
REPOSITORY_GRADED_CLAUSES: frozenset[str] = frozenset(
    {"acceptance_suite_committed", "runbook_in_readme"}
)

#: The four scripted faults of plan §10.7, in the order the plan names them,
#: and the ids a fault-injection record is filed under.
#:
#: A declared tuple rather than a scan of `tests/faults/`: the plan is what
#: says there are four and which four, and a scan of the suite would make the
#: gate's own definition of "scripted fault" move whenever someone added a
#: test class. `tests/test_gate_phase2_coverage.py` parses that suite and
#: asserts it defines exactly this many `TestFault*` cases, so the two cannot
#: drift silently in the other direction either.
SCRIPTED_FAULTS: tuple[str, ...] = (
    "spot_terminated_mid_job",
    "data_source_withheld",
    "router_returns_500",
    "stale_release_pointer",
)

#: The two fields an INDUCED fault-injection record must carry: the manifest
#: the fault produced, and the bus row it produced. Both, because either alone
#: is half the §10.7 exercise — a manifest with no bus row is a job that failed
#: and paged nobody, and a bus row with no manifest is a page about a run whose
#: cause was discarded.
#:
#: Only for `induced`. `alpha-engine-config-I10327`: two of plan §10.7's four
#: faults never produce a failed run, so requiring these two of every record
#: made the clause unsatisfiable for exactly the faults whose point is that the
#: system survived them. See :data:`FAULT_RECORD_OUTCOME_FIELD`.
FAULT_RECORD_MANIFEST_FIELD = "manifest_key"
FAULT_RECORD_BUS_FIELD = "bus_key"

#: The field saying HOW a fault ended, and therefore which evidence fields are
#: legal on the record (`alpha-engine-config-I10327`). The vocabulary lives in
#: `crucible.models.FAULT_OUTCOME_VALUES`, with the field-by-outcome matrix
#: enforced by `crucible.models.FaultRecordDocument`; this clause reads that
#: model rather than re-deriving the matrix, so a record shape the producer
#: could not write is also a record shape this reader refuses.
FAULT_RECORD_OUTCOME_FIELD = "outcome"

#: The one outcome whose `run_id` excuses a failed manifest from
#: `arc_runs_ok`/`replays_ok`. Named, because `_fault_excused_run_ids` filters
#: on it and a string literal there would be the kind of unexplained
#: comparison that gets "simplified" away.
FAULT_OUTCOME_INDUCED = "induced"

#: The README the phase-2 runbook deliverable lives in. Not a store artifact,
#: for the reason :data:`ACCEPTANCE_RATCHET_PATH` is not: the runbook's
#: existence is a property of the REPOSITORY and git is the durable record of
#: it. Read from a checkout; absent (a wheel install, where the README does
#: not ship) the clause is UNMET with the path named, never a pass.
README_PATH = Path(__file__).resolve().parent.parent / "README.md"

#: The runbook heading the five procedures sit under, and the heading depth
#: each procedure uses. Named rather than inlined so the extractor and the
#: requirement string cannot disagree about what they are looking for.
RUNBOOK_SECTION = "## Runbook"
RUNBOOK_PROCEDURE_PREFIX = "### "

#: The five procedures `alpha-engine-config-I9758` names, and whether each is
#: RESERVED — documented as a deliberate non-capability rather than as a
#: command.
#:
#: `unseal` is reserved by plan §9.4: unsealing is a human ruling, never an
#: automated action, and the README says so explicitly. Grading it as
#: "must contain a command that parses" would mean the runbook passes this
#: clause only by growing the exact CLI surface the plan forbids, so a
#: reserved procedure is graded the other way round: it must name NO command
#: and the CLI must carry no such job. The clause therefore refuses in both
#: directions — a missing procedure and an invented one.
RUNBOOK_PROCEDURES: tuple[tuple[str, bool], ...] = (
    ("rerun", False),
    ("replay", False),
    ("roll back", False),
    ("heal", False),
    ("unseal", True),
)

#: One `crucible <job> [args]` command line inside the runbook, with or
#: without the `uv run` prefix. Line-anchored, so a `crucible ...` mentioned
#: mid-sentence in prose is not extracted as a command — only the fenced
#: blocks an operator copies. Kept in the same shape as
#: `tests/test_runbook_new_experiment.py::_COMMAND`, which is the same
#: oracle applied to the other runbook (`crucible-PR163`).
_RUNBOOK_COMMAND_RE = re.compile(r"^(?:uv run )?crucible ([a-z][a-z0-9._-]*)(.*)$", re.MULTILINE)

#: The reason value every manifest's FIRST attempt carries. Everything else
#: in the schema's attempt vocabulary is a transient class that caused a
#: RETRY, which is what makes "the enum minus this value" the recordable
#: retry vocabulary rather than a second hand-kept list.
MANIFEST_ATTEMPT_INITIAL = "initial"

#: The slots whose recipes are `crucible.slots.arms.ArmSpec` documents and so
#: carry `params` — the only recipe shape with a place for
#: `LLM_ARM_CALLSITE_FIELD`. The M slot's recipes are `ModelRecipe`s (a design
#: matrix plus an estimator; no `params`, no model call) and the S slot's are
#: strategy configs, so neither can declare an LLM call site by contract and
#: neither is an LLM arm. `tests/test_gate_phases_2_5.py` pins that the M
#: recipe indeed has no such field, so widening this tuple is a deliberate
#: edit the day one of those contracts grows an LLM path.
LLM_ARM_RECIPE_SLOTS: tuple[str, ...] = ("u", "r")


def _clause_live_saturdays_first_attempt_ok(store: Store, window: list[dt.date]) -> Clause:
    """Plan §6 row 2 / §6.1: consecutive LIVE first-attempt `ok` Saturdays.

    **LIVE is read from the manifest, never inferred from the date.** A replay
    of a future Saturday and a re-run of a live one are both indistinguishable
    from their `trading_day`, and a gate that inferred liveness from the
    calendar would be satisfied by exactly the accelerated replay schedule
    §6.1 uses to build phase 2 — the "green on replays" failure the phase-2
    clause list was withheld for.

    **The window is anchored, never keyed on the render weekday.** A weekly
    artifact binds to a Friday close (§4.12), while `_window` steps back in
    raw calendar weeks from whatever day the caller passed and the ladder
    renders DAILY. Keying on those raw days would send Monday's read and
    Wednesday's read to different objects, so two flawless live Saturdays
    filed at Friday closes would read MET on a Friday and `never ran` on the
    other four weekdays — a contract unsatisfiable rather than merely unmet.
    Every window day therefore collapses through `weekly_anchor`, exactly as
    phase 0's `_clause_old_weekly_within_cadence` does
    (`alpha-engine-config-I9904`).
    """
    requirement = (
        f"{PHASE2_LIVE_SATURDAYS} consecutive LIVE Saturdays whose weekly run manifest "
        f"reads `status: ok` on its FIRST attempt (`attempts` is exactly one entry, "
        f"`n: 1`). LIVE is read from the manifest's `{MANIFEST_RUN_MODE_FIELD}` field, "
        "never inferred from the trading day; the Saturdays are the weekly anchors of "
        "the window, not the weekday the gate was rendered on"
    )
    name = "live_saturdays_first_attempt_ok"
    days = list(dict.fromkeys(weekly_anchor(day) for day in window[-PHASE2_LIVE_SATURDAYS:]))
    evidence = [manifest_key("weekly", day.isoformat()) for day in days]
    if len(days) != len(window[-PHASE2_LIVE_SATURDAYS:]):
        return Clause(
            name,
            requirement,
            False,
            f"{len(window[-PHASE2_LIVE_SATURDAYS:])} window weeks collapsed onto "
            f"{len(days)} week anchors ({', '.join(d.isoformat() for d in days)}); one "
            "manifest would be graded twice",
            tuple(evidence),
        )
    if MANIFEST_RUN_MODE_FIELD not in _manifest_property_names():
        return _unmeasurable(
            name,
            requirement,
            f"the current run manifest schema declares no `{MANIFEST_RUN_MODE_FIELD}` "
            "field, so no manifest can say whether its run was a live Saturday or a "
            "replay of a historical one. The schema sets `additionalProperties: "
            "false`, so a producer cannot add it either. Inferring liveness from the "
            "trading day is exactly what this clause must not do (§6.1 runs replays of "
            "past Saturdays on an accelerated schedule). The field was introduced by "
            f"`alpha-engine-config-I{MANIFEST_RUN_MODE_GAP_ISSUE}`",
            evidence,
        )
    missing: list[str] = []
    malformed: list[str] = []
    unreadable: list[str] = []
    replayed: list[str] = []
    predates_field: list[str] = []
    failed: list[str] = []
    retried: list[str] = []
    for day, key in zip(days, evidence, strict=True):
        read = _read_store_document(store, key)
        if read.problem is not None:
            (unreadable if read.access_problem else malformed).append(read.problem)
            continue
        if read.absent:
            missing.append(f"{key} is absent")
            continue
        document = read.document or {}
        if MANIFEST_RUN_MODE_FIELD not in document:
            # A manifest written before the field existed. It is UNMET, never
            # MET and never "malformed": the document was correct against the
            # contract it declares, and it simply cannot establish liveness.
            # Counting it live on the strength of its date is the one thing
            # this clause exists not to do.
            predates_field.append(
                f"{day.isoformat()}: {document.get('schema_version')!r} carries no "
                f"`{MANIFEST_RUN_MODE_FIELD}`, so this run cannot be shown to be live"
            )
            continue
        problem = _field(key, document, MANIFEST_RUN_MODE_FIELD, str)
        if problem is not None:
            malformed.append(problem)
            continue
        if document[MANIFEST_RUN_MODE_FIELD] != MANIFEST_RUN_MODE_LIVE:
            replayed.append(
                f"{day.isoformat()}: {MANIFEST_RUN_MODE_FIELD} is "
                f"{document[MANIFEST_RUN_MODE_FIELD]!r}, not {MANIFEST_RUN_MODE_LIVE!r}"
            )
            continue
        status, problem = _status(key, document)
        if problem is not None:
            malformed.append(problem)
            continue
        if status != "ok":
            failed.append(f"{day.isoformat()}: {document.get('reason')!r}")
            continue
        problem = _field(key, document, "attempts", list)
        if problem is not None:
            malformed.append(problem)
            continue
        attempts = document["attempts"]
        if len(attempts) != 1 or not isinstance(attempts[0], dict) or attempts[0].get("n") != 1:
            retried.append(
                f"{day.isoformat()}: {len(attempts)} attempt(s) — a retried run is not a "
                "first-attempt ok"
            )
    content_gap = bool(missing or malformed or replayed or predates_field or failed or retried)
    if unreadable and not content_gap:
        return _unmeasurable(name, requirement, "; ".join(unreadable), evidence)
    if content_gap or unreadable:
        parts: list[str] = []
        if unreadable:
            parts.append(f"{len(unreadable)} could not be read: {'; '.join(unreadable)}")
        for label, rows in (
            ("never ran", missing),
            ("malformed", malformed),
            ("not live", replayed),
            ("predate the live/replay field", predates_field),
            ("failed", failed),
            ("retried", retried),
        ):
            if rows:
                parts.append(f"{len(rows)} {label}: {'; '.join(rows)}")
        return Clause(name, requirement, False, "; ".join(parts), tuple(evidence))
    return Clause(
        name,
        requirement,
        True,
        f"{len(days)} consecutive live Saturdays "
        f"({days[0].isoformat()}..{days[-1].isoformat()}), each first-attempt ok",
        tuple(evidence),
    )


def _clause_replays_ok(
    store: Store, window: list[dt.date], registry: dict[str, Component]
) -> Clause:
    """Phase 1's replay predicate, re-read over phase 2's five replay Saturdays.

    `_clause_arc_runs_ok` verbatim — the same function phase 1 registers, over
    a five-week window anchored on the same trading day. Restating the
    predicate would let phase 2 and phase 1 disagree about what a good replay
    is, which is the drift `crucible.gate`'s single-reader discipline exists
    to prevent. Only the NAME and the requirement sentence change, so the two
    readings land on the ladder as two clauses rather than one.
    """
    # Anchored, not raw: phase 2's window is render-keyed (its live clause
    # collapses it itself), and the replay manifests are Friday closes. Read
    # 2026-09-03 on `main`, the raw form named `data.weekly@2026-08-06` — a
    # Thursday — as "never ran" (`alpha-engine-config-I9904`).
    replay_window = weekly_window(window[-1], PHASE2_REPLAY_SATURDAYS)
    clause = _clause_arc_runs_ok(store, replay_window, registry)
    return replace(
        clause,
        name="replays_ok",
        requirement=(
            f"the same predicate phase 1 grades ({clause.requirement}), re-read over the "
            f"{PHASE2_REPLAY_SATURDAYS} replay Saturdays "
            f"{replay_window[0].isoformat()}..{replay_window[-1].isoformat()}"
        ),
    )


def _s3_client() -> Any:
    """An S3 client for the CloudTrail archive read.

    A module-level function so a test can substitute it without reaching for a
    credential chain, and lazy for the reason `crucible.store.S3Store` is:
    importing `crucible.gate` must not require an AWS SDK.
    """
    import boto3  # noqa: PLC0415 - lazy on purpose

    return boto3.client("s3")


class LastChangeUnreadableError(RuntimeError):
    """One of the two inputs to the autonomy window's START could not be read.

    Raised rather than falling back to a window that starts earlier (or at
    zero). A window whose start defaulted when an input was unreadable would
    grade a span nobody established, and it fails in the direction that looks
    like diligence: the earlier the start, the more history, the more
    authoritative the reading appears. `alpha-engine-config-I10324` requires
    the opposite — either input unreadable is UNMEASURABLE.
    """


@dataclass(frozen=True)
class _SystemChange:
    """When the graded system last changed, and which input said so."""

    at: dt.datetime
    source: str
    pointer_at: dt.datetime
    stack_at: dt.datetime

    def provenance(self) -> str:
        return (
            f"window starts {self.at.isoformat()} ({self.source}); release pointer "
            f"{POINTER_KEY} flipped {self.pointer_at.isoformat()}, stack last updated "
            f"{self.stack_at.isoformat()}"
        )


def _pointer_flip_time(store: Store) -> dt.datetime:
    """When `releases/current` last moved, as an instant.

    The pointer is the ONE mutable object in the release layout
    (`crucible.release`), moved only by a conditional PUT on a green smoke, so
    its object modification time IS the flip time — there is no separate flip
    record to read, and `release.json` is deterministic across rebuilds by
    design (`alpha-engine-config-I9786`) so it carries no instant at all.

    `HeadObject` off the store's own client, mirroring
    `crucible.release_lock_sweep._read_one` rather than inventing a second
    way to reach the same header — that module already established the
    pattern (and the rule that ONLY `LastModified` is trusted out of the
    response).

    A non-S3 backend raises: a `LocalStore` directory's mtime is a fact about
    a laptop's filesystem, not a deploy record, and grading a live autonomy
    window off it would be an answer about the workbench — the defect this
    whole clause was rewritten to stop.
    """
    if not isinstance(store, S3Store):
        raise LastChangeUnreadableError(
            f"the store backend is {type(store).__name__}, not S3, so the release "
            f"pointer {POINTER_KEY} has no flip instant to read. A local directory's "
            "mtime is a fact about a filesystem, not a deploy"
        )
    try:
        head = store.client.head_object(Bucket=store.bucket, Key=store._s3_key(POINTER_KEY))
    except Exception as exc:
        # Broader than `ClientError` on purpose, and NOT a swallow: the caller
        # renders every branch of this function as UNMEASURABLE, and the
        # failures that actually happen here are as often `NoCredentialsError`
        # or an endpoint resolution error as they are a 403 or a 404. Catching
        # only `ClientError` would take `crucible gate`, `build_ladder` and the
        # board render down together over an expired credential — the same
        # reasoning `_clause_zero_human_mutating_calls`'s archive read carries,
        # and the exception class is named in the message either way.
        code = getattr(exc, "response", None) and store._error_code(exc)
        raise LastChangeUnreadableError(
            f"head_object({POINTER_KEY}) failed: {code or type(exc).__name__}: {exc}. That "
            "is a statement about our access or about the pointer being unset, not about "
            "the system being measured"
        ) from exc
    last_modified = head.get("LastModified")
    if last_modified is None:
        raise LastChangeUnreadableError(
            f"head_object({POINTER_KEY}) returned no LastModified, so the flip instant is unknown"
        )
    return _as_utc(last_modified)


def _stack_last_updated(cfn: Any | None = None, *, stack: str | None = None) -> dt.datetime:
    """When the graded stack was last applied, as an instant.

    `DescribeStacks` rather than the drift or event APIs: `LastUpdatedTime` is
    the one field that moves on every `aws cloudformation deploy` that changed
    anything, and it is present on a stack nobody has updated since creation
    only as `CreationTime` — which is why the fallback below is a fallback and
    not an error. A never-updated stack HAS a last change; it is the day it
    was created.

    ``stack`` defaults to `crucible.config.settings().stack_name`, the same
    `CRUCIBLE_STACK`-resolved name `crucible.autonomy.machine_principals` and
    `crucible.tags.audit_stack_tags` read — a second account or a renamed
    stack stays one variable.
    """
    from crucible.autonomy import _cfn_client  # noqa: PLC0415 - lazy, one call site
    from crucible.config import settings  # noqa: PLC0415 - one call site

    stack_name = stack or settings().stack_name
    # CONSTRUCTION is inside the guard, not above it (`alpha-engine-config-
    # I10328`). `boto3.client("cloudformation")` raises `NoRegionError` where no
    # region is configured, and the v2 box shell exports none — so with the
    # construction outside this try, every `weekly` arc on a box died at the
    # `console` stage rendering `gates/ladder.json`, which regressed phase 1's
    # `arc_runs_ok` from a gate READ. A clause that cannot build its client has
    # learned nothing about the system, which is UNMEASURABLE; it has not
    # learned that the system is broken.
    try:
        client = cfn if cfn is not None else _cfn_client()
        described = client.describe_stacks(StackName=stack_name)
    except Exception as exc:
        raise LastChangeUnreadableError(
            f"describe_stacks({stack_name!r}) failed: {type(exc).__name__}: {exc}. An "
            "undescribable stack — including one whose client could not be built at all, "
            "for want of a configured region — is UNMEASURABLE, never a window starting "
            "at zero"
        ) from exc
    stacks = described.get("Stacks") or []
    if not stacks:
        raise LastChangeUnreadableError(
            f"describe_stacks({stack_name!r}) returned no stack. The v2 environment is "
            "applied by CloudFormation, so no stack means there is nothing whose last "
            "change could bound a window"
        )
    when = stacks[0].get("LastUpdatedTime") or stacks[0].get("CreationTime")
    if when is None:
        raise LastChangeUnreadableError(
            f"stack {stack_name!r} reports neither LastUpdatedTime nor CreationTime, so "
            "the instant it last changed is unknown"
        )
    return _as_utc(when)


def _as_utc(value: dt.datetime) -> dt.datetime:
    """A tz-aware UTC instant. A naive datetime is REFUSED, not assumed UTC.

    Both producers here (`HeadObject`, `DescribeStacks`) return aware
    datetimes; a naive one means something substituted a value, and silently
    labelling it UTC would shift a window by up to a day in whichever
    direction the local zone happens to sit — a window boundary error that
    reads as a clean number.
    """
    if value.tzinfo is None:
        raise LastChangeUnreadableError(
            f"a change instant arrived without a timezone ({value.isoformat()}); "
            "assuming UTC would move the window boundary by the local offset"
        )
    return value.astimezone(dt.UTC)


def _last_system_change(store: Store, *, cfn: Any | None = None) -> _SystemChange:
    """`max(release pointer flip, stack last applied)`, with its provenance.

    **BOTH inputs, and neither substitutes for the other** (`alpha-engine-
    config-I10324`). A wheel flip changes what the box RUNS and leaves the
    stack untouched; a stack apply changes the box's ENVIRONMENT — its roles,
    schedules, topics, env vars — with no release flip at all, and the two
    fixes that made the v2 box able to page and able to run a full-universe
    weekly on 2026-09-09 were both the latter. Reading either alone would put
    the window's start before a change it could not see.
    """
    pointer_at = _pointer_flip_time(store)
    stack_at = _stack_last_updated(cfn)
    if stack_at > pointer_at:
        return _SystemChange(stack_at, "stack last applied", pointer_at, stack_at)
    if pointer_at > stack_at:
        return _SystemChange(pointer_at, "release pointer flip", pointer_at, stack_at)
    return _SystemChange(
        pointer_at, "release pointer flip and stack apply, same instant", pointer_at, stack_at
    )


def _event_instant(event_time: str) -> dt.datetime | None:
    """A CloudTrail `eventTime` as an instant, or None when it will not parse."""
    try:
        return _as_utc(dt.datetime.fromisoformat(event_time.replace("Z", "+00:00")))
    except (ValueError, TypeError, LastChangeUnreadableError):
        return None


def _clause_zero_human_mutating_calls(store: Store, window: list[dt.date]) -> Clause:
    """Plan §6 row 2 and §11 risk 8: zero human-originated mutating calls, over
    a window that starts at the system's LAST CHANGE.

    Read through `crucible.autonomy`, which walks the CloudTrail **S3
    archive**. `aws cloudtrail lookup-events` is forbidden there and the
    acceptance suite asserts the module has no path to it: the username lookup
    silently truncates to ~2 days, so a gate built on it reports zero because
    it looked at two days.

    **The window used to be the phase's rolling calendar window, and that
    measured the workbench** (`alpha-engine-config-I10324`). Measured
    2026-09-09 it read 167 human mutating calls over 2026-09-01..09-08 —
    dominated by the operator-gated CloudFormation applies and dispatches that
    MADE THE SYSTEM WORK. The v2 box could not page at all until 2026-09-09
    (`alpha-engine-config-I10156`) and could not run a full-universe
    `data.weekly` until the same day (the liquidity floor), so a window
    spanning the repair cannot tell the repair from an intervention. Brian's
    2026-09-04 ruling on phase 0 is the governing precedent: a phase completes
    on "fix and rerun until it is successful", not on a span nobody touched.

    So the window is `[last change, render day]`, and both halves of that
    construction are load-bearing:

    * `last change = max(release pointer flip, stack last applied)` —
      :func:`_last_system_change`, which documents why neither input covers the
      other. A pleasant consequence, and the reason there is NO carve-out here
      for operator-gated applies: an apply now DEFINES the window's start
      rather than violating it. A carve-out would be a second mechanism for the
      same thing, and the one that fails open the day an operator does
      something the allowlist did not anticipate.
    * the window must ALSO contain one complete weekly cycle whose close falls
      after the change, plus the daily cycles falling in that span
      (:func:`autonomy_earliest_satisfiable_render_day`). Without that, every
      change SHORTENS the window and makes this clause EASIER — an apply one
      minute before the read would leave a one-minute window containing nothing
      and read MET, which is the precise inverse of what the clause grades. Too
      short is UNMET, not unmeasurable: the input was perfectly readable and it
      said the system has not yet run a cycle unattended.

      **The minimum span is DERIVED from that cycle requirement, not declared
      beside it** (`alpha-engine-config-I10327`). It shipped as a
      `PHASE2_AUTONOMY_MIN_SPAN = 7 days` constant AND the cycle requirement,
      and for the attack the constant was written against the two are
      redundant: one hour cannot contain a complete weekly cycle. What the
      constant added was a second, arithmetically separate waiting period —
      four days past the weekly Step Function phase 2 has to exit on. Brian's
      2026-09-04 phase-0 ruling is the precedent and the same trade: *"we
      can't wait a week on phase 0, it should clear after this week's
      weekly"*, where the second week's protection was replaced by a daily
      guard rather than deleted. The anti-gaming property is unchanged and
      still tested.

    Actions are counted from the change instant, not from midnight of the
    change's day. `count_operator_actions` covers whole calendar days (the
    archive is day-partitioned), so the day of the change arrives carrying the
    operator's own applies; those are filtered here by `eventTime`. An
    `eventTime` that will not parse is COUNTED rather than dropped — for a
    clause asserting a count of zero, over-counting is investigated and
    under-counting is a gate that reads clean because it could not read, the
    same direction `crucible.autonomy._touches` chose.
    """
    from crucible.autonomy import (  # noqa: PLC0415 - heavy import, one call site
        ArchiveMissingError,
        count_operator_actions,
    )
    from crucible.config import settings  # noqa: PLC0415 - one call site

    name = "zero_human_mutating_calls"
    requirement = (
        "zero human-originated mutating calls touched a v2 resource between the "
        "system's last change (the later of the release pointer flip and the "
        f"{settings().stack_name} stack's last apply) and the render day, over a span "
        "containing one complete unattended weekly cycle closing after the change plus "
        f"at least {PHASE2_AUTONOMY_MIN_DAILY_CYCLES} complete daily cycle(s) in it, "
        "counted from the CloudTrail S3 archive (never `lookup-events`, which truncates "
        "its username lookup to ~2 days — §11 risk 8)"
    )
    archive = settings().cloudtrail_archive
    evidence = (archive, POINTER_KEY) if archive else (POINTER_KEY,)
    if not archive:
        return _unmeasurable(
            name,
            requirement,
            "no CloudTrail archive is configured (`CRUCIBLE_CLOUDTRAIL_ARCHIVE` is "
            "unset and there is no default, deliberately — a guessed bucket name "
            "produces a `NoSuchBucket` that reads like a permissions problem). "
            "Reporting 0 would make 'no trail' and 'no human touched it' the same "
            "answer",
            evidence,
        )
    try:
        change = _last_system_change(store)
    except LastChangeUnreadableError as exc:
        return _unmeasurable(
            name,
            requirement,
            f"the system's last change could not be established: {exc}",
            evidence,
        )
    render_day = window[-1]
    change_day = change.at.date()
    span = render_day - change_day
    cycle_close = weekly_anchor(render_day)
    daily_cycles = autonomy_daily_cycles_in_span(change_day, render_day)
    if cycle_close <= change_day:
        try:
            satisfiable_on = autonomy_earliest_satisfiable_render_day(change_day)
        except LastChangeUnreadableError as exc:
            return _unmeasurable(
                name,
                requirement,
                f"the earliest satisfiable render day could not be derived: {exc}",
                evidence,
            )
        return Clause(
            name,
            requirement,
            False,
            f"the most recent weekly close in the window, {cycle_close.isoformat()}, is "
            f"not after the system's last change ({change.at.isoformat()}, "
            f"{change.source}), so no complete weekly cycle has run unattended since "
            f"it — {span.days} day(s) to the render day {render_day.isoformat()}. This "
            f"reads UNMET until {satisfiable_on.isoformat()}, the first render day whose "
            "window contains a complete weekly cycle closing after the change plus "
            f"{PHASE2_AUTONOMY_MIN_DAILY_CYCLES} daily cycle(s). A window that shrank "
            "with every change would make this clause easier the more the system was "
            f"touched. {change.provenance()}",
            evidence,
        )
    if daily_cycles < PHASE2_AUTONOMY_MIN_DAILY_CYCLES:
        # UNMEASURABLE, not UNMET, and deliberately so: under the NYSE calendar
        # this branch is unreachable, because `weekly_anchor(render_day)` IS a
        # trading day in `(change_day, render_day]` whenever the guard above
        # passed, so a cycle closing after the change implies at least one daily
        # cycle. It is a self-check on that implication rather than a second
        # bar, and the honest reading when an implication the derivation rests
        # on fails is "the calendar contradicted itself", never a quiet UNMET
        # that would look like a system finding.
        return _unmeasurable(
            name,
            requirement,
            f"the window {change_day.isoformat()}..{render_day.isoformat()} carries a "
            f"weekly cycle closing {cycle_close.isoformat()}, after the change, and yet "
            f"counts {daily_cycles} complete daily cycle(s) — short of the "
            f"{PHASE2_AUTONOMY_MIN_DAILY_CYCLES} that close implies. The trading "
            "calendar contradicted itself, which is a statement about the calendar and "
            f"not about the system being measured. {change.provenance()}",
            evidence,
        )
    try:
        counted = count_operator_actions(
            _s3_client(),
            bucket=archive.removeprefix("s3://").partition("/")[0],
            prefix=archive.removeprefix("s3://").partition("/")[2],
            start=change.at.date(),
            end=render_day,
            # `alpha-engine-config-I10416` §11 row 9: the same reserved-
            # action exclusion the standing monthly board reading uses
            # (`crucible.board._read_human_touch_count`) — one config, both
            # callers, so an operator-declared exception applies wherever
            # "zero human mutating calls" is graded rather than only where
            # it happened to be added first.
            reserved=frozenset(settings().autonomy_reserved_events),
        )
    except ArchiveMissingError as exc:
        return _unmeasurable(name, requirement, f"ArchiveMissingError: {exc}", evidence)
    except Exception as exc:
        # Not a swallow: the failure mode is "the archive could not be read",
        # the primary deliverable (a gate reading) survives as an UNMEASURABLE
        # clause, and the recording surface is this clause's own detail, which
        # names the exception class. A raise here would take `crucible gate`,
        # `build_ladder` and the board render down together over a credential
        # that expired.
        return _unmeasurable(
            name,
            requirement,
            f"the CloudTrail archive could not be read: {type(exc).__name__}: {exc}. "
            "That is a statement about our access, not about the system being measured",
            evidence,
        )
    after_change = tuple(
        action
        for action in counted.actions
        if (instant := _event_instant(action.event_time)) is None or instant >= change.at
    )
    if after_change:
        offenders = ", ".join(
            f"{a.principal} {a.event_name}@{a.event_time}" for a in after_change[:4]
        )
        return Clause(
            name,
            requirement,
            False,
            f"{len(after_change)} human mutating call(s) over "
            f"{change.at.isoformat()}..{render_day.isoformat()}: {offenders}. "
            f"{change.provenance()}",
            evidence,
        )
    return Clause(
        name,
        requirement,
        True,
        f"0 human mutating calls over {span.days} days "
        f"({change.at.isoformat()}..{render_day.isoformat()}), spanning the weekly cycle "
        f"closing {cycle_close.isoformat()} and {daily_cycles} daily cycle(s), "
        f"{counted.records_scanned} records in {counted.objects_read} archive objects; "
        f"{len(counted.actions) - len(after_change)} call(s) on the change day itself "
        f"predate the change and are excluded. {change.provenance()}",
        evidence,
    )


def _clause_pages_within_ceiling(store: Store, window: list[dt.date]) -> Clause:
    """Plan §6 row 2: at most two paged INCIDENTS over the window.

    Incidents, not observations and not members: `crucible.alerts` files one
    bus row per incident under `alerts/{trading_day}/{incident}.json`, and
    five arms failing on one data outage is one page (§9.3).

    A listing that comes back empty is NOT zero pages unless something has
    actually swept. `alerts.sweep` is the only producer of that prefix, so an
    empty `runs/alerts.sweep/` means the ledger was never written by anything
    — no data, which is UNMEASURABLE rather than a clean month.

    **The count comes from `crucible.alerts.pages_in_range`, not from a copy
    of it here.** This clause and the `pages_per_20_trading_days` metric grade
    the same bus against the same ceiling; a second implementation here
    disagreed with the module at the first day of the window and restated the
    `alerts/{trading_day}/{incident}.json` shape as an integer arity outside
    `crucible.keys` — the failure class where `len(parts) != 4` silently
    dropped 100% of discriminated manifests (`alpha-engine-config-I9879`).

    **A deliberate exercise does not spend the budget** — Brian's 2026-09-09
    ruling, `alpha-engine-config-I10366` option (b), live from
    :data:`crucible.synthetic.SYNTHETIC_ROUTING_ACTIVE_FROM`. The clause read
    **6/2** on 2026-09-04 with induced and replayed runs making up the breach,
    and `nous-ergon-ops/runbooks/crucible-v2-fault-injection-2026-09-06.md`
    records that as the reason three of four plan §10.7 faults could not be
    induced for real: any day close enough to `now` for a live sweep to see a
    fresh failure is, by construction, inside the ceiling window. The
    exclusion reads the bus row's own `synthetic` field and nothing else.
    """
    name = "pages_within_ceiling"
    # `alpha-engine-config-I10366`, Brian's 2026-09-09 ruling (b): a
    # deliberate exercise does not spend a production alert budget. Resolved
    # from the moment of THIS READING, not from each row's day, so the whole
    # history is re-read under one rule on one day rather than the ceiling
    # carrying its old exercises for another twenty sessions. See
    # `crucible.synthetic.SYNTHETIC_ROUTING_ACTIVE_FROM` for why the switch is
    # a date and what it is not allowed to move before 2026-09-20.
    exclude_synthetic = synthetic_routing_active()
    requirement = (
        f"at most {PHASE2_MAX_PAGES} paged incidents over the window, counted from the "
        "alert bus (one row per incident, not per observation or per member)"
        + (
            "; a page whose bus row carries `synthetic` was a deliberate exercise and "
            "does not count against a production alert budget"
            if exclude_synthetic
            else ""
        )
    )
    sweeps = _list_store_keys(store, runs_prefix("alerts.sweep"))
    if sweeps.problem is not None:
        return _unmeasurable(name, requirement, sweeps.problem, (runs_prefix("alerts.sweep"),))
    if not sweeps.keys:
        return _unmeasurable(
            name,
            requirement,
            f"no `alerts.sweep` manifest exists under {runs_prefix('alerts.sweep')}, so "
            "nothing has ever written the alert bus. An empty bus beside a sweep that "
            "never ran is no data, not a month without pages",
            (runs_prefix("alerts.sweep"),),
        )
    start, end = window[0], window[-1]
    try:
        incidents = pages_in_range(store, start=start, end=end, exclude_synthetic=exclude_synthetic)
    except Exception as exc:
        # A denied or unreachable listing is a statement about OUR access. It
        # must be a red reading on the ladder, never an exception out of
        # `evaluate` taking `crucible gate` and the board render down with it.
        return _unmeasurable(
            name,
            requirement,
            f"listing {ALERTS_ROOT!r} could not be read: {type(exc).__name__}: {exc}. "
            "That is a statement about our access, not about the system being measured",
            (ALERTS_ROOT,),
        )
    if len(incidents) > PHASE2_MAX_PAGES:
        return Clause(
            name,
            requirement,
            False,
            f"{len(incidents)} paged incidents over {start.isoformat()}..{end.isoformat()}, "
            f"ceiling {PHASE2_MAX_PAGES}: {', '.join(sorted(incidents)[:4])}",
            tuple(sorted(incidents)),
        )
    return Clause(
        name,
        requirement,
        True,
        f"{len(incidents)} paged incident(s) over {start.isoformat()}..{end.isoformat()}, "
        f"ceiling {PHASE2_MAX_PAGES}",
        tuple(sorted(incidents)),
    )


def _job_stood_down(store: Store, job: str, trading_day: str) -> tuple[bool, str | None]:
    """Whether ``job``'s manifests for ``trading_day`` now include an `ok` one.

    Returns ``(stood_down, problem)``. A prefix that could not be listed is a
    problem naming the prefix, never "not stood down": the difference between
    "the rerun did not happen" and "we could not see whether it did" is the
    whole of this clause's honesty.
    """
    prefix = manifest_prefix(job, trading_day)
    listed = read_manifests_under(store, prefix)
    if listed.listing_problem is not None:
        return False, f"{prefix}: {listed.listing_problem}"
    for _key, manifest in listed.documents:
        if manifest.get("status") == "ok":
            return True, None
    return False, None


def _clause_pages_commissioned(store: Store) -> Clause:
    """Plan §9.3 "Commissioning" (a phase-2 exit-gate row): each page condition
    induced for real before it is trusted — fired, DELIVERED, and STOOD DOWN.

    The plan names the inducing means ("a forced failure and a withheld
    manifest"). This clause grades the OUTCOME those means exist to produce,
    read from the alert bus and the manifests, so a condition that fired on a
    genuine defect counts — and counts for more, since nothing about the
    condition was arranged. Measured 2026-09-06 on the production store:
    `failure.weekly@2026-09-04` (the first scheduled Saturday died on a
    missing IAM prefix) was delivered to `operator_chat` and stood down when
    the replay wrote an `ok` manifest; `absence@2026-09-04` (five arc stages
    past deadline) was delivered and stood down when the five manifests were
    written. Both conditions had therefore been commissioned by the real
    system a day before anyone forced them, and a clause that only counted a
    forced one would have read UNMET over that evidence.

    Three facts per condition, each from a durable artifact and none from a
    test double:

    1. **Fired** — a bus row under `alerts/{day}/{incident}.json` whose
       `condition` is this one. The bus is the machine-readable record §9.3
       requires; the transport's own log is not read.
    2. **Delivered** — that row's `sent` is `True`. `sent` records what the
       transport actually did, dedup- and mute-suppression subtracted
       (`crucible.alerts._transport_outcome`); a row with `sent: false` is a
       page that never left and proves the condition, not the path.
    3. **Stood down** — every member job named on the row now has an `ok`
       manifest for that trading day. Absence stands down when the manifest
       appears; failure stands down when a rerun overwrites the failed one.
       A condition that fired and was never cleared is an open incident, not
       a commissioned detector: the plan's row ends "and stood down".

    UNMEASURABLE when the sweep has never run (an empty bus beside no sweep
    is no data) or a listing could not be read. A row that is not an object,
    or names a condition this module does not know, is a malformed artifact
    and is reported in the detail rather than silently skipped; it never
    counts toward MET.

    Not windowed. Commissioning happens once per condition and stays true;
    re-reading it every render is what makes a later deletion of the bus
    visible, but the trailing window the other phase-2 clauses use would
    make a detector commissioned three weeks ago read as never commissioned.

    **A synthetic row commissions nothing** (`alpha-engine-config-I10366`,
    live from :data:`crucible.synthetic.SYNTHETIC_ROUTING_ACTIVE_FROM`). This
    is the clause the ruling's own Delta warns about: once a synthetic page
    routes to the muted topic its row would still read `sent: true` and could
    stand down, and this clause would go green on evidence that no human ever
    heard and that nothing about the real system produced. Commissioning is
    the claim that the detector fired on the REAL system — so a condition
    whose only rows are exercises reads "has never fired on the real system",
    with the exercises named rather than hidden.
    """
    name = "pages_commissioned"
    # The other half of Brian's 2026-09-09 ruling (`alpha-engine-config-I10366`),
    # and the one its own Delta names as the trap: routing synthetic pages
    # away from the operator topic would otherwise turn this clause green by
    # moving rows OUT OF ITS VIEW. A condition whose only evidence is an
    # exercise somebody launched on purpose is not commissioned — commissioning
    # is the claim that the detector fired on the REAL system and a human
    # heard it. Same date switch as the ceiling, one constant, so the two
    # clauses can never disagree about what a synthetic row is.
    exclude_synthetic = synthetic_routing_active()
    requirement = (
        "each page condition ("
        + ", ".join(PAGE_CONDITIONS)
        + ") has fired for real, been delivered (bus row `sent: true`) and stood down "
        "(every member job's manifest for that trading day now reads ok)"
        + (
            "; a row carrying `synthetic` is a deliberate exercise and commissions "
            "nothing — commissioning is a claim about the REAL system"
            if exclude_synthetic
            else ""
        )
    )
    sweeps = _list_store_keys(store, runs_prefix("alerts.sweep"))
    if sweeps.problem is not None:
        return _unmeasurable(name, requirement, sweeps.problem, (runs_prefix("alerts.sweep"),))
    if not sweeps.keys:
        return _unmeasurable(
            name,
            requirement,
            f"no `alerts.sweep` manifest exists under {runs_prefix('alerts.sweep')}, so "
            "nothing has ever evaluated either page condition. Nothing to commission",
            (runs_prefix("alerts.sweep"),),
        )
    rows = _list_store_keys(store, ALERTS_ROOT)
    if rows.problem is not None:
        return _unmeasurable(name, requirement, rows.problem, (ALERTS_ROOT,))

    stood_down: dict[str, list[str]] = {c: [] for c in PAGE_CONDITIONS}
    still_open: dict[str, list[str]] = {c: [] for c in PAGE_CONDITIONS}
    undelivered: dict[str, list[str]] = {c: [] for c in PAGE_CONDITIONS}
    synthetic: dict[str, list[str]] = {c: [] for c in PAGE_CONDITIONS}
    problems: list[str] = []
    access: list[str] = []
    for key in rows.keys:
        parsed = parse_bus_key(key)
        if parsed is None:
            continue
        day = parsed[0]
        read = _read_store_document(store, key)
        if read.problem is not None:
            (access if read.access_problem else problems).append(read.problem)
            continue
        row = read.document or {}
        condition = row.get("condition")
        if condition not in PAGE_CONDITIONS:
            problems.append(f"{key}: condition {condition!r} is not one of {PAGE_CONDITIONS}")
            continue
        if exclude_synthetic and row.get("synthetic"):
            # Counted and named, not dropped: a reader must be able to see
            # that the condition HAS fired on an exercise and still be told
            # it is not commissioned. Silently skipping it would make "never
            # fired" and "fired only in a drill" the same reading.
            synthetic[condition].append(key)
            continue
        if row.get("sent") is not True:
            undelivered[condition].append(key)
            continue
        problem = _field(key, row, "members", list)
        if problem is not None:
            problems.append(problem)
            continue
        jobs = sorted({str(m.get("job")) for m in row["members"] if isinstance(m, dict)})
        if not jobs:
            problems.append(f"{key}: `members` names no job")
            continue
        open_jobs: list[str] = []
        for job in jobs:
            down, why = _job_stood_down(store, job, day)
            if why is not None:
                access.append(why)
            if not down:
                open_jobs.append(job)
        if open_jobs:
            still_open[condition].append(f"{key} ({', '.join(open_jobs)} not yet ok)")
        else:
            stood_down[condition].append(key)

    met = all(stood_down[c] for c in PAGE_CONDITIONS)
    if not met and access:
        return _unmeasurable(
            name,
            requirement,
            f"{len(access)} read(s) could not be made, so at least one condition's standing-"
            "down could not be graded: " + " | ".join(sorted(access)[:3]),
            tuple(sorted(a.split(":")[0] for a in access)),
        )
    parts: list[str] = []
    for condition in PAGE_CONDITIONS:
        if stood_down[condition]:
            parts.append(
                f"{condition}: commissioned by {sorted(stood_down[condition])[0]}"
                + (
                    f" (+{len(stood_down[condition]) - 1} more)"
                    if len(stood_down[condition]) > 1
                    else ""
                )
            )
        elif still_open[condition]:
            parts.append(
                f"{condition}: delivered but not stood down: "
                + "; ".join(sorted(still_open[condition])[:2])
            )
        elif undelivered[condition]:
            parts.append(
                f"{condition}: fired but never delivered (`sent: false`): "
                + ", ".join(sorted(undelivered[condition])[:2])
            )
        elif synthetic[condition]:
            parts.append(
                f"{condition}: has never fired on the real system — "
                f"{len(synthetic[condition])} synthetic row(s) "
                f"({sorted(synthetic[condition])[0]}) are deliberate exercises and "
                "commission nothing"
            )
        else:
            parts.append(f"{condition}: has never fired")
    if problems:
        parts.append(f"{len(problems)} malformed bus row(s) ignored: {sorted(problems)[0]}")
    evidence = tuple(sorted(k for c in PAGE_CONDITIONS for k in stood_down[c]))
    return Clause(name, requirement, met, "; ".join(parts), evidence)


def _clause_two_page_conditions_on_real_channel(store: Store) -> Clause:
    """`alpha-engine-config-I9758` deliverable 2: both declared page
    conditions route to the REAL operator channel, read from the bus row's
    own transport fields.

    **How this differs from `pages_commissioned`, which reads the same rows.**
    That clause grades the LIFECYCLE of an incident — fired, delivered, stood
    down — and treats `sent: true` as the whole delivery fact, whatever
    channel the transport used. This one grades the CHANNEL and nothing else:
    a row's `destination`, which `crucible.alerts._transport_outcome` fills
    from what the transport actually did. The two come apart in both
    directions, which is why neither can stand in for the other:

    * a page routed to the muted v1 overlap topic (`legacy=True`) can read
      `sent: true` and stand down, so `pages_commissioned` counts it — and
      nobody on the operator channel ever heard it;
    * a page delivered on the real channel and never cleared is on the real
      channel and is not commissioned.

    **What it refuses to say.** The row records ONE destination — krepis'
    `telegram_destination` when the Telegram leg was reached, else the send's
    own state — so it can say which channel was reached, and cannot say that
    every declared leg was. This clause therefore grades the recorded
    destination and makes no claim about per-leg delivery; a clause that read
    `operator_chat` as proof the SNS leg also published would be inferring a
    fact from the absence of a field. The SNS leg's own liveness is the
    heartbeat's confirmed-subscriber reading, a different artifact with a
    different producer.

    **A condition with no row at all is UNMEASURABLE, not UNMET.** "Never
    fired" and "not deliverable" are different facts, and the store cannot
    tell them apart: a condition that has had nothing to page about has left
    no delivery record to read. A condition whose rows exist and are ALL
    undelivered or non-operator-destined is a different reading entirely —
    that is evidence, and it is UNMET.

    Not windowed, for `pages_commissioned`'s reason: a channel demonstrated
    once stays demonstrated, and a trailing window would make a condition
    proved three weeks ago read as never proved.
    """
    name = "two_page_conditions_on_real_channel"
    requirement = (
        "each page condition ("
        + ", ".join(PAGE_CONDITIONS)
        + ") has at least one delivered bus row (`sent: true`) whose `destination` is an "
        "operator-facing channel rather than a send state ("
        + ", ".join(sorted(NON_OPERATOR_DESTINATIONS))
        + ")"
    )
    rows = _list_store_keys(store, ALERTS_ROOT)
    if rows.problem is not None:
        return _unmeasurable(name, requirement, rows.problem, (ALERTS_ROOT,))

    delivered: dict[str, dict[str, list[str]]] = {c: {} for c in PAGE_CONDITIONS}
    withheld: dict[str, list[str]] = {c: [] for c in PAGE_CONDITIONS}
    problems: list[str] = []
    access: list[str] = []
    for key in sorted(rows.keys or []):
        if parse_bus_key(key) is None:
            continue
        read = _read_store_document(store, key)
        if read.problem is not None:
            (access if read.access_problem else problems).append(read.problem)
            continue
        row = read.document or {}
        condition = row.get("condition")
        if condition not in PAGE_CONDITIONS:
            problems.append(f"{key}: condition {condition!r} is not one of {PAGE_CONDITIONS}")
            continue
        destination = str(row.get("destination") or "").strip()
        if not destination:
            problems.append(
                f"{key}: carries no `destination`, so what channel this page reached "
                "cannot be read from the row at all"
            )
            continue
        if row.get("sent") is not True or destination in NON_OPERATOR_DESTINATIONS:
            withheld[condition].append(f"{key} (sent={row.get('sent')!r}, {destination})")
            continue
        delivered[condition].setdefault(destination, []).append(key)

    if access:
        return _unmeasurable(
            name,
            requirement,
            f"{len(access)} bus row(s) could not be read, so at least one condition's "
            "delivery channel could not be graded: " + " | ".join(sorted(access)[:3]),
            (ALERTS_ROOT,),
        )
    parts: list[str] = []
    contradicted: list[str] = []
    silent: list[str] = []
    for condition in PAGE_CONDITIONS:
        if delivered[condition]:
            channels = sorted(delivered[condition])
            first = sorted(delivered[condition][channels[0]])[0]
            parts.append(f"{condition}: delivered to {', '.join(channels)} ({first})")
        elif withheld[condition]:
            contradicted.append(condition)
            parts.append(
                f"{condition}: every row was withheld or non-operator-destined: "
                + "; ".join(sorted(withheld[condition])[:2])
            )
        else:
            silent.append(condition)
            parts.append(f"{condition}: no bus row exists, so no delivery record names a channel")
    if problems:
        parts.append(f"{len(problems)} malformed bus row(s) ignored: {sorted(problems)[0]}")
    evidence = tuple(
        sorted(k for c in PAGE_CONDITIONS for keys in delivered[c].values() for k in keys)
    )
    if contradicted:
        return Clause(name, requirement, False, "; ".join(parts), evidence)
    if silent:
        return _unmeasurable(name, requirement, "; ".join(parts), evidence or (ALERTS_ROOT,))
    return Clause(name, requirement, True, "; ".join(parts), evidence)


@lru_cache(maxsize=1)
def _manifest_retry_reasons() -> frozenset[str]:
    """The retry classes the CURRENT manifest schema can record, read from it.

    The schema's `AttemptRow.reason` enum minus
    :data:`MANIFEST_ATTEMPT_INITIAL`. Derived, never restated: the question
    the retry clause asks is whether a transient retry is RECORDABLE, and
    that is a question about the contract. A copy of the vocabulary here
    would answer it from something that drifts.
    """
    schema = load_schema()
    row = schema.get("$defs", {}).get("AttemptRow", {})
    enum = row.get("properties", {}).get("reason", {}).get("enum", [])
    return frozenset(str(value) for value in enum) - {MANIFEST_ATTEMPT_INITIAL}


def _clause_transient_retry_class_in_runner(store: Store, registry: dict[str, Component]) -> Clause:
    """`alpha-engine-config-I9758` deliverable 3: the §11.2 transient-retry
    class is implemented in the runner — graded from a manifest that RECORDS
    a retry, never from the runner's source.

    A clause that read `crucible/runner.py` would grade the code that wrote
    it, which is plan §11 risk 1 in one line: "v2 passes its gates because
    its tests were written to pass". What is graded here is the artifact the
    class produces when it fires — a run manifest whose `attempts[]` carries
    a second attempt naming one of the declared classes.

    **"Nothing retried" and "retry cannot be recorded" are two readings, and
    the contract is what separates them.** Before looking at any manifest
    this asks the schema which retry classes it can express, and compares
    that vocabulary with `crucible.runner.TRANSIENT_CLASSIFIERS`:

    * a class the runner declares and the schema cannot record is a retry
      that could fire and leave no trace — UNMEASURABLE, naming the classes,
      because no artifact could ever answer the question;
    * a class the schema records and the runner does not implement is the
      inverse defect, reported as a problem;
    * with the two in agreement and no retried attempt anywhere on the
      store, the reading is UNMEASURABLE and says so in those words: the
      class is implemented and recordable, and nothing transient has
      happened yet. Never MET — a retry nobody has seen fire is a detector
      nobody knows works (AGENTS.md, test discipline).

    Not windowed. A retry, once recorded, stays recorded, and phase 2's raw
    window is two dates a week apart — a window that would make this clause
    answer "no transient in the last fortnight" while a recorded one sat on
    the store. Every registry job's manifest prefix is read through
    `read_manifests_under`, the one sanctioned prefix reader (rule 1).
    """
    name = "transient_retry_class_in_runner"
    declared = frozenset(reason for reason, _types, _needles in TRANSIENT_CLASSIFIERS)
    requirement = (
        "a run manifest records a retried attempt whose `reason` is one of the runner's "
        f"declared transient classes ({', '.join(sorted(declared))}), and the manifest "
        "schema can record every class the runner declares"
    )
    recordable = _manifest_retry_reasons()
    if not recordable:
        return _unmeasurable(
            name,
            requirement,
            "the current run-manifest schema declares no retry vocabulary at all "
            f"(`$defs.AttemptRow.reason` beyond {MANIFEST_ATTEMPT_INITIAL!r}), so a "
            "transient retry could fire and leave no trace. Nothing on the store could "
            "answer this question",
        )
    unrecordable = sorted(declared - recordable)
    if unrecordable:
        return _unmeasurable(
            name,
            requirement,
            f"the runner declares transient classes the manifest schema cannot record: "
            f"{unrecordable}. A retry of one of these leaves no artifact, so its absence "
            "from the store says nothing about whether it fired",
        )
    unimplemented = sorted(recordable - declared)

    retried: list[str] = []
    retried_keys: list[str] = []
    unknown: list[str] = []
    listing_problems: list[str] = []
    faults: list[str] = []
    manifests = 0
    for job in sorted(registry):
        read = read_manifests_under(store, runs_prefix(job))
        if read.listing_problem is not None:
            listing_problems.append(read.listing_problem)
            continue
        faults.extend(f"{key}: {problem}" for key, problem in sorted(read.faults.items()))
        for key, document in read.documents:
            manifests += 1
            attempts = document.get("attempts")
            if not isinstance(attempts, list):
                faults.append(f"{key}: `attempts` is {type(attempts).__name__}, not a list")
                continue
            for attempt in attempts:
                if not isinstance(attempt, dict):
                    faults.append(f"{key}: an `attempts` row is not an object")
                    continue
                if attempt.get("n") in (None, 1):
                    continue
                reason = str(attempt.get("reason") or "")
                if reason in declared:
                    retried.append(f"{key} (attempt {attempt.get('n')}: {reason})")
                    retried_keys.append(key)
                else:
                    unknown.append(f"{key} (attempt {attempt.get('n')}: {reason!r})")

    if listing_problems:
        return _unmeasurable(
            name,
            requirement,
            f"{len(listing_problems)} manifest prefix(es) could not be listed, so a "
            "recorded retry could be sitting in the part that was not read: "
            + " | ".join(sorted(listing_problems)[:3]),
        )
    parts: list[str] = []
    if unimplemented:
        parts.append(f"the schema records classes the runner does not implement: {unimplemented}")
    if unknown:
        parts.append(
            f"{len(unknown)} retried attempt(s) name a reason outside the declared class: "
            + "; ".join(sorted(unknown)[:2])
        )
    if faults:
        parts.append(f"{len(faults)} unreadable manifest(s): {sorted(faults)[0]}")
    if unknown or unimplemented:
        return Clause(name, requirement, False, "; ".join(parts), tuple(sorted(retried_keys)))
    if retried:
        parts.insert(
            0,
            f"{len(retried)} recorded transient retry(ies) across {manifests} manifests: "
            + "; ".join(sorted(retried)[:2]),
        )
        return Clause(name, requirement, True, "; ".join(parts), tuple(sorted(retried_keys)))
    parts.insert(
        0,
        f"{manifests} manifests across {len(registry)} registry jobs record no attempt "
        f"beyond the first. The class is declared ({', '.join(sorted(declared))}) and the "
        "schema can record every one of them, so this reads as `nothing transient has "
        "occurred`, NOT as `retry is not implemented` — and neither of those is met",
    )
    return _unmeasurable(name, requirement, "; ".join(parts))


def _clause_fault_injection_against_scheduled_path(store: Store) -> Clause:
    """`alpha-engine-config-I9758` deliverable 4: plan §10.7's fault injection
    was run against the live scheduled path — graded from a durable record
    per scripted fault, each naming the manifest and the bus row it produced.

    §10.7's exercise leaves artifacts behind on the real path, and WHICH
    artifacts depends on how the fault ended. A record that merely says
    "fault 2 was induced" is a sentence; a record naming keys is checkable,
    and this clause checks them — every key it names must exist on the store
    and be of the right shape. A record whose named manifest is absent is a
    claim the store contradicts, and that is the reading this clause exists to
    make possible: the exercise used to be written up in a runbook nothing
    parses, so "we ran fault injection" and "we did not" rendered identically
    on every surface.

    **The record carries three OUTCOME KINDS and this clause grades each on
    its own evidence** (`alpha-engine-config-I10327`). It required
    `manifest_key` AND `bus_key` of every record, which made it unsatisfiable
    for the two faults that never produce a failed run — fault 1's designed
    outcome is a spot interruption the runner's declared transient class
    ABSORBS (the manifest reads `ok`), and fault 4's state cannot be entered
    at all:

    * `induced` — the fault fired and the job failed. Both keys required, both
      read off the store.
    * `absorbed` — the fault fired and the system handled it. `manifest_key`
      required; `bus_key` must be ABSENT, and a record carrying one is
      CONTRADICTED, not tolerated — a page here would mean the retry did not
      work, so the absence is part of the claim.
    * `unreachable` — the state cannot be entered. No `run_id` and no keys at
      all; the evidence is `closed_paths`, one machine-executed probe per
      closed path, which `crucible.faults` refused to file unless every probe
      observed what it required.

    The per-outcome matrix is NOT re-derived here: every record is validated
    against `crucible.models.FaultRecordDocument`, the same model the producer
    validates before writing, so a record shape the producer could not have
    written is a record this reader refuses. A second copy of the matrix in
    this function would be the half that drifts.

    **Three readings, deliberately distinct.**

    * A record that does not conform, or names keys the store does not hold,
      or keys of the wrong shape — UNMET. That is evidence, positively read.
    * No record for a fault — UNMEASURABLE, naming the exact key that is
      missing. An absent record means the exercise is unrecorded, never that
      it did not happen.
    * Every scripted fault has a conforming record whose named keys are
      present — MET.

    Not windowed: a fault exercise is a one-time event whose record stays
    true, the same reason `pages_commissioned` is not windowed.
    """
    name = "fault_injection_against_scheduled_path"
    requirement = (
        f"each of the {len(SCRIPTED_FAULTS)} scripted faults (plan §10.7) has a "
        f"conforming record under {FAULT_INJECTION_ROOT} declaring an "
        f"`{FAULT_RECORD_OUTCOME_FIELD}` and carrying exactly that outcome's evidence — "
        f"`{FAULT_RECORD_MANIFEST_FIELD}` plus `{FAULT_RECORD_BUS_FIELD}` for `induced`, "
        f"`{FAULT_RECORD_MANIFEST_FIELD}` with NO `{FAULT_RECORD_BUS_FIELD}` for "
        "`absorbed`, machine-checked `closed_paths` and no run for `unreachable` — with "
        "every key it names present on the store"
    )
    listed = _list_store_keys(store, FAULT_INJECTION_ROOT)
    if listed.problem is not None:
        return _unmeasurable(name, requirement, listed.problem, (FAULT_INJECTION_ROOT,))

    records: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    problems: list[str] = []
    access: list[str] = []
    for key in sorted(listed.keys or []):
        parsed = parse_fault_injection_key(key)
        if parsed is None:
            continue
        _day, fault_id = parsed
        read = _read_store_document(store, key)
        if read.problem is not None:
            (access if read.access_problem else problems).append(read.problem)
            continue
        records.setdefault(fault_id, []).append((key, read.document or {}))

    if access:
        return _unmeasurable(
            name,
            requirement,
            f"{len(access)} fault record(s) could not be read: " + " | ".join(sorted(access)[:3]),
            (FAULT_INJECTION_ROOT,),
        )
    evidence: list[str] = []
    missing: list[str] = []
    contradicted: list[str] = []
    parts: list[str] = []
    for fault in SCRIPTED_FAULTS:
        filed = records.get(fault) or []
        if not filed:
            missing.append(fault)
            parts.append(f"{fault}: no record at {FAULT_INJECTION_ROOT}<trading-day>/{fault}.json")
            continue
        key, document = sorted(filed)[-1]
        named: list[str] = []
        bad: list[str] = []
        # Conformance FIRST, through the producer's own model: the
        # field-by-outcome matrix (which evidence each kind requires and which
        # it forbids) is enforced in ONE place, and a record shape the producer
        # could not have written is a record this reader refuses rather than
        # grades on whichever fields happen to be populated. This is also what
        # catches an `absorbed` record carrying a `bus_key`, and an
        # `unreachable` one carrying a `run_id` it could excuse a manifest
        # with.
        try:
            FaultRecordDocument.model_validate(document)
        except ValidationError as exc:
            contradicted.append(fault)
            first = exc.errors()[0]
            where = ".".join(str(part) for part in first["loc"]) or "<root>"
            parts.append(
                f"{fault}: {key} does not conform to fault_record.v1 "
                f"({len(exc.errors())} error(s), first at {where}: {first['msg']})"
            )
            continue
        outcome = str(document.get(FAULT_RECORD_OUTCOME_FIELD))
        # Which keys this outcome names on the store. `unreachable` names
        # none: its evidence is `closed_paths`, already validated above, and a
        # reader that demanded a store key of it would be the defect I10327
        # removed.
        expected_keys = (
            ((FAULT_RECORD_MANIFEST_FIELD, is_manifest_key, "a run manifest key"),)
            if outcome == "absorbed"
            else ()
            if outcome != FAULT_OUTCOME_INDUCED
            else (
                (FAULT_RECORD_MANIFEST_FIELD, is_manifest_key, "a run manifest key"),
                (
                    FAULT_RECORD_BUS_FIELD,
                    lambda k: parse_bus_key(k) is not None,
                    "an alert bus key",
                ),
            )
        )
        for field_name, shape_ok, shape in expected_keys:
            value = str(document.get(field_name) or "").strip()
            if not value:
                bad.append(f"names no `{field_name}`")
                continue
            if not shape_ok(value):
                bad.append(f"`{field_name}` {value!r} is not {shape}")
                continue
            read = _read_store_bytes(store, value)
            if read.problem is not None:
                access.append(read.problem)
                continue
            if read.absent:
                bad.append(f"`{field_name}` names {value}, which the store does not hold")
                continue
            named.append(value)
        if bad:
            contradicted.append(fault)
            parts.append(f"{fault}: {key} ({outcome}) " + "; ".join(bad))
        else:
            evidence.append(key)
            evidence.extend(named)
            probes = document.get("closed_paths") or []
            shown = (
                ", ".join(str(row.get("probe")) for row in probes) if probes else ", ".join(named)
            )
            parts.append(f"{fault}: {key} ({outcome}) -> {shown}")
    if problems:
        parts.append(f"{len(problems)} malformed record(s) ignored: {sorted(problems)[0]}")
    if access:
        return _unmeasurable(
            name,
            requirement,
            f"{len(access)} artifact(s) a fault record names could not be read: "
            + " | ".join(sorted(access)[:3]),
            tuple(sorted(evidence)) or (FAULT_INJECTION_ROOT,),
        )
    if contradicted:
        return Clause(name, requirement, False, "; ".join(parts), tuple(sorted(evidence)))
    if missing:
        return _unmeasurable(
            name,
            requirement,
            "; ".join(parts),
            tuple(sorted(evidence)) or (FAULT_INJECTION_ROOT,),
        )
    return Clause(name, requirement, True, "; ".join(parts), tuple(sorted(evidence)))


def _runbook_sections(text: str) -> dict[str, str]:
    """Every `### <verb>` subsection of the README's Runbook section.

    The Runbook section only: a `### heal` under some other `##` heading is
    not the runbook, and a scan of the whole file would grade it as though it
    were. Keys are lowercased headings, values the body up to the next
    heading of either depth.
    """
    if RUNBOOK_SECTION not in text:
        return {}
    body = text.split(RUNBOOK_SECTION, 1)[1]
    for line in body.splitlines():
        if line.startswith("## "):
            body = body.split("\n" + line, 1)[0]
            break
    sections: dict[str, str] = {}
    current: str | None = None
    collected: list[str] = []
    for line in body.splitlines():
        if line.startswith(RUNBOOK_PROCEDURE_PREFIX):
            if current is not None:
                sections[current] = "\n".join(collected)
            current = line[len(RUNBOOK_PROCEDURE_PREFIX) :].strip().lower()
            collected = []
            continue
        if current is not None:
            collected.append(line)
    if current is not None:
        sections[current] = "\n".join(collected)
    return sections


def _parses_against_the_cli(job: str, rest: str) -> str | None:
    """``None`` if `crucible <job> <rest>` parses, else why it does not.

    The real parser is the oracle — a renamed flag, a removed job or a newly
    required argument fails HERE rather than in an operator's terminal at the
    moment they most need the runbook. Imported lazily: `crucible.cli`
    imports this module, and a top-level import would be a cycle.
    """
    from crucible.cli import JOBS, build_parser  # noqa: PLC0415 - cycle: cli imports gate

    if job not in JOBS:
        return f"names job {job!r}, which the CLI does not carry"
    try:
        argv = [job, *shlex.split(rest.split("#", 1)[0])]
    except ValueError as exc:
        return f"is not a parseable command line: {exc}"
    stderr = io.StringIO()
    try:
        with redirect_stderr(stderr):
            build_parser().parse_args(argv)
    except SystemExit:
        return f"does not parse: {' '.join(argv)} -> {stderr.getvalue().strip().splitlines()[-1:]}"
    return None


def _cli_carries_job(job: str) -> bool:
    """Whether the CLI carries ``job`` at all — the reserved procedures' test."""
    from crucible.cli import JOBS  # noqa: PLC0415 - cycle: cli imports gate

    return job in JOBS


def _clause_runbook_in_readme() -> Clause:
    """`alpha-engine-config-I9758` deliverable 6: the runbook is in the README,
    and every command it publishes still parses against the real CLI.

    Presence alone is not the deliverable and never was. `crucible-PR163` made
    the point for the other runbook: a procedure whose commands no longer parse
    is worse than no procedure, because it fails halfway, in front of somebody
    who reached for it under pressure. So the parser is the oracle here too —
    each command is extracted from the runbook's own fenced blocks and handed
    to `crucible.cli.build_parser`, and a renamed flag turns this clause red
    rather than an operator's terminal.

    **A reserved procedure is graded the other way round.** `unseal` is a
    human ruling and never an automated action (plan §9.4); the README says
    so and names no command. Requiring "a command that parses" from every
    procedure would mean this clause could only go green if someone built the
    exact CLI surface the plan forbids. So a reserved procedure must name NO
    command AND the CLI must carry no such job — the clause refuses a missing
    procedure and an invented one with equal force (see
    :data:`RUNBOOK_PROCEDURES`).

    Read from the repository, not the store, for :data:`ACCEPTANCE_RATCHET_PATH`'s
    reason: the runbook's existence is a property of the checkout and git is its
    durable record. Absent — a wheel install, where the README does not ship —
    the clause is UNMET with the path named, never a pass and never
    unmeasurable: this reader knows exactly which file it wanted and that it is
    not there.
    """
    name = "runbook_in_readme"
    requirement = (
        "README carries a "
        + RUNBOOK_SECTION
        + " section naming "
        + ", ".join(verb for verb, _reserved in RUNBOOK_PROCEDURES)
        + "; every `crucible ...` command it publishes parses against the real CLI, and "
        "each reserved procedure names no command and no such job exists"
    )
    evidence = (str(README_PATH),)
    if not README_PATH.is_file():
        return Clause(
            name,
            requirement,
            False,
            f"{README_PATH} is absent, so the runbook cannot be read here",
            evidence,
        )
    try:
        text = README_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        return _unmeasurable(name, requirement, f"{README_PATH} could not be read: {exc}", evidence)
    sections = _runbook_sections(text)
    if not sections:
        return Clause(
            name,
            requirement,
            False,
            f"{README_PATH} carries no `{RUNBOOK_SECTION}` section with "
            f"`{RUNBOOK_PROCEDURE_PREFIX}` procedures under it",
            evidence,
        )
    parts: list[str] = []
    problems: list[str] = []
    for verb, reserved in RUNBOOK_PROCEDURES:
        body = sections.get(verb)
        if body is None:
            problems.append(f"{verb}: no `{RUNBOOK_PROCEDURE_PREFIX}{verb}` section")
            continue
        commands = _RUNBOOK_COMMAND_RE.findall(body)
        if reserved:
            if commands:
                problems.append(
                    f"{verb}: reserved (plan §9.4) but publishes "
                    f"{len(commands)} command(s): {commands[0][0]}"
                )
            elif _cli_carries_job(verb):
                problems.append(
                    f"{verb}: documented as reserved, but the CLI carries a {verb!r} job"
                )
            else:
                parts.append(f"{verb}: reserved, no command, no such job")
            continue
        if not commands:
            problems.append(f"{verb}: the section publishes no `crucible ...` command")
            continue
        broken = [
            f"`crucible {job}{rest}` {why}"
            for job, rest in commands
            if (why := _parses_against_the_cli(job, rest)) is not None
        ]
        if broken:
            problems.extend(f"{verb}: {b}" for b in broken)
        else:
            parts.append(f"{verb}: {len(commands)} command(s), all parse")
    if problems:
        return Clause(name, requirement, False, "; ".join(problems), evidence)
    return Clause(name, requirement, True, "; ".join(parts), evidence)


#: Complete days whose TOTAL is graded against the monthly ceiling
#: (`alpha-engine-config-I9927`): the ceiling's own period, so a weekly batch
#: lands in the window four or five times and a month-boundary accrual once,
#: whatever weekday the gate is read on. Fewer complete daily periods than
#: this is UNMEASURABLE, never a guess.
COST_TRAILING_DAYS = 30

#: Complete days in the leading-pace grade: the trailing week's mean ×
#: COST_TRAILING_DAYS is a hard-UNMET when it exceeds the ceiling. Prefer a
#: short false-UNMET after a month-boundary lump in this window over a
#: false-MET while spend is accelerating (alpha-engine-config-I9927).
COST_LEADING_DAYS = 7


def _ce_client() -> Any:
    from crucible.cost import default_client  # noqa: PLC0415 - lazy on purpose

    return default_client()


def _clause_aws_cost_within_ceiling(
    window: list[dt.date], *, name: str, ceiling_usd: float, tagged: bool
) -> Clause:
    """Plan §6 row 2 (`<= $40`, tagged) and row 4 (`<= $70/mo`, whole account).

    One reader at two ceilings and two scopes, for the reason
    `_clause_old_weekly_within_cadence` takes a ceiling: a second copy of a
    reading is a second contract.

    **A denied, unregioned or uncredentialed read is UNMEASURABLE, never
    `$0.00`.** Measured 2026-09-03 from the laptop: the human admin identity
    CAN read Cost Explorer — the account total came back `$9.05` and the
    tag-filtered total `$0.00` — so the denial this was written against is
    not the whole gap. What remains is that the SCHEDULED reader (the board
    render) holds no such grant, and that the tag returning `$0.00` measures
    the filter rather than the spend. Both are tracked; the grant is
    deliberately not made in this PR, which changes no IAM.

    **A monthly ceiling is two readings, and a partial month is never MET on
    its own.** The first reading (`alpha-engine-config-I9913`) compared
    month-to-date spend against the whole month's ceiling, so on 2026-09-02
    phase 4 read `MET: $9.05 of $70.00` on two days of a month — green by
    default on the 1st of every month, able only to fall out of MET later
    (`alpha-engine-config-I9927`). A pro-rata line (`ceiling x elapsed /
    days_in_month`) was the first fix and was wrong the other way: this
    account posts a month-boundary lump on the 1st (`$15.59` on 2026-07-01,
    `$30.88` on 2026-08-01 — Savings Plan and reserved accruals against a
    `$2.50–4.00`/day baseline), so a compliant month reads OVER the line until
    the lump is amortised, deterministically, for most of the month.

    A projection from a trailing-week MEDIAN was the second cut and was wrong
    for the estate's real shape: eight of the eleven `components.yaml` rows
    are Saturday-weekly and the Saturday spot run is the expensive compute, so
    with one batch day in seven the median is a quiet day by construction —
    `$60` every Saturday and `$0.20` otherwise (~`$245`/month) projected
    `$66.00` MET until month-to-date itself crossed the ceiling on the 10th,
    after the month was blown. Today's daily v1 residue is the only thing that
    hid it, and it leaves at exactly the phase-4 exit this clause grades.

    So the clause matches the window to the ceiling's own period and grades:

    1. **Hard UNMET** when month-to-date spend already exceeds the FULL
       ceiling — the fast path; no trailing read is needed to know a breached
       month is breached.
    2. **The trailing COST_TRAILING_DAYS (30) complete days' TOTAL** against
       the full ceiling. No projection, no mean, no median: thirty days of
       spend is a month of spend whatever weekday the batch lands on, and the
       window crosses the month boundary, so it is measurable on day 1 and a
       month-boundary lump is one day in thirty. Fewer daily periods than
       days, an unreadable window, or a `$0.00` total is UNMEASURABLE.
    3. **Hard UNMET** when the leading COST_LEADING_DAYS (7) mean ×
       COST_TRAILING_DAYS exceeds the ceiling — the previously ungraded
       leading indicator, promoted. Trailing-30 alone false-METs an
       accelerating month (day-5 MTD `$40` → ~`$240`/mo implied while
       trailing `$65` still sits under a `$70` ceiling). Prefer a short
       false-UNMET after a month-boundary lump lands inside the leading
       window over a false-MET during acceleration.

    **A completed calendar month is also read, in this same function** — on
    renders taken the 1st through the 3rd of the following month, against
    this same ceiling (`alpha-engine-config-I9946`). Cost Explorer finalises
    a month a few days into the next, so a read taken any later would just
    restate month-to-date under another name; a render on the 4th or later
    does not attempt it. A closed month over the ceiling is a hard UNMET,
    independent of what the new month itself projects — a month that
    projected under all the way through and then closed over must leave a
    red row somewhere, and the in-progress readings above can never produce
    one for a month that has already ended.
    """
    from crucible.cost import (  # noqa: PLC0415
        CostUnreadableError,
        closed_month_usd,
        month_to_date_usd,
        trailing_daily_usd,
    )

    scope = f"tagged `{TAG_KEY}={TAG_VALUE}`" if tagged else "the whole account"
    requirement = (
        f"AWS spend for {scope} is at most ${ceiling_usd:.2f}/month, read from Cost "
        f"Explorer: UNMET once month-to-date exceeds it, otherwise graded on the total of "
        f"the trailing {COST_TRAILING_DAYS} complete days, UNMET when the leading "
        f"{COST_LEADING_DAYS}-day mean × {COST_TRAILING_DAYS} exceeds the ceiling, and UNMET "
        "when the prior CLOSED calendar month (read on the 1st-3rd) exceeded it"
    )
    evidence = ("ce:GetCostAndUsage",)

    # I9946: evaluated first and prefixed onto every detail below, so a
    # closed-month-over-ceiling UNMET is not shadowed by a MET in-progress
    # reading, and a closed-month-under or PROVISIONAL reading is visible
    # beside whichever in-progress verdict follows.
    closed_note = ""
    if window[-1].day <= 3:
        try:
            closed = closed_month_usd(_ce_client(), today=window[-1], tagged=tagged)
        except CostUnreadableError as exc:
            closed_note = f"prior closed month could not be read: CostUnreadableError: {exc}; "
        except Exception as exc:
            closed_note = (
                f"prior closed month could not be read: {type(exc).__name__}: {exc}. That is "
                "a statement about our access, not about what was spent; "
            )
        else:
            if closed.amount_usd == 0.0:
                # The same $0.00 trap as the in-progress readings, one level
                # up: a correctly-tagged closed month with no resources and an
                # entirely untagged estate both read $0.00 here.
                closed_note = (
                    f"closed month {closed.start.isoformat()}..{closed.end.isoformat()} for "
                    f"{closed.scope} read exactly $0.00 — not graded (that is what a "
                    "misfiltered or broken reading returns as well as a free month); "
                )
            elif closed.estimated:
                closed_note = (
                    f"closed month {closed.start.isoformat()}..{closed.end.isoformat()} "
                    f"${closed.amount_usd:.2f}: PROVISIONAL, Cost Explorer has not finalised "
                    "it yet — not graded; "
                )
            elif closed.amount_usd > ceiling_usd:
                return Clause(
                    name,
                    requirement,
                    False,
                    f"closed month {closed.start.isoformat()}..{closed.end.isoformat()} for "
                    f"{closed.scope} was ${closed.amount_usd:.2f}, over the "
                    f"${ceiling_usd:.2f}/month ceiling — UNMET regardless of what the new "
                    "month itself projects",
                    evidence,
                )
            else:
                closed_note = (
                    f"closed month {closed.start.isoformat()}..{closed.end.isoformat()} "
                    f"${closed.amount_usd:.2f}: under; "
                )

    try:
        reading = month_to_date_usd(_ce_client(), today=window[-1], tagged=tagged)
    except CostUnreadableError as exc:
        return _unmeasurable(name, requirement, f"CostUnreadableError: {exc}", evidence)
    except Exception as exc:
        # See `_clause_zero_human_mutating_calls`: an unavailable SDK or an
        # unresolvable credential chain must be a red clause, never an
        # exception out of `evaluate`.
        return _unmeasurable(
            name,
            requirement,
            f"Cost Explorer could not be reached: {type(exc).__name__}: {exc}. That is a "
            "statement about our access, not about what was spent",
            evidence,
        )
    if reading.amount_usd == 0.0:
        # The empty-set trap, one level down from phase 5's, and it applies to
        # BOTH scopes. A TAG-FILTERED total of exactly $0.00 is a property of
        # the FILTER, not of the spend: it is what a correctly-tagged month
        # with no resources, an entirely UNTAGGED estate, and a correctly
        # tagged estate whose tag key is Inactive in Billing all return, and
        # this account's untagged total was $230.21 on the day this was
        # written. An ACCOUNT total of exactly $0.00 is the same shape one
        # level up: for a live AWS estate it means Cost Explorer answered with
        # nothing chargeable — a broken reading, not a free month. Reading
        # either as "under the ceiling" would put a phase row green on the
        # evidence that the cost reading is not working. Whether the RESOURCES
        # carry the tag is `crucible.tags.audit_stack_tags`' question and
        # phase 0's `v2_resources_tagged_and_versioned` deliverable; whether
        # Billing has ACTIVATED the tag key at all is
        # `crucible.tags.cost_allocation_tag_status`' question, below.
        if not tagged:
            return _unmeasurable(
                name,
                requirement,
                f"Cost Explorer returned exactly $0.00 for {reading.scope} over "
                f"{reading.start.isoformat()}..{reading.end.isoformat()}: no spend "
                "recorded for the whole account. A total of zero is what a broken "
                "or misfiltered reading returns as well as a free month",
                evidence,
            )
        # The tag-filtered $0.00 has a second candidate cause one level above
        # the resources: Cost Explorer only indexes spend under a tag KEY that
        # Billing has activated as a cost-allocation tag, independent of
        # whether every resource carries it (`crucible.tags.audit_stack_tags`
        # answers that, separately). `alpha-engine-config-I10076`: the
        # `system` key read `Inactive` while the stack was fully tagged, and
        # this clause pointed the reader at the wrong audit.
        try:
            tag_status = cost_allocation_tag_status(_ce_client())
        except CostAllocationTagUnreadableError as exc:
            return _unmeasurable(
                name,
                requirement,
                f"Cost Explorer returned exactly $0.00 for {reading.scope} over "
                f"{reading.start.isoformat()}..{reading.end.isoformat()}, and whether "
                f"`{TAG_KEY}` is activated as a cost-allocation tag could not be read: "
                f"{exc}",
                (*evidence, "ce:ListCostAllocationTags"),
            )
        if not tag_status.active:
            return _unmeasurable(
                name,
                requirement,
                f"Cost Explorer returned exactly $0.00 for {reading.scope} over "
                f"{reading.start.isoformat()}..{reading.end.isoformat()}: `{TAG_KEY}` is "
                f"{tag_status.status} as a cost-allocation tag in Billing, so Cost "
                "Explorer does not index spend under this filter at all, regardless of "
                "whether the resources carry it — activate it with `aws ce "
                f"update-cost-allocation-tags-status --cost-allocation-tags-status "
                f"TagKey={TAG_KEY},Status=Active`",
                (*evidence, "ce:ListCostAllocationTags"),
            )
        since = tag_status.last_updated_date or "an unknown date"
        age_note = ""
        if tag_status.last_updated_date:
            try:
                activated = dt.date.fromisoformat(tag_status.last_updated_date[:10])
            except ValueError:
                age_note = ""
            else:
                age_note = f", activated {(window[-1] - activated).days} day(s) ago"
        return _unmeasurable(
            name,
            requirement,
            f"Cost Explorer returned exactly $0.00 for {reading.scope} over "
            f"{reading.start.isoformat()}..{reading.end.isoformat()}: `{TAG_KEY}` has been "
            f"Active since {since}{age_note} — Cost Explorer indexes forward from "
            "activation, so spend recorded is genuinely zero or not yet indexed",
            (*evidence, "ce:ListCostAllocationTags"),
        )
    # `end` is exclusive in Cost Explorer's grammar, so the day count is the
    # interval length: a reading taken on the 1st covers one day, not zero.
    days_elapsed = (reading.end - reading.start).days
    days_in_month = monthrange(reading.start.year, reading.start.month)[1]
    month_to_date = (
        f"{closed_note}${reading.amount_usd:.2f} month-to-date for {reading.scope} over "
        f"{days_elapsed} of {days_in_month} days "
        f"({reading.start.isoformat()}..{reading.end.isoformat()})"
    )
    if reading.amount_usd > ceiling_usd:
        return Clause(
            name,
            requirement,
            False,
            f"{month_to_date}: already over the ${ceiling_usd:.2f}/month ceiling",
            evidence,
        )
    try:
        trailing = trailing_daily_usd(
            _ce_client(), today=window[-1], days=COST_TRAILING_DAYS, tagged=tagged
        )
    except CostUnreadableError as exc:
        return _unmeasurable(
            name,
            requirement,
            f"{month_to_date}: under the ceiling so far, but the trailing "
            f"{COST_TRAILING_DAYS}-day total could not be read — CostUnreadableError: {exc}",
            evidence,
        )
    except Exception as exc:
        return _unmeasurable(
            name,
            requirement,
            f"{month_to_date}: under the ceiling so far, but the trailing "
            f"{COST_TRAILING_DAYS}-day read failed: {type(exc).__name__}: {exc}. That is a "
            "statement about our access, not about what was spent",
            evidence,
        )
    if trailing.total_usd == 0.0:
        # The `$0.00` trap again, on the trailing window: thirty free days is
        # what an untagged estate returns for the tagged scope, and what a
        # broken daily read returns for either.
        return _unmeasurable(
            name,
            requirement,
            f"{month_to_date}: the trailing {COST_TRAILING_DAYS} complete days "
            f"({trailing.start.isoformat()}..{trailing.end.isoformat()}) read exactly $0.00, "
            "which is what a misfiltered or broken daily read returns as well as a free month",
            evidence,
        )
    leading = trailing.amounts_usd[-COST_LEADING_DAYS:]
    leading_mean = sum(leading) / len(leading)
    leading_pace = leading_mean * COST_TRAILING_DAYS
    trailing_total = (
        f"trailing {COST_TRAILING_DAYS} complete days "
        f"({trailing.start.isoformat()}..{trailing.end.isoformat()}) ${trailing.total_usd:.2f}, "
        f"ceiling ${ceiling_usd:.2f}; leading {COST_LEADING_DAYS}-day mean "
        f"${leading_mean:.2f}/day × {COST_TRAILING_DAYS} = ${leading_pace:.2f}"
    )
    if trailing.total_usd > ceiling_usd:
        return Clause(
            name, requirement, False, f"{month_to_date}; {trailing_total}: OVER", evidence
        )
    if leading_pace > ceiling_usd:
        return Clause(
            name,
            requirement,
            False,
            f"{month_to_date}; {trailing_total}: OVER (leading pace)",
            evidence,
        )
    return Clause(name, requirement, True, f"{month_to_date}; {trailing_total}: under", evidence)


def _phase2(
    store: Store,
    window: list[dt.date],
    registry: dict[str, Component],
    *,
    trading_day: dt.date,
) -> list[Clause]:
    """Phase 2's exit gate (plan §6 row 2, §6.1's ruled minimum)."""
    _unused((trading_day,))
    return [
        _clause_live_saturdays_first_attempt_ok(store, window),
        _clause_replays_ok(store, window, registry),
        _clause_zero_human_mutating_calls(store, window),
        _clause_pages_within_ceiling(store, window),
        _clause_pages_commissioned(store),
        _clause_two_page_conditions_on_real_channel(store),
        _clause_transient_retry_class_in_runner(store, registry),
        _clause_fault_injection_against_scheduled_path(store),
        _clause_runbook_in_readme(),
        _clause_aws_cost_within_ceiling(
            window, name="aws_cost_within_ceiling", ceiling_usd=PHASE2_MAX_TAGGED_USD, tagged=True
        ),
    ]


# ---------------------------------------------------------------------------
# phase 3
# ---------------------------------------------------------------------------


def _promote_non_promotion(
    store: Store, slot: str, window: list[dt.date]
) -> tuple[bool, list[str], list[str], list[str], list[str]]:
    """Whether a verdict-backed NON-promotion was filed for ``slot``.

    Returns (found, evidence, problems, access_problems, silent). A promote
    run files one manifest per trading day under `runs/promote/{day}/run.json`
    with a `pointer_moved` metric per slot; the metric's `source_path` is that
    slot's arena cycle key, which is how one manifest answers for four slots
    without the key shape having to carry the slot.

    ``silent`` names every manifest that was FILED and read `ok` but carried
    no `pointer_moved` metric for this slot. "Nothing looked" and "promote ran
    and said nothing about this slot" are different facts with different
    owners (principle 1), and folding the second into the first makes the
    clause state something false about the store: the manifest is right there.
    """
    evidence: list[str] = []
    problems: list[str] = []
    access: list[str] = []
    silent: list[str] = []
    found = False
    for day in window:
        key = manifest_key("promote", day.isoformat())
        evidence.append(key)
        read = _read_store_document(store, key)
        if read.problem is not None:
            (access if read.access_problem else problems).append(read.problem)
            continue
        if read.absent:
            continue
        document = read.document or {}
        status, problem = _status(key, document)
        if problem is not None:
            problems.append(problem)
            continue
        if status != "ok":
            problems.append(f"{key}: promote run status {status!r}")
            continue
        problem = _field(key, document, "metrics", list)
        if problem is not None:
            problems.append(problem)
            continue
        wanted = arena_cycle_key(slot, day.isoformat())
        named = False
        for metric in document["metrics"]:
            if not isinstance(metric, dict) or metric.get("name") != "pointer_moved":
                continue
            if metric.get("source_path") != wanted:
                continue
            named = True
            if metric.get("value"):
                # The pointer MOVED. That is a promotion, graded by the
                # champion branch above, not a non-promotion.
                continue
            if str(metric.get("status_reason") or "").strip():
                found = True
            else:
                problems.append(
                    f"{key}: the `pointer_moved` metric for slot {slot!r} carries no "
                    "`status_reason` — a non-promotion with no stated reason is not a "
                    "verdict-backed one"
                )
        if not named:
            silent.append(
                f"{key}: read `ok` but carries no `pointer_moved` metric whose "
                f"`source_path` is {wanted}"
            )
    return found, evidence, problems, access, silent


def _clause_slot_promotion_or_non_promotion(
    store: Store, slot: str, window: list[dt.date]
) -> Clause:
    """Plan §6 row 3: one evidence-won promotion, or a verdict-backed
    non-promotion, in ``slot``.

    Two artifacts answer it and NEITHER existing is UNMEASURABLE, not UNMET:
    a slot with no champion pointer and no promote run has not held its
    pointer on the evidence — nothing has looked.
    """
    name = f"{slot}_promotion_or_verdict_backed_non_promotion"
    requirement = (
        f"slot {slot!r} either carries a champion promoted on EVIDENCE whose producing "
        "run manifest reads `ok`, or filed a promote run in the window whose "
        "`pointer_moved` metric records a non-promotion with a stated reason"
    )
    champion = champion_key(slot)
    read = _read_store_document(store, champion)
    problems: list[str] = []
    access: list[str] = []
    evidence: list[str] = [champion]
    if read.problem is not None:
        (access if read.access_problem else problems).append(read.problem)
    elif not read.absent:
        pointer = read.document or {}
        source_problem = _field(champion, pointer, "promotion_source", str)
        manifest_problem = _field(champion, pointer, "manifest_key", str)
        if source_problem is not None or manifest_problem is not None:
            problems.extend(p for p in (source_problem, manifest_problem) if p is not None)
        elif pointer["promotion_source"] != "evidence":
            problems.append(
                f"{champion}: `promotion_source` is "
                f"{pointer['promotion_source']!r}, not `evidence` — a bootstrap or an "
                "operator revert is not a promotion the system won"
            )
        else:
            producing = pointer["manifest_key"]
            evidence.append(producing)
            producing_read = _read_store_document(store, producing)
            if producing_read.problem is not None:
                (access if producing_read.access_problem else problems).append(
                    producing_read.problem
                )
            elif producing_read.absent:
                problems.append(
                    f"{producing} is absent — the champion names a producing run whose "
                    "manifest does not exist, so the promotion cannot be verified"
                )
            else:
                status, problem = _status(producing, producing_read.document or {})
                if problem is not None:
                    problems.append(problem)
                elif status != "ok":
                    problems.append(f"{producing}: producing run status {status!r}")
                else:
                    return Clause(
                        name,
                        requirement,
                        True,
                        f"{champion} names {pointer.get('arm_id')!r}, promoted on evidence "
                        f"by {producing} (`ok`)",
                        tuple(evidence),
                    )
    (
        found,
        promote_evidence,
        promote_problems,
        promote_access,
        promote_silent,
    ) = _promote_non_promotion(store, slot, window)
    evidence.extend(promote_evidence)
    problems.extend(promote_problems)
    access.extend(promote_access)
    if found:
        return Clause(
            name,
            requirement,
            True,
            f"no promotion, and a verdict-backed non-promotion was filed for slot {slot!r} "
            f"over {window[0].isoformat()}..{window[-1].isoformat()}",
            tuple(evidence),
        )
    if access and not problems:
        return _unmeasurable(name, requirement, "; ".join(access), evidence)
    if problems:
        detail = "; ".join(problems[:4])
        if access:
            detail = f"{detail}; {len(access)} could not be read: {'; '.join(access[:2])}"
        return Clause(name, requirement, False, detail, tuple(evidence))
    if promote_silent:
        # The manifest is IN the store and read `ok`; it simply said nothing
        # about this slot. Reporting that as "no promote run manifest was
        # filed" states something false about the store (F2, PR68 review).
        # It is a producer gap in `crucible.promote`, not an absence, and the
        # clause names the key and the missing field so the next action is in
        # the output.
        return _unmeasurable(
            name,
            requirement,
            f"{champion} is absent, and {len(promote_silent)} promote run manifest(s) in "
            f"{window[0].isoformat()}..{window[-1].isoformat()} were filed and read `ok` "
            f"but recorded no verdict for slot {slot!r}: {'; '.join(promote_silent[:4])}. "
            "`promote` ran and said nothing about this slot — a producer gap in "
            "`crucible.promote`, not a slot that was never looked at",
            evidence,
        )
    return _unmeasurable(
        name,
        requirement,
        f"neither artifact exists: {champion} is absent and no promote run manifest was "
        f"filed over {window[0].isoformat()}..{window[-1].isoformat()}. A slot that has "
        "never run `promote` has not held its pointer on the evidence — nothing looked",
        evidence,
    )


def _phase3(
    store: Store,
    window: list[dt.date],
    registry: dict[str, Component],
    *,
    trading_day: dt.date,
) -> list[Clause]:
    """Phase 3's exit gate (plan §6 row 3), one clause per slot in `SLOTS`."""
    _unused((registry, trading_day))
    return [_clause_slot_promotion_or_non_promotion(store, slot, window) for slot in sorted(SLOTS)]


# ---------------------------------------------------------------------------
# phase 4
# ---------------------------------------------------------------------------


def _clause_trader_week_on_v2_champion(store: Store, window: list[dt.date]) -> Clause:
    """Plan §6 row 4: the trader ran one week on the v2 champion.

    **The harness may not reach into the trader** (plan §3: separate systems,
    coupled by one contract). So this clause reads the CONSUMER EVIDENCE
    artifact the trader contract declares — and the contract declares none
    today, which is `alpha-engine-config-I9760`'s own scope, so the clause
    reads UNMEASURABLE naming the missing declaration rather than inventing a
    key the trader has never agreed to write.
    """
    name = "trader_one_week_on_v2_champion"
    requirement = (
        "the trader ran one week on the v2 champion, read from the consumer-evidence "
        "artifact the trader contract declares"
    )
    if TRADER_EVIDENCE_KEY is None:
        return _unmeasurable(
            name,
            requirement,
            "the trader contract declares no consumer-evidence artifact, so there is "
            "nothing in this store to read. The harness may not reach into the trader "
            "(plan §3), and inventing a key it has never agreed to write would grade a "
            f"contract that does not exist. Declaring one is `{phase_tracker('phase4')}`'s "
            "own scope",
        )
    key = TRADER_EVIDENCE_KEY
    read = _read_store_document(store, key)
    if read.problem is not None:
        if read.access_problem:
            return _unmeasurable(name, requirement, read.problem, (key,))
        return Clause(name, requirement, False, read.problem, (key,))
    if read.absent:
        return Clause(
            name,
            requirement,
            False,
            f"{key} is absent — the trader has filed no evidence of a week on the v2 champion",
            (key,),
        )
    document = read.document or {}
    problem = _field(key, document, "trading_days", int)
    if problem is not None:
        return Clause(name, requirement, False, problem, (key,))
    days = document["trading_days"]
    if days < TRADING_DAYS_PER_WEEK:
        return Clause(
            name,
            requirement,
            False,
            f"{key}: {days} trading day(s) on the v2 champion, {TRADING_DAYS_PER_WEEK} required",
            (key,),
        )
    return Clause(
        name,
        requirement,
        True,
        f"{key}: {days} trading day(s) on the v2 champion",
        (key,),
    )


def _phase4(
    store: Store,
    window: list[dt.date],
    registry: dict[str, Component],
    *,
    trading_day: dt.date,
) -> list[Clause]:
    """Phase 4's exit gate (plan §6 row 4)."""
    _unused((registry, trading_day))
    return [
        _clause_trader_week_on_v2_champion(store, window),
        _clause_aws_cost_within_ceiling(
            window, name="aws_total_within_ceiling", ceiling_usd=PHASE4_MAX_TOTAL_USD, tagged=False
        ),
        # The phase-0 reader at phase 4's ceiling: decommissioned means ZERO
        # starts, not "within cadence". `alpha-engine-config-I9860` adds the
        # preopen and postclose counts; until it files them this clause grades
        # the weekly pipeline only, and the two absent pipelines are named in
        # `alpha-engine-config-I9758`'s coverage rather than silently omitted.
        _clause_old_weekly_within_cadence(
            store,
            window,
            name="old_sf_execution_count_zero",
            maximum=0,
            # ZERO is the PASS here, so the floor is zero too. Phase 0 grades
            # a live pipeline and a silent week there is a missing run; phase
            # 4 grades a decommissioned one and a silent week is the whole
            # point. Stated here rather than derived from `maximum == 0`, for
            # the same reason as `skips_count_as_runs` below.
            minimum=0,
            # Decommissioned means the state machine emits NOTHING. A
            # `WeeklyRunDayGate` Succeed-skip is phase 0's fail-open margin
            # and phase 4's evidence that the trigger is still alive, so this
            # reading keeps counting every start where phase 0 stopped.
            skips_count_as_runs=True,
        ),
    ]


# ---------------------------------------------------------------------------
# phase 5
# ---------------------------------------------------------------------------


def _clause_every_llm_arm_has_a_verdict(store: Store, window: list[dt.date]) -> Clause:
    """Plan §6 row 5: every LLM arm has a verdict inside one weekly cycle.

    **Zero registered LLM arms is UNMEASURABLE, never MET.** A property over
    an empty set is vacuously true, and that exact trap put a v1 row green
    (`gotcha_a_test_that_passes_on_an_empty_list`: a null implementation
    returning `[]` from every query passed 7 of 11 tests).

    Two preconditions, both absent today and each named separately so the
    reason is actionable: no LLM call site is registered at all, and no arm
    recipe carries a field declaring which call site it reaches a model
    through.
    """
    name = "every_llm_arm_has_a_verdict_within_one_cycle"
    requirement = (
        "every ACTIVE registered arm whose recipe declares an LLM call site has a "
        "verdict in the most recent weekly cycle. Zero such arms is UNMEASURABLE, never "
        "met — a property over an empty set is vacuously true"
    )
    from crucible.llm import (  # noqa: PLC0415 - one call site
        CALLSITE_REGISTRY_PATH,
        LLM_CALLSITE_REGISTRY,
    )

    if not LLM_CALLSITE_REGISTRY:
        return _unmeasurable(
            name,
            requirement,
            "no LLM call site is registered (`crucible/llm_callsites.yaml` declares "
            "`callsites: {}`), so no arm can declare one and the LLM-arm set is empty. "
            "An empty set satisfies this property vacuously, which is why it reads "
            "unmeasurable rather than met",
            (str(CALLSITE_REGISTRY_PATH),),
        )
    if LLM_ARM_CALLSITE_FIELD is None:
        return _unmeasurable(
            name,
            requirement,
            f"{len(LLM_CALLSITE_REGISTRY)} LLM call site(s) are registered, but an arm "
            "recipe carries no field naming which one it uses (`ArmSpec` declares "
            "`name`, `slot`, `ranker`, `params`, `registered_at`, and the id-hashed "
            "`spec` carries no call-site reference), so the LLM-arm SET is not "
            f"derivable from the register. Filed as "
            f"`alpha-engine-config-I{LLM_ARM_CALLSITE_GAP_ISSUE}`",
            tuple(arm_register_key(slot) for slot in sorted(SLOTS)),
        )
    # The register carries an arm's `spec_hash`, never its spec
    # (`nousergon_lib.arena.arms.ArmRecord`), so the declaration is read from
    # the recipe files the register was folded from — the strategy tree synced
    # into the store under `strategy_arms_prefix(slot)` — and joined to the
    # ACTIVE set by arm id, which is the hash of that same spec. No producer
    # state is consulted: a recipe and a register row are both durable
    # artifacts anyone can read.
    from crucible.slots.arms import load_arm_specs  # noqa: PLC0415 - one call site

    day = window[-1]
    evidence: list[str] = []
    access: list[str] = []
    missing: list[str] = []
    llm_arms: list[str] = []
    for slot in LLM_ARM_RECIPE_SLOTS:
        key = arm_register_key(slot)
        evidence.append(key)
        active, _, problem, access_problem, register = _register_arms(store, slot)
        if problem is not None:
            (access if access_problem else missing).append(problem)
            continue
        # Controls are generated by the harness and never filed in the strategy
        # tree, so they are excluded before the join — through the one helper
        # that matches on the NAME component (I9757 F4), not a local set.
        filed = sorted(a for a in active if not is_control_arm(SLOTS[slot], a, register))
        if not filed:
            continue
        prefix = strategy_arms_prefix(slot)
        evidence.append(prefix)
        try:
            specs = {spec.arm_id: spec for spec in load_arm_specs(slot, store=store)}
        except (FileNotFoundError, ValueError) as exc:
            # A refused or absent recipe tree beside an active register is a
            # reading, not a crash: the arms are registered and cannot be
            # classified, which is UNMET naming why.
            missing.append(f"{prefix}: {exc}")
            continue
        except Exception as exc:  # noqa: BLE001 - classified into the reading below
            # Any other failure to list or read the tree (a permissions
            # denial, a transport error) is UNMEASURABLE naming the class —
            # the same treatment `_read_store_lines` gives the register.
            access.append(f"{prefix}: {type(exc).__name__}: {exc}")
            continue
        for arm_id in filed:
            spec = specs.get(arm_id)
            if spec is None:
                missing.append(
                    f"{arm_id}: active in {key} but no recipe under {prefix} derives that id"
                )
                continue
            if spec.params.get(LLM_ARM_CALLSITE_FIELD) in LLM_CALLSITE_REGISTRY:
                llm_arms.append(arm_id)
    if access and not missing:
        return _unmeasurable(name, requirement, "; ".join(access), evidence)
    if missing:
        return Clause(name, requirement, False, "; ".join(missing[:4]), tuple(evidence))
    if not llm_arms:
        return _unmeasurable(
            name,
            requirement,
            "no ACTIVE registered arm declares an LLM call site, so the set this "
            "property quantifies over is empty and the property holds vacuously. "
            "Phase 5 is the phase that ADDS those arms; until one exists there is "
            "nothing to measure",
            evidence,
        )
    unverdicted: list[str] = []
    for arm_id in llm_arms:
        key = verdict_key(arm_id, day.isoformat())
        evidence.append(key)
        read = _read_store_document(store, key)
        if read.problem is not None:
            (access if read.access_problem else unverdicted).append(read.problem)
            continue
        if read.absent:
            unverdicted.append(f"{arm_id}: no verdict at {key}")
    if access and not unverdicted:
        return _unmeasurable(name, requirement, "; ".join(access), evidence)
    if unverdicted:
        return Clause(
            name,
            requirement,
            False,
            f"{len(unverdicted)} of {len(llm_arms)} LLM arm(s) have no verdict for "
            f"{day.isoformat()}: {'; '.join(unverdicted[:4])}",
            tuple(evidence),
        )
    return Clause(
        name,
        requirement,
        True,
        f"{len(llm_arms)} LLM arm(s), each with a verdict for {day.isoformat()}",
        tuple(evidence),
    )


def _phase5(
    store: Store,
    window: list[dt.date],
    registry: dict[str, Component],
    *,
    trading_day: dt.date,
) -> list[Clause]:
    """Phase 5's exit gate (plan §6 row 5)."""
    _unused((registry, trading_day))
    return [_clause_every_llm_arm_has_a_verdict(store, window)]


#: The gates this command can read, and how wide a window each needs.
#:
#: **Every plan §6 phase is registered, and a phase with nothing to read is
#: UNMEASURABLE by clause, not blank** (`alpha-engine-config-I9913`). An
#: earlier revision withheld phases 2-5 on the argument that a phase-2 list
#: "would be a gate that could go green on replays" — but the result was four
#: ladder rows with no instrument behind them, and blank renders identically
#: to "no data yet", so nobody could tell from the board whether the
#: instrument for phase 3 existed or merely had nothing to read. The
#: green-on-replays risk is closed properly instead: the live clause reads the
#: manifest's own LIVE/REPLAY field rather than inferring it from the date,
#: and `Clause.unmeasurable` makes a clause whose input does not exist read
#: UNMEASURABLE with the reason, never MET.
#:
#: Windows, each from the plan rather than chosen here:
#:
#: * phase 0 — ONE week. `alpha-engine-config-I9756`'s original closes-when
#:   said "<= 1 start per calendar week for two consecutive weeks", and the
#:   second week bought exactly one thing: evidence that a disabled rerun
#:   issuer had not silently returned. Brian ruled on 2026-09-04, "we can't
#:   wait a week on phase 0. it should clear after this week's weekly sf."
#:   That protection is not deleted, it is REPLACED by a continuous one:
#:   `nousergon-data-PR1637` adds a `trigger-undeclared` finding to
#:   `automation_pause.py --check`, which runs DAILY and reports any live
#:   enabled trigger declared in neither manifest block. A standing detector
#:   is strictly stronger than one extra week of watching, so one graded week
#:   plus a daily guard beats two graded weeks and none. What makes the
#:   narrower window safe is `minimum=LEGACY_WEEKLY_MIN_RUNS_PER_WEEK` at the
#:   phase-0 call site: with one week in the window, a Saturday on which the
#:   pipeline never ran at all would otherwise satisfy "at most one" and read
#:   MET (`alpha-engine-config-I9962`).
#: * phase 1 — FIVE replay Saturdays (§6 row 1).
#: * phase 2 — TWO, §6.1's ruled minimum ("2 consecutive first-attempt `ok`
#:   Saturdays, not 4", the other two soak weeks traded for the five replays).
#:   Its `replays_ok` clause re-reads phase 1's predicate over its own
#:   five-week window, so the narrow live window does not narrow the replay
#:   one.
#: * phase 3 — FOUR, `promote_min_weeks` (§5.0): a promotion cannot be won on
#:   fewer paired weeks than the eligibility age requires, so a shorter window
#:   could only ever read UNMET.
#: * phase 4 — TWO, covering §6 row 4's "one week" trader claim plus a second
#:   week of cadence evidence for the SF count. Phase 0's window narrowed to
#:   one on 2026-09-04; phase 4's did NOT, and this is not an oversight. Phase
#:   0 asks whether a live pipeline is QUIET, which a daily trigger-declaration
#:   check now also watches continuously; phase 4 asks whether it is GONE, and
#:   nothing else grades that.
#: * phase 5 — ONE, §6 row 5's "verdict within one weekly cycle".
# Every clause is contained before any gate can reach one — below every
# `_clause_*` definition and above the registry that calls them
# (`alpha-engine-config-I10328`).
_contain_clause_exceptions()

GATES: dict[str, tuple[int, Any]] = {
    "phase0": (1, _phase0),
    "phase1": (5, _phase1),
    "phase2": (PHASE2_WINDOW_WEEKS, _phase2),
    "phase3": (4, _phase3),
    "phase4": (2, _phase4),
    "phase5": (1, _phase5),
}


def _window(trading_day: dt.date, weeks: int) -> list[dt.date]:
    """The ``weeks`` weekly trading days ending at ``trading_day``, oldest first.

    Weekly work binds to a Friday close (§4.12), so the window steps back in
    calendar weeks and every element is the same weekday as the anchor.

    This is the RAW window: its elements share the render day's weekday. A
    clause that reads a per-day key with it is keyed to the day somebody
    rendered the gate, which is the defect `weekly_anchor` documents — use
    `weekly_window` for any gate whose artifacts are filed at a weekly close.
    """
    if weeks < 1:
        raise ValueError("a gate window of fewer than one week measures nothing")
    return [trading_day - dt.timedelta(weeks=n) for n in reversed(range(weeks))]


def weekly_window(trading_day: dt.date, weeks: int) -> list[dt.date]:
    """The ``weeks`` weekly CLOSES a gate rendered on ``trading_day`` reads.

    `_window` stepped back from the render day, so a gate rendered on
    Wednesday 2026-09-02 read five Wednesdays and could not see the five
    replay runs keyed to Friday closes — MET on Fridays, UNMET the other four
    weekdays, forever (`alpha-engine-config-I9904`; the same class phase 0
    fixed in `crucible-PR46`). Every element here is `weekly_anchor` of the
    raw window's element: the Friday close strictly before it, resolved
    through the trading calendar, so a holiday Friday walks back to a real
    session and Monday through Friday of one week resolve to the identical
    key set. ONE producer of the anchor shape, shared with phase 0 and the
    phase-2 live clause — not a second stepping rule.
    """
    anchors = list(dict.fromkeys(weekly_anchor(day) for day in _window(trading_day, weeks)))
    if len(anchors) != weeks:
        # Two raw days a week apart always straddle one Friday close, so this
        # cannot happen through `weekly_anchor` as written — but a window that
        # silently shrank would grade fewer weeks than the gate declares and
        # read MET over the ones it kept. Same refusal as
        # `_clause_old_weekly_within_cadence`.
        raise ValueError(
            f"{weeks} window weeks from {trading_day.isoformat()} collapsed onto "
            f"{len(anchors)} weekly anchor(s) {[d.isoformat() for d in anchors]}; a gate may "
            "not grade fewer weeks than it declares"
        )
    return anchors


def _session_span(window: list[dt.date], render_day: dt.date) -> list[dt.date]:
    """Every trading session from the oldest anchor through ``render_day``.

    Two phase-1 clauses read artifacts that are NOT weekly closes: a review is
    filed on the day it was written and an `explain` run is keyed to the day
    it ran, both on any weekday. Reading them at five Friday keys would make a
    Wednesday review invisible — the I9904 defect reflected. So they read
    every session in the span the weekly window covers, up to and including
    the render day and never past it: a document dated after the render is a
    document that did not exist when the gate was read, and counting it would
    let a review filed tomorrow satisfy today's exit.

    Read amplification, stated: a five-week window is ~26 sessions, so the
    `explain` clause makes ~26 manifest GETs per render instead of five. That
    is the cost of reading the artifact at the key it is actually filed under;
    the review clause lists one prefix and is unaffected.
    """
    span: list[dt.date] = []
    day = window[0]
    while day <= render_day:
        if is_trading_day(day):
            span.append(day)
        day += dt.timedelta(days=1)
    return span


#: Gates whose per-day artifacts are weekly closes, so `evaluate` derives the
#: window through `weekly_window` and `GateResult.window` — the dates the
#: filed `gate.json` names — is the key set actually read.
#:
#: * phase 1 — the weekly arc's stage manifests, arena cycles and the
#:   attribution table are all keyed to the Friday close.
#: * phase 3 — `runs/promote/{day}` is filed by the Saturday promote run at
#:   the Friday close; over the raw window the clause could find it on a
#:   Friday render only, hidden behind "neither artifact exists" (PR80 review,
#:   B2). `champions/{slot}/current.json` is not date-keyed and is unaffected.
#: * phase 0 and phase 2 — NOT here: they collapse the raw window onto anchors
#:   inside their own clauses (`weekly_anchor` per element, and the phase-2
#:   replay clause through `weekly_window`), and their tests pin that shape.
#: * phase 4 — reads phase 0's cadence reader (anchored inside) and the
#:   calendar-month cost reader; nothing per-day-keyed.
#: * phase 5 — `verdict_key(arm, day)` per window day, an arena-cycle artifact
#:   keyed to the Friday close like phase 1's; it joins this set when its
#:   clause is next touched (`alpha-engine-config-I9920` owns that clause).
WEEKLY_ANCHORED_GATES: frozenset[str] = frozenset({"phase1", "phase3"})


def evaluate(
    store: Store,
    *,
    gate: str,
    trading_day: dt.date,
    weeks: int | None = None,
    registry: dict[str, Component] | None = None,
) -> GateResult:
    """Read ``gate``'s clauses out of ``store`` and report them."""
    try:
        default_weeks, clauses_fn = GATES[gate]
    except KeyError as exc:
        raise KeyError(
            f"unknown gate {gate!r}; the registered gates are {sorted(GATES)}. A gate "
            "that is not registered has no clause list, so running it would report a "
            "pass over nothing."
        ) from exc
    weeks_to_read = default_weeks if weeks is None else weeks
    if gate in WEEKLY_ANCHORED_GATES:
        window = weekly_window(trading_day, weeks_to_read)
    else:
        window = _window(trading_day, weeks_to_read)
    result = GateResult(gate=gate, trading_day=trading_day, window=window)
    result.clauses = list(
        clauses_fn(store, window, registry or load_registry(), trading_day=trading_day)
    )
    result.coverage = coverage_note(gate, [c.name for c in result.clauses])
    return result


def _unused(_: Iterable[Any]) -> None:
    return None


# ---------------------------------------------------------------------------
# The phase ladder — one durable row per phase.
#
# Until this existed, "which phase is Crucible v2 on, and is its gate met" was
# published only as prose in GitHub issue comments, written by hand. On
# `alpha-engine-config-I9757` the phase-1 reading moved 18/24 -> 19/23 -> 21/23
# across three comments, one of them explicitly correcting another, and phase 1
# was CLOSED on 2026-09-02 while phase 0's gate had never been read at all.
# Three defects, one cause: the number that says whether a phase is done had no
# surface of its own.
#
# The ladder is that surface. It is a PROJECTION over gate readings already in
# the store plus the registered clause lists — it runs nothing, and it invents
# no state a gate did not measure.
# ---------------------------------------------------------------------------

LADDER_SCHEMA_VERSION = "phase_ladder.v1"

#: Where the ladder is filed. ONE well-known object, rewritten by every reader,
#: because the console renders current state and never owns history
#: (`console-policy` §1). The ladder's history is the dated gate readings at
#: `gate_key`, which are never overwritten.
LADDER_KEY = "gates/ladder.json"

#: The tracker every phase issue lives on. The alpha-engine ecosystem files to
#: `alpha-engine-config` whatever repo the code lands in, so a phase row's
#: identifier is `alpha-engine-config-I<N>` — deliberately the SAME identifier
#: the console's `git-host` adapter mints for an issue, so the two claims merge
#: into one row (`console-policy` §2.5) instead of rendering the phase twice.
TRACKER_REPO = "nousergon/alpha-engine-config"


@dataclass(frozen=True)
class Phase:
    """One rung of the plan §6 ladder, and the gate that lets it exit."""

    id: str
    number: int
    title: str
    issue: int
    #: The registered gate name, or None when no gate has been written for this
    #: phase yet. None is never a pass: it renders `UNMEASURED`.
    gate: str | None

    @property
    def tracker(self) -> str:
        return f"alpha-engine-config-I{self.issue}"

    @property
    def tracker_url(self) -> str:
        return f"https://github.com/{TRACKER_REPO}/issues/{self.issue}"


#: The plan §6 ladder. EVERY phase carries a registered gate
#: (`alpha-engine-config-I9913`), and `tests/test_gate.py` asserts both
#: directions of that — `all(p.gate is not None for p in PHASES)` and
#: `set(GATES) == {p.gate for p in PHASES}` — so a plan phase can never again
#: render blank, and a gate can never be registered that no phase reads.
#:
#: `gate: None` is still representable, because a SIXTH phase added to the
#: plan before its clause list is written must render `UNMEASURED` rather than
#: fail an import; the structural test is what makes leaving it that way a red
#: CI run instead of a silent hole on the board.
PHASES: tuple[Phase, ...] = (
    Phase("phase0", 0, "Stop the bleeding", 9756, "phase0"),
    Phase("phase1", 1, "One command, locally", 9757, "phase1"),
    Phase("phase2", 2, "Unattended", 9758, "phase2"),
    Phase("phase3", 3, "All three slots", 9759, "phase3"),
    Phase("phase4", 4, "Trader on the contract; decommission", 9760, "phase4"),
    Phase("phase5", 5, "Grow", 9761, "phase5"),
)


def phase_tracker(phase_id: str) -> str:
    """The tracker identifier owning ``phase_id``, DERIVED from :data:`PHASES`.

    The one legitimate way for a clause reading to name a phase's issue.
    `tests/test_no_stale_tracker_literals.py` (`alpha-engine-config-I9839`)
    forbids the literal shape in any string a reader can reach, because a
    restated issue number stays pointed at a phase after that phase closes —
    which is exactly what happened to `I9757`, cited by phase-2 and phase-3
    acceptance clauses that had nothing to do with it.

    Defined below :data:`PHASES` and called at read time, never at import
    time, so a clause defined earlier in the module can still use it.
    """
    for phase in PHASES:
        if phase.id == phase_id:
            return phase.tracker
    raise KeyError(
        f"no registered phase {phase_id!r}; the registered phases are "
        f"{[p.id for p in PHASES]}. A tracker derived from an unregistered phase would "
        "be an invented issue number, which is the defect this function exists to remove."
    )


#: Environment variables a registered gate's OWN clause list reads, keyed by
#: gate name (`alpha-engine-config-I10492`). Distinct in kind from
#: `_contained`'s catch-all: that guard exists so a clause's UNPREDICTABLE
#: read failure — a denied AWS call, a missing region — cannot darken the
#: whole ladder, and it must stay that broad. THIS table is the narrow,
#: PREDICTABLE subset of that surface — "the operator never told this
#: process where to look" — and `crucible gate` / `crucible gate.close`
#: check it BEFORE reading anything, so that class refuses loudly instead of
#: folding into the same UNMEASURABLE string a real read failure produces.
#: "The world could not be read" and "this process was not told where to
#: look" must never be the same string on the artifact.
#:
#: `old_alerts_muted` (phase 0) calls `crucible.alerts.muted_topic`, which
#: raises via `crucible.required.require_env` when `MUTED_TOPIC_VAR` is
#: unset. `zero_human_mutating_calls` (phase 2) reads
#: `crucible.config.settings().cloudtrail_archive`, which does NOT raise —
#: an unset `CLOUDTRAIL_ARCHIVE_VAR` resolves to the deliberately empty
#: `DEFAULT_CLOUDTRAIL_ARCHIVE` and the clause reads a clean UNMEASURABLE —
#: so this table checks the raw environment directly rather than relying on
#: either clause's internal failure shape, and catches both cases the same
#: way.
#:
#: Kept honest by `tests/test_gate.py::test_gate_required_env_matches_clauses`:
#: for each entry, evaluating that gate with the named variable unset (and
#: every other required variable set) must produce an UNMEASURABLE reading on
#: exactly the clause this table exists for — so a clause gaining or losing a
#: required variable is a red CI run, not a silent hole in this table.
GATE_REQUIRED_ENV: dict[str, tuple[str, ...]] = {
    "phase0": (MUTED_TOPIC_VAR,),
    "phase2": (CLOUDTRAIL_ARCHIVE_VAR,),
}


def required_env_for_run() -> tuple[str, ...]:
    """Every env var ANY registered gate's live evaluation needs, sorted.

    Not just the selected gate's own entry in :data:`GATE_REQUIRED_ENV`:
    `build_ladder` live-evaluates EVERY registered phase on every call — the
    ladder answers "which phase is the rebuild on", which needs every rung,
    not only the one `--gate` named — so a `crucible gate --gate phase2`
    invocation still needs `phase0`'s required variables too. This is the
    union over the whole table, computed once here rather than at each call
    site, so a call site cannot narrow it by accident.
    """
    seen: dict[str, None] = {}
    for names in GATE_REQUIRED_ENV.values():
        for name in names:
            seen[name] = None
    return tuple(seen)


def missing_required_env() -> tuple[str, ...]:
    """Which of :func:`required_env_for_run`'s variables are unset right now.

    A direct `os.environ` check, not a call through `crucible.config.settings`
    or `crucible.alerts.muted_topic`: those two resolvers disagree about
    whether an unset variable raises (see :data:`GATE_REQUIRED_ENV`), and this
    function exists precisely so `crucible gate` does not have to care which
    shape either one takes.
    """
    return tuple(name for name in required_env_for_run() if not os.environ.get(name, "").strip())


#: The ladder's closed state vocabulary. Total, with no fall-through, and no
#: fifth member: `UNKNOWN`/`PENDING`/`N/A` are the shapes this exists to
#: remove (`observability-policy` §8.3, same posture as
#: `crucible.console.classify`).
#:
#: * `MET`           — the gate was read and every clause is met.
#: * `UNMET`         — the gate was read and at least one clause is not met.
#:                     A real result, not a failure.
#: * `UNMEASURED`    — no clause list is registered for this phase, or no
#:                     reading has ever been filed. Never green, never zero.
#: * `UNMEASURABLE`  — a read was attempted and could not complete: a store
#:                     access failure on one clause, or on the listing that
#:                     answers "when was this last read". Reusing
#:                     `crucible.board`'s vocabulary, not restating it —
#:                     `board.BOARD_STATES` already carries this exact word
#:                     and `LADDER_BOARD_STATE` maps it straight through
#:                     (`alpha-engine-config-I9869` round 3, finding 4). Red,
#:                     never folded into `UNMET`: "it says no" and "I could
#:                     not ask" are different facts, and the second is about
#:                     us.
#: * `OUT_OF_ORDER`  — a later phase is being graded while an earlier phase's
#:                     gate is not met. Brian's ruling of 2026-09-02: phase 0's
#:                     gate must READ before phase 2 opens.
LADDER_STATES: tuple[str, ...] = ("MET", "UNMET", "UNMEASURED", "UNMEASURABLE", "OUT_OF_ORDER")

#: How a ladder state renders on the fleet console, in `observability-policy`
#: §8.3's vocabulary. `UNMEASURED` maps to `UNREPORTED` and therefore counts
#: against the transparency gap whose objective is zero — a phase nobody can
#: read is unobserved, not healthy. `UNMEASURABLE` and `OUT_OF_ORDER` both map
#: to `FAILED`: one is an access fault, the other an invariant breach, but
#: neither is a slow phase or a plain shortfall.
LADDER_CONSOLE_STATE: dict[str, str] = {
    "MET": "HEALTHY",
    "UNMET": "DEGRADED",
    "UNMEASURED": "UNREPORTED",
    "UNMEASURABLE": "FAILED",
    "OUT_OF_ORDER": "FAILED",
}


def _check_ladder_console_coverage(states: Iterable[str], console_map: dict[str, str]) -> None:
    """Refuse a ladder state with no declared console rendering.

    A plain function, not a module-level `assert`: `assert` is compiled out
    under `python -O`/`PYTHONOPTIMIZE` — the one guard construct guaranteed
    absent in an optimized interpreter, and this is the guard that stops a
    ladder state reaching a console surface with no declared rendering
    (`alpha-engine-config-I9826`). Called once at import time below, so the
    failure is still caught at import, not deferred to the first render.
    """
    gap = set(states) - set(console_map)
    if gap:
        raise ValueError(
            f"LADDER_CONSOLE_STATE is missing {sorted(gap)} — every ladder "
            "state must declare how it renders before it can reach a surface."
        )


_check_ladder_console_coverage(LADDER_STATES, LADDER_CONSOLE_STATE)


def last_read(store: Store, gate: str) -> tuple[str | None, bool]:
    """The most recent trading day a reading of ``gate`` was filed for, and
    whether the listing itself could not be read.

    ``(None, False)`` when the gate has never been read. That is the field
    the ladder row publishes as "when was this last measured", and a row
    that cannot answer it says so rather than borrowing the ladder's own
    generation time — a freshly written ladder full of never-measured phases
    would otherwise look entirely fresh.

    ``(None, True)`` is a THIRD, different answer: the listing itself could
    not be read (a store access failure), not "never measured". A bare
    `store.list_keys` here used to raise straight out of `build_ladder`, the
    one guarded-single-key-read discipline round 2 established applied to
    every `_read_store_document`/`_read_store_lines` call but not to this
    listing (`alpha-engine-config-I9869` round 3, finding 2).
    """
    listed = _list_store_keys(store, gate_prefix(gate))
    if listed.problem is not None:
        return None, listed.access_problem
    days: list[str] = []
    for key in listed.keys or []:
        parts = key.split("/")
        if key.endswith("/gate.json") and len(parts) == 4:
            days.append(parts[2])
    return (max(days) if days else None), False


@dataclass(frozen=True)
class PhaseRow:
    """One phase, as the console reads it."""

    phase: Phase
    state: str
    gate_state: str
    detail: str
    clauses_met: int | None
    clauses_total: int | None
    #: How many of this phase's clauses read UNMEASURABLE — a store access
    #: failure, distinct from a clause that was read and found unmet. `None`
    #: exactly where `clauses_total` is `None` (no clause list registered at
    #: all); otherwise always a count, `0` included, so its absence on the
    #: wire is never ambiguous with "not counted"
    #: (`alpha-engine-config-I9869` round 3, finding 4).
    clauses_unmeasurable: int | None
    met_ratio: float | None
    read_on: str | None
    blocked_by: str | None

    def to_dict(self, generated_utc: str) -> dict[str, Any]:
        return {
            # The tracker ref IS the identifier (`console-policy` §2.1), so a
            # `git-host` claim about the same issue merges onto this row.
            "decision_id": self.phase.tracker,
            "phase": self.phase.id,
            "number": self.phase.number,
            "title": self.phase.title,
            "tracker": self.phase.tracker,
            "tracker_url": self.phase.tracker_url,
            "gate": self.phase.gate,
            "state": self.state,
            "console_state": LADDER_CONSOLE_STATE[self.state],
            "gate_state": self.gate_state,
            "detail": self.detail,
            "clauses_met": self.clauses_met,
            "clauses_total": self.clauses_total,
            "clauses_unmeasurable": self.clauses_unmeasurable,
            # `null`, never 0.0, when nothing was measured. Zero is a
            # measurement; absence is not, and rendering one as the other is
            # the whole defect (principle 7).
            "met_ratio": self.met_ratio,
            "read_on": self.read_on,
            "blocked_by": self.blocked_by,
            "generated_utc": generated_utc,
        }


@dataclass
class Ladder:
    """Every phase, in order, with the ladder-level counts."""

    trading_day: dt.date
    generated_utc: str
    rows: list[PhaseRow] = field(default_factory=list)

    @property
    def current_phase(self) -> str:
        """The lowest phase whose gate is not met — where the ladder actually is.

        Not "the highest phase with work in it". A phase whose gate has never
        been read is not behind us, and this is the field that says so.
        """
        for row in self.rows:
            if row.gate_state != "MET":
                return row.phase.id
        return "complete"

    @property
    def out_of_order(self) -> list[str]:
        return [r.phase.id for r in self.rows if r.state == "OUT_OF_ORDER"]

    @property
    def unmeasured(self) -> int:
        return sum(1 for r in self.rows if r.gate_state == "UNMEASURED")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": LADDER_SCHEMA_VERSION,
            "trading_day": self.trading_day.isoformat(),
            "generated_utc": self.generated_utc,
            "current_phase": self.current_phase,
            "phases_total": len(self.rows),
            "phases_met": sum(1 for r in self.rows if r.gate_state == "MET"),
            "unmeasured": self.unmeasured,
            "out_of_order": self.out_of_order,
            "phases": [r.to_dict(self.generated_utc) for r in self.rows],
        }

    def render(self) -> str:
        lines = [
            f"phase ladder @ {self.trading_day.isoformat()}: at {self.current_phase}, "
            f"{sum(1 for r in self.rows if r.gate_state == 'MET')}/{len(self.rows)} met, "
            f"{self.unmeasured} unmeasured"
        ]
        for row in self.rows:
            read = row.read_on or "never"
            lines.append(
                f"  [{row.phase.number}] {row.phase.id} {row.state}: {row.detail} "
                f"(tracker {row.phase.tracker}, last read {read})"
            )
        return "\n".join(lines)


def build_ladder(
    store: Store,
    *,
    trading_day: dt.date,
    registry: dict[str, Component] | None = None,
    now: dt.datetime | None = None,
    readings: dict[str, GateResult] | None = None,
) -> Ladder:
    """Read every phase's gate out of ``store`` and assemble the ladder.

    Reads only. Each registered gate is evaluated against the artifacts already
    filed — the same measurement `crucible gate` publishes — so the ladder can
    never disagree with the per-gate reading beside it.

    ``readings`` lets a caller that already evaluated a gate this run (`crucible
    gate`'s own handler) hand that reading in rather than have it re-evaluated
    here — `gate_handler` used to read the same gate twice in one job, doubling
    every store read the clause set makes (`alpha-engine-config-I9826`). Any
    gate not present in ``readings`` is still evaluated normally.
    """
    moment = (now or dt.datetime.now(dt.UTC)).astimezone(dt.UTC)
    reg = registry if registry is not None else load_registry()
    supplied = readings or {}

    gate_states: list[
        tuple[Phase, str, str, int | None, int | None, int | None, float | None, str | None]
    ] = []
    for phase in PHASES:
        if phase.gate is None or phase.gate not in GATES:
            gate_states.append(
                (
                    phase,
                    "UNMEASURED",
                    f"no clause list is registered for {phase.id}; nothing has ever "
                    "measured whether it may exit",
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            )
            continue
        reading = supplied.get(phase.gate) or evaluate(
            store, gate=phase.gate, trading_day=trading_day, registry=reg
        )
        met = sum(1 for c in reading.clauses if c.met)
        total = len(reading.clauses)
        unmeasurable_count = sum(1 for c in reading.clauses if c.unmeasurable)
        read_on, read_on_access_problem = last_read(store, phase.gate)
        if total == 0:
            # A gate with no clauses measured nothing. `met` would be
            # vacuously true, and `reading.met_ratio` is already `None` here
            # (GateResult.met_ratio, not re-derived) — both are the shape
            # that lets a phase close unmeasured, so the ladder refuses to
            # call it a reading. Passing `reading.met_ratio` through, rather
            # than constructing a second `None` here, is what makes the
            # ladder row and the dated gate artifact agree by construction:
            # there is exactly one place this ratio is computed.
            gate_states.append(
                (
                    phase,
                    "UNMEASURED",
                    f"gate {phase.gate} has no clauses",
                    0,
                    0,
                    0,
                    reading.met_ratio,
                    read_on,
                )
            )
            continue
        unmet = [c.name for c in reading.clauses if not c.met and not c.unmeasurable]
        unmeasurable_names = [c.name for c in reading.clauses if c.unmeasurable]
        detail = f"{met}/{total} clauses met"
        if unmeasurable_names:
            detail = f"{detail}; unmeasurable: {', '.join(unmeasurable_names)}"
        if unmet:
            detail = f"{detail}; holding: {', '.join(unmet)}"
        # The coverage line travels with the row. Without it a phase whose
        # gate grades a SUBSET of its deliverables renders MET on the ladder
        # and on the board with nothing saying so — partial coverage reported
        # as complete, which is the pattern `alpha-engine-config-I9837`
        # exists to make visible rather than one this row gets to repeat.
        if reading.coverage:
            detail = f"{detail}; {reading.coverage}"
        if read_on_access_problem:
            detail = (
                f"{detail}; could not determine when gate {phase.gate} was last read "
                "(store access failure)"
            )
        # `UNMEASURABLE` outranks `MET`/`UNMET`: a phase with one unreadable
        # clause, or whose own read-history listing failed, was rendering as
        # plain `UNMET` with a specific `met_ratio` — indistinguishable from
        # "we checked and it fell short" (`alpha-engine-config-I9869` round
        # 3, finding 4).
        #
        # The three-way choice itself is `gate_state_for`, not re-derived here:
        # the closing block a phase issue is closed with renders its
        # `gate_state` from the same function, so a ladder row and the block on
        # the tracker beside it cannot disagree about one reading
        # (`alpha-engine-config-I9967`).
        state = "UNMEASURABLE" if read_on_access_problem else gate_state_for(reading)
        gate_states.append(
            (
                phase,
                state,
                detail,
                met,
                total,
                unmeasurable_count,
                reading.met_ratio,
                read_on,
            )
        )

    # Out-of-order: a phase later than the lowest not-met phase that has
    # nevertheless been graded. Derived from readings alone, so nothing here
    # depends on a hand-maintained claim about which phase is "open".
    first_unmet = next((i for i, s in enumerate(gate_states) if s[1] != "MET"), len(gate_states))
    rows: list[PhaseRow] = []
    for index, (
        phase,
        gate_state,
        detail,
        met,
        total,
        unmeasurable_count,
        ratio,
        read_on,
    ) in enumerate(gate_states):
        state = gate_state
        blocked_by = None
        if index > first_unmet and read_on is not None:
            blocker = gate_states[first_unmet][0]
            state = "OUT_OF_ORDER"
            blocked_by = blocker.id
            detail = (
                f"{detail}; graded while {blocker.id} ({blocker.tracker}) is "
                f"{gate_states[first_unmet][1]} — a later phase may not be exited "
                "ahead of an earlier one"
            )
        rows.append(
            PhaseRow(
                phase=phase,
                state=state,
                gate_state=gate_state,
                detail=detail,
                clauses_met=met,
                clauses_total=total,
                clauses_unmeasurable=unmeasurable_count,
                met_ratio=ratio,
                read_on=read_on,
                blocked_by=blocked_by,
            )
        )

    return Ladder(
        trading_day=trading_day,
        generated_utc=moment.strftime("%Y-%m-%dT%H:%M:%SZ"),
        rows=rows,
    )


LADDER_SCHEMA_PATH = Path(__file__).parent / "schemas" / "phase_ladder.v1.json"


@lru_cache(maxsize=1)
def ladder_schema() -> dict[str, Any]:
    """The `phase_ladder.v1` JSON Schema, loaded once (`alpha-engine-config-I9825`).

    A missing schema is a broken build, not a degraded read — same posture as
    `crucible.champion.load_schema`.
    """
    if not LADDER_SCHEMA_PATH.is_file():
        raise FileNotFoundError(
            f"phase ladder schema missing at {LADDER_SCHEMA_PATH}. It ships inside "
            "the package; a missing schema means a broken build."
        )
    return json.loads(LADDER_SCHEMA_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _ladder_validator() -> Draft202012Validator:
    schema = ladder_schema()
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def validate_ladder_document(document: dict[str, Any]) -> None:
    """Refuse a ladder document that does not conform to `phase_ladder.v1`.

    Producer-side validation, the shape `crucible/slots/inputs.py::write_arm_
    predictions` uses — a malformed ladder is refused before it reaches the
    store, not discovered by whatever reads it next.

    `alpha-engine-config-I10045` row 8: validated through
    `crucible.models.PhaseLadderDocument` instead of the hand-rolled
    `_ladder_validator()`; `phase_ladder.v1.json` is now GENERATED from that
    model. `ladder_schema()`/`_ladder_validator()` stay unchanged (both are
    tested directly in `tests/test_phase_ladder.py`) and still read whichever
    file is committed.
    """
    try:
        PhaseLadderDocument.model_validate(document)
    except ValidationError as exc:
        detail = "\n".join(
            f"  - {'/'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}"
            for e in exc.errors()
        )
        raise ValueError(
            f"ladder document does not conform to {LADDER_SCHEMA_VERSION}:\n{detail}"
        ) from exc


def ladder_payload(ladder: Ladder) -> bytes:
    """The ladder artifact's bytes, as every publisher writes them.

    Validated against `phase_ladder.v1` before being returned — both
    publishers (`crucible gate` and `crucible console`) call this rather than
    re-serializing `ladder.to_dict()` by hand, so there is exactly one place
    the bytes on the wire are produced and checked (`alpha-engine-config-I9825`).
    """
    document = ladder.to_dict()
    validate_ladder_document(document)
    return json.dumps(document, indent=2, sort_keys=True).encode("utf-8")


# ---------------------------------------------------------------------------
# The closing reading — `alpha-engine-config-I9967`, deliverables 2 and 3.
#
# The ladder above answers "is phase N met" on a surface nobody has to close.
# The BACKLOG answers the same question by a human closing an issue, and on
# 2026-09-02 the two disagreed: `alpha-engine-config-I9757` was closed the
# moment its build PRs merged while its own gate read 1 of 6 clauses met and
# the phase beneath it was unmet. Two instruments, one question, and the one a
# human reads on a board was the wrong one.
#
# A convention ("paste the reading before you close it") has now failed twice.
# What follows is the machine-checkable form of that convention: one canonical
# block, rendered from a real reading by `crucible gate --closing-comment`,
# and one refusal predicate — `closing_reading_refusals` — that says why a
# CLOSED phase issue is not justified by the block on it.
#
# **Crucible renders and defines; it does not enforce.** Enforcement lives in
# `alpha-engine-config`, because that is the only side that can read BOTH
# facts: this package has an AWS identity and no GitHub credential for the
# private tracker, and the tracker's own CI has a GitHub credential and no
# read on this store. Splitting it any other way needs a new identity in one
# repo or the other; splitting it this way needs none.
# ---------------------------------------------------------------------------

CLOSING_READING_SCHEMA_VERSION = "phase_closing_reading.v1"

#: The fenced-code info string that marks a closing reading inside a GitHub
#: comment. A fence info string rather than an HTML comment: GitHub renders it
#: as a plain code block (so a human sees the reading), it survives quoting
#: and editing, and it is one token a `re` can find without parsing Markdown.
CLOSING_READING_FENCE = "crucible-gate-reading"

_CLOSING_READING_BLOCK = re.compile(
    r"```" + CLOSING_READING_FENCE + r"\s*\n(?P<body>.*?)\n?```",
    re.DOTALL,
)

#: Minimum length of the `commit` field, in lowercase hex characters. Twelve,
#: matching `crucible.release`'s `+g<sha12>` local version segment, so a block
#: rendered on a box that only knows its wheel's version can still carry real
#: provenance.
CLOSING_READING_COMMIT_MIN = 12

_COMMIT_RE = re.compile(r"^[0-9a-f]{12,40}$")


def gate_state_for(reading: GateResult) -> str:
    """`MET`, `UNMET` or `UNMEASURABLE` for one gate reading.

    The single derivation of a gate's own state from its clauses. `UNMEASURABLE`
    outranks both others: a reading with one unreadable clause is not "we
    checked and it fell short" (`alpha-engine-config-I9869` round 3). A gate
    with NO clauses is `UNMEASURED` on the ladder — that case is the ladder's,
    because it is a fact about registration rather than about a reading, and
    this function is only ever handed a reading that has clauses; it returns
    `UNMET` for an empty clause list, which `GateResult.met` already does and
    which is never a pass.

    `build_ladder` calls this rather than re-deriving the same three-way
    choice inline, so a ladder row and a closing block rendered from the same
    reading cannot disagree about its state.
    """
    if any(c.unmeasurable for c in reading.clauses):
        return "UNMEASURABLE"
    return "MET" if reading.met else "UNMET"


def phase_for_gate(gate: str) -> Phase:
    """The registered phase whose exit this ``gate`` grades.

    Raises rather than returning None: a closing block rendered for a gate no
    phase reads would name a tracker nobody could derive, which is the
    invented-issue-number defect `phase_tracker` exists to remove.
    """
    for phase in PHASES:
        if phase.gate == gate:
            return phase
    raise KeyError(
        f"no registered phase reads gate {gate!r}; the registered gates are "
        f"{sorted(p.gate for p in PHASES if p.gate)}. A closing reading for an "
        "unregistered gate would name no tracker."
    )


def closing_reading(
    reading: GateResult,
    *,
    store_uri: str,
    commit: str,
) -> dict[str, Any]:
    """The `phase_closing_reading.v1` document for one gate reading.

    A TRANSCRIPT, not a claim. Every count comes off ``reading`` — `met_ratio`
    from `GateResult.met_ratio` and the state from :func:`gate_state_for`, both
    of which the ladder also reads — so the block, the ladder row and the
    durable `gates/{gate}/{day}/gate.json` artifact are three renderings of one
    measurement rather than three numbers that happen to agree today.

    It is rendered for an UNMET gate exactly as readily as for a MET one. The
    honest block on a phase that may not close yet is the one that says so; a
    renderer that refused to produce it would leave "no block" meaning both
    "not measured" and "measured and failing".
    """
    phase = phase_for_gate(reading.gate)
    if not store_uri:
        raise ValueError(
            "a closing reading needs the store URI it was read against: without it "
            "`gate_artifact` names a key in no particular bucket and the block cannot "
            "be checked by anyone who doubts it."
        )
    if not _COMMIT_RE.match(commit):
        raise ValueError(
            f"commit {commit!r} is not at least {CLOSING_READING_COMMIT_MIN} lowercase hex "
            "characters. A reading with no commit cannot be re-run against the clause "
            "definitions that produced it, which is the reason it is recorded at all."
        )
    ratio = reading.met_ratio
    document: dict[str, Any] = {
        "schema_version": CLOSING_READING_SCHEMA_VERSION,
        "phase": phase.id,
        "tracker": phase.tracker,
        "tracker_url": phase.tracker_url,
        "gate": reading.gate,
        "gate_state": gate_state_for(reading),
        "clauses_met": sum(1 for c in reading.clauses if c.met),
        "clauses_total": len(reading.clauses),
        "clauses_unmeasurable": sum(1 for c in reading.clauses if c.unmeasurable),
        "met_ratio": None if ratio is None else round(ratio, 6),
        "coverage": reading.coverage,
        "trading_day": reading.trading_day.isoformat(),
        "generated_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "store": store_uri,
        "commit": commit,
        "gate_artifact": gate_key(reading.gate, reading.trading_day.isoformat()),
        "clauses": [
            {
                "name": c.name,
                "met": c.met,
                "unmeasurable": c.unmeasurable,
                "detail": c.detail,
            }
            for c in reading.clauses
        ],
    }
    validate_closing_reading_document(document)
    return document


CLOSING_READING_SCHEMA_PATH = Path(__file__).parent / "schemas" / "phase_closing_reading.v1.json"


@lru_cache(maxsize=1)
def closing_reading_schema() -> dict[str, Any]:
    """The `phase_closing_reading.v1` JSON Schema, loaded once.

    A missing schema is a broken build, not a degraded read — the posture
    :func:`ladder_schema` already takes.
    """
    if not CLOSING_READING_SCHEMA_PATH.is_file():
        raise FileNotFoundError(
            f"phase closing reading schema missing at {CLOSING_READING_SCHEMA_PATH}. It "
            "ships inside the package; a missing schema means a broken build."
        )
    return json.loads(CLOSING_READING_SCHEMA_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _closing_reading_validator() -> Draft202012Validator:
    schema = closing_reading_schema()
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def validate_closing_reading_document(document: dict[str, Any]) -> None:
    """Refuse a closing reading that does not conform to `phase_closing_reading.v1`.

    `alpha-engine-config-I10045` row 8: validated through
    `crucible.models.PhaseClosingReadingDocument` instead of the hand-rolled
    `_closing_reading_validator()`; `phase_closing_reading.v1.json` is now
    GENERATED from that model. `closing_reading_schema()`/
    `_closing_reading_validator()` stay unchanged and still read whichever
    file is committed.
    """
    try:
        PhaseClosingReadingDocument.model_validate(document)
    except ValidationError as exc:
        detail = "\n".join(
            f"  - {'/'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}"
            for e in exc.errors()
        )
        raise ValueError(
            f"closing reading does not conform to {CLOSING_READING_SCHEMA_VERSION}:\n{detail}"
        ) from exc


def render_closing_comment(document: dict[str, Any]) -> str:
    """The comment body a phase issue is closed with.

    A human-readable summary line, then the machine-readable block. Both, not
    one: a reader scrolling the issue sees the verdict without decoding JSON,
    and the sweep reads the block without parsing prose. The summary is
    DERIVED from the same document, so the two halves cannot drift.
    """
    validate_closing_reading_document(document)
    state = document["gate_state"]
    verdict = (
        f"`crucible gate --gate {document['gate']}` reads **{state}** "
        f"({document['clauses_met']}/{document['clauses_total']} clauses met"
    )
    if document["clauses_unmeasurable"]:
        verdict += f", {document['clauses_unmeasurable']} unmeasurable"
    verdict += f") on trading day {document['trading_day']}."
    lines = [
        verdict,
        "",
        f"Read from `{document['store']}` at commit `{document['commit']}`; the durable "
        f"artifact is `{document['gate_artifact']}`.",
    ]
    if document["coverage"]:
        lines += ["", f"Coverage: {document['coverage']}"]
    lines += [
        "",
        f"```{CLOSING_READING_FENCE}",
        json.dumps(document, indent=2, sort_keys=True),
        "```",
    ]
    return "\n".join(lines)


def parse_closing_comment(text: str) -> dict[str, Any] | None:
    """The closing reading inside ``text``, or ``None`` when there is none.

    ``None`` means "this comment carries no block" — a real answer, and the
    one the sweep turns into a refusal. A block that IS present and cannot be
    parsed raises: a malformed reading is not an absent one, and swallowing
    the difference would let a corrupted paste read as "no reading here yet".

    The LAST block wins when a comment carries several. A comment edited to
    correct a reading appends; the correction is the later block.
    """
    matches = list(_CLOSING_READING_BLOCK.finditer(text or ""))
    if not matches:
        return None
    body = matches[-1].group("body")
    try:
        document = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"a `{CLOSING_READING_FENCE}` block is present but is not valid JSON: {exc}. "
            "A malformed reading is not an absent one."
        ) from exc
    if not isinstance(document, dict):
        raise ValueError(
            f"a `{CLOSING_READING_FENCE}` block is present but is not a JSON object "
            f"(got {type(document).__name__})."
        )
    return document


def closing_reading_refusals(document: dict[str, Any] | None, *, phase_id: str) -> list[str]:
    """Why a CLOSED phase issue is NOT justified by ``document``. Empty means it is.

    The whole predicate, in one place, so the sweep that enforces it in
    `alpha-engine-config` mirrors a list rather than inventing one. Every check
    is stated as a refusal with its reason, because the sweep's output is read
    by whoever has to fix the issue.

    Rules 5–8 are the mutation guards: a block that merely SAYS `MET` while its
    own counts, ratio or per-clause verdicts disagree is refused. Forging one
    now takes editing five mutually-consistent fields, which is a different act
    from the carelessness that closed `alpha-engine-config-I9757` — a phase
    closed on merged PRs, with nobody having read anything.
    """
    if document is None:
        return [
            f"no `{CLOSING_READING_FENCE}` block on the closing comment — nothing records "
            f"what {phase_id}'s gate read when it was closed. Render one with "
            f"`crucible gate --gate {phase_id} --closing-comment --store <uri>`."
        ]
    problems: list[str] = []
    version = document.get("schema_version")
    if version != CLOSING_READING_SCHEMA_VERSION:
        return [
            f"the block declares schema_version {version!r}, not "
            f"{CLOSING_READING_SCHEMA_VERSION!r}; a reader that cannot read the version "
            "refuses the document rather than guessing at its fields."
        ]
    if document.get("phase") != phase_id:
        problems.append(
            f"the block was rendered for phase {document.get('phase')!r}, not {phase_id!r} "
            "— a reading pasted onto the wrong issue."
        )
    state = document.get("gate_state")
    if state != "MET":
        problems.append(
            f"the gate reads {state!r}, not 'MET'. A phase issue may not rest closed while "
            "its own exit gate says it has not exited."
        )
    total = document.get("clauses_total")
    met = document.get("clauses_met")
    unmeasurable = document.get("clauses_unmeasurable")
    clauses = document.get("clauses")
    if not isinstance(total, int) or total < 1:
        problems.append(
            f"the block records {total!r} clauses. A gate with no clauses measured nothing, "
            "and nothing is never a pass (plan §6 rule 1)."
        )
    elif met != total:
        problems.append(
            f"the block records {met} of {total} clauses met, not {total} of {total} — a "
            "phase whose gate holds on a clause has not exited."
        )
    if unmeasurable:
        problems.append(
            f"{unmeasurable} clause(s) read UNMEASURABLE. Unmeasurable is never met: it is a "
            "fact about our reading, not about the phase."
        )
    if document.get("met_ratio") != 1.0:
        problems.append(
            f"met_ratio is {document.get('met_ratio')!r}, not 1.0 — a fully met gate reads 1.0 "
            "and nothing else does."
        )
    if not isinstance(clauses, list):
        problems.append("the block carries no clause list, so its counts cannot be checked.")
    else:
        if isinstance(total, int) and len(clauses) != total:
            problems.append(
                f"the block lists {len(clauses)} clauses but claims {total} — the transcript "
                "does not match its own summary."
            )
        unmet = [
            c.get("name")
            for c in clauses
            if not isinstance(c, dict) or c.get("met") is not True or c.get("unmeasurable")
        ]
        if unmet:
            problems.append(
                f"clause(s) {unmet} are not met on the block's own transcript, while the block "
                "claims MET."
            )
    store = document.get("store")
    if not isinstance(store, str) or not store:
        problems.append(
            "the block names no store, so the gate artifact it cites cannot be fetched and the "
            "reading cannot be checked by anyone who doubts it."
        )
    commit = document.get("commit")
    if not isinstance(commit, str) or not _COMMIT_RE.match(commit):
        problems.append(
            f"commit {commit!r} is not at least {CLOSING_READING_COMMIT_MIN} lowercase hex "
            "characters, so the clause definitions that produced this reading cannot be "
            "recovered."
        )
    return problems
