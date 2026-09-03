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
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from crucible.alerts import pages_in_range
from crucible.calendar import TRADING_DAYS_PER_WEEK, resolve_trading_day
from crucible.components import Component, load_registry
from crucible.documents import DocumentRead
from crucible.documents import read_path_document as _read_path_document
from crucible.documents import read_store_document as _read_store_document
from crucible.keys import (
    ALERTS_ROOT,
    arena_cycle_key,
    arm_register_key,
    champion_key,
    gate_key,
    gate_prefix,
    legacy_weekly_executions_key,
    review_key,
    review_prefix,
    runs_prefix,
    verdict_key,
)  # noqa: F401 - re-exported
from crucible.manifest import load_schema, manifest_key
from crucible.release import POINTER_KEY
from crucible.report import attribution_key
from crucible.slots import SLOTS
from crucible.store import Store
from crucible.tags import TAG_KEY, TAG_VALUE
from crucible.weekly import arc_stages

__all__ = [
    "ACCEPTANCE_RATCHET_PATH",
    "GATE_SCHEMA_VERSION",
    "GATES",
    "LEGACY_WEEKLY_MAX_STARTS_PER_WEEK",
    "LADDER_CONSOLE_STATE",
    "LADDER_KEY",
    "LADDER_SCHEMA_VERSION",
    "LADDER_STATES",
    "GATE_DELIVERABLES",
    "LLM_ARM_CALLSITE_FIELD",
    "MANIFEST_RUN_MODE_FIELD",
    "MANIFEST_RUN_MODE_LIVE",
    "PHASE0_DELIVERABLES",
    "PHASE2_LIVE_SATURDAYS",
    "PHASE2_MAX_PAGES",
    "PHASE2_MAX_TAGGED_USD",
    "PHASE2_REPLAY_SATURDAYS",
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
    "legacy_weekly_executions_key",
    "review_key",
    "review_prefix",
    "weekly_anchor",
    "last_read",
    "ladder_schema",
    "validate_ladder_document",
]

GATE_SCHEMA_VERSION = "gate.v1"


@dataclass(frozen=True)
class Clause:
    """One gate condition and what the store said about it.

    A clause is met, unmet, or UNMEASURABLE, and unmeasurable is never met
    (module docstring; `alpha-engine-config-I9869` round 2). ``unmeasurable``
    is a fact about OUR access — a store read that raised (a permission
    denial, a transient AccessDenied) — never a fact about the system being
    graded, and it renders distinctly rather than being folded into "unmet"
    so an operator does not go fix a producer when the real fault is our own
    credentials.
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


@dataclass
class GateResult:
    """Every clause of one gate, and whether the phase may exit."""

    gate: str
    trading_day: dt.date
    window: list[dt.date]
    clauses: list[Clause] = field(default_factory=list)
    #: How much of the phase issue this clause list grades, or None when the
    #: gate declares no deliverable list. Not a clause: it is true by
    #: construction, so counting it in `met_ratio` would inflate the one
    #: figure plan §6 rule 3 exists to keep honest. Set by `evaluate`.
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


def _register_arms(store: Store, slot: str) -> tuple[set[str], str, str | None, bool]:
    """The slot's ACTIVE registered arm ids, the key, and the reason if the
    register could not be read (``problem``, ``access_problem``).

    Guarded through :func:`_read_store_lines` rather than the bare
    `json.loads` per line this used to run: an unreadable register is a red
    `arms_all_scored` reading naming the key, never an exception out of
    `evaluate` (`alpha-engine-config-I9869` round 2).
    """
    key = arm_register_key(slot)
    read = _read_store_lines(store, key)
    if read.problem is not None:
        return set(), key, read.problem, read.access_problem
    if read.absent:
        return set(), key, None, False
    from nousergon_lib.arena import ArmRegister  # noqa: PLC0415 - heavy import, one call site

    return set(ArmRegister.from_dicts(read.lines or []).active_arms()), key, None, False


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
    """The exhaustive `run_manifest.v1` `status` vocabulary, derived from the
    SCHEMA every manifest is validated against at write time — never a
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
    implication `run_manifest.v1`'s own conditional schema enforces at write
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


def _clause_arc_runs_ok(
    store: Store, window: list[dt.date], registry: dict[str, Component]
) -> Clause:
    requirement = (
        "every stage of the weekly arc wrote a manifest with status `ok` for each "
        "trading day in the window"
    )
    missing: list[str] = []
    malformed: list[str] = []
    unmeasurable: list[str] = []
    failed: list[str] = []
    evidence: list[str] = []
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
    return Clause(
        "arc_runs_ok",
        requirement,
        True,
        f"{len(evidence)} stage manifests over {len(window)} trading days, all ok",
        tuple(evidence),
    )


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
    for slot in SLOTS:
        registered, register_key, register_problem, register_access = _register_arms(store, slot)
        registers[slot] = (registered, register_key, register_problem, register_access)
        if register_key is not None:
            evidence.append(register_key)
        if register_problem is not None:
            (unmeasurable if register_access else gaps).append(register_problem)
    for day in window:
        for slot, spec in SLOTS.items():
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
        f"{len(SLOTS)} slots x {len(window)} days, every registered arm and both controls scored",
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


def _phase1(store: Store, window: list[dt.date], registry: dict[str, Component]) -> list[Clause]:
    return [
        _clause_arc_runs_ok(store, window, registry),
        _clause_arms_all_scored(store, window),
        _clause_attribution_renders(store, window),
        _clause_explain_walks_a_verdict(store, window),
        _clause_pointer_flipped_on_smoke(store, window),
        _clause_independently_reviewed(store, "phase1", window),
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
#: what it ends).
LEGACY_WEEKLY_MAX_STARTS_PER_WEEK = 1


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


def _clause_old_weekly_within_cadence(
    store: Store,
    window: list[dt.date],
    *,
    name: str = "old_weekly_within_cadence",
    maximum: int = LEGACY_WEEKLY_MAX_STARTS_PER_WEEK,
) -> Clause:
    """The v1 weekly start count, read from a filed document, against a ceiling.

    ``maximum`` is a parameter because plan §6 asks the SAME question at two
    ceilings: phase 0 tolerates one start a week while the v1 system is being
    quieted, and phase 4 requires ZERO once it is decommissioned. A second
    function for the second ceiling would be the same reader restated, and
    this repository has already found one of those drifting inside the change
    that introduced it (`_review_problem`). ``name`` travels with it so the
    two readings do not both land on the ladder under one clause name.
    """
    requirement = (
        f"the v1 weekly state machine started at most {maximum} "
        "execution in EACH week of the window, read from a filed count keyed on the "
        "week, not on the day the gate was read"
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
    over: list[str] = []
    for key in evidence:
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
        started = document.get("executions_started")
        if isinstance(started, bool) or not isinstance(started, int) or started < 0:
            malformed.append(f"{key}: `executions_started` is {started!r}, not a count")
            continue
        if started > maximum:
            over.append(f"{key}: {started} starts, ceiling {maximum}")
    if missing or malformed or over:
        parts: list[str] = []
        if missing:
            parts.append(
                f"{len(missing)} of {len(evidence)} weeks have no filed count: "
                f"{', '.join(missing)}. Nothing writes this key yet; a gate may not "
                "call `states:ListExecutions` to find out"
            )
        if malformed:
            parts.append(f"{len(malformed)} malformed: {'; '.join(malformed)}")
        if over:
            parts.append("; ".join(over))
        return Clause(name, requirement, False, "; ".join(parts), tuple(evidence))
    return Clause(
        name,
        requirement,
        True,
        f"{len(evidence)} consecutive weeks at <= {maximum} start each",
        tuple(evidence),
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
#: Two are graded. The other three are each a property of AWS that no artifact
#: in this store records, and a gate does not call AWS — so they are named here
#: with their reason and their real reading surface, and :func:`coverage_note`
#: publishes that subset onto every surface the reading reaches. The
#: alternative, an ungraded deliverable simply missing from the clause list, is
#: how "the gate is met" comes to mean less than a reader assumes.
PHASE0_DELIVERABLES: tuple[Deliverable, ...] = (
    Deliverable(
        "old_weekly_once_per_week",
        "the v1 weekly pipeline runs at most once a week, with no rerun issuers",
        "old_weekly_within_cadence",
    ),
    Deliverable(
        "dead_lambdas_deleted",
        "the six zero-invocation v1 functions are deleted via their owning IaC",
        None,
        "plan §6 names the deleted-Lambda count as a PROGRESS figure and explicitly "
        "not the gate; the only surface that answers it is a live AWS read, which a "
        "gate may not make. Graded by the owning repo's IaC drift check.",
    ),
    Deliverable(
        "old_alerts_muted",
        "v1 weekly/rehearsal alert emitters route to the muted topic, trading alerts unchanged",
        None,
        "the evidence is a forced v1 failure arriving in the muted topic and not in "
        "the paging one — an observation about a notification channel, which leaves "
        "no artifact in this store. Graded by the routing change's own PR.",
    ),
    Deliverable(
        "v2_resources_tagged_and_versioned",
        "S3 versioning on the v2 prefix and `system=crucible-v2` on every v2 resource",
        None,
        "`crucible.tags.audit_stack_tags` resolves this against live CloudFormation "
        "and IAM and raises when the stack is absent. It is carried as an acceptance "
        "clause (`TestCost::test_every_v2_resource_is_tagged_for_cost_attribution`), "
        "committed unmet; it becomes a gate clause the day that audit files its "
        "result to the store.",
    ),
    Deliverable(
        "acceptance_tests_written_as_failing_pytest",
        "the plan §2 clauses exist as pytest, written before the code that satisfies them",
        "acceptance_suite_committed",
    ),
)

#: Which phase issue's deliverables each gate is answerable for. A gate absent
#: from this table grades no declared deliverable list and gets no coverage
#: line — silence here is "not declared", never "grades everything".
GATE_DELIVERABLES: dict[str, tuple[Deliverable, ...]] = {
    "phase0": PHASE0_DELIVERABLES,
}


def coverage_note(gate: str, clause_names: Iterable[str]) -> str | None:
    """How much of ``gate``'s phase issue this clause list actually grades.

    Returns None for a gate with no declared deliverable list. Otherwise a
    single line naming the subset and every deliverable nothing measures —
    carried on :class:`GateResult`, printed by `render`, and appended to the
    ladder row's detail, so a phase cannot render MET on a surface that gives
    no sign three of its five deliverables were never looked at.

    **Raises** when the table and the clause list disagree: a deliverable
    naming a clause this reading does not contain, or an ungraded deliverable
    with no written reason. That is a defect in this file, not a condition of
    the store, and rule 5's default is `raise` — a reading published from an
    inconsistent table would quietly grade less than it claims.
    """
    deliverables = GATE_DELIVERABLES.get(gate)
    if not deliverables:
        return None
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
        return f"grades all {len(deliverables)} {tracker} deliverables"
    return (
        f"grades {len(deliverables) - len(ungraded)} of {len(deliverables)} {tracker} "
        f"deliverables; not gate-readable: {', '.join(d.id for d in ungraded)}"
    )


def _phase0(store: Store, window: list[dt.date], registry: dict[str, Component]) -> list[Clause]:
    """Phase 0's exit gate.

    ``registry`` is unused: phase 0 predates the weekly arc entirely — it is
    about the v1 system being quieted and the §2 clauses being written — so
    there is no `components.yaml` row to read. The parameter is kept because
    `GATES` holds one callable shape, and a second signature would be a
    per-gate special case in `evaluate`.
    """
    _unused((registry,))
    return [
        _clause_old_weekly_within_cadence(store, window),
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

#: How many first-attempt `ok` LIVE Saturdays phase 2 requires. §6.1's ruled
#: minimum — "2 consecutive first-attempt `ok` Saturdays, not 4", the other
#: two soak weeks traded for the five replays.
PHASE2_LIVE_SATURDAYS = 2

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

#: The arm-recipe field that would name which registered LLM call site an arm
#: reaches a model through — `None` because no such field exists.
#: `ArmSpec`/`REQUIRED_ARM_FIELDS` carry `name`, `slot`, `ranker`, `params`
#: and `registered_at`, and the id-hashed `spec` carries no call-site
#: reference, so "which arms are LLM arms" is not answerable from the register
#: today. Filed as `alpha-engine-config-I9920`.
LLM_ARM_CALLSITE_FIELD: str | None = None


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
    replay_window = _window(window[-1], PHASE2_REPLAY_SATURDAYS)
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


def _s3_client() -> Any:  # pragma: no cover - constructed only outside tests
    """An S3 client for the CloudTrail archive read.

    A module-level function so a test can substitute it without reaching for a
    credential chain, and lazy for the reason `crucible.store.S3Store` is:
    importing `crucible.gate` must not require an AWS SDK.
    """
    import boto3  # noqa: PLC0415 - lazy on purpose

    return boto3.client("s3")


def _clause_zero_human_mutating_calls(window: list[dt.date]) -> Clause:
    """Plan §6 row 2 and §11 risk 8: zero human-originated mutating calls.

    Read through `crucible.autonomy`, which walks the CloudTrail **S3
    archive** over the whole window. `aws cloudtrail lookup-events` is
    forbidden there and the acceptance suite asserts the module has no path to
    it: the username lookup silently truncates to ~2 days, so a gate built on
    it reports zero because it looked at two days.
    """
    name = "zero_human_mutating_calls"
    requirement = (
        "zero human-originated mutating calls touched a v2 resource over the window, "
        "counted from the CloudTrail S3 archive (never `lookup-events`, which truncates "
        "its username lookup to ~2 days — §11 risk 8)"
    )
    from crucible.autonomy import (  # noqa: PLC0415 - heavy import, one call site
        ArchiveMissingError,
        count_operator_actions,
    )
    from crucible.config import settings  # noqa: PLC0415 - one call site

    archive = settings().cloudtrail_archive
    evidence = (archive,) if archive else ()
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
    bucket, _, prefix = archive.removeprefix("s3://").partition("/")
    try:
        counted = count_operator_actions(
            _s3_client(),
            bucket=bucket,
            prefix=prefix,
            start=window[0],
            end=window[-1],
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
    if counted.count:
        offenders = ", ".join(
            f"{a.principal} {a.event_name}@{a.event_time}" for a in counted.actions[:4]
        )
        return Clause(
            name,
            requirement,
            False,
            f"{counted.count} human mutating call(s) over "
            f"{window[0].isoformat()}..{window[-1].isoformat()}: {offenders}",
            evidence,
        )
    return Clause(
        name,
        requirement,
        True,
        f"0 human mutating calls over {counted.records_scanned} records in "
        f"{counted.objects_read} archive objects",
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
    """
    name = "pages_within_ceiling"
    requirement = (
        f"at most {PHASE2_MAX_PAGES} paged incidents over the window, counted from the "
        "alert bus (one row per incident, not per observation or per member)"
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
        incidents = pages_in_range(store, start=start, end=end)
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


def _ce_client() -> Any:  # pragma: no cover - constructed only outside tests
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
    """
    from crucible.cost import CostUnreadableError, month_to_date_usd  # noqa: PLC0415

    scope = f"tagged `{TAG_KEY}={TAG_VALUE}`" if tagged else "the whole account"
    requirement = (
        f"month-to-date AWS spend for {scope} is at most ${ceiling_usd:.2f}, read from "
        "Cost Explorer"
    )
    evidence = ("ce:GetCostAndUsage",)
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
        # with no resources and an entirely UNTAGGED estate both return, and
        # this account's untagged total was $230.21 on the day this was
        # written. An ACCOUNT total of exactly $0.00 is the same shape one
        # level up: for a live AWS estate it means Cost Explorer answered with
        # nothing chargeable — a broken reading, not a free month. Reading
        # either as "under the ceiling" would put a phase row green on the
        # evidence that the cost reading is not working. Whether the tag is
        # actually applied is `crucible.tags.audit_stack_tags`' question, and
        # phase 0's `v2_resources_tagged_and_versioned` deliverable.
        return _unmeasurable(
            name,
            requirement,
            f"Cost Explorer returned exactly $0.00 for {reading.scope} over "
            f"{reading.start.isoformat()}..{reading.end.isoformat()}: no spend recorded "
            "under this filter — the tag or the account read is not evidence of cost "
            "under the ceiling. A total of zero is what a broken or misfiltered reading "
            "returns as well as a free month. `crucible.tags.audit_stack_tags` is the "
            "reading that says whether the tag is applied",
            evidence,
        )
    if reading.amount_usd > ceiling_usd:
        return Clause(
            name,
            requirement,
            False,
            f"${reading.amount_usd:.2f} month-to-date for {reading.scope} "
            f"({reading.start.isoformat()}..{reading.end.isoformat()}), ceiling "
            f"${ceiling_usd:.2f}",
            evidence,
        )
    return Clause(
        name,
        requirement,
        True,
        f"${reading.amount_usd:.2f} month-to-date for {reading.scope} "
        f"({reading.start.isoformat()}..{reading.end.isoformat()}), ceiling "
        f"${ceiling_usd:.2f}",
        evidence,
    )


def _phase2(store: Store, window: list[dt.date], registry: dict[str, Component]) -> list[Clause]:
    """Phase 2's exit gate (plan §6 row 2, §6.1's ruled minimum)."""
    return [
        _clause_live_saturdays_first_attempt_ok(store, window),
        _clause_replays_ok(store, window, registry),
        _clause_zero_human_mutating_calls(window),
        _clause_pages_within_ceiling(store, window),
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


def _phase3(store: Store, window: list[dt.date], registry: dict[str, Component]) -> list[Clause]:
    """Phase 3's exit gate (plan §6 row 3), one clause per slot in `SLOTS`."""
    _unused((registry,))
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


def _phase4(store: Store, window: list[dt.date], registry: dict[str, Component]) -> list[Clause]:
    """Phase 4's exit gate (plan §6 row 4)."""
    _unused((registry,))
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
            store, window, name="old_sf_execution_count_zero", maximum=0
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
    day = window[-1]
    evidence: list[str] = []
    access: list[str] = []
    missing: list[str] = []
    llm_arms: list[str] = []
    for slot in sorted(SLOTS):
        key = arm_register_key(slot)
        evidence.append(key)
        read = _read_store_lines(store, key)
        if read.problem is not None:
            (access if read.access_problem else missing).append(read.problem)
            continue
        if read.absent:
            continue
        active, _, problem, access_problem = _register_arms(store, slot)
        if problem is not None:
            (access if access_problem else missing).append(problem)
            continue
        specs = {
            str(event.get("arm_id")): (event.get("spec") or {})
            for event in (read.lines or [])
            if isinstance(event.get("spec"), dict)
        }
        for arm_id in sorted(active):
            params = specs.get(arm_id, {}).get("params")
            if not isinstance(params, dict):
                continue
            callsite = params.get(LLM_ARM_CALLSITE_FIELD)
            if callsite in LLM_CALLSITE_REGISTRY:
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


def _phase5(store: Store, window: list[dt.date], registry: dict[str, Component]) -> list[Clause]:
    """Phase 5's exit gate (plan §6 row 5)."""
    _unused((registry,))
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
#: * phase 0 — TWO weeks, `alpha-engine-config-I9756`'s own closes-when
#:   ("<= 1 start per calendar week for two consecutive weeks"); one quiet
#:   week is a gap between reruns, not a cadence.
#: * phase 1 — FIVE replay Saturdays (§6 row 1).
#: * phase 2 — TWO, §6.1's ruled minimum ("2 consecutive first-attempt `ok`
#:   Saturdays, not 4", the other two soak weeks traded for the five replays).
#:   Its `replays_ok` clause re-reads phase 1's predicate over its own
#:   five-week window, so the narrow live window does not narrow the replay
#:   one.
#: * phase 3 — FOUR, `promote_min_weeks` (§5.0): a promotion cannot be won on
#:   fewer paired weeks than the eligibility age requires, so a shorter window
#:   could only ever read UNMET.
#: * phase 4 — TWO, covering §6 row 4's "one week" trader claim plus the same
#:   two-week cadence evidence phase 0 needs for the SF count.
#: * phase 5 — ONE, §6 row 5's "verdict within one weekly cycle".
GATES: dict[str, tuple[int, Any]] = {
    "phase0": (2, _phase0),
    "phase1": (5, _phase1),
    "phase2": (PHASE2_LIVE_SATURDAYS, _phase2),
    "phase3": (4, _phase3),
    "phase4": (2, _phase4),
    "phase5": (1, _phase5),
}


def _window(trading_day: dt.date, weeks: int) -> list[dt.date]:
    """The ``weeks`` weekly trading days ending at ``trading_day``, oldest first.

    Weekly work binds to a Friday close (§4.12), so the window steps back in
    calendar weeks and every element is the same weekday as the anchor.
    """
    if weeks < 1:
        raise ValueError("a gate window of fewer than one week measures nothing")
    return [trading_day - dt.timedelta(weeks=n) for n in reversed(range(weeks))]


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
    window = _window(trading_day, default_weeks if weeks is None else weeks)
    result = GateResult(gate=gate, trading_day=trading_day, window=window)
    result.clauses = list(clauses_fn(store, window, registry or load_registry()))
    result.coverage = coverage_note(gate, [c.name for c in result.clauses])
    return result


def _unused(_: Iterable[Any]) -> None:  # pragma: no cover - typing shim
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
        if unmeasurable_count > 0 or read_on_access_problem:
            state = "UNMEASURABLE"
        else:
            state = "MET" if reading.met else "UNMET"
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
    """
    errors = sorted(_ladder_validator().iter_errors(document), key=lambda e: list(e.absolute_path))
    if errors:
        detail = "\n".join(
            f"  - {'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"
            for e in errors
        )
        raise ValueError(f"ladder document does not conform to {LADDER_SCHEMA_VERSION}:\n{detail}")


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
