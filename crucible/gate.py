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
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from crucible.calendar import resolve_trading_day
from crucible.components import Component, load_registry
from crucible.keys import (
    arena_cycle_key,
    arm_register_key,
    gate_key,
    legacy_weekly_executions_key,
    review_key,
    review_prefix,
    runs_prefix,
)  # noqa: F401 - re-exported
from crucible.manifest import manifest_key
from crucible.release import POINTER_KEY
from crucible.report import attribution_key
from crucible.slots import SLOTS
from crucible.store import Store
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
    "PHASE0_DELIVERABLES",
    "PHASES",
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
    """One gate condition and what the store said about it."""

    name: str
    requirement: str
    met: bool
    detail: str
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "requirement": self.requirement,
            "met": self.met,
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
        """
        if not self.clauses:
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
            lines.append(f"  [{'x' if clause.met else ' '}] {clause.name}: {clause.detail}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# reading helpers. Each returns None on absence rather than raising, because
# an absent artifact is a clause's ANSWER, not an error in the gate.
# ---------------------------------------------------------------------------


def _read_json(store: Store, key: str) -> dict[str, Any] | None:
    """Read a FIRST-PARTY artifact this repository wrote.

    Malformed is deliberately an exception here: every caller is a phase-1
    clause over a `run_manifest.v1`/arena artifact written by
    `crucible.runner` and validated against its schema at write time, so a
    malformed one is a broken build, not a producer we do not control. Reads
    of documents written OUTSIDE this repository go through
    :func:`_read_document`, which turns every malformed shape into a red
    clause instead (`alpha-engine-config-I9869` tracks giving the phase-1
    clauses the same treatment).
    """
    if not store.exists(key):
        return None
    return json.loads(store.get_bytes(key))


@dataclass(frozen=True)
class DocumentRead:
    """One external document, or the reason it could not be read.

    Three outcomes, never two: present-and-readable, ABSENT, and
    UNREADABLE. Collapsing the last two is the defect this exists to prevent
    — "no filed count" and "the file is corrupt" call for different actions,
    and only one of them is about the system rather than about us.
    """

    document: dict[str, Any] | None
    absent: bool
    problem: str | None


def _read_document(source: str, reader: Callable[[], bytes | None]) -> DocumentRead:
    """Read one document written by a producer OUTSIDE this repository.

    ``reader`` returns the raw bytes, or ``None`` when the source is absent.

    The shape of `crucible.board._fetch`, and here for the same reason: this
    is the only place in `gate.py` where a document nobody in this repository
    wrote is parsed, and **an exception raised here does not fail one clause —
    it propagates out of `evaluate` and takes `crucible gate`, `build_ladder`
    and the board render down together.** One malformed upstream file would
    then publish NOTHING where a red reading belongs, which is precisely the
    absence-instead-of-red failure the whole gate exists to refuse. So every
    failure mode below becomes a clause detail naming the source and the
    fault.

    The broad `except` is deliberate and is not a swallow: the failure mode
    caught is "an external document cannot be parsed", and the recording
    surface is the returned :class:`DocumentRead`, which every caller renders
    into an unmet clause. Nothing is discarded and nothing degrades silently.
    """
    try:
        raw = reader()
    except Exception as exc:
        return DocumentRead(
            None,
            False,
            f"{source} could not be read: {type(exc).__name__}: {exc}. That is a "
            "statement about our access, not about the system being measured",
        )
    if raw is None:
        return DocumentRead(None, True, None)
    try:
        document = json.loads(raw)
    except Exception as exc:
        return DocumentRead(
            None,
            False,
            f"{source} is present but is not readable JSON: {type(exc).__name__}: {exc}",
        )
    if document is None:
        return DocumentRead(
            None,
            False,
            f"{source} is present and its body is literal `null` — present-but-null is "
            "unreadable, not absent, and reporting it as absent names the wrong remedy",
        )
    if not isinstance(document, dict):
        return DocumentRead(
            None, False, f"{source} parsed to {type(document).__name__}, not an object with fields"
        )
    return DocumentRead(document, False, None)


def _read_store_document(store: Store, key: str) -> DocumentRead:
    """:func:`_read_document` over a store key."""

    def reader() -> bytes | None:
        return store.get_bytes(key) if store.exists(key) else None

    return _read_document(key, reader)


def _read_path_document(path: Path) -> DocumentRead:
    """:func:`_read_document` over a file in the checkout."""

    def reader() -> bytes | None:
        return path.read_bytes() if path.is_file() else None

    return _read_document(str(path), reader)


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


def _register_arms(store: Store, slot: str) -> tuple[set[str], str | None]:
    """The slot's ACTIVE registered arm ids, and the key they came from."""
    key = arm_register_key(slot)
    if not store.exists(key):
        return set(), None
    from nousergon_lib.arena import ArmRegister  # noqa: PLC0415 - heavy import, one call site

    events = [
        json.loads(line)
        for line in store.get_bytes(key).decode("utf-8").splitlines()
        if line.strip()
    ]
    return set(ArmRegister.from_dicts(events).active_arms()), key


# ---------------------------------------------------------------------------
# phase 1 clauses
# ---------------------------------------------------------------------------


def _clause_arc_runs_ok(
    store: Store, window: list[dt.date], registry: dict[str, Component]
) -> Clause:
    requirement = (
        "every stage of the weekly arc wrote a manifest with status `ok` for each "
        "trading day in the window"
    )
    missing: list[str] = []
    failed: list[str] = []
    evidence: list[str] = []
    for day in window:
        for stage in arc_stages(day, registry):
            key = manifest_key(stage.job, day.isoformat(), discriminator=stage.slot)
            evidence.append(key)
            document = _read_json(store, key)
            if document is None:
                missing.append(f"{stage.label}@{day.isoformat()}")
            elif document["status"] != "ok":
                failed.append(f"{stage.label}@{day.isoformat()}: {document['reason']}")
    if missing or failed:
        parts = []
        if missing:
            parts.append(f"{len(missing)} never ran ({', '.join(missing[:4])}...)")
        if failed:
            parts.append(f"{len(failed)} failed ({'; '.join(failed[:2])})")
        return Clause("arc_runs_ok", requirement, False, "; ".join(parts), tuple(evidence))
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
    evidence: list[str] = []
    for day in window:
        for slot, spec in SLOTS.items():
            key = arena_cycle_key(slot, day.isoformat())
            evidence.append(key)
            cycle = _read_json(store, key)
            if cycle is None:
                gaps.append(f"{slot}@{day.isoformat()}: no arena_cycle artifact")
                continue
            scored = set(cycle["scored_arms"])
            registered, _ = _register_arms(store, slot)
            unscored = registered - scored
            if unscored:
                gaps.append(f"{slot}@{day.isoformat()}: {sorted(unscored)} registered but unscored")
            controls = {c.arm_id for c in spec.control_arms}
            if not controls & {arm_name(a) for a in scored}:
                gaps.append(
                    f"{slot}@{day.isoformat()}: no control arm was scored — an unscored "
                    "control is an unverified grader (§10.1)"
                )
    if gaps:
        return Clause("arms_all_scored", requirement, False, "; ".join(gaps[:4]), tuple(evidence))
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
    document = _read_json(store, key)
    if document is None:
        return Clause("attribution_renders", requirement, False, f"{key} is absent", (key,))
    rows = document.get("rows", [])
    if len(rows) != 5:
        return Clause(
            "attribution_renders", requirement, False, f"{len(rows)} rows, expected 5", (key,)
        )
    silent = [
        r["name"]
        for r in rows
        if r.get("value") is None
        and not str(r.get("status", "")).startswith("N/A")
        or not r.get("status_reason")
    ]
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
    for key in evidence:
        document = _read_json(store, key)
        if document is None or document["status"] != "ok":
            continue
        if any("verdict.json" in i["key"] for i in document.get("inputs", [])):
            return Clause("explain_walks_a_verdict", requirement, True, f"walked at {key}", (key,))
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
    pointer = _read_json(store, POINTER_KEY)
    if pointer is None:
        return Clause(
            "pointer_flipped_on_smoke",
            requirement,
            False,
            f"{POINTER_KEY} is absent; no release has ever been published",
            (POINTER_KEY,),
        )
    sha = pointer.get("sha", "")
    evidence = [POINTER_KEY]
    for key in store.list_keys(runs_prefix("smoke")):
        if not key.endswith("run.json"):
            continue
        evidence.append(key)
        document = _read_json(store, key)
        if document and document["status"] == "ok" and document.get("release_sha") == sha:
            return Clause(
                "pointer_flipped_on_smoke",
                requirement,
                True,
                f"{POINTER_KEY} -> {sha[:12]}, smoked at {key}",
                tuple(evidence),
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
    keys = sorted(key for key in store.list_keys(prefix) if key.endswith(".json"))
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
    reviews: list[_Review] = []
    for key in keys:
        read = _read_store_document(store, key)
        if read.problem is not None:
            problems.append(read.problem)
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
    if problems:
        return Clause("independently_reviewed", requirement, False, "; ".join(problems), evidence)

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


def _clause_old_weekly_within_cadence(store: Store, window: list[dt.date]) -> Clause:
    requirement = (
        f"the v1 weekly state machine started at most {LEGACY_WEEKLY_MAX_STARTS_PER_WEEK} "
        "execution in EACH week of the window, read from a filed count keyed on the "
        "week, not on the day the gate was read"
    )
    anchors = list(dict.fromkeys(weekly_anchor(day) for day in window))
    evidence = [legacy_weekly_executions_key(a.isoformat()) for a in anchors]
    if len(anchors) != len(window):
        return Clause(
            "old_weekly_within_cadence",
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
        if started > LEGACY_WEEKLY_MAX_STARTS_PER_WEEK:
            over.append(f"{key}: {started} starts")
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
        return Clause(
            "old_weekly_within_cadence", requirement, False, "; ".join(parts), tuple(evidence)
        )
    return Clause(
        "old_weekly_within_cadence",
        requirement,
        True,
        f"{len(evidence)} consecutive weeks at <= {LEGACY_WEEKLY_MAX_STARTS_PER_WEEK} start each",
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


#: The gates this command can read, and how wide a window each needs. Phase 0's
#: window is TWO weeks, which is `alpha-engine-config-I9756`'s own closes-when
#: ("<= 1 start per calendar week for two consecutive weeks") — one quiet week
#: is a gap between reruns, not a cadence. Phase 1 is five replay Saturdays; the
#: phase-2 gate is two consecutive LIVE ones (§6.1's ruled minimum —
#: "2 consecutive first-attempt `ok` Saturdays, not 4", the other two soak weeks
#: traded for the five replays) and is not registered here, because it is
#: measured over production manifests that do not exist yet and a clause list
#: without them would be a gate that could go green on replays
#: (`alpha-engine-config-I9757`).
GATES: dict[str, tuple[int, Any]] = {
    "phase0": (2, _phase0),
    "phase1": (5, _phase1),
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


#: The plan §6 ladder. Phases 2-5 carry no registered gate: their clause lists
#: are not written yet, and inventing one here would be a gate that could go
#: green over nothing. They render `UNMEASURED` until a clause list exists in
#: `GATES` — which is exactly what principle 7 asks for, and is the fact that
#: was invisible when phase 1 closed ahead of phase 0. Phase 0's list was
#: written for `alpha-engine-config-I9804`, so the phase actually in flight is
#: measured rather than unreadable.
PHASES: tuple[Phase, ...] = (
    Phase("phase0", 0, "Stop the bleeding", 9756, "phase0"),
    Phase("phase1", 1, "One command, locally", 9757, "phase1"),
    Phase("phase2", 2, "Unattended", 9758, None),
    Phase("phase3", 3, "All three slots", 9759, None),
    Phase("phase4", 4, "Trader on the contract; decommission", 9760, None),
    Phase("phase5", 5, "Grow", 9761, None),
)

#: The ladder's closed state vocabulary. Total, with no fall-through, and no
#: fifth member: `UNKNOWN`/`PENDING`/`N/A` are the shapes this exists to
#: remove (`observability-policy` §8.3, same posture as
#: `crucible.console.classify`).
#:
#: * `MET`          — the gate was read and every clause is met.
#: * `UNMET`        — the gate was read and at least one clause is not met.
#:                    A real result, not a failure.
#: * `UNMEASURED`   — no clause list is registered for this phase, or no
#:                    reading has ever been filed. Never green, never zero.
#: * `OUT_OF_ORDER` — a later phase is being graded while an earlier phase's
#:                    gate is not met. Brian's ruling of 2026-09-02: phase 0's
#:                    gate must READ before phase 2 opens.
LADDER_STATES: tuple[str, ...] = ("MET", "UNMET", "UNMEASURED", "OUT_OF_ORDER")

#: How a ladder state renders on the fleet console, in `observability-policy`
#: §8.3's vocabulary. `UNMEASURED` maps to `UNREPORTED` and therefore counts
#: against the transparency gap whose objective is zero — a phase nobody can
#: read is unobserved, not healthy. `OUT_OF_ORDER` maps to `FAILED` because it
#: is an invariant breach, not a slow phase.
LADDER_CONSOLE_STATE: dict[str, str] = {
    "MET": "HEALTHY",
    "UNMET": "DEGRADED",
    "UNMEASURED": "UNREPORTED",
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


def last_read(store: Store, gate: str) -> str | None:
    """The most recent trading day a reading of ``gate`` was filed for.

    ``None`` when the gate has never been read. That is the field the ladder
    row publishes as "when was this last measured", and a row that cannot
    answer it says so rather than borrowing the ladder's own generation time —
    a freshly written ladder full of never-measured phases would otherwise look
    entirely fresh.
    """
    days: list[str] = []
    for key in store.list_keys(f"gates/{gate}/"):
        parts = key.split("/")
        if key.endswith("/gate.json") and len(parts) == 4:
            days.append(parts[2])
    return max(days) if days else None


@dataclass(frozen=True)
class PhaseRow:
    """One phase, as the console reads it."""

    phase: Phase
    state: str
    gate_state: str
    detail: str
    clauses_met: int | None
    clauses_total: int | None
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

    gate_states: list[tuple[Phase, str, str, int | None, int | None, float | None, str | None]] = []
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
                )
            )
            continue
        reading = supplied.get(phase.gate) or evaluate(
            store, gate=phase.gate, trading_day=trading_day, registry=reg
        )
        met = sum(1 for c in reading.clauses if c.met)
        total = len(reading.clauses)
        read_on = last_read(store, phase.gate)
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
                    reading.met_ratio,
                    read_on,
                )
            )
            continue
        unmet = [c.name for c in reading.clauses if not c.met]
        detail = (
            f"{met}/{total} clauses met"
            if not unmet
            else f"{met}/{total} clauses met; holding: {', '.join(unmet)}"
        )
        # The coverage line travels with the row. Without it a phase whose
        # gate grades a SUBSET of its deliverables renders MET on the ladder
        # and on the board with nothing saying so — partial coverage reported
        # as complete, which is the pattern `alpha-engine-config-I9837`
        # exists to make visible rather than one this row gets to repeat.
        if reading.coverage:
            detail = f"{detail}; {reading.coverage}"
        gate_states.append(
            (
                phase,
                "MET" if reading.met else "UNMET",
                detail,
                met,
                total,
                reading.met_ratio,
                read_on,
            )
        )

    # Out-of-order: a phase later than the lowest not-met phase that has
    # nevertheless been graded. Derived from readings alone, so nothing here
    # depends on a hand-maintained claim about which phase is "open".
    first_unmet = next((i for i, s in enumerate(gate_states) if s[1] != "MET"), len(gate_states))
    rows: list[PhaseRow] = []
    for index, (phase, gate_state, detail, met, total, ratio, read_on) in enumerate(gate_states):
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
