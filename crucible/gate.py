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

import datetime as dt
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from crucible.components import Component, load_registry
from crucible.keys import arena_cycle_key, arm_register_key
from crucible.manifest import manifest_key
from crucible.release import POINTER_KEY
from crucible.report import attribution_key
from crucible.slots import SLOTS
from crucible.store import Store
from crucible.weekly import arc_stages

__all__ = [
    "GATE_SCHEMA_VERSION",
    "GATES",
    "Clause",
    "GateResult",
    "arm_name",
    "evaluate",
    "gate_key",
]

GATE_SCHEMA_VERSION = "gate.v1"


def gate_key(gate: str, trading_day: str) -> str:
    """Where a gate reading is filed. Keyed by trading day like everything else."""
    return f"gates/{gate}/{trading_day}/gate.json"


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

    @property
    def met(self) -> bool:
        return bool(self.clauses) and all(c.met for c in self.clauses)

    @property
    def met_ratio(self) -> float:
        """Met clauses over total. Zero clauses is 0.0, never 1.0 — an empty
        gate is a gate that measured nothing, and vacuous truth is exactly the
        shape that let a phase close unmeasured."""
        if not self.clauses:
            return 0.0
        return sum(1 for c in self.clauses if c.met) / len(self.clauses)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": GATE_SCHEMA_VERSION,
            "gate": self.gate,
            "trading_day": self.trading_day.isoformat(),
            "window": [d.isoformat() for d in self.window],
            "met": self.met,
            "met_ratio": round(self.met_ratio, 6),
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
        for clause in self.clauses:
            lines.append(f"  [{'x' if clause.met else ' '}] {clause.name}: {clause.detail}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# reading helpers. Each returns None on absence rather than raising, because
# an absent artifact is a clause's ANSWER, not an error in the gate.
# ---------------------------------------------------------------------------


def _read_json(store: Store, key: str) -> dict[str, Any] | None:
    if not store.exists(key):
        return None
    return json.loads(store.get_bytes(key))


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
    for key in store.list_keys("runs/smoke/"):
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


def _phase1(store: Store, window: list[dt.date], registry: dict[str, Component]) -> list[Clause]:
    return [
        _clause_arc_runs_ok(store, window, registry),
        _clause_arms_all_scored(store, window),
        _clause_attribution_renders(store, window),
        _clause_explain_walks_a_verdict(store, window),
        _clause_pointer_flipped_on_smoke(store, window),
    ]


#: The gates this command can read, and how wide a window each needs. Phase 1
#: is five replay Saturdays; the phase-2 gate is four consecutive LIVE ones and
#: is not registered here, because it is measured over production manifests
#: that do not exist yet and a clause list without them would be a gate that
#: could go green on replays (`alpha-engine-config-I9757`).
GATES: dict[str, tuple[int, Any]] = {
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
    return result


def _unused(_: Iterable[Any]) -> None:  # pragma: no cover - typing shim
    return None
