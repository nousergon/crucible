"""The fully-declared board — every row red before the thing it measures exists.

Normative source: `alpha-engine-config-I9837`, on Brian's ruling of
2026-09-02: *"should we gauge the crucible v2 buildout against a fully red
console that already contains everything we plan to measure in the system?"*

**A board assembled from what already reports can only show you the parts that
report.** That is not a limitation of a particular implementation, it is what
the shape does. The v2 buildout is therefore gauged against a board whose rows
are declared FIRST — one per plan §2 objective, one per §6 phase gate, one per
`components.yaml` component, one per cutover predicate — so the gap between
what was promised and what is measurable is a rendered number from day one
rather than something a reader has to notice is missing.

The failure this exists to prevent has a date. `alpha-engine-config-I9757`
closed 2026-09-02T01:31Z, the moment its build PRs merged, with zero replay
Saturdays run and its own gate reading 1 of 5. Nothing rendered the gap, so
nothing objected. A row reading `UNMEASURED — 0 of 5 replay Saturdays` since
the first commit could not have been closed by accident.

**Rows are derived from four declarations, never hand-listed.** A hand-kept
board drifts from what the plan promises, which is the same failure one level
up. :func:`build_board` reads:

* `crucible/board.yaml` — the plan §2 objective table and the cutover
  predicates.
* :data:`crucible.gate.PHASES` and :data:`crucible.gate.GATES` — the §6 ladder.
* `crucible/components.yaml` — the observability registry.

A row whose source declaration disappears raises. A board that shrinks quietly
is the same defect as a green row over no data, wearing a different face.

**`a gate reads; it never runs` extends to the whole board.** :func:`build_board`
renders from artifacts already in the store. It must not execute a phase's
work in order to grade it, and a missing artifact makes a row `UNMEASURED`
with the key named — never `MET`, never absent from the board.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from crucible.calendar import resolve_trading_day
from crucible.components import Component, load_registry
from crucible.console.classify import STATES as COMPONENT_STATES
from crucible.console.classify import Classification
from crucible.gate import LADDER_STATES, Ladder, PhaseRow
from crucible.store import Store

__all__ = [
    "BOARD_CONSOLE_STATE",
    "BOARD_CURRENT_KEY",
    "BOARD_SCHEMA_VERSION",
    "BOARD_STATES",
    "COMPONENT_BOARD_STATE",
    "DECLARATION_PATH",
    "GREY_STATES",
    "LADDER_BOARD_STATE",
    "RED_STATES",
    "SOURCES",
    "Board",
    "BoardRow",
    "Declaration",
    "Declarations",
    "RowDelta",
    "board_delta",
    "board_key",
    "board_html_key",
    "board_payload",
    "render_board_html",
    "build_board",
    "load_declarations",
    "read_verdict",
]

BOARD_SCHEMA_VERSION = "board.v1"

DECLARATION_PATH = Path(__file__).parent / "board.yaml"

#: Where a day's board is filed, and the pointer the console reads. Keyed by
#: trading day like everything else (§4.12).
BOARD_CURRENT_KEY = "board/current.json"


#: The served page. `console-policy` requires every view to also serve the
#: JSON an agent reads, which is `BOARD_CURRENT_KEY` — a page whose numbers can
#: only be scraped out of HTML is a page the next automated reader re-derives
#: incorrectly.
BOARD_HTML_KEY = "board/index.html"


def board_key(trading_day: str) -> str:
    return f"board/{trading_day}/board.json"


def board_html_key() -> str:
    return BOARD_HTML_KEY


#: The four declarations a row can come from. Closed: a fifth source is a
#: design change visible in a diff, not a new dict key someone adds.
SOURCES: tuple[str, ...] = ("objective", "phase", "component", "cutover")

#: The board's closed state vocabulary. Total, with no fall-through.
#:
#: **Uniform red is noise.** If "not built yet", "built and failing" and
#: "cannot be measured" render identically, the board desensitises exactly as
#: the acceptance gate did — measured 2026-09-02, seven Dependabot PRs sat
#: UNSTABLE on one by-design red check, and merging any of them required a
#: human to decide, per PR, that a red check did not count. A board that
#: teaches its reader to discount red has cost more than it delivers.
#:
#: * `MET`           — read, and at target.
#: * `UNMET`         — read, and below target. A real result, not a failure.
#: * `UNMEASURED`    — built, never read, or no reading for the current day.
#:                     Red. This is the state that closed I9757.
#: * `UNMEASURABLE`  — a read was attempted and could not complete: no
#:                     credential, an absent artifact, a document missing the
#:                     field. Red, never green, and never silently folded into
#:                     `UNMET` — "it says no" and "I could not ask" are
#:                     different facts, and the second one is about us.
#: * `OUT_OF_ORDER`  — an invariant breach: a later phase graded while an
#:                     earlier phase's gate is unmet (Brian's ruling of
#:                     2026-09-02, phase 0 gates phase 2).
#: * `PLANNED`       — declared, not built. The expected state on day one.
#:                     Grey and counted separately, and DECLARED rather than
#:                     inferred from an absent artifact: "nobody built it" and
#:                     "it broke" want opposite responses.
#: * `DECLARED_OFF`  — deliberately not measured (a `DISABLED`, `DEPRECATED`
#:                     or `RETIRED` registry row). Grey. Rendering a decision
#:                     as a defect is how a board trains people to ignore it.
BOARD_STATES: tuple[str, ...] = (
    "MET",
    "UNMET",
    "UNMEASURED",
    "UNMEASURABLE",
    "OUT_OF_ORDER",
    "PLANNED",
    "DECLARED_OFF",
)

#: The states that are RED. Named as a set rather than "everything but MET"
#: because the grey states must never drift into the red count by omission.
RED_STATES: frozenset[str] = frozenset({"UNMET", "UNMEASURED", "UNMEASURABLE", "OUT_OF_ORDER"})

#: Declared, not measured. Counted separately and never as progress.
GREY_STATES: frozenset[str] = frozenset({"PLANNED", "DECLARED_OFF"})

#: How a board state renders on the fleet console, in `observability-policy`
#: §8.3's vocabulary — the SAME vocabulary `crucible.console.classify` uses,
#: so a board row and a component row can sit on one surface without two
#: colour schemes.
BOARD_CONSOLE_STATE: dict[str, str] = {
    "MET": "HEALTHY",
    "UNMET": "DEGRADED",
    "UNMEASURED": "UNREPORTED",
    "UNMEASURABLE": "UNREPORTED",
    "OUT_OF_ORDER": "FAILED",
    "PLANNED": "ARMED",
    "DECLARED_OFF": "DISABLED",
}

#: `crucible.gate.LADDER_STATES` -> board states. An identity map today, and
#: written out anyway: the point of the map is that it is TOTAL and checked,
#: so a fifth ladder state cannot reach the board with no declared rendering.
#: Extending the ladder's vocabulary rather than duplicating it is deliberate
#: — a second state enum is a contract restated in two places, and this
#: repository has already watched one of those drift.
LADDER_BOARD_STATE: dict[str, str] = {
    "MET": "MET",
    "UNMET": "UNMET",
    "UNMEASURED": "UNMEASURED",
    "OUT_OF_ORDER": "OUT_OF_ORDER",
}

#: `crucible.console.classify.STATES` -> board states. Total over all
#: fourteen.
#:
#: Two mappings are worth arguing rather than reading past:
#:
#: * `ARMED` (on-demand, silence carries no claim of a recent run) maps to
#:   `UNMEASURED`, not to a grey state. Silence carrying no claim IS the
#:   definition of unmeasured; treating "we never ask" as acceptable is how a
#:   component ends up unobserved and reported as fine.
#: * `RUNNING` maps to `UNMEASURED` too. The deadline has not passed, so
#:   there is no reading for today — which is a true statement about today,
#:   and the row goes green the moment the manifest lands.
COMPONENT_BOARD_STATE: dict[str, str] = {
    "HEALTHY": "MET",
    "RUNNING": "UNMEASURED",
    "DEGRADED": "UNMET",
    "FAILED": "UNMET",
    "STALLED": "UNMET",
    "MISSED": "UNMET",
    "NEVER_RAN": "UNMEASURED",
    "DISABLED": "DECLARED_OFF",
    "DEPRECATED": "DECLARED_OFF",
    "RETIRED": "DECLARED_OFF",
    "ABSENT": "UNMET",
    "UNREGISTERED": "UNMEASURABLE",
    "UNREPORTED": "UNMEASURED",
    "ARMED": "UNMEASURED",
}


def _check_total(
    name: str,
    domain: Iterable[str],
    mapping: dict[str, str],
    codomain: Iterable[str],
) -> None:
    """Refuse a source state with no declared board rendering.

    A plain function rather than a module-level `assert`, for the reason
    `crucible.gate._check_ladder_console_coverage` gives: `assert` is compiled
    out under `python -O`, which makes it the one guard construct guaranteed
    absent in an optimized interpreter — and this is the guard that stops a
    state reaching a surface with no declared rendering
    (`alpha-engine-config-I9826`).
    """
    gap = set(domain) - set(mapping)
    if gap:
        raise ValueError(
            f"{name} is missing {sorted(gap)} — every source state must declare how "
            "it renders on the board before it can reach a surface."
        )
    stray = set(mapping.values()) - set(codomain)
    if stray:
        raise ValueError(
            f"{name} maps to {sorted(stray)}, which are outside its declared codomain "
            f"{sorted(codomain)}. Both directions are checked: a mapping that is total "
            "over its inputs can still land on a state nothing renders."
        )


_check_total("BOARD_CONSOLE_STATE", BOARD_STATES, BOARD_CONSOLE_STATE, COMPONENT_STATES)
_check_total("LADDER_BOARD_STATE", LADDER_STATES, LADDER_BOARD_STATE, BOARD_STATES)
_check_total("COMPONENT_BOARD_STATE", COMPONENT_STATES, COMPONENT_BOARD_STATE, BOARD_STATES)

if RED_STATES | GREY_STATES | {"MET"} != set(BOARD_STATES):  # pragma: no cover - import guard
    raise ValueError(
        "every board state must be red, grey, or MET. A state in none of the three "
        "is one nobody has decided how to count, and it will be counted as progress."
    )


# ── Declarations ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Declaration:
    """One declared row, before anything has been read for it.

    ``reads`` is `None` exactly when the row is `PLANNED`. That is the whole
    encoding of "declared, not built", and it is a declaration rather than an
    inference on purpose: an absent artifact would otherwise be
    indistinguishable from a producer that broke.
    """

    id: str
    source: str
    title: str
    surface: str
    reads: str | None
    verdict: str | None
    means_when_red: str
    section: str = ""
    clause_class: str = ""
    planned_because: str = ""

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ValueError(f"{self.id}: source {self.source!r} not in {SOURCES}")
        if (self.reads is None) != (self.verdict is None):
            raise ValueError(
                f"{self.id}: `reads` and `verdict` must be set together. A key with no "
                "field to read cannot produce a state, and a field with no key has "
                "nowhere to come from."
            )
        if self.reads is None and not self.planned_because.strip():
            raise ValueError(
                f"{self.id} is PLANNED with no `planned_because`. A grey row with no "
                "stated reason is indistinguishable from one somebody forgot to wire, "
                "and grey rows are the ones nobody chases."
            )
        if self.reads is not None and self.planned_because.strip():
            raise ValueError(
                f"{self.id} declares both `reads` and `planned_because` — it cannot be "
                "both measured and not built."
            )
        if not self.means_when_red.strip():
            raise ValueError(
                f"{self.id} declares no `means_when_red`. A red row a reader has to ask "
                "an agent about is not actionable, which is the whole objection to the "
                "board this one replaces."
            )


@dataclass(frozen=True)
class Declarations:
    """`board.yaml`, parsed. Two of the board's four sources."""

    objectives: dict[str, Declaration]
    cutover: dict[str, Declaration]

    @property
    def all(self) -> dict[str, Declaration]:
        return {**self.objectives, **self.cutover}


def _declaration(row_id: str, source: str, body: dict[str, Any]) -> Declaration:
    known = {
        "section",
        "statement",
        "clause_class",
        "surface",
        "reads",
        "verdict",
        "planned_because",
        "means_when_red",
    }
    unknown = set(body) - known
    if unknown:
        raise ValueError(
            f"{row_id}: unknown field(s) {sorted(unknown)}. The declaration schema is "
            "closed — a typo'd key would otherwise be silently ignored, which on a "
            "board means a row measuring nothing while looking configured."
        )
    missing = {"statement", "surface", "means_when_red"} - set(body)
    if missing:
        raise ValueError(f"{row_id}: missing required field(s) {sorted(missing)}")
    return Declaration(
        id=row_id,
        source=source,
        title=str(body["statement"]).strip(),
        surface=str(body["surface"]),
        reads=body.get("reads"),
        verdict=body.get("verdict"),
        means_when_red=str(body.get("means_when_red", "")),
        section=str(body.get("section", "")),
        clause_class=str(body.get("clause_class", "")),
        planned_because=str(body.get("planned_because", "")),
    )


@lru_cache(maxsize=1)
def load_declarations(path: Path | None = None) -> Declarations:
    document = yaml.safe_load((path or DECLARATION_PATH).read_text())
    if document.get("version") != 1:
        raise ValueError(f"board.yaml version {document.get('version')!r} is not 1")
    objectives = {
        row_id: _declaration(row_id, "objective", body)
        for row_id, body in (document.get("objectives") or {}).items()
    }
    cutover = {
        row_id: _declaration(row_id, "cutover", body)
        for row_id, body in (document.get("cutover") or {}).items()
    }
    if not objectives:
        raise ValueError("board.yaml declares no objectives — the board would be blind")
    if not cutover:
        raise ValueError(
            "board.yaml declares no cutover predicates. The v1 -> v2 cutover is four "
            "individually readable facts; inferring it from a phase state is how a "
            "milestone gets declared from a merge."
        )
    overlap = set(objectives) & set(cutover)
    if overlap:
        raise ValueError(f"row id(s) {sorted(overlap)} declared in two sections")
    return Declarations(objectives=objectives, cutover=cutover)


# ── Reading ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Reading:
    """What the store said about one declared row.

    ``state`` is already a board state. ``detail`` says how it knows, because
    a dot that cannot say how it knows is not trustworthy
    (`crucible.console.classify.Classification`, same posture).
    """

    state: str
    detail: str
    last_read: str | None = None


def read_verdict(store: Store, declaration: Declaration) -> Reading:
    """Resolve one declared row against the store. Reads; never runs.

    The four outcomes are deliberately distinct, and three of them are red:

    * key absent            -> `UNMEASURED`, key named.
    * key unreadable        -> `UNMEASURABLE`, error named. A credential
      failure and an unmet objective are different facts, and reporting them
      identically is `alpha-engine-config-I9828`.
    * field absent          -> `UNMEASURABLE`, field named.
    * field present         -> `MET` / `UNMET`.
    """
    if declaration.reads is None:
        return Reading("PLANNED", declaration.planned_because.strip())

    key = declaration.reads
    try:
        present = store.exists(key)
    except Exception as exc:  # noqa: BLE001 - the error IS the reading
        return Reading(
            "UNMEASURABLE",
            f"could not establish whether {key} exists: {type(exc).__name__}: {exc}. "
            "This is not the same as the objective being unmet — it is a statement "
            "about our access, and it must never be counted as one about the system.",
        )
    if not present:
        return Reading("UNMEASURED", f"no artifact at {key} — nothing has filed a reading yet")

    try:
        document = json.loads(store.get_bytes(key))
    except Exception as exc:  # noqa: BLE001 - the error IS the reading
        return Reading(
            "UNMEASURABLE",
            f"{key} is present but could not be read: {type(exc).__name__}: {exc}",
        )
    if not isinstance(document, dict):
        return Reading(
            "UNMEASURABLE",
            f"{key} parsed to {type(document).__name__}, not an object with fields",
        )

    field_name = declaration.verdict
    assert field_name is not None  # Declaration.__post_init__ guarantees the pair
    if field_name not in document:
        return Reading(
            "UNMEASURABLE",
            f"{key} carries no field {field_name!r}. The document exists and does not "
            "answer the question, which is not the same as answering it 'no'.",
            last_read=_as_text(document.get("generated_at")),
        )
    value = document[field_name]
    if not isinstance(value, bool):
        return Reading(
            "UNMEASURABLE",
            f"{key}:{field_name} is {type(value).__name__} {value!r}, not a boolean. A "
            "verdict field that is not a verdict cannot be rendered either way.",
            last_read=_as_text(document.get("generated_at")),
        )
    return Reading(
        "MET" if value else "UNMET",
        f"{key}:{field_name} is {value}",
        last_read=_as_text(document.get("generated_at")),
    )


def _as_text(value: Any) -> str | None:
    return str(value) if isinstance(value, str) and value.strip() else None


# ── Rows and the board ────────────────────────────────────────────────────


@dataclass(frozen=True)
class BoardRow:
    """One row. Declared before it could be read, and named by a stable id.

    The id scheme is `<source>:<declared id>` and is a PUBLIC CONTRACT from
    the first commit — the console addresses rows by it, the digest reports
    deltas by it, and a renamed row is indistinguishable from a vanished one
    plus a new one to anything downstream.
    """

    id: str
    source: str
    title: str
    state: str
    detail: str
    surface: str
    artifact: str
    means_when_red: str
    section: str = ""
    last_read: str | None = None

    def __post_init__(self) -> None:
        if self.state not in BOARD_STATES:
            raise ValueError(f"{self.id}: {self.state!r} is not a board state")
        if not self.detail.strip():
            raise ValueError(f"{self.id} carries no detail — a dot that cannot say how")
        if not self.artifact.strip():
            raise ValueError(
                f"{self.id} names no artifact. Every row names the thing it reads — an "
                "S3 key, a log location or a registry path — so a red row is "
                "actionable without asking an agent what it means."
            )

    @property
    def console_state(self) -> str:
        return BOARD_CONSOLE_STATE[self.state]

    @property
    def red(self) -> bool:
        return self.state in RED_STATES

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "section": self.section,
            "title": self.title,
            "state": self.state,
            "console_state": self.console_state,
            "detail": self.detail,
            "surface": self.surface,
            "artifact": self.artifact,
            "means_when_red": self.means_when_red,
            "last_read": self.last_read,
        }


@dataclass
class Board:
    """Every declared row, in every source, with its reading."""

    trading_day: str
    generated_at: str
    rows: list[BoardRow] = field(default_factory=list)

    @property
    def red(self) -> list[BoardRow]:
        return [r for r in self.rows if r.red]

    @property
    def grey(self) -> list[BoardRow]:
        return [r for r in self.rows if r.state in GREY_STATES]

    @property
    def met(self) -> list[BoardRow]:
        return [r for r in self.rows if r.state == "MET"]

    def counts(self) -> dict[str, int]:
        """Every state, including the zeroes.

        Zeroes are emitted on purpose: a state missing from the counts reads
        as "not applicable" and is indistinguishable from a state the producer
        stopped being able to compute.
        """
        return {state: sum(1 for r in self.rows if r.state == state) for state in BOARD_STATES}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": BOARD_SCHEMA_VERSION,
            "trading_day": self.trading_day,
            "generated_at": self.generated_at,
            "counts": self.counts(),
            # The headline. Not "how many are green" — a board that leads with
            # its green count on day one leads with zero and gets closed.
            "red_count": len(self.red),
            "row_count": len(self.rows),
            "rows": [r.to_dict() for r in self.rows],
        }


def build_board(
    store: Store,
    *,
    now: dt.datetime | None = None,
    trading_day: dt.date | None = None,
    registry: dict[str, Component] | None = None,
    classifications: dict[str, Classification] | None = None,
    ladder: Ladder | None = None,
    declarations: Declarations | None = None,
) -> Board:
    """Assemble every declared row and read each one. Reads; never runs.

    ``classifications`` and ``ladder`` are supplied by the caller rather than
    computed here, because both are already produced by modules that own them
    (`crucible.console.render`, `crucible.gate.build_ladder`) and a second
    computation of either would put two surfaces out of step. Passing `None`
    for either is not "skip it": those rows still appear, `UNMEASURED`, with
    the reason stated.
    """
    moment = (now or dt.datetime.now(dt.UTC)).astimezone(dt.UTC)
    # The RUN's trading day when the caller has one, never the wall clock.
    # `crucible board --trading-day 2026-08-28` re-rendering under today's
    # date would file the board at a key that disagrees with the manifest
    # beside it, and §4.12 makes the trading day THE key — a replay that
    # landed under the wrong one is indistinguishable from a duplicate.
    day = (trading_day or resolve_trading_day(moment)).isoformat()
    decl = declarations if declarations is not None else load_declarations()
    reg = registry if registry is not None else load_registry()

    rows: list[BoardRow] = []

    for row_id, declaration in decl.objectives.items():
        rows.append(_declared_row(store, f"objective:{row_id}", declaration))

    rows.extend(_phase_rows(ladder))

    for name in sorted(reg):
        rows.append(_component_row(reg[name], (classifications or {}).get(name)))

    for row_id, declaration in decl.cutover.items():
        rows.append(_declared_row(store, f"cutover:{row_id}", declaration))

    return Board(
        trading_day=day,
        generated_at=moment.strftime("%Y-%m-%dT%H:%M:%SZ"),
        rows=rows,
    )


def _declared_row(store: Store, row_id: str, declaration: Declaration) -> BoardRow:
    reading = read_verdict(store, declaration)
    return BoardRow(
        id=row_id,
        source=declaration.source,
        section=declaration.section,
        title=declaration.title,
        state=reading.state,
        detail=reading.detail,
        surface=declaration.surface,
        artifact=declaration.reads or "(no producer yet)",
        means_when_red=declaration.means_when_red,
        last_read=reading.last_read,
    )


def _phase_rows(ladder: Ladder | None) -> list[BoardRow]:
    """One row per §6 phase, from `gate.PHASES` — never from a second list."""
    from crucible.gate import PHASES  # noqa: PLC0415 - avoids a module import cycle

    if ladder is None:
        return [
            BoardRow(
                id=f"phase:{phase.id}",
                source="phase",
                section=f"§6 phase {phase.number}",
                title=phase.title,
                state="UNMEASURED",
                detail=(
                    "no ladder was supplied to this render, so no gate was read. This "
                    "is a statement about the producer, not about the phase."
                ),
                surface="crucible/board",
                artifact=f"gates/{phase.gate or phase.id}/{{trading_day}}/gate.json",
                means_when_red=(
                    f"phase {phase.number}'s exit gate is not met. Tracker: "
                    f"{phase.tracker} ({phase.tracker_url})."
                ),
            )
            for phase in PHASES
        ]

    by_id = {row.phase.id: row for row in ladder.rows}
    missing = {phase.id for phase in PHASES} - set(by_id)
    if missing:
        raise ValueError(
            f"the ladder is missing phase row(s) {sorted(missing)} declared in "
            "gate.PHASES. A board that silently drops a declared row is the defect "
            "this board exists to make impossible."
        )
    rows = []
    for phase in PHASES:
        ladder_row = by_id[phase.id]
        rows.append(
            BoardRow(
                id=f"phase:{phase.id}",
                source="phase",
                section=f"§6 phase {phase.number}",
                title=phase.title,
                state=LADDER_BOARD_STATE[ladder_row.state],
                detail=_ladder_detail(ladder_row),
                surface="crucible/board",
                artifact=f"gates/{phase.gate or phase.id}/{{trading_day}}/gate.json",
                means_when_red=(
                    f"phase {phase.number}'s exit gate is not met. Tracker: "
                    f"{phase.tracker} ({phase.tracker_url})."
                ),
                last_read=ladder_row.read_on,
            )
        )
    return rows


def _ladder_detail(ladder_row: PhaseRow) -> str:
    """The ladder's own words, or a statement of what it could not say.

    `PhaseRow.detail` is required to be populated by `build_ladder`; the
    fallback exists so an empty one renders as a named gap rather than as a
    row with a blank explanation, which `BoardRow.__post_init__` would reject
    outright and take the whole board down over one row.
    """
    if ladder_row.detail.strip():
        return ladder_row.detail
    return (
        f"the ladder placed this phase {ladder_row.state} (gate {ladder_row.gate_state}) "
        "and supplied no detail — the state is usable, the reason is missing"
    )


def _component_row(component: Component, classification: Classification | None) -> BoardRow:
    if classification is None:
        state, detail = (
            "UNMEASURED",
            "no classification was supplied to this render, so this component was not "
            "read. A component nobody read is unobserved, not healthy.",
        )
        day = None
    else:
        state = COMPONENT_BOARD_STATE[classification.state]
        detail = f"{classification.state}: {classification.reason}"
        day = classification.trading_day
    return BoardRow(
        id=f"component:{component.name}",
        source="component",
        section="§9.2 registry",
        title=component.description,
        state=state,
        detail=detail,
        surface=component.console_surface,
        artifact=component.log_location,
        means_when_red=(
            f"{component.name} is not reporting healthy. Its absence is watched by "
            f"{component.absence_watched_by}; alerts reach {component.alert_channel}."
        ),
        last_read=day,
    )


def board_payload(board: Board) -> bytes:
    return json.dumps(board.to_dict(), indent=2, sort_keys=True).encode() + b"\n"


# ── The digest: deltas, not absolute state ────────────────────────────────


@dataclass(frozen=True)
class RowDelta:
    """One row that changed between two boards.

    ``was`` is `None` for a row that APPEARED and ``now`` is `None` for one
    that VANISHED. Both are reported, and a vanished row is a finding rather
    than a quiet omission: the row set is derived from declarations, so a row
    disappearing means a declaration disappeared.
    """

    id: str
    was: str | None
    now: str | None

    @property
    def kind(self) -> str:
        if self.was is None:
            return "appeared"
        if self.now is None:
            return "vanished"
        if self.now == "MET":
            return "earned"
        if self.was == "MET":
            return "lost"
        return "moved"

    def describe(self) -> str:
        if self.was is None:
            return f"{self.id}: APPEARED as {self.now}"
        if self.now is None:
            return (
                f"{self.id}: VANISHED (was {self.was}) — a declared row stopped being "
                "declared, which is a change to the plan, not to the system"
            )
        return f"{self.id}: {self.was} -> {self.now}"


def board_delta(previous: dict[str, Any] | None, current: Board) -> list[RowDelta]:
    """What changed since the last board.

    **The digest reports deltas, not absolute state.** A board reading almost
    entirely `PLANNED` for weeks is correct, and is also the thing people stop
    opening; the delta is the part that stays worth reading. The absolute
    board is one click away and does not need to be retold daily.

    A `previous` of `None` — the first ever run — yields no deltas rather than
    a full board's worth of `appeared` rows, which would be noise on the one
    day the board itself is the news.
    """
    if previous is None:
        return []
    before = {row["id"]: row["state"] for row in previous.get("rows", [])}
    after = {row.id: row.state for row in current.rows}
    deltas = [
        RowDelta(row_id, before.get(row_id), after.get(row_id))
        for row_id in sorted(set(before) | set(after))
        if before.get(row_id) != after.get(row_id)
    ]
    return deltas


# ── The page ──────────────────────────────────────────────────────────────

#: Every state gets its own colour. Uniform red is the failure mode named on
#: the issue: if "not built yet", "built and failing" and "cannot be measured"
#: render identically, the reader stops distinguishing them and then stops
#: looking. `UNMEASURABLE` is deliberately the loudest thing on the page —
#: it is the only state that is about US rather than about the system.
_SWATCH: dict[str, str] = {
    "MET": "#1a7f37",
    "UNMET": "#bf8700",
    "UNMEASURED": "#cf222e",
    "UNMEASURABLE": "#8250df",
    "OUT_OF_ORDER": "#a40e26",
    "PLANNED": "#6e7781",
    "DECLARED_OFF": "#8c959f",
}

_check_total("_SWATCH", BOARD_STATES, {k: k for k in _SWATCH}, BOARD_STATES)


def render_board_html(board: Board, deltas: list[RowDelta] | None = None) -> str:
    """One page, every declared row, grouped by source.

    The headline is the RED count, not the green one. A board that leads with
    "3 of 40 met" on day one leads with a number that looks like failure and
    is in fact the plan; a board that leads with what is not yet measurable
    leads with the thing that can be acted on.
    """
    counts = board.counts()
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<title>crucible v2 — the declared board</title>",
        "<style>",
        "body{font:14px/1.5 system-ui,sans-serif;margin:2rem;max-width:70rem}",
        "table{border-collapse:collapse;width:100%;margin:0 0 2rem}",
        "th,td{text-align:left;padding:.4rem .6rem;border-bottom:1px solid #d0d7de;"
        "vertical-align:top}",
        "th{font-weight:600;background:#f6f8fa}",
        ".s{font-weight:700;white-space:nowrap}",
        ".k{font-family:ui-monospace,monospace;font-size:12px;color:#57606a}",
        ".d{color:#57606a}",
        "</style></head><body>",
        "<h1>crucible v2 — the declared board</h1>",
        f"<p><strong>{len(board.red)} of {len(board.rows)} rows red.</strong> "
        f"{len(board.grey)} declared and not built; {len(board.met)} met. "
        f"Trading day {_esc(board.trading_day)}, rendered {_esc(board.generated_at)}.</p>",
        "<p class='d'>Every row below was declared before the thing it measures "
        "existed. Red is the correct day-one reading; what matters is which rows "
        "move, and in which direction. <strong>UNMEASURABLE</strong> is not a "
        "weaker UNMET — it means the reading could not be taken at all, which is "
        "a statement about our access rather than about the system.</p>",
        "<p>"
        + " · ".join(
            f"<span class='s' style='color:{_SWATCH[state]}'>{state}</span> {counts[state]}"
            for state in BOARD_STATES
        )
        + "</p>",
    ]

    if deltas:
        parts.append("<h2>Since the last board</h2><ul>")
        parts.extend(f"<li>{_esc(delta.describe())}</li>" for delta in deltas)
        parts.append("</ul>")
    elif deltas is not None:
        parts.append("<h2>Since the last board</h2><p class='d'>No row changed state.</p>")

    for source in SOURCES:
        rows = [r for r in board.rows if r.source == source]
        if not rows:
            continue
        parts.append(f"<h2>{_esc(source)} ({len(rows)})</h2>")
        parts.append(
            "<table><tr><th>state</th><th>row</th><th>how it knows</th>"
            "<th>artifact</th><th>what red means</th></tr>"
        )
        for row in rows:
            parts.append(
                f"<tr><td class='s' style='color:{_SWATCH[row.state]}'>{row.state}</td>"
                f"<td>{_esc(row.title)}<br><span class='k'>{_esc(row.id)}</span></td>"
                f"<td class='d'>{_esc(row.detail)}</td>"
                f"<td class='k'>{_esc(row.artifact)}</td>"
                f"<td class='d'>{_esc(row.means_when_red)}</td></tr>"
            )
        parts.append("</table>")

    parts.append("</body></html>")
    return "".join(parts)


def _esc(value: Any) -> str:
    import html  # noqa: PLC0415 - one call site, kept local to the renderer

    return html.escape("" if value is None else str(value))
