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

**Rows are derived from five declarations, never hand-listed.** A hand-kept
board drifts from what the plan promises, which is the same failure one level
up. :func:`build_board` reads:

* `crucible/board.yaml` — the plan §2 objective table and the cutover
  predicates.
* :data:`crucible.gate.PHASES` and :data:`crucible.gate.GATES` — the §6 ladder.
* `crucible/components.yaml` — the observability registry.
* :data:`crucible.schedule.MILESTONES` — the plan §6.1 milestone table
  (`alpha-engine-config-I9914`). Informational: a milestone row is never a
  page condition and never a gate input (§6.1: "weeks are sequencing, not
  commitments"); it renders the SAME ladder reading `crucible.gate` already
  produced for its phase, dated against the calendar the plan committed to.

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
from collections.abc import Callable, Iterable
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
    "board_payload",
    "pointer_may_move",
    "render_board_html",
    "build_board",
    "load_declarations",
    "read_acceptance",
    "READERS",
    "READER_ARTIFACTS",
    "read_declaration",
]

BOARD_SCHEMA_VERSION = "board.v1"

DECLARATION_PATH = Path(__file__).parent / "board.yaml"

#: Where a day's board is filed, and the pointer the console reads. Keyed by
#: trading day like everything else (§4.12).

#: The five declarations a row can come from. Closed: a sixth source is a
#: design change visible in a diff, not a new dict key someone adds. `schedule`
#: was added deliberately on `alpha-engine-config-I9914` — the plan §6.1
#: milestone table — rather than folded into `phase`, because a milestone is
#: a DATED reflection of a phase's ladder reading, not the reading itself, and
#: the two must be able to disagree (a milestone can be UNMET past its date
#: while the phase it names is later read MET) without one silently standing
#: in for the other.
SOURCES: tuple[str, ...] = ("objective", "phase", "schedule", "component", "cutover")

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
    # FAILED, not UNREPORTED. `UNMEASURED` and `UNMEASURABLE` are different
    # facts and this board argues that harder than it argues anything else —
    # so collapsing them onto one console state would lose the distinction on
    # the FLEET surface, which is the surface actually opened. `UNREPORTED` is
    # "nobody has read this"; a read that was ATTEMPTED and could not complete
    # is a fault in our access, and §8.3's word for a fault is FAILED.
    "UNMEASURABLE": "FAILED",
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
    # Added when the ladder gained the state (`alpha-engine-config-I9869`
    # round 3, finding 4) — the board already had this word (`BOARD_STATES`
    # above), the ladder did not, and this map's job is precisely to refuse
    # a ladder state with no declared board rendering.
    "UNMEASURABLE": "UNMEASURABLE",
    "OUT_OF_ORDER": "OUT_OF_ORDER",
}

#: `crucible.console.classify.STATES` -> board states. Total over all
#: fourteen.
#:
#: Two mappings are worth arguing rather than reading past:
#:
#: * `MISSED` maps to `UNMEASURED`, not `UNMET`: a schedule that fired with
#:   no run produced no reading, and "we did not measure" is not "we measured
#:   and it fell short". `ABSENT` stays `UNMET` because it IS a reading — a
#:   declared component established as not present.
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
    # `MISSED` is "its schedule fired and no run started" — no reading was
    # taken at all, which is this board's definition of UNMEASURED, not of a
    # measured shortfall. `ABSENT` below stays UNMET on purpose and the
    # difference is worth stating: ABSENT is a positive finding that a
    # DECLARED thing is not there, which is itself a reading.
    "MISSED": "UNMEASURED",
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
    """Refuse a mapping that is not total, in EITHER direction, over a non-empty domain.

    A plain function rather than a module-level `assert`, for the reason
    `crucible.gate._check_ladder_console_coverage` gives: `assert` is compiled
    out under `python -O`, which makes it the one guard construct guaranteed
    absent in an optimized interpreter, and this is the guard that stops a
    state reaching a surface with no declared rendering
    (`alpha-engine-config-I9826`).

    Three checks, and the first two were each missing once:

    * **non-empty domain** — an empty domain makes every other assertion
      vacuously true. A guard that passes over nothing is the failure this
      repository grades other systems on, so it is refused rather than trusted
      to be impossible.
    * **domain ⊆ keys** — every source state has a rendering.
    * **keys ⊆ domain** — no rendering exists for a state that no longer
      exists. Dead data today; a renamed state tomorrow, still mapped under
      its old name, silently unrendered under the new one.
    * **values ⊆ codomain** — a mapping total over its inputs can still land
      on a state nothing renders.
    """
    states, targets = set(domain), set(codomain)
    if not states:
        raise ValueError(
            f"{name} was checked against an EMPTY domain, which makes every assertion "
            "about it vacuously true. A guard that passes over nothing is not a guard."
        )
    missing = states - set(mapping)
    if missing:
        raise ValueError(
            f"{name} is missing {sorted(missing)} — every source state must declare how "
            "it renders on the board before it can reach a surface."
        )
    orphaned = set(mapping) - states
    if orphaned:
        raise ValueError(
            f"{name} maps {sorted(orphaned)}, which are not source states. A rendering "
            "for a state that no longer exists is how a renamed state ends up unrendered."
        )
    stray = set(mapping.values()) - targets
    if stray:
        raise ValueError(
            f"{name} maps to {sorted(stray)}, which are outside its declared codomain "
            f"{sorted(targets)}."
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

    ``reader`` is `None` exactly when the row is `PLANNED`. That is the whole
    encoding of "declared, not built", and it is a declaration rather than an
    inference on purpose: an absent artifact would otherwise be
    indistinguishable from a producer that broke.

    ``artifact`` names the key a reader resolves, for the READER OF THE BOARD.
    It is asserted equal to the key the registered reader actually derives, so
    it cannot drift into decoration — and it is stated even for a `PLANNED`
    row, because "here is the objective and here is the artifact nobody writes
    yet" is actionable and a bare grey dot is not.
    """

    id: str
    source: str
    title: str
    surface: str
    reader: str | None
    artifact: str
    means_when_red: str
    section: str = ""
    clause_class: str = ""
    planned_because: str = ""

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ValueError(f"{self.id}: source {self.source!r} not in {SOURCES}")
        if not self.artifact.strip():
            raise ValueError(
                f"{self.id} names no artifact. Every row names the thing it reads or is "
                "waiting for — a red row a reader has to ask an agent about is not "
                "actionable, which is the whole objection to the surface this replaces."
            )
        if self.reader is not None and self.reader not in READERS:
            raise ValueError(
                f"{self.id}: reader {self.reader!r} is not registered in board.READERS "
                f"({sorted(READERS)}). A declaration naming a reader that does not exist "
                "would render UNMEASURABLE forever for a reason about US, not about the "
                "objective."
            )
        if self.reader is None and not self.planned_because.strip():
            raise ValueError(
                f"{self.id} is PLANNED with no `planned_because`. A grey row with no "
                "stated reason is indistinguishable from one somebody forgot to wire, "
                "and grey rows are the ones nobody chases."
            )
        if self.reader is not None and self.planned_because.strip():
            raise ValueError(
                f"{self.id} declares both a reader and `planned_because` — it cannot be "
                "both measured and not built."
            )
        if not self.means_when_red.strip():
            raise ValueError(f"{self.id} declares no `means_when_red`")


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
        "reader",
        "artifact",
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
    missing = {"statement", "surface", "artifact", "means_when_red"} - set(body)
    if missing:
        raise ValueError(f"{row_id}: missing required field(s) {sorted(missing)}")
    return Declaration(
        id=row_id,
        source=source,
        title=str(body["statement"]).strip(),
        surface=str(body["surface"]),
        reader=body.get("reader"),
        artifact=str(body["artifact"]).strip(),
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


def _fetch(
    store: Store, key: str, trading_day: str
) -> tuple[dict[str, Any] | None, Reading | None]:
    """Read one day-keyed JSON document. Returns `(document, None)` or `(None, reading)`.

    Every failure mode is a DIFFERENT red, and three of them are about us
    rather than about the system:

    * key absent           -> `UNMEASURED`, key named.
    * key unreadable       -> `UNMEASURABLE`, error named. A credential failure
      and an unmet objective reported identically is
      `alpha-engine-config-I9828`, whose cost clause has been unreadable in CI
      for its whole life for exactly this reason.
    * not a JSON object    -> `UNMEASURABLE`, type named.
    * document's own day disagrees with the board's -> `UNMEASURED`, both days
      named. The key is day-scoped so this should be impossible; it is checked
      anyway, because "the artifact says it is about a different day" is the
      one thing that would make a stale document read as a current reading,
      and `crucible.gate.last_read` already records that a freshly written
      artifact full of never-measured content otherwise looks entirely fresh.
    """
    try:
        present = store.exists(key)
    except Exception as exc:  # noqa: BLE001 - the error IS the reading
        return None, Reading(
            "UNMEASURABLE",
            f"could not establish whether {key} exists: {type(exc).__name__}: {exc}. This "
            "is not the same as the objective being unmet — it is a statement about our "
            "access, and it must never be counted as one about the system.",
        )
    if not present:
        return None, Reading(
            "UNMEASURED", f"no artifact at {key} — nothing has filed a reading for this day"
        )
    try:
        document = json.loads(store.get_bytes(key))
    except Exception as exc:  # noqa: BLE001 - the error IS the reading
        return None, Reading(
            "UNMEASURABLE", f"{key} is present but could not be read: {type(exc).__name__}: {exc}"
        )
    if not isinstance(document, dict):
        return None, Reading(
            "UNMEASURABLE", f"{key} parsed to {type(document).__name__}, not an object with fields"
        )
    stamped = document.get("trading_day")
    if stamped is not None and stamped != trading_day:
        return None, Reading(
            "UNMEASURED",
            f"{key} carries trading_day {stamped!r}, not {trading_day!r}. A document about "
            "another day is not a reading for this one, however recently it was written.",
            last_read=str(stamped),
        )
    return document, None


def _provenance(document: dict[str, Any]) -> str | None:
    for field_name in ("generated_utc", "generated_at", "trading_day"):
        value = document.get(field_name)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _read_attribution(store: Store, trading_day: str) -> Reading:
    """Plan §2 row 5 — the five-row attribution table, read from the report card.

    Key AND vocabulary from `crucible.report`, which is the module that WRITES
    it. Not restated here — the first version of this reader filtered on
    ``status == "UNREPORTED"``, a token `report.py` never emits: attribution
    rows are statused by `krepis.metrics.derive_status`, whose
    not-measured vocabulary is the `N/A-*` family, and `report.py:278` itself
    filters on ``startswith("N/A")``. So a table of five blank rows read MET,
    with a detail asserting "with a value each" over five nulls — the exact
    failure the previous paragraph of this docstring claimed to prevent, and
    the exact bug class of a test that fabricates the keys its code invented.

    MET therefore requires all three: the declared number of rows, no row in
    the `N/A-*` family, and no row whose `value` is null. A row count is not a
    reading, and `build_attribution` already refuses to emit the wrong number
    of rows — so counting them grades the one thing that cannot go wrong.
    """
    from crucible.report import ROWS, attribution_key  # noqa: PLC0415 - avoids a cycle

    key = attribution_key(trading_day)
    document, failure = _fetch(store, key, trading_day)
    if failure is not None:
        return failure
    assert document is not None
    rows = document.get("rows")
    if not isinstance(rows, list):
        return Reading(
            "UNMEASURABLE",
            f"{key} carries no `rows` list — the document exists and does not answer the "
            "question, which is not the same as answering it 'no'.",
            last_read=_provenance(document),
        )
    provenance = _provenance(document)
    if len(rows) != len(ROWS):
        return Reading(
            "UNMET",
            f"{key} grades {len(rows)} of {len(ROWS)} declared layer(s)",
            last_read=provenance,
        )
    blind = [_row_name(r) for r in rows if _row_measured_nothing(r)]
    if blind:
        return Reading(
            "UNMET",
            f"{key} grades all {len(rows)} layers and {len(blind)} of them measured "
            f"nothing ({', '.join(blind)}). A row that measured nothing is not evidence "
            "of health, whatever the table's own row count says.",
            last_read=provenance,
        )
    return Reading(
        "MET",
        f"{key} grades all {len(rows)} declared layers, each with a measured value",
        provenance,
    )


def read_acceptance(store: Store, trading_day: str) -> tuple[dict[str, Any] | None, str]:
    """The §2 acceptance reading for ``trading_day``, or `None` and why not.

    Routed through :func:`_fetch` so an absent artifact, an unreadable one and
    a credential failure are three different sentences on the page rather
    than one silent gap — the same distinction `crucible.morning` makes for
    the same artifact, made once here so the page and the message cannot
    disagree about what they could read.
    """
    from crucible.keys import acceptance_reading_key  # noqa: PLC0415 - avoids a cycle

    key = acceptance_reading_key(trading_day)
    document, failure = _fetch(store, key, trading_day)
    if failure is not None:
        return None, failure.detail
    return document, ""


def _row_name(row: Any) -> str:
    return str(row.get("name", "?")) if isinstance(row, dict) else "?"


def _row_measured_nothing(row: Any) -> bool:
    """Whether one attribution row carries no measurement.

    Two conditions, because either alone is beatable. `report.py` writes the
    `N/A-*` status family through `krepis.metrics.derive_status` for a row it
    could not measure — but a row can also carry a measured-looking status and
    a null value, and a null value is not a measurement whatever the status
    says. A malformed row (not a dict at all) counts as blind: a row this
    reader cannot parse has not been shown to measure anything.
    """
    if not isinstance(row, dict):
        return True
    if str(row.get("status", "")).startswith("N/A"):
        return True
    return row.get("value") is None


#: The registered readers, by the name a declaration uses.
#:
#: A reader takes `(store, trading_day)` and returns a `Reading`. It derives
#: its own key from the module that OWNS the artifact — never from a literal
#: in `board.yaml`, which is what the first draft of this board did for
#: fourteen keys, eight of them wrong.
#:
#: The registry is deliberately small. Most §2 objectives have no producer at
#: all yet, and those rows are `PLANNED` with the artifact they are waiting for
#: named. Inventing a reader for an artifact nobody writes would produce a red
#: about the KEY rather than about the objective, indistinguishable from the
#: gap this board exists to show.
READERS: dict[str, Callable[[Store, str], Reading]] = {
    "attribution_table_is_complete": _read_attribution,
}

#: What key each reader resolves, as a template, for the board's reader.
#: `test_every_declared_artifact_matches_the_key_its_reader_uses` asserts each
#: entry against the key the reader really derives for a sample trading day,
#: so the human-facing string cannot drift away from the code.
READER_ARTIFACTS: dict[str, str] = {
    "attribution_table_is_complete": "report/{trading_day}/attribution.json",
}


def read_declaration(store: Store, declaration: Declaration, trading_day: str) -> Reading:
    """Resolve one declared row against the store. Reads; never runs."""
    if declaration.reader is None:
        return Reading("PLANNED", declaration.planned_because.strip())
    return READERS[declaration.reader](store, trading_day)


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
    #: The gate clauses behind this row, each `{name, met, requirement,
    #: detail}` — or `None` when this row has no clause list to show.
    #:
    #: `None` and `()` are DIFFERENT facts and are kept apart deliberately.
    #: `None` is "this render was given no gate reading for this row", which
    #: is a statement about the producer; `()` is "the gate was read and
    #: declares no clauses", which is `crucible.gate`'s own UNMEASURED case.
    #: A consumer that could not tell them apart would render "0/0 clauses
    #: met" over a render that never took a reading — the fabricated
    #: measurement this whole board exists to make impossible.
    #:
    #: Carried on the ROW rather than left in the free-text `detail` because
    #: `crucible.morning` has to print the unmet clause NAMES
    #: (`alpha-engine-config-I9921`), and the alternative is parsing them back
    #: out of an English sentence — a contract restated as a regex, which is
    #: the bug class this repository has already paid for twice.
    clauses: tuple[dict[str, Any], ...] | None = None

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
            # `null` for "no reading was taken", `[]` for "read, no clauses".
            # See the field's own comment: collapsing the two would let a
            # consumer print 0/0 over a render that measured nothing.
            "clauses": None if self.clauses is None else [dict(c) for c in self.clauses],
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
    readings: dict[str, Any] | None = None,
) -> Board:
    """Assemble every declared row and read each one. Reads; never runs.

    ``classifications`` and ``ladder`` are supplied by the caller rather than
    computed here, because both are already produced by modules that own them
    (`crucible.console.render`, `crucible.gate.build_ladder`) and a second
    computation of either would put two surfaces out of step. Passing `None`
    for either is not "skip it": those rows still appear, `UNMEASURED`, with
    the reason stated.

    ``readings`` is the same argument one level further in: the
    `{gate: GateResult}` the caller ALREADY evaluated for `build_ladder`,
    handed on so each phase row can carry its clause list
    (:attr:`BoardRow.clauses`, `alpha-engine-config-I9921`) without this
    function evaluating a single gate itself. Re-reading them here would
    double every store read the clause set makes — the defect
    `alpha-engine-config-I9826` fixed in `gate_handler` — and would break
    "a gate reads; it never runs" the moment the two evaluations disagreed.
    `None` leaves every phase row's `clauses` `None`, which renders as "this
    build took no reading" rather than as an empty clause list.
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
        rows.append(_declared_row(store, f"objective:{row_id}", declaration, day))

    rows.extend(_phase_rows(ladder, day, readings or {}))
    rows.extend(_schedule_rows(ladder, day))

    for name in sorted(reg):
        rows.append(_component_row(reg[name], (classifications or {}).get(name)))

    for row_id, declaration in decl.cutover.items():
        rows.append(_declared_row(store, f"cutover:{row_id}", declaration, day))

    return Board(
        trading_day=day,
        generated_at=moment.strftime("%Y-%m-%dT%H:%M:%SZ"),
        rows=rows,
    )


def _declared_row(
    store: Store, row_id: str, declaration: Declaration, trading_day: str
) -> BoardRow:
    reading = read_declaration(store, declaration, trading_day)
    return BoardRow(
        id=row_id,
        source=declaration.source,
        section=declaration.section,
        title=declaration.title,
        state=reading.state,
        detail=reading.detail,
        surface=declaration.surface,
        # The template with the day substituted where there is one to
        # substitute. A literal `{trading_day}` left on the surface is the
        # fleet's documented placeholder gotcha: a key nobody can copy, and
        # indistinguishable from one that was never written.
        artifact=declaration.artifact.replace("{trading_day}", trading_day),
        means_when_red=declaration.means_when_red,
        last_read=reading.last_read,
    )


def _clause_views(reading: Any | None) -> tuple[dict[str, Any], ...] | None:
    """One gate's clauses as plain dicts, or `None` when nothing was read.

    Plain dicts rather than `Clause` objects because this travels onto
    `board/{day}/board.json` and out to two consumers — the page and the
    morning report — and a surface that could only be read by importing
    `crucible.gate` is a contract only this repository can consume.
    """
    if reading is None:
        return None
    return tuple(
        {
            "name": clause.name,
            "met": bool(clause.met),
            "requirement": clause.requirement,
            "detail": clause.detail,
        }
        for clause in reading.clauses
    )


def _phase_rows(
    ladder: Ladder | None,
    trading_day: str,
    readings: dict[str, Any] | None = None,
) -> list[BoardRow]:
    """One row per §6 phase, from `gate.PHASES` — never from a second list."""
    from crucible.gate import PHASES, gate_key  # noqa: PLC0415 - avoids a module import cycle

    supplied = readings or {}

    def artifact(phase: Any) -> str:
        # `phase.gate or phase.id` FABRICATES a key for every unregistered
        # phase: `gate_key` is keyed by the REGISTERED gate name, so
        # `gates/phase2/…` is a path no producer will ever write. Copying it
        # yields nothing, which is indistinguishable from a gate that ran and
        # produced nothing — the exact confusion this board exists to remove.
        if phase.gate is None:
            return "(no gate is registered for this phase)"
        return gate_key(phase.gate, trading_day)

    def means_when_red(phase: Any) -> str:
        if phase.gate is None:
            return (
                f"no clause list is registered for phase {phase.number}, so its gate "
                f"cannot be read at all — it is not unmet, it does not exist. Tracker: "
                f"{phase.tracker} ({phase.tracker_url})."
            )
        return (
            f"phase {phase.number}'s exit gate is not met. Tracker: "
            f"{phase.tracker} ({phase.tracker_url})."
        )

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
                artifact=artifact(phase),
                means_when_red=means_when_red(phase),
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
                artifact=artifact(phase),
                means_when_red=means_when_red(phase),
                last_read=ladder_row.read_on,
                clauses=_clause_views(supplied.get(phase.gate) if phase.gate else None),
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


def _schedule_rows(ladder: Ladder | None, trading_day: str) -> list[BoardRow]:
    """One row per plan §6.1 milestone (`alpha-engine-config-I9914`).

    Each row quotes the ladder reading of the phase the milestone names —
    never a second evaluation of that phase's clauses. Closed three-way state:

    * ``MET``     — the named phase's ladder row already reads `MET`, on or
                    before the milestone's own trading-day anchor.
    * ``UNMET``   — the trading day being rendered is on or past the
                    milestone's anchor and the phase has not read `MET`. The
                    row's own detail begins with `OVERDUE` (`morning.py`
                    repeats that word as the first word of its line for the
                    same row).
    * ``PLANNED`` — the anchor is still ahead. There is no reader to register
                    for a schedule row (`Declaration.reader`, `board.yaml`'s
                    contract) because a milestone is never read from the
                    store on its own account; the row still names the date
                    and the phase/clause it is waiting on, which is the same
                    information `planned_because` carries for a declared row.

    A milestone naming a phase absent from the supplied ladder — or no ladder
    at all — renders `UNMEASURED`, exactly like `_phase_rows` does for the
    same absence: a row this build could not read is red for a reason about
    the RENDER, never silently promoted to `PLANNED` or `MET`.
    """
    from crucible.gate import PHASES, gate_key  # noqa: PLC0415 - avoids a module import cycle
    from crucible.schedule import MILESTONES

    phases_by_id = {phase.id: phase for phase in PHASES}
    by_id = {row.phase.id: row for row in ladder.rows} if ladder is not None else {}
    today = dt.date.fromisoformat(trading_day)

    rows: list[BoardRow] = []
    for milestone in MILESTONES:
        phase = phases_by_id.get(milestone.phase_id)
        if phase is None:  # pragma: no cover - guarded by test_schedule.py
            raise ValueError(
                f"schedule milestone {milestone.id!r} names phase {milestone.phase_id!r}, "
                "which is not in crucible.gate.PHASES. A milestone naming a phase that "
                "does not exist would render forever with no way to ever read MET."
            )
        phase_row = by_id.get(milestone.phase_id)
        due = milestone.trading_day.isoformat()

        if phase_row is None:
            state: str | None = "UNMEASURED"
            quote = f"{phase.id} UNMEASURED — no ladder was supplied to this render"
        elif phase_row.gate_state == "UNMEASURED":
            # The ladder read this phase, but the reading itself is
            # UNMEASURED (no registered gate yet) — a breach can only be
            # derived from a reading that exists, so this renders UNMEASURED
            # regardless of the date, never a date-driven UNMET.
            state = "UNMEASURED"
            quote = f"waiting on {phase.id}, which has no reading"
        else:
            quote = (
                f"{phase.id} {phase_row.clauses_met}/{phase_row.clauses_total}"
                if phase_row.clauses_total
                else f"{phase.id} {phase_row.gate_state}"
            )
            state = "MET" if phase_row.gate_state == "MET" else None

        if state is None:
            # Strictly PAST the anchor: the anchor day itself is the session
            # the milestone measures (rule 3), so the milestone is not yet
            # overdue on that day — only once a later trading day is reached.
            state = "UNMET" if today > milestone.trading_day else "PLANNED"

        plan_date = milestone.plan_date.isoformat()
        if state == "MET":
            detail = f"met — due {due} (plan {plan_date}), reads: {quote}"
        elif state == "UNMET":
            detail = f"OVERDUE since {due} (plan {plan_date}) — reads: {quote}"
        elif state == "UNMEASURED":
            detail = f"due {due} (plan {plan_date}) — {quote}"
        else:  # PLANNED
            detail = f"due {due} (plan {plan_date}), waiting on {quote}"

        rows.append(
            BoardRow(
                id=f"schedule:{milestone.id}",
                source="schedule",
                section="§6.1 schedule",
                title=milestone.what,
                state=state,
                detail=detail,
                surface="crucible/board",
                artifact=gate_key(phase.gate, trading_day)
                if phase.gate
                else (f"(no gate is registered for {phase.id} yet)"),
                means_when_red=(
                    f"the plan's {due} milestone — {milestone.what} — is not met. "
                    f"Tracker: {phase.tracker} ({phase.tracker_url})."
                ),
                last_read=phase_row.read_on if phase_row is not None else None,
            )
        )
    return rows


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


def render_board_html(
    board: Board,
    deltas: list[RowDelta] | None = None,
    *,
    acceptance: dict[str, Any] | None = None,
    acceptance_note: str = "",
) -> str:
    """One page, every declared row, grouped by source.

    The headline is the RED count, not the green one. A board that leads with
    "3 of 40 met" on day one leads with a number that looks like failure and
    is in fact the plan; a board that leads with what is not yet measurable
    leads with the thing that can be acted on.

    **This page is the detailed artifact; the Telegram report is a pointer to
    it** (`alpha-engine-config-I9921`, Brian 2026-09-03: *"I don't find the
    report detailed enough"*). Everything the message has to truncate to fit
    4096 characters is here in full — every row's store key and last-read
    stamp, every phase's per-clause state and reason, the acceptance clause
    list, and the §6.1 schedule.

    ``acceptance`` is the `report/acceptance/{day}.json` document
    (`crucible.keys.acceptance_reading_key`) when one could be read, and
    ``acceptance_note`` says why not when it could not. Both, never one: a
    page that simply omitted the section when the artifact was missing would
    render identically to one whose producer broke, and §12 rule 3's only
    progress figure is the last number this page may go quiet about.
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
        "ul.c{margin:.4rem 0 0;padding-left:1.1rem}",
        "ul.c li{margin:.15rem 0}",
        ".met{color:#1a7f37;font-weight:700}",
        ".unmet{color:#cf222e;font-weight:700}",
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

    parts.extend(_acceptance_section(acceptance, acceptance_note))

    for source in SOURCES:
        rows = [r for r in board.rows if r.source == source]
        if not rows:
            continue
        parts.append(f"<h2>{_esc(source)} ({len(rows)})</h2>")
        parts.append(
            "<table><tr><th>state</th><th>row</th><th>how it knows</th>"
            # `last read`, not `generated at`. The cell under it renders
            # `BoardRow.last_read` — when this ROW's reading was taken — which
            # is a different fact from when the artifact behind it was
            # generated, and a header naming the wrong one makes every stamp
            # in the column a misquotation.
            "<th>store key</th><th>last read</th><th>what red means</th></tr>"
        )
        for row in rows:
            parts.append(
                f"<tr><td class='s' style='color:{_SWATCH[row.state]}'>{row.state}</td>"
                f"<td>{_esc(row.title)}<br><span class='k'>{_esc(row.id)}</span></td>"
                f"<td class='d'>{_esc(row.detail)}{_clause_list_html(row)}</td>"
                f"<td class='k'>{_esc(row.artifact)}</td>"
                # `last_read` is the row's OWN provenance stamp and is
                # rendered as "never read" rather than blank when absent: an
                # empty cell reads as a formatting gap, and this is the field
                # that says whether the reading beside it is from today.
                f"<td class='k'>{_esc(row.last_read) if row.last_read else 'never read'}</td>"
                f"<td class='d'>{_esc(row.means_when_red)}</td></tr>"
            )
        parts.append("</table>")

    parts.append("</body></html>")
    return "".join(parts)


def _clause_list_html(row: BoardRow) -> str:
    """One row's gate clauses, each with its own state and reason.

    Three renderings, because `None`, `()` and a populated list are three
    different facts (:attr:`BoardRow.clauses`) and a page that showed nothing
    for the first two would make an unread gate look like a gate with no
    conditions.
    """
    if row.clauses is None:
        # Only phase rows ever carry a clause list; saying "no reading" under
        # every objective and component row would be noise, not honesty.
        if row.source != "phase":
            return ""
        return (
            "<br><span class='d'>no gate reading was supplied to this render, so no "
            "clause states are shown — a statement about the producer, not the phase.</span>"
        )
    if not row.clauses:
        return "<br><span class='d'>this gate declares no clauses, so it measured nothing.</span>"
    met = sum(1 for c in row.clauses if c["met"])
    items = "".join(
        f"<li><span class='{'met' if c['met'] else 'unmet'}'>"
        f"{'MET' if c['met'] else 'UNMET'}</span> <span class='k'>{_esc(c['name'])}</span> — "
        f"{_esc(c['detail'])}<br><span class='d'>requires: {_esc(c['requirement'])}</span></li>"
        for c in row.clauses
    )
    return (
        f"<br><span class='d'>{met}/{len(row.clauses)} clauses met</span><ul class='c'>{items}</ul>"
    )


def _acceptance_section(acceptance: dict[str, Any] | None, note: str) -> list[str]:
    """§12 rule 3's one progress figure, and the clauses behind it.

    Never omitted. When the artifact cannot be read the section states that
    in the words the reader needs — an absent section and a broken producer
    look identical, and this is the one number the plan calls progress.

    **The completeness rule is not stated here.** It is
    `crucible.keys.parse_acceptance_reading`, which `crucible.morning` reads
    the same artifact through, so this page and the message that links to it
    cannot disagree about whether a document is a reading. They did: the
    adversarial review on `alpha-engine-config-I9921` fed ONE document to both
    and got `acceptance count: not on any artifact` in the message beside
    `21 met / 2 unmet / 1 unmeasurable ... at commit UNKNOWN` on the page,
    because this reader required only the three integers and rendered an
    absent `commit` as the literal `UNKNOWN`. A commit is what makes the
    count reproducible (plan §6 rule 2); `UNKNOWN` in its place is a figure
    nobody can check, printed under a link that says it does not exist.
    """
    from crucible.keys import (  # noqa: PLC0415 - avoids a cycle
        ACCEPTANCE_REQUIRED_FIELDS,
        parse_acceptance_reading,
    )

    parts = ["<h2>acceptance (plan §2 — the only progress figure, §12 rule 3)</h2>"]
    if acceptance is None:
        parts.append(
            f"<p class='d'>No acceptance reading on this store: {_esc(note)}. "
            "The count is a property of the repository (<span class='k'>"
            "tests/acceptance/ratchet.json</span>) until a producer republishes it; "
            "reading it out of whichever checkout rendered this page would be a "
            "fabricated provenance rather than a missing one.</p>"
        )
        return parts
    reading = parse_acceptance_reading(acceptance)
    if reading is None:
        parts.append(
            "<p class='d'>The acceptance artifact is present and does not carry "
            f"{' / '.join(ACCEPTANCE_REQUIRED_FIELDS)} in the shape the producer contract "
            "declares, so it answers a different question than the one asked. Rendering "
            "part of it would look exactly like a measurement.</p>"
        )
        return parts
    parts.append(
        f"<p><strong>{reading.met} met / {reading.unmet} unmet / "
        f"{reading.unmeasurable} unmeasurable</strong> of {reading.total} clauses, at commit "
        f"<span class='k'>{_esc(reading.commit)}</span>, measured "
        f"{_esc(reading.measured_at or 'at an unrecorded time')}.</p>"
    )
    for named, label in (
        (reading.unmet_clauses, "unmet"),
        (reading.unmeasurable_clauses, "unmeasurable"),
    ):
        if not named:
            parts.append(
                f"<p class='d'>The artifact names no {label} clause ids, so this page "
                f"cannot list which {label} clauses they are — only how many.</p>"
            )
            continue
        parts.append(f"<p class='d'>{label} clauses:</p><ul class='c'>")
        parts.extend(f"<li><span class='k'>{_esc(cid)}</span></li>" for cid in named)
        parts.append("</ul>")
    return parts


def _esc(value: Any) -> str:
    import html  # noqa: PLC0415 - one call site, kept local to the renderer

    return html.escape("" if value is None else str(value))


def pointer_may_move(previous: dict[str, Any] | None, board: Board) -> tuple[bool, str]:
    """Whether `board/current.json` may be repointed at this board.

    **A replay must not clobber the pointer with an older board.** Measured on
    this branch before the guard existed: `crucible board --date 2026-09-01`
    then `--date 2026-08-28` left `current.json` reading 2026-08-28, and that
    key is what the fleet-console adapter reads
    (`alpha-engine-config-I9800`) and what the workflow's summary step
    downloads. The dated key is always written; only the POINTER is guarded,
    for the same reason `crucible.release` guards its release pointer rather
    than its releases.

    An unreadable incumbent does NOT license a move. "I could not read what is
    there" is not "what is there is older", and on a pointer those two want
    opposite actions.
    """
    if previous is None:
        return True, "no incumbent pointer"
    incumbent = previous.get("trading_day")
    if not isinstance(incumbent, str) or not incumbent:
        return False, (
            "the incumbent board/current.json carries no trading_day, so this board "
            "cannot be shown to be newer. Refusing to move the pointer: an unreadable "
            "incumbent is not evidence that it is stale."
        )
    if incumbent > board.trading_day:
        return False, (
            f"the incumbent board/current.json is for {incumbent}, which is later than "
            f"this board's {board.trading_day}. A replay does not repoint the pointer — "
            "the dated board was still written."
        )
    return True, f"incumbent is {incumbent}, this board is {board.trading_day}"
