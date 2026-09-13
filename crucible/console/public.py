"""The PUBLIC surface — one static render, from v2 artifacts, with no alpha on it.

Normative sources: `alpha-engine-config-I10223` (this module's tracker),
`alpha-engine-config-I10215` (Brian's 2026-09-09 ruling: `/live`'s default page
is the Promotion view), `repository-tiering-policy.md` (publish-by-purpose),
`console-policy` (the row contract and the JSON twin),
`private-docs/architecture.d/069` (the audience split).

**Why this is a second render and not a flag on the first one.**
:mod:`crucible.console.render` builds the OPERATOR console: fourteen-state
component rows, `log_location`, `absence_watched_by`, week cost, deploy SHAs,
the unreadable-artifact list. Every one of those is tier-2 under
`repository-tiering-policy.md` §2 — it names live infrastructure — and the
attribution table it renders carries the graded `value` of
`portfolio_excess_return_s_ratio`, which is the alpha figure `I10215` ruled
off the public surface. A `public=True` parameter threaded through that
builder would make the public/private boundary a branch inside a renderer,
which is exactly the "enforced by frontend discipline" shape `I10223` names
as unacceptable. The boundary is instead the **projection**: this module
never copies a document through. Every field on :class:`PublicPage` is
constructed by name from a named source field, so a field a producer adds
tomorrow reaches the public page only when someone writes the line that puts
it there.

**What it reads, and nothing else.** Five v2 artifacts the runtime already
publishes:

* `gates/ladder.json` — the plan §6 phase ladder, READ (never recomputed:
  `alpha-engine-config-I10575`), through
  :func:`crucible.console.render._read_ladder` so the two surfaces cannot
  disagree about whether a ladder is stale.
* `board/current.json` — for its `standing` rows only. Brian's 2026-09-13
  ruling took three calendar-floored readings off phase 2's exit gate and
  made them standing SLOs; they are the rows that say whether the system is
  running itself, which is the public claim this surface actually makes.
* `arena/{slot}/{trading_day}/arena_cycle.json` — the four slots' arms and
  the cycle's verdict.
* `champions/{slot}/current.json` — who is serving.
* `report/{trading_day}/attribution.json` — for the STATUS of each graded
  layer, never its value. "Measurement integrity leads, performance is not a
  grade" (`architecture.d/069` §1): that a layer was graded at all, and
  whether the grade breached, is a statement about the instrument. The number
  is the experiment's, and it is not published.

Plus the morning report's delivered headline
(:func:`crucible.keys.morning_report_key`), filtered — see
:func:`_read_headline`.

**Three things are structurally absent, not merely omitted:**

1. **No figure with a magnitude.** No `value`, no `cost_usd`, no score, no
   ratio, no return, no p-value, no n. :data:`FORBIDDEN_SOURCE_FIELDS` names
   the fields this module refuses to read even when they sit in a document it
   opens, and `tests/test_console_public.py` feeds every source artifact a
   distinctive figure and asserts none of them reaches either output.
2. **No free-text `reason`/`detail` from a producer.** An arena cycle's
   `decision.reason` and a gate clause's `detail` are English written for an
   operator; they routinely quote the score that decided the comparison. The
   page renders the declared verdict vocabulary instead, plus a static legend
   this module owns. The cost is real — a public red row here says *what*,
   not *why* — and it is the right trade for a surface whose failure mode is
   publishing a number, not under-explaining one.
3. **No infrastructure identifier.** No `tracker_url` (the tracker is
   private), no `log_location`, no bucket, no role, no run id. The evidence
   column is the artifact's own STORE KEY — a bucket-relative path, which is
   what `console-policy`'s row contract asks for and carries nothing an
   outside reader can use.

**Absence is rendered, never skipped.** Every section that could not be read
says so in the page's own red, with the key named — principle 7, and the same
posture the operator console takes. A public page that quietly drops a
section it could not read is the false-green this repository has already paid
for twice.
"""

from __future__ import annotations

import datetime as dt
import html
import json
import re
from dataclasses import dataclass, field
from typing import Any

from crucible.calendar import resolve_trading_day
from crucible.console.render import STATUS_COLORS, _read_ladder, _state_class
from crucible.documents import read_store_document
from crucible.gate import LADDER_KEY
from crucible.keys import (
    BOARD_CURRENT_KEY,
    PUBLIC_JSON_KEY,
    PUBLIC_KEY,
    arena_cycle_key,
    attribution_key,
    champion_key,
    morning_report_key,
)
from crucible.manifest import manifest_prefix
from crucible.slots import SLOTS
from crucible.store import Store

__all__ = [
    "FORBIDDEN_SOURCE_FIELDS",
    "HEADLINE_JOB",
    "PUBLIC_JSON_KEY",
    "PUBLIC_KEY",
    "STANDING_SOURCE",
    "VERDICT_LEGEND",
    "PublicPage",
    "build_public_page",
    "render_public_html",
    "write_public_page",
]

#: Fields this module refuses to carry onto the public surface even when they
#: sit in a document it legitimately opens. Not a scrubber — nothing here is
#: stripped out of a value at render time, because a scrubber that mutates
#: text is a control nobody can audit and one that silently changes what a
#: figure says. It is a REGISTER: `tests/test_console_public.py` asserts that
#: no name in this set appears as a key anywhere in the published JSON, so a
#: future edit that reaches for `row["value"]` fails a test rather than
#: shipping a number.
#:
#: `value`/`unit` are the attribution row's graded figure (the alpha).
#: `cost_usd` is spend. `score`, `metric`, `ic`, `sharpe`, `ratio`, `n`,
#: `p_value`, `returns`, `excess_return`, `nav`, `pnl` are the arena's and the
#: report's own magnitudes. `reason`, `detail`, `status_reason` and
#: `means_when_red` are producer free text that routinely quotes one of the
#: above. `tracker_url`, `log_location`, `run_id`, `release_sha` and
#: `source_path` are infrastructure identifiers under
#: `repository-tiering-policy.md` §2 step 2.
FORBIDDEN_SOURCE_FIELDS: frozenset[str] = frozenset(
    {
        "cost_usd",
        "detail",
        "excess_return",
        "ic",
        "log_location",
        "means_when_red",
        "metric",
        "n",
        "nav",
        "p_value",
        "pnl",
        "ratio",
        "reason",
        "release_sha",
        "returns",
        "run_id",
        "score",
        "sharpe",
        "source_path",
        "status_reason",
        "tracker_url",
        "unit",
        "value",
    }
)

#: The board source whose rows this page renders. `crucible.board.SOURCES`
#: carries seven; six of them are internal build state (objectives, phases,
#: schedule milestones, component health, cutover predicates, producer
#: liveness) and belong on the operator console. `standing` is the one that
#: answers the public question — is this thing running itself — which is the
#: system property Brian's 2026-09-13 statement of intent is about.
STANDING_SOURCE = "standing"

#: The component whose delivered message carries the headline. Named rather
#: than spelled at the call site so `morning_report_key`'s prefix and this
#: constant cannot disagree about which job wrote the file.
HEADLINE_JOB = "report.morning"

#: What each `arena_cycle.decision.status` means, in this module's own words.
#: A static legend rather than the producer's `reason` field: see the module
#: docstring, point 2. Keyed by the library contract's own enum
#: (`nousergon_lib/contracts/arena_cycle.schema.json`), and
#: :func:`_verdict_label` raises on a status absent from it, so a sixth status
#: reaches this page as a failing test rather than as a blank cell.
VERDICT_LEGEND: dict[str, str] = {
    "decided": "a comparison ran and chose the serving arm",
    "held": "a comparison ran and the incumbent kept the pointer",
    "unmeasurable": "the cycle could not be scored; no arm was chosen",
    "unservable": "no arm cleared its serving preconditions",
    "bootstrap": "the slot's first cycle; there was no incumbent to beat",
}

#: How a verdict renders. `unmeasurable`/`unservable` are RED for the same
#: reason `UNMEASURED` is on the operator console: nothing trustworthy was
#: produced, and a calm colour over no data is the failure mode principle 7
#: names. `held` and `bootstrap` are grey — a true statement about a cycle
#: that ran and did not move the pointer.
VERDICT_STATE: dict[str, str] = {
    "decided": "HEALTHY",
    "held": "RUNNING",
    "bootstrap": "ARMED",
    "unmeasurable": "UNREPORTED",
    "unservable": "FAILED",
}

#: A line of the delivered headline is dropped when it matches this: a link
#: line (the tracker permalink and the presigned board URL), or the pending
#: operator action (an instruction to Brian, which is internal ops and not a
#: public claim). Matched on the RAW line before tags are stripped, so an
#: `<a href>` cannot survive by being formatted differently.
_HEADLINE_DROP = re.compile(r"<a\s|https?://|^\s*pending operator action", re.IGNORECASE)

_TAG = re.compile(r"<[^>]+>")

#: A trading day used ONLY to ask `crucible.keys.morning_report_key` what it
#: names the delivered file, so the basename below is derived from the key
#: builder rather than restated as a literal here (`tests/test_no_inline_store
#: _keys.py`, and the repo rule that a key shape has exactly one owner). Any
#: trading day answers the question; 2026-01-02 is a Friday session.
_BASENAME_PROBE_DAY = "2026-01-02"

#: What `report.morning` calls the delivered message, derived not restated.
_MESSAGE_BASENAME = morning_report_key(_BASENAME_PROBE_DAY, _BASENAME_PROBE_DAY).rsplit("/", 1)[-1]

#: Refused outright rather than dropped line-by-line: any character sequence
#: that still looks like a URL after the drop filter and tag strip has run.
#: Belt and braces on the one field of this page that carries producer prose.
_URL = re.compile(r"https?://|www\.", re.IGNORECASE)


@dataclass
class PublicPage:
    """Everything the public page shows, as data.

    Every field is a projection built by name. No source document is carried
    through, and `faults` is a first-class published field for the same reason
    :class:`crucible.console.render.ConsolePage` publishes `unreadable`: a
    section that could not be read must say so on the surface itself, with the
    key named, rather than vanishing.
    """

    trading_day: str
    generated_utc: str
    #: The morning report's headline, filtered to the lines that carry no
    #: link and no operator instruction. Empty when it could not be read;
    #: ``headline_fault`` then says why.
    headline: list[str] = field(default_factory=list)
    headline_fault: str | None = None
    #: One row per plan §6 phase, from `gates/ladder.json`: number, title,
    #: state, clauses met of total. No `detail`, no `tracker_url`.
    phases: list[dict[str, Any]] = field(default_factory=list)
    phase_ladder_fault: str | None = None
    #: One row per slot in `crucible.slots.SLOTS` order: the serving arm, the
    #: cycle's verdict, and how many arms were active and scored.
    slots: list[dict[str, Any]] = field(default_factory=list)
    #: The `standing` rows of `board/current.json`: title, state, last read.
    slo: list[dict[str, Any]] = field(default_factory=list)
    slo_fault: str | None = None
    #: One row per graded attribution layer: its name and its STATUS. Never
    #: its value.
    integrity: list[dict[str, Any]] = field(default_factory=list)
    integrity_fault: str | None = None
    #: `{"key": ..., "fault": ...}` for every artifact this page tried to read
    #: and could not, deduplicated by key.
    faults: list[dict[str, str]] = field(default_factory=list)

    def to_json(self) -> bytes:
        return json.dumps(self.__dict__, indent=2, sort_keys=True).encode("utf-8")


def _record(faults: list[dict[str, str]], key: str, fault: str) -> str:
    """Record ``fault`` against ``key`` once, and hand it back for a section field.

    The returned sentence always NAMES the key, even when the fault it was
    handed does not: `DocumentRead.require` reports "required field 'rows' is
    str, not list", which is true and actionable only if you already know
    which artifact it is about — and a section field is rendered on its own,
    away from the faults table. Naming it here rather than at each call site
    means one answer instead of four.
    """
    named = fault if key in fault else f"{key}: {fault}"
    if not any(entry["key"] == key for entry in faults):
        faults.append({"key": key, "fault": named})
    return named


def _read_text(store: Store, key: str) -> tuple[str | None, str | None]:
    """``(text, None)`` or ``(None, why)`` for one store key holding plain text.

    The key arrives from a LISTING, so it is a parameter here rather than a
    subscript at the call site: `crucible.keys` owns every key SHAPE, and a
    shape it already owns must not be re-spelled at the point of the read
    (`tests/test_no_inline_store_keys.py`).

    The broad `except` is deliberate and is not a swallow — the same argument
    `crucible.documents.read_document` makes for the JSON path, which this
    cannot reuse because the delivered headline is text, not a document. The
    failure mode caught is "this artifact cannot be read"; the primary
    deliverable (every other section of the page) survives; and the recording
    surface is the returned sentence, which the caller publishes as
    ``headline_fault`` AND as a row in ``faults``.
    """
    try:
        return store.get_bytes(key).decode("utf-8"), None
    except Exception as exc:  # noqa: BLE001 - reported, never swallowed; see above
        return None, f"{key} could not be read: {type(exc).__name__}: {exc}"


def _read_headline(store: Store, trading_day: str) -> tuple[list[str], str | None]:
    """The delivered morning headline, filtered, or why there is none.

    The delivered artifact is Telegram HTML written for Brian: it carries a
    permalink to the PRIVATE tracker comment, a link to the rolling issue, and
    (when there is one) a pending operator action. Those are tier-2 and tier-5
    under `repository-tiering-policy.md` and none of them has a public job.

    So the filter is a drop list on whole LINES, applied before tags are
    stripped (:data:`_HEADLINE_DROP`), followed by a refusal: if anything that
    still looks like a URL survives, the headline is not rendered at all and
    the fault says so. A partial render of a document whose shape changed
    under us is how a link ends up on a public page; refusing is the honest
    outcome and it is visible, because ``headline_fault`` is published.

    The day's calendar-dated file is found by listing the job's own manifest
    prefix rather than by guessing today's calendar date: the report is
    delivered on a CALENDAR morning against a TRADING day (§4.12), and on a
    Monday those differ by three days.
    """
    prefix = manifest_prefix(HEADLINE_JOB, trading_day)
    try:
        candidates = sorted(
            k for k in store.list_keys(prefix) if k.rsplit("/", 1)[-1] == _MESSAGE_BASENAME
        )
    except Exception as exc:  # noqa: BLE001 - reported, never swallowed; see below
        # Not a swallow: the failure mode is "we could not LIST the morning
        # job's prefix", the primary deliverable (the rest of the page)
        # survives because a headline is one section, and the recording
        # surface is `headline_fault` plus the published `faults` list, both
        # of which name the prefix and the cause.
        return [], f"{prefix} could not be listed: {type(exc).__name__}: {exc}"
    if not candidates:
        return [], (
            f"no morning headline under {prefix} — `crucible report.morning` "
            "has delivered nothing for this trading day."
        )
    key = candidates[-1]
    raw, fault = _read_text(store, key)
    if raw is None:
        return [], fault or f"{key} could not be read"
    kept: list[str] = []
    for line in raw.splitlines():
        if not line.strip() or _HEADLINE_DROP.search(line):
            continue
        kept.append(html.unescape(_TAG.sub("", line)).strip())
    if any(_URL.search(line) for line in kept):
        return [], (
            f"{key} still carries a URL after the public filter ran. The headline is "
            "withheld rather than partially published — a link on this page would reach "
            "the private tracker."
        )
    if not kept:
        return [], f"{key} carries no publishable line once links and operator actions are dropped."
    return kept, None


def _phase_rows(ladder: dict[str, Any]) -> list[dict[str, Any]]:
    """The ladder's phases, projected by name.

    `number`, `title`, `state` and the two clause counts. `detail` and
    `tracker_url` are deliberately absent — the first is operator prose that
    quotes gate readings, the second points at a private repository.
    """
    rows: list[dict[str, Any]] = []
    for phase in ladder.get("phases", []):
        if not isinstance(phase, dict):
            continue
        rows.append(
            {
                "number": phase.get("number"),
                "title": phase.get("title"),
                "state": phase.get("state"),
                "clauses_met": phase.get("clauses_met"),
                "clauses_total": phase.get("clauses_total"),
                "artifact": LADDER_KEY,
            }
        )
    return rows


def _verdict_label(status: Any) -> str:
    """:data:`VERDICT_LEGEND`'s sentence for ``status``, refusing an unknown one.

    The same posture :func:`crucible.console.render._state_class` takes: a
    verdict with no declared public meaning is refused rather than rendered
    blank, because a blank cell beside a slot name reads as "nothing
    happened" and the truth would be "something happened that this page has
    never been taught to describe".
    """
    if status not in VERDICT_LEGEND:
        raise KeyError(
            f"arena decision status {status!r} has no entry in VERDICT_LEGEND, so the "
            "public page has no declared wording for it — add one before it can render."
        )
    return VERDICT_LEGEND[status]


def _slot_rows(
    store: Store, trading_day: str, faults: list[dict[str, str]]
) -> list[dict[str, Any]]:
    """One row per slot: who serves it, and what this cycle decided.

    Two artifacts per slot, each read independently, because they fail
    independently: a slot can hold a valid champion pointer and have produced
    no cycle today (the ordinary state between cadences), and it can produce a
    cycle whose pointer has never been written (bootstrap). Collapsing them
    would render one absence as the other.
    """
    rows: list[dict[str, Any]] = []
    for slot in SLOTS:
        row: dict[str, Any] = {
            "slot": slot,
            "serving_arm": None,
            "serving_as_of": None,
            "verdict": None,
            "verdict_state": "UNREPORTED",
            "verdict_means": None,
            "moved": None,
            "active_arms": None,
            "scored_arms": None,
            "artifact": arena_cycle_key(slot, trading_day),
            "fault": None,
        }
        pointer_key = champion_key(slot)
        pointer = read_store_document(store, pointer_key)
        if pointer.problem is not None:
            row["fault"] = _record(faults, pointer_key, f"{pointer_key}: {pointer.problem}")
        elif pointer.document is not None:
            # Projected field by field. The pointer also carries provenance
            # (`decided_at`, `release_sha`, the producing `run_id`) that is
            # infrastructure, not a public claim.
            row["serving_arm"] = pointer.document.get("arm_id")
            row["serving_as_of"] = pointer.document.get("as_of")
        cycle_key = row["artifact"]
        cycle = read_store_document(store, cycle_key)
        if cycle.problem is not None:
            row["fault"] = _record(faults, cycle_key, f"{cycle_key}: {cycle.problem}")
        elif cycle.document is not None:
            decision = cycle.document.get("decision")
            if not isinstance(decision, dict):
                row["fault"] = _record(
                    faults,
                    cycle_key,
                    f"{cycle_key} carries no `decision` object — the cycle exists and does "
                    "not say what it decided, which is not the same as having decided "
                    "nothing.",
                )
            else:
                status = decision.get("status")
                row["verdict"] = status
                row["verdict_means"] = _verdict_label(status)
                row["verdict_state"] = VERDICT_STATE[status]
                row["moved"] = decision.get("moved")
            active = cycle.document.get("active_arms")
            scored = cycle.document.get("scored_arms")
            row["active_arms"] = len(active) if isinstance(active, list) else None
            row["scored_arms"] = len(scored) if isinstance(scored, list) else None
        rows.append(row)
    return rows


def _slo_rows(
    store: Store, faults: list[dict[str, str]]
) -> tuple[list[dict[str, Any]], str | None]:
    """The board's `standing` rows, projected to title, state and last read.

    `detail` and `means_when_red` are board free text: the first quotes the
    reading (a count of operator actions, a date), the second is a sentence
    about what an operator should do. Neither has a public job, and the row's
    own `artifact` — a bucket-relative store key — is what `console-policy`'s
    row contract asks the evidence column to carry.
    """
    read = read_store_document(store, BOARD_CURRENT_KEY)
    if read.absent:
        return [], _record(
            faults,
            BOARD_CURRENT_KEY,
            f"{BOARD_CURRENT_KEY} is absent — `crucible board` has published nothing.",
        )
    problem = read.require("rows", list)
    if problem is not None:
        return [], _record(faults, BOARD_CURRENT_KEY, problem)
    assert read.document is not None  # `require` returned None, so it read
    rows = [
        {
            "id": row.get("id"),
            "title": row.get("title"),
            "state": row.get("state"),
            "last_read": row.get("last_read"),
            "artifact": BOARD_CURRENT_KEY,
        }
        for row in read.document["rows"]
        if isinstance(row, dict) and row.get("source") == STANDING_SOURCE
    ]
    if not rows:
        # Honestly empty, and said so: a board with no standing rows is a
        # board published before the 2026-09-13 ruling landed, not a system
        # with no standing obligations.
        return [], _record(
            faults,
            BOARD_CURRENT_KEY,
            f"{BOARD_CURRENT_KEY} carries no `{STANDING_SOURCE}` rows — this board predates "
            "the standing-SLO source, so no standing obligation is being rendered.",
        )
    return rows, None


def _integrity_rows(
    store: Store, trading_day: str, faults: list[dict[str, str]]
) -> tuple[list[dict[str, Any]], str | None]:
    """Each graded layer's NAME and STATUS from the attribution artifact.

    Never its `value` and never its `unit`. The distinction is the whole
    audience split (`architecture.d/069` §1): that a layer was graded, and
    whether the grade breached, is a statement about the INSTRUMENT and is the
    claim this surface exists to make. The magnitude is the experiment's
    result, which `alpha-engine-config-I10215` ruled off the public surface.
    """
    key = attribution_key(trading_day)
    read = read_store_document(store, key)
    if read.absent:
        return [], _record(
            faults,
            key,
            f"{key} is absent — no report card has been graded for this trading day.",
        )
    problem = read.require("rows", list)
    if problem is not None:
        return [], _record(faults, key, problem)
    assert read.document is not None  # `require` returned None, so it read
    return [
        {"layer": row.get("name"), "status": row.get("status")}
        for row in read.document["rows"]
        if isinstance(row, dict)
    ], None


def build_public_page(store: Store, *, now: dt.datetime | None = None) -> PublicPage:
    """Assemble the public page from the store. Reads artifacts; computes no state.

    Deliberately shares :func:`crucible.console.render._read_ladder` with the
    operator console rather than re-reading `gates/ladder.json` its own way.
    Two surfaces with two staleness rules for one artifact is the drift this
    repository has already paid for on the ladder itself
    (`alpha-engine-config-I10575`), and a PUBLIC surface rendering a stale
    ladder the operator console refuses would be the worse half of that pair.
    """
    moment = (now or dt.datetime.now(dt.UTC)).astimezone(dt.UTC)
    trading_day = resolve_trading_day(moment)
    day = trading_day.isoformat()
    faults: list[dict[str, str]] = []

    ladder, ladder_fault = _read_ladder(store, trading_day)
    if ladder_fault is not None:
        _record(faults, LADDER_KEY, ladder_fault)
    headline, headline_fault = _read_headline(store, day)
    if headline_fault is not None:
        _record(faults, manifest_prefix(HEADLINE_JOB, day), headline_fault)
    slo, slo_fault = _slo_rows(store, faults)
    integrity, integrity_fault = _integrity_rows(store, day, faults)

    return PublicPage(
        trading_day=day,
        generated_utc=moment.strftime("%Y-%m-%dT%H:%M:%SZ"),
        headline=headline,
        headline_fault=headline_fault,
        phases=_phase_rows(ladder),
        phase_ladder_fault=ladder_fault,
        slots=_slot_rows(store, day, faults),
        slo=slo,
        slo_fault=slo_fault,
        integrity=integrity,
        integrity_fault=integrity_fault,
        faults=faults,
    )


_STYLE = (
    """
:root { color-scheme: light dark; --fg:#111; --bg:#fff; --muted:#666; --line:#ddd; }
@media (prefers-color-scheme: dark) {
  :root { --fg:#e8e8e8; --bg:#111; --muted:#999; --line:#333; }
}
body { font: 15px/1.6 ui-sans-serif, system-ui, sans-serif; color: var(--fg);
       background: var(--bg); margin: 0 auto; padding: 24px; max-width: 60rem; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 15px; margin: 28px 0 8px; }
.sub { color: var(--muted); margin: 0 0 16px; }
.wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--line);
         vertical-align: top; }
th { font-weight: 600; white-space: nowrap; }
code { font: 12px ui-monospace, monospace; }
.note { color: var(--muted); max-width: 68ch; }
.state { font-weight: 600; white-space: nowrap; }
pre.headline { font: 13px/1.5 ui-monospace, monospace; white-space: pre-wrap;
               border-left: 3px solid var(--line); padding: 8px 12px; margin: 0; }
"""
    + "\n".join(f".s-{name} {{ color: {color}; }}" for name, color in STATUS_COLORS.items())
    + """
.empty { color: var(--muted); font-style: italic; }
"""
)

#: The page's own statement of what it is and what it deliberately does not
#: show. Rendered, not a comment: a reader who cannot tell that alpha is
#: absent ON PURPOSE reads its absence as the system having none
#: (`alpha-engine-config-I10215`, `architecture.d/069` §1).
_PREAMBLE = (
    "Crucible is an experiment harness. It runs concurrent experiments across four "
    "slots — universe, research, model and strategy — and grades each one against its "
    "own benchmark on a trading-day axis. This page reports whether the harness is "
    "running, what it decided, and whether its own measurements are sound. It does not "
    "publish any experiment's performance: no return, no alpha, no score. Those are the "
    "experiments' results and they are not a public claim."
)


def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _fault_note(fault: str | None) -> str:
    """An UNREPORTED paragraph naming the cause, in the page's own red."""
    return (
        f'<p class="note {_state_class("UNREPORTED")}">{_e(fault)}</p>'
        if fault
        else '<p class="empty">Nothing to report.</p>'
    )


def render_public_html(page: PublicPage) -> str:
    """One page, no JavaScript, no polling, no third-party request.

    Everything is inline: a public page that fetched a font or a script from
    a CDN would put a third party between this system's claims and whoever is
    reading them, and would make the surface's availability depend on a vendor
    nobody chose.
    """
    parts = [
        "<title>Crucible — experiment harness</title>",
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<style>{_STYLE}</style>",
        "<h1>Crucible</h1>",
        f'<p class="sub">Trading day <code>{_e(page.trading_day)}</code> · rendered '
        f"<code>{_e(page.generated_utc)}</code></p>",
        f'<p class="note">{_e(_PREAMBLE)}</p>',
        "<h2>Morning report</h2>",
    ]
    if page.headline:
        parts.append(f'<pre class="headline">{_e(chr(10).join(page.headline))}</pre>')
    else:
        parts.append(_fault_note(page.headline_fault))

    parts.append("<h2>Build ladder</h2>")
    if page.phases:
        parts.append(
            '<div class="wrap"><table><thead><tr><th>Phase</th><th>Title</th>'
            "<th>State</th><th>Clauses met</th><th>Evidence</th>"
            "</tr></thead><tbody>"
        )
        for row in page.phases:
            met, total = row["clauses_met"], row["clauses_total"]
            fraction = "—" if met is None or total is None else f"{met}/{total}"
            parts.append(
                "<tr>"
                f"<td>{_e(row['number'])}</td>"
                f"<td>{_e(row['title'])}</td>"
                f'<td class="state {_state_class(row["state"])}">{_e(row["state"])}</td>'
                f"<td>{_e(fraction)}</td>"
                f"<td><code>{_e(row['artifact'])}</code></td>"
                "</tr>"
            )
        parts.append("</tbody></table></div>")
    else:
        parts.append(_fault_note(page.phase_ladder_fault))

    parts.append("<h2>Slots</h2>")
    parts.append(
        '<div class="wrap"><table><thead><tr><th>Slot</th><th>Serving arm</th>'
        "<th>Since</th><th>Verdict</th><th>Pointer moved</th><th>Arms active</th>"
        "<th>Arms scored</th><th>Evidence</th></tr></thead><tbody>"
    )
    for row in page.slots:
        moved = "—" if row["moved"] is None else ("yes" if row["moved"] else "no")
        verdict = row["verdict"] or "no cycle"
        parts.append(
            "<tr>"
            f"<td><code>{_e(row['slot'])}</code></td>"
            f"<td><code>{_e(row['serving_arm'] or '—')}</code></td>"
            f"<td>{_e(row['serving_as_of'] or '—')}</td>"
            f'<td class="state {_state_class(row["verdict_state"])}" '
            f'title="{_e(row["verdict_means"] or "")}">{_e(verdict)}</td>'
            f"<td>{_e(moved)}</td>"
            f"<td>{_e('—' if row['active_arms'] is None else row['active_arms'])}</td>"
            f"<td>{_e('—' if row['scored_arms'] is None else row['scored_arms'])}</td>"
            f"<td><code>{_e(row['artifact'])}</code></td>"
            "</tr>"
        )
    parts.append("</tbody></table></div>")
    parts.append(
        '<p class="note">'
        + _e(" · ".join(f"{name}: {meaning}" for name, meaning in sorted(VERDICT_LEGEND.items())))
        + "</p>"
    )

    parts.append("<h2>Standing obligations</h2>")
    if page.slo:
        parts.append(
            '<div class="wrap"><table><thead><tr><th>Obligation</th><th>State</th>'
            "<th>Last read</th><th>Evidence</th></tr></thead><tbody>"
        )
        for row in page.slo:
            parts.append(
                "<tr>"
                f"<td>{_e(row['title'])}</td>"
                f'<td class="state {_state_class(row["state"])}">{_e(row["state"])}</td>'
                f"<td>{_e(row['last_read'] or '—')}</td>"
                f"<td><code>{_e(row['artifact'])}</code></td>"
                "</tr>"
            )
        parts.append("</tbody></table></div>")
    else:
        parts.append(_fault_note(page.slo_fault))

    parts.append("<h2>Measurement integrity</h2>")
    parts.append(
        '<p class="note">Whether each layer of the report card was graded at all, and '
        "whether the grade breached its declared objective. The graded values are the "
        "experiments' results and are not published here.</p>"
    )
    if page.integrity:
        parts.append(
            '<div class="wrap"><table><thead><tr><th>Layer</th><th>Status</th></tr></thead><tbody>'
        )
        for row in page.integrity:
            parts.append(
                "<tr>"
                f"<td><code>{_e(row['layer'])}</code></td>"
                f'<td class="state {_state_class(row["status"])}">{_e(row["status"])}</td>'
                "</tr>"
            )
        parts.append("</tbody></table></div>")
    else:
        parts.append(_fault_note(page.integrity_fault))

    if page.faults:
        parts.append("<h2>Artifacts this page could not read</h2>")
        parts.append(
            '<div class="wrap"><table><thead><tr><th>Artifact</th><th>Why</th></tr></thead><tbody>'
        )
        for entry in page.faults:
            parts.append(
                "<tr>"
                f"<td><code>{_e(entry['key'])}</code></td>"
                f'<td class="note">{_e(entry["fault"])}</td>'
                "</tr>"
            )
        parts.append("</tbody></table></div>")

    parts.append(
        f'<p class="note">The machine-readable twin of this page is '
        f"<code>{_e(PUBLIC_JSON_KEY)}</code>, served at the same path with "
        "<code>.json</code>.</p>"
    )
    return "\n".join(parts)


def write_public_page(store: Store, page: PublicPage) -> tuple[str, ...]:
    """Write the public HTML and its JSON twin. Returns the keys.

    The JSON is not an extra: `console-policy` §3.8 makes an agent a
    first-class reader of every view, and a public page whose numbers can only
    be scraped out of markup is a page the next automated reader re-derives
    incorrectly.
    """
    store.put_bytes(PUBLIC_KEY, render_public_html(page).encode("utf-8"))
    store.put_bytes(PUBLIC_JSON_KEY, page.to_json())
    return PUBLIC_KEY, PUBLIC_JSON_KEY
