"""Build the console page's data, then render it.

Split in two on purpose: :func:`build_page` produces a plain dict that a test
can assert against and an agent can read, and :func:`render_html` turns that
into the page. A renderer that computed its own numbers would put the console
and its JSON out of step, and the JSON is what the next reader parses.
"""

from __future__ import annotations

import datetime as dt
import html
import json
from dataclasses import dataclass, field, replace
from typing import Any, get_args

from krepis.metrics import StatusLiteral

from crucible.calendar import previous_trading_day, resolve_trading_day
from crucible.components import Component, load_registry
from crucible.console.classify import STATES, Classification, classify
from crucible.documents import (
    DocumentRead,
    read_listed_document,
    read_manifests_under,
    read_store_document,
)
from crucible.gate import LADDER_KEY, LADDER_SCHEMA_VERSION, LADDER_STATES, PHASES
from crucible.gate import validate_ladder_document as _validate_ladder_document
from crucible.keys import (
    CONSOLE_JSON_KEY,
    CONSOLE_KEY,
    RUNS_ROOT,
    attribution_key,
    champion_key,
    is_manifest_key,
    parse_manifest_key,
    runs_prefix,
)
from crucible.manifest import manifest_prefix
from crucible.slots import SLOTS, dispatchable_slots
from crucible.store import Store
from crucible.weekly import ARC_SLOT_JOBS

__all__ = [
    "ATTRIBUTION_STATUSES",
    "classify_registry",
    "CONSOLE_KEY",
    "LADDER_KEY",
    "ConsolePage",
    "STATUS_COLORS",
    "build_page",
    "render_html",
    "write_page",
]

#: The phase whose track-C work introduced the closed status vocabulary these
#: two guards defend. Derived rather than hardcoded so a phase renumbering
#: cannot leave the message pointing at a finished issue
#: (alpha-engine-config-I9839).
_C14_PHASE = next(p for p in PHASES if p.id == "phase1")

# `CONSOLE_KEY` / `CONSOLE_JSON_KEY` live in `crucible.keys` like every other
# store key shape (alpha-engine-config-I9899, round 2) and are imported above;
# `CONSOLE_KEY` stays in this module's `__all__` for its existing readers.


@dataclass
class ConsolePage:
    """Everything the page shows, as data.

    ``unreported`` is a top-level field rather than something a reader counts
    off the rows: §8.4 makes it the transparency-gap count with an objective
    of zero, and a number nobody publishes is a number nobody is held to. It
    is the sum of components classified `UNREPORTED` *and* every individual
    metric, on any row, whose own status is `UNREPORTED` — a component can
    classify `HEALTHY` or `DEGRADED` overall and still owe this count a
    metric that went blind underneath it (alpha-engine-config-I9757, C5).
    """

    trading_day: str
    generated_utc: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    #: Every artifact this page tried to read and could not, as
    #: `{"key": ..., "fault": ...}`. A first-class published field rather than
    #: a log line: `alpha-engine-config-I9900` is the case where ONE
    #: unreadable artifact raised out of the builder and published no page at
    #: all, and the remedy is not a quieter failure but a page that renders
    #: every other row and says, on its own surface, exactly which key it
    #: could not read and why.
    unreadable: list[dict[str, str]] = field(default_factory=list)
    #: The plan §6 phase ladder — one row per phase — as READ from
    #: `gates/ladder.json`, never as computed here. The console is a consumer
    #: of that key and not a producer of it (`alpha-engine-config-I10575`):
    #: it re-evaluated every gate under the console runtime's own
    #: environment, which deliberately carries neither `CRUCIBLE_MUTED_TOPIC`
    #: nor `CRUCIBLE_CLOUDTRAIL_ARCHIVE` nor `ce:GetCostAndUsage`, and on
    #: 2026-09-12 republished a phase0 UNMEASURABLE 0/5 ladder over
    #: `gate.close`'s phase2 7/10 reading while reporting `status: ok`.
    #: Empty when the ladder could not be read, could not be validated, or is
    #: older than this page's trading day — ``phase_ladder_fault`` then says
    #: which, and the panel renders UNREPORTED rather than a recomputed
    #: ladder.
    phase_ladder: dict[str, Any] = field(default_factory=dict)
    #: Why ``phase_ladder`` is empty, as a sentence naming `gates/ladder.json`
    #: and the cause; None when the ladder read. A published field rather than
    #: a log line: the panel's red has to name what to fix, and the JSON an
    #: agent reads has to carry the same cause the HTML shows.
    phase_ladder_fault: str | None = None
    attribution: list[dict[str, Any]] = field(default_factory=list)
    #: The champion pointer document per slot, or None when no pointer has
    #: been written OR the pointer could not be read. Which of those two it is
    #: lives in ``champion_faults``: a reader-state field carried INSIDE the
    #: pointer document (`{"unreadable": ...}`) overloaded the trader's
    #: contract namespace with a fact about our read of it
    #: (`alpha-engine-config-I9931` item 3).
    champions: dict[str, Any] = field(default_factory=dict)
    #: `{slot: fault}` for every champion pointer that is present and could
    #: not be read. Empty when every pointer read or is honestly absent.
    champion_faults: dict[str, str] = field(default_factory=dict)
    deploys: list[dict[str, Any]] = field(default_factory=list)
    week_cost_usd: float = 0.0
    unreported: int = 0
    population: int = 0

    def to_json(self) -> bytes:
        return json.dumps(self.__dict__, indent=2, sort_keys=True).encode("utf-8")


#: The ONE guarded reader this module parses stored JSON with — imported, not
#: redefined. Every JSON read below goes through it: the representative
#: manifest, the week-cost roll-up, the attribution artifact and each champion
#: pointer. `alpha-engine-config-I9900` is what an unguarded read costs — a
#: `json.loads` on a `message.txt` sitting beside a run manifest raised
#: `JSONDecodeError` out of `_read_representative_manifest`, out of
#: `classify_registry`, out of `crucible board`, and published no board and no
#: ladder for seven hours.
_read = read_store_document


def _fault(read: DocumentRead, key: str) -> str | None:
    """``read``'s fault as a sentence naming ``key``, or None when it read.

    An ABSENT artifact is not a fault: absence is an answer this page renders
    (`no pointer written`, `no attribution artifact`). Present-and-unreadable
    is, and it names the key, because a detail that says only "unreadable" is
    a detail nobody can act on.
    """
    if read.problem is None:
        return None
    return f"{read.problem}" if key in read.problem else f"{key}: {read.problem}"


@dataclass(frozen=True)
class ManifestRead:
    """The representative manifest for one component, or why there is none.

    ``manifest`` and ``problem`` are never both set, and both being None means
    the component filed nothing for this trading day — the ordinary absence
    the classifier's four absence branches already answer. ``problem`` set is
    the third outcome `classify` grew an argument for: something WAS filed and
    we could not read it.
    """

    manifest: dict[str, Any] | None
    problem: str | None
    #: `{key: fault}` for every key under the prefix that could not be read.
    #: Keyed by STORE KEY rather than by component so the page's unreadable
    #: section can deduplicate against the week-cost roll-up, which lists
    #: `RUNS_ROOT` and reaches the same objects by a different route.
    faults: dict[str, str] = field(default_factory=dict)


def _read_representative_manifest(store: Store, job: str, trading_day: str) -> ManifestRead:
    """One manifest to classify ``job`` for ``trading_day`` by, from however
    many its writers produced.

    **Scope, since `alpha-engine-config-I9818`:** this collapse is used for
    every job EXCEPT the two named in :data:`crucible.weekly.ARC_SLOT_JOBS`
    (`experiment.run`, `experiment.grade`), which :func:`build_page` renders
    one row per writer for instead — see :func:`_read_manifests_by_slot`.
    `alerts.sweep` still collapses through this function: it discriminates by
    `calendar_date`, not by a registry-derived slot set the way the slot jobs
    do (its Friday/Saturday/Sunday writers share one trading day by calendar
    construction, not by a dispatchable-slot count that grows), and I9818's
    own measured facts scope the per-writer redesign to the slot-expanded
    stages only. A per-writer console row for `alerts.sweep` is a follow-up
    this function's continued use here deliberately leaves open, not an
    oversight.

    A job that carries a discriminator (`experiment.run`/`experiment.grade`
    by slot, `alerts.sweep` by `calendar_date`) can have written several
    manifests here since I9781 fixed the collision that used to leave
    exactly one, last-writer-wins. This row still renders one classification,
    so a `failed` manifest wins over an `ok` one — a component is DEGRADED
    the moment any one of its writers failed, never masked by a healthier
    sibling — and otherwise the lexicographically-last manifest is used, on
    the same principle `Store.list_keys` orders by: deterministic, not a
    claim about recency.

    **A manifest prefix is a namespace, not a manifest list.** The listing is
    narrowed by :func:`crucible.keys.is_manifest_key` before anything is read,
    because a job may legitimately file its own evidence beside its manifest
    and `report.morning` does exactly that — `runs/report.morning/{day}/
    {calendar_date}/message.txt`, the delivered text, under the job's own
    prefix so the delivery needs no second IAM grant. Parsing that as a
    manifest is `alpha-engine-config-I9900`, and it is why the board published
    nothing between 14:26Z and this fix.

    An unreadable manifest is reported, never skipped and never raised: it
    becomes this row's `problem`, which `classify` renders UNREPORTED with the
    key and the fault in the detail. A key that was listed and had vanished by
    the time it was read is a fault too, not an absence — the `runs/` loop in
    :func:`build_page` already recorded that race and this function silently
    dropped it (`alpha-engine-config-I9931` item 2); both now read through
    :func:`crucible.documents.read_manifests_under`, so there is one answer.
    """
    listed = read_manifests_under(store, manifest_prefix(job, trading_day))
    if listed.listing_problem is not None:
        # "Could not ask", rendered as itself. The console has a fault channel
        # already; folding an unlistable prefix into `no manifests` would put a
        # blank row where an access failure belongs, and *no data* is never
        # rendered green (principle 7, alpha-engine-config-I9960).
        return ManifestRead(None, listed.listing_problem, {})
    candidates = [document for _key, document in listed.documents]
    faults = dict(listed.faults)
    if faults:
        # Reported even when a readable sibling exists. A component with one
        # corrupt writer and one healthy one is not healthy — the same rule
        # the `failed`-wins branch below states, applied to the manifest we
        # could not read at all rather than to the one that said it failed.
        return ManifestRead(None, "; ".join(faults[k] for k in sorted(faults)), faults)
    if not candidates:
        return ManifestRead(None, None)
    for manifest in candidates:
        if manifest.get("status") == "failed":
            return ManifestRead(manifest, None)
    return ManifestRead(candidates[-1], None)


def _has_history(store: Store, job: str) -> bool:
    """Whether this component has ever produced a manifest.

    The whole difference between MISSED and NEVER_RAN, so it is a real
    listing rather than an assumption. Short-circuits on the first hit — the
    question is existential, not a count.
    """
    # Not `manifest_prefix(job, trading_day)`: this checks whether the
    # component has EVER produced a manifest, across every trading day.
    prefix = runs_prefix(job)
    for key in store.list_keys(prefix):
        # `is_manifest_key`, not a `"/run.json"` suffix literal: the basename
        # is `crucible.keys`' to own, and the predicate checks the root and
        # the arity too (alpha-engine-config-I9900).
        if is_manifest_key(key):
            return True
    return False


def _discriminated_slots() -> list[str]:
    """The slots :data:`crucible.weekly.ARC_SLOT_JOBS` expand into, in
    `SLOTS` order.

    The EXACT expression `crucible.weekly.arc_stages` builds the weekly arc
    from — `[slot for slot in SLOTS if slot in dispatchable_slots()]` —
    reused here rather than re-derived, so the console's row count and the
    arc's own stage count can never disagree about which slots are live. Not
    hand-listed: `dispatchable_slots()` reads off which slot modules expose
    `produce`/`grade`, so a slot landing on the CLI (M, then S, at phase 3)
    grows this list, and therefore the console's row count, with no change
    here (`alpha-engine-config-I9818`).
    """
    return [slot for slot in SLOTS if slot in dispatchable_slots()]


def _read_manifests_by_slot(
    store: Store, job: str, trading_day: str, slots: list[str]
) -> dict[str, ManifestRead]:
    """One :class:`ManifestRead` per ``slot`` in ``slots`` — never collapsed
    across slots the way :func:`_read_representative_manifest` collapses
    across every discriminator.

    `alpha-engine-config-I9818`: a job in `ARC_SLOT_JOBS` writes one manifest
    per dispatchable slot at `runs/{job}/{trading_day}/{slot}/run.json`, and
    the console used to reduce all of them to one row — a healthy slot and a
    failed slot read as one status, and the row's `run_id`/`trading_day`
    belonged to whichever manifest won the collapse's tiebreak. This
    function is the replacement: it still reuses `_read_representative
    _manifest`'s worst-status-wins rule, but applies it WITHIN one slot's own
    candidates only (defensive — `crucible.runner` writes exactly one
    manifest per `(job, trading_day, discriminator)` key, so a slot should
    never have more than one candidate in practice), never across slots.

    The whole prefix is listed exactly ONCE (`read_manifests_under`) and then
    partitioned by discriminator, so N slot rows cost one listing rather than
    N — the same care `crucible.aggregation`'s callers take not to reduce a
    reduction's own reads into a second N+1 query shape.
    """
    listed = read_manifests_under(store, manifest_prefix(job, trading_day))
    if listed.listing_problem is not None:
        return {slot: ManifestRead(None, listed.listing_problem, {}) for slot in slots}
    docs_by_slot: dict[str, list[dict[str, Any]]] = {slot: [] for slot in slots}
    faults_by_slot: dict[str, dict[str, str]] = {slot: {} for slot in slots}
    for key, document in listed.documents:
        parsed = parse_manifest_key(key)
        assert parsed is not None  # `read_manifests_under` already filtered to manifest keys
        _, _, discriminator = parsed
        if discriminator in docs_by_slot:
            docs_by_slot[discriminator].append(document)
        # A discriminator outside `slots` (a slot the registry no longer
        # dispatches, or a stray key) is not this row's business — it is
        # neither dropped nor mistaken for one of the expected slots.
    for key, fault in listed.faults.items():
        parsed = parse_manifest_key(key)
        assert parsed is not None
        _, _, discriminator = parsed
        if discriminator in faults_by_slot:
            faults_by_slot[discriminator][key] = fault
    reads: dict[str, ManifestRead] = {}
    for slot in slots:
        faults = faults_by_slot[slot]
        if faults:
            reads[slot] = ManifestRead(None, "; ".join(faults[k] for k in sorted(faults)), faults)
            continue
        candidates = docs_by_slot[slot]
        if not candidates:
            reads[slot] = ManifestRead(None, None)
            continue
        failed = next((m for m in candidates if m.get("status") == "failed"), None)
        reads[slot] = ManifestRead(failed if failed is not None else candidates[-1], None)
    return reads


def _has_history_by_slot(store: Store, job: str, slots: list[str]) -> dict[str, bool]:
    """The multi-slot analogue of :func:`_has_history`: whether each slot in
    ``slots`` has EVER produced a manifest for ``job``, from one listing of
    the job's whole history.

    Needed because MISSED vs NEVER_RAN becomes a per-writer fact once a job
    renders one row per writer: a slot that has never run must not read
    RUNNING or MISSED just because a sibling slot has history, which is
    exactly what calling the job-wide `_has_history` per slot would do.
    """
    found = dict.fromkeys(slots, False)
    for key in store.list_keys(runs_prefix(job)):
        parsed = parse_manifest_key(key)
        if parsed is not None and parsed[2] in found:
            found[parsed[2]] = True
    return found


def classify_registry(
    store: Store,
    registry: dict[str, Component],
    *,
    now: dt.datetime,
    trading_day: dt.date,
    faults: dict[str, str] | None = None,
    skip: frozenset[str] = frozenset(),
) -> tuple[dict[str, Classification], dict[str, dict[str, Any] | None]]:
    """Classify every registry row once, and hand back the manifests too.

    Extracted so the console page and the declared board
    (`alpha-engine-config-I9837`) read the SAME classification rather than
    each computing its own. Two surfaces classifying the same components
    independently is a contract restated in two places, and this repository
    has already watched one of those drift — the board's whole argument is
    that a second copy of a declaration is the defect, so it would be an odd
    thing for the board itself to introduce.

    Returns `(classifications, manifests)` keyed by component name, both
    covering every registry row not named in ``skip``. A component with no
    manifest is present in both maps with a `None` manifest, never absent —
    an absent key would make a caller's `.get()` return `None` for "no such
    component" and "no run today" alike.

    ``skip`` names registry rows this call does not classify at all — absent
    from both returned maps. `alpha-engine-config-I9818`: `build_page` passes
    `ARC_SLOT_JOBS` here and classifies each of those two jobs itself, one row
    PER WRITER, instead of through this function's worst-status collapse —
    asking this function to collapse them too would be wasted work and would
    report the same unreadable manifest through two routes into `faults`.
    Every other caller (the board) passes nothing and sees every registry
    row, exactly as before.

    ``faults`` is an optional mapping this fills with `{store key: fault}` for
    every manifest that existed and could not be read. An
    out-parameter rather than a third return value because
    `crucible.track_c.board_handler` unpacks the pair and the board does not
    render an unreadable-artifact section — the console does, and it is the
    only caller that asks. The classification itself already carries the fault
    in its reason for every caller, so nothing is lost by not asking.
    """
    classifications: dict[str, Classification] = {}
    manifests: dict[str, dict[str, Any] | None] = {}
    for name, component in sorted(registry.items()):
        if name in skip:
            continue
        read = _read_representative_manifest(store, name, trading_day.isoformat())
        manifests[name] = read.manifest
        if faults is not None:
            faults.update(read.faults)
        classifications[name] = classify(
            component,
            read.manifest,
            now=now,
            # An unreadable manifest still counts as history: something ran.
            history=read.manifest is not None
            or read.problem is not None
            or _has_history(store, name),
            unreadable=read.problem,
        )
    return classifications, manifests


def _read_ladder(store: Store, trading_day: dt.date) -> tuple[dict[str, Any], str | None]:
    """`gates/ladder.json` as the gate publisher wrote it, or why it is unusable.

    The console READS this key. It does not compute it and it does not write
    it — `crucible gate --publish` / `crucible gate.close`
    (`crucible.gate.ladder_payload`) is the one producer
    (`alpha-engine-config-I10575`).

    Four outcomes, and three of them are an UNREPORTED panel rather than a
    ladder:

    * **absent** — nothing has published a ladder yet;
    * **unreadable** — present and not parseable as an object, or our access
      to it failed;
    * **invalid** — present, parseable, and not conformant to
      `phase_ladder.v1`;
    * **stale** — present and valid, carrying a `trading_day` EARLIER than
      the day this page is keyed to, which is the shape of a publisher that
      stopped running.

    A later `trading_day` is NOT stale and is not refused: a ladder published
    for a day after this page's is a replay of an older page against a
    current ladder, and rendering the current reading is correct there. What
    is refused is showing a reading from before the day being rendered as
    though it were this day's.

    Nothing here re-evaluates the gates. A recomputed ladder is exactly
    the defect: it reads green-or-red under whatever environment the console
    process happens to hold, which is not the environment the gate is
    measured in.
    """
    read = _read(store, LADDER_KEY)
    if read.absent:
        return {}, (
            f"{LADDER_KEY} is absent — no gate publisher has written a ladder. "
            "`crucible gate.close` (or `crucible gate --publish`) is its one producer; "
            "the console reads it and never computes it."
        )
    if read.problem is not None or read.document is None:
        return {}, _fault(read, LADDER_KEY) or str(read.problem)
    document = read.document
    try:
        _validate_ladder_document(document)
    except ValueError as exc:
        # Not a swallow: the failure mode is "the published ladder does not
        # conform to phase_ladder.v1", and the recording surface is the
        # UNREPORTED ladder panel plus this page's JSON, both of which carry
        # the validator's own sentence naming the offending field.
        return {}, f"{LADDER_KEY} does not validate against {LADDER_SCHEMA_VERSION}: {exc}"
    published_day = document.get("trading_day")
    if not isinstance(published_day, str) or published_day < trading_day.isoformat():
        return {}, (
            f"{LADDER_KEY} is stale: it was built for trading day {published_day!r}, "
            f"before this page's {trading_day.isoformat()}. A reading from an earlier "
            "day rendered as this day's is the false-green this panel refuses."
        )
    return document, None


def build_page(
    store: Store,
    *,
    now: dt.datetime | None = None,
    registry: dict[str, Component] | None = None,
    run_days: int = 5,
) -> ConsolePage:
    """Assemble the page from the store. Reads artifacts; computes no state.

    ``run_days`` is a count of TRADING days (§4.12), so a holiday week shows
    the same number of sessions as any other rather than four.
    """
    moment = (now or dt.datetime.now(dt.UTC)).astimezone(dt.UTC)
    reg = registry if registry is not None else load_registry()
    trading_day = resolve_trading_day(moment)

    days = [trading_day]
    for _ in range(run_days - 1):
        days.append(previous_trading_day(days[-1]))
    day_set = {d.isoformat() for d in days}

    rows: list[dict[str, Any]] = []
    metric_gap = 0
    unreadable: list[dict[str, str]] = []
    manifest_faults: dict[str, str] = {}
    # `ARC_SLOT_JOBS` is skipped here and classified below instead, one row
    # PER WRITER rather than through the worst-status collapse
    # (`alpha-engine-config-I9818`) — see `classify_registry`'s `skip` and
    # `_read_manifests_by_slot`.
    classifications, manifests = classify_registry(
        store, reg, now=moment, trading_day=trading_day, faults=manifest_faults, skip=ARC_SLOT_JOBS
    )
    slot_order = _discriminated_slots()
    row_components: dict[str, Component] = {}
    for name, component in sorted(reg.items()):
        if name in ARC_SLOT_JOBS:
            slot_reads = _read_manifests_by_slot(store, name, trading_day.isoformat(), slot_order)
            history_by_slot = _has_history_by_slot(store, name, slot_order)
            for slot in slot_order:
                row_name = f"{name}[{slot}]"
                read = slot_reads[slot]
                manifests[row_name] = read.manifest
                manifest_faults.update(read.faults)
                row_components[row_name] = replace(component, name=row_name)
                classifications[row_name] = classify(
                    row_components[row_name],
                    read.manifest,
                    now=moment,
                    # An unreadable manifest still counts as history: something
                    # ran. History is scoped to THIS slot — a sibling slot's
                    # history must never make an empty slot read RUNNING or
                    # MISSED instead of NEVER_RAN.
                    history=read.manifest is not None
                    or read.problem is not None
                    or history_by_slot[slot],
                    unreadable=read.problem,
                )
        else:
            row_components[name] = component

    unreadable.extend(
        {"key": key, "fault": fault} for key, fault in sorted(manifest_faults.items())
    )
    for row_name, component in sorted(row_components.items()):
        manifest = manifests[row_name]
        classification = classifications[row_name]
        rows.append(_row(component, classification, manifest))
        # alpha-engine-config-I9757 (C5): the transparency-gap count read
        # only component STATES, never the metric statuses inside a
        # manifest that classified HEALTHY or DEGRADED — so a run with two
        # of three metrics UNREPORTED (§10.5) contributed zero to "objective
        # 0". Counting each UNREPORTED metric directly, in addition to
        # UNREPORTED component rows below, makes the count see what the row
        # color alone cannot: a component can be DEGRADED rather than
        # UNREPORTED and still owe the gap count every metric it went blind
        # on.
        metric_gap += sum(
            1 for m in (manifest or {}).get("metrics", []) if m.get("status") == "UNREPORTED"
        )

    week_cost = 0.0
    deploys: list[dict[str, Any]] = []
    for key in store.list_keys(RUNS_ROOT):
        parsed = parse_manifest_key(key)
        if parsed is None:
            continue
        job, trading_day_str, _discriminator = parsed
        if trading_day_str not in day_set:
            continue
        # `read_listed_document`, not `_read`: this key came from a listing, so
        # "gone by the time it was read" is a race to record, never an absence.
        read = read_listed_document(store, key)
        if read.problem is not None or read.document is None:
            # A manifest in the window we cannot read is a hole in the week's
            # cost, and a hole nobody publishes reads as $0 spent. Named on
            # the page instead, and the row it belongs to is already
            # UNREPORTED above (alpha-engine-config-I9900).
            # Deduplicated against the per-component pass above: the same
            # object is reachable by two routes (a manifest prefix and
            # `RUNS_ROOT`) and listing it twice would make the page's own
            # unreadable count depend on how many readers happened to touch it.
            if key not in manifest_faults:
                # `read.problem` is set on every non-document outcome of a
                # listed read, so the fault is never an empty string.
                unreadable.append({"key": key, "fault": _fault(read, key) or str(read.problem)})
            continue
        manifest = read.document
        cost = manifest.get("cost_usd", 0.0)
        if isinstance(cost, bool) or not isinstance(cost, int | float):
            unreadable.append(
                {"key": key, "fault": f"cost_usd is {type(cost).__name__}, not a number ({cost!r})"}
            )
        else:
            week_cost += float(cost)
        if job == "deploy":
            deploys.append(
                {
                    "trading_day": manifest.get("trading_day"),
                    "status": manifest.get("status"),
                    "release_sha": manifest.get("release_sha"),
                    "reason": manifest.get("reason"),
                    "run_id": manifest.get("run_id"),
                }
            )

    attribution_key_ = attribution_key(trading_day.isoformat())
    attribution_read = _read(store, attribution_key_)
    attribution_fault = _fault(attribution_read, attribution_key_)
    if attribution_fault is not None:
        unreadable.append({"key": attribution_key_, "fault": attribution_fault})
    attribution_rows = _attribution_rows(attribution_read, attribution_key_, unreadable)
    champions, champion_faults = _champions(store, unreadable)
    ladder, ladder_fault = _read_ladder(store, trading_day)

    return ConsolePage(
        trading_day=trading_day.isoformat(),
        generated_utc=moment.strftime("%Y-%m-%dT%H:%M:%SZ"),
        rows=rows,
        # Track A owns `report`; until it lands there is no attribution
        # artifact, and the page says so rather than showing an empty table
        # that reads like five rows of zero.
        attribution=attribution_rows,
        champions=champions,
        champion_faults=champion_faults,
        deploys=sorted(deploys, key=lambda d: str(d.get("trading_day"))),
        week_cost_usd=round(week_cost, 4),
        phase_ladder=ladder,
        phase_ladder_fault=ladder_fault,
        unreadable=unreadable,
        unreported=sum(1 for r in rows if r["state"] == "UNREPORTED") + metric_gap,
        population=len(rows),
    )


def _row(
    component: Component,
    classification: Classification,
    manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "component": component.name,
        "state": classification.state,
        "reason": classification.reason,
        "lifecycle": component.lifecycle,
        "schedule": component.schedule,
        "deadline": component.deadline.describe() if component.deadline else None,
        "absence_watched_by": component.absence_watched_by,
        "console_surface": component.console_surface,
        "log_location": component.log_location,
        "run_id": classification.run_id,
        "trading_day": classification.trading_day,
        "cost_usd": (manifest or {}).get("cost_usd"),
        "attempts": len((manifest or {}).get("attempts", []) or []) or None,
    }


def _attribution_rows(
    read: DocumentRead, key: str, unreadable: list[dict[str, str]]
) -> list[dict[str, Any]]:
    """The attribution table's rows, or an empty table plus a named fault.

    `rows` is the one required field of this artifact, and it must be a list.
    A document that parses to an object and then carries `rows` as a string or
    a mapping used to reach the renderer and raise there instead — one line
    further down, with a less obvious traceback and the same outcome: no page
    at all.
    """
    if read.absent:
        return []
    problem = read.require("rows", list)
    if problem is not None:
        if not any(entry["key"] == key for entry in unreadable):
            unreadable.append({"key": key, "fault": problem})
        return []
    assert read.document is not None  # `require` returned None, so it read
    return list(read.document["rows"])


def _champions(
    store: Store, unreadable: list[dict[str, str]]
) -> tuple[dict[str, Any], dict[str, str]]:
    """``(pointer per slot, fault per unreadable slot)``.

    Track B owns `promote` and therefore the pointers. Reading them here is a
    one-way dependency on an artifact contract, not on their code.

    A pointer that is present and unreadable is neither a champion nor "no
    pointer written": it is recorded in ``unreadable`` AND in the returned
    faults, and the slot renders as an unreadable pointer, because a corrupt
    pointer rendered as an absent one says the slot has never promoted when
    in fact the trader's whole read surface is broken. The fault travels in
    its own map rather than as a field inside the pointer slot: the pointer
    document is the trader's contract, and a reader-state key smuggled into
    it (`{"unreadable": ...}`) is a second schema nobody declared
    (`alpha-engine-config-I9931` item 3).
    """
    out: dict[str, Any] = {}
    faults: dict[str, str] = {}
    for slot in ("u", "r", "m", "s"):
        # `keys.champion_key`, not a restatement of its shape. Handed over
        # from the I9807 sweep, which fixed the other five instances and could
        # not touch this file while this branch owned it. The module's own
        # docstring is the argument: "a key format restated at each call site
        # is a contract restated fifty times, and one of them has already
        # drifted" — and one of them had, in `explain.py`.
        key = champion_key(slot)
        read = _read(store, key)
        fault = _fault(read, key)
        if fault is not None:
            unreadable.append({"key": key, "fault": fault})
            faults[slot] = fault
            out[slot] = None
            continue
        out[slot] = read.document if read.document else None
    return out, faults


#: alpha-engine-config-I9757 (C14): the stylesheet used to hand-list CSS
#: rules against the fourteen component states only, so any OTHER status
#: rendered through the same `class="state s-{status}"` template — the
#: attribution table's own `OK`/`RED`/`GREEN`/`BREACH`/`N/A-NOT-RUN`/
#: `N/A-NOT-IMPL`, or a drift `Band` status — had no rule at all and
#: rendered in plain text, visually identical to a HEALTHY green row. A
#: fully-unmeasured report card was indistinguishable from a green one.
#:
#: `STATUS_COLORS` is the single source of every color a status can render
#: in; the stylesheet's `.s-*` rules are generated FROM it below rather than
#: hand-listed a second time, so the two structurally cannot drift apart.
#: `_state_class` (used at every render call site) raises for any status not
#: a key here — a rendered status with no color is refused rather than
#: silently rendered plain, per the fleet's no-silent-swallow rule.
_GREEN = "#1a7f37"
_GRAY = "#6e7781"
_AMBER = "#9a6700"
_RED = "#cf222e"
_PURPLE = "#8250df"

#: Attribution rows carry their own closed vocabulary, not `classify.STATES`
#: — owned by whichever module reduces the week's manifests into
#: `report/{trading_day}/attribution.json` (plan §4.5, track A; the handler
#: is currently a stub). No schema is registered for it yet, so this tuple
#: is the vocabulary the C14 reproduction demonstrated as actually
#: rendered — kept here, next to the stylesheet it must cover, until track A
#: lands a real schema this can import instead.
#: Every status an attribution row can carry. DERIVED from
#: `krepis.metrics.StatusLiteral` — the vocabulary `derive_status` returns —
#: plus the two the report adds itself. The hand-written form of this tuple
#: omitted `WATCH`, `N/A-LOW-N` and `N/A-MISSING-INPUT`, all three of which
#: `derive_status` returns and `crucible.report` therefore writes: with
#: `_state_class` now raising on an unregistered status, a WATCH row would
#: have taken the console down. A list of things to keep in sync is a list
#: that goes stale in the direction of omitting the case nobody hit yet.
ATTRIBUTION_STATUSES: tuple[str, ...] = ("OK", "BREACH", *get_args(StatusLiteral))

STATUS_COLORS: dict[str, str] = {
    # The fourteen component states (`crucible.console.classify.STATES`).
    "HEALTHY": _GREEN,
    "RUNNING": _GRAY,
    "ARMED": _GRAY,
    "DEGRADED": _AMBER,
    "FAILED": _RED,
    "STALLED": _RED,
    "MISSED": _RED,
    "ABSENT": _RED,
    "UNREPORTED": _RED,
    "UNREGISTERED": _RED,
    "NEVER_RAN": _RED,
    "DISABLED": _PURPLE,
    "DEPRECATED": _PURPLE,
    "RETIRED": _PURPLE,
    # `crucible.drift.Band.status()` — not yet rendered through a dedicated
    # Metrics section, colored here so that day does not repeat C14.
    "WATCH": _AMBER,
    # The phase ladder's own state vocabulary (`crucible.gate.LADDER_STATES`).
    # `UNMEASURED` is RED, not gray: a phase whose gate has never been read is
    # unobserved, not "fine so far" (principle 7). `OUT_OF_ORDER` is PURPLE —
    # it is not a phase running badly, it is the ladder itself being violated,
    # and giving it FAILED's red would hide it among ordinary unmet clauses.
    # `UNMEASURABLE` aliases `UNMEASURED`'s red — both are "we have nothing
    # trustworthy to show", the same posture `board.BOARD_CONSOLE_STATE`
    # takes mapping it to FAILED rather than a third color
    # (`alpha-engine-config-I9869` round 3, finding 4).
    "MET": _GREEN,
    "UNMET": _AMBER,
    "UNMEASURED": _RED,
    "UNMEASURABLE": _RED,
    "OUT_OF_ORDER": _PURPLE,
    # The attribution table's own vocabulary (`ATTRIBUTION_STATUSES`). `OK`
    # and `GREEN` alias `HEALTHY`'s color, `BREACH`/`RED` alias `FAILED`'s,
    # and the two `N/A-*` statuses are red rather than a neutral gray:
    # principle 7 — no data is never rendered as green, and a status meaning
    # "not measured" earns the same loud color as one meaning "measured and
    # bad", never the calm one.
    "OK": _GREEN,
    "GREEN": _GREEN,
    "RED": _RED,
    "BREACH": _RED,
    # Every `N/A-*` state is RED, not gray. Principle 7: a row that measured
    # nothing is unobserved, not healthy, and rendering it in the same colour
    # as a quiet-but-fine state is how `no data` becomes green by degrees.
    "N/A-NOT-RUN": _RED,
    "N/A-NOT-IMPL": _RED,
    "N/A-LOW-N": _RED,
    "N/A-MISSING-INPUT": _RED,
}

# Completeness guard, enforced at import time rather than left to be
# noticed at render time: every declared component state and every declared
# attribution status must have a color. A 15th component state (impossible
# under `console-policy`'s add-by-PR-only closed vocabulary, but checked
# anyway) or a new attribution status added to `ATTRIBUTION_STATUSES`
# without a matching entry here fails the import, not a rendered page.
_missing = (set(STATES) | set(ATTRIBUTION_STATUSES) | set(LADDER_STATES)) - set(STATUS_COLORS)
assert not _missing, (
    f"STATUS_COLORS is missing a rule for {sorted(_missing)} — every status in "
    "classify.STATES or ATTRIBUTION_STATUSES must have a color before it can render "
    f"({_C14_PHASE.tracker}, C14)."
)

_STYLE = (
    """
:root { color-scheme: light dark; --fg:#111; --bg:#fff; --muted:#666; --line:#ddd; }
@media (prefers-color-scheme: dark) {
  :root { --fg:#e8e8e8; --bg:#111; --muted:#999; --line:#333; }
}
body { font: 14px/1.5 ui-sans-serif, system-ui, sans-serif; color: var(--fg);
       background: var(--bg); margin: 0; padding: 24px; }
h1 { font-size: 20px; margin: 0 0 4px; }
h2 { font-size: 15px; margin: 28px 0 8px; }
.sub { color: var(--muted); margin: 0 0 16px; }
.wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--line);
         vertical-align: top; }
th { font-weight: 600; white-space: nowrap; }
code { font: 12px ui-monospace, monospace; }
.reason { color: var(--muted); max-width: 60ch; }
.state { font-weight: 600; white-space: nowrap; }
"""
    + "\n".join(f".s-{name} {{ color: {color}; }}" for name, color in STATUS_COLORS.items())
    + """
.gap { font-weight: 600; }
.gap-zero { color: #1a7f37; } .gap-nonzero { color: #cf222e; }
.empty { color: var(--muted); font-style: italic; }
"""
)


def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _state_class(status: Any) -> str:
    """The `s-{status}` class for a status, refusing one with no color.

    C14 was exactly the absence of this check: a status could reach the
    template and render with no stylesheet rule at all, silently
    indistinguishable from HEALTHY. Raising here is the fail-loud posture
    every producer in this repo takes — a console page that errors is a
    worse-looking but more honest outcome than one that quietly mis-colors
    an unmeasured row as green.
    """
    if status not in STATUS_COLORS:
        raise KeyError(
            f"status {status!r} has no entry in STATUS_COLORS, so it has no "
            "stylesheet rule — register a color for it before it can render "
            f"({_C14_PHASE.tracker}, C14)."
        )
    return f"s-{_e(status)}"


def render_html(page: ConsolePage) -> str:
    """One page, no JavaScript, no polling.

    The transparency-gap count is rendered at the top and is red whenever it
    is non-zero — including when it is non-zero for a good reason. §8.4 makes
    it the number that says the surface is complete, and a count that renders
    calmly is a count nobody acts on.
    """
    gap_class = "gap-zero" if page.unreported == 0 else "gap-nonzero"
    unreadable_class = "gap-zero" if not page.unreadable else "gap-nonzero"
    parts = [
        "<title>Crucible v2</title>",
        f"<style>{_STYLE}</style>",
        "<h1>Crucible v2</h1>",
        f'<p class="sub">Trading day <code>{_e(page.trading_day)}</code> · generated '
        f"<code>{_e(page.generated_utc)}</code> · week cost "
        f"<code>${page.week_cost_usd:.2f}</code> · population {page.population} · "
        f'transparency gap <span class="gap {gap_class}">{page.unreported}</span> '
        f'(objective 0) · unreadable artifacts <span class="gap {unreadable_class}">'
        f"{len(page.unreadable)}</span></p>",
        *_unreadable_section(page),
        *_ladder_section(page),
        "<h2>Components</h2>",
        '<div class="wrap"><table><thead><tr>'
        "<th>Component</th><th>State</th><th>Schedule</th><th>Deadline</th>"
        "<th>Absence watched by</th><th>Run</th><th>Cost</th><th>Why</th>"
        "</tr></thead><tbody>",
    ]
    for row in page.rows:
        cost = "" if row["cost_usd"] is None else f"${float(row['cost_usd']):.4f}"
        attempts = f" ({row['attempts']}x)" if (row["attempts"] or 1) > 1 else ""
        parts.append(
            "<tr>"
            f"<td><code>{_e(row['component'])}</code></td>"
            f'<td class="state {_state_class(row["state"])}">{_e(row["state"])}</td>'
            f"<td>{_e(row['schedule'])}</td>"
            f"<td>{_e(row['deadline'])}</td>"
            f"<td><code>{_e(row['absence_watched_by'])}</code></td>"
            f"<td><code>{_e(row['run_id'])}</code>{attempts}</td>"
            f"<td>{_e(cost)}</td>"
            f'<td class="reason">{_e(row["reason"])}</td>'
            "</tr>"
        )
    parts.append("</tbody></table></div>")

    parts.append("<h2>Attribution</h2>")
    if page.attribution:
        parts.append(
            '<div class="wrap"><table><thead><tr><th>Row</th><th>Value</th>'
            "<th>Status</th><th>Why</th></tr></thead><tbody>"
        )
        for row in page.attribution:
            parts.append(
                "<tr>"
                f"<td><code>{_e(row.get('name'))}</code></td>"
                f"<td>{_e(row.get('value'))} {_e(row.get('unit'))}</td>"
                f'<td class="state {_state_class(row.get("status"))}">{_e(row.get("status"))}</td>'
                f'<td class="reason">{_e(row.get("status_reason"))}</td>'
                "</tr>"
            )
        parts.append("</tbody></table></div>")
    else:
        # An unreadable attribution artifact is NOT an absent one, and saying
        # "crucible report has not written it" when it did and the bytes are
        # corrupt names the wrong remedy (alpha-engine-config-I9900).
        fault = next(
            (e for e in page.unreadable if str(e.get("key", "")).startswith("report/")), None
        )
        if fault is not None:
            parts.append(
                f'<p class="reason {_state_class("UNREPORTED")}">The attribution artifact '
                f"<code>{_e(fault.get('key'))}</code> exists and could not be read: "
                f"{_e(fault.get('fault'))}. This is a corrupt artifact, not an absent one.</p>"
            )
        else:
            parts.append(
                '<p class="empty">No attribution artifact for this trading day. This is an '
                "absence, not five rows of zero — <code>crucible report</code> has not "
                "written <code>report/{trading_day}/attribution.json</code>.</p>"
            )

    parts.append("<h2>Champions</h2>")
    parts.append(
        '<div class="wrap"><table><thead><tr><th>Slot</th><th>Champion</th>'
        "<th>Since</th></tr></thead><tbody>"
    )
    for slot, champ in sorted(page.champions.items()):
        if slot in page.champion_faults:
            parts.append(
                f"<tr><td><code>{_e(slot)}</code></td>"
                f'<td class="state {_state_class("UNREPORTED")}" colspan="2">unreadable: '
                f"{_e(page.champion_faults[slot])}</td></tr>"
            )
        elif champ:
            parts.append(
                f"<tr><td><code>{_e(slot)}</code></td>"
                f"<td><code>{_e(champ.get('arm_id'))}</code></td>"
                f"<td>{_e(champ.get('promoted_trading_day'))}</td></tr>"
            )
        else:
            parts.append(
                f"<tr><td><code>{_e(slot)}</code></td>"
                '<td class="empty" colspan="2">no pointer written</td></tr>'
            )
    parts.append("</tbody></table></div>")

    parts.append("<h2>Deploys</h2>")
    if page.deploys:
        parts.append(
            '<div class="wrap"><table><thead><tr><th>Trading day</th><th>Status</th>'
            "<th>Release</th><th>Why</th></tr></thead><tbody>"
        )
        for dep in page.deploys:
            state = "HEALTHY" if dep.get("status") == "ok" else "FAILED"
            parts.append(
                f"<tr><td>{_e(dep.get('trading_day'))}</td>"
                f'<td class="state {_state_class(state)}">{_e(dep.get("status"))}</td>'
                f"<td><code>{_e(str(dep.get('release_sha'))[:12])}</code></td>"
                f'<td class="reason">{_e(dep.get("reason"))}</td></tr>'
            )
        parts.append("</tbody></table></div>")
    else:
        parts.append('<p class="empty">No deploy manifests in the window.</p>')

    return "\n".join(parts)


def _unreadable_section(page: ConsolePage) -> list[str]:
    """Every artifact this page could not read, at the TOP of the page.

    Rendered above the ladder and the component table because it is the
    section that says how much of everything below it is trustworthy. It is
    omitted entirely when it is empty — the header already publishes the count
    as a zero, so an empty table here would be a second rendering of the same
    zero rather than a fact.
    """
    if not page.unreadable:
        return []
    parts = [
        "<h2>Unreadable artifacts</h2>",
        '<p class="sub">Read and could not be parsed. Every row below that depends on one '
        "of these keys is UNREPORTED, not green, and the page renders regardless — one "
        "unreadable artifact publishing no page at all is the defect this section "
        "exists to have prevented.</p>",
        '<div class="wrap"><table><thead><tr><th>Key</th><th>Fault</th></tr></thead><tbody>',
    ]
    for entry in page.unreadable:
        parts.append(
            "<tr>"
            f"<td><code>{_e(entry.get('key'))}</code></td>"
            f'<td class="reason {_state_class("UNREPORTED")}">{_e(entry.get("fault"))}</td>'
            "</tr>"
        )
    parts.append("</tbody></table></div>")
    return parts


def _ladder_section(page: ConsolePage) -> list[str]:
    """The phase ladder, rendered at the TOP of the page.

    Above Components deliberately: "which phase is this rebuild on, and is
    anything being graded out of order" is the question the whole surface
    exists to answer, and it is the one that was previously answerable only by
    reading three GitHub issue comments and knowing which of them superseded
    the others.
    """
    ladder = page.phase_ladder
    if not ladder:
        # UNREPORTED, in the page's own red, never `empty`'s gray and never a
        # recomputed ladder (`alpha-engine-config-I10575`). The console is a
        # CONSUMER of `gates/ladder.json`; when that key is absent, unreadable,
        # malformed or older than this page's trading day, the honest reading
        # is "we have nothing trustworthy to show" plus the cause — principle
        # 7: no data is never rendered as green.
        cause = page.phase_ladder_fault or (
            f"<code>{_e(LADDER_KEY)}</code> was not read for this page."
        )
        return [
            "<h2>Phase ladder</h2>",
            '<p class="sub">'
            f'<span class="state {_state_class("UNREPORTED")}">UNREPORTED</span> · '
            "the console reads <code>{key}</code> and never computes it; its one "
            "producer is <code>crucible gate.close</code> / "
            "<code>crucible gate --publish</code></p>".format(key=_e(LADDER_KEY)),
            f'<p class="reason {_state_class("UNREPORTED")}">{_e(cause)}</p>',
        ]
    out_of_order = ladder.get("out_of_order") or []
    parts = [
        "<h2>Phase ladder</h2>",
        f'<p class="sub">At <code>{_e(ladder.get("current_phase"))}</code> · '
        f"{_e(ladder.get('phases_met'))}/{_e(ladder.get('phases_total'))} gates met · "
        f'<span class="gap {"gap-zero" if not ladder.get("unmeasured") else "gap-nonzero"}">'
        f"{_e(ladder.get('unmeasured'))} unmeasured</span> · "
        f'<span class="gap {"gap-zero" if not out_of_order else "gap-nonzero"}">'
        f"{len(out_of_order)} out of order</span></p>",
        '<div class="wrap"><table><thead><tr><th>Phase</th><th>State</th>'
        "<th>Clauses</th><th>Last read</th><th>Tracker</th><th>Why</th>"
        "</tr></thead><tbody>",
    ]
    for row in ladder.get("phases", []):
        met, total = row.get("clauses_met"), row.get("clauses_total")
        # `never measured`, never `0/0` and never a blank cell: an unread gate
        # and a gate that read zero met clauses are different facts.
        clauses = "never measured" if total is None else f"{met}/{total}"
        read_on = row.get("read_on") or "never"
        parts.append(
            "<tr>"
            f"<td><code>{_e(row.get('phase'))}</code> {_e(row.get('title'))}</td>"
            f'<td class="state {_state_class(row.get("state"))}">{_e(row.get("state"))}</td>'
            f"<td>{_e(clauses)}</td>"
            f"<td>{_e(read_on)}</td>"
            f'<td><a href="{_e(row.get("tracker_url"))}"><code>'
            f"{_e(row.get('tracker'))}</code></a></td>"
            f'<td class="reason">{_e(row.get("detail"))}</td>'
            "</tr>"
        )
    parts.append("</tbody></table></div>")
    return parts


def write_page(store: Store, page: ConsolePage) -> tuple[str, ...]:
    """Write the HTML and the JSON. Returns the keys.

    The JSON is not an extra: `console-policy` requires every view to serve
    the JSON an agent reads, and a page whose numbers can only be scraped out
    of HTML is a page the next automated reader re-derives incorrectly.

    **`gates/ladder.json` is NOT written here** (`alpha-engine-config-I10575`).
    The ladder has exactly one producer — `crucible gate --publish` /
    `crucible gate.close`, via `crucible.gate.ladder_payload` — and this
    function used to be a second one, re-evaluating every phase gate under
    the console runtime's own environment and republishing the result over
    the gate publisher's. That runtime deliberately lacks
    `CRUCIBLE_MUTED_TOPIC`, `CRUCIBLE_CLOUDTRAIL_ARCHIVE` and
    `ce:GetCostAndUsage`, so on the first live Saturday (2026-09-12) the
    11:04Z console run overwrote the 02:06Z `phase2` 7/10 ladder with a
    `phase0` UNMEASURABLE 0/5 one and reported `status: ok`. The fleet
    console's `s3-records` adapter still reads the key directly; it now reads
    the only reading there is.
    """
    store.put_bytes(CONSOLE_KEY, render_html(page).encode("utf-8"))
    store.put_bytes(CONSOLE_JSON_KEY, page.to_json())
    return CONSOLE_KEY, CONSOLE_JSON_KEY
