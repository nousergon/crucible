"""Two page conditions, one heartbeat, and nothing else.

Normative source: plan §4.6, §9.3.

    1. **Absence** — a scheduled job's manifest does not exist by its deadline.
    2. **Failure** — a manifest exists with `status: failed`.

That is the exhaustive list. Everything else is a row on a page nobody is
paged to. The v1 system carried 185 alert rules and 153 alarms and still
missed three consecutive Saturday failures, because the signal was buried in
the ones that fire for nothing.

**The deadline table is declarative and lives in `components.yaml`**, ≈6 rows
rather than 185, and it is read from there rather than restated here — a
contract restated in two places has already drifted.

**Deadlines are trading-calendar-relative** (§4.12): "present by *T* after the
close of trading day *d*". A Monday holiday therefore raises no page, and
nobody has to mute anything to get that.

**Alert count is itself a metric with a ceiling.** Exceeding it is a defect in
this module, never a reason to add a suppression — and this repository carries
no suppression collection at all (§11.1).

**The heartbeat is not a third page condition.** It is the answer to "is the
alerting path itself alive": a component emitting nothing is unobserved, not
healthy, and that includes the thing whose job is to notice silence.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from crucible.calendar import (
    TRADING_DAYS_PER_WEEK,
    previous_trading_day,
    resolve_trading_day,
)
from crucible.components import Component, load_registry, scheduled_components
from crucible.documents import load_store_document, read_listed_document, read_manifests_under
from crucible.keys import (
    ALERTS_ROOT,
    RUNS_ROOT,
    is_manifest_key,
    parse_bus_key,
    parse_manifest_key,
)
from crucible.manifest import manifest_prefix
from crucible.store import Store

__all__ = [
    "ALERT_BUS_SCHEMA_VERSION",
    "CATCH_UP_TRADING_DAYS",
    "CAUSE_MATCHERS",
    "CEILING_WINDOW_TRADING_DAYS",
    "MUTED_TOPIC",
    "MUTED_TOPIC_ARN_VAR",
    "PAGES_PER_MONTH_CEILING",
    "PAGES_TOPIC",
    "PAGES_TOPIC_ARN_VAR",
    "PAGE_CONDITIONS",
    "PENDING_CONFIRMATION",
    "SUBSCRIBERS_METRIC",
    "SWEEP_JOB",
    "Page",
    "PageGroup",
    "SubscriberReading",
    "StoreAccessError",
    "TopicUnresolvedError",
    "bus_key",
    "bus_row",
    "cause_key",
    "ceiling_metric",
    "days_to_evaluate",
    "dedup_key",
    "emit",
    "evaluate_absence",
    "evaluate_failure",
    "group_pages",
    "heartbeat",
    "incident_id",
    "incident_key",
    "pages_in_range",
    "pages_in_window",
    "pages_topic_subscribers",
    "send",
    "sweep",
    "topic_arn",
]

#: Exhaustive. Adding a third member is a design change, and it is visible in
#: a diff as one — which is the point of writing it down as a closed set
#: rather than as prose in a runbook.
PageCondition = Literal["absence", "failure"]
PAGE_CONDITIONS: tuple[str, ...] = ("absence", "failure")


@dataclass(frozen=True)
class Page:
    """One page. Carries its correlation identity, or it is unactionable.

    ``run_id`` is None only for an ABSENCE page — there was no run, so there
    is no id. That is the one legitimate case, and it is why the field is
    optional rather than why it is often missing.
    """

    condition: PageCondition
    job: str
    trading_day: dt.date
    reason: str
    run_id: str | None = None

    def __post_init__(self) -> None:
        if self.condition not in PAGE_CONDITIONS:
            raise ValueError(
                f"{self.condition!r} is not a page condition. There are exactly two "
                f"({PAGE_CONDITIONS}); a third is a design change, not a config value."
            )
        if not self.reason.strip():
            raise ValueError(
                f"a {self.condition} page for {self.job} on {self.trading_day} carries "
                "no reason. A page an operator cannot act on is noise with an alarm "
                "attached."
            )
        if self.condition == "failure" and self.run_id is None:
            raise ValueError(
                "a failure page must carry the run_id of the failed run — the manifest "
                "exists, so its correlation identity does. Only an ABSENCE page "
                "legitimately has none."
            )


def dedup_key(condition: str, subject: str, trading_day: dt.date) -> str:
    """One key per (condition, incident SUBJECT, trading day). §4.6.

    Not per attempt and not per condition-instance: a job that fails, is
    retried by the declared transient class and fails again is ONE incident,
    and paging twice for it is how a two-page-per-month ceiling is blown by a
    single bad Saturday.

    ``subject`` is what the incident is ABOUT — the cause its members share
    (:func:`incident_key` supplies it from the group's own cause key), never
    one member's job. The defect this parameter's name records: the key was
    built from ``group.members[0].job``, the alphabetically-first member, so
    it CHANGED as membership changed. One absent Friday artifact observed by
    three consecutive nightly sweeps produced two different keys — the
    forever-dedup window never engaged, and one incident paged three times
    against a ceiling of two a month.
    """
    return f"{condition}:{subject}:{trading_day.isoformat()}"


@dataclass(frozen=True)
class PageGroup:
    """One page, N members, one cause.

    §9.3: "five arms failing on one data outage is one page naming five
    members". The group is the unit that is delivered and the unit that is
    counted against the ceiling — counting members instead would make one
    outage look like five incidents and would blow a two-page-a-month target
    on a single bad Saturday.
    """

    cause_key: str
    members: tuple[Page, ...]

    def __post_init__(self) -> None:
        if not self.members:
            raise ValueError("a page group with no members is not a page")

    @property
    def condition(self) -> str:
        return self.members[0].condition

    @property
    def trading_day(self) -> dt.date:
        return self.members[0].trading_day

    @property
    def severity(self) -> str:
        """Both conditions are incident-tier.

        There is no warning tier here: a condition that does not warrant a
        push is a console row, and §4.6 says everything that is not one of
        the two conditions IS a console row. A third severity would be a
        third page condition wearing a different field name.
        """
        return "error"

    def render(self) -> str:
        jobs = ", ".join(sorted({m.job for m in self.members}))
        head = (
            f"[crucible-v2] {self.condition.upper()} on {self.trading_day.isoformat()} "
            f"({len(self.members)} member{'s' if len(self.members) > 1 else ''}): {jobs}"
        )
        lines = [head, f"cause: {self.cause_key}"]
        for m in sorted(self.members, key=lambda x: x.job):
            run = f" run_id={m.run_id}" if m.run_id else ""
            lines.append(f"  - {m.job}:{run} {m.reason}")
        return "\n".join(lines)


# ── The causal grouping key ───────────────────────────────────────────────
#
# What "one cause" means, stated as data rather than as prose in a runbook.
# Exhaustive and ordered: the first matcher whose substring appears in the
# reason claims the page. A reason matching nothing groups on the job itself,
# which degrades to one page per job — the pre-grouping behaviour — rather
# than to one page for everything, because collapsing unrelated failures into
# a single page would hide N-1 of them.
CAUSE_MATCHERS: tuple[tuple[str, str], ...] = (
    ("data_source_unavailable", "data source"),
    ("data_source_unavailable", "yfinance"),
    ("data_source_unavailable", "fred"),
    ("router_unavailable", "router"),
    ("router_unavailable", "provider_5xx"),
    ("router_unavailable", "provider_timeout"),
    ("spot_interruption", "spot_interruption"),
    ("stale_release_pointer", "StaleReleasePointerError"),
    ("s3_unavailable", "s3_throttling"),
)


def cause_key(page: Page) -> str:
    """The key N members share when they failed for one reason.

    An ABSENCE page groups on the deadline it missed, not on a reason: every
    absent manifest for one trading day is the same operator action — go and
    find out why nothing ran — and paging separately for six of them on a
    morning when the scheduler was down is the noise that buries the signal.
    """
    if page.condition == "absence":
        return f"absence:{page.trading_day.isoformat()}"
    haystack = page.reason.lower()
    for key, needle in CAUSE_MATCHERS:
        if needle.lower() in haystack:
            return f"{key}:{page.trading_day.isoformat()}"
    return f"{page.job}:{page.trading_day.isoformat()}"


def group_pages(pages: Sequence[Page]) -> list[PageGroup]:
    """Collapse pages onto their shared causes, deterministically ordered."""
    buckets: dict[str, list[Page]] = {}
    for page in pages:
        buckets.setdefault(cause_key(page), []).append(page)
    return [
        PageGroup(cause_key=key, members=tuple(sorted(buckets[key], key=lambda p: p.job)))
        for key in sorted(buckets)
    ]


# ── Incident identity ─────────────────────────────────────────────────────
#
# One INCIDENT, not one observation of it. The sweep runs nightly; a wrongness
# that persists is observed by every sweep until it clears, and §2 row 6
# ("alerts only when wrong") means a persistent, unchanged wrongness is ONE
# alert. Everything downstream — the transport dedup key, the bus key, and
# therefore the ceiling metric — is derived from this identity rather than
# from the observation that happened to notice it.

#: A store-key segment. Deliberately narrow: an incident id becomes a path
#: component, and a subject carrying `/`, `:` or a date would either break
#: the key or smuggle a second date past
#: :meth:`Store.assert_keys_bind_to_trading_days`.
INCIDENT_ID_RE = re.compile(r"[A-Za-z0-9_.]+")


def _incident_subject(group: PageGroup) -> str:
    """The cause half of ``group.cause_key`` — what the incident is about.

    :func:`cause_key` renders ``"{subject}:{trading_day}"``. The trading day
    is carried separately by every consumer here, so it is split off rather
    than repeated. A cause key with no separator at all (a hand-built group
    in a test) is its own subject; that is a defined answer rather than a
    raise, because the subject is only ever an identity component and a
    hand-built one is still stable.
    """
    subject, separator, _day = group.cause_key.rpartition(":")
    return subject if separator else group.cause_key


def incident_key(group: PageGroup) -> str:
    """The stable identity of one incident. §4.6, §2 row 6.

    Stable across SWEEPS — the same absence seen on Friday, Saturday and
    Sunday nights yields one key — and stable across MEMBERSHIP changes,
    because it is derived from the shared cause rather than from whichever
    member happens to sort first. Those are the two properties the previous
    key lacked, and between them they are why one absent artifact could page
    three times.
    """
    return dedup_key(group.condition, _incident_subject(group), group.trading_day)


def incident_id(group: PageGroup) -> str:
    """:func:`incident_key` as ONE store-key segment.

    The trading day is dropped because the bus key already carries it, and a
    second date inside the segment would be a second date in the key.

    Raises rather than sanitising an unexpected subject: silently rewriting
    two distinct subjects into one id would merge two incidents into one bus
    row, which is the failure this whole identity exists to prevent.
    """
    subject = _incident_subject(group)
    if not INCIDENT_ID_RE.fullmatch(subject):
        raise ValueError(
            f"{subject!r} is not usable as an incident id segment (expected "
            f"{INCIDENT_ID_RE.pattern}). Sanitising it would risk mapping two distinct "
            "causes onto one bus row, which would hide one of them entirely."
        )
    return f"{group.condition}.{subject}"


# ── The days a sweep is answerable for ────────────────────────────────────

#: The name of the sweep's own registry row. Used to read the sweep's own
#: manifests — which is how the sweep knows which days it did not run.
SWEEP_JOB = "alerts.sweep"

#: How far back a sweep looks for days it did not run. One trading week.
#:
#: Both page conditions used to evaluate ONLY ``resolve_trading_day(now)``, so
#: a failed or absent artifact on a day the sweep did not run was never paged
#: once the trading day advanced past it — a permanent blind spot, not a
#: delayed page. A gap longer than a week is the sweep itself being down,
#: which is the heartbeat's row to raise (§9.3), not something a longer
#: window here would fix.
CATCH_UP_TRADING_DAYS = TRADING_DAYS_PER_WEEK


def days_to_evaluate(
    store: Store,
    moment: dt.datetime,
    *,
    window_trading_days: int = CATCH_UP_TRADING_DAYS,
) -> list[dt.date]:
    """Today's trading day, plus every day in the window since the sweep's
    first observed run — whether or not it ran on them.

    The sweep writes its own ``run.json`` like every other job, so the day it
    first existed is readable from the store rather than inferred. Everything
    at or after that day is re-evaluated on every pass, because a pass over
    day *d* is a reading taken at one instant and not a verdict on *d*: a
    manifest written after it, and a deadline that had not yet arrived when
    it looked, are both facts about *d* that the pass could not have seen.
    Restricting this to the days the sweep MISSED made those facts
    permanently invisible (alpha-engine-config-I9960; see the body).

    **Bounded below by the sweep's own first observed run in the window.**
    The sweep was not blind on a day before it existed; claiming otherwise
    would make a cold start report a week of absences it was never there to
    watch, and a first run that BREACHES the ceiling is a false reading of
    the one metric §11 risk 2 holds this module to.

    Re-paging is impossible by construction, not by luck: a caught-up day
    resolves to the same :func:`incident_key` the missed sweep would have
    produced, and :func:`emit` records a second observation of an open
    incident rather than sending again.
    """
    today = resolve_trading_day(moment)
    candidates: list[dt.date] = []
    day = today
    for _ in range(window_trading_days):
        day = previous_trading_day(day)
        candidates.append(day)
    # `alerts.sweep` now discriminates its own manifest by `calendar_date`
    # (alpha-engine-config-I9781), so a trading day can carry several — this
    # only needs to know whether ANY exist, hence a prefix listing rather
    # than an exact-key check.
    # `is_manifest_key`, not `any(...)` over the raw listing: a manifest prefix
    # is a namespace and a job may file its own evidence beside its manifest
    # (`report.morning` writes `message.txt` there). Counting a non-manifest
    # object as "the sweep ran" would delete a real missed day from the backlog
    # (alpha-engine-config-I9900).
    ran = {
        d
        for d in candidates
        if any(
            is_manifest_key(k) for k in store.list_keys(manifest_prefix(SWEEP_JOB, d.isoformat()))
        )
    }
    if not ran:
        # No evidence the sweep ran on any day in the window: either it is a
        # cold start or the sweep has been down for longer than the window,
        # and the second is the heartbeat's finding (§9.3) rather than a
        # backlog this run should invent.
        return [today]
    # EVERY candidate at or after the first observed run — not only the ones
    # the sweep missed. "The sweep ran on day d" is not "day d was fully
    # evaluated": both conditions are evaluated against `moment`, and a row
    # whose deadline is anchored `next_calendar_day_at` is ALWAYS still in
    # the future when the 21:00 ET sweep looks at its own trading day. Those
    # rows were therefore never absence-checked by any sweep, ever — the day
    # they came due was the one day the old set excluded, precisely because
    # the sweep had run on it (alpha-engine-config-I9960).
    #
    # MEASURED 2026-09-04: `report.morning`'s 13:00Z occurrence for trading
    # day 2026-09-03 did not deliver; `runs/report.morning/2026-09-03/` was
    # empty and `runs/alerts.sweep/2026-09-03/2026-09-04/run.json` existed,
    # so the missed report was structurally unreachable by every future
    # sweep. `data.weekly` (`next_calendar_day_at 09:00`) carries the same
    # shape. This is the class, not the instance: any fact about day d that
    # becomes true after the sweep's own pass over d — a late manifest, a
    # deadline that had not yet arrived — was invisible forever.
    #
    # Re-paging is still impossible by construction: a re-evaluated day
    # resolves to the same :func:`incident_key`, and :func:`emit` records a
    # second observation of an open incident rather than sending again.
    first_seen = min(ran)
    return sorted({today, *(d for d in candidates if d >= first_seen)})


class StoreAccessError(RuntimeError):
    """One or more manifest prefixes could not be LISTED at all.

    Distinct from "listed, and nothing was there", which is an absence page.
    A listing that fails is a statement about our access, not about the
    system being measured — the same distinction
    :func:`crucible.gate._list_store_keys` draws, mirrored here rather than
    collapsed: an absence page raised on a permissions error would name a job
    that may well have delivered, and an operator who acts on it twice stops
    reading the third one.

    It is raised, never returned as a page: §4.6 admits exactly two page
    conditions, and "we could not tell" is neither. The sweep's own manifest
    is where it lands, `status: failed` with this as the reason, which the
    FAILURE condition then pages on — one surface, the declared one.
    """


@dataclass(frozen=True)
class _KeysRead:
    """A prefix listing, or the reason it could not be taken. Never both."""

    keys: tuple[str, ...] | None
    problem: str | None


def _list_manifest_keys(store: Store, prefix: str) -> _KeysRead:
    try:
        keys = tuple(store.list_keys(prefix))
    except Exception as exc:
        return _KeysRead(
            None,
            f"listing {prefix!r} could not be read: {type(exc).__name__}: {exc}. That is a "
            "statement about our access, not about the job being watched",
        )
    return _KeysRead(keys, None)


# ── The two page conditions ───────────────────────────────────────────────


def evaluate_absence(
    store: Store,
    *,
    now: dt.datetime | None = None,
    registry: dict[str, Component] | None = None,
    watched_by: str = SWEEP_JOB,
    access_faults: list[str] | None = None,
) -> list[Page]:
    """Page for every scheduled job whose manifest is missing past its deadline.

    Reads the deadline table from `components.yaml` through
    :mod:`crucible.components` — never a second copy — and resolves each
    deadline against the trading calendar, so a market holiday moves the
    deadline rather than producing a page that has to be dismissed.

    **Only past deadlines are evaluated.** A job whose deadline is still in
    the future is not absent, it is not due; reporting it would make the
    absence condition fire every time the sweep ran early.

    **Only the rows this watcher owns are evaluated, by declaration.**
    ``watched_by`` is matched against the registry's `absence_watched_by`, so
    the split is read from the file rather than being a name embedded in this
    function. The sweep passes the default and therefore never evaluates
    `alerts.sweep` itself; :func:`heartbeat` passes ``"heartbeat"`` and
    evaluates exactly that row, which is what makes the registry's declared
    watcher a real one rather than a claim.

    **Every day the sweep is answerable for**, not only today's — see
    :func:`days_to_evaluate`. A missed sweep day used to be a permanent
    blind spot; it is now a caught-up page carrying the day it belongs to.

    ``access_faults``: pass a list to collect "could not ask" problems and
    keep evaluating; leave it ``None`` and the first one raises
    :class:`StoreAccessError`. A caller that pages (:func:`sweep`,
    :func:`heartbeat`) passes the list, so one unreadable prefix cannot
    withhold the pages the same pass already found, and then raises once it
    has emitted them. A caller that only wants the reading gets the loud
    default.
    """
    moment = (now or dt.datetime.now(dt.UTC)).astimezone(dt.UTC)
    reg = scheduled_components(registry)
    pages: list[Page] = []
    for trading_day in days_to_evaluate(store, moment):
        for name, component in sorted(reg.items()):
            if component.absence_watched_by != watched_by:
                continue
            assert component.deadline is not None  # Component.__post_init__ guarantees it
            due = component.deadline.due_at(trading_day)
            if moment < due:
                continue
            # A prefix listing, not an exact-key check: a job that carries a
            # discriminator (experiment.run/grade by slot, alerts.sweep by
            # calendar_date) writes ANY number of manifests under this
            # trading day, and absence means none of them exist — not that
            # the one bare key is missing (alpha-engine-config-I9781).
            # Narrowed to MANIFEST keys: a job's own evidence filed beside its
            # manifest (`report.morning`'s `message.txt`) is not a manifest,
            # and letting it satisfy this check would suppress a real absence
            # page for the one job that files evidence (alpha-engine-config-I9900).
            #
            # Guarded, and the guard is the point (alpha-engine-config-I9960):
            # a listing that RAISES used to abort the whole pass, so one
            # unreadable prefix withheld every real page the same run had
            # already found. The access problem is now carried out to the
            # caller — which pages what it found, then fails loudly naming
            # every prefix it could not read — while a prefix we could not
            # ask about produces no page, because we did not observe an
            # absence there.
            prefix = manifest_prefix(name, trading_day.isoformat())
            read = _list_manifest_keys(store, prefix)
            if read.problem is not None:
                if access_faults is None:
                    raise StoreAccessError(read.problem)
                access_faults.append(read.problem)
                continue
            if any(is_manifest_key(k) for k in read.keys or ()):
                continue
            pages.append(
                Page(
                    condition="absence",
                    job=name,
                    trading_day=trading_day,
                    reason=(
                        f"no manifest under {manifest_prefix(name, trading_day.isoformat())}; due "
                        f"{component.deadline.describe(trading_day)} "
                        f"({due.strftime('%Y-%m-%dT%H:%M:%SZ')}), now "
                        f"{moment.strftime('%Y-%m-%dT%H:%M:%SZ')}"
                    ),
                )
            )
    return pages


def evaluate_failure(
    store: Store,
    *,
    now: dt.datetime | None = None,
    registry: dict[str, Component] | None = None,
    access_faults: list[str] | None = None,
) -> list[Page]:
    """Page for every manifest written with `status: failed`.

    The page's reason is the manifest's `reason` VERBATIM. Re-deriving a
    cause here would let the page and the artifact disagree, and an operator
    reading two accounts of one failure has to work out which is authoritative
    before doing anything about it.

    A manifest that will not parse or will not validate is itself a failure
    page: an unreadable manifest is indistinguishable from a lie, and reading
    past it would drop the very run most likely to be broken.

    **Every day the sweep is answerable for**, not only today's — see
    :func:`days_to_evaluate`. A `status: failed` manifest written on a day
    the sweep did not run used to go unpaged forever once the trading day
    advanced past it.
    """
    moment = (now or dt.datetime.now(dt.UTC)).astimezone(dt.UTC)
    reg = registry if registry is not None else load_registry()
    pages: list[Page] = []
    for trading_day in days_to_evaluate(store, moment):
        for name, component in sorted(reg.items()):
            if component.lifecycle != "ACTIVE":
                continue
            # A job that carries a discriminator (experiment.run/grade by
            # slot, alerts.sweep by calendar_date) can have written several
            # manifests under this trading day; each is its own writer and
            # each is checked, rather than the single bare key that used to
            # be the only place a failure could be recorded — and the only
            # one four colliding writers could share (alpha-engine-config-I9781).
            # One guarded prefix reader (`crucible.documents`), never a listing
            # parsed key by key: it keeps manifests only — `report.morning`'s
            # delivered `message.txt` is filed under its own manifest prefix
            # by design and used to page a FAILURE for a job that succeeded,
            # every night (alpha-engine-config-I9900) — and it turns a manifest
            # whose body is an array or a string into a fault instead of an
            # `AttributeError` out of the sweep (alpha-engine-config-I9931).
            listed = read_manifests_under(store, manifest_prefix(name, trading_day.isoformat()))
            if listed.listing_problem is not None:
                # Same split as the absence condition (I9960): a prefix we
                # could not LIST tells us nothing about whether a manifest
                # there says `failed`, and it must not abort the rows we can
                # still read. Carried out; raised by the caller after it has
                # paged what it observed.
                if access_faults is None:
                    raise StoreAccessError(listed.listing_problem)
                access_faults.append(listed.listing_problem)
                continue
            for key, problem in sorted(listed.faults.items()):
                pages.append(
                    Page(
                        condition="failure",
                        job=name,
                        trading_day=trading_day,
                        reason=f"manifest at {key} is unreadable: {problem}",
                        run_id=_UNPARSEABLE_RUN_ID,
                    )
                )
            for _key, manifest in listed.documents:
                if manifest.get("status") == "failed":
                    pages.append(
                        Page(
                            condition="failure",
                            job=name,
                            trading_day=trading_day,
                            reason=manifest.get("reason") or "(the manifest recorded no reason)",
                            run_id=manifest.get("run_id") or _UNPARSEABLE_RUN_ID,
                        )
                    )
    return pages


#: A failure page needs a run_id (Page.__post_init__), and a manifest that
#: will not parse has none to give. This sentinel is that fact stated, rather
#: than the check relaxed: relaxing it would let every failure page ship
#: without correlation identity, to accommodate the rarest case.
_UNPARSEABLE_RUN_ID = "0" * 26


# ── The bus, the transport, and the ceiling ───────────────────────────────


def bus_key(group: PageGroup) -> str:
    """`alerts/{trading_day}/{incident_id}.json` (§9.3).

    Stays here rather than in `crucible.keys` (alpha-engine-config-I9807
    class sweep): the key is a function of `incident_id(group)`, this
    module's own derived identity for a `PageGroup` — moving `bus_key` alone
    would either duplicate that derivation in `crucible.keys` or make the
    generic key module import alert-domain logic, which is backwards.

    **One row per INCIDENT, not per observation.** §9.3 writes the shape as
    `alerts/{date}/{run_id}.json`; keying on the observation's id meant the
    nightly sweep wrote a fresh row for the same unchanged absence every
    night — three rows, three transport sends and three counts against a
    two-a-month ceiling for one absent artifact. The delta is deliberate and
    it costs nothing a reader needs: `alert_id` and every member's `run_id`
    are fields ON the row, so the join to the manifest §9.3 asks for is
    intact, while the key is now the thing the ceiling should be counting.
    """
    return f"alerts/{group.trading_day.isoformat()}/{incident_id(group)}.json"


def bus_row(
    group: PageGroup,
    *,
    alert_id: str | None,
    sent: bool,
    destination: str,
    first_observed_utc: str,
    last_observed_utc: str,
    observations: int = 1,
) -> dict[str, Any]:
    """The machine-readable row. §7.3: a human-only alert is invisible.

    ``sent`` records what actually happened on the transport, not what was
    intended. A row claiming delivery for a page that never left is worse
    than no row: the response plane would read it as handled.

    ``observations`` is how many sweeps have seen this incident still open.
    It is what makes one row per incident lossless: "absent since Friday,
    seen by three sweeps" is strictly more information than three rows that
    each look like a separate incident.
    """
    return {
        "schema_version": ALERT_BUS_SCHEMA_VERSION,
        "alert_id": alert_id,
        "alert_id_is_run_id": _alert_id_joins_to_a_manifest(group, alert_id),
        "condition": group.condition,
        "cause_key": group.cause_key,
        "trading_day": group.trading_day.isoformat(),
        "dedup_key": incident_key(group),
        "incident_key": incident_key(group),
        # `members` and `rendered` are a record of the page that WENT OUT and
        # are never rewritten; `members_now` is the latest observation. An
        # incident that grows from three arms to five is one incident, and
        # both facts are worth keeping — but not in one field, where a reader
        # could not tell which vintage they were holding.
        "members": _members(group),
        "members_now": _members(group),
        "sent": sent,
        "destination": destination,
        "first_observed_utc": first_observed_utc,
        "last_observed_utc": last_observed_utc,
        "observations": observations,
        "rendered": group.render(),
    }


def _members(group: PageGroup) -> list[dict[str, Any]]:
    return [
        {"job": m.job, "run_id": m.run_id, "reason": m.reason}
        for m in sorted(group.members, key=lambda x: x.job)
    ]


def _alert_id_joins_to_a_manifest(group: PageGroup, alert_id: str | None) -> bool:
    """Does ``alert_id`` actually join to a run manifest's `run_id`?

    §9.3 says the response plane reads this bus, so this is the field that
    decides whether a machine reader goes looking for a manifest. It was
    wrong in both directions at once: `bus_row` computed it as
    ``alert_id != _sweep_alert_id_marker(alert_id)`` where that helper was
    the IDENTITY function, so it was always False even for a real run id;
    and :func:`emit` then overwrote it with ``condition == "failure"``, which
    is True for a failure page whose manifest would not parse and whose
    "run id" is therefore the all-zeros sentinel joining to nothing.

    Computed once, here, from the only three facts that decide it: an
    absence has no run, a sentinel is not a run id, and the id must be one
    the group's own members carry.
    """
    if alert_id is None or group.condition != "failure":
        return False
    if alert_id == _UNPARSEABLE_RUN_ID:
        return False
    return any(member.run_id == alert_id for member in group.members)


#: Bumped from `alert_bus.v1` with the incident-keyed row: the key shape
#: changed and `first_observed_utc`, `last_observed_utc`, `observations` and
#: `incident_key` are new. A consumer pinned to v1 should see a version it
#: does not know rather than a v1-shaped row that means something else.
ALERT_BUS_SCHEMA_VERSION = "alert_bus.v2"

#: §11 risk 2 / §2 row 6: pages per month is a metric with a CEILING.
#: Exceeding it is a defect in this module — never a reason to add a
#: suppression, and this repository carries no suppression collection at all.
PAGES_PER_MONTH_CEILING = 2

#: Trading days in the window the ceiling is measured over (§4.12: every
#: window is trading days). Four trading weeks.
CEILING_WINDOW_TRADING_DAYS = 20

#: Where anything belonging to the SUPERSEDED v1 system is routed during the
#: three-week overlap (§11 risk 5). Muting at the destination rather than at
#: the producer: a mute implemented by not emitting is indistinguishable from
#: a producer that died.
#:
#: A topic NAME, and named as one. It was previously passed straight into
#: `krepis.alerts.publish(sns_topic_arn=...)`, whose resolver returns an
#: explicit value VERBATIM — so the legacy path would have handed SNS a bare
#: name where an ARN is required and taken an `InvalidParameter`. Resolved
#: through :func:`topic_arn` now, like every other topic.
MUTED_TOPIC = "alpha-engine-alerts-muted"

#: The topic v2 pages go to. Created, tagged and exported by the `crucible-v2`
#: CloudFormation stack, and the ONLY topic besides :data:`MUTED_TOPIC` the
#: v2 RuntimeRole is granted `sns:Publish` on.
PAGES_TOPIC = "crucible-v2-pages"

#: The declared adapter's inputs (principle 8): the topic ARN is read from
#: the environment, never composed from a literal here and never left to a
#: provider default. `CRUCIBLE_PAGES_TOPIC_ARN` carries the `crucible-v2`
#: stack's `PagesTopicArn` output; a `crucible.config` setting for it is the
#: right long-term home and is tracked on alpha-engine-config-I9757.
PAGES_TOPIC_ARN_VAR = "CRUCIBLE_PAGES_TOPIC_ARN"
MUTED_TOPIC_ARN_VAR = "CRUCIBLE_MUTED_TOPIC_ARN"


class TopicUnresolvedError(RuntimeError):
    """The SNS half of a page has no topic it is allowed to publish to.

    Raised rather than falling through to `krepis.alerts`' own default. That
    default is `alpha-engine-alerts`, which the v2 RuntimeRole is NOT granted
    — so every page's SNS half would have been `AccessDenied` on the day the
    stack was applied, while Telegram succeeded, `any_ok` stayed True and
    nothing failed loudly. A page delivered on one of its two channels, with
    no surface saying so, is the shape of failure this module exists to
    remove from everything else.
    """


def topic_arn(*, legacy: bool = False) -> str | None:
    """The SNS topic ARN this process publishes to. ONE resolution point.

    Principle 8: no topic ARN, account id or region is composed at a call
    site. The value comes from the environment — an output of the stack that
    grants the publish, so the grant and the target cannot drift apart — and
    this function is the whole adapter.

    Returns ``None`` when unset, which :func:`_krepis_publish` turns into a
    :class:`TopicUnresolvedError` at the point of a REAL send. The check is
    there rather than here so that an injected transport (the fault-injection
    suite, every unit test) needs no AWS configuration to exercise the
    grouping and bus logic, while nothing can ever reach SNS without a topic
    this deployment was actually granted.

    A value that is set but is not an ARN raises immediately: that is the
    exact defect the legacy path carried, and it is a configuration error
    everywhere, not only on the send path.
    """
    variable = MUTED_TOPIC_ARN_VAR if legacy else PAGES_TOPIC_ARN_VAR
    expected_name = MUTED_TOPIC if legacy else PAGES_TOPIC
    raw = os.environ.get(variable, "").strip()
    if not raw:
        return None
    if not raw.startswith("arn:aws:sns:"):
        raise TopicUnresolvedError(
            f"{variable}={raw!r} is not an SNS topic ARN. `krepis.alerts` returns an "
            "explicit topic value verbatim, so a bare topic NAME reaches SNS as one and "
            "is refused with InvalidParameter."
        )
    if raw.rsplit(":", 1)[-1] != expected_name:
        raise TopicUnresolvedError(
            f"{variable}={raw!r} names topic {raw.rsplit(':', 1)[-1]!r}, not "
            f"{expected_name!r}. The v2 RuntimeRole is granted sns:Publish on "
            f"{PAGES_TOPIC} and {MUTED_TOPIC} only; publishing anywhere else is an "
            "AccessDenied that Telegram's success would hide."
        )
    return raw


def send(
    group: PageGroup,
    *,
    alert_id: str,
    legacy: bool = False,
    transport: Callable[..., Any] | None = None,
) -> tuple[bool, str]:
    """Deliver one page group through `krepis.alerts`. Returns ``(sent, destination)``.

    **Delivery failure RAISES.** An alert that could not be sent and was
    logged instead is an outage nobody hears about, and the send path is the
    last place a silent swallow belongs. `krepis.alerts.publish` already
    raises when the alert reached nothing at all; this passes
    ``raise_on_total_failure=True`` explicitly rather than relying on the
    default, so a change to that default cannot quietly turn this into a
    logging call.

    ``legacy=True`` routes to the muted topic — the v1 system's alerts during
    the overlap. It is a parameter rather than a separate function so the
    routing decision appears at the call site in a diff.

    ``transport`` is injectable for the fault-injection suite, which asserts
    "exactly one page" against a captured transport. It defaults to the real
    one; a test double is never the default.

    ``alert_id`` is the row's correlation identity, carried here so the call
    site reads as one act. The dedup identity is deliberately NOT it: an id
    that changes per observation is exactly what stopped krepis' forever
    window from ever engaging.
    """
    publish = transport if transport is not None else _krepis_publish
    result = publish(
        group.render(),
        severity=group.severity,
        source=f"crucible-v2/{group.condition}",
        dedup_key=incident_key(group),
        dedup_window_min=None,
        sns_topic_arn=topic_arn(legacy=legacy),
        raise_on_total_failure=True,
    )
    return _transport_outcome(result, legacy=legacy)


def _transport_outcome(result: Any, *, legacy: bool) -> tuple[bool, str]:
    """What the transport ACTUALLY did — never what was asked of it.

    `PublishResult.any_ok` is True on a DEDUP-SUPPRESSED and on a MUTED
    publish, by krepis' own documented contract: the alert is "logically in
    the operator's hands" by virtue of an earlier send. That is a reasonable
    thing for a Bash caller's `|| echo failed` to want and a false thing to
    write into `sent`, whose docstring says it records what happened on the
    transport and whose reader is the response plane. Both suppression flags
    are subtracted here.

    There is no optimistic default left. `sent` used to be
    ``bool(getattr(result, "any_ok", True))`` — a transport whose result did
    not answer the question was recorded as having delivered.

    ``dedup_skipped`` and ``muted`` are read with ``False`` defaults, and
    that is the one accommodation: they are fields of krepis'
    `PublishResult` (defaulting to False there) and a test double that omits
    them is asserting no suppression occurred. FAILURE MODE SWALLOWED: a
    double that suppresses a send without setting either flag would still be
    recorded as delivered. RECORDING SURFACE: the bus row's `sent` field,
    and `tests/test_alerts_grouping.py::TestDeliveryHonesty`, which drives
    the real flags. Closing it fully needs `tests/conftest.py`'s
    `CapturingTransport` to set `dedup_skipped` on its own dedup branch —
    named in the PR body, not this agent's file to change.
    """
    if not hasattr(result, "any_ok"):
        raise TypeError(
            f"{type(result).__name__} carries no `any_ok`; a transport result that "
            "cannot say whether the page was delivered cannot be recorded as delivered. "
            "`krepis.alerts.PublishResult` is the contract."
        )
    suppressed = bool(getattr(result, "dedup_skipped", False)) or bool(
        getattr(result, "muted", False)
    )
    sent = bool(result.any_ok) and not suppressed
    # krepis names it `telegram_destination` and leaves it None when the
    # Telegram leg was not reached; the previous `getattr(result,
    # "destination", ...)` read an attribute `PublishResult` has never had,
    # so every real bus row recorded the literal default rather than a
    # destination. Both names are read, in the order of authority.
    destination = (
        getattr(result, "telegram_destination", None)
        or getattr(result, "destination", None)
        or ("suppressed" if suppressed else "muted" if legacy else "sns_only")
    )
    return sent, str(destination)


def _krepis_publish(*args: Any, **kwargs: Any) -> Any:
    """The real transport, imported at call time.

    Lazy because `crucible --help`, the tests and every laptop run import
    this module, and none of them should pull an SNS client onto the import
    path.

    Refuses a publish with no topic. `krepis.alerts._resolve_sns_topic_arn`
    composes `alpha-engine-alerts` when it is handed None, and the v2
    RuntimeRole holds no grant on that topic — the publish would be
    AccessDenied, Telegram would succeed, `any_ok` would be True, and the
    only symptom would be an SNS subscriber that never heard from v2.
    """
    if kwargs.get("sns_topic_arn") is None:
        raise TopicUnresolvedError(
            f"{PAGES_TOPIC_ARN_VAR} (or {MUTED_TOPIC_ARN_VAR} for the legacy path) is "
            f"unset, so this page has no topic. Set it to the `crucible-v2` stack's "
            f"PagesTopicArn output. Falling through would publish to "
            f"alpha-engine-alerts, where this role has no grant."
        )
    from krepis.alerts import publish  # noqa: PLC0415

    return publish(*args, **kwargs)


def emit(
    store: Store,
    groups: Sequence[PageGroup],
    *,
    sweep_run_id: str | None,
    legacy: bool = False,
    transport: Callable[..., Any] | None = None,
    now: dt.datetime | None = None,
) -> list[str]:
    """Send each INCIDENT once and write its bus row. Returns the bus keys.

    The bus row is written **after** the send and records the send's real
    outcome. Writing it first would produce a durable claim about something
    that had not happened yet, and the response plane reads the bus.

    **An incident whose row already exists is not sent again.** §2 row 6 is
    "alerts only when wrong", and a persistent, unchanged wrongness is one
    wrongness: the nightly sweep re-observing Friday's missing artifact on
    Saturday and again on Sunday is one incident, not three. The suppression
    is not silence — the existing row's `observations` and
    `last_observed_utc` advance, so "still open, seen by three sweeps" is
    readable from the artifact, and the incident is counted once against the
    ceiling instead of once per sweep execution.

    Suppressing HERE rather than relying on the transport's dedup marker is
    deliberate: the marker lives in another bucket and its absence is
    invisible from these artifacts, while this store is the thing principle 1
    says the account must be reconstructible from. The transport's own
    `dedup_key` is still passed, as a second line.

    ``sweep_run_id`` may be None for an observer that has no run of its own;
    the row then records `alert_id: null` rather than a fabricated id.
    """
    moment = (now or dt.datetime.now(dt.UTC)).astimezone(dt.UTC)
    stamp = moment.strftime("%Y-%m-%dT%H:%M:%SZ")
    keys: list[str] = []
    for gp in groups:
        key = bus_key(gp)
        if store.exists(key):
            _record_reobservation(store, key, gp, stamp)
            keys.append(key)
            continue
        alert_id = _alert_id_for(gp, sweep_run_id)
        sent, destination = send(gp, alert_id=alert_id, legacy=legacy, transport=transport)
        row = bus_row(
            gp,
            alert_id=alert_id,
            sent=sent,
            destination=destination,
            first_observed_utc=stamp,
            last_observed_utc=stamp,
        )
        store.put_bytes(key, _dump_row(row))
        keys.append(key)
    return keys


def _alert_id_for(group: PageGroup, sweep_run_id: str | None) -> str | None:
    """The correlation id the row carries.

    A failure group's is the failed run's id — that is the join §9.3 wants.
    The all-zeros sentinel is explicitly NOT used: it joins to nothing, and a
    row whose `alert_id` is a plausible-looking id that resolves to no
    manifest costs a reader more than a row that names the sweep that saw it.
    """
    first = group.members[0]
    if group.condition == "failure" and first.run_id and first.run_id != _UNPARSEABLE_RUN_ID:
        return first.run_id
    return sweep_run_id


def _record_reobservation(store: Store, key: str, group: PageGroup, stamp: str) -> None:
    """Advance an open incident's row instead of paging for it again.

    `sent`, `destination`, `alert_id`, `members` and `rendered` describe the
    page that was DELIVERED and are left alone — rewriting a delivery record
    with facts from a later observation would make the row claim it sent
    something it did not. `members_now`, `observations` and
    `last_observed_utc` are the current picture.

    Conditional on the row's version. This is one key written by more than
    one execution — the fleet's last-writer-wins class — and a blind
    overwrite here would let a concurrent sweep drop an observation. A lost
    conditional write RAISES (:class:`crucible.store.PointerConflictError`)
    rather than retrying: two sweeps racing on one incident is itself a fact
    an operator should see.
    """
    expected = store.etag(key)
    # The STRICT face of the one reader: this is a writer about to swap a row
    # it has just read, and a bus row that is not an object is a reason to
    # stop the sweep with the cause named, never a fault row to publish.
    row = load_store_document(store, key)
    row["observations"] = int(row["observations"]) + 1
    row["last_observed_utc"] = stamp
    row["members_now"] = _members(group)
    store.compare_and_swap(key, expected, _dump_row(row))


def _dump_row(row: dict[str, Any]) -> bytes:
    return json.dumps(row, indent=2, sort_keys=True).encode("utf-8")


def pages_in_window(
    store: Store,
    *,
    now: dt.datetime | None = None,
    window_trading_days: int = CEILING_WINDOW_TRADING_DAYS,
) -> int:
    """How many INCIDENTS were paged over the trailing window.

    Three properties, and the ceiling means nothing without all three:

    * **Groups, not members** — five arms failing on one data outage is one
      page (§9.3), and counting members would read a single bad Saturday as
      five incidents against a two-a-month target.
    * **Incidents, not observations** — one bus row per incident, advanced
      rather than re-created by each later sweep that still sees it. The row
      shape used to be per-observation, so this number counted SWEEP CADENCE:
      one absent Friday artifact read as 1 on Friday night, 2 on Saturday and
      3 — a BREACH — on Sunday, without anything new having gone wrong.
    * **Counted from the bus**, not from an in-process counter, so it is
      reconstructible from artifacts by someone who was not here (principle
      1).

    The counting itself is :func:`pages_in_range`; this function only decides
    which trading days the trailing window covers.
    """
    moment = (now or dt.datetime.now(dt.UTC)).astimezone(dt.UTC)
    end = resolve_trading_day(moment)
    start = end
    # `window_trading_days` days INCLUSIVE of `end`, so the walk back takes
    # one step fewer than the count: `pages_in_range` is closed on both ends
    # (see its docstring for why one boundary rule, not two).
    for _ in range(window_trading_days - 1):
        start = previous_trading_day(start)
    return len(pages_in_range(store, start=start, end=end))


def pages_in_range(store: Store, *, start: dt.date, end: dt.date) -> list[str]:
    """Every bus row keyed on a day in ``start..end``, INCLUSIVE of both ends.

    **The one implementation of "pages over a span".** `crucible.gate`'s
    phase-2 ceiling clause grades the same bus against the same ceiling; when
    it carried its own copy the two disagreed at the first day of the window
    (`start <= day` here against `start < day` there), so the gate and the
    `pages_per_20_trading_days` metric could report different counts for the
    same bus. A second copy of a reading is a second contract.

    Closed on both ends because that is what every caller means by "over the
    window": `pages_in_window` subtracts one from its walk-back to keep
    counting exactly ``window_trading_days`` sessions, and the gate passes the
    first and last day of its window as written.

    Keys are parsed through :func:`crucible.keys.parse_bus_key`, never by
    positional index or an arity restated as an integer.
    """
    keys: list[str] = []
    for key in store.list_keys(ALERTS_ROOT):
        parsed = parse_bus_key(key)
        if parsed is None:
            continue
        try:
            day = dt.date.fromisoformat(parsed[0])
        except ValueError:
            continue
        if start <= day <= end:
            keys.append(key)
    return sorted(keys)


def ceiling_metric(count: int, *, now: dt.datetime) -> dict[str, Any]:
    """Pages-per-window as a MetricRecord with its ceiling as the baseline.

    `status` is BREACH rather than a soft word when the ceiling is exceeded:
    §11 risk 2 sets the target at two pages a month, and a number over target
    that renders as "watch" is a target nobody is held to.
    """
    breach = count > PAGES_PER_MONTH_CEILING
    return {
        "name": "pages_per_20_trading_days",
        "module": "crucible.alerts",
        "metric_type": "operational",
        "value": float(count),
        "unit": "pages",
        "n_floor": 0,
        "status": "BREACH" if breach else "OK",
        "status_reason": (
            f"{count} incident(s) paged in the trailing {CEILING_WINDOW_TRADING_DAYS} "
            f"trading days against a ceiling of {PAGES_PER_MONTH_CEILING}. "
            + (
                "Exceeding the ceiling is a defect in crucible.alerts, never a reason to "
                "add a suppression."
                if breach
                else "Within the declared ceiling."
            )
        ),
        "source_path": "alerts/{trading_day}/{alert_id}.json",
        "last_updated_utc": now.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "horizon_trading_days": CEILING_WINDOW_TRADING_DAYS,
    }


#: The heartbeat's reading of who would actually receive a page
#: (alpha-engine-config-I10024). Filed on `runs/heartbeat/{day}/run.json`.
SUBSCRIBERS_METRIC = "pages_topic_confirmed_subscribers"

#: The literal SNS returns as `SubscriptionArn` for a leg nobody has confirmed.
#: Not a real ARN, and the only way the API says "pending".
PENDING_CONFIRMATION = "PendingConfirmation"

#: A Lambda leg is a machine reader (the Telegram backstop forwarder), not a
#: human one. Everything else — email, sms, https — is a leg a person reads.
_MACHINE_PROTOCOLS = frozenset({"lambda", "sqs", "firehose", "application"})


@dataclass(frozen=True)
class SubscriberReading:
    """Who is subscribed to the pages topic, by leg, as SNS reports it.

    ``confirmed_human_legs`` and ``pending_legs`` hold PROTOCOL names so the
    metric can say "email pending" rather than a count a reader has to go
    and resolve. No endpoint addresses are kept: an email address is not a
    fact a manifest needs.
    """

    topic: str
    confirmed_human_legs: tuple[str, ...]
    pending_legs: tuple[str, ...]
    lambda_legs: int

    @property
    def met(self) -> bool:
        """At least one confirmed human leg AND the machine leg.

        Both, deliberately. The Lambda leg alone proves a page reaches a
        forwarder; the human leg alone proves it reaches a person only while
        the forwarder is the thing that noticed the forwarder died.
        """
        return bool(self.confirmed_human_legs) and self.lambda_legs >= 1

    def metric(self, *, now: dt.datetime) -> dict[str, Any]:
        if self.met:
            reason = (
                f"{len(self.confirmed_human_legs)} confirmed human leg(s) "
                f"({', '.join(self.confirmed_human_legs)}) and {self.lambda_legs} lambda "
                f"leg(s) on {PAGES_TOPIC}."
            )
        else:
            missing = []
            if not self.confirmed_human_legs:
                missing.append(
                    "no CONFIRMED human leg"
                    + (
                        f" ({', '.join(self.pending_legs)} still {PENDING_CONFIRMATION})"
                        if self.pending_legs
                        else " (none subscribed)"
                    )
                )
            if self.lambda_legs < 1:
                missing.append("no lambda leg (the Telegram backstop forwarder is not subscribed)")
            reason = (
                f"{PAGES_TOPIC}: " + "; ".join(missing) + ". A page published here reaches "
                "nobody who can act on it. Subscribe and confirm the leg; never soften this row."
            )
        return _subscribers_metric(
            value=float(len(self.confirmed_human_legs)),
            status="OK" if self.met else "FAIL",
            reason=reason,
            now=now,
        )


def _subscribers_metric(
    *, value: float, status: str, reason: str, now: dt.datetime
) -> dict[str, Any]:
    return {
        "name": SUBSCRIBERS_METRIC,
        "module": "crucible.alerts",
        "metric_type": "operational",
        "value": value,
        "unit": "confirmed_human_legs",
        "n_floor": 0,
        "status": status,
        "status_reason": reason,
        "source_path": f"sns:ListSubscriptionsByTopic on ${PAGES_TOPIC_ARN_VAR}",
        "last_updated_utc": now.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _subscribers_unmeasurable(reason: str, *, now: dt.datetime) -> dict[str, Any]:
    """The row when the list could not be read. `unmeasurable`, never `OK`
    and never a zero: no data about subscribers is not "no subscribers"."""
    return _subscribers_metric(value=0.0, status="unmeasurable", reason=reason, now=now)


def _default_sns() -> Any:
    """An SNS client, constructed lazily — same reason as `crucible.cost.default_client`."""
    import boto3  # noqa: PLC0415 - lazy on purpose

    return boto3.client("sns")


def pages_topic_subscribers(topic: str, *, sns: Any) -> SubscriberReading:
    """Read every subscription on ``topic`` and sort it into legs.

    Raises whatever the client raises. A denied `ListSubscriptionsByTopic` is
    a fact about OUR grant, and the caller (:func:`heartbeat`) records it as
    an access fault and fails the run after the heartbeat message is sent —
    the same shape as an unlistable manifest prefix (I9960). It is never
    read as "no subscribers".
    """
    confirmed: list[str] = []
    pending: list[str] = []
    lambdas = 0
    token: str | None = None
    while True:
        request: dict[str, Any] = {"TopicArn": topic}
        if token:
            request["NextToken"] = token
        response = sns.list_subscriptions_by_topic(**request)
        for sub in response.get("Subscriptions", []):
            protocol = str(sub.get("Protocol", "")).lower()
            arn = str(sub.get("SubscriptionArn", ""))
            if protocol in _MACHINE_PROTOCOLS:
                if arn.startswith("arn:aws:sns:"):
                    lambdas += 1
                continue
            if arn == PENDING_CONFIRMATION or not arn.startswith("arn:aws:sns:"):
                pending.append(protocol)
            else:
                confirmed.append(protocol)
        token = response.get("NextToken")
        if not token:
            break
    return SubscriberReading(
        topic=topic,
        confirmed_human_legs=tuple(sorted(confirmed)),
        pending_legs=tuple(sorted(pending)),
        lambda_legs=lambdas,
    )


def heartbeat(
    store: Store,
    *,
    now: dt.datetime | None = None,
    transport: Callable[..., Any] | None = None,
    run_id: str | None = None,
    dry_run: bool = False,
    sns: Any | None = None,
) -> dict[str, Any]:
    """Emit proof that the alerting path itself ran, and return its summary.

    Not a third page condition. It is the number that distinguishes "nothing
    was wrong this week" from "the thing that notices was dead this week", and
    its ABSENCE is what surfaces — to the operator, because the machine that
    would notice is the machine that would be dead.

    It is delivered on the same transport as a page ON PURPOSE. A heartbeat
    carried by a healthy second channel would prove that channel alive and say
    nothing about the one the pages use — which is the failure it exists to
    detect, dressed as its own detector.

    **And it evaluates the rows that declare IT as their watcher.**
    `components.yaml` gives `alerts.sweep` `absence_watched_by: heartbeat` —
    the one row the sweep cannot honestly watch, because a sweep that never
    ran cannot report itself missing. That declaration was a claim and not a
    mechanism: this function read the sweep's manifests nowhere, so the row
    rendered as covered while a sweep that stopped running was noticed by
    nobody. It now runs the same absence condition over its own rows, through
    the same :func:`evaluate_absence` — one implementation, selected by the
    declared watcher — and a finding is a real page with a real bus row,
    counted against the same ceiling. A dead sweep is an incident, not a
    footnote in an info-severity message.

    ``run_id`` is this heartbeat's own run id, carried onto any bus row it
    writes. Absent, the row records `alert_id: null`: an absence page has no
    run to join to anyway, and inventing an id would be worse than saying
    there is none.

    ``dry_run=True`` evaluates the same watched-absence condition and
    computes the same summary, but never calls :func:`emit` (no bus row, no
    page for a genuine watched absence) and never calls the final `publish`
    below (no heartbeat message sent at all) — alpha-engine-config-I9922
    R2-1: a dry-run heartbeat that still messaged the real channel would be a
    worse defect than the store-write bug this closes, since a message send
    is not something the store guard (`crucible.store.read_only`) can see or
    refuse.
    """
    moment = (now or dt.datetime.now(dt.UTC)).astimezone(dt.UTC)
    trading_day = resolve_trading_day(moment)
    access_faults: list[str] = []
    watched = evaluate_absence(
        store, now=moment, watched_by="heartbeat", access_faults=access_faults
    )
    if dry_run:
        bus_keys: list[str] = []
    else:
        bus_keys = emit(
            store,
            group_pages(watched),
            sweep_run_id=run_id,
            transport=transport,
            now=moment,
        )
    # After emit, so a sweep-absence raised by THIS run is inside the number
    # the same run reports. A count taken first would publish a heartbeat
    # whose own finding was missing from its own metric.
    pages = pages_in_window(store, now=moment)
    runs_ok, runs_failed, spend = _week_summary(store, trading_day)
    unwatched = sorted({page.job for page in watched})
    # Who would receive the page this heartbeat is about to send
    # (alpha-engine-config-I10024). Read-only, so it runs under `dry_run`
    # too. With no topic configured there is nothing to list: the row reads
    # `unmeasurable` naming the variable, and on a real box the publish
    # below refuses for the same reason. A DENIED list is recorded as an
    # access fault and raised after the heartbeat is sent — the reading's
    # absence must never look like a confirmed subscriber.
    topic = topic_arn()
    if topic is None:
        subscribers = _subscribers_unmeasurable(
            f"{PAGES_TOPIC_ARN_VAR} is unset, so there is no topic whose subscriptions "
            "could be listed. Not a reading of zero subscribers.",
            now=moment,
        )
    else:
        try:
            subscribers = pages_topic_subscribers(
                topic, sns=sns if sns is not None else _default_sns()
            ).metric(now=moment)
        except Exception as exc:  # noqa: BLE001 - re-raised via access_faults below
            access_faults.append(
                f"sns:ListSubscriptionsByTopic on {topic}: {type(exc).__name__}: {exc}"
            )
            subscribers = _subscribers_unmeasurable(
                f"sns:ListSubscriptionsByTopic on {topic} raised {type(exc).__name__}: {exc}. "
                "A statement about this identity's grant, not about who is subscribed.",
                now=moment,
            )
    summary = {
        "trading_day": trading_day.isoformat(),
        "runs_ok": runs_ok,
        "runs_failed": runs_failed,
        "cost_usd": round(spend, 4),
        "pages_in_window": pages,
        "watched_absences": unwatched,
        "bus_keys": bus_keys,
        "metric": ceiling_metric(pages, now=moment),
        "subscribers": subscribers,
    }
    message = (
        f"[crucible-v2] alive {trading_day.isoformat()}: {runs_ok} run(s) ok, "
        f"{runs_failed} failed, ${spend:.2f}, {pages} incident(s) paged in the trailing "
        f"{CEILING_WINDOW_TRADING_DAYS} trading days "
        f"(ceiling {PAGES_PER_MONTH_CEILING})."
    )
    if unwatched:
        message += (
            " ABSENT, and watched by nothing else: "
            + ", ".join(unwatched)
            + ". The alerting path did not run; neither page condition was evaluated "
            "on the day(s) named in the bus row."
        )
    nobody_listening = subscribers["status"] == "FAIL"
    if nobody_listening:
        # Said on the channel that still works (the Lambda leg forwards to
        # Telegram even when the human leg is gone), because the row on the
        # manifest is read by a console and this is read by a person.
        message += " " + str(subscribers["status_reason"])
    if not dry_run:
        publish = transport if transport is not None else _krepis_publish
        publish(
            message,
            severity="error" if (unwatched or nobody_listening) else "info",
            source="crucible-v2/heartbeat",
            dedup_key=f"heartbeat:{trading_day.isoformat()}",
            dedup_window_min=None,
            sns_topic_arn=topic_arn(),
            raise_on_total_failure=True,
        )
    # After the heartbeat has been SENT: a pass that could not list a prefix
    # still owes the operator the proof that the alerting path is alive, and
    # then fails loudly naming what it could not read (I9960).
    _raise_on_access_faults(access_faults)
    # Always carried, not only under `dry_run`: the caller's own print (under
    # `--dry-run`, `heartbeat_handler`) needs the exact text that either was,
    # or would have been, sent — one string, not two call sites composing it
    # differently.
    summary["message"] = message
    return summary


def _week_summary(store: Store, trading_day: dt.date) -> tuple[int, int, float]:
    """Runs ok, runs failed and spend over the trailing trading week.

    Reads the manifests, not a rollup. A rollup would be a second place the
    same facts live, and the week's manifests are already the durable record
    the console renders from.
    """
    days = {trading_day}
    day = trading_day
    for _ in range(TRADING_DAYS_PER_WEEK - 1):
        day = previous_trading_day(day)
        days.add(day)
    ok = failed = 0
    spend = 0.0
    for key in store.list_keys(RUNS_ROOT):
        parsed = parse_manifest_key(key)
        if parsed is None:
            continue
        _job, trading_day_str, _discriminator = parsed
        try:
            if dt.date.fromisoformat(trading_day_str) not in days:
                continue
        except ValueError:
            # A trading-day segment that fails ISO parsing means this key
            # was never written by crucible.keys.manifest_key — but it still
            # sits under a real job/day prefix and still cost something, so
            # counting it failed (matching the manifest-read failure below)
            # keeps this summary from reading healthier than reality the
            # worse the store gets; a silent `continue` here would have done
            # exactly that.
            failed += 1
            continue
        read = read_listed_document(store, key)
        if read.problem is not None or read.document is None:
            # Counted as failed, never skipped: an unreadable manifest is the
            # run most likely to be broken, and dropping it from the summary
            # would make the heartbeat read healthier the worse things got.
            # Through the one guarded reader, so an array-bodied manifest is
            # counted here rather than raising `AttributeError` two lines
            # down (alpha-engine-config-I9931).
            failed += 1
            continue
        manifest = read.document
        if manifest.get("status") == "ok":
            ok += 1
        else:
            failed += 1
        spend += float(manifest.get("cost_usd", 0.0))
    return ok, failed, spend


def _raise_on_access_faults(faults: Sequence[str]) -> None:
    """Raise if any prefix could not be listed. Called after emitting."""
    if faults:
        raise StoreAccessError(
            f"{len(faults)} manifest prefix(es) could not be listed, so ABSENCE was "
            "not evaluated for them. This is not an absence: " + " | ".join(sorted(faults))
        )


def sweep(
    store: Store,
    *,
    now: dt.datetime | None = None,
    registry: dict[str, Component] | None = None,
    transport: Callable[..., Any] | None = None,
    sweep_run_id: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Evaluate both conditions, group by cause, page once per group.

    The whole alerting pass, in one function, so "what does the alerter do"
    has one answer and the fault-injection suite has one thing to call.

    ``dry_run=True`` evaluates and groups exactly as a real sweep would —
    both conditions are pure reads — but never calls :func:`emit`: no page is
    sent and no bus row is written (alpha-engine-config-I9922 R2-1).
    `pages_emitted` and `bus_keys` read as `0`/`()` on this path (nothing was
    emitted, by definition); `incidents_open` and `members` are the real
    counts, since both come from evaluating and reading the bus, not from
    writing to it.
    """
    moment = (now or dt.datetime.now(dt.UTC)).astimezone(dt.UTC)
    access_faults: list[str] = []
    pages = evaluate_absence(
        store, now=moment, registry=registry, access_faults=access_faults
    ) + evaluate_failure(store, now=moment, registry=registry, access_faults=access_faults)
    groups = group_pages(pages)
    if dry_run:
        _raise_on_access_faults(access_faults)
        count = pages_in_window(store, now=moment)
        return {
            "pages_emitted": 0,
            "incidents_open": len(groups),
            "members": len(pages),
            "bus_keys": [],
            "metric": ceiling_metric(count, now=moment),
        }
    # Read before emitting: `pages_emitted` is what this run actually SENT,
    # not how many incidents it saw. A run that re-observed three open
    # incidents and paged for none of them reporting "3 pages emitted" is the
    # same false claim the bus rows used to make, one layer up.
    already_open = {bus_key(gp) for gp in groups if store.exists(bus_key(gp))}
    keys = emit(store, groups, sweep_run_id=sweep_run_id, transport=transport, now=moment)
    # AFTER emit, deliberately. Everything this pass could observe has now
    # been paged; what it could not observe is a failure of this job, and it
    # exits non-zero through `run_job`'s `try/finally` with the prefixes
    # named in its own manifest (alpha-engine-config-I9960).
    _raise_on_access_faults(access_faults)
    count = pages_in_window(store, now=moment)
    return {
        "pages_emitted": len([k for k in keys if k not in already_open]),
        "incidents_open": len(groups),
        "members": len(pages),
        "bus_keys": keys,
        "metric": ceiling_metric(count, now=moment),
    }
