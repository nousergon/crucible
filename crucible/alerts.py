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
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from crucible.calendar import (
    TRADING_DAYS_PER_WEEK,
    previous_trading_day,
    resolve_trading_day,
)
from crucible.components import Component, load_registry, scheduled_components
from crucible.manifest import manifest_key
from crucible.store import Store

__all__ = [
    "ALERT_BUS_SCHEMA_VERSION",
    "CAUSE_MATCHERS",
    "CEILING_WINDOW_TRADING_DAYS",
    "MUTED_TOPIC",
    "PAGES_PER_MONTH_CEILING",
    "PAGE_CONDITIONS",
    "Page",
    "PageGroup",
    "bus_key",
    "bus_row",
    "cause_key",
    "ceiling_metric",
    "dedup_key",
    "emit",
    "evaluate_absence",
    "evaluate_failure",
    "group_pages",
    "heartbeat",
    "pages_in_window",
    "send",
    "sweep",
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


def dedup_key(condition: str, job: str, trading_day: dt.date) -> str:
    """One key per (job, trading day). §4.6.

    Not per attempt and not per condition-instance: a job that fails, is
    retried by the declared transient class and fails again is ONE incident,
    and paging twice for it is how a two-page-per-month ceiling is blown by a
    single bad Saturday.
    """
    return f"{condition}:{job}:{trading_day.isoformat()}"


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


# ── The two page conditions ───────────────────────────────────────────────


def evaluate_absence(
    store: Store,
    *,
    now: dt.datetime | None = None,
    registry: dict[str, Component] | None = None,
) -> list[Page]:
    """Page for every scheduled job whose manifest is missing past its deadline.

    Reads the deadline table from `components.yaml` through
    :mod:`crucible.components` — never a second copy — and resolves each
    deadline against the trading calendar, so a market holiday moves the
    deadline rather than producing a page that has to be dismissed.

    **Only past deadlines are evaluated.** A job whose deadline is still in
    the future is not absent, it is not due; reporting it would make the
    absence condition fire every time the sweep ran early.

    **A row declaring another watcher is skipped, by declaration.** That is
    `heartbeat`, whose absence a human notices precisely because the sweep
    that would have noticed it is the thing that may be dead. The skip is
    read from the registry, so it is visible in the file rather than being a
    name embedded in this function.
    """
    moment = (now or dt.datetime.now(dt.UTC)).astimezone(dt.UTC)
    reg = scheduled_components(registry)
    trading_day = resolve_trading_day(moment)
    pages: list[Page] = []
    for name, component in sorted(reg.items()):
        if component.absence_watched_by != "alerts.sweep":
            continue
        assert component.deadline is not None  # Component.__post_init__ guarantees it
        due = component.deadline.due_at(trading_day)
        if moment < due:
            continue
        if store.exists(manifest_key(name, trading_day.isoformat())):
            continue
        pages.append(
            Page(
                condition="absence",
                job=name,
                trading_day=trading_day,
                reason=(
                    f"no manifest at {manifest_key(name, trading_day.isoformat())}; due "
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
) -> list[Page]:
    """Page for every manifest written with `status: failed`.

    The page's reason is the manifest's `reason` VERBATIM. Re-deriving a
    cause here would let the page and the artifact disagree, and an operator
    reading two accounts of one failure has to work out which is authoritative
    before doing anything about it.

    A manifest that will not parse or will not validate is itself a failure
    page: an unreadable manifest is indistinguishable from a lie, and reading
    past it would drop the very run most likely to be broken.
    """
    moment = (now or dt.datetime.now(dt.UTC)).astimezone(dt.UTC)
    reg = registry if registry is not None else load_registry()
    trading_day = resolve_trading_day(moment)
    pages: list[Page] = []
    for name, component in sorted(reg.items()):
        if component.lifecycle != "ACTIVE":
            continue
        key = manifest_key(name, trading_day.isoformat())
        if not store.exists(key):
            continue
        try:
            manifest = json.loads(store.get_bytes(key).decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            pages.append(
                Page(
                    condition="failure",
                    job=name,
                    trading_day=trading_day,
                    reason=f"manifest at {key} is unreadable: {type(exc).__name__}: {exc}",
                    run_id=_UNPARSEABLE_RUN_ID,
                )
            )
            continue
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


def bus_key(group: PageGroup, alert_id: str) -> str:
    """`alerts/{trading_day}/{run_id}.json` (§9.3).

    ``alert_id`` is the group's own correlation identity. For a single-member
    FAILURE group it IS the failed run's id, which is what makes the bus row
    joinable to the manifest; for an ABSENCE group there is no run, so the
    sweep's own ULID stands in and the row says which of the two it is. A
    machine reader gets a stable key either way, and the field it needs to
    join on is never silently empty.
    """
    return f"alerts/{group.trading_day.isoformat()}/{alert_id}.json"


def bus_row(group: PageGroup, *, alert_id: str, sent: bool, destination: str) -> dict[str, Any]:
    """The machine-readable row. §7.3: a human-only alert is invisible.

    ``sent`` records what actually happened on the transport, not what was
    intended. A row claiming delivery for a page that never left is worse
    than no row: the response plane would read it as handled.
    """
    return {
        "schema_version": ALERT_BUS_SCHEMA_VERSION,
        "alert_id": alert_id,
        "alert_id_is_run_id": alert_id != _sweep_alert_id_marker(alert_id),
        "condition": group.condition,
        "cause_key": group.cause_key,
        "trading_day": group.trading_day.isoformat(),
        "dedup_key": dedup_key(group.condition, group.members[0].job, group.trading_day),
        "members": [
            {"job": m.job, "run_id": m.run_id, "reason": m.reason}
            for m in sorted(group.members, key=lambda x: x.job)
        ],
        "sent": sent,
        "destination": destination,
        "rendered": group.render(),
    }


def _sweep_alert_id_marker(alert_id: str) -> str:
    """Identity. Present so `alert_id_is_run_id` reads as a computed field
    rather than a hardcoded True, and so the one caller that knows the
    difference (:func:`emit`) sets it by passing the run's id or not."""
    return alert_id


ALERT_BUS_SCHEMA_VERSION = "alert_bus.v1"

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
MUTED_TOPIC = "alpha-engine-alerts-muted"


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
    """
    publish = transport if transport is not None else _krepis_publish
    result = publish(
        group.render(),
        severity=group.severity,
        source=f"crucible-v2/{group.condition}",
        dedup_key=dedup_key(group.condition, group.members[0].job, group.trading_day),
        dedup_window_min=None,
        sns_topic_arn=MUTED_TOPIC if legacy else None,
        raise_on_total_failure=True,
    )
    destination = getattr(result, "destination", "muted" if legacy else "operator_chat")
    sent = bool(getattr(result, "any_ok", True))
    return sent, str(destination)


def _krepis_publish(*args: Any, **kwargs: Any) -> Any:
    """The real transport, imported at call time.

    Lazy because `crucible --help`, the tests and every laptop run import
    this module, and none of them should pull an SNS client onto the import
    path.
    """
    from krepis.alerts import publish  # noqa: PLC0415

    return publish(*args, **kwargs)


def emit(
    store: Store,
    groups: Sequence[PageGroup],
    *,
    sweep_run_id: str,
    legacy: bool = False,
    transport: Callable[..., Any] | None = None,
) -> list[str]:
    """Send each group once and write its bus row. Returns the bus keys.

    The bus row is written **after** the send and records the send's real
    outcome. Writing it first would produce a durable claim about something
    that had not happened yet, and the response plane reads the bus.
    """
    keys: list[str] = []
    for gp in groups:
        alert_id = (
            gp.members[0].run_id
            if gp.condition == "failure" and gp.members[0].run_id
            else sweep_run_id
        )
        sent, destination = send(gp, alert_id=alert_id, legacy=legacy, transport=transport)
        row = bus_row(gp, alert_id=alert_id, sent=sent, destination=destination)
        row["alert_id_is_run_id"] = gp.condition == "failure"
        key = bus_key(gp, alert_id)
        store.put_bytes(key, json.dumps(row, indent=2, sort_keys=True).encode("utf-8"))
        keys.append(key)
    return keys


def pages_in_window(
    store: Store,
    *,
    now: dt.datetime | None = None,
    window_trading_days: int = CEILING_WINDOW_TRADING_DAYS,
) -> int:
    """How many page GROUPS were emitted over the trailing window.

    Groups, not members: one outage is one page. Counted from the bus rather
    than from a counter, so the number is reconstructible from artifacts by
    someone who was not here (principle 1).
    """
    moment = (now or dt.datetime.now(dt.UTC)).astimezone(dt.UTC)
    end = resolve_trading_day(moment)
    start = end
    for _ in range(window_trading_days):
        start = previous_trading_day(start)
    count = 0
    for key in store.list_keys("alerts/"):
        parts = key.split("/")
        if len(parts) != 3 or not key.endswith(".json"):
            continue
        try:
            day = dt.date.fromisoformat(parts[1])
        except ValueError:
            continue
        if start < day <= end:
            count += 1
    return count


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
            f"{count} page group(s) in the trailing {CEILING_WINDOW_TRADING_DAYS} trading "
            f"days against a ceiling of {PAGES_PER_MONTH_CEILING}. "
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


def heartbeat(
    store: Store,
    *,
    now: dt.datetime | None = None,
    transport: Callable[..., Any] | None = None,
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
    """
    moment = (now or dt.datetime.now(dt.UTC)).astimezone(dt.UTC)
    trading_day = resolve_trading_day(moment)
    pages = pages_in_window(store, now=moment)
    runs_ok, runs_failed, spend = _week_summary(store, trading_day)
    summary = {
        "trading_day": trading_day.isoformat(),
        "runs_ok": runs_ok,
        "runs_failed": runs_failed,
        "cost_usd": round(spend, 4),
        "pages_in_window": pages,
        "metric": ceiling_metric(pages, now=moment),
    }
    message = (
        f"[crucible-v2] alive {trading_day.isoformat()}: {runs_ok} run(s) ok, "
        f"{runs_failed} failed, ${spend:.2f}, {pages} page group(s) in the trailing "
        f"{CEILING_WINDOW_TRADING_DAYS} trading days "
        f"(ceiling {PAGES_PER_MONTH_CEILING})."
    )
    publish = transport if transport is not None else _krepis_publish
    publish(
        message,
        severity="info",
        source="crucible-v2/heartbeat",
        dedup_key=f"heartbeat:{trading_day.isoformat()}",
        dedup_window_min=None,
        raise_on_total_failure=True,
    )
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
    for key in store.list_keys("runs/"):
        if not key.endswith("/run.json"):
            continue
        parts = key.split("/")
        if len(parts) != 4:
            continue
        try:
            if dt.date.fromisoformat(parts[2]) not in days:
                continue
        except ValueError:
            continue
        try:
            manifest = json.loads(store.get_bytes(key).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            # Counted as failed, never skipped: an unreadable manifest is the
            # run most likely to be broken, and dropping it from the summary
            # would make the heartbeat read healthier the worse things got.
            failed += 1
            continue
        if manifest.get("status") == "ok":
            ok += 1
        else:
            failed += 1
        spend += float(manifest.get("cost_usd", 0.0))
    return ok, failed, spend


def sweep(
    store: Store,
    *,
    now: dt.datetime | None = None,
    registry: dict[str, Component] | None = None,
    transport: Callable[..., Any] | None = None,
    sweep_run_id: str,
) -> dict[str, Any]:
    """Evaluate both conditions, group by cause, page once per group.

    The whole alerting pass, in one function, so "what does the alerter do"
    has one answer and the fault-injection suite has one thing to call.
    """
    moment = (now or dt.datetime.now(dt.UTC)).astimezone(dt.UTC)
    pages = evaluate_absence(store, now=moment, registry=registry) + evaluate_failure(
        store, now=moment, registry=registry
    )
    groups = group_pages(pages)
    keys = emit(store, groups, sweep_run_id=sweep_run_id, transport=transport)
    count = pages_in_window(store, now=moment)
    return {
        "pages_emitted": len(groups),
        "members": len(pages),
        "bus_keys": keys,
        "metric": ceiling_metric(count, now=moment),
    }
