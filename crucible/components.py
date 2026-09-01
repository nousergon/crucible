"""The observability registry, read as data.

Normative source: plan §9.2, §4.6, §4.12.

`components.yaml` is the declaration; this module is the only reader. It
exists so the deadline table has exactly one parser: §4.6's absence condition
resolves a deadline against the trading calendar, and a deadline expressed as
English prose would need that parse written twice — once in the alerter and
once in whatever reads the file next — which is the contract-restated-twice
shape that has already drifted elsewhere in the fleet.

**Deadlines are structured, and the sentence is rendered from the structure.**
The YAML carries an anchor and an offset; :meth:`Deadline.describe` produces
"3h after the close of trading day 2026-08-28". The prose is therefore a
projection of the data rather than a second copy of it, and the two cannot
disagree.

**Two anchors, exhaustively.** Adding a third is a design change visible in a
diff, for the same reason the two page conditions are a closed set.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

import yaml

from crucible.calendar import assert_trading_day

__all__ = [
    "ANCHORS",
    "Component",
    "DISPATCHES",
    "Deadline",
    "LIFECYCLES",
    "NYSE_TZ",
    "REGISTRY_PATH",
    "load_registry",
    "scheduled_components",
]

REGISTRY_PATH = Path(__file__).parent / "components.yaml"

NYSE_TZ = ZoneInfo("America/New_York")

#: The regular NYSE close. Half-day sessions close at 13:00 ET; the deadline
#: table anchors on the *regular* close on purpose — a deadline that moved
#: three hours earlier on a half day would page for an artifact that was
#: never late, and the half-day list is not a thing an operator carries in
#: their head at 3am.
REGULAR_CLOSE = dt.time(16, 0)

#: Exhaustive. `close_plus` is "T hours after the close of trading day d";
#: `next_calendar_day_at` is "HH:MM ET on the calendar day after trading day
#: d" — which for a Friday d is the Saturday the weekly work runs on.
Anchor = Literal["close_plus", "next_calendar_day_at"]
ANCHORS: tuple[str, ...] = ("close_plus", "next_calendar_day_at")

#: §8.3 of observability-policy: DISABLED and RETIRED are DECLARED, never
#: inferred. A job inferred to be retired because it stopped producing is
#: indistinguishable from one that broke.
LIFECYCLES: tuple[str, ...] = ("ACTIVE", "DISABLED", "RETIRED")

#: WHO starts a scheduled job. Exhaustive, and declared per row rather than
#: inferred from the schedule string: `arc` means the weekly driver runs it as
#: a stage (`crucible.weekly`), `scheduler` means it has an EventBridge
#: schedule of its own. Before this field existed, six rows said "weekly,
#: Saturday" and exactly one of them was dispatched by anything — the other
#: five were declared, deadlined, watched for absence, and started by nobody.
#: A schedule string is a description; this is the wiring.
DISPATCHES: tuple[str, ...] = ("arc", "scheduler")


@dataclass(frozen=True)
class Deadline:
    """When a scheduled job's manifest must exist, relative to a session.

    Resolved, never approximated: :meth:`due_at` returns an aware UTC instant
    for a given trading day, so "is it late" is a comparison rather than a
    judgement.
    """

    anchor: Anchor
    offset_hours: float | None = None
    at: dt.time | None = None

    def __post_init__(self) -> None:
        if self.anchor not in ANCHORS:
            raise ValueError(
                f"{self.anchor!r} is not a deadline anchor. There are exactly "
                f"{ANCHORS}; a third is a design change, not a config value."
            )
        if self.anchor == "close_plus" and self.offset_hours is None:
            raise ValueError("a close_plus deadline needs offset_hours")
        if self.anchor == "next_calendar_day_at" and self.at is None:
            raise ValueError("a next_calendar_day_at deadline needs `at`")

    def due_at(self, trading_day: dt.date) -> dt.datetime:
        """The UTC instant by which trading day ``trading_day``'s manifest
        must exist.

        ``trading_day`` is asserted to be a session first. Resolving a
        deadline against a non-session would produce a plausible instant for
        a day the market never opened, and the page that followed would be
        for an artifact nothing was ever going to write.
        """
        assert_trading_day(trading_day, context=f"deadline anchor {self.anchor}")
        if self.anchor == "close_plus":
            base = dt.datetime.combine(trading_day, REGULAR_CLOSE, tzinfo=NYSE_TZ)
            return (base + dt.timedelta(hours=float(self.offset_hours))).astimezone(dt.UTC)
        # next_calendar_day_at. Calendar, deliberately: this is a wall-clock
        # scheduling quantity, which is one of §4.12's exhaustive exceptions.
        assert self.at is not None  # narrowed by __post_init__
        base = dt.datetime.combine(trading_day + dt.timedelta(days=1), self.at, tzinfo=NYSE_TZ)
        return base.astimezone(dt.UTC)

    def describe(self, trading_day: dt.date | None = None) -> str:
        """The operator-facing sentence, rendered FROM the structure.

        Not stored beside it: a sentence stored beside the data it describes
        is a second declaration, and the fleet has measured what happens to
        those.
        """
        day = trading_day.isoformat() if trading_day else "d"
        if self.anchor == "close_plus":
            hours = self.offset_hours
            rendered = f"{hours:g}h"
            return f"{rendered} after the close of trading day {day}"
        assert self.at is not None
        return f"{self.at.strftime('%H:%M')} ET the calendar day after trading day {day}"

    @classmethod
    def from_yaml(cls, raw: dict[str, Any] | None) -> Deadline | None:
        if raw is None:
            return None
        at = raw.get("at")
        return cls(
            anchor=raw["anchor"],
            offset_hours=raw.get("offset_hours"),
            at=dt.time.fromisoformat(at) if at else None,
        )


@dataclass(frozen=True)
class Component:
    """One registry row. Every field is required, including the null ones.

    A component with no row is unobserved, not healthy; a row with an
    *omitted* signal class is indistinguishable from a forgotten one, which
    is why `signals` carries all five with explicit nulls.
    """

    name: str
    description: str
    lifecycle: str
    signals: dict[str, str | None]
    log_location: str
    log_retention_days: int
    alert_channel: str
    console_surface: str
    artifact_retention: str
    schedule: str | None
    deadline: Deadline | None
    #: `arc` | `scheduler` for a scheduled row, `None` for an on-demand one.
    #: Required of every scheduled row IN THE FILE — enforced by
    #: :func:`load_registry`, not by `__post_init__`, because the YAML is the
    #: declaration surface and the dataclass is a value object a test may
    #: construct for one narrow purpose.
    #: The cross-repo half of this contract — that every `scheduler` row has a
    #: real `AWS::Scheduler::Schedule` and no `arc` row has one — is asserted
    #: in `nous-ergon-ops` against the CloudFormation template, because that is
    #: where the other half of the pair lives.
    dispatch: str | None = None
    #: WHO would notice this row's absence. Almost always `alerts.sweep`.
    #: The two exceptions are the sweep itself (a sweep that never ran
    #: cannot report itself missing) and the heartbeat, whose watcher is
    #: `operator` — because if the alerting path is dead, so is any machine
    #: watcher living inside it. Declared per row so the exception is
    #: visible in the file rather than being a name embedded in the
    #: alerter, and so a row can never end up watching itself.
    absence_watched_by: str = "alerts.sweep"

    @property
    def scheduled(self) -> bool:
        return self.schedule is not None

    def __post_init__(self) -> None:
        if self.dispatch is not None and self.dispatch not in DISPATCHES:
            raise ValueError(f"{self.name}: dispatch {self.dispatch!r} not in {DISPATCHES}")
        if self.lifecycle not in LIFECYCLES:
            raise ValueError(f"{self.name}: lifecycle {self.lifecycle!r} not in {LIFECYCLES}")
        if self.scheduled and self.deadline is None:
            raise ValueError(
                f"{self.name} is scheduled but declares no deadline — its absence could "
                "never page, which is the blindness §4.6 exists to remove."
            )
        if self.absence_watched_by == self.name:
            raise ValueError(
                f"{self.name} declares itself its own absence watcher. A component "
                "that never ran cannot report itself missing, so this row would be "
                "unwatched while reading as covered."
            )
        if not self.scheduled and self.deadline is not None:
            raise ValueError(
                f"{self.name} is on-demand but declares a deadline; its absence is not a "
                "fact about the system and the deadline would page for nothing."
            )


@lru_cache(maxsize=1)
def load_registry(path: str | None = None) -> dict[str, Component]:
    """Parse `components.yaml` into :class:`Component` rows.

    Cached on the path: the file ships inside the package and does not change
    within a process. A missing file raises rather than yielding an empty
    registry — an empty registry would make every absence check vacuously
    pass, which is the loudest possible way to be silently blind.
    """
    target = Path(path) if path else REGISTRY_PATH
    if not target.is_file():
        raise FileNotFoundError(
            f"the observability registry is missing at {target}. An empty registry "
            "makes every absence check vacuously pass; that is a broken build, not a "
            "degraded run."
        )
    raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    out: dict[str, Component] = {}
    for name, row in raw["components"].items():
        _assert_dispatch_declared(name, row)
        out[name] = Component(
            name=name,
            description=row["description"],
            lifecycle=row.get("lifecycle", "ACTIVE"),
            signals=row["signals"],
            log_location=row["log_location"],
            log_retention_days=row["log_retention_days"],
            alert_channel=row["alert_channel"],
            console_surface=row["console_surface"],
            artifact_retention=row["artifact_retention"],
            schedule=row["schedule"],
            dispatch=row["dispatch"],
            absence_watched_by=row.get("absence_watched_by", "alerts.sweep"),
            deadline=Deadline.from_yaml(row["deadline"]),
        )
    return out


def _assert_dispatch_declared(name: str, row: dict[str, Any]) -> None:
    """Every scheduled row in the file names who starts it; no other row does.

    Enforced here rather than on the dataclass because this is the only place
    the FILE is read, and the file is the declaration. The failure it prevents
    is the one that shipped: six rows read `schedule: weekly, Saturday` while
    the scheduler dispatched exactly one of them, so five components were
    deadlined and watched for absence and started by nobody.
    """
    if "dispatch" not in row:
        raise ValueError(
            f"{name} declares no `dispatch`. Every row states who starts it — `arc` "
            "(a stage of `crucible weekly`), `scheduler` (its own EventBridge "
            "schedule), or `null` for on-demand."
        )
    dispatch = row["dispatch"]
    scheduled = row["schedule"] is not None
    if scheduled and dispatch is None:
        raise ValueError(
            f"{name} is scheduled but its dispatch is null. Something has to start "
            "it, and a row naming no starter is a job whose absence pages every "
            "cycle for work nobody was going to run."
        )
    if not scheduled and dispatch is not None:
        raise ValueError(
            f"{name} is on-demand but declares dispatch {dispatch!r}; an unscheduled "
            "job is started by a person or another job, and naming a starter here "
            "would claim a cadence it does not have."
        )


def scheduled_components(registry: dict[str, Component] | None = None) -> dict[str, Component]:
    """The rows whose absence is a fact about the system.

    `DISABLED` and `RETIRED` rows are excluded here — that is the one place
    the declaration is honoured, and it is honoured by reading the declared
    field rather than by inferring anything from silence.
    """
    reg = registry if registry is not None else load_registry()
    return {name: c for name, c in reg.items() if c.scheduled and c.lifecycle == "ACTIVE"}
