"""The trading-day axis. A thin wrapper over `krepis.trading_calendar`.

Normative source: `crucible_v2_rebuild_plan_260901.md` §4.12 (Brian ruling,
2026-09-01) — *"each data point should bind to a trading day, not a calendar
day. We should make all decisions based on trading days."*

**Thin on purpose.** The NYSE calendar — holidays, observed holidays, early
closes, the 16:00 ET close threshold — has exactly one implementation in the
fleet, in `krepis.trading_calendar`, and this module never re-derives any of
it. What it adds is the three things v2 needs and krepis does not owe it:

* a single `resolve_trading_day` that every job calls, so "which day is this
  run for" is answered in one place rather than per job;
* `trading_weeks_between`, because a ladder rung is five trading days and a
  holiday week is still one rung;
* `assert_trading_day` and :class:`NonTradingDayKeyError`, the failure the
  §4.12 contract test raises when a key does not bind to a session.

Everything here is backward-looking. A run binds to the last *completed*
session, never to a session still open — a morning job keyed to `today`
would publish an artifact for a day with no close.
"""

from __future__ import annotations

import datetime as dt
import re

from krepis.trading_calendar import (
    count_trading_days,
    last_closed_trading_day,
)
from krepis.trading_calendar import (
    is_trading_day as _krepis_is_trading_day,
)

__all__ = [
    "ISO_DATE_RE",
    "NonTradingDayKeyError",
    "TRADING_DAYS_PER_WEEK",
    "assert_trading_day",
    "is_trading_day",
    "resolve_trading_day",
    "trading_weeks_between",
]

#: §4.12: "Ladder rungs, promote_min_weeks, grace_weeks,
#: retired_trailing_cycles [are in] trading weeks = 5 trading days".
TRADING_DAYS_PER_WEEK = 5

#: Matches an ISO-8601 date component in a store key. Deliberately strict:
#: `2026-8-29` does NOT match, and the walker treats a near-miss as an error
#: rather than as a dateless key (see :meth:`Store.assert_keys_bind_to_trading_days`).
ISO_DATE_RE = re.compile(r"(?<![0-9])([0-9]{4}-[0-9]{2}-[0-9]{2})(?![0-9])")

#: A date-shaped path segment that is not a well-formed ISO date. Present so
#: a malformed date is caught rather than read as "this key has no date".
NEAR_MISS_DATE_RE = re.compile(r"(?<![0-9])([0-9]{4}-[0-9]{1,2}-[0-9]{1,2})(?![0-9])")


class NonTradingDayKeyError(ValueError):
    """A date that is not an NYSE session was used where a session is required.

    Always a raise, never a warning. A non-trading-day key silently drops out
    of every paired window downstream — the arm looks like it missed the
    cycle, and nothing anywhere reports why.
    """


def is_trading_day(day: dt.date) -> bool:
    """True when ``day`` is a full or half NYSE session.

    Straight delegation. Present so callers import one calendar module rather
    than two, and so a future calendar swap has one adapter to edit
    (principle 8).
    """
    return _krepis_is_trading_day(day)


def resolve_trading_day(now: dt.datetime | None = None) -> dt.date:
    """The trading day a run launched at ``now`` binds to.

    The last session whose close has passed: 09:00 on a trading day binds to
    the *previous* session, 16:30 binds to the same one, and a Saturday or a
    weekday holiday walks backward past it. `now` naive is read as NYSE
    local time; aware is converted.

    Idempotent by construction — resolving a resolved day at its own close
    returns it unchanged — which is what lets a pipeline resolve at several
    stages without walking backwards one session per stage.
    """
    return last_closed_trading_day(now)


def trading_weeks_between(start: dt.date, end: dt.date, *, rungs: bool = False) -> float | int:
    """Trading weeks in the half-open interval ``(start, end]``.

    ``rungs=False`` (default) returns the fractional count — trading days
    divided by five — which is what a duration wants.

    ``rungs=True`` returns the number of ladder rungs, counting a partial
    week as a whole one. §4.12 is explicit that "a holiday week is still one
    rung": the week of 2026-07-03 has four sessions, and a ladder that
    counted it as 0.8 of a rung would stretch every arm's eligibility clock
    by whatever holidays happened to fall inside it.

    Returns 0 when ``end <= start`` rather than a negative count: a negative
    number of weeks is not a thing a caller can act on, and an interval given
    backwards is a caller bug the caller should see as an empty window.
    """
    days = count_trading_days(start, end)
    if days <= 0:
        return 0 if rungs else 0.0
    if rungs:
        return -(-days // TRADING_DAYS_PER_WEEK)  # ceiling division
    return days / TRADING_DAYS_PER_WEEK


def assert_trading_day(day: dt.date | str, *, context: str) -> bool:
    """Raise :class:`NonTradingDayKeyError` unless ``day`` is an NYSE session.

    ``context`` is the key, field or window the date came from and is
    mandatory: an error reading "2026-08-29 is not a trading day" tells an
    operator nothing about which writer produced it.
    """
    if isinstance(day, str):
        try:
            parsed = dt.date.fromisoformat(day)
        except ValueError as exc:
            raise NonTradingDayKeyError(
                f"{day!r} in {context!r} is not a well-formed ISO-8601 date: {exc}"
            ) from exc
    else:
        parsed = day

    if not is_trading_day(parsed):
        raise NonTradingDayKeyError(
            f"{parsed.isoformat()} in {context!r} is not an NYSE trading day. "
            "Every key, window and horizon binds to a session (plan §4.12); a run "
            "launched on a non-trading day binds to the last completed session, "
            f"which is {resolve_trading_day(dt.datetime.combine(parsed, dt.time(23, 59)))}."
        )
    return True
