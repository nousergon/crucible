"""Month-to-date AWS spend, read from Cost Explorer.

Normative source: plan §6 row 2 (`AWS <= $40`) and row 4 (`AWS total <=
$70/mo`) — the two phase gates that are a dollar figure.

**A cost ceiling read from an unreachable API is UNMEASURABLE, never zero.**
`ce:GetCostAndUsage` is denied to most identities by default, Cost Explorer
has no per-account default region, and a laptop with no credentials raises
before any request leaves. Each of those produces *no number*, and the one
answer they must never produce is `$0.00 — under the ceiling`. So every
failure here RAISES :class:`CostUnreadableError` carrying the underlying
exception class, and `crucible.gate` turns that into an UNMEASURABLE clause
naming the class rather than a met one.

**Tag-filtered and unfiltered are the same reader.** Phase 2 grades the spend
carrying `system=crucible-v2`; phase 4 grades the account total. A second
function for the second question is how two spellings of one reading drift
apart, so the tag filter is a parameter and the ceiling belongs to the caller.

**No account id, no ARN, no region literal.** The client is injected, and the
tag key/value come from :mod:`crucible.tags`, which already owns them.

**Every read here is cached and budgeted (`alpha-engine-config-I10389`).**
CloudTrail measured 44,167 `ce:GetCostAndUsage` calls / $441.67 over four days
(2026-09-03..09-06) from this module's own shape — an unfiltered
`MONTHLY`/`UnblendedCost` read of a CLOSED historical window
(`2026-08-01..2026-08-28`), issued thousands of times against an answer that
cannot change. `ce:GetCostAndUsage` is billed **per request at $0.01**, three
orders of magnitude more than a typical AWS read, so a cache miss here is a
spending decision and a retry is a cost multiplier — the fleet's normal
"retry transient failures" reasoning does not transfer to this API.

**Why a cache and not just fewer call sites.** A closed historical window
(the interval this request names has already fully elapsed in real wall-clock
time) is immutable: the same `(start, end, granularity, tagged)` request
returns the same amount every time, forever, for this process — memoized with
no TTL. A window reaching into today is still accruing and could in
principle answer differently a moment later, so it gets a bounded TTL
instead. **Cost Explorer's own data lags ~24h regardless** — re-asking a
still-open window sooner than the TTL does not produce a fresher number, it
produces the same number at a higher price — so the TTL exists only to bound
how long one long-running process trusts a single reading, not to chase
freshness. See :class:`CostExplorerCache`.

**The budget is a hard, named failure, never a silent spend.** Every cache
miss counts against a per-process call budget (`CRUCIBLE_CE_CALL_BUDGET`,
default :data:`DEFAULT_CE_CALL_BUDGET`); exceeding it raises
:class:`CostExplorerBudgetExceededError` before the request is made, in the
same shape :class:`CostUnreadableError` already uses — `crucible.gate` turns
either into an UNMEASURABLE clause rather than letting the process keep
spending.
"""

from __future__ import annotations

import datetime as dt
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from crucible.tags import TAG_KEY, TAG_VALUE

__all__ = [
    "ClosedMonthReading",
    "CostExplorerBudgetExceededError",
    "CostExplorerCache",
    "CostReading",
    "CostUnreadableError",
    "DailyReading",
    "DEFAULT_CE_CALL_BUDGET",
    "DEFAULT_CE_OPEN_WINDOW_TTL_SECONDS",
    "closed_month_usd",
    "default_cache",
    "default_client",
    "month_to_date_usd",
    "reset_default_cache",
    "trailing_daily_usd",
]


class CostUnreadableError(RuntimeError):
    """Cost Explorer could not answer, so there is no number.

    Raised rather than returning 0.0. An identity without `ce:GetCostAndUsage`
    and an account that genuinely spent nothing produce the same float, and
    the gate must be able to tell them apart.
    """


class CostExplorerBudgetExceededError(RuntimeError):
    """A process asked Cost Explorer more times than its declared budget.

    Raised BEFORE the request is made, rather than spending past the budget
    silently. `ce:GetCostAndUsage` and `ce:ListCostAllocationTags` are billed
    per request at $0.01 — an unbounded retry loop or a clause re-evaluated
    once per caller is a spending decision, not a resilience measure
    (`alpha-engine-config-I10389`: 44,167 uncached calls / $441.67 in four
    days, eight times either of the account's two prior FULL months' bills).
    `crucible.gate` turns this into an UNMEASURABLE clause the same way it
    already does :class:`CostUnreadableError` — this is a producer, and the
    fleet's default is fail loud rather than silently degrade.
    """


#: Per-process `ce:GetCostAndUsage` / `ce:ListCostAllocationTags` call budget.
#: Sized for one `crucible gate`/`crucible board` invocation, which reads at
#: most a handful of distinct `(start, end, granularity, tagged)` windows —
#: not for a test suite or a script that legitimately wants many distinct
#: windows in one process, which should pass its own `CostExplorerCache` or
#: raise `CRUCIBLE_CE_CALL_BUDGET`.
DEFAULT_CE_CALL_BUDGET = 25

#: How long a reading for a window that has not fully closed is trusted
#: before this process asks again. Cost Explorer's own data lags ~24h, so a
#: shorter TTL does not buy a fresher number — it only bounds how long a
#: long-running process (a daemon, a `--watch` loop) keeps citing one
#: reading before paying for another.
DEFAULT_CE_OPEN_WINDOW_TTL_SECONDS = 900


@dataclass
class _CacheEntry:
    value: Any
    fetched_monotonic: float
    closed: bool


class CostExplorerCache:
    """Per-process memoization + hard call budget for Cost Explorer reads.

    Keyed on the exact parameters of one request (`kind`, `start`, `end`,
    `granularity`, `tagged`, ...). A **closed** window — one whose own `end`
    boundary has already fully elapsed in real wall-clock time — cannot
    change, so it is cached for the life of the process with no TTL. An
    **open** window (it reaches into today) is cached for
    `open_window_ttl_seconds` — long enough to absorb the several reads one
    clause evaluation makes, short enough that a long-running process still
    re-checks occasionally rather than trusting one number forever.

    Every cache MISS increments a hard counter and raises
    :class:`CostExplorerBudgetExceededError` once `budget` is exceeded,
    BEFORE `fetch` is called — a failed or denied fetch still counts, since
    it still represents an attempted, billable request.

    `clock` is injectable so a test can pin "real wall-clock today" without
    reaching for `dt.date.today()` — this repo's own rule (`AGENTS.md`
    "Trading days, always" / test discipline) is to use fixed date literals,
    never live clock arithmetic, in a test.
    """

    def __init__(
        self,
        *,
        budget: int = DEFAULT_CE_CALL_BUDGET,
        open_window_ttl_seconds: float = DEFAULT_CE_OPEN_WINDOW_TTL_SECONDS,
        clock: Callable[[], dt.date] = dt.date.today,
    ) -> None:
        self._budget = budget
        self._ttl = open_window_ttl_seconds
        self._clock = clock
        self._store: dict[tuple[Any, ...], _CacheEntry] = {}
        self._calls = 0
        self._lock = threading.Lock()

    @property
    def calls(self) -> int:
        """Requests actually issued (cache misses), this process."""
        return self._calls

    @property
    def budget(self) -> int:
        return self._budget

    def is_window_closed(self, end: dt.date) -> bool:
        """Whether a request whose exclusive `End` is `end` has fully elapsed.

        `end` is exclusive in Cost Explorer's own grammar, so the last day
        actually covered is `end - 1 day`; a window is closed once that day
        is strictly before real wall-clock today. Deliberately conservative
        at the boundary — a window whose `end` IS today is treated as open
        (TTL'd, not cached forever), since the most recent day it names may
        not have finished settling under Cost Explorer's own ~24h ingestion
        lag.
        """
        return end < self._clock()

    def get_or_fetch(self, key: tuple[Any, ...], *, closed: bool, fetch: Callable[[], Any]) -> Any:
        with self._lock:
            entry = self._store.get(key)
            if entry is not None:
                if entry.closed or (time.monotonic() - entry.fetched_monotonic) < self._ttl:
                    return entry.value
            if self._calls >= self._budget:
                raise CostExplorerBudgetExceededError(
                    f"Cost Explorer call budget of {self._budget} exhausted for this "
                    f"process, on key {key!r}. `ce:GetCostAndUsage` and "
                    "`ce:ListCostAllocationTags` are billed per request at $0.01 -- "
                    "raising rather than spending past the declared budget. Set "
                    "CRUCIBLE_CE_CALL_BUDGET if this process genuinely reads more "
                    "distinct windows than that."
                )
            self._calls += 1
            value = fetch()
            self._store[key] = _CacheEntry(
                value=value, fetched_monotonic=time.monotonic(), closed=closed
            )
            return value

    def snapshot(self) -> dict[str, Any]:
        """Every cached reading's key and closedness, for a caller that wants
        to persist this process's Cost Explorer activity onto a run manifest
        or other durable artifact, rather than re-deriving it from CloudTrail."""
        return {
            "calls": self._calls,
            "budget": self._budget,
            "entries": [{"key": list(key), "closed": e.closed} for key, e in self._store.items()],
        }


def _budget_from_env() -> int:
    raw = os.environ.get("CRUCIBLE_CE_CALL_BUDGET")
    if raw:
        try:
            return int(raw)
        except ValueError:
            pass
    return DEFAULT_CE_CALL_BUDGET


_default_cache: CostExplorerCache | None = None
_default_cache_lock = threading.Lock()


def default_cache() -> CostExplorerCache:
    """The process-wide cache + budget, created lazily on first use.

    Shared by every `crucible.cost` and `crucible.tags` Cost Explorer reader
    in this process (both modules call this rather than each holding their
    own), so one budget bounds the whole process's Cost Explorer spend
    rather than one function's calls in isolation.
    """
    global _default_cache
    with _default_cache_lock:
        if _default_cache is None:
            _default_cache = CostExplorerCache(budget=_budget_from_env())
        return _default_cache


def reset_default_cache() -> None:
    """Test-only: drop the process-wide cache so a test starts clean.

    Without this, two tests in the same pytest process would share cached
    Cost Explorer readings and call-budget state across test boundaries.
    """
    global _default_cache
    with _default_cache_lock:
        _default_cache = None


@dataclass(frozen=True)
class CostReading:
    """One month-to-date amount, and exactly what was asked for."""

    start: dt.date
    end: dt.date
    amount_usd: float
    tag_filter: str | None

    @property
    def scope(self) -> str:
        return self.tag_filter or "the whole account"

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "amount_usd": self.amount_usd,
            "tag_filter": self.tag_filter,
        }


@dataclass(frozen=True)
class DailyReading:
    """One amount per COMPLETE calendar day over ``[start, end)``, oldest first.

    ``end`` is exclusive and is the day the reading was taken, so every day
    carried here has closed — Cost Explorer's figure for the current day is
    partial by construction and would read as a cheap day on every render.
    """

    start: dt.date
    end: dt.date
    amounts_usd: tuple[float, ...]
    tag_filter: str | None

    @property
    def scope(self) -> str:
        return self.tag_filter or "the whole account"

    @property
    def total_usd(self) -> float:
        return sum(self.amounts_usd)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "amounts_usd": list(self.amounts_usd),
            "tag_filter": self.tag_filter,
        }


@dataclass(frozen=True)
class ClosedMonthReading:
    """The unblended total for one CLOSED prior calendar month.

    Normative source: `alpha-engine-config-I9946`. `month_to_date_usd` and
    `trailing_daily_usd` both grade a month IN PROGRESS; neither ever compares
    a completed month against its own ceiling, so a month that projected
    under all the way through and closed over left no red row anywhere.

    ``estimated`` is Cost Explorer's own `Estimated` flag on this period, not
    derived: Cost Explorer finalises a month a few days into the next one, so
    a reading taken while it is still `True` is a statement that can move —
    reading it as final would grade a closed month against a number that
    is not actually closed yet.
    """

    start: dt.date
    end: dt.date
    amount_usd: float
    estimated: bool
    tag_filter: str | None

    @property
    def scope(self) -> str:
        return self.tag_filter or "the whole account"

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "amount_usd": self.amount_usd,
            "estimated": self.estimated,
            "tag_filter": self.tag_filter,
        }


def default_client() -> Any:
    """A Cost Explorer client, constructed lazily.

    `boto3` is imported inside the function for the same reason
    `crucible.store.S3Store` does it: importing `crucible.gate` must not
    require an AWS SDK to be installed or credentials to be resolvable, and a
    module-level client would make every unit test that imports the gate
    reach for a credential chain.
    """
    import boto3  # noqa: PLC0415 - lazy on purpose; see the docstring

    return boto3.client("ce")


def month_to_date_usd(
    client: Any,
    *,
    today: dt.date,
    tagged: bool,
    cache: CostExplorerCache | None = None,
) -> CostReading:
    """Unblended USD spent so far this calendar month.

    Calendar month, not trading days — one of §4.12's exhaustive exceptions,
    and not a discretionary one: the plan's ceiling is "$40" and "$70/mo", AWS
    bills on the calendar month, and grading a monthly ceiling over a
    trading-day window would compare a number to a budget it was never drawn
    against.

    ``end`` is exclusive in Cost Explorer's own grammar and must be strictly
    after ``start``, so a reading taken on the 1st of a month asks for
    [1st, 2nd) rather than the empty interval the naive month-to-date
    expression produces — an empty interval returns no `ResultsByTime` group
    at all, which would read as `$0.00` on exactly one day a month.
    """
    start = today.replace(day=1)
    end = max(today, start + dt.timedelta(days=1))
    amounts = _amounts(
        client, start=start, end=end, granularity="MONTHLY", tagged=tagged, cache=cache
    )
    return CostReading(
        start=start,
        end=end,
        amount_usd=sum(amounts),
        tag_filter=f"{TAG_KEY}={TAG_VALUE}" if tagged else None,
    )


def trailing_daily_usd(
    client: Any,
    *,
    today: dt.date,
    days: int,
    tagged: bool,
    cache: CostExplorerCache | None = None,
) -> DailyReading:
    """Unblended USD for each of the ``days`` complete days before ``today``.

    The window is ``[today - days, today)`` — it ends at yesterday's close and
    may cross a month boundary, because a spending PACE is a property of the
    estate, not of the calendar month it happens to be read in. Cost Explorer
    must answer with exactly one period per day; fewer periods than days is
    not "the rest were free", it is an incomplete answer and raises.
    """
    if days < 1:
        raise ValueError("a trailing window of fewer than one day measures nothing")
    start = today - dt.timedelta(days=days)
    amounts = _amounts(
        client, start=start, end=today, granularity="DAILY", tagged=tagged, cache=cache
    )
    if len(amounts) != days:
        raise CostUnreadableError(
            f"Cost Explorer returned {len(amounts)} daily period(s) for "
            f"{start.isoformat()}..{today.isoformat()}, not {days}. A day the API did not "
            "answer for is not a day that cost nothing."
        )
    return DailyReading(
        start=start,
        end=today,
        amounts_usd=tuple(amounts),
        tag_filter=f"{TAG_KEY}={TAG_VALUE}" if tagged else None,
    )


def closed_month_usd(
    client: Any,
    *,
    today: dt.date,
    tagged: bool,
    cache: CostExplorerCache | None = None,
) -> ClosedMonthReading:
    """The PRIOR calendar month's closed total (`alpha-engine-config-I9946`).

    The prior month is derived from ``today`` regardless of which day of the
    month it names — the CALL SITE decides when it is worth asking (only the
    first few days of a month; a read taken mid-month would just restate
    month-to-date under another name). One request, MONTHLY granularity,
    over exactly ``[prior month's 1st, this month's 1st)`` — Cost Explorer
    answers with exactly one period for that interval, and this reads its
    `Estimated` flag rather than assuming the number is final.
    """
    end = today.replace(day=1)
    start = (end - dt.timedelta(days=1)).replace(day=1)
    amount, estimated = _single_period(client, start=start, end=end, tagged=tagged, cache=cache)
    return ClosedMonthReading(
        start=start,
        end=end,
        amount_usd=amount,
        estimated=estimated,
        tag_filter=f"{TAG_KEY}={TAG_VALUE}" if tagged else None,
    )


def _single_period(
    client: Any,
    *,
    start: dt.date,
    end: dt.date,
    tagged: bool,
    cache: CostExplorerCache | None = None,
) -> tuple[float, bool]:
    """One MONTHLY period's amount and its `Estimated` flag.

    Shares `_amounts`'s request shape, caching and exception handling
    exactly; the difference is this reader also needs the `Estimated` bit
    `_amounts` discards, since a month-close reading must say when the
    number can still move (`alpha-engine-config-I9946`).
    """
    response = _get_cost_and_usage(
        client, start=start, end=end, granularity="MONTHLY", tagged=tagged, cache=cache
    )
    periods = response.get("ResultsByTime") or []
    if len(periods) != 1:
        raise CostUnreadableError(
            f"Cost Explorer returned {len(periods)} period(s) for "
            f"{start.isoformat()}..{end.isoformat()}, not 1. A closed-month reading needs "
            "exactly one MONTHLY period."
        )
    period = periods[0]
    amount = ((period.get("Total") or {}).get("UnblendedCost") or {}).get("Amount")
    if amount is None:
        raise CostUnreadableError(
            f"Cost Explorer returned a period with no `UnblendedCost` amount for "
            f"{start.isoformat()}..{end.isoformat()}. A missing amount is not zero."
        )
    try:
        amount_usd = float(amount)
    except (TypeError, ValueError) as exc:
        raise CostUnreadableError(
            f"Cost Explorer returned {amount!r} as an amount, which is not a number: "
            f"{type(exc).__name__}"
        ) from exc
    return amount_usd, bool(period.get("Estimated", False))


def _get_cost_and_usage(
    client: Any,
    *,
    start: dt.date,
    end: dt.date,
    granularity: str,
    tagged: bool,
    cache: CostExplorerCache | None,
) -> dict[str, Any]:
    """The cached, budgeted `ce:GetCostAndUsage` call shared by every reader.

    One request builder for both `_amounts` and `_single_period` — they
    differ only in how they parse the response, and two request builders
    would be two places for the tag filter, and the cache key, to drift.

    Caching is keyed on the exact request (`start`, `end`, `granularity`,
    `tagged`): a CLOSED window's answer is memoized for the process's
    lifetime (`alpha-engine-config-I10389` — the defect this closes was
    exactly this call, unfiltered and uncached, issued thousands of times
    against an immutable prior-month window); an OPEN window (reaching into
    today) is memoized with a TTL. See `CostExplorerCache`.
    """
    cache = cache or default_cache()
    request: dict[str, Any] = {
        "TimePeriod": {"Start": start.isoformat(), "End": end.isoformat()},
        "Granularity": granularity,
        "Metrics": ["UnblendedCost"],
    }
    if tagged:
        request["Filter"] = {"Tags": {"Key": TAG_KEY, "Values": [TAG_VALUE]}}
    key = ("get_cost_and_usage", start.isoformat(), end.isoformat(), granularity, tagged)

    def fetch() -> dict[str, Any]:
        try:
            return client.get_cost_and_usage(**request)
        except Exception as exc:
            # The failure mode swallowed: none. This converts an exception
            # into a NAMED exception whose message carries the original
            # class, and the recording surface is the gate clause that
            # renders it UNMEASURABLE. Catching broadly is deliberate:
            # `AccessDenied`, `NoRegionError`, `NoCredentialsError`,
            # `EndpointConnectionError` and an absent `boto3` are five
            # unrelated types that mean one thing here — there is no number —
            # and enumerating them would let the sixth read as green.
            raise CostUnreadableError(
                f"Cost Explorer could not be read for "
                f"{start.isoformat()}..{end.isoformat()}: {type(exc).__name__}: {exc}. "
                "That is a statement about our access, not about what was spent."
            ) from exc

    return cache.get_or_fetch(key, closed=cache.is_window_closed(end), fetch=fetch)


def _amounts(
    client: Any,
    *,
    start: dt.date,
    end: dt.date,
    granularity: str,
    tagged: bool,
    cache: CostExplorerCache | None = None,
) -> list[float]:
    """The `UnblendedCost` amount of every period Cost Explorer returns.

    One request shape for both readers — the month-to-date total and the
    trailing daily pace differ only in granularity and interval, and two
    request builders would be two places for the tag filter to drift.
    """
    response = _get_cost_and_usage(
        client, start=start, end=end, granularity=granularity, tagged=tagged, cache=cache
    )
    periods = response.get("ResultsByTime") or []
    if not periods:
        raise CostUnreadableError(
            f"Cost Explorer returned no period for {start.isoformat()}..{end.isoformat()}. "
            "An empty response is not a spend of zero."
        )
    amounts: list[float] = []
    for period in periods:
        amount = ((period.get("Total") or {}).get("UnblendedCost") or {}).get("Amount")
        if amount is None:
            raise CostUnreadableError(
                f"Cost Explorer returned a period with no `UnblendedCost` amount for "
                f"{start.isoformat()}..{end.isoformat()}. A missing amount is not zero."
            )
        try:
            amounts.append(float(amount))
        except (TypeError, ValueError) as exc:
            raise CostUnreadableError(
                f"Cost Explorer returned {amount!r} as an amount, which is not a "
                f"number: {type(exc).__name__}"
            ) from exc
    return amounts
