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
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

from crucible.tags import TAG_KEY, TAG_VALUE

__all__ = [
    "CostReading",
    "CostUnreadableError",
    "DailyReading",
    "default_client",
    "month_to_date_usd",
    "trailing_daily_usd",
]


class CostUnreadableError(RuntimeError):
    """Cost Explorer could not answer, so there is no number.

    Raised rather than returning 0.0. An identity without `ce:GetCostAndUsage`
    and an account that genuinely spent nothing produce the same float, and
    the gate must be able to tell them apart.
    """


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
    amounts = _amounts(client, start=start, end=end, granularity="MONTHLY", tagged=tagged)
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
    amounts = _amounts(client, start=start, end=today, granularity="DAILY", tagged=tagged)
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


def _amounts(
    client: Any, *, start: dt.date, end: dt.date, granularity: str, tagged: bool
) -> list[float]:
    """The `UnblendedCost` amount of every period Cost Explorer returns.

    One request shape for both readers — the month-to-date total and the
    trailing daily pace differ only in granularity and interval, and two
    request builders would be two places for the tag filter to drift.
    """
    request: dict[str, Any] = {
        "TimePeriod": {"Start": start.isoformat(), "End": end.isoformat()},
        "Granularity": granularity,
        "Metrics": ["UnblendedCost"],
    }
    if tagged:
        request["Filter"] = {"Tags": {"Key": TAG_KEY, "Values": [TAG_VALUE]}}
    try:
        response = client.get_cost_and_usage(**request)
    except Exception as exc:
        # The failure mode swallowed: none. This converts an exception into a
        # NAMED exception whose message carries the original class, and the
        # recording surface is the gate clause that renders it UNMEASURABLE.
        # Catching broadly is deliberate: `AccessDenied`, `NoRegionError`,
        # `NoCredentialsError`, `EndpointConnectionError` and an absent
        # `boto3` are five unrelated types that mean one thing here — there is
        # no number — and enumerating them would let the sixth read as green.
        raise CostUnreadableError(
            f"Cost Explorer could not be read for "
            f"{start.isoformat()}..{end.isoformat()}: {type(exc).__name__}: {exc}. "
            "That is a statement about our access, not about what was spent."
        ) from exc
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
