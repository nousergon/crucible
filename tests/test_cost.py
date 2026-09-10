"""`crucible.cost` caches and budgets every Cost Explorer read.

Normative source: `alpha-engine-config-I10389`. CloudTrail measured 44,167
`ce:GetCostAndUsage` calls / $441.67 over four days (2026-09-03..09-06) —
`crucible.cost`'s own request shape, unfiltered and uncached, issued
thousands of times against a CLOSED historical window whose answer cannot
change. These tests assert the two properties the issue's `Closes-when`
names: one API call per distinct window per process, and a hard budget that
RAISES rather than spends past it.
"""

from __future__ import annotations

import datetime as dt

import pytest

from crucible.cost import (
    CostExplorerBudgetExceededError,
    CostExplorerCache,
    CostUnreadableError,
    closed_month_usd,
    month_to_date_usd,
    trailing_daily_usd,
)
from crucible.tags import cost_allocation_tag_status


class _CostClient:
    """A Cost Explorer stand-in that counts every real request it answers."""

    def __init__(self, amount: str = "9.05", daily: str = "1.00") -> None:
        self.amount = amount
        self.daily = daily
        self.requests: list[dict] = []
        self.tag_requests: list[dict] = []

    def get_cost_and_usage(self, **request) -> dict:
        self.requests.append(request)
        if request["Granularity"] == "DAILY":
            start = dt.date.fromisoformat(request["TimePeriod"]["Start"])
            end = dt.date.fromisoformat(request["TimePeriod"]["End"])
            days = (end - start).days
            return {
                "ResultsByTime": [
                    {"Total": {"UnblendedCost": {"Amount": self.daily}}} for _ in range(days)
                ]
            }
        return {
            "ResultsByTime": [
                {"Total": {"UnblendedCost": {"Amount": self.amount}, "Estimated": False}}
            ]
        }

    def list_cost_allocation_tags(self, **request) -> dict:
        self.tag_requests.append(request)
        return {
            "CostAllocationTags": [
                {"TagKey": "system", "Status": "Active", "LastUpdatedDate": "2026-08-01"}
            ]
        }


#: A date safely before this repo's real wall-clock "today", so every window
#: built from it is CLOSED under `CostExplorerCache.is_window_closed` no
#: matter when the suite runs.
TODAY = dt.date(2026, 8, 28)


class TestClosedWindowsAreCachedForTheProcess:
    def test_repeated_month_to_date_reads_issue_one_api_call(self) -> None:
        client = _CostClient()
        cache = CostExplorerCache()
        for _ in range(50):
            reading = month_to_date_usd(client, today=TODAY, tagged=False, cache=cache)
        assert reading.amount_usd == 9.05
        assert len(client.requests) == 1
        assert cache.calls == 1

    def test_repeated_closed_month_reads_issue_one_api_call(self) -> None:
        """The exact shape CloudTrail measured: MONTHLY/UnblendedCost over a
        closed prior month, re-evaluated many times in one process."""
        client = _CostClient()
        cache = CostExplorerCache()
        for _ in range(2000):
            reading = closed_month_usd(client, today=dt.date(2026, 9, 3), tagged=False, cache=cache)
        assert reading.amount_usd == 9.05
        assert len(client.requests) == 1
        assert cache.calls == 1

    def test_repeated_trailing_daily_reads_issue_one_api_call(self) -> None:
        client = _CostClient()
        cache = CostExplorerCache()
        for _ in range(50):
            reading = trailing_daily_usd(client, today=TODAY, days=30, tagged=True, cache=cache)
        assert reading.total_usd == pytest.approx(30.0)
        assert len(client.requests) == 1
        assert cache.calls == 1

    def test_distinct_windows_are_not_conflated(self) -> None:
        """Caching must not merge two genuinely different requests."""
        client = _CostClient()
        cache = CostExplorerCache()
        month_to_date_usd(client, today=TODAY, tagged=False, cache=cache)
        month_to_date_usd(client, today=TODAY, tagged=True, cache=cache)  # different tag filter
        trailing_daily_usd(client, today=TODAY, days=30, tagged=False, cache=cache)
        closed_month_usd(client, today=dt.date(2026, 9, 3), tagged=False, cache=cache)
        assert len(client.requests) == 4
        assert cache.calls == 4

    def test_a_closed_reading_survives_past_the_ttl(self) -> None:
        """A closed window is cached with NO ttl -- it must still be served
        from cache long after `open_window_ttl_seconds` would have expired an
        open one."""
        client = _CostClient()
        cache = CostExplorerCache(open_window_ttl_seconds=0.001)
        month_to_date_usd(client, today=TODAY, tagged=False, cache=cache)
        import time

        time.sleep(0.01)
        month_to_date_usd(client, today=TODAY, tagged=False, cache=cache)
        assert len(client.requests) == 1


class TestOpenWindowsGetATtlNotForever:
    def test_a_window_reaching_real_today_is_not_cached_forever(self, monkeypatch) -> None:
        """`is_window_closed` is conservative at the boundary: a request whose
        exclusive `End` IS real wall-clock today is OPEN, since Cost
        Explorer's own ~24h ingestion lag means the most recent day it names
        may not have settled yet."""
        real_today = dt.date(2026, 9, 9)
        client = _CostClient()
        cache = CostExplorerCache(open_window_ttl_seconds=1000, clock=lambda: real_today)

        clock_ticks = [0.0]
        monkeypatch.setattr("crucible.cost.time.monotonic", lambda: clock_ticks[0])

        month_to_date_usd(client, today=real_today, tagged=False, cache=cache)
        assert len(client.requests) == 1

        clock_ticks[0] = 500.0  # inside the TTL
        month_to_date_usd(client, today=real_today, tagged=False, cache=cache)
        assert len(client.requests) == 1, "still within the TTL -- must be served from cache"

        clock_ticks[0] = 1500.0  # past the TTL
        month_to_date_usd(client, today=real_today, tagged=False, cache=cache)
        assert len(client.requests) == 2, "past the TTL -- an open window must be re-read"


class TestTheCallBudgetRaisesRatherThanSpends:
    def test_exceeding_the_budget_raises_named_exception(self) -> None:
        client = _CostClient()
        cache = CostExplorerCache(budget=3)
        # Three genuinely distinct windows (different months) exhaust the budget...
        month_to_date_usd(client, today=dt.date(2026, 6, 15), tagged=False, cache=cache)
        month_to_date_usd(client, today=dt.date(2026, 7, 15), tagged=False, cache=cache)
        month_to_date_usd(client, today=dt.date(2026, 8, 15), tagged=False, cache=cache)
        assert cache.calls == 3
        # ... and a fourth distinct window is refused before it is ever sent.
        with pytest.raises(CostExplorerBudgetExceededError):
            month_to_date_usd(client, today=dt.date(2026, 9, 3), tagged=False, cache=cache)
        assert len(client.requests) == 3, "the over-budget call must never reach the client"

    def test_a_cache_hit_never_counts_against_the_budget(self) -> None:
        """44,167 REPEATS of the same window must cost one call, not one per
        repeat -- the budget exists for distinct windows, not for re-reading
        an answer already known."""
        client = _CostClient()
        cache = CostExplorerCache(budget=1)
        for _ in range(10_000):
            month_to_date_usd(client, today=TODAY, tagged=False, cache=cache)
        assert len(client.requests) == 1
        assert cache.calls == 1

    def test_a_denied_read_still_counts_as_an_attempt(self) -> None:
        """A failing fetch still represents a billable, attempted request --
        the budget must not let a process retry a denial indefinitely for
        free."""

        class _AlwaysDenies:
            def get_cost_and_usage(self, **_request):
                raise RuntimeError("AccessDeniedException")

        cache = CostExplorerCache(budget=2)
        client = _AlwaysDenies()
        for expected_today in (dt.date(2026, 6, 15), dt.date(2026, 7, 15)):
            with pytest.raises(CostUnreadableError):
                month_to_date_usd(client, today=expected_today, tagged=False, cache=cache)
        assert cache.calls == 2
        with pytest.raises(CostExplorerBudgetExceededError):
            month_to_date_usd(client, today=dt.date(2026, 8, 15), tagged=False, cache=cache)


class TestCostAllocationTagStatusSharesTheSameCacheAndBudget:
    def test_repeated_reads_issue_one_api_call(self) -> None:
        client = _CostClient()
        cache = CostExplorerCache()
        for _ in range(50):
            status = cost_allocation_tag_status(client, cache=cache)
        assert status.active
        assert len(client.tag_requests) == 1
        assert cache.calls == 1

    def test_it_shares_the_budget_with_get_cost_and_usage(self) -> None:
        client = _CostClient()
        cache = CostExplorerCache(budget=1)
        month_to_date_usd(client, today=TODAY, tagged=False, cache=cache)
        with pytest.raises(CostExplorerBudgetExceededError):
            cost_allocation_tag_status(client, cache=cache)


class TestDefaultCacheIsProcessWideAndSharedByBothModules:
    def test_default_cache_is_a_singleton_across_calls(self) -> None:
        from crucible.cost import default_cache, reset_default_cache

        reset_default_cache()
        try:
            assert default_cache() is default_cache()
        finally:
            reset_default_cache()

    def test_reset_default_cache_drops_prior_state(self) -> None:
        from crucible.cost import default_cache, reset_default_cache

        reset_default_cache()
        try:
            client = _CostClient()
            month_to_date_usd(client, today=TODAY, tagged=False)  # uses default_cache()
            assert default_cache().calls == 1
            reset_default_cache()
            assert default_cache().calls == 0
        finally:
            reset_default_cache()
