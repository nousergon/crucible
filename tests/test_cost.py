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


class TestTheBudgetIsEnvOverridableAndZeroMeansZero:
    """`CRUCIBLE_CE_CALL_BUDGET` is the documented escape hatch for a process
    that legitimately reads more distinct windows than `DEFAULT_CE_CALL_BUDGET`
    (`crucible/cost.py` module docstring). It was documented and never tested,
    and the fleet has repeatedly shipped guards that were never shown to fire
    (`alpha-engine-config-I11201`).

    The `0` case is the one that matters: `0` is the value an operator reaches
    for to mean "make no Cost Explorer calls at all", and a truthiness check on
    the raw string would silently give them the default of 25 instead -- a
    setting that reads as a hard stop while permitting 25 billed requests.
    """

    def test_the_env_var_sets_the_default_cache_budget(self, monkeypatch) -> None:
        from crucible.cost import default_cache, reset_default_cache

        monkeypatch.setenv("CRUCIBLE_CE_CALL_BUDGET", "3")
        reset_default_cache()
        try:
            assert default_cache().budget == 3
        finally:
            reset_default_cache()

    def test_zero_refuses_the_first_call_rather_than_disabling_the_budget(
        self, monkeypatch
    ) -> None:
        from crucible.cost import default_cache, reset_default_cache

        monkeypatch.setenv("CRUCIBLE_CE_CALL_BUDGET", "0")
        reset_default_cache()
        try:
            assert default_cache().budget == 0
            client = _CostClient()
            with pytest.raises(CostExplorerBudgetExceededError):
                month_to_date_usd(client, today=TODAY, tagged=False)
            assert client.requests == [], "nothing may be sent under a zero budget"
        finally:
            reset_default_cache()

    def test_an_unparseable_value_falls_back_to_the_declared_default(self, monkeypatch) -> None:
        """A typo must not silently mean "unbounded". It means the default."""
        from crucible.cost import DEFAULT_CE_CALL_BUDGET, default_cache, reset_default_cache

        monkeypatch.setenv("CRUCIBLE_CE_CALL_BUDGET", "lots")
        reset_default_cache()
        try:
            assert default_cache().budget == DEFAULT_CE_CALL_BUDGET
        finally:
            reset_default_cache()


# ---------------------------------------------------------------------------
# The expense collector's series is the spend source (alpha-engine-config-I11707)
# ---------------------------------------------------------------------------


class _FakeS3:
    def __init__(self, doc: dict) -> None:
        import json  # noqa: PLC0415 - local to the fake

        self.body = json.dumps(doc).encode()
        self.gets = 0

    def get_object(self, *, Bucket: str, Key: str) -> dict:
        assert (Bucket, Key) == ("bucket", "expenses/latest.json")
        self.gets += 1
        import io  # noqa: PLC0415 - local to the fake

        return {"Body": io.BytesIO(self.body)}


def _rollup(
    *,
    as_of: str = "2026-09-29T12:15:51+00:00",
    start: dt.date = dt.date(2026, 8, 1),
    end: dt.date = dt.date(2026, 9, 29),
    tagged: float = 2.0,
    untagged: float = 1.0,
    complete: bool = True,
    estimated_from: dt.date = dt.date(2026, 9, 27),
) -> dict:
    days, d = [], start
    while d < end:
        days.append(
            {
                "date": d.isoformat(),
                "estimated": d >= estimated_from,
                "by_system_usd": {"crucible-v2": tagged, "(untagged)": untagged},
            }
        )
        d += dt.timedelta(days=1)
    return {
        "as_of": as_of,
        "providers": [
            {"key": "openrouter", "detail": {}},
            {
                "key": "aws",
                "detail": {
                    "daily_by_system": {
                        "tag_key": "system",
                        "untagged_key": "(untagged)",
                        "start": start.isoformat(),
                        "end": end.isoformat(),
                        "complete": complete,
                        "days": days,
                    }
                },
            },
        ],
    }


def _client(doc: dict, *, now: str = "2026-09-29T21:30:00+00:00"):
    from crucible.cost import CollectorSpendClient  # noqa: PLC0415

    return CollectorSpendClient(
        "s3://bucket/expenses/latest.json",
        s3=_FakeS3(doc),
        now=lambda: dt.datetime.fromisoformat(now),
    )


class TestTheCollectorSeriesAnswersEveryReader:
    """The three readers the dollar clause uses, driven through the real
    request builder and cache against the collector's series."""

    def test_month_to_date_tagged_and_whole_account(self) -> None:
        from crucible.cost import month_to_date_usd  # noqa: PLC0415

        client = _client(_rollup())
        cache = CostExplorerCache(budget=10)
        tagged = month_to_date_usd(client, today=dt.date(2026, 9, 29), tagged=True, cache=cache)
        whole = month_to_date_usd(client, today=dt.date(2026, 9, 29), tagged=False, cache=cache)
        assert tagged.amount_usd == pytest.approx(28 * 2.0)
        assert whole.amount_usd == pytest.approx(28 * 3.0)

    def test_trailing_thirty_days_is_thirty_daily_periods(self) -> None:
        from crucible.cost import trailing_daily_usd  # noqa: PLC0415

        reading = trailing_daily_usd(
            _client(_rollup()),
            today=dt.date(2026, 9, 29),
            days=30,
            tagged=False,
            cache=CostExplorerCache(budget=10),
        )
        assert len(reading.amounts_usd) == 30
        assert reading.total_usd == pytest.approx(90.0)

    def test_the_closed_month_carries_the_estimated_flag(self) -> None:
        from crucible.cost import closed_month_usd  # noqa: PLC0415

        final = closed_month_usd(
            _client(_rollup()),
            today=dt.date(2026, 9, 2),
            tagged=True,
            cache=CostExplorerCache(budget=10),
        )
        assert final.amount_usd == pytest.approx(31 * 2.0)
        assert final.estimated is False
        provisional = closed_month_usd(
            _client(_rollup(estimated_from=dt.date(2026, 8, 30))),
            today=dt.date(2026, 9, 2),
            tagged=True,
            cache=CostExplorerCache(budget=10),
        )
        assert provisional.estimated is True

    def test_the_rollup_is_read_once_per_client(self) -> None:
        from crucible.cost import month_to_date_usd, trailing_daily_usd  # noqa: PLC0415

        client = _client(_rollup())
        cache = CostExplorerCache(budget=10)
        month_to_date_usd(client, today=dt.date(2026, 9, 29), tagged=True, cache=cache)
        trailing_daily_usd(client, today=dt.date(2026, 9, 29), days=30, tagged=True, cache=cache)
        assert client._s3.gets == 1


class TestTheCollectorSeriesRaisesWhenItCannotAnswer:
    """Every one of these is no number, and the gate renders it UNMEASURABLE."""

    def _mtd(self, client):
        from crucible.cost import month_to_date_usd  # noqa: PLC0415

        return month_to_date_usd(
            client, today=dt.date(2026, 9, 29), tagged=False, cache=CostExplorerCache(budget=10)
        )

    def test_a_stale_rollup(self) -> None:
        with pytest.raises(CostUnreadableError, match="older than"):
            self._mtd(_client(_rollup(as_of="2026-09-27T12:15:00+00:00")))

    def test_a_truncated_series(self) -> None:
        with pytest.raises(CostUnreadableError, match="truncated"):
            self._mtd(_client(_rollup(complete=False)))

    def test_a_day_outside_the_series(self) -> None:
        with pytest.raises(CostUnreadableError, match="has no 2026-09-28"):
            self._mtd(_client(_rollup(end=dt.date(2026, 9, 28))))

    def test_a_missing_series_names_the_collectors_error(self) -> None:
        doc = _rollup()
        doc["providers"][1]["detail"] = {"daily_by_system_error": "RuntimeError: AccessDenied"}
        with pytest.raises(CostUnreadableError, match="AccessDenied"):
            self._mtd(_client(doc))

    def test_an_unknown_filter(self) -> None:
        with pytest.raises(CostUnreadableError, match="filter"):
            _client(_rollup()).get_cost_and_usage(
                TimePeriod={"Start": "2026-09-01", "End": "2026-09-29"},
                Granularity="MONTHLY",
                Filter={"Dimensions": {"Key": "SERVICE", "Values": ["Amazon EC2"]}},
            )


class TestTagActivationIsEstablishedFromEvidence:
    def test_tagged_spend_proves_active(self) -> None:
        status = cost_allocation_tag_status(_client(_rollup()), cache=CostExplorerCache(budget=5))
        assert status.active

    def test_no_tagged_spend_proves_nothing(self) -> None:
        from crucible.tags import CostAllocationTagUnreadableError  # noqa: PLC0415

        with pytest.raises(CostAllocationTagUnreadableError):
            cost_allocation_tag_status(
                _client(_rollup(tagged=0.0)), cache=CostExplorerCache(budget=5)
            )
