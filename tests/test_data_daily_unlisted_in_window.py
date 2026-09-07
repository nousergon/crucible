"""`run_daily`'s `symbols_unlisted_in_window` metric and the coverage floor
that must stay reachable around it (`alpha-engine-config-I10127`).

The expected-universe denominator never shrinks for an unlisted symbol —
it counts against coverage exactly like any other absent ticker — so a
handful of new listings dropping into a historical window degrades the
ratio instead of making the run unmeasurable, and a day that is ACTUALLY
too thin still fails.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from crucible.data import CoverageError, FramePriceSource, run_daily
from crucible.data.heal import run_heal, sessions_in_range
from crucible.keys import data_panel_key
from crucible.runner import run_job


def _extend(frames: dict[str, object], ticker: str, dates: list[dt.date]) -> dict[str, object]:
    rows = pd.DataFrame(
        {
            "open_raw": [1.0] * len(dates),
            "high_raw": [1.0] * len(dates),
            "low_raw": [1.0] * len(dates),
            "close_raw": [1.0] * len(dates),
            "volume_raw": [1e6] * len(dates),
        },
        index=pd.Index(dates, name="trading_day"),
    )
    out = dict(frames)
    out[ticker] = rows
    return out


class TestSymbolsUnlistedInWindowMetric:
    def test_an_unlisted_symbol_is_named_in_the_metric_and_coverage_still_passes(
        self, store, frames, cycle_date
    ) -> None:
        newco_dates = [cycle_date + dt.timedelta(days=200)]
        with_newco = _extend(frames, "NEWCO", newco_dates)
        expected = [*sorted(frames), "NEWCO"]

        ctx = run_job(
            "data.daily",
            lambda c: run_daily(c, source=FramePriceSource(with_newco), expected_symbols=expected),
            store=store,
            trading_day=cycle_date,
        )

        unlisted_metric = next(m for m in ctx.metrics if m["name"] == "symbols_unlisted_in_window")
        assert unlisted_metric["value"] == 1.0
        assert unlisted_metric["status"] == "OK"
        assert "NEWCO" in unlisted_metric["status_reason"]
        assert cycle_date.isoformat() in unlisted_metric["status_reason"]

        coverage_metric = next(m for m in ctx.metrics if m["name"] == "universe_coverage_ratio")
        expected_ratio = len(frames) / len(expected)
        assert coverage_metric["value"] == pytest.approx(expected_ratio)
        assert coverage_metric["status"] == "OK", (
            "an unlisted symbol counts against the SAME denominator as any other "
            "absent ticker — it degrades the ratio, it does not shrink it away"
        )
        assert store.exists(data_panel_key(cycle_date.isoformat())), (
            "unlisted is not a failure: the day still compiles"
        )

    def test_the_metric_is_zero_and_still_ok_when_nothing_is_unlisted(
        self, store, source, frames, cycle_date
    ) -> None:
        ctx = run_job(
            "data.daily",
            lambda c: run_daily(c, source=source, expected_symbols=sorted(frames)),
            store=store,
            trading_day=cycle_date,
        )
        metric = next(m for m in ctx.metrics if m["name"] == "symbols_unlisted_in_window")
        assert metric["value"] == 0.0
        assert metric["status"] == "OK", (
            "zero unlisted symbols is still a measured OK reading, never an absent metric"
        )

    def test_enough_unlisted_symbols_still_fails_the_coverage_floor(
        self, store, frames, cycle_date
    ) -> None:
        """Unlisted is a fact about the market, not an exemption from the floor:
        a day whose REAL coverage is too thin still fails, whether the absent
        tickers are outages or symbols that have not listed yet."""
        far_future = cycle_date + dt.timedelta(days=400)
        with_newco = dict(frames)
        newco_names = [f"NEWCO{i}" for i in range(60)]
        for name in newco_names:
            with_newco = _extend(with_newco, name, [far_future])
        expected = [*sorted(frames), *newco_names]

        with pytest.raises(CoverageError, match="below the floor"):
            run_job(
                "data.daily",
                lambda c: run_daily(
                    c, source=FramePriceSource(with_newco), expected_symbols=expected
                ),
                store=store,
                trading_day=cycle_date,
            )


class TestHealAcrossAListingBoundary:
    def test_a_symbol_that_lists_mid_range_is_unlisted_on_early_sessions_only(
        self, store, frames, cycle_date, monkeypatch
    ) -> None:
        """Three fixture sessions, one symbol listing on the middle day: the
        first heal session counts it unlisted, the later two include it."""
        monkeypatch.setattr("crucible.data.heal.in_region", lambda: (True, "test stand-in"))
        # Real NYSE sessions, taken from the fixture panel's own index rather
        # than computed by hand, so the range is guaranteed to be three
        # trading days `sessions_in_range` will accept.
        any_ticker_dates = sorted(next(iter(frames.values())).index)
        first, mid, last = any_ticker_dates[-3:]
        assert sessions_in_range(first, last) == [first, mid, last]

        midlist_dates = [mid, last]
        with_midlist = _extend(frames, "MIDLIST", midlist_dates)
        expected = [*sorted(frames), "MIDLIST"]

        ctx = run_job(
            "data.heal",
            lambda c: run_heal(
                c,
                source=FramePriceSource(with_midlist),
                start=first,
                end=last,
                gap="test-listing-boundary",
                expected_symbols=expected,
            ),
            store=store,
            trading_day=last,
        )

        unlisted_metrics = [m for m in ctx.metrics if m["name"] == "symbols_unlisted_in_window"]
        assert len(unlisted_metrics) == 3, "one run_daily call per healed session"
        by_reason = {m["status_reason"]: m["value"] for m in unlisted_metrics}
        assert any(
            v == 1.0 and "MIDLIST" in reason and first.isoformat() in reason
            for reason, v in by_reason.items()
        ), "the first session predates MIDLIST's listing: named, counted, never raised"
        later_values = [m["value"] for m in unlisted_metrics if m["value"] == 0.0]
        assert len(later_values) == 2, "mid and last both fall inside MIDLIST's stored history"
