"""The data layer's refusals — the behaviour that matters is what it will not do."""

from __future__ import annotations

import datetime as dt
import json

import pytest

from crucible.data import (
    CoverageError,
    DataGapError,
    FramePriceSource,
    MissingSourceError,
    NotInRegionError,
    run_daily,
    run_heal,
    run_weekly,
    week_sessions,
)
from crucible.data.daily import UndeclaredUniverseError
from crucible.data.heal import LAPTOP_SESSION_ALLOWANCE
from crucible.keys import coverage_key, data_panel_key, features_key
from crucible.manifest import read_manifest
from crucible.runner import run_job


class TestMissingSourceIsFailedNeverZeroFilled:
    def test_an_empty_source_fails_the_run_and_writes_a_failed_manifest(
        self, store, cycle_date
    ) -> None:
        source = FramePriceSource({}, snapshot="frames:empty")
        with pytest.raises(MissingSourceError):
            run_job(
                "data.daily",
                lambda c: run_daily(c, source=source, expected_symbols=["AAA"]),
                store=store,
                trading_day=cycle_date,
            )
        manifest = read_manifest(store, "data.daily", cycle_date.isoformat())
        assert manifest["status"] == "failed"
        assert "MissingSourceError" in manifest["reason"]
        assert not store.exists(data_panel_key(cycle_date.isoformat())), (
            "a failed compile must leave no panel behind; a partial panel that a later "
            "job reads is the zero-fill by another route"
        )

    def test_a_frame_short_a_column_is_refused_rather_than_read_as_zero(
        self, frames, cycle_date
    ) -> None:
        broken = dict(frames)
        first = next(iter(broken))
        broken[first] = broken[first].drop(columns=["volume_raw"])
        with pytest.raises(MissingSourceError, match="missing column"):
            FramePriceSource(broken).load_panel(end=cycle_date, lookback_days=400)

    def test_a_stale_source_reads_differently_from_an_empty_one(self, frames, cycle_date) -> None:
        with pytest.raises(MissingSourceError, match="no rows fall inside the window"):
            FramePriceSource(frames).load_panel(
                end=cycle_date + dt.timedelta(days=900), lookback_days=5
            )

    def test_a_present_but_empty_per_ticker_frame_is_raised_not_dropped(
        self, frames, cycle_date
    ) -> None:
        """Defect #14 (2026-09-01 adversarial review): a partial outage was invisible.

        `normalize_panel` used to `continue` past any per-ticker frame that
        was `None` or empty, while raising loudly only when ZERO tickers
        survived. A dropped ticker among many is exactly a partial source
        outage — a well-formed, thin panel with nothing to say it happened.
        """
        import pandas as pd

        broken = dict(frames)
        first = next(iter(broken))
        broken[first] = pd.DataFrame(columns=broken[first].columns)
        with pytest.raises(MissingSourceError, match="are `None` or empty"):
            FramePriceSource(broken).load_panel(end=cycle_date, lookback_days=400)

    def test_a_partial_arctic_outage_that_omits_a_ticker_key_is_raised(
        self, frames, cycle_date, monkeypatch
    ) -> None:
        """The other shape of the same defect: `load_universe_ohlcv` "drops
        per-ticker read failures at WARNING and returns what it got" (its
        own docstring) — the failed ticker's key is entirely absent from the
        returned dict, not present with an empty value. `ArcticPriceSource`
        used to check nothing against `symbols` and hand the thinned dict
        straight to `normalize_panel`, which had no way to know a ticker was
        even requested.
        """
        from crucible.data.sources import ArcticPriceSource

        requested = sorted(frames)
        thinned = {t: f for t, f in frames.items() if t != requested[0]}

        def fake_load_universe_ohlcv(bucket, *, symbols, lookback_days, end, region=None):
            return thinned

        monkeypatch.setattr(
            "nousergon_lib.arcticdb.load_universe_ohlcv", fake_load_universe_ohlcv
        )
        source = ArcticPriceSource("test-bucket")
        with pytest.raises(MissingSourceError, match="dropped"):
            source.load_panel(end=cycle_date, lookback_days=400, symbols=requested)


class TestCoverage:
    def test_no_expected_set_fails_rather_than_reporting_a_flattering_ratio(
        self, store, source, cycle_date
    ) -> None:
        """Defect #2 (2026-09-01 adversarial review), restored and re-closed.

        `expected_symbols=None` used to write `status: "OK"` with
        `value: None` and skip `COVERAGE_FLOOR_RATIO` entirely — the ONLY
        path that runs in production, since nothing resolves a universe
        automatically and `--symbols` is hand-typed. A run declaring no
        universe must fail loudly, not report a coverage metric that looks
        benign, and it must fail BEFORE writing a panel, coverage record or
        feature layer at all.
        """
        with pytest.raises(UndeclaredUniverseError, match="no --symbols"):
            run_job(
                "data.daily",
                lambda c: run_daily(c, source=source),
                store=store,
                trading_day=cycle_date,
            )
        manifest = read_manifest(store, "data.daily", cycle_date.isoformat())
        assert manifest["status"] == "failed"
        assert not store.exists(data_panel_key(cycle_date.isoformat())), (
            "a run with no declared universe must leave no panel behind, exactly like "
            "any other refused day"
        )

    def test_an_explicit_empty_symbol_list_is_also_undeclared(
        self, store, source, cycle_date
    ) -> None:
        """`--symbols` resolving to an empty list is the same failure as omitting it."""
        with pytest.raises(UndeclaredUniverseError):
            run_job(
                "data.daily",
                lambda c: run_daily(c, source=source, expected_symbols=[]),
                store=store,
                trading_day=cycle_date,
            )

    def test_a_thin_day_fails_instead_of_degrading(self, store, frames, cycle_date) -> None:
        expected = sorted(frames) + [f"ABSENT{i}" for i in range(60)]
        thin = FramePriceSource(frames)
        with pytest.raises(MissingSourceError, match="absent from the source"):
            run_job(
                "data.daily",
                lambda c: run_daily(c, source=thin, expected_symbols=expected),
                store=store,
                trading_day=cycle_date,
            )

    def test_coverage_below_the_floor_raises(self, store, frames, cycle_date) -> None:
        """A ticker present in the source but with no close on the day."""
        holed = dict(frames)
        for name in sorted(holed)[:20]:
            block = holed[name]
            holed[name] = block[[d != cycle_date for d in block.index]]
        with pytest.raises(CoverageError, match="below the floor"):
            run_job(
                "data.daily",
                lambda c: run_daily(
                    c, source=FramePriceSource(holed), expected_symbols=sorted(holed)
                ),
                store=store,
                trading_day=cycle_date,
            )


class TestArtifacts:
    def test_a_successful_day_writes_panel_coverage_and_features(
        self, store, source, frames, cycle_date
    ) -> None:
        run_job(
            "data.daily",
            lambda c: run_daily(c, source=source, expected_symbols=sorted(frames)),
            store=store,
            trading_day=cycle_date,
        )
        from crucible.features import DEFAULT_FEATURE_VERSION

        assert store.exists(data_panel_key(cycle_date.isoformat()))
        assert store.exists(coverage_key(cycle_date.isoformat()))
        assert store.exists(features_key(DEFAULT_FEATURE_VERSION, cycle_date.isoformat()))
        coverage = json.loads(store.get_bytes(coverage_key(cycle_date.isoformat())))
        assert coverage["data_snapshot_id"] == "frames:conftest-seed-20260901", (
            "§9.7: the run that read a given version of the source must be identifiable"
        )

    def test_every_key_written_binds_to_a_trading_day(
        self, store, source, frames, cycle_date
    ) -> None:
        run_job(
            "data.daily",
            lambda c: run_daily(c, source=source, expected_symbols=sorted(frames)),
            store=store,
            trading_day=cycle_date,
        )
        store.assert_keys_bind_to_trading_days()


class TestWeekly:
    def test_the_week_is_enumerated_from_the_calendar(self, cycle_date) -> None:
        sessions = week_sessions(cycle_date)
        assert cycle_date in sessions
        assert all(s <= cycle_date for s in sessions)
        assert len(sessions) == 5

    def test_a_gap_fails_the_week_and_names_the_heal_command(
        self, store, source, frames, cycle_date
    ) -> None:
        with pytest.raises(DataGapError) as excinfo:
            run_job(
                "data.weekly",
                lambda c: run_weekly(c, source=source, expected_symbols=sorted(frames)),
                store=store,
                trading_day=cycle_date,
            )
        assert "crucible data.heal --from" in str(excinfo.value)

    def test_a_gap_fails_the_week_even_though_no_flag_exists_to_suppress_it(
        self, store, source, frames, cycle_date
    ) -> None:
        """Defect #3 (2026-09-01 adversarial review): `--allow-week-gap` is gone.

        There used to be a `require_full_week` parameter that let a gap
        write a `FAIL` `week_sessions_compiled` MetricRecord inside an `ok`
        manifest — the excluded third state, reintroduced at the only level
        a human actually reads (§2 row 4, §4.6). `run_weekly` now has no
        parameter that can suppress the raise; this test pins that the
        signature itself no longer accepts one.
        """
        import inspect

        assert "require_full_week" not in inspect.signature(run_weekly).parameters
        with pytest.raises(DataGapError):
            run_job(
                "data.weekly",
                lambda c: run_weekly(c, source=source, expected_symbols=sorted(frames)),
                store=store,
                trading_day=cycle_date,
            )
        manifest = read_manifest(store, "data.weekly", cycle_date.isoformat())
        assert manifest["status"] == "failed", (
            "a week with a gap must never produce an `ok` manifest, regardless of how "
            "the gap is described — the FAIL belongs to the run, not to a metric a "
            "human has to find inside a successful one"
        )

    def test_a_complete_week_passes(self, store, source, frames, cycle_date) -> None:
        for day in week_sessions(cycle_date)[:-1]:
            run_job(
                "data.daily",
                lambda c: run_daily(c, source=source, expected_symbols=sorted(frames)),
                store=store,
                trading_day=day,
            )
        ctx = run_job(
            "data.weekly",
            lambda c: run_weekly(c, source=source, expected_symbols=sorted(frames)),
            store=store,
            trading_day=cycle_date,
        )
        metric = next(m for m in ctx.metrics if m["name"] == "week_sessions_compiled")
        assert metric["value"] == metric["baseline"]


class TestHeal:
    def test_a_bulk_heal_refuses_off_ec2_and_prints_the_in_region_command(
        self, store, source, cycle_date, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "crucible.data.heal.in_region", lambda: (False, "IMDSv2 did not answer")
        )
        sessions = week_sessions(cycle_date)
        with pytest.raises(NotInRegionError) as excinfo:
            run_job(
                "data.heal",
                lambda c: run_heal(
                    c, source=source, start=sessions[0], end=sessions[-1], gap="test"
                ),
                store=store,
                trading_day=cycle_date,
            )
        assert "crucible data.heal --from" in str(excinfo.value)
        assert "--i-am-in-region" in str(excinfo.value)

    def test_the_explicit_override_is_the_only_one(
        self, store, source, cycle_date, monkeypatch
    ) -> None:
        monkeypatch.setattr("crucible.data.heal.in_region", lambda: (False, "laptop"))
        monkeypatch.setenv("CRUCIBLE_IN_REGION", "true")
        sessions = week_sessions(cycle_date)
        with pytest.raises(NotInRegionError):
            run_job(
                "data.heal",
                lambda c: run_heal(
                    c, source=source, start=sessions[0], end=sessions[-1], gap="test"
                ),
                store=store,
                trading_day=cycle_date,
            )

    def test_a_small_diagnostic_range_is_allowed_locally(
        self, store, source, frames, cycle_date, monkeypatch
    ) -> None:
        monkeypatch.setattr("crucible.data.heal.in_region", lambda: (False, "laptop"))
        sessions = week_sessions(cycle_date)[-LAPTOP_SESSION_ALLOWANCE:]
        ctx = run_job(
            "data.heal",
            lambda c: run_heal(
                c,
                source=source,
                start=sessions[0],
                end=sessions[-1],
                gap="diagnostic",
                expected_symbols=sorted(frames),
            ),
            store=store,
            trading_day=cycle_date,
        )
        for day in sessions:
            assert store.exists(data_panel_key(day.isoformat()))
        assert ctx.rows_out == len(sessions)

    def test_a_heal_is_idempotent_by_content(
        self, store, source, frames, cycle_date, monkeypatch
    ) -> None:
        monkeypatch.setattr("crucible.data.heal.in_region", lambda: (True, "EC2 instance i-test"))
        sessions = week_sessions(cycle_date)
        run_job(
            "data.heal",
            lambda c: run_heal(
                c,
                source=source,
                start=sessions[0],
                end=sessions[-1],
                gap="first",
                expected_symbols=sorted(frames),
            ),
            store=store,
            trading_day=cycle_date,
        )
        before = {
            day.isoformat(): store.get_bytes(data_panel_key(day.isoformat())) for day in sessions
        }
        run_job(
            "data.heal",
            lambda c: run_heal(
                c,
                source=source,
                start=sessions[0],
                end=sessions[-1],
                gap="second",
                expected_symbols=sorted(frames),
            ),
            store=store,
            trading_day=cycle_date,
        )
        after = {
            day.isoformat(): store.get_bytes(data_panel_key(day.isoformat())) for day in sessions
        }
        assert before == after, "a rerun producing different bytes is not idempotent"

    def test_a_non_session_bound_is_refused_rather_than_snapped(
        self, store, source, cycle_date
    ) -> None:
        from crucible.calendar import NonTradingDayKeyError
        from crucible.data.heal import sessions_in_range

        saturday = dt.date(2026, 8, 29)
        with pytest.raises(NonTradingDayKeyError):
            sessions_in_range(saturday, cycle_date)
