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
                lambda c: run_daily(c, source=source),
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


class TestCoverage:
    def test_no_expected_set_means_no_ratio_rather_than_a_flattering_one(
        self, store, source, cycle_date
    ) -> None:
        ctx = run_job(
            "data.daily",
            lambda c: run_daily(c, source=source),
            store=store,
            trading_day=cycle_date,
        )
        metric = next(m for m in ctx.metrics if m["name"] == "universe_coverage_ratio")
        assert metric["value"] is None
        assert metric["unit"] is None
        assert "no denominator" in metric["status_reason"]

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
        self, store, source, cycle_date
    ) -> None:
        run_job(
            "data.daily",
            lambda c: run_daily(c, source=source),
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

    def test_every_key_written_binds_to_a_trading_day(self, store, source, cycle_date) -> None:
        run_job(
            "data.daily",
            lambda c: run_daily(c, source=source),
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
        self, store, source, cycle_date
    ) -> None:
        with pytest.raises(DataGapError) as excinfo:
            run_job(
                "data.weekly",
                lambda c: run_weekly(c, source=source),
                store=store,
                trading_day=cycle_date,
            )
        assert "crucible data.heal --from" in str(excinfo.value)

    def test_a_complete_week_passes(self, store, source, cycle_date) -> None:
        for day in week_sessions(cycle_date)[:-1]:
            run_job(
                "data.daily",
                lambda c: run_daily(c, source=source),
                store=store,
                trading_day=day,
            )
        ctx = run_job(
            "data.weekly",
            lambda c: run_weekly(c, source=source),
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
        self, store, source, cycle_date, monkeypatch
    ) -> None:
        monkeypatch.setattr("crucible.data.heal.in_region", lambda: (False, "laptop"))
        sessions = week_sessions(cycle_date)[-LAPTOP_SESSION_ALLOWANCE:]
        ctx = run_job(
            "data.heal",
            lambda c: run_heal(
                c, source=source, start=sessions[0], end=sessions[-1], gap="diagnostic"
            ),
            store=store,
            trading_day=cycle_date,
        )
        for day in sessions:
            assert store.exists(data_panel_key(day.isoformat()))
        assert ctx.rows_out == len(sessions)

    def test_a_heal_is_idempotent_by_content(self, store, source, cycle_date, monkeypatch) -> None:
        monkeypatch.setattr("crucible.data.heal.in_region", lambda: (True, "EC2 instance i-test"))
        sessions = week_sessions(cycle_date)
        run_job(
            "data.heal",
            lambda c: run_heal(c, source=source, start=sessions[0], end=sessions[-1], gap="first"),
            store=store,
            trading_day=cycle_date,
        )
        before = {
            day.isoformat(): store.get_bytes(data_panel_key(day.isoformat())) for day in sessions
        }
        run_job(
            "data.heal",
            lambda c: run_heal(c, source=source, start=sessions[0], end=sessions[-1], gap="second"),
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
