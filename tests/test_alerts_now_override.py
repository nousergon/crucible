"""`crucible alerts.sweep --now`: an operator override of the wall-clock
instant the sweep evaluates against.

Normative source: `alpha-engine-config-I10125`. `crucible.alerts.days_to_evaluate`
and `crucible.alerts.pages_in_window` are both anchored to wall-clock `now`
by construction, which makes them structurally coincident with
`crucible.gate._clause_pages_within_ceiling`'s own live grading window — a
fault could not be swept for real without landing inside a window already
being graded. `--now` lets an operator sweep a historical trading day instead,
without touching the live window, and the manifest that run writes must be
distinguishable from a natural one.

Every test here was seen failing before this module existed — there was no
`parse_now_override`, no `--now` flag, and no `now_override_utc` manifest
field.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json

import pytest

from crucible import alerts, track_c
from crucible.calendar import NonTradingDayKeyError, resolve_trading_day
from crucible.keys import manifest_key
from crucible.manifest import validate
from crucible.runner import run_job
from crucible.store import LocalStore

# A real NYSE session (Friday), well clear of a holiday — matches the fault-
# injection runbook's own safe historical day (I10125's "Measured" section).
HISTORICAL_TRADING_DAY = "2026-08-07"
HISTORICAL_SATURDAY = "2026-08-08"


class TestParseNowOverride:
    """`crucible.alerts.parse_now_override` — the pure parsing/validation."""

    def test_resolves_to_the_days_close_in_utc(self) -> None:
        # 21:00 ET on 2026-08-07 (EDT, UTC-4) is 2026-08-08T01:00:00Z.
        moment = alerts.parse_now_override(
            HISTORICAL_TRADING_DAY,
            actual_now=dt.datetime(2026, 9, 9, tzinfo=dt.UTC),
        )
        assert moment == dt.datetime(2026, 8, 8, 1, 0, tzinfo=dt.UTC)

    def test_rejects_a_malformed_date_rather_than_guessing(self) -> None:
        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            alerts.parse_now_override("not-a-date")

    def test_rejects_a_non_trading_day_rather_than_silently_resolving_it(self) -> None:
        """Rule 3 (crucible/AGENTS.md): an explicit non-trading day is
        refused, never quietly walked back to the prior session."""
        with pytest.raises(NonTradingDayKeyError):
            alerts.parse_now_override(
                HISTORICAL_SATURDAY, actual_now=dt.datetime(2026, 9, 9, tzinfo=dt.UTC)
            )

    def test_rejects_a_future_override_rather_than_clamping_it(self) -> None:
        # 2026-08-14 is a real trading day; the override is only "future"
        # relative to the injected actual_now, so this cannot depend on the
        # real wall clock at test-run time.
        with pytest.raises(ValueError, match="future"):
            alerts.parse_now_override(
                "2026-08-14", actual_now=dt.datetime(2026, 8, 10, tzinfo=dt.UTC)
            )

    def test_a_day_exactly_at_actual_now_is_not_future(self) -> None:
        """The boundary is inclusive: the override's own close is not
        strictly after `actual_now` when `actual_now` IS that close."""
        moment = alerts.parse_now_override(
            HISTORICAL_TRADING_DAY,
            actual_now=dt.datetime(2026, 8, 8, 1, 0, tzinfo=dt.UTC),
        )
        assert moment == dt.datetime(2026, 8, 8, 1, 0, tzinfo=dt.UTC)


class TestSweepHonoursTheOverride:
    """`alerts.sweep(now=...)` actually evaluates the overridden instant,
    not wall-clock now — the whole point of the flag."""

    def test_days_to_evaluate_anchors_to_the_override_not_the_real_clock(self, store) -> None:
        real_now = dt.datetime(2026, 9, 9, 12, 0, tzinfo=dt.UTC)
        override = alerts.parse_now_override(HISTORICAL_TRADING_DAY, actual_now=real_now)

        overridden_days = alerts.days_to_evaluate(store, override)
        real_days = alerts.days_to_evaluate(store, real_now)

        assert resolve_trading_day(override) in overridden_days
        assert resolve_trading_day(override) not in real_days
        assert overridden_days != real_days

    def test_pages_in_window_counts_the_overridden_windows_bus_rows_only(self, store) -> None:
        """A bus row keyed near the historical override must not be counted
        by a real-wall-clock ceiling read, and vice versa — the two windows
        stay decoupled, which is the whole fix."""
        real_now = dt.datetime(2026, 9, 9, 12, 0, tzinfo=dt.UTC)
        override = alerts.parse_now_override(HISTORICAL_TRADING_DAY, actual_now=real_now)
        historical_day = resolve_trading_day(override)

        group = alerts.PageGroup(
            cause_key=f"absence:{historical_day.isoformat()}",
            members=(
                alerts.Page(
                    condition="absence",
                    job="data.daily",
                    trading_day=historical_day,
                    reason="test fixture",
                ),
            ),
        )
        store.put_bytes(
            alerts.bus_key(group),
            json.dumps(
                alerts.bus_row(
                    group,
                    alert_id=None,
                    sent=True,
                    destination="test",
                    first_observed_utc="2026-08-08T01:00:00Z",
                    last_observed_utc="2026-08-08T01:00:00Z",
                )
            ).encode("utf-8"),
        )

        assert alerts.pages_in_window(store, now=override) >= 1
        assert alerts.pages_in_window(store, now=real_now) == 0


class TestRunJobRecordsTheOverrideHonestly:
    """`crucible.runner.run_job(now_override=...)` — the manifest-provenance
    half. A run under an overridden `now` must be distinguishable from a
    natural one in its own manifest."""

    def test_an_overridden_run_carries_now_override_utc(self, tmp_path) -> None:
        local = LocalStore(tmp_path)
        override = dt.datetime(2026, 8, 8, 1, 0, tzinfo=dt.UTC)

        run_job(
            "alerts.sweep",
            lambda ctx: None,
            store=local,
            trading_day=dt.date(2026, 8, 28),
            now_override=override,
        )

        doc = json.loads(local.get_bytes(manifest_key("alerts.sweep", "2026-08-28")))
        validate(doc)
        assert doc["now_override_utc"] == "2026-08-08T01:00:00Z"

    def test_a_natural_run_carries_no_now_override_field_at_all(self, tmp_path) -> None:
        """Never written as null — omitted entirely, so a natural run's
        manifest is unaffected by this field existing at all."""
        local = LocalStore(tmp_path)

        run_job(
            "alerts.sweep",
            lambda ctx: None,
            store=local,
            trading_day=dt.date(2026, 8, 28),
        )

        doc = json.loads(local.get_bytes(manifest_key("alerts.sweep", "2026-08-28")))
        validate(doc)
        assert "now_override_utc" not in doc

    def test_started_and_finished_stay_the_real_wall_clock_under_an_override(
        self, tmp_path
    ) -> None:
        """`now_override` changes what the job's OWN logic reasons from, not
        when the process actually ran — conflating the two would make the
        manifest lie about when this run really happened."""
        local = LocalStore(tmp_path)
        real_run_time = dt.datetime(2026, 8, 28, 21, 0, tzinfo=dt.UTC)
        override = dt.datetime(2026, 8, 8, 1, 0, tzinfo=dt.UTC)

        run_job(
            "alerts.sweep",
            lambda ctx: None,
            store=local,
            trading_day=dt.date(2026, 8, 28),
            now=real_run_time,
            now_override=override,
        )

        doc = json.loads(local.get_bytes(manifest_key("alerts.sweep", "2026-08-28")))
        assert doc["started"] == "2026-08-28T21:00:00Z"
        assert doc["now_override_utc"] == "2026-08-08T01:00:00Z"


class TestCLIWiring:
    """`--now` exists on `alerts.sweep` and nowhere else, and the scheduled
    path — which never passes it — is provably unaffected by its existence."""

    def test_now_flag_is_declared_on_alerts_sweep(self) -> None:
        from crucible.cli import build_parser

        parser = build_parser()
        sweep_parser = parser._subparsers._group_actions[0].choices["alerts.sweep"]
        assert any(a.dest == "now" for a in sweep_parser._actions)

    def test_now_flag_is_not_declared_on_other_jobs(self) -> None:
        from crucible.cli import build_parser

        parser = build_parser()
        for name in ("smoke", "heartbeat", "board", "report", "weekly"):
            sub = parser._subparsers._group_actions[0].choices[name]
            assert not any(a.dest == "now" for a in sub._actions), name

    def test_a_bad_override_is_refused_before_any_manifest_write(self, tmp_path) -> None:
        args = argparse.Namespace(
            store=str(tmp_path),
            dry_run=False,
            run_mode="live",
            trading_day=dt.date(2026, 9, 4),
            now="not-a-date",
        )
        with pytest.raises(SystemExit):
            track_c.sweep_handler(args)
        assert list(tmp_path.rglob("run.json")) == []

    def test_a_future_override_is_refused_before_any_manifest_write(self, tmp_path) -> None:
        args = argparse.Namespace(
            store=str(tmp_path),
            dry_run=False,
            run_mode="live",
            trading_day=dt.date(2026, 9, 4),
            now="2099-01-06",
        )
        with pytest.raises(SystemExit):
            track_c.sweep_handler(args)
        assert list(tmp_path.rglob("run.json")) == []

    def test_the_scheduled_path_passes_no_override_and_behaves_exactly_as_before(
        self, tmp_path, monkeypatch
    ) -> None:
        """The scheduled/unattended sweep never sets `--now`; asserting this
        keeps `alerts.sweep(now=None)` — real wall clock — the path an
        un-overridden invocation takes, exactly as before this flag existed.
        """
        captured: dict[str, object] = {}
        real_sweep = alerts.sweep

        def spy(store, **kwargs):
            captured.update(kwargs)
            return real_sweep(store, **kwargs)

        monkeypatch.setattr(track_c.alerts, "sweep", spy)

        args = argparse.Namespace(
            store=str(tmp_path),
            dry_run=True,
            run_mode="live",
            trading_day=dt.date(2026, 9, 4),
            now=None,
        )
        track_c.sweep_handler(args)

        assert captured["now"] is None
        doc = json.loads(
            (tmp_path / "runs" / "alerts.sweep" / "2026-09-04" / "run.json").read_text()
            if (tmp_path / "runs" / "alerts.sweep" / "2026-09-04" / "run.json").exists()
            else "{}"
        )
        # `--dry-run` writes no manifest at all (alpha-engine-config-I9922);
        # the point of this test is `captured["now"] is None` above — a
        # scheduled sweep is unaffected by `--now`'s mere existence.
        assert doc == {}
