"""`data.daily`'s holiday guard (alpha-engine-config-I9781).

The EventBridge schedule fires every weekday close, and a market holiday is
still a weekday: without a guard, `data.daily` resolves `trading_day` back to
the last real session (legitimate — rule 3) and reruns the whole compile for
it, overwriting that session's already-good manifest with a second, unrelated
writer's output. `crucible.track_a.handle_data_daily` refuses to run at all
when the schedule fires on a day that is not itself an NYSE session and no
explicit `--date` was given.
"""

from __future__ import annotations

import argparse
import datetime as dt

import pytest

import crucible.track_a as track_a
from crucible.calendar import resolve_trading_day
from crucible.manifest import manifest_key
from crucible.store import LocalStore

#: 2026-09-07 is Labor Day: a Monday, so `is_trading_day` refuses it, and the
#: last session before it is Friday 2026-09-04.
LABOR_DAY = dt.date(2026, 9, 7)
PRIOR_FRIDAY = dt.date(2026, 9, 4)


def _args(**over) -> argparse.Namespace:
    base = dict(
        job="data.daily",
        date=None,
        store=None,
        symbols=None,
        feature_version=None,
        dry_run=False,
    )
    base.update(over)
    ns = argparse.Namespace(**base)
    ns.trading_day = resolve_trading_day(dt.datetime.combine(LABOR_DAY, dt.time(21, 0)))
    return ns


class TestHolidayGuard:
    def test_a_holiday_firing_with_no_explicit_date_is_a_clean_no_op(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        monkeypatch.setattr(track_a, "_today", lambda: LABOR_DAY)
        store = LocalStore(tmp_path)
        # A good manifest already sits under the resolved trading day, as if
        # Friday's real run had written it.
        store.put_bytes(
            manifest_key("data.daily", PRIOR_FRIDAY.isoformat()),
            b'{"status": "ok", "sentinel": "friday-was-here"}',
        )

        args = _args(store=str(tmp_path))
        code = track_a.handle_data_daily(args)

        assert code == 0
        assert (
            store.get_bytes(manifest_key("data.daily", PRIOR_FRIDAY.isoformat()))
            == b'{"status": "ok", "sentinel": "friday-was-here"}'
        )
        assert "not an NYSE trading day" in capsys.readouterr().out

    def test_an_explicit_date_on_a_holiday_is_still_honoured(self, monkeypatch) -> None:
        """Rule 3: an explicit `--date` is a deliberate backfill/replay and
        is never silently skipped, holiday or not — only the auto-scheduled
        firing is guarded."""
        monkeypatch.setattr(track_a, "_today", lambda: LABOR_DAY)
        called = {}

        def fake_settings(args):
            called["ran"] = True
            raise SystemExit("stopped before touching a real store")

        monkeypatch.setattr(track_a, "_settings", fake_settings)
        args = _args(date=LABOR_DAY.isoformat())
        with pytest.raises(SystemExit):
            track_a.handle_data_daily(args)
        assert called.get("ran") is True

    def test_a_trading_day_firing_is_unaffected(self, monkeypatch) -> None:
        friday = dt.date(2026, 9, 4)
        monkeypatch.setattr(track_a, "_today", lambda: friday)
        called = {}

        def fake_settings(args):
            called["ran"] = True
            raise SystemExit("stopped before touching a real store")

        monkeypatch.setattr(track_a, "_settings", fake_settings)
        args = _args()
        args.trading_day = friday
        with pytest.raises(SystemExit):
            track_a.handle_data_daily(args)
        assert called.get("ran") is True
