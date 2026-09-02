"""`data.daily`'s holiday guard (alpha-engine-config-I9781).

The EventBridge schedule fires every weekday close, and a market holiday is
still a weekday: without a guard, `data.daily` resolves `trading_day` back to
the last real session (legitimate — rule 3) and reruns the whole compile for
it, overwriting that session's already-good manifest with a second, unrelated
writer's output. `crucible.track_a.handle_data_daily` never re-compiles on a
day that is not itself an NYSE session with no explicit `--date` given — but
per rule 2 ("a job that had nothing to do produced a complete correct
result — that is `ok`"), it still runs the job and writes a real `ok`
manifest, discriminated by the wall-clock firing date so it cannot collide
with the prior session's real manifest. A silent `return 0` with no manifest
at all was the shape rules 1 and 2 forbid: it left a holiday no-op and a
`data.daily` that had stopped working indistinguishable.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json

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
    def test_a_holiday_firing_with_no_explicit_date_does_not_touch_the_real_manifest(
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

    def test_a_holiday_firing_still_writes_a_real_ok_manifest_of_its_own(
        self, tmp_path, monkeypatch
    ) -> None:
        """Rules 1 and 2: a holiday no-op is a run that happened, not a
        vanished third status wearing exit 0 — it goes through `run_job` and
        writes its own `ok` manifest, discriminated so it cannot collide
        with the real session's manifest at the same `trading_day`."""
        monkeypatch.setattr(track_a, "_today", lambda: LABOR_DAY)
        store = LocalStore(tmp_path)
        args = _args(store=str(tmp_path))

        code = track_a.handle_data_daily(args)

        assert code == 0
        holiday_key = manifest_key(
            "data.daily", PRIOR_FRIDAY.isoformat(), discriminator=LABOR_DAY.isoformat()
        )
        assert store.exists(holiday_key)
        manifest = json.loads(store.get_bytes(holiday_key))
        assert manifest["status"] == "ok"
        assert manifest["reason"] == ""
        assert manifest["discriminator"] == LABOR_DAY.isoformat()
        # And the bare (undiscriminated) key is still untouched.
        assert not store.exists(manifest_key("data.daily", PRIOR_FRIDAY.isoformat()))

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
