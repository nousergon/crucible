"""`crucible data.weekly` — the week's refresh and coverage pass.

Normative source: plan §4.3, §9.7, §4.12.

The weekly job compiles the last completed trading day of the week and then
does the one thing the daily job cannot: it looks BACK across the week and
**refuses to run on a gap it did not heal**.

That refusal is §9.7's row, made mechanical. A weekly pass that quietly
computed over four of five sessions produces a verdict with an undeclared
denominator — which is the shape of every "the system stopped thinking while
every detector stayed green" incident in the register. So:

* the week's sessions are enumerated from the trading calendar, not from
  what happens to be in the store — a holiday is not a gap, and a missing
  Tuesday is not a holiday;
* a session with no `data/{day}/panel.parquet` is a GAP, named individually;
* a gap FAILS the run with `crucible data.heal --from --to` in the reason,
  because the operator's next action should be in the failure, not in a
  runbook they have to find.

**There is no flag that lets a gap pass.** An earlier `--allow-week-gap`
recorded the gap as a `FAIL` MetricRecord while still writing an `ok`
manifest — the excluded third state (§2 row 4: "no skip flags, no
fail-open, no degraded-SUCCEEDED") reintroduced at the only level a human
actually reads, since the console pages on absence or `status: failed`
(§4.6), never on a metric buried inside an `ok` run (defect #3, 2026-09-01
adversarial review). `run_weekly` now always raises on a gap; there is no
parameter that suppresses it.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Any

from crucible.calendar import assert_trading_day, is_trading_day
from crucible.data.daily import DEFAULT_LOOKBACK_DAYS, run_daily
from crucible.data.sources import PriceSource
from crucible.keys import data_panel_key

if TYPE_CHECKING:
    from crucible.runner import RunContext

__all__ = ["DataGapError", "run_weekly", "week_sessions"]

#: A trading week is five sessions (§4.12). A holiday week is still one week;
#: it simply has four sessions in it, and four of four is complete.
TRADING_WEEK_CALENDAR_SPAN_DAYS = 6


class DataGapError(RuntimeError):
    """The week contains a session with no compiled panel. The run fails."""


def week_sessions(end: dt.date) -> list[dt.date]:
    """Every NYSE session in the trading week ending at ``end``, inclusive.

    Enumerated from the calendar rather than from the store. A set built from
    what the store happens to hold cannot report a gap, because the gap is
    exactly what is not in it.
    """
    assert_trading_day(end, context=f"data.weekly week ending {end}")
    start = end - dt.timedelta(days=TRADING_WEEK_CALENDAR_SPAN_DAYS)
    day = start
    sessions: list[dt.date] = []
    while day <= end:
        if is_trading_day(day):
            sessions.append(day)
        day += dt.timedelta(days=1)
    return sessions


def run_weekly(
    ctx: RunContext,
    *,
    source: PriceSource,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    expected_symbols: list[str] | None = None,
) -> dict[str, Any]:
    """Compile the week's last session, then assert the week has no gap.

    The order matters: the day is compiled FIRST, so a run whose own session
    is the missing one fails on the source rather than on its own absence,
    which names the actual cause.

    No ``feature_version`` parameter, for the same reason `run_daily` has
    none (`alpha-engine-config-I9816`): the version is derived inside
    `run_daily` from the catalogue, and there is nothing here for a caller
    to override.
    """
    trading_day = ctx.trading_day
    coverage = run_daily(
        ctx,
        source=source,
        lookback_days=lookback_days,
        expected_symbols=expected_symbols,
    )

    sessions = week_sessions(trading_day)
    gaps = [
        day
        for day in sessions
        if day != trading_day and not ctx.store.exists(data_panel_key(day.isoformat()))
    ]

    ctx.record_metric(
        {
            "name": "week_sessions_compiled",
            "module": "crucible.data.weekly",
            "metric_type": "coverage",
            "value": float(len(sessions) - len(gaps)),
            "unit": "sessions",
            "n_floor": 1,
            "status": "OK" if not gaps else "FAIL",
            "status_reason": (
                f"{len(sessions) - len(gaps)} of {len(sessions)} sessions in the trading "
                f"week ending {trading_day} have a compiled panel"
                + (f"; gaps: {[d.isoformat() for d in gaps]}" if gaps else "")
            ),
            "source_path": f"data/*/panel.parquet (week ending {trading_day})",
            "last_updated_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "baseline": float(len(sessions)),
        }
    )

    if gaps:
        first, last = gaps[0].isoformat(), gaps[-1].isoformat()
        raise DataGapError(
            f"the trading week ending {trading_day} has {len(gaps)} session(s) with no "
            f"compiled panel: {[d.isoformat() for d in gaps]}. The weekly pass does not "
            "run on a gap it did not heal — a week computed over four of five sessions "
            "carries an undeclared denominator into every verdict downstream. Heal it "
            f"in region first:\n"
            f"    crucible data.heal --from {first} --to {last} --gap missing-panel"
        )

    coverage["week_sessions"] = [d.isoformat() for d in sessions]
    coverage["week_gaps"] = [d.isoformat() for d in gaps]
    return coverage
