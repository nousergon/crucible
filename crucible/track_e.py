"""Track E's handler: `crucible report`.

Its own module rather than a branch inside another track's, so three tracks
land code in the same release without editing one another's lines. `cli.py`
carries one line naming it.

The job does two things, and the second is the reason the first can be
trusted: it reduces the week into `report/{trading_day}/attribution.json`, and
it publishes the run's LLM spend against the declared cap. A report card that
did not say what it cost to produce would be one more number with no
denominator (§2 row 3).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json

from crucible.config import settings as load_settings
from crucible.llm import SpendCap, cap_metric, spend_pace, week_to_date_llm_spend
from crucible.report import (
    ATTRIBUTION_SCHEMA_VERSION,
    attribution_key,
    build_attribution,
    rows_complete_metric,
)
from crucible.runner import RunContext, run_job
from crucible.store import open_store

__all__ = ["report_handler"]


def report_handler(args: argparse.Namespace) -> int:
    """`crucible report [--date YYYY-MM-DD]`.

    Runs through `run_job` like every other job, so a reduction that raises —
    a champion carrying verdicts at two horizons, a corrupt manifest — leaves
    a `failed` manifest naming the cause instead of an absent report card
    whose absence looks like a scheduler that never fired.
    """
    store = open_store(getattr(args, "store", None))
    config = load_settings()

    def body(ctx: RunContext) -> None:
        now = dt.datetime.now(dt.UTC)
        document, sources = build_attribution(
            store, trading_day=ctx.trading_day, now=now, run_id=ctx.run_id
        )
        for key in sources:
            ctx.record_input(key, store.get_bytes(key))
        payload = json.dumps(document, indent=2, sort_keys=True).encode("utf-8")
        key = attribution_key(ctx.trading_day.isoformat())
        ctx.record_output(key, payload, schema_version=ATTRIBUTION_SCHEMA_VERSION)
        ctx.record_rows(rows_in=len(sources), rows_out=len(document["rows"]))

        # §9.2 class 5: every row of the table is a manifest metric too, so the
        # report card's numbers reach the telemetry surface by the same route
        # as everything else rather than only through their own artifact.
        for row in document["rows"]:
            ctx.record_metric(row)
        ctx.record_metric(rows_complete_metric(document, now=now, source_path=key))

        # §2 row 3: the cap is published on every run, spend or none. It is
        # read off the week's manifests rather than off this process, because
        # the cap bounds the weekly RUN and a weekly run is several jobs.
        window_start = dt.date.fromisoformat(document["window_sessions"][0])
        cap = SpendCap(
            cap_usd=config.llm_cap_usd,
            spent_usd=week_to_date_llm_spend(
                store, window_start=window_start, trading_day=ctx.trading_day
            ),
        )
        ctx.record_metric(
            cap_metric(cap, now=now, source_path=f"runs/*/{ctx.trading_day}/run.json")
        )
        pace = spend_pace(
            cap.spent_usd,
            cap_usd=cap.cap_usd,
            now=now,
            anchor=dt.datetime.combine(window_start, dt.time.min, tzinfo=dt.UTC),
        )
        ctx.record_metric(
            {
                "name": "llm_spend_pace_overrun_ratio",
                "module": "crucible.llm",
                "metric_type": "ratio",
                "value": round(pace.overrun, 6),
                "unit": "ratio",
                "n_floor": 1,
                "status": "WATCH" if pace.exceeded else "OK",
                "status_reason": (
                    f"{pace.used_frac:.1%} of the ${cap.cap_usd:.2f} weekly LLM cap spent "
                    f"against {pace.elapsed_frac:.1%} of the window elapsed; a positive "
                    "overrun is ahead of a straight-line pace, which is visible here days "
                    "before a fixed threshold on the cap itself would fire"
                ),
                "source_path": f"runs/*/{ctx.trading_day}/run.json",
                "last_updated_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "baseline": 0.0,
            }
        )

    run_job("report", body, store=store, trading_day=args.trading_day)
    return 0
