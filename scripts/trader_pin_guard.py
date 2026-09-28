"""Read whether this session's post-close pipeline finished cleanly, for
`.github/workflows/trader-pin.yml` (alpha-engine-config-I11545).

    uv run python scripts/trader_pin_guard.py

Brian's ruling: a queued trader pin is applied by "the next clean post-close".
This script answers the "clean post-close" half and nothing else; whether
anything is queued, and what to do about it, is `crucible release.pin_apply`'s,
which takes this reading as `--postclose`.

**The session** is the last CLOSED NYSE session, `crucible.calendar.
resolve_trading_day` — the same resolution every crucible job uses. The cron
fires at 23:15 UTC on weekdays and GitHub delivers it up to ~5 hours late
(`components.yaml`, `gate.close`), so a delivery after midnight New York time
still belongs to the previous evening: the NOMINAL day is the New York date of
the run, less one day before noon. When that nominal day was not a session (an
NYSE holiday), this prints `skip` and the workflow does nothing, green — the
job's absence deadline is trading-calendar relative, so a holiday owes no
manifest.

**Clean** means: at least one execution of the post-close state machine that
STARTED on the session's New York date SUCCEEDED, and none that started that
day is still RUNNING. A failed attempt followed by a successful redrive is
clean; a pipeline still running, or one that never started, is not.

Reads `POSTCLOSE_STATE_MACHINE_ARN` from the environment (the workflow builds
it; this public tree carries no account number). Calls exactly one AWS API,
`states:ListExecutions`, on exactly that state machine. Writes `session=` and
`postclose=` (`clean` / `unclean` / `skip`) to `$GITHUB_OUTPUT` when set.

Exit 0 for every READING, including `unclean`. A non-zero exit means the
reading could not be taken at all (unset variable, AWS refused), and the
workflow fails naming it — "could not ask" never reads as "not clean".
"""

from __future__ import annotations

import datetime as dt
import os
import sys
from dataclasses import dataclass
from typing import Any
from zoneinfo import ZoneInfo

from crucible.calendar import is_trading_day, resolve_trading_day

ET = ZoneInfo("America/New_York")

#: Before this New York wall time the run is a late delivery of the PREVIOUS
#: evening's cron. The cron is 23:15 UTC (19:15 EDT / 18:15 EST), and the
#: measured delivery tail is ~5 hours, so every legitimate run lands between
#: 18:15 and ~00:30 New York time; noon splits that band from nothing.
NOMINAL_DAY_ROLLOVER_ET = dt.time(12, 0)

STATE_MACHINE_ARN_VAR = "POSTCLOSE_STATE_MACHINE_ARN"


@dataclass(frozen=True)
class Reading:
    session: dt.date | None
    postclose: str  # "clean" | "unclean" | "skip"
    detail: str


def nominal_day(now: dt.datetime) -> dt.date:
    """The New York date whose evening cron this run is, late delivery included."""
    local = now.astimezone(ET)
    if local.time() < NOMINAL_DAY_ROLLOVER_ET:
        return local.date() - dt.timedelta(days=1)
    return local.date()


def _started_et_date(execution: dict[str, Any]) -> dt.date:
    started = execution["startDate"]
    if started.tzinfo is None:
        raise ValueError(f"execution {execution.get('name')!r} carries a naive startDate")
    return started.astimezone(ET).date()


def session_executions(pages: Any, session: dt.date) -> list[dict[str, Any]]:
    """The executions that started on ``session``'s New York date.

    ``pages`` is `ListExecutions` output, newest first (the API's order), so
    reading stops at the first page reaching back before the session.
    """
    found: list[dict[str, Any]] = []
    for page in pages:
        older = False
        for execution in page.get("executions", []):
            day = _started_et_date(execution)
            if day == session:
                found.append(execution)
            elif day < session:
                older = True
        if older:
            break
    return found


def read(now: dt.datetime, list_pages: Any) -> Reading:
    """The guard's reading at ``now``. ``list_pages`` yields `ListExecutions` pages."""
    if now.tzinfo is None:
        raise ValueError(f"{now!r} is naive; the guard reads New York wall time")
    nominal = nominal_day(now)
    session = resolve_trading_day(now)
    if not is_trading_day(nominal) or session != nominal:
        return Reading(
            None,
            "skip",
            f"{nominal.isoformat()} was not an NYSE session (last closed session "
            f"{session.isoformat()}): nothing is due tonight",
        )
    executions = session_executions(list_pages(), session)
    statuses = sorted({str(e.get("status")) for e in executions})
    succeeded = [e for e in executions if e.get("status") == "SUCCEEDED"]
    running = [e for e in executions if e.get("status") == "RUNNING"]
    if succeeded and not running:
        return Reading(
            session,
            "clean",
            f"post-close for {session.isoformat()} SUCCEEDED "
            f"({len(executions)} execution(s) that day: {', '.join(statuses)})",
        )
    if not executions:
        why = "no post-close execution started that day"
    elif running:
        why = f"{len(running)} post-close execution(s) still RUNNING"
    else:
        why = f"no post-close execution SUCCEEDED (statuses: {', '.join(statuses)})"
    return Reading(session, "unclean", f"no clean post-close for {session.isoformat()}: {why}")


def main(
    argv: list[str] | None = None, *, now: dt.datetime | None = None, client: Any = None
) -> int:
    if argv:
        print(f"usage: {sys.argv[0]} (takes no arguments; got {argv})", file=sys.stderr)
        return 2
    arn = os.environ.get(STATE_MACHINE_ARN_VAR, "").strip()
    if not arn:
        print(f"::error::{STATE_MACHINE_ARN_VAR} is unset; the guard cannot name the post-close")
        return 1
    if client is None:
        import boto3  # noqa: PLC0415 - only the live path needs the SDK

        client = boto3.client("stepfunctions")
    paginator = client.get_paginator("list_executions")
    reading = read(
        now or dt.datetime.now(dt.UTC),
        lambda: paginator.paginate(stateMachineArn=arn, PaginationConfig={"PageSize": 100}),
    )
    print(f"trader-pin guard: {reading.postclose}: {reading.detail}")
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(f"session={reading.session.isoformat() if reading.session else ''}\n")
            handle.write(f"postclose={reading.postclose}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
