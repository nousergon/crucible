"""Every NYSE session gets an acceptance reading, whether or not anyone merged.

Normative source: `alpha-engine-config-I12483`. The plan §12 rule-3 reading is
filed at `crucible.keys.acceptance_reading_key(day)` for
`day = crucible.calendar.resolve_trading_day(started)` — the last session whose
close had passed when the job ran — and `crucible report.morning` reads exactly
that day's key, with no fallback. `ci.yml`'s acceptance job ran on
`push: [main]` only, so a session got a reading only if some merge happened to
land between its close and the next session's close. 2026-09-29, 2026-10-12,
2026-10-15 and 2026-10-19 had none, and the morning report's history rows for
them read `acceptance_met: null`.

The pieces asserted here are the ones that close that gap, held against the
trading calendar rather than against a restated time:

* `ci.yml` carries a weekday cron whose every firing binds to THAT session,
  in both EST and EDT, and still binds to it after GitHub's multi-hour
  scheduled-event lag;
* the acceptance job runs on that arm, and the tree-only `test` job does not;
* the registry row is scheduled, names `ci.yml` as its starter, and declares a
  deadline after the cron, so `alerts.sweep` pages a session whose reading
  never arrived instead of the morning report rendering a null.
"""

from __future__ import annotations

import datetime as dt
import pathlib
import re

import pytest
import yaml

from crucible.acceptance_publish import ACCEPTANCE_PUBLISH_JOB
from crucible.calendar import is_trading_day, resolve_trading_day
from crucible.cli import JOBS
from crucible.components import load_registry

REPO = pathlib.Path(__file__).resolve().parents[1]
CI_WORKFLOW = REPO / ".github" / "workflows" / "ci.yml"

#: The four sessions `alpha-engine-config-I12483` found with no reading.
MISSED_SESSIONS = (
    dt.date(2026, 9, 29),
    dt.date(2026, 10, 12),
    dt.date(2026, 10, 15),
    dt.date(2026, 10, 19),
)

#: GitHub's scheduled-event delivery lag, as measured on this fleet
#: (`morning-report.yml`'s header: 3-5h, chronic). Doubled-ish for margin.
SCHEDULE_LAG = dt.timedelta(hours=8)

#: A span crossing both DST transitions (2026-11-01, 2027-03-14), a weekday
#: holiday (Thanksgiving) and an early close (the day after it).
SPAN = (dt.date(2026, 9, 1), dt.date(2027, 6, 30))

_SIMPLE_CRON = re.compile(r"^(\d{1,2}) (\d{1,2}) \* \* (\d)-(\d)$")


def _workflow() -> dict:
    return yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))


def _triggers(workflow: dict) -> dict:
    # YAML 1.1 reads a bare `on` as the boolean True.
    return workflow.get(True, workflow.get("on")) or {}


def _crons() -> list[str]:
    schedule = _triggers(_workflow()).get("schedule") or []
    return [entry["cron"] for entry in schedule]


def _firings(cron: str, start: dt.date, end: dt.date) -> list[dt.datetime]:
    """Every UTC instant ``cron`` fires in ``[start, end]``.

    Only the `M H * * a-b` shape is understood, and any other shape FAILS
    rather than being approximated: a parser that guessed would make every
    assertion below a statement about the guess.
    """
    match = _SIMPLE_CRON.match(cron)
    assert match, (
        f"ci.yml's cron {cron!r} is not the `M H * * a-b` shape this test can "
        "evaluate. Extend the parser alongside the cron; never loosen it."
    )
    minute, hour, first, last = (int(group) for group in match.groups())
    # cron day-of-week: 0 = Sunday. Python weekday(): 0 = Monday.
    allowed = {(dow - 1) % 7 for dow in range(first, last + 1)}
    out = []
    day = start
    while day <= end:
        if day.weekday() in allowed:
            out.append(dt.datetime.combine(day, dt.time(hour, minute), tzinfo=dt.UTC))
        day += dt.timedelta(days=1)
    return out


def _sessions(start: dt.date, end: dt.date) -> list[dt.date]:
    out = []
    day = start
    while day <= end:
        if is_trading_day(day):
            out.append(day)
        day += dt.timedelta(days=1)
    return out


@pytest.fixture(scope="module")
def cron() -> str:
    crons = _crons()
    assert len(crons) == 1, (
        f"ci.yml declares {crons!r}; expected exactly one post-close cron for the "
        "acceptance reading. With none, a session with no post-close merge has "
        "no reading and the morning report reads null."
    )
    return crons[0]


class TestEverySessionIsFiled:
    def test_every_session_in_the_span_is_bound_by_some_firing(self, cron: str) -> None:
        """The property the issue's closes-when names, over a whole span and
        not only the four days that were missed."""
        start, end = SPAN
        bound = {resolve_trading_day(at) for at in _firings(cron, start, end)}
        unfiled = [day for day in _sessions(start, end) if day not in bound]
        assert not unfiled, (
            f"sessions no firing of {cron!r} binds to: {unfiled}. Each of them would "
            "have no acceptance reading unless a merge landed after its close."
        )

    def test_each_session_is_bound_by_its_own_days_firing(self, cron: str) -> None:
        """Not merely covered by SOME firing: the firing on day D binds to D.
        A cron before the close in either DST regime would bind to the
        previous session, file that key twice and leave D's to the next day."""
        start, end = SPAN
        wrong = [
            (at.isoformat(), resolve_trading_day(at).isoformat())
            for at in _firings(cron, start, end)
            if is_trading_day(at.date()) and resolve_trading_day(at) != at.date()
        ]
        assert not wrong, f"firings that bind to another session: {wrong[:5]}"

    def test_a_late_delivery_still_binds_to_the_same_session(self, cron: str) -> None:
        """GitHub delivers scheduled events hours late. Any delivery before the
        next session's close must still file the same day's key, or a slow
        queue silently moves the reading onto the wrong day."""
        start, end = SPAN
        moved = [
            at.isoformat()
            for at in _firings(cron, start, end)
            if is_trading_day(at.date())
            and resolve_trading_day(at + SCHEDULE_LAG) != resolve_trading_day(at)
        ]
        assert not moved, f"firings an {SCHEDULE_LAG} lag moves to another day: {moved[:5]}"

    @pytest.mark.parametrize("session", MISSED_SESSIONS, ids=lambda d: d.isoformat())
    def test_the_sessions_that_were_missed_are_filed(self, cron: str, session: dt.date) -> None:
        """The four days `alpha-engine-config-I12483` measured, by name."""
        firings = _firings(cron, session, session)
        assert firings, f"{cron!r} does not fire on {session}"
        assert resolve_trading_day(firings[0]) == session


class TestTheWorkflowRunsTheRightJobs:
    def test_the_acceptance_job_runs_on_the_scheduled_arm(self) -> None:
        job = _workflow()["jobs"]["acceptance"]
        assert job.get("if") == "github.event_name != 'pull_request'", (
            "the acceptance job's `if:` changed; re-check that it still runs on "
            "`schedule` — the arm that files a session with no merge."
        )

    def test_the_tree_only_test_job_does_not_run_on_the_scheduled_arm(self) -> None:
        """An unchanged tree regraded every weekday is metered minutes buying
        nothing (plan §4.11); only the acceptance reading is owed per day."""
        condition = _workflow()["jobs"]["test"].get("if", "")
        assert condition == "github.event_name == 'push' || github.event_name == 'pull_request'"

    def test_a_manual_catch_up_exists(self) -> None:
        """A dropped scheduled event is recoverable by hand the same day."""
        assert "workflow_dispatch" in _triggers(_workflow())


class TestTheRegistryDeclaresItDue:
    def test_the_row_is_scheduled_by_ci_yml(self) -> None:
        row = load_registry()[ACCEPTANCE_PUBLISH_JOB]
        assert row.schedule is not None
        assert row.dispatch == "github-actions"
        assert row.dispatch_workflow == CI_WORKFLOW.name
        assert JOBS[ACCEPTANCE_PUBLISH_JOB].scheduled is True

    def test_the_deadline_is_daily_and_after_every_firing(self, cron: str) -> None:
        """Before the cron, the sweep would page a reading that is not yet due;
        weekly, it would never notice a missed weekday."""
        deadline = load_registry()[ACCEPTANCE_PUBLISH_JOB].deadline
        assert deadline is not None
        assert deadline.anchor == "close_plus"
        assert deadline.cadence == "daily"
        start, end = SPAN
        early = [
            at.isoformat()
            for at in _firings(cron, start, end)
            if is_trading_day(at.date()) and deadline.due_at(at.date()) <= at
        ]
        assert not early, f"deadline falls before the cron on: {early[:5]}"
