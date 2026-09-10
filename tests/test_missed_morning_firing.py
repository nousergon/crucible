"""A morning report that never fired is DETECTED — alpha-engine-config-I9960.

`report.morning` is the accountability surface Brian relies on. It is started
by a GitHub `schedule:` cron, which is best-effort by documented design: the
one scheduled firing this repository has on record (`board.yml`, due
2026-09-03T21:30Z) ran 113 minutes late, and `morning-report.yml`'s first due
occurrence, 2026-09-04T13:00Z, produced no run at all.

The defect this file pins is not the cron. It is that **nothing noticed**.
`alerts.sweep` declares `report.morning` under its ABSENCE condition and read
as covered, while the day that row came due was the one day
:func:`crucible.alerts.days_to_evaluate` structurally excluded — because the
sweep had RUN on it, hours before the deadline arrived. Every row anchored
`next_calendar_day_at` (`report.morning`, `data.weekly`) was therefore
absence-checked by no sweep, ever.

Measured 2026-09-04 against the live store:
``runs/report.morning/2026-09-03/`` was empty and
``runs/alerts.sweep/2026-09-03/2026-09-04/run.json`` existed.

The cron itself was `0 13 * * *` (13:00 UTC) at measurement time and was
recalibrated to `0 10 * * *` (10:00 UTC) on 2026-09-06
(`alpha-engine-config-I9966`) — GitHub was delivering it 3-5h after its
declared instant, so `06:00 PT` cron text was landing near 09:07 PT.
:data:`OCCURRENCE` below tracks the CURRENT cron
(`crucible.morning.DELIVERY_CRON_UTC`), not the literal in effect when this
incident was measured.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from crucible.alerts import StoreAccessError, days_to_evaluate, evaluate_absence, sweep
from crucible.components import load_registry
from crucible.keys import TRIGGER_UNKNOWN, morning_trigger_key
from crucible.manifest import manifest_key
from crucible.morning import DELIVERY_CRON_UTC_HOUR, resolve_trigger
from crucible.store import LocalStore

#: Fixed literals, never `today` arithmetic (AGENTS.md test discipline). Both
#: are NYSE sessions; the pair is the live 2026-09-03/2026-09-04 case.
THURSDAY = dt.date(2026, 9, 3)
FRIDAY = dt.date(2026, 9, 4)

#: Friday 21:00 ET — the `alerts.sweep` schedule's own slot, one calendar day
#: after Thursday's report was due to be delivered.
FRIDAY_SWEEP = dt.datetime(2026, 9, 5, 1, 0, tzinfo=dt.UTC)

#: The occurrence `morning-report.yml`'s cron
#: (`crucible.morning.DELIVERY_CRON_UTC`) names for THURSDAY's report: its
#: declared UTC hour on the following calendar day. Recalibrated
#: 2026-09-06 (alpha-engine-config-I9966): the cron moved from 13:00 UTC to
#: 10:00 UTC, three hours earlier, so it delivers near 06:00 PT instead of
#: three hours after it.
OCCURRENCE = dt.datetime(2026, 9, 4, DELIVERY_CRON_UTC_HOUR, 0, tzinfo=dt.UTC)

#: The floor I9960/I9966 set for the deadline, from the worst of four
#: measured `nous-ergon-ops` GitHub Actions cron delays on 2026-09-04
#: (authority-surface, +304 min) — not from the single 113-minute
#: `board.yml` sample, which was this account's best case, not its
#: envelope. See `components.yaml`'s `report.morning.deadline` comment for
#: the full four-workflow table.
LATENCY_FLOOR_MINUTES = 304

MORNING = "report.morning"


def _manifest(store: LocalStore, job: str, day: dt.date, discriminator: str | None = None) -> None:
    payload = {
        "schema_version": "run_manifest.v1",
        "run_id": "01JG000000000000000000000" + job[0].upper(),
        "job": job,
        "trading_day": day.isoformat(),
        "calendar_date": (discriminator or day.isoformat()),
        "status": "ok",
        "reason": "",
        "cost_usd": 0.0,
        "attempts": [{"n": 1, "reason": "initial"}],
    }
    if discriminator is not None:
        payload["discriminator"] = discriminator
    store.put_bytes(
        manifest_key(job, day.isoformat(), discriminator=discriminator),
        json.dumps(payload).encode(),
    )


def _live_shape(store: LocalStore) -> None:
    """The store exactly as it stood at 2026-09-04T14:35Z.

    The sweep ran on THURSDAY. The report for THURSDAY, due 13:00Z FRIDAY,
    never arrived. Nothing else about the day is unusual.
    """
    _manifest(store, "alerts.sweep", THURSDAY, discriminator=FRIDAY.isoformat())


def _absent(pages, job: str) -> set[dt.date]:
    return {p.trading_day for p in pages if p.job == job and p.condition == "absence"}


class TestTheMissedFiringIsSeen:
    def test_the_day_a_next_calendar_day_deadline_comes_due_is_evaluated(self, tmp_path) -> None:
        """The whole defect, in one assertion.

        THURSDAY's deadline falls on FRIDAY. The sweep RAN on Thursday — hours
        before that deadline existed — so the old day set, which re-evaluated
        only the days the sweep MISSED, excluded Thursday from every
        subsequent pass. Thursday was never re-examined by anything.
        """
        store = LocalStore(tmp_path)
        _live_shape(store)
        assert THURSDAY in days_to_evaluate(store, FRIDAY_SWEEP)

    def test_an_undelivered_morning_report_pages_absence(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _live_shape(store)
        assert THURSDAY in _absent(evaluate_absence(store, now=FRIDAY_SWEEP), MORNING)

    def test_the_same_store_with_the_report_delivered_pages_nothing_for_it(self, tmp_path) -> None:
        """The other half of the demonstration: a detector that fires on a
        healthy store is not a detector, it is noise with a deadline.
        """
        store = LocalStore(tmp_path)
        _live_shape(store)
        _manifest(store, MORNING, THURSDAY, discriminator=FRIDAY.isoformat())
        assert _absent(evaluate_absence(store, now=FRIDAY_SWEEP), MORNING) == set()

    def test_the_sweep_is_not_blind_before_it_first_ran(self, tmp_path) -> None:
        """Widening the day set must not manufacture a backlog. A cold start
        still reports only today — a first run that BREACHED the two-a-month
        ceiling would be a false reading of the one metric §11 risk 2 holds
        this module to.
        """
        store = LocalStore(tmp_path)
        assert days_to_evaluate(store, FRIDAY_SWEEP) == [FRIDAY]


class TestTheDeadlineComesFromAMeasurement:
    def test_the_registry_allows_at_least_the_measured_account_envelope(self) -> None:
        """N is set from the worst of four measured account-wide cron delays,
        not from a preference for a faster page. A deadline tighter than the
        delivery envelope this account has actually measured produces a
        near-daily page about GitHub's scheduler rather than about a report
        that did not go out, and an operator who dismisses one page dismisses
        the next.
        """
        deadline = load_registry()[MORNING].deadline
        assert deadline is not None
        slack = (deadline.due_at(THURSDAY) - OCCURRENCE).total_seconds() / 60
        assert slack >= LATENCY_FLOOR_MINUTES

    def test_the_deadline_is_still_the_same_calendar_morning(self) -> None:
        """And bounded above: a deadline that slid past the next occurrence
        would let one absent day hide behind the following one.
        """
        deadline = load_registry()[MORNING].deadline
        assert deadline is not None
        assert deadline.due_at(THURSDAY) < OCCURRENCE + dt.timedelta(hours=24)


class TestTheCalibratedThresholdChangesTheReading:
    """§7.4: the constant is not decorative. A report delivered 300 minutes
    after its occurrence is inside the account's measured worst cron delay
    (304 minutes, `authority-surface`, 2026-09-04) — the recalibrated
    360-minute deadline reads it as on time, and the original 210-minute one
    (9:30 ET, set from the single 113-minute `board.yml` sample, and shifted
    three hours earlier here alongside `OCCURRENCE` by the 2026-09-06
    `alpha-engine-config-I9966` cron recalibration — it was 12:30 ET against
    the pre-recalibration 13:00 UTC/09:00 ET occurrence) would have read it
    as absent.
    """

    ARRIVAL = OCCURRENCE + dt.timedelta(minutes=300)

    def test_the_new_360_minute_deadline_reads_a_300_minute_arrival_as_on_time(self) -> None:
        deadline = load_registry()[MORNING].deadline
        assert deadline is not None
        assert self.ARRIVAL <= deadline.due_at(THURSDAY)

    def test_the_original_210_minute_deadline_would_have_read_it_as_absent(self) -> None:
        from crucible.components import Deadline

        old_deadline = Deadline(anchor="next_calendar_day_at", cadence="daily", at=dt.time(9, 30))
        assert self.ARRIVAL > old_deadline.due_at(THURSDAY)

    def test_the_two_readings_differ(self) -> None:
        """The load-bearing assertion: the same arrival, judged by the two
        constants, does not produce the same verdict.
        """
        from crucible.components import Deadline

        new_deadline = load_registry()[MORNING].deadline
        assert new_deadline is not None
        old_deadline = Deadline(anchor="next_calendar_day_at", cadence="daily", at=dt.time(9, 30))

        new_reads_on_time = self.ARRIVAL <= new_deadline.due_at(THURSDAY)
        old_reads_on_time = self.ARRIVAL <= old_deadline.due_at(THURSDAY)

        assert new_reads_on_time is True
        assert old_reads_on_time is False
        assert new_reads_on_time != old_reads_on_time


class _Unlistable(LocalStore):
    """A store whose listing fails for ONE prefix. Everything else works."""

    def __init__(self, root, blocked: str) -> None:
        super().__init__(root)
        self._blocked = blocked

    def list_keys(self, prefix: str):
        if prefix.startswith(self._blocked):
            raise PermissionError("AccessDenied: s3:ListBucket")
        return super().list_keys(prefix)


class TestCouldNotAskIsNotNothingThere:
    """`crucible.gate._list_store_keys` draws this distinction; the sweep now
    draws the same one. A listing failure is a fact about our access.
    """

    def test_an_unreadable_prefix_does_not_become_an_absence_page(self, tmp_path) -> None:
        store = _Unlistable(tmp_path, f"runs/{MORNING}/")
        _live_shape(store)
        faults: list[str] = []
        pages = evaluate_absence(store, now=FRIDAY_SWEEP, access_faults=faults)
        assert _absent(pages, MORNING) == set()
        assert any("could not be read" in f for f in faults)

    def test_an_unreadable_prefix_raises_rather_than_reading_as_covered(self, tmp_path) -> None:
        store = _Unlistable(tmp_path, f"runs/{MORNING}/")
        _live_shape(store)
        with pytest.raises(StoreAccessError, match=MORNING):
            evaluate_absence(store, now=FRIDAY_SWEEP)

    def test_one_unreadable_prefix_does_not_withhold_the_other_pages(
        self, tmp_path, transport
    ) -> None:
        """The bug class this guards (`alpha-engine-config-I9955`): one arm
        that cannot be built refusing the whole slot. A pass that cannot list
        one prefix still owes the operator every page it DID observe, and
        then fails loudly.
        """
        store = _Unlistable(tmp_path, f"runs/{MORNING}/")
        _live_shape(store)
        with pytest.raises(StoreAccessError):
            sweep(store, now=FRIDAY_SWEEP, transport=transport, sweep_run_id="0" * 26)
        assert transport.pages, "pages observed before the fault were withheld"


class TestTheTriggerIsEvidence:
    """I9896 / I9914 / I9921 close on "a delivery no human dispatched".
    Nothing in `run_manifest.v2` can say that — `run_mode` is live-vs-replay,
    and a dispatched run is just as live as a scheduled one.
    """

    def test_the_platforms_own_variable_wins(self) -> None:
        """`GITHUB_EVENT_NAME` first: GitHub sets it, and `GITHUB_*` is a
        reserved prefix a workflow's `env:` block cannot override — so a
        dispatched run cannot file evidence claiming a schedule.
        """
        assert (
            resolve_trigger(
                {"GITHUB_EVENT_NAME": "workflow_dispatch", "CRUCIBLE_TRIGGER": "schedule"}
            )
            == "workflow_dispatch"
        )

    def test_a_non_github_dispatcher_can_still_say(self) -> None:
        assert resolve_trigger({"CRUCIBLE_TRIGGER": "eventbridge"}) == "eventbridge"

    def test_an_invocation_that_said_nothing_is_recorded_as_unknown(self) -> None:
        assert resolve_trigger({}) == TRIGGER_UNKNOWN

    def test_a_trigger_that_is_not_key_safe_is_refused(self) -> None:
        """It becomes an S3 key segment. Refusing is the only reading that is
        not a guess.
        """
        with pytest.raises(ValueError, match="not a usable trigger name"):
            resolve_trigger({"GITHUB_EVENT_NAME": "../../etc"})

    def test_the_predicate_key_is_the_answer_itself(self) -> None:
        """No body parse: the sweep that closes a tracker issue on evidence
        evaluates an S3 `exists` op, and the trigger is in the key.
        """
        key = morning_trigger_key("2026-09-03", "2026-09-04", "schedule")
        assert key == "runs/report.morning/2026-09-03/2026-09-04/trigger.schedule"

    def test_the_evidence_is_not_a_manifest(self) -> None:
        """Or it would satisfy the absence check it exists alongside — the
        `message.txt` defect (`alpha-engine-config-I9900`) one object over.
        """
        from crucible.keys import is_manifest_key

        assert not is_manifest_key(morning_trigger_key("2026-09-03", "2026-09-04", "schedule"))
