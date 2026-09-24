"""alpha-engine-config-I10134 deliverable 2: a dispatch record with no
manifest past its horizon is an ABSENCE, for the on-demand jobs
`components.yaml` carries no schedule or deadline for at all.

Normative source: `crucible/AGENTS.md` §Alerting, plan §4.6 (exactly two
page conditions — this is a third INPUT to the first, not a third
condition).
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from crucible.alerts import (
    DISPATCH_ABSENCE_HORIZON,
    InstanceReading,
    Page,
    evaluate_dispatch_absence,
    sweep,
)
from crucible.keys import dispatch_exit_key, dispatch_key, manifest_key, migration_key
from crucible.store import LocalStore

#: A Friday, well after that trading day's close (~20:00 UTC — see
#: `test_alerts_grouping.py`'s SATURDAY_NIGHT for the same boundary).
DISPATCHED_AT = dt.datetime(2026, 8, 28, 4, 3, tzinfo=dt.UTC)
#: Resolves to the PRIOR session (2026-08-27): before close on the 28th.
DISPATCH_TRADING_DAY = dt.date(2026, 8, 27)

#: The day the DEFAULT fixture dispatch's manifest actually lands under —
#: `--to`, not the dispatch clock (`alpha-engine-config-I10696`). A range job
#: (`data.heal`, `experiment.backfill`) passes `trading_day=end` to `run_job`
#: and carries no `--date` at all, so grading it against the wall clock
#: listed a prefix the manifest was never going to be written to and paged an
#: ABSENCE for a run whose artifact exists. `experiment.backfill` is
#: dispatched in chunks over historical ranges, so every one of them would
#: have produced that false page.
DISPATCH_TARGET_TRADING_DAY = dt.date(2025, 1, 21)

#: Ten hours after DISPATCHED_AT — the exact gap the issue was filed over,
#: comfortably past DISPATCH_ABSENCE_HORIZON.
PAST_HORIZON = DISPATCHED_AT + dt.timedelta(hours=10)
#: One hour after DISPATCHED_AT — inside the horizon; the box may still be
#: mid-run.
WITHIN_HORIZON = DISPATCHED_AT + dt.timedelta(hours=1)


def _no_log(job: str, instance_id: str) -> None:
    """A fake `read_box_log_tail` that finds no stream — the default for
    every test that is not exercising the CloudWatch rung, so a unit test
    never makes a real `logs:GetLogEvents` call."""
    return None


def _no_reason(instance_id: str) -> InstanceReading:
    """A fake `describe_instance_state_reason` that finds nothing — the
    default for every test that is not exercising classification itself, so
    a unit test never makes a real `ec2:DescribeInstances` call.

    `known=False` is the PRODUCTION reading (`alpha-engine-config-I11049`):
    EC2 purges a terminated instance about an hour after termination, and a
    page this detector fires without an exit record is at least
    DISPATCH_ABSENCE_HORIZON old."""
    return InstanceReading(False, None)


def _write_dispatch(
    store: LocalStore,
    *,
    job: str = "data.heal",
    dispatch_id: str = "01abc",
    args: str = "--from 2025-01-21 --to 2025-01-21",
    instance_id: str = "i-0cb52a780eb7eb90c",
    dispatched_at: dt.datetime = DISPATCHED_AT,
) -> None:
    store.put_bytes(
        dispatch_key(job, dispatch_id),
        json.dumps(
            {
                "schema_version": "dispatch_record.v1",
                "job": job,
                "args": args,
                "instance_id": instance_id,
                "requested_by": "test-dispatcher",
                "dispatched_at_utc": dispatched_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        ).encode(),
    )


#: A manifest that this dispatch's own run wrote: `started` at or after the
#: dispatch. `started` is REQUIRED on every `run_manifest.v2` document, which
#: is what makes it usable as the identity test (`alpha-engine-config-I10981`).
AFTER_DISPATCH = DISPATCHED_AT + dt.timedelta(minutes=4)
#: A manifest an EARLIER run left at the same key. This is the I10981 shape:
#: `runs/data.heal/2026-01-30/run.json` held the prior heal's success while
#: the dispatch that was supposed to write there died four minutes in.
BEFORE_DISPATCH = DISPATCHED_AT - dt.timedelta(days=1)


def _manifest_body(
    started: dt.datetime | None,
    *,
    run_id: str = "01M2GPZ56BVJRBJHHSMHZRQSEM",
    status: str = "ok",
) -> bytes:
    body: dict[str, object] = {"status": status, "run_id": run_id}
    if started is not None:
        body["started"] = started.strftime("%Y-%m-%dT%H:%M:%SZ")
    return json.dumps(body).encode()


class TestDispatchAbsence:
    def test_a_dispatch_past_horizon_with_no_manifest_pages_absence(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _write_dispatch(store)
        pages = evaluate_dispatch_absence(
            store,
            now=PAST_HORIZON,
            describe_instance_state_reason=_no_reason,
            read_box_log_tail=_no_log,
        )
        assert pages == [
            Page(
                condition="absence",
                job="data.heal",
                trading_day=DISPATCH_TARGET_TRADING_DAY,
                reason=pages[0].reason,
            )
        ]
        assert "i-0cb52a780eb7eb90c" in pages[0].reason
        assert "--from 2025-01-21 --to 2025-01-21" in pages[0].reason

    def test_a_dispatch_still_inside_the_horizon_is_not_yet_due(self, tmp_path) -> None:
        """The same 'only past deadlines are evaluated' rule
        `evaluate_absence` states for scheduled jobs: a box mid-run must not
        page for being slow."""
        store = LocalStore(tmp_path)
        _write_dispatch(store)
        assert (
            evaluate_dispatch_absence(
                store,
                now=WITHIN_HORIZON,
                describe_instance_state_reason=_no_reason,
                read_box_log_tail=_no_log,
            )
            == []
        )

    def test_a_manifest_this_dispatch_wrote_clears_it(self, tmp_path) -> None:
        """AMENDED, not deleted (`alpha-engine-config-I10981` deliverable 4).

        This assertion used to write a bare `{"status": "ok"}` with no run_id
        and no timestamp and assert that it cleared the dispatch — mere key
        existence. Its original case is still served and still asserted here:
        a dispatch whose run really did write the manifest at the expected key
        is cleared and does not page. That case was the SCHEDULED one
        (`alpha-engine-config-I10134`), where a stale manifest from a prior
        dispatch of the same slot cannot normally exist; the rule was never
        designed against the on-demand RE-DISPATCH, where a genuinely older
        unrelated run's manifest already sits at the target key. The sibling
        below is that case."""
        store = LocalStore(tmp_path)
        _write_dispatch(store)
        store.put_bytes(
            manifest_key("data.heal", DISPATCH_TARGET_TRADING_DAY.isoformat()),
            _manifest_body(AFTER_DISPATCH),
        )
        assert (
            evaluate_dispatch_absence(
                store,
                now=PAST_HORIZON,
                describe_instance_state_reason=_no_reason,
                read_box_log_tail=_no_log,
            )
            == []
        )

    def test_a_discriminated_manifest_also_clears_the_dispatch(self, tmp_path) -> None:
        """Same rule `evaluate_absence` follows: a discriminated manifest
        under the job's prefix is still a manifest for that trading day."""
        store = LocalStore(tmp_path)
        _write_dispatch(store)
        store.put_bytes(
            manifest_key(
                "data.heal", DISPATCH_TARGET_TRADING_DAY.isoformat(), discriminator="r1of4"
            ),
            _manifest_body(AFTER_DISPATCH),
        )
        assert (
            evaluate_dispatch_absence(
                store,
                now=PAST_HORIZON,
                describe_instance_state_reason=_no_reason,
                read_box_log_tail=_no_log,
            )
            == []
        )

    def test_a_dispatch_record_with_an_unparseable_time_pages_rather_than_vanishing(
        self, tmp_path
    ) -> None:
        store = LocalStore(tmp_path)
        store.put_bytes(
            dispatch_key("data.heal", "badtime"),
            json.dumps(
                {
                    "job": "data.heal",
                    "args": "--from x",
                    "instance_id": "i-bad",
                    "dispatched_at_utc": "not-a-timestamp",
                }
            ).encode(),
        )
        pages = evaluate_dispatch_absence(
            store,
            now=PAST_HORIZON,
            describe_instance_state_reason=_no_reason,
            read_box_log_tail=_no_log,
        )
        assert len(pages) == 1
        assert "cannot be graded against the absence horizon" in pages[0].reason

    def test_a_dispatch_record_with_a_wrongly_typed_field_is_an_access_fault(
        self, tmp_path
    ) -> None:
        """`alpha-engine-config-I9847` (wave 2): `args` written as a list
        (not the string every real dispatcher writes) used to reach
        `args_synthetic_marker` unnamed. Validated through
        `DispatchRecordDocument` before either helper sees it — with no
        `access_faults` list supplied, the sweep raises `StoreAccessError`
        naming the record, the same shape an unreadable record already
        raised through before this migration."""
        from crucible.alerts import StoreAccessError

        store = LocalStore(tmp_path)
        store.put_bytes(
            dispatch_key("data.heal", "badargs"),
            json.dumps(
                {
                    "job": "data.heal",
                    "args": ["--from", "x"],
                    "instance_id": "i-bad",
                    "dispatched_at_utc": "2026-08-28T04:03:00Z",
                }
            ).encode(),
        )
        with pytest.raises(StoreAccessError, match="does not conform"):
            evaluate_dispatch_absence(
                store,
                now=PAST_HORIZON,
                describe_instance_state_reason=_no_reason,
                read_box_log_tail=_no_log,
            )

    def test_this_is_the_evidence_a_matching_scheduled_absence_groups_with(self, tmp_path) -> None:
        """§9.3: both share `cause_key` `absence:{trading_day}`, so an
        operator sees one incident naming both — never two separate pages
        for one bad day (`crucible.alerts.cause_key`)."""
        from crucible.alerts import cause_key

        store = LocalStore(tmp_path)
        _write_dispatch(store)
        [dispatch_page] = evaluate_dispatch_absence(
            store,
            now=PAST_HORIZON,
            describe_instance_state_reason=_no_reason,
            read_box_log_tail=_no_log,
        )
        scheduled_page = Page(
            condition="absence",
            job="data.daily",
            trading_day=DISPATCH_TARGET_TRADING_DAY,
            reason="no manifest under runs/data.daily/2025-01-21/; due ...",
        )
        assert cause_key(dispatch_page) == cause_key(scheduled_page)

    def test_dry_run_sweep_never_writes_a_bus_row_for_a_dispatch_absence(self, tmp_path) -> None:
        """Same `alpha-engine-config-I9922` R2-1 guarantee `sweep(dry_run=True)`
        already gives scheduled absences: reading, never emitting.

        An UNDATED dispatch (no `--date`, no `--to`) so this stays a
        single-incident fixture: a dated or ranged one is graded against the
        day it names, which is a different `cause_key` from the scheduled
        absences the same sweep finds — correct, and beside the point here.
        """
        store = LocalStore(tmp_path)
        _write_dispatch(store, args="--gap missing-panel")
        summary = sweep(
            store,
            now=PAST_HORIZON,
            sweep_run_id="run1",
            dry_run=True,
            describe_instance_state_reason=_no_reason,
            read_box_log_tail=_no_log,
        )
        assert summary["pages_emitted"] == 0
        assert summary["incidents_open"] == 1
        assert list(store.list_keys("alerts/")) == []


#: `migrate.arm_filed_on`/`migrate.code_sha` stamp `migration_run_id` from
#: `dt.datetime.now(dt.UTC)` at the START of the run, in this exact format
#: (`crucible.migrate.run_migrate_arm_filed_on`).
_MIGRATION_RUN_ID_FORMAT = "%Y%m%dT%H%M%S%fZ"


class TestANonJobHandlerClearsOnTheMigrationsPrefix:
    """`alpha-engine-config-I11022` residual finding, posted after the
    dispatch-script fix landed (`nous-ergon-ops-PR1322`).

    `crucible.cli.NON_JOB_HANDLERS` (`migrate.code_sha`, `migrate.arm_filed_on`)
    write no `run_manifest.v2` document, so grading them against
    `manifest_prefix(job, trading_day)` — the premise `evaluate_dispatch_absence`
    used for every job — pages every one of them, every night, even after a
    correction runs successfully. Their only durable record is
    `migrations/{trading_day}/{run_id}.json`, and this class asserts the
    sweep reads it instead.
    """

    def test_a_non_job_handler_dispatch_with_no_migration_record_pages_absence(
        self, tmp_path
    ) -> None:
        store = LocalStore(tmp_path)
        _write_dispatch(store, job="migrate.arm_filed_on", args="--dry-run")
        pages = evaluate_dispatch_absence(
            store,
            now=PAST_HORIZON,
            describe_instance_state_reason=_no_reason,
            read_box_log_tail=_no_log,
        )
        assert len(pages) == 1
        assert pages[0].job == "migrate.arm_filed_on"
        assert "migrations/" in pages[0].reason

    def test_a_migration_record_written_after_the_dispatch_clears_it(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _write_dispatch(store, job="migrate.arm_filed_on", args="")
        run_id = AFTER_DISPATCH.strftime(_MIGRATION_RUN_ID_FORMAT)
        store.put_bytes(
            migration_key(DISPATCH_TARGET_TRADING_DAY.isoformat(), run_id),
            json.dumps(
                {"migration_run_id": run_id, "dry_run": False, "corrected": [], "refused": []}
            ).encode(),
        )
        assert (
            evaluate_dispatch_absence(
                store,
                now=PAST_HORIZON,
                describe_instance_state_reason=_no_reason,
                read_box_log_tail=_no_log,
            )
            == []
        )

    def test_a_migration_record_written_before_the_dispatch_does_not_clear_it(
        self, tmp_path
    ) -> None:
        """The I10981 shape, in the migration prefix: a record left by an
        earlier attempt is not evidence of THIS dispatch."""
        store = LocalStore(tmp_path)
        _write_dispatch(store, job="migrate.arm_filed_on", args="")
        run_id = BEFORE_DISPATCH.strftime(_MIGRATION_RUN_ID_FORMAT)
        store.put_bytes(
            migration_key(DISPATCH_TARGET_TRADING_DAY.isoformat(), run_id),
            json.dumps(
                {"migration_run_id": run_id, "dry_run": False, "corrected": [], "refused": []}
            ).encode(),
        )
        pages = evaluate_dispatch_absence(
            store,
            now=PAST_HORIZON,
            describe_instance_state_reason=_no_reason,
            read_box_log_tail=_no_log,
        )
        assert len(pages) == 1
        assert pages[0].job == "migrate.arm_filed_on"

    def test_a_ulid_migration_record_from_migrate_history_does_not_falsely_clear(
        self, tmp_path
    ) -> None:
        """`migrate.history` (a real `JOBS` member) writes to the SAME
        `migrations/` root, keyed by `ctx.run_id` — a ULID, not a UTC
        timestamp. That record must not be misread as evidence for an
        unrelated non-job-handler dispatch."""
        store = LocalStore(tmp_path)
        _write_dispatch(store, job="migrate.arm_filed_on", args="")
        store.put_bytes(
            migration_key(DISPATCH_TARGET_TRADING_DAY.isoformat(), "01M2GPZ56BVJRBJHHSMHZRQSEM"),
            json.dumps({"schema_version": "migration.v1"}).encode(),
        )
        pages = evaluate_dispatch_absence(
            store,
            now=PAST_HORIZON,
            describe_instance_state_reason=_no_reason,
            read_box_log_tail=_no_log,
        )
        assert len(pages) == 1


def test_the_horizon_is_stated_and_bounded() -> None:
    """Deliverable 2: 'a horizon that is stated, not implied.' Set from the
    `alpha-engine-config-I11032` measurement (slowest legitimate dispatch
    5.09h, an `experiment.backfill` chunk) and bounded well inside the
    daily sweep cadence, so it defers a page by at most one sweep."""
    assert DISPATCH_ABSENCE_HORIZON == dt.timedelta(hours=8)
    assert DISPATCH_ABSENCE_HORIZON < dt.timedelta(hours=24)


#: The slowest legitimate dispatch measured for `alpha-engine-config-I11032`:
#: an `experiment.backfill` replay chunk dispatched 2026-09-17T01:57:33Z that
#: booted in 3 minutes and ran 5.04h before writing its own `ok` manifest.
BACKFILL_DISPATCHED_AT = dt.datetime(2026, 9, 17, 1, 57, 33, tzinfo=dt.UTC)
BACKFILL_ARGS = "--date 2026-09-11 --run-mode replay --slot m --arm residual_momentum"


class TestTheHorizonIsSetFromTheMeasuredRuntimes:
    """`alpha-engine-config-I11032`. Once `-I10981` made a manifest clear only
    its own dispatch, the horizon decided, and 3h had been measured on
    `data.heal` alone."""

    def test_a_backfill_chunk_still_running_at_4h_does_not_page(self, tmp_path) -> None:
        """The case the issue was filed over: a healthy chunk, mid-run, read
        by a sweep four hours after its dispatch. Under 3h this paged."""
        store = LocalStore(tmp_path)
        _write_dispatch(
            store,
            job="experiment.backfill",
            args=BACKFILL_ARGS,
            dispatched_at=BACKFILL_DISPATCHED_AT,
        )
        assert (
            evaluate_dispatch_absence(
                store,
                now=BACKFILL_DISPATCHED_AT + dt.timedelta(hours=4),
                describe_instance_state_reason=_no_reason,
                read_box_log_tail=_no_log,
            )
            == []
        )

    def test_a_box_that_never_ends_still_pages_and_the_page_names_the_horizon(
        self, tmp_path
    ) -> None:
        """Widening is not suppression: past the horizon, with no manifest
        and no recorded end, the dispatch pages, and says after how long."""
        store = LocalStore(tmp_path)
        _write_dispatch(
            store,
            job="experiment.backfill",
            args=BACKFILL_ARGS,
            dispatched_at=BACKFILL_DISPATCHED_AT,
        )
        [page] = evaluate_dispatch_absence(
            store,
            now=BACKFILL_DISPATCHED_AT + dt.timedelta(hours=9),
            describe_instance_state_reason=_no_reason,
            read_box_log_tail=_no_log,
        )
        assert page.condition == "absence"
        assert page.job == "experiment.backfill"
        assert "after 8h" in page.reason


class TestARecordedEndIsGradedAtOnce:
    """`alpha-engine-config-I11032` deliverable 2: the exit record is a real
    end condition (the shape `crucible_manifest_wait.sh` uses), so a wider
    horizon does not delay the page for a box that died and said so."""

    def test_an_ended_dispatch_with_no_manifest_pages_inside_the_horizon(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _write_dispatch(store)
        _write_exit_record(store, _exit_record())
        [page] = evaluate_dispatch_absence(
            store,
            now=WITHIN_HORIZON,
            describe_instance_state_reason=_no_reason,
            read_box_log_tail=_no_log,
        )
        assert "the box recorded its own exit: failed (code 1)" in page.reason

    def test_an_ended_dispatch_whose_own_manifest_landed_does_not_page(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _write_dispatch(store)
        _write_exit_record(store, _exit_record(exit_code=0, exit_class="ok", manifest_written=True))
        store.put_bytes(
            manifest_key("data.heal", DISPATCH_TARGET_TRADING_DAY.isoformat()),
            _manifest_body(AFTER_DISPATCH),
        )
        assert (
            evaluate_dispatch_absence(
                store,
                now=WITHIN_HORIZON,
                describe_instance_state_reason=_no_reason,
                read_box_log_tail=_no_log,
            )
            == []
        )

    def test_no_recorded_end_inside_the_horizon_is_still_not_due(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _write_dispatch(store)
        assert (
            evaluate_dispatch_absence(
                store,
                now=WITHIN_HORIZON,
                describe_instance_state_reason=_no_reason,
                read_box_log_tail=_no_log,
            )
            == []
        )

    def test_an_unreadable_exit_record_is_not_an_end(self, tmp_path) -> None:
        """A writer defect must not grade a box that may still be running.
        The horizon still applies, and the page names the defect once due."""
        store = LocalStore(tmp_path)
        _write_dispatch(store)
        store.put_bytes(dispatch_exit_key("data.heal", "01abc"), b"{not json")
        assert (
            evaluate_dispatch_absence(
                store,
                now=WITHIN_HORIZON,
                describe_instance_state_reason=_no_reason,
                read_box_log_tail=_no_log,
            )
            == []
        )


class TestDispatchAbsenceClassification:
    """alpha-engine-config-I10149 part B. `evaluate_dispatch_absence` must
    say WHICH kind of gone the dispatched instance is — a spot reclamation
    calls for a re-dispatch, a box that died mid-run or never started calls
    for investigation — rather than reporting a bare, unclassified absence
    identically for both.
    """

    def test_a_spot_reclaimed_instance_is_named_as_such(self, tmp_path) -> None:
        """Measured 2026-09-07: `StateReason.Message` on a spot-reclaimed
        box names `Server.SpotInstanceTermination`."""
        store = LocalStore(tmp_path)
        _write_dispatch(store, instance_id="i-096d52ca7a0c2ff21")

        def reclaimed(instance_id: str) -> InstanceReading:
            assert instance_id == "i-096d52ca7a0c2ff21"
            return InstanceReading(
                True, "Server.SpotInstanceTermination: Spot instance termination"
            )

        [page] = evaluate_dispatch_absence(
            store,
            now=PAST_HORIZON,
            describe_instance_state_reason=reclaimed,
            read_box_log_tail=_no_log,
        )
        assert "reclaimed by AWS" in page.reason
        assert "Server.SpotInstanceTermination" in page.reason
        assert "re-dispatch" in page.reason

    def test_a_box_that_died_mid_run_is_named_as_unreclaimed(self, tmp_path) -> None:
        """A `StateReason.Message` that is present and NOT the spot marker —
        e.g. a manual terminate, or a normal shutdown the box never reached
        its EXIT trap to log — reads as "investigate", not "reclaimed"."""
        store = LocalStore(tmp_path)
        _write_dispatch(store, instance_id="i-0deadbox00000000")

        def user_terminated(instance_id: str) -> InstanceReading:
            return InstanceReading(True, "Client.UserInitiatedShutdown: User initiated shutdown")

        [page] = evaluate_dispatch_absence(
            store,
            now=PAST_HORIZON,
            describe_instance_state_reason=user_terminated,
            read_box_log_tail=_no_log,
        )
        assert "reclaimed" not in page.reason
        assert "Client.UserInitiatedShutdown" in page.reason
        assert "investigate" in page.reason

    def test_a_describe_instances_failure_still_pages_unclassified(self, tmp_path) -> None:
        """The load-bearing case: a classifier that CANNOT determine the
        cause must never silence the page it enriches — the page fires with
        an unclassified reason naming the lookup failure, not a swallowed
        absence."""
        store = LocalStore(tmp_path)
        _write_dispatch(store, instance_id="i-0accessdenied0000")

        def denied(instance_id: str) -> InstanceReading:
            raise Exception("AccessDenied: not authorized to perform ec2:DescribeInstances")

        pages = evaluate_dispatch_absence(
            store,
            now=PAST_HORIZON,
            describe_instance_state_reason=denied,
            read_box_log_tail=_no_log,
        )
        assert len(pages) == 1
        assert pages[0].condition == "absence"
        assert "termination cause unknown" in pages[0].reason
        assert "AccessDenied" in pages[0].reason
        assert "classify by hand" in pages[0].reason

    def test_no_recorded_instance_id_is_named_rather_than_looked_up(self, tmp_path) -> None:
        """A record with no `instance_id` at all (an even older/malformed
        record) must not be handed to `ec2:DescribeInstances` as the literal
        string `"unknown"` — it is named as unrecorded instead."""
        store = LocalStore(tmp_path)
        store.put_bytes(
            dispatch_key("data.heal", "noinstance"),
            json.dumps(
                {
                    "job": "data.heal",
                    "args": "--from x",
                    "dispatched_at_utc": DISPATCHED_AT.strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
            ).encode(),
        )

        def fail_if_called(instance_id: str) -> InstanceReading:
            raise AssertionError("must not be called for a record with no instance_id")

        [page] = evaluate_dispatch_absence(
            store,
            now=PAST_HORIZON,
            describe_instance_state_reason=fail_if_called,
            read_box_log_tail=_no_log,
        )
        assert "no instance_id recorded" in page.reason


def test_classify_dispatch_absence_never_raises_on_a_lookup_failure(tmp_path) -> None:
    """Self-test of the classifier's own fail-safety, at the unit under it:
    `_classify_dispatch_absence` is the one place a transport failure on any
    rung of the ladder is folded into text rather than propagated."""
    from crucible.alerts import _classify_dispatch_absence

    def boom(instance_id: str) -> InstanceReading:
        raise RuntimeError("boom")

    text = _classify_dispatch_absence(
        LocalStore(tmp_path), "data.heal", "01abc", "i-0x", boom, _no_log
    )
    assert "termination cause unknown" in text
    assert "boom" in text


def test_classify_dispatch_absence_names_a_reclamation(tmp_path) -> None:
    from crucible.alerts import _classify_dispatch_absence

    text = _classify_dispatch_absence(
        LocalStore(tmp_path),
        "data.heal",
        "01abc",
        "i-0x",
        lambda _: InstanceReading(
            True, "Server.SpotInstanceTermination: Spot instance termination"
        ),
        _no_log,
    )
    assert "reclaimed by AWS" in text
    assert "re-dispatch" in text


def test_classify_dispatch_absence_when_ec2_described_it_with_no_reason(tmp_path) -> None:
    """The instance describes cleanly but carries no `StateReason.Message`
    — still running under a different lifecycle state, or terminated with
    nothing recorded. Neither "reclaimed" nor a swallowed lookup failure."""
    from crucible.alerts import _classify_dispatch_absence

    text = _classify_dispatch_absence(
        LocalStore(tmp_path),
        "data.heal",
        "01abc",
        "i-0x",
        lambda _: InstanceReading(True, None),
        _no_log,
    )
    assert "reclaimed" not in text
    assert "EC2 knows i-0x and recorded no state reason" in text


def test_classify_dispatch_absence_when_ec2_no_longer_knows_the_instance(tmp_path) -> None:
    """`alpha-engine-config-I11049`, the PRODUCTION case. Measured
    2026-09-18 under `ne-admin`: `describe-instances` returned
    `{"Reservations": []}` for all six instances named by that night's pages.
    That is "EC2 cannot answer a question this old", which the old text
    ("no termination reason available … investigate the box directly")
    reported as "the box died without saying why"."""
    from crucible.alerts import _classify_dispatch_absence

    text = _classify_dispatch_absence(
        LocalStore(tmp_path),
        "data.heal",
        "01abc",
        "i-02d1de194ea8420d5",
        lambda _: InstanceReading(False, None),
        _no_log,
    )
    assert "EC2 no longer knows i-02d1de194ea8420d5" in text
    assert "purged from DescribeInstances" in text
    assert "no termination reason available" not in text


class TestTheGradedDayComesFromTheArgsNotTheClock:
    """`_dispatch_target_trading_day` landed with the resolver but without a
    test asserting the prefix it produces, which is the thing that pages.
    These pin it in both directions, and pin the window that bounds the whole
    input.

    Measured 2026-09-09 against the live store: 20 of the 28 dispatch records
    under `runs/_dispatch/` carry an explicit `--date`. Nine `data.weekly` /
    `weekly` / `fault.probe` replays fired on 2026-09-09 for `--date
    2026-08-07`, `-08-14`, `-08-21` and `-09-11` had every manifest present at
    that date's prefix and were still paging ABSENCE against the dispatch
    day's — fourteen of the twenty-two members of that day's page, none of
    which could ever have cleared. In the other direction, two `alerts.sweep
    --now 2026-08-07` replays wrote no manifest anywhere and were cleared by
    the SCHEDULED sweep's manifest sitting under the dispatch day's prefix.
    """

    #: A replay dispatched on 2026-08-28 for a trading day five weeks earlier.
    REPLAY_ARGS = "--date 2026-07-24 --run-mode replay"
    REPLAY_TARGET = dt.date(2026, 7, 24)

    def test_a_manifest_at_the_dated_target_clears_the_dispatch(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _write_dispatch(store, args=self.REPLAY_ARGS)
        store.put_bytes(
            manifest_key("data.heal", self.REPLAY_TARGET.isoformat()),
            _manifest_body(AFTER_DISPATCH),
        )
        assert (
            evaluate_dispatch_absence(
                store,
                now=PAST_HORIZON,
                describe_instance_state_reason=_no_reason,
                read_box_log_tail=_no_log,
            )
            == []
        )

    def test_a_manifest_under_the_dispatch_day_does_not_clear_a_dated_dispatch(
        self, tmp_path
    ) -> None:
        """The false negative, and the one that matters: a replay that
        produced nothing must not be cleared by whatever the scheduled run of
        the same job happened to write that day."""
        store = LocalStore(tmp_path)
        _write_dispatch(store, args=self.REPLAY_ARGS)
        store.put_bytes(
            manifest_key("data.heal", DISPATCH_TRADING_DAY.isoformat()),
            _manifest_body(AFTER_DISPATCH),
        )
        [page] = evaluate_dispatch_absence(
            store,
            now=PAST_HORIZON,
            describe_instance_state_reason=_no_reason,
            read_box_log_tail=_no_log,
        )
        assert page.trading_day == self.REPLAY_TARGET
        assert self.REPLAY_TARGET.isoformat() in page.reason

    def test_an_undated_dispatch_still_grades_against_the_dispatch_clock(self, tmp_path) -> None:
        """The on-demand case `alpha-engine-config-I10134` was written for is
        unchanged — this is a refinement of that behaviour, not a replacement.

        A dispatch naming NEITHER `--date` nor `--to` — `alpha-engine-config-
        I10696` added the second — is the case that genuinely has nothing but
        the clock to be graded against.
        """
        store = LocalStore(tmp_path)
        _write_dispatch(store, args="--gap missing-panel")
        [page] = evaluate_dispatch_absence(
            store,
            now=PAST_HORIZON,
            describe_instance_state_reason=_no_reason,
            read_box_log_tail=_no_log,
        )
        assert page.trading_day == DISPATCH_TRADING_DAY

    def test_a_record_older_than_the_catch_up_window_is_no_longer_graded(self, tmp_path) -> None:
        """The bound every other input already had (`days_to_evaluate`). A
        dispatch record is written once and never rewritten, so without it the
        set of things the sweep grades grows for the life of the store and one
        dispatch that genuinely never landed is re-evaluated every night
        forever."""
        store = LocalStore(tmp_path)
        _write_dispatch(store)
        long_after = DISPATCHED_AT + dt.timedelta(days=30)
        assert (
            evaluate_dispatch_absence(
                store,
                now=long_after,
                describe_instance_state_reason=_no_reason,
                read_box_log_tail=_no_log,
            )
            == []
        )


class TestAManifestClearsOnlyItsOwnDispatch:
    """`alpha-engine-config-I10981`.

    The clearing predicate was the mere EXISTENCE of a manifest key. A
    crucible manifest key is `runs/{job}/{trading_day}/run.json` and an
    undiscriminated job overwrites it, so on a re-dispatch over a range that
    was healed before, an earlier run's `status: ok` manifest already occupies
    that exact key — and a dispatched job that dies mid-range is invisible at
    precisely the artifact anyone would check.

    The rule these tests pin is lifted from
    `nous-ergon-ops/scripts/lib/crucible_manifest_wait.sh`, which already
    grades a dispatch against a manifest whose identity DIFFERS from the one
    that was at the key before (`policy-shared-code`'s second adoption). Three
    outcomes, never two: only evidence of THIS run is a clear.
    """

    def test_a_manifest_that_predates_the_dispatch_pages(self, tmp_path) -> None:
        """The guard firing. Seen failing before the fix: under the old
        predicate this returned `[]`."""
        store = LocalStore(tmp_path)
        _write_dispatch(store)
        store.put_bytes(
            manifest_key("data.heal", DISPATCH_TARGET_TRADING_DAY.isoformat()),
            _manifest_body(BEFORE_DISPATCH),
        )
        [page] = evaluate_dispatch_absence(
            store,
            now=PAST_HORIZON,
            describe_instance_state_reason=_no_reason,
            read_box_log_tail=_no_log,
        )
        assert page.condition == "absence"
        assert page.job == "data.heal"
        assert "PREDATE" in page.reason, "the page says WHY the manifest did not clear"
        assert "01M2GPZ56BVJRBJHHSMHZRQSEM" in page.reason, (
            "the page names the run_id sitting at the key, so an operator can tell "
            "the stale manifest from the one that never arrived"
        )

    def test_the_i10981_instance_shape_pages(self, tmp_path) -> None:
        """The measured instance, reproduced: dispatch `441e55ef…` for
        `data.heal --date 2026-01-30 --from 2025-07-01 --to 2026-01-30`, made
        2026-09-15T01:20:57Z on instance `i-02d1de194ea8420d5`, against the
        prior I10703 heal's `status: ok` manifest (`run_id
        01M2GPZ56BVJRBJHHSMHZRQSEM`) already sitting at
        `runs/data.heal/2026-01-30/run.json`.

        The real dispatch is outside the catch-up window now, so the shape is
        reproduced on a fixture rather than re-graded live — which is what the
        issue's third closes-when clause asks for."""
        dispatched_at = dt.datetime(2026, 8, 28, 1, 20, 57, tzinfo=dt.UTC)
        store = LocalStore(tmp_path)
        _write_dispatch(
            store,
            dispatch_id="441e55ef6f6d6478f42e84919b81a960",
            args="--date 2026-01-30 --from 2025-07-01 --to 2026-01-30 --run-mode live",
            instance_id="i-02d1de194ea8420d5",
            dispatched_at=dispatched_at,
        )
        store.put_bytes(
            manifest_key("data.heal", "2026-01-30"),
            _manifest_body(dispatched_at - dt.timedelta(hours=4)),
        )
        [page] = evaluate_dispatch_absence(
            store,
            now=dispatched_at + dt.timedelta(hours=10),
            describe_instance_state_reason=_no_reason,
            read_box_log_tail=_no_log,
        )
        assert page.trading_day == dt.date(2026, 1, 30)
        assert "i-02d1de194ea8420d5" in page.reason
        assert "01M2GPZ56BVJRBJHHSMHZRQSEM" in page.reason

    def test_a_manifest_with_no_started_does_not_clear(self, tmp_path) -> None:
        """Fail loud. A manifest we cannot age against the dispatch is not a
        clear — the posture `check_feature_layer_provenance` takes for
        UNREADABLE. Treating unparseable as cleared reintroduces this bug in a
        new shape, and it is exactly the bare `{"status": "ok"}` blob the
        amended test used to write."""
        store = LocalStore(tmp_path)
        _write_dispatch(store)
        store.put_bytes(
            manifest_key("data.heal", DISPATCH_TARGET_TRADING_DAY.isoformat()),
            _manifest_body(None),
        )
        [page] = evaluate_dispatch_absence(
            store,
            now=PAST_HORIZON,
            describe_instance_state_reason=_no_reason,
            read_box_log_tail=_no_log,
        )
        assert "could not be aged" in page.reason
        assert "not a clear" in page.reason

    def test_a_manifest_with_an_unparseable_started_does_not_clear(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _write_dispatch(store)
        store.put_bytes(
            manifest_key("data.heal", DISPATCH_TARGET_TRADING_DAY.isoformat()),
            json.dumps({"status": "ok", "run_id": "01X", "started": "not a time"}).encode(),
        )
        [page] = evaluate_dispatch_absence(
            store,
            now=PAST_HORIZON,
            describe_instance_state_reason=_no_reason,
            read_box_log_tail=_no_log,
        )
        assert "could not be aged" in page.reason

    def test_a_stale_manifest_beside_a_fresh_one_still_clears(self, tmp_path) -> None:
        """The other direction, so the fix cannot become a source of false
        pages: a discriminated job writes several manifests under one prefix,
        and one of them being older than the dispatch is normal."""
        store = LocalStore(tmp_path)
        _write_dispatch(store)
        day = DISPATCH_TARGET_TRADING_DAY.isoformat()
        store.put_bytes(
            manifest_key("data.heal", day, discriminator="chunk1"),
            _manifest_body(BEFORE_DISPATCH, run_id="01OLD"),
        )
        store.put_bytes(
            manifest_key("data.heal", day, discriminator="chunk2"),
            _manifest_body(AFTER_DISPATCH, run_id="01NEW"),
        )
        assert (
            evaluate_dispatch_absence(
                store,
                now=PAST_HORIZON,
                describe_instance_state_reason=_no_reason,
                read_box_log_tail=_no_log,
            )
            == []
        )


# -- alpha-engine-config-I11048: the range-job rebinding ---------------------
#
# The REAL records, copied verbatim out of the v2 store on 2026-09-18
# rather than invented: the defect was that the tests were written
# from the docstring's premise ("a range job carries no `--date` at all"),
# which was false for every dispatch the fleet has ever made.

#: `runs/_dispatch/data.heal/a6e08e3207b59295f0d34c13d6aa84c3.json`, verbatim.
#: Its manifest is at `runs/data.heal/2026-02-02/run.json` (`--to`), status
#: ok; the detector looked under `runs/data.heal/2026-08-14/` (`--date`) and
#: paged.
REAL_HEAL_RECORD = {
    "args": (
        "--date 2026-08-14 --run-mode live --from 2025-11-17 --to 2026-02-02 "
        "--gap i10733-inst-ownership-null-band-v553618c991dd"
    ),
    "attempts": [{"n": 1, "reason": "initial"}],
    "dispatched_at_utc": "2026-09-17T00:24:50Z",
    "instance_id": "i-07d9f2bbdc6ee5843",
    "job": "data.heal",
    "placed": {"instance_type": "r5a.large", "market": "spot", "subnet_id": "subnet-c670118d"},
    "schema_version": "dispatch_record.v1",
}

REAL_HEAL_DISPATCHED_AT = dt.datetime(2026, 9, 17, 0, 24, 50, tzinfo=dt.UTC)


def _write_real_record(store: LocalStore, record: dict) -> None:
    store.put_bytes(
        dispatch_key(record["job"], "a6e08e3207b59295f0d34c13d6aa84c3"),
        json.dumps(record).encode(),
    )


def _write_manifest(
    store: LocalStore, job: str, day: str, started: dt.datetime, *, discriminator: str | None = None
) -> None:
    store.put_bytes(
        manifest_key(job, day, discriminator=discriminator)
        if discriminator
        else manifest_key(job, day),
        json.dumps(
            {
                "run_id": "01M2PC6SMMCM6VEMFBGMQZQ5A2",
                "status": "ok",
                "started": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        ).encode(),
    )


class TestARangeJobIsGradedAgainstTheKeyItsManifestBindsTo:
    """`alpha-engine-config-I11048`. `--date` is injected UNCONDITIONALLY by
    `nous-ergon-ops/scripts/dispatch_crucible_v2_job.sh`, so the tie-break
    written as unreachable is the one that fires on every range dispatch."""

    def test_the_real_record_with_both_flags_does_not_page_when_its_manifest_exists(
        self, tmp_path
    ) -> None:
        """The issue's own `Closes-when`, on the real record."""
        store = LocalStore(tmp_path)
        _write_real_record(store, REAL_HEAL_RECORD)
        _write_manifest(
            store, "data.heal", "2026-02-02", REAL_HEAL_DISPATCHED_AT + dt.timedelta(hours=2)
        )
        assert (
            evaluate_dispatch_absence(
                store,
                now=REAL_HEAL_DISPATCHED_AT + dt.timedelta(hours=10),
                describe_instance_state_reason=_no_reason,
                read_box_log_tail=_no_log,
            )
            == []
        )

    def test_a_manifest_under_the_date_flag_does_NOT_clear_a_range_job(self, tmp_path) -> None:
        """The other direction, and the one that matters: a range job that
        really did die must page, named against the day its manifest WOULD
        have been written under — a day somebody can go and look at."""
        store = LocalStore(tmp_path)
        _write_real_record(store, REAL_HEAL_RECORD)
        _write_manifest(
            store, "data.heal", "2026-08-14", REAL_HEAL_DISPATCHED_AT + dt.timedelta(hours=2)
        )
        [page] = evaluate_dispatch_absence(
            store,
            now=REAL_HEAL_DISPATCHED_AT + dt.timedelta(hours=10),
            describe_instance_state_reason=_no_reason,
            read_box_log_tail=_no_log,
        )
        assert page.trading_day == dt.date(2026, 2, 2)
        assert "runs/data.heal/2026-02-02/" in page.reason

    def test_a_non_range_job_still_reads_its_date(self, tmp_path) -> None:
        """`alpha-engine-config-I10696`'s fix, which this must not undo:
        `fault.probe` binds to `--date`, and a `--to` in its argv (there is
        none today) must never outrank it."""
        store = LocalStore(tmp_path)
        _write_dispatch(
            store,
            job="fault.probe",
            args="--date 2026-09-11 --run-mode replay --to 2026-09-01",
        )
        [page] = evaluate_dispatch_absence(
            store,
            now=PAST_HORIZON,
            describe_instance_state_reason=_no_reason,
            read_box_log_tail=_no_log,
        )
        assert page.trading_day == dt.date(2026, 9, 11)

    def test_the_arm_discriminator_still_clears_a_backfill(self, tmp_path) -> None:
        """Deliverable 3: the detector lists `manifest_prefix(job, day)` and
        accepts any manifest beneath it, so an arm-scoped key
        (`m.residual_momentum`) clears — asserted so it stays true now that
        the day underneath it changed."""
        store = LocalStore(tmp_path)
        _write_dispatch(
            store,
            job="experiment.backfill",
            args=(
                "--date 2026-09-11 --run-mode replay --slot m --arm residual_momentum "
                "--from 2024-05-01 --to 2026-06-04"
            ),
        )
        _write_manifest(
            store,
            "experiment.backfill",
            "2026-06-04",
            DISPATCHED_AT + dt.timedelta(hours=1),
            discriminator="m.residual_momentum",
        )
        assert (
            evaluate_dispatch_absence(
                store,
                now=PAST_HORIZON,
                describe_instance_state_reason=_no_reason,
                read_box_log_tail=_no_log,
            )
            == []
        )


# -- alpha-engine-config-I11049/-I11050/-I11051 -----------------------------


def _exit_record(**overrides) -> dict:
    document = {
        "schema_version": "dispatch_exit.v1",
        "dispatch_id": "01abc",
        "instance_id": "i-0cb52a780eb7eb90c",
        "job": "data.heal",
        "argv": "--date 2025-01-21 --from 2025-01-21 --to 2025-01-21",
        "exit_code": 1,
        "exit_class": "failed",
        "last_error_line": (
            "crucible.data.heal.MissingArtifactError: no fundamentals for 2025-01-21"
        ),
        "console_tail": "…\ncrucible data.heal exited 1",
        "log_group": "/crucible/data.heal",
        "log_stream": "i-0cb52a780eb7eb90c",
        "expected_manifest_key": "runs/data.heal/2025-01-21/run.json",
        "manifest_written": False,
        "redispatch_expected": False,
        "next_attempt_dispatch_id": None,
        "attempts": [{"n": 1, "reason": "initial"}],
        "finished_at_utc": "2026-08-28T04:30:00Z",
    }
    document.update(overrides)
    return document


def _write_exit_record(store: LocalStore, record: dict) -> None:
    store.put_bytes(
        dispatch_exit_key(record["job"], record["dispatch_id"]), json.dumps(record).encode()
    )


class TestThePageNamesTheCauseFromTheBoxsOwnExitRecord:
    """`alpha-engine-config-I11050`'s `Closes-when`: an absence page names
    the exit code and the failing line WITHOUT any human reading a log."""

    def test_the_exit_record_is_read_before_any_aws_api(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _write_dispatch(store)
        _write_exit_record(store, _exit_record())

        def fail_if_called(instance_id: str) -> InstanceReading:
            raise AssertionError("the EC2 rung must not be reached when an exit record exists")

        def no_logs(job: str, instance_id: str) -> str | None:
            raise AssertionError("the CloudWatch rung must not be reached either")

        [page] = evaluate_dispatch_absence(
            store,
            now=PAST_HORIZON,
            describe_instance_state_reason=fail_if_called,
            read_box_log_tail=no_logs,
        )
        assert "the box recorded its own exit: failed (code 1)" in page.reason
        assert "MissingArtifactError" in page.reason
        assert "/crucible/data.heal :: i-0cb52a780eb7eb90c" in page.reason
        assert "investigate the box directly" not in page.reason

    def test_an_argparse_refusal_reads_as_a_malformed_dispatch(self, tmp_path) -> None:
        """Measured 2026-09-17 on `i-04653230d4780dede`: `unrecognized
        arguments: --date 2026-09-17`, exit 2 — the box, the wheel, the store
        and the data were all fine."""
        store = LocalStore(tmp_path)
        _write_dispatch(store)
        _write_exit_record(
            store,
            _exit_record(
                exit_code=2,
                exit_class="refused",
                last_error_line="crucible: error: unrecognized arguments: --date 2026-09-17",
            ),
        )
        [page] = evaluate_dispatch_absence(
            store,
            now=PAST_HORIZON,
            describe_instance_state_reason=_no_reason,
            read_box_log_tail=_no_log,
        )
        assert "refused (code 2)" in page.reason
        assert "unrecognized arguments" in page.reason

    def test_a_clean_exit_with_no_manifest_pages_repo_rule_one(self, tmp_path) -> None:
        """Where repo rule 1 moved to (`alpha-engine-config-I11050`,
        2026-09-18). `DispatchExitDocument` used to REFUSE `exit_class: ok`
        with `manifest_written: false`, on the reasoning that an exit record
        must not excuse a missing manifest. The refusal ran inside a dying
        box and its stderr fell into a console already shipped, so the fleet
        wrote no exit records at all. The finding belongs on a surface that
        reaches a human: this page."""
        store = LocalStore(tmp_path)
        _write_dispatch(store)
        _write_exit_record(
            store,
            _exit_record(
                exit_code=0,
                exit_class="ok",
                last_error_line=None,
                manifest_written=False,
                redispatch_expected=False,
                next_attempt_dispatch_id=None,
            ),
        )
        [page] = evaluate_dispatch_absence(
            store,
            now=PAST_HORIZON,
            describe_instance_state_reason=_no_reason,
            read_box_log_tail=_no_log,
        )
        assert "THE JOB EXITED 0 AND NO MANIFEST IS AT THE KEY IT OWED" in page.reason
        assert "manifest or it did not happen" in page.reason
        assert "investigate the box directly" not in page.reason

    def test_an_unreadable_exit_record_is_not_silently_a_missing_one(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _write_dispatch(store)
        store.put_bytes(dispatch_exit_key("data.heal", "01abc"), b"{not json")
        [page] = evaluate_dispatch_absence(
            store,
            now=PAST_HORIZON,
            describe_instance_state_reason=_no_reason,
            read_box_log_tail=_no_log,
        )
        assert "could not be read" in page.reason
        assert "01abc.exit.json" in page.reason


class TestTheCloudWatchRungIsReadWhenThereIsNoExitRecord:
    """`alpha-engine-config-I11050`: the evidence was never missing. Every
    one of the six instances measured on 2026-09-18 had a complete stream."""

    def test_the_last_error_line_of_the_stream_is_rendered(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _write_dispatch(store, instance_id="i-025a6e0286a1bc4d0")

        def stream(job: str, instance_id: str) -> str:
            assert (job, instance_id) == ("data.heal", "i-025a6e0286a1bc4d0")
            return (
                "starting crucible data.heal\n"
                "crucible.weekly.ArcStageFailed: experiment.run[r] raised "
                "MissingArtifactError\n"
                "crucible data.heal exited 1\n"
            )

        [page] = evaluate_dispatch_absence(
            store,
            now=PAST_HORIZON,
            describe_instance_state_reason=_no_reason,
            read_box_log_tail=stream,
        )
        assert "ArcStageFailed" in page.reason
        assert "/crucible/data.heal :: i-025a6e0286a1bc4d0" in page.reason

    def test_nothing_anywhere_is_its_own_louder_page(self, tmp_path) -> None:
        """Deliverable 4: no manifest AND no exit record AND no stream means
        the box never reached its trap — a boot failure or a hard reclaim,
        which is a genuinely different remediation."""
        store = LocalStore(tmp_path)
        _write_dispatch(store)
        [page] = evaluate_dispatch_absence(
            store,
            now=PAST_HORIZON,
            describe_instance_state_reason=_no_reason,
            read_box_log_tail=_no_log,
        )
        assert "NOTHING recorded this box's exit" in page.reason
        assert "died before reaching its trap" in page.reason


class TestAnUnkeptRedispatchIsNamedAsOne:
    """`alpha-engine-config-I11051`. The reclaimed path suppresses its
    manifest on the promise that the dispatcher re-launches the job. On
    2026-09-15 that promise was not kept for one of eight EDGAR re-heal
    chunks and nothing said so for three days."""

    RECLAIMED = dict(
        exit_code=1,
        exit_class="spot_reclaimed",
        last_error_line=(
            "crucible.runner.SpotInterruptionError: spot_interruption: received signal 15"
        ),
        redispatch_expected=True,
        next_attempt_dispatch_id="01abc-r2",
    )

    def test_a_reclaimed_attempt_with_no_successor_pages_as_unkept(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _write_dispatch(store)
        _write_exit_record(store, _exit_record(**self.RECLAIMED))
        [page] = evaluate_dispatch_absence(
            store,
            now=PAST_HORIZON,
            describe_instance_state_reason=_no_reason,
            read_box_log_tail=_no_log,
        )
        assert "RECLAIMED AND THE RE-DISPATCH WAS NOT KEPT" in page.reason
        assert "01abc-r2" in page.reason

    def test_a_reclaimed_attempt_whose_successor_exists_says_so(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _write_dispatch(store)
        _write_exit_record(store, _exit_record(**self.RECLAIMED))
        store.put_bytes(
            dispatch_key("data.heal", "01abc-r2"),
            json.dumps(
                {
                    "job": "data.heal",
                    "args": "--from 2025-01-21 --to 2025-01-21",
                    "instance_id": "i-0successor",
                    "dispatched_at_utc": (DISPATCHED_AT + dt.timedelta(minutes=8)).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    ),
                    "attempts": [{"n": 1, "reason": "initial"}, {"n": 2, "reason": "spot"}],
                }
            ).encode(),
        )
        pages = evaluate_dispatch_absence(
            store,
            now=PAST_HORIZON,
            describe_instance_state_reason=_no_reason,
            read_box_log_tail=_no_log,
        )
        first = next(page for page in pages if "01abc.json" in page.reason)
        assert "re-dispatched as 01abc-r2" in first.reason
        assert "NOT KEPT" not in first.reason

    def test_the_attempt_ceiling_matches_the_runners(self) -> None:
        """Two constants, one fact — asserted rather than left to agree."""
        from crucible.alerts import MAX_DISPATCH_ATTEMPTS
        from crucible.runner import MAX_ATTEMPTS

        assert MAX_DISPATCH_ATTEMPTS == MAX_ATTEMPTS


class TestAHoleInAFannedOutGapIsVisibleAsAHole:
    """`alpha-engine-config-I11051` deliverable 4: seven of eight succeeding
    was indistinguishable from eight of eight on every surface."""

    def test_the_page_names_how_many_siblings_landed(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        gap = "i10733-edgar-fundamentals-reheal-v553618c991dd"
        landed = [("2022-07-29", "chunk1"), ("2023-09-29", "chunk2")]
        for day, dispatch_id in landed:
            _write_dispatch(
                store,
                dispatch_id=dispatch_id,
                args=f"--date {day} --from 2022-01-03 --to {day} --gap {gap} --run-mode live",
            )
            _write_manifest(store, "data.heal", day, DISPATCHED_AT + dt.timedelta(hours=1))
        _write_dispatch(
            store,
            dispatch_id="chunk3",
            args=f"--date 2026-01-30 --from 2025-07-01 --to 2026-01-30 --gap {gap} --run-mode live",
        )
        pages = evaluate_dispatch_absence(
            store,
            now=PAST_HORIZON,
            describe_instance_state_reason=_no_reason,
            read_box_log_tail=_no_log,
        )
        [page] = [page for page in pages if page.trading_day == dt.date(2026, 1, 30)]
        assert f"one of 3 chunks fanned out for gap {gap!r}" in page.reason
        assert "2 of them have a manifest" in page.reason
        assert "INCONSISTENT" in page.reason

    def test_a_dispatch_with_no_gap_says_nothing_about_fan_out(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _write_dispatch(store)
        [page] = evaluate_dispatch_absence(
            store,
            now=PAST_HORIZON,
            describe_instance_state_reason=_no_reason,
            read_box_log_tail=_no_log,
        )
        assert "chunks fanned out" not in page.reason


def test_an_exit_record_is_never_read_back_as_a_dispatch_record(tmp_path) -> None:
    """The trap this key shape sets: both documents live under
    `runs/_dispatch/{job}/` and both end `.json`. An exit record parsed as a
    dispatch record would be a dispatch with no manifest anywhere near it —
    the detector meant to explain absences would MANUFACTURE one per box."""
    store = LocalStore(tmp_path)
    _write_dispatch(store)
    _write_exit_record(store, _exit_record(exit_code=0, exit_class="ok", manifest_written=True))
    pages = evaluate_dispatch_absence(
        store,
        now=PAST_HORIZON,
        describe_instance_state_reason=_no_reason,
        read_box_log_tail=_no_log,
    )
    assert len(pages) == 1
