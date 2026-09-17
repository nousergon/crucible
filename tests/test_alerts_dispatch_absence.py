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
    Page,
    evaluate_dispatch_absence,
    sweep,
)
from crucible.keys import dispatch_key, manifest_key
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


def _no_reason(instance_id: str) -> None:
    """A fake `describe_instance_state_reason` that finds nothing — the
    default for every test that is not exercising classification itself, so
    a unit test never makes a real `ec2:DescribeInstances` call."""
    return None


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
            store, now=PAST_HORIZON, describe_instance_state_reason=_no_reason
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
                store, now=WITHIN_HORIZON, describe_instance_state_reason=_no_reason
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
                store, now=PAST_HORIZON, describe_instance_state_reason=_no_reason
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
                store, now=PAST_HORIZON, describe_instance_state_reason=_no_reason
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
            store, now=PAST_HORIZON, describe_instance_state_reason=_no_reason
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
                store, now=PAST_HORIZON, describe_instance_state_reason=_no_reason
            )

    def test_this_is_the_evidence_a_matching_scheduled_absence_groups_with(self, tmp_path) -> None:
        """§9.3: both share `cause_key` `absence:{trading_day}`, so an
        operator sees one incident naming both — never two separate pages
        for one bad day (`crucible.alerts.cause_key`)."""
        from crucible.alerts import cause_key

        store = LocalStore(tmp_path)
        _write_dispatch(store)
        [dispatch_page] = evaluate_dispatch_absence(
            store, now=PAST_HORIZON, describe_instance_state_reason=_no_reason
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
        )
        assert summary["pages_emitted"] == 0
        assert summary["incidents_open"] == 1
        assert list(store.list_keys("alerts/")) == []


def test_the_horizon_is_stated_and_bounded() -> None:
    """Deliverable 2: 'a horizon that is stated, not implied.'"""
    assert DISPATCH_ABSENCE_HORIZON == dt.timedelta(hours=3)


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

        def reclaimed(instance_id: str) -> str:
            assert instance_id == "i-096d52ca7a0c2ff21"
            return "Server.SpotInstanceTermination: Spot instance termination"

        [page] = evaluate_dispatch_absence(
            store, now=PAST_HORIZON, describe_instance_state_reason=reclaimed
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

        def user_terminated(instance_id: str) -> str:
            return "Client.UserInitiatedShutdown: User initiated shutdown"

        [page] = evaluate_dispatch_absence(
            store, now=PAST_HORIZON, describe_instance_state_reason=user_terminated
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

        def denied(instance_id: str) -> str:
            raise Exception("AccessDenied: not authorized to perform ec2:DescribeInstances")

        pages = evaluate_dispatch_absence(
            store, now=PAST_HORIZON, describe_instance_state_reason=denied
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

        def fail_if_called(instance_id: str) -> str:
            raise AssertionError("must not be called for a record with no instance_id")

        [page] = evaluate_dispatch_absence(
            store, now=PAST_HORIZON, describe_instance_state_reason=fail_if_called
        )
        assert "no instance_id recorded" in page.reason


def test_classify_dispatch_absence_never_raises_on_a_lookup_failure() -> None:
    """Self-test of the classifier's own fail-safety, at the unit under it:
    `_classify_dispatch_absence` is the one place a describe-instances
    failure is folded into text rather than propagated."""
    from crucible.alerts import _classify_dispatch_absence

    def boom(instance_id: str) -> str:
        raise RuntimeError("boom")

    text = _classify_dispatch_absence("i-0x", boom)
    assert "termination cause unknown" in text
    assert "boom" in text


def test_classify_dispatch_absence_names_a_reclamation() -> None:
    from crucible.alerts import _classify_dispatch_absence

    text = _classify_dispatch_absence(
        "i-0x", lambda _: "Server.SpotInstanceTermination: Spot instance termination"
    )
    assert "reclaimed by AWS" in text
    assert "re-dispatch" in text


def test_classify_dispatch_absence_with_no_reason_at_all() -> None:
    """The instance describes cleanly but carries no `StateReason.Message`
    — still running under a different lifecycle state, or terminated with
    nothing recorded. Neither "reclaimed" nor a swallowed lookup failure."""
    from crucible.alerts import _classify_dispatch_absence

    text = _classify_dispatch_absence("i-0x", lambda _: None)
    assert "reclaimed" not in text
    assert "no termination reason available" in text
    assert "investigate" in text


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
                store, now=PAST_HORIZON, describe_instance_state_reason=_no_reason
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
            store, now=PAST_HORIZON, describe_instance_state_reason=_no_reason
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
            store, now=PAST_HORIZON, describe_instance_state_reason=_no_reason
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
                store, now=long_after, describe_instance_state_reason=_no_reason
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
            store, now=PAST_HORIZON, describe_instance_state_reason=_no_reason
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
            store, now=PAST_HORIZON, describe_instance_state_reason=_no_reason
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
            store, now=PAST_HORIZON, describe_instance_state_reason=_no_reason
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
                store, now=PAST_HORIZON, describe_instance_state_reason=_no_reason
            )
            == []
        )
