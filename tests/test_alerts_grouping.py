"""Causal grouping, the bus, the ceiling, and the heartbeat.

Normative source: plan §4.6, §9.3, §11 risk 2.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from crucible.alerts import (
    ALERT_BUS_SCHEMA_VERSION,
    CEILING_WINDOW_TRADING_DAYS,
    MUTED_TOPIC_ARN_VAR,
    MUTED_TOPIC_VAR,
    PAGES_PER_MONTH_CEILING,
    PAGES_TOPIC_ARN_VAR,
    PAGES_TOPIC_VAR,
    Page,
    PageGroup,
    TopicUnresolvedError,
    _week_summary,
    cause_key,
    ceiling_metric,
    days_to_evaluate,
    emit,
    evaluate_failure,
    group_pages,
    heartbeat,
    incident_key,
    muted_topic,
    pages_in_window,
    pages_topic,
    sweep,
    topic_arn,
)
from crucible.manifest import manifest_key
from crucible.store import LocalStore

FRIDAY = dt.date(2026, 8, 28)
SATURDAY_NIGHT = dt.datetime(2026, 8, 29, 23, 30, tzinfo=dt.UTC)

#: The five sessions before FRIDAY, most recent first. The catch-up window.
THURSDAY = dt.date(2026, 8, 27)
WEDNESDAY = dt.date(2026, 8, 26)
TUESDAY = dt.date(2026, 8, 25)
MONDAY = dt.date(2026, 8, 24)

#: The two topic NAMES are read from the environment (`alpha-engine-config-
#: I10156`), which the autouse `declared_topics` fixture in `conftest.py`
#: sets to a synthetic value before every test — so the two ARNs below are
#: built inside each test that needs them, never at import time.


def _failed(job: str, reason: str, run_id: str = "01JG0000000000000000000001") -> Page:
    return Page(condition="failure", job=job, trading_day=FRIDAY, reason=reason, run_id=run_id)


def _write_manifest(
    store,
    job: str,
    *,
    status: str,
    reason: str = "",
    cost: float = 0.0,
    trading_day: dt.date = FRIDAY,
    calendar_date: dt.date | None = None,
    discriminator: str | None = None,
):
    payload = {
        "schema_version": "run_manifest.v1",
        "run_id": "01JG000000000000000000000" + job[0].upper(),
        "job": job,
        "trading_day": trading_day.isoformat(),
        "calendar_date": (calendar_date or trading_day).isoformat(),
        "status": status,
        "reason": reason,
        "cost_usd": cost,
        "attempts": [{"n": 1, "reason": "initial"}],
    }
    if discriminator is not None:
        payload["discriminator"] = discriminator
    store.put_bytes(
        manifest_key(job, trading_day.isoformat(), discriminator=discriminator),
        json.dumps(payload).encode(),
    )
    return payload


class TestCausalGrouping:
    def test_five_arms_failing_on_one_data_outage_is_one_page(self) -> None:
        """§9.3, stated as the requirement: one page naming five members."""
        pages = [
            _failed(job, "RuntimeError: data source yfinance returned nothing")
            for job in ("data.daily", "experiment.run", "experiment.grade", "report", "drift")
        ]
        groups = group_pages(pages)
        assert len(groups) == 1
        assert len(groups[0].members) == 5
        assert "5 members" in groups[0].render()

    def test_unrelated_failures_are_not_collapsed_into_one_page(self) -> None:
        """The failure mode of grouping: collapsing everything hides N-1 of
        them. A reason matching no cause groups on its own job, which
        degrades to the pre-grouping behaviour rather than to one page."""
        groups = group_pages(
            [
                _failed("data.daily", "ValueError: schema drift in the fundamentals frame"),
                _failed("report", "KeyError: missing attribution row"),
            ]
        )
        assert len(groups) == 2

    def test_every_absence_on_one_day_is_one_page(self) -> None:
        """Six absent manifests on a morning the scheduler was down is one
        operator action, not six pages."""
        pages = [
            Page(condition="absence", job=job, trading_day=FRIDAY, reason="no manifest; due …")
            for job in ("data.weekly", "experiment.run", "experiment.grade")
        ]
        assert len(group_pages(pages)) == 1

    @pytest.mark.parametrize(
        ("reason", "expected"),
        [
            ("RuntimeError: spot_interruption: signal 15", "spot_interruption"),
            ("HTTPError: provider_5xx 503 from the router", "router_unavailable"),
            ("StaleReleasePointerError: releases/current names …", "stale_release_pointer"),
            ("ClientError: s3_throttling SlowDown", "s3_unavailable"),
        ],
    )
    def test_the_declared_causes_claim_their_failures(self, reason: str, expected: str) -> None:
        assert cause_key(_failed("data.daily", reason)).startswith(expected)

    def test_a_group_with_no_members_is_not_a_page(self) -> None:
        with pytest.raises(ValueError, match="no members"):
            PageGroup("x", ())


class TestFailureCondition:
    def test_the_page_reason_is_the_manifest_reason_verbatim(self, tmp_path) -> None:
        """A page and its artifact disagreeing is worse than either alone."""
        store = LocalStore(tmp_path)
        _write_manifest(store, "data.daily", status="failed", reason="RuntimeError: boom at x.py:3")
        pages = evaluate_failure(store, now=SATURDAY_NIGHT)
        assert [p.reason for p in pages] == ["RuntimeError: boom at x.py:3"]

    def test_an_ok_manifest_pages_for_nothing(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _write_manifest(store, "data.daily", status="ok")
        assert evaluate_failure(store, now=SATURDAY_NIGHT) == []

    def test_an_unreadable_manifest_is_itself_a_failure_page(self, tmp_path) -> None:
        """Reading past it would drop the run most likely to be broken."""
        store = LocalStore(tmp_path)
        store.put_bytes(manifest_key("data.daily", FRIDAY.isoformat()), b"{not json")
        pages = evaluate_failure(store, now=SATURDAY_NIGHT)
        assert len(pages) == 1
        assert "unreadable" in pages[0].reason


class TestDiscriminatedManifests:
    """alpha-engine-config-I9781: `alerts.sweep` fires every calendar day, and
    Friday/Saturday/Sunday all resolve to Friday's trading day. Each firing
    now writes its own manifest (discriminated by `calendar_date`) instead of
    the last one silently overwriting the first two; the sweep's own reader
    logic has to see all of them, not just whichever key it used to guess."""

    def test_evaluate_failure_sees_every_discriminated_manifest_not_just_one(
        self, tmp_path
    ) -> None:
        store = LocalStore(tmp_path)
        _write_manifest(
            store,
            "alerts.sweep",
            status="ok",
            trading_day=FRIDAY,
            calendar_date=FRIDAY,
            discriminator="2026-08-28",
        )
        _write_manifest(
            store,
            "alerts.sweep",
            status="failed",
            reason="RuntimeError: SNS publish timed out",
            trading_day=FRIDAY,
            calendar_date=dt.date(2026, 8, 29),
            discriminator="2026-08-29",
        )
        _write_manifest(
            store,
            "alerts.sweep",
            status="ok",
            trading_day=FRIDAY,
            calendar_date=dt.date(2026, 8, 30),
            discriminator="2026-08-30",
        )
        pages = evaluate_failure(store, now=SATURDAY_NIGHT)
        assert [p.reason for p in pages] == ["RuntimeError: SNS publish timed out"]

    def test_days_to_evaluate_treats_any_discriminated_firing_as_the_sweep_having_run(
        self, tmp_path
    ) -> None:
        store = LocalStore(tmp_path)
        _write_manifest(
            store,
            "alerts.sweep",
            status="ok",
            trading_day=FRIDAY,
            calendar_date=dt.date(2026, 8, 30),
            discriminator="2026-08-30",
        )
        # Only the Sunday firing's manifest exists; the sweep still counts
        # as having run for FRIDAY, not as three consecutive absences.
        assert days_to_evaluate(store, SATURDAY_NIGHT) == [FRIDAY]

    def test_a_bare_and_a_discriminated_manifest_for_the_same_job_both_read(self, tmp_path) -> None:
        """Backward compatibility: a job with no discriminator (every job but
        the two I9781 gave one to) still pages exactly as it always did."""
        store = LocalStore(tmp_path)
        _write_manifest(store, "data.daily", status="failed", reason="boom")
        pages = evaluate_failure(store, now=SATURDAY_NIGHT)
        assert [p.reason for p in pages] == ["boom"]


class TestBus:
    def test_every_page_is_also_a_machine_readable_row(self, tmp_path, transport) -> None:
        """§7.3: a human-only alert is invisible to the response plane."""
        store = LocalStore(tmp_path)
        group = group_pages([_failed("data.daily", "RuntimeError: boom")])[0]
        keys = emit(store, [group], sweep_run_id="0" * 26, transport=transport)
        row = json.loads(store.get_bytes(keys[0]))
        assert row["schema_version"] == ALERT_BUS_SCHEMA_VERSION
        assert row["sent"] is True
        assert row["members"][0]["job"] == "data.daily"
        assert keys[0].startswith(f"alerts/{FRIDAY.isoformat()}/")

    def test_a_failure_rows_alert_id_is_the_failed_runs_id(self, tmp_path, transport) -> None:
        """That is what makes the bus row joinable to the manifest."""
        store = LocalStore(tmp_path)
        group = group_pages([_failed("data.daily", "boom", run_id="01JG00000000000000000000ZZ")])[0]
        keys = emit(store, [group], sweep_run_id="0" * 26, transport=transport)
        row = json.loads(store.get_bytes(keys[0]))
        assert row["alert_id"] == "01JG00000000000000000000ZZ"
        assert row["alert_id_is_run_id"] is True
        assert row["members"][0]["run_id"] == "01JG00000000000000000000ZZ"

    def test_an_absence_row_says_its_id_is_not_a_run_id(self, tmp_path, transport) -> None:
        """There was no run, so there is no run id. The field a machine joins
        on is never silently empty."""
        store = LocalStore(tmp_path)
        page = Page(condition="absence", job="data.weekly", trading_day=FRIDAY, reason="late")
        keys = emit(store, group_pages([page]), sweep_run_id="0" * 26, transport=transport)
        assert json.loads(store.get_bytes(keys[0]))["alert_id_is_run_id"] is False

    def test_an_unjoinable_id_is_not_claimed_as_a_run_id(self, tmp_path, transport) -> None:
        """C9. A manifest that will not parse has no run_id to give, so the
        failure page carries the all-zeros sentinel. The row must not then
        assert a join to a manifest that carries no such id — the response
        plane reads this field to decide whether to go looking (§9.3).

        The two defects this replaces cancelled in exactly the wrong place:
        `bus_row` computed the flag as `alert_id != _sweep_alert_id_marker(
        alert_id)` where the helper was the IDENTITY function — always False,
        even for a real run id — and `emit` then overwrote it with
        `condition == "failure"`, which is True for this case.
        """
        store = LocalStore(tmp_path)
        store.put_bytes(manifest_key("data.daily", FRIDAY.isoformat()), b"{not json")
        result = sweep(store, now=SATURDAY_NIGHT, transport=transport, sweep_run_id="Z" * 26)
        rows = [json.loads(store.get_bytes(k)) for k in result["bus_keys"]]
        failures = [r for r in rows if r["condition"] == "failure"]
        assert len(failures) == 1
        assert failures[0]["alert_id_is_run_id"] is False, (
            "the sentinel run_id joins to nothing; a row claiming it does sends the "
            "response plane after a manifest that does not exist"
        )
        assert failures[0]["alert_id"] == "Z" * 26, (
            "with no joinable run id the row names the sweep that saw it, rather than "
            "a plausible-looking id that resolves to nothing"
        )

    def test_the_row_records_what_the_transport_actually_did(self, tmp_path, transport) -> None:
        store = LocalStore(tmp_path)
        group = group_pages([_failed("data.daily", "boom")])[0]
        emit(store, [group], sweep_run_id="0" * 26, transport=transport)
        assert transport.pages == 1
        assert transport.calls[0].kwargs["dedup_window_min"] is None


class TestDeliveryHonesty:
    """C10. `sent` is what happened on the transport, never what was asked."""

    def test_a_dedup_suppressed_publish_is_not_recorded_as_delivered(self, tmp_path) -> None:
        """krepis sets `any_ok=True` on a dedup-suppressed publish by its own
        documented contract — "logically in the operator's hands" by virtue
        of an earlier send. Writing that into `sent` produces exactly the row
        the field's docstring calls worse than no row: a durable claim of
        delivery for a page that never left."""
        store = LocalStore(tmp_path)

        class Suppressed:
            any_ok = True
            dedup_skipped = True
            muted = False
            telegram_destination = None

        keys = emit(
            store,
            group_pages([_failed("data.daily", "boom")]),
            sweep_run_id="0" * 26,
            transport=lambda *a, **k: Suppressed(),
        )
        assert json.loads(store.get_bytes(keys[0]))["sent"] is False

    def test_a_muted_publish_is_not_recorded_as_delivered(self, tmp_path) -> None:
        store = LocalStore(tmp_path)

        class Muted:
            any_ok = True
            dedup_skipped = False
            muted = True
            telegram_destination = None

        keys = emit(
            store,
            group_pages([_failed("data.daily", "boom")]),
            sweep_run_id="0" * 26,
            transport=lambda *a, **k: Muted(),
        )
        assert json.loads(store.get_bytes(keys[0]))["sent"] is False

    def test_a_result_that_cannot_answer_is_not_read_as_delivered(self, tmp_path) -> None:
        """The second half of the same fault: `bool(getattr(result, "any_ok",
        True))` recorded an unknown outcome as a delivery. An optimistic
        default on a delivery claim is the fleet's `getattr(x, attr,
        <optimistic>)` pattern, and it raises here instead."""
        store = LocalStore(tmp_path)
        with pytest.raises(TypeError, match="cannot say whether"):
            emit(
                store,
                group_pages([_failed("data.daily", "boom")]),
                sweep_run_id="0" * 26,
                transport=lambda *a, **k: object(),
            )

    def test_the_destination_is_read_from_the_field_krepis_actually_sets(self, tmp_path) -> None:
        """`PublishResult` has never carried a `destination` attribute; it
        carries `telegram_destination`. Reading the name it does not have
        meant every real bus row recorded the literal default string."""
        store = LocalStore(tmp_path)

        class Delivered:
            any_ok = True
            dedup_skipped = False
            muted = False
            telegram_destination = "operator_incident_chat"

        keys = emit(
            store,
            group_pages([_failed("data.daily", "boom")]),
            sweep_run_id="0" * 26,
            transport=lambda *a, **k: Delivered(),
        )
        assert json.loads(store.get_bytes(keys[0]))["destination"] == "operator_incident_chat"


class TestIncidentIdentity:
    """C2. One incident pages once, however many sweeps observe it."""

    def test_the_key_does_not_move_when_the_membership_does(self) -> None:
        """The measured defect: the key was built from `members[0].job`, the
        alphabetically-first member. Friday's sweep saw one absent job and
        Saturday's saw two, so the same unchanged absence produced two
        different keys and the forever-dedup window never engaged."""
        friday = group_pages(
            [Page(condition="absence", job="data.daily", trading_day=FRIDAY, reason="late")]
        )[0]
        saturday = group_pages(
            [
                Page(condition="absence", job=job, trading_day=FRIDAY, reason="late")
                for job in ("console", "data.daily")
            ]
        )[0]
        assert friday.members[0].job != saturday.members[0].job
        assert incident_key(friday) == incident_key(saturday)

    def test_a_persistent_absence_pages_once_and_writes_one_row(self, tmp_path, transport):
        """The reproduction, run: the sweep fires 21:00 ET every calendar day
        and Friday, Saturday and Sunday nights all resolve to Friday's
        session. One absent Friday artifact produced three transport sends,
        three bus rows and a BREACH of the declared two-a-month ceiling
        without anything new having gone wrong."""
        store = LocalStore(tmp_path)
        nights = [
            dt.datetime(2026, 8, 29, 1, 0, tzinfo=dt.UTC),
            dt.datetime(2026, 8, 30, 1, 0, tzinfo=dt.UTC),
            dt.datetime(2026, 8, 31, 1, 0, tzinfo=dt.UTC),
        ]
        for index, night in enumerate(nights):
            result = sweep(store, now=night, transport=transport, sweep_run_id=f"{index}" * 26)
        assert transport.pages == 1, "one incident, one page"
        rows = sorted(store.list_keys("alerts/"))
        assert len(rows) == 1, "one incident, one bus row"
        assert result["metric"]["status"] == "OK", (
            "the ceiling metric counted bus rows, so it counted SWEEP CADENCE: a single "
            "absence read as a BREACH by its third night"
        )
        row = json.loads(store.get_bytes(rows[0]))
        assert row["observations"] == 3
        assert row["last_observed_utc"] > row["first_observed_utc"]

    def test_a_reobservation_records_the_membership_it_saw(self, tmp_path, transport) -> None:
        """Suppressing the page is not the same as recording nothing: the row
        keeps the page that went out and carries the current picture beside
        it."""
        store = LocalStore(tmp_path)
        first = group_pages([_failed("data.daily", "yfinance is down")])[0]
        emit(store, [first], sweep_run_id="0" * 26, transport=transport)
        second = group_pages(
            [_failed(job, "yfinance is down") for job in ("data.daily", "report")]
        )[0]
        keys = emit(store, [second], sweep_run_id="1" * 26, transport=transport)
        row = json.loads(store.get_bytes(keys[0]))
        assert transport.pages == 1
        assert [m["job"] for m in row["members"]] == ["data.daily"]
        assert [m["job"] for m in row["members_now"]] == ["data.daily", "report"]


class TestMutedRouting:
    def test_legacy_pages_go_to_the_muted_topics_arn(self, tmp_path, transport, monkeypatch):
        """§11 risk 5: the old system's alerts during the overlap. Muted at
        the destination, never by not emitting — a mute implemented as
        silence is indistinguishable from a dead producer.

        C8: what was passed was the bare topic NAME. `krepis.alerts`'
        `_resolve_sns_topic_arn` returns an explicit value verbatim, so SNS
        would have refused it with InvalidParameter. The previous test
        asserted the kwarg equalled the same bare constant — true for any
        implementation, including the broken one."""
        muted_arn = f"arn:aws:sns:us-east-1:123456789012:{muted_topic()}"
        monkeypatch.setenv(MUTED_TOPIC_ARN_VAR, muted_arn)
        store = LocalStore(tmp_path)
        group = group_pages([_failed("data.daily", "boom")])[0]
        emit(store, [group], sweep_run_id="0" * 26, legacy=True, transport=transport)
        arn = transport.calls[0].kwargs["sns_topic_arn"]
        assert arn == muted_arn
        assert arn.startswith("arn:aws:sns:"), "SNS refuses a bare topic name"


class TestTopicResolution:
    """C8. One adapter, and it names the topic the stack grants."""

    def test_pages_resolve_to_the_topic_the_stack_grants(self, monkeypatch) -> None:
        """The stack creates, tags and exports the pages topic and grants
        `sns:Publish` on that topic and the muted one ONLY. Passing None let
        krepis resolve its own default, an unwatched topic, where the v2
        RuntimeRole has no grant: every page's SNS half would have been
        AccessDenied on the day the stack was applied, while Telegram
        succeeded and `any_ok` stayed True."""
        pages_arn = f"arn:aws:sns:us-east-1:123456789012:{pages_topic()}"
        monkeypatch.setenv(PAGES_TOPIC_ARN_VAR, pages_arn)
        assert topic_arn() == pages_arn
        assert topic_arn().rsplit(":", 1)[-1] == pages_topic()

    def test_a_bare_topic_name_is_refused(self, monkeypatch) -> None:
        monkeypatch.setenv(PAGES_TOPIC_ARN_VAR, pages_topic())
        with pytest.raises(TopicUnresolvedError, match="not an SNS topic ARN"):
            topic_arn()

    def test_a_topic_the_role_is_not_granted_is_refused(self, monkeypatch) -> None:
        monkeypatch.setenv(
            PAGES_TOPIC_ARN_VAR, "arn:aws:sns:us-east-1:123456789012:test-unwatched-topic"
        )
        with pytest.raises(TopicUnresolvedError, match="test-unwatched-topic"):
            topic_arn()

    def test_the_real_transport_refuses_to_publish_with_no_topic(self, monkeypatch) -> None:
        """Rather than falling through to krepis' default. The whole failure
        mode is that the fallback SUCCEEDS on Telegram, so nothing surfaces."""
        from crucible.alerts import _krepis_publish

        monkeypatch.delenv(PAGES_TOPIC_ARN_VAR, raising=False)
        with pytest.raises(TopicUnresolvedError, match="has no grant"):
            _krepis_publish("body", sns_topic_arn=topic_arn())


class TestCatchUp:
    """C18. The sweep's own downtime was a permanent blind spot."""

    def test_a_day_the_sweep_did_not_run_is_evaluated(self, tmp_path) -> None:
        """Both conditions read only `resolve_trading_day(now)`, so a failed
        manifest written on a day the sweep was down was never paged once the
        trading day advanced past it. Not a delayed page — no page, ever."""
        store = LocalStore(tmp_path)
        for day in (MONDAY, TUESDAY):
            _write_manifest(store, "alerts.sweep", status="ok", trading_day=day)
        _write_manifest(
            store, "data.daily", status="failed", reason="RuntimeError: boom", trading_day=WEDNESDAY
        )
        friday_night = dt.datetime(2026, 8, 28, 23, 30, tzinfo=dt.UTC)
        days = days_to_evaluate(store, friday_night)
        assert WEDNESDAY in days and THURSDAY in days
        pages = evaluate_failure(store, now=friday_night)
        assert [(p.job, p.trading_day) for p in pages] == [("data.daily", WEDNESDAY)]

    def test_a_caught_up_day_does_not_page_an_incident_already_paged(
        self, tmp_path, transport
    ) -> None:
        """Catch-up and dedup are one mechanism, not two that must agree: the
        caught-up day resolves to the same incident key the missed sweep
        would have produced."""
        store = LocalStore(tmp_path)
        for day in (MONDAY, TUESDAY):
            _write_manifest(store, "alerts.sweep", status="ok", trading_day=day)
        _write_manifest(
            store, "data.daily", status="failed", reason="RuntimeError: boom", trading_day=WEDNESDAY
        )
        friday_night = dt.datetime(2026, 8, 28, 23, 30, tzinfo=dt.UTC)
        sweep(store, now=friday_night, transport=transport, sweep_run_id="0" * 26)
        before = transport.pages
        sweep(store, now=friday_night, transport=transport, sweep_run_id="1" * 26)
        assert transport.pages == before

    def test_the_sweep_is_not_blind_for_days_before_it_first_ran(self, tmp_path) -> None:
        """A cold start has no missed days. Claiming a week of them would
        BREACH the one metric §11 risk 2 holds this module to, on the first
        run, for nothing."""
        store = LocalStore(tmp_path)
        assert days_to_evaluate(store, SATURDAY_NIGHT) == [FRIDAY]


class TestSweepIsWatched:
    """C11. `components.yaml` declares `alerts.sweep: absence_watched_by:
    heartbeat`. That was a claim, not a mechanism."""

    def test_the_heartbeat_pages_when_the_sweep_did_not_run(self, tmp_path, transport) -> None:
        """`heartbeat()` never read `alerts.sweep`'s manifest, so the one row
        the sweep cannot honestly watch was watched by nobody while rendering
        as covered — and a sweep that stops running is the failure that hides
        every other failure."""
        store = LocalStore(tmp_path)
        summary = heartbeat(store, now=SATURDAY_NIGHT, transport=transport, run_id="H" * 26)
        assert summary["watched_absences"] == ["alerts.sweep"]
        assert summary["bus_keys"], "§7.3: a human-only alert is invisible"
        row = json.loads(store.get_bytes(summary["bus_keys"][0]))
        assert row["condition"] == "absence"
        assert [m["job"] for m in row["members"]] == ["alerts.sweep"]
        severities = [c.kwargs["severity"] for c in transport.calls]
        assert "error" in severities, "a dead alerting path is not an info-severity footnote"

    def test_a_sweep_that_ran_raises_nothing(self, tmp_path, transport) -> None:
        store = LocalStore(tmp_path)
        _write_manifest(store, "alerts.sweep", status="ok")
        summary = heartbeat(store, now=SATURDAY_NIGHT, transport=transport, run_id="H" * 26)
        assert summary["watched_absences"] == []
        assert summary["bus_keys"] == []
        assert [c.kwargs["severity"] for c in transport.calls] == ["info"]

    def test_the_sweep_still_never_pages_for_itself(self, tmp_path) -> None:
        """The registry refuses a row that watches itself; so does this. A
        sweep that never ran cannot report itself missing."""
        from crucible.alerts import evaluate_absence

        store = LocalStore(tmp_path)
        jobs = {p.job for p in evaluate_absence(store, now=SATURDAY_NIGHT)}
        assert "alerts.sweep" not in jobs
        assert "heartbeat" not in jobs, "its declared watcher is the operator"


class TestCeiling:
    def test_the_ceiling_is_a_metric_with_a_breach_status(self) -> None:
        assert ceiling_metric(0, now=SATURDAY_NIGHT)["status"] == "OK"
        breached = ceiling_metric(PAGES_PER_MONTH_CEILING + 1, now=SATURDAY_NIGHT)
        assert breached["status"] == "BREACH"
        assert "never a reason to add a suppression" in breached["status_reason"]
        assert breached["horizon_trading_days"] == CEILING_WINDOW_TRADING_DAYS

    def test_pages_are_counted_in_groups_not_members(self, tmp_path, transport) -> None:
        """One outage is one page. Counting members would make a single bad
        Saturday read as five incidents against a two-a-month target."""
        store = LocalStore(tmp_path)
        pages = [
            _failed(job, "RuntimeError: data source yfinance is down")
            for job in ("data.daily", "report", "drift")
        ]
        emit(store, group_pages(pages), sweep_run_id="0" * 26, transport=transport)
        assert pages_in_window(store, now=SATURDAY_NIGHT) == 1

    def test_pages_are_counted_in_incidents_not_observations(self, tmp_path, transport) -> None:
        """The other half, and the one that was wrong: the metric counted bus
        rows, and the bus wrote one row per SWEEP. Three nightly sweeps over
        one unchanged failure read as three pages against a ceiling of two."""
        store = LocalStore(tmp_path)
        groups = group_pages([_failed("data.daily", "RuntimeError: yfinance is down")])
        for index in range(3):
            emit(store, groups, sweep_run_id=f"{index}" * 26, transport=transport)
        assert pages_in_window(store, now=SATURDAY_NIGHT) == 1
        assert (
            ceiling_metric(pages_in_window(store, now=SATURDAY_NIGHT), now=SATURDAY_NIGHT)["status"]
            == "OK"
        )


class TestHeartbeat:
    def test_it_reports_the_week_and_goes_out_on_the_pages_channel(
        self, tmp_path, transport
    ) -> None:
        """Same transport as a page ON PURPOSE: a heartbeat carried by a
        healthy second channel proves that channel alive and says nothing
        about the one the pages use."""
        store = LocalStore(tmp_path)
        _write_manifest(store, "alerts.sweep", status="ok")
        _write_manifest(store, "data.daily", status="ok", cost=0.25)
        _write_manifest(store, "report", status="failed", reason="boom")
        summary = heartbeat(store, now=SATURDAY_NIGHT, transport=transport)
        assert summary["runs_ok"] == 2
        assert summary["runs_failed"] == 1
        assert summary["cost_usd"] == 0.25
        assert transport.pages == 1
        assert "alive" in transport.calls[0].message

    def test_an_unreadable_manifest_counts_as_failed_not_as_absent(
        self, tmp_path, transport
    ) -> None:
        """Dropping it would make the heartbeat read healthier the worse
        things got."""
        store = LocalStore(tmp_path)
        store.put_bytes(manifest_key("data.daily", FRIDAY.isoformat()), b"{oops")
        assert heartbeat(store, now=SATURDAY_NIGHT, transport=transport)["runs_failed"] == 1

    def test_discriminated_manifests_are_counted_not_dropped(self, tmp_path, transport) -> None:
        """alpha-engine-config-I9879: `_week_summary` used to filter store
        keys on `len(parts) != 4`, the BARE manifest shape
        (`runs/{job}/{trading_day}/run.json`). A discriminated manifest
        (`runs/{job}/{trading_day}/{discriminator}/run.json`, 5 segments) —
        what `experiment.run`/`experiment.grade` (by slot) and
        `alerts.sweep` (by `calendar_date`) actually write — was silently
        skipped, so the ok/failed counters and spend both undercounted the
        highest-volume writers. Shown RED against pre-fix `main` in the PR
        body; this asserts a bare AND a discriminated manifest for the SAME
        job/trading-day are both counted.
        """
        store = LocalStore(tmp_path)
        _write_manifest(store, "data.daily", status="ok", cost=0.25)
        _write_manifest(store, "experiment.run", status="ok", cost=1.5, discriminator="r")
        _write_manifest(
            store, "experiment.run", status="failed", reason="boom", cost=0.75, discriminator="m"
        )
        ok, failed, spend = _week_summary(store, FRIDAY)
        assert ok == 2
        assert failed == 1
        assert spend == pytest.approx(2.5)


class TestSweep:
    def test_it_evaluates_both_conditions_and_pages_once_per_cause(
        self, tmp_path, transport
    ) -> None:
        store = LocalStore(tmp_path)
        _write_manifest(store, "data.daily", status="failed", reason="RuntimeError: boom")
        result = sweep(store, now=SATURDAY_NIGHT, transport=transport, sweep_run_id="0" * 26)
        # One absence group covering every job with no manifest, plus the
        # data.daily failure.
        assert result["pages_emitted"] == 2
        assert result["members"] > 2
        assert transport.pages == 2

    def test_pages_emitted_counts_what_was_sent_not_what_was_seen(
        self, tmp_path, transport
    ) -> None:
        """A second sweep over the same two open incidents sends nothing, and
        says so — reporting "2 pages emitted" would be the bus row's own
        false claim one layer up."""
        store = LocalStore(tmp_path)
        _write_manifest(store, "data.daily", status="failed", reason="RuntimeError: boom")
        sweep(store, now=SATURDAY_NIGHT, transport=transport, sweep_run_id="0" * 26)
        again = sweep(store, now=SATURDAY_NIGHT, transport=transport, sweep_run_id="1" * 26)
        assert again["pages_emitted"] == 0
        assert again["incidents_open"] == 2
        assert transport.pages == 2


class TestTopicNamesRaiseOnUnset:
    """`alpha-engine-config-I10156`: the topic NAMES are no longer literals,
    so the raise-on-unset path is the only thing standing between a
    forgotten environment variable and a page silently sent nowhere (or
    the muted topic, which is the same failure with a delay)."""

    def test_muted_topic_raises_when_unset(self, monkeypatch) -> None:
        monkeypatch.delenv(MUTED_TOPIC_VAR, raising=False)
        with pytest.raises(RuntimeError, match=MUTED_TOPIC_VAR):
            muted_topic()

    def test_muted_topic_raises_when_empty(self, monkeypatch) -> None:
        monkeypatch.setenv(MUTED_TOPIC_VAR, "")
        with pytest.raises(RuntimeError, match=MUTED_TOPIC_VAR):
            muted_topic()

    def test_pages_topic_raises_when_unset(self, monkeypatch) -> None:
        monkeypatch.delenv(PAGES_TOPIC_VAR, raising=False)
        with pytest.raises(RuntimeError, match=PAGES_TOPIC_VAR):
            pages_topic()

    def test_pages_topic_raises_when_empty(self, monkeypatch) -> None:
        monkeypatch.setenv(PAGES_TOPIC_VAR, "")
        with pytest.raises(RuntimeError, match=PAGES_TOPIC_VAR):
            pages_topic()
