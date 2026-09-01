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
    MUTED_TOPIC,
    PAGES_PER_MONTH_CEILING,
    Page,
    PageGroup,
    cause_key,
    ceiling_metric,
    emit,
    evaluate_failure,
    group_pages,
    heartbeat,
    pages_in_window,
    sweep,
)
from crucible.manifest import manifest_key
from crucible.store import LocalStore

FRIDAY = dt.date(2026, 8, 28)
SATURDAY_NIGHT = dt.datetime(2026, 8, 29, 23, 30, tzinfo=dt.UTC)


def _failed(job: str, reason: str, run_id: str = "01JG0000000000000000000001") -> Page:
    return Page(condition="failure", job=job, trading_day=FRIDAY, reason=reason, run_id=run_id)


def _write_manifest(store, job: str, *, status: str, reason: str = "", cost: float = 0.0):
    payload = {
        "schema_version": "run_manifest.v1",
        "run_id": "01JG000000000000000000000" + job[0].upper(),
        "job": job,
        "trading_day": FRIDAY.isoformat(),
        "calendar_date": FRIDAY.isoformat(),
        "status": status,
        "reason": reason,
        "cost_usd": cost,
        "attempts": [{"n": 1, "reason": "initial"}],
    }
    store.put_bytes(manifest_key(job, FRIDAY.isoformat()), json.dumps(payload).encode())
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
        assert keys[0].endswith("01JG00000000000000000000ZZ.json")
        assert json.loads(store.get_bytes(keys[0]))["alert_id_is_run_id"] is True

    def test_an_absence_row_says_its_id_is_not_a_run_id(self, tmp_path, transport) -> None:
        """There was no run, so there is no run id. The field a machine joins
        on is never silently empty."""
        store = LocalStore(tmp_path)
        page = Page(condition="absence", job="data.weekly", trading_day=FRIDAY, reason="late")
        keys = emit(store, group_pages([page]), sweep_run_id="0" * 26, transport=transport)
        assert json.loads(store.get_bytes(keys[0]))["alert_id_is_run_id"] is False

    def test_the_row_records_what_the_transport_actually_did(self, tmp_path, transport) -> None:
        store = LocalStore(tmp_path)
        group = group_pages([_failed("data.daily", "boom")])[0]
        emit(store, [group], sweep_run_id="0" * 26, transport=transport)
        assert transport.pages == 1
        assert transport.calls[0].kwargs["dedup_window_min"] is None


class TestMutedRouting:
    def test_legacy_pages_go_to_the_muted_topic(self, tmp_path, transport) -> None:
        """§11 risk 5: the old system's alerts during the overlap. Muted at
        the destination, never by not emitting — a mute implemented as
        silence is indistinguishable from a dead producer."""
        store = LocalStore(tmp_path)
        group = group_pages([_failed("data.daily", "boom")])[0]
        emit(store, [group], sweep_run_id="0" * 26, legacy=True, transport=transport)
        assert transport.calls[0].kwargs["sns_topic_arn"] == MUTED_TOPIC


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


class TestHeartbeat:
    def test_it_reports_the_week_and_goes_out_on_the_pages_channel(
        self, tmp_path, transport
    ) -> None:
        """Same transport as a page ON PURPOSE: a heartbeat carried by a
        healthy second channel proves that channel alive and says nothing
        about the one the pages use."""
        store = LocalStore(tmp_path)
        _write_manifest(store, "data.daily", status="ok", cost=0.25)
        _write_manifest(store, "report", status="failed", reason="boom")
        summary = heartbeat(store, now=SATURDAY_NIGHT, transport=transport)
        assert summary["runs_ok"] == 1
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
