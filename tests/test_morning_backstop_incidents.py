"""report.morning files each exhausted weekly-arc backstop as ONE incident.

`alpha-engine-config-I13712`. The crucible-v2 dispatcher relaunches a failed
weekly arc from its weekday `data-daily` firing at most three times, then
writes `runs/_dispatch/weekly-backstop/{week}/exhausted.json` and stops. The
morning report turns that marker into a tracked incident: titled by week and
failed stage, labelled `incident`, found-or-created by exact title including
CLOSED issues, so a week is filed once and never again after a human closes
it. No test here reaches the network: the tracker's two primitives are stubbed.
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
from typing import Any

import nousergon_lib.gates.tracker as tracker_module
import pytest

from crucible.keys import (
    parse_dispatch_exit_key,
    parse_dispatch_key,
    parse_weekly_backstop_exhausted_key,
    weekly_backstop_exhausted_key,
)
from crucible.morning import (
    INCIDENT_LABEL,
    file_weekly_backstop_incidents,
    weekly_backstop_incident_title,
)
from crucible.store import LocalStore

TODAY = dt.date(2026, 11, 18)
WEEK = "2026-11-13"
REASON = "weekly arc stage experiment.grade[s] raised TurnoverBudgetError: 0.25 > 0.20"


def _marker(store: LocalStore, week: str = WEEK, **fields: Any) -> str:
    key = weekly_backstop_exhausted_key(week)
    body = {
        "schema": "weekly_backstop_exhausted.v1",
        "week": week,
        "attempts": 3,
        "failed_stage": "experiment.grade",
        "reason": REASON,
        **fields,
    }
    store.put_bytes(key, json.dumps(body).encode("utf-8"))
    return key


@pytest.fixture
def tracker(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Any]]:
    """Stub the two primitives `find_or_create_issue` composes, and refuse the
    real HTTP opener outright."""
    calls: dict[str, list[Any]] = {"find": [], "create": [], "existing": []}

    def _find(self: Any, title: str, *, include_closed: bool = False) -> int | None:
        calls["find"].append((title, include_closed))
        return 40 if title in calls["existing"] else None

    def _create(self: Any, *, title: str, body: str, labels: Any = ()) -> int:
        calls["create"].append((title, body, list(labels)))
        return 41

    def _refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a test reached the tracker adapter's real HTTP opener")

    monkeypatch.setattr(tracker_module, "_default_opener", _refuse)
    monkeypatch.setattr(tracker_module.Tracker, "find_issue_by_title", _find)
    monkeypatch.setattr(tracker_module.Tracker, "create_issue", _create)
    return calls


def test_the_marker_key_round_trips_and_no_dispatch_reader_claims_it() -> None:
    key = weekly_backstop_exhausted_key(WEEK)
    assert key == "runs/_dispatch/weekly-backstop/2026-11-13/exhausted.json"
    assert parse_weekly_backstop_exhausted_key(key) == WEEK
    attempt = "runs/_dispatch/weekly-backstop/2026-11-13/attempt-1.json"
    assert parse_weekly_backstop_exhausted_key(attempt) is None
    for k in (key, attempt):
        assert parse_dispatch_key(k) is None
        assert parse_dispatch_exit_key(k) is None
    with pytest.raises(ValueError):
        weekly_backstop_exhausted_key("not-a-date")


def test_an_exhausted_week_is_filed_once_with_the_incident_label(
    tmp_path: pathlib.Path, tracker
) -> None:
    store = LocalStore(tmp_path)
    key = _marker(store)
    assert file_weekly_backstop_incidents(store, trading_day=TODAY) == [(WEEK, 41)]
    title = "[incident] crucible-v2 weekly arc 2026-11-13 failed at experiment.grade"
    assert tracker["find"] == [(title, True)]
    ((created_title, body, labels),) = tracker["create"]
    assert created_title == title
    assert labels == [INCIDENT_LABEL] == ["incident"]
    assert key in body and "TurnoverBudgetError" in body and "3 times" in body


def test_a_week_already_filed_open_or_closed_is_not_filed_again(
    tmp_path: pathlib.Path, tracker
) -> None:
    store = LocalStore(tmp_path)
    _marker(store)
    tracker["existing"].append(weekly_backstop_incident_title(WEEK, "experiment.grade"))
    assert file_weekly_backstop_incidents(store, trading_day=TODAY) == [(WEEK, 40)]
    assert tracker["create"] == []
    assert tracker["find"][0][1] is True, "dedup must count a closed issue as filed"


def test_attempt_markers_and_old_weeks_are_not_incidents(tmp_path: pathlib.Path, tracker) -> None:
    store = LocalStore(tmp_path)
    store.put_bytes("runs/_dispatch/weekly-backstop/2026-11-13/attempt-1.json", b"{}")
    _marker(store, week="2026-09-04")
    assert file_weekly_backstop_incidents(store, trading_day=TODAY) == []
    assert tracker["find"] == [] and tracker["create"] == []


def test_no_marker_means_no_tracker_call_at_all(tmp_path: pathlib.Path, tracker) -> None:
    assert file_weekly_backstop_incidents(LocalStore(tmp_path), trading_day=TODAY) == []
    assert tracker["find"] == []


def test_a_marker_without_a_stage_still_files(tmp_path: pathlib.Path, tracker) -> None:
    store = LocalStore(tmp_path)
    _marker(store, failed_stage=None)
    file_weekly_backstop_incidents(store, trading_day=TODAY)
    assert tracker["create"][0][0].endswith("failed at an unnamed stage")


def test_an_unreadable_marker_raises_rather_than_going_unfiled(
    tmp_path: pathlib.Path, tracker
) -> None:
    store = LocalStore(tmp_path)
    store.put_bytes(weekly_backstop_exhausted_key(WEEK), b"{not json")
    with pytest.raises(RuntimeError, match="could not be read"):
        file_weekly_backstop_incidents(store, trading_day=TODAY)
    assert tracker["create"] == []
