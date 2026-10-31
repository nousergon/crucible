"""The weekly arc's completion notice (Brian 2026-10-31: no run alert for a
clean v2 weekly). The negative cases: a failed arc says so with its reason,
issues are listed and capped, and a Telegram outage never fails the arc."""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from crucible import track_f
from crucible.store import LocalStore
from crucible.weekly_notice import (
    MAX_ISSUE_LINES,
    NOTICE_SOURCE,
    issues,
    render_notice,
    send_notice,
)

DAY = "2026-10-30"


@dataclass
class Result:
    any_ok: bool = True
    telegram_destination: str = "operator_chat"


class Transport:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, message: str, **kwargs: Any) -> Result:
        self.calls.append((message, kwargs))
        return Result()


def arc(status: str = "ok", reason: str = "", inputs: list[str] | None = None) -> dict[str, Any]:
    return {
        "status": status,
        "reason": reason,
        "started": "2026-10-31T11:02:59Z",
        "finished": "2026-10-31T12:01:30Z",
        "release_sha": "5ca466173f4c371aaef2936a6ae2b72ac800e122",
        "run_mode": "live",
        "inputs": [{"key": k} for k in (inputs or [])],
    }


def stage(status: str = "ok", metrics: list[tuple[str, str]] = (), attempts: int = 1) -> dict:
    return {
        "status": status,
        "attempts": [{"n": i + 1} for i in range(attempts)],
        "metrics": [{"name": n, "status": s} for n, s in metrics],
    }


GRADE_M = "runs/experiment.grade/2026-10-30/m/run.json"
DRIFT = "runs/drift/2026-10-30/run.json"
PROMOTE = "runs/promote/2026-10-30/u/run.json"


class TestIssues:
    def test_clean_statuses_are_not_issues(self):
        assert issues([(PROMOTE, stage(metrics=[("pointer_moved", "held"), ("x", "OK")]))]) == []

    def test_non_clean_metrics_are_listed_by_stage_and_counted(self):
        doc = stage(metrics=[("ic", "unmeasurable"), ("ic", "unmeasurable"), ("z", "FAIL")])
        assert issues([(GRADE_M, doc)]) == [
            "experiment.grade m: ic unmeasurable x2",
            "experiment.grade m: z FAIL",
        ]

    def test_a_failed_or_retried_stage_is_listed(self):
        assert issues([(DRIFT, stage(status="failed", attempts=2))]) == [
            "drift: status failed",
            "drift: needed 2 attempts",
        ]


class TestRender:
    def test_a_clean_week_says_ok_and_no_issues(self):
        text = render_notice(DAY, arc(), [(PROMOTE, stage())], planned=1)
        assert "Crucible v2 weekly 2026-10-30: OK" in text
        assert "11:02-12:01Z (59 min), 1/1 stages, release 5ca46617, live" in text
        assert "No stage issues." in text

    def test_a_failed_arc_names_its_reason(self):
        text = render_notice(DAY, arc("failed", "stage drift failed: <boom>"), [], planned=22)
        assert ": FAILED" in text
        assert "Reason: stage drift failed: &lt;boom&gt;" in text
        assert "0/22 stages" in text

    def test_issue_lines_are_capped(self):
        doc = stage(metrics=[(f"m{i}", "RED") for i in range(MAX_ISSUE_LINES + 3)])
        text = render_notice(DAY, arc(), [(DRIFT, doc)], planned=1)
        assert f"Issues ({MAX_ISSUE_LINES + 3})" in text
        assert "and 3 more in the stage manifests" in text

    def test_unparseable_times_do_not_raise(self):
        assert "duration unknown" in render_notice(DAY, {"status": "ok"}, [], planned=0)


def put(store: LocalStore, key: str, doc: dict) -> None:
    store.put_bytes(key, json.dumps(doc).encode())


class TestSend:
    def test_reads_the_arc_and_its_stages_and_delivers_info_on_telegram(self, tmp_path: Path):
        store = LocalStore(tmp_path)
        put(store, "runs/weekly/2026-10-30/run.json", arc(inputs=[DRIFT]))
        put(store, DRIFT, stage(metrics=[("feature_psi_max_ratio", "BREACH")]))
        transport = Transport()
        send_notice(store, DAY, 1, transport=transport)
        ((message, kwargs),) = transport.calls
        assert "drift: feature_psi_max_ratio BREACH" in message
        assert kwargs["severity"] == "info" and kwargs["source"] == NOTICE_SOURCE
        assert kwargs["sns"] is False and kwargs["telegram"] is True

    def test_a_missing_arc_manifest_raises(self, tmp_path: Path):
        with pytest.raises(Exception):  # noqa: B017 - the store's own not-found error
            send_notice(LocalStore(tmp_path), DAY, 1, transport=Transport())


def test_an_undelivered_notice_never_fails_the_arc(monkeypatch, capsys, tmp_path: Path):
    def boom(*_a: Any, **_k: Any) -> str:
        raise RuntimeError("telegram down")

    monkeypatch.setattr(track_f, "send_notice", boom)
    track_f._send_weekly_notice(LocalStore(tmp_path), dt.date(2026, 10, 30))
    assert "weekly notice NOT delivered: RuntimeError: telegram down" in capsys.readouterr().err
