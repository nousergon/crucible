"""`alpha-engine-config-I13055`: a weekly arc that fails part-way still files
`report/{day}/weekly_summary.json`, marked failed, naming the failed stage and
every declared stage it never reached. Before this, the summary was built only
after `run_arc` returned, so a failed week showed on the console as the
previous week going stale and the failed stage was named nowhere on the page.
Negative cases: an earlier run's `ok` manifest at an unreached key is not
read as a stage that ran, and a summary that cannot be filed never replaces
the stage error the arc must raise."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import pytest

from crucible import track_f
from crucible.cli import main
from crucible.keys import manifest_key, weekly_summary_key
from crucible.store import LocalStore
from crucible.weekly import ArcStageFailed, Stage

DAY = "2026-10-30"
DUE = dt.datetime(2026, 10, 31, 11, 0, tzinfo=dt.UTC)
STAGES = [
    Stage("data.weekly", None, DUE),
    Stage("experiment.run", "r", DUE),
    Stage("report", None, DUE),
]


def key(stage: Stage) -> str:
    return manifest_key(stage.job, DAY, discriminator=stage.slot)


def put(store: LocalStore, k: str, doc: dict[str, Any]) -> None:
    store.put_bytes(k, json.dumps(doc).encode())


@pytest.fixture
def arc(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> LocalStore:
    monkeypatch.setattr(track_f, "arc_stages", lambda _day: list(STAGES))
    monkeypatch.setattr(track_f, "send_notice", lambda *_a, **_k: "")
    return LocalStore(tmp_path)


def run_weekly(store: LocalStore) -> int:
    return main(["weekly", "--store", str(store.root), "--date", DAY])


def test_a_stage_failure_files_a_failed_summary(monkeypatch, arc: LocalStore) -> None:
    put(arc, key(STAGES[0]), {"status": "ok", "started": "2026-10-31T11:00:00Z"})
    put(arc, key(STAGES[1]), {"status": "failed", "started": "2026-10-31T11:10:00Z"})
    # An earlier run's manifest at a stage this run never reached.
    put(arc, key(STAGES[2]), {"status": "ok", "started": "2026-10-24T11:00:00Z"})

    def failing_arc(*_a: Any, **_k: Any) -> list[Stage]:
        raise ArcStageFailed(
            f"weekly arc stage experiment.run[r] exited 1 for {DAY}; stop",
            stage=STAGES[1],
            completed=STAGES[:1],
        )

    monkeypatch.setattr(track_f, "run_arc", failing_arc)
    with pytest.raises(ArcStageFailed):
        run_weekly(arc)
    summary = json.loads(arc.get_bytes(weekly_summary_key(DAY)))
    assert summary["arc"]["status"] == "failed"
    assert summary["arc"]["failed_stage"] == "experiment.run[r]"
    assert "experiment.run[r] exited 1" in summary["arc"]["reason"]
    assert [(r["job"], r["status"]) for r in summary["stages"]] == [
        ("data.weekly", "ok"),
        ("experiment.run", "failed"),
        ("report", "not-run"),
    ]
    arc_manifest = json.loads(arc.get_bytes(manifest_key("weekly", DAY)))
    assert arc_manifest["status"] == "failed"
    assert weekly_summary_key(DAY) in {o["key"] for o in arc_manifest["outputs"]}


def test_a_refusal_before_stage_one_marks_every_stage_not_run(monkeypatch, arc: LocalStore) -> None:
    def refusing_arc(*_a: Any, **_k: Any) -> list[Stage]:
        raise RuntimeError("champion pointer names an arm the release cannot produce")

    monkeypatch.setattr(track_f, "run_arc", refusing_arc)
    with pytest.raises(RuntimeError, match="champion pointer"):
        run_weekly(arc)
    summary = json.loads(arc.get_bytes(weekly_summary_key(DAY)))
    assert summary["arc"]["status"] == "failed" and summary["arc"]["failed_stage"] is None
    assert [r["status"] for r in summary["stages"]] == ["not-run"] * len(STAGES)


def test_a_summary_that_cannot_be_filed_never_masks_the_stage_error(
    monkeypatch, capsys, arc: LocalStore
) -> None:
    def failing_arc(*_a: Any, **_k: Any) -> list[Stage]:
        # `completed` names a stage whose manifest is absent: the read raises.
        raise ArcStageFailed(
            "weekly arc stage report exited 1", stage=STAGES[2], completed=STAGES[:2]
        )

    monkeypatch.setattr(track_f, "run_arc", failing_arc)
    with pytest.raises(ArcStageFailed, match="report exited 1"):
        run_weekly(arc)
    assert "weekly summary NOT filed for failed arc 2026-10-30" in capsys.readouterr().err
    assert not arc.exists(weekly_summary_key(DAY))
