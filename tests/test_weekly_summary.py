"""`weekly/{day}/summary.json` (Director 2026-10-31, the console weekly-run
page). Negative cases: an unknown instance type gives a null cost with a
reason (never a guessed price), an absent arena cycle is marked absent, an arm
with no shared dates compares to nothing, and a corrupt jsonl line raises."""

from __future__ import annotations

from pathlib import Path

import pytest

from crucible.documents import UnreadableDocumentError, load_store_jsonl
from crucible.keys import (
    arena_cycle_key,
    arm_series_key,
    attribution_key,
    coverage_key,
    experiments_key,
    weekly_summary_key,
)
from crucible.store import LocalStore
from crucible.weekly_summary import (
    COMPARISON_DATES,
    WEEKLY_SUMMARY_SCHEMA,
    build_summary,
    compute_estimate,
)

DAY = "2026-10-30"
CHAMP = "m:champ:aaaaaaaaaaaa"
RIVAL = "m:rival:bbbbbbbbbbbb"
LATE = "m:late:cccccccccccc"
CTRL = "m:control_null_m:dddddddddddd"


def dates(n: int) -> list[str]:
    return [f"2026-09-{d:02d}" for d in range(1, n + 1)]


def docs() -> dict[str, dict]:
    many = dates(25)
    return {
        coverage_key(DAY): {
            "data_snapshot_id": "snap",
            "feature_version": "v1",
            "coverage_ratio": 1.0,
        },
        attribution_key(DAY): {"grade": "B", "grade_reason": "r", "members": []},
        arena_cycle_key("m", DAY): {
            "benchmark": "population",
            "scored_arms": [CHAMP, RIVAL, LATE, CTRL],
            "decision": {
                "champion": CHAMP,
                "status": "held",
                "moved": False,
                "reason": "x",
                "ineligible": {CTRL: [{"name": "not_a_control_arm", "passed": False}]},
            },
        },
        arm_series_key("m", CHAMP): {"scores": dict.fromkeys(many, 0.01)},
        arm_series_key("m", RIVAL): {"scores": dict.fromkeys(many, 0.03)},
        arm_series_key("m", LATE): {"scores": {"2026-10-20": 0.5}},
        arm_series_key("m", CTRL): {"scores": dict.fromkeys(many, 0.0)},
    }


STAGE = (
    "runs/data.weekly/2026-10-30/run.json",
    {
        "status": "ok",
        "started": "2026-10-31T11:03:02Z",
        "finished": "2026-10-31T11:04:35Z",
        "attempts": [{"n": 1}],
        "cost_usd": 0.25,
        "release_sha": "abc",
        "code_sha": "abc",
        "wheelhouse_digest": "d",
        "resource": {"instance_type": "r5a.large", "spot": True},
    },
)


def summary(**kw):
    store = docs()
    return build_summary(
        trading_day=DAY,
        run_id="R",
        arc_started="2026-10-31T11:00:00Z",
        arc_finished="2026-10-31T12:00:00Z",
        stages=[STAGE],
        read=store.get,
        read_events=lambda key: (
            [{"kind": "negative_result"}] if key == experiments_key(DAY) else []
        ),
        generated_utc="2026-10-31T12:00:01Z",
        **kw,
    )


class TestSummary:
    def test_headline_fields(self):
        s = summary()
        assert s["schema_version"] == WEEKLY_SUMMARY_SCHEMA and s["backfilled"] is False
        assert s["arc"]["wall_seconds"] == 3600.0
        assert s["release"] == {"release_sha": "abc", "code_sha": "abc", "wheelhouse_digest": "d"}
        assert s["data"]["data_snapshot_id"] == "snap"
        assert s["stages"][0]["job"] == "data.weekly" and s["stages"][0]["wall_seconds"] == 93.0
        assert s["llm_cost_usd"] == 0.25
        assert s["decisions"] == [{"kind": "negative_result"}]
        assert s["attribution"]["grade"] == "B"
        assert summary(backfilled=True)["backfilled"] is True

    def test_compute_is_estimated_hours_times_declared_price(self):
        c = summary()["compute"]
        assert c["estimated"] is True and c["instance_hours"] == 1.0
        assert c["cost_usd"] == 0.113 and "list price" in c["price_source"]

    def test_every_arm_gets_a_row_beside_the_champion(self):
        m = summary()["slots"]["m"]
        assert m["champion"] == CHAMP and m["challengers_scored"] == 2 and m["controls_scored"] == 1
        rows = {r["arm_id"]: r for r in m["arms"]}
        assert rows[RIVAL]["n_dates"] == COMPARISON_DATES
        assert rows[RIVAL]["diff_vs_champion"] == pytest.approx(0.02)
        assert rows[CTRL]["control"] is True
        assert rows[CHAMP]["is_champion"] and rows[CHAMP]["diff_vs_champion"] == 0.0

    def test_an_arm_with_no_shared_dates_compares_to_nothing(self):
        late = next(r for r in summary()["slots"]["m"]["arms"] if r["arm_id"] == LATE)
        assert late["n_dates"] == 0 and late["diff_vs_champion"] is None

    def test_an_absent_cycle_is_marked_absent(self):
        assert summary()["slots"]["u"] == {"cycle_key": arena_cycle_key("u", DAY), "present": False}


class TestComputeEstimate:
    def test_an_undeclared_type_is_null_with_a_reason(self):
        c = compute_estimate("x9.huge", 3600, False)
        assert c["cost_usd"] is None and "no declared price" in c["reason"]

    def test_unknown_wall_time_is_null_with_a_reason(self):
        c = compute_estimate("r5a.large", None, True)
        assert c["cost_usd"] is None and c["reason"] == "arc wall time unknown"


class TestJsonl:
    def test_reads_rows_and_absent_is_empty(self, tmp_path: Path):
        store = LocalStore(tmp_path)
        assert load_store_jsonl(store, "a/b.jsonl") == []
        store.put_bytes("a/b.jsonl", b'{"x": 1}\n\n{"x": 2}\n')
        assert load_store_jsonl(store, "a/b.jsonl") == [{"x": 1}, {"x": 2}]

    def test_a_corrupt_line_raises_naming_it(self, tmp_path: Path):
        store = LocalStore(tmp_path)
        store.put_bytes("a/b.jsonl", b'{"x": 1}\nnot json\n')
        with pytest.raises(UnreadableDocumentError, match="line 2"):
            load_store_jsonl(store, "a/b.jsonl")


def test_the_key():
    assert weekly_summary_key(DAY) == "weekly/2026-10-30/summary.json"
    with pytest.raises(ValueError):
        weekly_summary_key("2026-10-31")
