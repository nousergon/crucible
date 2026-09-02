"""`crucible report` — the five-row attribution table (plan §2 row 5 / §4.5).

Every test here asserts against a REAL store written by the real jobs: the
coverage row is reduced from manifests `run_job` wrote, and the slot rows from
verdict artifacts in the shape `experiment.grade` writes them. A reducer
tested against hand-built inputs it will never see in production is a reducer
tested against itself.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json

import pytest

from crucible.champion import ChampionPointer
from crucible.data.daily import COVERAGE_FLOOR_RATIO
from crucible.keys import champion_key, verdict_key
from crucible.manifest import manifest_key
from crucible.report import (
    RANK_IC_N_FLOOR,
    REPORT_WINDOW_TRADING_DAYS,
    ROWS,
    SLOT_WINDOW_TRADING_DAYS,
    attribution_key,
    build_attribution,
)
from crucible.runner import run_job
from crucible.slots.grading import CROSS_SECTION_MIN_NAMES, cross_section_settled_key
from crucible.store import LocalStore
from crucible.track_e import report_handler

DAY = dt.date(2026, 8, 28)
NOW = dt.datetime(2026, 8, 29, 12, 0, tzinfo=dt.UTC)
ARM = "r:momentum_sleeve:ab12cd"
S_ARM = "s:momentum_sleeve:ab12cd"


def _store(tmp_path) -> LocalStore:
    return LocalStore(tmp_path / "store")


def _write_data_day(store: LocalStore, day: dt.date, ratio: float | None) -> None:
    """A real `data.daily` manifest for ``day``, through the real runner."""

    def body(ctx):
        if ratio is None:
            return
        ctx.record_metric(
            {
                "name": "universe_coverage_ratio",
                "module": "crucible.data",
                "metric_type": "coverage",
                "value": ratio,
                "unit": "ratio",
                "n_floor": 1,
                "status": "OK",
                "status_reason": f"{ratio:.3f} of the expected universe compiled",
                "source_path": f"data/{day.isoformat()}/coverage.json",
                "last_updated_utc": "2026-08-28T21:00:00Z",
            }
        )

    run_job("data.daily", body, store=store, trading_day=day, now=NOW)


def _write_champion(store: LocalStore, slot: str, arm_id: str) -> None:
    pointer = ChampionPointer(
        slot=slot,
        arm_id=arm_id,
        as_of=DAY.isoformat(),
        decided_at="2026-08-28T21:00:00Z",
        run_id="01JG0000000000000000000000",
        code_sha="0" * 40,
        promotion_source="evidence",
        manifest_key=manifest_key("promote", DAY.isoformat()),
    )
    store.put_bytes(
        champion_key(slot), json.dumps(pointer.to_dict(), sort_keys=True).encode("utf-8")
    )


def _write_verdicts(
    store: LocalStore, arm_id: str, scores: dict[str, float], horizon: int = 21
) -> None:
    for day, score in scores.items():
        store.put_bytes(
            verdict_key(arm_id, day),
            json.dumps(
                {
                    "schema_version": "verdict.v1",
                    "arm_id": arm_id,
                    "slot": arm_id.split(":")[0],
                    "trading_day": day,
                    "score_ratio": score,
                    "horizon_trading_days": horizon,
                    "benchmark": "population",
                    "control": False,
                    "detail": {},
                },
                sort_keys=True,
            ).encode("utf-8"),
        )


#: A perfectly rank-correlated cross-section (score and realized return agree
#: on every ordering), sized above :data:`CROSS_SECTION_MIN_NAMES` so a date
#: built from it always contributes an IC of ``+1.0``.
_PERFECT_RANK_SECTION: tuple[tuple[str, float, float], ...] = (
    ("AAA", 6.0, 0.060),
    ("BBB", 5.0, 0.050),
    ("CCC", 4.0, 0.040),
    ("DDD", 3.0, 0.030),
    ("EEE", 2.0, 0.020),
    ("FFF", 1.0, 0.010),
)


def _write_cross_sections_settled(
    store: LocalStore,
    arm_id: str,
    days: list[str],
    *,
    ranks: tuple[tuple[str, float, float | None], ...] = _PERFECT_RANK_SECTION,
    horizon: int = 21,
) -> None:
    """One `cross_section_settled.json` per day — the shape `experiment.grade`
    writes via `crucible.slots.grading.settle_cross_section`."""
    for day in days:
        document = {
            "schema_version": "cross_section_settled.v2",
            "arm_id": arm_id,
            "trading_day": day,
            "horizon_trading_days": horizon,
            "settled_on": day,
            "population_size": len(ranks),
            "n_settled": sum(1 for _, _, r in ranks if r is not None),
            "ranks": [
                {"ticker": t, "score": s, "rank": i + 1, "realized_forward_return_ratio": r}
                for i, (t, s, r) in enumerate(ranks)
            ],
        }
        store.put_bytes(
            cross_section_settled_key(arm_id, day),
            json.dumps(document, sort_keys=True).encode("utf-8"),
        )


class TestShape:
    def test_the_table_always_has_the_five_declared_rows_in_order(self, tmp_path) -> None:
        document, _ = build_attribution(_store(tmp_path), trading_day=DAY, now=NOW, run_id="R" * 26)
        assert [r["name"] for r in document["rows"]] == [s.name for s in ROWS]
        assert document["schema_version"] == "attribution.v1"

    def test_plan_row_is_gone(self, tmp_path) -> None:
        """alpha-engine-config-I9778: the indirection existed only because the
        R/M rows disagreed with §4.5's wording. They no longer do — the row
        name IS the wording now — so there is nothing left for the field to
        carry."""
        document, _ = build_attribution(_store(tmp_path), trading_day=DAY, now=NOW, run_id="R" * 26)
        for row in document["rows"]:
            assert "plan_row" not in row

    def test_every_row_carries_value_ci_n_baseline_and_status(self, tmp_path) -> None:
        document, _ = build_attribution(_store(tmp_path), trading_day=DAY, now=NOW, run_id="R" * 26)
        for row in document["rows"]:
            for field in ("value", "ci_low", "ci_high", "n_samples", "baseline", "status"):
                assert field in row, f"{row['name']} carries no {field}"
            assert row["status_reason"] and len(row["status_reason"]) > 40

    def test_an_empty_store_grades_nothing_green_and_names_what_is_missing(self, tmp_path) -> None:
        """Principle 7: *no data* is never rendered as green."""
        document, sources = build_attribution(
            _store(tmp_path), trading_day=DAY, now=NOW, run_id="R" * 26
        )
        assert sources == []
        for row in document["rows"]:
            assert row["status"].startswith("N/A"), row
            assert row["value"] is None
            assert row["unit"] is None, "a row with no value declares no unit"
        reasons = {r["name"]: r["status_reason"] for r in document["rows"]}
        assert "data.daily" in reasons["data_coverage_ratio"]
        assert "champions/r/current.json" in reasons["signal_rank_ic_r"]
        assert "trader" in reasons["execution_shortfall_bps"]

    def test_the_execution_row_is_not_implemented_rather_than_absent(self, tmp_path) -> None:
        document, _ = build_attribution(_store(tmp_path), trading_day=DAY, now=NOW, run_id="R" * 26)
        execution = document["rows"][-1]
        assert execution["name"] == "execution_shortfall_bps"
        assert execution["status"] == "N/A-NOT-IMPL"
        assert execution["unit"] is None


SESSIONS = [
    dt.date(2026, 8, 24),
    dt.date(2026, 8, 25),
    dt.date(2026, 8, 26),
    dt.date(2026, 8, 27),
    DAY,
]


class TestCoverageRow:
    def test_it_reduces_the_manifests_the_data_job_wrote(self, tmp_path) -> None:
        store = _store(tmp_path)
        for day, ratio in zip(SESSIONS, (0.98, 0.97, 0.99, 0.98, 0.96), strict=True):
            _write_data_day(store, day, ratio)
        document, sources = build_attribution(store, trading_day=DAY, now=NOW, run_id="R" * 26)
        row = document["rows"][0]
        assert row["value"] == pytest.approx(0.976)
        assert row["unit"] == "ratio"
        assert row["n_samples"] == 5
        assert row["baseline"] == COVERAGE_FLOOR_RATIO
        assert row["status"] == "GREEN"
        assert row["ci_low"] is not None and row["ci_high"] is not None
        assert manifest_key("data.daily", DAY.isoformat()) in sources

    def test_a_session_with_no_manifest_is_ABSENT_not_an_observation_of_zero(
        self, tmp_path
    ) -> None:
        """The E4 reproduction, as a test.

        The shape this replaces appended 0.0 for each session with no
        `data.daily` manifest and then reported ``n_samples = 5`` against
        ``n_floor = 5``: one real observation cleared its own floor and
        carried a `bootstrap-percentile-2000` interval over four imputed
        zeros. Every assertion below is false under that shape — it produced
        ``value 0.196``, ``n_samples 5``, ``ci [0.0, 0.588]``, ``RED``.

        The value is now the mean over the session that ran, and it is the
        *evidence* that the absences degrade: n against the full window's
        floor, which is what keeps the row out of GREEN.
        """
        store = _store(tmp_path)
        _write_data_day(store, DAY, 1.0)
        document, _ = build_attribution(store, trading_day=DAY, now=NOW, run_id="R" * 26)
        row = document["rows"][0]
        assert row["value"] == pytest.approx(1.0)
        assert row["n_samples"] == 1, "four sessions that never ran are not four samples"
        assert row["n_floor"] == 5
        assert row["ci_low"] is None and row["ci_high"] is None
        assert row["ci_method"] is None, "no interval is claimed over data that does not exist"
        assert row["status"] == "N/A-LOW-N"
        assert "2026-08-24" in row["status_reason"]
        assert "ABSENT" in row["status_reason"]

    def test_a_partly_covered_window_can_never_read_green(self, tmp_path) -> None:
        """The escape the zero-fill was defending against, closed by n.

        Three perfect sessions of five is a perfect mean — and WATCH, because
        n sits below the window's floor. The missing days cannot be made to
        flatter the row.
        """
        store = _store(tmp_path)
        for day in SESSIONS[:3]:
            _write_data_day(store, day, 1.0)
        document, _ = build_attribution(store, trading_day=DAY, now=NOW, run_id="R" * 26)
        row = document["rows"][0]
        assert row["value"] == pytest.approx(1.0)
        assert row["n_samples"] == 3 and row["n_floor"] == 5
        assert row["status"] == "WATCH"

    def test_a_failed_data_run_covers_nothing_and_IS_a_sample(self, tmp_path) -> None:
        """A layer that ran and covered nothing is a measurement of zero;
        only a layer that never ran is an absence."""
        store = _store(tmp_path)

        def boom(ctx):
            raise RuntimeError("the price source withheld the session")

        with pytest.raises(RuntimeError):
            run_job(
                "data.daily", boom, store=store, trading_day=DAY, now=NOW, transient_retry=False
            )
        document, _ = build_attribution(store, trading_day=DAY, now=NOW, run_id="R" * 26)
        row = document["rows"][0]
        assert row["value"] == pytest.approx(0.0)
        assert row["n_samples"] == 1


class TestPortfolioAlphaRow:
    """S: unchanged by I9778 — still `_slot_row`'s realized excess return,
    reduced from `verdict.json`, because portfolio alpha is a claim about
    what was actually earned, not a ranking to correlate."""

    def test_the_champions_settled_verdicts_become_the_row(self, tmp_path) -> None:
        store = _store(tmp_path)
        _write_champion(store, "s", S_ARM)
        _write_verdicts(
            store,
            S_ARM,
            {
                "2026-07-01": 0.02,
                "2026-07-02": 0.01,
                "2026-07-06": 0.03,
                "2026-07-07": 0.015,
                "2026-07-08": 0.025,
                "2026-07-09": 0.02,
            },
        )
        document, sources = build_attribution(store, trading_day=DAY, now=NOW, run_id="R" * 26)
        row = next(r for r in document["rows"] if r["name"] == "portfolio_excess_return_s_ratio")
        assert row["value"] == pytest.approx(0.02, abs=1e-9)
        assert row["n_samples"] == 6
        assert row["baseline"] == 0.0
        assert row["horizon_trading_days"] == 21
        assert row["status"] == "GREEN"
        assert row["ci_low"] > 0.0
        assert verdict_key(S_ARM, "2026-07-01") in sources
        assert champion_key("s") in sources

    def test_too_few_settled_dates_is_low_n_not_a_number(self, tmp_path) -> None:
        store = _store(tmp_path)
        _write_champion(store, "s", S_ARM)
        _write_verdicts(store, S_ARM, {"2026-07-01": 0.9})
        document, _ = build_attribution(store, trading_day=DAY, now=NOW, run_id="R" * 26)
        row = next(r for r in document["rows"] if r["name"] == "portfolio_excess_return_s_ratio")
        assert row["status"] == "N/A-LOW-N"

    def test_a_champion_with_no_settled_verdict_is_missing_input(self, tmp_path) -> None:
        store = _store(tmp_path)
        _write_champion(store, "s", S_ARM)
        document, _ = build_attribution(store, trading_day=DAY, now=NOW, run_id="R" * 26)
        row = next(r for r in document["rows"] if r["name"] == "portfolio_excess_return_s_ratio")
        assert row["status"] == "N/A-MISSING-INPUT"

    def test_verdicts_at_two_horizons_are_refused_rather_than_averaged(self, tmp_path) -> None:
        store = _store(tmp_path)
        _write_champion(store, "s", S_ARM)
        _write_verdicts(store, S_ARM, {"2026-07-01": 0.02}, horizon=21)
        _write_verdicts(store, S_ARM, {"2026-07-02": 0.02}, horizon=63)
        with pytest.raises(ValueError, match="horizons"):
            build_attribution(store, trading_day=DAY, now=NOW, run_id="R" * 26)


class TestRankICRow:
    """R and M: reduced from `shadow.v2` settled cross-sections into a true
    Spearman rank IC (alpha-engine-config-I9778), not the realized excess
    return `signal_excess_return_r_ratio` used to carry under a `plan_row`
    admitting the mismatch."""

    def test_the_champions_settled_cross_sections_become_the_row(self, tmp_path) -> None:
        store = _store(tmp_path)
        _write_champion(store, "r", ARM)
        days = [
            "2026-07-01",
            "2026-07-02",
            "2026-07-06",
            "2026-07-07",
            "2026-07-08",
            "2026-07-09",
        ]
        _write_cross_sections_settled(store, ARM, days)
        document, sources = build_attribution(store, trading_day=DAY, now=NOW, run_id="R" * 26)
        row = next(r for r in document["rows"] if r["name"] == "signal_rank_ic_r")
        assert row["value"] == pytest.approx(1.0, abs=1e-9)
        assert row["n_samples"] == len(days)
        assert row["n_floor"] == RANK_IC_N_FLOOR
        assert row["baseline"] == 0.0
        assert row["horizon_trading_days"] == 21
        assert row["unit"] == "ic"
        assert row["status"] == "GREEN"
        assert cross_section_settled_key(ARM, days[0]) in sources
        assert champion_key("r") in sources

    def test_a_cross_section_after_the_report_day_is_not_read(self, tmp_path) -> None:
        store = _store(tmp_path)
        _write_champion(store, "r", ARM)
        _write_cross_sections_settled(store, ARM, ["2026-07-01", "2026-09-01"])
        document, _ = build_attribution(store, trading_day=DAY, now=NOW, run_id="R" * 26)
        row = next(r for r in document["rows"] if r["name"] == "signal_rank_ic_r")
        assert row["n_samples"] == 1

    def test_too_few_settled_dates_is_low_n_not_a_number(self, tmp_path) -> None:
        store = _store(tmp_path)
        _write_champion(store, "r", ARM)
        _write_cross_sections_settled(store, ARM, ["2026-07-01"])
        document, _ = build_attribution(store, trading_day=DAY, now=NOW, run_id="R" * 26)
        row = next(r for r in document["rows"] if r["name"] == "signal_rank_ic_r")
        assert row["status"] == "N/A-LOW-N"

    def test_a_champion_with_no_settled_cross_section_is_missing_input(self, tmp_path) -> None:
        store = _store(tmp_path)
        _write_champion(store, "r", ARM)
        document, _ = build_attribution(store, trading_day=DAY, now=NOW, run_id="R" * 26)
        row = next(r for r in document["rows"] if r["name"] == "signal_rank_ic_r")
        assert row["status"] == "N/A-MISSING-INPUT"
        assert "shadow.v2" in row["status_reason"]

    def test_a_date_below_the_names_floor_is_skipped_not_counted(self, tmp_path) -> None:
        """§4.12/`CROSS_SECTION_MIN_NAMES`: a cross-section with too few
        paired names contributes NO observation — it does not get counted as
        a zero, and it does not get counted as a date at all."""
        store = _store(tmp_path)
        _write_champion(store, "r", ARM)
        thin = tuple(_PERFECT_RANK_SECTION[: CROSS_SECTION_MIN_NAMES - 1])
        _write_cross_sections_settled(store, ARM, ["2026-07-01"], ranks=thin)
        document, _ = build_attribution(store, trading_day=DAY, now=NOW, run_id="R" * 26)
        row = next(r for r in document["rows"] if r["name"] == "signal_rank_ic_r")
        assert row["n_samples"] == 0
        assert row["status"].startswith("N/A")
        assert str(CROSS_SECTION_MIN_NAMES) in row["status_reason"]

    def test_a_cross_section_outside_the_declared_window_is_not_read(self, tmp_path) -> None:
        store = _store(tmp_path)
        _write_champion(store, "r", ARM)
        inside = ["2026-06-08", "2026-07-06", "2026-08-28"]
        outside = ["2026-01-05", "2026-02-05", "2026-03-05", "2026-04-06", "2026-05-05"]
        _write_cross_sections_settled(store, ARM, inside + outside)
        document, sources = build_attribution(store, trading_day=DAY, now=NOW, run_id="R" * 26)
        row = next(r for r in document["rows"] if r["name"] == "signal_rank_ic_r")
        assert row["n_samples"] == len(inside)
        assert cross_section_settled_key(ARM, "2026-01-05") not in sources
        assert cross_section_settled_key(ARM, "2026-08-28") in sources

    def test_every_row_declares_the_window_it_was_reduced_over(self, tmp_path) -> None:
        store = _store(tmp_path)
        _write_champion(store, "r", ARM)
        _write_cross_sections_settled(
            store, ARM, ["2026-07-01", "2026-07-02", "2026-07-06", "2026-07-07", "2026-07-08"]
        )
        document, _ = build_attribution(store, trading_day=DAY, now=NOW, run_id="R" * 26)
        assert document["window_scope"] == "per-row"
        by_name = {r["name"]: r for r in document["rows"]}

        coverage = by_name["data_coverage_ratio"]
        assert coverage["window_trading_days"] == REPORT_WINDOW_TRADING_DAYS
        assert coverage["window_start"] == document["window_sessions"][0]
        assert coverage["window_end"] == document["window_sessions"][-1]

        for name in (
            "signal_rank_ic_r",
            "prediction_rank_ic_m",
            "portfolio_excess_return_s_ratio",
        ):
            row = by_name[name]
            assert row["window_trading_days"] == SLOT_WINDOW_TRADING_DAYS
            assert row["window_end"] == DAY.isoformat()
            assert row["window_start"] < document["window_sessions"][0], (
                "a slot row graded over the header's five sessions could never reach its "
                "own n_floor, so it declares a longer window rather than a false one"
            )

        assert (
            by_name["signal_rank_ic_r"]["window_start"]
            in (by_name["signal_rank_ic_r"]["status_reason"])
        ), "the row that HAS a champion names its window in words too"

        execution = by_name["execution_shortfall_bps"]
        assert execution["window_trading_days"] is None, "an unimplemented row grades no window"

    def test_identical_ics_do_not_buy_a_zero_width_interval(self, tmp_path) -> None:
        store = _store(tmp_path)
        _write_champion(store, "r", ARM)
        days = [
            "2026-07-01",
            "2026-07-02",
            "2026-07-06",
            "2026-07-07",
            "2026-07-08",
            "2026-07-09",
        ]
        _write_cross_sections_settled(store, ARM, days)
        document, _ = build_attribution(store, trading_day=DAY, now=NOW, run_id="R" * 26)
        row = next(r for r in document["rows"] if r["name"] == "signal_rank_ic_r")
        assert row["n_samples"] == len(days)
        assert row["ci_low"] is None and row["ci_high"] is None
        assert row["ci_method"] is None, (
            "a zero-width interval is a bootstrap with nothing to resample, not a method"
        )
        assert "identical observations" in row["status_reason"]

    def test_cross_sections_at_two_horizons_are_refused_rather_than_averaged(
        self, tmp_path
    ) -> None:
        store = _store(tmp_path)
        _write_champion(store, "r", ARM)
        _write_cross_sections_settled(store, ARM, ["2026-07-01"], horizon=21)
        _write_cross_sections_settled(store, ARM, ["2026-07-02"], horizon=63)
        with pytest.raises(ValueError, match="horizons"):
            build_attribution(store, trading_day=DAY, now=NOW, run_id="R" * 26)

    def test_a_constant_ranker_date_does_not_clear_the_floor_it_carries_no_information(
        self, tmp_path
    ) -> None:
        """`alpha-engine-config-I9778` review, finding 1, reproduced and closed.

        Five genuine dates alone sit below `RANK_IC_N_FLOOR` (6) and read
        `WATCH`. Before the fix, adding a SIXTH date whose ranker emitted a
        constant score — carrying zero information — was scored as a real
        `ic=0.0` observation, pushed `n_samples` to 6, and cleared the floor
        into `GREEN`. That is principle 7 inverted: a date with no data was
        the thing that turned the row green. After the fix the degenerate
        date contributes neither a value nor a count: `n_samples` stays 5
        and the row stays out of GREEN.
        """
        store = _store(tmp_path)
        _write_champion(store, "r", ARM)
        genuine_days = [
            "2026-07-01",
            "2026-07-02",
            "2026-07-06",
            "2026-07-07",
            "2026-07-08",
        ]
        _write_cross_sections_settled(store, ARM, genuine_days)

        # Five genuine dates alone: below the floor, WATCH.
        document, _ = build_attribution(store, trading_day=DAY, now=NOW, run_id="R" * 26)
        row = next(r for r in document["rows"] if r["name"] == "signal_rank_ic_r")
        assert row["n_samples"] == 5
        assert row["status"] == "WATCH"

        # Add a sixth date with a constant ranker score across every name —
        # the "hard-zeroed feature column" shape the review demonstrated.
        constant_section = tuple(
            (ticker, 1.0, realized) for ticker, _score, realized in _PERFECT_RANK_SECTION
        )
        _write_cross_sections_settled(store, ARM, ["2026-07-09"], ranks=constant_section)
        document, _ = build_attribution(store, trading_day=DAY, now=NOW, run_id="R" * 26)
        row = next(r for r in document["rows"] if r["name"] == "signal_rank_ic_r")
        assert row["n_samples"] == 5, "the degenerate date must not increment n"
        assert row["status"] != "GREEN", "no data must never render as GREEN"
        assert row["status"] == "WATCH"
        assert (
            "degenerate" in row["status_reason"]
            or "undefined correlation" in (row["status_reason"])
        )


class TestTheJob:
    def _run(self, tmp_path) -> tuple[LocalStore, dict]:
        store = _store(tmp_path)
        _write_data_day(store, DAY, 0.99)
        args = argparse.Namespace(
            job="report", date=None, trading_day=DAY, dry_run=False, store=str(store.root)
        )
        assert report_handler(args) == 0
        manifest = json.loads(
            store.get_bytes(manifest_key("report", DAY.isoformat())).decode("utf-8")
        )
        return store, manifest

    def test_it_writes_the_artifact_and_records_it_as_an_output(self, tmp_path) -> None:
        store, manifest = self._run(tmp_path)
        key = attribution_key(DAY.isoformat())
        assert store.exists(key)
        assert manifest["status"] == "ok"
        assert any(o["key"] == key for o in manifest["outputs"])
        assert any(i["key"].startswith("runs/data.daily/") for i in manifest["inputs"])

    def test_every_row_and_the_completeness_count_reach_the_manifest(self, tmp_path) -> None:
        _, manifest = self._run(tmp_path)
        names = [m["name"] for m in manifest["metrics"]]
        for spec in ROWS:
            assert spec.name in names
        complete = next(m for m in manifest["metrics"] if m["name"] == "attribution_rows_complete")
        assert complete["value"] == float(len(ROWS))
        assert complete["status"] == "OK"

    def test_the_run_publishes_its_llm_spend_against_the_declared_cap(self, tmp_path) -> None:
        """§2 row 3: a cap nobody publishes a figure against is a cap nobody
        is held to, and zero spend is a measurement, not a silence."""
        _, manifest = self._run(tmp_path)
        spend = next(m for m in manifest["metrics"] if m["name"] == "llm_spend_usd")
        assert spend["value"] == 0.0
        assert spend["unit"] == "usd"
        assert spend["baseline"] == 5.00
        assert spend["status"] == "OK"
        # `alpha-engine-config-I9823`: the manifest is where `Settings
        # .llm_cap_usd_measured` now reaches a consumer — it was previously
        # written only to `Settings.to_dict()`, which nothing in this
        # package read.
        assert spend["cap_usd_measured"] is False
        pace = next(m for m in manifest["metrics"] if m["name"] == "llm_spend_pace_overrun_ratio")
        assert pace["value"] <= 0.0

    def test_the_console_renders_the_table_it_wrote(self, tmp_path) -> None:
        """The one-line hook already existed: `build_page` reads
        `report/{trading_day}/attribution.json`. Asserted rather than assumed —
        a producer and a consumer agreeing about a key is the M0 contract."""
        from crucible.console.render import build_page, render_html

        store, _ = self._run(tmp_path)
        page = build_page(store, now=dt.datetime(2026, 8, 28, 22, 0, tzinfo=dt.UTC))
        assert [r["name"] for r in page.attribution] == [s.name for s in ROWS]
        assert "data_coverage_ratio" in render_html(page)
