"""The earliest-snapshot sector backfill is a named, flagged source mode.

Brian's ruling (a) on `alpha-engine-config-I10733` (2026-09-14): a session
before the first universe-covering constituents snapshot (2026-05-01) uses the
earliest such snapshot's sector map. The known look-ahead that carries (the
2023-03-20 GICS change moved V, MA, PYPL and others from Information Technology
to Financials) must be DATA on every artifact that rests on it: the source
reading, every feature row, and the grade of any arm scored over such sessions.
"""

from __future__ import annotations

import datetime as dt
import json

import pandas as pd
import pytest
from conftest import SESSIONS, sessions_ending, synthetic_frames

from crucible.config import Settings
from crucible.data import FramePriceSource, run_daily
from crucible.data.point_in_time import (
    EARLIEST_SNAPSHOT_BACKFILL_MODE,
    KNOWN_GICS_RECLASSIFICATIONS,
    POINT_IN_TIME_MODE,
    MappingSnapshotReader,
    SnapshotPointInTimeSource,
    constituents_key,
)
from crucible.features import build_features
from crucible.keys import arena_cycle_key, arm_series_key, coverage_key
from crucible.runner import run_job
from crucible.slots import universe
from crucible.slots.cycle import SECTOR_SOURCE_MODE_FIELD

SECTORS = ("Information Technology", "Financials", "Industrials", "Health Care")
#: The fixture's first universe-covering snapshot, fetched the evening before
#: its label exactly as production's are (2026-05-01).
FIRST_FULL = dt.date(2026, 5, 1)
FIRST_FULL_FETCHED = "2026-04-30T22:10:00+00:00"


def _constituents(tickers: list[str], label: dt.date, fetched_at: str, sector: str | None) -> bytes:
    return json.dumps(
        {
            "date": label.isoformat(),
            "tickers": tickers,
            "sector_map": {t: sector or SECTORS[i % len(SECTORS)] for i, t in enumerate(tickers)},
            "fetched_at": fetched_at,
        }
    ).encode()


def _objects(tickers: list[str]) -> dict[str, bytes]:
    """An S&P-500-only map (half the universe) before the first full one, as in
    production, plus a later full map with a DIFFERENT sector for every name."""
    from test_point_in_time_attractiveness import _fundamentals, _inst, _parquet

    from crucible.keys import fundamental_snapshot_key, inst_ownership_key

    return {
        constituents_key(dt.date(2026, 3, 2)): _constituents(
            tickers[: len(tickers) // 2], dt.date(2026, 3, 2), "2026-03-02T12:00:00+00:00", None
        ),
        constituents_key(FIRST_FULL): _constituents(tickers, FIRST_FULL, FIRST_FULL_FETCHED, None),
        constituents_key(dt.date(2026, 5, 11)): _constituents(
            tickers, dt.date(2026, 5, 11), "2026-05-11T12:00:00+00:00", "Utilities"
        ),
        fundamental_snapshot_key(dt.date(2026, 9, 9)): _parquet(
            _fundamentals(tickers, dt.date(2026, 9, 9), seed=9)
        ),
        inst_ownership_key(2026, 1): _parquet(_inst(tickers, seed=1)),
    }


def _tickers(n: int = 40) -> list[str]:
    return [f"T{i:03d}" for i in range(n)]


def _source(objects: dict[str, bytes]) -> SnapshotPointInTimeSource:
    return SnapshotPointInTimeSource(MappingSnapshotReader(objects), label="fixture")


def _sector(inputs):
    return next(r for r in inputs.readings if r.group == "sector")


class TestSourceMode:
    def test_a_session_before_the_first_full_snapshot_resolves_the_earliest_map_flagged(
        self,
    ) -> None:
        tickers = _tickers()
        day = dt.date(2023, 3, 1)
        inputs = _source(_objects(tickers)).load(trading_day=day, symbols=tickers)
        reading = _sector(inputs)
        assert reading.state == "measured"
        assert reading.source_mode == EARLIEST_SNAPSHOT_BACKFILL_MODE
        assert inputs.sector_source_mode == EARLIEST_SNAPSHOT_BACKFILL_MODE
        # The EARLIEST full map, not the S&P-500-only one and not the later one.
        assert reading.snapshot == constituents_key(FIRST_FULL)
        assert reading.knowledge_date == "2026-04-30"
        assert "Utilities" not in set(inputs.frame["sector_raw"])
        assert inputs.frame["sector_raw"].notna().all()
        # The known look-ahead is data, and it names V, MA and PYPL.
        assert reading.known_look_ahead == KNOWN_GICS_RECLASSIFICATIONS
        named = {t for r in reading.known_look_ahead for t in r.tickers}
        assert {"V", "MA", "PYPL"} <= named
        document = inputs.to_dict()
        assert document["sector_source_mode"] == EARLIEST_SNAPSHOT_BACKFILL_MODE
        sector_doc = next(r for r in document["readings"] if r["group"] == "sector")
        assert sector_doc["source_mode"] == EARLIEST_SNAPSHOT_BACKFILL_MODE
        assert sector_doc["known_look_ahead"][0]["effective_session"] == "2023-03-20"

    def test_a_backfilled_session_after_the_2023_change_names_no_look_ahead(self) -> None:
        tickers = _tickers()
        inputs = _source(_objects(tickers)).load(trading_day=dt.date(2024, 6, 3), symbols=tickers)
        reading = _sector(inputs)
        assert reading.source_mode == EARLIEST_SNAPSHOT_BACKFILL_MODE
        assert reading.known_look_ahead == ()

    def test_a_below_coverage_map_before_the_first_full_one_is_superseded_by_the_backfill(
        self,
    ) -> None:
        """2026-03-10 has an admissible S&P-500-only map; the ruling still applies."""
        tickers = _tickers()
        inputs = _source(_objects(tickers)).load(trading_day=dt.date(2026, 3, 10), symbols=tickers)
        assert _sector(inputs).source_mode == EARLIEST_SNAPSHOT_BACKFILL_MODE
        assert _sector(inputs).state == "measured"

    def test_a_session_after_the_first_full_snapshot_is_point_in_time_and_unflagged(
        self,
    ) -> None:
        tickers = _tickers()
        inputs = _source(_objects(tickers)).load(trading_day=dt.date(2026, 5, 4), symbols=tickers)
        reading = _sector(inputs)
        assert reading.state == "measured"
        assert reading.source_mode == POINT_IN_TIME_MODE
        assert reading.known_look_ahead == ()
        assert reading.snapshot == constituents_key(FIRST_FULL)
        assert inputs.sector_source_mode == POINT_IN_TIME_MODE
        for other in inputs.readings:
            assert other.source_mode == POINT_IN_TIME_MODE

    def test_without_any_full_snapshot_nothing_is_backfilled(self) -> None:
        tickers = _tickers()
        objects = {k: v for k, v in _objects(tickers).items() if "2026-05" not in k}
        inputs = _source(objects).load(trading_day=dt.date(2026, 3, 10), symbols=tickers)
        assert _sector(inputs).state == "below_coverage"
        assert _sector(inputs).source_mode == POINT_IN_TIME_MODE
        assert inputs.sector_source_mode is None


class TestFeatureRows:
    @pytest.fixture(autouse=True)
    def _liquidity_floor(self, monkeypatch):
        from crucible.features.compute import LIQUIDITY_FLOOR_VAR

        monkeypatch.setenv(LIQUIDITY_FLOOR_VAR, "1000000")

    @pytest.mark.parametrize(
        ("end", "expected"), [(dt.date(2026, 4, 1), 1.0), (dt.date(2026, 5, 12), 0.0)]
    )
    def test_every_row_records_the_mode(self, end: dt.date, expected: float) -> None:
        tickers = _tickers()
        frames = synthetic_frames(end=end, sessions=SESSIONS, names=tickers)
        panel = pd.concat([f.reset_index() for f in frames.values()], ignore_index=True)
        inputs = _source(_objects(tickers)).load(trading_day=end, symbols=tickers)
        features, _ = build_features(panel, point_in_time=inputs)
        assert (features["sector_earliest_snapshot_backfill_raw"] == expected).all()


HORIZON = 21
DECISION_DATES = 6


class TestGradeArtifact:
    def test_the_flag_reaches_the_series_and_the_arena_cycle(
        self, store, strategy_dir, cycle_date, benchmark_frames, tmp_path, monkeypatch
    ) -> None:
        """A U slot graded over sessions whose features were compiled under the
        backfill carries the mode on every scored date, on the `scores/`
        document, on the ladder lineage and on the `arena_cycle` artifact."""
        from crucible.features.compute import LIQUIDITY_FLOOR_VAR

        monkeypatch.setenv(LIQUIDITY_FLOOR_VAR, "1000000")
        frames = synthetic_frames(end=cycle_date)
        source = FramePriceSource({**frames, **benchmark_frames}, snapshot="frames:sector-backfill")
        tickers = sorted(frames)
        # The first full map is fetched AFTER the cycle date, so every
        # decision date resolves the backfill.
        late = dt.date(2026, 9, 1)
        objects = _objects(tickers)
        objects = {k: v for k, v in objects.items() if not k.startswith("market_data/")}
        objects[constituents_key(late)] = _constituents(
            tickers, late, "2026-09-01T12:00:00+00:00", None
        )
        point_in_time = _source(objects)

        settings = Settings(
            store_uri=str(tmp_path / "store"),
            arctic_bucket="unused-in-this-test",
            strategy_dir=strategy_dir,
            origins={"store_uri": "test", "strategy_dir": "test"},
        )
        sessions = sessions_ending(cycle_date, HORIZON + DECISION_DATES + 1)
        decision_days = sessions[:DECISION_DATES]
        for day in [*decision_days, cycle_date]:
            run_job(
                "data.daily",
                lambda c: run_daily(
                    c, point_in_time=point_in_time, source=source, expected_symbols=tickers
                ),
                store=store,
                trading_day=day,
            )
        coverage = json.loads(store.get_bytes(coverage_key(decision_days[0].isoformat())))
        assert coverage["point_in_time"]["sector_source_mode"] == EARLIEST_SNAPSHOT_BACKFILL_MODE
        for day in decision_days:
            run_job(
                "experiment.run",
                lambda c: universe.produce(c, settings=settings),
                store=store,
                trading_day=day,
            )
        run_job(
            "experiment.grade",
            lambda c: universe.grade(c, settings=settings),
            store=store,
            trading_day=cycle_date,
        )

        cycle = json.loads(store.get_bytes(arena_cycle_key("u", cycle_date.isoformat())))
        by_arm = cycle[SECTOR_SOURCE_MODE_FIELD]
        real = {a: modes for a, modes in by_arm.items() if modes}
        assert real, "no arm carried a sector source mode, so the flag never reached the grade"
        for arm_id, modes in real.items():
            assert list(modes) == [EARLIEST_SNAPSHOT_BACKFILL_MODE], arm_id
            series = json.loads(store.get_bytes(arm_series_key("u", arm_id)))
            assert series[SECTOR_SOURCE_MODE_FIELD] == modes
            assert sorted(series["scores"]) == modes[EARLIEST_SNAPSHOT_BACKFILL_MODE]
        ladders = {ladder["arm_id"]: ladder for ladder in cycle["ladders"]}
        for arm_id in real:
            assert ladders[arm_id]["lineage"][SECTOR_SOURCE_MODE_FIELD] == [
                EARLIEST_SNAPSHOT_BACKFILL_MODE
            ]
