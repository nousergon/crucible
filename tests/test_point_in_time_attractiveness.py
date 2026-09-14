"""Point-in-time fundamentals, sector and 13F inputs, and the attractiveness pillars.

`alpha-engine-config-I10721`. Two halves, both tested against what they must
REFUSE as well as what they compute:

* `crucible.data.point_in_time` admits a snapshot only for sessions strictly
  after its knowledge date, nulls a group that is stale or thin and a field
  that is non-null but uninformative, and refuses an empty source outright.
* `crucible.features.compute` turns those inputs into v1's within-sector
  percentile pillars. Every pillar value is checked against an independent
  calculation written here with plain loops (average-tie ranks counted by hand,
  no pandas `rank`), and the attractiveness rankers of `crucible-PR281` are run
  on the resulting frame to prove the column names line up.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import math
import random

import pandas as pd
import pytest

from crucible.data import FramePriceSource, MissingSourceError, run_daily
from crucible.data.point_in_time import (
    FUNDAMENTAL_FIELD_COLUMNS,
    MappingSnapshotReader,
    SnapshotPointInTimeSource,
    UnavailablePointInTimeSource,
    constituents_key,
    distinctness_floor,
    fundamental_snapshot_key,
    inst_ownership_key,
    quarter_knowledge_date,
)
from crucible.features import build_features, catalog_column_depths, feature_names
from crucible.features.compute import PILLAR_COMPONENTS
from crucible.features.depth import check_feature_layer_completeness
from crucible.features.registry import feature_version
from crucible.runner import run_job
from crucible.slots.rankers import (
    ATTRACTIVENESS_PILLAR_COLUMNS,
    MOMENTUM_12_1_PILLAR_COLUMN,
    DegenerateFeatureError,
    get_ranker,
    rank_with,
)
from crucible.store import LocalStore
from tests.conftest import SESSIONS, synthetic_frames

END = dt.date(2026, 9, 11)
SECTORS = ("Energy", "Health Care", "Information Technology")
ATTRACTIVENESS_RANKERS = (
    "attractiveness_blend",
    "attractiveness_momzero_blend",
    "attractiveness_mom121_blend",
    "attractiveness_hard3_blend",
)


@pytest.fixture(autouse=True)
def _liquidity_floor(monkeypatch):
    from crucible.features.compute import LIQUIDITY_FLOOR_VAR

    monkeypatch.setenv(LIQUIDITY_FLOOR_VAR, "1000000")


# -- fixtures -----------------------------------------------------------------


def _parquet(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


def _fundamentals(tickers: list[str], label: dt.date, *, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for ticker in tickers:
        row = {"ticker": ticker, "date": label.isoformat()}
        for v1_field in FUNDAMENTAL_FIELD_COLUMNS:
            row[v1_field] = round(rng.uniform(-0.5, 1.5), 6)
        rows.append(row)
    return pd.DataFrame(rows)


def _constituents(tickers: list[str], label: dt.date, fetched_at: str) -> bytes:
    return json.dumps(
        {
            "date": label.isoformat(),
            "tickers": tickers,
            "sector_map": {t: SECTORS[i % len(SECTORS)] for i, t in enumerate(tickers)},
            "fetched_at": fetched_at,
        }
    ).encode()


def _inst(tickers: list[str], *, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    return pd.DataFrame(
        {
            "ticker": tickers,
            "quarter": ["2026Q1"] * len(tickers),
            "n_funds_increasing": [float(rng.randint(0, 900)) for _ in tickers],
            "n_funds_decreasing": [float(rng.randint(0, 900)) for _ in tickers],
        }
    )


def _objects(tickers: list[str]) -> dict[str, bytes]:
    """A snapshot store shaped like the production data bucket, fully measured on END."""
    return {
        fundamental_snapshot_key(dt.date(2026, 9, 9)): _parquet(
            _fundamentals(tickers, dt.date(2026, 9, 9), seed=9)
        ),
        fundamental_snapshot_key(dt.date(2026, 9, 10)): _parquet(
            _fundamentals(tickers, dt.date(2026, 9, 10), seed=10)
        ),
        constituents_key(dt.date(2026, 9, 9)): _constituents(
            tickers, dt.date(2026, 9, 9), "2026-09-10T12:18:52+00:00"
        ),
        inst_ownership_key(2026, 1): _parquet(_inst(tickers, seed=1)),
    }


def _tickers(n: int = 40) -> list[str]:
    return [f"T{i:03d}" for i in range(n)]


def _source(objects: dict[str, bytes]) -> SnapshotPointInTimeSource:
    return SnapshotPointInTimeSource(MappingSnapshotReader(objects), label="fixture")


def _panel(tickers: list[str]) -> pd.DataFrame:
    frames = synthetic_frames(end=END, sessions=SESSIONS, names=tickers)
    return pd.concat(
        [frame.reset_index() for frame in frames.values()], ignore_index=True
    ).sort_values(["trading_day", "ticker"])


# -- the source ---------------------------------------------------------------


class TestKnowledgeTime:
    def test_a_fundamentals_snapshot_is_admitted_only_after_its_label(self) -> None:
        tickers = _tickers()
        objects = _objects(tickers)
        on_label = _source(objects).load(trading_day=dt.date(2026, 9, 10), symbols=tickers)
        after = _source(objects).load(trading_day=END, symbols=tickers)

        expected_on_label = _fundamentals(tickers, dt.date(2026, 9, 9), seed=9).set_index("ticker")
        expected_after = _fundamentals(tickers, dt.date(2026, 9, 10), seed=10).set_index("ticker")
        got_on_label = on_label.frame.set_index("ticker")["roe_ratio"]
        got_after = after.frame.set_index("ticker")["roe_ratio"]
        assert got_on_label.to_dict() == expected_on_label["roe"].to_dict(), (
            "the session labelled 2026-09-10 read its own day's snapshot: a value written "
            "on a session is not known before that session is decided"
        )
        assert got_after.to_dict() == expected_after["roe"].to_dict()

    def test_the_sector_map_is_admitted_by_its_fetch_time_not_its_folder_label(self) -> None:
        tickers = _tickers()
        objects = _objects(tickers)
        # Labelled for 09-10 but fetched on the morning of 09-11 — the measured
        # v1 shape since 2026-06-25. It must not be read on 09-11.
        objects[constituents_key(dt.date(2026, 9, 10))] = json.dumps(
            {
                "tickers": tickers,
                "sector_map": dict.fromkeys(tickers, "Utilities"),
                "fetched_at": "2026-09-11T12:18:51+00:00",
            }
        ).encode()
        inputs = _source(objects).load(trading_day=END, symbols=tickers)
        sector = next(r for r in inputs.readings if r.group == "sector")
        assert sector.snapshot == constituents_key(dt.date(2026, 9, 9))
        assert sector.knowledge_date == "2026-09-10"
        assert "Utilities" not in set(inputs.frame["sector_raw"])

    def test_a_13f_quarter_is_admitted_the_session_after_its_filing_deadline(self) -> None:
        tickers = _tickers()
        objects = _objects(tickers)
        assert quarter_knowledge_date(2026, 1) == dt.date(2026, 5, 15)
        assert quarter_knowledge_date(2026, 4) == dt.date(2027, 2, 14)

        on_deadline = _source(objects).load(trading_day=dt.date(2026, 5, 15), symbols=tickers)
        after = _source(objects).load(trading_day=dt.date(2026, 5, 18), symbols=tickers)
        assert _reading(on_deadline, "institutional").state == "predates_source"
        assert _reading(after, "institutional").state == "measured"


class TestUnmeasured:
    def test_an_empty_source_is_an_outage_not_a_market_without_fundamentals(self) -> None:
        with pytest.raises(MissingSourceError, match="holds no object"):
            _source({}).load(trading_day=END, symbols=_tickers())

    def test_a_session_before_the_first_snapshot_is_unmeasurable_not_failed(self) -> None:
        """Fundamentals and 13F still read `predates_source`. Sector does not:
        Brian's ruling (a) on alpha-engine-config-I10733 backfills it, flagged
        (`TestSectorBackfill`)."""
        tickers = _tickers()
        inputs = _source(_objects(tickers)).load(trading_day=dt.date(2026, 1, 5), symbols=tickers)
        for reading in inputs.readings:
            if reading.group == "sector":
                continue
            assert reading.state == "predates_source"
            assert reading.metric_status == "unmeasurable"
        assert inputs.frame.drop(columns=["ticker", "sector_raw"]).isna().all().all()

    def test_a_stale_fundamentals_snapshot_nulls_the_group_and_fails(self) -> None:
        tickers = _tickers()
        objects = {k: v for k, v in _objects(tickers).items() if not k.startswith("features/")}
        objects[fundamental_snapshot_key(dt.date(2026, 8, 3))] = _parquet(
            _fundamentals(tickers, dt.date(2026, 8, 3), seed=3)
        )
        inputs = _source(objects).load(trading_day=END, symbols=tickers)
        reading = _reading(inputs, "fundamental")
        assert reading.state == "stale"
        assert reading.metric_status == "FAIL"
        assert inputs.frame[list(FUNDAMENTAL_FIELD_COLUMNS.values())].isna().all().all()

    def test_a_snapshot_covering_too_few_names_is_not_used(self) -> None:
        tickers = _tickers()
        objects = _objects(tickers)
        objects[constituents_key(dt.date(2026, 9, 9))] = _constituents(
            tickers[:20], dt.date(2026, 9, 9), "2026-09-10T12:18:52+00:00"
        )
        inputs = _source(objects).load(trading_day=END, symbols=tickers)
        assert _reading(inputs, "sector").state == "below_coverage"
        assert inputs.frame["sector_raw"].isna().all()

    def test_a_non_null_placeholder_field_is_unmeasured_by_name(self) -> None:
        """The v1 `fcf_yield` shape before 2026-08-19: one value across the universe."""
        tickers = _tickers()
        objects = _objects(tickers)
        frame = _fundamentals(tickers, dt.date(2026, 9, 10), seed=10)
        frame["fcf_yield"] = 0.0
        objects[fundamental_snapshot_key(dt.date(2026, 9, 10))] = _parquet(frame)
        inputs = _source(objects).load(trading_day=END, symbols=tickers)
        reading = _reading(inputs, "fundamental")
        assert reading.state == "measured"
        assert set(reading.unmeasured_fields) == {"fcf_yield_ratio"}
        assert reading.metric_status == "FAIL"
        assert inputs.frame["fcf_yield_ratio"].isna().all()
        assert inputs.frame["roe_ratio"].notna().all()

    def test_the_distinctness_floor_is_v1s_hundred_capped_at_half_the_names(self) -> None:
        assert distinctness_floor(903) == 100
        assert distinctness_floor(40) == 20
        assert distinctness_floor(25) == 13, (
            "v1 exempted every universe under 200 names, which admitted the 25-name "
            "all-zero 2026-03-27 snapshot"
        )


# -- the pillars --------------------------------------------------------------


def _average_tie_pct(value: float, population: list[float]) -> float:
    """Percentile rank (0-100) with average ties, counted by hand."""
    below = sum(1 for v in population if v < value)
    equal = sum(1 for v in population if v == value)
    return (below + (equal + 1) / 2.0) / len(population) * 100.0


def _independent_pillar(features: pd.DataFrame, pillar: str) -> dict[str, float]:
    rows = features.set_index("ticker")
    out: dict[str, float] = {}
    for ticker, row in rows.iterrows():
        sector = row["sector_raw"]
        if not isinstance(sector, str):
            out[ticker] = math.nan
            continue
        peers = rows[rows["sector_raw"] == sector]
        numerator = 0.0
        denominator = 0.0
        for column, weight, invert in PILLAR_COMPONENTS[pillar]:
            value = row[column]
            if value is None or (isinstance(value, float) and math.isnan(value)):
                continue
            population = [float(v) for v in peers[column] if not pd.isna(v)]
            pct = _average_tie_pct(float(value), population)
            numerator += weight * (100.0 - pct if invert else pct)
            denominator += weight
        out[ticker] = numerator / denominator if denominator else math.nan
    return out


def _built(tickers: list[str], objects: dict[str, bytes]) -> pd.DataFrame:
    panel = _panel(tickers)
    inputs = _source(objects).load(trading_day=END, symbols=tickers)
    features, _ = build_features(panel, point_in_time=inputs)
    return features


class TestPillars:
    def test_every_pillar_matches_an_independent_within_sector_calculation(self) -> None:
        tickers = _tickers()
        objects = _objects(tickers)
        frame = _fundamentals(tickers, dt.date(2026, 9, 10), seed=10)
        # One name missing one quality component: its pillar re-weights over
        # the three it has, v1's partial-coverage rule.
        frame.loc[frame["ticker"] == "T005", "roe"] = None
        # A tie inside a sector, so the average-tie convention is exercised.
        frame.loc[frame["ticker"].isin(["T000", "T003"]), "pb_ratio"] = 0.25
        objects[fundamental_snapshot_key(dt.date(2026, 9, 10))] = _parquet(frame)
        features = _built(tickers, objects)

        assert features["sector_raw"].notna().all()
        for pillar in PILLAR_COMPONENTS:
            expected = _independent_pillar(features, pillar)
            got = features.set_index("ticker")[pillar].to_dict()
            for ticker in tickers:
                assert got[ticker] == pytest.approx(expected[ticker], abs=1e-9), (
                    pillar,
                    ticker,
                )
            assert features[pillar].between(0.0, 100.0).all(), pillar

    def test_derived_inputs_follow_v1(self) -> None:
        tickers = _tickers()
        objects = _objects(tickers)
        inst = _inst(tickers, seed=1)
        inst.loc[inst["ticker"] == "T001", ["n_funds_increasing", "n_funds_decreasing"]] = [2, 0]
        inst.loc[inst["ticker"] == "T002", ["n_funds_increasing", "n_funds_decreasing"]] = [10, 4]
        objects[inst_ownership_key(2026, 1)] = _parquet(inst)
        features = _built(tickers, objects).set_index("ticker")

        assert features.loc["T001", "institutional_accumulation_raw"] == 0.0, (
            "two funds moving is below v1's three-fund gate"
        )
        assert features.loc["T002", "institutional_accumulation_raw"] == 6.0
        fundamentals = _fundamentals(tickers, dt.date(2026, 9, 10), seed=10).set_index("ticker")
        for ticker in tickers:
            expected = fundamentals.loc[ticker, "roe"] * (
                1 - fundamentals.loc[ticker, "payout_ratio"]
            )
            assert features.loc[ticker, "sustainable_growth_rate_ratio"] == pytest.approx(expected)

    def test_a_pillar_with_an_unmeasured_component_is_null_for_the_whole_session(self) -> None:
        tickers = _tickers()
        objects = _objects(tickers)
        frame = _fundamentals(tickers, dt.date(2026, 9, 10), seed=10)
        frame["fcf_yield"] = 0.0
        objects[fundamental_snapshot_key(dt.date(2026, 9, 10))] = _parquet(frame)
        features = _built(tickers, objects)

        assert features["value_pillar_pct"].isna().all(), (
            "v1 re-weighted value over P/E and P/B while fcf_yield rode one placeholder "
            "value; the pillar must be absent instead"
        )
        assert features["quality_pillar_pct"].notna().all()
        # hard3 reads the value pillar, so it refuses by name; momzero does not.
        params = {f"{p}_weight": 1.0 for p in ATTRACTIVENESS_PILLAR_COLUMNS}
        with pytest.raises(DegenerateFeatureError, match="value"):
            rank_with(
                "attractiveness_hard3_blend",
                features,
                _params_for("attractiveness_hard3_blend", params),
            )

    def test_without_a_sector_map_every_pillar_is_null_and_every_ranker_refuses(self) -> None:
        tickers = _tickers()
        panel = _panel(tickers)
        inputs = UnavailablePointInTimeSource(reason="test").load(trading_day=END, symbols=tickers)
        features, _ = build_features(panel, point_in_time=inputs)
        for pillar in PILLAR_COMPONENTS:
            assert features[pillar].isna().all(), pillar
        params = {f"{p}_weight": 1.0 for p in ATTRACTIVENESS_PILLAR_COLUMNS}
        for name in ATTRACTIVENESS_RANKERS:
            with pytest.raises(DegenerateFeatureError):
                rank_with(name, features, _params_for(name, params))

    def test_inputs_resolved_for_another_session_are_refused(self) -> None:
        tickers = _tickers()
        inputs = _source(_objects(tickers)).load(trading_day=dt.date(2026, 9, 10), symbols=tickers)
        with pytest.raises(ValueError, match="another session's prices"):
            build_features(_panel(tickers), point_in_time=inputs)

    def test_a_session_is_reproduced_from_a_panel_truncated_at_it(self) -> None:
        tickers = _tickers()
        objects = _objects(tickers)
        panel = _panel(tickers)
        inputs = _source(objects).load(trading_day=END, symbols=tickers)
        extended = pd.concat(
            [panel, panel[panel["trading_day"] == END].assign(trading_day=dt.date(2026, 9, 14))]
        )
        full, _ = build_features(extended, point_in_time=inputs, as_of=END)
        truncated, _ = build_features(panel, point_in_time=inputs)
        pd.testing.assert_frame_equal(full, truncated)


def _params_for(ranker: str, weights: dict[str, float]) -> dict[str, float]:
    return {k: v for k, v in weights.items() if k in get_ranker(ranker).params}


class TestRankersReadTheCatalogue:
    def test_every_attractiveness_ranker_reads_only_catalogue_columns(self) -> None:
        catalogue = set(feature_names())
        assert set(ATTRACTIVENESS_PILLAR_COLUMNS.values()) <= catalogue
        assert MOMENTUM_12_1_PILLAR_COLUMN in catalogue
        for name in ATTRACTIVENESS_RANKERS:
            assert set(get_ranker(name).reads) <= catalogue, name

    def test_every_attractiveness_ranker_ranks_the_built_layer(self) -> None:
        tickers = _tickers()
        features = _built(tickers, _objects(tickers))
        weights = {f"{p}_weight": 1.0 for p in ATTRACTIVENESS_PILLAR_COLUMNS}
        for name in ATTRACTIVENESS_RANKERS:
            scores = rank_with(name, features, _params_for(name, weights))
            assert len(scores) == len(tickers), name

    def test_pillar_depth_is_its_deepest_component(self) -> None:
        depths = catalog_column_depths()
        assert depths["momentum_12_1_pillar_pct"] == depths["mom_12_1_log_return"]
        assert depths["defensiveness_pillar_pct"] == depths["vol_ratio_10_60_ratio"]
        assert depths["quality_pillar_pct"] == 1


class TestDailyCompile:
    def test_a_measured_session_reads_complete_and_an_unavailable_one_does_not(
        self, tmp_path
    ) -> None:
        tickers = _tickers()
        frames = synthetic_frames(end=END, sessions=SESSIONS, names=[*tickers, "SPY"])
        for label, point_in_time in (
            ("measured", _source(_objects(tickers))),
            ("unavailable", UnavailablePointInTimeSource(reason="test")),
        ):
            store = LocalStore(tmp_path / label)
            ctx = run_job(
                "data.daily",
                lambda c, pit=point_in_time: run_daily(
                    c,
                    source=FramePriceSource(frames),
                    point_in_time=pit,
                    expected_symbols=tickers,
                ),
                store=store,
                trading_day=END,
            )
            metrics = {m["name"]: m for m in ctx.metrics}
            reading = check_feature_layer_completeness(store, live_version=feature_version())
            if label == "measured":
                assert reading.state == "GREEN", reading.detail
                assert metrics["point_in_time_fundamental_coverage_ratio"]["status"] == "OK"
            else:
                assert reading.state == "RED"
                assert set(PILLAR_COMPONENTS) <= set(reading.dead_columns)
                assert metrics["point_in_time_sector_coverage_ratio"]["status"] == "unmeasurable"


def _reading(inputs, group):
    return next(r for r in inputs.readings if r.group == group)
