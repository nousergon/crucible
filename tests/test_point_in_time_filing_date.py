"""`FilingDatePointInTimeSource`: the SEC EDGAR filing-date fundamentals
source, `alpha-engine-config-I10733`.

Mirrors `tests/test_point_in_time_attractiveness.py`'s style and reuses its
production resolution path (`MappingSnapshotReader`), but is scoped to the
one thing that changes for this source: the fundamental group's key shape,
and its extra producer-contract refusals. Sector and 13F are asserted to
resolve identically to `SnapshotPointInTimeSource` on the SAME objects,
since this class inherits both unchanged.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import random

import pandas as pd
import pytest

from crucible.data import FramePriceSource, MissingSourceError, run_daily
from crucible.data.point_in_time import (
    EDGAR_SESSION_SCHEMA_VERSION,
    FUNDAMENTAL_FIELD_COLUMNS,
    FilingDatePointInTimeSource,
    MappingSnapshotReader,
    SnapshotPointInTimeSource,
)
from crucible.features import build_features
from crucible.features.registry import feature_version
from crucible.keys import (
    constituents_key,
    edgar_fundamentals_session_key,
    fundamental_snapshot_key,
    inst_ownership_key,
)
from crucible.runner import run_job
from crucible.store import LocalStore
from tests.conftest import SESSIONS, synthetic_frames

END = dt.date(2026, 9, 11)
SECTORS = ("Energy", "Health Care", "Information Technology")

_RAW_COLUMNS = (
    "close_raw",
    "market_cap_raw",
    "shares_outstanding_raw",
    "net_income_ttm_raw",
    "revenue_ttm_raw",
    "gross_profit_ttm_raw",
    "equity_raw",
    "total_debt_raw",
    "assets_current_raw",
    "liabilities_current_raw",
    "fcf_ttm_raw",
    "dividends_ttm_raw",
)


def _parquet(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


def _edgar_fundamentals(
    tickers: list[str],
    label: dt.date,
    *,
    seed: int,
    knowledge_date: dt.date | None = None,
    schema_version: int | None = EDGAR_SESSION_SCHEMA_VERSION,
    latest_filed: dt.date | None = None,
    null_latest_filed: bool = False,
) -> pd.DataFrame:
    """A fixture EDGAR session, fully conformant to the producer contract by
    default. Every keyword lets one test bend exactly one rule.
    """
    rng = random.Random(seed)
    knowledge_date = knowledge_date if knowledge_date is not None else label
    filed = latest_filed if latest_filed is not None else label - dt.timedelta(days=1)
    rows = []
    for i, ticker in enumerate(tickers):
        row: dict[str, object] = {
            "ticker": ticker,
            "cik": 1000000 + i,
            "knowledge_date": knowledge_date.isoformat(),
            "schema_version": schema_version,
            "latest_filed": None if null_latest_filed else filed.isoformat(),
            "latest_accession": f"0001-26-{i:06d}",
        }
        for v1_field in FUNDAMENTAL_FIELD_COLUMNS:
            row[v1_field] = round(rng.uniform(-0.5, 1.5), 6)
        for raw_column in _RAW_COLUMNS:
            row[raw_column] = round(rng.uniform(1.0, 1000.0), 2)
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
    """A snapshot store shaped like the production data bucket, fully measured
    on END, through the EDGAR key shape."""
    return {
        edgar_fundamentals_session_key(dt.date(2026, 9, 9)): _parquet(
            _edgar_fundamentals(tickers, dt.date(2026, 9, 9), seed=9)
        ),
        edgar_fundamentals_session_key(dt.date(2026, 9, 10)): _parquet(
            _edgar_fundamentals(tickers, dt.date(2026, 9, 10), seed=10)
        ),
        constituents_key(dt.date(2026, 9, 9)): _constituents(
            tickers, dt.date(2026, 9, 9), "2026-09-10T12:18:52+00:00"
        ),
        inst_ownership_key(2026, 1): _parquet(_inst(tickers, seed=1)),
    }


def _tickers(n: int = 40) -> list[str]:
    return [f"T{i:03d}" for i in range(n)]


def _source(objects: dict[str, bytes]) -> FilingDatePointInTimeSource:
    return FilingDatePointInTimeSource(MappingSnapshotReader(objects), label="fixture")


def _panel(tickers: list[str]) -> pd.DataFrame:
    frames = synthetic_frames(end=END, sessions=SESSIONS, names=tickers)
    return pd.concat(
        [frame.reset_index() for frame in frames.values()], ignore_index=True
    ).sort_values(["trading_day", "ticker"])


def _reading(inputs, group):
    return next(r for r in inputs.readings if r.group == group)


class TestKeyShape:
    def test_the_edgar_session_key_is_the_declared_literal(self) -> None:
        assert (
            edgar_fundamentals_session_key(dt.date(2026, 9, 9))
            == "fundamentals_pit/edgar/v1/sessions/2026-09-09.parquet"
        )


class TestKnowledgeTime:
    def test_a_session_is_admitted_only_strictly_after_its_label(self) -> None:
        tickers = _tickers()
        objects = _objects(tickers)
        on_label = _source(objects).load(trading_day=dt.date(2026, 9, 10), symbols=tickers)
        after = _source(objects).load(trading_day=END, symbols=tickers)

        expected_on_label = _edgar_fundamentals(tickers, dt.date(2026, 9, 9), seed=9).set_index(
            "ticker"
        )
        expected_after = _edgar_fundamentals(tickers, dt.date(2026, 9, 10), seed=10).set_index(
            "ticker"
        )
        got_on_label = on_label.frame.set_index("ticker")["roe_ratio"]
        got_after = after.frame.set_index("ticker")["roe_ratio"]
        assert got_on_label.to_dict() == expected_on_label["roe"].to_dict(), (
            "the session labelled 2026-09-10 must read the PREVIOUS file: a value filed "
            "that day is not known before that day is decided"
        )
        assert got_after.to_dict() == expected_after["roe"].to_dict()

    def test_sector_and_13f_resolve_identically_to_the_snapshot_source(self) -> None:
        """Both groups are inherited unchanged — same objects, same readings.

        The v1 snapshot comparator needs its OWN fundamentals object too
        (`SnapshotPointInTimeSource.load` resolves all three groups), so one
        is added under the v1 key shape; only the sector and institutional
        readings are compared, which do not read it.
        """
        tickers = _tickers()
        objects = _objects(tickers)
        snapshot_objects = dict(objects)
        snapshot_objects[fundamental_snapshot_key(dt.date(2026, 9, 10))] = _parquet(
            _edgar_fundamentals(tickers, dt.date(2026, 9, 10), seed=10)
        )
        edgar_inputs = _source(objects).load(trading_day=END, symbols=tickers)
        snapshot_inputs = SnapshotPointInTimeSource(
            MappingSnapshotReader(snapshot_objects), label="fixture"
        ).load(trading_day=END, symbols=tickers)

        for group in ("sector", "institutional"):
            edgar_reading = _reading(edgar_inputs, group)
            snapshot_reading = _reading(snapshot_inputs, group)
            assert edgar_reading.state == snapshot_reading.state == "measured"
            assert edgar_reading.snapshot == snapshot_reading.snapshot
            assert edgar_reading.knowledge_date == snapshot_reading.knowledge_date
            assert edgar_reading.covered == snapshot_reading.covered
        assert (
            edgar_inputs.frame["sector_raw"].tolist()
            == snapshot_inputs.frame["sector_raw"].tolist()
        )
        assert (
            edgar_inputs.frame["n_funds_increasing_raw"].tolist()
            == snapshot_inputs.frame["n_funds_increasing_raw"].tolist()
        )


class TestUnmeasured:
    def test_an_empty_edgar_prefix_is_an_outage_like_the_snapshot_source(self) -> None:
        tickers = _tickers()
        objects = {
            k: v for k, v in _objects(tickers).items() if not k.startswith("fundamentals_pit/")
        }
        with pytest.raises(MissingSourceError, match="holds no object"):
            _source(objects).load(trading_day=END, symbols=tickers)

    def test_a_stale_edgar_session_nulls_the_group_and_fails(self) -> None:
        tickers = _tickers()
        objects = {
            k: v for k, v in _objects(tickers).items() if not k.startswith("fundamentals_pit/")
        }
        objects[edgar_fundamentals_session_key(dt.date(2026, 8, 3))] = _parquet(
            _edgar_fundamentals(tickers, dt.date(2026, 8, 3), seed=3)
        )
        inputs = _source(objects).load(trading_day=END, symbols=tickers)
        reading = _reading(inputs, "fundamental")
        assert reading.state == "stale"
        assert reading.metric_status == "FAIL"
        assert inputs.frame[list(FUNDAMENTAL_FIELD_COLUMNS.values())].isna().all().all()

    def test_a_snapshot_covering_too_few_names_is_not_used(self) -> None:
        tickers = _tickers()
        objects = _objects(tickers)
        thin = _edgar_fundamentals(tickers[:5], dt.date(2026, 9, 10), seed=10)
        objects[edgar_fundamentals_session_key(dt.date(2026, 9, 10))] = _parquet(thin)
        inputs = _source(objects).load(trading_day=END, symbols=tickers)
        reading = _reading(inputs, "fundamental")
        assert reading.state == "below_coverage"
        assert inputs.frame["roe_ratio"].isna().all()

    def test_a_non_null_placeholder_field_is_unmeasured_by_name(self) -> None:
        tickers = _tickers()
        objects = _objects(tickers)
        frame = _edgar_fundamentals(tickers, dt.date(2026, 9, 10), seed=10)
        frame["fcf_yield"] = 0.0
        objects[edgar_fundamentals_session_key(dt.date(2026, 9, 10))] = _parquet(frame)
        inputs = _source(objects).load(trading_day=END, symbols=tickers)
        reading = _reading(inputs, "fundamental")
        assert reading.state == "measured"
        assert set(reading.unmeasured_fields) == {"fcf_yield_ratio"}
        assert inputs.frame["fcf_yield_ratio"].isna().all()
        assert inputs.frame["roe_ratio"].notna().all()


class TestProducerContractRefusals:
    """A row violating the producer's OWN declared contract stops the
    compile — a look-ahead or a version drift must never read as an
    ordinary outage or be silently averaged into a pillar.
    """

    def _one_bad_session(self, tickers: list[str], frame: pd.DataFrame) -> dict[str, bytes]:
        objects = _objects(tickers)
        objects[edgar_fundamentals_session_key(dt.date(2026, 9, 10))] = _parquet(frame)
        return objects

    def test_latest_filed_after_the_label_is_refused(self) -> None:
        tickers = _tickers()
        frame = _edgar_fundamentals(
            tickers,
            dt.date(2026, 9, 10),
            seed=10,
            latest_filed=dt.date(2026, 9, 11),
        )
        objects = self._one_bad_session(tickers, frame)
        with pytest.raises(MissingSourceError, match="latest_filed"):
            _source(objects).load(trading_day=END, symbols=tickers)

    def test_a_null_latest_filed_is_refused(self) -> None:
        tickers = _tickers()
        frame = _edgar_fundamentals(tickers, dt.date(2026, 9, 10), seed=10, null_latest_filed=True)
        objects = self._one_bad_session(tickers, frame)
        with pytest.raises(MissingSourceError, match="latest_filed"):
            _source(objects).load(trading_day=END, symbols=tickers)

    def test_a_schema_version_other_than_one_is_refused(self) -> None:
        tickers = _tickers()
        frame = _edgar_fundamentals(tickers, dt.date(2026, 9, 10), seed=10, schema_version=2)
        objects = self._one_bad_session(tickers, frame)
        with pytest.raises(MissingSourceError, match="schema_version"):
            _source(objects).load(trading_day=END, symbols=tickers)

    def test_a_knowledge_date_unequal_to_its_own_label_is_refused(self) -> None:
        tickers = _tickers()
        frame = _edgar_fundamentals(
            tickers,
            dt.date(2026, 9, 10),
            seed=10,
            knowledge_date=dt.date(2026, 9, 9),
        )
        objects = self._one_bad_session(tickers, frame)
        with pytest.raises(MissingSourceError, match="knowledge_date"):
            _source(objects).load(trading_day=END, symbols=tickers)


class TestFeatureVersionUnchanged:
    def test_feature_version_matches_the_committed_catalog_pin(self) -> None:
        """The eleven EDGAR field names equal `FUNDAMENTAL_FIELD_COLUMNS`'
        keys — no catalogue expression or column name changes for this
        source, so the committed pin must still match the live catalogue.
        `tests/test_features_catalog_pin.py` is the normative pin check;
        this asserts the same fact from this PR's own vantage point.
        """
        from pathlib import Path

        pin_path = (
            Path(__file__).resolve().parent.parent / "crucible" / "features" / "CATALOG_VERSION"
        )
        assert feature_version() == pin_path.read_text().strip()


class TestEndToEnd:
    def test_a_measured_session_reads_complete_through_the_edgar_source(self, tmp_path) -> None:
        tickers = _tickers()
        frames = synthetic_frames(end=END, sessions=SESSIONS, names=[*tickers, "SPY"])
        store = LocalStore(tmp_path / "edgar")
        source = _source(_objects(tickers))
        ctx = run_job(
            "data.daily",
            lambda c: run_daily(
                c,
                source=FramePriceSource(frames),
                point_in_time=source,
                expected_symbols=tickers,
            ),
            store=store,
            trading_day=END,
        )
        metrics = {m["name"]: m for m in ctx.metrics}
        assert metrics["point_in_time_fundamental_coverage_ratio"]["status"] == "OK"

        panel = _panel(tickers)
        inputs = source.load(trading_day=END, symbols=tickers)
        features, _ = build_features(panel, point_in_time=inputs)
        assert features["quality_pillar_pct"].notna().all()
        assert features["value_pillar_pct"].notna().all()
        assert features["growth_pillar_pct"].notna().all()
