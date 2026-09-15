"""Consumer contract test — the nousergon-data 13F institutional-ownership row shape
(P-07, alpha-engine-config-I10870, D39).

A pinned copy of `nousergon-data/contracts/inst_ownership.schema.json` lives at
`tests/contracts/inst_ownership.schema.json` (crucible never imports nousergon-data;
the versioned JSON schema is the entire coupling, same discipline as
`tests/test_arctic_universe_contract.py`). v2's
`crucible.data.point_in_time.SnapshotPointInTimeSource._load_institutional` is the
first-class consumer of `data/inst_ownership/{quarter}/latest.parquet`
(`crucible.keys.inst_ownership_key`), hard-requiring `ticker`, `n_funds_increasing`,
`n_funds_decreasing` (`INSTITUTIONAL_COLUMNS`) — a missing column raises
`MissingSourceError`, never a zero-fill.

nousergon-data's D39 descriptor and `consumers` list did not previously name crucible
as a consumer of this key (that correction is a separate, sibling PR); this test pins
the row shape the real reader already depends on regardless.

This test:
  1. checks the pinned schema is itself a valid JSON Schema;
  2. builds schema-conformant rows per the pinned contract, turns them into the raw
     parquet-shaped bytes `MappingSnapshotReader` serves, and feeds them through the
     REAL `SnapshotPointInTimeSource._load_institutional` (via `.load()`) so the
     actual read/refuse contract is exercised, not a parallel one;
  3. asserts a row missing `n_funds_increasing` fails schema validation AND that the
     source independently refuses the same shape via `MissingSourceError`.
"""

from __future__ import annotations

import datetime as dt
import io
import json
from pathlib import Path

import jsonschema
import pandas as pd
import pytest

from crucible.data.point_in_time import (
    FUNDAMENTAL_FIELD_COLUMNS,
    MappingSnapshotReader,
    MissingSourceError,
    SnapshotPointInTimeSource,
    constituents_key,
    fundamental_snapshot_key,
    inst_ownership_key,
)

SCHEMA_PATH = Path(__file__).parent / "contracts" / "inst_ownership.schema.json"
_SECTORS = ("Energy", "Health Care", "Information Technology")


def _fundamentals(tickers: list[str], label: dt.date) -> pd.DataFrame:
    rows = []
    for i, ticker in enumerate(tickers):
        row = {"ticker": ticker, "date": label.isoformat()}
        for j, v1_field in enumerate(FUNDAMENTAL_FIELD_COLUMNS):
            row[v1_field] = round(((i + 1) * (j + 1) % 97) / 97.0, 6)
        rows.append(row)
    return pd.DataFrame(rows)


def _constituents(tickers: list[str], label: dt.date, fetched_at: str) -> bytes:
    return json.dumps(
        {
            "date": label.isoformat(),
            "tickers": tickers,
            "sector_map": {t: _SECTORS[i % len(_SECTORS)] for i, t in enumerate(tickers)},
            "fetched_at": fetched_at,
        }
    ).encode()


def _base_objects(tickers: list[str]) -> dict[str, bytes]:
    """Fundamentals + sector snapshots so `.load()`'s other two groups measure —
    this test's subject is the institutional group; the others must not raise."""
    buf9, buf10 = io.BytesIO(), io.BytesIO()
    _fundamentals(tickers, dt.date(2026, 9, 9)).to_parquet(buf9, index=False)
    _fundamentals(tickers, dt.date(2026, 9, 10)).to_parquet(buf10, index=False)
    return {
        fundamental_snapshot_key(dt.date(2026, 9, 9)): buf9.getvalue(),
        fundamental_snapshot_key(dt.date(2026, 9, 10)): buf10.getvalue(),
        constituents_key(dt.date(2026, 9, 9)): _constituents(
            tickers, dt.date(2026, 9, 9), "2026-09-10T12:18:52+00:00"
        ),
    }


@pytest.fixture(scope="module")
def schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


def test_pinned_schema_is_valid(schema):
    jsonschema.Draft202012Validator.check_schema(schema)


def _row(ticker: str, *, n_inc: int, n_dec: int) -> dict:
    return {
        "ticker": ticker,
        "quarter": "2026Q1",
        "schema_version": 1,
        "n_funds_holding": 18,
        "total_shares_held": 450_200_000.0,
        "total_value_usd": 90_000_000_000.0,
        "shares_qoq_change": 2_100_000.0,
        "value_qoq_change": 500_000_000.0,
        "top5_concentration_pct": 8.2,
        "n_funds_increasing": n_inc,
        "n_funds_decreasing": n_dec,
        "n_funds_new": 1,
        "n_funds_exited": 0,
        "put_call_ratio": None,
    }


def test_row_fixture_validates_against_pinned_schema(schema):
    jsonschema.validate(instance=_row("AAA", n_inc=12, n_dec=3), schema=schema)


def test_row_missing_n_funds_increasing_fails_schema(schema):
    row = _row("AAA", n_inc=12, n_dec=3)
    del row["n_funds_increasing"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=row, schema=schema)


def _frame_from_rows(rows: list[dict]) -> bytes:
    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


def test_schema_conformant_rows_flow_through_the_real_institutional_source(schema):
    """The real consumer path: schema-conformant rows -> parquet bytes ->
    SnapshotPointInTimeSource.load() -> a measured `institutional` GroupReading."""
    tickers = [f"T{i:03d}" for i in range(40)]
    rows = [
        _row(t, n_inc=(7 * i) % 41, n_dec=(11 * i) % 37)
        for i, t in enumerate(tickers)
    ]
    for row in rows:
        jsonschema.validate(instance=row, schema=schema)
    objects = {**_base_objects(tickers), inst_ownership_key(2026, 1): _frame_from_rows(rows)}
    source = SnapshotPointInTimeSource(MappingSnapshotReader(objects), label="fixture")

    inputs = source.load(trading_day=dt.date(2026, 9, 11), symbols=tickers)
    institutional = next(r for r in inputs.readings if r.group == "institutional")

    assert institutional.state == "measured"
    assert institutional.covered == len(tickers)
    assert institutional.snapshot == inst_ownership_key(2026, 1)


def test_row_missing_a_required_institutional_column_is_refused_by_both_schema_and_source(
    schema,
):
    """The schema's `required` list and the source's own refusal must agree: a frame
    short `n_funds_decreasing` is never silently zero-filled by either."""
    tickers = [f"T{i:03d}" for i in range(40)]
    rows = [_row(t, n_inc=(7 * i) % 41, n_dec=(11 * i) % 37) for i, t in enumerate(tickers)]

    broken = _row(tickers[0], n_inc=1, n_dec=1)
    del broken["n_funds_decreasing"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=broken, schema=schema)

    broken_rows = [dict(r) for r in rows]
    for r in broken_rows:
        del r["n_funds_decreasing"]
    objects = {
        **_base_objects(tickers),
        inst_ownership_key(2026, 1): _frame_from_rows(broken_rows),
    }
    source = SnapshotPointInTimeSource(MappingSnapshotReader(objects), label="fixture")
    with pytest.raises(MissingSourceError, match="missing column"):
        source.load(trading_day=dt.date(2026, 9, 11), symbols=tickers)
