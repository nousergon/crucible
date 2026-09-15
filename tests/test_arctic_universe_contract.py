"""Consumer contract test — the ArcticDB `universe` library row shape (P-07,
alpha-engine-config-I10774).

A pinned copy of `nousergon-data/contracts/arctic_universe.schema.json` lives at
`tests/contracts/arctic_universe.schema.json` (crucible never imports nousergon-data;
the versioned JSON schema is the entire coupling, same discipline as
`metron/tests/contracts/*.schema.json`). v2's `data.daily` is the first-class consumer
of this library (plan §3 boundary table row 3), reached through
`crucible.data.sources.ArcticPriceSource.load_panel` -> `normalize_panel`, which renames
the raw `Open`/`High`/`Low`/`Close`/`Volume` columns to the `PANEL_COLUMNS` contract
(`open_raw`/`high_raw`/`low_raw`/`close_raw`/`volume_raw`) and raises `MissingSourceError`
on any frame missing one of them -- never zero-fills.

This test:
  1. checks the pinned schema is itself a valid JSON Schema;
  2. builds one schema-conformant row per the pinned contract, turns it into the raw
     ArcticDB-shaped frame (`Open`/`High`/`Low`/`Close`/`Volume` on a tz-naive
     `DatetimeIndex` -- exactly what `nousergon_lib.arcticdb.load_universe_ohlcv` returns),
     and feeds it through the REAL `ArcticPriceSource.load_panel` (via the same
     `load_universe_ohlcv` monkeypatch seam `tests/test_data_sources_library_override.py`
     uses) so the actual rename/refuse contract is exercised, not a parallel one;
  3. asserts a row missing a required OHLCV column fails schema validation AND that
     `ArcticPriceSource` independently refuses the same shape via `MissingSourceError` --
     the schema and the code's own guarantee agree.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from crucible.data.sources import PANEL_COLUMNS, ArcticPriceSource, MissingSourceError

SCHEMA_PATH = Path(__file__).parent / "contracts" / "arctic_universe.schema.json"


@pytest.fixture(scope="module")
def schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


def test_pinned_schema_is_valid(schema):
    jsonschema.Draft202012Validator.check_schema(schema)


def _row(
    symbol: str,
    index_date: str,
    *,
    open_=100.0,
    high=101.0,
    low=99.0,
    close=100.5,
    volume=1_000_000.0,
) -> dict:
    return {
        "symbol": symbol,
        "index_date": index_date,
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    }


def test_row_fixture_validates_against_pinned_schema(schema):
    jsonschema.validate(instance=_row("AAA", "2026-06-26"), schema=schema)


def test_row_missing_close_fails_schema(schema):
    row = _row("AAA", "2026-06-26")
    del row["Close"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=row, schema=schema)


def test_row_with_bitemporal_additive_columns_still_validates(schema):
    """settled/as_of/source_tier/valid_date/knowledge_time (config#2459) are additive --
    a row carrying them is still a valid `universe` library row."""
    row = _row("AAA", "2026-06-26")
    row.update(
        {
            "settled": True,
            "as_of": "2026-06-26T21:15:00Z",
            "source_tier": "primary",
            "valid_date": "2026-06-26",
            "knowledge_time": "2026-06-26T21:15:00Z",
            "source": "polygon",
        }
    )
    jsonschema.validate(instance=row, schema=schema)


def _frame_from_rows(rows: list[dict]):
    """Turn a list of schema-conformant rows (one symbol) into the raw ArcticDB-shaped
    frame `load_universe_ohlcv` returns: `Open`/`High`/`Low`/`Close`/`Volume` on a
    tz-naive `DatetimeIndex`."""
    import pandas as pd

    df = pd.DataFrame(rows).set_index("index_date")
    df.index = pd.to_datetime(df.index)
    df.index.name = None
    return df[["Open", "High", "Low", "Close", "Volume"]]


def test_schema_conformant_rows_flow_through_arctic_price_source(monkeypatch, cycle_date):
    """The real consumer path: schema-conformant rows -> raw Arctic frame ->
    `ArcticPriceSource.load_panel` -> the `PANEL_COLUMNS` contract, values intact."""
    rows = [
        _row("AAA", "2026-06-24", close=100.0),
        _row("AAA", "2026-06-25", close=101.0),
        _row("AAA", "2026-06-26", close=102.0),
    ]
    for row in rows:
        jsonschema.validate(instance=row, schema=json.loads(SCHEMA_PATH.read_text()))
    frames = {"AAA": _frame_from_rows(rows)}

    monkeypatch.setattr("nousergon_lib.arcticdb.load_universe_ohlcv", lambda *a, **k: frames)

    source = ArcticPriceSource("test-bucket")
    panel = source.load_panel(end=cycle_date, lookback_days=400, symbols=["AAA"])

    assert list(panel.columns) == list(PANEL_COLUMNS)
    assert set(panel["ticker"]) == {"AAA"}
    assert panel.sort_values("trading_day")["close_raw"].tolist()[-1] == pytest.approx(102.0)


def test_row_missing_a_required_ohlcv_column_is_refused_by_both_schema_and_source(
    monkeypatch,
    cycle_date,
    schema,
):
    """The schema's `required` list and `ArcticPriceSource`'s own refusal must agree: a
    frame short a column is never silently zero-filled by either."""
    row = _row("AAA", "2026-06-26")
    del row["Volume"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=row, schema=schema)

    broken_frame = _frame_from_rows([_row("AAA", "2026-06-26")]).drop(columns=["Volume"])
    monkeypatch.setattr(
        "nousergon_lib.arcticdb.load_universe_ohlcv", lambda *a, **k: {"AAA": broken_frame}
    )
    source = ArcticPriceSource("test-bucket")
    with pytest.raises(MissingSourceError, match="missing column"):
        source.load_panel(end=cycle_date, lookback_days=400, symbols=["AAA"])
