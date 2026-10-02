"""Consumer contract test for the weekly S&P 500/400 constituents document (P-07).

`tests/contracts/constituents.schema.json` is a byte copy of
`nousergon-data/contracts/constituents.schema.json`, the contract of
`market_data/weekly/{date}/constituents.json` (nousergon-data unit D01).
crucible never imports nousergon-data, so this versioned JSON schema is the
whole coupling. Without a pinned copy the data gate graded D01's
`schema_contract` UNMET: "declared consumer(s) with no pinned contract file:
crucible:crucible/data/universe.py" (alpha-engine-config-I11282). The gate now
compares this copy's validation shape with the producer's on every run, so a
stale copy reads UNMET there.

The fixture is trimmed from the real 2026-09-25 document (903 members, read
read-only from `s3://alpha-engine-research/market_data/weekly/2026-09-25/`), which
validates against the producer schema as published. Only three members are kept,
and the counts are restated to match them.

This test:
  1. checks the pinned schema is itself a valid JSON Schema;
  2. validates the fixture against the pinned copy, then resolves it through the
     REAL `load_declared_universe`. It goes in by the same `latest_weekly.json`
     pointer the live universe URI names, through the `read=` seam, so the field
     this consumer depends on (`tickers`) is exercised as well as declared;
  3. asserts that a document missing `tickers` fails the schema AND is refused by
     the loader. The schema and the code's own guarantee agree.

To re-pin after nousergon-data ships a new contract, copy the producer file again
and update the fixture until this file passes.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema
import pytest

from crucible.data.universe import MalformedUniverseError, load_declared_universe

SCHEMA_PATH = Path(__file__).parent / "contracts" / "constituents.schema.json"

POINTER_URI = "s3://alpha-engine-research/market_data/latest_weekly.json"
TARGET_URI = "s3://alpha-engine-research/market_data/weekly/2026-09-25/constituents.json"

_POINTER = {"date": "2026-09-25", "s3_prefix": "market_data/weekly/2026-09-25/"}

_CONSTITUENTS = {
    "date": "2026-09-25",
    "tickers": ["AAPL", "ACM", "MSFT"],
    "sp500_tickers": ["AAPL", "MSFT"],
    "sp400_tickers": ["ACM"],
    "sector_map": {
        "AAPL": "Information Technology",
        "ACM": "Industrials",
        "MSFT": "Information Technology",
    },
    "sector_etf_map": {"AAPL": "XLK", "ACM": "XLI", "MSFT": "XLK"},
    "sub_industry_map": {
        "AAPL": "Technology Hardware, Storage & Peripherals",
        "ACM": "Construction & Engineering",
        "MSFT": "Systems Software",
    },
    "sub_sector_etf_map": {"AAPL": "XLK", "ACM": "XLI", "MSFT": "IGV"},
    "sector_fallback": {},
    "sp500_count": 2,
    "sp400_count": 1,
    "total_count": 3,
    "weight_map": {
        "AAPL": 0.07460106380451607,
        "ACM": 0.002279422855926449,
        "MSFT": 0.0574433294342991,
    },
    "index_of": {"AAPL": "S&P 500", "ACM": "S&P 400", "MSFT": "S&P 500"},
    "weight_method": "ssga_holdings_file",
    "weight_sum_raw_sp500": 99.979459,
    "weight_sum_raw_sp400": 99.144395,
    "fetched_at": "2026-09-28T12:19:00.405228+00:00",
}


@pytest.fixture(scope="module")
def schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


def _reader(documents: dict[str, dict]):
    def read(uri: str) -> bytes:
        if uri not in documents:
            raise FileNotFoundError(uri)
        return json.dumps(documents[uri]).encode("utf-8")

    return read


def test_pinned_schema_is_valid(schema) -> None:
    jsonschema.Draft202012Validator.check_schema(schema)


def test_fixture_validates_and_the_real_loader_resolves_it_through_the_pointer(schema) -> None:
    jsonschema.validate(instance=_CONSTITUENTS, schema=schema)
    declared = load_declared_universe(
        POINTER_URI,
        origin="environ:CRUCIBLE_UNIVERSE_URI",
        read=_reader({POINTER_URI: _POINTER, TARGET_URI: _CONSTITUENTS}),
    )
    assert declared.symbols == ("AAPL", "ACM", "MSFT")
    assert declared.source_uri == TARGET_URI
    assert len(declared.symbols) == _CONSTITUENTS["total_count"]


def test_a_document_without_tickers_fails_the_schema_and_the_loader(schema) -> None:
    document = copy.deepcopy(_CONSTITUENTS)
    del document["tickers"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=document, schema=schema)
    with pytest.raises(MalformedUniverseError):
        load_declared_universe(TARGET_URI, origin="argument", read=_reader({TARGET_URI: document}))


def test_a_percent_scaled_weight_fails_the_schema(schema) -> None:
    """`weight_map` holds FRACTIONS normalised within each index. 7.46 is AAPL's
    weight as a percent, which is the units drift the pin exists to catch."""
    document = copy.deepcopy(_CONSTITUENTS)
    document["weight_map"]["AAPL"] = 7.460106380451607
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=document, schema=schema)


def test_an_unknown_top_level_field_fails_the_schema(schema) -> None:
    """The producer contract is closed (`additionalProperties: false`), so a field
    added on the producer side without a re-pin here fails this test."""
    document = {**copy.deepcopy(_CONSTITUENTS), "sp600_tickers": ["XYZ"]}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=document, schema=schema)
