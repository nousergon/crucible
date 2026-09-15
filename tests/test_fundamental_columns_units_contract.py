"""Consumer contract test -- data-collector plan P-14 (alpha-engine-config-I10781).

Pinned copy of nousergon-data's `features/registry.py::CATALOG` fundamental-group
column names/dtypes crucible actually reads, in
`tests/contracts/fundamental_snapshot_columns.schema.json` (mirrors the P-07
precedent: metron-PR457, crucible-PR302 -- crucible has no importable dependency
on nousergon-data, so the versioned pinned schema IS the coupling).

Root cause this class of test exists (fleet CLAUDE.md): `avg_volume_20d` was
emitted as a normalized ratio and consumed as raw shares -- 901/903 tickers
silently failed the scanner liquidity gate for months, because no consumer
pinned what it expected the producer's column to mean. This file is the crucible
half of the fix for `crucible.data.point_in_time.PointInTimeSource`'s fundamentals
group, which reads exactly the 11 nousergon-data CATALOG columns pinned below
(verified via `FUNDAMENTAL_FIELD_COLUMNS` -- the dict whose KEYS are the
nousergon-data column names read from `features/{date}/fundamental.parquet`,
and whose VALUES are crucible's own locally-renamed columns).

Each test:
  1. checks the pinned schema is itself a valid JSON Schema;
  2. checks the pin's column set is EXACTLY what `point_in_time.py` reads
     today -- drift between the pin and the reader is a failure in either
     direction;
  3. asserts a deliberately suffix-renamed or dtype-changed fixture fails
     schema validation -- the drift alarm a pinned copy exists to provide.

If nousergon-data ships a units-suffix or dtype change to any of these 11
columns, re-pin this schema AND `FUNDAMENTAL_FIELD_COLUMNS` in the SAME PR, or
this test starts disagreeing with the reader it is meant to describe.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from crucible.data.point_in_time import FUNDAMENTAL_FIELD_COLUMNS

CONTRACTS_DIR = Path(__file__).parent / "contracts"
SCHEMA_PATH = CONTRACTS_DIR / "fundamental_snapshot_columns.schema.json"


def _schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


def _pinned_instance() -> dict[str, str]:
    """The 'live' shape this schema validates: {column_name: dtype} for every
    nousergon-data column `FUNDAMENTAL_FIELD_COLUMNS` reads today, at the
    dtype the pin declares (nousergon-data `FeatureEntry.dtype` default,
    `float32`, for every fundamental-group entry)."""
    return {v1_field: "float32" for v1_field in FUNDAMENTAL_FIELD_COLUMNS}


def test_the_pinned_schema_is_itself_valid_json_schema():
    jsonschema.Draft202012Validator.check_schema(_schema())


def test_the_pin_names_exactly_the_columns_point_in_time_reads():
    """Drift in EITHER direction is a failure: a column point_in_time.py
    reads but the pin doesn't know about, or a pinned column point_in_time.py
    no longer reads."""
    pinned = set(_schema()["required"])
    read = set(FUNDAMENTAL_FIELD_COLUMNS)
    assert pinned == read, (
        f"pin vs FUNDAMENTAL_FIELD_COLUMNS mismatch -- pinned only: "
        f"{sorted(pinned - read)}, read only: {sorted(read - pinned)}"
    )


def test_the_live_reader_shape_validates_against_the_pin():
    jsonschema.validate(instance=_pinned_instance(), schema=_schema())


def test_a_units_suffix_rename_fails_the_pin():
    """The avg_volume_20d shape: nousergon-data renames `pe_ratio` to
    `pe_ratio_ratio` (adds a units suffix per its own new-field rule) without
    crucible re-pinning -- the instance now carries a key the schema's
    `additionalProperties: false` refuses, and drops a required one."""
    instance = _pinned_instance()
    instance["pe_ratio_ratio"] = instance.pop("pe_ratio")
    errors = list(jsonschema.Draft202012Validator(_schema()).iter_errors(instance))
    assert errors, "a renamed (suffix-added) column must fail the pinned schema"


def test_a_dtype_change_fails_the_pin():
    """The other half of the avg_volume_20d class: same name, different type
    -- e.g. nousergon-data starts emitting a fundamental column as a plain
    Python object (post-processed string) instead of float32."""
    instance = _pinned_instance()
    instance["roe"] = "object"
    errors = list(jsonschema.Draft202012Validator(_schema()).iter_errors(instance))
    assert errors, "a dtype change on a pinned column must fail the pinned schema"


def test_a_missing_required_column_fails_the_pin():
    """nousergon-data drops a column crucible's fundamentals group depends on
    -- caught here rather than only when point_in_time.py raises at read time."""
    instance = _pinned_instance()
    del instance["fcf_yield"]
    errors = list(jsonschema.Draft202012Validator(_schema()).iter_errors(instance))
    assert errors, "a dropped required column must fail the pinned schema"
