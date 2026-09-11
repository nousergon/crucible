"""Producer/consumer contract test for `factor_attribution.v1` (M0 rule).

Every new cross-module artifact gets a versioned schema plus a
producer/consumer contract test at birth. The producer is
`crucible.attribution.compute_factor_attribution` +
`attribution_metric_record`; the consumer is the predicate
`manifest_records_factor_attribution`, shaped like
`crucible.portfolio.manifest_records_portfolio_engine` so a future phase-3
gate clause can call it directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from crucible.attribution import (
    ATTRIBUTION_METRIC_NAME,
    FACTOR_ATTRIBUTION_SCHEMA_VERSION,
    AttributionFactorParams,
    FactorDef,
    attribution_metric_record,
    compute_factor_attribution,
    manifest_records_factor_attribution,
)

_SCHEMA_PATH = (
    Path(__file__).parent.parent
    / "crucible"
    / "schemas"
    / f"{FACTOR_ATTRIBUTION_SCHEMA_VERSION}.json"
)

_PARAMS = AttributionFactorParams(
    factors={
        "market": FactorDef(category="beta", proxy="SPY"),
        "sector_tech": FactorDef(category="sector", proxy="XLK"),
        "size_factor": FactorDef(category="size", proxy="IWM"),
    },
    benchmark_proxy="SPY",
)

_MARKET = [0.01, -0.005, 0.008, 0.002, -0.011, 0.006]
_SECTOR = [0.004, 0.006, -0.002, 0.009, -0.003, 0.001]
_SIZE = [-0.002, 0.001, 0.003, -0.004, 0.005, 0.002]
_HOLDING_RETURNS = {
    "AAA": [1.0 * m + 0.5 * s + 0.2 * z for m, s, z in zip(_MARKET, _SECTOR, _SIZE, strict=True)],
    "BBB": [0.8 * m + 0.0 * s + 0.3 * z for m, s, z in zip(_MARKET, _SECTOR, _SIZE, strict=True)],
}
_WEIGHTS = {"AAA": 0.6, "BBB": 0.4}


def _build_evidence() -> dict:
    gross = sum(_WEIGHTS[t] * sum(_HOLDING_RETURNS[t]) for t in _WEIGHTS)
    return compute_factor_attribution(
        trading_day="2026-07-06",
        window_sessions=6,
        holding_returns=_HOLDING_RETURNS,
        weights=_WEIGHTS,
        factor_returns={"market": _MARKET, "sector_tech": _SECTOR, "size_factor": _SIZE},
        params=_PARAMS,
        gross_return=gross,
        cost_bps_total=10.0,
    )


def test_schema_file_is_a_valid_draft_2020_12_schema() -> None:
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)


def test_producer_output_validates_against_its_own_schema() -> None:
    evidence = _build_evidence()
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    errors = list(validator.iter_errors(evidence))
    assert not errors, [e.message for e in errors]


def test_metric_record_round_trips_through_a_manifest_reader() -> None:
    evidence = _build_evidence()
    row = attribution_metric_record(evidence, now_utc="2026-07-06T21:00:00Z")
    manifest = {
        "job": "some.job",
        "trading_day": "2026-07-06",
        "metrics": [row, {"name": "unrelated", "value": 1.0}],
    }
    read_back = manifest_records_factor_attribution(manifest)
    assert read_back == evidence


def test_manifest_with_no_matching_metric_row_reads_as_none() -> None:
    manifest = {"job": "some.job", "trading_day": "2026-07-06", "metrics": [{"name": "other"}]}
    assert manifest_records_factor_attribution(manifest) is None


def test_metric_row_missing_its_payload_raises_rather_than_reading_as_absent() -> None:
    manifest = {
        "job": "some.job",
        "trading_day": "2026-07-06",
        "metrics": [{"name": ATTRIBUTION_METRIC_NAME}],  # no factor_attribution payload
    }
    with pytest.raises(ValueError, match="no `factor_attribution` payload"):
        manifest_records_factor_attribution(manifest)


def test_metric_row_with_a_future_schema_version_raises() -> None:
    evidence = dict(_build_evidence())
    evidence["schema_version"] = "factor_attribution.v2"
    manifest = {
        "job": "some.job",
        "trading_day": "2026-07-06",
        "metrics": [{"name": ATTRIBUTION_METRIC_NAME, "factor_attribution": evidence}],
    }
    with pytest.raises(ValueError, match="read wrong, not read approximately"):
        manifest_records_factor_attribution(manifest)


def test_producer_refuses_to_emit_a_document_missing_a_required_field() -> None:
    """A hand-corrupted document must fail the same validator the producer runs at birth."""
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    evidence = _build_evidence()
    del evidence["residual_alpha"]
    errors = list(validator.iter_errors(evidence))
    assert errors, "removing a required field must fail schema validation"
