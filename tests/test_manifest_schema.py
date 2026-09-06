"""Contract test for the CURRENT run manifest schema.

(`crucible/schemas/run_manifest.v2.json` today — read through
`crucible.manifest.load_schema`, never by filename, so a version bump does
not need this file edited to keep testing the shipped contract. The
live-or-replay field v2 added has its own producer/consumer contract test in
`tests/test_run_mode_contract.py`.)

Written before the schema (fleet TDD rule) and seen failing.

The manifest schema is not documentation — it is the *enforcement surface*
for the plan's hardest guarantee (§4.2): "A run either writes a complete
run.json with `status: ok` or it is FAILED and pages. The manifest schema
forbids the third state." Every assertion below tests that the schema
refuses something, because a schema that only accepts valid documents has
not been shown to constrain anything.

**`alpha-engine-config-I10045` row 1**: `run_manifest.v2.json` is now
GENERATED from `crucible.models.RunManifestV2`
(`TestTheV2SchemaIsGeneratedFromTheModel` below), and the status<->reason
cross-field rule moved onto the model as `RunManifestV2._status_and_reason_agree`
— it is deliberately not in the schema's `allOf` any more (see
`crucible/models.py`'s docstring), so the one test that rule used to make
the raw `Draft202012Validator` reject now goes through
`crucible.manifest.validate` instead, which is the boundary that still
enforces it.
"""

from __future__ import annotations

import copy
import json
import pathlib

import pytest
from jsonschema import Draft202012Validator, ValidationError

from crucible.manifest import (
    RUN_MANIFEST_SCHEMA_VERSION,
    ManifestValidationError,
    load_schema,
    validate,
)
from crucible.models import RunManifestV2

SCHEMA_PATH = (
    pathlib.Path(__file__).resolve().parents[1] / "crucible" / "schemas" / "run_manifest.v2.json"
)

# Every state the v1 system produced that v2 excludes BY SCHEMA, not by
# policy: the audit's `cycle_verdict: "unknown"` (I9729) and the 32-of-33
# no-op skip flags (I9721). `degraded` is the "degraded-SUCCEEDED" shape §2
# names outright.
FORBIDDEN_STATUSES = ("partial", "skipped", "unknown", "degraded", "success", "OK", "")

# §4.12: a horizon expressed in a calendar unit fails schema validation.
CALENDAR_UNIT_HORIZONS = ("1 month", "3mo", "1y", "30d_calendar", "quarter")


def _valid_manifest() -> dict:
    """The minimum a job must write. Kept deliberately close to the floor:
    a fixture carrying every optional field would not prove the required
    ones are required."""
    return {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": "01JG0000000000000000000000",
        "job": "experiment.run",
        # v2's one addition (alpha-engine-config-I9918). Required with no
        # default, so it belongs in the FLOOR fixture: an optional field here
        # would prove nothing about a producer that omitted it.
        "run_mode": "live",
        "trading_day": "2026-08-28",
        "calendar_date": "2026-08-29",
        "status": "ok",
        "reason": "",
        "started": "2026-08-29T13:00:00Z",
        "finished": "2026-08-29T13:04:11Z",
        "code_sha": "0" * 40,
        "release_sha": "1" * 40,
        "seed": 20260828,
        "inputs": [
            {
                "key": "features/v3/2026-08-28.parquet",
                "sha256": "a" * 64,
                "schema_version": "v3",
            }
        ],
        "outputs": [
            {
                "key": "signals/2026-08-28/signals.json",
                "sha256": "b" * 64,
                "schema_version": "v1",
            }
        ],
        "rows_in": 903,
        "rows_out": 40,
        "rows_rejected": [{"reason": "no_price_history", "count": 3}],
        "cost_usd": 0.41,
        "llm_calls": [
            {
                "callsite_id": "research.rank.v1",
                "model_requested": "tier:high",
                "model_served": "glm-4.6",
                "route_degraded": False,
                "fallback_used": False,
                "served_deployment": "high-1",
                "tokens_in": 12000,
                "tokens_out": 900,
                "cache_read": 11000,
                "cache_write": 0,
                "usd": 0.41,
            }
        ],
        "resource": {
            "instance_type": "c7i.xlarge",
            "spot": True,
            "escalated_to_on_demand": False,
            "interruptions": 0,
            "mem_peak_mb": 2100,
            "disk_free_mb": 41000,
        },
        "metrics": [
            {
                "name": "signal_ic_21d",
                "module": "research",
                "metric_type": "ic",
                "value": 0.031,
                "unit": "ratio",
                "n_floor": 60,
                "status": "OK",
                "status_reason": "Rank IC over 903 paired names, 21 trading-day horizon.",
                "source_path": "s3://crucible/signals/2026-08-28/signals.json",
                "last_updated_utc": "2026-08-29T13:04:11Z",
                "horizon_trading_days": 21,
            }
        ],
        "attempts": [{"n": 1, "reason": "initial"}],
    }


@pytest.fixture(scope="module")
def validator() -> Draft202012Validator:
    schema = load_schema()
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def test_new_manifest_is_not_a_live_stub() -> None:
    """alpha-engine-config-I9757 defect #13: `crucible.manifest.new_manifest`
    was a `NotImplementedError` stub whose message said "is track A's" —
    track A landed (this module, `crucible.runner._write_manifest`, builds
    the manifest dict inline) and neither implemented nor removed it, and it
    had zero callers anywhere in the tree. Dead code claiming outstanding
    work, in the module that defines the central contract."""
    import crucible.manifest as manifest_module

    assert not hasattr(manifest_module, "new_manifest"), (
        "new_manifest must be removed, not merely left unimplemented — its only "
        "caller was never written, and a stub with none is dead code"
    )


def test_a_complete_manifest_validates(validator: Draft202012Validator) -> None:
    validator.validate(_valid_manifest())


def test_schema_is_closed(validator: Draft202012Validator) -> None:
    """additionalProperties: false, at the top level and inside every object.

    An open schema silently accepts `skip_reason: "no new data"` — the exact
    field the plan exists to make unrepresentable."""
    doc = _valid_manifest()
    doc["skip_reason"] = "no new data"
    with pytest.raises(ValidationError):
        validator.validate(doc)


@pytest.mark.parametrize("bad_status", FORBIDDEN_STATUSES)
def test_only_ok_and_failed_are_representable(
    validator: Draft202012Validator, bad_status: str
) -> None:
    doc = _valid_manifest()
    doc["status"] = bad_status
    with pytest.raises(ValidationError):
        validator.validate(doc)


def test_failed_requires_a_non_empty_reason() -> None:
    """`reason` is mandatory on failure. A failed run with an empty reason is
    the shape that made three Saturdays' failures indistinguishable.

    This rule is `RunManifestV2._status_and_reason_agree`, not the raw
    schema's `allOf` (dropped, `alpha-engine-config-I10045` row 1 — see
    `crucible/models.py`'s docstring), so it is checked through
    `crucible.manifest.validate`, the boundary that still enforces it.
    """
    doc = _valid_manifest()
    doc["status"] = "failed"
    doc["reason"] = ""
    with pytest.raises(ManifestValidationError):
        validate(doc)

    doc["reason"] = "SpotInterruption: instance reclaimed at 13:02Z"
    validate(doc)


def test_ok_requires_an_empty_reason() -> None:
    """The other direction of the same rule: an `ok` run with something to
    explain is a degraded-SUCCEEDED in disguise."""
    doc = _valid_manifest()
    doc["status"] = "ok"
    doc["reason"] = "ran fine, mostly"
    with pytest.raises(ManifestValidationError):
        validate(doc)


@pytest.mark.parametrize(
    "field",
    [
        "schema_version",
        "run_id",
        "job",
        "trading_day",
        "calendar_date",
        "status",
        "reason",
        "started",
        "finished",
        "code_sha",
        "inputs",
        "outputs",
        "cost_usd",
        "llm_calls",
        "resource",
        "metrics",
        "attempts",
    ],
)
def test_every_required_field_is_required(validator: Draft202012Validator, field: str) -> None:
    doc = _valid_manifest()
    del doc[field]
    with pytest.raises(ValidationError):
        validator.validate(doc)


def test_trading_day_and_calendar_date_are_separate_typed_fields(
    validator: Draft202012Validator,
) -> None:
    """§4.12: the key is the trading day; `calendar_date` is provenance.

    They are separate fields precisely so a Saturday run keyed to Friday is
    representable and legible."""
    doc = _valid_manifest()
    assert doc["trading_day"] != doc["calendar_date"]
    validator.validate(doc)

    doc["trading_day"] = "2026/08/28"
    with pytest.raises(ValidationError):
        validator.validate(doc)


@pytest.mark.parametrize("horizon", CALENDAR_UNIT_HORIZONS)
def test_a_calendar_unit_horizon_fails_validation(
    validator: Draft202012Validator, horizon: str
) -> None:
    """§4.12: "a horizon expressed as a calendar unit fails schema
    validation". The horizon is an integer count of TRADING days; there is no
    string form for it to smuggle a month into."""
    doc = _valid_manifest()
    doc["metrics"][0]["horizon_trading_days"] = horizon
    with pytest.raises(ValidationError):
        validator.validate(doc)


def test_llm_call_records_the_model_actually_served(validator: Draft202012Validator) -> None:
    """§9.2 class 2: requested AND served, because a routed call that
    silently served a different model is the failure this field exists for."""
    doc = _valid_manifest()
    del doc["llm_calls"][0]["model_served"]
    with pytest.raises(ValidationError):
        validator.validate(doc)


def test_llm_call_records_whether_the_route_was_degraded(
    validator: Draft202012Validator,
) -> None:
    """`alpha-engine-config-I9969`: a fallback-served call is
    DISTINGUISHABLE from a primary-served one in the durable record.

    Required rather than optional, and boolean rather than truthy: an absent
    field would make "the primary served" and "nobody recorded which served"
    the same reading, which is the `no data rendered as green` shape
    principle 7 forbids.
    """
    doc = _valid_manifest()
    del doc["llm_calls"][0]["route_degraded"]
    with pytest.raises(ValidationError):
        validator.validate(doc)

    doc = _valid_manifest()
    doc["llm_calls"][0]["route_degraded"] = "false"
    with pytest.raises(ValidationError):
        validator.validate(doc)


def test_the_call_time_fallback_fact_is_required_and_boolean(
    validator: Draft202012Validator,
) -> None:
    """`alpha-engine-config-I10006`: the CALL-time fact, beside the
    resolve-time one.

    `route_degraded` answers whether the ROUTE object declared a degraded
    shape at resolution; on the router-edge route the chain is walked by the
    proxy afterwards, so it can read `false` on a call the primary never
    answered. Required for the same reason `route_degraded` is: an absent
    field makes "the primary served" and "nobody recorded which served" one
    reading.
    """
    doc = _valid_manifest()
    del doc["llm_calls"][0]["fallback_used"]
    with pytest.raises(ValidationError):
        validator.validate(doc)

    doc = _valid_manifest()
    doc["llm_calls"][0]["fallback_used"] = "false"
    with pytest.raises(ValidationError):
        validator.validate(doc)


def test_served_deployment_is_required_and_admits_null_but_not_a_number(
    validator: Draft202012Validator,
) -> None:
    """`null` is the router reporting no deployment — an ANSWER, not an
    absence — so the key is required and the value is nullable.

    A reader must be able to tell "the route reported none" from "this
    producer does not record the field", which is the same distinction
    `Store.get_bytes` raising on a missing key makes everywhere else.
    """
    doc = _valid_manifest()
    del doc["llm_calls"][0]["served_deployment"]
    with pytest.raises(ValidationError):
        validator.validate(doc)

    doc = _valid_manifest()
    doc["llm_calls"][0]["served_deployment"] = None
    validator.validate(doc)

    doc = _valid_manifest()
    doc["llm_calls"][0]["served_deployment"] = 7
    with pytest.raises(ValidationError):
        validator.validate(doc)


def test_rejected_rows_carry_a_reason(validator: Draft202012Validator) -> None:
    """§9.2 class 4: rows rejected WITH REASON. A bare count cannot be acted
    on, and a rejected-row count with no reason is how 901 of 903 tickers
    failed a liquidity gate for months."""
    doc = _valid_manifest()
    doc["rows_rejected"] = [{"count": 3}]
    with pytest.raises(ValidationError):
        validator.validate(doc)


def test_resource_class_records_spot_escalation(validator: Draft202012Validator) -> None:
    """§9.2 class 3 / I5727: spot-to-on-demand fallback is a COUNTABLE
    metric, so it is a required boolean, not an optional annotation."""
    doc = _valid_manifest()
    del doc["resource"]["escalated_to_on_demand"]
    with pytest.raises(ValidationError):
        validator.validate(doc)


def test_attempts_records_the_declared_transient_retry(
    validator: Draft202012Validator,
) -> None:
    """§11 risk 2: a retried run carries BOTH attempts in the manifest, so a
    silent retry cannot look like a clean first-attempt run."""
    doc = _valid_manifest()
    doc["attempts"] = [
        {"n": 1, "reason": "spot_interruption"},
        {"n": 2, "reason": "initial"},
    ]
    validator.validate(doc)

    doc["attempts"] = []
    with pytest.raises(ValidationError):
        validator.validate(doc)


@pytest.mark.parametrize(
    "bad_metric_status",
    ["DEGRADED_BY_OPERATOR_CONSENT", "DEGRADED", "PARTIAL", "SKIPPED", "UNKNOWN", "ok", ""],
)
def test_a_metric_cannot_spell_a_third_state_either(
    validator: Draft202012Validator, bad_metric_status: str
) -> None:
    """alpha-engine-config-I9757 defect #3: the run-level `status` is closed
    to `ok`/`failed` by schema, but a metric nested inside an `ok` manifest
    is a level a human reads too — `migrate.history --allow-missing` once
    wrote `DEGRADED_BY_OPERATOR_CONSENT` there, inside a manifest whose own
    `status` was `ok`. `metricRecord.status` is now a closed enum for the
    same reason `status` itself is."""
    doc = _valid_manifest()
    doc["metrics"][0]["status"] = bad_metric_status
    with pytest.raises(ValidationError):
        validator.validate(doc)


@pytest.mark.parametrize(
    "ok_metric_status",
    [
        "OK",
        "FAIL",
        "BREACH",
        "unservable",
        "bootstrap",
        "unmeasurable",
        "measured",
        "decided",
        "held",
    ],
)
def test_every_currently_used_metric_status_still_validates(
    validator: Draft202012Validator, ok_metric_status: str
) -> None:
    """Every value a live producer writes today — `crucible.data`/`alerts`/
    `deploy`/`slots.cycle`/`track_c`'s OK/FAIL/BREACH vocabulary, and
    `crucible.promote`'s forwarded `nousergon_lib.arena.engine` decision
    vocabulary — must still be representable; closing the enum must not
    silently break a producer this PR does not own."""
    doc = _valid_manifest()
    doc["metrics"][0]["status"] = ok_metric_status
    validator.validate(doc)


def test_the_valid_fixture_is_not_mutated_between_tests() -> None:
    """Guards the parametrized tests above: they all mutate a fresh copy."""
    a = _valid_manifest()
    b = _valid_manifest()
    a["status"] = "failed"
    assert b["status"] == "ok"
    assert b == copy.deepcopy(_valid_manifest())


class TestTheMetricStatusVocabularyIsDerivedNotRestated:
    """`main` was red for eleven minutes on 2026-09-01 because it was restated.

    One PR closed `metricRecord.status` to an enum built by auditing the call
    sites that existed on its branch. Another, on a different branch, added
    `crucible.report`'s five attribution rows, whose statuses come from
    `krepis.metrics.derive_status`. Both were green; together they were red,
    and every `crucible report` run failed schema validation at the moment it
    tried to write its manifest.

    The enum stays closed — that is the point of it, and `§2 row 4` is a claim
    about every level a human reads. What changes is that the requirement is
    now DERIVED from the producer's own type rather than transcribed from it,
    so a state krepis adds fails this test instead of failing a Saturday run.
    """

    def test_the_enum_admits_every_status_krepis_can_return(self) -> None:
        from typing import get_args

        from krepis.metrics import StatusLiteral

        from crucible.manifest import load_schema

        schema = load_schema()
        enum = set(_status_enum(schema))
        missing = set(get_args(StatusLiteral)) - enum
        assert not missing, (
            f"`metricRecord.status` cannot spell {sorted(missing)}, which "
            "`krepis.metrics.derive_status` returns and `crucible.report` writes. A "
            "manifest carrying one fails validation inside the runner, so the job "
            "fails on the path where its telemetry is written."
        )

    def test_the_enum_is_still_closed(self) -> None:
        """Derived, not opened. A metric that could spell any word would
        reintroduce the third state the top-level `status` enum forbids —
        `migrate.history --allow-missing` once wrote
        `DEGRADED_BY_OPERATOR_CONSENT` here, inside an `ok` manifest."""
        from crucible.manifest import load_schema

        enum = _status_enum(load_schema())
        assert enum, "the status property must carry an enum, not a bare string type"
        assert "DEGRADED_BY_OPERATOR_CONSENT" not in enum


def _status_enum(schema: dict) -> list[str]:
    """The `metricRecord.status` enum, found wherever the schema keeps it."""

    def walk(node: object) -> list[str] | None:
        if isinstance(node, dict):
            if node.get("type") == "string" and isinstance(node.get("enum"), list):
                if "BREACH" in node["enum"]:
                    return node["enum"]
            for value in node.values():
                found = walk(value)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for value in node:
                found = walk(value)
                if found is not None:
                    return found
        return None

    found = walk(schema)
    assert found is not None, "no metricRecord status enum in the schema"
    return found


class TestTheV2SchemaIsGeneratedFromTheModel:
    """`alpha-engine-config-I10045` row 1: one source of truth, not two that
    must agree — the same rule `crucible-PR111` established for
    `components_registry.v1.json`."""

    def test_the_committed_schema_is_byte_identical_to_the_generated_one(self) -> None:
        generated = json.dumps(RunManifestV2.model_json_schema(), indent=2, sort_keys=True) + "\n"
        committed = SCHEMA_PATH.read_text(encoding="utf-8")
        assert committed == generated, (
            f"{SCHEMA_PATH.name} has drifted from `crucible.models.RunManifestV2`. The "
            "schema is GENERATED, never hand-edited: regenerate it in the same commit "
            "as the model change."
        )

    def test_the_valid_fixture_validates_against_the_committed_schema(self) -> None:
        """A schema nobody has validated a real document against is a schema
        nobody knows is right."""
        Draft202012Validator(json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))).validate(
            _valid_manifest()
        )

    def test_the_schema_version_literal_matches_the_module_constant(self) -> None:
        """`RunManifestV2.schema_version` is a bare `Literal`, restated from
        `crucible.manifest.RUN_MANIFEST_SCHEMA_VERSION` rather than imported
        — `crucible.manifest` imports `crucible.models`, so the reverse
        import would be circular. Pinned here so the two cannot drift
        silently."""
        doc = _valid_manifest()
        assert doc["schema_version"] == RUN_MANIFEST_SCHEMA_VERSION
        RunManifestV2.model_validate(doc)

    def test_the_attempt_reason_vocabulary_matches_the_transient_retry_set(self) -> None:
        """`crucible.models.ATTEMPT_REASON_VALUES` is restated, not imported
        (see `crucible/models.py`'s docstring on the import cycle); this is
        the drift guard."""
        from crucible.manifest import TRANSIENT_RETRY_REASONS
        from crucible.models import ATTEMPT_REASON_VALUES

        assert ATTEMPT_REASON_VALUES == ("initial", *TRANSIENT_RETRY_REASONS)


class TestAMalformedV2ManifestNamesTheFieldAtTheBoundary:
    """`alpha-engine-config-I10045` deliverable 4: a named field error at the
    boundary, not a `KeyError` several functions in."""

    def test_a_missing_required_field_names_it(self) -> None:
        doc = _valid_manifest()
        del doc["run_id"]

        with pytest.raises(ManifestValidationError) as excinfo:
            validate(doc)

        assert "run_id" in str(excinfo.value)

    def test_a_wrongly_typed_field_names_the_field(self) -> None:
        doc = _valid_manifest()
        doc["rows_in"] = "nine hundred three"

        with pytest.raises(ManifestValidationError) as excinfo:
            validate(doc)

        assert "rows_in" in str(excinfo.value)

    def test_an_UNKNOWN_top_level_key_is_refused(self) -> None:
        doc = _valid_manifest()
        doc["skip_reason"] = "no new data"

        with pytest.raises(ManifestValidationError) as excinfo:
            validate(doc)

        assert "skip_reason" in str(excinfo.value)

    def test_a_metric_value_with_no_unit_is_refused(self) -> None:
        """`MetricRecordRow._unit_required_when_value_present`: a numeric
        value with no declared unit is the defect that emitted
        `avg_volume_20d` as a ratio and consumed it as raw shares."""
        doc = _valid_manifest()
        del doc["metrics"][0]["unit"]

        with pytest.raises(ManifestValidationError) as excinfo:
            validate(doc)

        assert "no unit" in str(excinfo.value)

    def test_a_metric_type_outside_krepis_still_validates(self) -> None:
        """`MetricRecordRow` is deliberately not `krepis.metrics.MetricRecord`
        subclassed — a live producer's `metric_type` (`operational`,
        `coverage`, ...) is outside `krepis.metrics.MetricTypeLiteral` and
        must keep validating."""
        doc = _valid_manifest()
        doc["metrics"][0]["metric_type"] = "operational"
        validate(doc)

    def test_a_served_deployment_of_null_is_an_answer_not_an_absence(self) -> None:
        doc = _valid_manifest()
        doc["llm_calls"][0]["served_deployment"] = None
        validate(doc)

    def test_a_served_deployment_key_missing_entirely_is_refused(self) -> None:
        doc = _valid_manifest()
        del doc["llm_calls"][0]["served_deployment"]

        with pytest.raises(ManifestValidationError) as excinfo:
            validate(doc)

        assert "served_deployment" in str(excinfo.value)
