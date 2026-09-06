"""The manifest ledger vs. the fleet cost-sink ledger must sum to the same
number, within rounding — or the run's own `llm_cost_reconciliation_usd`
metric says `FAIL` and names the discrepancy.

Normative source: `alpha-engine-config-I9986` deliverable 2 (`I9972`
deliverable 3). `crucible.llm.week_to_date_llm_spend` reads
`manifest["llm_calls"][].usd` for the per-weekly-run cap and MUST keep doing
so; this reconciliation is a second, independent check against the fleet
cost-sink rows `krepis.cost_sink.S3JsonlCostSink` writes under
`{prefix}/{date}/{run_id}/{callsite_id}.{seq}.jsonl` — the layout
`crucible.runner.run_job` now makes joinable by exporting `KREPIS_RUN_ID`
before a job's body runs (see `tests/test_runner.py::TestKrepisRunIdExport`).

A mismatch is a METRIC with `status: FAIL` on the manifest, never a log line
(§9.2 class 5) — `reconcile_run_cost` does not raise on a mismatch. It DOES
raise when a cost-sink object exists and cannot be read: an unreadable row is
a pipeline defect, not a disagreement the two ledgers are allowed to have.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from crucible.llm import (
    RECONCILIATION_TOLERANCE_USD,
    CostSinkReconciliationError,
    reconcile_run_cost,
)

BUCKET = "test-cost-sink-bucket"
PREFIX = "ops/llm/cost/crucible-v2"
RUN_ID = "01J8Z9K3M4N5P6Q7R8S9T0V1W2"
CALENDAR_DATE = dt.date(2026, 8, 29)


def _manifest(*, llm_calls_usd: list[float], calendar_date: dt.date = CALENDAR_DATE) -> dict:
    return {
        "run_id": RUN_ID,
        "calendar_date": calendar_date.isoformat(),
        "llm_calls": [
            {"callsite_id": "test.probe", "usd": usd, "model_served": "m"} for usd in llm_calls_usd
        ],
    }


def _put_sink_object(
    fake_s3, *, date: dt.date, run_id: str, callsite_id: str, rows: list[dict]
) -> None:
    key = f"{PREFIX}/{date.isoformat()}/{run_id}/{callsite_id}.0.jsonl"
    body = "\n".join(json.dumps(r) for r in rows).encode("utf-8")
    fake_s3.put_object(Bucket=BUCKET, Key=key, Body=body)


class TestAgreement:
    def test_the_two_ledgers_agreeing_reads_ok(self, fake_s3) -> None:
        manifest = _manifest(llm_calls_usd=[0.4, 0.35])
        _put_sink_object(
            fake_s3,
            date=CALENDAR_DATE,
            run_id=RUN_ID,
            callsite_id="test.probe",
            rows=[{"cost_usd": 0.4}, {"cost_usd": 0.35}],
        )

        metric = reconcile_run_cost(manifest, bucket=BUCKET, prefix=PREFIX, s3_client=fake_s3)

        assert metric["status"] == "OK"
        assert metric["name"] == "llm_cost_reconciliation_usd"
        assert metric["value"] == pytest.approx(0.75)
        assert metric["sink_usd"] == pytest.approx(0.75)
        assert metric["discrepancy_usd"] == pytest.approx(0.0)
        assert metric["run_id"] == RUN_ID
        assert metric["sink_objects_read"] == 1

    def test_a_discrepancy_within_the_rounding_tolerance_still_reads_ok(self, fake_s3) -> None:
        manifest = _manifest(llm_calls_usd=[1.000004])
        _put_sink_object(
            fake_s3,
            date=CALENDAR_DATE,
            run_id=RUN_ID,
            callsite_id="test.probe",
            rows=[{"cost_usd": 1.0}],
        )

        metric = reconcile_run_cost(manifest, bucket=BUCKET, prefix=PREFIX, s3_client=fake_s3)
        assert metric["status"] == "OK"

    def test_a_run_that_spent_and_sank_nothing_reads_ok(self, fake_s3) -> None:
        """A run with zero LLM calls reconciles against zero sink rows —
        the metric is emitted on EVERY run that could have spent, same as
        `cap_metric`, not only the ones that did."""
        manifest = _manifest(llm_calls_usd=[])

        metric = reconcile_run_cost(manifest, bucket=BUCKET, prefix=PREFIX, s3_client=fake_s3)
        assert metric["status"] == "OK"
        assert metric["value"] == 0.0
        assert metric["sink_usd"] == 0.0
        assert metric["sink_objects_read"] == 0


class TestMismatch:
    def test_the_sink_missing_a_call_the_manifest_recorded_fails_by_name(self, fake_s3) -> None:
        """The exact shape `alpha-engine-config-I7407`/`I9694` describe: spend
        that reached one ledger and not the other."""
        manifest = _manifest(llm_calls_usd=[0.75])
        # No sink object PUT at all — the manifest recorded a call the sink
        # never received.

        metric = reconcile_run_cost(manifest, bucket=BUCKET, prefix=PREFIX, s3_client=fake_s3)

        assert metric["status"] == "FAIL"
        assert "MISMATCH" in metric["status_reason"]
        assert RUN_ID in metric["status_reason"]
        assert metric["value"] == pytest.approx(0.75)
        assert metric["sink_usd"] == 0.0
        assert metric["discrepancy_usd"] == pytest.approx(0.75)

    def test_the_sink_holding_a_call_the_manifest_never_recorded_also_fails(self, fake_s3) -> None:
        """The reconciliation is symmetric: a sink row with no manifest
        counterpart is just as much a divergence as the reverse."""
        manifest = _manifest(llm_calls_usd=[0.10])
        _put_sink_object(
            fake_s3,
            date=CALENDAR_DATE,
            run_id=RUN_ID,
            callsite_id="test.probe",
            rows=[{"cost_usd": 0.10}, {"cost_usd": 5.00}],
        )

        metric = reconcile_run_cost(manifest, bucket=BUCKET, prefix=PREFIX, s3_client=fake_s3)

        assert metric["status"] == "FAIL"
        assert metric["sink_usd"] == pytest.approx(5.10)
        assert metric["discrepancy_usd"] == pytest.approx(-5.00)

    def test_this_reconciliation_has_been_seen_failing(self, fake_s3) -> None:
        """`Closes-when`: the assertion must have been SEEN failing, not only
        theorized. This test is that observation, pinned."""
        manifest = _manifest(llm_calls_usd=[2.00])
        metric = reconcile_run_cost(manifest, bucket=BUCKET, prefix=PREFIX, s3_client=fake_s3)
        assert metric["status"] == "FAIL", "the mismatch this assertion exists to catch"


class TestUnreadableSinkRow:
    def test_a_row_with_no_numeric_cost_usd_raises_rather_than_summing_as_zero(
        self, fake_s3
    ) -> None:
        """`cost_source == 'usage_unreported'` writes `cost_usd: None`
        (`krepis.cost.record_llm_call`). Summing that as zero would silently
        understate `sink_usd` by exactly the gap this function exists to
        surface."""
        manifest = _manifest(llm_calls_usd=[0.5])
        _put_sink_object(
            fake_s3,
            date=CALENDAR_DATE,
            run_id=RUN_ID,
            callsite_id="test.probe",
            rows=[{"cost_usd": None, "cost_source": "usage_unreported"}],
        )

        with pytest.raises(CostSinkReconciliationError, match="no numeric"):
            reconcile_run_cost(manifest, bucket=BUCKET, prefix=PREFIX, s3_client=fake_s3)

    def test_a_malformed_jsonl_line_raises_naming_the_key_and_line(self, fake_s3) -> None:
        key = f"{PREFIX}/{CALENDAR_DATE.isoformat()}/{RUN_ID}/test.probe.0.jsonl"
        fake_s3.put_object(Bucket=BUCKET, Key=key, Body=b"{not json}\n")
        manifest = _manifest(llm_calls_usd=[0.0])

        with pytest.raises(CostSinkReconciliationError, match=key):
            reconcile_run_cost(manifest, bucket=BUCKET, prefix=PREFIX, s3_client=fake_s3)


class TestDateWindow:
    def test_a_call_that_landed_the_day_after_calendar_date_is_still_found(self, fake_s3) -> None:
        """`S3JsonlCostSink` partitions by the record's own UTC `ts`, not by
        the run's start time — a job that straddles midnight files under the
        NEXT calendar day. The default search window covers it without the
        caller naming it."""
        manifest = _manifest(llm_calls_usd=[0.20], calendar_date=CALENDAR_DATE)
        _put_sink_object(
            fake_s3,
            date=CALENDAR_DATE + dt.timedelta(days=1),
            run_id=RUN_ID,
            callsite_id="test.probe",
            rows=[{"cost_usd": 0.20}],
        )

        metric = reconcile_run_cost(manifest, bucket=BUCKET, prefix=PREFIX, s3_client=fake_s3)
        assert metric["status"] == "OK"
        assert metric["sink_objects_read"] == 1

    def test_an_explicit_dates_argument_overrides_the_default_window(self, fake_s3) -> None:
        manifest = _manifest(llm_calls_usd=[0.20], calendar_date=CALENDAR_DATE)
        far_date = CALENDAR_DATE + dt.timedelta(days=5)
        _put_sink_object(
            fake_s3,
            date=far_date,
            run_id=RUN_ID,
            callsite_id="test.probe",
            rows=[{"cost_usd": 0.20}],
        )

        # Default window (calendar_date, calendar_date + 1) misses it.
        default_metric = reconcile_run_cost(
            manifest, bucket=BUCKET, prefix=PREFIX, s3_client=fake_s3
        )
        assert default_metric["status"] == "FAIL"

        explicit_metric = reconcile_run_cost(
            manifest, bucket=BUCKET, prefix=PREFIX, s3_client=fake_s3, dates=[far_date]
        )
        assert explicit_metric["status"] == "OK"


class TestDeclaration:
    def test_the_tolerance_is_a_positive_number_of_dollars(self) -> None:
        assert RECONCILIATION_TOLERANCE_USD > 0
