"""The weekly heartbeat reconciles the week's manifests against the fleet cost sink.

`alpha-engine-config-I9986` deliverable 2: "for a weekly run, sum(llm_calls[].usd)
across that run's manifests equals the sum of the fleet cost rows under
`{prefix}/{date}/{run_id}/`, within rounding. A discrepancy is a FAILED run,
not a note." The heartbeat is the "later job" the issue's gotcha names — the
sink flushes at `atexit`, so reconciling inside the run that spent would read
an incomplete sink.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest
from krepis.cost_sink import BUCKET_ENV_VAR, PREFIX_ENV_VAR

from crucible.alerts import PAGES_TOPIC_ARN_VAR, StoreAccessError, heartbeat
from crucible.llm import RECONCILIATION_METRIC, CostSinkReconciliationError
from crucible.manifest import manifest_key, validate
from crucible.runner import run_job
from crucible.store import LocalStore

FRIDAY = dt.date(2026, 8, 28)
SATURDAY_NIGHT = dt.datetime(2026, 8, 29, 23, 30, tzinfo=dt.UTC)
BUCKET = "test-cost-sink-bucket"
PREFIX = "ops/llm/cost/crucible-v2"
RUN_ID = "01J8Z9K3M4N5P6Q7R8S9T0V1W2"


def _write_run(store: LocalStore, *, job: str, llm_usd: list[float], run_id: str = RUN_ID) -> None:
    store.put_bytes(
        manifest_key(job, FRIDAY.isoformat()),
        json.dumps(
            {
                "schema_version": "run_manifest.v2",
                "status": "ok",
                "reason": "",
                "run_id": run_id,
                "calendar_date": FRIDAY.isoformat(),
                "trading_day": FRIDAY.isoformat(),
                "job": job,
                "cost_usd": 0.0,
                "llm_calls": [{"callsite_id": "t", "usd": u, "model_served": "m"} for u in llm_usd],
            }
        ).encode(),
    )


def _sink_rows(fake_s3, *, run_id: str, usd: list[float]) -> None:
    key = f"{PREFIX}/{FRIDAY.isoformat()}/{run_id}/t.0.jsonl"
    body = "\n".join(json.dumps({"cost_usd": u}) for u in usd).encode()
    fake_s3.put_object(Bucket=BUCKET, Key=key, Body=body)


def _configured(monkeypatch) -> None:
    monkeypatch.setenv(BUCKET_ENV_VAR, BUCKET)
    monkeypatch.setenv(PREFIX_ENV_VAR, PREFIX)
    monkeypatch.delenv(PAGES_TOPIC_ARN_VAR, raising=False)


class TestTheHeartbeatReconcilesTheWeek:
    def test_no_sink_configured_reads_unmeasurable_naming_both_variables(
        self, tmp_path, transport, monkeypatch
    ) -> None:
        monkeypatch.delenv(BUCKET_ENV_VAR, raising=False)
        monkeypatch.delenv(PREFIX_ENV_VAR, raising=False)
        store = LocalStore(tmp_path)
        _write_run(store, job="alerts.sweep", llm_usd=[])
        summary = heartbeat(store, now=SATURDAY_NIGHT, transport=transport)
        row = summary["cost_reconciliation"]
        assert row["name"] == RECONCILIATION_METRIC
        assert row["status"] == "unmeasurable"
        assert BUCKET_ENV_VAR in row["status_reason"] and PREFIX_ENV_VAR in row["status_reason"]
        assert [c.kwargs["severity"] for c in transport.calls] == ["info"]

    def test_agreeing_ledgers_read_ok_and_the_row_validates_on_a_manifest(
        self, tmp_path, transport, monkeypatch, fake_s3
    ) -> None:
        _configured(monkeypatch)
        store = LocalStore(tmp_path)
        _write_run(store, job="alerts.sweep", llm_usd=[])
        _write_run(store, job="experiment.run", llm_usd=[0.25, 0.5], run_id="R" * 26)
        _sink_rows(fake_s3, run_id="R" * 26, usd=[0.25, 0.5])
        summary = heartbeat(store, now=SATURDAY_NIGHT, transport=transport, s3=fake_s3)
        row = summary["cost_reconciliation"]
        assert row["status"] == "OK", row["status_reason"]
        assert row["runs_reconciled"] == 2
        assert row["value"] == 0.75

        def body(ctx) -> None:
            ctx.record_metric(row)

        run_job("heartbeat", body, store=store, trading_day=FRIDAY, now=SATURDAY_NIGHT)
        manifest = json.loads(store.get_bytes(manifest_key("heartbeat", FRIDAY.isoformat())))
        validate(manifest)
        assert RECONCILIATION_METRIC in [m["name"] for m in manifest["metrics"]]

    def test_a_run_whose_sink_rows_are_missing_fails_the_heartbeat_after_it_is_sent(
        self, tmp_path, transport, monkeypatch, fake_s3
    ) -> None:
        """The divergence I7407 / I9694 had: spend the manifest admits to that
        reached no fleet ledger. A failed run, after the proof of life."""
        _configured(monkeypatch)
        store = LocalStore(tmp_path)
        _write_run(store, job="alerts.sweep", llm_usd=[], run_id="S" * 26)
        _write_run(store, job="experiment.run", llm_usd=[0.25])
        with pytest.raises(CostSinkReconciliationError, match="MANIFEST-VS-SINK"):
            heartbeat(store, now=SATURDAY_NIGHT, transport=transport, s3=fake_s3)
        assert transport.pages == 1, "the heartbeat itself still went out"
        assert transport.calls[0].kwargs["severity"] == "error"
        assert "disagree" in transport.calls[0].message

    def test_sink_rows_for_a_run_that_recorded_no_calls_also_fail(
        self, tmp_path, transport, monkeypatch, fake_s3
    ) -> None:
        """The other half: spend that reached the fleet ledger and not the
        manifest. Reconciling only the runs that admit to spending would
        never see it, which is why every manifest of the week is checked."""
        _configured(monkeypatch)
        store = LocalStore(tmp_path)
        _write_run(store, job="alerts.sweep", llm_usd=[])
        _sink_rows(fake_s3, run_id=RUN_ID, usd=[0.10])
        with pytest.raises(CostSinkReconciliationError):
            heartbeat(store, now=SATURDAY_NIGHT, transport=transport, s3=fake_s3)

    def test_a_half_configured_sink_fails_the_run_as_an_access_fault(
        self, tmp_path, transport, monkeypatch
    ) -> None:
        """krepis refuses exactly-one-variable (I5206); so does this, loudly."""
        monkeypatch.setenv(BUCKET_ENV_VAR, BUCKET)
        monkeypatch.delenv(PREFIX_ENV_VAR, raising=False)
        monkeypatch.delenv(PAGES_TOPIC_ARN_VAR, raising=False)
        store = LocalStore(tmp_path)
        _write_run(store, job="alerts.sweep", llm_usd=[])
        with pytest.raises(StoreAccessError, match=PREFIX_ENV_VAR):
            heartbeat(store, now=SATURDAY_NIGHT, transport=transport)
        assert transport.pages == 1

    def test_a_denied_sink_read_fails_the_run_after_it_is_sent(
        self, tmp_path, transport, monkeypatch
    ) -> None:
        class _Denied:
            def get_paginator(self, name):
                raise PermissionError("AccessDenied")

        _configured(monkeypatch)
        store = LocalStore(tmp_path)
        _write_run(store, job="alerts.sweep", llm_usd=[])
        with pytest.raises(StoreAccessError, match="cost-sink read"):
            heartbeat(store, now=SATURDAY_NIGHT, transport=transport, s3=_Denied())
        assert transport.pages == 1

    def test_a_dry_run_still_reconciles_but_sends_nothing(
        self, tmp_path, transport, monkeypatch, fake_s3
    ) -> None:
        _configured(monkeypatch)
        store = LocalStore(tmp_path)
        _write_run(store, job="experiment.run", llm_usd=[0.25])
        _sink_rows(fake_s3, run_id=RUN_ID, usd=[0.25])
        summary = heartbeat(
            store, now=SATURDAY_NIGHT, transport=transport, dry_run=True, s3=fake_s3
        )
        assert summary["cost_reconciliation"]["status"] == "OK"
        assert transport.pages == 0
