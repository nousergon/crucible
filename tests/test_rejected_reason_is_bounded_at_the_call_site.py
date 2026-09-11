"""A job-contributed field with a schema cap is bounded where the job writes it.

`alpha-engine-config-I10484`, the producer-side half of `-I10482`.

`run_manifest.v2.json` caps one rejection reason at 200 characters. Over that,
the manifest fails validation at write time and the write routes through
`_minimal_failed_manifest`, whose `_FALLBACK_DROPPED_FIELDS` includes
`rows_rejected` — so the run loses `inputs`, `outputs`, `metrics`, `llm_calls`
and every row count because one string was long.

That fallback behaviour is correct and deliberate for a job-contributed field
(`crucible-PR201` reserved it for exactly this). What was wrong is that the job
could introduce the violation silently and only learn of it as a schema path in
a manifest that had already lost everything else.

Raising rather than fitting is the other half, and it turns on who owns the
string. The manifest's own `reason` is FITTED by `_write_manifest`: the runner
renders it from an exception whose message it does not control. A rejection
reason is authored by the job, which can shorten it — clipping it silently
would discard a category the job chose, at the one place §9.2 class 4 says the
category is the point.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from crucible.manifest import load_schema, validate
from crucible.runner import _schema_max_length, run_job
from crucible.store import LocalStore

ROW_CAP = load_schema()["$defs"]["RejectedRow"]["properties"]["reason"]["maxLength"]


def test_the_cap_is_read_from_the_schema_not_restated() -> None:
    """Two different caps on two fields both named `reason`, reached by two
    different callers. A literal in either place diverges the day one moves.
    """
    assert _schema_max_length("reason", defs="RejectedRow") == ROW_CAP
    assert _schema_max_length("reason") == load_schema()["properties"]["reason"]["maxLength"]
    assert _schema_max_length("reason") != ROW_CAP, (
        "the manifest reason and a rejection reason must not share a cap by accident"
    )
    # An absent field, and an absent $defs entry, answer None rather than guess.
    assert _schema_max_length("run_id") is None
    assert _schema_max_length("reason", defs="NoSuchDef") is None


def test_a_reason_at_the_cap_is_accepted(tmp_path) -> None:
    store = LocalStore(tmp_path)

    def body(ctx) -> None:
        ctx.record_rejected("x" * ROW_CAP, 3)

    run_job(
        "data.daily",
        body,
        store=store,
        trading_day=dt.date(2026, 9, 9),
        run_mode="replay",
    )
    manifest = json.loads(store.get_bytes("runs/data.daily/2026-09-09/run.json"))
    validate(manifest)
    assert manifest["rows_rejected"] == [{"reason": "x" * ROW_CAP, "count": 3}]


def test_a_reason_over_the_cap_raises_where_the_job_wrote_it(tmp_path) -> None:
    """And the raise names both numbers, so the fix is obvious without opening
    the schema."""
    store = LocalStore(tmp_path)
    overlong = "y" * (ROW_CAP + 1)

    def body(ctx) -> None:
        ctx.record_rejected(overlong, 1)

    with pytest.raises(ValueError) as caught:
        run_job(
            "data.daily",
            body,
            store=store,
            trading_day=dt.date(2026, 9, 9),
            run_mode="replay",
        )
    message = str(caught.value)
    assert str(ROW_CAP) in message
    assert str(ROW_CAP + 1) in message


def test_the_failure_is_the_jobs_own_and_the_manifest_keeps_its_fields(tmp_path) -> None:
    """The point of moving the bound earlier. The run still fails — a job that
    tried to record an over-long category has a defect — but it fails as
    ITSELF, with a manifest that kept everything else it recorded, instead of
    as a validation error over a document stripped of every job field.
    """
    store = LocalStore(tmp_path)

    def body(ctx) -> None:
        ctx.record_rows(rows_in=900, rows_out=898)
        ctx.record_rejected("z" * (ROW_CAP * 5), 2)

    with pytest.raises(ValueError):
        run_job(
            "data.daily",
            body,
            store=store,
            trading_day=dt.date(2026, 9, 9),
            run_mode="replay",
            transient_retry=False,
        )
    manifest = json.loads(store.get_bytes("runs/data.daily/2026-09-09/run.json"))
    validate(manifest)
    assert manifest["status"] == "failed"
    assert manifest["rows_in"] == 900, "the job's own fields must survive"
    assert manifest["rows_out"] == 898
    assert "does not validate" not in manifest["reason"], (
        "the fallback must not be reached — the violation was refused before the write"
    )
    assert "at most" in manifest["reason"], "the manifest must carry the refusal's own words"


def test_an_empty_reason_is_still_refused(tmp_path) -> None:
    """The pre-existing guard, unchanged: a bare count is unactionable."""
    store = LocalStore(tmp_path)
    with pytest.raises(ValueError, match="bare count"):
        run_job(
            "data.daily",
            lambda ctx: ctx.record_rejected("", 1),
            store=store,
            trading_day=dt.date(2026, 9, 9),
            run_mode="replay",
        )
