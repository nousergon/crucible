"""The fallback manifest must be valid BY CONSTRUCTION, not by hoping.

`alpha-engine-config-I10410` established the invariant and built the
fallback: a validator that can veto the write inverts
manifest-or-it-didn't-happen, so when the assembled manifest will not
validate, a minimal one stands in its place. `crucible-PR192` shipped that
on 2026-09-10.

**It failed the same day, on a path it did not cover.** Measured
2026-09-10T20:51Z on `i-0fd915a1eec8ff5c5`, release `4218c63`:

    ManifestValidationError: run manifest does not conform to run_manifest.v2:
      - reason: String should have at most 2000 characters
    crucible fault.probe exited 1

`runs/fault.probe/2026-09-09/` gained nothing. The run was a real FAILURE and
became an ABSENCE with its cause surviving only in a 90-day CloudWatch log —
precisely the outcome `-I10410` exists to prevent, one field over.

The reason is structural. `_FALLBACK_DROPPED_FIELDS` drops what a JOB
contributes, because one of those is normally what the validator rejected.
But `reason` is not job-contributed in that sense — the runner writes it —
and the fallback EMBEDS the original reason plus the validator's complaint.
So when an over-long `reason` is the cause, the fallback is strictly LONGER
than the document that just failed, fails identically, and that second
`validate()` raises with nothing left to catch it.

A fallback that can fail the same way as the thing it replaces is not a
fallback.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from crucible.manifest import ManifestValidationError, load_schema, validate
from crucible.runner import _fit, _schema_max_length, run_job
from crucible.store import LocalStore

REASON_CAP = load_schema()["properties"]["reason"]["maxLength"]


def test_the_cap_is_read_from_the_schema_not_restated() -> None:
    """A hardcoded 2000 is a second declaration of a fact that already has
    one, and the two diverge the day somebody widens the schema.
    """
    assert _schema_max_length("reason") == REASON_CAP
    # A field with no declared cap must answer None rather than a guess.
    assert _schema_max_length("run_id") is None


def test_fit_is_a_no_op_below_the_cap() -> None:
    assert _fit("short", REASON_CAP) == "short"
    assert _fit("x" * REASON_CAP, REASON_CAP) == "x" * REASON_CAP


def test_fit_marks_where_it_cut() -> None:
    """Truncation is visible on purpose: a reader who cannot tell a complete
    reason from a clipped one chases the missing half as if it were never
    written.
    """
    out = _fit("x" * (REASON_CAP * 3), REASON_CAP)
    assert len(out) <= REASON_CAP
    assert "truncated" in out
    assert out.startswith("x"), "the HEAD is kept — an exception's message leads"


def _overlong_reason() -> str:
    overlong = "the router refused: " + ("member skipped; " * 400)
    assert len(overlong) > REASON_CAP
    return overlong


def test_a_job_whose_reason_exceeds_the_cap_still_writes_a_manifest(tmp_path) -> None:
    """The live case, end to end. Before this change the store was left
    EMPTY and the process exited non-zero, so `crucible.alerts` reported an
    ABSENCE — the condition for "nothing was even attempted".
    """
    store = LocalStore(tmp_path)

    with pytest.raises(RuntimeError):
        run_job(
            "fault.probe",
            lambda ctx: (_ for _ in ()).throw(RuntimeError(_overlong_reason())),
            store=store,
            trading_day=dt.date(2026, 9, 9),
            run_mode="replay",
            transient_retry=False,
        )

    written = store.get_bytes("runs/fault.probe/2026-09-09/run.json")
    manifest = json.loads(written)
    validate(manifest)
    assert manifest["status"] == "failed"
    assert len(manifest["reason"]) <= REASON_CAP


def test_an_overlong_reason_never_costs_the_job_its_own_fields(tmp_path) -> None:
    """The half `crucible-PR200` left open, and the reason this file grew.

    A `reason` too long is the RUNNER's overflow — it renders an exception
    whose message it does not control. Routing that through the fallback made
    a long string cost the run its inputs, outputs, metrics, llm_calls and
    row counts. Measured 2026-09-09 on the live
    `runs/fault.probe/2026-09-09/run.json`: `llm_calls: []`, `metrics: []`,
    `outputs: []`, `cost_usd: 0.0` — every one of them recorded by the run and
    none of them written.

    So the primary assembly fits `reason` to the schema's cap, and this run —
    which used to land in the fallback — now writes the full document.
    """
    store = LocalStore(tmp_path)

    def body(ctx) -> None:
        ctx.record_metric(
            {
                "name": "probe_reached_the_router",
                "module": "crucible.fault_probe",
                "metric_type": "gauge",
                "n_floor": 1,
                "status": "OK",
                "status_reason": "the probe reached the router before it was refused",
                "source_path": "crucible.fault_probe.probe_body",
                "last_updated_utc": "2026-09-09T00:00:00Z",
                "value": 1.0,
                "unit": "bool",
            }
        )
        ctx.rows_in = 7
        raise RuntimeError(_overlong_reason())

    with pytest.raises(RuntimeError):
        run_job(
            "fault.probe",
            body,
            store=store,
            trading_day=dt.date(2026, 9, 9),
            run_mode="replay",
            transient_retry=False,
        )

    manifest = json.loads(store.get_bytes("runs/fault.probe/2026-09-09/run.json"))
    validate(manifest)
    assert manifest["status"] == "failed"
    assert len(manifest["reason"]) <= REASON_CAP
    assert "truncated" in manifest["reason"], "a clipped reason must say it was clipped"
    assert "does not validate" not in manifest["reason"], (
        "an over-long reason must no longer reach the fallback at all — the "
        "fallback is for a JOB-contributed field the validator rejected"
    )
    assert manifest["rows_in"] == 7, "the job's own fields must survive"
    assert [m["name"] for m in manifest["metrics"]] == ["probe_reached_the_router"]


def test_the_fallback_keeps_the_validators_complaint_not_only_the_original(
    tmp_path,
) -> None:
    """What an operator needs in order to FIX the producer is the validator's
    complaint; what they came for is the original reason. When both cannot
    fit, the complaint leads — the original is recoverable from the box log,
    the complaint names the field nobody would otherwise know to look at.
    """
    store = LocalStore(tmp_path)

    def body(ctx) -> None:
        # A JOB-contributed violation, which is what the fallback is FOR:
        # `rows_in` is declared `minimum: 0`. An over-long `reason` no longer
        # reaches here — the primary assembly fits it — so triggering the
        # fallback with one would be testing a path that can no longer occur.
        ctx.rows_in = -1
        raise RuntimeError("the job also failed, and this is why")

    with pytest.raises(RuntimeError):
        run_job(
            "fault.probe",
            body,
            store=store,
            trading_day=dt.date(2026, 9, 9),
            run_mode="replay",
            transient_retry=False,
        )
    manifest = json.loads(store.get_bytes("runs/fault.probe/2026-09-09/run.json"))
    assert "does not validate" in manifest["reason"]
    assert "The validator said" in manifest["reason"]
    assert "the job also failed" in manifest["reason"]


def test_the_fallback_itself_never_raises_validation(tmp_path) -> None:
    """The property that makes it a fallback. Whatever the job did to its own
    manifest, the replacement validates — so `run_job`'s `finally` always
    leaves a document behind and the exception that propagates is the JOB's,
    never the validator's.
    """
    store = LocalStore(tmp_path)
    for length in (REASON_CAP - 1, REASON_CAP, REASON_CAP + 1, REASON_CAP * 10):
        day = dt.date(2026, 9, 9)
        with pytest.raises(RuntimeError) as caught:
            run_job(
                "fault.probe",
                lambda ctx, n=length: (_ for _ in ()).throw(RuntimeError("q" * n)),
                store=store,
                trading_day=day,
                run_mode="replay",
                transient_retry=False,
            )
        assert not isinstance(caught.value, ManifestValidationError), (
            f"a reason of {length} characters let the validator's exception escape "
            "instead of the job's"
        )
        validate(json.loads(store.get_bytes("runs/fault.probe/2026-09-09/run.json")))
