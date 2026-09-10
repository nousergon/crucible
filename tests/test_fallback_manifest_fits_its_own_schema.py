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


def test_a_job_whose_reason_exceeds_the_cap_still_writes_a_manifest(tmp_path) -> None:
    """The live case, end to end. Before this change the store was left
    EMPTY and the process exited non-zero, so `crucible.alerts` reported an
    ABSENCE — the condition for "nothing was even attempted".
    """
    store = LocalStore(tmp_path)
    overlong = "the router refused: " + ("member skipped; " * 400)
    assert len(overlong) > REASON_CAP

    with pytest.raises(RuntimeError):
        run_job(
            "fault.probe",
            lambda ctx: (_ for _ in ()).throw(RuntimeError(overlong)),
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


def test_the_fallback_keeps_the_validators_complaint_not_only_the_original(
    tmp_path,
) -> None:
    """What an operator needs in order to FIX the producer is the validator's
    complaint; what they came for is the original reason. When both cannot
    fit, the complaint leads — the original is recoverable from the box log,
    the complaint names the field nobody would otherwise know to look at.
    """
    store = LocalStore(tmp_path)
    with pytest.raises(RuntimeError):
        run_job(
            "fault.probe",
            lambda ctx: (_ for _ in ()).throw(RuntimeError("z" * (REASON_CAP * 2))),
            store=store,
            trading_day=dt.date(2026, 9, 9),
            run_mode="replay",
            transient_retry=False,
        )
    manifest = json.loads(store.get_bytes("runs/fault.probe/2026-09-09/run.json"))
    assert "does not validate" in manifest["reason"]
    assert "The validator said" in manifest["reason"]


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
