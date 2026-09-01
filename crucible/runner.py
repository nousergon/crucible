"""The job wrapper. Every job in the system runs through this function.

Normative source: plan §4.2, §4.6, §9.2.

One guarantee, and everything else is in service of it:

    **A job writes its manifest, whether it succeeds or dies.**

The write happens in a ``finally``, so an exception produces a `failed`
manifest carrying its cause, its spend and whatever lineage the job had
accumulated — and then the exception continues to propagate, so the process
exits non-zero and the scheduler sees a failure. `try/finally`, never
`try/except`: swallowing here would turn a failure into a silent success,
which is the single defect the whole plan is a reaction to.

**There is no third status, and no API to request one.** The status is
derived from whether the callable returned or raised. A job cannot declare
itself skipped, partial or degraded, because there is nothing to call. That
makes §11.1 structural rather than a rule someone has to remember.

**The manifest is validated before it is written.** A runner that could emit
a non-conformant document defeats the schema, and the failure path — where
fields are missing because the job never reached them — is exactly where
that would happen.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import random
import subprocess
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from crucible.calendar import assert_trading_day, resolve_trading_day
from crucible.manifest import (
    RUN_MANIFEST_SCHEMA_VERSION,
    STATUSES,
    manifest_key,
    validate,
)
from crucible.store import Store, sha256_hex

__all__ = ["RunContext", "run_job"]

_ULID_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"  # Crockford base32
_UNKNOWN_SHA = "0" * 40


def _new_run_id(now: dt.datetime) -> str:
    """A ULID: 48 bits of millisecond timestamp then 80 bits of randomness.

    Lexically sortable by creation time, which is what makes a listing of
    run ids readable without parsing them.
    """
    ms = int(now.timestamp() * 1000)
    rand = random.getrandbits(80)
    value = (ms << 80) | rand
    out = []
    for _ in range(26):
        out.append(_ULID_ALPHABET[value & 0x1F])
        value >>= 5
    return "".join(reversed(out))


def _code_sha() -> str:
    """The commit that is running.

    Falls back to the all-zero sha when git is unavailable (inside a wheel on
    a spot box, there is no repository). That is a *declared* unknown carried
    in a required field, not an omitted field: the manifest still validates
    and `explain` still reports honestly that the sha could not be read.
    """
    env = os.environ.get("CRUCIBLE_CODE_SHA")
    if env and len(env) == 40:
        return env
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
    except (OSError, subprocess.SubprocessError):
        return _UNKNOWN_SHA
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and len(sha) == 40 else _UNKNOWN_SHA


def _utc(now: dt.datetime) -> str:
    return now.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class RunContext:
    """What a job is handed, and the only way it contributes to its manifest.

    Every method here adds telemetry. None of them can change the status:
    that is derived, not declared.
    """

    run_id: str
    job: str
    trading_day: dt.date
    calendar_date: dt.date
    store: Store
    seed: int
    started: dt.datetime

    inputs: list[dict[str, Any]] = field(default_factory=list)
    outputs: list[dict[str, Any]] = field(default_factory=list)
    rows_in: int = 0
    rows_out: int = 0
    rows_rejected: list[dict[str, Any]] = field(default_factory=list)
    cost_usd: float = 0.0
    llm_calls: list[dict[str, Any]] = field(default_factory=list)
    metrics: list[dict[str, Any]] = field(default_factory=list)
    attempts: list[dict[str, Any]] = field(default_factory=lambda: [{"n": 1, "reason": "initial"}])
    resource: dict[str, Any] = field(
        default_factory=lambda: {
            "instance_type": os.environ.get("CRUCIBLE_INSTANCE_TYPE", "local"),
            "spot": os.environ.get("CRUCIBLE_SPOT", "").lower() == "true",
            "escalated_to_on_demand": False,
            "interruptions": 0,
            "mem_peak_mb": 0.0,
            "disk_free_mb": 0.0,
        }
    )

    def set_status(self, status: str) -> None:
        """Always raises. Present so the attempt is a loud, findable error.

        A job may not declare its own status. If this method did not exist,
        someone would add it; existing and refusing is what makes the refusal
        greppable and the reason readable at the call site.
        """
        raise ValueError(
            f"a job cannot set its own status (asked for {status!r}). The status is "
            f"derived from whether the job returned or raised, and is one of {STATUSES}. "
            "A job with nothing to do still produced a complete, correct result — that "
            "is `ok` — and a job that could not do its work raises."
        )

    def record_input(self, key: str, payload: bytes, schema_version: str = "v1") -> None:
        self.inputs.append(
            {"key": key, "sha256": sha256_hex(payload), "schema_version": schema_version}
        )

    def record_output(self, key: str, payload: bytes, schema_version: str = "v1") -> None:
        """Write ``payload`` to the store and record it as an output."""
        digest = self.store.put_bytes(key, payload)
        self.outputs.append({"key": key, "sha256": digest, "schema_version": schema_version})

    def record_rows(self, *, rows_in: int, rows_out: int) -> None:
        self.rows_in = rows_in
        self.rows_out = rows_out

    def record_rejected(self, reason: str, count: int) -> None:
        """§9.2 class 4: rejections carry a reason, always. A bare count
        cannot be acted on."""
        if not reason:
            raise ValueError("a rejected-row entry needs a reason; a bare count is unactionable")
        self.rows_rejected.append({"reason": reason, "count": count})

    def record_cost(self, usd: float) -> None:
        self.cost_usd += usd

    def record_llm_call(self, call: dict[str, Any]) -> None:
        """§9.2 class 2. The caller supplies the record; the runner adds its
        `usd` to the run total so spend is never counted in one place only."""
        self.llm_calls.append(call)
        self.cost_usd += float(call.get("usd", 0.0))

    def record_metric(self, metric: dict[str, Any]) -> None:
        """§9.2 class 5. MetricRecord-shaped; the schema enforces that a
        value carries a unit and that a horizon is in trading days."""
        self.metrics.append(metric)


def run_job(
    job: str,
    fn: Callable[[RunContext], Any],
    *,
    store: Store,
    trading_day: dt.date | None = None,
    now: dt.datetime | None = None,
    seed: int | None = None,
    release_sha: str | None = None,
) -> RunContext:
    """Run ``fn`` as job ``job`` and write its manifest, whatever happens.

    ``trading_day`` may be passed explicitly (a backfill, a replay); it is
    then *asserted*, never silently corrected. A caller that asked for a
    Saturday has a bug, and quietly resolving it to Friday hides the bug
    while producing plausible output. When it is omitted the day is resolved
    from ``now``.

    Returns the :class:`RunContext` on success. Re-raises on failure, after
    the manifest is on disk.
    """
    started = now or dt.datetime.now(dt.UTC)
    if trading_day is None:
        trading_day = resolve_trading_day(started)
    else:
        assert_trading_day(trading_day, context=manifest_key(job, trading_day.isoformat()))

    calendar_date = started.date()
    resolved_seed = seed if seed is not None else int(trading_day.strftime("%Y%m%d"))

    ctx = RunContext(
        run_id=_new_run_id(started),
        job=job,
        trading_day=trading_day,
        calendar_date=calendar_date,
        store=store,
        seed=resolved_seed,
        started=started,
    )

    status = "ok"
    reason = ""
    try:
        fn(ctx)
    except BaseException as exc:
        # BaseException, not Exception: a spot reclamation arrives as a
        # signal, so catching only Exception would leave the fleet's single
        # most common real failure with no manifest at all — an ABSENCE page
        # instead of a FAILURE page, with the cause discarded.
        status = "failed"
        reason = _reason_from(exc)
        raise
    finally:
        # `finally`, so the manifest exists on both paths. The exception is
        # NOT swallowed: the raise above continues after this block runs.
        finished = dt.datetime.now(dt.UTC) if now is None else now
        manifest = {
            "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
            "run_id": ctx.run_id,
            "job": ctx.job,
            "trading_day": ctx.trading_day.isoformat(),
            "calendar_date": ctx.calendar_date.isoformat(),
            "status": status,
            "reason": reason,
            "started": _utc(started),
            "finished": _utc(finished),
            "code_sha": _code_sha(),
            "release_sha": release_sha or os.environ.get("CRUCIBLE_RELEASE_SHA") or _code_sha(),
            "seed": ctx.seed,
            "inputs": ctx.inputs,
            "outputs": ctx.outputs,
            "rows_in": ctx.rows_in,
            "rows_out": ctx.rows_out,
            "rows_rejected": ctx.rows_rejected,
            "cost_usd": round(ctx.cost_usd, 6),
            "llm_calls": ctx.llm_calls,
            "resource": ctx.resource,
            "metrics": ctx.metrics,
            "attempts": ctx.attempts,
        }
        # Validated BEFORE the write. A non-conformant manifest is a bug in
        # the runner, and it must surface here rather than at read time on a
        # Saturday morning.
        validate(manifest)
        store.put_bytes(
            manifest_key(job, ctx.trading_day.isoformat()),
            json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"),
        )

    return ctx


def _reason_from(exc: BaseException) -> str:
    """An operator-readable cause that is never empty.

    `raise ValueError()` carries no message, and a `reason` of `""` on a
    failure is the shape that made three consecutive Saturday failures
    indistinguishable from each other. The type name is always present, and
    the innermost frame is appended so the reason names a line.
    """
    kind = type(exc).__name__
    message = str(exc).strip()
    tb = exc.__traceback__
    location = ""
    if tb is not None:
        frames = traceback.extract_tb(tb)
        if frames:
            last = frames[-1]
            location = f" at {os.path.basename(last.filename)}:{last.lineno}"
    if message:
        return f"{kind}: {message}{location}"
    return f"{kind} raised with no message{location}"
