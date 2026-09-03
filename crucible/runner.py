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
import signal
import subprocess
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from crucible.calendar import assert_trading_day, resolve_trading_day
from crucible.manifest import (
    RUN_MANIFEST_SCHEMA_VERSION,
    STATUSES,
    manifest_key,
    validate,
)
from crucible.runmode import resolve_run_mode
from crucible.store import Store, sha256_hex

__all__ = [
    "MAX_ATTEMPTS",
    "RunContext",
    "SpotInterruptionError",
    "TRANSIENT_CLASSIFIERS",
    "classify_transient",
    "run_job",
    "spot_interruption_guard",
]

#: One retry, never a loop (§11 risk 2). Two attempts is the whole ladder:
#: a transient fault that survives a fresh instance is not transient, and a
#: retry budget larger than one is how a job that will never succeed burns
#: an afternoon of spot time before anyone is told.
MAX_ATTEMPTS = 2


class SpotInterruptionError(BaseException):
    """The instance is being reclaimed.

    Derives from BaseException, not Exception, deliberately: it is raised
    from a SIGTERM handler, and a job's own `except Exception` must not be
    able to swallow the reclamation and carry on writing to a machine that
    is about to disappear. The runner catches BaseException, so the manifest
    is still written.
    """


@contextmanager
def spot_interruption_guard() -> Iterator[None]:
    """Turn SIGTERM into :class:`SpotInterruptionError` for the duration.

    A spot reclamation arrives as SIGTERM about two minutes before the
    instance goes away. Without this the process dies with no manifest at
    all — which surfaces as an ABSENCE page with the cause discarded,
    rather than as a FAILURE page naming the reclamation and retried once
    on a fresh instance.

    Restores the previous handler on exit, including on the raise, so a
    caller that wraps two runs does not leave the second one holding the
    first one's handler. Signal handlers can only be installed on the main
    thread; off the main thread this is a documented no-op rather than a
    crash, because refusing to run a job for want of a signal handler would
    trade a real failure mode for a certain one.
    """

    def _handler(signum: int, frame: Any) -> None:
        raise SpotInterruptionError(
            f"spot_interruption: received signal {signum}; the instance is being reclaimed"
        )

    try:
        previous = signal.signal(signal.SIGTERM, _handler)
    except ValueError:
        yield
        return
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


#: **The declared transient class (§11 risk 2), frozen.** A failure outside
#: this set pages immediately. Each row is (reason, exception-type names,
#: message substrings); a match on EITHER the type or a substring claims the
#: failure. The reasons are exactly the `attempts[].reason` enum in the
#: manifest schema, so a class that is not in the schema cannot be recorded
#: and therefore cannot be retried.
#:
#: **This tuple grows only by PR, with the failure named.** That is the whole
#: control: a retry class that can be widened at runtime is a retry class
#: that eventually swallows a real defect, because the moment a defect looks
#: transient is the moment someone is under pressure to make it go away.
TRANSIENT_CLASSIFIERS: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    (
        "spot_interruption",
        ("SpotInterruptionError",),
        ("spot_interruption", "instance is being reclaimed"),
    ),
    (
        "provider_timeout",
        ("ReadTimeout", "ConnectTimeout", "ReadTimeoutError", "TimeoutError"),
        ("provider_timeout", "read timeout", "connection timed out"),
    ),
    (
        "provider_5xx",
        (),
        (
            "provider_5xx",
            " 500",
            " 502",
            " 503",
            " 504",
            "internal server error",
            "bad gateway",
            "service unavailable",
        ),
    ),
    (
        "s3_throttling",
        ("SlowDown",),
        ("s3_throttling", "slowdown", "requestlimitexceeded", "reduce your request rate"),
    ),
)


def classify_transient(exc: BaseException) -> str | None:
    """The declared class of ``exc``, or None if it is not in the class.

    None is the default and the safe answer: an unrecognised failure pages
    immediately. A classifier whose fall-through was "probably transient"
    would retry real defects and halve the rate at which they are noticed.
    """
    names = {type(exc).__name__, *(b.__name__ for b in type(exc).__mro__)}
    haystack = f"{type(exc).__name__}: {exc}".lower()
    for reason, types, needles in TRANSIENT_CLASSIFIERS:
        if names & set(types):
            return reason
        if any(n in haystack for n in needles):
            return reason
    return None


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

    #: `live` or `replay`, resolved by `run_job` from the INVOCATION before
    #: the job body runs — never from `trading_day` or `calendar_date`, which
    #: are identical for a live Saturday and for a replay of one. A job body
    #: may read it (a replay legitimately behaves differently) but never
    #: assigns it: the mode is a property of how the process was launched.
    #:
    #: The empty default is not a value — `run_manifest.v2` has no empty
    #: member in its `run_mode` enum, so a context that reached the write path
    #: without `run_job` setting it fails validation loudly instead of filing
    #: a run under a mode nobody declared.
    run_mode: str = ""

    #: Set by `run_job` from its own `discriminator` argument, once
    #: `calendar_date` is known — never by the job body. Mirrors
    #: `crucible.manifest.manifest_key`'s `discriminator`: absent for a job
    #: that writes at most one manifest per trading day (alpha-engine-config-I9781).
    discriminator: str | None = None

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

    def record_output_cas(
        self, key: str, expected: str, payload: bytes, schema_version: str = "v1"
    ) -> str:
        """Write ``payload`` at ``key`` by compare-and-swap, and record it as
        an output.

        For a pointer more than one actor can write — a champion pointer, a
        release pointer — a bare :meth:`record_output` is the wrong tool
        twice over: it writes last-writer-wins through :meth:`Store.put_bytes`
        (§4.11's whole reason for :meth:`Store.compare_and_swap` to exist),
        and a caller that reaches for `store.put_bytes` directly to get the
        CAS semantics back skips the lineage half, so the pointer never
        enters `outputs[]` and `crucible explain` cannot name the run that
        moved it. This method is the one call that gets both: the write is
        conditional, and win or lose, a successful write is recorded exactly
        like any other output.

        Raises :class:`~crucible.store.PointerConflictError` on a lost race,
        same as a bare `compare_and_swap` — never retried here, for the same
        reason the store itself never retries: the caller re-reads and
        decides rather than overwriting whatever the winner just published.
        """
        digest = self.store.compare_and_swap(key, expected, payload)
        self.outputs.append({"key": key, "sha256": digest, "schema_version": schema_version})
        return digest


def run_job(
    job: str,
    fn: Callable[[RunContext], Any],
    *,
    store: Store,
    trading_day: dt.date | None = None,
    now: dt.datetime | None = None,
    seed: int | None = None,
    release_sha: str | None = None,
    transient_retry: bool = True,
    discriminator: str | Callable[[RunContext], str] | None = None,
    run_mode: str | None = None,
) -> RunContext:
    """Run ``fn`` as job ``job`` and write its manifest, whatever happens.

    ``discriminator`` distinguishes concurrent or repeated writers of the
    same `job`+`trading_day` (alpha-engine-config-I9781) — see
    `crucible.manifest.manifest_key`. Pass a plain string when the value is
    known before the run starts (a `--slot`); pass a callable taking the
    freshly-built :class:`RunContext` when it can only be derived after
    `trading_day`/`calendar_date` resolve (`alerts.sweep`'s own
    `calendar_date`, so Friday/Saturday/Sunday firings that all resolve to
    Friday's trading day still write three distinct manifests). Omitted for
    every job that writes at most one manifest per trading day, which is
    every job but those two today.

    ``trading_day`` may be passed explicitly (a backfill, a replay); it is
    then *asserted*, never silently corrected. A caller that asked for a
    Saturday has a bug, and quietly resolving it to Friday hides the bug
    while producing plausible output. When it is omitted the day is resolved
    from ``now``.

    **One retry, for one declared class** (§11 risk 2). When the first
    attempt raises something :func:`classify_transient` recognises — a spot
    reclamation, a provider 5xx or timeout, an S3 throttle — the job runs once
    more with a **fresh** :class:`RunContext`, and both attempts appear in
    ``manifest.attempts[]``. Exactly one manifest is written either way, so a
    retried success is visibly a retried success rather than a clean first
    attempt, and a page fires only when the retry also fails.

    The retry rests on jobs being idempotent, which is what content-addressed
    outputs are for: a rerun producing identical bytes is a no-op. A job that
    is not idempotent is a defect in that job, not a reason to remove the
    retry — and the fresh context is what makes the second attempt's lineage
    its own rather than a merge of two runs.

    ``transient_retry=False`` disables it for a caller that must see the first
    failure — the fault-injection suite asserts the no-retry path as well as
    the retry path.

    ``run_mode`` is `live` or `replay` and is REQUIRED, in the sense that
    omitting it here falls through to ``$CRUCIBLE_RUN_MODE`` and then to a
    refusal (:class:`crucible.runmode.RunModeError`) — there is no default at
    any layer, and the resolution never looks at the date. A replay of a past
    Saturday is indistinguishable from a live one by `trading_day`, and phase
    2's exit gate counts live Saturdays from this field, so a guessed value
    would be a false claim about production in the one place that matters
    (alpha-engine-config-I9918). Resolved BEFORE the job body runs, so an
    undeclared invocation costs nothing.

    Returns the :class:`RunContext` on success. Re-raises on failure, after
    the manifest is on disk.
    """
    started = now or dt.datetime.now(dt.UTC)
    # Resolved first, before the trading day and before any work: an
    # invocation that never said whether it was live or a replay is refused
    # while refusing is still free.
    resolved_run_mode = resolve_run_mode(run_mode)
    if trading_day is None:
        trading_day = resolve_trading_day(started)
    else:
        assert_trading_day(trading_day, context=manifest_key(job, trading_day.isoformat()))

    resolved_seed = seed if seed is not None else int(trading_day.strftime("%Y%m%d"))
    attempts: list[dict[str, Any]] = [{"n": 1, "reason": "initial"}]

    while True:
        ctx = RunContext(
            run_id=_new_run_id(dt.datetime.now(dt.UTC) if now is None else now),
            job=job,
            trading_day=trading_day,
            calendar_date=started.date(),
            store=store,
            seed=resolved_seed,
            started=started,
            run_mode=resolved_run_mode,
        )
        ctx.attempts = [dict(a) for a in attempts]
        ctx.discriminator = discriminator(ctx) if callable(discriminator) else discriminator

        status = "ok"
        reason = ""
        transient: str | None = None
        try:
            # The guard is installed HERE, inside the runner, rather than left
            # to each call site. `spot_interruption_guard` is defined in this
            # module and only `track_c.py`'s smoke job ever wrapped a call
            # with it — every other job, including `data.weekly`, the
            # 1-3 hour job plan §4.7 puts on a spot instance, ran unguarded.
            # A guard every caller must remember to install is a guard that
            # is missing from whichever caller was written last; the manifest
            # guarantee lives in `run_job`, so the guard does too. Nesting is
            # safe — a caller that also wraps `run_job` itself (as smoke
            # still does, harmlessly) restores its own handler on exit same
            # as this one does.
            with spot_interruption_guard():
                fn(ctx)
        except BaseException as exc:
            # BaseException, not Exception: a spot reclamation arrives as a
            # signal, so catching only Exception would leave the fleet's single
            # most common real failure with no manifest at all — an ABSENCE page
            # instead of a FAILURE page, with the cause discarded.
            status = "failed"
            reason = _reason_from(exc)
            if transient_retry and len(attempts) < MAX_ATTEMPTS:
                transient = classify_transient(exc)
            if transient is None:
                raise
        finally:
            # `finally`, so the manifest exists on both paths — EXCEPT on the
            # one path where another attempt is about to run and will write the
            # manifest itself. Writing a `failed` manifest for attempt 1 and an
            # `ok` one for attempt 2 at the same key would leave the store's
            # answer to "did this run work" decided by write ordering, which is
            # the last-writer-wins shape this system is built to refuse.
            if transient is None:
                _write_manifest(
                    ctx,
                    store=store,
                    status=status,
                    reason=reason,
                    started=started,
                    now=now,
                    release_sha=release_sha,
                )

        if transient is None:
            return ctx
        attempts.append({"n": len(attempts) + 1, "reason": transient})


def _write_manifest(
    ctx: RunContext,
    *,
    store: Store,
    status: str,
    reason: str,
    started: dt.datetime,
    now: dt.datetime | None,
    release_sha: str | None,
) -> dict[str, Any]:
    """Assemble, validate and write one manifest. The single writer.

    Validated BEFORE the write. A non-conformant manifest is a bug in the
    runner, and it must surface here rather than at read time on a Saturday
    morning.
    """
    finished = dt.datetime.now(dt.UTC) if now is None else now
    manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": ctx.run_id,
        "job": ctx.job,
        # From the invocation, carried on the context since before the job
        # body ran. Never recomputed here, and never derived from either date
        # field below.
        "run_mode": ctx.run_mode,
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
    if ctx.discriminator is not None:
        # Omitted entirely rather than written as null: the schema declares
        # it optional, not nullable, so a job with one writer per trading day
        # produces a manifest byte-identical to one from before I9781.
        manifest["discriminator"] = ctx.discriminator
    # cost_usd must be >= the sum of llm_calls[].usd (schemas/run_manifest.v1.json,
    # `cost_usd` description) — a schema cannot cross-reference two fields of
    # the same document, so the runner asserts it here, before the write, per
    # that field's own claim. `record_llm_call` always adds its `usd` to
    # `cost_usd`, so this can only fail if a job's own bookkeeping (a
    # negative `record_cost`, most plausibly) pulled the total back below
    # what the calls themselves report — a defect worth surfacing at write
    # time, not discovered by a reader doing the arithmetic on a Saturday
    # morning.
    llm_usd_total = round(sum(float(call.get("usd", 0.0)) for call in ctx.llm_calls), 6)
    if manifest["cost_usd"] + 1e-9 < llm_usd_total:
        raise ValueError(
            f"cost_usd ({manifest['cost_usd']}) is less than the sum of llm_calls[].usd "
            f"({llm_usd_total}) for run {ctx.run_id} ({ctx.job}). The schema's own "
            "description promises the runner asserts this; it must never be false."
        )
    validate(manifest)
    store.put_bytes(
        manifest_key(ctx.job, ctx.trading_day.isoformat(), discriminator=ctx.discriminator),
        json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"),
    )
    return manifest


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
