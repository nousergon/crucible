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
import re
import resource as posix_resource
import shutil
import signal
import subprocess
import sys
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from crucible.calendar import assert_trading_day, resolve_trading_day
from crucible.manifest import (
    RUN_MANIFEST_SCHEMA_VERSION,
    STATUSES,
    ManifestValidationError,
    load_schema,
    manifest_key,
    validate,
)
from crucible.runmode import resolve_run_mode
from crucible.store import Store, sha256_hex

__all__ = [
    "MAX_ATTEMPTS",
    "CODE_SHA_ENV",
    "CodeShaError",
    "RunContext",
    "SPOT_INTERRUPTION_REASON",
    "SpotInterruptionError",
    "TRANSIENT_CLASSIFIERS",
    "classify_transient",
    "resolve_code_sha",
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

    Without this the process dies with no manifest at all — which surfaces
    as an ABSENCE page with the cause discarded, rather than as a FAILURE
    page naming the reclamation and retried once inside the notice window.

    **Who delivers that SIGTERM, corrected 2026-09-10
    (`alpha-engine-config-I10438`).** This docstring used to say "a spot
    reclamation arrives as SIGTERM about two minutes before the instance
    goes away", and AWS does no such thing: a reclamation is announced on
    IMDS at `/latest/meta-data/spot/instance-action` and carries no signal.
    The only SIGTERM AWS causes arrives at OS shutdown, at the END of the
    two-minute notice, which leaves the retry the shutdown grace period
    rather than two minutes — so the retry this guard exists to enable
    could never complete, and `transient_retry_class_in_runner` read
    UNMEASURABLE for as long as the premise stood (112 manifests across 24
    jobs, not one attempt past the first).

    The two minutes are real only because the box's own shell polls that
    IMDS path and signals THIS process on a notice
    (`nous-ergon-ops/infrastructure/cloudformation/crucible-v2.yaml`,
    `_user_data`). The guard is therefore correct and the wiring is what
    was missing; stated here because a docstring naming the wrong signal
    source is how the gap survived being read.

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
#:
#: `SPOT_INTERRUPTION_REASON` names the first row's reason once, rather than
#: repeating the string literal at both this definition and wherever
#: `resource.interruptions` is derived from `attempts[]` below
#: (`alpha-engine-config-I10463`) — the two would otherwise be two spellings
#: of the same fact with no test forcing them to agree.
SPOT_INTERRUPTION_REASON = "spot_interruption"

TRANSIENT_CLASSIFIERS: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    (
        SPOT_INTERRUPTION_REASON,
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


#: A real, non-placeholder git sha: forty lowercase hex characters, and
#: explicitly NOT the all-zero placeholder (`alpha-engine-config-I10454`).
#: `run_manifest.v2`'s `code_sha` pattern mirrors this exactly
#: (`crucible.models._CODE_SHA_PATTERN`) — the two are asserted equal by
#: `tests/test_runner.py`, so this module and the schema cannot drift into
#: two different ideas of "real".
_REAL_SHA_RE = re.compile(r"^(?!0{40}$)[0-9a-f]{40}$")

#: The box's dispatcher exports this from the SAME `releases/current` sha it
#: already reads `CRUCIBLE_RELEASE_SHA` from
#: (`nous-ergon-ops/infrastructure/cloudformation/crucible-v2.yaml`) — a
#: wheel install carries no git checkout, so the commit it was built from has
#: to be carried IN the release rather than read off a working tree that
#: does not exist there.
CODE_SHA_ENV = "CRUCIBLE_CODE_SHA"


class CodeShaError(RuntimeError):
    """`code_sha` could not be resolved to a real, measured commit sha.

    Raised, never defaulted around (repo rule 5): a `0`*40 placeholder used
    to validate and answer nothing (`alpha-engine-config-I10454`) — half of
    `explain`'s answer to "why did it do that" was silently absent on every
    manifest a dispatched box ever wrote. Raised BEFORE `run_job` writes
    anything, mirroring `crucible.runmode.RunModeError`'s shape: refusing
    here, before any work starts, is what keeps this refusal from colliding
    with "manifest or it did not happen" — the process never reaches a job
    that would need one.
    """


def resolve_code_sha() -> str:
    """The commit sha of the crucible tree that is running, or raise
    :class:`CodeShaError`.

    `$CRUCIBLE_CODE_SHA` wins when set — the box's own answer, carried in
    the release rather than read from a working tree a wheel install does
    not have. Off the box (a laptop or CI run, inside a real git checkout)
    the variable is normally unset and `git rev-parse HEAD` in the tree this
    module ships from is the real answer.

    Either source producing something other than a real 40-character
    lowercase git sha — unset and no git, a malformed export, a detached
    checkout with no commits — is refused rather than written as the
    all-zero placeholder that used to validate and answer nothing.
    """
    env = os.environ.get(CODE_SHA_ENV)
    if env is not None:
        if not _REAL_SHA_RE.match(env):
            raise CodeShaError(
                f"${CODE_SHA_ENV}={env!r} is not a real 40-character lowercase git sha (or "
                "is the all-zero placeholder). The box's dispatcher exports this from the "
                "sha under `releases/current`; a malformed value there is a deploy-time "
                "defect, and code_sha cannot be written as a value nobody measured "
                "(repo rule 5)."
            )
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
    except (OSError, subprocess.SubprocessError) as exc:
        raise CodeShaError(
            f"${CODE_SHA_ENV} is unset and `git rev-parse HEAD` could not run ({exc}). "
            "code_sha cannot be written as a value nobody measured (repo rule 5) — export "
            f"${CODE_SHA_ENV} on a box with no git checkout, or run from inside one."
        ) from exc
    sha = out.stdout.strip()
    if out.returncode != 0 or not _REAL_SHA_RE.match(sha):
        raise CodeShaError(
            f"${CODE_SHA_ENV} is unset and `git rev-parse HEAD` did not return a real sha "
            f"(exit {out.returncode}, stdout {sha!r}). code_sha cannot be written as a value "
            "nobody measured (repo rule 5)."
        )
    return sha


def _utc(now: dt.datetime) -> str:
    return now.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


#: The closed vocabulary EC2's own `instance-life-cycle` metadata answers.
_LIFECYCLE_VALUES = ("spot", "on-demand")


def _spot_lifecycle() -> tuple[bool, bool]:
    """Read `$CRUCIBLE_LIFECYCLE`, exported by the box shell from
    `/latest/meta-data/instance-life-cycle` via the same IMDSv2 token dance
    as `CRUCIBLE_INSTANCE_TYPE` (alpha-engine-config-I10328, second half).
    Returns `(spot, escalated_to_on_demand)`.

    Every crucible-v2 dispatch requests spot first and only falls back to
    on-demand on a declared capacity error (`crucible-v2.yaml`'s
    `CAPACITY_ERRORS`), so on this substrate `on-demand` always means an
    escalation happened — there is no code path that launches on-demand for
    any other reason.

    Before this, `resource.spot` came from an unconditionally exported
    `CRUCIBLE_SPOT=true`, so it was structurally `true` on every run
    including the two that escalated on 2026-09-09 — the CloudWatch
    `SpotEscalation` metric and the run manifest disagreed about the same
    launch.

    **Fail loud (repo rule 5), not a third default.** Unset means no box
    shell ran this process at all — a laptop or CI run — and reads as
    `(False, False)`, the same as `instance_type`'s own `local` default. Any
    other value (`unknown`, if the box's IMDS curl itself failed, or
    anything outside the closed EC2 vocabulary) RAISES rather than guess:
    `spot`/`escalated_to_on_demand` are required booleans in the schema, so
    there is no "absent" to fall back to that isn't also a fabricated
    measurement, and the run fails loudly instead of filing a manifest that
    asserts a resource fact nobody actually observed.
    """
    lifecycle = os.environ.get("CRUCIBLE_LIFECYCLE")
    if lifecycle is None:
        return False, False
    if lifecycle not in _LIFECYCLE_VALUES:
        raise RuntimeError(
            f"CRUCIBLE_LIFECYCLE={lifecycle!r} is not one of {_LIFECYCLE_VALUES}. The box "
            "shell's IMDS read for /latest/meta-data/instance-life-cycle failed or returned "
            "something crucible.runner does not recognise, and resource.spot/"
            "resource.escalated_to_on_demand cannot be defaulted without asserting a "
            "measurement nobody took (repo rule 5) — the run fails rather than files a "
            "manifest that claims to know."
        )
    return lifecycle == "spot", lifecycle == "on-demand"


def _measured_mem_peak_mb() -> float:
    """Peak resident set size of this process so far, in MB.

    `ru_maxrss` is a running maximum the kernel tracks for the life of the
    process, so reading it at manifest-write time (after `fn(ctx)` ran)
    reports the true peak of the whole run, not a point sample. POSIX
    leaves the unit unspecified: Linux (every crucible-v2 box, and CI)
    reports kilobytes; Darwin (a laptop dev run) reports bytes.
    """
    peak = posix_resource.getrusage(posix_resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return round(peak / divisor, 3)


def _measured_disk_free_mb(path: str = "/") -> float:
    """Free space on the filesystem `path` lives on, in MB. `/` on every
    substrate this runs on today — every crucible-v2 box is a fresh EC2
    root volume with no separate data mount, and a laptop or CI run reads
    its own root filesystem's headroom the same way."""
    return round(shutil.disk_usage(path).free / (1024.0 * 1024.0), 3)


def _initial_resource() -> dict[str, Any]:
    """The `resource` block's starting values, computed once per
    `RunContext`. `mem_peak_mb`/`disk_free_mb` are placeholders here —
    `_write_manifest` overwrites both at write time, when they mean
    something."""
    spot, escalated = _spot_lifecycle()
    return {
        "instance_type": os.environ.get("CRUCIBLE_INSTANCE_TYPE", "local"),
        "spot": spot,
        "escalated_to_on_demand": escalated,
        "interruptions": 0,
        "mem_peak_mb": 0.0,
        "disk_free_mb": 0.0,
    }


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

    #: Set by `run_job` from its own `now_override` argument — never by the
    #: job body. `alpha-engine-config-I10125`: an OPERATOR OVERRIDE of the
    #: wall-clock instant a job's own evaluation logic reasons from (today,
    #: only `alerts.sweep`'s catch-up/ceiling windows), distinct from
    #: `started`/`finished`, which stay the REAL wall clock this process ran
    #: at. Mirrors `discriminator`'s shape: `None` for every job that never
    #: passes it, and omitted from the manifest entirely rather than written
    #: as null (see `_write_manifest`) — a natural run's manifest stays
    #: byte-identical to one from before this field existed.
    now_override: dt.datetime | None = None

    #: Set by `run_job` from its own `fault_capability_class` argument —
    #: never by the job body. `alpha-engine-config-I10343`: the capability
    #: class every LLM call this run makes is REDIRECTED to, so plan §10.7
    #: fault 3 can be induced against the real dispatched path. `crucible.llm
    #: .call` reads it (`effective_capability_class`) and the value may only
    #: be one of `crucible.llm.FAULT_INJECTION_CAPABILITY_CLASSES`. Same
    #: "omitted entirely rather than null" shape as `now_override` above, and
    #: for the same reason: a natural run's manifest is byte-identical to one
    #: from before this field existed, and a run that deliberately routed to a
    #: broken group is structurally distinguishable from one that failed on
    #: its own.
    fault_capability_class: str | None = None

    inputs: list[dict[str, Any]] = field(default_factory=list)
    outputs: list[dict[str, Any]] = field(default_factory=list)
    rows_in: int = 0
    rows_out: int = 0
    rows_rejected: list[dict[str, Any]] = field(default_factory=list)
    cost_usd: float = 0.0
    llm_calls: list[dict[str, Any]] = field(default_factory=list)
    metrics: list[dict[str, Any]] = field(default_factory=list)
    attempts: list[dict[str, Any]] = field(default_factory=lambda: [{"n": 1, "reason": "initial"}])
    #: `spot`/`escalated_to_on_demand` are measured once, here, at RunContext
    #: construction — the box's lifecycle is a property of how the process
    #: was launched and does not change mid-run. `mem_peak_mb`/`disk_free_mb`
    #: start as placeholders and are OVERWRITTEN at write time
    #: (`_write_manifest`), which is when a peak-so-far and a free-space
    #: reading are actually meaningful — never left at these zeros in a
    #: written manifest. See `_spot_lifecycle`, `_measured_mem_peak_mb`,
    #: `_measured_disk_free_mb` (alpha-engine-config-I10328, second half).
    resource: dict[str, Any] = field(default_factory=_initial_resource)

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
        # Hashed here, not taken from the write's return value. Every
        # `put_bytes` happens to return the content digest today, but the
        # store contract does not promise it -- `compare_and_swap`, right
        # below, promises a VERSION TOKEN and returns an S3 ETag, and reading
        # `sha256` out of a write's return value is exactly how that value
        # ended up in this field (see `record_output_cas`).
        self.store.put_bytes(key, payload)
        self.outputs.append(
            {"key": key, "sha256": sha256_hex(payload), "schema_version": schema_version}
        )

    def record_rows(self, *, rows_in: int, rows_out: int) -> None:
        self.rows_in = rows_in
        self.rows_out = rows_out

    def record_rejected(self, reason: str, count: int) -> None:
        """§9.2 class 4: rejections carry a reason, always. A bare count
        cannot be acted on.

        The reason is bounded HERE, at the call site, and by raising rather
        than truncating (`alpha-engine-config-I10484`).

        Both halves are deliberate. The schema caps a rejection reason at 200
        characters; over that, the manifest fails validation at write time and
        the write routes through `_minimal_failed_manifest`, whose
        `_FALLBACK_DROPPED_FIELDS` includes `rows_rejected` — so a long string
        costs the run its inputs, outputs, metrics, llm_calls and every row
        count. That fallback behaviour is correct for a job-contributed field
        and is not what changes; what changes is that the job no longer
        introduces the violation silently and learn about it as a schema path
        in a manifest that lost everything else.

        Raising, not fitting, because of who owns the string. The manifest's
        own `reason` is fitted by `_write_manifest`: the runner renders it
        from an exception whose message it does not control, so the overflow
        is the runner's to absorb. A REJECTION reason is authored by the job,
        which can shorten it — silently clipping it would discard a category
        the job chose, at the one place `§9.2 class 4` says the category is
        the point.

        The cap is read from the schema, never restated: a second declaration
        of a number that already has one diverges the day somebody widens it.
        """
        if not reason:
            raise ValueError("a rejected-row entry needs a reason; a bare count is unactionable")
        cap = _schema_max_length("reason", defs="RejectedRow")
        if cap is not None and len(reason) > cap:
            raise ValueError(
                f"a rejected-row reason may be at most {cap} characters and this one is "
                f"{len(reason)}. Shorten it here — over the cap the manifest fails "
                "validation at write time and the fallback drops every job-contributed "
                f"field. Reason began: {reason[:120]!r}"
            )
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
        version = self.store.compare_and_swap(key, expected, payload)
        # `sha256` is the CONTENT digest, exactly as in `record_output`. What
        # `compare_and_swap` returns is the store's new VERSION TOKEN — its
        # own docstring says so — and the two are not the same value on every
        # backend. `LocalStore` makes its etag the content digest, so the two
        # coincide there and every local test agreed with the wrong answer;
        # `S3Store` returns the object's `ETag`, which is not a sha256 at all
        # for a multipart write and is not one BY CONTRACT for any write.
        #
        # This recorded the version token in the `sha256` field. On S3 that
        # fails `run_manifest.v2`'s `^[0-9a-f]{64}$` at the end of the job, in
        # the `finally` that writes the manifest -- AFTER the work has been
        # done and the object written. Measured 2026-09-09: the `Gate close`
        # run at 01:03:47Z filed BOTH phase-0 and phase-1 closing records by
        # CAS and then died on `outputs/0/sha256` and `outputs/1/sha256`, so
        # it never reached the step that posts the closing reading to the
        # phase tracker issues. The phase-exit loop failed on the first day it
        # ever succeeded, and the two trackers stayed open with no comment
        # explaining why (`alpha-engine-config-I9967`).
        #
        # The version token is still RETURNED, because a caller needs it as
        # the `expected` of its next conditional write.
        self.outputs.append(
            {"key": key, "sha256": sha256_hex(payload), "schema_version": schema_version}
        )
        return version


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
    dry_run: bool = False,
    write_manifest: bool = True,
    run_mode: str | None = None,
    now_override: dt.datetime | None = None,
    fault_capability_class: str | None = None,
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

    ``dry_run=True`` runs ``fn`` exactly as a real invocation would — a caller
    still needs the rendered/resolved result to report it — but skips the
    write path entirely: no `run.json` is written and nothing `fn` recorded
    is folded into a manifest, since there is no manifest. One line is
    printed instead, naming the job and trading day, so a dry run is visibly
    a dry run in the operator's own terminal rather than silent. This is the
    fix for alpha-engine-config-I9922: `report.morning --dry-run` is
    documented as "renders and files nothing", but before this flag existed
    the write below ran unconditionally, so a dry run against the production
    store left a real `ok` firing in `runs/` for `alerts.sweep` and the board
    to read. `fn` itself must still avoid any OTHER real write (a store put
    outside `ctx.record_output`, an external delivery) — this flag only
    covers the one write `run_job` itself makes; the store every CLI handler
    resolves is ALSO read-only under `--dry-run` (`crucible.store.read_only`),
    which is what actually stops `fn`'s own writes.

    ``write_manifest=False`` suppresses the manifest write for an invocation
    that publishes NOTHING — `alpha-engine-config-I10576`. It is deliberately
    a second parameter rather than a reuse of ``dry_run``, and the reason is
    the printed line: `crucible gate` without `--publish` is a READ, not a
    dry run, and telling an operator who asked for a live reading that they
    performed a "dry_run" is a false statement about what just ran on the one
    surface they are looking at. The parameter names what actually differs.

    The rule it carves out of is rule 1 ("manifest or it did not happen"), and
    the carve-out is exact: a manifest records a run's LINEAGE — what it read,
    what it published, what it spent. A run that publishes no shared artifact
    has no lineage a later reader could need, and the manifest over it is
    itself a write. Every write by a human principal is a counted touch: the
    measured case was a laptop `crucible gate` read writing
    `runs/gate/{day}/run.json`, which `crucible.autonomy` then counted against
    phase 2's own `zero_human_mutating_calls` clause — the documented read
    recipe making the gate it reads unmeetable. A job that PUBLISHES still
    writes its manifest unconditionally; only the read does not.

    ``run_mode`` is `live` or `replay` and is REQUIRED, in the sense that
    omitting it here falls through to ``$CRUCIBLE_RUN_MODE`` and then to a
    refusal (:class:`crucible.runmode.RunModeError`) — there is no default at
    any layer, and the resolution never looks at the date. A replay of a past
    Saturday is indistinguishable from a live one by `trading_day`, and phase
    2's exit gate counts live Saturdays from this field, so a guessed value
    would be a false claim about production in the one place that matters
    (alpha-engine-config-I9918). Resolved BEFORE the job body runs, so an
    undeclared invocation costs nothing. Resolved (and required) on the
    ``dry_run=True`` path too: a dry run still writes nothing regardless of
    `run_mode`, but a caller that cannot say whether it is live or a replay
    has the same bug whether or not `--dry-run` is also set.

    ``now_override`` records an operator override of the wall-clock instant a
    job's OWN evaluation logic reasons from — distinct from ``now`` above,
    which is when this process actually ran and stays real regardless.
    Today only `alerts.sweep` passes this (`alpha-engine-config-I10125`): its
    catch-up horizon and ceiling window are anchored to wall-clock `now` by
    construction, which makes them structurally coincident with
    `crucible.gate._clause_pages_within_ceiling`'s live grading window — a
    fault could not be swept for real without landing inside a window
    already being graded. This parameter is the honesty half of that fix: it
    is never read to decide what the job body does (the CLI resolves the
    override and passes it separately into `alerts.sweep(now=...)` itself);
    it exists so `_write_manifest` can record `now_override_utc` on the
    manifest whether or not the job body remembered to say so, making an
    overridden run structurally distinguishable from a natural one rather
    than relying on the body's own bookkeeping. `None` (the default) writes
    nothing, so every other job's manifest is unaffected.

    ``fault_capability_class`` records — and effects — a deliberate
    fault-injection redirect of this run's LLM routing
    (`alpha-engine-config-I10343`). It is the honesty half of §10.7 fault 3
    in exactly the shape ``now_override`` is for I10125: the value lands on
    the manifest as `fault_capability_class`, so a run that failed because an
    operator pointed it at a group contracted never to serve can never be
    read as a run that failed on its own. Unlike ``now_override`` it is also
    the ACTING half — `crucible.llm.call` reads it off the context, which is
    what lets a job that knows nothing about fault injection be faulted.
    `None` (the default) writes nothing and changes no routing, so every
    other job is unaffected.

    Returns the :class:`RunContext` on success. Re-raises on failure, after
    the manifest is on disk (or, on ``dry_run=True``, after the one line is
    printed in its place).
    """
    started = now or dt.datetime.now(dt.UTC)
    # Resolved first, before the trading day and before any work: an
    # invocation that never said whether it was live or a replay is refused
    # while refusing is still free.
    resolved_run_mode = resolve_run_mode(run_mode)
    # Same shape, same reason (alpha-engine-config-I10454): a code_sha this
    # process cannot measure for real is refused HERE, before any manifest
    # write is attempted — never written as the all-zero placeholder, and
    # never deferred to the write path, where a raise would collide with
    # "manifest or it did not happen" (see `CodeShaError`).
    resolved_code_sha = resolve_code_sha()
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
        ctx.now_override = now_override
        ctx.fault_capability_class = fault_capability_class

        # `alpha-engine-config-I9986` deliverable 1: the fleet cost-sink
        # partitions every row under `{prefix}/{date}/{run_id}/`
        # (`krepis.cost_sink.S3JsonlCostSink`), and `resolve_run_id()` reads
        # `KREPIS_RUN_ID` when set, otherwise minting a random per-process id
        # that cannot be joined back to this manifest. This is the one place
        # the harness's own run id is known before a job's body can reach a
        # model, so it is exported HERE, immediately, and before `fn(ctx)`
        # runs — never after. `krepis.cost_sink.default_sink_from_env` caches
        # its sink on `(bucket, prefix, KREPIS_RUN_ID)`, keyed at the sink's
        # first construction inside the job (an `LLMClient` is built lazily,
        # on the first call): an export issued after that construction would
        # already be too late for the sink that call built, so setting it
        # after `fn(ctx)` starts would be a no-op that looks like a fix. Set
        # fresh on every attempt, since a retried job gets a fresh `run_id`
        # too (`ctx` above is rebuilt each iteration) and the two attempts'
        # cost rows must not be joined to the same key.
        os.environ["KREPIS_RUN_ID"] = ctx.run_id

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
                if dry_run:
                    # No manifest, no outputs record — `dry_run` means this
                    # function makes exactly zero writes of its own. `fn` may
                    # still have returned data for the caller to print; what
                    # it must not have done is write to `store` itself, which
                    # is the caller's obligation (mirrored in `report.morning`
                    # and `board`, both of which branch on the same flag
                    # before touching the store).
                    print(
                        f"dry_run: {job} {ctx.trading_day.isoformat()} — no manifest written, "
                        f"no outputs recorded (status would have been {status!r})"
                    )
                elif not write_manifest:
                    # A REAL run that published nothing (alpha-engine-config-
                    # I10576). Distinct wording from the `dry_run` line above
                    # on purpose: this run read live state and reported it,
                    # which is not a dry run in the operator's sense, and a
                    # line claiming otherwise would be the third instance of
                    # the class this issue belongs to — a surface asserting
                    # something that is not what happened.
                    print(
                        f"read-only: {job} {ctx.trading_day.isoformat()} — no manifest written "
                        f"(this invocation published nothing; status {status!r})"
                    )
                else:
                    _write_manifest(
                        ctx,
                        store=store,
                        status=status,
                        reason=reason,
                        started=started,
                        now=now,
                        release_sha=release_sha,
                        code_sha=resolved_code_sha,
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
    code_sha: str,
) -> dict[str, Any]:
    """Assemble, validate and write one manifest. The single writer.

    Validated BEFORE the write. A non-conformant manifest is a bug in the
    runner, and it must surface here rather than at read time on a Saturday
    morning.
    """
    finished = dt.datetime.now(dt.UTC) if now is None else now
    # Measured here, not at RunContext construction: `ru_maxrss` is a
    # running peak, so reading it now reports the peak of the whole run
    # rather than a point sample taken before `fn(ctx)` did any work, and
    # disk headroom is only meaningful after whatever the job wrote.
    # alpha-engine-config-I10328, second half: these were an unconditional
    # `0.0` on every manifest ever written, worse than absent because they
    # read as a measurement nobody took.
    ctx.resource["mem_peak_mb"] = _measured_mem_peak_mb()
    ctx.resource["disk_free_mb"] = _measured_disk_free_mb()
    # DERIVED from `ctx.attempts`, never a second counter (alpha-engine-config-
    # I10463): a genuinely absorbed spot interruption produced `status: ok`,
    # `attempts: [initial, spot_interruption]` and `resource.interruptions: 0`
    # — the guard raised, the retry succeeded, and nothing ever incremented
    # anything. `ctx.attempts` is the one record that cannot exist without the
    # guard having raised (`run_job`'s `except BaseException` only appends a
    # `spot_interruption` entry after `classify_transient` names that reason),
    # so counting it here means the two fields can never disagree — there is
    # no second write path to drift. Computed at write time, after every
    # retry has appended its own entry, so this covers both the absorbed
    # (`ok`, N>=1 interruptions) and the exhausted-retry (`failed`) shapes.
    ctx.resource["interruptions"] = sum(
        1 for attempt in ctx.attempts if attempt.get("reason") == SPOT_INTERRUPTION_REASON
    )
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
        # Fitted to the schema's own cap HERE, at the primary assembly, not
        # only in the fallback below. `reason` is written by this module —
        # `_reason_from(exc)` renders an exception whose message the runner
        # does not control and cannot bound (the fault probe's is a LiteLLM
        # error carrying the router's entire fallback map) — so an over-long
        # reason is a runner-side overflow, never a claim the job made. Left
        # unfitted it fails validation and routes the write through
        # `_minimal_failed_manifest`, which drops every job-contributed field:
        # inputs, outputs, metrics, llm_calls and all three row counts, thrown
        # away because a string was long. Measured 2026-09-09 on
        # `runs/fault.probe/2026-09-09/run.json` — a run whose spend, calls and
        # lineage were all recorded and none of them survived.
        #
        # crucible-PR200 made the fallback survive that case; this makes the
        # case stop reaching it. The fallback is for a JOB-contributed field
        # the validator rejects, which is what its dropped-field list is
        # chosen against; a runner-written field overflowing its own declared
        # cap is the runner's to fit before it asks anyone to validate it.
        "reason": _fit(reason, _schema_max_length("reason")),
        "started": _utc(started),
        "finished": _utc(finished),
        "code_sha": code_sha,
        "release_sha": release_sha or os.environ.get("CRUCIBLE_RELEASE_SHA") or code_sha,
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
    if ctx.now_override is not None:
        # Same "omitted entirely" shape as discriminator, same reason: a run
        # that never overrode `now` produces a manifest byte-identical to one
        # from before this field existed. `alpha-engine-config-I10125`: this
        # is the field that makes an overridden sweep structurally
        # distinguishable from a natural one — a hand-chosen historical
        # window must never read as though it were observed live.
        manifest["now_override_utc"] = _utc(ctx.now_override)
    if ctx.fault_capability_class is not None:
        # Same "omitted entirely" shape, same reason (alpha-engine-config-I10343).
        # This is the field that makes an INDUCED router failure distinguishable
        # from a real one: `crucible fault.record --outcome induced` names a
        # failed manifest, and a reader has to be able to tell that the failure
        # was arranged.
        manifest["fault_capability_class"] = ctx.fault_capability_class
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
    key = manifest_key(ctx.job, ctx.trading_day.isoformat(), discriminator=ctx.discriminator)
    try:
        validate(manifest)
    except ManifestValidationError as exc:
        # A validator that can veto the write INVERTS the one guarantee this
        # system rests on. See :func:`_minimal_failed_manifest`.
        manifest = _minimal_failed_manifest(
            ctx,
            status=status,
            reason=reason,
            started=started,
            finished=finished,
            detail=str(exc),
            code_sha=code_sha,
            release_sha=release_sha or os.environ.get("CRUCIBLE_RELEASE_SHA") or code_sha,
        )
        validate(manifest)
    store.put_bytes(key, json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"))
    return manifest


#: The manifest fields a run is answerable for no matter what else went
#: wrong. Everything a job CONTRIBUTES — outputs, inputs, metrics, llm_calls,
#: the row counts — is dropped from the fallback below, because the whole
#: reason the fallback exists is that one of those is what the validator
#: rejected. Keeping any of them would risk the fallback failing the same way
#: the manifest it replaces did.
_FALLBACK_DROPPED_FIELDS = (
    "inputs",
    "outputs",
    "metrics",
    "llm_calls",
    "rows_in",
    "rows_out",
    "rows_rejected",
)


def _schema_max_length(field: str, *, defs: str | None = None) -> int | None:
    """The declared `maxLength` for a manifest field, or None.

    Read from the schema rather than restated, so a cap that moves there
    moves here. A hardcoded 2000 is the second declaration of a fact that
    already has one, and the two diverge on the day somebody widens the
    schema and nothing tells this module.

    ``defs`` names a `$defs` entry to read the field from instead of the
    document root — `_schema_max_length("reason", defs="RejectedRow")` is the
    200-character cap on ONE rejection row, a different number from the
    2000-character cap on the manifest's own `reason` and reached by a
    different caller.
    """
    schema = load_schema()
    root = schema.get("$defs", {}).get(defs, {}) if defs else schema
    prop = root.get("properties", {}).get(field, {})
    cap = prop.get("maxLength")
    return int(cap) if isinstance(cap, int) else None


def _fit(text: str, cap: int | None) -> str:
    """``text`` shortened to ``cap``, saying so where it was cut.

    Truncation is visible on purpose: a reader who cannot tell a complete
    reason from a clipped one will chase the missing half as if it were
    never written. The HEAD is kept because an exception's type and message
    lead, and its stack trails.
    """
    if cap is None or len(text) <= cap:
        return text
    marker = f" …[truncated to the schema's {cap}-character limit]"
    return text[: max(0, cap - len(marker))] + marker


def _minimal_failed_manifest(
    ctx: RunContext,
    *,
    status: str,
    reason: str,
    started: dt.datetime,
    finished: dt.datetime,
    detail: str,
    code_sha: str,
    release_sha: str,
) -> dict[str, Any]:
    """The manifest written when the real one will not validate.

    **Manifest-or-it-didn't-happen is the invariant; the validator is not
    allowed to break it.** `validate` raising here used to mean no write at
    all: the exception propagated out of `run_job`'s `finally`, the job
    exited non-zero with nothing in the store, and `crucible.alerts` reported
    the run as an ABSENCE — the condition for "nothing was even attempted" —
    with the cause discarded. That is strictly worse than the failure it was
    guarding against: an unreadable manifest at least names a run, and this
    named nothing.

    Measured 2026-09-09 (`alpha-engine-config-I10410`). `gate.close` for
    trading day 2026-09-08 (GitHub Actions run 34297633089) filed two closing
    records, recorded them as outputs whose `sha256` field carried a store
    version token rather than a content digest, and died on
    `outputs/0/sha256: String should match pattern '^[0-9a-f]{64}$'`.
    `runs/gate.close/2026-09-08/` is empty to this day, and the only surviving
    account of a run that DID file two phase closing records is a GitHub
    Actions log with a 90-day retention. The `sha256` defect itself is fixed
    (see `record_output`); this is the class above it, which is not.

    The fallback keeps only the fields the runner itself owns and computes,
    and drops every field a job contributes (:data:`_FALLBACK_DROPPED_FIELDS`)
    — one of those is what the validator rejected, so carrying them forward
    would reproduce the same failure. `status` is forced to `failed`: a run
    whose own account of itself is malformed has not succeeded, whatever it
    thought, and recording `ok` on a document we had to rebuild would be the
    graceful-degrade this repository refuses.

    `reason` carries BOTH the original reason (if the job also failed) and
    the validator's full complaint, because the second is what an operator
    needs in order to fix the producer and the first is what they came for.
    """
    rebuilt: dict[str, Any] = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": ctx.run_id,
        "job": ctx.job,
        "run_mode": ctx.run_mode,
        "trading_day": ctx.trading_day.isoformat(),
        "calendar_date": ctx.calendar_date.isoformat(),
        "status": "failed",
        "reason": _fit(
            (
                f"the manifest this run assembled does not validate, so this minimal "
                f"record stands in its place — every job-contributed field "
                f"({', '.join(_FALLBACK_DROPPED_FIELDS)}) is dropped, because one of "
                f"them is what was rejected. The validator said: {detail}. The run's "
                f"own status was {status!r}"
                + (f" with reason {reason!r}" if reason else " with no reason recorded")
            ),
            _schema_max_length("reason"),
        ),
        "started": _utc(started),
        "finished": _utc(finished),
        "code_sha": code_sha,
        "release_sha": release_sha,
        "seed": ctx.seed,
        "inputs": [],
        "outputs": [],
        "rows_in": 0,
        "rows_out": 0,
        "rows_rejected": [],
        "cost_usd": 0.0,
        "llm_calls": [],
        "resource": ctx.resource,
        "metrics": [],
        "attempts": ctx.attempts,
    }
    if ctx.discriminator is not None:
        rebuilt["discriminator"] = ctx.discriminator
    if ctx.now_override is not None:
        rebuilt["now_override_utc"] = _utc(ctx.now_override)
    if ctx.fault_capability_class is not None:
        rebuilt["fault_capability_class"] = ctx.fault_capability_class
    return rebuilt


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
