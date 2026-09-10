"""`crucible fault.probe` — the dispatched job that induces plan §10.7 fault 3.

Normative source: `alpha-engine-config-I10343` (`chaos_probe` is declared as a
model group and no crucible call site could ask for it), whose parent
`-I10126` recorded fault 3 as having "no authorized production seam": the unit
suite induces a router failure by `monkeypatch`-ing `krepis.llm.LLMClient`,
which is a pytest-only seam, and nothing in the CLI or the environment
redirected a real dispatched job to a broken transport.

**Why this is a JOB and not a test hook.** §10.7 asks that the fault produce
`status: failed` with the right reason, full telemetry and exactly one page —
properties of the real dispatched path, none of which a pytest monkeypatch can
evidence. The only thing that can produce them is a process the v2
dispatcher launches onto a spot box, installed from `releases/current`,
running through `crucible.runner.run_job` and swept by
`crucible.alerts`. This module is that process, and it is deliberately the
SMALLEST thing that is one: it reaches the router once, through the same door
every future LLM arm reaches it through, and reports what happened.

**It cannot serve, and it cannot spend.** Its registered call site
(`faults.router_probe` in `crucible/llm_callsites.yaml`) declares
`capability_class: chaos_probe` — a group whose registry contract is that
every member routes to a real upstream host with a model string that host will
never answer. So there is no invocation of this job that reaches a serving
model, and none that produces a bill: the call fails at the provider before a
completion exists to be charged for. Both properties are what make a
permanently-failing job in a production CLI acceptable rather than a liability.

**The request is explicit, and the manifest records it.**
`--fault-capability-class` is REQUIRED (`crucible.cli`), validated by
`crucible.llm.parse_fault_capability_class`, and written onto the manifest as
`fault_capability_class` by `run_job` — the `now_override_utc` shape from
`-I10125`, for the same reason. A fault record filed as `induced` names a
FAILED manifest; a reader has to be able to tell that particular failure was
arranged rather than observed, and the field is the only thing that says so.
Because `crucible.llm.call` reads the same context attribute, the mechanism
generalises: pointing `experiment.run` at a fault-injection class in phase 5
faults a job that knows nothing about fault injection, which is how a real
router outage arrives.

**A probe that SUCCEEDS is a failure of the seam**, and raises. A completion
from a group contracted never to serve means the fault-injection target has
started serving — the seam is gone, and every future `induced` record written
against it would be evidence about nothing. That is a louder defect than the
transport failure this job exists to produce, so it is reported as its own
error rather than as an `ok` run.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from crucible import llm
from crucible.runner import RunContext, classify_transient

__all__ = [
    "FAULT_PROBE_CALLSITE_ID",
    "FAULT_PROBE_JOB",
    "PROBE_OUTCOMES",
    "PROBE_OUTCOME_MARKER",
    "PROBE_OUTCOME_ROUTING_REFUSAL",
    "PROBE_OUTCOME_UPSTREAM_REFUSAL",
    "PROBE_OUTCOME_UPSTREAM_TRANSPORT_FAILURE",
    "FaultProbeFailure",
    "FaultProbeServedError",
    "classify_probe_failure",
    "fault_probe_handler",
    "probe_body",
    "probe_outcome_from_reason",
]

#: The job name registered in `crucible.cli.JOBS`, `crucible/components.yaml`
#: and `crucible.models.JOB_VALUES` — a constant imported by `cli.py` rather
#: than a literal restated at each of those sites, the same shape
#: `crucible.faults.FAULT_RECORD_JOB` uses.
FAULT_PROBE_JOB = "fault.probe"

#: The registered call site this job reaches the router through. It must be a
#: literal at the `crucible.llm.call` keyword itself for
#: `crucible.llm.audit_call_sites` to resolve it statically, so the literal
#: below is the one the audit reads and this constant is what every other
#: reader (and the containment test) compares against.
FAULT_PROBE_CALLSITE_ID = "faults.router_probe"

#: The one message the probe sends. Fixed and content-free: it is never
#: answered, and a prompt that looked like real work would invite somebody to
#: reuse this call site for real work.
PROBE_MESSAGES: tuple[dict[str, str], ...] = (
    {"role": "user", "content": "crucible fault-injection probe; no answer is expected"},
)


#: **The three ways this probe's one router call can end**
#: (`alpha-engine-config-I10367` deliverable 3), and only one of them is
#: evidence of plan §10.7 fault 3.
#:
#: Fault 3 is "the LLM router returns 500". The retry class it exists to
#: exercise is keyed on `provider_5xx`/`provider_timeout`
#: (`crucible.runner.TRANSIENT_CLASSIFIERS`), so a failure outside that class
#: is a fact about something else — and two of the three cases below are
#: exactly that, both of them measured live rather than imagined.

#: **(i)** Nothing upstream was ever contacted. `crucible.llm` refusing the
#: capability class, `krepis.router` refusing the group, the edge answering
#: 401 before LiteLLM sees the request, a connect timeout, or LiteLLM failing
#: to reach its own upstream. It says the route is broken, which is a real
#: finding and is not this fault.
PROBE_OUTCOME_ROUTING_REFUSAL = "routing_refusal"

#: **(ii)** An upstream answered with a 5xx, or was reached and then timed
#: out. The ONLY honest `induced` evidence for fault 3.
PROBE_OUTCOME_UPSTREAM_TRANSPORT_FAILURE = "upstream_transport_failure"

#: **(iii)** An upstream answered and refused with a 4xx. This is what the
#: `chaos_probe` group produced through the live edge on 2026-09-10 — both
#: chain members walked, both refused at the provider's REQUEST VALIDATOR.
#: A record filed off it would be evidence about that validator.
PROBE_OUTCOME_UPSTREAM_REFUSAL = "upstream_refusal"

#: The closed vocabulary. A fourth case is a PR that has to argue for it.
PROBE_OUTCOMES: tuple[str, ...] = (
    PROBE_OUTCOME_ROUTING_REFUSAL,
    PROBE_OUTCOME_UPSTREAM_TRANSPORT_FAILURE,
    PROBE_OUTCOME_UPSTREAM_REFUSAL,
)

#: How the outcome reaches a reader. The manifest's `reason` is the durable
#: record of why a run failed, and `crucible.faults` re-reads it rather than
#: trusting anything the recording operator typed — so the classification is
#: carried IN that string, as a marker no ordinary failure message produces.
#:
#: Deliberately not a new manifest FIELD: `run_manifest.v2.json` is the
#: contract every already-written manifest is read against, and a field only
#: one of thirteen jobs ever sets would be a schema change bought for one
#: consumer. The marker is greppable, is preserved verbatim by
#: `crucible.runner._reason_from`, and cannot be forged by a job that did not
#: classify anything, because nothing else writes it.
PROBE_OUTCOME_MARKER = "fault_probe_outcome="

#: Message substrings that mean the transport never got an answer from an
#: upstream at all — whatever HTTP status the layer above chose to report.
#:
#: **This list is the whole point of the classifier.** Measured 2026-09-10
#: against `litellm` with the generated `chaos_probe` group pointed at a dead
#: port: LiteLLM maps its own failure to CONNECT onto an HTTP **500** whose
#: body reads `OpenAIException - Connection error.`, and
#: `crucible.runner.classify_transient` correctly reads that as
#: `provider_5xx` — it is looking at the status, which is all a retry
#: classifier needs. A fault RECORD needs more: that 500 is a routing refusal
#: wearing fault 3's clothes, and filing it would attest that the router
#: surfaced an upstream failure when no upstream was reached.
_NO_UPSTREAM_ANSWER_NEEDLES: tuple[str, ...] = (
    "connection error",
    "connection refused",
    "cannot connect",
    "name or service not known",
    "nodename nor servname",
    "max retries exceeded",
    "route_unconfigured",
    "connection timed out",
)

#: Exception type names that mean the connection itself never came up. Kept
#: separate from the timeout row of `TRANSIENT_CLASSIFIERS` on purpose: that
#: table puts `ConnectTimeout` and `ReadTimeout` in ONE retry class, correctly
#: — both are worth one more attempt — while only the second of them reached
#: an upstream.
_NO_UPSTREAM_ANSWER_TYPES: tuple[str, ...] = ("ConnectTimeout", "APIConnectionError")

#: Statuses the ROUTER EDGE answers with before the router process is
#: consulted (`nous-ergon-ops/.../nginx/conf.d/litellm-router.conf`,
#: `location /`: `if ($router_consumer = "") { return 401; }`). They are 4xx,
#: but they are not an upstream refusing anything.
_EDGE_REFUSAL_STATUSES = frozenset({401, 403})


def classify_probe_failure(exc: BaseException) -> str:
    """Which of :data:`PROBE_OUTCOMES` ``exc`` is evidence of.

    Ordered so the cheap-and-wrong reading never wins: the connection-level
    tests run FIRST, because a failure to reach an upstream can arrive
    carrying any status code the layer above chose, and the status code is
    the only thing a status-based classifier looks at.

    The fall-through is :data:`PROBE_OUTCOME_ROUTING_REFUSAL`, the outcome
    that is REFUSED for `induced`. An unrecognised failure is not evidence
    of fault 3, and a classifier whose default was the accepting answer
    would turn every unfamiliar exception into an attestation.
    """
    names = {type(exc).__name__, *(b.__name__ for b in type(exc).__mro__)}
    haystack = f"{type(exc).__name__}: {exc}".lower()
    status = getattr(exc, "status_code", None)

    if names & set(_NO_UPSTREAM_ANSWER_TYPES):
        return PROBE_OUTCOME_ROUTING_REFUSAL
    if any(needle in haystack for needle in _NO_UPSTREAM_ANSWER_NEEDLES):
        return PROBE_OUTCOME_ROUTING_REFUSAL
    if classify_transient(exc) == "provider_timeout":
        # Everything that reads as a timeout and is NOT a connect timeout:
        # the upstream was reached and stopped answering, which §10.7's
        # retry class names alongside a 5xx.
        return PROBE_OUTCOME_UPSTREAM_TRANSPORT_FAILURE
    if not isinstance(status, int):
        # No HTTP exchange happened at all — `crucible.llm` or
        # `krepis.router` refused before a client existed.
        return PROBE_OUTCOME_ROUTING_REFUSAL
    if status in _EDGE_REFUSAL_STATUSES:
        return PROBE_OUTCOME_ROUTING_REFUSAL
    if status >= 500:
        return PROBE_OUTCOME_UPSTREAM_TRANSPORT_FAILURE
    if 400 <= status < 500:
        return PROBE_OUTCOME_UPSTREAM_REFUSAL
    return PROBE_OUTCOME_ROUTING_REFUSAL


def probe_outcome_from_reason(reason: str) -> str | None:
    """The outcome a manifest `reason` records, or None.

    None for a reason carrying no marker AND for one carrying a marker whose
    value is not declared: an undeclared outcome is not a fourth case, it is
    a string somebody wrote, and reading it as anything else would be the
    hole this vocabulary exists to close.
    """
    _, marker, rest = reason.partition(PROBE_OUTCOME_MARKER)
    if not marker:
        return None
    candidate = rest.split(":", 1)[0].split()[0] if rest.split() else ""
    return candidate if candidate in PROBE_OUTCOMES else None


class FaultProbeFailure(RuntimeError):
    """The probe's router call failed, classified.

    Wraps the transport's own exception rather than replacing it: the
    manifest `reason` is the only durable record of what the router actually
    said, and a wrapper that discarded it would leave a reader with a
    category and no evidence for it. `str(cause)` is appended for that
    reason, and `__cause__` is set by the `raise ... from` at the call site.
    """

    def __init__(self, outcome: str, cause: BaseException) -> None:
        if outcome not in PROBE_OUTCOMES:
            raise ValueError(
                f"{outcome!r} is not a declared fault-probe outcome. Declared: "
                f"{list(PROBE_OUTCOMES)}. A fourth case is a deliberate edit to "
                "PROBE_OUTCOMES, not a string a call site can invent — "
                "`crucible.faults` decides what may be filed `induced` from this "
                "vocabulary alone."
            )
        self.outcome = outcome
        self.cause = cause
        super().__init__(
            f"{PROBE_OUTCOME_MARKER}{outcome}: the fault-injection probe's router call "
            f"failed and was classified {outcome!r}. Underlying "
            f"{type(cause).__name__}: {cause}"
        )


class FaultProbeServedError(RuntimeError):
    """A fault-injection capability class returned a completion.

    The seam plan §10.7 fault 3 rests on is a router group contracted NEVER to
    serve. If it served, that contract has been broken somewhere above this
    package — a real model added to the group, or a member's `model` string
    becoming one its upstream host answers — and every `induced` fault record
    written against this path from now on would be evidence about a fault that
    no longer exists.

    Raised rather than returned so this run reads `status: failed` and pages:
    an `ok` here is exactly the "green over no data" reading the harness
    refuses everywhere else, and the group's purity is enforced in the private
    registry (`validate_llm_model_registry.py` invariant 21) where this
    package cannot see it. This is the runtime half of that boundary.
    """


def probe_body(ctx: RunContext, *, client_factory: Any = None) -> None:
    """Reach the router once, through the door, and refuse a completion.

    Separated from the handler so the whole body is exercisable against a
    :class:`~crucible.runner.RunContext` without a CLI, a store URI or a
    provider — the same split `crucible.faults.record_fault` uses.

    ``capability_class`` is read off the context rather than passed: the
    override is a property of how the RUN was launched, and a call site that
    took it as an argument would be a call site that could choose to ignore
    it. `crucible.llm.call` resolves the effective class from the same
    attribute, so this is the honest statement of what is being asked for
    rather than a second source of it.
    """
    requested = ctx.fault_capability_class
    if requested is None:
        raise ValueError(
            f"{FAULT_PROBE_JOB} ran with no fault-injection capability class. This job "
            "exists only to induce plan §10.7 fault 3, so there is no default and no "
            "natural invocation: `--fault-capability-class` is required, and a context "
            "that reached this body without it did not come from the CLI."
        )
    try:
        result = llm.call(
            ctx,
            callsite_id="faults.router_probe",
            capability_class=requested,
            messages=[dict(message) for message in PROBE_MESSAGES],
            cap=llm.SpendCap(cap_usd=llm.DEFAULT_LLM_CAP_USD),
            estimate_usd=0.0,
            client_factory=client_factory,
        )
    except Exception as exc:
        # `Exception`, deliberately NOT `BaseException`
        # (`alpha-engine-config-I10367` deliverable 3).
        # `crucible.runner.SpotInterruptionError` is a BaseException so that a
        # job's own handler cannot swallow a reclamation, and this handler is
        # one of those: catching it would classify plan §10.7 fault 1 as a
        # fault-3 outcome and file the record under the wrong fault. It
        # propagates untouched, and `run_job` writes the reclamation manifest
        # it already writes for every other job.
        raise FaultProbeFailure(classify_probe_failure(exc), exc) from exc
    raise FaultProbeServedError(
        f"capability class {requested!r} returned a completion (model "
        f"{getattr(result, 'model', '<unreported>')!r}). It is declared a FAULT-INJECTION "
        "target — a router group whose every member is contracted never to serve — so a "
        "completion means that contract is broken above this package and plan §10.7 fault "
        "3 no longer has a seam. Fix the group's purity in the model registry; do not "
        "relax this refusal, and do not file a fault record against this run."
    )


def fault_probe_handler(args: argparse.Namespace) -> int:
    """`crucible fault.probe --fault-capability-class <class>`.

    Writes nothing but its own manifest — no store artifact, no fault record.
    Filing the record is `crucible fault.record`'s job and is a separate,
    deliberate act that re-reads this manifest and refuses if it does not read
    `status: failed`: a job that both induced a fault and attested to it would
    be the rubber stamp `crucible.faults`' refusals exist to prevent.

    Exits non-zero, always, by design — `run_job` re-raises after the manifest
    is on disk.
    """
    from crucible.cli import _resolve_store
    from crucible.runner import run_job

    store = _resolve_store(args)
    # Already validated by `crucible.cli.main`'s usage block, which refuses a
    # bad value with `UsageError` (exit 2) BEFORE any handler is reached. It
    # was validated here, at exit 1, and the box wrapper reports exit 1 as
    # "if no run manifest exists the harness died before writing one" -- so a
    # mistyped flag was reported as a dead harness. Re-validated rather than
    # trusted: `fault_probe_handler` is also called directly by tests and by
    # anything that builds a Namespace by hand, and `parse_fault_capability_class`
    # is pure and offline, so the second call costs nothing and the guarantee
    # is local.
    requested = llm.parse_fault_capability_class(args.fault_capability_class)
    ctx = run_job(
        FAULT_PROBE_JOB,
        probe_body,
        store=store,
        trading_day=args.trading_day,
        dry_run=bool(args.dry_run),
        run_mode=getattr(args, "run_mode", None),
        fault_capability_class=requested,
        # The FIRST failure is the exercise. A retried probe would write one
        # manifest carrying two attempts, and `crucible fault.record`
        # distinguishes `induced` from `absorbed` on exactly that: a transient
        # retry in `attempts[]` is the signature of an ABSORBED fault, and a
        # probe whose whole point is a terminal transport failure must not
        # produce it. The retry path itself is exercised by the fault suite,
        # which asserts both directions (`tests/faults/`).
        transient_retry=False,
    )
    print(json.dumps({"run_id": ctx.run_id, "job": ctx.job}, indent=2))
    return 0
