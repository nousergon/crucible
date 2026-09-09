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
from crucible.runner import RunContext

__all__ = [
    "FAULT_PROBE_CALLSITE_ID",
    "FAULT_PROBE_JOB",
    "FaultProbeServedError",
    "fault_probe_handler",
    "probe_body",
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
    result = llm.call(
        ctx,
        callsite_id="faults.router_probe",
        capability_class=requested,
        messages=[dict(message) for message in PROBE_MESSAGES],
        cap=llm.SpendCap(cap_usd=llm.DEFAULT_LLM_CAP_USD),
        estimate_usd=0.0,
        client_factory=client_factory,
    )
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
