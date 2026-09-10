"""Was this run SYNTHETIC? One answer, derived from what the run declared.

Normative source: `alpha-engine-config-I10126`/`-I10343` (plan §10.7 fault
injection against the real dispatched path) and the class this module was
written for: **a deliberate probe and a real production failure were
indistinguishable at the notification boundary.**

Measured 2026-09-09, both on the real box and the real store:

* `runs/data.weekly/2026-08-07/run.json` — `run_mode: replay`, reason naming
  a ticker that exists nowhere but a fault-injection dispatch — produced
  `alerts/2026-08-07/failure.data.weekly.json` with `sent: true` and
  `destination: operator_chat`. A human was paged, at incident severity, by
  an exercise somebody had just launched on purpose.
* `runs/fault.probe/2026-09-11/run.json` — `fault_capability_class:
  chaos_probe`, a job whose docstring says it exits non-zero ALWAYS by design
  — paged the same topic through the box wrapper's generic
  `crucible-v2 job exited $code` branch.

**The marker is derived, never matched.** The obvious fix — recognise the
fault-injection sentinel ticker, or the `fault.probe` job name — grades the
one instance and misses the class: the next fault is induced with a different
sentinel, on a different job, and pages exactly as this one did. What is
common to every synthetic run is that the INVOCATION declared it: `run_mode`
(`crucible.runmode`, required with no default) and `fault_capability_class`
(`crucible.llm.parse_fault_capability_class`) are both operator statements
recorded on the manifest, and both are the only reason the run exists. So
this module reads those two fields and nothing else.

**It marks, and — from :data:`SYNTHETIC_ROUTING_ACTIVE_FROM` — it also
re-routes. It never silences.** Every synthetic page still writes its bus row,
still reaches a transport, still lands on a durable topic, still renders on
the console. What changes on that day is the *delivery* half and the
*accounting* half, and only those: the page goes to the muted topic instead of
the operator one, and `crucible.gate`'s page-ceiling clause stops counting an
exercise against a production alert budget. That is `observability-policy.md`
§7.2a's line exactly — suppression is a delivery decision, never a recording
one — and it is why this is not the suppression collection AGENTS.md rule 4
forbids: nothing stops being written down, and a reader can still count every
synthetic page there has ever been.

**Brian ruled this on 2026-09-09** (`alpha-engine-config-I10366`, option (b)),
"executed after 2026-09-19". See :data:`SYNTHETIC_ROUTING_ACTIVE_FROM` for why
the date is a constant in this file rather than a label on a draft PR.
"""

from __future__ import annotations

import datetime as dt
import shlex
from typing import Any

from crucible.runmode import RUN_MODE_REPLAY

__all__ = [
    "SYNTHETIC_ROUTING_ACTIVE_FROM",
    "SYNTHETIC_SUBJECT_PREFIX",
    "args_synthetic_marker",
    "manifest_synthetic_marker",
    "synthetic_marker",
    "synthetic_routing_active",
]

#: Prepended to an incident SUBJECT so a synthetic incident and a real one can
#: never land in the same page group, the same dedup key or the same bus row.
#: A dot, because `crucible.alerts.INCIDENT_ID_RE` admits `[A-Za-z0-9_.]` and
#: the subject becomes one store-key segment.
#:
#: Grouping them would be the worst possible outcome of this whole change: a
#: page reading "SYNTHETIC" that also carried a real production failure as its
#: second member is a page an operator is now TRAINED to ignore.
SYNTHETIC_SUBJECT_PREFIX = "synthetic."


def synthetic_marker(*, run_mode: str | None, fault_capability_class: str | None) -> str | None:
    """The human-readable descriptor, or ``None`` for a real live run.

    Two facts, deliberately both rendered rather than collapsed to a boolean:
    "replay" and "fault-injected" are different reasons not to be alarmed, and
    an operator who is told only "synthetic" has to go and find out which.
    """
    parts: list[str] = []
    if run_mode == RUN_MODE_REPLAY:
        parts.append("replay")
    if fault_capability_class:
        parts.append(f"fault-injected: {fault_capability_class}")
    return "; ".join(parts) if parts else None


def manifest_synthetic_marker(manifest: Any) -> str | None:
    """:func:`synthetic_marker` for a run manifest document.

    Tolerant of a non-mapping because it is called from the alert sweep, which
    reaches manifests that failed validation on purpose — a document that
    cannot be read as a mapping is not evidence that a run was synthetic, and
    the honest answer for it is "no marker", not a crash inside the paging
    path. FAILURE MODE SWALLOWED: a synthetic run whose manifest is corrupt
    pages unmarked. RECORDING SURFACE: that manifest is ALREADY a failure page
    of its own (`crucible.alerts.evaluate_failure` pages every unreadable
    manifest), so the corruption is never silent — only its synthetic-ness is,
    and erring toward "treat it as real" is the safe direction.
    """
    if not isinstance(manifest, dict):
        return None
    return synthetic_marker(
        run_mode=manifest.get("run_mode"),
        fault_capability_class=manifest.get("fault_capability_class"),
    )


def args_synthetic_marker(args: Any) -> str | None:
    """:func:`synthetic_marker` for a dispatch record's ``args`` string.

    The absence conditions grade a dispatch that produced NO manifest, so
    there is no manifest to read the two fields off. The dispatch record's
    ``args`` is the same operator statement one layer earlier — it is the
    literal argv the box was launched with — and parsing it is how an absence
    page for a deliberate probe is marked as one.

    Parsed with :mod:`shlex` and read positionally (``--flag value`` and
    ``--flag=value`` both), never with a substring test: ``--run-mode replay``
    appearing anywhere in a free-text field is not the same statement as the
    flag being set, and a marker derived from a substring is the string match
    this module exists to avoid.
    """
    if not isinstance(args, str) or not args.strip():
        return None
    try:
        tokens = shlex.split(args)
    except ValueError:
        return None
    values = _flag_values(tokens, ("--run-mode", "--fault-capability-class"))
    return synthetic_marker(
        run_mode=values.get("--run-mode"),
        fault_capability_class=values.get("--fault-capability-class"),
    )


def _flag_values(tokens: list[str], flags: tuple[str, ...]) -> dict[str, str]:
    found: dict[str, str] = {}
    for index, token in enumerate(tokens):
        for flag in flags:
            if token == flag and index + 1 < len(tokens):
                found[flag] = tokens[index + 1]
            elif token.startswith(f"{flag}="):
                found[flag] = token.split("=", 1)[1]
    return found


#: The day the SECOND half of this module's job switches on: a synthetic page
#: routes to the muted topic and leaves the phase-2 page ceiling.
#: **Brian's ruling, 2026-09-09 (`alpha-engine-config-I10366`), option (b),
#: "executed after 2026-09-19".**
#:
#: WHY A DATE AND NOT A LABEL, A FLAG OR A FOLLOW-UP PR
#: ---------------------------------------------------
#: Phase 2's two live first-attempt Saturdays are **2026-09-12** and
#: **2026-09-19**. Four `crucible.gate` clauses are being graded across that
#: window — `live_saturdays_first_attempt_ok`, `replays_ok`,
#: `pages_within_ceiling`, `pages_commissioned` — and re-reading the last two
#: mid-window makes the phase-2 exit unauditable: the same store would answer
#: the same clause differently depending on which day somebody looked, with
#: nothing in the artifacts saying why.
#:
#: The obvious alternatives were each rejected for a named reason:
#:
#: * **A `gate:*` label on a draft PR** — the activation then depends on a
#:   human remembering to come back on 2026-09-20. `~/Development/CLAUDE.md`
#:   is explicit that a post-merge manual step is not a legitimate form, and
#:   `alpha-engine-config-I1906` is the recorded instance of exactly that
#:   shape being closed as *fixed* with the step never performed.
#: * **An environment variable / feature flag** — same defect wearing a
#:   configuration hat, plus a second place for the answer to live. The box's
#:   shell would carry one value and the laptop reading `crucible gate`
#:   another, and the clause reading would then depend on WHERE it was read.
#: * **A second PR opened later** — the work would sit unlanded through the
#:   exact week it is most likely to be forgotten, and the ruling would be
#:   recorded nowhere executable.
#:
#: What a date buys is that the guarantee is **structural**: the code before
#: this day is provably the code that shipped, and
#: `tests/test_synthetic_routing_activation.py` asserts that both sides of the
#: boundary — same store, same rows, one day apart — differ in exactly the two
#: readings the ruling moves and in nothing else.
#:
#: The honest cost, stated rather than hidden: a dated behaviour change is a
#: scheduled change nobody is standing next to when it fires. It is mitigated
#: by there being exactly ONE constant (every reader below derives from it,
#: none carries its own copy), by the post-activation clause detail SAYING it
#: is active so the flip is visible on the surface it changes, and by the
#: boundary test. It is not mitigated by anyone remembering.
SYNTHETIC_ROUTING_ACTIVE_FROM = dt.date(2026, 9, 20)


def synthetic_routing_active(*, on: dt.date | dt.datetime | None = None) -> bool:
    """Whether synthetic pages route to the muted topic and leave the ceiling.

    ``on`` is the moment being asked about — the moment of a send for the
    routing decision, the moment of the READING for a gate clause. It defaults
    to today in UTC.

    Deliberately a function of the moment and not of the ROW: the gate clauses
    grade a whole store at once, so a per-row cutoff would leave the ceiling
    permanently carrying the exercises that polluted it (it read **6/2** on
    2026-09-04, and the fault-injection runbook records that as the reason
    three of four plan §10.7 faults could not be induced for real). One flip,
    one day, the whole history re-read under one rule.
    """
    if on is None:
        moment = dt.datetime.now(dt.UTC).date()
    elif isinstance(on, dt.datetime):
        moment = on.astimezone(dt.UTC).date()
    else:
        moment = on
    return moment >= SYNTHETIC_ROUTING_ACTIVE_FROM
