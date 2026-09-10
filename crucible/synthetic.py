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

**It marks; it does not silence.** Everything a synthetic page produces today
it still produces: the bus row is written, the transport is reached, the
ceiling still counts it. What changes is that the row and the rendered page
say what they are, so no reader — human or machine — has to ask "was that
real?" and no reader has to answer it by recognising a ticker. Suppressing
here instead would be a suppression collection (AGENTS.md rule 4) and would
also move two of `crucible.gate`'s live phase-2 clauses, which is not a thing
to do in the week two live Saturdays are being graded.
"""

from __future__ import annotations

import shlex
from typing import Any

from crucible.runmode import RUN_MODE_REPLAY

__all__ = [
    "SYNTHETIC_SUBJECT_PREFIX",
    "args_synthetic_marker",
    "manifest_synthetic_marker",
    "synthetic_marker",
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
