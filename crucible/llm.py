"""The one way this package reaches a model: the cap, the registry, the audit.

Normative source: plan §2 row 3 (cost), §2 row 7 (transparency), §4.8 (LLM
access), §9.2 class 2 (token/cost telemetry).

Three things live here, and they are one thing:

1. **A declared, enforced spend cap.** :data:`DEFAULT_LLM_CAP_USD` is the
   per-weekly-run ceiling, resolvable per deployment through
   `crucible.config.Settings.llm_cap_usd`. A call that would carry the run
   past it RAISES :class:`LlmSpendCapExceeded` **before the provider is
   touched**, so the run fails with a reason naming the cap and the spend
   that would have happened does not happen. Three ceilings bind, and each
   one REFUSES rather than adjusts: an estimate above the call site's own
   `max_usd_per_call` is refused (:class:`LlmCallCeilingExceeded`), never
   admitted at a lowered reservation; admission against the weekly cap is on
   the site's ceiling, the largest bill the call may produce, not on an
   optimistic estimate; and the actual cost is re-checked against what was
   admitted after the provider bills (:class:`LlmSpendOverrun`), so an
   overrun fails the run instead of being booked in silence. Pacing through
   the week comes from `krepis.usage_pacing`, which is what turns "we blew
   the budget on Saturday" into "we were ahead of pace on Tuesday".

2. **A call-site registry.** ``LLM_CALLSITE_REGISTRY`` is loaded from
   ``llm_callsites.yaml``; :func:`call` refuses an id that is not in it. The
   id is the join key between spend and the code that caused it, and
   `krepis.llm.LLMClient` requires it too — but krepis is a public library
   and can only check that it is a non-empty string, so membership is
   checked here.

3. **An enumerator that reads the CODE.** :func:`audit_call_sites` walks the
   package's ASTs and reports every place a model can be reached: a call to
   :func:`call` whose ``callsite_id`` is absent, computed, or unregistered,
   and — the half that matters — any import of a provider or router module
   from a module that is not this one. Coverage measured from a list is a
   list checking itself; the audit's central finding was a loop that ran for
   months measuring nothing.

**Why the adapter is the only door.** Principle 8: no call site names a
provider, a base url, a model id or an SDK client. A capability class or a
router group goes in, the router decides what serves it, and both the
requested and the served model land in the manifest — a routed call that
silently served something else is exactly what that pair exists to expose.
"""

from __future__ import annotations

import ast
import datetime as dt
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from krepis.usage_pacing import PaceStatus, pace_check
from pydantic import ValidationError

from crucible.documents import load_store_document
from crucible.keys import RUNS_ROOT, is_manifest_key
from crucible.models import LlmCallsiteRegistryDocument
from crucible.store import Store

__all__ = [
    "CALLSITE_REGISTRY_PATH",
    "CAPABILITY_CLASS_GROUPS",
    "EXEC_CONTEXT_ENV",
    "CallSite",
    "CapabilityClassNotRouted",
    "CostSinkReconciliationError",
    "DEFAULT_LLM_CAP_USD",
    "DEFAULT_LLM_CAP_USD_MEASURED",
    "Finding",
    "LLM_CALLSITE_REGISTRY",
    "LlmCallCeilingExceeded",
    "LlmSpendCapExceeded",
    "LlmSpendOverrun",
    "PACE_OVERRUN_MARGIN",
    "PACING_PERIOD",
    "PROVIDER_MODULES",
    "RECONCILIATION_METRIC",
    "RECONCILIATION_TOLERANCE_USD",
    "SpendCap",
    "audit_call_sites",
    "call",
    "capability_group",
    "load_capability_classes",
    "load_registry",
    "pace_metric",
    "reconcile_manifests_cost",
    "reconcile_run_cost",
    "reconciliation_unmeasurable",
    "spend_pace",
    "week_to_date_llm_spend",
]

#: The per-weekly-run ceiling on LLM spend, in USD.
#:
#: **A declared ceiling, not a measurement, and it says so.** Phase 1 carries
#: zero LLM arms, so there is no observed distribution to size this against;
#: what it is sized to is the plan's cost row (§2 row 3), where LLM spend sits
#: beside a $40/mo AWS ceiling and must not dominate it. It is deliberately
#: small enough that the first LLM arm's first cycle hits it and someone has
#: to look, rather than large enough that nobody ever does.
#:
#: It is re-set from the cost sink once an LLM arm has run a full cycle —
#: that is a measurement, and it belongs to phase 5.
DEFAULT_LLM_CAP_USD = 5.00

#: **Machine-checkable, not only prose.** `alpha-engine-config-I9778`: the
#: paragraph above already SAID the cap is unmeasured, but nothing in the
#: artifacts a run writes could be compared against that claim — a manifest
#: or a report generated under the default looked identical to one generated
#: under a cap someone had actually re-set. This flag is that comparison
#: point: :func:`~crucible.config.settings` cannot set it (there is no
#: `CRUCIBLE_LLM_CAP_MEASURED` env var — measurement is a fact about how the
#: number was chosen, not a runtime knob), so it flips to `True` only when a
#: future edit HERE, beside the value it describes, records that a phase-5
#: cost-sink cycle re-set :data:`DEFAULT_LLM_CAP_USD`. Track: a phase-5
#: follow-up files "re-set `DEFAULT_LLM_CAP_USD` from the first LLM arm's
#: first full weekly cycle's `llm_spend_usd` metric, then flip this to
#: `True` in the same PR that changes the value" — the two edits belong in
#: one PR precisely so this flag can never say `True` beside a number nobody
#: measured.
DEFAULT_LLM_CAP_USD_MEASURED = False

#: The window the cap is paced across. The cap is per WEEKLY RUN and the
#: weekly cycle is the window, so `krepis.usage_pacing` compares spend against
#: the fraction of the week elapsed: a run that has burned 80% of the cap by
#: Tuesday is ahead of pace even though nothing has breached yet.
PACING_PERIOD = dt.timedelta(days=7)

#: Modules through which a process can reach a model. An import of any of them
#: from a module other than this one is an adapter bypass, and
#: :func:`audit_call_sites` reports it as an unregistered call site — because
#: that is what it is: a place spend can happen that the registry cannot see.
#:
#: The router modules are first-party and the rest are vendor SDKs. Both are
#: refused for the same reason (principle 8): a call site holding an SDK
#: client is a call site shaped around one provider.
#:
#: ``krepis.llm_config`` is listed for a sharper reason
#: (`alpha-engine-config-I9969`). It is the PRE-router flip surface: its
#: ``resolve_model_spec`` reads a ``provider:model`` string out of SSM and
#: hands back a spec naming a vendor endpoint directly — no router edge, no
#: cross-provider fallback chain, no per-consumer attribution, and outside
#: the egress proxy. Those are verbatim the three objections
#: ``krepis.llm_config.ModelSpec.__post_init__`` writes down as its reason
#: for refusing ``provider="litellm"`` at construction, and this adapter
#: reached all three through that function for the whole of phase 1. It is
#: refused here so the same shape cannot come back anywhere in the package —
#: and, because THIS module is the exempt path, a dedicated assertion in
#: ``tests/test_llm_router_route.py`` refuses it here too.
PROVIDER_MODULES: frozenset[str] = frozenset(
    {
        "krepis.llm",
        "krepis.llm_config",
        "krepis.llm_search",
        "krepis.router",
        "litellm",
        "openai",
        "anthropic",
        "cohere",
        "mistralai",
        "ollama",
        "google.generativeai",
    }
)

CALLSITE_REGISTRY_PATH = Path(__file__).parent / "llm_callsites.yaml"

#: This module. The adapter is the one place a provider module may be
#: imported, and the audit compares against this path rather than against a
#: set of exempt files — a second entry beside it is an edit to the rule, in
#: the place the rule is written down.
_ADAPTER_PATH = Path(__file__).resolve()


class LlmSpendCapExceeded(RuntimeError):
    """A call was refused because it would carry the run past its cap.

    Raised BEFORE the provider is reached, so the spend it names never
    happens. It propagates: the runner writes `status: failed` with this
    reason, which is the plan's answer to "what does a run do when it would
    exceed the cap" — it fails, rather than overspending and reporting `ok`.

    The two subclasses below name the other two ways spend can escape a
    declared ceiling. They subclass rather than stand beside it because every
    consumer of the cap wants the same answer to all three — the run failed
    because the money would not fit — and only a reader diagnosing *which*
    ceiling bound needs the distinction.
    """


class LlmCallCeilingExceeded(LlmSpendCapExceeded):
    """An estimate above the call site's own declared per-call ceiling.

    **This is a refusal, never a smaller reservation.** The defect this class
    exists to make impossible: reserving ``min(estimate, ceiling)`` admits a
    call estimated at $100 against $0.05 of headroom because the `min` shrinks
    the *reservation* to something that fits, while the call itself still
    costs $100. A ceiling that lowers what is booked rather than refusing what
    is asked is not a ceiling — measured 2026-09-01: provider constructed,
    weekly spend $9.85 against a declared $5.00 cap, run status ``ok``.
    """


class LlmSpendOverrun(LlmSpendCapExceeded):
    """A completed call billed more than the ceiling it was admitted under.

    The money is already spent, so this is raised AFTER the cost is booked
    into :attr:`SpendCap.spent_usd` and after the call lands in the run
    manifest's ``llm_calls``: the durable record tells the truth about what
    happened, and the run then fails rather than continuing to spend on top
    of an overrun nobody has looked at. Booking silently and returning `ok`
    is the failure mode plan §2 row 3 forbids — the overrun would exist only
    in the provider's invoice.
    """


@dataclass(frozen=True)
class CallSite:
    """One registered place this package can reach a model from."""

    callsite_id: str
    purpose: str
    capability_class: str
    max_usd_per_call: float
    owner: str


# --------------------------------------------------------------------------
# Capability class -> router model group.
# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _capability_classes() -> frozenset[str]:
    """Every capability class this package may ask the router for.

    **An ALLOWLIST, not a denylist of vendor name fragments.** The shape here
    used to be nine substrings — ``gpt-``, ``claude-``, ``llama`` and so on —
    and ``grok-``, ``qwen``, ``command-r``, ``nova-`` and ``kimi`` all walked
    straight through it, as did a bare base url and a bare provider name.
    Principle 8 is not "refuse the model ids somebody thought of"; it is that
    a call site addresses a capability and nothing else, which is a
    *membership* question and therefore has a positive answer. A denylist is
    wrong by default and wrong again with every vendor that launches.

    Two sources, unioned, and neither is a restatement of the other:

    ``krepis.router.TIER_GROUPS``
        the router's own tier-to-group mapping, read rather than copied, so a
        group added there is askable here without an edit and a group removed
        there stops being askable.

    ``capability_classes`` in ``llm_callsites.yaml``
        the classes this deployment's router serves beyond the bare tiers,
        declared once beside the call sites that use them. A call site cannot
        add its own — the list is a deliberate edit in the file where the
        reason for each name is written down.
    """
    from krepis.router import TIER_GROUPS

    return (
        frozenset(TIER_GROUPS)
        | frozenset(TIER_GROUPS.values())
        | frozenset(load_capability_classes())
    )


def _require_capability_class(value: str, *, callsite_id: str) -> None:
    allowed = _capability_classes()
    if value in allowed:
        return
    raise ValueError(
        f"call site {callsite_id!r} asked for {value!r}, which is not a router capability "
        f"class. The router declares {sorted(allowed)}; anything else — a vendor model id, "
        "a base url, a provider name — is the lock-in principle 8 forbids, and is refused "
        "by membership rather than by a list of model-name fragments that a new vendor "
        "walks through."
    )


class CapabilityClassNotRouted(RuntimeError):
    """A declared capability class that addresses no ruled router group.

    Raised BEFORE a provider, an endpoint or a credential is reached, and
    deliberately in preference to the two alternatives
    (`alpha-engine-config-I9969`):

    * **A silent default.** Resolving an unmapped class to `high`, or to any
      other group, ships a model nobody chose — strictly worse than a crash,
      because a run would then produce a graded result attributed to a
      capability class that never served it.
    * **A pre-router flip surface.** The shape this replaces read
      a per-class SSM parameter through
      ``krepis.llm_config.resolve_model_spec``, so an unmapped class was not
      an error at all — it was a second, one-rung routing plane naming a
      provider directly.

    A class reaching this exception is a MAPPING that has not been made, and
    the message says whose it is to make.
    """


#: Declared capability class -> the registry MODEL GROUP it addresses.
#:
#: **The one place the mapping is written down, and it is checkable.** A
#: class absent from this table addresses the group of the SAME NAME: `high`
#: means the registry's `high` group, and `krepis.router` refuses a name the
#: registry does not declare, naming the groups it does — so identity needs
#: no entry here and cannot drift. An entry exists only for a class whose
#: name is NOT a group name, and it carries one of two things:
#:
#: ``a group name``
#:     the ruled mapping. Everything below the group — which model, which
#:     provider, which endpoint, which credential, which reasoning params,
#:     and the cross-provider fallback chain — stays a registry decision
#:     resolved above this consumer (`model-router-policy` §2 layer 5).
#:
#: ``None``
#:     the mapping is a RULING that has not been made. Every use of the class
#:     raises :class:`CapabilityClassNotRouted` naming the ruling. This is
#:     not a placeholder to be filled in by whoever next needs the class: it
#:     is the refusal that keeps an unruled name from quietly acquiring an
#:     answer.
#:
#: `reasoning_high` is the live instance. It is declared in
#: `llm_callsites.yaml` as the class the phase-5 research arms ask for, and
#: it is a group in NO registry — measured 2026-09-04, the registry declares
#: exactly `low`, `med`, `high`, `ultra`. Which of those the phase-5 arms and
#: their judge address is Brian's ruling, open as
#: `alpha-engine-config-I9970`; the mechanism that routes them is this file's
#: to own and does not wait on it.
CAPABILITY_CLASS_GROUPS: dict[str, str | None] = {
    "reasoning_high": None,
}


def capability_group(capability_class: str) -> str:
    """The router model group *capability_class* addresses.

    Pure, offline and total: it reads :data:`CAPABILITY_CLASS_GROUPS` and
    nothing else, so it can be called at registry-load time — the earliest
    point at which an unrouted class is knowable — without a registry file,
    a network, or an AWS credential.

    Whether the returned name is a group the registry actually declares is
    the ROUTER's question, answered by `krepis.router` against the registry
    document with a `ValueError` naming every available group. Restating the
    group set here would be the copied list `alpha-engine-config-I9971`
    already records against this package's capability-class allowlist.
    """
    if capability_class in CAPABILITY_CLASS_GROUPS:
        group = CAPABILITY_CLASS_GROUPS[capability_class]
        if group is None:
            # The tracker for the ruling is cited in `CAPABILITY_CLASS_GROUPS`'s
            # own documentation above, not here: a tracker literal in a raised
            # message is the stale-pointer class `tests/
            # test_no_stale_tracker_literals.py` refuses package-wide.
            raise CapabilityClassNotRouted(
                f"capability class {capability_class!r} addresses no router model group. "
                "Which group it maps to is an open RULING — see the tracker named in "
                "`crucible.llm.CAPABILITY_CLASS_GROUPS`'s documentation — and until that "
                "ruling is made this class REFUSES rather than resolving: a default here "
                "would ship a model nobody chose, and the manifest would attribute the "
                "result to a class that never served it. Rule the group, then write it "
                f"beside {capability_class!r} in `crucible.llm.CAPABILITY_CLASS_GROUPS`."
            )
        return group
    return capability_class


#: The fleet's ONE name for "where is this code running" — krepis' own
#: variable, not a crucible-shaped second one. `krepis.router` reads it when a
#: caller passes no `exec_context`, and every launcher and deploy config in
#: the fleet already sets it.
EXEC_CONTEXT_ENV = "KREPIS_EXEC_CONTEXT"


def _exec_context() -> str:
    """Where this process is running, DECLARED — never inferred, never defaulted.

    `model-router-policy` R28/R29: which registry entries are reachable is a
    function of the execution context, and the context may not be guessed. A
    wrong guess produces a resolution that reads as a health failure — a spot
    box with no local egress proxy handed a `laptop`-reachable endpoint fails
    with an opaque "no model reachable" instead of an honest "you never said
    where you are".

    krepis' own resolution currently DEFAULTS an undeclared context to
    `laptop` with a warning — a staged migration
    (`alpha-engine-config-I7409`) toward raising, kept permissive for the
    call sites that predate the rule. This package has none: it is refused
    here, so crucible cannot be one of the call sites that migration is
    waiting on, and a log line nobody reads is never what stands between a
    run and the wrong endpoint.
    """
    declared = (os.environ.get(EXEC_CONTEXT_ENV) or "").strip()
    if not declared:
        from krepis.router import EXEC_CONTEXTS

        raise ValueError(
            f"{EXEC_CONTEXT_ENV} is not set, so this process has not said where it is "
            f"running. Which models are reachable depends on it (model-router-policy "
            f"R28/R29) and it may not be inferred — a guess hands a spot box an endpoint "
            f"only a laptop can reach and reports it as a health failure. Set it to one of "
            f"{list(EXEC_CONTEXTS)} in this job's launcher or deploy config."
        )
    return declared


def _read_registry_document() -> LlmCallsiteRegistryDocument:
    """Parse and validate `llm_callsites.yaml` as one document.

    `alpha-engine-config-I10045` row 4: `load_registry` and
    `load_capability_classes` each used to `yaml.safe_load` this file
    independently and hand-check the piece they needed — a row missing
    `max_usd_per_call` fell through to `float(row["max_usd_per_call"])`,
    which raises a bare `KeyError`/`TypeError` naming neither the call site
    nor the field. Both callers now validate through
    `crucible.models.LlmCallsiteRegistryDocument`; each still does its own
    `yaml.safe_load` and its own call into this function (no shared cache
    between them) so the existing test suite's direct
    `load_capability_classes.cache_clear()` /
    `_capability_classes.cache_clear()` calls keep working unchanged.
    """
    if not CALLSITE_REGISTRY_PATH.is_file():
        raise FileNotFoundError(
            f"the LLM call-site registry is missing at {CALLSITE_REGISTRY_PATH}. It "
            "ships inside the package; its absence is a broken build, not an empty "
            "registry."
        )
    raw = yaml.safe_load(CALLSITE_REGISTRY_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{CALLSITE_REGISTRY_PATH} is not a mapping; got {type(raw).__name__}")
    try:
        return LlmCallsiteRegistryDocument.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"{CALLSITE_REGISTRY_PATH}: {exc}") from exc


@lru_cache(maxsize=1)
def load_registry() -> dict[str, CallSite]:
    """``LLM_CALLSITE_REGISTRY``, read from ``llm_callsites.yaml``.

    A missing or malformed file RAISES. An unreadable registry read as empty
    would make every call site unregistered and every coverage check vacuous
    at the same time — the registry would report 100% coverage of nothing.
    """
    document = _read_registry_document()
    registry: dict[str, CallSite] = {}
    for callsite_id, row in document.callsites.items():
        _require_capability_class(row.capability_class, callsite_id=callsite_id)
        # The EARLIEST point an unrouted class is knowable: a row declaring a
        # class that addresses no ruled group is refused when the registry
        # loads, not on the first call in the first weekly run. Offline and
        # pure — no registry document, no network, no credential — so this
        # holds in CI, where none of those exist
        # (`alpha-engine-config-I9969`).
        capability_group(row.capability_class)
        registry[callsite_id] = CallSite(
            callsite_id=callsite_id,
            purpose=row.purpose,
            capability_class=row.capability_class,
            max_usd_per_call=row.max_usd_per_call,
            owner=row.owner,
        )
    return registry


@lru_cache(maxsize=1)
def load_capability_classes() -> tuple[str, ...]:
    """``capability_classes`` from ``llm_callsites.yaml``.

    A missing key RAISES, like a missing ``callsites`` mapping: an allowlist
    read as empty would refuse every call, which looks like a broken router
    rather than a broken registry, and an allowlist that silently defaulted
    to "anything" would be the denylist this replaced with extra steps. An
    empty allowlist is written ``capability_classes: []``.
    """
    return tuple(_read_registry_document().capability_classes)


#: The registry itself. A mapping id -> :class:`CallSite`; empty in phase 1,
#: and the enumerator below is what makes that emptiness checkable.
LLM_CALLSITE_REGISTRY: dict[str, CallSite] = load_registry()


# --------------------------------------------------------------------------
# The cap.
# --------------------------------------------------------------------------


def week_to_date_llm_spend(store: Store, *, window_start: dt.date, trading_day: dt.date) -> float:
    """LLM spend already recorded between ``window_start`` and ``trading_day``.

    Summed from ``llm_calls[].usd`` across every run manifest in the window —
    the durable record — rather than from a process-local counter, so a cap
    spanning several jobs in one weekly run is enforced against what actually
    happened rather than against what this process remembers.
    """
    total = 0.0
    for key in store.list_keys(RUNS_ROOT):
        # `is_manifest_key`, not a `"/run.json"` suffix literal: the basename
        # is `crucible.keys`' to own, and the predicate checks the root and the
        # arity too (alpha-engine-config-I9900).
        if not is_manifest_key(key):
            continue
        # STRICT face of the one reader: a cap that skipped an unreadable
        # manifest would under-count spend in exactly the run most likely to
        # have overspent, so a corrupt one stops the job with its key named.
        manifest = load_store_document(store, key)
        day = dt.date.fromisoformat(manifest["trading_day"])
        if not window_start <= day <= trading_day:
            continue
        for record in manifest.get("llm_calls", []):
            total += float(record.get("usd", 0.0))
    return total


def spend_pace(
    spent_usd: float, *, cap_usd: float, now: dt.datetime, anchor: dt.datetime
) -> PaceStatus:
    """Where this run's spend sits against a straight line through the week.

    `krepis.usage_pacing` rather than a threshold: a fixed 85%-of-cap alarm
    only fires once most of the budget is gone, which on a weekly cadence
    means it fires after the week is already lost. The linear-pace comparison
    catches a front-loaded burst at any point in the window.
    """
    if cap_usd <= 0:
        raise ValueError(
            f"the LLM cap must be positive; got {cap_usd}. A cap of zero is spelled by "
            "registering no call sites, not by a ceiling no call can clear."
        )
    return pace_check(used_frac=spent_usd / cap_usd, now=now, anchor=anchor, period=PACING_PERIOD)


#: The materiality margin an ahead-of-pace reading must clear before `call`
#: writes a durable artifact about it (`alpha-engine-config-I9977`).
#:
#: **The early-window case, handled deliberately.** `PaceStatus.exceeded` is
#: `overrun > 0`, which is trivially true for almost any nonzero spend in the
#: first moments of the window: at `elapsed_frac == 0.001` (a few minutes
#: after the weekly anchor), a single $0.05 call against a $5.00 cap reads
#: `used_frac=0.01 > elapsed_frac=0.001` — technically ahead of a
#: straight-line pace, and not actionable about anything. Acting on `exceeded`
#: alone would write the "ahead of pace" artifact on the first call of every
#: week, every week, which is exactly the fixed-threshold noise this module's
#: own docstring says the linear-pace comparison exists to avoid, and this
#: repo's alerting rule caps pages per month for the same reason. The margin
#: is a materiality floor on `pace.overrun`, not a change to the math:
#: `spend_pace`/`pace_check` keep reporting the honest, unfiltered sign — a
#: front-loaded burst still clears 0.05 easily (see
#: `tests/test_llm_cap.py::TestPaceAtCallTime`) — this constant only decides
#: whether writing an artifact about a given reading is worth doing.
PACE_OVERRUN_MARGIN = 0.05


def pace_metric(
    pace: PaceStatus, *, cap: SpendCap, now: dt.datetime, source_path: str
) -> dict[str, Any]:
    """The MetricRecord an ahead-of-pace reading writes AT CALL TIME.

    Same metric name and shape the terminal `report` job already publishes
    (`crucible.track_e.report_handler`) — this is the same quantity, read
    earlier: `report` still runs the weekly summary reading unchanged, and
    this is the reading the module's own docstring promised ("we were ahead
    of pace on Tuesday") and that nothing took until now
    (`alpha-engine-config-I9977`). `status` is always `"WATCH"`: this
    function is only called once :data:`PACE_OVERRUN_MARGIN` is cleared, so
    every call of it is, by construction, a reading worth recording.
    """
    return {
        "name": "llm_spend_pace_overrun_ratio",
        "module": "crucible.llm",
        "metric_type": "ratio",
        "value": round(pace.overrun, 6),
        "unit": "ratio",
        "n_floor": 1,
        "status": "WATCH",
        "status_reason": (
            f"{pace.used_frac:.1%} of the ${cap.cap_usd:.2f} weekly LLM cap spent "
            f"(worst case, including this call's own ceiling) against {pace.elapsed_frac:.1%} "
            f"of the window elapsed — {pace.overrun:.1%} past the {PACE_OVERRUN_MARGIN:.0%} "
            "materiality margin, ahead of a straight-line pace at call time, days before "
            "the terminal report job would otherwise have taken this reading first"
        ),
        "source_path": source_path,
        "last_updated_utc": now.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "baseline": 0.0,
    }


@dataclass
class SpendCap:
    """The enforcement point: nothing is spent that this has not admitted.

    ``spent_usd`` is seeded from the week's manifests, so the cap is a
    property of the weekly RUN rather than of whichever process happens to be
    making the call.
    """

    cap_usd: float
    spent_usd: float = 0.0
    anchor: dt.datetime | None = None
    #: `alpha-engine-config-I9823`: whether ``cap_usd`` traces to a phase-5
    #: cost-sink measurement (`crucible.config.Settings.llm_cap_usd_measured`,
    #: itself only ever :data:`DEFAULT_LLM_CAP_USD_MEASURED`). Carried on the
    #: cap the caller constructs rather than re-derived here, so this module
    #: never re-implements config's resolution — but it now RIDES somewhere:
    #: :func:`cap_metric` publishes it on every run's `llm_spend_usd` metric,
    #: closing the half of I9823 where the flag was written only to
    #: `Settings.to_dict()`, which no caller in this package read.
    cap_usd_measured: bool = False

    def headroom_usd(self) -> float:
        return self.cap_usd - self.spent_usd

    def reserve(self, ceiling_usd: float, *, callsite_id: str) -> float:
        """Admit a call that may cost up to ``ceiling_usd``, or refuse it.

        **Admission is on the worst case, not on the estimate.** The amount
        checked against the cap is the call's declared per-call ceiling —
        `CallSite.max_usd_per_call` — because that is the largest bill the
        call is allowed to produce, and a cap that admits on an optimistic
        estimate is a cap that is crossed every time an estimate is low. The
        pessimism is transient: :meth:`record` books what the call actually
        cost, so the headroom the next `reserve` sees is the real figure, not
        the reservation.

        Returns the admitted ceiling, which the caller hands back to
        :meth:`record` so an actual cost above it is caught rather than
        booked.
        """
        if ceiling_usd < 0:
            raise ValueError(f"a call ceiling cannot be negative; got {ceiling_usd}")
        if self.spent_usd + ceiling_usd > self.cap_usd:
            raise LlmSpendCapExceeded(
                f"call site {callsite_id!r} was refused: it would carry this weekly run to "
                f"${self.spent_usd + ceiling_usd:.4f} against a declared cap of "
                f"${self.cap_usd:.2f} (already spent ${self.spent_usd:.4f}). The run fails "
                "rather than overspending; raise `llm_cap_usd` deliberately in config, or "
                "cut what the run asks for."
            )
        return ceiling_usd

    def record(self, usd: float, *, reserved_usd: float, callsite_id: str) -> None:
        """Book what a completed call actually cost, and RE-CHECK it.

        ``reserved_usd`` is what :meth:`reserve` admitted. The cost is booked
        first — the money is gone and the ledger says so whatever happens
        next — and then a bill above the reservation, or a total above the
        cap, raises :class:`LlmSpendOverrun`. Booking without the re-check is
        how an overrun becomes silent: the old shape here added the actual
        cost to ``spent_usd`` and returned, so a run could finish ``ok``
        having spent double its declared cap and nothing in the artifacts
        said so.

        Both arguments are keyword-only and required, so a caller cannot
        reach the un-checked path by omitting them.
        """
        usd = float(usd)
        if usd < 0:
            raise ValueError(f"a completed call cannot have cost {usd}")
        self.spent_usd += usd
        if usd > reserved_usd:
            raise LlmSpendOverrun(
                f"call site {callsite_id!r} was admitted under a per-call ceiling of "
                f"${reserved_usd:.4f} and billed ${usd:.4f}. The cost is booked and in the "
                "run manifest — the money is spent — and the run now fails rather than "
                "continuing on top of an overrun. Either the provider's price moved or the "
                "site's `max_usd_per_call` is wrong; both are edits somebody makes on "
                "purpose."
            )
        if self.spent_usd > self.cap_usd:
            raise LlmSpendOverrun(
                f"call site {callsite_id!r} carried this weekly run to "
                f"${self.spent_usd:.4f} against a declared cap of ${self.cap_usd:.2f}. The "
                "cost is booked and in the run manifest; the run fails rather than "
                "reporting `ok` over a breached cap."
            )


def cap_metric(cap: SpendCap, *, now: dt.datetime, source_path: str) -> dict[str, Any]:
    """The MetricRecord that makes the cap observable (principle 7).

    Emitted on EVERY run that holds a cap, including the ones that spend
    nothing: a cap nobody publishes a figure against is a cap nobody is held
    to, and zero spend is a measurement.

    Carries ``cap_usd_measured`` (`alpha-engine-config-I9823`): whether
    ``cap.cap_usd_measured`` traces to a phase-5 cost-sink cycle rather than
    the declared-not-measured default. Before this, a manifest or a report
    generated under the default looked identical to one generated under a
    cap someone had actually re-set — the flag existed in
    `crucible.config.Settings` but reached no consumer. This is that
    consumer: the field is on `metricRecord`, which is
    ``additionalProperties: true`` in the run-manifest schema, so it needs no
    schema change to become machine-checkable off the manifest a run already
    writes.
    """
    fraction = cap.spent_usd / cap.cap_usd if cap.cap_usd else 0.0
    status = "BREACH" if cap.spent_usd > cap.cap_usd else "OK"
    measured_phrase = "measured" if cap.cap_usd_measured else "declared, not measured"
    return {
        "name": "llm_spend_usd",
        "module": "crucible.llm",
        "metric_type": "count",
        "value": round(cap.spent_usd, 6),
        "unit": "usd",
        "n_floor": 0,
        "status": status,
        "status_reason": (
            f"${cap.spent_usd:.4f} of a {measured_phrase} ${cap.cap_usd:.2f} per-weekly-run "
            f"cap ({fraction:.1%}); a call that would exceed it is refused before the "
            "provider is reached, so the run fails rather than overspending"
        ),
        "source_path": source_path,
        "last_updated_utc": now.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "baseline": cap.cap_usd,
        "cap_usd_measured": cap.cap_usd_measured,
    }


# --------------------------------------------------------------------------
# The manifest-vs-sink reconciliation (`alpha-engine-config-I9986`).
# --------------------------------------------------------------------------

#: A rounding allowance, not a materiality threshold. `krepis.cost.record_llm_call`
#: and this package's own `cost_usd` both round to six decimal places, but at
#: different times relative to token-to-price conversion, so two honestly
#: identical totals can differ in the trailing digits. A cent is generous
#: enough to absorb that and tight enough that a run actually missing a
#: cost-sink row (or double-counting one) still fails.
RECONCILIATION_TOLERANCE_USD = 0.01


class CostSinkReconciliationError(RuntimeError):
    """A cost-sink object under the run's own key exists and could not be read.

    Raised rather than silently excluded from the sum: a row this function
    cannot parse or price is a row that would otherwise vanish from
    `sink_usd`, which makes an unreadable object look identical to a quiet
    run — the same shape `crucible.cost.CostUnreadableError` refuses for
    Cost Explorer, and the same reasoning `crucible-research/scripts/
    aggregate_costs.py::_read_jsonl_rows` already applies to this exact key
    layout in the sibling repo (`policy-shared-code`: mirrored, not
    reinvented).
    """


def _cost_sink_client() -> Any:
    """A boto3 S3 client, constructed lazily.

    Mirrors `crucible.cost.default_client`: importing this module — or
    calling `reconcile_run_cost` in a test that injects its own `s3_client`
    — must not require `boto3` to be installed or a credential chain to
    resolve.
    """
    import boto3  # noqa: PLC0415 - lazy on purpose; see the docstring

    return boto3.client("s3")


def _cost_sink_jsonl_keys(
    s3_client: Any, *, bucket: str, prefix: str, date: dt.date, run_id: str
) -> list[str]:
    """Every `.jsonl` key under `{prefix}/{date}/{run_id}/`.

    `S3JsonlCostSink`'s own key layout (`krepis.cost_sink.S3JsonlCostSink`
    docstring): `{prefix}/{date}/{run_id}/{callsite_id}.{seq}.jsonl`. Paginated,
    the same shape `crucible-research/scripts/aggregate_costs.py::_list_jsonl_keys`
    uses for the identical sink.
    """
    key_prefix = f"{prefix.rstrip('/')}/{date.isoformat()}/{run_id}/"
    keys: list[str] = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=key_prefix):
        for obj in page.get("Contents") or []:
            key = obj.get("Key", "")
            if key.endswith(".jsonl"):
                keys.append(key)
    return keys


def _cost_sink_row_usd(s3_client: Any, *, bucket: str, key: str) -> float:
    """The summed `cost_usd` of every row in one cost-sink JSONL object.

    A row with no numeric `cost_usd` RAISES — `krepis.cost.record_llm_call`
    writes `cost_usd: None` for `cost_source == "usage_unreported"` (a
    provider that did not report usage), and summing `None` as zero would
    understate `sink_usd` by exactly the amount a mismatch is supposed to
    catch.
    """
    body = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
    total = 0.0
    for lineno, line in enumerate(body.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CostSinkReconciliationError(
                f"s3://{bucket}/{key} line {lineno} is not valid JSON: {exc}"
            ) from exc
        usd = row.get("cost_usd")
        if not isinstance(usd, (int, float)) or isinstance(usd, bool):
            raise CostSinkReconciliationError(
                f"s3://{bucket}/{key} line {lineno} carries no numeric `cost_usd` "
                f"(got {usd!r}). A row that cannot be priced is not a row that cost "
                "nothing, and summing it as zero would hide exactly the gap this "
                "reconciliation exists to catch."
            )
        total += float(usd)
    return total


def reconcile_run_cost(
    manifest: dict[str, Any],
    *,
    bucket: str,
    prefix: str,
    s3_client: Any = None,
    dates: list[dt.date] | None = None,
    now: dt.datetime | None = None,
) -> dict[str, Any]:
    """The manifest-vs-sink reconciliation MetricRecord for one run.

    `alpha-engine-config-I9986` deliverable 2 (`I9972` deliverable 3): the
    two spend ledgers this run wrote — `manifest["llm_calls"][].usd` (this
    package's own cap ledger, `crucible.llm.week_to_date_llm_spend`'s source
    of truth, which keeps reading the manifest and never this function) and
    the fleet cost-sink rows `krepis.cost_sink.S3JsonlCostSink` wrote under
    `{prefix}/{date}/{manifest['run_id']}/` — must sum to the same figure
    within :data:`RECONCILIATION_TOLERANCE_USD`, or the two ledgers have
    silently diverged, which is the shape of `alpha-engine-config-I7407` and
    `I9694`.

    **A mismatch is a metric with `status: FAIL` and a stated reason on the
    manifest — never a log line.** This function does not raise on a
    mismatch and does not write anything itself; it returns a MetricRecord
    (the §9.2 class 5 shape `cap_metric` also returns) for the caller to
    hand to `ctx.record_metric(...)`, which is what makes the discrepancy
    part of the durable, greppable record `crucible explain` reads rather
    than something only a log aggregator saw. It DOES raise
    :class:`CostSinkReconciliationError` when a cost-sink object exists and
    cannot be read — an unreadable row is a defect in the pipeline, not a
    discrepancy this run's ledgers can disagree about.

    ``dates`` defaults to the run's own `calendar_date` plus the following
    day: `S3JsonlCostSink` partitions each row by the record's own UTC `ts`
    (set when the call completes), not by the run's start time, so a job
    that straddles UTC midnight can file rows one calendar day after the run
    started. The manifest carries no field for "every date this run's calls
    landed on" — only `calendar_date`, the run's own start — so the search
    window is bounded rather than exact; pass ``dates`` explicitly for a job
    known to run longer than a day.

    ``s3_client`` is the injection point tests use to avoid a real AWS
    credential; production callers omit it and get a lazily-constructed
    `boto3.client("s3")` (:func:`_cost_sink_client`), the same pattern
    `crucible.cost.default_client` uses for Cost Explorer.
    """
    run_id = manifest["run_id"]
    manifest_usd = round(sum(float(c.get("usd", 0.0)) for c in manifest.get("llm_calls", [])), 6)
    calendar_date = dt.date.fromisoformat(manifest["calendar_date"])
    search_dates = (
        dates if dates is not None else [calendar_date, calendar_date + dt.timedelta(days=1)]
    )
    client = s3_client if s3_client is not None else _cost_sink_client()

    keys: list[str] = []
    for date in search_dates:
        keys.extend(
            _cost_sink_jsonl_keys(client, bucket=bucket, prefix=prefix, date=date, run_id=run_id)
        )
    sink_usd = round(sum(_cost_sink_row_usd(client, bucket=bucket, key=key) for key in keys), 6)

    discrepancy = round(manifest_usd - sink_usd, 6)
    ok = abs(discrepancy) <= RECONCILIATION_TOLERANCE_USD
    finished = now or dt.datetime.now(dt.UTC)
    prefix_display = f"s3://{bucket}/{prefix.rstrip('/')}"
    if ok:
        reason = (
            f"manifest llm_calls[].usd=${manifest_usd:.6f} agrees with "
            f"{prefix_display}/.../{run_id}/ (${sink_usd:.6f} across {len(keys)} "
            f"object(s)) within the ${RECONCILIATION_TOLERANCE_USD:.2f} rounding tolerance"
        )
    else:
        reason = (
            f"MANIFEST-VS-SINK MISMATCH for run {run_id!r}: manifest "
            f"llm_calls[].usd=${manifest_usd:.6f} vs {prefix_display}/.../{run_id}/ "
            f"=${sink_usd:.6f} across {len(keys)} object(s) — a ${discrepancy:.6f} "
            f"discrepancy, past the ${RECONCILIATION_TOLERANCE_USD:.2f} rounding "
            "tolerance. The two spend ledgers have diverged."
        )
    return {
        "name": "llm_cost_reconciliation_usd",
        "module": "crucible.llm",
        "metric_type": "count",
        "value": manifest_usd,
        "unit": "usd",
        "n_floor": 0,
        "status": "OK" if ok else "FAIL",
        "status_reason": reason,
        "source_path": f"{prefix_display}/",
        "last_updated_utc": finished.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "baseline": manifest_usd,
        "run_id": run_id,
        "sink_usd": sink_usd,
        "sink_objects_read": len(keys),
        "discrepancy_usd": discrepancy,
    }


#: The aggregate row the weekly heartbeat files (`alpha-engine-config-I9986`
#: deliverable 2: "for a weekly run ... a discrepancy is a failed run").
RECONCILIATION_METRIC = "llm_cost_reconciliation_usd"


def reconciliation_unmeasurable(reason: str, *, now: dt.datetime) -> dict[str, Any]:
    """The aggregate row when the sink could not be consulted at all.

    `unmeasurable`, never `OK` and never `$0.00`: no reading of the sink is
    not a reading that the two ledgers agree.
    """
    return {
        "name": RECONCILIATION_METRIC,
        "module": "crucible.llm",
        "metric_type": "count",
        "value": 0.0,
        "unit": "usd",
        "n_floor": 0,
        "status": "unmeasurable",
        "status_reason": reason,
        "source_path": "runs/{job}/{trading_day}/run.json:llm_calls",
        "last_updated_utc": now.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def reconcile_manifests_cost(
    manifests: list[dict[str, Any]],
    *,
    bucket: str,
    prefix: str,
    s3_client: Any = None,
    now: dt.datetime | None = None,
) -> dict[str, Any]:
    """One aggregate MetricRecord over a set of run manifests — the week's.

    Every manifest that carries a `run_id` and a `calendar_date` is
    reconciled through :func:`reconcile_run_cost`, INCLUDING manifests whose
    `llm_calls` is empty: a run that recorded no call but whose key holds
    cost-sink rows is the other half of the divergence, and reconciling
    only the runs that admit to spending would never see it. `FAIL` if any
    run mismatched, naming the first few; `OK` otherwise, with the count of
    runs and the total either ledger reports. A manifest missing either
    field is a malformed manifest and is reported in the reason, never
    silently skipped — an unreadable run is not a run that spent nothing.

    Raises :class:`CostSinkReconciliationError` from the per-run read when
    a sink object exists and cannot be read, and lets a denied or failed S3
    call propagate: both are facts about the pipeline or the grant, and the
    caller (`crucible.alerts.heartbeat`) records them as access faults and
    fails the run after its proof of life is sent.
    """
    finished = now or dt.datetime.now(dt.UTC)
    client = s3_client if s3_client is not None else _cost_sink_client()
    mismatched: list[str] = []
    malformed: list[str] = []
    checked = 0
    manifest_total = 0.0
    sink_total = 0.0
    for manifest in manifests:
        run_id = manifest.get("run_id")
        calendar_date = manifest.get("calendar_date")
        if not isinstance(run_id, str) or not isinstance(calendar_date, str):
            malformed.append(f"{manifest.get('job', '?')}@{manifest.get('trading_day', '?')}")
            continue
        row = reconcile_run_cost(
            manifest, bucket=bucket, prefix=prefix, s3_client=client, now=finished
        )
        checked += 1
        manifest_total += float(row["value"])
        sink_total += float(row["sink_usd"])
        if row["status"] != "OK":
            mismatched.append(f"{run_id}: {row['status_reason']}")
    prefix_display = f"s3://{bucket}/{prefix.rstrip('/')}"
    if mismatched:
        reason = (
            f"{len(mismatched)} of {checked} run(s) this week disagree between their "
            f"manifest llm_calls ledger and {prefix_display}: " + " | ".join(mismatched[:3])
        )
    else:
        reason = (
            f"{checked} run(s) this week reconcile: manifests ${manifest_total:.6f}, "
            f"{prefix_display} ${sink_total:.6f}, within "
            f"${RECONCILIATION_TOLERANCE_USD:.2f} per run."
        )
    if malformed:
        reason += (
            f" {len(malformed)} manifest(s) carried no run_id/calendar_date and could not "
            f"be reconciled: {', '.join(malformed[:3])}."
        )
    return {
        "name": RECONCILIATION_METRIC,
        "module": "crucible.llm",
        "metric_type": "count",
        "value": round(manifest_total, 6),
        "unit": "usd",
        "n_floor": 0,
        "status": "FAIL" if mismatched or malformed else "OK",
        "status_reason": reason,
        "source_path": f"{prefix_display}/",
        "last_updated_utc": finished.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "baseline": round(sink_total, 6),
        "runs_reconciled": checked,
        "runs_mismatched": len(mismatched),
    }


# --------------------------------------------------------------------------
# The one call.
# --------------------------------------------------------------------------


def _system_and_user_content(messages: list[dict[str, str]]) -> tuple[str, str]:
    """Translate an OpenAI-style ``messages`` list into the pair the
    installed `krepis.llm.LLMClient.complete` actually accepts.

    **Found while building `alpha-engine-config-I9986`.** `LLMClient.complete`
    (krepis 0.59.50, the pinned version) takes ``system: str`` and
    ``user_content: str`` — it has no ``messages=`` parameter at all. Every
    existing test in this package stubs `krepis.llm.LLMClient` out entirely
    (`tests/test_llm_cap.py`, `tests/test_llm_router_route.py`), so a call
    against the REAL class — which phase 5's first arm will make, and which
    `alpha-engine-config-I9972` deliverable 4 (this issue's deliverable 3)
    needs in order to exercise the real DLP/cost-sink preconditions — raised
    `TypeError: complete() got an unexpected keyword argument 'messages'`
    before reaching any of them. Fixed here rather than filed, since the
    reconciliation and precondition tests below are meaningless against a
    stub that accepts any keyword.

    A single system message (or none) plus exactly one user message is the
    only shape translated — refused rather than guessed for anything else,
    since a call site is either OpenAI-chat-shaped by construction (today,
    every registered site is) or it is a caller that needs a real multi-turn
    `LLMClient` method this package does not yet expose, and silently
    flattening a multi-turn conversation into one `user_content` string would
    be a plausible wrong prompt reaching the provider.
    """
    system_parts = [m["content"] for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]
    if len(rest) != 1 or rest[0].get("role") != "user":
        raise ValueError(
            f"crucible.llm.call only translates a single optional system message plus "
            f"exactly one user message into krepis.llm.LLMClient.complete's "
            f"(system, user_content) pair; got {len(messages)} message(s) with roles "
            f"{[m.get('role') for m in messages]!r}. A multi-turn conversation needs a "
            "translation this adapter does not yet implement, not a silently flattened one."
        )
    return "\n\n".join(system_parts), rest[0]["content"]


def call(
    ctx: Any,
    *,
    callsite_id: str,
    capability_class: str,
    messages: list[dict[str, str]],
    cap: SpendCap,
    estimate_usd: float,
    client_factory: Any = None,
    registry: dict[str, CallSite] | None = None,
    now: dt.datetime | None = None,
    **kwargs: Any,
) -> Any:
    """Reach a model, once, through the router — the only door in this package.

    ``ctx`` is the :class:`~crucible.runner.RunContext`: the call's tokens,
    cost, requested model and served model land in its manifest, so §9.2
    class 2 is a property of the door rather than of each caller remembering.

    ``capability_class`` is a router group, or a declared class that
    :func:`capability_group` maps to one. A value that looks like a vendor
    model id is REFUSED — a call site naming a model is a call site that has
    to be edited when the provider changes (principle 8).

    **The group is resolved through `krepis.router`, and only through it**
    (`alpha-engine-config-I9969`). What the call site states is a capability;
    which model serves it, at which endpoint, on which credential, and in
    what order its cross-provider fallback chain is walked are registry
    decisions resolved above this package (`model-router-policy` §2 layer 5).
    The row this writes to the manifest carries every half of the answer:
    ``model_served`` (which model actually answered), ``route_degraded``
    (whether RESOLUTION had already fallen past the group's primary), and —
    the call-time facts, `alpha-engine-config-I10006` — ``fallback_used`` and
    ``served_deployment``, read off :class:`krepis.llm.LLMResult`.

    **The resolve-time and call-time facts are not interchangeable.** On the
    router-edge route the chain is walked by the proxy AFTER resolution, so
    ``route_is_degraded`` can report a healthy route on a call the primary
    never answered — and it returned ``False`` unconditionally on
    ``litellm_proxy``, the route krepis prefers whenever the health probe
    answers and therefore the one v2 takes. Recording only the resolve-time
    predicate does not leave the fact missing; it leaves it stated FALSE.

    ``registry`` and ``client_factory`` are injection points, in that order:
    the first lets a test exercise the cap against a call site that does not
    exist in production, the second lets it do so without a provider. Neither
    weakens the audit, which reads the code rather than this function's
    arguments — a call site injecting its own registry is still a call site,
    and :func:`audit_call_sites` still demands a literal registered id.

    ``now`` is the same kind of injection point, for the pace reading below —
    production omits it and gets ``dt.datetime.now(dt.UTC)``; a test wanting
    a fixed reading against a fixed anchor passes it explicitly, per this
    repo's "fixed date literals, never `today` arithmetic" test-discipline
    rule.

    **Pace, on the same cap admission just used (`alpha-engine-config-I9977`).**
    Once :meth:`SpendCap.reserve` admits the call, the worst-case spend this
    call could produce — ``cap.spent_usd + reserved_usd`` — is read against a
    straight-line pace through the week via :func:`spend_pace`, using
    ``cap.anchor``: the same instance the admission check just used, never a
    fresh read of the store (that would put an S3 list on the hot path of
    every LLM call). A cap constructed with no ``anchor`` — a test cap, or a
    caller outside the weekly cadence — gets no pace reading; that is a
    distinct opt-in from cap enforcement, which stays unconditional above. An
    ahead-of-pace reading past :data:`PACE_OVERRUN_MARGIN` lands on the
    *calling job's own* manifest as a metric (:func:`pace_metric`), before
    the provider is reached — not only in the terminal `report` job's weekly
    summary, which still runs unchanged and still reads the pace as of the
    week's last manifest.
    """
    known = load_registry() if registry is None else registry
    site = known.get(callsite_id)
    if site is None:
        raise KeyError(
            f"call site {callsite_id!r} is not in LLM_CALLSITE_REGISTRY "
            f"({CALLSITE_REGISTRY_PATH}). Register it — with its purpose, capability "
            "class, per-call ceiling and owner — before it can spend; an unregistered "
            "call site is spend nobody can attribute."
        )
    _require_capability_class(capability_class, callsite_id=callsite_id)
    if estimate_usd < 0:
        raise ValueError(f"a call estimate cannot be negative; got {estimate_usd}")
    if estimate_usd > site.max_usd_per_call:
        raise LlmCallCeilingExceeded(
            f"call site {callsite_id!r} estimates ${estimate_usd:.4f} against its own "
            f"declared ceiling of ${site.max_usd_per_call:.4f}. The call is REFUSED. A "
            "ceiling that reserved the smaller of the two would admit this call and then "
            "let it bill the estimate, which is a cap that binds the bookkeeping and not "
            "the spend; raise `max_usd_per_call` in llm_callsites.yaml deliberately, or "
            "ask for less."
        )
    reserved_usd = cap.reserve(site.max_usd_per_call, callsite_id=callsite_id)

    # Pace, on the same cap the admission check above just used, before the
    # provider is reached (`alpha-engine-config-I9977` deliverable 1). Worst
    # case — already-spent plus this call's own admitted ceiling — is the
    # same pessimism `reserve` itself applies, for the same reason: pacing on
    # an optimistic estimate is pacing that is behind every time the estimate
    # is low. `cap.anchor is None` is an opt-out (no weekly anchor wired),
    # not a failure — pace enforcement above stays unconditional either way.
    if cap.anchor is not None:
        pace_now = now if now is not None else dt.datetime.now(dt.UTC)
        pace = spend_pace(
            cap.spent_usd + reserved_usd,
            cap_usd=cap.cap_usd,
            now=pace_now,
            anchor=cap.anchor,
        )
        # PACE_OVERRUN_MARGIN's docstring: the early-window case, handled
        # deliberately — a raw `pace.exceeded` fires on nearly any nonzero
        # spend in the first moments of the window, so the durable artifact
        # is written only once the reading clears the materiality margin, a
        # front-loaded burst does (see TestPaceAtCallTime), a trivial early
        # call does not.
        if pace.overrun > PACE_OVERRUN_MARGIN:
            ctx.record_metric(
                pace_metric(
                    pace,
                    cap=cap,
                    now=pace_now,
                    source_path=f"runs/{ctx.job}/{ctx.trading_day.isoformat()}/run.json",
                )
            )

    # THE ROUTER, and nothing beside it (`alpha-engine-config-I9969`).
    #
    # `resolve_group_spec` is the supported way to address a model GROUP: it
    # returns a spec pointing at the authenticated router edge, behind which
    # the cross-provider fallback chain is walked, the request body is
    # scanned by the egress proxy, and the call is attributed to this
    # consumer. What it is NOT is the shape this replaces —
    # `krepis.llm_config.resolve_model_spec`, which read a `provider:model`
    # string out of a per-class SSM parameter and returned a spec naming one
    # vendor endpoint directly. That is a second routing plane with a single
    # rung: one 429, one 5xx or one read timeout is terminal, exactly the
    # shape of #9728, and nothing about it is visible to the proxy.
    #
    # `wire="openai"`: this call site builds an `LLMClient` on the openai
    # transport, so asking for the anthropic wire would let a fallback hand
    # it a URL its transport cannot speak.
    from krepis.llm import LLMClient
    from krepis.router import resolve_group_spec, route_is_degraded

    group = capability_group(capability_class)
    spec, route = resolve_group_spec(
        group,
        exec_context=_exec_context(),
        wire="openai",
    )
    client = LLMClient(spec, callsite_id=callsite_id, client_factory=client_factory)
    system, user_content = _system_and_user_content(messages)
    result = client.complete(system=system, user_content=user_content, **kwargs)
    usage = result.usage
    usd = float(usage.provider_cost_usd or 0.0)
    # The manifest is written BEFORE the cap re-check, so an overrun that
    # raises below still lands in `llm_calls` and `cost_usd`: the run fails
    # AND the artifact says what the money bought (§2 row 7).
    ctx.record_llm_call(
        {
            "callsite_id": callsite_id,
            "model_requested": capability_class,
            "model_served": result.model,
            # RESOLUTION already fell past the group's primary entry
            # (`model-router-policy` R12: serving from a fallback is an
            # alert, not a log line). Recorded on the row rather than
            # re-derived by each reader, so a fallback-served call is
            # DISTINGUISHABLE from a primary-served one in the durable
            # record — the half `crucible-evaluator/director/agent.py`
            # already carries on its artifact, and the half this package
            # could not carry at all while the spec came from SSM.
            #
            # It answers the resolve-time question only. On the router-edge
            # route the chain is walked by the proxy, so WHICH entry served
            # arrives at call time — as `fallback_used` and
            # `served_deployment` below, which is why the fields are recorded
            # together and none of them replaces another.
            "route_degraded": bool(route_is_degraded(route)),
            # The CALL-TIME answer (`alpha-engine-config-I10006`), from the
            # result rather than from the route. `route_is_degraded` asks
            # whether the route OBJECT declares a degraded shape; on the
            # `litellm_proxy` route — the one krepis prefers whenever the
            # health probe answers, and therefore the one v2 actually takes —
            # it returned `False` unconditionally, so the predicate could not
            # fire at all on the live path. `LLMResult.fallback_used` is the
            # fact about THIS call: the primary failed and the chain was
            # walked. Stamping the weaker predicate here does not leave the
            # field missing, it leaves it FALSE beside a fallback-served
            # call, and a run that grades a model nobody selected is worse
            # than one that admits it cannot say.
            #
            # Read as an ATTRIBUTE, not with a `getattr(..., False)` default:
            # a krepis that withdrew the field would then record `false` on
            # every call forever, which is the `dropped_params` failure mode
            # (I7232) the field itself was added to end. Rule 5 — an absent
            # contract raises here rather than degrading into a plausible
            # answer.
            "fallback_used": bool(result.fallback_used),
            # Which deployment the router reported. `None` is the router
            # reporting none, kept distinct from the field being absent:
            # `model_served` is the resolved upstream id, two deployments can
            # share one, and the comparison deciding `fallback_used` happens
            # at the deployment layer — so this is not reconstructible from
            # `model_served` and is recorded rather than derived.
            "served_deployment": result.served_deployment,
            "tokens_in": int(usage.input_tokens),
            "tokens_out": int(usage.output_tokens),
            "cache_read": int(usage.cache_read_tokens),
            "cache_write": int(usage.cache_create_tokens),
            "usd": usd,
        }
    )
    cap.record(usd, reserved_usd=reserved_usd, callsite_id=callsite_id)
    return result


# --------------------------------------------------------------------------
# The enumerator.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    """One place a model can be reached that the registry does not cover."""

    path: str
    lineno: int
    kind: str
    detail: str

    def describe(self) -> str:
        return f"{self.path}:{self.lineno}: {self.kind} — {self.detail}"


def audit_call_sites(root: Path, *, registry: dict[str, CallSite] | None = None) -> list[Finding]:
    """Every unregistered way to reach a model under ``root``, from the AST.

    Enumerated from the CODE, never from a list: the point of the clause is
    that a call site added tomorrow is caught without anyone remembering to
    add it anywhere. Two kinds are reported, and the second is the one that
    makes the first meaningful:

    ``unregistered_callsite``
        a :func:`call` whose ``callsite_id`` is missing, computed rather than
        literal, or absent from the registry. Computed counts: an id a static
        reader cannot resolve is an id the cost join cannot resolve either.

    ``adapter_bypass``
        a module other than this one importing a provider or router module.
        Without this, coverage would only ever be as good as everyone's
        willingness to use the door.

    A file that does not parse is itself a finding. Skipping it would make a
    syntax error the way to become invisible to the audit.
    """
    known = load_registry() if registry is None else registry
    findings: list[Finding] = []
    for path in sorted(Path(root).rglob("*.py")):
        if any(part in {"__pycache__", ".venv", "build", "dist"} for part in path.parts):
            continue
        source = path.read_text(encoding="utf-8")
        display = str(path)
        try:
            tree = ast.parse(source, filename=display)
        except SyntaxError as exc:
            findings.append(
                Finding(display, exc.lineno or 0, "unparseable", f"cannot be audited: {exc.msg}")
            )
            continue
        is_adapter = path.resolve() == _ADAPTER_PATH
        findings.extend(_audit_module(tree, display, known, is_adapter=is_adapter))
    return findings


def _audit_module(
    tree: ast.Module, display: str, known: dict[str, CallSite], *, is_adapter: bool
) -> list[Finding]:
    findings: list[Finding] = []
    door_names: set[str] = set()
    module_aliases: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not is_adapter and _is_provider_module(alias.name):
                    findings.append(
                        Finding(
                            display,
                            node.lineno,
                            "adapter_bypass",
                            f"imports {alias.name!r}; every model call goes through "
                            "crucible.llm.call, which is where the cap and the registry are",
                        )
                    )
                if alias.name == "crucible.llm":
                    module_aliases.add(alias.asname or "crucible.llm")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if not is_adapter and _is_provider_module(module):
                findings.append(
                    Finding(
                        display,
                        node.lineno,
                        "adapter_bypass",
                        f"imports from {module!r}; every model call goes through "
                        "crucible.llm.call, which is where the cap and the registry are",
                    )
                )
            if module in ("crucible.llm", "llm"):
                for alias in node.names:
                    if alias.name == "call":
                        door_names.add(alias.asname or "call")
            if module == "crucible" and any(a.name == "llm" for a in node.names):
                module_aliases.update(a.asname or "llm" for a in node.names if a.name == "llm")

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not _is_door_call(
            node.func, door_names, module_aliases
        ):
            continue
        literal = _literal_callsite_id(node)
        if literal is None:
            findings.append(
                Finding(
                    display,
                    node.lineno,
                    "unregistered_callsite",
                    "reaches a model with no literal `callsite_id=`; an id a static reader "
                    "cannot resolve is an id the spend join cannot resolve either",
                )
            )
        elif literal not in known:
            findings.append(
                Finding(
                    display,
                    node.lineno,
                    "unregistered_callsite",
                    f"callsite_id {literal!r} is absent from {CALLSITE_REGISTRY_PATH.name}",
                )
            )
    return findings


def _is_provider_module(name: str) -> bool:
    return any(name == module or name.startswith(f"{module}.") for module in PROVIDER_MODULES)


def _is_door_call(func: ast.expr, door_names: set[str], module_aliases: set[str]) -> bool:
    if isinstance(func, ast.Name):
        return func.id in door_names
    if isinstance(func, ast.Attribute) and func.attr == "call":
        return _dotted(func.value) in module_aliases | {"crucible.llm"}
    return False


def _dotted(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_dotted(node.value)}.{node.attr}"
    return ""


def _literal_callsite_id(node: ast.Call) -> str | None:
    for keyword in node.keywords:
        if keyword.arg != "callsite_id":
            continue
        if isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
            return keyword.value.value
        return None
    return None
