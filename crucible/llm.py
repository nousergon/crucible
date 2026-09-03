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
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from krepis.usage_pacing import PaceStatus, pace_check

from crucible.keys import RUNS_ROOT, is_manifest_key
from crucible.store import Store

__all__ = [
    "CALLSITE_REGISTRY_PATH",
    "CallSite",
    "DEFAULT_LLM_CAP_USD",
    "DEFAULT_LLM_CAP_USD_MEASURED",
    "Finding",
    "LLM_CALLSITE_REGISTRY",
    "LlmCallCeilingExceeded",
    "LlmSpendCapExceeded",
    "LlmSpendOverrun",
    "PACING_PERIOD",
    "PROVIDER_MODULES",
    "SpendCap",
    "audit_call_sites",
    "call",
    "load_capability_classes",
    "load_registry",
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
PROVIDER_MODULES: frozenset[str] = frozenset(
    {
        "krepis.llm",
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


@lru_cache(maxsize=1)
def load_registry() -> dict[str, CallSite]:
    """``LLM_CALLSITE_REGISTRY``, read from ``llm_callsites.yaml``.

    A missing or malformed file RAISES. An unreadable registry read as empty
    would make every call site unregistered and every coverage check vacuous
    at the same time — the registry would report 100% coverage of nothing.
    """
    if not CALLSITE_REGISTRY_PATH.is_file():
        raise FileNotFoundError(
            f"the LLM call-site registry is missing at {CALLSITE_REGISTRY_PATH}. It "
            "ships inside the package; its absence is a broken build, not an empty "
            "registry."
        )
    document = yaml.safe_load(CALLSITE_REGISTRY_PATH.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or "callsites" not in document:
        raise ValueError(
            f"{CALLSITE_REGISTRY_PATH} carries no `callsites` mapping. An empty "
            "registry is written `callsites: {}`, so that 'no call sites' is a "
            "recorded fact rather than a parse that fell through."
        )
    rows = document["callsites"] or {}
    if not isinstance(rows, dict):
        raise ValueError(f"{CALLSITE_REGISTRY_PATH}: `callsites` must be a mapping of id -> row")
    registry: dict[str, CallSite] = {}
    for callsite_id, row in rows.items():
        missing = [
            f for f in ("purpose", "capability_class", "max_usd_per_call", "owner") if f not in row
        ]
        if missing:
            raise ValueError(
                f"call site {callsite_id!r} declares no {', '.join(missing)}. Every field "
                "is required: a row that names no owner or no ceiling records the id and "
                "nothing anyone can act on."
            )
        _require_capability_class(str(row["capability_class"]), callsite_id=str(callsite_id))
        registry[str(callsite_id)] = CallSite(
            callsite_id=str(callsite_id),
            purpose=str(row["purpose"]),
            capability_class=str(row["capability_class"]),
            max_usd_per_call=float(row["max_usd_per_call"]),
            owner=str(row["owner"]),
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
    document = yaml.safe_load(CALLSITE_REGISTRY_PATH.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or "capability_classes" not in document:
        raise ValueError(
            f"{CALLSITE_REGISTRY_PATH} declares no `capability_classes` list. It is the "
            "allowlist `crucible.llm.call` admits against; its absence is a broken build, "
            "not an empty allowlist, which is written `capability_classes: []`."
        )
    rows = document["capability_classes"] or []
    if not isinstance(rows, list) or any(not isinstance(r, str) or not r for r in rows):
        raise ValueError(
            f"{CALLSITE_REGISTRY_PATH}: `capability_classes` must be a list of non-empty "
            "strings naming router capability classes."
        )
    return tuple(rows)


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
        manifest = json.loads(store.get_bytes(key).decode("utf-8"))
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
# The one call.
# --------------------------------------------------------------------------


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
    **kwargs: Any,
) -> Any:
    """Reach a model, once, through the router — the only door in this package.

    ``ctx`` is the :class:`~crucible.runner.RunContext`: the call's tokens,
    cost, requested model and served model land in its manifest, so §9.2
    class 2 is a property of the door rather than of each caller remembering.

    ``capability_class`` is a router group. A value that looks like a vendor
    model id is REFUSED — a call site naming a model is a call site that has
    to be edited when the provider changes (principle 8).

    ``registry`` and ``client_factory`` are injection points, in that order:
    the first lets a test exercise the cap against a call site that does not
    exist in production, the second lets it do so without a provider. Neither
    weakens the audit, which reads the code rather than this function's
    arguments — a call site injecting its own registry is still a call site,
    and :func:`audit_call_sites` still demands a literal registered id.
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

    from krepis.llm import LLMClient
    from krepis.llm_config import resolve_model_spec

    spec = resolve_model_spec(
        ssm_param=f"/crucible/llm/{capability_class}",
        env_var=f"CRUCIBLE_LLM_{capability_class.upper()}",
    )
    client = LLMClient(spec, callsite_id=callsite_id, client_factory=client_factory)
    result = client.complete(messages=messages, **kwargs)
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
            "tokens_in": int(usage.input_tokens),
            "tokens_out": int(usage.output_tokens),
            "cache_read": int(usage.cache_read_tokens),
            "cache_write": int(usage.cache_create_tokens),
            "usd": usd,
        }
    )
    cap.record(usd, reserved_usd=reserved_usd, callsite_id=callsite_id)
    return result


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
