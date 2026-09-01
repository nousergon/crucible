"""The one way this package reaches a model: the cap, the registry, the audit.

Normative source: plan §2 row 3 (cost), §2 row 7 (transparency), §4.8 (LLM
access), §9.2 class 2 (token/cost telemetry).

Three things live here, and they are one thing:

1. **A declared, enforced spend cap.** :data:`DEFAULT_LLM_CAP_USD` is the
   per-weekly-run ceiling, resolvable per deployment through
   `crucible.config.Settings.llm_cap_usd`. A call that would carry the run
   past it RAISES :class:`LlmSpendCapExceeded` **before the provider is
   touched**, so the run fails with a reason naming the cap and the spend
   that would have happened does not happen. Pacing through the week comes
   from `krepis.usage_pacing`, which is what turns "we blew the budget on
   Saturday" into "we were ahead of pace on Tuesday".

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

from crucible.store import Store

__all__ = [
    "CALLSITE_REGISTRY_PATH",
    "CallSite",
    "DEFAULT_LLM_CAP_USD",
    "Finding",
    "LLM_CALLSITE_REGISTRY",
    "LlmSpendCapExceeded",
    "PACING_PERIOD",
    "PROVIDER_MODULES",
    "SpendCap",
    "audit_call_sites",
    "call",
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
        registry[str(callsite_id)] = CallSite(
            callsite_id=str(callsite_id),
            purpose=str(row["purpose"]),
            capability_class=str(row["capability_class"]),
            max_usd_per_call=float(row["max_usd_per_call"]),
            owner=str(row["owner"]),
        )
    return registry


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
    for key in store.list_keys("runs/"):
        if not key.endswith("/run.json"):
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

    def headroom_usd(self) -> float:
        return self.cap_usd - self.spent_usd

    def reserve(self, estimate_usd: float, *, callsite_id: str) -> None:
        """Admit a call of at most ``estimate_usd``, or refuse it.

        Called before the provider is reached. A call admitted here may still
        cost more than its estimate — providers bill what they bill — so
        :meth:`record` is what closes the loop, and the next `reserve` sees
        the real figure.
        """
        if estimate_usd < 0:
            raise ValueError(f"a call estimate cannot be negative; got {estimate_usd}")
        if self.spent_usd + estimate_usd > self.cap_usd:
            raise LlmSpendCapExceeded(
                f"call site {callsite_id!r} was refused: it would carry this weekly run to "
                f"${self.spent_usd + estimate_usd:.4f} against a declared cap of "
                f"${self.cap_usd:.2f} (already spent ${self.spent_usd:.4f}). The run fails "
                "rather than overspending; raise `llm_cap_usd` deliberately in config, or "
                "cut what the run asks for."
            )

    def record(self, usd: float) -> None:
        """Book what a completed call actually cost."""
        self.spent_usd += float(usd)


def cap_metric(cap: SpendCap, *, now: dt.datetime, source_path: str) -> dict[str, Any]:
    """The MetricRecord that makes the cap observable (principle 7).

    Emitted on EVERY run that holds a cap, including the ones that spend
    nothing: a cap nobody publishes a figure against is a cap nobody is held
    to, and zero spend is a measurement.
    """
    fraction = cap.spent_usd / cap.cap_usd if cap.cap_usd else 0.0
    status = "BREACH" if cap.spent_usd > cap.cap_usd else "OK"
    return {
        "name": "llm_spend_usd",
        "module": "crucible.llm",
        "metric_type": "count",
        "value": round(cap.spent_usd, 6),
        "unit": "usd",
        "n_floor": 0,
        "status": status,
        "status_reason": (
            f"${cap.spent_usd:.4f} of a declared ${cap.cap_usd:.2f} per-weekly-run cap "
            f"({fraction:.1%}); a call that would exceed it is refused before the provider "
            "is reached, so the run fails rather than overspending"
        ),
        "source_path": source_path,
        "last_updated_utc": now.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "baseline": cap.cap_usd,
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
    if _looks_like_a_model_id(capability_class):
        raise ValueError(
            f"call site {callsite_id!r} asked for {capability_class!r}, which is a provider "
            "model id. Ask the router for a capability class or a registry group; a model "
            "id at a call site is the lock-in principle 8 forbids."
        )
    cap.reserve(min(estimate_usd, site.max_usd_per_call), callsite_id=callsite_id)

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
    cap.record(usd)
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
    return result


_MODEL_ID_MARKERS = (
    "gpt-",
    "claude-",
    "gemini-",
    "llama",
    "mistral",
    "glm-",
    "deepseek",
    "o3",
    "o4",
)


def _looks_like_a_model_id(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in _MODEL_ID_MARKERS)


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
