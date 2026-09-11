"""Factor-neutral performance attribution — beta/sector/size/residual.

Normative source: `alpha-engine-config-I10501` (phase-3 deliverable, quoted
verbatim from `crucible/gate.py::PHASE3_DELIVERABLES`):

    factor_neutral_attribution
      "factor-neutral attribution (beta/sector/size/residual, OLS on ArcticDB
      ETF series)"

This is WIRING, not a second implementation. The OLS regression, the
Ledoit-Wolf-shrunk factor covariance, and the ex-ante risk decomposition all
come from `nousergon_lib.quant.factor_risk` — this module supplies the
crucible-side shape: a strategy-tree-sourced factor spec (which ETF proxies
are the beta/sector/size factors — strategy edge, never hardcoded here,
mirroring how `crucible.portfolio.load_portfolio_params` resolves its
parameters), the realized-return decomposition on top of the fitted model,
and the versioned evidence document a phase-3 gate clause can read.

**Not the phase-1 attribution table.** `crucible/report.py` already writes
`report/{trading_day}/attribution.json` (`attribution.v1`) — a five-row
per-stage status table with zero references to beta, sector or residual. This
module's `factor_attribution.v1` is a different artifact for a different
question and does not touch that one.

**Return decomposition, in the OLS model's own units.** `estimate_factor_model`
fits, for each holding, ``y = a + B·x + e`` by OLS over the window — so summed
over the window, ``Σy = n·a + B·Σx + Σe``. The portfolio-level factor exposure
``x_p = Bᵀw`` (the same quantity `factor_risk.portfolio_risk` reports) times
the summed factor-return series is therefore the additive, in-model return
each factor is credited with; what the caller's *realized* portfolio return
does not carry in that sum is `residual_alpha` — the intercept, holding
selection, and everything the named factors do not span. This is a top-down
decomposition (gross return minus explained return), not a re-estimate of
`Σe` per holding: the realized gross return is the caller's own measured
figure (whatever priced the book), and only that figure ties exactly to what
was actually earned.

**Data provenance.** Every input return series (holdings, factors, benchmark)
is read by the caller through `crucible.data.sources.PriceSource` — this
module takes plain ``{name: return_series}`` mappings and never opens a
source itself, exactly as `nousergon_lib.quant.factor_risk` is
data-source-agnostic and `crucible.portfolio` composes it without becoming a
data client.

**Not yet wired to a gate clause or a scheduled job.** See the PR body for
what a follow-up must call.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator
from nousergon_lib.quant.factor_risk import estimate_factor_model, portfolio_risk

__all__ = [
    "ATTRIBUTION_FACTOR_CATEGORIES",
    "ATTRIBUTION_METRIC_NAME",
    "FACTOR_ATTRIBUTION_SCHEMA_VERSION",
    "AttributionFactorParams",
    "AttributionParamsError",
    "FactorDef",
    "attribution_metric_record",
    "compute_factor_attribution",
    "load_attribution_params",
    "manifest_records_factor_attribution",
    "params_digest",
]

#: The three factor categories the phase-3 deliverable names, plus the
#: residual is not a category — it is what the three categories don't span.
ATTRIBUTION_FACTOR_CATEGORIES: tuple[str, ...] = ("beta", "sector", "size")

FACTOR_ATTRIBUTION_SCHEMA_VERSION = "factor_attribution.v1"

#: The metric row name a gate clause matches on.
ATTRIBUTION_METRIC_NAME = "factor_attribution"

_SCHEMA_PATH = Path(__file__).parent / "schemas" / f"{FACTOR_ATTRIBUTION_SCHEMA_VERSION}.json"


def _evidence_validator() -> Draft202012Validator:
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


class AttributionParamsError(ValueError):
    """The strategy-tree attribution factor spec is absent or malformed."""


@dataclass(frozen=True)
class FactorDef:
    """One named factor: which category it is, and the ETF proxy that sources it."""

    category: str
    proxy: str

    def __post_init__(self) -> None:
        if self.category not in ATTRIBUTION_FACTOR_CATEGORIES:
            raise AttributionParamsError(
                f"factor category {self.category!r} is not one of {ATTRIBUTION_FACTOR_CATEGORIES}"
            )
        if not self.proxy:
            raise AttributionParamsError("a factor's proxy ticker must be non-empty")


@dataclass(frozen=True)
class AttributionFactorParams:
    """The factor spec a window is attributed against — strategy edge, not code.

    ``factors`` maps a factor name (e.g. ``"market"``, ``"sector_tech"``,
    ``"size"``) to its :class:`FactorDef`. Which tickers are the beta/sector/
    size proxies is a tuned choice that lives in the private strategy tree
    (`alpha-engine-config/strategy/`), loaded here exactly the way
    `crucible.portfolio.load_portfolio_params` resolves `PortfolioParams` —
    this module declares the SHAPE and refuses one short a category; it
    declares no ticker.
    """

    factors: dict[str, FactorDef]
    benchmark_proxy: str
    shrinkage: str = "ledoit_wolf"

    def __post_init__(self) -> None:
        if not self.factors:
            raise AttributionParamsError(
                "an attribution factor spec with no factors cannot decompose anything"
            )
        present = {f.category for f in self.factors.values()}
        missing = [c for c in ATTRIBUTION_FACTOR_CATEGORIES if c not in present]
        if missing:
            noun = "category" if len(missing) == 1 else "categories"
            raise AttributionParamsError(
                f"the attribution factor spec is missing {noun} {missing}; the phase-3 "
                "deliverable names beta, sector and size and a spec silently short one "
                "would decompose against fewer factors than it claims to."
            )
        if not self.benchmark_proxy:
            raise AttributionParamsError("benchmark_proxy must be a non-empty ticker")
        if self.shrinkage not in ("ledoit_wolf", "sample"):
            raise AttributionParamsError(
                f"shrinkage must be 'ledoit_wolf' or 'sample', got {self.shrinkage!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_proxy": self.benchmark_proxy,
            "shrinkage": self.shrinkage,
            "factors": {
                name: {"category": fd.category, "proxy": fd.proxy}
                for name, fd in sorted(self.factors.items())
            },
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], *, source: str) -> AttributionFactorParams:
        if "factors" not in payload or not isinstance(payload["factors"], Mapping):
            raise AttributionParamsError(f"{source}: expected an 'attribution.factors' mapping")
        if "benchmark_proxy" not in payload:
            raise AttributionParamsError(f"{source}: 'attribution.benchmark_proxy' is required")
        factors: dict[str, FactorDef] = {}
        for name, block in payload["factors"].items():
            if not isinstance(block, Mapping) or "category" not in block or "proxy" not in block:
                raise AttributionParamsError(
                    f"{source}: factor {name!r} must declare both 'category' and 'proxy'"
                )
            category = str(block["category"])
            proxy = str(block["proxy"])
            factors[str(name)] = FactorDef(category=category, proxy=proxy)
        return cls(
            factors=factors,
            benchmark_proxy=str(payload["benchmark_proxy"]),
            shrinkage=str(payload.get("shrinkage", "ledoit_wolf")),
        )


def load_attribution_params(path: Path | str) -> AttributionFactorParams:
    """Read the attribution factor spec from the private strategy tree.

    `alpha-engine-config/strategy/slots/attribution.yaml` in production
    (opened as a separate worktree/PR against `alpha-engine-config`, per
    `repository-tiering-policy` test 2 — the factor/proxy list is a tuned
    choice, not framework). An absent file RAISES: there is no default factor
    spec this repository can supply.
    """
    resolved = Path(path)
    if not resolved.exists():
        raise AttributionParamsError(
            f"no attribution factor spec at {resolved}. The beta/sector/size ETF proxy "
            "list is strategy edge and lives in the private strategy tree; there is no "
            "default this repository can invent."
        )
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or "attribution" not in payload:
        raise AttributionParamsError(
            f"{resolved}: expected a mapping carrying an 'attribution' block; "
            "the file declares none."
        )
    return AttributionFactorParams.from_mapping(payload["attribution"], source=str(resolved))


def params_digest(params: AttributionFactorParams) -> str:
    """A stable digest over the factor spec — see `crucible.portfolio.params_digest`."""
    canonical = json.dumps(params.to_dict(), sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_factor_attribution(
    *,
    trading_day: str,
    window_sessions: int,
    holding_returns: Mapping[str, Sequence[float]],
    weights: Mapping[str, float],
    factor_returns: Mapping[str, Sequence[float]],
    params: AttributionFactorParams,
    gross_return: float,
    cost_bps_total: float,
) -> dict[str, Any]:
    """Fit the factor model through `nousergon_lib.quant.factor_risk` and decompose.

    ``holding_returns`` / ``factor_returns`` are ``{name: daily_return_series}``
    maps, every series exactly ``window_sessions`` long, aligned to the same
    trading days — the caller (a future scheduled job) sources these through
    `crucible.data.sources.PriceSource`; this function never reads a store.

    ``gross_return`` is the caller's own measured realized portfolio return
    over the window (whatever priced the actual book); ``cost_bps_total`` is
    the same cost-model charge `crucible.portfolio.portfolio_evidence` already
    records, carried here rather than re-priced so gross and net tie to one
    number.

    Returns the validated `factor_attribution.v1` evidence document.
    """
    if window_sessions < 1:
        raise ValueError(
            f"attribution over {window_sessions} sessions; a window that spans no "
            "session produced no evidence and must not be recorded as though it had"
        )
    factor_names = set(params.factors)
    given_names = set(factor_returns)
    if factor_names != given_names:
        missing = sorted(factor_names - given_names)
        extra = sorted(given_names - factor_names)
        raise ValueError(
            f"factor_returns does not match the attribution spec's factors "
            f"(missing={missing}, extra={extra}). A factor the spec names but that "
            "carries no return series would be silently dropped from the model."
        )
    for name, series in factor_returns.items():
        if len(series) != window_sessions:
            raise ValueError(
                f"factor {name!r} has {len(series)} observations, expected {window_sessions}"
            )
    for ticker, series in holding_returns.items():
        if len(series) != window_sessions:
            raise ValueError(
                f"holding {ticker!r} has {len(series)} observations, expected {window_sessions}"
            )

    model = estimate_factor_model(holding_returns, factor_returns, shrinkage=params.shrinkage)
    risk = portfolio_risk(model, weights)
    exposures: dict[str, float] = risk["factor_exposures"]

    factor_rows: list[dict[str, Any]] = []
    category_totals = {c: 0.0 for c in ATTRIBUTION_FACTOR_CATEGORIES}
    explained_return = 0.0
    for name in sorted(params.factors):
        fdef = params.factors[name]
        exposure = float(exposures[name])
        # Additive in-model return contribution: exposure × the factor's own
        # summed return over the window (see module docstring — Σy = n·a +
        # B·Σx + Σe, so exposure·Σx_k is exactly what factor k is credited
        # with in that sum).
        contribution = exposure * float(sum(factor_returns[name]))
        category_totals[fdef.category] += contribution
        explained_return += contribution
        factor_rows.append(
            {
                "name": name,
                "category": fdef.category,
                "proxy": fdef.proxy,
                "exposure": exposure,
                "contribution_return": contribution,
            }
        )

    residual_alpha = float(gross_return) - explained_return
    net_return = float(gross_return) - float(cost_bps_total) / 1.0e4

    document = {
        "schema_version": FACTOR_ATTRIBUTION_SCHEMA_VERSION,
        "trading_day": trading_day,
        "engine": "crucible.attribution",
        "window_sessions": int(window_sessions),
        "factors": factor_rows,
        "category_totals": {
            "beta": category_totals["beta"],
            "sector": category_totals["sector"],
            "size": category_totals["size"],
            "residual": residual_alpha,
        },
        "gross_return": float(gross_return),
        "net_return": net_return,
        "cost_bps_total": float(cost_bps_total),
        "residual_alpha": residual_alpha,
        "risk": {
            "total_vol": float(risk["total_vol"]),
            "factor_vol": float(risk["factor_vol"]),
            "idio_vol": float(risk["idio_vol"]),
        },
        "model": {
            "shrinkage": params.shrinkage,
            "n_obs": int(window_sessions),
            "n_factors": len(params.factors),
            "n_holdings": len(holding_returns),
        },
        "params_digest": params_digest(params),
    }
    errors = sorted(_evidence_validator().iter_errors(document), key=lambda e: list(e.path))
    if errors:
        paths = "; ".join(
            f"{'/'.join(str(part) for part in e.path) or '<root>'}: {e.message}" for e in errors
        )
        raise ValueError(
            f"factor attribution evidence does not satisfy "
            f"{FACTOR_ATTRIBUTION_SCHEMA_VERSION}: {paths}"
        )
    return document


def attribution_metric_record(evidence: Mapping[str, Any], *, now_utc: str) -> dict[str, Any]:
    """The manifest metric row carrying the factor-attribution evidence.

    Mirrors `crucible.portfolio.portfolio_metric_record`: the required core of
    `run_manifest.v2`'s open MetricRecord shape is filled here so a job body
    records it with one call. ``now_utc`` is passed rather than read from a
    clock, per the same rule.
    """
    return {
        "name": ATTRIBUTION_METRIC_NAME,
        "module": "crucible.attribution",
        "metric_type": "attribution",
        "n_floor": 1,
        "status": "OK",
        "status_reason": (
            f"factor attribution over {evidence['window_sessions']} session(s): "
            f"gross={evidence['gross_return']:.6f} net={evidence['net_return']:.6f} "
            f"residual_alpha={evidence['residual_alpha']:.6f}"
        ),
        "source_path": "crucible/attribution.py",
        "last_updated_utc": now_utc,
        "value": float(evidence["residual_alpha"]),
        "unit": "return_fraction",
        "horizon_trading_days": int(evidence["window_sessions"]),
        "factor_attribution": dict(evidence),
    }


def manifest_records_factor_attribution(manifest: Mapping[str, Any]) -> dict[str, Any] | None:
    """The attribution evidence on ``manifest``, or None if it carries none.

    The predicate a future phase-3 gate clause reads — mirrors
    `crucible.portfolio.manifest_records_portfolio_engine` exactly, for the
    same reason: the shape lives beside the producer so a reader calls one
    function rather than restating the row's shape.
    """
    for row in manifest.get("metrics") or ():
        if not isinstance(row, Mapping):
            continue
        if row.get("name") != ATTRIBUTION_METRIC_NAME:
            continue
        evidence = row.get("factor_attribution")
        if not isinstance(evidence, Mapping):
            raise ValueError(
                f"manifest carries a {ATTRIBUTION_METRIC_NAME!r} metric row with no "
                "`factor_attribution` payload. A row that names the engine without "
                "carrying its evidence would read as proof to any clause counting rows."
            )
        if evidence.get("schema_version") != FACTOR_ATTRIBUTION_SCHEMA_VERSION:
            raise ValueError(
                f"factor attribution evidence declares schema_version "
                f"{evidence.get('schema_version')!r}; this reader understands "
                f"{FACTOR_ATTRIBUTION_SCHEMA_VERSION!r}. A document read against the "
                "wrong version is read wrong, not read approximately."
            )
        return dict(evidence)
    return None
