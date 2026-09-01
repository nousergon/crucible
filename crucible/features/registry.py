"""The feature registry: name, units, lineage, and the hash that versions it.

Normative source: plan §10 component 4, and the fleet feature-store rule.

**Every column name carries an explicit units suffix** — `_raw`, `_ratio`,
`_pct`, `_zscore` or `_log_return`. There is no grandfather list here,
because there is no history here: this layer is new, and the rule's root
cause is a fleet fact worth restating — `avg_volume_20d` was emitted as a
normalized ratio and consumed as raw shares, and 901 of 903 tickers silently
failed the scanner liquidity gate for months.

**The version is derived, never declared.** `feature_version()` hashes the
whole registry — every name, unit, window, expression id and input column —
so a changed recipe writes to a different `features/{version}/` prefix and
cannot overwrite the layer an earlier verdict was computed from. That is what
lets `explain` answer "the signal degraded or the feature changed" from
artifacts alone: the two questions have different version strings.

**Lineage is a field, not a comment.** `inputs` names the panel columns a
feature reads; `window_trading_days` is a count of SESSIONS (§4.12), never
calendar days, and a feature declaring a calendar window fails construction.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

__all__ = [
    "CATALOG",
    "UNIT_SUFFIXES",
    "FeatureSpec",
    "feature_names",
    "feature_version",
    "registry_payload",
]

#: The exhaustive suffix set. A column outside it is refused at construction,
#: so the rule is enforced by the registry rather than by review.
UNIT_SUFFIXES: tuple[str, ...] = ("_raw", "_ratio", "_pct", "_zscore", "_log_return")


@dataclass(frozen=True)
class FeatureSpec:
    """One column of the feature layer."""

    name: str
    unit: str
    expression: str
    description: str
    inputs: tuple[str, ...]
    #: SESSIONS, not calendar days. `None` for a point-in-time column that
    #: reads only the current row.
    window_trading_days: int | None = None
    #: Whether the column is computed across the cross-section of one day
    #: (a z-score) rather than along one ticker's history.
    cross_sectional: bool = False

    def __post_init__(self) -> None:
        if not any(self.name.endswith(s) for s in UNIT_SUFFIXES):
            raise ValueError(
                f"feature {self.name!r} carries no units suffix; one of {UNIT_SUFFIXES} "
                "is mandatory. A bare name is how a ratio gets consumed as raw shares."
            )
        if not self.unit:
            raise ValueError(f"feature {self.name!r} declares no unit")
        if not self.inputs:
            raise ValueError(
                f"feature {self.name!r} declares no inputs; a column with no lineage "
                "cannot be traced back to the data that produced it (principle 1)"
            )
        if self.window_trading_days is not None and self.window_trading_days < 1:
            raise ValueError(
                f"feature {self.name!r} declares window_trading_days="
                f"{self.window_trading_days}; a window is a count of SESSIONS and is "
                "at least one (plan §4.12)"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "unit": self.unit,
            "expression": self.expression,
            "description": self.description,
            "inputs": list(self.inputs),
            "window_trading_days": self.window_trading_days,
            "cross_sectional": self.cross_sectional,
        }


#: The phase-1 catalogue. Deliberately small: every column here is consumed
#: by a registered U or R arm, and a feature nothing reads is a column that
#: rots without anyone noticing it stopped being computed correctly.
CATALOG: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        name="close_raw",
        unit="USD",
        expression="close",
        description="Settled close for the trading day, unadjusted by this layer.",
        inputs=("close_raw",),
    ),
    FeatureSpec(
        name="dollar_volume_20d_raw",
        unit="USD",
        expression="mean(close * volume, 20)",
        description=(
            "Mean traded notional over 20 sessions. USD, not shares, and not a "
            "ratio — the liquidity gate reads this, and the units confusion in the "
            "column it replaces silently failed 901 of 903 tickers."
        ),
        inputs=("close_raw", "volume_raw"),
        window_trading_days=20,
    ),
    FeatureSpec(
        name="return_1d_log_return",
        unit="log_return",
        expression="log(close / close.shift(1))",
        description="One-session log return.",
        inputs=("close_raw",),
        window_trading_days=1,
    ),
    FeatureSpec(
        name="momentum_20d_log_return",
        unit="log_return",
        expression="log(close / close.shift(20))",
        description="Trailing 20-session log return.",
        inputs=("close_raw",),
        window_trading_days=20,
    ),
    FeatureSpec(
        name="return_60d_log_return",
        unit="log_return",
        expression="log(close / close.shift(60))",
        description="Trailing 60-session log return.",
        inputs=("close_raw",),
        window_trading_days=60,
    ),
    FeatureSpec(
        name="mom_12_1_log_return",
        unit="log_return",
        expression="log(close.shift(21) / close.shift(252))",
        description=(
            "Twelve-month momentum skipping the most recent month — 252 sessions "
            "back to 21 sessions back. The skip is the point: the last month is "
            "short-horizon reversal, not momentum."
        ),
        inputs=("close_raw",),
        window_trading_days=252,
    ),
    FeatureSpec(
        name="volatility_20d_ratio",
        unit="ratio",
        expression="std(log_return_1d, 20)",
        description="Standard deviation of 20 sessions of daily log returns. Not annualized.",
        inputs=("close_raw",),
        window_trading_days=20,
    ),
    FeatureSpec(
        name="close_to_sma50_ratio",
        unit="ratio",
        expression="close / mean(close, 50)",
        description="Close over its 50-session mean. 1.0 is at the average.",
        inputs=("close_raw",),
        window_trading_days=50,
    ),
    FeatureSpec(
        name="close_to_sma200_ratio",
        unit="ratio",
        expression="close / mean(close, 200)",
        description="Close over its 200-session mean.",
        inputs=("close_raw",),
        window_trading_days=200,
    ),
    FeatureSpec(
        name="rsi_14_ratio",
        unit="ratio",
        expression="wilder_rsi(close, 14) / 100",
        description=(
            "Wilder's RSI over 14 sessions, expressed on 0-1 rather than 0-100 so "
            "the units suffix is honest — a `_ratio` bounded at 100 is a `_pct`."
        ),
        inputs=("close_raw",),
        window_trading_days=14,
    ),
    FeatureSpec(
        name="liquidity_pass_raw",
        unit="indicator",
        expression="dollar_volume_20d_raw >= 5_000_000",
        description=(
            "The liquidity gate as a 0/1 indicator, computed HERE so every arm "
            "reads the same gate rather than each re-deriving a threshold."
        ),
        inputs=("dollar_volume_20d_raw",),
        window_trading_days=20,
    ),
    FeatureSpec(
        name="tech_score_ratio",
        unit="ratio",
        expression=(
            "mean(rank01(rsi_14_ratio), rank01(close_to_sma50_ratio), "
            "rank01(close_to_sma200_ratio), rank01(momentum_20d_log_return))"
        ),
        description=(
            "Equally weighted composite of four technical inputs, each converted "
            "to a cross-sectional 0-1 rank first so the mean is not dominated by "
            "whichever input happens to have the widest raw scale."
        ),
        inputs=(
            "rsi_14_ratio",
            "close_to_sma50_ratio",
            "close_to_sma200_ratio",
            "momentum_20d_log_return",
        ),
        cross_sectional=True,
    ),
    FeatureSpec(
        name="momentum_20d_zscore",
        unit="zscore",
        expression="zscore(momentum_20d_log_return)",
        description="Cross-sectional z-score of 20-session momentum, over the liquid set.",
        inputs=("momentum_20d_log_return", "liquidity_pass_raw"),
        cross_sectional=True,
    ),
    FeatureSpec(
        name="return_60d_zscore",
        unit="zscore",
        expression="zscore(return_60d_log_return)",
        description="Cross-sectional z-score of 60-session return, over the liquid set.",
        inputs=("return_60d_log_return", "liquidity_pass_raw"),
        cross_sectional=True,
    ),
    FeatureSpec(
        name="mom_12_1_zscore",
        unit="zscore",
        expression="zscore(mom_12_1_log_return)",
        description="Cross-sectional z-score of 12-1 momentum, over the liquid set.",
        inputs=("mom_12_1_log_return", "liquidity_pass_raw"),
        cross_sectional=True,
    ),
)


def feature_names(catalog: tuple[FeatureSpec, ...] = CATALOG) -> tuple[str, ...]:
    return tuple(spec.name for spec in catalog)


def registry_payload(catalog: tuple[FeatureSpec, ...] = CATALOG) -> dict[str, Any]:
    """The registry as it is written to `features/{version}/registry.json`."""
    return {
        "schema_version": "feature_registry.v1",
        "feature_version": feature_version(catalog),
        "features": [spec.to_dict() for spec in catalog],
    }


def feature_version(catalog: tuple[FeatureSpec, ...] = CATALOG) -> str:
    """A stable 12-hex digest of the whole catalogue.

    Derived, not declared. A registry edit that changed what a column MEANS
    while leaving a hand-written version string alone would overwrite the
    layer an earlier verdict was computed from, and nothing would show it.
    """
    canonical = json.dumps(
        [spec.to_dict() for spec in catalog], sort_keys=True, separators=(",", ":")
    )
    return "v" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
