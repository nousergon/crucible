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
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from crucible.models import FeatureRegistryDocument

__all__ = [
    "CATALOG",
    "FEATURE_REGISTRY_SCHEMA_PATH",
    "FEATURE_REGISTRY_SCHEMA_VERSION",
    "UNIT_SUFFIXES",
    "FeatureRegistryValidationError",
    "FeatureSpec",
    "PENDING_COLUMNS",
    "PendingColumn",
    "feature_names",
    "feature_version",
    "load_registry_schema",
    "registry_payload",
    "render_catalog_markdown",
    "validate_registry_payload",
]

#: The versioned producer/consumer contract of the feature layer, and the
#: `schema_version` the artifact at `features/{version}/registry.json`
#: carries. Written at BIRTH of the interface rather than lifted out of it
#: later (M0 contract discipline): the two tracks that build the producer and
#: the consumer read the same declared document instead of each holding a
#: defensible reading of an undeclared one, which is how
#: `alpha-engine-config-I9772`'s three disagreements happened.
FEATURE_REGISTRY_SCHEMA_VERSION = "feature_registry.v1"
FEATURE_REGISTRY_SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent / "schemas" / "feature_registry.v1.json"
)


class FeatureRegistryValidationError(ValueError):
    """A registry payload that does not conform to its own schema.

    Always raised, never logged and swallowed. The layer's whole claim is
    that R and M read the same declared columns; a registry document nobody
    validated, read as valid, is worse than none.
    """


#: The exhaustive suffix set. A column outside it is refused at construction,
#: so the rule is enforced by the registry rather than by review.
UNIT_SUFFIXES: tuple[str, ...] = ("_raw", "_ratio", "_pct", "_zscore", "_log_return")

#: The suffix-to-unit contract, exhaustive over `UNIT_SUFFIXES`. A
#: NORMALIZED suffix (`_ratio`, `_pct`, `_zscore`, `_log_return`) pins the
#: `unit` field to exactly one string — the suffix names the unit, so there
#: is nothing else it could legitimately be. `_raw` is the one suffix with
#: no single unit, because "raw" means "not normalized by this layer", and
#: an unnormalized column can carry any concrete unit (USD, shares, a beta
#: coefficient, a 0/1 indicator) — what it may NOT do is claim a normalized
#: unit while being unnormalized, which is exactly the defect this maps
#: closes: `avg_volume_20d` was emitted as a ratio and consumed as raw
#: shares, and a construction-time check that compared only the SUFFIX to
#: the allowed-suffix list — never the suffix to the declared `unit` — let
#: `FeatureSpec(name="avg_volume_20d_raw", unit="ratio", ...)` construct
#: successfully (defect #11, 2026-09-01 adversarial review).
_NORMALIZED_UNIT_BY_SUFFIX: dict[str, str] = {
    "_ratio": "ratio",
    "_pct": "pct",
    "_zscore": "zscore",
    "_log_return": "log_return",
}

#: The unit words `_raw` may never declare — each one is already the exact
#: unit a normalized suffix owns above. A `_raw` column claiming one of
#: these is a normalized value wearing the unnormalized suffix, the mirror
#: image of the bug this registry exists to catch.
_NORMALIZED_UNIT_WORDS: frozenset[str] = frozenset(_NORMALIZED_UNIT_BY_SUFFIX.values())


@dataclass(frozen=True)
class FeatureSpec:
    """One column of the feature layer."""

    name: str
    unit: str
    expression: str
    description: str
    inputs: tuple[str, ...]
    #: Whether the column's VALUE varies across the day's cross-section
    #: (`False`) or is one value repeated identically across every ticker on
    #: a day, by construction (`True`) — `market_return_1d_log_return` is
    #: the phase-1 example. **Declared, never inferred from variance at
    #: runtime**: a column that happens to be constant on one quiet day is
    #: not market-wide, and a market-wide column's whole point is that it is
    #: constant on EVERY day. No default on purpose — a new catalogue entry
    #: that omits this argument fails at construction (`TypeError`, before
    #: any test runs), so a column cannot silently inherit a guess about its
    #: own distributional shape. `crucible.drift_inputs` reads this to
    #: choose the comparison a PSI-style drift check runs: a cross-sectional
    #: column keeps the cross-section-vs-cross-section reading, a
    #: market-wide one is compared ALONG TIME instead, because a point mass
    #: measured against pooled point masses reads BREACH by construction
    #: (`alpha-engine-config-I10071`) regardless of what the market did.
    market_wide: bool
    #: SESSIONS, not calendar days. `None` for a point-in-time column that
    #: reads only the current row.
    window_trading_days: int | None = None
    #: Whether the column is COMPUTED across the cross-section of one day
    #: (a z-score or a cross-sectional mean) rather than along one ticker's
    #: history. Orthogonal to `market_wide`: `tech_score_ratio` is computed
    #: cross-sectionally (`cross_sectional=True`) but its VALUE still varies
    #: per ticker (`market_wide=False`); `market_return_1d_log_return` is
    #: both (computed as a cross-sectional mean, and its value is identical
    #: for every ticker that day).
    cross_sectional: bool = False

    def __post_init__(self) -> None:
        matched = [s for s in UNIT_SUFFIXES if self.name.endswith(s)]
        if not matched:
            raise ValueError(
                f"feature {self.name!r} carries no units suffix; one of {UNIT_SUFFIXES} "
                "is mandatory. A bare name is how a ratio gets consumed as raw shares."
            )
        if not self.unit:
            raise ValueError(f"feature {self.name!r} declares no unit")
        # The longest matching suffix: `_log_return` must not be shadowed by a
        # coincidental shorter match, though none of `UNIT_SUFFIXES` overlaps
        # today — this is the check staying correct if one ever does.
        suffix = max(matched, key=len)
        if suffix in _NORMALIZED_UNIT_BY_SUFFIX:
            expected_unit = _NORMALIZED_UNIT_BY_SUFFIX[suffix]
            if self.unit != expected_unit:
                raise ValueError(
                    f"feature {self.name!r} carries suffix {suffix!r} but declares "
                    f"unit={self.unit!r}; suffix {suffix!r} means unit={expected_unit!r} "
                    "and nothing else. A suffix that disagrees with the declared unit is "
                    "the `avg_volume_20d` defect: emitted as a ratio, consumed as raw "
                    "shares, and 901 of 903 tickers silently failed the liquidity gate "
                    "for months."
                )
        elif self.unit in _NORMALIZED_UNIT_WORDS:
            raise ValueError(
                f"feature {self.name!r} carries suffix '_raw' but declares "
                f"unit={self.unit!r}, which is a NORMALIZED unit. '_raw' means "
                "unnormalized — a raw column claiming a normalized unit is the same "
                "defect in the other direction: the name promises the consumer an "
                "unnormalized value and the declared unit says otherwise."
            )
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
        # `market_wide` IS written here (`alpha-engine-config-I10114`): the
        # document at `features/{version}/registry.json` is the declared
        # producer/consumer contract for the feature layer, and a reader of
        # that document alone previously could not tell a market-wide column
        # from a cross-sectional one without also importing `CATALOG`. The
        # schema (`feature_registry.v1.json`) declares `market_wide` as an
        # ADDITIVE OPTIONAL property rather than required, because a
        # document already written under the old schema (before this field
        # existed) lacks it and must keep validating — the same shape as
        # `window_trading_days` joining the schema in an earlier revision.
        # This dataclass still declares `market_wide` with no default, so
        # every catalogue column carries a real value here; "optional" is a
        # property of the WIRE SCHEMA (old documents may omit it), not of
        # what this producer emits (it never omits it).
        #
        # Deliberately EXCLUDED from `feature_version()`'s hash input (see
        # `_hashed_dict` below): `market_wide` is descriptive of an existing
        # constant, not a change to what any column computes, and folding it
        # into the hash the moment the schema changed would flip
        # `feature_version()` for the whole `CATALOG`, writing every column
        # to a new `features/{version}/` prefix and orphaning every
        # `run.json`/verdict that pointed at the old one.
        return {
            "name": self.name,
            "unit": self.unit,
            "expression": self.expression,
            "description": self.description,
            "inputs": list(self.inputs),
            "window_trading_days": self.window_trading_days,
            "cross_sectional": self.cross_sectional,
            "market_wide": self.market_wide,
        }


#: The phase-1 catalogue. Deliberately small: every column here is consumed
#: by a registered U or R arm, and a feature nothing reads is a column that
#: rots without anyone noticing it stopped being computed correctly.
CATALOG: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        name="close_raw",
        market_wide=False,
        unit="USD",
        expression="close",
        description="Settled close for the trading day, unadjusted by this layer.",
        inputs=("close_raw",),
    ),
    FeatureSpec(
        name="dollar_volume_20d_raw",
        market_wide=False,
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
        market_wide=False,
        unit="log_return",
        expression="log(close / close.shift(1))",
        description="One-session log return.",
        inputs=("close_raw",),
        window_trading_days=1,
    ),
    FeatureSpec(
        name="momentum_20d_log_return",
        market_wide=False,
        unit="log_return",
        expression="log(close / close.shift(20))",
        description="Trailing 20-session log return.",
        inputs=("close_raw",),
        window_trading_days=20,
    ),
    FeatureSpec(
        name="return_60d_log_return",
        market_wide=False,
        unit="log_return",
        expression="log(close / close.shift(60))",
        description="Trailing 60-session log return.",
        inputs=("close_raw",),
        window_trading_days=60,
    ),
    FeatureSpec(
        name="mom_12_1_log_return",
        market_wide=False,
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
        market_wide=False,
        unit="ratio",
        expression="std(log_return_1d, 20)",
        description="Standard deviation of 20 sessions of daily log returns. Not annualized.",
        inputs=("close_raw",),
        window_trading_days=20,
    ),
    FeatureSpec(
        name="close_to_sma50_ratio",
        market_wide=False,
        unit="ratio",
        expression="close / mean(close, 50)",
        description="Close over its 50-session mean. 1.0 is at the average.",
        inputs=("close_raw",),
        window_trading_days=50,
    ),
    FeatureSpec(
        name="close_to_sma200_ratio",
        market_wide=False,
        unit="ratio",
        expression="close / mean(close, 200)",
        description="Close over its 200-session mean.",
        inputs=("close_raw",),
        window_trading_days=200,
    ),
    FeatureSpec(
        name="rsi_14_ratio",
        market_wide=False,
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
        market_wide=False,
        unit="indicator",
        expression="dollar_volume_20d_raw >= liquidity_floor_usd()",
        description=(
            "The liquidity gate as a 0/1 indicator, computed HERE so every arm "
            "reads the same gate rather than each re-deriving a threshold."
        ),
        inputs=("dollar_volume_20d_raw",),
        window_trading_days=20,
    ),
    FeatureSpec(
        name="tech_score_ratio",
        market_wide=False,
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
        market_wide=False,
        unit="zscore",
        expression="zscore(momentum_20d_log_return)",
        description="Cross-sectional z-score of 20-session momentum, over the liquid set.",
        inputs=("momentum_20d_log_return", "liquidity_pass_raw"),
        cross_sectional=True,
    ),
    FeatureSpec(
        name="return_60d_zscore",
        market_wide=False,
        unit="zscore",
        expression="zscore(return_60d_log_return)",
        description="Cross-sectional z-score of 60-session return, over the liquid set.",
        inputs=("return_60d_log_return", "liquidity_pass_raw"),
        cross_sectional=True,
    ),
    FeatureSpec(
        name="mom_12_1_zscore",
        market_wide=False,
        unit="zscore",
        expression="zscore(mom_12_1_log_return)",
        description="Cross-sectional z-score of 12-1 momentum, over the liquid set.",
        inputs=("mom_12_1_log_return", "liquidity_pass_raw"),
        cross_sectional=True,
    ),
    # -- the M slot's declared demand (alpha-engine-config-I9765) ----------
    #
    # Lifted from the v1 implementation named in that issue's origin table,
    # `crucible-predictor/data/residual_momentum_features.py` with the
    # windows from `config/predictor.sample.yaml::residual_momentum`
    # (beta_window 60, window 252, skip_days 21, vol_window 20,
    # change_window 21). Two DECLARED deltas from v1, both documented on the
    # columns below: the market leg is the equal-weighted cross-section
    # rather than a sector ETF with a SPY fallback (plan §4.4 — the
    # benchmark is the population drawn from, never SPY; and this layer
    # carries no sector map), and the return stream is the layer's log
    # return rather than v1's simple `pct_change`, so a cumulative residual
    # is a sum rather than an approximation of one.
    FeatureSpec(
        name="market_return_1d_log_return",
        market_wide=True,
        unit="log_return",
        expression="mean(return_1d_log_return) over the day's cross-section",
        description=(
            "The market leg: the equal-weighted mean one-session log return across "
            "the day's whole cross-section. Identical for every ticker on a day, by "
            "construction — it is a market factor carried as a column so the beta "
            "below has a named input rather than a hidden intermediate. Not SPY: "
            "plan §4.4 grades against the population drawn from, and this layer "
            "carries no sector map to reproduce v1's sector-ETF benchmark."
        ),
        inputs=("return_1d_log_return",),
        window_trading_days=1,
        cross_sectional=True,
    ),
    FeatureSpec(
        name="beta_60d_raw",
        market_wide=False,
        unit="beta",
        expression=(
            "cov(return_1d_log_return, market_return_1d_log_return, 60) / "
            "var(market_return_1d_log_return, 60), shifted one session"
        ),
        description=(
            "Point-in-time market beta over 60 sessions. SHIFTED BY ONE SESSION, "
            "which is load-bearing: the beta used to residualize the return at t is "
            "estimated only on data through t-1, so the residual at t is not "
            "explained partly by itself."
        ),
        inputs=("return_1d_log_return", "market_return_1d_log_return"),
        window_trading_days=60,
    ),
    FeatureSpec(
        name="residual_return_1d_log_return",
        market_wide=False,
        unit="log_return",
        expression="return_1d_log_return - beta_60d_raw * market_return_1d_log_return",
        description=(
            "The idiosyncratic one-session return: what is left after the market "
            "leg is removed at the ticker's own point-in-time beta."
        ),
        inputs=(
            "return_1d_log_return",
            "market_return_1d_log_return",
            "beta_60d_raw",
        ),
        window_trading_days=1,
    ),
    FeatureSpec(
        name="residual_vol_20d_ratio",
        market_wide=False,
        unit="ratio",
        expression="std(residual_return_1d_log_return, 20)",
        description=(
            "Standard deviation of 20 sessions of residual daily log returns. Not "
            "annualized, matching `volatility_20d_ratio`. This is the denominator of "
            "the vol scaling below and an M-arm column in its own right."
        ),
        inputs=("residual_return_1d_log_return",),
        window_trading_days=20,
    ),
    FeatureSpec(
        name="residual_momentum_252d_skip21d_ratio",
        market_wide=False,
        unit="ratio",
        expression=(
            "sum(residual_return_1d_log_return, 231).shift(21) / "
            "(residual_vol_20d_ratio * sqrt(231))"
        ),
        description=(
            "Vol-scaled cumulative residual momentum (Blitz/Hanauer): 252 sessions "
            "back to 21 sessions back — the 12-1 skip-month convention, because the "
            "most recent month is short-horizon reversal — divided by the "
            "window-level residual volatility. An information ratio, so `_ratio`: "
            "a raw cumulative residual return would put the signal on a magnitude "
            "scale rather than a Sharpe-like one."
        ),
        inputs=("residual_return_1d_log_return", "residual_vol_20d_ratio"),
        window_trading_days=252,
    ),
    FeatureSpec(
        name="residual_momentum_252d_skip21d_zscore",
        market_wide=False,
        unit="zscore",
        expression="zscore(residual_momentum_252d_skip21d_ratio)",
        description=(
            "Cross-sectional z-score of vol-scaled residual momentum, over the "
            "liquid set. This is the column both M arms rank first."
        ),
        inputs=("residual_momentum_252d_skip21d_ratio", "liquidity_pass_raw"),
        cross_sectional=True,
    ),
    FeatureSpec(
        name="momentum_change_21d_log_return",
        market_wide=False,
        unit="log_return",
        expression=("sum(return_1d_log_return, 21) - sum(return_1d_log_return, 21).shift(21)"),
        description=(
            "Momentum acceleration: the trailing 21-session return minus the 21 "
            "sessions before it. Two consecutive windows, so 42 sessions of history."
        ),
        inputs=("return_1d_log_return",),
        window_trading_days=42,
    ),
    FeatureSpec(
        name="momentum_change_21d_zscore",
        market_wide=False,
        unit="zscore",
        expression="zscore(momentum_change_21d_log_return)",
        description=("Cross-sectional z-score of momentum acceleration, over the liquid set."),
        inputs=("momentum_change_21d_log_return", "liquidity_pass_raw"),
        cross_sectional=True,
    ),
    # -- the v3.0-meta L1 inputs (alpha-engine-config-I10695) -----------------
    #
    # The columns v1's serving model `v3.0-meta` reads for its momentum and
    # volatility heads that this layer did not already produce. Each is the v1
    # definition (`nousergon-data/features/feature_engineer.py`) re-expressed
    # under this layer's rules: a units suffix, a finite window (so a value
    # never depends on where the producer's panel starts), and a null where
    # v1 substituted a constant. The deltas are named per column.
    FeatureSpec(
        name="momentum_5d_log_return",
        market_wide=False,
        unit="log_return",
        expression="log(close) - log(close).shift(5)",
        description=(
            "Trailing 5-session log return. v1's `momentum_5d` was the simple return "
            "over the same window."
        ),
        inputs=("close_raw",),
        window_trading_days=5,
    ),
    FeatureSpec(
        name="atr_14_ratio",
        market_wide=False,
        unit="ratio",
        expression=(
            "mean(max(high - low, |high - close.shift(1)|, |low - close.shift(1)|), 14) / close"
        ),
        description=(
            "14-session average true range over close. A simple mean rather than v1's "
            "`atr_14_pct` EWM: an EWM has infinite memory, so its value would depend on "
            "how deep a panel the producer was handed and a heal would not reproduce "
            "a daily compile."
        ),
        inputs=("high_raw", "low_raw", "close_raw"),
        window_trading_days=14,
    ),
    FeatureSpec(
        name="vol_ratio_10_60_ratio",
        market_wide=False,
        unit="ratio",
        expression="std(return_1d_log_return, 10) / std(return_1d_log_return, 60)",
        description=(
            "Short over long realised volatility of daily log returns (sample std, "
            "ddof=1, as v1). Null where the 60-session volatility is zero; v1 filled "
            "that case with 1.0."
        ),
        inputs=("return_1d_log_return",),
        window_trading_days=60,
    ),
    FeatureSpec(
        name="dist_from_52w_high_ratio",
        market_wide=False,
        unit="ratio",
        expression="close / max(close, 252) - 1",
        description=(
            "Fractional distance below the 252-session closing high; 0 at the high, "
            "negative below it."
        ),
        inputs=("close_raw",),
        window_trading_days=252,
    ),
    FeatureSpec(
        name="dist_from_52w_low_ratio",
        market_wide=False,
        unit="ratio",
        expression="close / min(close, 252) - 1",
        description=(
            "Fractional distance above the 252-session closing low; 0 at the low, "
            "positive above it."
        ),
        inputs=("close_raw",),
        window_trading_days=252,
    ),
    # -- the attractiveness inputs (alpha-engine-config-I10721) ---------------
    #
    # v1's `scanner_cut` arms rank on six within-sector percentile pillars
    # (`crucible-research/scoring/factor_scoring.py::_BASELINE_COMPOSITE_DEFS`,
    # mapped to pillars by `scoring/composite.py::_PILLAR_TO_FACTOR_KEY`) plus a
    # 12-1 momentum variant (`_CHALLENGER_MOMENTUM_DEF`). The non-price inputs
    # arrive through `crucible.data.point_in_time`, which admits a value only
    # for sessions strictly after it became known and nulls a field that is
    # non-null but uninformative. A column below is null on a session its
    # input did not measure — never zero.
    FeatureSpec(
        name="sector_raw",
        market_wide=False,
        unit="gics_sector_label",
        expression="constituents.sector_map[ticker], fetched before the session",
        description=(
            "GICS sector name from the newest constituents snapshot FETCHED before the "
            "session (its recorded `fetched_at`, not its folder label). A label, not a "
            "number: the pillars rank within it. Null when the snapshot covers under 90% "
            "of the universe — every snapshot before 2026-04-30 carried the S&P 500 only."
        ),
        inputs=("point_in_time.sector.sector_map",),
    ),
    FeatureSpec(
        name="sector_earliest_snapshot_backfill_raw",
        market_wide=True,
        unit="indicator",
        expression=(
            "1 if point_in_time.sector.source_mode == 'earliest_snapshot_backfill', "
            "0 if 'point_in_time', null when the sector group measured nothing"
        ),
        description=(
            "Whether `sector_raw` and every within-sector pillar on this session rest on a "
            "BACKFILLED sector map (1) or on one fetched before the session (0). A backfill "
            "is used when no universe-covering constituents snapshot existed yet (before "
            "2026-05-01) and carries the known look-ahead of every GICS change between the "
            "session and that snapshot, e.g. the 2023-03-20 move of V, MA and PYPL from "
            "Information Technology to Financials. One value per session."
        ),
        # Brian's ruling (a) on alpha-engine-config-I10733 (2026-09-14).
        inputs=("point_in_time.sector.source_mode",),
    ),
    FeatureSpec(
        name="roe_ratio",
        market_wide=False,
        unit="ratio",
        expression="fundamental.roe (TTM return on equity, decimal, clipped [-1, 1])",
        description="Return on equity as v1 stores it.",
        inputs=("point_in_time.fundamental.roe",),
    ),
    FeatureSpec(
        name="debt_to_equity_div2_ratio",
        market_wide=False,
        unit="ratio",
        expression="fundamental.debt_to_equity (total debt / equity / 2, clipped [-3, 3])",
        description=(
            "Debt to equity DIVIDED BY TWO, as v1's collector normalises it. The divisor "
            "is in the name because a consumer reading this as D/E is off by a factor of 2."
        ),
        inputs=("point_in_time.fundamental.debt_to_equity",),
    ),
    FeatureSpec(
        name="gross_margin_ratio",
        market_wide=False,
        unit="ratio",
        expression="fundamental.gross_margin (TTM, 0-1 fraction)",
        description="Gross margin as a 0-1 fraction.",
        inputs=("point_in_time.fundamental.gross_margin",),
    ),
    FeatureSpec(
        name="current_ratio_div3_ratio",
        market_wide=False,
        unit="ratio",
        expression="fundamental.current_ratio (current assets / liabilities / 3, clipped [0, 3])",
        description="Current ratio DIVIDED BY THREE, as v1's collector normalises it.",
        inputs=("point_in_time.fundamental.current_ratio",),
    ),
    FeatureSpec(
        name="pe_div30_ratio",
        market_wide=False,
        unit="ratio",
        expression="fundamental.pe_ratio (trailing P/E / 30, clipped [-3, 3])",
        description=(
            "Trailing P/E DIVIDED BY THIRTY and clipped, as v1 stores it. Negative for a "
            "loss-making name, which the value pillar's inverted rank reads as cheap — "
            "v1's definition, carried unchanged."
        ),
        inputs=("point_in_time.fundamental.pe_ratio",),
    ),
    FeatureSpec(
        name="pb_div5_ratio",
        market_wide=False,
        unit="ratio",
        expression="fundamental.pb_ratio (price / book / 5, clipped [-3, 3])",
        description="Price to book DIVIDED BY FIVE and clipped, as v1 stores it.",
        inputs=("point_in_time.fundamental.pb_ratio",),
    ),
    FeatureSpec(
        name="fcf_yield_ratio",
        market_wide=False,
        unit="ratio",
        expression="fundamental.fcf_yield (TTM free cash flow / market cap, clipped [-0.5, 0.5])",
        description=(
            "Free-cash-flow yield. Unmeasured on every v1 snapshot before 2026-08-19, "
            "where it carried one distinct value across the universe."
        ),
        inputs=("point_in_time.fundamental.fcf_yield",),
    ),
    FeatureSpec(
        name="revenue_growth_3y_ratio",
        market_wide=False,
        unit="ratio",
        expression="fundamental.revenue_growth_3y (3-year revenue CAGR, decimal)",
        description="Three-year revenue CAGR.",
        inputs=("point_in_time.fundamental.revenue_growth_3y",),
    ),
    FeatureSpec(
        name="eps_growth_3y_ratio",
        market_wide=False,
        unit="ratio",
        expression="fundamental.eps_growth_3y (3-year EPS CAGR, decimal)",
        description="Three-year EPS CAGR.",
        inputs=("point_in_time.fundamental.eps_growth_3y",),
    ),
    FeatureSpec(
        name="capex_growth_5y_ratio",
        market_wide=False,
        unit="ratio",
        expression="fundamental.capex_growth_5y (5-year capex growth, decimal)",
        description="Five-year capital-expenditure growth; reinvestment intensity.",
        inputs=("point_in_time.fundamental.capex_growth_5y",),
    ),
    FeatureSpec(
        name="payout_ratio",
        market_wide=False,
        unit="ratio",
        expression="fundamental.payout_ratio (TTM dividends / net income, clipped [0, 2])",
        description="Dividend payout ratio as a decimal.",
        inputs=("point_in_time.fundamental.payout_ratio",),
    ),
    FeatureSpec(
        name="sustainable_growth_rate_ratio",
        market_wide=False,
        unit="ratio",
        expression="roe_ratio * (1 - payout_ratio)",
        description="Sustainable growth rate: return on equity retained. v1's derived factor.",
        inputs=("roe_ratio", "payout_ratio"),
    ),
    FeatureSpec(
        name="institutional_accumulation_raw",
        market_wide=False,
        unit="funds",
        expression=(
            "n_funds_increasing - n_funds_decreasing, 0 where fewer than 3 funds moved, "
            "from the newest 13F quarter whose filing deadline precedes the session"
        ),
        description=(
            "Net count of 13F filers adding to the position over the quarter. A count of "
            "funds, not a ratio. v1 also multiplied by `institutional_boost`; a positive "
            "scale does not change a percentile rank, so it is not carried. The quarter "
            "is admitted from the day after its 45-day filing deadline."
        ),
        inputs=(
            "point_in_time.institutional.n_funds_increasing",
            "point_in_time.institutional.n_funds_decreasing",
        ),
    ),
    FeatureSpec(
        name="return_120d_log_return",
        market_wide=False,
        unit="log_return",
        expression="log(close) - log(close).shift(120)",
        description=(
            "Trailing 120-session log return. v1's `return_120d` is the simple return "
            "over the same window; the two are monotone in each other, so every rank the "
            "pillars take of them is identical."
        ),
        inputs=("close_raw",),
        window_trading_days=120,
    ),
    FeatureSpec(
        name="quality_pillar_pct",
        market_wide=False,
        unit="pct",
        expression=(
            "wmean(sector_pct(roe_ratio) .30, 100 - sector_pct(debt_to_equity_div2_ratio) "
            ".25, sector_pct(gross_margin_ratio) .25, sector_pct(current_ratio_div3_ratio) .20)"
        ),
        description=(
            "v1 `quality_score`. Each component is a 0-100 percentile rank within the "
            "session's GICS sector; the weighted mean renormalises over the components a "
            "ticker has. Null for the whole session when the sector or any component is "
            "unmeasured on it, so the pillar never silently changes composition."
        ),
        inputs=(
            "sector_raw",
            "roe_ratio",
            "debt_to_equity_div2_ratio",
            "gross_margin_ratio",
            "current_ratio_div3_ratio",
        ),
        cross_sectional=True,
    ),
    FeatureSpec(
        name="value_pillar_pct",
        market_wide=False,
        unit="pct",
        expression=(
            "wmean(100 - sector_pct(pe_div30_ratio) .40, 100 - sector_pct(pb_div5_ratio) .30, "
            "sector_pct(fcf_yield_ratio) .30)"
        ),
        description="v1 `value_score`, within-sector, as the quality pillar.",
        inputs=("sector_raw", "pe_div30_ratio", "pb_div5_ratio", "fcf_yield_ratio"),
        cross_sectional=True,
    ),
    FeatureSpec(
        name="momentum_pillar_pct",
        market_wide=False,
        unit="pct",
        expression=(
            "wmean(sector_pct(momentum_20d_log_return) .30, sector_pct(return_60d_log_return) "
            ".25, sector_pct(return_120d_log_return) .20, sector_pct(dist_from_52w_high_ratio) "
            ".15, sector_pct(momentum_5d_log_return) .10)"
        ),
        description=(
            "v1 `momentum_score`, within-sector. The log returns rank identically to v1's "
            "simple returns."
        ),
        inputs=(
            "sector_raw",
            "momentum_20d_log_return",
            "return_60d_log_return",
            "return_120d_log_return",
            "dist_from_52w_high_ratio",
            "momentum_5d_log_return",
        ),
        cross_sectional=True,
    ),
    FeatureSpec(
        name="growth_pillar_pct",
        market_wide=False,
        unit="pct",
        expression=(
            "wmean(sector_pct(revenue_growth_3y_ratio) .30, sector_pct(eps_growth_3y_ratio) "
            ".30, sector_pct(sustainable_growth_rate_ratio) .25, "
            "sector_pct(capex_growth_5y_ratio) .15)"
        ),
        description="v1 `growth_score`, within-sector.",
        inputs=(
            "sector_raw",
            "revenue_growth_3y_ratio",
            "eps_growth_3y_ratio",
            "sustainable_growth_rate_ratio",
            "capex_growth_5y_ratio",
        ),
        cross_sectional=True,
    ),
    FeatureSpec(
        name="stewardship_pillar_pct",
        market_wide=False,
        unit="pct",
        expression=(
            "wmean(100 - sector_pct(payout_ratio) .35, sector_pct(capex_growth_5y_ratio) .35, "
            "sector_pct(institutional_accumulation_raw) .30)"
        ),
        description="v1 `stewardship_score` (config#2428 three-component form), within-sector.",
        inputs=(
            "sector_raw",
            "payout_ratio",
            "capex_growth_5y_ratio",
            "institutional_accumulation_raw",
        ),
        cross_sectional=True,
    ),
    FeatureSpec(
        name="defensiveness_pillar_pct",
        market_wide=False,
        unit="pct",
        expression=(
            "wmean(100 - sector_pct(volatility_20d_ratio) .50, "
            "100 - sector_pct(vol_ratio_10_60_ratio) .30, 100 - sector_pct(atr_14_ratio) .20)"
        ),
        description=(
            "v1 `low_vol_score`, the defensiveness pillar. Two declared deltas: realised "
            "volatility is the std of daily LOG returns (v1: annualised std of simple "
            "returns — a constant scale, rank-invariant, plus a second-order log/simple "
            "difference), and ATR is a 14-session simple mean (v1: an EWM, whose value "
            "depends on panel depth)."
        ),
        inputs=("sector_raw", "volatility_20d_ratio", "vol_ratio_10_60_ratio", "atr_14_ratio"),
        cross_sectional=True,
    ),
    FeatureSpec(
        name="momentum_12_1_pillar_pct",
        market_wide=False,
        unit="pct",
        expression=(
            "wmean(sector_pct(mom_12_1_log_return) .40, sector_pct(return_120d_log_return) "
            ".25, sector_pct(dist_from_52w_high_ratio) .20, sector_pct(return_60d_log_return) "
            ".15)"
        ),
        description=(
            "v1's 12-1 momentum challenger pillar (`_CHALLENGER_MOMENTUM_DEF`), within-sector."
        ),
        inputs=(
            "sector_raw",
            "mom_12_1_log_return",
            "return_120d_log_return",
            "dist_from_52w_high_ratio",
            "return_60d_log_return",
        ),
        cross_sectional=True,
    ),
)


@dataclass(frozen=True)
class PendingColumn:
    """A column the feature layer does not produce YET, and what will.

    `alpha-engine-config-I11030`. Some R rankers read a column no
    :data:`CATALOG` entry produces, on purpose: the recipe is registered
    now and waits for a producer that another track or phase builds. Such an
    arm is refused by name at every `experiment.run` and writes nothing. That
    is the design, not a defect, but only this declaration lets a surface
    tell it apart from a recipe that reads a column nothing will ever
    produce.

    Keyed by COLUMN, never by arm: an arm cannot declare itself waiting. It
    waits only while every column it lacks is listed here and absent from
    :data:`CATALOG`. When a catalogue entry for the column lands, the wait
    lapses on its own, and an arm still silent after that reads mute again.
    """

    name: str
    #: What will materialise the column into the feature layer.
    producer: str
    #: The plan milestone that builds that producer (a track or a phase).
    milestone: str

    def describe(self) -> str:
        return f"{self.name} ({self.producer}, {self.milestone})"


#: Every column a registered recipe may wait on, and who produces it. Not in
#: :func:`feature_version`: a declared future producer changes no column the
#: layer writes today.
PENDING_COLUMNS: dict[str, PendingColumn] = {
    column.name: column
    for column in (
        PendingColumn(
            name="predicted_alpha_ratio",
            producer="the M slot materializes it into the feature layer",
            milestone="track B",
        ),
        PendingColumn(
            name="thinktank_rating_ratio",
            producer="an LLM arm produces the per-ticker rating",
            milestone="phase 5 (§9.1)",
        ),
    )
}


def feature_names(catalog: tuple[FeatureSpec, ...] = CATALOG) -> tuple[str, ...]:
    return tuple(spec.name for spec in catalog)


@lru_cache(maxsize=1)
def load_registry_schema() -> dict[str, Any]:
    """The v1 feature-registry schema, as a dict.

    Cached: the validator is constructed per process and the file never
    changes under a running one.
    """
    return json.loads(FEATURE_REGISTRY_SCHEMA_PATH.read_text(encoding="utf-8"))


def validate_registry_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Refuse a registry document that does not conform. Returns it on success.

    Raises rather than warning: this is the producer side of the contract,
    and a producer that emits a document its own consumer will refuse has
    failed, not degraded (`AGENTS.md`, fail loud).

    `alpha-engine-config-I10045` row 7: validated through
    `crucible.models.FeatureRegistryDocument` instead of a hand-rolled
    `jsonschema.Draft202012Validator` — `feature_registry.v1.json` is now
    GENERATED from that model. Error paths are still `/`-joined (not
    `.`-joined, unlike this migration's other rows) to match
    `tests/test_feature_registry_contract.py::
    test_validate_registry_payload_names_the_offending_path`'s existing
    `features/0/inputs`-style assertion, unchanged by this PR.
    """
    try:
        FeatureRegistryDocument.model_validate(payload)
    except ValidationError as exc:
        detail = "; ".join(
            f"{'/'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}" for e in exc.errors()
        )
        raise FeatureRegistryValidationError(
            f"the feature registry does not conform to {FEATURE_REGISTRY_SCHEMA_VERSION}: "
            f"{detail}. The registry is the producer/consumer contract of the feature "
            "layer — a document the consumer would refuse must never be written beside a "
            "day's parquet, because the parquet would then be read against a registry "
            "nothing validated."
        ) from exc
    return payload


def registry_payload(catalog: tuple[FeatureSpec, ...] = CATALOG) -> dict[str, Any]:
    """The registry as it is written to `features/{version}/registry.json`.

    Validated against `schemas/feature_registry.v1.json` on the way out, so
    the producer structurally cannot emit a payload the consumer's contract
    refuses — no caller has to remember to validate, and `crucible/data/`'s
    write path inherits the check without naming it.
    """
    return validate_registry_payload(
        {
            "schema_version": FEATURE_REGISTRY_SCHEMA_VERSION,
            "feature_version": feature_version(catalog),
            "features": [spec.to_dict() for spec in catalog],
        }
    )


def _hashed_dict(spec: FeatureSpec) -> dict[str, Any]:
    """The fields `feature_version()` hashes: `to_dict()` minus `market_wide`.

    `market_wide` joined `to_dict()` after `feature_version()` had already
    been hashing catalogue dicts for every existing `features/{version}/`
    prefix (`alpha-engine-config-I10114`). It is excluded here on purpose:
    it is a declared fact about an EXISTING constant's distributional shape,
    not a change to what a column computes, and folding it into the hash
    input would change `feature_version()` for the whole `CATALOG` the
    instant the schema gained the field — writing every column to a new
    prefix and orphaning every `run.json`/verdict that pointed at the old
    one, even though no expression, input or computed value changed. A
    future edit that changes what `market_wide` means for an EXISTING
    column already changes the hash through `expression`/`inputs` below,
    since a value's distributional shape does not change without its
    computation changing too.
    """
    d = spec.to_dict()
    del d["market_wide"]
    return d


def feature_version(catalog: tuple[FeatureSpec, ...] = CATALOG) -> str:
    """A stable 12-hex digest of the whole catalogue.

    Derived, not declared. A registry edit that changed what a column MEANS
    while leaving a hand-written version string alone would overwrite the
    layer an earlier verdict was computed from, and nothing would show it.
    """
    canonical = json.dumps(
        [_hashed_dict(spec) for spec in catalog], sort_keys=True, separators=(",", ":")
    )
    return "v" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


#: The markers `docs/FEATURE_CATALOG.md` carries around its generated table.
#: The prose outside them is written by hand; the rows between them are
#: rendered from `CATALOG` and asserted equal by
#: `tests/test_feature_registry_contract.py`, so a column added without its
#: documentation row is a test failure rather than a doc that quietly stops
#: describing the layer.
CATALOG_TABLE_BEGIN = "<!-- BEGIN GENERATED CATALOG TABLE -->"
CATALOG_TABLE_END = "<!-- END GENERATED CATALOG TABLE -->"


def render_catalog_markdown(catalog: tuple[FeatureSpec, ...] = CATALOG) -> str:
    """The catalogue as the documentation table, one row per column.

    Rendered rather than hand-maintained: the fleet rule asks for a
    documentation row per feature column, and a hand-written table is a
    contract restated in a second place, which has already drifted once in
    this fleet.
    """
    header = (
        "| Column | Unit | Window (sessions) | Cross-sectional | Expression | Inputs |\n"
        "|---|---|---|---|---|---|"
    )
    rows = []
    for spec in catalog:
        window = "—" if spec.window_trading_days is None else str(spec.window_trading_days)
        rows.append(
            f"| `{spec.name}` | {spec.unit} | {window} | "
            f"{'yes' if spec.cross_sectional else 'no'} | "
            f"`{spec.expression}` | {', '.join(f'`{i}`' for i in spec.inputs)} |"
        )
    return "\n".join([header, *rows])
