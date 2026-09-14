"""The ranker registry: what an arm's recipe RESOLVES to.

Normative source: plan §4.4 (an arm is an immutable recipe whose id is the
hash of its spec) and §10 component 4 (U and R read the materialized feature
layer, and only that).

An arm recipe in `alpha-engine-config/strategy/arms/{slot}/` names a
`ranker` and supplies its parameters. This module holds the ranking
functions themselves — pure, deterministic, and reading nothing but the
feature frame and their own declared parameters. That split is what makes
the crucible repo publishable while the tuned values stay private.

**Two rules the registry enforces mechanically:**

1. **Every column a ranker reads is declared.** `RankerSpec.reads` is
   checked against the feature frame BEFORE the ranker runs, so a missing
   column is a named refusal rather than an all-null ranking that scores
   like a coin flip.
2. **No two live arms may share a ranking callable.** Policy §4's vacuity
   rule: an arm sharing the champion's callable is `inapplicable` and
   refused at load. :func:`ranker_identity` is what the loader compares, so
   two arms differing only in a comment cannot both stand.

**The serving path never imports a ranker directly.** It resolves the
champion pointer and looks the name up here; `tests/test_slots_arms.py`
asserts no module outside this one imports a ranking function by name.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "ATTRACTIVENESS_PILLAR_COLUMNS",
    "MOMENTUM_12_1_PILLAR_COLUMN",
    "RANKERS",
    "DegenerateFeatureError",
    "MissingFeatureError",
    "RankerSpec",
    "get_ranker",
    "rank_with",
    "ranker_identity",
]


class MissingFeatureError(RuntimeError):
    """A ranker's declared input column is absent from the feature layer.

    A hard raise. A ranker whose input is missing produces an all-null
    ranking, which sorts stably and looks exactly like a considered ordering.
    """


class DegenerateFeatureError(MissingFeatureError):
    """A declared input column is present and carries no cross-sectional information.

    A subclass of :class:`MissingFeatureError` so the produce loop routes it the
    same way — `TrainingIntegrityError`, slot-wide (plan §4.4). Non-null is not
    the same as informative: v1's attractiveness board carried three pillars
    that rode near-constant for at least 18 days with coverage reading 897-903
    of 903 (`alpha-engine-config-I8255`). Z-scoring a constant divides by zero,
    and quietly dropping the pillar would re-weight the composite into a
    different arm under the same name.
    """


RankFn = Callable[["pd.DataFrame", Mapping[str, Any]], "pd.Series"]


@dataclass(frozen=True)
class RankerSpec:
    """One ranking rule: its function, what it reads, and what it accepts."""

    name: str
    fn: RankFn
    reads: tuple[str, ...]
    params: tuple[str, ...]
    description: str

    def score(self, features: pd.DataFrame, params: Mapping[str, Any]) -> pd.Series:
        missing = [c for c in self.reads if c not in features.columns]
        if missing:
            raise MissingFeatureError(
                f"ranker {self.name!r} reads {list(self.reads)} and the feature layer is "
                f"missing {missing}. Ranking on an absent column yields an all-null "
                "series, which sorts stably and is indistinguishable from a considered "
                "ordering."
            )
        unknown = sorted(set(params) - set(self.params))
        if unknown:
            raise ValueError(
                f"ranker {self.name!r} accepts {list(self.params)}; the arm supplied "
                f"unknown parameter(s) {unknown}. An ignored parameter is a recipe that "
                "does not describe what ran, and the spec hash would claim it did."
            )
        return self.fn(features, params)


def _liquid(features: pd.DataFrame) -> pd.DataFrame:
    """The liquid subset. The gate is a feature, computed once, read by all."""
    return features[features["liquidity_pass_raw"] == 1.0]


def _momentum_sleeve(features: pd.DataFrame, params: Mapping[str, Any]) -> pd.Series:
    """Mean of two momentum z-scores over the liquid set."""
    block = _liquid(features)
    score = (block["momentum_20d_zscore"] + block["return_60d_zscore"]) / 2.0
    return score.dropna()


def _tech_score_gate(features: pd.DataFrame, params: Mapping[str, Any]) -> pd.Series:
    """The displaced incumbent: an equally weighted four-input technical rank."""
    block = _liquid(features)
    return block["tech_score_ratio"].dropna()


def _mom_12_1_sleeve(features: pd.DataFrame, params: Mapping[str, Any]) -> pd.Series:
    """Twelve-month-minus-one momentum alone."""
    block = _liquid(features)
    return block["mom_12_1_zscore"].dropna()


def _quant_composite(features: pd.DataFrame, params: Mapping[str, Any]) -> pd.Series:
    """A weighted composite of momentum, trend and mean-reversion.

    The v1 `no_agent_quant` producer's composite, re-expressed on the
    feature layer. The weights are the arm's parameters and live in the
    private strategy tree; the shape is here, in the open, because the shape
    is not the edge.
    """
    block = _liquid(features)
    w_momentum = float(params.get("momentum_weight", 0.5))
    w_trend = float(params.get("trend_weight", 0.3))
    w_reversal = float(params.get("reversal_weight", 0.2))
    total = w_momentum + w_trend + w_reversal
    if total <= 0:
        raise ValueError(
            "quant_composite weights sum to zero; every name would score identically "
            "and the arm would rank alphabetically while reporting a considered order"
        )
    momentum = block["momentum_20d_zscore"]
    trend = block["close_to_sma50_ratio"] - 1.0
    # RSI above the midpoint is stretched; the reversal leg is short it.
    reversal = 0.5 - block["rsi_14_ratio"]
    score = (w_momentum * momentum + w_trend * trend + w_reversal * reversal) / total
    return score.dropna()


def _predicted_alpha_direct(features: pd.DataFrame, params: Mapping[str, Any]) -> pd.Series:
    """Rank by the model slot's `predicted_alpha`.

    Declared here so the arm is REGISTERED — policy §3's registration gate —
    even though its input arrives with the M slot (track B). The feature
    layer does not carry `predicted_alpha_ratio`, so
    :meth:`RankerSpec.score` refuses by name rather than this function
    inventing a substitute ordering. An arm whose input is absent fails the
    slot's cycle (plan §4.4); it does not quietly record a miss, because a
    missing INPUT and "this arm had nothing to say" are different facts.
    """
    block = _liquid(features)
    return block["predicted_alpha_ratio"].dropna()


def _predicted_alpha_within_cut(features: pd.DataFrame, params: Mapping[str, Any]) -> pd.Series:
    """Rank by `predicted_alpha` inside the top slice of the momentum cut.

    v1's `scanner_top20_predictor`: the same predictor score as
    `scanner_predictor_direct`, over a NARROWER pool. It is a separate
    callable rather than the same function with a parameter because policy
    §4 refuses two arms that share a ranking callable — and here they genuinely
    are two rules, since the pool is part of the recipe, not a setting on it.

    Like `predicted_alpha_direct`, it reads a column the phase-1 feature layer
    does not carry, so it refuses BY NAME until the M slot materializes it.
    """
    block = _liquid(features)
    cut = int(params.get("cut_top_n", 20))
    pool = block.nlargest(cut, "momentum_20d_zscore")
    return pool["predicted_alpha_ratio"].dropna()


def _thinktank_rating_direct(features: pd.DataFrame, params: Mapping[str, Any]) -> pd.Series:
    """Rank by the Think Tank's own per-ticker rating.

    v1's `thinktank_coverage`. Registered so the arm exists in the register
    with its lineage intact (policy §3: an arm producing output without a
    register row is the defect this arm's own name commemorates). Its input
    is an LLM-derived column, which phase 1 does not produce — LLM arms are
    phase 5 (§9.1) — so it refuses by name rather than inventing an ordering.
    """
    block = _liquid(features)
    return block["thinktank_rating_ratio"].dropna()


# ---------------------------------------------------------------------------
# The attractiveness family — v1's `scanner_cut` slot (`alpha-engine-config-I10715`).
#
# v1 ran two universe decisions over the SAME scanned population at the SAME
# width: `scanner_spec` (momentum_sleeve / mom_12_1_sleeve / tech_score_gate)
# and `scanner_cut` (attractiveness_top_60 and its variants), each graded
# against the population it drew from, count-matched at 60. They are one
# decision — "which 60 names advance" — ranked by different rules, so v2
# carries the cut's arms as U arms (plan §4.4 row U; policy §2: one slot, one
# decision) rather than as a second stage whose output nothing but the first
# stage's population could explain.
#
# v1 method (`crucible-research/scoring/universe_board.py`, schema_version 3):
#   z_{i,p}  = clip((pillar_{i,p} - mean_p) / sd_p, -3, +3)
#   blend_i  = sum_{p in avail} w_p * z_{i,p} / sum_{p in avail} w_p
# then a terminal percentile rank, which is monotone and so changes no top-N.
# The six pillars are within-sector percentile composites of raw factors,
# computed upstream of any ranking — so here they are FEATURE-LAYER columns
# and these rankers only blend them. Until the catalogue declares them, every
# arm below is refused BY NAME at registration
# (`crucible.slots.cycle.partition_by_catalog`), never ranked on nulls.
#
# **Each variant is its own callable, deliberately.** The vacuity guard
# (`crucible.slots.arms._assert_applicable`) compares callables, so a single
# parameterised blend would make `momzero` and the champion "not two arms".
# And the variant's defining property is STRUCTURAL: `momzero` does not read
# the momentum pillar at all, rather than reading it at weight zero, so a
# broken momentum column cannot fail an arm whose hypothesis is that momentum
# does not belong. The per-pillar weights stay parameters, supplied by the
# private recipe; each must be strictly positive, because a zero weight is a
# different ranker wearing this one's name.
# ---------------------------------------------------------------------------

#: Pillar -> feature column. v1 `PILLAR_ORDER_FOR_WEIGHTS` order; v1 maps
#: `defensiveness` to its `low_vol_score` composite
#: (`crucible-research/scoring/composite.py::_PILLAR_TO_FACTOR_KEY`).
ATTRACTIVENESS_PILLAR_COLUMNS: dict[str, str] = {
    "quality": "quality_pillar_pct",
    "value": "value_pillar_pct",
    "momentum": "momentum_pillar_pct",
    "growth": "growth_pillar_pct",
    "stewardship": "stewardship_pillar_pct",
    "defensiveness": "defensiveness_pillar_pct",
}

#: The momentum pillar re-composed on the 12-1 skip-month horizon — v1's
#: `mom121` challenger, which differs from the champion in this pillar ONLY.
MOMENTUM_12_1_PILLAR_COLUMN = "momentum_12_1_pillar_pct"

#: v1's winsorisation bound on the per-pillar cross-sectional z.
_PILLAR_Z_CLIP = 3.0


def _columns_for(pillars: tuple[str, ...], **overrides: str) -> dict[str, str]:
    return {p: overrides.get(p, ATTRACTIVENESS_PILLAR_COLUMNS[p]) for p in pillars}


def _pillar_weight_params(pillars: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(f"{p}_weight" for p in pillars)


def _resolved_pillar_weights(
    ranker: str, params: Mapping[str, Any], pillars: Mapping[str, str]
) -> dict[str, float]:
    missing = [key for key in _pillar_weight_params(pillars) if key not in params]
    if missing:
        raise ValueError(
            f"ranker {ranker!r} requires an explicit weight per pillar it reads; the arm "
            f"omits {missing}. A defaulted weight is a tuned value the recipe does not "
            "declare, and the spec hash would not see it."
        )
    weights: dict[str, float] = {}
    for pillar in pillars:
        raw = params[f"{pillar}_weight"]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw):
            raise ValueError(
                f"ranker {ranker!r}: {pillar}_weight must be a finite number; got {raw!r}"
            )
        if raw <= 0:
            raise ValueError(
                f"ranker {ranker!r}: {pillar}_weight must be strictly positive; got {raw!r}. "
                "A pillar at weight zero is a different arm (see the momzero and hard3 "
                "rankers, which do not read the pillar at all)."
            )
        weights[pillar] = float(raw)
    return weights


def _attractiveness_pillar_blend(
    features: pd.DataFrame,
    params: Mapping[str, Any],
    *,
    ranker: str,
    pillars: Mapping[str, str],
) -> pd.Series:
    """Coverage-renormalised weighted mean of winsorised pillar z-scores.

    Z-scores are taken over the LIQUID set, i.e. the population being ranked,
    before any selection: v1 measured that scoring a wider population and
    filtering afterwards moved 860 of 902 scores and the ordering from rank 26
    (`alpha-engine-config-I7844`). A name with no pillar available is dropped —
    unrankable, never coerced to the bottom.
    """
    weights = _resolved_pillar_weights(ranker, params, pillars)
    block = _liquid(features)
    numerator = None
    denominator = None
    for pillar, column in pillars.items():
        values = block[column].astype(float)
        present = values.dropna()
        spread = float(present.std(ddof=0)) if len(present) else 0.0
        if not len(present) or not spread > 0.0:
            raise DegenerateFeatureError(
                f"ranker {ranker!r}: pillar {pillar!r} ({column}) has "
                f"{len(present)} non-null value(s) and dispersion {spread!r} over the "
                f"{len(block)}-name liquid set. A pillar with no cross-sectional spread "
                "carries no ordering, and dropping it would silently re-weight the "
                "composite into a different arm."
            )
        z = ((values - float(present.mean())) / spread).clip(-_PILLAR_Z_CLIP, _PILLAR_Z_CLIP)
        weighted = z.fillna(0.0) * weights[pillar]
        available = z.notna().astype(float) * weights[pillar]
        numerator = weighted if numerator is None else numerator + weighted
        denominator = available if denominator is None else denominator + available
    assert numerator is not None and denominator is not None  # pillars is never empty
    rankable = denominator > 0.0
    return (numerator[rankable] / denominator[rankable]).dropna()


_ATTRACTIVENESS_SIX = _columns_for(
    ("quality", "value", "momentum", "growth", "stewardship", "defensiveness")
)
_ATTRACTIVENESS_MOMZERO = _columns_for(
    ("quality", "value", "growth", "stewardship", "defensiveness")
)
_ATTRACTIVENESS_MOM121 = _columns_for(
    ("quality", "value", "momentum", "growth", "stewardship", "defensiveness"),
    momentum=MOMENTUM_12_1_PILLAR_COLUMN,
)
_ATTRACTIVENESS_HARD3 = _columns_for(("value", "momentum", "defensiveness"))


def _attractiveness_blend(features: pd.DataFrame, params: Mapping[str, Any]) -> pd.Series:
    """v1 `attractiveness_top_60`: all six pillars. The `scanner_cut` champion."""
    return _attractiveness_pillar_blend(
        features, params, ranker="attractiveness_blend", pillars=_ATTRACTIVENESS_SIX
    )


def _attractiveness_momzero_blend(features: pd.DataFrame, params: Mapping[str, Any]) -> pd.Series:
    """v1 `attractiveness_momzero_top_60`: the five non-momentum pillars. Isolates EXPOSURE."""
    return _attractiveness_pillar_blend(
        features, params, ranker="attractiveness_momzero_blend", pillars=_ATTRACTIVENESS_MOMZERO
    )


def _attractiveness_mom121_blend(features: pd.DataFrame, params: Mapping[str, Any]) -> pd.Series:
    """v1 `attractiveness_mom121_top_60`: momentum pillar on the 12-1 horizon. Isolates HORIZON."""
    return _attractiveness_pillar_blend(
        features, params, ranker="attractiveness_mom121_blend", pillars=_ATTRACTIVENESS_MOM121
    )


def _attractiveness_hard3_blend(features: pd.DataFrame, params: Mapping[str, Any]) -> pd.Series:
    """v1 `attractiveness_hard3_top_60`: value, momentum, defensiveness only.

    The vendor-fundamental half (quality, growth, stewardship) is not read.
    """
    return _attractiveness_pillar_blend(
        features, params, ranker="attractiveness_hard3_blend", pillars=_ATTRACTIVENESS_HARD3
    )


def _attractiveness_spec(
    name: str, fn: RankFn, pillars: Mapping[str, str], description: str
) -> RankerSpec:
    return RankerSpec(
        name=name,
        fn=fn,
        reads=("liquidity_pass_raw", *pillars.values()),
        params=("top_n", *_pillar_weight_params(pillars)),
        description=description,
    )


RANKERS: dict[str, RankerSpec] = {
    spec.name: spec
    for spec in (
        RankerSpec(
            name="momentum_sleeve",
            fn=_momentum_sleeve,
            reads=("liquidity_pass_raw", "momentum_20d_zscore", "return_60d_zscore"),
            params=("top_n",),
            description="Mean of 20-session and 60-session momentum z-scores.",
        ),
        RankerSpec(
            name="tech_score_gate",
            fn=_tech_score_gate,
            reads=("liquidity_pass_raw", "tech_score_ratio"),
            params=("top_n",),
            description="Equally weighted RSI / MA50 / MA200 / 20-session momentum rank.",
        ),
        RankerSpec(
            name="mom_12_1_sleeve",
            fn=_mom_12_1_sleeve,
            reads=("liquidity_pass_raw", "mom_12_1_zscore"),
            params=("top_n",),
            description="Twelve-month momentum skipping the most recent month.",
        ),
        RankerSpec(
            name="quant_composite",
            fn=_quant_composite,
            reads=(
                "liquidity_pass_raw",
                "momentum_20d_zscore",
                "close_to_sma50_ratio",
                "rsi_14_ratio",
            ),
            params=("top_n", "momentum_weight", "trend_weight", "reversal_weight"),
            description="Weighted momentum / trend / mean-reversion composite.",
        ),
        RankerSpec(
            name="predicted_alpha_within_cut",
            fn=_predicted_alpha_within_cut,
            reads=("liquidity_pass_raw", "momentum_20d_zscore", "predicted_alpha_ratio"),
            params=("top_n", "cut_top_n"),
            description=(
                "Predictor score inside the top slice of the momentum cut. Servable "
                "when the M slot materializes predicted_alpha_ratio (track B)."
            ),
        ),
        RankerSpec(
            name="thinktank_rating_direct",
            fn=_thinktank_rating_direct,
            reads=("liquidity_pass_raw", "thinktank_rating_ratio"),
            params=("top_n",),
            description=(
                "Think Tank per-ticker rating. Servable when an LLM arm exists to "
                "produce it — phase 5 (§9.1), not phase 1."
            ),
        ),
        RankerSpec(
            name="predicted_alpha_direct",
            fn=_predicted_alpha_direct,
            reads=("liquidity_pass_raw", "predicted_alpha_ratio"),
            params=("top_n",),
            description=(
                "Rank by the model slot's predicted_alpha. Registered now, servable "
                "when the M slot materializes predicted_alpha_ratio into the feature "
                "layer (track B)."
            ),
        ),
        _attractiveness_spec(
            "attractiveness_blend",
            _attractiveness_blend,
            _ATTRACTIVENESS_SIX,
            "Six-pillar attractiveness composite (v1 scanner_cut champion). Servable when "
            "the feature layer declares the pillar columns.",
        ),
        _attractiveness_spec(
            "attractiveness_momzero_blend",
            _attractiveness_momzero_blend,
            _ATTRACTIVENESS_MOMZERO,
            "Attractiveness without the momentum pillar (v1 momzero challenger).",
        ),
        _attractiveness_spec(
            "attractiveness_mom121_blend",
            _attractiveness_mom121_blend,
            _ATTRACTIVENESS_MOM121,
            "Attractiveness with a 12-1 skip-month momentum pillar (v1 mom121 challenger).",
        ),
        _attractiveness_spec(
            "attractiveness_hard3_blend",
            _attractiveness_hard3_blend,
            _ATTRACTIVENESS_HARD3,
            "Value, momentum and defensiveness pillars only (v1 hard3 challenger).",
        ),
    )
}


def get_ranker(name: str) -> RankerSpec:
    """The spec for ``name``. Raises on an unknown ranker — never returns None."""
    try:
        return RANKERS[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown ranker {name!r}; the registered rankers are {sorted(RANKERS)}. An "
            "arm naming a ranker that does not exist has no recipe, and its spec hash "
            "would be the hash of a recipe nothing can run."
        ) from exc


def ranker_identity(name: str) -> int:
    """The identity of a ranker's callable, for the §4 vacuity check.

    Two arms sharing a callable are not two arms: policy §4 calls that
    `inapplicable` and refuses it at import, because a comparison of a rule
    with itself always reads as a tie and consumes a slot in the pool.
    """
    return id(get_ranker(name).fn)


def rank_with(name: str, features: pd.DataFrame, params: Mapping[str, Any]) -> pd.Series:
    """Score the cross-section with the named ranker. Index is the ticker."""
    scores = get_ranker(name).score(features, params)
    ticker_column = features.loc[scores.index, "ticker"]
    scores = scores.copy()
    scores.index = ticker_column.to_numpy()
    return scores.sort_values(ascending=False)
