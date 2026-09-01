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

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

__all__ = [
    "RANKERS",
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
