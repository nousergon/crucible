"""Materializing the feature layer from the price panel.

Normative source: plan §10 component 4.

One function, `build_features`, turns the long price panel into the
cross-section for its last trading day. That cross-section is the artifact U
and R read — **the only** artifact they read for features, so the two cannot
recompute a feature differently from each other.

**The layer never fills.** A ticker without enough history for a 252-session
window gets a null in that column, and the arm that consumes it drops the
ticker with a recorded reason. Forward-filling would put a value that is not
a measurement into a column an arm ranks on.

**The layer never looks ahead.** Every window ends at the row's own trading
day. The one construct that could reach forward — a cross-sectional rank —
ranks within a single day. `tests/test_features.py` asserts that recomputing
a day's features from a panel truncated at that day reproduces the same
bytes, which is the property a look-ahead breaks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from crucible.features.registry import CATALOG, FeatureSpec, feature_version

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

__all__ = ["LIQUIDITY_FLOOR_USD", "build_features"]

#: The liquidity gate, in USD of mean 20-session traded notional. A single
#: declared constant read by the one feature that expresses the gate, so no
#: arm re-derives a threshold of its own.
LIQUIDITY_FLOOR_USD = 5_000_000.0


def _rank01(series: pd.Series) -> pd.Series:
    """Cross-sectional rank on 0-1. Ties share the average rank."""
    valid = series.notna().sum()
    if valid < 2:
        # A single observation has no cross-section to rank within. It is
        # left null rather than assigned 0.5, which would be an invented
        # measurement occupying a real column.
        return series * float("nan")
    return series.rank(pct=True, na_option="keep")


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if not std or std != std:
        # Zero dispersion: every name is identical on this axis, so no name
        # is above or below average. Null, not zero — a zero z-score is a
        # measured "exactly at the mean" and would rank ties as a real tie.
        return series * float("nan")
    return (series - series.mean()) / std


def _wilder_rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, float("nan"))
    return 100.0 - (100.0 / (1.0 + rs))


def build_features(
    panel: pd.DataFrame,
    *,
    version: str | None = None,
    catalog: tuple[FeatureSpec, ...] = CATALOG,
    as_of: object = None,
) -> tuple[pd.DataFrame, tuple[FeatureSpec, ...]]:
    """The feature cross-section for ``as_of`` (default: the panel's last day).

    Returns the frame and the catalogue it was built from, so the caller
    writes the registry that matches the columns rather than one that
    happens to be importable.
    """
    import numpy as np

    if panel.empty:
        raise ValueError(
            "cannot build features from an empty panel; the data layer raises before "
            "it can hand one over, so reaching here means a caller bypassed it"
        )

    resolved_version = version or feature_version(catalog)
    day = as_of if as_of is not None else max(panel["trading_day"])

    frame = panel.sort_values(["ticker", "trading_day"]).copy()
    grouped = frame.groupby("ticker", sort=False)["close_raw"]

    log_close = np.log(frame["close_raw"].where(frame["close_raw"] > 0))
    frame["_log_close"] = log_close
    log_grouped = frame.groupby("ticker", sort=False)["_log_close"]

    frame["return_1d_log_return"] = log_grouped.diff(1)
    frame["momentum_20d_log_return"] = log_grouped.diff(20)
    frame["return_60d_log_return"] = log_grouped.diff(60)
    frame["mom_12_1_log_return"] = log_grouped.shift(21) - log_grouped.shift(252)

    frame["volatility_20d_ratio"] = frame.groupby("ticker", sort=False)[
        "return_1d_log_return"
    ].transform(lambda s: s.rolling(20, min_periods=20).std(ddof=0))

    frame["close_to_sma50_ratio"] = frame["close_raw"] / grouped.transform(
        lambda s: s.rolling(50, min_periods=50).mean()
    )
    frame["close_to_sma200_ratio"] = frame["close_raw"] / grouped.transform(
        lambda s: s.rolling(200, min_periods=200).mean()
    )
    frame["rsi_14_ratio"] = (
        frame.groupby("ticker", sort=False)["close_raw"].transform(lambda s: _wilder_rsi(s, 14))
        / 100.0
    )

    notional = frame["close_raw"] * frame["volume_raw"]
    frame["_notional"] = notional
    frame["dollar_volume_20d_raw"] = frame.groupby("ticker", sort=False)["_notional"].transform(
        lambda s: s.rolling(20, min_periods=20).mean()
    )

    cross = frame[frame["trading_day"] == day].copy()
    if cross.empty:
        raise ValueError(
            f"the panel carries no rows for {day}; features are a cross-section of one "
            "trading day and there is nothing to cut"
        )

    cross["liquidity_pass_raw"] = (cross["dollar_volume_20d_raw"] >= LIQUIDITY_FLOOR_USD).astype(
        "float64"
    )
    # A null liquidity input is not a failed gate — it is an unmeasured one.
    cross.loc[cross["dollar_volume_20d_raw"].isna(), "liquidity_pass_raw"] = float("nan")

    liquid = cross["liquidity_pass_raw"] == 1.0
    for source_col, target in (
        ("momentum_20d_log_return", "momentum_20d_zscore"),
        ("return_60d_log_return", "return_60d_zscore"),
        ("mom_12_1_log_return", "mom_12_1_zscore"),
    ):
        cross[target] = float("nan")
        # Z-scored over the LIQUID set only: an illiquid tail with wild
        # returns would set the scale for every liquid name, and the arms
        # that rank on these z-scores only ever consider liquid names.
        cross.loc[liquid, target] = _zscore(cross.loc[liquid, source_col])

    cross["tech_score_ratio"] = (
        _rank01(cross["rsi_14_ratio"])
        + _rank01(cross["close_to_sma50_ratio"])
        + _rank01(cross["close_to_sma200_ratio"])
        + _rank01(cross["momentum_20d_log_return"])
    ) / 4.0

    columns = ["trading_day", "ticker"] + [spec.name for spec in catalog]
    out = cross[columns].sort_values("ticker").reset_index(drop=True)
    out.attrs["feature_version"] = resolved_version
    return out, catalog


def read_features(payload: bytes) -> pd.DataFrame:
    """Parse a materialized feature artifact back into a frame."""
    import io

    import pandas as pd

    return pd.read_parquet(io.BytesIO(payload))
