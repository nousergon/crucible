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

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "BETA_WINDOW_TRADING_DAYS",
    "LIQUIDITY_FLOOR_VAR",
    "liquidity_floor_usd",
    "MOMENTUM_CHANGE_WINDOW_TRADING_DAYS",
    "RESIDUAL_MOMENTUM_CUM_TRADING_DAYS",
    "RESIDUAL_MOMENTUM_SKIP_TRADING_DAYS",
    "RESIDUAL_MOMENTUM_WINDOW_TRADING_DAYS",
    "RESIDUAL_VOL_WINDOW_TRADING_DAYS",
    "build_features",
]

#: The environment variable carrying the liquidity gate, in USD of mean
#: 20-session traded notional.
LIQUIDITY_FLOOR_VAR = "CRUCIBLE_LIQUIDITY_FLOOR_USD"


def liquidity_floor_usd() -> float:
    """The liquidity gate, read from the environment.

    Was a literal in this file until 2026-09-07. It moved because it is the
    one genuinely TUNED number in this module: a belief about what is tradeable,
    which `repository-tiering-policy.md` test 3 sends to the private tree, and
    which `alpha-engine-config/strategy/README.md` calls strategy edge in as
    many words. Its neighbours below did NOT move, and the distinction is the
    point — a 252-session window with a 21-session skip is the textbook 12-1
    residual-momentum construction, published in the literature long before
    this system existed, while a five-million-dollar floor is a position.

    Still a SINGLE declared value read by the one feature that expresses the
    gate, so no arm re-derives a threshold of its own; only its home changed.

    **Resolved through :func:`crucible.required.require_env`, not by reading
    the environment here.** That is what enrols this variable in the two
    guards that derive the required set from `require_env` call sites --
    `crucible`'s `tests/test_workflow_required_env.py` and
    `nous-ergon-ops`' `tests/crossrepo/test_crucible_box_shell_required_env.py`.
    It previously raised through a hand-rolled `RuntimeError`, so neither
    guard could see it, and the dispatched box never declared it: measured
    2026-09-09 on `i-04afbe045e0ca7354`, every full-universe `data.weekly` on
    the box died on `CRUCIBLE_LIQUIDITY_FLOOR_USD is unset` after this value
    was extracted on 2026-09-07. A second raising path outside the shared
    resolver is a fourth instance of `alpha-engine-config-I10156`'s class, and
    the resolver is the only thing that makes the guards exhaustive rather
    than merely populated.

    RAISES rather than defaulting. A floor of zero passes every name and a
    floor guessed high passes none, and both produce a `liquidity_pass_raw`
    column that looks computed — the exact shape of the `avg_volume_20d`
    defect this layer's units contract exists to prevent, where 901 of 903
    tickers failed the gate silently for months.
    """
    from crucible.required import require_env  # noqa: PLC0415 - one call site

    raw = require_env(
        LIQUIDITY_FLOOR_VAR,
        refusing_to="compute liquidity_pass_raw against a guessed threshold",
    )
    try:
        floor = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{LIQUIDITY_FLOOR_VAR}={raw!r} is not a number") from exc
    if floor <= 0:
        raise RuntimeError(
            f"{LIQUIDITY_FLOOR_VAR}={raw!r} is not positive; a floor of zero or less "
            "passes every name and is indistinguishable from no gate at all."
        )
    return floor


#: The residual-momentum window set, lifted verbatim from the v1 recipe this
#: layer reproduces (`crucible-predictor/config/predictor.sample.yaml::
#: residual_momentum`, alpha-engine-config-I9765). Named constants rather
#: than literals in the expressions below because the registry's `expression`
#: strings quote these numbers, and two declarations of one window is how a
#: recipe and its implementation drift apart.
BETA_WINDOW_TRADING_DAYS = 60
RESIDUAL_MOMENTUM_WINDOW_TRADING_DAYS = 252
RESIDUAL_MOMENTUM_SKIP_TRADING_DAYS = 21
RESIDUAL_VOL_WINDOW_TRADING_DAYS = 20
MOMENTUM_CHANGE_WINDOW_TRADING_DAYS = 21

#: The cumulation window of the 12-1 residual momentum: the lookback less the
#: skipped month, exactly as v1 computed it.
RESIDUAL_MOMENTUM_CUM_TRADING_DAYS = (
    RESIDUAL_MOMENTUM_WINDOW_TRADING_DAYS - RESIDUAL_MOMENTUM_SKIP_TRADING_DAYS
)

#: Guards the information-ratio division. v1's `_EPS`, carried across so the
#: two implementations do not disagree on a near-zero denominator.
_EPS = 1e-8


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
    catalog: tuple[FeatureSpec, ...] = CATALOG,
    as_of: object = None,
) -> tuple[pd.DataFrame, tuple[FeatureSpec, ...]]:
    """The feature cross-section for ``as_of`` (default: the panel's last day).

    Returns the frame and the catalogue it was built from, so the caller
    writes the registry that matches the columns rather than one that
    happens to be importable.

    There is deliberately no ``version`` parameter. The version is a hash of
    ``catalog`` (`feature_version`, `registry.py`) and nothing else — a
    caller cannot pass one in, so the frame's own `attrs["feature_version"]`
    and the registry document `registry_payload(catalog)` writes beside it
    are computed from the same catalogue by construction, never two values
    a caller could make disagree (`alpha-engine-config-I9816`).
    """
    import numpy as np

    if panel.empty:
        raise ValueError(
            "cannot build features from an empty panel; the data layer raises before "
            "it can hand one over, so reaching here means a caller bypassed it"
        )

    resolved_version = feature_version(catalog)
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

    # -- the residual / idiosyncratic-momentum block (I9765) ---------------
    #
    # The market leg is the day's equal-weighted cross-sectional mean log
    # return. It is a same-day statistic, so it reaches no further forward
    # than the row it sits on, and truncating the panel at a day leaves every
    # earlier day's value unchanged — the property `TestNoLookAhead` asserts.
    frame["market_return_1d_log_return"] = frame.groupby("trading_day", sort=False)[
        "return_1d_log_return"
    ].transform("mean")

    # Point-in-time beta by the population moments of the two return series
    # over the window. Written as moments rather than `rolling().cov()` so the
    # numerator and denominator share one ddof and it cancels exactly.
    frame["_rrm"] = frame["return_1d_log_return"] * frame["market_return_1d_log_return"]
    frame["_rm2"] = frame["market_return_1d_log_return"] ** 2
    by_ticker = frame.groupby("ticker", sort=False)

    def _rolling_mean(column: str, window: int) -> pd.Series:
        return by_ticker[column].transform(lambda s: s.rolling(window, min_periods=window).mean())

    mean_r = _rolling_mean("return_1d_log_return", BETA_WINDOW_TRADING_DAYS)
    mean_rm = _rolling_mean("market_return_1d_log_return", BETA_WINDOW_TRADING_DAYS)
    mean_rrm = _rolling_mean("_rrm", BETA_WINDOW_TRADING_DAYS)
    mean_rm2 = _rolling_mean("_rm2", BETA_WINDOW_TRADING_DAYS)
    covariance = mean_rrm - mean_r * mean_rm
    variance = mean_rm2 - mean_rm**2
    # A market leg with no dispersion over the window leaves beta NULL, not
    # zero: an unmeasurable beta is not a beta of zero, and a zero would make
    # the residual return equal to the raw return without saying so.
    frame["beta_60d_raw"] = (
        (covariance / variance.where(variance > 0)).groupby(frame["ticker"], sort=False).shift(1)
    )

    frame["residual_return_1d_log_return"] = (
        frame["return_1d_log_return"] - frame["beta_60d_raw"] * frame["market_return_1d_log_return"]
    )

    residual_by_ticker = frame.groupby("ticker", sort=False)["residual_return_1d_log_return"]
    frame["residual_vol_20d_ratio"] = residual_by_ticker.transform(
        lambda s: s.rolling(
            RESIDUAL_VOL_WINDOW_TRADING_DAYS, min_periods=RESIDUAL_VOL_WINDOW_TRADING_DAYS
        ).std(ddof=0)
    )
    cumulative_residual = residual_by_ticker.transform(
        lambda s: (
            s.rolling(
                RESIDUAL_MOMENTUM_CUM_TRADING_DAYS,
                min_periods=RESIDUAL_MOMENTUM_CUM_TRADING_DAYS,
            )
            .sum()
            .shift(RESIDUAL_MOMENTUM_SKIP_TRADING_DAYS)
        )
    )
    frame["residual_momentum_252d_skip21d_ratio"] = cumulative_residual / (
        frame["residual_vol_20d_ratio"] * np.sqrt(RESIDUAL_MOMENTUM_CUM_TRADING_DAYS) + _EPS
    )

    recent_momentum = by_ticker["return_1d_log_return"].transform(
        lambda s: s.rolling(
            MOMENTUM_CHANGE_WINDOW_TRADING_DAYS,
            min_periods=MOMENTUM_CHANGE_WINDOW_TRADING_DAYS,
        ).sum()
    )
    frame["momentum_change_21d_log_return"] = recent_momentum - recent_momentum.groupby(
        frame["ticker"], sort=False
    ).shift(MOMENTUM_CHANGE_WINDOW_TRADING_DAYS)

    cross = frame[frame["trading_day"] == day].copy()
    if cross.empty:
        raise ValueError(
            f"the panel carries no rows for {day}; features are a cross-section of one "
            "trading day and there is nothing to cut"
        )

    cross["liquidity_pass_raw"] = (cross["dollar_volume_20d_raw"] >= liquidity_floor_usd()).astype(
        "float64"
    )
    # A null liquidity input is not a failed gate — it is an unmeasured one.
    cross.loc[cross["dollar_volume_20d_raw"].isna(), "liquidity_pass_raw"] = float("nan")

    liquid = cross["liquidity_pass_raw"] == 1.0
    for source_col, target in (
        ("momentum_20d_log_return", "momentum_20d_zscore"),
        ("return_60d_log_return", "return_60d_zscore"),
        ("mom_12_1_log_return", "mom_12_1_zscore"),
        ("residual_momentum_252d_skip21d_ratio", "residual_momentum_252d_skip21d_zscore"),
        ("momentum_change_21d_log_return", "momentum_change_21d_zscore"),
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
