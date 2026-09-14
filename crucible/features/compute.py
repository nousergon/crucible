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
    "SMA_LONG_WINDOW_TRADING_DAYS",
    "build_features",
    "catalog_column_depths",
    "min_panel_trading_days",
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

#: The two simple-moving-average windows. Named for the same reason as the
#: residual block above: :func:`catalog_column_depths` quotes them, and a
#: window declared twice is a window that drifts.
SMA_SHORT_WINDOW_TRADING_DAYS = 50
SMA_LONG_WINDOW_TRADING_DAYS = 200

#: Wilder's RSI window.
RSI_WINDOW_TRADING_DAYS = 14

#: The v3.0-meta L1 input windows (alpha-engine-config-I10695), lifted from
#: v1's `crucible-predictor/config/predictor.sample.yaml::features`
#: (`momentum_short`, `atr_period`, `vol_short_window`, `vol_long_window`) and
#: `nousergon-data/features/feature_engineer.py` (the 252-session year).
MOMENTUM_SHORT_WINDOW_TRADING_DAYS = 5
ATR_WINDOW_TRADING_DAYS = 14
VOL_RATIO_SHORT_WINDOW_TRADING_DAYS = 10
VOL_RATIO_LONG_WINDOW_TRADING_DAYS = 60
FIFTY_TWO_WEEK_WINDOW_TRADING_DAYS = 252

#: Guards the information-ratio division. v1's `_EPS`, carried across so the
#: two implementations do not disagree on a near-zero denominator.
_EPS = 1e-8

#: Sessions of a ticker's own history before `return_1d_log_return` — the
#: first link of every chained column below — has a value. A one-session diff
#: needs two rows.
_RETURN_DEPTH_TRADING_DAYS = 2

#: Sessions before `beta_60d_raw` has a value. The rolling moments consume
#: `BETA_WINDOW_TRADING_DAYS` sessions OF RETURNS (which themselves start one
#: session in), and the result is then shifted one session so a row's beta is
#: estimated strictly before the row it prices.
_BETA_DEPTH_TRADING_DAYS = _RETURN_DEPTH_TRADING_DAYS - 1 + BETA_WINDOW_TRADING_DAYS + 1

#: Sessions before `residual_momentum_252d_skip21d_ratio` has a value — the
#: DEEPEST column in the catalogue, and the reason this arithmetic is written
#: down rather than left implicit. The residual return stream starts where
#: beta does; the cumulation consumes `RESIDUAL_MOMENTUM_CUM_TRADING_DAYS`
#: sessions OF THAT STREAM; the 12-1 skip then shifts the result
#: `RESIDUAL_MOMENTUM_SKIP_TRADING_DAYS` sessions further.
#:
#: **It is 313, not 252.** The column's declared `window_trading_days` is 252
#: — a true statement about the economic lookback, and a false one about the
#: panel the producer needs, because the window is composed over a residual
#: stream that is itself 61 sessions deep. Reading the declared window as the
#: panel requirement is exactly how `alpha-engine-config-I10688` happened:
#: `crucible.data.daily.DEFAULT_LOOKBACK_DAYS` was 400 CALENDAR days ("a
#: little over 252 sessions plus slack"), which is 275 sessions, and this
#: column was therefore null for 903 of 903 tickers on every one of the 536
#: sessions the layer had been compiled for — a column that looked computed
#: and measured nothing, the `avg_volume_20d` class this module's units
#: contract exists to prevent.
_RESIDUAL_MOMENTUM_DEPTH_TRADING_DAYS = (
    _BETA_DEPTH_TRADING_DAYS
    - 1
    + RESIDUAL_MOMENTUM_CUM_TRADING_DAYS
    + RESIDUAL_MOMENTUM_SKIP_TRADING_DAYS
)


def catalog_column_depths() -> dict[str, int]:
    """`{column: sessions of a ticker's own history before it has a value}`.

    DERIVED from the window constants above, never restated: the whole point
    is that one edit to a window moves both the computation and the panel the
    producer asks for. A column absent from this mapping reads only the
    current row and needs one session.

    Cross-sectional columns (`_zscore`, `tech_score_ratio`) inherit the depth
    of the deepest column they are computed from, because a z-score of a null
    is a null.
    """
    residual_vol = _BETA_DEPTH_TRADING_DAYS - 1 + RESIDUAL_VOL_WINDOW_TRADING_DAYS
    momentum_change = _RETURN_DEPTH_TRADING_DAYS - 1 + 2 * MOMENTUM_CHANGE_WINDOW_TRADING_DAYS
    mom_12_1 = RESIDUAL_MOMENTUM_WINDOW_TRADING_DAYS + 1
    momentum_20d = 21
    return_60d = 61
    depths = {
        "close_raw": 1,
        "return_1d_log_return": _RETURN_DEPTH_TRADING_DAYS,
        "momentum_20d_log_return": momentum_20d,
        "return_60d_log_return": return_60d,
        "mom_12_1_log_return": mom_12_1,
        "volatility_20d_ratio": _RETURN_DEPTH_TRADING_DAYS - 1 + 20,
        "close_to_sma50_ratio": SMA_SHORT_WINDOW_TRADING_DAYS,
        "close_to_sma200_ratio": SMA_LONG_WINDOW_TRADING_DAYS,
        "rsi_14_ratio": RSI_WINDOW_TRADING_DAYS + 1,
        "dollar_volume_20d_raw": 20,
        "liquidity_pass_raw": 20,
        "market_return_1d_log_return": _RETURN_DEPTH_TRADING_DAYS,
        "beta_60d_raw": _BETA_DEPTH_TRADING_DAYS,
        "residual_return_1d_log_return": _BETA_DEPTH_TRADING_DAYS,
        "residual_vol_20d_ratio": residual_vol,
        "residual_momentum_252d_skip21d_ratio": _RESIDUAL_MOMENTUM_DEPTH_TRADING_DAYS,
        "momentum_change_21d_log_return": momentum_change,
        "momentum_5d_log_return": MOMENTUM_SHORT_WINDOW_TRADING_DAYS + 1,
        # A true range needs the PRIOR close, so the first one is on a
        # ticker's second session; the mean then consumes the window of them.
        "atr_14_ratio": ATR_WINDOW_TRADING_DAYS + 1,
        "vol_ratio_10_60_ratio": _RETURN_DEPTH_TRADING_DAYS
        - 1
        + VOL_RATIO_LONG_WINDOW_TRADING_DAYS,
        "dist_from_52w_high_ratio": FIFTY_TWO_WEEK_WINDOW_TRADING_DAYS,
        "dist_from_52w_low_ratio": FIFTY_TWO_WEEK_WINDOW_TRADING_DAYS,
    }
    depths["momentum_20d_zscore"] = depths["momentum_20d_log_return"]
    depths["return_60d_zscore"] = depths["return_60d_log_return"]
    depths["mom_12_1_zscore"] = depths["mom_12_1_log_return"]
    depths["residual_momentum_252d_skip21d_zscore"] = depths["residual_momentum_252d_skip21d_ratio"]
    depths["momentum_change_21d_zscore"] = depths["momentum_change_21d_log_return"]
    depths["tech_score_ratio"] = max(
        depths["rsi_14_ratio"],
        depths["close_to_sma50_ratio"],
        depths["close_to_sma200_ratio"],
        depths["momentum_20d_log_return"],
    )
    return depths


def min_panel_trading_days(catalog: tuple[FeatureSpec, ...] = CATALOG) -> int:
    """Sessions of history every column of ``catalog`` needs to be computable.

    The producer's declared demand on the price panel, and the number
    `crucible.data.daily` sizes its trailing window from. A panel shallower
    than this does not produce a thinner cross-section — it produces a
    cross-section whose deepest columns are null for EVERY ticker, which no
    consumer can distinguish from a layer that was never built.

    A column the catalogue declares and this function does not know the depth
    of raises rather than defaulting to one session: a new column whose depth
    nobody wrote down is how the panel silently stops covering the catalogue.
    """
    depths = catalog_column_depths()
    unknown = sorted(spec.name for spec in catalog if spec.name not in depths)
    if unknown:
        raise ValueError(
            f"catalogue column(s) {unknown} declare no lookback depth in "
            "`catalog_column_depths`. The depth is what sizes the producer's price "
            "panel; a column whose depth is unwritten is a column the panel may be "
            "too short for, null on every ticker and indistinguishable from a layer "
            "that was never compiled."
        )
    return max(depths[spec.name] for spec in catalog)


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
        lambda s: s.rolling(
            SMA_SHORT_WINDOW_TRADING_DAYS, min_periods=SMA_SHORT_WINDOW_TRADING_DAYS
        ).mean()
    )
    frame["close_to_sma200_ratio"] = frame["close_raw"] / grouped.transform(
        lambda s: s.rolling(
            SMA_LONG_WINDOW_TRADING_DAYS, min_periods=SMA_LONG_WINDOW_TRADING_DAYS
        ).mean()
    )
    frame["rsi_14_ratio"] = (
        frame.groupby("ticker", sort=False)["close_raw"].transform(
            lambda s: _wilder_rsi(s, RSI_WINDOW_TRADING_DAYS)
        )
        / 100.0
    )

    # -- the v3.0-meta L1 inputs (alpha-engine-config-I10695) --------------
    frame["momentum_5d_log_return"] = log_grouped.diff(MOMENTUM_SHORT_WINDOW_TRADING_DAYS)

    # `np.maximum` propagates a null rather than skipping it, so a ticker's
    # first session (no prior close) has no true range instead of a partial
    # one computed from high - low alone.
    prior_close = grouped.shift(1)
    frame["_true_range"] = np.maximum(
        np.maximum(frame["high_raw"] - frame["low_raw"], (frame["high_raw"] - prior_close).abs()),
        (frame["low_raw"] - prior_close).abs(),
    )
    frame["atr_14_ratio"] = (
        frame.groupby("ticker", sort=False)["_true_range"].transform(
            lambda s: s.rolling(ATR_WINDOW_TRADING_DAYS, min_periods=ATR_WINDOW_TRADING_DAYS).mean()
        )
        / frame["close_raw"]
    )

    returns_by_ticker = frame.groupby("ticker", sort=False)["return_1d_log_return"]
    short_vol = returns_by_ticker.transform(
        lambda s: s.rolling(
            VOL_RATIO_SHORT_WINDOW_TRADING_DAYS, min_periods=VOL_RATIO_SHORT_WINDOW_TRADING_DAYS
        ).std(ddof=1)
    )
    long_vol = returns_by_ticker.transform(
        lambda s: s.rolling(
            VOL_RATIO_LONG_WINDOW_TRADING_DAYS, min_periods=VOL_RATIO_LONG_WINDOW_TRADING_DAYS
        ).std(ddof=1)
    )
    # A zero long-window volatility is an unmeasurable ratio, not a ratio of
    # one — v1's `fillna(1.0)` is the substituted constant this layer refuses.
    frame["vol_ratio_10_60_ratio"] = short_vol / long_vol.where(long_vol > 0)

    rolling_high = grouped.transform(
        lambda s: s.rolling(
            FIFTY_TWO_WEEK_WINDOW_TRADING_DAYS, min_periods=FIFTY_TWO_WEEK_WINDOW_TRADING_DAYS
        ).max()
    )
    rolling_low = grouped.transform(
        lambda s: s.rolling(
            FIFTY_TWO_WEEK_WINDOW_TRADING_DAYS, min_periods=FIFTY_TWO_WEEK_WINDOW_TRADING_DAYS
        ).min()
    )
    frame["dist_from_52w_high_ratio"] = (
        frame["close_raw"] / rolling_high.where(rolling_high > 0) - 1.0
    )
    frame["dist_from_52w_low_ratio"] = frame["close_raw"] / rolling_low.where(rolling_low > 0) - 1.0

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
