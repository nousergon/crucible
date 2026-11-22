"""A deterministic, point-in-time market-state score from the cross-section.

Pure functions over the long panel `crucible.features.compute` works from: one
row per (``trading_day``, ``ticker``), carrying ``market_return_1d_log_return``,
``momentum_20d_log_return``, ``close_to_sma50_ratio`` and
``close_to_sma200_ratio``. No I/O, no clock, no network: a DataFrame in, a
Series or a document out.

The windows below are methodology, not edge, so they are module constants. The
band edges that turn a score into a state ARE tuned, so they carry no default
here: they live in ``alpha-engine-config`` (Regime 3, aec#14083) and arrive as a
mapping, refused by name when short a key, carrying an unknown one, or
misordered.

Point-in-time: every statistic for day D reads rows dated <= D and nothing
else, so appending or reordering rows after D leaves D's output unchanged.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import pandas as pd  # pyright: ignore[reportMissingTypeStubs]

REGIME_SCHEMA: Final = "crucible_regime.v1"

#: Realised-vol window for ``market_vol_ratio`` and the return window behind dispersion.
VOL_WINDOW_SESSIONS: Final = 20
#: Trailing high / median window, and the minimum history before any score exists.
LEVEL_WINDOW_SESSIONS: Final = 252
#: Trailing window each input is z-scored against.
ZSCORE_WINDOW_SESSIONS: Final = 504

#: Fewest observations of an input a z-score is taken over. The windowed inputs
#: (vol ratio, dispersion) are themselves NaN for their first ~270 sessions, so
#: this bounds how soon after they appear a score exists; it is not tuned.
ZSCORE_MIN_OBSERVATIONS: Final = 60

#: Panel columns this module reads.
MARKET_RETURN: Final = "market_return_1d_log_return"
RETURN_20D: Final = "momentum_20d_log_return"
SMA50_RATIO: Final = "close_to_sma50_ratio"
SMA200_RATIO: Final = "close_to_sma200_ratio"
REQUIRED_COLUMNS: Final = ("trading_day", MARKET_RETURN, RETURN_20D, SMA50_RATIO, SMA200_RATIO)

#: Input name -> sign making higher = more stress. A deeper drawdown is a more
#: negative number and thinner breadth is a smaller one, so both are negated.
INPUT_SIGNS: Final[Mapping[str, float]] = {
    "market_drawdown": -1.0,
    "market_vol_ratio": 1.0,
    "breadth_50": -1.0,
    "breadth_200": -1.0,
    "dispersion_20d": 1.0,
}

STATES: Final = ("calm", "normal", "stressed", "crisis")


class RegimeParamsError(ValueError):
    """A regime parameter set is absent, short a key, carries an unknown one, or is misordered."""


@dataclass(frozen=True)
class RegimeParams:
    """The band edges of ``regime_state``. Every field is REQUIRED; none has a default.

    ``score < normal_at`` is calm; ``normal_at <= score < stressed_at`` is
    normal; ``stressed_at <= score < crisis_at`` is stressed; ``crisis_at <=
    score`` is crisis. An edge belongs to the more stressed state.
    """

    normal_at: float
    stressed_at: float
    crisis_at: float

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object], *, source: str) -> RegimeParams:
        """Validate and build. Refuses a missing, unknown or misordered key BY NAME."""
        if not isinstance(payload, Mapping):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise RegimeParamsError(
                f"{source}: regime parameters must be a mapping, got {type(payload).__name__}"
            )
        fields = ("normal_at", "stressed_at", "crisis_at")
        missing = [name for name in fields if name not in payload]
        if missing:
            raise RegimeParamsError(
                f"{source}: regime parameters are missing {missing}. Every key is required "
                "and none has a default."
            )
        unknown = sorted(set(payload) - set(fields))
        if unknown:
            raise RegimeParamsError(
                f"{source}: regime parameters declare {unknown}, which this engine does not "
                f"read. The keys it reads are {list(fields)}."
            )
        values: dict[str, float] = {}
        for name in fields:
            raw = payload[name]
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise RegimeParamsError(f"{source}: {name!r} must be a number; got {raw!r}")
            if not math.isfinite(raw):
                raise RegimeParamsError(f"{source}: {name!r} must be finite; got {raw!r}")
            values[name] = float(raw)
        if not values["normal_at"] < values["stressed_at"] < values["crisis_at"]:
            raise RegimeParamsError(
                f"{source}: bands must satisfy normal_at < stressed_at < crisis_at; got "
                f"normal_at={values['normal_at']}, stressed_at={values['stressed_at']}, "
                f"crisis_at={values['crisis_at']}"
            )
        return cls(**values)

    def to_dict(self) -> dict[str, float]:
        return {
            "normal_at": self.normal_at,
            "stressed_at": self.stressed_at,
            "crisis_at": self.crisis_at,
        }

    def digest(self) -> str:
        """sha256 of the canonical JSON of the bands: names the parameter set a state used."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


class _Panel(Protocol):
    """The slice of a DataFrame this module reads: column names and column access.

    Declared structurally so the numeric core is typed numpy, not pandas.
    """

    @property
    def columns(self) -> Iterable[object]: ...

    def __getitem__(self, key: str, /) -> object: ...


Floats = npt.NDArray[np.float64]


def _require_columns(frame: _Panel) -> None:
    present = {str(name) for name in frame.columns}
    absent = [name for name in REQUIRED_COLUMNS if name not in present]
    if absent:
        raise ValueError(f"the regime frame is missing columns {absent}")


def _floats(frame: _Panel, name: str) -> Floats:
    return np.asarray(frame[name], dtype=np.float64)


def _trailing(
    values: Floats,
    window: int,
    min_periods: int,
    reduce: Callable[[Floats], float],
) -> Floats:
    """``reduce`` over the finite values of the trailing window ending AT each index.

    NaN when the window holds fewer than ``min_periods`` finite values. Index i
    reads ``values[:i + 1]`` and nothing after it.
    """
    out = np.full(values.shape, np.nan)
    for i in range(len(values)):
        block = values[max(0, i - window + 1) : i + 1]
        block = block[np.isfinite(block)]
        if len(block) >= min_periods:
            out[i] = reduce(block)
    return out


def _std0(block: Floats) -> float:
    return float(np.std(block))


def _ratio_to_median(values: Floats) -> Floats:
    """``values`` over the median of its own trailing 252 sessions (NaN if that is not positive)."""
    median = _trailing(values, LEVEL_WINDOW_SESSIONS, LEVEL_WINDOW_SESSIONS, np.median)
    return np.where(median > 0, values / np.where(median > 0, median, 1.0), np.nan)


def _day_mean(inverse: npt.NDArray[np.intp], values: Floats, n_days: int) -> Floats:
    finite = np.isfinite(values)
    counts = np.bincount(inverse, weights=finite.astype(np.float64), minlength=n_days)
    sums = np.bincount(inverse, weights=np.where(finite, values, 0.0), minlength=n_days)
    return np.where(counts > 0, sums / np.where(counts > 0, counts, 1.0), np.nan)


def _day_std0(inverse: npt.NDArray[np.intp], values: Floats, n_days: int) -> Floats:
    finite = np.isfinite(values)
    counts = np.bincount(inverse, weights=finite.astype(np.float64), minlength=n_days)
    mean = _day_mean(inverse, values, n_days)
    centred = np.where(finite, values - mean[inverse], 0.0)
    ss = np.bincount(inverse, weights=centred * centred, minlength=n_days)
    return np.where(counts > 0, np.sqrt(ss / np.where(counts > 0, counts, 1.0)), np.nan)


def _day_share_above_one(inverse: npt.NDArray[np.intp], values: Floats, n_days: int) -> Floats:
    finite = np.isfinite(values)
    counts = np.bincount(inverse, weights=finite.astype(np.float64), minlength=n_days)
    above = np.bincount(
        inverse, weights=(finite & (values > 1.0)).astype(np.float64), minlength=n_days
    )
    return np.where(counts > 0, above / np.where(counts > 0, counts, 1.0), np.nan)


def _input_arrays(frame: _Panel) -> tuple[npt.NDArray[np.datetime64], dict[str, Floats]]:
    _require_columns(frame)
    days = np.asarray(frame["trading_day"], dtype="datetime64[ns]")
    unique_days, inverse = np.unique(days, return_inverse=True)
    inverse = inverse.astype(np.intp)
    n_days = len(unique_days)

    market = _day_mean(inverse, _floats(frame, MARKET_RETURN), n_days)
    index = np.exp(np.cumsum(np.nan_to_num(market)))
    high = _trailing(index, LEVEL_WINDOW_SESSIONS, LEVEL_WINDOW_SESSIONS, np.max)
    drawdown = index / high - 1.0

    realised = _trailing(market, VOL_WINDOW_SESSIONS, VOL_WINDOW_SESSIONS, _std0)
    dispersion = _day_std0(inverse, _floats(frame, RETURN_20D), n_days)

    return unique_days, {
        "market_drawdown": drawdown,
        "market_vol_ratio": _ratio_to_median(realised),
        "breadth_50": _day_share_above_one(inverse, _floats(frame, SMA50_RATIO), n_days),
        "breadth_200": _day_share_above_one(inverse, _floats(frame, SMA200_RATIO), n_days),
        "dispersion_20d": _ratio_to_median(dispersion),
    }


def _scores(inputs: Mapping[str, Floats]) -> Floats:
    zscores: list[Floats] = []
    for name, sign in INPUT_SIGNS.items():
        series = inputs[name]
        mean = _trailing(series, ZSCORE_WINDOW_SESSIONS, ZSCORE_MIN_OBSERVATIONS, np.mean)
        spread = _trailing(series, ZSCORE_WINDOW_SESSIONS, ZSCORE_MIN_OBSERVATIONS, _std0)
        safe = np.where(spread > 0, spread, 1.0)
        zscores.append(np.where(spread > 0, sign * (series - mean) / safe, np.nan))
    score = np.mean(np.stack(zscores), axis=0)
    score[: LEVEL_WINDOW_SESSIONS - 1] = np.nan
    return score


def regime_inputs(frame: _Panel) -> pd.DataFrame:
    """The five raw inputs by trading day (index ``trading_day``, ascending).

    A value is NaN until the history it needs exists; it is never a guess.
    """
    import pandas as pd  # pyright: ignore[reportMissingTypeStubs]  # noqa: PLC0415 - pandas ships no stubs here; keeps the module importable without it

    days, inputs = _input_arrays(frame)
    return pd.DataFrame(inputs, index=pd.DatetimeIndex(days, name="trading_day"))


def stress_score(frame: _Panel) -> pd.Series:
    """The daily stress score: the mean of each input's signed trailing z-score.

    Each input is z-scored against its own trailing 504-session window ending
    at the day itself, once it has 60 observations. Higher = more stress. NaN
    for a day with fewer than 252 sessions of history, or whose window has no
    spread, or any of whose inputs is NaN: a partial mean would be a different
    number under the same name.
    """
    import pandas as pd  # pyright: ignore[reportMissingTypeStubs]  # noqa: PLC0415 - see regime_inputs

    days, inputs = _input_arrays(frame)
    return pd.Series(
        _scores(inputs), index=pd.DatetimeIndex(days, name="trading_day"), name="stress_score"
    )


def regime_state(score: float, params: RegimeParams) -> str:
    """``calm | normal | stressed | crisis`` for one score; an edge is the more stressed state."""
    if not math.isfinite(score):
        raise ValueError(f"a stress score of {score!r} has no state; it is not a measured number")
    if score >= params.crisis_at:
        return "crisis"
    if score >= params.stressed_at:
        return "stressed"
    if score >= params.normal_at:
        return "normal"
    return "calm"


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


@dataclass(frozen=True)
class RegimeDay:
    """One day's regime reading. ``stress_score`` and ``state`` are None before history exists."""

    day: str
    inputs: Mapping[str, float | None]
    stress_score: float | None
    state: str | None
    params_digest: str

    def to_document(self) -> dict[str, object]:
        return {
            "schema": REGIME_SCHEMA,
            "day": self.day,
            "inputs": dict(self.inputs),
            "stress_score": self.stress_score,
            "state": self.state,
            "params_digest": self.params_digest,
        }


def regime_days(frame: _Panel, params: RegimeParams) -> list[RegimeDay]:
    """A ``RegimeDay`` for every trading day in ``frame``, ascending."""
    days, inputs = _input_arrays(frame)
    scores = _scores(inputs)
    digest = params.digest()
    out: list[RegimeDay] = []
    for i, day in enumerate(days):
        value = float(scores[i])
        measured = math.isfinite(value)
        out.append(
            RegimeDay(
                day=str(day.astype("datetime64[D]")),
                inputs={name: _finite_or_none(float(inputs[name][i])) for name in INPUT_SIGNS},
                stress_score=value if measured else None,
                state=regime_state(value, params) if measured else None,
                params_digest=digest,
            )
        )
    return out
