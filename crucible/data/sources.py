"""Price sources behind one adapter. ArcticDB is one implementation, not the shape.

Normative source: plan §4.3, principle 8 (substitutability).

The data layer reads a **panel**: one row per (trading_day, ticker), with the
canonical OHLCV columns. Where those bytes come from is a
:class:`PriceSource`, and the production implementation
(:class:`ArcticPriceSource`) is the only thing in the package that knows
ArcticDB exists. Swapping the store means writing one class here; nothing in
`crucible/data/daily.py`, `crucible/features/` or `crucible/slots/` changes.

**A missing source is a failure, never a zero-fill.** Every path that cannot
produce the panel raises :class:`MissingSourceError`, which the runner turns
into `status: failed` with the source named in `reason`. The
`SYSTEM_OPTIMIZED.md` §3 lesson is that a zero-filled input is a well-formed
artifact containing nothing, and every downstream gate passes on it.

**The panel's columns carry units suffixes** (fleet rule; root cause:
`avg_volume_20d` emitted as a ratio and consumed as raw shares). Raw prices
and share counts are `_raw`; nothing here is normalized.
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

__all__ = [
    "PANEL_COLUMNS",
    "ArcticPriceSource",
    "FramePriceSource",
    "MissingSourceError",
    "PriceSource",
    "normalize_panel",
]

#: The canonical panel. Every source returns exactly these columns, in this
#: order, so a swap cannot quietly change the shape the feature layer reads.
PANEL_COLUMNS: tuple[str, ...] = (
    "trading_day",
    "ticker",
    "open_raw",
    "high_raw",
    "low_raw",
    "close_raw",
    "volume_raw",
)

_OHLCV_RENAME = {
    "Open": "open_raw",
    "High": "high_raw",
    "Low": "low_raw",
    "Close": "close_raw",
    "Volume": "volume_raw",
    "open": "open_raw",
    "high": "high_raw",
    "low": "low_raw",
    "close": "close_raw",
    "volume": "volume_raw",
}


class MissingSourceError(RuntimeError):
    """A required input could not be read. The run fails; it does not degrade.

    Carries the source name and the concrete thing that was missing, because
    "data unavailable" on a Saturday morning is the reason that made three
    consecutive weekly failures indistinguishable from one another.
    """


class PriceSource(ABC):
    """One adapter: give me a panel ending at this trading day."""

    #: Recorded in the manifest so `explain` can say which source a run read.
    name: str = "abstract"

    @abstractmethod
    def load_panel(
        self,
        *,
        end: dt.date,
        lookback_days: int,
        symbols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Rows for every session in the window, for every symbol available.

        Raises :class:`MissingSourceError` when the source cannot be reached
        or returns nothing. Returning an empty frame is NOT an option: an
        empty panel is the well-formed-artifact-containing-nothing shape.
        """

    @abstractmethod
    def snapshot_id(self) -> str:
        """An identifier for the version of the source that was read.

        §9.7: a restatement is versioned, not overwritten, and the run that
        read the old value must be identifiable. This value lands in the
        data artifact as `data_snapshot_id`.
        """


def normalize_panel(frames: dict[str, Any], *, end: dt.date, lookback_days: int) -> pd.DataFrame:
    """A ``{ticker: OHLCV frame}`` mapping to the canonical long panel.

    Shared by every source so the dedup, the tz-flattening, the column
    renaming and the window trim happen once. A source that normalized its
    own way would produce a panel differing from another source's in a way
    no test would see.
    """
    import pandas as pd

    start = end - dt.timedelta(days=lookback_days)
    rows: list[pd.DataFrame] = []
    for ticker, frame in sorted(frames.items()):
        if frame is None or len(frame) == 0:
            continue
        block = frame.rename(columns=_OHLCV_RENAME).copy()
        if "trading_day" not in block.columns:
            index = pd.to_datetime(block.index)
            try:
                index = index.tz_localize(None)
            except TypeError:
                index = index.tz_convert(None)
            block["trading_day"] = index.date
        block["ticker"] = ticker
        missing = [c for c in PANEL_COLUMNS if c not in block.columns]
        if missing:
            raise MissingSourceError(
                f"source frame for {ticker!r} is missing column(s) {missing}. The panel "
                f"contract is {list(PANEL_COLUMNS)}; a frame short a column would be "
                "read as a zero column downstream, which is the zero-fill this layer "
                "exists to refuse."
            )
        rows.append(block[list(PANEL_COLUMNS)])

    if not rows:
        raise MissingSourceError(
            "no source frame carried any rows; a panel with zero tickers is a FAILED "
            "run, never an empty-but-successful one (plan §4.3)"
        )

    panel = pd.concat(rows, ignore_index=True)
    panel["trading_day"] = pd.to_datetime(panel["trading_day"]).dt.date
    panel = panel[(panel["trading_day"] > start) & (panel["trading_day"] <= end)]
    panel = panel.drop_duplicates(subset=["trading_day", "ticker"], keep="last")
    panel = panel.sort_values(["trading_day", "ticker"]).reset_index(drop=True)
    if panel.empty:
        raise MissingSourceError(
            f"no rows fall inside the window ({start}, {end}] after normalization. "
            "The source returned data, but none of it covers the requested trading "
            "days — a stale source, not an empty one, and the two must not read alike."
        )
    return panel


class FramePriceSource(PriceSource):
    """An in-memory panel. The test backend, and the fixture-replay backend.

    Not a mock: it runs `normalize_panel`, the same function the production
    source runs, so a test against it tests the real contract rather than a
    parallel one.
    """

    name = "frames"

    def __init__(self, frames: dict[str, Any], *, snapshot: str = "frames:in-memory") -> None:
        self._frames = frames
        self._snapshot = snapshot

    def load_panel(
        self,
        *,
        end: dt.date,
        lookback_days: int,
        symbols: list[str] | None = None,
    ) -> pd.DataFrame:
        frames = self._frames
        if symbols is not None:
            wanted = set(symbols)
            absent = sorted(wanted - set(frames))
            if absent:
                raise MissingSourceError(
                    f"requested symbol(s) absent from the source: {absent}. A requested "
                    "symbol that silently drops out of the panel is survivorship bias "
                    "introduced by the loader."
                )
            frames = {t: f for t, f in frames.items() if t in wanted}
        return normalize_panel(frames, end=end, lookback_days=lookback_days)

    def snapshot_id(self) -> str:
        return self._snapshot


class ArcticPriceSource(PriceSource):
    """The production source: the ArcticDB universe library, read-only.

    **Read-only from anywhere; written from in-region only.** Nothing in this
    class writes, and `crucible/data/heal.py` carries the in-region guard for
    the one job that does. A ~900-ticker write is dominated by S3 round-trip
    latency, and a 20-40 minute in-region job took 3+ hours from a laptop
    (measured 2026-07-15).

    The `arcticdb` extra is imported lazily and its absence is reported by
    name. It is never substituted with a different source: a run that
    silently read a fallback measured something other than what it reports.
    """

    name = "arcticdb:universe"

    def __init__(self, bucket: str, *, region: str | None = None) -> None:
        self.bucket = bucket
        self.region = region

    def load_panel(
        self,
        *,
        end: dt.date,
        lookback_days: int,
        symbols: list[str] | None = None,
    ) -> pd.DataFrame:
        try:
            from nousergon_lib.arcticdb import load_universe_ohlcv
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise MissingSourceError(
                "the ArcticDB price source needs the `arcticdb` extra: install "
                "`crucible[arcticdb]`. This run is NOT falling back to another "
                f"source — a substituted source measures something else ({exc})."
            ) from exc

        try:
            frames = load_universe_ohlcv(
                self.bucket,
                symbols=symbols,
                lookback_days=lookback_days,
                end=str(end),
                region=self.region,
            )
        except Exception as exc:
            raise MissingSourceError(
                f"ArcticDB universe library on bucket {self.bucket!r} could not be read "
                f"for the window ending {end}: {type(exc).__name__}: {exc}"
            ) from exc

        if not frames:
            raise MissingSourceError(
                f"ArcticDB universe library on bucket {self.bucket!r} returned zero "
                f"symbols for the window ending {end}. `load_universe_ohlcv` drops "
                "per-ticker read failures at WARNING and returns what it got, so an "
                "empty result is an outage or a wrong bucket, not an empty market."
            )
        return normalize_panel(frames, end=end, lookback_days=lookback_days)

    def snapshot_id(self) -> str:
        """`arcticdb:{bucket}` — the library is versioned, this read is not.

        ArcticDB versions every write, so a restatement is recoverable; what
        this identifier pins is WHICH store was read. Pinning a per-symbol
        version tuple would be an artifact of its own, and is the follow-up
        named in the PR body rather than a value invented here.
        """
        return f"arcticdb:{self.bucket}"
