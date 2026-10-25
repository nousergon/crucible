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

**An absent symbol is two different facts, and only one of them is an
outage** (`alpha-engine-config-I10127`). Four `data.heal` backfills for the
window ending 2024-08-07 all raised `MissingSourceError` on ``['FDXF',
'HONA', 'Q', 'SARO', 'SNDK', 'SOLS']`` — six 2025-26 listings that simply
did not exist yet in that 2024 window, surfaced only because today's
903-name universe was applied to a historical session. Before this, any
symbol a source's windowed read dropped was treated as a partial outage,
which made `COVERAGE_FLOOR_RATIO` unreachable for any historical session —
a threshold nothing could ever enforce, since the run failed before the
floor got a chance to fire. Every :class:`PriceSource` that drops a symbol
now asks the store what it actually knows about that symbol's stored
history before raising: a symbol not in the store at all, or whose stored
history OVERLAPS the requested window yet still came back empty, or whose
history could not be read, stays exactly the outage it always was
(`MissingSourceError`, unchanged). A symbol whose entire stored history
lies outside the window is a fact about the market, not the source — it is
returned alongside the panel, in ``panel.attrs["unlisted_in_window"]``,
rather than failing the run. This is never guessed from an empty frame
alone: both implementations resolve real bounds before choosing either
branch.

**A departed symbol keeps its history** (`alpha-engine-config-I12287`).
The data layer prunes a delisted name from the live price library by
MOVING its whole stored history into a separate retention library
(``delisted_history``, written by the data repository's
``prune_delisted_tickers``), so a reader of the live library alone sees a
delisted name vanish from every session it ever traded, not just the ones
after it stopped. :meth:`PriceSource.retained_symbols` names the retained
symbols whose history reaches into a window, and
:meth:`PriceSource.load_departed` reads departed names from whichever
library holds them. A departed symbol held in neither is NAMED
(``DepartedHistory.without_history``), never invented and never silently
dropped. Point-in-time constituents from a provider stay phase-5 work —
see the binding plan §10.4 and `alpha-engine-config-I9761`.
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

__all__ = [
    "PANEL_COLUMNS",
    "RETENTION_LIBRARY",
    "ArcticPriceSource",
    "DepartedHistory",
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

#: The ArcticDB library a delisted symbol's full stored history is MOVED to
#: when the data layer prunes it from the live `universe` library — the data
#: repository's ``store.arctic_store.get_delisted_history_lib`` contract
#: (`alpha-engine-config-I1943`, leg 3). Read here, never written.
RETENTION_LIBRARY = "delisted_history"

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


def _classify_absent_symbols(
    absent: list[str],
    bounds: dict[str, tuple[dt.date, dt.date] | None],
    *,
    window_start: dt.date,
    window_end: dt.date,
) -> tuple[list[str], list[str]]:
    """Split ``absent`` into ``(missing, unlisted_in_window)``.

    ``bounds[sym]`` is the symbol's FULL stored-history ``(first, last)``
    date pair, or ``None`` when the symbol could not be resolved at all —
    not in the store, or its description could not be read. A symbol whose
    bounds fall entirely outside ``(window_start, window_end]`` is a fact
    about the market: the window predates the listing, or postdates the
    delisting. Everything else — unresolved, or bounds that DO overlap the
    window yet the source still returned nothing for it — stays a
    :class:`MissingSourceError`: an overlapping window with no data is a
    partial outage, and the two must never be told apart by guessing from
    an empty frame alone.
    """
    missing: list[str] = []
    unlisted: list[str] = []
    for sym in absent:
        b = bounds.get(sym)
        if b is None:
            missing.append(sym)
            continue
        b_start, b_end = b
        if b_end < window_start or b_start > window_end:
            unlisted.append(sym)
        else:
            missing.append(sym)
    return sorted(missing), sorted(unlisted)


def _frame_dates(frame: Any) -> np.ndarray:
    """Every date a source frame carries, as a numpy array of `dt.date`.

    Mirrors `normalize_panel`'s own index handling exactly (``trading_day``
    column if present, else the index; tz-stripped) so a value computed here
    can never disagree with what `normalize_panel` would have read from the
    same frame.
    """
    import pandas as pd

    values = frame["trading_day"] if "trading_day" in frame.columns else frame.index
    index = pd.to_datetime(values)
    try:
        index = index.tz_localize(None)
    except TypeError:
        index = index.tz_convert(None)
    return index.date


def _frame_date_bounds(frame: Any) -> tuple[dt.date, dt.date] | None:
    """The ``(first, last)`` date carried by a source frame, or ``None`` if empty."""
    if frame is None or len(frame) == 0:
        return None
    dates = _frame_dates(frame)
    return (min(dates), max(dates))


def _overlaps_window(
    bounds: tuple[dt.date, dt.date] | None, *, window_start: dt.date, window_end: dt.date
) -> bool:
    """Whether stored-history ``bounds`` reach into ``(window_start, window_end]``."""
    if bounds is None:
        return False
    first, last = bounds
    return last > window_start and first <= window_end


def _empty_panel() -> pd.DataFrame:
    import pandas as pd

    return pd.DataFrame({column: [] for column in PANEL_COLUMNS})


@dataclass(frozen=True)
class DepartedHistory:
    """What a source holds for symbols that have LEFT the declared universe.

    ``panel`` is the canonical long panel (:data:`PANEL_COLUMNS`) of every
    departed symbol with a stored row inside the window — each up to its own
    last stored session and no further, because nothing here fills forward.
    It may be empty: a window no departure reached is not an outage.

    ``without_history`` names the departed symbols the source holds NO stored
    history for in any library — the "never written" case. Named rather than
    raised: a departed name is outside the coverage denominator, so its
    absence cannot fail the day, and naming it is what lets a consumer tell
    a name with genuinely no data from a panel that dropped one.
    """

    panel: pd.DataFrame
    without_history: tuple[str, ...]


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

        The returned frame carries ``attrs["unlisted_in_window"]``: a sorted
        list of requested symbols whose entire stored history falls outside
        ``(end - lookback_days, end]`` — never a source failure, and never a
        reason to shrink the panel's expected-universe denominator. Always
        present (``[]`` when empty), so its absence is never mistaken for
        "not measured".
        """

    @abstractmethod
    def retained_symbols(self, *, end: dt.date, lookback_days: int) -> list[str]:
        """Every symbol the source RETAINS after it left the live listing whose
        stored history reaches into ``(end - lookback_days, end]``, sorted.

        Abstract rather than defaulted to ``[]``: a source that could not say
        what it retains would silently produce a survivors-only panel, which
        is the defect `alpha-engine-config-I12287` exists to close.
        """

    @abstractmethod
    def load_departed(
        self, *, end: dt.date, lookback_days: int, symbols: list[str]
    ) -> DepartedHistory:
        """The window's history for ``symbols`` — names that LEFT the declared
        universe — read from the live listing first and the retention store
        second.

        Never raises for a symbol held nowhere (that is
        :attr:`DepartedHistory.without_history`), nor for one whose history
        lies wholly outside the window (it simply contributes no rows). Still
        raises :class:`MissingSourceError` for a symbol whose stored history
        overlaps the window yet came back empty: that is a partial outage,
        exactly as it is for a universe name.
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

    **A ticker present in ``frames`` with a `None` or empty value is a
    per-ticker read failure, and it RAISES rather than being dropped.**
    Before this, a `continue` here silently thinned the panel by exactly the
    tickers a partial source outage failed on, while the ONLY loud failure
    in this module was zero tickers surviving at all — a partial ArcticDB
    outage produced a well-formed, thin panel and nothing downstream could
    tell. `ArcticPriceSource` and every other :class:`PriceSource` must
    either omit a failed ticker's key entirely (caught below, at
    :func:`ArcticPriceSource.load_panel`, when the caller named it in
    ``symbols``) or raise before handing this function a frame it knows is
    empty — never carry the empty value in silently.
    """
    import pandas as pd

    start = end - dt.timedelta(days=lookback_days)
    dropped = sorted(t for t, f in frames.items() if f is None or len(f) == 0)
    if dropped:
        raise MissingSourceError(
            f"source frame(s) for {dropped} are `None` or empty. A per-ticker read "
            "failure that survives into `normalize_panel` as an empty value is a "
            "partial outage, and dropping it silently is the shape that turns a "
            "well-formed panel into one covering a fraction of the universe with "
            "nothing downstream able to tell (plan §4.3, principle 7 — no data is "
            "never rendered as green)."
        )
    rows: list[pd.DataFrame] = []
    for ticker, frame in sorted(frames.items()):
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

    def __init__(
        self,
        frames: dict[str, Any],
        *,
        retained: dict[str, Any] | None = None,
        snapshot: str = "frames:in-memory",
    ) -> None:
        self._frames = frames
        # The retention store's stand-in: frames for names that LEFT the live
        # listing. Kept apart from `frames` exactly as the production source
        # keeps the two libraries apart, so `load_panel` never sees them.
        self._retained = dict(retained or {})
        self._snapshot = snapshot

    def retained_symbols(self, *, end: dt.date, lookback_days: int) -> list[str]:
        window_start = end - dt.timedelta(days=lookback_days)
        return sorted(
            ticker
            for ticker, frame in self._retained.items()
            if _overlaps_window(
                _frame_date_bounds(frame), window_start=window_start, window_end=end
            )
        )

    def load_departed(
        self, *, end: dt.date, lookback_days: int, symbols: list[str]
    ) -> DepartedHistory:
        window_start = end - dt.timedelta(days=lookback_days)
        present: dict[str, Any] = {}
        without: list[str] = []
        overlapping_but_empty: list[str] = []
        for ticker in sorted(set(symbols)):
            frame = self._frames.get(ticker)
            if frame is None:
                frame = self._retained.get(ticker)
            bounds = _frame_date_bounds(frame)
            if bounds is None:
                without.append(ticker)
                continue
            dates = _frame_dates(frame)
            if ((dates > window_start) & (dates <= end)).any():
                present[ticker] = frame
            elif _overlaps_window(bounds, window_start=window_start, window_end=end):
                overlapping_but_empty.append(ticker)
        if overlapping_but_empty:
            raise MissingSourceError(
                f"departed symbol(s) {overlapping_but_empty} have stored history overlapping "
                f"the window ending {end} but no row inside it — a partial outage, not a "
                "departure."
            )
        panel = (
            normalize_panel(present, end=end, lookback_days=lookback_days)
            if present
            else _empty_panel()
        )
        return DepartedHistory(panel=panel, without_history=tuple(without))

    def load_panel(
        self,
        *,
        end: dt.date,
        lookback_days: int,
        symbols: list[str] | None = None,
    ) -> pd.DataFrame:
        frames = self._frames
        unlisted: list[str] = []
        if symbols is not None:
            wanted = set(symbols)
            window_start = end - dt.timedelta(days=lookback_days)
            # Mirror `ArcticPriceSource`'s windowed `read_batch`, which drops a
            # ticker's key entirely (rather than returning an empty frame) when
            # its stored history has zero rows inside the requested date range —
            # so this fixture source classifies the SAME two ways: `present` if
            # the ticker's own stored history has any row in the window, else
            # its full-history bounds decide `missing` vs `unlisted_in_window`
            # via `_classify_absent_symbols`.
            bounds: dict[str, tuple[dt.date, dt.date] | None] = {}
            present: set[str] = set()
            for ticker in wanted:
                frame = frames.get(ticker)
                b = _frame_date_bounds(frame)
                bounds[ticker] = b
                if b is None:
                    continue
                dates = _frame_dates(frame)
                if ((dates > window_start) & (dates <= end)).any():
                    present.add(ticker)
            absent = sorted(wanted - present)
            if absent:
                missing, unlisted = _classify_absent_symbols(
                    absent, bounds, window_start=window_start, window_end=end
                )
                if missing:
                    raise MissingSourceError(
                        f"requested symbol(s) absent from the source: {missing}. A "
                        "requested symbol that silently drops out of the panel is "
                        "survivorship bias introduced by the loader."
                    )
            frames = {t: f for t, f in frames.items() if t in present}
        panel = normalize_panel(frames, end=end, lookback_days=lookback_days)
        panel.attrs["unlisted_in_window"] = unlisted
        return panel

    def snapshot_id(self) -> str:
        return self._snapshot

    def symbols(self) -> list[str]:
        """Every ticker this source can serve.

        The coverage DENOMINATOR for a fixture or replay run. `run_daily` now
        refuses a run with no declared universe — a run with no denominator
        reports `OK` over a ratio it never computed, which is the
        901-of-903 bug class — and a caller replaying a fixture has to say what
        it expected. Reading it off the source is the honest answer for that
        caller and it is NOT available on the production source by design:
        asking ArcticDB "what do you have" and then measuring coverage against
        the reply is a denominator that moves with the outage.
        """
        return sorted(self._frames)


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

    **`library` is a dedicated-library override** (`alpha-engine-config-
    I10457`). Default (`None`) is production, byte-for-byte unchanged: the
    read still goes through `nousergon_lib.arcticdb.load_universe_ohlcv`/
    `open_universe_lib`, both hard-wired to the single production `universe`
    library. Passing a name routes through the generic
    `open_arctic(bucket).get_library(name, create_if_missing=True)` +
    `_load_arctic_frames` path instead — the same low-level path
    `tests/integration/conftest.py::arctic_library` already uses to avoid
    ever touching `open_universe_lib`'s hard-wired name. The two paths never
    cross: a dedicated-library read that fell through to the production
    helpers on any failure would silently read `universe` from a call that
    asked for something else.
    """

    name = "arcticdb:universe"

    def __init__(
        self, bucket: str, *, region: str | None = None, library: str | None = None
    ) -> None:
        # `crucible/config.py::DEFAULT_ARCTIC_BUCKET` carries no default bucket
        # name (`alpha-engine-config-I9906` finding 3 — a bucket name in a
        # public repo's package source is an infrastructure identifier
        # `crucible/AGENTS.md` forbids). An empty bucket here means
        # `CRUCIBLE_ARCTIC_BUCKET` was never set, and failing loud here — at
        # construction, before any S3 call — beats the opaque `NoSuchBucket`
        # or empty-name error the SDK would otherwise raise deep inside
        # `load_panel`.
        if not bucket:
            raise ValueError(
                "ArcticPriceSource needs a bucket name — set CRUCIBLE_ARCTIC_BUCKET "
                "(there is no default; see crucible/config.py::DEFAULT_ARCTIC_BUCKET)"
            )
        self.bucket = bucket
        self.region = region
        self.library = library

    def _library_label(self) -> str:
        return f"library {self.library!r}" if self.library else "universe library"

    def _load_dedicated_library_frames(
        self, *, end: dt.date, lookback_days: int, symbols: list[str] | None
    ) -> dict[str, Any]:
        """The generic read path a dedicated `library` override takes.

        Never calls `load_universe_ohlcv`/`open_universe_lib` — both are
        hard-wired to `UNIVERSE_LIB` and a fallback to either here would
        silently read the production library from a call that named a
        different one.
        """
        from nousergon_lib.arcticdb import _load_arctic_frames, open_arctic

        library = self.library
        if not library:
            raise MissingSourceError("a dedicated-library read needs a named library")
        arctic = open_arctic(self.bucket, region=self.region)
        lib = arctic.get_library(library, create_if_missing=True)
        resolved_symbols = symbols if symbols is not None else sorted(lib.list_symbols())
        return _load_arctic_frames(
            lib,
            resolved_symbols,
            lookback_days=lookback_days,
            end=str(end),
            columns=None,
            label=f"ArcticPriceSource(library={self.library!r})",
        )

    def load_panel(
        self,
        *,
        end: dt.date,
        lookback_days: int,
        symbols: list[str] | None = None,
    ) -> pd.DataFrame:
        try:
            if self.library:
                from nousergon_lib.arcticdb import _load_arctic_frames, open_arctic  # noqa: F401
            else:
                from nousergon_lib.arcticdb import load_universe_ohlcv
        except ImportError as exc:
            raise MissingSourceError(
                "the ArcticDB price source needs the `arcticdb` extra: install "
                "`crucible[arcticdb]`. This run is NOT falling back to another "
                f"source — a substituted source measures something else ({exc})."
            ) from exc

        try:
            if self.library:
                frames = self._load_dedicated_library_frames(
                    end=end, lookback_days=lookback_days, symbols=symbols
                )
            else:
                frames = load_universe_ohlcv(
                    self.bucket,
                    symbols=symbols,
                    lookback_days=lookback_days,
                    end=str(end),
                    region=self.region,
                )
        except Exception as exc:
            raise MissingSourceError(
                f"ArcticDB {self._library_label()} on bucket {self.bucket!r} could not be "
                f"read for the window ending {end}: {type(exc).__name__}: {exc}"
            ) from exc

        if not frames:
            raise MissingSourceError(
                f"ArcticDB {self._library_label()} on bucket {self.bucket!r} returned zero "
                f"symbols for the window ending {end}. The batch read drops per-ticker "
                "read failures at WARNING and returns what it got, so an empty result is "
                "an outage or a wrong bucket/library, not an empty market."
            )
        unlisted: list[str] = []
        if symbols is not None:
            # `load_universe_ohlcv` "drops per-ticker read failures at WARNING and
            # returns what it got" (its own docstring) — a PARTIAL outage returns a
            # non-empty `frames` dict that is simply missing the failed tickers'
            # keys. It ALSO returns a dict missing a ticker's key when that
            # ticker's windowed `read_batch` call legitimately found zero rows —
            # e.g. its stored history starts after `end` (a 2025-26 listing read
            # against a 2024 window, `alpha-engine-config-I10127`). Both shapes
            # look identical here: an absent key. Resolved below by asking the
            # library what it actually knows about each absent symbol's stored
            # history, never by guessing from the empty read alone.
            absent = sorted(set(symbols) - set(frames))
            if absent:
                window_start = end - dt.timedelta(days=lookback_days)
                bounds = self._symbol_bounds(absent)
                missing, unlisted = _classify_absent_symbols(
                    absent, bounds, window_start=window_start, window_end=end
                )
                if missing:
                    raise MissingSourceError(
                        f"ArcticDB universe library on bucket {self.bucket!r} dropped "
                        f"{len(missing)} of {len(symbols)} requested symbol(s) for the "
                        f"window ending {end}: "
                        f"{missing[:20]}{'…' if len(missing) > 20 else ''}. A requested "
                        "symbol that silently drops out of the panel is survivorship "
                        "bias introduced by the source — a partial outage must fail the "
                        "run, not thin the panel."
                    )
        panel = normalize_panel(frames, end=end, lookback_days=lookback_days)
        panel.attrs["unlisted_in_window"] = unlisted
        return panel

    def _retention_lib(self) -> Any:
        """The retention library, opened read-only (never created).

        ``None`` under a dedicated-``library`` override: that override names
        ONE library for an isolated read (`alpha-engine-config-I10457`), and
        reaching from it into the production retention store would mix two
        stores in one panel — the cross-over the override's docstring
        forbids. An absent retention library on the production path is an
        outage, not an empty store: the data layer creates it on its first
        prune, so a production bucket without it is the wrong bucket.
        """
        if self.library:
            return None
        from nousergon_lib.arcticdb import open_arctic

        try:
            return open_arctic(self.bucket, region=self.region).get_library(RETENTION_LIBRARY)
        except Exception as exc:
            raise MissingSourceError(
                f"ArcticDB retention library {RETENTION_LIBRARY!r} on bucket {self.bucket!r} "
                f"could not be opened: {type(exc).__name__}: {exc}. Without it every "
                "delisted name's history is invisible and the panel is survivors-only."
            ) from exc

    def _live_lib(self) -> Any:
        from nousergon_lib.arcticdb import open_arctic, open_universe_lib

        if self.library:
            return open_arctic(self.bucket, region=self.region).get_library(
                self.library, create_if_missing=True
            )
        return open_universe_lib(self.bucket, region=self.region)

    def retained_symbols(self, *, end: dt.date, lookback_days: int) -> list[str]:
        import pandas as pd

        retention = self._retention_lib()
        if retention is None:
            return []
        window_start = end - dt.timedelta(days=lookback_days)
        try:
            symbols = sorted(retention.list_symbols())
            descriptions = retention.get_description_batch(symbols) if symbols else []
        except Exception as exc:
            raise MissingSourceError(
                f"ArcticDB retention library {RETENTION_LIBRARY!r} on bucket {self.bucket!r} "
                f"could not be listed: {type(exc).__name__}: {exc}"
            ) from exc
        out: list[str] = []
        for sym, description in zip(symbols, descriptions, strict=True):
            start_ts, end_ts = description.date_range
            if pd.isna(start_ts) or pd.isna(end_ts):
                raise MissingSourceError(
                    f"retained symbol {sym!r} carries no date range in {RETENTION_LIBRARY!r}; "
                    "whether its history reaches this window cannot be decided, and "
                    "guessing either way would add or drop a name silently."
                )
            bounds = (pd.Timestamp(start_ts).date(), pd.Timestamp(end_ts).date())
            if _overlaps_window(bounds, window_start=window_start, window_end=end):
                out.append(sym)
        return out

    def load_departed(
        self, *, end: dt.date, lookback_days: int, symbols: list[str]
    ) -> DepartedHistory:
        import pandas as pd
        from nousergon_lib.arcticdb import _load_arctic_frames

        wanted = sorted(set(symbols))
        if not wanted:
            return DepartedHistory(panel=_empty_panel(), without_history=())
        try:
            live = self._live_lib()
            retention = self._retention_lib()
            by_lib: dict[str, list[str]] = {"live": [], "retained": []}
            without: list[str] = []
            for sym in wanted:
                if live.has_symbol(sym):
                    by_lib["live"].append(sym)
                elif retention is not None and retention.has_symbol(sym):
                    by_lib["retained"].append(sym)
                else:
                    without.append(sym)
            frames: dict[str, Any] = {}
            bounds: dict[str, tuple[dt.date, dt.date] | None] = {}
            for key, lib in (("live", live), ("retained", retention)):
                if not by_lib[key]:
                    continue
                frames.update(
                    _load_arctic_frames(
                        lib,
                        by_lib[key],
                        lookback_days=lookback_days,
                        end=str(end),
                        columns=None,
                        label=f"ArcticPriceSource.load_departed({key})",
                    )
                )
                for sym in by_lib[key]:
                    if sym in frames:
                        continue
                    start_ts, end_ts = lib.get_description(sym).date_range
                    bounds[sym] = (
                        None
                        if pd.isna(start_ts) or pd.isna(end_ts)
                        else (pd.Timestamp(start_ts).date(), pd.Timestamp(end_ts).date())
                    )
        except MissingSourceError:
            raise
        except Exception as exc:
            raise MissingSourceError(
                f"ArcticDB departed-symbol read on bucket {self.bucket!r} failed for the window "
                f"ending {end}: {type(exc).__name__}: {exc}"
            ) from exc

        window_start = end - dt.timedelta(days=lookback_days)
        # A held symbol whose windowed read came back empty is either history
        # wholly outside the window (contributes nothing) or an outage — the
        # same two facts `load_panel` separates, decided the same way.
        missing, _outside = _classify_absent_symbols(
            sorted(bounds), bounds, window_start=window_start, window_end=end
        )
        if missing:
            raise MissingSourceError(
                f"departed symbol(s) {missing} are held on bucket {self.bucket!r} with history "
                f"overlapping the window ending {end}, yet the read returned nothing for them "
                "— a partial outage, which must fail the run rather than thin the panel."
            )
        panel = (
            normalize_panel(frames, end=end, lookback_days=lookback_days)
            if frames
            else _empty_panel()
        )
        return DepartedHistory(panel=panel, without_history=tuple(without))

    def _symbol_bounds(self, symbols: list[str]) -> dict[str, tuple[dt.date, dt.date] | None]:
        """Each symbol's FULL stored-history ``(first, last)`` date, or ``None``.

        ``None`` covers every way a symbol's history cannot be resolved: not
        present in the library (``has_symbol`` False), a description read that
        raises, or a description whose own ``date_range`` is unset (``NaT`` —
        an unsorted or non-timestamp-indexed symbol, per `SymbolDescription`'s
        own contract). Every one of those stays classified as a genuine
        `MissingSourceError` by `_classify_absent_symbols` — resolving to
        ``None`` here never itself decides "unlisted", only "unresolved".
        """
        import pandas as pd

        try:
            if self.library:
                from nousergon_lib.arcticdb import open_arctic

                lib = open_arctic(self.bucket, region=self.region).get_library(
                    self.library, create_if_missing=True
                )
            else:
                from nousergon_lib.arcticdb import open_universe_lib

                lib = open_universe_lib(self.bucket, region=self.region)
        except Exception:
            # The library itself could not be opened: every symbol is equally
            # unresolved (`None`), so `_classify_absent_symbols` classifies all
            # of them `missing` and the caller's `MissingSourceError` fires,
            # naming the bucket and window — this is not a second failure path,
            # only the input to the one that already exists.
            return dict.fromkeys(symbols, None)

        bounds: dict[str, tuple[dt.date, dt.date] | None] = {}
        for sym in symbols:
            try:
                if not lib.has_symbol(sym):
                    bounds[sym] = None
                    continue
                start_ts, end_ts = lib.get_description(sym).date_range
                if pd.isna(start_ts) or pd.isna(end_ts):
                    bounds[sym] = None
                    continue
                bounds[sym] = (pd.Timestamp(start_ts).date(), pd.Timestamp(end_ts).date())
            except Exception:
                # A `has_symbol`/`get_description` failure for THIS symbol is not
                # swallowed: it resolves to `None` (unresolved), and
                # `_classify_absent_symbols` treats every unresolved symbol as
                # `missing` — the caller's `MissingSourceError` still fires and
                # still names it. This is the strict default, never a silent skip.
                bounds[sym] = None
        return bounds

    def snapshot_id(self) -> str:
        """`arcticdb:{bucket}[:{library}]` — the library is versioned, this
        read is not.

        ArcticDB versions every write, so a restatement is recoverable; what
        this identifier pins is WHICH store — bucket AND library — was read.
        The `:{library}` suffix only appears with a dedicated-library
        override (`alpha-engine-config-I10457`): the production default
        (`library=None`) keeps the exact string every already-written
        manifest carries, so this is additive, not a provenance format
        change for existing runs. Pinning a per-symbol version tuple would be
        an artifact of its own, and is the follow-up named in the PR body
        rather than a value invented here.
        """
        if self.library:
            return f"arcticdb:{self.bucket}:{self.library}"
        return f"arcticdb:{self.bucket}"
