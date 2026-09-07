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
branch. Survivorship (a delisted symbol dropped from *today's* universe
list before ever reaching this module) is unaffected and stays phase-5 work
— see the binding plan §10.4 and `alpha-engine-config-I9761`.
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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


def _frame_dates(frame: Any):
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
    """

    name = "arcticdb:universe"

    def __init__(self, bucket: str, *, region: str | None = None) -> None:
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

    def load_panel(
        self,
        *,
        end: dt.date,
        lookback_days: int,
        symbols: list[str] | None = None,
    ) -> pd.DataFrame:
        try:
            from nousergon_lib.arcticdb import load_universe_ohlcv
        except ImportError as exc:
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
        from nousergon_lib.arcticdb import open_universe_lib

        try:
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
        """`arcticdb:{bucket}` — the library is versioned, this read is not.

        ArcticDB versions every write, so a restatement is recoverable; what
        this identifier pins is WHICH store was read. Pinning a per-symbol
        version tuple would be an artifact of its own, and is the follow-up
        named in the PR body rather than a value invented here.
        """
        return f"arcticdb:{self.bucket}"
