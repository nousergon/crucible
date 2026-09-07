"""A dropped symbol is two different facts, and only one is an outage.

`alpha-engine-config-I10127`: four `data.heal` backfills for the window
ending 2024-08-07 raised `MissingSourceError` on six 2025-26 listings
that simply did not exist yet in that 2024 window — today's universe
applied to a historical session. `ArcticPriceSource.load_panel` used to
treat every dropped symbol as a partial outage, which made
`COVERAGE_FLOOR_RATIO` unreachable for any historical session.

These tests pin the fix at the `PriceSource` boundary: a symbol whose
stored history lies entirely outside the requested window is reported via
`panel.attrs["unlisted_in_window"]`, never raised; everything else that
drops a symbol — absent from the store entirely, an overlapping window
that still came back empty, or an unreadable description — stays exactly
the `MissingSourceError` it always was. No case here infers its answer
from an empty frame alone; each supplies real stored-history bounds.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from crucible.data.sources import ArcticPriceSource, FramePriceSource, MissingSourceError

END = dt.date(2026, 8, 28)
LOOKBACK = 30


def _frame(dates: list[dt.date]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open_raw": [1.0] * len(dates),
            "high_raw": [1.0] * len(dates),
            "low_raw": [1.0] * len(dates),
            "close_raw": [1.0] * len(dates),
            "volume_raw": [100.0] * len(dates),
        },
        index=pd.Index(dates, name="trading_day"),
    )


def _daily(start: dt.date, end: dt.date) -> list[dt.date]:
    out: list[dt.date] = []
    day = start
    while day <= end:
        out.append(day)
        day += dt.timedelta(days=1)
    return out


PRESENT = _frame(_daily(END - dt.timedelta(days=LOOKBACK), END))


class TestFramePriceSourceClassification:
    def test_a_symbol_whose_history_starts_after_the_window_is_unlisted_not_missing(
        self,
    ) -> None:
        newco = _frame(_daily(END + dt.timedelta(days=200), END + dt.timedelta(days=230)))
        source = FramePriceSource({"AAA": PRESENT, "NEWCO": newco})
        panel = source.load_panel(end=END, lookback_days=LOOKBACK, symbols=["AAA", "NEWCO"])
        assert panel.attrs["unlisted_in_window"] == ["NEWCO"]
        assert "NEWCO" not in set(panel["ticker"])

    def test_a_symbol_absent_from_the_source_entirely_still_raises(self) -> None:
        source = FramePriceSource({"AAA": PRESENT})
        with pytest.raises(MissingSourceError, match="absent from the source"):
            source.load_panel(end=END, lookback_days=LOOKBACK, symbols=["AAA", "GHOST"])

    def test_a_symbol_with_overlapping_bounds_but_no_rows_in_the_window_still_raises(
        self,
    ) -> None:
        # Real rows on both sides of the window, none inside it — the full
        # stored-history bounds overlap the window, so this stays an outage
        # rather than being read as "unlisted" from the empty windowed slice.
        gappy = pd.concat(
            [
                _frame([END - dt.timedelta(days=LOOKBACK + 60)]),
                _frame([END + dt.timedelta(days=60)]),
            ]
        )
        source = FramePriceSource({"AAA": PRESENT, "GAPPY": gappy})
        with pytest.raises(MissingSourceError, match="absent from the source"):
            source.load_panel(end=END, lookback_days=LOOKBACK, symbols=["AAA", "GAPPY"])

    def test_no_symbols_declared_unlisted_is_reported_as_an_empty_list_not_absent(self) -> None:
        source = FramePriceSource({"AAA": PRESENT})
        panel = source.load_panel(end=END, lookback_days=LOOKBACK, symbols=["AAA"])
        assert panel.attrs["unlisted_in_window"] == []


class _FakeDescription:
    def __init__(self, date_range: tuple[dt.date, dt.date]) -> None:
        self.date_range = date_range


class _FakeLib:
    """A `Library` stand-in whose bounds are supplied per symbol.

    ``bounds[sym] == "raise"`` simulates a `get_description` read failure;
    a symbol absent from ``bounds`` simulates `has_symbol` returning False.
    """

    def __init__(self, bounds: dict[str, object]) -> None:
        self._bounds = bounds

    def has_symbol(self, sym: str) -> bool:
        return sym in self._bounds

    def get_description(self, sym: str) -> _FakeDescription:
        value = self._bounds[sym]
        if value == "raise":
            raise RuntimeError("description read failed")
        return _FakeDescription(value)  # type: ignore[arg-type]


class TestArcticPriceSourceClassification:
    """`frames`/`cycle_date` come from `tests/conftest.py`: a real synthetic
    panel and a real NYSE session, so only the classification path under
    test is faked."""

    def test_a_symbol_whose_stored_history_is_entirely_outside_the_window_is_unlisted(
        self, frames, cycle_date, monkeypatch
    ) -> None:
        requested = sorted(frames)
        newco = requested[0]
        thinned = {t: f for t, f in frames.items() if t != newco}
        fake_lib = _FakeLib(
            {newco: (cycle_date + dt.timedelta(days=30), cycle_date + dt.timedelta(days=60))}
        )
        monkeypatch.setattr("nousergon_lib.arcticdb.load_universe_ohlcv", lambda *a, **k: thinned)
        monkeypatch.setattr(
            "nousergon_lib.arcticdb.open_universe_lib", lambda bucket, region=None: fake_lib
        )
        source = ArcticPriceSource("test-bucket")
        panel = source.load_panel(end=cycle_date, lookback_days=400, symbols=requested)
        assert panel.attrs["unlisted_in_window"] == [newco]
        assert newco not in set(panel["ticker"])

    def test_a_symbol_not_present_in_the_library_at_all_still_raises(
        self, frames, cycle_date, monkeypatch
    ) -> None:
        requested = sorted(frames)
        ghost = requested[0]
        thinned = {t: f for t, f in frames.items() if t != ghost}
        monkeypatch.setattr("nousergon_lib.arcticdb.load_universe_ohlcv", lambda *a, **k: thinned)
        monkeypatch.setattr(
            "nousergon_lib.arcticdb.open_universe_lib", lambda bucket, region=None: _FakeLib({})
        )
        source = ArcticPriceSource("test-bucket")
        with pytest.raises(MissingSourceError, match="dropped"):
            source.load_panel(end=cycle_date, lookback_days=400, symbols=requested)

    def test_a_symbol_with_overlapping_stored_history_but_an_empty_read_still_raises(
        self, frames, cycle_date, monkeypatch
    ) -> None:
        requested = sorted(frames)
        outage = requested[0]
        thinned = {t: f for t, f in frames.items() if t != outage}
        fake_lib = _FakeLib(
            {outage: (cycle_date - dt.timedelta(days=10), cycle_date + dt.timedelta(days=10))}
        )
        monkeypatch.setattr("nousergon_lib.arcticdb.load_universe_ohlcv", lambda *a, **k: thinned)
        monkeypatch.setattr(
            "nousergon_lib.arcticdb.open_universe_lib", lambda bucket, region=None: fake_lib
        )
        source = ArcticPriceSource("test-bucket")
        with pytest.raises(MissingSourceError, match="dropped"):
            source.load_panel(end=cycle_date, lookback_days=400, symbols=requested)

    def test_a_symbol_whose_description_read_raises_still_raises(
        self, frames, cycle_date, monkeypatch
    ) -> None:
        requested = sorted(frames)
        broken = requested[0]
        thinned = {t: f for t, f in frames.items() if t != broken}
        fake_lib = _FakeLib({broken: "raise"})
        monkeypatch.setattr("nousergon_lib.arcticdb.load_universe_ohlcv", lambda *a, **k: thinned)
        monkeypatch.setattr(
            "nousergon_lib.arcticdb.open_universe_lib", lambda bucket, region=None: fake_lib
        )
        source = ArcticPriceSource("test-bucket")
        with pytest.raises(MissingSourceError, match="dropped"):
            source.load_panel(end=cycle_date, lookback_days=400, symbols=requested)

    def test_a_symbol_whose_description_has_no_date_range_still_raises(
        self, frames, cycle_date, monkeypatch
    ) -> None:
        """`SymbolDescription.date_range` is `(NaT, NaT)` for a symbol with no
        timestamp index or an UNSORTED one (its own documented contract) —
        unresolvable, never a license to guess "unlisted" from it."""
        requested = sorted(frames)
        unsorted = requested[0]
        thinned = {t: f for t, f in frames.items() if t != unsorted}
        fake_lib = _FakeLib({unsorted: (pd.NaT, pd.NaT)})
        monkeypatch.setattr("nousergon_lib.arcticdb.load_universe_ohlcv", lambda *a, **k: thinned)
        monkeypatch.setattr(
            "nousergon_lib.arcticdb.open_universe_lib", lambda bucket, region=None: fake_lib
        )
        source = ArcticPriceSource("test-bucket")
        with pytest.raises(MissingSourceError, match="dropped"):
            source.load_panel(end=cycle_date, lookback_days=400, symbols=requested)

    def test_the_library_itself_failing_to_open_still_raises_for_every_absent_symbol(
        self, frames, cycle_date, monkeypatch
    ) -> None:
        requested = sorted(frames)
        newco = requested[0]
        thinned = {t: f for t, f in frames.items() if t != newco}

        def _boom(bucket, region=None):
            raise RuntimeError("library open failed")

        monkeypatch.setattr("nousergon_lib.arcticdb.load_universe_ohlcv", lambda *a, **k: thinned)
        monkeypatch.setattr("nousergon_lib.arcticdb.open_universe_lib", _boom)
        source = ArcticPriceSource("test-bucket")
        with pytest.raises(MissingSourceError, match="dropped"):
            source.load_panel(end=cycle_date, lookback_days=400, symbols=requested)
