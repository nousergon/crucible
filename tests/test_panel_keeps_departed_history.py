"""A name that leaves the universe keeps its price history in the panel.

`alpha-engine-config-I12287`: the 2026-10-09 v2 panel carried no row at all
for three names that had left the declared universe that day — not just no
row after they stopped trading, but none of their history either. Two
things dropped it, and each is pinned here:

* `run_daily` only ever asked the source for TODAY's declared universe, so a
  name absent from it was never requested;
* the production source only ever read the LIVE price library, and a name
  the data layer prunes on delisting has its whole history MOVED to a
  retention library, so even a request for it would have found nothing.

What these tests assert is the survivorship property itself: a name that
delists mid-window has a close on every session up to its last trading day
and on none after, it never enters the coverage denominator or the feature
cross-section, and a departed name with no stored history anywhere is named
in the coverage record rather than silently missing.
"""

from __future__ import annotations

import datetime as dt
import json

import pandas as pd
import pytest
from conftest import synthetic_frames

from crucible.data import FramePriceSource, MissingSourceError, run_daily
from crucible.data.point_in_time import UnavailablePointInTimeSource
from crucible.data.sources import ArcticPriceSource
from crucible.features import read_features
from crucible.keys import coverage_key, data_panel_key, declared_universe_key, features_key
from crucible.runner import run_job

NO_FUNDAMENTALS = "synthetic fixture market carries no fundamentals"


def _read_panel(store, day: dt.date) -> pd.DataFrame:
    import io

    return pd.read_parquet(io.BytesIO(store.get_bytes(data_panel_key(day.isoformat()))))


def _panel_sessions(panel: pd.DataFrame) -> list[dt.date]:
    """Every session the compiled panel spans, read off a surviving name."""
    return sorted(panel.loc[panel["ticker"] == "T000", "trading_day"])


def _sessions(frames: dict[str, pd.DataFrame]) -> list[dt.date]:
    return sorted(next(iter(frames.values())).index)


def _delisting(cycle_date: dt.date, *, last_day: dt.date, name: str) -> pd.DataFrame:
    """A synthetic name whose stored history STOPS at ``last_day``."""
    frame = synthetic_frames(end=cycle_date, names=[name], seed=7)[name]
    return frame[frame.index <= last_day]


def _declare(store, day: dt.date, symbols: list[str]) -> None:
    """A recorded declared universe, in the exact shape `DeclaredUniverse.record` writes."""
    store.put_bytes(
        declared_universe_key(day.isoformat()),
        json.dumps(
            {
                "schema_version": "declared_universe.v1",
                "trading_day": day.isoformat(),
                "source_uri": "argv:--symbols",
                "source_sha256": "0" * 64,
                "origin": "argument",
                "count": len(symbols),
                "symbols": sorted(symbols),
            }
        ).encode("utf-8"),
    )


def _compile(store, source, *, cycle_date, expected):
    return run_job(
        "data.daily",
        lambda c: run_daily(
            c,
            point_in_time=UnavailablePointInTimeSource(reason=NO_FUNDAMENTALS),
            source=source,
            expected_symbols=expected,
        ),
        store=store,
        trading_day=cycle_date,
    )


class TestADelistedNameKeepsItsHistory:
    """The shape the data layer produces on a delisting: the name left the
    universe, and its stored history moved out of the live library into the
    retention store (`FramePriceSource(retained=...)`)."""

    def test_it_has_a_close_on_every_session_before_it_delisted_and_none_after(
        self, store, frames, benchmark_frames, cycle_date
    ) -> None:
        days = _sessions(frames)
        last_day = days[len(days) // 2]
        delisted = _delisting(cycle_date, last_day=last_day, name="GONE")
        source = FramePriceSource({**frames, **benchmark_frames}, retained={"GONE": delisted})

        _compile(store, source, cycle_date=cycle_date, expected=sorted(frames))

        panel = _read_panel(store, cycle_date)
        gone_days = sorted(panel.loc[panel["ticker"] == "GONE", "trading_day"])
        assert gone_days == [d for d in _panel_sessions(panel) if d <= last_day], (
            "a delisted name keeps a close on every session of the panel up to its last trading day"
        )
        assert not [d for d in gone_days if d > last_day], "and has none after it"
        assert panel.loc[panel["ticker"] == "GONE", "close_raw"].notna().all()

    def test_it_never_enters_the_coverage_denominator_or_the_feature_cross_section(
        self, store, tmp_path, frames, benchmark_frames, cycle_date
    ) -> None:
        from crucible.store import LocalStore

        days = _sessions(frames)
        delisted = _delisting(cycle_date, last_day=days[-5], name="GONE")
        with_departed = FramePriceSource(
            {**frames, **benchmark_frames}, retained={"GONE": delisted}
        )
        ctx = _compile(store, with_departed, cycle_date=cycle_date, expected=sorted(frames))

        coverage = json.loads(store.get_bytes(coverage_key(cycle_date.isoformat())))
        assert coverage["expected_tickers"] == sorted(frames)
        assert coverage["coverage_ratio"] == 1.0
        assert coverage["departed_tickers"] == ["GONE"]
        assert coverage["departed_without_history"] == []
        metric = next(m for m in ctx.metrics if m["name"] == "departed_symbols_in_panel")
        assert metric["value"] == 1.0
        assert "GONE" in metric["status_reason"]

        # The feature layer is the tradeable cross-section M scores and U
        # selects from: a departed name's history must not change one byte of
        # it. Compiled again without the departed name, the features match.
        bare_store = LocalStore(tmp_path / "bare")
        _compile(
            bare_store,
            FramePriceSource({**frames, **benchmark_frames}),
            cycle_date=cycle_date,
            expected=sorted(frames),
        )
        key = features_key(coverage["feature_version"], cycle_date.isoformat())
        with_gone = read_features(store.get_bytes(key))
        without_gone = read_features(bare_store.get_bytes(key))
        assert "GONE" not in set(with_gone["ticker"])
        pd.testing.assert_frame_equal(with_gone, without_gone)

    def test_history_that_ended_before_the_window_adds_nothing(
        self, store, frames, benchmark_frames, cycle_date
    ) -> None:
        long_gone = synthetic_frames(end=cycle_date - dt.timedelta(days=900), names=["OLD"])
        source = FramePriceSource({**frames, **benchmark_frames}, retained=long_gone)

        _compile(store, source, cycle_date=cycle_date, expected=sorted(frames))

        assert "OLD" not in set(_read_panel(store, cycle_date)["ticker"])
        coverage = json.loads(store.get_bytes(coverage_key(cycle_date.isoformat())))
        assert coverage["departed_tickers"] == []


class TestANameThatLeftTheDeclaredUniverseKeepsItsHistory:
    """The window between leaving the universe and being pruned: the name is
    no longer in today's declared universe, but its history still sits in
    the LIVE library. The 2026-10-09 compile was exactly this case for all
    three names it dropped."""

    def test_a_name_declared_earlier_in_the_window_is_carried_up_to_its_last_day(
        self, store, frames, benchmark_frames, cycle_date
    ) -> None:
        days = _sessions(frames)
        last_day = days[-10]
        left = _delisting(cycle_date, last_day=last_day, name="LEFT")
        _declare(store, days[-20], [*sorted(frames), "LEFT"])
        source = FramePriceSource({**frames, **benchmark_frames, "LEFT": left})

        _compile(store, source, cycle_date=cycle_date, expected=sorted(frames))

        panel = _read_panel(store, cycle_date)
        assert sorted(panel.loc[panel["ticker"] == "LEFT", "trading_day"]) == [
            d for d in _panel_sessions(panel) if d <= last_day
        ]

    def test_a_declared_name_with_no_stored_history_anywhere_is_named_not_dropped(
        self, store, frames, benchmark_frames, cycle_date
    ) -> None:
        """Issue item 4's case: history that was never written. The compile
        cannot invent it, and must not lose the fact that it is missing — the
        S grade reads this list to tell a name with genuinely no data from a
        panel that dropped one."""
        days = _sessions(frames)
        _declare(store, days[-20], [*sorted(frames), "NEVERWRITTEN"])

        ctx = _compile(
            store,
            FramePriceSource({**frames, **benchmark_frames}),
            cycle_date=cycle_date,
            expected=sorted(frames),
        )

        coverage = json.loads(store.get_bytes(coverage_key(cycle_date.isoformat())))
        assert coverage["departed_without_history"] == ["NEVERWRITTEN"]
        assert coverage["departed_tickers"] == []
        metric = next(m for m in ctx.metrics if m["name"] == "departed_symbols_without_history")
        assert metric["value"] == 1.0
        assert "NEVERWRITTEN" in metric["status_reason"]
        assert "NEVERWRITTEN" not in set(_read_panel(store, cycle_date)["ticker"])

    def test_a_declared_universe_recorded_after_the_day_is_not_read(
        self, store, frames, benchmark_frames, cycle_date
    ) -> None:
        """A record for a LATER session is not membership this panel could
        have known — reading it would make a recompile depend on when it ran."""
        _declare(store, cycle_date + dt.timedelta(days=3), [*sorted(frames), "LATER"])

        _compile(
            store,
            FramePriceSource({**frames, **benchmark_frames}),
            cycle_date=cycle_date,
            expected=sorted(frames),
        )

        coverage = json.loads(store.get_bytes(coverage_key(cycle_date.isoformat())))
        assert coverage["departed_without_history"] == []


class _Description:
    def __init__(self, first: dt.date, last: dt.date) -> None:
        self.date_range = (pd.Timestamp(first), pd.Timestamp(last))


class _Lib:
    """A `Library` stand-in holding real frames, so bounds come from data."""

    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self.frames = frames

    def has_symbol(self, sym: str) -> bool:
        return sym in self.frames

    def list_symbols(self) -> list[str]:
        return sorted(self.frames)

    def get_description(self, sym: str) -> _Description:
        index = self.frames[sym].index
        return _Description(min(index), max(index))

    def get_description_batch(self, symbols: list[str]) -> list[_Description]:
        return [self.get_description(s) for s in symbols]


class TestArcticPriceSourceReadsTheRetentionLibrary:
    """`ArcticPriceSource` resolves a departed name from the live library
    first and the retention library second — the production half of the
    drop, faked only at the library boundary."""

    @pytest.fixture
    def libs(self, frames, cycle_date, monkeypatch):
        days = _sessions(frames)
        live = _Lib({"LEFT": _delisting(cycle_date, last_day=days[-10], name="LEFT")})
        retained = _Lib(
            {
                "GONE": _delisting(cycle_date, last_day=days[-30], name="GONE"),
                "OLD": synthetic_frames(end=cycle_date - dt.timedelta(days=900), names=["OLD"])[
                    "OLD"
                ],
            }
        )

        class _Arctic:
            def get_library(self, name, create_if_missing=False):
                assert name == "delisted_history"
                assert not create_if_missing, "a read path never creates a library"
                return retained

        def _load(lib, symbols, *, lookback_days, end, columns, label):
            stop = dt.date.fromisoformat(end)
            start = stop - dt.timedelta(days=lookback_days)
            out = {}
            for sym in symbols:
                frame = lib.frames[sym]
                window = frame[[start < d <= stop for d in frame.index]]
                if len(window):
                    out[sym] = window
            return out

        monkeypatch.setattr("nousergon_lib.arcticdb.open_universe_lib", lambda *a, **k: live)
        monkeypatch.setattr("nousergon_lib.arcticdb.open_arctic", lambda *a, **k: _Arctic())
        monkeypatch.setattr("nousergon_lib.arcticdb._load_arctic_frames", _load)
        return days

    def test_retained_symbols_are_those_whose_history_reaches_into_the_window(
        self, libs, cycle_date
    ) -> None:
        source = ArcticPriceSource("test-bucket")
        assert source.retained_symbols(end=cycle_date, lookback_days=400) == ["GONE"]

    def test_a_departed_name_is_read_from_whichever_library_holds_it(
        self, libs, cycle_date
    ) -> None:
        days = libs
        source = ArcticPriceSource("test-bucket")
        departed = source.load_departed(
            end=cycle_date, lookback_days=400, symbols=["GONE", "LEFT", "NEVERWRITTEN"]
        )
        in_window = [d for d in days if d > cycle_date - dt.timedelta(days=400)]
        by_name = departed.panel.groupby("ticker")["trading_day"].apply(sorted).to_dict()
        assert by_name["LEFT"] == [d for d in in_window if d <= days[-10]]
        assert by_name["GONE"] == [d for d in in_window if d <= days[-30]]
        assert departed.without_history == ("NEVERWRITTEN",)

    def test_a_departed_name_whose_window_read_comes_back_empty_is_an_outage(
        self, libs, cycle_date, monkeypatch
    ) -> None:
        """Stored history overlapping the window that the read still dropped
        is a partial outage, exactly as it is for a universe name."""
        monkeypatch.setattr("nousergon_lib.arcticdb._load_arctic_frames", lambda *a, **k: {})
        source = ArcticPriceSource("test-bucket")
        with pytest.raises(MissingSourceError, match="GONE"):
            source.load_departed(end=cycle_date, lookback_days=400, symbols=["GONE"])
