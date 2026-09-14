"""The compiled panel carries every slot's declared benchmark (`alpha-engine-config-I10635`).

S is the one slot graded against a market index —
`crucible.slots.SLOTS["s"].benchmark == "SPY"` — and
`crucible.slots.strategy.grade_arm` refuses a book whose benchmark has no
panel row. `run_daily` derives the set of required benchmark symbols from
`crucible.slots.SLOTS` (never a second literal) and fetches each one through
the same `PriceSource` the declared universe is read from, so a slot that
starts declaring a new index benchmark tomorrow makes this contract test red
at compile time rather than discovered as a grading refusal on a box.
"""

from __future__ import annotations

import io

import pandas as pd
import pytest

from crucible.data import FramePriceSource, MissingSourceError, run_daily
from crucible.data.point_in_time import UnavailablePointInTimeSource
from crucible.keys import data_panel_key
from crucible.manifest import read_manifest
from crucible.runner import run_job
from crucible.slots import SLOTS, declared_benchmark_symbols


class TestPanelCarriesEveryDeclaredBenchmark:
    def test_every_non_population_slot_benchmark_is_a_real_symbol(self) -> None:
        """The contract this whole module tests only means something if at
        least one slot actually declares an index benchmark — otherwise
        every assertion below passes vacuously."""
        assert declared_benchmark_symbols() == {"SPY"}, (
            "S is the one slot declared against a market index today; if this "
            "changes, the fixtures and assertions below must change with it, "
            "not silently stop covering anything"
        )

    def test_declared_benchmark_symbols_is_derived_never_hand_kept(self) -> None:
        """A hand-kept second list of benchmark symbols is exactly the
        units-mismatch shape this repository exists to catch
        (`avg_volume_20d`, 901/903 tickers silently failing the liquidity
        gate for months) — this pins that the set is read off `SLOTS`
        itself, not restated."""
        expected = {spec.benchmark for spec in SLOTS.values() if spec.benchmark != "population"}
        assert declared_benchmark_symbols() == expected

    def test_a_compiled_panel_contains_a_row_for_every_declared_benchmark(
        self, store, source, frames, cycle_date
    ) -> None:
        run_job(
            "data.daily",
            lambda c: run_daily(
                c,
                point_in_time=UnavailablePointInTimeSource(
                    reason="synthetic fixture market carries no fundamentals"
                ),
                source=source,
                expected_symbols=sorted(frames),
            ),
            store=store,
            trading_day=cycle_date,
        )
        manifest = read_manifest(store, "data.daily", cycle_date.isoformat())
        assert manifest["status"] == "ok", manifest.get("reason")

        panel_bytes = store.get_bytes(data_panel_key(cycle_date.isoformat()))
        panel = pd.read_parquet(io.BytesIO(panel_bytes))
        tickers_on_day = set(panel.loc[panel["trading_day"] == cycle_date, "ticker"])

        for symbol in declared_benchmark_symbols():
            assert symbol in tickers_on_day, (
                f"the compiled panel for {cycle_date} carries no row for {symbol!r}, "
                "a symbol a slot declares as its own benchmark — grade_arm would "
                "refuse every book for that slot against this panel"
            )

    def test_the_panels_column_set_is_unchanged(self, store, source, frames, cycle_date) -> None:
        """Carrying the benchmark's OHLCV is adding a ROW, never a column —
        the panel contract (`crucible.data.sources.PANEL_COLUMNS`) is
        untouched by this fix."""
        from crucible.data.sources import PANEL_COLUMNS

        run_job(
            "data.daily",
            lambda c: run_daily(
                c,
                point_in_time=UnavailablePointInTimeSource(
                    reason="synthetic fixture market carries no fundamentals"
                ),
                source=source,
                expected_symbols=sorted(frames),
            ),
            store=store,
            trading_day=cycle_date,
        )
        panel_bytes = store.get_bytes(data_panel_key(cycle_date.isoformat()))
        panel = pd.read_parquet(io.BytesIO(panel_bytes))
        assert tuple(panel.columns) == PANEL_COLUMNS

    def test_a_source_that_cannot_serve_the_benchmark_fails_the_compile(
        self, store, frames, cycle_date
    ) -> None:
        """The refusal this fix exists to make loud at compile time: a
        source missing the benchmark's data must not write a panel that
        omits it silently — that is exactly the gap this issue measured on
        `s3://<v2 store>/crucible/data/2026-09-11/panel.parquet`."""
        no_benchmark_source = FramePriceSource(frames, snapshot="frames:no-benchmark")
        with pytest.raises(MissingSourceError, match="SPY"):
            run_job(
                "data.daily",
                lambda c: run_daily(
                    c,
                    point_in_time=UnavailablePointInTimeSource(
                        reason="synthetic fixture market carries no fundamentals"
                    ),
                    source=no_benchmark_source,
                    expected_symbols=sorted(frames),
                ),
                store=store,
                trading_day=cycle_date,
            )
        assert not store.exists(data_panel_key(cycle_date.isoformat())), (
            "a compile that cannot serve a declared benchmark must leave no panel "
            "behind, exactly like any other refused day"
        )

    def test_a_benchmark_symbol_already_in_the_declared_universe_is_not_refetched(
        self, store, benchmark_frames, frames, cycle_date
    ) -> None:
        """When the declared universe already names the benchmark symbol,
        `run_daily` must not issue a second `load_panel` call for it — the
        source already served it once, in the primary fetch."""
        merged = {**frames, **benchmark_frames}
        benchmark_symbol = next(iter(benchmark_frames))
        expected = [*sorted(frames), benchmark_symbol]

        calls: list[list[str] | None] = []
        real_source = FramePriceSource(merged, snapshot="frames:already-declared")
        original_load_panel = real_source.load_panel

        def _tracking_load_panel(*, end, lookback_days, symbols=None):
            calls.append(list(symbols) if symbols is not None else None)
            return original_load_panel(end=end, lookback_days=lookback_days, symbols=symbols)

        real_source.load_panel = _tracking_load_panel  # type: ignore[method-assign]

        run_job(
            "data.daily",
            lambda c: run_daily(
                c,
                point_in_time=UnavailablePointInTimeSource(
                    reason="synthetic fixture market carries no fundamentals"
                ),
                source=real_source,
                expected_symbols=expected,
            ),
            store=store,
            trading_day=cycle_date,
        )

        assert len(calls) == 1, (
            "the benchmark symbol was already part of the declared universe's own "
            "fetch; a second call for the same symbol is redundant and, against a "
            "real source, a second round trip nothing needs"
        )
