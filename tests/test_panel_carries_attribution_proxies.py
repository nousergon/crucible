"""The compiled panel carries a row for every factor-attribution proxy
(`alpha-engine-config-I10683`).

`crucible.slots.strategy.grade` calls `crucible.attribution.compute_factor_
attribution` (through `_attribute_book`), which raises `MissingArtifactError`
the moment a proxy the attribution spec names
(`strategy/slots/attribution.yaml`) has no row in the compiled panel.
`declared_benchmark_symbols` alone only ever carried S's own benchmark
(`SPY`) into the panel — the other proxies a factor spec names (`IWM`, the
sector ETFs) are not any slot's `SlotSpec.benchmark`, so `run_daily` never
fetched them and every real S-slot grading run raised. `run_daily` now
derives a SECOND set of required symbols from
`crucible.attribution.AttributionFactorParams.factors` (via
`crucible.slots.attribution_factor_symbols`) and fetches every one through
the same `PriceSource`, exactly like the benchmark-row precedent
(`alpha-engine-config-I10635`, `crucible-PR244`).
"""

from __future__ import annotations

import io

import pandas as pd
import pytest
import yaml

from crucible.data import FramePriceSource, MissingSourceError, run_daily
from crucible.keys import data_panel_key, strategy_slot_key
from crucible.manifest import read_manifest
from crucible.runner import run_job
from crucible.slots import attribution_factor_symbols
from tests.conftest import synthetic_frames

# A minimal factor spec naming three proxies beyond S's own SPY benchmark
# already carries: one raw (SPY, doubling as both the beta factor's proxy
# and S's own benchmark — same symbol, two declared reasons for it) and two
# non-benchmark proxies (`IWM` for size, `XLK` for the tech sector) that
# `declared_benchmark_symbols` alone never named.
_ATTRIBUTION_SPEC = {
    "attribution": {
        "benchmark_proxy": "SPY",
        "shrinkage": "ledoit_wolf",
        "factors": {
            "market": {"category": "beta", "proxy": "SPY"},
            "size_factor": {"category": "size", "proxy": "IWM"},
            "sector_tech": {"category": "sector", "proxy": "XLK"},
        },
    }
}

_NON_BENCHMARK_PROXIES = ("IWM", "XLK")


def _write_attribution_spec(store) -> None:
    store.put_bytes(
        strategy_slot_key("attribution"), yaml.safe_dump(_ATTRIBUTION_SPEC).encode("utf-8")
    )


class TestAttributionFactorSymbols:
    def test_no_spec_declared_yields_no_extra_symbols(self, store) -> None:
        """Absence of an attribution spec is a legitimate state (an
        environment that has not adopted factor attribution yet) — never a
        MissingArtifactError-shaped refusal at compile time for a run that
        never declared one."""
        assert attribution_factor_symbols(store=store) == frozenset()

    def test_symbols_are_derived_from_the_declared_spec_never_hand_kept(self, store) -> None:
        _write_attribution_spec(store)
        assert attribution_factor_symbols(store=store) == {"SPY", "IWM", "XLK"}

    def test_no_spec_declared_in_a_checkout_directory_yields_no_extra_symbols(
        self, tmp_path
    ) -> None:
        """Mirrors the store-absent case above for the `strategy_dir` branch —
        a checkout with no `slots/attribution.yaml` at all is the same
        legitimate absence, not a refusal."""
        strategy_dir = tmp_path / "strategy"
        (strategy_dir / "slots").mkdir(parents=True)
        assert attribution_factor_symbols(strategy_dir=strategy_dir) == frozenset()

    def test_a_checkout_directory_spec_is_derived_too(self, tmp_path) -> None:
        strategy_dir = tmp_path / "strategy"
        (strategy_dir / "slots").mkdir(parents=True)
        (strategy_dir / "slots" / "attribution.yaml").write_text(
            yaml.safe_dump(_ATTRIBUTION_SPEC), encoding="utf-8"
        )
        assert attribution_factor_symbols(strategy_dir=strategy_dir) == {"SPY", "IWM", "XLK"}

    def test_a_spread_factors_short_proxy_is_included_too(self, store) -> None:
        """`alpha-engine-config-I10592`: a spread factor's short leg is read
        by `crucible.attribution.factor_return_series` exactly like the long
        leg, so both must have a panel row."""
        spec = {
            "attribution": {
                "benchmark_proxy": "SPY",
                "factors": {
                    "size_factor": {
                        "category": "size",
                        "proxy": "IWM",
                        "short_proxy": "SPY",
                    },
                    "sector_tech": {"category": "sector", "proxy": "XLK"},
                    "market": {"category": "beta", "proxy": "SPY"},
                },
            }
        }
        store.put_bytes(strategy_slot_key("attribution"), yaml.safe_dump(spec).encode("utf-8"))
        assert attribution_factor_symbols(store=store) == {"SPY", "IWM", "XLK"}

    def test_neither_a_store_nor_a_strategy_dir_raises(self) -> None:
        with pytest.raises(ValueError, match="needs either a store or a strategy_dir"):
            attribution_factor_symbols()


class TestPanelCarriesEveryAttributionProxy:
    def test_a_compiled_panel_contains_a_row_for_every_attribution_proxy(
        self, store, frames, benchmark_frames, cycle_date
    ) -> None:
        _write_attribution_spec(store)
        proxy_frames = synthetic_frames(
            end=cycle_date, names=list(_NON_BENCHMARK_PROXIES), seed=20260913
        )
        source = FramePriceSource(
            {**frames, **benchmark_frames, **proxy_frames},
            snapshot="frames:attribution-proxies",
        )
        run_job(
            "data.daily",
            lambda c: run_daily(c, source=source, expected_symbols=sorted(frames)),
            store=store,
            trading_day=cycle_date,
        )
        manifest = read_manifest(store, "data.daily", cycle_date.isoformat())
        assert manifest["status"] == "ok", manifest.get("reason")

        panel_bytes = store.get_bytes(data_panel_key(cycle_date.isoformat()))
        panel = pd.read_parquet(io.BytesIO(panel_bytes))
        tickers_on_day = set(panel.loc[panel["trading_day"] == cycle_date, "ticker"])

        for symbol in ("SPY", *_NON_BENCHMARK_PROXIES):
            assert symbol in tickers_on_day, (
                f"the compiled panel for {cycle_date} carries no row for {symbol!r}, "
                "a ticker the attribution spec proxies a factor with — "
                "`crucible.attribution.compute_factor_attribution` would raise "
                "MissingArtifactError against this panel"
            )

    def test_a_proxy_absent_from_the_price_source_fails_loud(
        self, store, frames, benchmark_frames, cycle_date
    ) -> None:
        """The mutation case: a factor proxy the spec names but the source
        cannot serve must FAIL the compile, never silently omit the row —
        no zero-fill, no partial panel written."""
        _write_attribution_spec(store)
        source_missing_proxies = FramePriceSource(
            {**frames, **benchmark_frames}, snapshot="frames:missing-attribution-proxy"
        )
        with pytest.raises(MissingSourceError, match="IWM"):
            run_job(
                "data.daily",
                lambda c: run_daily(
                    c, source=source_missing_proxies, expected_symbols=sorted(frames)
                ),
                store=store,
                trading_day=cycle_date,
            )
        assert not store.exists(data_panel_key(cycle_date.isoformat())), (
            "a compile that cannot serve a declared attribution proxy must leave no "
            "panel behind, exactly like any other refused day"
        )

    def test_no_attribution_spec_declared_does_not_break_an_ordinary_compile(
        self, store, frames, source, cycle_date
    ) -> None:
        """An environment carrying no attribution spec at all (this repo's
        own other fixtures, chiefly) must keep compiling exactly as before —
        `attribution_factor_symbols` returning empty is not itself a
        refusal."""
        run_job(
            "data.daily",
            lambda c: run_daily(c, source=source, expected_symbols=sorted(frames)),
            store=store,
            trading_day=cycle_date,
        )
        manifest = read_manifest(store, "data.daily", cycle_date.isoformat())
        assert manifest["status"] == "ok", manifest.get("reason")

    def test_an_attribution_proxy_already_in_the_declared_universe_is_not_refetched(
        self, store, frames, benchmark_frames, cycle_date
    ) -> None:
        _write_attribution_spec(store)
        proxy_frames = synthetic_frames(
            end=cycle_date, names=list(_NON_BENCHMARK_PROXIES), seed=20260913
        )
        merged = {**frames, **benchmark_frames, **proxy_frames}
        expected = [*sorted(frames), "IWM"]

        calls: list[list[str] | None] = []
        real_source = FramePriceSource(merged, snapshot="frames:proxy-already-declared")
        original_load_panel = real_source.load_panel

        def _tracking_load_panel(*, end, lookback_days, symbols=None):
            calls.append(list(symbols) if symbols is not None else None)
            return original_load_panel(end=end, lookback_days=lookback_days, symbols=symbols)

        real_source.load_panel = _tracking_load_panel  # type: ignore[method-assign]

        run_job(
            "data.daily",
            lambda c: run_daily(c, source=real_source, expected_symbols=expected),
            store=store,
            trading_day=cycle_date,
        )

        # One call for the declared universe (which already carries IWM), one
        # for the remaining extra symbols (SPY, XLK) — never a third call for
        # IWM specifically, since it was already part of the primary fetch.
        assert len(calls) == 2, calls
        assert "IWM" not in (calls[1] or [])
