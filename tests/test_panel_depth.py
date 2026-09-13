"""The trailing panel must be deep enough for the catalogue that reads it.

Normative source: `alpha-engine-config-I10688`. The producer's window was
sized from the deepest DECLARED `window_trading_days` (252) rather than from
the deepest CHAINED one (313), so `residual_momentum_252d_skip21d_ratio` was
null for 903 of 903 tickers on every one of the 536 sessions the layer had
been compiled for — a column that looked computed and measured nothing, and
the reason the M arm that ranks on it could never fit on any date.

Coverage and depth are orthogonal and this file exists because only one of
them was measured. A panel can be 1.00 on coverage — every expected ticker
closed on the day — and still be too shallow for the catalogue, and the
artifact it writes is not thinner, it is wrong in a way no consumer can see.
"""

from __future__ import annotations

import pytest

from crucible.data import run_daily
from crucible.data.daily import (
    DEFAULT_LOOKBACK_DAYS,
    MIN_PANEL_TRADING_DAYS,
    PanelDepthError,
)
from crucible.features import min_panel_trading_days
from crucible.keys import data_panel_key, features_key
from crucible.manifest import read_manifest
from crucible.runner import run_job


class TestTheDefaultWindowCoversTheCatalogue:
    def test_the_declared_minimum_comes_from_the_producer_not_a_literal(self) -> None:
        assert MIN_PANEL_TRADING_DAYS == min_panel_trading_days()

    def test_the_default_lookback_covers_the_declared_minimum(self) -> None:
        """486 calendar days, not 400.

        Measured over the NYSE calendar krepis covers: the worst calendar span
        of 313 consecutive sessions is 460 days. 400 calendar days is 275
        sessions — which is what the production panel for 2026-09-11 actually
        carried, measured from `data/2026-09-11/panel.parquet`.
        """
        assert DEFAULT_LOOKBACK_DAYS >= 460
        assert DEFAULT_LOOKBACK_DAYS < 400 * 2


class TestAShallowPanelFailsTheDayRatherThanWritingANullColumn:
    def test_a_lookback_below_the_catalogues_depth_raises(
        self, store, source, frames, cycle_date
    ) -> None:
        with pytest.raises(PanelDepthError, match="session"):
            run_job(
                "data.daily",
                lambda c: run_daily(
                    c,
                    source=source,
                    expected_symbols=sorted(frames),
                    # 400 calendar days: the exact value that shipped, and the
                    # exact reason the layer carried a dead column.
                    lookback_days=400,
                ),
                store=store,
                trading_day=cycle_date,
            )

    def test_the_refusal_names_the_column_that_would_be_dead(
        self, store, source, frames, cycle_date
    ) -> None:
        with pytest.raises(PanelDepthError, match="residual_momentum_252d_skip21d_ratio"):
            run_job(
                "data.daily",
                lambda c: run_daily(
                    c, source=source, expected_symbols=sorted(frames), lookback_days=400
                ),
                store=store,
                trading_day=cycle_date,
            )

    def test_the_failure_writes_a_failed_manifest_and_no_feature_artifact(
        self, store, source, frames, cycle_date
    ) -> None:
        with pytest.raises(PanelDepthError):
            run_job(
                "data.daily",
                lambda c: run_daily(
                    c, source=source, expected_symbols=sorted(frames), lookback_days=400
                ),
                store=store,
                trading_day=cycle_date,
            )
        manifest = read_manifest(store, "data.daily", cycle_date.isoformat())
        assert manifest["status"] == "failed"
        assert "PanelDepthError" in manifest["reason"]

    def test_the_default_window_compiles_the_day_and_the_column_measures(
        self, store, source, frames, cycle_date
    ) -> None:
        """The green side, and the one that proves the refusal is not a wall."""
        import pandas as pd

        captured: dict[str, str] = {}

        def _compile(c):
            coverage = run_daily(c, source=source, expected_symbols=sorted(frames))
            captured["feature_version"] = coverage["feature_version"]
            return coverage

        run_job("data.daily", _compile, store=store, trading_day=cycle_date)
        assert store.exists(data_panel_key(cycle_date.isoformat()))
        key = features_key(captured["feature_version"], cycle_date.isoformat())
        import io

        features = pd.read_parquet(io.BytesIO(store.get_bytes(key)))
        assert features["residual_momentum_252d_skip21d_ratio"].notna().any(), (
            "the whole defect in one assertion: at the shipped window this column was "
            "null for every ticker on every compiled session"
        )

    def test_the_depth_is_measured_on_the_manifest_even_when_it_passes(
        self, store, source, frames, cycle_date
    ) -> None:
        """Principle 7: a component emitting nothing is unobserved, not healthy."""
        run_job(
            "data.daily",
            lambda c: run_daily(c, source=source, expected_symbols=sorted(frames)),
            store=store,
            trading_day=cycle_date,
        )
        manifest = read_manifest(store, "data.daily", cycle_date.isoformat())
        rows = [m for m in manifest["metrics"] if m["name"] == "panel_depth_trading_days"]
        assert len(rows) == 1
        assert rows[0]["status"] == "OK"
        assert rows[0]["value"] >= MIN_PANEL_TRADING_DAYS
        assert rows[0]["baseline"] == float(MIN_PANEL_TRADING_DAYS)
