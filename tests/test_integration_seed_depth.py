"""The integration tier's seed depth is derived, never a literal.

Normative source: `alpha-engine-config-I10701`, measured against run
34797033396: `tests/integration/conftest.py::integration_arctic_symbols`
seeded a hand-counted 300 trading sessions, and `crucible-PR266` sized
`crucible.data.daily`'s `PanelDepthError` guard from
`crucible.features.min_panel_trading_days()` (313 sessions today) — so the
literal fell behind the producer's own guard the moment that PR landed, and
`test_data_daily`/`test_data_weekly`/`test_data_heal` failed with "the
trailing panel ... carries 300 session(s), below the 313 the feature
catalogue's deepest column needs".

This module imports `tests/integration/conftest.py` directly (never a
fixture) — it is a plain function, no `pytest` fixture machinery and no
`require_env` call at import time, so it is importable and assertable from
the blocking suite without any of the dedicated S3/ArcticDB environment
`tests/integration/` needs to actually run. That is the property this test
protects: the derivation is checked on every PR, even though the fixture
that consumes it only runs nightly.
"""

from __future__ import annotations

from crucible.data.daily import MIN_PANEL_TRADING_DAYS
from crucible.features import min_panel_trading_days
from tests.integration.conftest import (
    _SEED_DEPTH_MARGIN_TRADING_DAYS,
    integration_seed_trading_days,
)


class TestIntegrationSeedDepth:
    def test_derived_from_the_producer_function_not_a_literal(self) -> None:
        assert (
            integration_seed_trading_days()
            == min_panel_trading_days() + _SEED_DEPTH_MARGIN_TRADING_DAYS
        )

    def test_clears_the_producer_guard_with_margin_to_spare(self) -> None:
        """Strictly above `MIN_PANEL_TRADING_DAYS`, the exact threshold
        `crucible.data.daily.run_daily` compares the trailing panel's session
        count against (`PanelDepthError` fires when the panel is BELOW it) —
        equal would still be one flaky session away from the failure this
        test exists to prevent recurring.
        """
        assert integration_seed_trading_days() > MIN_PANEL_TRADING_DAYS

    def test_moves_on_its_own_if_the_catalogue_deepens(self) -> None:
        """A future catalogue column deepening the minimum moves this seed
        with it, with no edit here — the same property `MIN_PANEL_TRADING_
        DAYS` itself already has over a literal 313."""
        assert (
            integration_seed_trading_days() - min_panel_trading_days()
            == _SEED_DEPTH_MARGIN_TRADING_DAYS
        )
