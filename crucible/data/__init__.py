"""The data layer: compile, refresh, heal — and refuse rather than fill.

Normative source: plan §4.3 and §9.7.

Three jobs and one adapter:

* :func:`~crucible.data.daily.run_daily` — one trading day's panel, its
  coverage record and the materialized feature layer;
* :func:`~crucible.data.weekly.run_weekly` — the week's last session, plus
  the refusal to run on a gap it did not heal;
* :func:`~crucible.data.heal.run_heal` — an idempotent in-region repair of a
  named range;
* :mod:`crucible.data.sources` — the `PriceSource` adapter, of which
  ArcticDB is one implementation and not the shape.

The rule under all four: **a missing source is `status: failed`, never a
zero-fill.** It is not stated as a policy anywhere in this package, because
there is no code path that could do otherwise.
"""

from __future__ import annotations

from crucible.data.daily import COVERAGE_FLOOR_RATIO, CoverageError, run_daily
from crucible.data.heal import LAPTOP_SESSION_ALLOWANCE, NotInRegionError, run_heal
from crucible.data.sources import (
    PANEL_COLUMNS,
    ArcticPriceSource,
    FramePriceSource,
    MissingSourceError,
    PriceSource,
)
from crucible.data.weekly import DataGapError, run_weekly, week_sessions

__all__ = [
    "COVERAGE_FLOOR_RATIO",
    "LAPTOP_SESSION_ALLOWANCE",
    "PANEL_COLUMNS",
    "ArcticPriceSource",
    "CoverageError",
    "DataGapError",
    "FramePriceSource",
    "MissingSourceError",
    "NotInRegionError",
    "PriceSource",
    "run_daily",
    "run_heal",
    "run_weekly",
    "week_sessions",
]
