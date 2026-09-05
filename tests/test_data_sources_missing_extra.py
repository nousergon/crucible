"""`ArcticPriceSource` reports a missing `arcticdb` extra by name and never
substitutes another source. The branch carried `pragma: no cover - depends
on the install`; it depends on nothing but `sys.modules`, which a test owns.
"""

from __future__ import annotations

import datetime as dt
import sys

import pytest

from crucible.data.sources import ArcticPriceSource, MissingSourceError


def test_a_missing_arcticdb_extra_is_reported_by_name_not_substituted(monkeypatch) -> None:
    # A `None` entry in sys.modules makes `from nousergon_lib.arcticdb import ...`
    # raise ImportError — the documented way to simulate an uninstalled module.
    monkeypatch.setitem(sys.modules, "nousergon_lib.arcticdb", None)
    source = ArcticPriceSource("some-bucket")
    with pytest.raises(MissingSourceError, match=r"crucible\[arcticdb\]"):
        source.load_panel(end=dt.date(2026, 9, 4), lookback_days=5)
