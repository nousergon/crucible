"""A v1 promotion date with no NYSE session binds to the session before it.

`alpha-engine-config-I12020`: v1 promoted R on its Saturday weekly, so
`config/predictor_champion.json :: promoted_at` read 2026-10-10. The import
stamped that straight onto the arm's `registered_at`, and the 10-09 arc's
`migrate.history` stage failed `NonTradingDayKeyError` on it. Every v2 key
binds to a session (plan §4.12); a decision filed on a non-session is about the
last session before it.
"""

from __future__ import annotations

import pytest

from crucible.calendar import NonTradingDayKeyError, assert_trading_day
from crucible.migrate import SOURCES, MigrationSourceMissing, _date_of

R_SOURCE = next(s for s in SOURCES if s.slot == "r" and s.date_field)


def test_a_saturday_promotion_binds_to_friday() -> None:
    day, provenance = _date_of(
        {R_SOURCE.date_field: "2026-10-10T09:41:00Z"}, source=R_SOURCE, recipe=None
    )
    assert day == "2026-10-09"
    assert assert_trading_day(day, context="test")
    assert "2026-10-10 has no NYSE session" in provenance


def test_a_holiday_promotion_walks_past_the_weekend() -> None:
    # 2026-09-07 is Labor Day; the session before it is Friday 09-04.
    day, _ = _date_of({R_SOURCE.date_field: "2026-09-07"}, source=R_SOURCE, recipe=None)
    assert day == "2026-09-04"


def test_a_session_date_is_unchanged() -> None:
    day, provenance = _date_of({R_SOURCE.date_field: "2026-07-13"}, source=R_SOURCE, recipe=None)
    assert day == "2026-07-13"
    assert provenance == f"{R_SOURCE.key}:{R_SOURCE.date_field}"


def test_a_missing_date_still_refuses() -> None:
    with pytest.raises(MigrationSourceMissing):
        _date_of({}, source=R_SOURCE, recipe=None)


def test_the_unbound_date_is_what_the_key_check_refuses() -> None:
    with pytest.raises(NonTradingDayKeyError):
        assert_trading_day("2026-10-10", context="test")
