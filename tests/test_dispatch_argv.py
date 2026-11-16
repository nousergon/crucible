"""Unit tests for parsing the argv recorded on a dispatch."""

from __future__ import annotations

import datetime as dt

import pytest

from crucible import dispatch_argv


def _fixed_trading_day(value: dt.datetime | None = None) -> dt.date:
    return dt.date(2026, 11, 13)


def test_flag_value_reads_separate_and_equals_forms() -> None:
    assert dispatch_argv.flag_value("--date 2026-11-13 --mode replay", "--date") == "2026-11-13"
    assert dispatch_argv.flag_value("--label='two words' --mode replay", "--label") == "two words"


def test_flag_value_handles_empty_missing_and_non_string_argv() -> None:
    assert dispatch_argv.flag_value("", "--date") is None
    assert dispatch_argv.flag_value("--mode replay --date", "--date") is None
    assert dispatch_argv.flag_value(["--date", "2026-11-13"], "--date") is None


def test_flag_value_rejects_malformed_shell_syntax() -> None:
    # Mutating the shlex failure path to return a value must make this assertion fail.
    assert dispatch_argv.flag_value("--date '2026-11-13", "--date") is None


def test_dispatch_target_trading_day_prefers_date_for_ordinary_jobs() -> None:
    actual = dispatch_argv.dispatch_target_trading_day(
        "fault.probe",
        "--to 2026-11-12 --date 2026-11-13",
        dt.datetime(2026, 11, 16, 12, tzinfo=dt.UTC),
    )

    assert actual == dt.date(2026, 11, 13)


def test_dispatch_target_trading_day_prefers_range_end_for_range_jobs() -> None:
    actual = dispatch_argv.dispatch_target_trading_day(
        "data.heal",
        "--date 2026-11-12 --to=2026-11-13",
        dt.datetime(2026, 11, 16, 12, tzinfo=dt.UTC),
    )

    assert actual == dt.date(2026, 11, 13)


def test_dispatch_target_trading_day_falls_back_when_no_day_is_named(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatched_at = dt.datetime(2026, 11, 16, 12, tzinfo=dt.UTC)
    monkeypatch.setattr(dispatch_argv, "resolve_trading_day", _fixed_trading_day)

    assert dispatch_argv.dispatch_target_trading_day(
        "fault.probe", "--mode replay", dispatched_at
    ) == dt.date(2026, 11, 13)


def test_dispatch_target_trading_day_falls_back_on_an_invalid_explicit_date(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatched_at = dt.datetime(2026, 11, 16, 12, tzinfo=dt.UTC)
    monkeypatch.setattr(dispatch_argv, "resolve_trading_day", _fixed_trading_day)

    # Mutating the invalid-date branch to accept the later --to must make this fail.
    actual = dispatch_argv.dispatch_target_trading_day(
        "fault.probe", "--date not-a-date --to 2026-11-12", dispatched_at
    )

    assert actual == dt.date(2026, 11, 13)
