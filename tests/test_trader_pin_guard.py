"""`scripts/trader_pin_guard.py`: "was tonight's post-close clean", against a
fake `states:ListExecutions` (alpha-engine-config-I11545).

Fixed date literals only. Tuesday 2026-09-08 is the session; Monday
2026-09-07 is Labor Day, the holiday a `weekday() < 5` check gets wrong.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import pathlib
import sys
from zoneinfo import ZoneInfo

import pytest

_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "trader_pin_guard.py"
_spec = importlib.util.spec_from_file_location("trader_pin_guard", _PATH)
guard = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = guard  # a dataclass resolves its module by name
_spec.loader.exec_module(guard)

ET = ZoneInfo("America/New_York")
SESSION = dt.date(2026, 9, 8)
#: The cron's own instant on the session: 23:15 UTC = 19:15 EDT.
ON_TIME = dt.datetime(2026, 9, 8, 23, 15, tzinfo=dt.UTC)
#: The same cron delivered five hours late: 00:15 EDT the next calendar day.
LATE = dt.datetime(2026, 9, 9, 4, 15, tzinfo=dt.UTC)


def _execution(status: str, start_et: dt.datetime, name: str = "x") -> dict:
    return {"name": name, "status": status, "startDate": start_et.astimezone(dt.UTC)}


def _at(day: dt.date, hour: int, minute: int = 5) -> dt.datetime:
    return dt.datetime(day.year, day.month, day.day, hour, minute, tzinfo=ET)


def _pages(*pages: list[dict]):
    calls: list[int] = []

    def list_pages():
        calls.append(1)
        for page in pages:
            yield {"executions": page}

    return list_pages, calls


class TestNominalDay:
    def test_an_on_time_run_is_the_evening_it_fired(self) -> None:
        assert guard.nominal_day(ON_TIME) == SESSION

    def test_a_run_delivered_after_midnight_is_still_the_previous_evening(self) -> None:
        assert guard.nominal_day(LATE) == SESSION


class TestReading:
    def test_one_succeeded_execution_that_day_is_clean(self) -> None:
        pages, _ = _pages([_execution("SUCCEEDED", _at(SESSION, 16))])
        reading = guard.read(ON_TIME, pages)
        assert (reading.session, reading.postclose) == (SESSION, "clean")

    def test_a_late_delivery_reads_the_same_session(self) -> None:
        pages, _ = _pages([_execution("SUCCEEDED", _at(SESSION, 16))])
        assert guard.read(LATE, pages).session == SESSION

    def test_a_failure_followed_by_a_successful_redrive_is_clean(self) -> None:
        pages, _ = _pages(
            [
                _execution("SUCCEEDED", _at(SESSION, 17, 30), "redrive"),
                _execution("FAILED", _at(SESSION, 16), "first"),
            ]
        )
        assert guard.read(ON_TIME, pages).postclose == "clean"

    def test_a_still_running_execution_is_not_clean(self) -> None:
        pages, _ = _pages(
            [
                _execution("RUNNING", _at(SESSION, 18)),
                _execution("SUCCEEDED", _at(SESSION, 16)),
            ]
        )
        reading = guard.read(ON_TIME, pages)
        assert reading.postclose == "unclean" and "RUNNING" in reading.detail

    def test_only_failures_is_not_clean(self) -> None:
        pages, _ = _pages([_execution("FAILED", _at(SESSION, 16))])
        reading = guard.read(ON_TIME, pages)
        assert reading.postclose == "unclean" and "FAILED" in reading.detail

    def test_no_execution_that_day_is_not_clean(self) -> None:
        """Yesterday's success does not certify tonight."""
        pages, _ = _pages([_execution("SUCCEEDED", _at(dt.date(2026, 9, 4), 16))])
        reading = guard.read(ON_TIME, pages)
        assert reading.postclose == "unclean" and "no post-close execution" in reading.detail

    def test_the_next_days_execution_does_not_certify_this_session(self) -> None:
        """A late delivery must not read the following morning's run as tonight's."""
        pages, _ = _pages([_execution("SUCCEEDED", _at(dt.date(2026, 9, 9), 0, 1))])
        assert guard.read(LATE, pages).postclose == "unclean"

    def test_reading_stops_at_the_first_page_that_reaches_before_the_session(self) -> None:
        pages_seen: list[int] = []

        def list_pages():
            for n, page in enumerate(
                [
                    [_execution("SUCCEEDED", _at(SESSION, 16))],
                    [_execution("SUCCEEDED", _at(dt.date(2026, 9, 4), 16))],
                    [_execution("RUNNING", _at(SESSION, 18))],  # never reached
                ]
            ):
                pages_seen.append(n)
                yield {"executions": page}

        assert guard.read(ON_TIME, list_pages).postclose == "clean"
        assert pages_seen == [0, 1]

    def test_a_holiday_is_a_skip_and_never_lists_anything(self) -> None:
        pages, calls = _pages([_execution("SUCCEEDED", _at(dt.date(2026, 9, 7), 16))])
        labor_day = dt.datetime(2026, 9, 7, 23, 15, tzinfo=dt.UTC)
        reading = guard.read(labor_day, pages)
        assert (reading.session, reading.postclose) == (None, "skip")
        assert calls == []

    def test_a_naive_instant_is_refused(self) -> None:
        pages, _ = _pages([])
        with pytest.raises(ValueError, match="naive"):
            guard.read(dt.datetime(2026, 9, 8, 23, 15), pages)


class _FakeStates:
    """A stepfunctions client whose only API is `ListExecutions`."""

    def __init__(self, executions: list[dict]):
        self.executions = executions
        self.calls: list[dict] = []

    def get_paginator(self, name: str):
        assert name == "list_executions", (
            f"the guard called {name}; it may call ListExecutions only"
        )
        fake = self

        class _Paginator:
            def paginate(self, **kwargs):
                fake.calls.append(kwargs)
                yield {"executions": fake.executions}

        return _Paginator()


class TestMain:
    ARN = "arn:aws:states:us-east-1:123456789012:stateMachine:postclose"

    def test_it_writes_the_reading_to_the_step_outputs(self, tmp_path, monkeypatch) -> None:
        output = tmp_path / "out"
        monkeypatch.setenv("GITHUB_OUTPUT", str(output))
        monkeypatch.setenv(guard.STATE_MACHINE_ARN_VAR, self.ARN)
        client = _FakeStates([_execution("SUCCEEDED", _at(SESSION, 16))])
        assert guard.main([], now=ON_TIME, client=client) == 0
        assert output.read_text() == "session=2026-09-08\npostclose=clean\n"
        assert client.calls[0]["stateMachineArn"] == self.ARN

    def test_unclean_is_a_reading_and_exits_zero(self, tmp_path, monkeypatch) -> None:
        output = tmp_path / "out"
        monkeypatch.setenv("GITHUB_OUTPUT", str(output))
        monkeypatch.setenv(guard.STATE_MACHINE_ARN_VAR, self.ARN)
        assert guard.main([], now=ON_TIME, client=_FakeStates([])) == 0
        assert "postclose=unclean" in output.read_text()

    def test_a_holiday_writes_skip(self, tmp_path, monkeypatch) -> None:
        output = tmp_path / "out"
        monkeypatch.setenv("GITHUB_OUTPUT", str(output))
        monkeypatch.setenv(guard.STATE_MACHINE_ARN_VAR, self.ARN)
        labor_day = dt.datetime(2026, 9, 7, 23, 15, tzinfo=dt.UTC)
        assert guard.main([], now=labor_day, client=_FakeStates([])) == 0
        assert output.read_text() == "session=\npostclose=skip\n"

    def test_an_unset_state_machine_fails_rather_than_reading_unclean(self, monkeypatch) -> None:
        monkeypatch.delenv(guard.STATE_MACHINE_ARN_VAR, raising=False)
        assert guard.main([], now=ON_TIME, client=_FakeStates([])) == 1

    def test_a_refused_call_propagates(self, monkeypatch) -> None:
        monkeypatch.setenv(guard.STATE_MACHINE_ARN_VAR, self.ARN)

        class _Denied(_FakeStates):
            def get_paginator(self, name):
                raise PermissionError("AccessDeniedException: states:ListExecutions")

        with pytest.raises(PermissionError):
            guard.main([], now=ON_TIME, client=_Denied([]))

    def test_it_takes_no_arguments(self) -> None:
        assert guard.main(["--force"], now=ON_TIME, client=_FakeStates([])) == 2
