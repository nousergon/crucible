"""`board/current.json.human_touch_count` — `alpha-engine-config-I10416`.

Autonomy as a STANDING monthly number, not a gate that measured one window
once. `crucible.board._read_human_touch_count` generalises the SAME archive
reader the phase-2 gate clause uses (`crucible.autonomy.count_operator_
actions`) over the trailing calendar month — no second implementation of the
archive walk, the machine-principal allowlist, or the reserved-action
exclusion.
"""

from __future__ import annotations

import datetime as dt

from crucible.autonomy import ArchiveMissingError
from crucible.board import HumanTouchReading, _read_human_touch_count, build_board
from crucible.store import LocalStore


class _FakeCounted:
    def __init__(self, count: int, *, records_scanned: int = 10, objects_read: int = 2) -> None:
        self.count = count
        self.records_scanned = records_scanned
        self.objects_read = objects_read
        self.actions = tuple(
            {
                "event_time": f"2026-09-0{i + 1}T00:00:00Z",
                "event_name": "UpdateFunctionCode",
                "event_source": "lambda.amazonaws.com",
                "principal": f"human-{i}",
                "principal_type": "AssumedRole",
                "request_id": f"req-{i}",
            }
            for i in range(count)
        )


class _FakeOperatorAction:
    """Stands in for `crucible.autonomy.OperatorAction` — only `to_dict` is used."""

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def to_dict(self) -> dict:
        return self._payload


def _counted(count: int) -> _FakeCounted:
    fake = _FakeCounted(count)
    fake.actions = tuple(_FakeOperatorAction(a) for a in fake.actions)
    return fake


class TestNoArchiveConfigured:
    def test_reads_unmeasured_never_zero(self, monkeypatch) -> None:
        monkeypatch.delenv("CRUCIBLE_CLOUDTRAIL_ARCHIVE", raising=False)
        reading = _read_human_touch_count(dt.date(2026, 9, 10))
        assert reading.measured is False
        assert reading.count == 0
        assert "no CloudTrail archive" in reading.detail


class TestAConfiguredArchive:
    def test_a_clean_month_reads_zero_and_measured(self, monkeypatch) -> None:
        monkeypatch.setenv("CRUCIBLE_CLOUDTRAIL_ARCHIVE", "s3://trail/prefix")
        monkeypatch.setattr("crucible.autonomy.count_operator_actions", lambda *a, **k: _counted(0))
        reading = _read_human_touch_count(dt.date(2026, 9, 10))
        assert reading.measured is True
        assert reading.count == 0
        assert reading.month == "2026-09"

    def test_a_non_zero_month_is_a_finding_naming_the_calls(self, monkeypatch) -> None:
        """A non-zero month is a FINDING, never a gate quietly re-passed —
        the reading names the calls rather than collapsing to a bare count."""
        monkeypatch.setenv("CRUCIBLE_CLOUDTRAIL_ARCHIVE", "s3://trail/prefix")
        monkeypatch.setattr("crucible.autonomy.count_operator_actions", lambda *a, **k: _counted(2))
        reading = _read_human_touch_count(dt.date(2026, 9, 10))
        assert reading.count == 2
        assert reading.measured is True
        assert len(reading.actions) == 2
        assert reading.to_dict()["actions"][0]["event_name"] == "UpdateFunctionCode"

    def test_a_missing_archive_error_reads_unmeasured_not_zero(self, monkeypatch) -> None:
        monkeypatch.setenv("CRUCIBLE_CLOUDTRAIL_ARCHIVE", "s3://trail/prefix")

        def _raise(*a, **k):
            raise ArchiveMissingError("no trail")

        monkeypatch.setattr("crucible.autonomy.count_operator_actions", _raise)
        reading = _read_human_touch_count(dt.date(2026, 9, 10))
        assert reading.measured is False
        assert reading.count == 0

    def test_an_unexpected_failure_is_unmeasured_not_a_crashed_board(self, monkeypatch) -> None:
        """A credential or network failure must not take the whole board
        render down — the same posture `_clause_zero_human_mutating_calls`
        takes for the identical read."""
        monkeypatch.setenv("CRUCIBLE_CLOUDTRAIL_ARCHIVE", "s3://trail/prefix")

        def _raise(*a, **k):
            raise RuntimeError("boom")

        monkeypatch.setattr("crucible.autonomy.count_operator_actions", _raise)
        reading = _read_human_touch_count(dt.date(2026, 9, 10))
        assert reading.measured is False
        assert "boom" in reading.detail

    def test_the_window_passed_is_month_to_date(self, monkeypatch) -> None:
        monkeypatch.setenv("CRUCIBLE_CLOUDTRAIL_ARCHIVE", "s3://trail/prefix")
        captured = {}

        def _capture(*a, **k):
            captured.update(k)
            return _counted(0)

        monkeypatch.setattr("crucible.autonomy.count_operator_actions", _capture)
        _read_human_touch_count(dt.date(2026, 9, 10))
        assert captured["start"] == dt.date(2026, 9, 1)
        assert captured["end"] == dt.date(2026, 9, 10)

    def test_the_reserved_events_setting_is_forwarded(self, monkeypatch) -> None:
        monkeypatch.setenv("CRUCIBLE_CLOUDTRAIL_ARCHIVE", "s3://trail/prefix")
        monkeypatch.setenv("CRUCIBLE_AUTONOMY_RESERVED_EVENTS", "ReservedEvent")
        captured = {}

        def _capture(*a, **k):
            captured.update(k)
            return _counted(0)

        monkeypatch.setattr("crucible.autonomy.count_operator_actions", _capture)
        _read_human_touch_count(dt.date(2026, 9, 10))
        assert captured["reserved"] == frozenset({"ReservedEvent"})


class TestBuildBoardCarriesTheReading:
    def test_build_board_populates_human_touch_by_default(self, tmp_path, monkeypatch) -> None:
        """`alpha-engine-config-I10416`: every EXISTING `build_board` caller
        gets `human_touch_count` without a call-site change — this is the
        self-load-unless-supplied shape `registry`/`declarations` already
        use."""
        monkeypatch.delenv("CRUCIBLE_CLOUDTRAIL_ARCHIVE", raising=False)
        store = LocalStore(tmp_path)
        board = build_board(store, now=dt.datetime(2026, 9, 10, 12, 0, tzinfo=dt.UTC))
        assert board.human_touch is not None
        assert board.human_touch.measured is False
        document = board.to_dict()
        assert document["human_touch_count"]["month"] == "2026-09"
        assert document["human_touch_count"]["measured"] is False

    def test_a_supplied_reading_is_used_verbatim(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        supplied = HumanTouchReading("2026-09", 3, (), "fixture", measured=True)
        board = build_board(
            store,
            now=dt.datetime(2026, 9, 10, 12, 0, tzinfo=dt.UTC),
            human_touch=supplied,
        )
        assert board.human_touch is supplied
        assert board.to_dict()["human_touch_count"]["count"] == 3
