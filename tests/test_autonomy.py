"""The autonomy count reads the CloudTrail ARCHIVE, never `lookup-events`.

Normative source: plan §2 row 1 and §11 risk 8. The correct implementation and
the wrong one produce the same SHAPE of output — a small integer — and only
the wrong one is easy, so the first test here is about the API that is not
called.
"""

from __future__ import annotations

import ast
import datetime as dt
import gzip
import inspect
import json
import threading
import time

import pytest

from crucible import autonomy
from crucible.autonomy import (
    MACHINE_PRINCIPALS_VAR,
    ArchiveMissingError,
    count_operator_actions,
    machine_principals,
)

START = dt.date(2026, 8, 3)
END = dt.date(2026, 8, 4)


class _FakeS3:
    """An S3 client over an in-memory `{key: records}` archive."""

    def __init__(self, objects: dict[str, list[dict]]) -> None:
        self._objects = objects

    def get_paginator(self, name: str):
        assert name == "list_objects_v2"
        objects = self._objects

        class _Paginator:
            def paginate(  # noqa: N803 - boto3's shape
                self, *, Bucket: str, Prefix: str, Delimiter: str | None = None
            ):
                keys = [k for k in sorted(objects) if k.startswith(Prefix)]
                if Delimiter is None:
                    yield {"Contents": [{"Key": k} for k in keys]}
                    return
                # S3's own semantics: a key with the delimiter after the
                # prefix is rolled up into CommonPrefixes and does NOT appear
                # in Contents. Modelled rather than approximated — the region
                # discovery this fixture now exercises reads only the rolled-up
                # half, so a fake that returned everything in Contents would
                # pass a reader that asked for the wrong thing.
                common: dict[str, None] = {}
                contents: list[dict[str, str]] = []
                for key in keys:
                    rest = key[len(Prefix) :]
                    head, sep, _ = rest.partition(Delimiter)
                    if sep:
                        common[f"{Prefix}{head}{Delimiter}"] = None
                    else:
                        contents.append({"Key": key})
                yield {
                    "Contents": contents,
                    "CommonPrefixes": [{"Prefix": p} for p in common],
                }

        return _Paginator()

    def get_object(self, *, Bucket: str, Key: str):  # noqa: N803 - boto3's shape
        payload = gzip.compress(json.dumps({"Records": self._objects[Key]}).encode("utf-8"))

        class _Body:
            def read(self) -> bytes:
                return payload

        return {"Body": _Body()}


class _ConcurrencyTrackingS3(_FakeS3):
    """A fake that records how many `get_object` calls overlapped.

    The property under test is not speed — a timing assertion would be a test
    of the machine — but whether more than one round-trip is ever in flight.
    Each call holds briefly so overlap is observable without making the suite
    slow.
    """

    def __init__(self, objects: dict[str, list[dict]]) -> None:
        super().__init__(objects)
        self._lock = threading.Lock()
        self._in_flight = 0
        self.peak_in_flight = 0

    def get_object(self, *, Bucket: str, Key: str):  # noqa: N803 - boto3's shape
        with self._lock:
            self._in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self._in_flight)
        try:
            time.sleep(0.02)
            return super().get_object(Bucket=Bucket, Key=Key)
        finally:
            with self._lock:
                self._in_flight -= 1


def _record(**over) -> dict:
    document = {
        "eventTime": "2026-08-03T18:00:00Z",
        "eventName": "UpdateFunctionCode",
        "eventSource": "lambda.amazonaws.com",
        "readOnly": False,
        "requestID": "req-1",
        # "crucible-v2" is the marker `count_operator_actions` matches on by
        # default (the stack name, which legitimately stays a literal —
        # `crucible/config.py`'s own `DEFAULT_STACK_NAME`); the suffix here
        # is a synthetic fixture function, never a real Lambda name.
        "requestParameters": {"functionName": "crucible-v2-fixture-function"},
        "userIdentity": {
            "type": "AssumedRole",
            "arn": "arn:aws:sts::123456789012:assumed-role/AWSReservedSSO_admin/a-human",
            "sessionContext": {"sessionIssuer": {"userName": "AWSReservedSSO_admin"}},
        },
    }
    document.update(over)
    return document


#: The prefix `fleet-cloudtrail.yaml` exports — it stops ABOVE the region.
ARCHIVE_PREFIX = "AWSLogs/123456789012/CloudTrail"


#: Every calendar day the default fixture archive delivers an object for.
WINDOW = (START, END)


def _archive(
    records_by_day: dict[dt.date, list[dict]],
    *,
    regions: tuple[str, ...] = ("us-east-1",),
    cover: tuple[dt.date, ...] = WINDOW,
) -> _FakeS3:
    """A fixture archive that DELIVERS on every day in ``cover``.

    A real trail always delivers — even a silent account gets periodic
    objects — so a day with no interesting records still has an object with an
    empty `Records` list. Modelling that is not decoration: the reader raises
    on any day the archive delivered nothing for, and a fixture that skipped
    quiet days would make every ordinary assertion below exercise the
    uncovered-day path instead of the thing it names.

    ``cover=()`` is the archive that does not exist.
    """
    objects: dict[str, list[dict]] = {}
    for region in regions:
        for day in cover:
            objects[f"{ARCHIVE_PREFIX}/{region}/{day:%Y/%m/%d}/part.json.gz"] = []
        for day, records in records_by_day.items():
            objects[f"{ARCHIVE_PREFIX}/{region}/{day:%Y/%m/%d}/part.json.gz"] = records
    return _FakeS3(objects)


def _count(client, **over):
    kwargs = {
        "bucket": "trail",
        # The region-scoped spelling, deliberately: every existing assertion
        # below keeps reading through a prefix that already names a region, so
        # region DISCOVERY cannot quietly become the only shape that works.
        "prefix": f"{ARCHIVE_PREFIX}/us-east-1",
        "start": START,
        "end": END,
    }
    kwargs.update(over)
    return count_operator_actions(client, **kwargs)


class TestTheForbiddenApi:
    def test_the_module_never_reaches_for_lookup_events(self) -> None:
        """§11 risk 8: `lookup-events` truncates its username lookup to about
        two days and returns a plausible short answer rather than erroring, so
        a multi-week gate built on it reports zero because it looked at two
        days. The module is asserted to contain no path to it at all."""
        tree = ast.parse(inspect.getsource(autonomy))
        # The AST, not a grep over the text: the module's own docstring names
        # `lookup-events` in order to explain why it is not used, and a text
        # scan that had to exempt the explanation would be a scan nobody could
        # keep honest. Attribute access and call names are what matter.
        called = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)} | {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        }
        assert not {name for name in called if "lookup" in name.lower()}
        clients = [
            node.args[0].value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "client"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ]
        assert "cloudtrail" not in clients

    def test_the_reader_takes_an_s3_client_and_nothing_else(self) -> None:
        signature = inspect.signature(count_operator_actions)
        assert list(signature.parameters)[0] == "client"
        assert "cloudtrail" not in str(signature).lower()


class TestUnmeasurableIsNeverZero:
    def test_no_configured_bucket_raises(self) -> None:
        with pytest.raises(ArchiveMissingError, match="no CloudTrail archive"):
            _count(_archive({}), bucket="")

    def test_an_empty_window_raises_rather_than_reporting_zero(self) -> None:
        """A trail that was never created and a perfectly autonomous month
        produce the same artifact set — no objects — and the two must not be
        the same answer (principle 7)."""
        with pytest.raises(ArchiveMissingError, match="UNMEASURABLE, not zero"):
            _count(_archive({}, cover=()))

    def test_a_day_the_trail_did_not_cover_raises_and_names_it(self) -> None:
        """`alpha-engine-config-I9928`, round 2. Coverage was asserted ONCE
        over the whole window, so a window the trail existed for only part of
        produced a count read from the covered fraction and presented as the
        whole — the §11 risk 8 failure with a different cause.

        MEASURED: the fleet trail was created 2026-09-01, and a phase-2 window
        of 2026-08-26..2026-09-02 had six of its eight days uncovered."""
        with pytest.raises(ArchiveMissingError) as caught:
            _count(_archive({START: [_record()]}, cover=(START,)))
        message = str(caught.value)
        assert END.isoformat() in message, "the uncovered day must be NAMED, not counted"
        assert "1 of the 2 calendar days" in message
        assert "UNMEASURABLE, not zero" in message

    def test_a_partially_covered_window_never_returns_a_count(self) -> None:
        """The dangerous shape is not the exception — it is the plausible
        integer. A partial window must never come back as a number, whatever
        that number is."""
        for supplied in ([_record()], []):
            with pytest.raises(ArchiveMissingError):
                _count(_archive({START: supplied}, cover=(START,)))


class TestCounting:
    def test_a_human_mutating_call_on_a_v2_resource_counts(self) -> None:
        result = _count(_archive({START: [_record()]}))
        assert result.count == 1
        assert result.actions[0].principal == "AWSReservedSSO_admin"
        assert result.actions[0].event_name == "UpdateFunctionCode"
        # Two: one object per covered day, and the window is two days. A trail
        # delivers on the quiet days too.
        assert result.objects_read == 2

    def test_a_declared_machine_principal_does_not_count(self) -> None:
        machine = _record(
            userIdentity={
                "type": "AssumedRole",
                "arn": "arn:aws:sts::123456789012:assumed-role/test-runtime/i-1",
                "sessionContext": {"sessionIssuer": {"userName": "test-runtime"}},
            }
        )
        assert _count(_archive({START: [machine]})).count == 0

    def test_an_unknown_principal_counts_as_human(self) -> None:
        """The fall-through is the whole control. A classifier defaulting to
        'probably automation' would make the gate read clean on the day a new
        role appears — exactly when it should not."""
        stranger = _record(
            userIdentity={
                "type": "AssumedRole",
                "arn": "arn:aws:sts::123456789012:assumed-role/some-new-role/x",
                "sessionContext": {"sessionIssuer": {"userName": "some-new-role"}},
            }
        )
        assert "some-new-role" not in machine_principals()
        assert _count(_archive({START: [stranger]})).count == 1

    def test_a_read_only_call_does_not_count(self) -> None:
        assert _count(_archive({START: [_record(readOnly=True)]})).count == 0

    def test_a_record_with_no_readonly_field_counts(self) -> None:
        """CloudTrail omits the field for some services. Defaulting the
        unknown to read-only would drop precisely the events nobody has
        classified yet."""
        naked = _record()
        del naked["readOnly"]
        assert _count(_archive({START: [naked]})).count == 1

    def test_a_call_that_names_no_v2_resource_does_not_count(self) -> None:
        assert (
            _count(_archive({START: [_record(requestParameters={"functionName": "eod"})]})).count
            == 0
        )

    def test_the_marker_is_matched_anywhere_in_the_record(self) -> None:
        """CloudTrail populates `resources[]` for some services and not
        others, so an ARN-only filter silently misses the ones it does not
        populate. Over-counting is the safe direction for a gate asserting a
        count of ZERO."""
        buried = _record(
            requestParameters={},
            resources=[{"ARN": "arn:aws:s3:::x/crucible-v2/releases/current"}],
        )
        assert _count(_archive({START: [buried]})).count == 1

    def test_every_calendar_day_of_the_window_is_read(self) -> None:
        """Calendar days, not trading days — CloudTrail delivers on wall-clock
        time, and a gate skipping weekends would skip exactly the hours when a
        Saturday run is repaired by hand."""
        result = _count(
            _archive({START: [_record()], END: [_record(eventTime="2026-08-04T01:00:00Z")]})
        )
        assert result.objects_read == 2
        assert result.count == 2

    def test_the_actions_are_ordered_by_time(self) -> None:
        late = _record(eventTime="2026-08-03T23:00:00Z", requestID="req-late")
        early = _record(eventTime="2026-08-03T01:00:00Z", requestID="req-early")
        result = _count(_archive({START: [late, early]}))
        assert [a.request_id for a in result.actions] == ["req-early", "req-late"]

    def test_the_reading_serializes_with_its_window_and_evidence(self) -> None:
        document = _count(_archive({START: [_record()]})).to_dict()
        assert document["start"] == START.isoformat()
        assert document["end"] == END.isoformat()
        assert document["objects_read"] == 2
        assert document["count"] == 1
        assert document["actions"][0]["event_name"] == "UpdateFunctionCode"


class TestTheRegionPartitionIsDiscovered:
    """`alpha-engine-config-I9928`: the archive URI stops above `{region}/`.

    `fleet-cloudtrail.yaml` exports one URI —
    `s3://<bucket>/AWSLogs/<account>/CloudTrail` — and `crucible-v2.yaml`
    already exports THAT value into every job's shell as
    `CRUCIBLE_CLOUDTRAIL_ARCHIVE`. Read as a date prefix it names
    `.../CloudTrail/2026/09/03/`, which holds nothing, so the gate raised
    `ArchiveMissingError` on a trail that was delivering — UNMEASURABLE for a
    reason that was configuration, not evidence.
    """

    def test_the_exported_region_agnostic_prefix_reads_the_archive(self) -> None:
        result = _count(_archive({START: [_record()]}), prefix=ARCHIVE_PREFIX)
        assert result.objects_read == 2
        assert result.count == 1

    def test_every_delivered_region_is_counted(self) -> None:
        """A multi-region trail delivers one directory per region. Counting
        only one of them narrows a ZERO-assertion gate by configuration, in
        the direction that reads clean."""
        result = _count(
            _archive({START: [_record()]}, regions=("us-east-1", "us-west-2")),
            prefix=ARCHIVE_PREFIX,
        )
        # Two regions x two covered days.
        assert result.objects_read == 4
        assert result.count == 2

    def test_a_prefix_that_already_names_a_region_is_honoured(self) -> None:
        """Its immediate children are years, not regions. Descending a level
        there would build `.../us-east-1/2026/2026/08/03/` and read nothing."""
        partitions = autonomy.date_partitions(
            _archive({START: [_record()]}),
            bucket="trail",
            prefix=f"{ARCHIVE_PREFIX}/us-east-1",
        )
        assert partitions == (f"{ARCHIVE_PREFIX}/us-east-1",)

    def test_an_empty_archive_still_raises_rather_than_reporting_zero(self) -> None:
        """Region discovery must not become a second place "there is no trail"
        is decided — the coverage branch owns that judgement."""
        with pytest.raises(ArchiveMissingError, match="UNMEASURABLE, not zero"):
            _count(_archive({}, cover=()), prefix=ARCHIVE_PREFIX)


class TestTheReadIsStreamedNotAccumulated:
    """`alpha-engine-config-I9928`, round 2.

    MEASURED 2026-09-03 against the production archive: one calendar day is
    ~1,123 objects / ~36 MB / ~181k records, and ~150 s read sequentially. A
    phase-2 fortnight is therefore ~9,000 objects and ~1.4M records — about 20
    minutes and 6-7 GB resident if every record is held to be filtered at the
    end. The board job's timeout is ten minutes, so the gate would have died
    of its own reading rather than reported one.

    Both halves are asserted here because both are invisible in the output: a
    slow, memory-hungry reader and a fast, lean one return the identical
    `OperatorActionCount`, and only the first one is what you get by writing
    the obvious loop.
    """

    def test_only_the_candidate_records_are_ever_held(self) -> None:
        """The peak is the size of the ANSWER, not of the archive."""
        noise = [
            _record(readOnly=True, requestID=f"read-{i}", eventName="GetFunction")
            for i in range(200)
        ]
        offender = _record(requestID="the-one")
        read = autonomy.iter_archive_records(
            _archive({START: [*noise, offender]}),
            bucket="trail",
            prefix=ARCHIVE_PREFIX,
            start=START,
            end=END,
            keep=lambda r: r.get("readOnly") is not True,
        )
        assert len(read.records) == 1, "read-only records must not be accumulated"
        assert read.records[0]["requestID"] == "the-one"
        # …and the count of what was LOOKED AT survives the filtering, because
        # "0 over 3 records" and "0 over 181k records" are different readings.
        assert read.records_scanned == 201

    def test_the_scanned_count_is_reported_even_when_nothing_is_kept(self) -> None:
        read = autonomy.iter_archive_records(
            _archive({START: [_record(readOnly=True) for _ in range(5)]}),
            bucket="trail",
            prefix=ARCHIVE_PREFIX,
            start=START,
            end=END,
            keep=lambda r: r.get("readOnly") is not True,
        )
        assert read.records == []
        assert read.records_scanned == 5

    def test_a_days_objects_are_fetched_concurrently(self) -> None:
        """Sequential is the shape that times out. Asserted by observing that
        more than one `get_object` is in flight at once — the property that
        makes ~9,000 round-trips fit in a job timeout — rather than by timing,
        which would be a flaky test of the machine it runs on."""
        keys = {
            f"{ARCHIVE_PREFIX}/us-east-1/{START:%Y/%m/%d}/part-{i}.json.gz": [_record()]
            for i in range(16)
        }
        keys[f"{ARCHIVE_PREFIX}/us-east-1/{END:%Y/%m/%d}/part.json.gz"] = []
        client = _ConcurrencyTrackingS3(keys)
        autonomy.iter_archive_records(
            client,
            bucket="trail",
            prefix=ARCHIVE_PREFIX,
            start=START,
            end=END,
        )
        assert client.peak_in_flight > 1, (
            "objects within a day must be fetched concurrently; sequentially a "
            "phase-2 window takes ~20 minutes and the board job is killed at its timeout"
        )

    def test_every_record_still_reaches_the_caller_when_nothing_is_filtered(self) -> None:
        """`keep=None` is the unfiltered read the public signature promises.
        Concurrency must not lose an object on the way."""
        keys = {
            f"{ARCHIVE_PREFIX}/us-east-1/{START:%Y/%m/%d}/part-{i}.json.gz": [
                _record(requestID=f"req-{i}")
            ]
            for i in range(16)
        }
        keys[f"{ARCHIVE_PREFIX}/us-east-1/{END:%Y/%m/%d}/part.json.gz"] = []
        read = autonomy.iter_archive_records(
            _FakeS3(keys),
            bucket="trail",
            prefix=ARCHIVE_PREFIX,
            start=START,
            end=END,
        )
        assert read.records_scanned == 16
        assert {r["requestID"] for r in read.records} == {f"req-{i}" for i in range(16)}


class TestMachinePrincipalsRaisesOnUnset:
    """`alpha-engine-config-I10156`: the allowlist is no longer a literal, so
    the raise-on-unset path is the only thing standing between a forgotten
    environment variable and a fully autonomous month graded as fully
    manual (every action falls through to "human")."""

    def test_unset_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(MACHINE_PRINCIPALS_VAR, raising=False)
        with pytest.raises(RuntimeError, match=MACHINE_PRINCIPALS_VAR):
            machine_principals()

    def test_empty_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(MACHINE_PRINCIPALS_VAR, "")
        with pytest.raises(RuntimeError, match=MACHINE_PRINCIPALS_VAR):
            machine_principals()

    def test_blank_entries_alone_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A value of only commas and whitespace strips to nothing, and must
        raise the same as an unset variable rather than returning an empty
        tuple silently."""
        monkeypatch.setenv(MACHINE_PRINCIPALS_VAR, " , , ")
        with pytest.raises(RuntimeError, match=MACHINE_PRINCIPALS_VAR):
            machine_principals()
