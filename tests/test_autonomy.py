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

import pytest

from crucible import autonomy
from crucible.autonomy import (
    MACHINE_PRINCIPALS,
    ArchiveMissingError,
    count_operator_actions,
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
            def paginate(self, *, Bucket: str, Prefix: str):  # noqa: N803 - boto3's shape
                contents = [{"Key": k} for k in sorted(objects) if k.startswith(Prefix)]
                yield {"Contents": contents}

        return _Paginator()

    def get_object(self, *, Bucket: str, Key: str):  # noqa: N803 - boto3's shape
        payload = gzip.compress(json.dumps({"Records": self._objects[Key]}).encode("utf-8"))

        class _Body:
            def read(self) -> bytes:
                return payload

        return {"Body": _Body()}


def _record(**over) -> dict:
    document = {
        "eventTime": "2026-08-03T18:00:00Z",
        "eventName": "UpdateFunctionCode",
        "eventSource": "lambda.amazonaws.com",
        "readOnly": False,
        "requestID": "req-1",
        "requestParameters": {"functionName": "crucible-v2-dispatcher"},
        "userIdentity": {
            "type": "AssumedRole",
            "arn": "arn:aws:sts::711398986525:assumed-role/AWSReservedSSO_admin/brian",
            "sessionContext": {"sessionIssuer": {"userName": "AWSReservedSSO_admin"}},
        },
    }
    document.update(over)
    return document


def _archive(records_by_day: dict[dt.date, list[dict]]) -> _FakeS3:
    objects: dict[str, list[dict]] = {}
    for day, records in records_by_day.items():
        objects[f"AWSLogs/711398986525/CloudTrail/us-east-1/{day:%Y/%m/%d}/part.json.gz"] = records
    return _FakeS3(objects)


def _count(client, **over):
    kwargs = {
        "bucket": "trail",
        "prefix": "AWSLogs/711398986525/CloudTrail/us-east-1",
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
            _count(_archive({}))


class TestCounting:
    def test_a_human_mutating_call_on_a_v2_resource_counts(self) -> None:
        result = _count(_archive({START: [_record()]}))
        assert result.count == 1
        assert result.actions[0].principal == "AWSReservedSSO_admin"
        assert result.actions[0].event_name == "UpdateFunctionCode"
        assert result.objects_read == 1

    def test_a_declared_machine_principal_does_not_count(self) -> None:
        machine = _record(
            userIdentity={
                "type": "AssumedRole",
                "arn": "arn:aws:sts::711398986525:assumed-role/crucible-v2-runtime/i-1",
                "sessionContext": {"sessionIssuer": {"userName": "crucible-v2-runtime"}},
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
                "arn": "arn:aws:sts::711398986525:assumed-role/some-new-role/x",
                "sessionContext": {"sessionIssuer": {"userName": "some-new-role"}},
            }
        )
        assert "some-new-role" not in MACHINE_PRINCIPALS
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
        assert document["objects_read"] == 1
        assert document["count"] == 1
        assert document["actions"][0]["event_name"] == "UpdateFunctionCode"
