"""How many human-originated mutating calls touched v2 over a window.

Normative source: plan §2 row 1 ("runs autonomously, minimal input") and
§11 risk 8, which is the whole reason this module is not three lines of
`aws cloudtrail lookup-events`:

    The "0 human mutating calls" gate is measured by CloudTrail, whose
    username lookup silently truncates to ~2 days. The gate reads clean
    because the query could not see the week.

`lookup-events` returns a plausible short answer instead of an error, so a
multi-week gate built on it reports zero because it looked at two days. This
module reads the **CloudTrail S3 archive** — the gzipped JSON objects a trail
delivers — over the whole window, and `tests/test_autonomy.py` asserts the
module never imports or calls the lookup API at all. That assertion is the
point: the correct implementation and the wrong one produce the same shape of
output, and only the wrong one is easy.

**Unknown principals count as HUMAN.** A classifier whose fall-through was
"probably automation" would make the autonomy gate read clean the day a new
role appears, which is exactly when it should not. The declared machine
principals are the exhaustive allowlist; everything else that mutates counts.

**No archive is UNMEASURABLE, never zero.** A missing trail is the loudest
possible way to have no human calls, and rendering it as `0` would be *no
data* painted green (principle 7).

**The region partition is discovered, never configured.** The archive URI is
the one `fleet-cloudtrail.yaml` exports, which stops above CloudTrail's
`{region}/` level. Pinning a region there would narrow a zero-assertion gate
by configuration — it would stop counting the day the trail delivered a second
region, in the direction that reads clean. :func:`date_partitions` lists that
level instead, and still honours a prefix that already names one region.
"""

from __future__ import annotations

import datetime as dt
import gzip
import json
import re
from dataclasses import dataclass
from typing import Any

__all__ = [
    "MACHINE_PRINCIPALS",
    "ArchiveMissingError",
    "OperatorAction",
    "OperatorActionCount",
    "count_operator_actions",
    "date_partitions",
    "iter_archive_records",
]

#: A date partition's first level. CloudTrail's key layout is
#: ``AWSLogs/{account}/CloudTrail/{region}/{YYYY}/{MM}/{DD}/``, so the only
#: thing distinguishing "this prefix already names a region" from "this prefix
#: is the region LEVEL" is whether its immediate children are years.
_YEAR = re.compile(r"^\d{4}$")

#: The exhaustive set of principals whose mutating calls are the system
#: working rather than a human touching it. Matched against the role or user
#: NAME, not an ARN, because the account id is environment and would make the
#: allowlist wrong in a second account.
#:
#: **This tuple grows only by PR, with the automation named.** An allowlist
#: that can be widened at read time is an allowlist that eventually contains
#: whoever ran the query.
MACHINE_PRINCIPALS: tuple[str, ...] = (
    "crucible-v2-runtime",
    "crucible-v2-dispatcher",
    "crucible-v2-scheduler",
    "crucible-v2-github-deploy",
    "crucible-v2-stack-check",
)


class ArchiveMissingError(RuntimeError):
    """The CloudTrail archive this gate reads does not exist.

    Raised rather than returning zero. A trail that was never created, or was
    deleted, produces exactly the artifact set a perfectly autonomous month
    produces — no objects — and the two must not be the same answer.
    """


@dataclass(frozen=True)
class OperatorAction:
    """One human-originated mutating call against a v2 resource."""

    event_time: str
    event_name: str
    event_source: str
    principal: str
    principal_type: str
    request_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_time": self.event_time,
            "event_name": self.event_name,
            "event_source": self.event_source,
            "principal": self.principal,
            "principal_type": self.principal_type,
            "request_id": self.request_id,
        }


@dataclass(frozen=True)
class OperatorActionCount:
    """The window, what was read to cover it, and what was found."""

    start: dt.date
    end: dt.date
    objects_read: int
    records_scanned: int
    actions: tuple[OperatorAction, ...]

    @property
    def count(self) -> int:
        return len(self.actions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "objects_read": self.objects_read,
            "records_scanned": self.records_scanned,
            "count": self.count,
            "actions": [a.to_dict() for a in self.actions],
        }


def _principal(record: dict[str, Any]) -> tuple[str, str]:
    """The acting principal's NAME and CloudTrail identity type.

    The name is taken from `sessionIssuer.userName` for an assumed role — the
    role, not the session — because the session name is the caller's and
    would make every automation run look like a different principal.
    """
    identity = record.get("userIdentity", {}) or {}
    kind = identity.get("type", "Unknown")
    issuer = (identity.get("sessionContext", {}) or {}).get("sessionIssuer", {}) or {}
    name = (
        issuer.get("userName") or identity.get("userName") or identity.get("arn", "") or "unknown"
    )
    return str(name), str(kind)


def _is_machine(name: str) -> bool:
    return name in MACHINE_PRINCIPALS


def _touches(record: dict[str, Any], marker: str) -> bool:
    """Whether the record names a v2 resource anywhere in its body.

    A substring scan over the serialized record, deliberately, rather than
    only `resources[].ARN`: CloudTrail populates `resources` for some services
    and not others, so an ARN-only filter would silently miss the services it
    does not populate. Over-counting is the safe direction for a gate that
    asserts a count of ZERO — a false positive is investigated, a false
    negative is a gate that reads clean because it could not see.
    """
    return marker in json.dumps(record, separators=(",", ":"))


def date_partitions(client: Any, *, bucket: str, prefix: str) -> tuple[str, ...]:
    """The prefixes a day's objects hang from, one per delivered region.

    CloudTrail's layout is `.../CloudTrail/{region}/{YYYY}/{MM}/{DD}/`, and the
    archive URI this module is pointed at deliberately stops ABOVE the region:
    a multi-region trail delivers one region directory per region it sees, and
    a configured region would make the gate stop counting the day a second one
    appears — silently, and in the direction that reads clean. **A gate that
    asserts a count of ZERO must never be narrowed by configuration.** So the
    region level is DISCOVERED, once, by listing one level with a delimiter.

    A prefix that already names a region is honoured as given: its immediate
    children are years, and the whole archive is then that single partition.
    That is the shape `tests/test_autonomy.py` fixtures use and the shape a
    single-region trail would be configured with, and neither should have to
    change to be read correctly.

    An unlistable or empty level yields the prefix itself, so the caller's
    "no objects" branch — not this one — is what raises
    :class:`ArchiveMissingError`. Deciding "there is no trail" from a listing
    that returned no common prefixes would put that judgement in two places.
    """
    base = prefix.rstrip("/")
    children: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{base}/", Delimiter="/"):
        for entry in page.get("CommonPrefixes") or []:
            children.append(entry["Prefix"].rstrip("/").rsplit("/", 1)[-1])
    partitions = tuple(f"{base}/{child}" for child in sorted(children) if not _YEAR.match(child))
    return partitions or (base,)


def iter_archive_records(
    client: Any,
    *,
    bucket: str,
    prefix: str,
    start: dt.date,
    end: dt.date,
) -> tuple[list[dict[str, Any]], int]:
    """Every CloudTrail record delivered for the calendar days in the window.

    The archive's layout is `.../CloudTrail/{region}/{YYYY}/{MM}/{DD}/*.json.gz`,
    so the region partitions are resolved once (:func:`date_partitions`) and the
    window is then walked one calendar day at a time, each day's objects listed
    under their own prefix. Listing the whole trail and filtering client-side
    would work and would also read years of objects to answer a question about
    a handful of weeks.

    Calendar days, not trading days — one of §4.12's exhaustive exceptions.
    CloudTrail delivers on wall-clock time, and a gate that skipped weekends
    would skip precisely the hours when a Saturday run is repaired by hand.
    """
    records: list[dict[str, Any]] = []
    objects = 0
    paginator = client.get_paginator("list_objects_v2")
    partitions = date_partitions(client, bucket=bucket, prefix=prefix)
    for partition in partitions:
        day = start
        while day <= end:
            day_prefix = f"{partition}/{day:%Y/%m/%d}/"
            for page in paginator.paginate(Bucket=bucket, Prefix=day_prefix):
                for entry in page.get("Contents", []):
                    body = client.get_object(Bucket=bucket, Key=entry["Key"])["Body"].read()
                    payload = json.loads(gzip.decompress(body).decode("utf-8"))
                    records.extend(payload.get("Records", []))
                    objects += 1
            day += dt.timedelta(days=1)
    return records, objects


def count_operator_actions(
    client: Any,
    *,
    bucket: str,
    prefix: str,
    start: dt.date,
    end: dt.date,
    marker: str = "crucible-v2",
) -> OperatorActionCount:
    """Count human-originated mutating calls against v2 over the window.

    ``client`` is an S3 client. There is no CloudTrail client here and there
    is not meant to be one: the API this gate is forbidden to use is the only
    thing a CloudTrail client would be for.
    """
    if not bucket:
        raise ArchiveMissingError(
            "no CloudTrail archive bucket is configured. The autonomy gate counts "
            "human mutating calls from the delivered archive over the full window; "
            "without a trail there is nothing to read, and reporting 0 would make "
            "'no trail' and 'no human touched it' the same answer."
        )
    records, objects = iter_archive_records(
        client, bucket=bucket, prefix=prefix, start=start, end=end
    )
    if objects == 0:
        raise ArchiveMissingError(
            f"the CloudTrail archive s3://{bucket}/{prefix} delivered no objects for "
            f"{start.isoformat()}..{end.isoformat()}. A trail always delivers — even a "
            "silent account gets periodic objects — so an empty window means the trail "
            "does not cover it. UNMEASURABLE, not zero."
        )
    actions: list[OperatorAction] = []
    for record in records:
        # An ABSENT `readOnly` counts as mutating. CloudTrail omits the field
        # for some services, and defaulting the unknown to read-only would
        # drop exactly the events nobody has classified yet.
        if record.get("readOnly") is True:
            continue
        if not _touches(record, marker):
            continue
        name, kind = _principal(record)
        if _is_machine(name):
            continue
        actions.append(
            OperatorAction(
                event_time=record.get("eventTime", ""),
                event_name=record.get("eventName", ""),
                event_source=record.get("eventSource", ""),
                principal=name,
                principal_type=kind,
                request_id=record.get("requestID", ""),
            )
        )
    return OperatorActionCount(
        start=start,
        end=end,
        objects_read=objects,
        records_scanned=len(records),
        actions=tuple(sorted(actions, key=lambda a: a.event_time)),
    )
