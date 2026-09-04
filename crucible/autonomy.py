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
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

__all__ = [
    "MACHINE_PRINCIPALS",
    "ArchiveMissingError",
    "ArchiveRead",
    "OperatorAction",
    "OperatorActionCount",
    "count_operator_actions",
    "date_partitions",
    "iter_archive_records",
]

#: How many archive objects are fetched at once within one calendar day. Each
#: is one S3 round-trip and a gzip decode, so the work is latency, not CPU.
#: Measured 2026-09-03: one production day is ~1,123 objects / ~36 MB, which
#: took ~150 s sequentially — a fortnight is ~20 minutes, past the board job's
#: timeout. Bounded rather than unbounded: a day with tens of thousands of
#: objects must not open a socket per object.
_ARCHIVE_WORKERS = 32

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
class ArchiveRead:
    """What one pass over the archive read, and what it FAILED to read.

    ``objects_by_day`` carries a key for every calendar day asked for, holding
    zero where the trail delivered nothing. A single total cannot distinguish
    "eight covered days, no human touched anything" from "two covered days and
    six the trail did not exist for", and those are the two answers §11 risk 8
    exists to keep apart.
    """

    records: list[dict[str, Any]]
    objects_by_day: dict[dt.date, int]
    records_scanned: int

    @property
    def objects_read(self) -> int:
        return sum(self.objects_by_day.values())

    @property
    def uncovered_days(self) -> tuple[dt.date, ...]:
        """Every day the archive delivered no object for, in order."""
        return tuple(day for day in sorted(self.objects_by_day) if not self.objects_by_day[day])


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


def _fetch_records(client: Any, *, bucket: str, key: str) -> list[dict[str, Any]]:
    """One archive object's records. The unit of work a worker thread does."""
    body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    payload = json.loads(gzip.decompress(body).decode("utf-8"))
    return list(payload.get("Records") or [])


def iter_archive_records(
    client: Any,
    *,
    bucket: str,
    prefix: str,
    start: dt.date,
    end: dt.date,
    keep: Callable[[dict[str, Any]], bool] | None = None,
) -> ArchiveRead:
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

    **Coverage is reported PER DAY, and that is the point.** A single count
    over the whole window cannot tell "the trail covered every day and nobody
    touched anything" from "the trail started on Tuesday". Only the caller can
    decide what a hole means, but it cannot decide what it never sees, so
    :attr:`ArchiveRead.objects_by_day` carries a key for every calendar day in
    the window — including the ones that yielded nothing.

    **``keep`` filters per object, before anything accumulates.** Measured
    2026-09-03 against the production archive: one day is ~1,123 objects and
    ~181k records, so a fortnight is ~1.4M records and 6–7 GB of resident
    dicts if every record is held. The gate wants the mutating records that
    name a v2 resource — a handful — so the predicate is applied as each
    object is decoded and peak memory is the size of the ANSWER rather than
    the size of the archive. ``records_scanned`` still counts everything read,
    because "how much was looked at" is half of what makes the count credible.

    Objects within one day are fetched concurrently: the work is a few hundred
    milliseconds of S3 round-trip each and nothing else, and sequentially a
    fortnight took ~20 minutes — past the board job's timeout, which is a
    measurement failing for a reason that has nothing to do with what is being
    measured.
    """
    kept: list[dict[str, Any]] = []
    scanned = 0
    objects_by_day: dict[dt.date, int] = {}
    paginator = client.get_paginator("list_objects_v2")
    partitions = date_partitions(client, bucket=bucket, prefix=prefix)
    day = start
    while day <= end:
        keys: list[str] = []
        for partition in partitions:
            day_prefix = f"{partition}/{day:%Y/%m/%d}/"
            for page in paginator.paginate(Bucket=bucket, Prefix=day_prefix):
                keys.extend(entry["Key"] for entry in page.get("Contents") or [])
        objects_by_day[day] = len(keys)
        if keys:
            workers = min(_ARCHIVE_WORKERS, len(keys))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(_fetch_records, client, bucket=bucket, key=key) for key in keys
                ]
                for future in futures:
                    records = future.result()
                    scanned += len(records)
                    kept.extend(r for r in records if keep is None or keep(r))
        day += dt.timedelta(days=1)
    return ArchiveRead(records=kept, objects_by_day=objects_by_day, records_scanned=scanned)


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

    **Coverage is asserted PER CALENDAR DAY.** A window total greater than
    zero says only that the trail existed for SOME of it, and a count read
    from a covered fraction reads exactly like a count read from the whole —
    the §11 risk 8 failure with a different cause. Measured 2026-09-03: the
    fleet trail was created 2026-09-01, so a phase-2 window of
    2026-08-26..2026-09-02 had six of eight days with nothing delivered, and
    the count taken over it would have been presented as a full-window figure.
    Any uncovered day therefore raises :class:`ArchiveMissingError` NAMING the
    days, which the gate renders as UNMEASURABLE — the honest answer until the
    trail has the window's worth of history.
    """
    if not bucket:
        raise ArchiveMissingError(
            "no CloudTrail archive bucket is configured. The autonomy gate counts "
            "human mutating calls from the delivered archive over the full window; "
            "without a trail there is nothing to read, and reporting 0 would make "
            "'no trail' and 'no human touched it' the same answer."
        )

    def _is_candidate(record: dict[str, Any]) -> bool:
        # An ABSENT `readOnly` counts as mutating. CloudTrail omits the field
        # for some services, and defaulting the unknown to read-only would
        # drop exactly the events nobody has classified yet.
        return record.get("readOnly") is not True and _touches(record, marker)

    read = iter_archive_records(
        client, bucket=bucket, prefix=prefix, start=start, end=end, keep=_is_candidate
    )
    uncovered = read.uncovered_days
    if uncovered:
        shown = ", ".join(day.isoformat() for day in uncovered[:8])
        more = "" if len(uncovered) <= 8 else f" (and {len(uncovered) - 8} more)"
        raise ArchiveMissingError(
            f"the CloudTrail archive s3://{bucket}/{prefix} delivered no objects for "
            f"{len(uncovered)} of the {len(read.objects_by_day)} calendar days in "
            f"{start.isoformat()}..{end.isoformat()}: {shown}{more}. A trail always "
            "delivers — even a silent account gets periodic objects — so an uncovered "
            "day means the trail does not cover it. Counting over the covered days "
            "would report a fraction of the window as if it were the whole. "
            "UNMEASURABLE, not zero."
        )
    actions: list[OperatorAction] = []
    for record in read.records:
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
        objects_read=read.objects_read,
        records_scanned=read.records_scanned,
        actions=tuple(sorted(actions, key=lambda a: a.event_time)),
    )
