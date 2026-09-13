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
role appears, which is exactly when it should not. The machine allowlist is
:func:`machine_principals`, and everything else that mutates counts.

**The allowlist is DERIVED from the `crucible-v2` stack's own IAM::Role
resources, never hand-kept (`alpha-engine-config-I10307`).** It was a
five-literal tuple until `alpha-engine-config-I10156` (2026-09-07) made role
names unpublishable in this now-public tree and moved it to
`CRUCIBLE_MACHINE_PRINCIPALS`, a hand-kept comma-separated environment
variable. That variable went stale the moment the stack grew past the five
roles it was extracted from: measured 2026-09-09, the stack creates fourteen
roles, and nine machine identities — including `gate-close`, which writes
phase closing records, and `strategy-publish`, `morning-report` and `review`,
which all write — scored as human touches on the one clause that must read
zero. This is the bug class recorded 2026-09-06 (`ops-PR1087`,
`alpha-engine-config-I10121`): a checker needing a hand-written twin of a
source it could read. Nothing failed loud when the twin drifted; it produced
a false UNMET nobody could see the cause of. `crucible-v2.yaml`'s own
`system=crucible-v2` tag is asserted exhaustive over the stack's resources by
phase 0's `v2_resources_tagged_and_versioned` clause, so
`cloudformation:ListStackResources` is an equally exhaustive source and one
this module can read directly rather than trust an operator to keep in sync.

**Deriving from the stack does not weaken the "grows only by PR" invariant —
it makes the PR mandatory instead of advisory.** The prior docstring's
concern was that "an allowlist that can be widened at read time eventually
contains whoever ran the query." An environment variable is exactly that: an
operator sets it with one `gh variable set` command, no review, no template
change. Widening a stack-derived allowlist requires creating a role in
`nous-ergon-ops/infrastructure/cloudformation/crucible-v2.yaml` — a template
PR plus an operator-gated `aws cloudformation deploy` — which is strictly
stronger than the mechanism it replaces, not weaker. Leaving the old
rationale in place beside the new mechanism is how a carve-out outlives its
justification; `tests/test_no_infra_literals.py`'s preamble records that
exact failure happening in this repo once already, over the `tests/`
exemption.

**Why `list_stack_resources` rather than `iam:list-role-tags`
(`system=crucible-v2`) directly**, though phase 0 asserts the two sets equal:
one CloudFormation call already returns every resource's logical id, type and
physical id in one paginated read, `crucible.tags._stack_resources` already
implements exactly that read with the "stack does not exist" / "stack has no
resources" refusals this module needs verbatim, and IAM tag reads are a
second API surface, a second permission
(`iam:ListRoleTags`, distinct from `cloudformation:ListStackResources`) and a
second network round trip per role for no additional exhaustiveness — the tag
audit exists to prove the two answers agree, not because either is more
authoritative. Filtering the stack's own `AWS::IAM::Role` resources is the
narrower, single-call read.

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

from crucible.models import CloudTrailRecord

__all__ = [
    "machine_principals",
    "ArchiveMissingError",
    "ArchiveRead",
    "OperatorAction",
    "OperatorActionCount",
    "PointerAttribution",
    "PointerWrite",
    "StackUnmeasurableError",
    "attribute_pointer_writes",
    "count_operator_actions",
    "date_partitions",
    "iter_archive_records",
    "trailing_calendar_month",
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


def _cfn_client() -> Any:
    """A CloudFormation client for the stack-resource read.

    A module-level function, lazy and substitutable, for the same reason
    `crucible.gate._s3_client` is: importing this module must not require an
    AWS SDK, and a test replaces this rather than reaching for a credential
    chain.
    """
    import boto3  # noqa: PLC0415 - lazy on purpose

    return boto3.client("cloudformation")


class StackUnmeasurableError(RuntimeError):
    """The `crucible-v2` stack could not be described, or names no role.

    Raised rather than returning an empty tuple. A stack that does not exist,
    cannot be reached, or (having been edited some other way) genuinely lists
    no `AWS::IAM::Role` resource is the loudest possible way to have no
    machine principals — and grading against an empty allowlist would score
    every machine action as a human touch, rendering a fully autonomous month
    as a fully manual one. That is the same UNMEASURABLE-never-zero shape
    :class:`ArchiveMissingError` enforces for the archive read this allowlist
    feeds; the two are kept as separate exception types because the caller
    (`crucible.gate._clause_zero_human_mutating_calls`) needs one story
    either way — both are caught by its existing broad `except Exception`
    and rendered UNMEASURABLE with the exception's own detail.
    """


def machine_principals(cfn: Any | None = None, *, stack: str | None = None) -> tuple[str, ...]:
    """The exhaustive machine allowlist, derived from the `crucible-v2`
    stack's own `AWS::IAM::Role` resources.

    See the module docstring ("The allowlist is DERIVED...") for why this
    reads the stack rather than a hand-kept environment variable, and why
    `cloudformation:ListStackResources` rather than an IAM tag read.

    ``cfn`` defaults to a lazily-constructed boto3 client (substituted in
    tests); ``stack`` defaults to `crucible.config.settings().stack_name` —
    the same `CRUCIBLE_STACK`-resolved name `crucible.tags.audit_stack_tags`
    reads, so a second account or a renamed stack is one variable, not two.

    Matched against the role NAME (CloudFormation's `PhysicalResourceId` for
    an `AWS::IAM::Role`), not an ARN, because the account id is environment
    and would make the allowlist wrong in a second account — the same
    invariant the prior environment-variable form stated.
    """
    from crucible.config import settings  # noqa: PLC0415 - avoid an import cycle at module load
    from crucible.tags import StackNotAppliedError, _stack_resources  # noqa: PLC0415

    stack_name = stack or settings().stack_name
    client = cfn if cfn is not None else _cfn_client()
    try:
        resources = _stack_resources(client, stack_name)
    except StackNotAppliedError as exc:
        raise StackUnmeasurableError(str(exc)) from exc
    names = sorted(
        {
            r["PhysicalResourceId"]
            for r in resources
            if r["ResourceType"] == "AWS::IAM::Role" and r.get("PhysicalResourceId")
        }
    )
    if not names:
        raise StackUnmeasurableError(
            f"stack {stack_name!r} lists no AWS::IAM::Role resource. An empty derived "
            "allowlist would score every machine action as a human touch, exactly as an "
            "unreadable stack does — this must never be reported as zero principals."
        )
    return tuple(names)


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


def _principal(record: CloudTrailRecord) -> tuple[str, str]:
    """The acting principal's NAME and CloudTrail identity type.

    The name is taken from `sessionIssuer.userName` for an assumed role — the
    role, not the session — because the session name is the caller's and
    would make every automation run look like a different principal.

    `alpha-engine-config-I10045` row 12: `record` is now a validated
    `crucible.models.CloudTrailRecord` rather than a raw dict walked three
    levels deep by hand with `.get(..., {})` at each level — a typo'd key
    at any level used to resolve silently to "no issuer" instead of
    surfacing.
    """
    identity = record.userIdentity
    issuer = identity.sessionContext.sessionIssuer if identity.sessionContext else None
    name = (issuer.userName if issuer else None) or identity.userName or identity.arn or "unknown"
    return str(name), str(identity.type)


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


def trailing_calendar_month(render_day: dt.date) -> tuple[dt.date, dt.date]:
    """The trailing calendar month for ``render_day``: month-to-date.

    `alpha-engine-config-I10416`: the standing monthly reading is a
    CALENDAR window (one of §4.12's exhaustive trading-days exceptions —
    CloudTrail windows), not a trading-day one, for the same reason
    :func:`iter_archive_records` walks calendar days: the archive delivers on
    wall-clock time and an operator apply on a Saturday must count.

    ``[first day of render_day's month, render_day]``, inclusive of both
    ends — month-to-date rather than the PRIOR completed month, so the board
    reads today's accumulation rather than a number that is up to a month
    stale. A caller wanting the fully-closed prior month passes the last day
    of that month as ``render_day``.
    """
    return render_day.replace(day=1), render_day


def count_operator_actions(
    client: Any,
    *,
    bucket: str,
    prefix: str,
    start: dt.date,
    end: dt.date,
    marker: str = "crucible-v2",
    cfn: Any | None = None,
    reserved: frozenset[str] = frozenset(),
) -> OperatorActionCount:
    """Count human-originated mutating calls against v2 over the window.

    ``client`` is an S3 client. There is no CloudTrail client here and there
    is not meant to be one: the API this gate is forbidden to use is the only
    thing a CloudTrail client would be for. ``cfn`` is a SEPARATE, optional
    CloudFormation client — forwarded to :func:`machine_principals` for the
    allowlist derivation, substitutable in tests the same way ``client`` is;
    it defaults to `machine_principals`'s own lazily-constructed client, so a
    production caller (`crucible.gate._clause_zero_human_mutating_calls`)
    passes none.

    ``reserved`` is `alpha-engine-config-I10416`'s §11 row 9 exclusion list —
    CloudTrail `eventName`s that are human-originated and mutating and are
    excluded anyway: IB Gateway paper re-auth, privileged SSO actions,
    rulings on holdout unseal and trader release pin. It is CONFIG
    (`crucible.config.Settings.autonomy_reserved_events`), never hardcoded
    here — this function only applies whatever set a caller hands it, and
    the default is empty, so a caller that passes nothing gets the prior,
    unreserved behaviour exactly. Applied AFTER the machine-principal
    allowlist, on the event's OWN `eventName` regardless of who the
    principal was, so a reserved action never has to also be a registered
    machine principal to be excluded.

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
    # Derived ONCE per call, after the archive-coverage checks above rather
    # than per candidate record: a bad bucket or an uncovered window must
    # raise on the archive alone, with no CloudFormation call in between, and
    # the read.records this loop walks is already the "a handful" set
    # `_is_candidate` filtered down to — one stack read for all of them.
    principals = machine_principals(cfn)
    actions: list[OperatorAction] = []
    for raw_record in read.records:
        # `alpha-engine-config-I10045` row 12: validated only HERE, on the
        # already-filtered KEPT records (a handful) — not on every scanned
        # record, which stays a raw dict on the memory/throughput-critical
        # hot path `_is_candidate`/`_touches` walk (`CloudTrailRecord`'s
        # own docstring names the measured cost of doing otherwise).
        record = CloudTrailRecord.model_validate(raw_record)
        name, kind = _principal(record)
        if name in principals:
            continue
        if record.eventName in reserved:
            continue
        actions.append(
            OperatorAction(
                event_time=record.eventTime,
                event_name=record.eventName,
                event_source=record.eventSource,
                principal=name,
                principal_type=kind,
                request_id=record.requestID,
            )
        )
    return OperatorActionCount(
        start=start,
        end=end,
        objects_read=read.objects_read,
        records_scanned=read.records_scanned,
        actions=tuple(sorted(actions, key=lambda a: a.event_time)),
    )


# ── alpha-engine-config-I10608: which pointer flips were HUMAN ─────────────
# Brian's ruling 2026-09-12 (option (a)): the autonomy window restarts on a
# human-originated change only. A release-pointer flip written by a stack
# machine principal — `deploy.yml` under `DeployRole` — IS the autonomy the
# phase certifies, not an interruption of it, so it must not restart the
# window. Only the pointer's OWN writer can settle that, and the pointer
# document carries no writer: `releases/current` is a deterministic
# `release.json` reference (`alpha-engine-config-I9786`), identical whoever
# put it there. So the writer is read from the same CloudTrail S3 archive
# this module already walks, never inferred from the document.

#: The S3 API calls that can move `releases/current`. `PutObject` is what
#: `crucible.release` issues; the other two are included because a gate whose
#: attribution set is narrower than the API surface fails in the direction
#: that reads clean — a flip written by a call this set does not name would
#: be UNATTRIBUTABLE, which is handled, rather than silently absent.
_POINTER_WRITE_EVENTS = frozenset({"PutObject", "CopyObject", "CompleteMultipartUpload"})

#: How far apart the pointer object's `LastModified` and the CloudTrail
#: `eventTime` for the same write may sit and still be the same event. Both
#: are second-granularity and stamped by the same service within one request,
#: so seconds is the real distance; five minutes is slack, not a tolerance
#: being leaned on. Its ONLY use is deciding whether the flip we can see
#: (`HeadObject`) is the flip we attributed — a flip with no matching record
#: is treated as human by the caller.
POINTER_ATTRIBUTION_TOLERANCE = dt.timedelta(minutes=5)


@dataclass(frozen=True)
class PointerWrite:
    """One archived write of the release pointer, and who issued it."""

    at: dt.datetime
    event_name: str
    principal: str
    principal_type: str
    machine: bool


@dataclass(frozen=True)
class PointerAttribution:
    """What the archive could say about who has moved the pointer.

    ``unattributable`` is a REASON string when the archive could not settle
    the question and `None` when it could. It is never an empty answer
    dressed as a clean one: the caller
    (`crucible.gate._last_system_change`) turns any reason at all into
    "treat the flip as human", which restarts the window and keeps the
    clause UNMET for longer. Over-counting human changes is the safe
    direction for a clause asserting a count of zero — the same direction
    :func:`_touches` chose, and the inverse of a gate that reads clean
    because it could not see.
    """

    writes: tuple[PointerWrite, ...]
    latest_human: dt.datetime | None
    unattributable: str | None
    objects_read: int
    records_scanned: int


def attribute_pointer_writes(
    client: Any,
    *,
    bucket: str,
    prefix: str,
    object_bucket: str,
    object_key: str,
    since: dt.datetime,
    until: dt.datetime,
    cfn: Any | None = None,
) -> PointerAttribution:
    """Who wrote ``object_key`` in ``(since, until]``, from the archive.

    ``bucket``/``prefix`` locate the CloudTrail archive;
    ``object_bucket``/``object_key`` are the v2 store's bucket and the
    pointer's FULL key (`S3Store._s3_key(POINTER_KEY)`), matched against the
    record's own `requestParameters` rather than by substring — this is the
    one read in this module asking about a single named object, and
    :func:`_touches`'s deliberately broad substring scan would match any
    record that merely mentions the key.

    ``since`` is the floor the answer has to beat: the stack's last apply,
    which is human today and already starts the window. A pointer write at or
    before it cannot move the start, so the scan walks calendar days BACKWARD
    from ``until`` and stops at the first day carrying a human write — the
    latest human write is on the newest day that has one, and nothing older
    can change the answer. On a system flipping the pointer daily that is one
    day of archive, not the whole span.

    A day the trail delivered nothing for is `unattributable`, not "no writes
    that day": a trail always delivers, so an empty day means the trail does
    not cover it, and counting a gap as silence is the §11 risk 8 failure
    with a different cause.
    """
    if not bucket:
        raise ArchiveMissingError(
            "no CloudTrail archive bucket is configured, so no release-pointer write "
            "can be attributed to a principal."
        )

    def _is_pointer_write(record: dict[str, Any]) -> bool:
        if record.get("eventName") not in _POINTER_WRITE_EVENTS:
            return False
        params = record.get("requestParameters")
        if not isinstance(params, dict):
            return False
        if params.get("bucketName") != object_bucket:
            return False
        return str(params.get("key") or "").lstrip("/") == object_key

    principals: tuple[str, ...] | None = None
    writes: list[PointerWrite] = []
    objects_read = 0
    records_scanned = 0
    day = until.date()
    floor_day = since.date()
    while day >= floor_day:
        read = iter_archive_records(
            client, bucket=bucket, prefix=prefix, start=day, end=day, keep=_is_pointer_write
        )
        objects_read += read.objects_read
        records_scanned += read.records_scanned
        if read.uncovered_days:
            return PointerAttribution(
                writes=tuple(sorted(writes, key=lambda w: w.at)),
                latest_human=None,
                unattributable=(
                    f"the CloudTrail archive s3://{bucket}/{prefix} delivered no objects "
                    f"for {day.isoformat()}, so who moved the pointer that day cannot be "
                    "read. A trail always delivers, so an uncovered day is a gap, not "
                    "silence"
                ),
                objects_read=objects_read,
                records_scanned=records_scanned,
            )
        found_human = False
        for raw_record in read.records:
            record = CloudTrailRecord.model_validate(raw_record)
            instant = _pointer_event_instant(record.eventTime)
            if instant is None:
                return PointerAttribution(
                    writes=tuple(sorted(writes, key=lambda w: w.at)),
                    latest_human=None,
                    unattributable=(
                        f"a pointer write carried an eventTime that will not parse "
                        f"({record.eventTime!r}), so it cannot be placed relative to the "
                        "window's floor"
                    ),
                    objects_read=objects_read,
                    records_scanned=records_scanned,
                )
            if principals is None:
                # Derived lazily and ONCE: a span with no pointer write at all
                # must not require a CloudFormation call to answer.
                principals = machine_principals(cfn)
            name, kind = _principal(record)
            write = PointerWrite(
                at=instant,
                event_name=record.eventName,
                principal=name,
                principal_type=kind,
                machine=name in principals,
            )
            writes.append(write)
            if not write.machine and write.at > since:
                found_human = True
        if found_human:
            break
        day -= dt.timedelta(days=1)
    ordered = tuple(sorted(writes, key=lambda w: w.at))
    humans = [w.at for w in ordered if not w.machine and w.at > since]
    if humans:
        return PointerAttribution(ordered, max(humans), None, objects_read, records_scanned)
    if not any(abs(w.at - until) <= POINTER_ATTRIBUTION_TOLERANCE for w in ordered):
        return PointerAttribution(
            writes=ordered,
            latest_human=None,
            unattributable=(
                f"the pointer's own flip instant {until.isoformat()} matches no archived "
                f"{'/'.join(sorted(_POINTER_WRITE_EVENTS))} on s3://{object_bucket}/"
                f"{object_key} within {POINTER_ATTRIBUTION_TOLERANCE}, so the writer of "
                f"the flip that is actually there is unknown ({len(ordered)} pointer "
                "write(s) were archived over the scanned days)"
            ),
            objects_read=objects_read,
            records_scanned=records_scanned,
        )
    return PointerAttribution(ordered, None, None, objects_read, records_scanned)


def _pointer_event_instant(event_time: str) -> dt.datetime | None:
    """A CloudTrail `eventTime` as a UTC instant, or `None` if it will not
    parse. Mirrors `crucible.gate._event_instant` rather than importing it —
    that module imports this one, not the other way round."""
    try:
        parsed = dt.datetime.fromisoformat(event_time.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(dt.UTC)
