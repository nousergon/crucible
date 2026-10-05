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
import hashlib
import json
import re
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

from crucible.models import CloudTrailRecord

__all__ = [
    "machine_principals",
    "ArchiveMissingError",
    "ArchiveRead",
    "CACHE_CLOSED_AFTER_DAYS",
    "DayCache",
    "S3DayCache",
    "OperatorAction",
    "OperatorActionCount",
    "PointerAttribution",
    "StackApply",
    "StackApplyAttribution",
    "PointerWrite",
    "StackUnmeasurableError",
    "attribute_pointer_writes",
    "attribute_stack_applies",
    "configured_day_cache",
    "count_operator_actions",
    "date_partitions",
    "iter_archive_records",
    "literal_needle",
    "trailing_calendar_month",
]

#: How many archive objects are fetched at once within one calendar day. Each
#: is one S3 round-trip and a gzip decode, so the work is latency, not CPU.
#: Measured 2026-09-03 at ~1,123 objects / ~36 MB a day (~150 s sequentially);
#: restated 2026-09-23 (`alpha-engine-config-I11448`): 09-14..09-22 ran
#: ~1,750 objects/day (1,207–2,436), 74–98 MB/day compressed and ~465k
#: records/day, and one day through :func:`iter_archive_records` took 28.5 s
#: wall at this concurrency. Past the fetch, the floor is `json.loads` —
#: ~12 s/day and GIL-bound, so more workers do not buy it back; the ``needle``
#: pre-filter is what does. Bounded rather than unbounded: a day with tens of
#: thousands of objects must not open a socket per object.
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
    #: Objects a ``needle`` ruled out before decode (`alpha-engine-config-I11448`).
    #: Counted in ``objects_by_day``; their records are not in ``records_scanned``.
    objects_prefiltered: int = 0
    #: Covered days whose kept records came from the per-day result cache
    #: rather than a download (`alpha-engine-config-I11792`). Their objects
    #: are still counted in ``objects_by_day`` — the listing that proves
    #: coverage is never cached — and their scanned/prefiltered counts are the
    #: ones recorded when the day was first read.
    days_from_cache: int = 0
    #: One line per cache read or write that failed. A failed cache is never
    #: fatal — the day is scanned, which is the uncached behaviour exactly —
    #: but it is reported rather than swallowed.
    cache_failures: tuple[str, ...] = ()

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


def literal_needle(value: str) -> bytes | None:
    """``value`` as a byte needle for :func:`iter_archive_records`, or `None`
    when its raw-JSON spelling cannot be proven to contain it literally.

    A JSON encoder is free to escape any character as ``\\uXXXX``, must
    escape ``"``, ``\\`` and control characters, and may escape ``/`` as
    ``\\/``. :func:`_needle_rules_out` falls through on ``\\u`` and — for a
    needle holding ``/`` — on ``\\/``, so the only characters left that could
    hide a match are the ones that MUST be escaped. A value carrying one of
    those, or anything outside printable ASCII, gets no needle: an unfiltered
    read is slower, never wrong (`alpha-engine-config-I11448`).
    """
    if not value or not value.isascii() or not value.isprintable():
        return None
    if '"' in value or "\\" in value:
        return None
    return value.encode("ascii")


def _needle_rules_out(raw: bytes, needle: bytes) -> bool:
    """Whether NO record in the decompressed object ``raw`` can carry a JSON
    string whose decoded value contains ``needle``.

    A strict NECESSARY condition, never a heuristic, because the readers it
    serves assert a count of zero and a false skip is a gate reading clean
    because it could not see. A decoded string holding ``needle`` is spelled
    in the raw text either literally or with at least one escape; every escape
    that can spell a :func:`literal_needle` character is ``\\u`` or, for
    ``/``, ``\\/``. So an object is ruled out only when the needle is absent
    AND neither escape is present. A stray ``\\\\u`` (an escaped backslash
    before a ``u``) also falls through — the conservative direction.
    """
    if needle in raw:
        return False
    if b"\\u" in raw:
        return False
    return not (b"/" in needle and b"\\/" in raw)


def _fetch_records(
    client: Any, *, bucket: str, key: str, needle: bytes | None = None
) -> list[dict[str, Any]] | None:
    """One archive object's records. The unit of work a worker thread does.

    With ``needle``, an object :func:`_needle_rules_out` excludes returns
    `None` before `json.loads` — the per-day floor measured 2026-09-23 at
    ~12 s of GIL-bound decode over ~465k records (`alpha-engine-config-I11448`).
    The bytes are still gunzipped (CRC-checked) and UTF-8 decoded first, so a
    truncated or mis-encoded object still raises rather than being skipped;
    only an object that decodes cleanly and cannot hold a match is.
    """
    body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    raw = gzip.decompress(body)
    text = raw.decode("utf-8")
    if needle is not None and _needle_rules_out(raw, needle):
        return None
    payload = json.loads(text)
    return list(payload.get("Records") or [])


# ── alpha-engine-config-I11792: a per-calendar-day result cache ────────────
# Every month-to-date board render re-downloaded every closed day of the
# month — ~95 MB compressed per day, measured 2026-10-05 over 09-20..10-04
# (86–106 MB/day, 831–3,244 objects/day) — so a late-month render pulled
# 2–3 GB of archive to re-derive answers that cannot have changed. That is
# billed S3 internet egress from a GitHub-hosted runner. A CLOSED day's kept
# records are a pure function of (the day's object set, the keep predicate,
# the needle), so they are read once and stored as one small JSON document.

#: A day is cacheable only once it is at least this many days before TODAY
#: (UTC): ``day <= today - CACHE_CLOSED_AFTER_DAYS``. CloudTrail files each
#: object under the UTC date it was DELIVERED, so no new object lands in a
#: day's folder once that day has ended — measured 2026-10-05, the latest PUT
#: into each of 09-28..10-04 was between 23:55:54 and 23:59:31 of the same
#: day. Delivery is documented as typically ~5 minutes and NOT guaranteed, so
#: the margin is a whole day of slack past "the day has ended" (today-1 would
#: cache yesterday minutes after midnight, while S3 PUTs for 23:5x deliveries
#: may still be landing), never a tolerance being leaned on. It is also not
#: the only guard: every read re-lists the day and a cached entry is used only
#: when the listed key set matches the one it was computed from, so an object
#: that does arrive late invalidates the entry instead of being missed.
CACHE_CLOSED_AFTER_DAYS = 2

#: Bumped whenever the cached document's shape or meaning changes; a reader
#: ignores every document carrying any other value.
_CACHE_SCHEMA = 1


class DayCache(Protocol):
    """Where per-day results live. ``get`` returns `None` for a miss."""

    def get(self, scope: str, day: dt.date) -> dict[str, Any] | None: ...

    def put(self, scope: str, day: dt.date, document: dict[str, Any]) -> None: ...


def _scope_digest(scope: str) -> str:
    return hashlib.sha256(scope.encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True)
class S3DayCache:
    """A :class:`DayCache` over ``s3://{bucket}/{prefix}``.

    One object per (scope, day):
    ``{prefix}/v{schema}/{sha256(scope)[:24]}/{YYYY}/{MM}/{DD}.json``. The
    scope is hashed into the key so a scope carrying a bucket or object key
    never shapes a path, and is stored verbatim in the document so a reader
    can refuse a collision rather than trust the hash.
    """

    client: Any
    bucket: str
    prefix: str

    def key(self, scope: str, day: dt.date) -> str:
        base = self.prefix.strip("/")
        tail = f"v{_CACHE_SCHEMA}/{_scope_digest(scope)}/{day:%Y/%m/%d}.json"
        return f"{base}/{tail}" if base else tail

    def get(self, scope: str, day: dt.date) -> dict[str, Any] | None:
        try:
            body = self.client.get_object(Bucket=self.bucket, Key=self.key(scope, day))["Body"]
        except Exception as exc:  # noqa: BLE001 - a miss and a denial both mean "scan"
            if _is_missing_key(exc):
                return None
            raise
        document = json.loads(body.read().decode("utf-8"))
        return document if isinstance(document, dict) else None

    def put(self, scope: str, day: dt.date, document: dict[str, Any]) -> None:
        self.client.put_object(
            Bucket=self.bucket,
            Key=self.key(scope, day),
            Body=json.dumps(document, separators=(",", ":"), sort_keys=True).encode("utf-8"),
            ContentType="application/json",
        )


def _is_missing_key(exc: Exception) -> bool:
    """Whether ``exc`` is S3 saying the key does not exist (a cache MISS)."""
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        code = str((response.get("Error") or {}).get("Code") or "")
        return code in {"NoSuchKey", "404", "NotFound"}
    return isinstance(exc, KeyError | FileNotFoundError)


def configured_day_cache(client: Any) -> S3DayCache | None:
    """The day cache ``CRUCIBLE_CLOUDTRAIL_DAY_CACHE`` names, or `None`.

    Unset means no cache — the uncached read, exactly as before. It is a
    per-workflow setting, never a default, because each scheduled identity
    may write only its OWN prefix: the board role caches under the
    ``board/`` prefix it already owns. A cache one identity writes and
    another grades from would let the writer shape the reader's answer.
    """
    from crucible.config import settings  # noqa: PLC0415 - avoid an import cycle at module load

    location = settings().cloudtrail_day_cache
    if not location:
        return None
    if not location.startswith("s3://"):
        raise ValueError(f"CRUCIBLE_CLOUDTRAIL_DAY_CACHE must be an s3:// URI, got {location!r}")
    bucket, _, prefix = location.removeprefix("s3://").strip("/").partition("/")
    return S3DayCache(client, bucket, prefix)


def _keys_digest(keys: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(keys)).encode("utf-8")).hexdigest()


def _cached_day(
    document: dict[str, Any] | None, *, scope: str, day: dt.date, keys: list[str]
) -> tuple[list[dict[str, Any]], int, int] | None:
    """A cached day's (records, scanned, prefiltered), or `None` to rescan.

    Every field is checked against what THIS read just listed. The coverage
    decision is never taken from the cache — ``objects_by_day`` comes from the
    live listing — and a document computed over a different object set (a
    late delivery, a different region set) is a miss, not a hit.
    """
    if not isinstance(document, dict):
        return None
    if (
        document.get("schema") != _CACHE_SCHEMA
        or document.get("scope") != scope
        or document.get("day") != day.isoformat()
        or document.get("objects") != len(keys)
        or document.get("keys_sha256") != _keys_digest(keys)
    ):
        return None
    records = document.get("records")
    scanned = document.get("records_scanned")
    prefiltered = document.get("objects_prefiltered")
    if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
        return None
    if not isinstance(scanned, int) or not isinstance(prefiltered, int):
        return None
    return records, scanned, prefiltered


def _cacheable(day: dt.date, today: dt.date) -> bool:
    return day <= today - dt.timedelta(days=CACHE_CLOSED_AFTER_DAYS)


def _utc_today() -> dt.date:
    return dt.datetime.now(dt.UTC).date()


def _note_cache_failure(failures: list[str], what: str, day: dt.date, exc: Exception) -> None:
    line = f"day cache {what} for {day.isoformat()} failed ({type(exc).__name__}: {exc})"
    failures.append(line)
    print(f"crucible.autonomy: {line}; the day was scanned instead", file=sys.stderr)


def iter_archive_records(
    client: Any,
    *,
    bucket: str,
    prefix: str,
    start: dt.date,
    end: dt.date,
    keep: Callable[[dict[str, Any]], bool] | None = None,
    needle: bytes | None = None,
    cache: DayCache | None = None,
    cache_scope: str | None = None,
    today: dt.date | None = None,
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
    2026-09-03 against the production archive at ~1,123 objects and ~181k
    records a day; restated 2026-09-23 (`alpha-engine-config-I11448`) at
    ~1,750 objects, 74–98 MB compressed and ~465k records a day, so a
    fortnight is ~6.5M records and far more resident dicts than a runner
    holds if every record is kept. The gate wants the mutating records that
    name a v2 resource — a handful — so the predicate is applied as each
    object is decoded and peak memory is the size of the ANSWER rather than
    the size of the archive. ``records_scanned`` still counts everything read,
    because "how much was looked at" is half of what makes the count credible.

    Objects within one day are fetched concurrently: the work is a few hundred
    milliseconds of S3 round-trip each and nothing else, and sequentially a
    fortnight took ~20 minutes — past the board job's timeout, which is a
    measurement failing for a reason that has nothing to do with what is being
    measured.

    **``needle`` skips whole objects before `json.loads`** — an ASCII byte
    string (build it with :func:`literal_needle`) that every record ``keep``
    accepts is PROVEN to carry literally in its raw JSON. It is the caller's
    proof, not this function's: a needle ``keep`` does not imply turns a
    zero-asserting read into one that cannot see. An object it rules out
    still counts in ``objects_by_day`` (it was delivered and read) and in
    :attr:`ArchiveRead.objects_prefiltered`, but its records are never
    decoded, so ``records_scanned`` counts only the records of objects that
    were.

    **``cache`` + ``cache_scope`` reuse a CLOSED day's kept records**
    (`alpha-engine-config-I11792`). ``cache_scope`` is the caller's name for
    exactly what ``keep`` and ``needle`` select — two different predicates
    must never share a scope, and a predicate whose meaning changes must
    change its scope. Both are required, and ``keep`` must be given: caching
    an unfiltered day would store the whole archive. Three properties hold:

    * the day is ALWAYS listed, and ``objects_by_day`` always comes from that
      listing, so a cached day can never turn an uncovered day into a covered
      one — a day with no delivered objects is never looked up or stored;
    * a cached day is used only when its stored key set matches the listing
      (count and digest), so a late delivery is a miss, never a stale hit;
    * only a day ``<= today - CACHE_CLOSED_AFTER_DAYS`` (UTC) is ever read
      from or written to the cache — today and yesterday are always scanned.

    A failed cache read or write falls back to the scan and is reported in
    :attr:`ArchiveRead.cache_failures`; a cache can make a read cheaper, never
    different.
    """
    use_cache = cache is not None and bool(cache_scope) and keep is not None
    if cache is not None and not use_cache:
        raise ValueError(
            "a day cache needs both a cache_scope naming the keep predicate and the "
            "keep predicate itself; caching an unfiltered read would store the archive"
        )
    reference_day = today if today is not None else _utc_today()
    kept: list[dict[str, Any]] = []
    prefiltered = 0
    scanned = 0
    days_from_cache = 0
    cache_failures: list[str] = []
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
        cacheable = use_cache and bool(keys) and _cacheable(day, reference_day)
        if cacheable:
            assert cache is not None and cache_scope is not None  # narrowed by use_cache
            try:
                hit = _cached_day(
                    cache.get(cache_scope, day), scope=cache_scope, day=day, keys=keys
                )
            except Exception as exc:  # noqa: BLE001 - reported, and the day is scanned
                _note_cache_failure(cache_failures, "read", day, exc)
                hit = None
            if hit is not None:
                records_hit, scanned_hit, prefiltered_hit = hit
                kept.extend(records_hit)
                scanned += scanned_hit
                prefiltered += prefiltered_hit
                days_from_cache += 1
                day += dt.timedelta(days=1)
                continue
        if keys:
            day_kept: list[dict[str, Any]] = []
            day_scanned = 0
            day_prefiltered = 0
            workers = min(_ARCHIVE_WORKERS, len(keys))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(_fetch_records, client, bucket=bucket, key=key, needle=needle)
                    for key in keys
                ]
                for future in futures:
                    records = future.result()
                    if records is None:
                        day_prefiltered += 1
                        continue
                    day_scanned += len(records)
                    day_kept.extend(r for r in records if keep is None or keep(r))
            kept.extend(day_kept)
            scanned += day_scanned
            prefiltered += day_prefiltered
            if cacheable:
                assert cache is not None and cache_scope is not None
                try:
                    cache.put(
                        cache_scope,
                        day,
                        {
                            "schema": _CACHE_SCHEMA,
                            "scope": cache_scope,
                            "day": day.isoformat(),
                            "objects": len(keys),
                            "keys_sha256": _keys_digest(keys),
                            "records_scanned": day_scanned,
                            "objects_prefiltered": day_prefiltered,
                            "records": day_kept,
                        },
                    )
                except Exception as exc:  # noqa: BLE001 - reported; the answer is unaffected
                    _note_cache_failure(cache_failures, "write", day, exc)
        day += dt.timedelta(days=1)
    return ArchiveRead(
        records=kept,
        objects_by_day=objects_by_day,
        records_scanned=scanned,
        objects_prefiltered=prefiltered,
        days_from_cache=days_from_cache,
        cache_failures=tuple(cache_failures),
    )


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
    cache: DayCache | None = None,
    today: dt.date | None = None,
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

    **``cache`` stores the RAW candidates, never the count**
    (`alpha-engine-config-I11792`). What a closed day caches is the
    mutating, ``marker``-naming records :func:`iter_archive_records` kept —
    before the machine-principal allowlist and ``reserved`` are applied. Both
    of those are per-caller and change with config (a role added to the
    stack, an operator-declared reservation), so they are applied fresh to
    every read, cached day or not; a cached filtered count would silently
    keep yesterday's answer to a question whose terms changed.
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
        client,
        bucket=bucket,
        prefix=prefix,
        start=start,
        end=end,
        keep=_is_candidate,
        cache=cache,
        # Names `_is_candidate` exactly: its only parameter is ``marker``.
        # Bump the version whenever `_is_candidate` or `_touches` changes.
        cache_scope=f"operator-candidates/v1/marker={marker}",
        today=today,
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
    cache: DayCache | None = None,
    today: dt.date | None = None,
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

    # `alpha-engine-config-I11448`: ``_is_pointer_write`` accepts a record
    # only when ``str(requestParameters.key).lstrip("/") == object_key``. The
    # key is a letter-bearing path, so no JSON number, bool, null, list or
    # object stringifies to it: the value is a JSON STRING whose decoded text
    # ends with ``object_key``, which :func:`literal_needle` then proves is
    # spelled literally in the raw bytes unless an escape (which
    # `_needle_rules_out` falls through on) is present. `None` — unfiltered
    # — for a key that proof does not cover.
    needle = literal_needle(object_key)
    # Names `_is_pointer_write` and the needle exactly (I11792). The machine
    # allowlist is applied below, on every read, never cached.
    scope = (
        f"pointer-writes/v1/events={','.join(sorted(_POINTER_WRITE_EVENTS))}"
        f"/bucket={object_bucket}/key={object_key}/needle={needle!r}"
    )
    principals: tuple[str, ...] | None = None
    writes: list[PointerWrite] = []
    objects_read = 0
    records_scanned = 0
    day = until.date()
    floor_day = since.date()
    while day >= floor_day:
        read = iter_archive_records(
            client,
            bucket=bucket,
            prefix=prefix,
            start=day,
            end=day,
            keep=_is_pointer_write,
            needle=needle,
            cache=cache,
            cache_scope=scope,
            today=today,
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


# ── alpha-engine-config-I10609: which stack applies were HUMAN ─────────────
# Brian ruled option (a) on `alpha-engine-config-I10609`: the `crucible-v2`
# CloudFormation apply is automated on merge under a least-privilege machine
# role, as five sibling stacks in `nous-ergon-ops` already are. That makes
# "who applied the stack" exactly the question :func:`attribute_pointer_writes`
# answers for the release pointer, and it has to be answered the same way —
# from the archive — because `DescribeStacks` reports WHEN the stack last
# changed and never WHO changed it. A stack carries no writer any more than
# the pointer document does.
#
# The machine allowlist needs no extension to cover the new role: it is
# DERIVED from the stack's own `AWS::IAM::Role` resources
# (:func:`machine_principals`), and the apply role is declared IN that
# template, so it joins the allowlist the moment the stack is applied. No
# role name is written into this public tree, here or anywhere else.

#: The CloudFormation calls that can change a stack. Broad on purpose, for
#: the same reason `_POINTER_WRITE_EVENTS` is: an attribution set narrower
#: than the API surface fails in the direction that reads clean, and an apply
#: performed by a call this set does not name is UNATTRIBUTABLE — handled by
#: the caller as HUMAN — rather than silently absent. `CreateChangeSet` is
#: included although it changes nothing by itself: it is mutating, it names
#: the stack, and `count_operator_actions` already counts it as a human
#: action, so excluding it here would make the two readings disagree about
#: the same event.
_STACK_APPLY_EVENTS = frozenset(
    {"CreateStack", "UpdateStack", "ExecuteChangeSet", "CreateChangeSet", "DeleteStack"}
)

#: The service the calls above are recorded under.
_CLOUDFORMATION_SOURCE = "cloudformation.amazonaws.com"

#: How far apart the stack's own change instant and the CloudTrail
#: `eventTime` for the call that caused it may sit and still be the same
#: event. Wider than `POINTER_ATTRIBUTION_TOLERANCE` and for a measured
#: reason: the stack instant `crucible.gate._stack_last_updated` reports is
#: the newest RESOURCE settle time, which trails `ExecuteChangeSet` by the
#: length of the apply — 2-4 minutes for this template, and
#: `nous-ergon-ops`'s own stack detector calls an apply STUCK rather than in
#: flight at ten. Fifteen minutes is past the longest apply the fleet treats
#: as healthy; it is slack, not a tolerance being leaned on.
STACK_APPLY_ATTRIBUTION_TOLERANCE = dt.timedelta(minutes=15)

#: How many calendar days back the apply attribution will walk before giving
#: up and answering UNATTRIBUTABLE. The pointer read needs no such bound —
#: its floor is the stack apply, which is by construction recent. This read's
#: floor is the stack's CREATION, which recedes without limit as machine
#: applies accumulate, so an unbounded walk would read months of archive
#: inside the board job's timeout (one production day measured 2026-09-23 at
#: ~1,750 objects and 28.5 s wall unfiltered at `_ARCHIVE_WORKERS`
#: concurrency, `alpha-engine-config-I11448`). Reaching the bound resolves
#: to UNATTRIBUTABLE, which the caller reads as HUMAN — the strict direction,
#: never "no human applied it".
_STACK_ATTRIBUTION_MAX_DAYS = 45


@dataclass(frozen=True)
class StackApply:
    """One archived call that changed the graded stack, and who issued it."""

    at: dt.datetime
    event_name: str
    principal: str
    principal_type: str
    machine: bool


@dataclass(frozen=True)
class StackApplyAttribution:
    """What the archive could say about who has applied the stack.

    ``unattributable`` is a REASON string when the archive could not settle
    the question and `None` when it could — the same contract
    :class:`PointerAttribution` carries, and for the same reason: the caller
    (`crucible.gate._human_stack_apply`) turns any reason at all into "treat
    the apply as human", which restarts the autonomy window and keeps the
    clause UNMET for longer. Over-counting human changes is the safe
    direction for a clause asserting a count of zero.
    """

    applies: tuple[StackApply, ...]
    latest_human: dt.datetime | None
    unattributable: str | None
    objects_read: int
    records_scanned: int


def _names_stack(record: dict[str, Any], stack_name: str) -> bool:
    """Whether ``record``'s `requestParameters` name ``stack_name``.

    Matched on the parameter rather than by substring — `_touches`'s
    deliberately broad scan would match any record merely mentioning the
    stack — and accepting both spellings, because `ExecuteChangeSet` names
    the stack by ARN where `UpdateStack` names it bare. A matcher that
    compared only the bare name would attribute nothing on the one call that
    actually applies a template.
    """
    params = record.get("requestParameters")
    if not isinstance(params, dict):
        return False
    named = str(params.get("stackName") or "")
    return named == stack_name or f":stack/{stack_name}/" in named


def attribute_stack_applies(
    client: Any,
    *,
    bucket: str,
    prefix: str,
    stack_name: str,
    since: dt.datetime,
    until: dt.datetime,
    cfn: Any | None = None,
    cache: DayCache | None = None,
    today: dt.date | None = None,
) -> StackApplyAttribution:
    """Who applied ``stack_name`` in ``(since, until]``, from the archive.

    ``bucket``/``prefix`` locate the CloudTrail archive. ``since`` is the
    floor the answer has to beat — the stack's CREATION, whose own apply was
    the operator bootstrap and is human by construction. The walk goes
    BACKWARD from ``until`` and stops at the first day carrying a human
    apply: the latest human apply is on the newest day that has one, and
    nothing older can change the answer.

    A day the trail delivered nothing for is `unattributable`, not "nobody
    applied the stack that day": a trail always delivers, so an empty day
    means the trail does not cover it, and counting a gap as silence is the
    §11 risk 8 failure with a different cause. So is running out of
    :data:`_STACK_ATTRIBUTION_MAX_DAYS` without reaching ``since``.
    """
    if not bucket:
        raise ArchiveMissingError(
            "no CloudTrail archive bucket is configured, so no stack apply can be "
            "attributed to a principal."
        )

    def _is_stack_apply(record: dict[str, Any]) -> bool:
        if record.get("eventSource") != _CLOUDFORMATION_SOURCE:
            return False
        if record.get("eventName") not in _STACK_APPLY_EVENTS:
            return False
        return _names_stack(record, stack_name)

    # `alpha-engine-config-I11448`: ``_is_stack_apply`` accepts a record only
    # when ``eventSource == _CLOUDFORMATION_SOURCE`` — a JSON STRING (no other
    # JSON value compares equal to a str) whose decoded text IS the needle.
    # The constant is printable ASCII with no character JSON must escape, so
    # the raw bytes carry it literally unless a ``\u`` escape is present.
    # The stack name would be an equally sound needle (`_names_stack` requires
    # it as a substring of `stackName`), but measured 2026-09-21 it matched
    # 216 of ~1,664 objects against this one's 116, and one needle suffices.
    needle = literal_needle(_CLOUDFORMATION_SOURCE)
    # Names `_is_stack_apply` and the needle exactly (I11792). The machine
    # allowlist is applied below, on every read, never cached.
    scope = (
        f"stack-applies/v1/source={_CLOUDFORMATION_SOURCE}"
        f"/events={','.join(sorted(_STACK_APPLY_EVENTS))}/stack={stack_name}/needle={needle!r}"
    )
    principals: tuple[str, ...] | None = None
    applies: list[StackApply] = []
    objects_read = 0
    records_scanned = 0
    day = until.date()
    floor_day = since.date()
    scanned_days = 0
    while day >= floor_day:
        if scanned_days >= _STACK_ATTRIBUTION_MAX_DAYS:
            return StackApplyAttribution(
                applies=tuple(sorted(applies, key=lambda a: a.at)),
                latest_human=None,
                unattributable=(
                    f"scanned {scanned_days} calendar day(s) back from "
                    f"{until.date().isoformat()} without reaching the stack's creation "
                    f"({floor_day.isoformat()}) or finding a human apply, which is this "
                    "read's bound — so who last applied the stack by hand is unknown"
                ),
                objects_read=objects_read,
                records_scanned=records_scanned,
            )
        read = iter_archive_records(
            client,
            bucket=bucket,
            prefix=prefix,
            start=day,
            end=day,
            keep=_is_stack_apply,
            needle=needle,
            cache=cache,
            cache_scope=scope,
            today=today,
        )
        objects_read += read.objects_read
        records_scanned += read.records_scanned
        scanned_days += 1
        if read.uncovered_days:
            return StackApplyAttribution(
                applies=tuple(sorted(applies, key=lambda a: a.at)),
                latest_human=None,
                unattributable=(
                    f"the CloudTrail archive s3://{bucket}/{prefix} delivered no objects "
                    f"for {day.isoformat()}, so who applied the stack that day cannot be "
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
                return StackApplyAttribution(
                    applies=tuple(sorted(applies, key=lambda a: a.at)),
                    latest_human=None,
                    unattributable=(
                        f"a stack apply carried an eventTime that will not parse "
                        f"({record.eventTime!r}), so it cannot be placed relative to the "
                        "window's floor"
                    ),
                    objects_read=objects_read,
                    records_scanned=records_scanned,
                )
            if principals is None:
                # Derived lazily and ONCE, exactly as the pointer read does: a
                # span with no apply at all must not require a CloudFormation
                # call to answer.
                principals = machine_principals(cfn)
            name, kind = _principal(record)
            applied = StackApply(
                at=instant,
                event_name=record.eventName,
                principal=name,
                principal_type=kind,
                machine=name in principals,
            )
            applies.append(applied)
            if not applied.machine and applied.at > since:
                found_human = True
        if found_human:
            break
        day -= dt.timedelta(days=1)
    ordered = tuple(sorted(applies, key=lambda a: a.at))
    humans = [a.at for a in ordered if not a.machine and a.at > since]
    if humans:
        return StackApplyAttribution(ordered, max(humans), None, objects_read, records_scanned)
    if not any(abs(a.at - until) <= STACK_APPLY_ATTRIBUTION_TOLERANCE for a in ordered):
        return StackApplyAttribution(
            applies=ordered,
            latest_human=None,
            unattributable=(
                f"the stack's own change instant {until.isoformat()} matches no archived "
                f"{'/'.join(sorted(_STACK_APPLY_EVENTS))} on stack {stack_name!r} within "
                f"{STACK_APPLY_ATTRIBUTION_TOLERANCE}, so the principal behind the change "
                f"that is actually there is unknown ({len(ordered)} stack apply call(s) "
                "were archived over the scanned days)"
            ),
            objects_read=objects_read,
            records_scanned=records_scanned,
        )
    return StackApplyAttribution(ordered, None, None, objects_read, records_scanned)
