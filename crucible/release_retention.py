"""`crucible release.lock <sha>` — apply Object Lock retention to a release
published before the write-time fix.

Normative source: alpha-engine-config-I9898.

`alpha-engine-config-I9787` made :func:`crucible.release.publish_release`
apply Object Lock retention on the SAME `PutObject` that writes a release
identity object, closing the two-call-sequence window a process dying
mid-publish used to leave open. It does nothing for a release object
published BEFORE that fix landed — `crucible-PR56`'s read-only sweep
(`crucible.release_lock_sweep`) found exactly one: the wheel and
`release.json` of `cafc3947faaed4258d11ca7526428b06b0a23b30`, both with no
Object Lock retention at all.

This module is the REPAIR for that one class of finding, run as a job rather
than a bare `aws s3api put-object-retention`: a hand-run CLI call leaves no
manifest, no lineage, and after phase 2 it counts as a human mutating call
the console cannot see. `crucible release.lock <sha>` runs through
`crucible.runner.run_job` like every other job, so a repair leaves the same
trail a publish does.

**Scope: the two release IDENTITY objects, never provenance.**
`crucible.release.publish_release` applies Object Lock to exactly the wheel
and `release.json` — `crucible.release.ReleaseProvenance` is written
"Unconditional and unlocked" (`publish_release`'s own docstring): one record
per publish ATTEMPT, deliberately never lock-checked. Locking it here would
lock an object the writer itself designed to stay mutable-by-convention (it
is never overwritten in practice, but nothing declares it immutable the way
`release.json`/the wheel are). So this job reuses
`crucible.release_lock_sweep`'s own identity-object pattern
(`_RELEASE_OBJECT_RE`) to select exactly the same two keys per sha that the
sweep already reads and `publish_release` already locks — "every object
under `releases/{sha}/`" in the issue's deliverable is, for THIS release
layout, exactly those two.

**Exactly what publish would have applied, computed from the object's own
`LastModified`, never restated.** `crucible.release.release_object_lock_params`
is the one function that knows the mode and period
(`RELEASE_OBJECT_LOCK_RETENTION`, GOVERNANCE) — passing it the object's
`LastModified` as `now` reproduces precisely the `(mode, retain_until)` a
publish AT THAT INSTANT would have computed, which is what a repair is: not
"lock it for ten years from today", but "apply what publish should have
applied, back-dated to when it ran".

**Never shortens.** GOVERNANCE retention can be extended freely but shortened
only with `s3:BypassGovernanceRetention` (never used here); COMPLIANCE cannot
be shortened by anyone, ever. The comparison below is written so a shorter
computed target than what is already on the object is simply left alone —
`unchanged`, not attempted — rather than relying on S3 to reject the call.
And a `PutObjectRetention` extending an already-locked object reuses that
object's OWN current mode rather than substituting GOVERNANCE: an object
already under COMPLIANCE cannot be moved to GOVERNANCE (S3 refuses it), and
this job's job is to close a retention gap, never to weaken or alter an
existing lock's mode.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import asdict, dataclass
from typing import Literal

from crucible.release import (
    RELEASE_OBJECT_LOCK_RETENTION,
    release_json_key,
    release_object_lock_params,
    release_prefix,
    retention_meets_target,
)
from crucible.release_lock_sweep import _RELEASE_OBJECT_RE
from crucible.runner import RunContext, run_job
from crucible.store import S3Store, Store, open_store

__all__ = [
    "RELEASE_LOCK_JOB",
    "RETENTION_APPLIED",
    "RETENTION_EXTENDED",
    "RETENTION_UNCHANGED",
    "ReleaseHasNoRecordError",
    "RetentionApplyResult",
    "apply_release_retention",
    "release_lock_handler",
    "release_lock_object_keys",
]

#: `crucible release.lock <sha>`. Named to match `release.pin`'s own
#: `release.<verb>` shape (cli.py's existing convention), not `release.repair`
#: or a fresh vocabulary for what is, mechanically, the same "point/lock one
#: release" family of one-shot operator jobs.
RELEASE_LOCK_JOB = "release.lock"

RETENTION_APPLIED: Literal["applied"] = "applied"
RETENTION_EXTENDED: Literal["extended"] = "extended"
RETENTION_UNCHANGED: Literal["unchanged"] = "unchanged"

RetentionAction = Literal["applied", "extended", "unchanged"]


class ReleaseHasNoRecordError(RuntimeError):
    """``sha`` has no `release.json` — refuse. A repair with nothing to
    repair against would either no-op silently or, worse, apply retention to
    a stray object under a prefix nothing ever published."""


@dataclass(frozen=True)
class RetentionApplyResult:
    """One object's before/after. Never a bare bool — same reasoning as
    :class:`crucible.release_lock_sweep.ReleaseLockReading`: a caller (and a
    manifest reader) needs to see WHAT changed, not just that something ran.
    """

    key: str
    action: RetentionAction
    before_mode: str | None
    before_retain_until: str | None
    after_mode: str
    after_retain_until: str
    detail: str

    def __post_init__(self) -> None:
        if self.action not in (RETENTION_APPLIED, RETENTION_EXTENDED, RETENTION_UNCHANGED):
            raise ValueError(f"{self.action!r} is not a retention-apply action")


def release_lock_object_keys(store: Store, sha: str) -> list[str]:
    """The wheel and `release.json` keys for ``sha``, sorted for determinism.

    Reuses `crucible.release_lock_sweep`'s own identity-object pattern rather
    than re-deriving which two keys under `releases/{sha}/` are lockable —
    see the module docstring for why provenance is excluded.
    """
    return sorted(
        key for key in store.list_keys(release_prefix(sha)) if _RELEASE_OBJECT_RE.match(key)
    )


def _current_retention(store: S3Store, s3_key: str) -> tuple[str | None, dt.datetime | None]:
    """``(mode, retain_until)``, or ``(None, None)`` when the object carries
    no Object Lock retention at all.

    Mirrors `crucible.release_lock_sweep._read_one`'s own reasoning for using
    `GetObjectRetention` rather than `HeadObject`: `HeadObject` omits the
    lock fields silently on a 200 when the caller lacks
    `s3:GetObjectRetention`, indistinguishable from a genuinely unlocked
    object. `NoSuchObjectLockConfiguration` is the one error this function
    treats as "confirmed absent"; every other `ClientError` propagates,
    because a denial or a transient failure read as "absent" here would let
    this job apply a *second*, possibly weaker, lock over one it simply could
    not see — the opposite of a repair.
    """
    from botocore.exceptions import ClientError  # noqa: PLC0415 - lazy, mirrors crucible.store

    try:
        resp = store.client.get_object_retention(Bucket=store.bucket, Key=s3_key)
    except ClientError as exc:
        if store._error_code(exc) == "NoSuchObjectLockConfiguration":
            return None, None
        raise
    retention = resp.get("Retention") or {}
    return retention.get("Mode"), retention.get("RetainUntilDate")


def _plan_one(store: S3Store, key: str) -> tuple[RetentionApplyResult, dict[str, object] | None]:
    """The reading for ``key`` and, when a write is needed, the
    `PutObjectRetention` kwargs to make it. Returns ``(result, None)`` when
    nothing needs writing — the caller never calls `put_object_retention` on
    a ``None`` plan, which is what makes a dry run and the idempotent-rerun
    case share this exact function with the real apply path.
    """
    s3_key = store._s3_key(key)
    head = store.client.head_object(Bucket=store.bucket, Key=s3_key)
    last_modified = head["LastModified"]
    target_mode, target_retain_until = release_object_lock_params(store, now=last_modified)
    # S3Store, so release_object_lock_params never returns (None, None) here.
    assert target_mode is not None
    assert target_retain_until is not None

    current_mode, current_retain_until = _current_retention(store, s3_key)

    if current_mode is not None and current_retain_until is not None:
        if retention_meets_target(current_retain_until, target_retain_until):
            return (
                RetentionApplyResult(
                    key,
                    RETENTION_UNCHANGED,
                    current_mode,
                    current_retain_until.isoformat(),
                    current_mode,
                    current_retain_until.isoformat(),
                    f"{key}: already retained under {current_mode} until "
                    f"{current_retain_until.isoformat()}, which meets or exceeds the "
                    f"{RELEASE_OBJECT_LOCK_RETENTION.days}-day policy computed from "
                    f"this object's own publish instant ({last_modified.isoformat()}). "
                    "Never shortened.",
                ),
                None,
            )
        # Extend in place: the object's OWN current mode, never substituted
        # for `target_mode` — an object already under COMPLIANCE cannot be
        # moved to GOVERNANCE, and this job only ever closes a retention
        # gap, never changes an existing lock's mode.
        write_mode = current_mode
        result = RetentionApplyResult(
            key,
            RETENTION_EXTENDED,
            current_mode,
            current_retain_until.isoformat(),
            write_mode,
            target_retain_until.isoformat(),
            f"{key}: retained under {current_mode} until "
            f"{current_retain_until.isoformat()}, short of the "
            f"{RELEASE_OBJECT_LOCK_RETENTION.days}-day policy computed from this "
            f"object's own publish instant ({last_modified.isoformat()}) — extending to "
            f"{target_retain_until.isoformat()}.",
        )
    else:
        write_mode = target_mode
        result = RetentionApplyResult(
            key,
            RETENTION_APPLIED,
            None,
            None,
            write_mode,
            target_retain_until.isoformat(),
            f"{key}: no Object Lock retention. Applying {write_mode} until "
            f"{target_retain_until.isoformat()} — exactly what "
            "crucible.release.release_object_lock_params would have applied at this "
            f"object's own publish instant ({last_modified.isoformat()}).",
        )
    return result, {
        "Bucket": store.bucket,
        "Key": s3_key,
        "Retention": {"Mode": write_mode, "RetainUntilDate": target_retain_until},
    }


def apply_release_retention(
    store: Store, sha: str, *, dry_run: bool = False
) -> list[RetentionApplyResult]:
    """Apply (or, with ``dry_run=True``, plan) retention for every release
    identity object of ``sha``.

    Raises :class:`ReleaseHasNoRecordError` when `release.json` for ``sha``
    is absent — never applies retention to a stray object under a prefix
    nothing ever published. Raises whatever the underlying `ClientError` is
    when `PutObjectRetention` fails (an `AccessDenied` included) — the job
    FAILS rather than swallowing it; a repair that could not confirm its own
    write happened must never report success.

    Idempotent: a second call against an already-repaired release returns
    every reading as `unchanged` and issues no `PutObjectRetention` calls at
    all — `_plan_one` returns a ``None`` write plan for exactly that case, so
    there is nothing for this function to have gotten wrong the second time.
    """
    if not isinstance(store, S3Store):
        raise TypeError(
            f"release.lock is only meaningful against S3Store — {type(store).__name__} "
            "has no Object Lock concept at all (crucible.store.LocalStore.put_bytes "
            "raises rather than fake one)."
        )
    if not store.exists(release_json_key(sha)):
        raise ReleaseHasNoRecordError(
            f"{sha}: no {release_json_key(sha)} — refusing. There is nothing published "
            "for this sha to apply retention to."
        )

    results: list[RetentionApplyResult] = []
    for key in release_lock_object_keys(store, sha):
        result, write_kwargs = _plan_one(store, key)
        if write_kwargs is not None and not dry_run:
            store.client.put_object_retention(**write_kwargs)
        results.append(result)
    return results


def _store(args: argparse.Namespace) -> Store:
    return open_store(getattr(args, "store", None))


def release_lock_handler(args: argparse.Namespace) -> int:
    """`crucible release.lock <sha> [--dry-run]`.

    **`--dry-run` never calls `run_job` and writes nothing** — same shape as
    `crucible.track_a`'s data jobs: it resolves the same plan a real apply
    would (:func:`apply_release_retention` with ``dry_run=True``, which
    reads current retention but never calls `PutObjectRetention`) and prints
    it, so an operator — or this PR's own evidence — can see exactly what
    would change against the live store before anything is mutated.

    The real apply runs through `run_job` like every other job: an
    `AccessDenied` (or any other) failure from `PutObjectRetention` inside
    `apply_release_retention` propagates out of the job body uncaught, which
    is what makes `run_job` write a `failed` manifest naming the reason —
    never a swallow, and never a manifest claiming `ok` for a repair that did
    not actually happen.
    """
    store = _store(args)
    sha = args.sha

    if args.dry_run:
        results = apply_release_retention(store, sha, dry_run=True)
        print(
            json.dumps(
                {"sha": sha, "dry_run": True, "objects": [asdict(r) for r in results]}, indent=2
            )
        )
        return 0

    def body(ctx: RunContext) -> None:
        results = apply_release_retention(store, sha, dry_run=False)
        changed = [r for r in results if r.action != RETENTION_UNCHANGED]
        for r in results:
            # One MetricRecord per object, carrying its own before/after in
            # `status_reason` — the deliverable's "record each object ...
            # with before/after", satisfied per-object rather than folded
            # into one aggregate line an operator would have to unpack.
            ctx.record_metric(
                {
                    "name": "release_object_retention",
                    "module": "crucible.release_retention",
                    "metric_type": "operational",
                    "value": 1.0 if r.action != RETENTION_UNCHANGED else 0.0,
                    "unit": "objects",
                    "n_floor": 0,
                    "status": "OK",
                    "status_reason": r.detail,
                    "source_path": r.key,
                    "last_updated_utc": ctx.started.astimezone(dt.UTC).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    ),
                }
            )
        ctx.record_metric(
            {
                "name": "release_objects_locked",
                "module": "crucible.release_retention",
                "metric_type": "operational",
                "value": float(len(changed)),
                "unit": "objects",
                "n_floor": 0,
                "status": "OK",
                "status_reason": (
                    f"{len(changed)} of {len(results)} object(s) for {sha} needed a "
                    "retention write on this run"
                    + (
                        ": " + "; ".join(f"{r.key} ({r.action})" for r in changed)
                        if changed
                        else " — already fully retained; a second run against an "
                        "already-repaired release is a clean idempotent no-op."
                    )
                ),
                "source_path": release_json_key(sha),
                "last_updated_utc": ctx.started.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )
        print(json.dumps({"sha": sha, "objects": [asdict(r) for r in results]}, indent=2))

    run_job(
        RELEASE_LOCK_JOB,
        body,
        store=store,
        trading_day=args.trading_day,
        release_sha=sha,
        # `discriminator=sha` (alpha-engine-config-I9781's parameter,
        # already used by `crucible.track_c`): without it, two repairs on
        # one trading day overwrite one another's
        # `runs/release.lock/{trading_day}/run.json` — rule 1, manifest or
        # it did not happen, defeated at the second invocation. A repair
        # session over `crucible/releases/`'s five unprovenanced shas is the
        # expected multi-sha use, not a hypothetical.
        discriminator=sha,
    )
    return 0
