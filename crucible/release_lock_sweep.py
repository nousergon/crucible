"""Read-only sweep for a published release object with no Object Lock retention.

Normative source: alpha-engine-config-I9787 (the write-time fix) and
alpha-engine-config-I9798 (this sweep).

I9787 made :meth:`crucible.store.S3Store.put_bytes` accept
``object_lock_mode`` / ``object_lock_retain_until`` and pass them on the
``PutObject`` itself, closing the window a process dying mid-publish used to
leave open. It does not detect a release object published under the old
two-call sequence (2026-09-01 or earlier), or one whose lock was lost some
other way — nothing checked for that until this module.

Lives outside `crucible.store` and `crucible.release` on purpose — both are
frozen to the generic :class:`~crucible.store.Store` interface by design
(`Store`'s own class docstring: "anything larger is a backend feature
leaking into the callers and pinning us to one provider"). Object Lock
retention is exactly such a backend feature: it has no `LocalStore` analogue
at all, so this module reaches `S3Store.client` directly, rather than
widening `Store` for a single caller.

**`GetObjectRetention`, never `HeadObject`, for the retention fields**
(alpha-engine-config-I9798 round 2, adversarial review). `HeadObject`
returns 200 with `ObjectLockMode`/`ObjectLockRetainUntilDate` simply
*omitted* when the caller lacks `s3:GetObjectRetention` — a role change
that narrowed permissions would silently flip every future run's "no
`ObjectLockMode`" into a false UNMET, indistinguishable from a genuinely
unlocked object. `GetObjectRetention` is the API S3 dedicates to exactly
this question: it raises a *distinct*, named error
(`NoSuchObjectLockConfiguration`) when the object truly has no retention,
so "denied" and "absent" are two different exceptions rather than one
ambiguous 200. `HeadObject` is still called once per key, but only for
`LastModified` — the retention-shorter-than-policy comparison's baseline —
never for the lock fields themselves.

A read is a **typed reading**, never a swallowed exception. An
`AccessDenied` from either call is UNMEASURABLE, not UNMET: reporting it as
UNMET would page for a retention that may well be present and simply
unreachable from here, which is the same "absence read as a page" defect
`S3Store`'s own docstring calls out for `exists`/`get_bytes` — a statement
about *our* access, never rendered as a statement about the object. A
`GetObjectRetention` failure that names the *key* rather than our access
(e.g. `NoSuchKey` — the release was deleted between the `list_keys` walk and
this read) is UNMEASURABLE too, but named as the race it is, distinctly from
an access failure: an operator reading "AccessDenied" would go check IAM,
which is the wrong action for a key that simply is not there any more.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import Any, Literal

from crucible.release import (
    POINTER_KEY,
    RELEASE_OBJECT_LOCK_RETENTION,
    RETENTION_CLOCK_SKEW_SLACK,
    retention_meets_target,
)
from crucible.store import S3Store, Store

__all__ = [
    "ReleaseLockReading",
    "ReleaseLockState",
    "release_lock_findings",
    "release_lock_metric",
]

ReleaseLockState = Literal["MET", "UNMET", "UNMEASURABLE"]

#: A key under `releases/{sha}/` that is a published release IDENTITY
#: object — the wheel or `release.json` — and nothing else. Never a
#: provenance record (one per publish ATTEMPT, never lock-checked:
#: `crucible.release.ReleaseProvenance`'s own docstring) and never
#: `releases/current` (a pointer, deliberately unlocked —
#: `crucible.release.POINTER_KEY`'s docstring; it also has no `{sha}/`
#: segment, so it cannot match this pattern regardless).
#:
#: The wheel half matches ANY `crucible-*.whl` directly under the sha
#: prefix, not a filename derived from the sha (alpha-engine-config-I9908):
#: the wheel is now named by its PEP 440 version
#: (`crucible.release.wheel_filename_for`), which embeds only the sha's
#: first 12 hex characters as a local version segment, not the full
#: 40-character sha this regex's own directory segment captures — a
#: same-sha backreference on the filename half can never match the real
#: object again. A release published before I9908 was named
#: `crucible-{sha40}-py3-none-any.whl` (also matched here, since that name
#: still starts with `crucible-` and ends in `.whl`), so this sweep keeps
#: covering every release object ever published, old naming and new, with
#: one pattern rather than two.
#:
#: The filename half is `[^/]+`, never `.+` (alpha-engine-config-I9917 item
#: 2): `.` matches `/`, so `.+` made the pattern match at ANY depth below the
#: sha prefix as long as the first segment started with `crucible-` — e.g.
#: `releases/{sha}/crucible-staging/inner/build.whl`. That contradicted this
#: docstring's own "directly under the sha prefix" invariant, and
#: `publish_release` writes exactly two keys there, so anything deeper is not
#: a published release object: reporting a lock finding against one would
#: grade something the release contract does not own and inflate the sweep's
#: denominator with it.
_RELEASE_OBJECT_RE = re.compile(
    r"^releases/(?P<sha>[0-9a-f]{40})/(?:release\.json|crucible-[^/]+\.whl)$"
)

#: `GetObjectRetention`'s error code for "this object has no Object Lock
#: retention set" — the genuinely-unlocked case, distinct from every other
#: `ClientError` this call can raise (access denial, the key having
#: disappeared between the list and this read).
_NO_RETENTION_ERROR_CODE = "NoSuchObjectLockConfiguration"

#: The `releases/` root, derived from `crucible.release.POINTER_KEY`
#: ("releases/current") rather than restated as a fresh literal —
#: `crucible/keys.py` (alpha-engine-config-I9875, a live concurrent PR owns
#: it) has no `RELEASES_ROOT` constant today; naming the gap here rather
#: than editing `keys.py` avoids a conflicting concurrent edit to a file
#: this track does not own. `tests/test_no_inline_store_keys.py` requires
#: `list_keys`'s first argument to be a `Name`/`Attribute`/`Call`, never an
#: inline string constant — this satisfies that by construction, not by
#: coincidence.
_RELEASES_ROOT = POINTER_KEY.rsplit("/", 1)[0] + "/"


@dataclass(frozen=True)
class ReleaseLockReading:
    """One key's Object Lock reading. Never a bare bool — a bool cannot
    distinguish "read, and found unlocked" from "could not read"."""

    key: str
    state: ReleaseLockState
    detail: str

    def __post_init__(self) -> None:
        if self.state not in ("MET", "UNMET", "UNMEASURABLE"):
            raise ValueError(f"{self.state!r} is not a release-lock reading state")
        if not self.detail.strip():
            raise ValueError(f"{self.key}: a {self.state} reading with no detail is not actionable")


def _release_object_keys(store: Store) -> list[str]:
    """Every published release identity object, sorted for determinism.

    Filters `store.list_keys("releases/")` to keys the sha-scoped release
    identity pattern matches — never a bare suffix check, which cannot tell
    a release identity object from a same-named object living somewhere
    else under the prefix.
    """
    return sorted(key for key in store.list_keys(_RELEASES_ROOT) if _RELEASE_OBJECT_RE.match(key))


def _read_one(store: S3Store, key: str) -> ReleaseLockReading:
    from botocore.exceptions import ClientError  # noqa: PLC0415 - lazy, mirrors crucible.store

    s3_key = store._s3_key(key)

    # `HeadObject` first, but ONLY for `LastModified` — the baseline the
    # retention-shorter-than-policy comparison needs. Never read
    # `ObjectLockMode`/`ObjectLockRetainUntilDate` from this response: S3
    # omits both, silently, on a 200, when the caller lacks
    # `s3:GetObjectRetention` (round 2 finding 1) — indistinguishable from a
    # genuinely unlocked object unless the lock fields are never trusted
    # from this call at all.
    try:
        head = store.client.head_object(Bucket=store.bucket, Key=s3_key)
    except ClientError as exc:
        code = store._error_code(exc)
        return ReleaseLockReading(
            key,
            "UNMEASURABLE",
            f"head_object({key}) failed: {code or type(exc).__name__}. This is a "
            "statement about our access, not about the object's retention, and it "
            "must never be counted as an UNMET finding.",
        )
    last_modified = head.get("LastModified")

    try:
        retention_response = store.client.get_object_retention(Bucket=store.bucket, Key=s3_key)
    except ClientError as exc:
        code = store._error_code(exc)
        if code == "AccessDenied":
            return ReleaseLockReading(
                key,
                "UNMEASURABLE",
                f"get_object_retention({key}) denied: {code}. This is a statement "
                "about our access, not about the object's retention, and it must "
                "never be counted as an UNMET finding.",
            )
        if code == _NO_RETENTION_ERROR_CODE:
            return ReleaseLockReading(
                key,
                "UNMET",
                f"{key}: no Object Lock retention (get_object_retention: {code})",
            )
        if code == "NoSuchKey":
            # The one code that names the RACE, not our access or the
            # object's state: `head_object` just confirmed the key exists,
            # so `get_object_retention` losing it a moment later is a
            # release deleted between the `list_keys` walk and this read —
            # distinct wording from every other failure below, since an
            # operator reading a generic "failed" would go check IAM, the
            # wrong action for a key that is simply gone (round 2,
            # should-fix finding 3).
            return ReleaseLockReading(
                key,
                "UNMEASURABLE",
                f"get_object_retention({key}) returned {code}, right after "
                f"head_object({key}) succeeded — the object was very likely deleted "
                "between the list and this read (a release deleted mid-sweep), not a "
                "statement about its retention.",
            )
        # Every OTHER ClientError — Throttling, InternalError, an SDK/API
        # code this module has never seen — is neither "denied", "no
        # retention" nor "gone". Round 2 re-verification: labelling ALL of
        # these as a mid-sweep deletion gave a false diagnosis for a
        # transient failure like Throttling. Generic and accurate: the call
        # failed, naming the code, and nothing more is claimed about why.
        return ReleaseLockReading(
            key,
            "UNMEASURABLE",
            f"get_object_retention({key}) failed: {code or type(exc).__name__}. This "
            "is a statement about the call failing, not about the object's "
            "retention, and it must never be counted as an UNMET finding.",
        )

    retention = retention_response.get("Retention") or {}
    mode = retention.get("Mode")
    retain_until = retention.get("RetainUntilDate")
    if not mode or retain_until is None:
        # `GetObjectRetention` succeeding without an error is contractually
        # supposed to always carry both fields (unlike `HeadObject`, which
        # can omit them silently). A success response missing them is a
        # malformed answer this module cannot make sense of — UNMEASURABLE,
        # never UNMET, so a genuine future API/SDK shape change is loud
        # rather than read as "confirmed unlocked".
        return ReleaseLockReading(
            key,
            "UNMEASURABLE",
            f"get_object_retention({key}) succeeded but returned no Mode/"
            f"RetainUntilDate ({retention!r}) — a malformed response, not a "
            "confirmed absence of retention.",
        )
    if last_modified is not None:
        target_retain_until = last_modified.astimezone(dt.UTC) + RELEASE_OBJECT_LOCK_RETENTION
        # `retention_meets_target` (`crucible.release`) — the SAME
        # comparison, with the SAME `RETENTION_CLOCK_SKEW_SLACK`,
        # `crucible.release_retention`'s repair now uses (I9898 round 2):
        # this sweep and that repair can no longer disagree about whether a
        # given object is already compliant.
        if not retention_meets_target(retain_until, target_retain_until):
            return ReleaseLockReading(
                key,
                "UNMET",
                f"{key}: retained until {retain_until.isoformat()}, short of the "
                f"declared {RELEASE_OBJECT_LOCK_RETENTION.days}-day policy "
                f"(published {last_modified.isoformat()}), beyond the "
                f"{RETENTION_CLOCK_SKEW_SLACK.days}-day clock-skew slack.",
            )
    return ReleaseLockReading(key, "MET", f"{key}: {mode} until {retain_until.isoformat()}")


def release_lock_findings(store: Store) -> list[ReleaseLockReading]:
    """Read-only: one typed reading per published release identity object.

    Never against a non-S3 backend's own Object Lock concept —
    `LocalStore` has none at all (`LocalStore.put_bytes` raises rather than
    fake one), so every key is reported UNMEASURABLE there rather than
    silently skipped: an operator who pointed this sweep at the wrong store
    by mistake sees why nothing was checked instead of a clean, wrong
    "0 findings".
    """
    keys = _release_object_keys(store)
    if not isinstance(store, S3Store):
        return [
            ReleaseLockReading(
                key,
                "UNMEASURABLE",
                f"{key}: store is {type(store).__name__}, which has no Object Lock "
                "concept — this sweep is only meaningful against S3Store.",
            )
            for key in keys
        ]
    return [_read_one(store, key) for key in keys]


def release_lock_metric(findings: list[ReleaseLockReading], *, now: dt.datetime) -> dict[str, Any]:
    """A MetricRecord over `findings`.

    BREACH beats `unmeasurable` beats OK — a single unlocked object must
    never be masked by nine unreadable ones. Status values are drawn from
    `run_manifest.v1.json`'s closed `metricRecord.status` enum: `BREACH`/
    `OK` are the coverage-and-ceiling vocabulary `crucible.alerts` and
    `crucible.track_c` already use; `unmeasurable` (lowercase) is the arena's
    own not-measured spelling forwarded verbatim by that same enum — this
    module does not invent a third one.
    """
    unmet = [f for f in findings if f.state == "UNMET"]
    unmeasurable = [f for f in findings if f.state == "UNMEASURABLE"]
    if unmet:
        status = "BREACH"
        status_reason = (
            f"{len(unmet)} of {len(findings)} release object(s) lack Object Lock "
            f"retention: {', '.join(f.key for f in unmet)}"
        )
    elif unmeasurable:
        status = "unmeasurable"
        status_reason = (
            f"{len(unmeasurable)} of {len(findings)} release object(s) could not be "
            f"read for retention: {', '.join(f.key for f in unmeasurable)}"
        )
    else:
        status = "OK"
        status_reason = f"{len(findings)} release object(s), all Object Lock retained"
    return {
        "name": "release_objects_unlocked",
        "module": "crucible.release_lock_sweep",
        "metric_type": "operational",
        "value": float(len(unmet)),
        "unit": "objects",
        "n_floor": 0,
        "n_samples": len(findings),
        "status": status,
        "status_reason": status_reason,
        "source_path": "releases/{sha}/release.json",
        "last_updated_utc": now.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
