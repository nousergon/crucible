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
at all, so this module reaches `S3Store.client.head_object` directly — the
same way `release.py`'s pre-I9787 `_lock_release_object` reached the client
— rather than widening `Store` for a single caller.

A read is a **typed reading**, never a swallowed exception. A
`head_object` `AccessDenied` is UNMEASURABLE, not UNMET: reporting it as
UNMET would page for a retention that may well be present and simply
unreachable from here, which is the same "absence read as a page" defect
`S3Store`'s own docstring calls out for `exists`/`get_bytes` — a statement
about *our* access, never rendered as a statement about the object.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import Any, Literal

from crucible.release import POINTER_KEY, RELEASE_OBJECT_LOCK_RETENTION
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
#: segment, so it cannot match this pattern regardless). The sha is
#: back-referenced into the wheel's own filename so a key whose directory
#: and filename name two different shas — which should never happen, but a
#: sweep exists to catch "should never happen" — is not silently treated as
#: this sha's own wheel.
_RELEASE_OBJECT_RE = re.compile(
    r"^releases/(?P<sha>[0-9a-f]{40})/(?:release\.json|crucible-(?P=sha)-py3-none-any\.whl)$"
)

#: Slack subtracted from the declared minimum retain-until before comparing
#: it against `HeadObject`'s `ObjectLockRetainUntilDate`. `release.py` stamps
#: `retain_until` as `now() + RELEASE_OBJECT_LOCK_RETENTION` at publish time,
#: and `HeadObject`'s `LastModified` is that same publish instant as S3
#: recorded it — the two should agree exactly, but a reading that failed on
#: sub-second clock skew between the SDK call and S3's own stamp would be a
#: false UNMET about US, not about the retention. A day is generous against
#: a single request round trip and costs nothing: a retention genuinely short
#: by a day is still short against the plan's ten-year horizon.
_CLOCK_SKEW_SLACK = dt.timedelta(days=1)

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

    try:
        response = store.client.head_object(Bucket=store.bucket, Key=store._s3_key(key))
    except ClientError as exc:
        code = store._error_code(exc)
        return ReleaseLockReading(
            key,
            "UNMEASURABLE",
            f"head_object({key}) failed: {code or type(exc).__name__}. This is a "
            "statement about our access, not about the object's retention, and it "
            "must never be counted as an UNMET finding.",
        )
    mode = response.get("ObjectLockMode")
    retain_until = response.get("ObjectLockRetainUntilDate")
    if not mode or retain_until is None:
        return ReleaseLockReading(
            key, "UNMET", f"{key}: no Object Lock retention (ObjectLockMode={mode!r})"
        )
    last_modified = response.get("LastModified")
    if last_modified is not None:
        floor = last_modified.astimezone(dt.UTC) + RELEASE_OBJECT_LOCK_RETENTION - _CLOCK_SKEW_SLACK
        if retain_until.astimezone(dt.UTC) < floor:
            return ReleaseLockReading(
                key,
                "UNMET",
                f"{key}: retained until {retain_until.isoformat()}, short of the "
                f"declared {RELEASE_OBJECT_LOCK_RETENTION.days}-day policy "
                f"(published {last_modified.isoformat()})",
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
