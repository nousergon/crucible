"""alpha-engine-config-I9798: the sweep for a published release object with
no Object Lock retention — asserted against a fake S3 client, never a mock
of `crucible.release_lock_sweep` itself.
"""

from __future__ import annotations

import datetime as dt

from botocore.exceptions import ClientError

from crucible.release import RELEASE_OBJECT_LOCK_RETENTION, release_json_key, wheel_key
from crucible.release_lock_sweep import (
    ReleaseLockReading,
    release_lock_findings,
    release_lock_metric,
)
from crucible.store import LocalStore, S3Store

SHA_LOCKED = "a" * 40
SHA_UNLOCKED = "b" * 40
SHA_DENIED = "c" * 40
SHA_HEAD_ONLY_DENIED = "d" * 40
SHA_MALFORMED = "e" * 40
SHA_RACE = "f" * 40

_PUBLISHED_AT = dt.datetime(2026, 8, 1, 12, 0, tzinfo=dt.UTC)


class _FakeS3Client:
    """Models `head_object` and `get_object_retention` as the two SEPARATE
    real S3 calls they are, never one call standing in for the other.

    This is the direct fix for round 2's blocking finding 1:
    `HeadObject` returns 200 with `ObjectLockMode`/`ObjectLockRetainUntilDate`
    *omitted* — never an error — when the caller lacks
    `s3:GetObjectRetention`. A fake that put the lock fields on
    `head_object`'s response (the round-1 shape) could not exercise that
    failure mode at all. Here, `head_object` only ever answers existence and
    `LastModified`; `get_object_retention` is the only source of lock info,
    and it fails in three district ways: `AccessDenied` (denied), the
    `NoSuchObjectLockConfiguration` code (genuinely unlocked), or `NoSuchKey`
    (the object vanished between `list_keys` and this read — the race
    finding 3 asks to be named distinctly).

    Deliberately separate from `tests/test_release.py::_FakeS3Client` — that
    fixture is owned by that module's tests and has no
    `get_object_retention` concept at all.
    """

    def __init__(self) -> None:
        # key -> None (absent entirely) | dict with "LastModified" (existence)
        self.objects: dict[str, dict | None] = {}
        # key -> "AccessDenied" | "NoSuchObjectLockConfiguration" | "NoSuchKey"
        #        | dict with "Mode"/"RetainUntilDate" | {} (malformed 200)
        self.retentions: dict[str, str | dict] = {}

    def put(self, key: str, *, locked: bool = True, retain_until=None, last_modified=None) -> None:
        self.objects[key] = {"LastModified": last_modified or _PUBLISHED_AT}
        if locked:
            self.retentions[key] = {
                "Mode": "GOVERNANCE",
                "RetainUntilDate": retain_until or (_PUBLISHED_AT + RELEASE_OBJECT_LOCK_RETENTION),
            }
        else:
            self.retentions[key] = "NoSuchObjectLockConfiguration"

    def deny_head(self, key: str) -> None:
        self.objects[key] = None

    def deny_retention(self, key: str, *, last_modified=None) -> None:
        self.objects[key] = {"LastModified": last_modified or _PUBLISHED_AT}
        self.retentions[key] = "AccessDenied"

    def malform_retention(self, key: str, *, last_modified=None) -> None:
        """`get_object_retention` succeeds (200) but the payload carries
        neither `Mode` nor `RetainUntilDate` — never observed against real
        S3, but a shape this module must not silently read as UNMET."""
        self.objects[key] = {"LastModified": last_modified or _PUBLISHED_AT}
        self.retentions[key] = {}

    def race(self, key: str, *, last_modified=None) -> None:
        """`head_object` sees the key; by the time `get_object_retention`
        runs, it is gone — a release deleted mid-sweep."""
        self.objects[key] = {"LastModified": last_modified or _PUBLISHED_AT}
        self.retentions[key] = "NoSuchKey"

    def head_object(self, **kw) -> dict:
        key = kw["Key"]
        if key not in self.objects:
            raise ClientError({"Error": {"Code": "404"}}, "head_object")
        response = self.objects[key]
        if response is None:
            raise ClientError({"Error": {"Code": "AccessDenied"}}, "head_object")
        return response

    def get_object_retention(self, **kw) -> dict:
        key = kw["Key"]
        outcome = self.retentions.get(key)
        if isinstance(outcome, str):
            raise ClientError({"Error": {"Code": outcome}}, "get_object_retention")
        return {"Retention": outcome or {}}

    def get_paginator(self, name: str) -> _Paginator:
        assert name == "list_objects_v2"
        return _Paginator(self)


class _Paginator:
    """Enough of boto3's list_objects_v2 paginator for `S3Store.list_keys`.

    A key that will later be denied or raced still LISTS — S3 lists what
    exists regardless of whether a subsequent call on it succeeds — only the
    per-object read fails.
    """

    def __init__(self, fake: _FakeS3Client) -> None:
        self.fake = fake

    def paginate(self, **kw) -> list:
        prefix = kw.get("Prefix") or ""
        keys = sorted(k for k in self.fake.objects if k.startswith(prefix))
        return [{"Contents": [{"Key": k} for k in keys]}]


def _store(client: _FakeS3Client) -> S3Store:
    return S3Store("bucket", "crucible", client=client)


class TestFindings:
    def test_a_locked_object_reads_met(self) -> None:
        client = _FakeS3Client()
        store = _store(client)
        key = release_json_key(SHA_LOCKED)
        client.put(f"crucible/{key}", locked=True)

        findings = release_lock_findings(store)

        assert findings == [ReleaseLockReading(key, "MET", findings[0].detail)]
        assert "GOVERNANCE" in findings[0].detail

    def test_an_unlocked_object_reads_unmet(self) -> None:
        """`get_object_retention` raising `NoSuchObjectLockConfiguration` —
        the genuinely-unlocked case."""
        client = _FakeS3Client()
        store = _store(client)
        key = wheel_key(SHA_UNLOCKED)
        client.put(f"crucible/{key}", locked=False)

        findings = release_lock_findings(store)

        assert len(findings) == 1
        assert findings[0].key == key
        assert findings[0].state == "UNMET"
        assert "no Object Lock retention" in findings[0].detail

    def test_head_object_omitting_lock_fields_on_200_is_never_read(self) -> None:
        """Round 2, blocking finding 1: `HeadObject` returns 200 with the
        lock fields OMITTED, not an error, when the caller lacks
        `s3:GetObjectRetention`. This fake's `head_object` never carries
        lock fields at all — proving the sweep's `_read_one` gets its
        reading from `get_object_retention` alone. A locked object still
        reads MET even though `head_object`'s own response here has no
        `ObjectLockMode`/`ObjectLockRetainUntilDate` key whatsoever."""
        client = _FakeS3Client()
        store = _store(client)
        key = release_json_key(SHA_LOCKED)
        client.put(f"crucible/{key}", locked=True)
        assert "ObjectLockMode" not in client.objects[f"crucible/{key}"]

        findings = release_lock_findings(store)

        assert findings[0].state == "MET"

    def test_get_object_retention_denied_is_unmeasurable_never_unmet(self) -> None:
        client = _FakeS3Client()
        store = _store(client)
        key = release_json_key(SHA_DENIED)
        client.deny_retention(f"crucible/{key}")

        findings = release_lock_findings(store)

        assert len(findings) == 1
        assert findings[0].state == "UNMEASURABLE"
        assert "AccessDenied" in findings[0].detail

    def test_head_object_denied_is_unmeasurable_never_unmet(self) -> None:
        client = _FakeS3Client()
        store = _store(client)
        key = release_json_key(SHA_HEAD_ONLY_DENIED)
        client.deny_head(f"crucible/{key}")

        findings = release_lock_findings(store)

        assert len(findings) == 1
        assert findings[0].state == "UNMEASURABLE"
        assert "head_object" in findings[0].detail

    def test_a_malformed_200_is_unmeasurable_never_unmet(self) -> None:
        """`get_object_retention` succeeding without `Mode`/`RetainUntilDate`
        has no real-S3 precedent, but the sweep must not read a shape it
        cannot make sense of as a confirmed absence of retention."""
        client = _FakeS3Client()
        store = _store(client)
        key = release_json_key(SHA_MALFORMED)
        client.malform_retention(f"crucible/{key}")

        findings = release_lock_findings(store)

        assert findings[0].state == "UNMEASURABLE"
        assert "malformed" in findings[0].detail

    def test_a_release_deleted_between_head_and_retention_is_named_as_a_race(self) -> None:
        """Round 2, should-fix finding 3: `NoSuchKey` from
        `get_object_retention` right after `head_object` succeeded is a
        release deleted mid-sweep, named DISTINCTLY from an access
        failure — an operator reading "AccessDenied" would check IAM, the
        wrong action for a key that is simply gone."""
        client = _FakeS3Client()
        store = _store(client)
        key = release_json_key(SHA_RACE)
        client.race(f"crucible/{key}")

        findings = release_lock_findings(store)

        assert findings[0].state == "UNMEASURABLE"
        assert "deleted" in findings[0].detail
        assert "AccessDenied" not in findings[0].detail

    def test_locked_unlocked_and_denied_together_all_three_readings(self) -> None:
        client = _FakeS3Client()
        store = _store(client)
        locked_key = release_json_key(SHA_LOCKED)
        unlocked_key = wheel_key(SHA_UNLOCKED)
        denied_key = release_json_key(SHA_DENIED)
        client.put(f"crucible/{locked_key}", locked=True)
        client.put(f"crucible/{unlocked_key}", locked=False)
        client.deny_retention(f"crucible/{denied_key}")

        findings = {f.key: f.state for f in release_lock_findings(store)}

        assert findings == {
            locked_key: "MET",
            unlocked_key: "UNMET",
            denied_key: "UNMEASURABLE",
        }

    def test_retention_shorter_than_the_declared_policy_is_unmet(self) -> None:
        client = _FakeS3Client()
        store = _store(client)
        key = release_json_key(SHA_LOCKED)
        client.put(
            f"crucible/{key}",
            locked=True,
            retain_until=_PUBLISHED_AT + dt.timedelta(days=30),
        )

        findings = release_lock_findings(store)

        assert findings[0].state == "UNMET"
        assert "short of the declared" in findings[0].detail

    def test_the_pointer_is_never_checked(self) -> None:
        """`releases/current` never matches the release-object key pattern,
        so it is never even a candidate — checked implicitly, since the fake
        client raises 404 for any key it was not told about and this test
        would fail loudly if the sweep tried to head it."""
        client = _FakeS3Client()
        store = _store(client)
        client.put(f"crucible/{release_json_key(SHA_LOCKED)}", locked=True)

        findings = release_lock_findings(store)

        assert all(f.key != "releases/current" for f in findings)

    def test_a_provenance_record_is_never_checked(self) -> None:
        client = _FakeS3Client()
        store = _store(client)
        provenance_key = f"releases/{SHA_LOCKED}/provenance/1-1.json"
        client.put(f"crucible/{provenance_key}", locked=False)  # would be UNMET if checked
        client.put(f"crucible/{release_json_key(SHA_LOCKED)}", locked=True)

        findings = release_lock_findings(store)

        assert [f.key for f in findings] == [release_json_key(SHA_LOCKED)]

    def test_a_non_s3_backend_reads_unmeasurable_never_a_silent_zero_findings(
        self, tmp_path
    ) -> None:
        """`LocalStore` has no Object Lock concept. A release object present
        there must read UNMEASURABLE, not be skipped into a clean "0
        findings" that an operator pointing the sweep at the wrong store
        would read as "all good"."""
        store = LocalStore(tmp_path)
        key = release_json_key(SHA_LOCKED)
        store.put_bytes(key, b"{}")

        findings = release_lock_findings(store)

        assert len(findings) == 1
        assert findings[0].state == "UNMEASURABLE"
        assert "LocalStore" in findings[0].detail


class TestMetric:
    def test_all_met_is_ok(self) -> None:
        findings = [ReleaseLockReading("k", "MET", "locked")]
        metric = release_lock_metric(findings, now=_PUBLISHED_AT)
        assert metric["status"] == "OK"
        assert metric["value"] == 0.0

    def test_any_unmet_breaches_even_alongside_met_and_unmeasurable(self) -> None:
        findings = [
            ReleaseLockReading("locked", "MET", "locked"),
            ReleaseLockReading("unlocked", "UNMET", "no retention"),
            ReleaseLockReading("denied", "UNMEASURABLE", "AccessDenied"),
        ]
        metric = release_lock_metric(findings, now=_PUBLISHED_AT)
        assert metric["status"] == "BREACH"
        assert metric["value"] == 1.0
        assert "unlocked" in metric["status_reason"]

    def test_unmeasurable_without_any_unmet_is_unmeasurable_not_ok(self) -> None:
        findings = [
            ReleaseLockReading("locked", "MET", "locked"),
            ReleaseLockReading("denied", "UNMEASURABLE", "AccessDenied"),
        ]
        metric = release_lock_metric(findings, now=_PUBLISHED_AT)
        assert metric["status"] == "unmeasurable"

    def test_metric_shape_matches_the_run_manifest_metricrecord_contract(self) -> None:
        findings = [ReleaseLockReading("k", "MET", "locked")]
        metric = release_lock_metric(findings, now=_PUBLISHED_AT)
        for field in (
            "name",
            "module",
            "metric_type",
            "n_floor",
            "status",
            "status_reason",
            "source_path",
            "last_updated_utc",
        ):
            assert field in metric
