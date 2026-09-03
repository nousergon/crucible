"""alpha-engine-config-I9898: `crucible release.lock <sha>` — the repair job
that applies Object Lock retention to a release published before I9787's
write-time fix, against a fake S3 client, never a mock of
`crucible.release_retention` itself.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest
from botocore.exceptions import ClientError

from crucible.release import (
    RELEASE_OBJECT_LOCK_RETENTION,
    release_json_key,
    wheel_key,
)
from crucible.release_retention import (
    RETENTION_APPLIED,
    RETENTION_EXTENDED,
    RETENTION_UNCHANGED,
    ReleaseHasNoRecordError,
    apply_release_retention,
    release_lock_handler,
)
from crucible.store import LocalStore, S3Store

SHA = "a" * 40
_PUBLISHED_AT = dt.datetime(2026, 8, 1, 12, 0, tzinfo=dt.UTC)


class _Paginator:
    def __init__(self, fake: _FakeS3Client) -> None:
        self.fake = fake

    def paginate(self, **kw) -> list:
        prefix = kw.get("Prefix") or ""
        keys = sorted(k for k in self.fake.objects if k.startswith(prefix))
        return [{"Contents": [{"Key": k} for k in keys]}]


class _FakeS3Client:
    """Enough of boto3's S3 surface for `apply_release_retention` and for
    `crucible.runner.run_job` to write its own manifest through the same
    store — a real run against this fake exercises both, same as
    `crucible.track_c`'s own handler tests do.

    Deliberately separate from `tests/test_release_lock_sweep.py`'s fixture
    of the same shape: that one has no `put_object`/`put_object_retention`
    concept, and this one has no need for its race/malform/throttle
    vocabulary — each fixture models exactly what its own module calls.
    """

    def __init__(self) -> None:
        self.objects: dict[str, dict] = {}
        self.retentions: dict[str, dict | str | None] = {}
        self.put_object_retention_calls: list[dict] = []
        self.deny_put_retention_for: set[str] = set()
        self.blobs: dict[str, bytes] = {}

    def put(
        self,
        key: str,
        *,
        locked: bool = False,
        mode: str = "GOVERNANCE",
        retain_until: dt.datetime | None = None,
        last_modified: dt.datetime | None = None,
    ) -> None:
        self.objects[key] = {"LastModified": last_modified or _PUBLISHED_AT}
        if locked:
            self.retentions[key] = {"Mode": mode, "RetainUntilDate": retain_until}
        else:
            self.retentions[key] = "NoSuchObjectLockConfiguration"

    def head_object(self, **kw) -> dict:
        key = kw["Key"]
        if key not in self.objects:
            raise ClientError({"Error": {"Code": "404"}}, "head_object")
        return self.objects[key]

    def get_object_retention(self, **kw) -> dict:
        key = kw["Key"]
        outcome = self.retentions.get(key)
        if isinstance(outcome, str):
            raise ClientError({"Error": {"Code": outcome}}, "get_object_retention")
        return {"Retention": outcome or {}}

    def put_object_retention(self, **kw) -> dict:
        key = kw["Key"]
        self.put_object_retention_calls.append(kw)
        if key in self.deny_put_retention_for:
            raise ClientError({"Error": {"Code": "AccessDenied"}}, "put_object_retention")
        self.retentions[key] = dict(kw["Retention"])
        return {}

    def put_object(self, **kw) -> dict:
        # The manifest writer's own call — `S3Store.put_bytes`. Recorded so
        # a test can assert the manifest landed, never inspected for lock
        # kwargs (this fake's `put_object_retention` is the only place those
        # are asserted).
        key = kw["Key"]
        self.blobs[key] = kw["Body"]
        self.objects.setdefault(key, {"LastModified": _PUBLISHED_AT})
        return {"ETag": '"fake"'}

    def get_object(self, **kw) -> dict:
        key = kw["Key"]
        if key not in self.blobs:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "get_object")

        class _Body:
            def __init__(self, data: bytes) -> None:
                self._data = data

            def read(self) -> bytes:
                return self._data

        return {"Body": _Body(self.blobs[key])}

    def get_paginator(self, name: str) -> _Paginator:
        assert name == "list_objects_v2"
        return _Paginator(self)


def _store(client: _FakeS3Client) -> S3Store:
    return S3Store("bucket", "crucible", client=client)


def _publish(client: _FakeS3Client, sha: str, **kw) -> None:
    client.put(f"crucible/{release_json_key(sha)}", **kw)
    client.put(f"crucible/{wheel_key(sha)}", **kw)


class TestApplyReleaseRetention:
    def test_absent_retention_is_applied(self) -> None:
        client = _FakeS3Client()
        store = _store(client)
        _publish(client, SHA, locked=False, last_modified=_PUBLISHED_AT)

        results = apply_release_retention(store, SHA)

        assert {r.action for r in results} == {RETENTION_APPLIED}
        assert len(results) == 2
        assert len(client.put_object_retention_calls) == 2
        for call in client.put_object_retention_calls:
            assert call["Retention"]["Mode"] == "GOVERNANCE"
            assert call["Retention"]["RetainUntilDate"] == (
                _PUBLISHED_AT + RELEASE_OBJECT_LOCK_RETENTION
            )
        for r in results:
            assert r.before_mode is None
            assert r.after_mode == "GOVERNANCE"

    def test_present_and_long_enough_is_untouched(self) -> None:
        client = _FakeS3Client()
        store = _store(client)
        far_future = _PUBLISHED_AT + RELEASE_OBJECT_LOCK_RETENTION + dt.timedelta(days=3650)
        _publish(client, SHA, locked=True, retain_until=far_future, last_modified=_PUBLISHED_AT)

        results = apply_release_retention(store, SHA)

        assert {r.action for r in results} == {RETENTION_UNCHANGED}
        assert client.put_object_retention_calls == []
        for r in results:
            assert r.before_retain_until == far_future.isoformat()
            assert r.after_retain_until == far_future.isoformat()

    def test_present_and_shorter_is_extended(self) -> None:
        client = _FakeS3Client()
        store = _store(client)
        short = _PUBLISHED_AT + dt.timedelta(days=30)
        _publish(
            client,
            SHA,
            locked=True,
            mode="GOVERNANCE",
            retain_until=short,
            last_modified=_PUBLISHED_AT,
        )

        results = apply_release_retention(store, SHA)

        assert {r.action for r in results} == {RETENTION_EXTENDED}
        assert len(client.put_object_retention_calls) == 2
        target = _PUBLISHED_AT + RELEASE_OBJECT_LOCK_RETENTION
        for call in client.put_object_retention_calls:
            # Extended in place under the SAME mode already on the object —
            # never substituted for a fresh target mode.
            assert call["Retention"]["Mode"] == "GOVERNANCE"
            assert call["Retention"]["RetainUntilDate"] == target
        for r in results:
            assert r.before_retain_until == short.isoformat()
            assert r.after_retain_until == target.isoformat()

    def test_never_shortens_a_compliance_lock_and_preserves_its_mode(self) -> None:
        """A COMPLIANCE-mode object that already exceeds the target is left
        alone — same `unchanged` path GOVERNANCE takes, never attempted."""
        client = _FakeS3Client()
        store = _store(client)
        far_future = _PUBLISHED_AT + RELEASE_OBJECT_LOCK_RETENTION + dt.timedelta(days=1)
        _publish(
            client,
            SHA,
            locked=True,
            mode="COMPLIANCE",
            retain_until=far_future,
            last_modified=_PUBLISHED_AT,
        )

        results = apply_release_retention(store, SHA)

        assert {r.action for r in results} == {RETENTION_UNCHANGED}
        assert client.put_object_retention_calls == []

    def test_a_target_500ms_short_from_second_precision_skew_is_unchanged(self) -> None:
        """alpha-engine-config-I9898 round-2 finding 1, reproduced: S3 stores
        `RetainUntilDate` at millisecond precision, but the target this job
        recomputes comes from `HeadObject`'s `LastModified`, which S3
        truncates to the SECOND. A correctly-locked object is therefore
        always a few hundred milliseconds "short" of a target recomputed
        this way — `retention_meets_target`'s clock-skew slack must absorb
        that, or every already-compliant release is read as `extended` and
        gets a redundant `PutObjectRetention` forever."""
        client = _FakeS3Client()
        store = _store(client)
        target = _PUBLISHED_AT + RELEASE_OBJECT_LOCK_RETENTION
        stored = target - dt.timedelta(milliseconds=500)
        _publish(client, SHA, locked=True, retain_until=stored, last_modified=_PUBLISHED_AT)

        results = apply_release_retention(store, SHA)

        assert {r.action for r in results} == {RETENTION_UNCHANGED}
        assert client.put_object_retention_calls == []

    def test_a_target_2_days_short_is_extended_despite_the_skew_slack(self) -> None:
        """A gap larger than the clock-skew slack is a real shortfall, not
        skew — it must still extend."""
        client = _FakeS3Client()
        store = _store(client)
        target = _PUBLISHED_AT + RELEASE_OBJECT_LOCK_RETENTION
        stored = target - dt.timedelta(days=2)
        _publish(client, SHA, locked=True, retain_until=stored, last_modified=_PUBLISHED_AT)

        results = apply_release_retention(store, SHA)

        assert {r.action for r in results} == {RETENTION_EXTENDED}
        assert len(client.put_object_retention_calls) == 2
        for call in client.put_object_retention_calls:
            assert call["Retention"]["RetainUntilDate"] == target

    def test_a_sha_with_no_release_json_is_refused(self) -> None:
        client = _FakeS3Client()
        store = _store(client)
        # The wheel exists but release.json does not — a partial, never-
        # published prefix.
        client.put(f"crucible/{wheel_key(SHA)}", locked=False)

        with pytest.raises(ReleaseHasNoRecordError):
            apply_release_retention(store, SHA)

    def test_a_local_store_is_refused(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        with pytest.raises(TypeError):
            apply_release_retention(store, SHA)

    def test_dry_run_never_writes(self) -> None:
        client = _FakeS3Client()
        store = _store(client)
        _publish(client, SHA, locked=False, last_modified=_PUBLISHED_AT)

        results = apply_release_retention(store, SHA, dry_run=True)

        assert {r.action for r in results} == {RETENTION_APPLIED}
        assert client.put_object_retention_calls == []

    def test_a_second_run_is_a_clean_no_op(self) -> None:
        """Idempotent: applying twice changes nothing on the second call."""
        client = _FakeS3Client()
        store = _store(client)
        _publish(client, SHA, locked=False, last_modified=_PUBLISHED_AT)

        first = apply_release_retention(store, SHA)
        assert {r.action for r in first} == {RETENTION_APPLIED}
        assert len(client.put_object_retention_calls) == 2

        second = apply_release_retention(store, SHA)
        assert {r.action for r in second} == {RETENTION_UNCHANGED}
        assert len(client.put_object_retention_calls) == 2  # unchanged from the first run

    def test_access_denied_on_put_object_retention_raises(self) -> None:
        client = _FakeS3Client()
        store = _store(client)
        _publish(client, SHA, locked=False, last_modified=_PUBLISHED_AT)
        client.deny_put_retention_for = {f"crucible/{release_json_key(SHA)}"}

        with pytest.raises(ClientError) as excinfo:
            apply_release_retention(store, SHA)
        assert excinfo.value.response["Error"]["Code"] == "AccessDenied"


class TestReleaseLockHandler:
    """`crucible release.lock <sha>` through `run_job`, exactly like every
    other job — a real manifest, `ok` on success, `failed` (never a swallow)
    when `PutObjectRetention` is denied."""

    def _args(self, *, sha: str, dry_run: bool = False, store):
        import argparse

        return argparse.Namespace(
            sha=sha,
            dry_run=dry_run,
            store=None,
            trading_day=dt.date(2026, 8, 28),
        )

    def test_a_successful_run_writes_an_ok_manifest(self) -> None:
        client = _FakeS3Client()
        store = _store(client)
        _publish(client, SHA, locked=False, last_modified=_PUBLISHED_AT)

        from unittest.mock import patch

        with patch("crucible.release_retention.open_store", return_value=store):
            rc = release_lock_handler(self._args(sha=SHA, store=store))

        assert rc == 0
        # The manifest landed at runs/release.lock/{trading_day}/{sha}/run.json
        # — discriminated by sha (finding 2), not a bare
        # runs/release.lock/{trading_day}/run.json.
        manifest_bytes = client.blobs[f"crucible/runs/release.lock/2026-08-28/{SHA}/run.json"]
        manifest = json.loads(manifest_bytes)
        assert manifest["status"] == "ok"
        assert manifest["job"] == "release.lock"
        names = {m["name"] for m in manifest["metrics"]}
        assert "release_objects_locked" in names
        assert "release_object_retention" in names

    def test_two_shas_repaired_on_one_trading_day_write_two_manifests(self) -> None:
        """alpha-engine-config-I9898 round-2 finding 2: without
        `discriminator=sha`, two repairs on the same trading day both write
        `runs/release.lock/{trading_day}/run.json` and the second silently
        overwrites the first — rule 1, manifest or it did not happen,
        defeated at the second invocation."""
        client = _FakeS3Client()
        store = _store(client)
        sha_a = "a" * 40
        sha_b = "b" * 40
        _publish(client, sha_a, locked=False, last_modified=_PUBLISHED_AT)
        _publish(client, sha_b, locked=False, last_modified=_PUBLISHED_AT)

        from unittest.mock import patch

        with patch("crucible.release_retention.open_store", return_value=store):
            rc_a = release_lock_handler(self._args(sha=sha_a, store=store))
            rc_b = release_lock_handler(self._args(sha=sha_b, store=store))

        assert rc_a == 0
        assert rc_b == 0
        key_a = f"crucible/runs/release.lock/2026-08-28/{sha_a}/run.json"
        key_b = f"crucible/runs/release.lock/2026-08-28/{sha_b}/run.json"
        assert key_a != key_b
        assert key_a in client.blobs
        assert key_b in client.blobs
        manifest_a = json.loads(client.blobs[key_a])
        manifest_b = json.loads(client.blobs[key_b])
        assert manifest_a["status"] == "ok"
        assert manifest_b["status"] == "ok"

    def test_access_denied_fails_the_job_with_the_reason_named(self) -> None:
        client = _FakeS3Client()
        store = _store(client)
        _publish(client, SHA, locked=False, last_modified=_PUBLISHED_AT)
        client.deny_put_retention_for = {f"crucible/{release_json_key(SHA)}"}

        from unittest.mock import patch

        with patch("crucible.release_retention.open_store", return_value=store):
            with pytest.raises(ClientError):
                release_lock_handler(self._args(sha=SHA, store=store))

        manifest_bytes = client.blobs[f"crucible/runs/release.lock/2026-08-28/{SHA}/run.json"]
        manifest = json.loads(manifest_bytes)
        assert manifest["status"] == "failed"
        assert "AccessDenied" in manifest["reason"]

    def test_dry_run_writes_no_manifest_at_all(self) -> None:
        """alpha-engine-config-I9922 round-2 review, non-blocking note:
        `store.client` passes through `crucible.store.read_only` unchanged
        (it is not in `Store.MUTATORS`), and `apply_release_retention`
        mutates Object Lock via `store.client.put_object_retention` directly
        — a second write path the store guard cannot see at all, double
        -guarded today only because `release_lock_handler`'s own `--dry-run`
        branch (`release_retention.py`) never calls `apply_release_retention`
        with `dry_run=False` in the first place. Pinned here against the
        FAKE CLIENT itself, not just the higher-level manifest/pointer
        assertions: no `put_object_retention` call AND no `put_object` call
        of any kind landed — `client.blobs` is empty, so nothing was written
        through `S3Store.put_bytes` either (the manifest included, already
        covered by the assertion below)."""
        client = _FakeS3Client()
        store = _store(client)
        _publish(client, SHA, locked=False, last_modified=_PUBLISHED_AT)

        from unittest.mock import patch

        with patch("crucible.release_retention.open_store", return_value=store):
            rc = release_lock_handler(self._args(sha=SHA, dry_run=True, store=store))

        assert rc == 0
        assert client.put_object_retention_calls == []
        assert client.blobs == {}
        assert f"crucible/runs/release.lock/2026-08-28/{SHA}/run.json" not in client.blobs
