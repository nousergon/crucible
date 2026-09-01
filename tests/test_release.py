"""One artifact, one pointer, one verified flip — asserted, not described.

Normative source: plan §4.11.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from crucible.release import (
    POINTER_KEY,
    TRADER_PIN_KEY,
    ReleaseImmutabilityError,
    StaleReleasePointerError,
    assert_immutable_write,
    current_release,
    flip_on_smoke,
    pin,
    publish_release,
    read_pointer,
    release_json_key,
    resolve_release,
    wheel_key,
)
from crucible.store import ETAG_ABSENT, LocalStore, PointerConflictError, S3Store

SHA_A = "a" * 40
SHA_B = "b" * 40


class _FakeS3Client:
    """Just enough of the boto3 S3 surface for the Object Lock test below.

    Deliberately NOT `tests/conftest.py`'s `FakeS3` fixture: this module owns
    only `crucible/release.py`'s tests, and `put_object_retention` is not
    part of that shared fake's boto3 surface (its owner is a sibling change
    in this same issue's session) — adding it there would edit a fixture
    other tests in this suite depend on. This fake exists to prove one
    thing: that `PutObjectRetention` is actually requested on the object,
    not merely that the bucket-level flag exists (the issue's Gotcha).
    """

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.retentions: dict[str, dict] = {}

    def put_object(self, **kw) -> dict:
        self.objects[kw["Key"]] = kw["Body"]
        return {"ETag": '"fake"'}

    def get_object(self, **kw) -> dict:
        class _Body:
            def __init__(self, payload: bytes) -> None:
                self._payload = payload

            def read(self) -> bytes:
                return self._payload

        return {"Body": _Body(self.objects[kw["Key"]])}

    def head_object(self, **kw) -> dict:
        if kw["Key"] not in self.objects:
            from botocore.exceptions import ClientError

            raise ClientError({"Error": {"Code": "404"}}, "head_object")
        return {"ETag": '"fake"'}

    def put_object_retention(self, **kw) -> dict:
        self.retentions[kw["Key"]] = kw["Retention"]
        return {}


def _published(store, sha=SHA_A):
    return publish_release(
        store,
        sha=sha,
        wheel=b"PK\x03\x04 wheel bytes",
        lockfile=b"# uv.lock",
        test_summary="42 passed",
        workflow_run_url="https://github.com/nousergon/crucible/actions/runs/1",
        now=dt.datetime(2026, 8, 28, 21, 0, tzinfo=dt.UTC),
    )


def _smoke_manifest(sha=SHA_A, status="ok"):
    return {
        "job": "smoke",
        "release_sha": sha,
        "status": status,
        "reason": "" if status == "ok" else "RuntimeError: boom",
    }


class TestLayout:
    def test_the_release_is_addressed_by_sha_not_by_version(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _published(store)
        assert store.exists(wheel_key(SHA_A))
        assert store.exists(release_json_key(SHA_A))

    def test_release_json_records_the_lockfile_hash(self, tmp_path) -> None:
        """Two wheels built from one commit against two resolved dependency
        trees are two different artifacts, and only this says which."""
        store = LocalStore(tmp_path)
        record = _published(store)
        assert record.lockfile_sha256
        payload = json.loads(store.get_bytes(release_json_key(SHA_A)))
        assert payload["lockfile_sha256"] == record.lockfile_sha256

    def test_publishing_does_not_touch_the_pointer(self, tmp_path) -> None:
        """Uploading is safe and repeatable; moving the pointer is the act
        with consequences, and the smoke gate sits between them."""
        store = LocalStore(tmp_path)
        _published(store)
        assert current_release(store) is None

    @pytest.mark.parametrize("bad", ["abc", SHA_A.upper(), "a" * 39, ""])
    def test_an_abbreviated_or_uppercase_sha_is_refused(self, tmp_path, bad: str) -> None:
        with pytest.raises(ValueError, match="40-character lowercase"):
            wheel_key(bad)

    def test_an_empty_wheel_is_refused(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="empty wheel"):
            publish_release(
                LocalStore(tmp_path),
                sha=SHA_A,
                wheel=b"",
                lockfile=b"x",
                test_summary="",
                workflow_run_url="",
            )


class TestObjectLock:
    """§4.11's immutability claim, defended past the writer (I9782).

    `assert_immutable_write` (`TestImmutability` below) refuses a differing
    overwrite from THIS code path; these tests are for the layer that
    defends against a writer that skips this module entirely.
    """

    def test_publish_locks_the_wheel_and_release_json_in_governance_mode(self) -> None:
        client = _FakeS3Client()
        store = S3Store("bucket", "crucible", client=client)
        _published(store)
        for key in (wheel_key(SHA_A), release_json_key(SHA_A)):
            s3_key = f"crucible/{key}"
            assert s3_key in client.retentions, f"no PutObjectRetention requested for {key}"
            retention = client.retentions[s3_key]
            assert retention["Mode"] == "GOVERNANCE"
            assert retention["RetainUntilDate"] > dt.datetime(2026, 8, 28, 21, 0, tzinfo=dt.UTC)

    def test_the_pointer_is_never_locked(self) -> None:
        """`releases/current` is a pointer, mutable by design, moved by
        conditional PUT — the exact opposite of what Object Lock defends."""
        client = _FakeS3Client()
        store = S3Store("bucket", "crucible", client=client)
        _published(store)
        pin(store, SHA_A, expect=ETAG_ABSENT)
        assert f"crucible/{POINTER_KEY}" not in client.retentions
        # And no PUT to the pointer's key was ever asked to lock the wheel's key
        # or vice versa — the two writes go through entirely separate calls.
        assert all(key != f"crucible/{POINTER_KEY}" for key in client.retentions)

    def test_object_lock_is_never_requested_off_s3(self, tmp_path) -> None:
        """`LocalStore` has no Object Lock concept; the helper is a no-op
        there rather than an error, so the laptop/test backend needs no
        special-casing."""
        store = LocalStore(tmp_path)
        _published(store)  # would raise on any attempt to reach a client


class TestPointer:
    def test_pinning_to_an_unpublished_sha_is_refused(self, tmp_path) -> None:
        """A pointer to an artifact that is not there is stale the moment it
        is written."""
        with pytest.raises(StaleReleasePointerError, match="no wheel at"):
            pin(LocalStore(tmp_path), SHA_A)

    def test_pin_then_read_round_trips(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _published(store)
        pin(store, SHA_A)
        assert current_release(store) == SHA_A

    def test_the_trader_pin_is_a_separate_pointer(self, tmp_path) -> None:
        """A trader that followed `current` would be promoted by every merge."""
        store = LocalStore(tmp_path)
        _published(store)
        _published(store, SHA_B)
        pin(store, SHA_A, target="trader")
        pin(store, SHA_B, target="current")
        assert json.loads(store.get_bytes(TRADER_PIN_KEY))["sha"] == SHA_A
        assert json.loads(store.get_bytes(POINTER_KEY))["sha"] == SHA_B

    def test_an_unknown_pin_target_is_refused(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _published(store)
        with pytest.raises(ValueError, match="pin target"):
            pin(store, SHA_A, target="staging")

    def test_a_swap_against_a_stale_version_loses(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _published(store)
        _published(store, SHA_B)
        pin(store, SHA_A, expect=ETAG_ABSENT)
        with pytest.raises(PointerConflictError):
            pin(store, SHA_B, expect=ETAG_ABSENT)


class TestStalePointer:
    def test_an_unset_pointer_refuses_rather_than_choosing(self, tmp_path) -> None:
        with pytest.raises(StaleReleasePointerError, match="unset"):
            resolve_release(LocalStore(tmp_path))

    def test_a_pointer_to_a_deleted_release_is_a_named_failure(self, tmp_path) -> None:
        """§10.7 fault 4. A job that installed nothing and carried on would
        run whatever was already on the box — the silent version of every
        deploy bug."""
        store = LocalStore(tmp_path)
        _published(store)
        pin(store, SHA_A)
        (tmp_path / wheel_key(SHA_A)).unlink()
        with pytest.raises(StaleReleasePointerError, match="whose wheel is not at"):
            resolve_release(store)


class TestSmokeGate:
    def test_an_ok_smoke_flips_the_pointer(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _published(store)
        _, version = read_pointer(store)
        assert flip_on_smoke(store, sha=SHA_A, smoke_manifest=_smoke_manifest(), expect=version)
        assert current_release(store) == SHA_A

    def test_a_failed_smoke_leaves_the_pointer_exactly_where_it_was(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _published(store)
        _published(store, SHA_B)
        pin(store, SHA_A)
        _, version = read_pointer(store)
        flipped = flip_on_smoke(
            store, sha=SHA_B, smoke_manifest=_smoke_manifest(SHA_B, "failed"), expect=version
        )
        assert flipped is False
        assert current_release(store) == SHA_A

    def test_another_builds_smoke_cannot_promote_this_one(self, tmp_path) -> None:
        """The gate failing open: promoting on a smoke that verified a
        different artifact."""
        store = LocalStore(tmp_path)
        _published(store)
        _published(store, SHA_B)
        with pytest.raises(ValueError, match="not"):
            flip_on_smoke(
                store, sha=SHA_B, smoke_manifest=_smoke_manifest(SHA_A), expect=ETAG_ABSENT
            )

    def test_a_non_smoke_manifest_cannot_be_the_gate(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        _published(store)
        manifest = _smoke_manifest()
        manifest["job"] = "data.daily"
        with pytest.raises(ValueError, match="gate is the smoke run"):
            flip_on_smoke(store, sha=SHA_A, smoke_manifest=manifest, expect=ETAG_ABSENT)

    def test_a_deploy_that_raced_another_deploy_fails_rather_than_overwriting(
        self, tmp_path
    ) -> None:
        store = LocalStore(tmp_path)
        _published(store)
        _published(store, SHA_B)
        _, version = read_pointer(store)  # both deploys read ETAG_ABSENT
        pin(store, SHA_B, expect=version)  # the other deploy wins
        with pytest.raises(PointerConflictError):
            flip_on_smoke(store, sha=SHA_A, smoke_manifest=_smoke_manifest(), expect=version)


class TestImmutability:
    """§4.11 claims "immutable versioned artifacts", and the whole rollback
    story rests on it: `crucible release.pin <prior-sha>` is a rollback only
    if the wheel under that prefix is still the wheel that was tested.

    The IAM grant is `s3:PutObject` on `crucible/releases/*` with no Object
    Lock, so nothing below the application layer prevents an overwrite. These
    are the application half; the CloudFormation half lives in
    `nous-ergon-ops`.
    """

    def test_republishing_a_sha_with_different_bytes_is_refused(self, tmp_path) -> None:
        """A `workflow_dispatch` re-run for the same sha rebuilds the wheel.
        Overwriting it changes the artifact a prior consumer installed, and
        `release.json.wheel_sha256` changes under it."""
        store = LocalStore(tmp_path)
        _published(store)
        original = store.get_bytes(wheel_key(SHA_A))
        with pytest.raises(ReleaseImmutabilityError, match="already exists with different bytes"):
            publish_release(
                store,
                sha=SHA_A,
                wheel=b"a REBUILT wheel",
                lockfile=b"# uv.lock",
                test_summary="42 passed",
                workflow_run_url="",
                now=dt.datetime(2026, 8, 28, 21, 0, tzinfo=dt.UTC),
            )
        assert store.get_bytes(wheel_key(SHA_A)) == original

    def test_the_refusal_names_the_operation_that_is_correct_instead(self, tmp_path) -> None:
        """Re-promoting an existing build is `release.pin` and needs no
        rebuild; publishing different bytes needs a new commit. Neither is
        "retry the deploy", so the message has to say which."""
        store = LocalStore(tmp_path)
        _published(store)
        with pytest.raises(ReleaseImmutabilityError, match="release.pin"):
            publish_release(
                store,
                sha=SHA_A,
                wheel=b"different",
                lockfile=b"x",
                test_summary="",
                workflow_run_url="",
            )

    def test_nothing_is_written_when_the_wheel_is_refused(self, tmp_path) -> None:
        """A wheel from one build beside a release.json from another is worse
        than either, so both keys are checked before either is written."""
        store = LocalStore(tmp_path)
        _published(store)
        before = store.get_bytes(release_json_key(SHA_A))
        with pytest.raises(ReleaseImmutabilityError):
            publish_release(
                store,
                sha=SHA_A,
                wheel=b"different",
                lockfile=b"x",
                test_summary="",
                workflow_run_url="",
            )
        assert store.get_bytes(release_json_key(SHA_A)) == before

    def test_an_identical_republish_is_a_no_op_not_a_failure(self, tmp_path) -> None:
        """A retried deploy against the same artifact must stay idempotent, or
        it becomes indistinguishable from a corrupted one."""
        store = LocalStore(tmp_path)
        _published(store)
        _published(store)  # same bytes, same `now` — must not raise
        assert store.exists(wheel_key(SHA_A))

    def test_the_comparison_is_on_the_bytes_not_on_a_recorded_digest(self, tmp_path) -> None:
        """The thing being protected is precisely the case where a recorded
        claim and the object have diverged, so a digest the writer supplies
        about its own payload cannot be the comparison."""
        store = LocalStore(tmp_path)
        store.put_bytes("releases/x", b"one")
        assert assert_immutable_write(store, "releases/x", b"one") is False
        assert assert_immutable_write(store, "releases/absent", b"one") is True
        with pytest.raises(ReleaseImmutabilityError):
            assert_immutable_write(store, "releases/x", b"two")
