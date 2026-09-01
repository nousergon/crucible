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
    StaleReleasePointerError,
    current_release,
    flip_on_smoke,
    pin,
    publish_release,
    read_pointer,
    release_json_key,
    resolve_release,
    wheel_key,
)
from crucible.store import ETAG_ABSENT, LocalStore, PointerConflictError

SHA_A = "a" * 40
SHA_B = "b" * 40


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
