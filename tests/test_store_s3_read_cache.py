"""`alpha-engine-config-I11792` — each manifest body crosses the wire once per job.

The 2026-10-09 access logs showed `gate-close.yml`'s `gate-publish` job read the
same 20-30 MB `runs/experiment.backfill/**/run.json` 12 times in one run, once
per `crucible gate --gate <phase>` process, all billed as internet egress. The
cache these tests pin is opt-in (`CRUCIBLE_S3_READ_CACHE_DIR`), shared across
processes through a directory, and validated by ETag on every read, so it can
never return bytes S3 no longer holds.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pytest
from botocore.exceptions import ClientError

from crucible import store as store_mod
from crucible.store import READ_CACHE_DIR_VAR, S3Store


class _Body:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data


class _ConditionalS3:
    """An S3 that honours `IfNoneMatch` the way the service does.

    It counts the GET requests it answers and the body BYTES it sends, because
    the bytes are what this cache exists to stop paying for.
    """

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.gets = 0
        self.not_modified = 0
        self.bytes_sent = 0

    @staticmethod
    def _etag(body: bytes) -> str:
        return hashlib.md5(body).hexdigest()  # noqa: S324 - S3's own ETag shape

    @staticmethod
    def _error(code: str) -> ClientError:
        return ClientError({"Error": {"Code": code, "Message": code}}, "GetObject")

    def put_object(self, **kw: Any) -> dict[str, Any]:
        self.objects[kw["Key"]] = kw["Body"]
        return {"ETag": f'"{self._etag(kw["Body"])}"'}

    def get_object(self, **kw: Any) -> dict[str, Any]:
        self.gets += 1
        key = kw["Key"]
        if key not in self.objects:
            raise self._error("NoSuchKey")
        body = self.objects[key]
        if kw.get("IfNoneMatch") == self._etag(body):
            self.not_modified += 1
            raise self._error("304")
        self.bytes_sent += len(body)
        return {"Body": _Body(body), "ETag": f'"{self._etag(body)}"'}


KEY = "runs/experiment.backfill/2026-09-11/m.residual_momentum/run.json"
BODY = b'{"run": "' + b"x" * 4096 + b'"}'


@pytest.fixture
def s3() -> _ConditionalS3:
    fake = _ConditionalS3()
    fake.objects[f"crucible/{KEY}"] = BODY
    return fake


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv(READ_CACHE_DIR_VAR, str(tmp_path / "cache"))
    monkeypatch.setattr(store_mod, "_READ_CACHES", {})
    return tmp_path / "cache"


def test_without_the_variable_every_read_is_a_plain_get(s3, monkeypatch) -> None:
    monkeypatch.delenv(READ_CACHE_DIR_VAR, raising=False)
    store = S3Store("bucket", "crucible", client=s3)
    for _ in range(3):
        assert store.get_bytes(KEY) == BODY
    assert s3.gets == 3
    assert s3.bytes_sent == 3 * len(BODY)


def test_one_process_downloads_the_body_once(s3, cache_dir) -> None:
    store = S3Store("bucket", "crucible", client=s3)
    for _ in range(5):
        assert store.get_bytes(KEY) == BODY
    assert s3.bytes_sent == len(BODY), "the body must cross the wire exactly once"
    assert s3.not_modified == 4, "every later read still asks S3, conditionally"


def test_separate_processes_sharing_the_directory_download_it_once(s3, cache_dir) -> None:
    """The measured shape: one `crucible gate` process per phase, in a loop."""
    for _ in range(6):
        store_mod._READ_CACHES.clear()  # a fresh process has an empty memory layer
        assert S3Store("bucket", "crucible", client=s3).get_bytes(KEY) == BODY
    assert s3.bytes_sent == len(BODY)


def test_a_changed_object_is_read_fresh_never_served_stale(s3, cache_dir) -> None:
    store = S3Store("bucket", "crucible", client=s3)
    assert store.get_bytes(KEY) == BODY
    s3.objects[f"crucible/{KEY}"] = b'{"run": "rewritten by another writer"}'
    assert store.get_bytes(KEY) == b'{"run": "rewritten by another writer"}'
    store_mod._READ_CACHES.clear()
    assert store.get_bytes(KEY) == b'{"run": "rewritten by another writer"}'


def test_a_write_through_the_store_drops_the_entry(s3, cache_dir) -> None:
    store = S3Store("bucket", "crucible", client=s3)
    store.get_bytes(KEY)
    store.put_bytes(KEY, b'{"run": "mine"}')
    assert store.get_bytes(KEY) == b'{"run": "mine"}'


def test_a_deleted_object_still_raises_key_error(s3, cache_dir) -> None:
    store = S3Store("bucket", "crucible", client=s3)
    store.get_bytes(KEY)
    del s3.objects[f"crucible/{KEY}"]
    with pytest.raises(KeyError):
        store.get_bytes(KEY)


def test_a_corrupt_entry_is_a_miss_not_an_error(s3, cache_dir) -> None:
    store = S3Store("bucket", "crucible", client=s3)
    store.get_bytes(KEY)
    for entry in cache_dir.iterdir():
        entry.write_bytes(b"")
    store_mod._READ_CACHES.clear()
    assert store.get_bytes(KEY) == BODY
