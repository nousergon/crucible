"""`open_store` and `store_from_uri` share one scheme validator
(`alpha-engine-config-I10519`).

Before this, `crucible.store.open_store` and `crucible.config.store_from_uri`
each hand-maintained an independent copy of the same three-branch check
(`s3://` / any other `://` scheme / a local path). Both now delegate scheme
classification to `crucible.store.parse_store_scheme`; this test proves the
two public entry points reject the same bad input identically rather than by
drift-prone duplicate logic.
"""

from __future__ import annotations

import pytest

from crucible.config import store_from_uri
from crucible.store import LocalStore, S3Store, open_store, parse_store_scheme


class TestBothEntryPointsRejectTheSameBadInputIdentically:
    @pytest.mark.parametrize("bad_uri", ["file://./store", "ftp://host/path", "gs://bucket/x"])
    def test_an_unsupported_scheme_is_refused_with_the_same_message(self, bad_uri: str) -> None:
        with pytest.raises(ValueError) as open_store_exc:
            open_store(bad_uri)
        with pytest.raises(ValueError) as store_from_uri_exc:
            store_from_uri(bad_uri)
        assert str(open_store_exc.value) == str(store_from_uri_exc.value)
        assert "unsupported store scheme" in str(open_store_exc.value)


class TestParseStoreScheme:
    def test_an_s3_uri_classifies_as_s3(self) -> None:
        assert parse_store_scheme("s3://my-bucket/crucible") == ("s3", "my-bucket/crucible")

    def test_a_local_path_classifies_as_local_unchanged(self) -> None:
        assert parse_store_scheme("./store") == ("local", "./store")

    def test_an_unsupported_scheme_raises(self) -> None:
        with pytest.raises(ValueError, match="unsupported store scheme"):
            parse_store_scheme("file://./store")


class TestEachCallerStillConstructsItsOwnBackend:
    def test_open_store_resolves_an_s3_uri_to_s3store(self) -> None:
        assert isinstance(open_store("s3://my-bucket/crucible"), S3Store)

    def test_store_from_uri_resolves_an_s3_uri_to_s3store(self) -> None:
        assert isinstance(store_from_uri("s3://my-bucket/crucible"), S3Store)

    def test_open_store_resolves_a_local_path_to_localstore(self, tmp_path) -> None:
        assert isinstance(open_store(str(tmp_path)), LocalStore)

    def test_store_from_uri_resolves_a_local_path_to_localstore(self, tmp_path) -> None:
        assert isinstance(store_from_uri(str(tmp_path)), LocalStore)

    def test_store_from_uri_still_refuses_an_empty_s3_bucket_by_name(self) -> None:
        """`store_from_uri`'s specific contract, preserved: `open_store` never
        made this check (an empty-bucket `s3://` URI there raised
        `S3Store`'s own generic "needs a bucket" error instead), and this PR
        does not widen `open_store`'s behaviour to add it."""
        with pytest.raises(ValueError, match="names no bucket"):
            store_from_uri("s3://")
