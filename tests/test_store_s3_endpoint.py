"""The S3-compatible endpoint setting on `crucible.store.S3Store`.

`alpha-engine-config-I12531` (staging plan S-44): the store must work against
any S3-compatible service (MinIO, Cloudflare R2) given an endpoint URL, and
must stay exactly AWS S3 when none is given.

The negative cases carry the weight here. Unset has to mean the same
`boto3.client("s3")` call production makes today, with no extra keyword.
A malformed value has to raise rather than fall back to AWS, because a run
meant for a test store would otherwise write to production.

These are unit tests of the client construction. The integration test against
a MinIO container (conditional writes and ArcticDB) is a CI follow-up on the
issue: no container runtime is available where this was written.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

from crucible.config import store_from_uri
from crucible.store import ENDPOINT_URL_VAR, S3Store, endpoint_url_from_env, open_store

MINIO = "http://localhost:9000"

#: botocore's own endpoint variables. They are cleared so an ambient value on
#: a runner cannot make the AWS-default assertions pass or fail by accident.
_BOTOCORE_ENDPOINT_VARS = ("AWS_ENDPOINT_URL", "AWS_ENDPOINT_URL_S3", "AWS_USE_FIPS_ENDPOINT")


class _RecordingBoto3:
    """A stand-in `boto3` module that records every `client(...)` call."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple, dict]] = []

    def client(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return SimpleNamespace(args=args, kwargs=kwargs)


@pytest.fixture
def boto3_calls(monkeypatch) -> list[tuple[tuple, dict]]:
    fake = _RecordingBoto3()
    monkeypatch.setitem(sys.modules, "boto3", fake)
    monkeypatch.delenv(ENDPOINT_URL_VAR, raising=False)
    return fake.calls


class TestUnsetIsTheAwsDefaultUnchanged:
    def test_unset_builds_exactly_the_production_client_call(self, boto3_calls) -> None:
        store = S3Store("bucket", "crucible")
        assert store.endpoint_url is None
        store.client  # noqa: B018 - the property builds the client
        assert boto3_calls == [(("s3",), {})], "no endpoint_url keyword, not even None"

    @pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
    def test_blank_is_unset(self, boto3_calls, monkeypatch, blank) -> None:
        monkeypatch.setenv(ENDPOINT_URL_VAR, blank)
        assert endpoint_url_from_env() is None
        S3Store("bucket").client  # noqa: B018
        assert boto3_calls == [(("s3",), {})]

    def test_an_injected_client_is_used_as_is(self, boto3_calls, monkeypatch) -> None:
        monkeypatch.setenv(ENDPOINT_URL_VAR, MINIO)
        injected = object()
        assert S3Store("bucket", client=injected).client is injected
        assert boto3_calls == []

    def test_the_client_is_built_once(self, boto3_calls, monkeypatch) -> None:
        monkeypatch.setenv(ENDPOINT_URL_VAR, MINIO)
        store = S3Store("bucket")
        assert store.client is store.client
        assert len(boto3_calls) == 1


class TestAnEndpointIsPassedThrough:
    def test_the_environment_variable_reaches_the_client(self, boto3_calls, monkeypatch) -> None:
        monkeypatch.setenv(ENDPOINT_URL_VAR, MINIO)
        store = S3Store("bucket", "crucible")
        assert store.endpoint_url == MINIO
        store.client  # noqa: B018
        assert boto3_calls == [(("s3",), {"endpoint_url": MINIO})]

    def test_surrounding_whitespace_is_stripped(self, boto3_calls, monkeypatch) -> None:
        monkeypatch.setenv(ENDPOINT_URL_VAR, f"  {MINIO}\n")
        assert S3Store("bucket").endpoint_url == MINIO

    def test_an_explicit_argument_wins_over_the_environment(self, boto3_calls, monkeypatch) -> None:
        monkeypatch.setenv(ENDPOINT_URL_VAR, "http://ignored:9000")
        r2 = "https://acct.r2.cloudflarestorage.com"
        S3Store("bucket", endpoint_url=r2).client  # noqa: B018
        assert boto3_calls == [(("s3",), {"endpoint_url": r2})]

    def test_open_store_honours_the_setting(self, boto3_calls, monkeypatch) -> None:
        monkeypatch.setenv(ENDPOINT_URL_VAR, MINIO)
        store = open_store("s3://bucket/prefix")
        assert isinstance(store, S3Store)
        assert store.endpoint_url == MINIO

    def test_a_dry_run_store_honours_the_setting(self, boto3_calls, monkeypatch) -> None:
        monkeypatch.setenv(ENDPOINT_URL_VAR, MINIO)
        assert open_store("s3://bucket/prefix", dry_run=True).endpoint_url == MINIO

    def test_config_store_from_uri_honours_the_setting(self, boto3_calls, monkeypatch) -> None:
        monkeypatch.setenv(ENDPOINT_URL_VAR, MINIO)
        store = store_from_uri("s3://bucket/prefix")
        assert isinstance(store, S3Store)
        assert store.endpoint_url == MINIO

    def test_unset_open_store_has_no_endpoint(self, boto3_calls) -> None:
        assert open_store("s3://bucket/prefix").endpoint_url is None
        assert store_from_uri("s3://bucket/prefix").endpoint_url is None


class TestAMalformedEndpointIsRefusedNotIgnored:
    @pytest.mark.parametrize(
        "value",
        ["localhost:9000", "s3://bucket", "ftp://host:21", "http://", "https:///path", "minio"],
    )
    def test_from_the_environment(self, boto3_calls, monkeypatch, value) -> None:
        monkeypatch.setenv(ENDPOINT_URL_VAR, value)
        with pytest.raises(ValueError, match=ENDPOINT_URL_VAR):
            S3Store("bucket")
        with pytest.raises(ValueError, match="would point this store at AWS S3"):
            open_store("s3://bucket/prefix")
        assert boto3_calls == []

    @pytest.mark.parametrize("value", ["", "localhost:9000", "http://"])
    def test_as_an_argument(self, boto3_calls, value) -> None:
        with pytest.raises(ValueError, match="endpoint_url"):
            S3Store("bucket", endpoint_url=value)
        assert boto3_calls == []


class TestTheRealSdkClient:
    """Offline checks against the real `boto3`: no request is sent. The
    client's resolved endpoint and a presigned URL show where requests go."""

    @pytest.fixture(autouse=True)
    def _offline_credentials(self, monkeypatch) -> None:
        for name in (ENDPOINT_URL_VAR, *_BOTOCORE_ENDPOINT_VARS, "AWS_PROFILE"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
        monkeypatch.setenv("AWS_CONFIG_FILE", os.devnull)
        monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", os.devnull)

    def _presigned(self, store: S3Store) -> str:
        url = store.client.generate_presigned_url(
            "get_object", Params={"Bucket": store.bucket, "Key": "k"}, ExpiresIn=60
        )
        return url.split("?", 1)[0]

    def test_unset_resolves_to_aws_s3(self) -> None:
        store = S3Store("bucket")
        assert store.client.meta.endpoint_url == "https://s3.amazonaws.com"
        presigned = urlsplit(self._presigned(store))
        assert (presigned.scheme, presigned.hostname, presigned.path) == (
            "https",
            "bucket.s3.amazonaws.com",
            "/k",
        )

    def test_a_minio_endpoint_is_used_path_style(self, monkeypatch) -> None:
        """Path-style (`host/bucket/key`), which a MinIO on `localhost`
        needs: a virtual-hosted `bucket.localhost` would not resolve."""
        monkeypatch.setenv(ENDPOINT_URL_VAR, MINIO)
        store = S3Store("bucket")
        assert store.client.meta.endpoint_url == MINIO
        assert self._presigned(store) == f"{MINIO}/bucket/k"
