"""alpha-engine-config-I9798: the sweep for a published release object with
no Object Lock retention — asserted against a fake S3 client, never a mock
of `crucible.release_lock_sweep` itself.
"""

from __future__ import annotations

import datetime as dt

import pytest
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

    def throttle(self, key: str, *, last_modified=None) -> None:
        """A transient `get_object_retention` failure that is neither
        denial, absence, nor the key having vanished."""
        self.objects[key] = {"LastModified": last_modified or _PUBLISHED_AT}
        self.retentions[key] = "Throttling"

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

    def test_a_transient_failure_is_unmeasurable_and_named_never_a_false_deletion(self) -> None:
        """Round 2 re-verification: labelling EVERY non-AccessDenied,
        non-NoSuchObjectLockConfiguration `ClientError` as a mid-sweep
        deletion gave `Throttling` a false deletion diagnosis. `Throttling`
        is neither denial, absence, nor the key having vanished — it must
        read `UNMEASURABLE` and its detail must name `Throttling`, not
        claim the object was deleted."""
        client = _FakeS3Client()
        store = _store(client)
        key = release_json_key(SHA_RACE)
        client.throttle(f"crucible/{key}")

        findings = release_lock_findings(store)

        assert findings[0].state == "UNMEASURABLE"
        assert "Throttling" in findings[0].detail
        assert "deleted" not in findings[0].detail

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

    def test_a_wheel_below_the_sha_prefix_is_never_checked(self) -> None:
        """alpha-engine-config-I9917 item 2. The pattern's own docstring says
        "directly under the sha prefix", but `crucible-.+\\.whl` let `.` match
        `/`, so a key whose FIRST segment merely starts with `crucible-`
        matched at any depth below it — the regex contradicted the invariant
        it was documented to enforce.

        The issue's own example (`releases/{sha}/nested/crucible-x.whl`) does
        NOT reproduce it: the segment after the sha has to start with
        `crucible-` for `.+` to be reached at all. The case below does, and is
        the one this narrowing actually closes.

        A nested key is not a published release object: `publish_release`
        writes exactly two keys directly under the prefix. Checking one would
        report an Object Lock finding against something the release contract
        does not own, and the sweep's denominator would count it."""
        client = _FakeS3Client()
        store = _store(client)
        nested = f"releases/{SHA_LOCKED}/crucible-staging/inner/build.whl"
        client.put(f"crucible/{nested}", locked=False)  # would be UNMET if checked
        client.put(f"crucible/{release_json_key(SHA_LOCKED)}", locked=True)

        findings = release_lock_findings(store)

        assert [f.key for f in findings] == [release_json_key(SHA_LOCKED)]

    def test_a_wheel_directly_under_the_sha_prefix_is_still_checked(self) -> None:
        """The other direction, so the narrowing cannot be satisfied by a
        pattern that matches nothing. Both the current PEP 440 name and the
        pre-I9908 legacy name still match."""
        for filename in (
            "crucible-0.1.0+gaaaaaaaaaaaa-py3-none-any.whl",
            f"crucible-{SHA_UNLOCKED}-py3-none-any.whl",
        ):
            client = _FakeS3Client()
            store = _store(client)
            key = f"releases/{SHA_UNLOCKED}/{filename}"
            client.put(f"crucible/{key}", locked=False)

            findings = release_lock_findings(store)

            assert [f.key for f in findings] == [key], filename

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
            ReleaseLockReading(
                "denied", "UNMEASURABLE", "AccessDenied",
                cause="get_object_retention:AccessDenied",
            ),
        ]
        metric = release_lock_metric(findings, now=_PUBLISHED_AT)
        assert metric["status"] == "BREACH"
        assert metric["value"] == 1.0
        assert "unlocked" in metric["status_reason"]

    def test_unmeasurable_without_any_unmet_is_unmeasurable_not_ok(self) -> None:
        findings = [
            ReleaseLockReading("locked", "MET", "locked"),
            ReleaseLockReading(
                "denied", "UNMEASURABLE", "AccessDenied",
                cause="get_object_retention:AccessDenied",
            ),
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


# --- The unmeasurable summary must be actionable (alpha-engine-config-I9952) -
#
# The first live sweep (2026-09-04T01:01:30Z) wrote
# "120 of 120 release object(s) could not be read for retention: <120 keys>"
# -- over 12 KB of one manifest field, naming no cause. Four conditions reach
# UNMEASURABLE (denied / deleted mid-sweep / transient / malformed) and they
# have four different fixes, so a reader could not act on it. It took a
# `simulate-principal-policy` call to learn what the sweep already knew on
# every single reading.


class TestUnmeasurableSummaryNamesTheCause:

    def test_an_unmeasurable_reading_without_a_cause_is_refused(self) -> None:
        """The field is not optional where it is load-bearing. An
        unattributed UNMEASURABLE is exactly the state I9952 removes, so it
        fails at construction rather than surfacing as an unactionable
        summary a day later."""
        with pytest.raises(ValueError, match="cause"):
            ReleaseLockReading("k", "UNMEASURABLE", "something went wrong")

    def test_met_and_unmet_need_no_cause(self) -> None:
        """Nothing is unexplained about them, so requiring a token there
        would be ceremony."""
        assert ReleaseLockReading("k", "MET", "locked").cause == ""
        assert ReleaseLockReading("k", "UNMET", "no retention").cause == ""

    def test_the_reason_counts_by_cause_commonest_first(self) -> None:
        findings = [
            ReleaseLockReading(
                f"denied-{i}", "UNMEASURABLE", "denied",
                cause="get_object_retention:AccessDenied",
            )
            for i in range(3)
        ] + [
            ReleaseLockReading(
                "gone", "UNMEASURABLE", "deleted",
                cause="get_object_retention:NoSuchKey (deleted mid-sweep)",
            ),
        ]
        reason = release_lock_metric(findings, now=_PUBLISHED_AT)["status_reason"]
        assert "3 x get_object_retention:AccessDenied" in reason
        assert "1 x get_object_retention:NoSuchKey (deleted mid-sweep)" in reason
        # Commonest first: the majority cause is the one to act on.
        assert reason.index("3 x ") < reason.index("1 x ")

    def test_the_real_incident_reads_as_one_grant_to_add(self) -> None:
        """The 2026-09-04 shape, reproduced: 120 keys, all AccessDenied on
        `get_object_retention`. The summary must name the call and the code,
        which together name the missing IAM action."""
        findings = [
            ReleaseLockReading(
                f"releases/{i:040x}/release.json", "UNMEASURABLE", "denied",
                cause="get_object_retention:AccessDenied",
            )
            for i in range(120)
        ]
        metric = release_lock_metric(findings, now=_PUBLISHED_AT)
        assert metric["status"] == "unmeasurable"
        assert "120 x get_object_retention:AccessDenied" in metric["status_reason"]

    def test_the_reason_truncates_the_key_list_and_says_that_it_did(self) -> None:
        """12 KB of keys in one field is not a summary. Truncation must be
        stated, never silent -- a reader who cannot tell a list of five from
        a list of 120 is being misled about the blast radius."""
        findings = [
            ReleaseLockReading(
                f"k{i}", "UNMEASURABLE", "denied",
                cause="get_object_retention:AccessDenied",
            )
            for i in range(120)
        ]
        reason = release_lock_metric(findings, now=_PUBLISHED_AT)["status_reason"]
        assert "and 115 more" in reason
        assert len(reason) < 500
        # The full record survives elsewhere: every key is on its own reading,
        # which `track_c.sweep_handler` writes to `release_lock_findings`.
        assert "k0" in reason

    def test_a_short_list_is_not_truncated(self) -> None:
        findings = [
            ReleaseLockReading(
                f"k{i}", "UNMEASURABLE", "denied",
                cause="get_object_retention:AccessDenied",
            )
            for i in range(2)
        ]
        reason = release_lock_metric(findings, now=_PUBLISHED_AT)["status_reason"]
        assert "more" not in reason
        assert "k0" in reason
        assert "k1" in reason

    def test_the_breach_reason_is_capped_too(self) -> None:
        """A BREACH over every object has the same 12 KB problem, and BREACH
        is the case somebody actually reads."""
        findings = [
            ReleaseLockReading(f"k{i}", "UNMET", "no retention") for i in range(120)
        ]
        metric = release_lock_metric(findings, now=_PUBLISHED_AT)
        assert metric["status"] == "BREACH"
        assert "and 115 more" in metric["status_reason"]
