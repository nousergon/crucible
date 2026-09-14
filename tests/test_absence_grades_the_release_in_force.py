"""A scheduler / GitHub Actions row is due on a day only if the release in
force at that day's deadline declared it (alpha-engine-config-I10718).

**The measured defect.** `crucible.alerts.evaluate_absence` graded every past
day against the registry in the grading process. Read-only on the live store,
2026-09-14 (with crucible-PR279 applied), it still paged:

* `gate.close` for 2026-09-03: its `dispatch: github-actions` landed in
  caefd0e (2026-09-06T20:42Z). No starter in force on 09-03 would run it.
* `gate` for 2026-09-09: its deadline and `gate-publish` starter landed in
  4da73b1 (2026-09-12T02:04Z). Before that the row had no deadline.
* `gate.close` for 2026-09-08: declared, and GitHub Actions runs 34174942075
  and 34297633089 failed. A REAL miss, which must still page.

The release in force at an instant comes from the object-version history of
`releases/current`; its registry comes from that release's published wheel.
Both directions are pinned here, plus every shape that is not a readable
declaration, each of which must leave the page standing.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import zipfile

import pytest
import yaml

from crucible.alerts import SWEEP_JOB, StoreAccessError, evaluate_absence
from crucible.components import REGISTRY_PATH
from crucible.keys import POINTER_KEY, manifest_key
from crucible.release import ReleaseRecord, release_json_key, wheel_key_for
from crucible.release_history import (
    DECLARED,
    UNDECLARED,
    UNKNOWN,
    PointerVersion,
    ReleaseInForce,
    _s3_pointer_versions,
)
from crucible.store import LocalStore, S3Store

#: Releases standing for the three registries the live pointer named.
BEFORE_GATE_CLOSE = "a" * 40  # neither `gate.close` nor a deadlined `gate`
BEFORE_GATE_PUBLISH = "c" * 40  # caefd0e: `gate.close` declared, `gate` not
TODAY = "e" * 40  # 4da73b1 onward: both declared

FLIPS = (
    (dt.datetime(2026, 9, 2, 0, 0, tzinfo=dt.UTC), BEFORE_GATE_CLOSE),
    (dt.datetime(2026, 9, 6, 20, 50, tzinfo=dt.UTC), BEFORE_GATE_PUBLISH),
    (dt.datetime(2026, 9, 12, 2, 10, tzinfo=dt.UTC), TODAY),
)

#: Two instants, because the catch-up window is five trading days and 09-03
#: and 09-09 are not both inside one of them.
AT_09_10 = dt.datetime(2026, 9, 10, 13, 0, tzinfo=dt.UTC)
AT_09_14 = dt.datetime(2026, 9, 14, 13, 0, tzinfo=dt.UTC)

D_0903 = dt.date(2026, 9, 3)
D_0908 = dt.date(2026, 9, 8)
D_0909 = dt.date(2026, 9, 9)


def _registry_without(*, gate_close: bool, gate_deadline: bool) -> bytes:
    document = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    rows = document["components"]
    if not gate_close:
        del rows["gate.close"]
    if not gate_deadline:
        rows["gate"]["deadline"] = None
        rows["gate"]["dispatch"] = None
        rows["gate"]["schedule"] = None
    return yaml.safe_dump(document).encode()


def _publish(store: LocalStore, sha: str, registry: bytes) -> None:
    wheel_filename = f"crucible-0.1.0+g{sha[:12]}-py3-none-any.whl"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("crucible/components.yaml", registry)
    store.put_bytes(wheel_key_for(sha, wheel_filename), buffer.getvalue())
    record = ReleaseRecord(
        schema_version="release.v3",
        sha=sha,
        lockfile_sha256="0" * 64,
        wheel_sha256="0" * 64,
        wheel_filename=wheel_filename,
    )
    store.put_bytes(release_json_key(sha), record.to_json())


def _pointer(sha: str) -> bytes:
    return json.dumps(
        {"sha": sha, "target": "current", "pinned_at": "2026-09-01T00:00:00Z"}
    ).encode()


def _history(store: LocalStore, flips=FLIPS, **overrides) -> ReleaseInForce:
    versions = [PointerVersion(at, f"v-{sha}") for at, sha in flips]
    bodies = {f"v-{sha}": _pointer(sha) for _, sha in flips}
    return ReleaseInForce(
        store,
        overrides.get("list_versions", lambda: versions),
        overrides.get("read_version", lambda version_id: bodies[version_id]),
    )


def _store(tmp_path) -> LocalStore:
    """The live shape: the three releases, the sweep running every day, and
    `gate.close` manifests only where the live store has them."""
    store = LocalStore(tmp_path)
    _publish(store, BEFORE_GATE_CLOSE, _registry_without(gate_close=False, gate_deadline=False))
    _publish(store, BEFORE_GATE_PUBLISH, _registry_without(gate_close=True, gate_deadline=False))
    _publish(store, TODAY, REGISTRY_PATH.read_bytes())
    day = dt.date(2026, 8, 31)
    while day <= dt.date(2026, 9, 14):
        if day.weekday() < 5:
            key = manifest_key(SWEEP_JOB, day.isoformat(), discriminator=day.isoformat())
            store.put_bytes(key, b'{"status": "ok"}')
        day += dt.timedelta(days=1)
    for live in ("2026-09-04", "2026-09-09", "2026-09-10", "2026-09-11"):
        store.put_bytes(manifest_key("gate.close", live), b'{"status": "ok"}')
    return store


def _paged(store, at, history=None, **kwargs) -> set[tuple[str, dt.date]]:
    return {
        (p.job, p.trading_day)
        for p in evaluate_absence(store, now=at, release_history=history, **kwargs)
    }


class TestTheLiveShape:
    def test_gate_close_is_not_absent_on_a_day_before_its_starter_was_declared(
        self, tmp_path
    ) -> None:
        store = _store(tmp_path)
        assert ("gate.close", D_0903) not in _paged(store, AT_09_10, _history(store))

    def test_gate_is_not_absent_on_a_day_before_its_publisher_was_declared(self, tmp_path) -> None:
        store = _store(tmp_path)
        assert ("gate", D_0909) not in _paged(store, AT_09_14, _history(store))

    def test_the_real_gate_close_miss_on_09_08_still_pages(self, tmp_path) -> None:
        store = _store(tmp_path)
        assert ("gate.close", D_0908) in _paged(store, AT_09_10, _history(store))
        assert ("gate.close", D_0908) in _paged(store, AT_09_14, _history(store))

    def test_without_the_history_both_retroactive_pages_come_back(self, tmp_path) -> None:
        """The narrowing is the history's doing: a LocalStore keeps no
        pointer versions, so the default reads no evidence."""
        store = _store(tmp_path)
        assert ("gate.close", D_0903) in _paged(store, AT_09_10)
        assert ("gate", D_0909) in _paged(store, AT_09_14)


class TestEveryShapeThatIsNotADeclarationLeavesThePage:
    def test_no_pointer_version_at_or_before_the_deadline(self, tmp_path) -> None:
        store = _store(tmp_path)
        history = _history(store, flips=FLIPS[1:])
        assert (
            history.declaration("gate.close", dt.datetime(2026, 9, 4, tzinfo=dt.UTC)).verdict
            == UNKNOWN
        )
        assert ("gate.close", D_0903) in _paged(store, AT_09_10, history)

    def test_a_delete_marker_in_force_is_no_evidence(self, tmp_path) -> None:
        store = _store(tmp_path)
        versions = [
            PointerVersion(FLIPS[0][0], f"v-{BEFORE_GATE_CLOSE}"),
            PointerVersion(dt.datetime(2026, 9, 3, tzinfo=dt.UTC), "gone", is_delete_marker=True),
        ]
        history = _history(store, list_versions=lambda: versions)
        assert ("gate.close", D_0903) in _paged(store, AT_09_10, history)

    def test_a_release_whose_wheel_is_missing_is_no_evidence(self, tmp_path) -> None:
        store = _store(tmp_path)
        history = _history(store, flips=((FLIPS[0][0], "f" * 40),))
        declaration = history.declaration("gate.close", dt.datetime(2026, 9, 4, tzinfo=dt.UTC))
        assert declaration.verdict == UNKNOWN
        assert not declaration.access_problem
        assert ("gate.close", D_0903) in _paged(store, AT_09_10, history)

    def test_an_unparseable_pointer_version_is_no_evidence(self, tmp_path) -> None:
        store = _store(tmp_path)
        history = _history(store, read_version=lambda version_id: b"{truncated")
        assert ("gate.close", D_0903) in _paged(store, AT_09_10, history)

    def test_arc_rows_are_not_graded_by_the_release_history(self, tmp_path) -> None:
        """`data.weekly` is an arc row; the arc's own manifest owns that
        question (crucible-PR279). A release declaring nothing does not
        narrow it."""
        store = _store(tmp_path)
        _publish(store, "9" * 40, b"version: 1\ncomponents: {}\n")
        history = _history(store, flips=((FLIPS[0][0], "9" * 40),))
        assert ("data.weekly", dt.date(2026, 9, 4)) in _paged(store, AT_09_10, history)


class TestAnUnreadableHistoryIsAnAccessFault:
    def test_listing_denied_is_carried_out_and_the_page_still_stands(self, tmp_path) -> None:
        store = _store(tmp_path)

        def denied():
            raise PermissionError("AccessDenied: s3:ListBucketVersions")

        history = _history(store, list_versions=denied)
        faults: list[str] = []
        assert ("gate.close", D_0903) in _paged(store, AT_09_10, history, access_faults=faults)
        assert len(faults) == 1 and "ListBucketVersions" in faults[0]
        with pytest.raises(StoreAccessError):
            _paged(store, AT_09_10, _history(store, list_versions=denied))

    def test_a_version_read_denied_is_an_access_fault(self, tmp_path) -> None:
        store = _store(tmp_path)

        def denied(version_id: str) -> bytes:
            raise PermissionError("AccessDenied: s3:GetObjectVersion")

        declaration = _history(store, read_version=denied).declaration(
            "gate.close", dt.datetime(2026, 9, 4, tzinfo=dt.UTC)
        )
        assert declaration.verdict == UNKNOWN and declaration.access_problem


class TestTheDeclarationItself:
    def test_both_directions(self, tmp_path) -> None:
        store = _store(tmp_path)
        history = _history(store)
        before = dt.datetime(2026, 9, 5, tzinfo=dt.UTC)
        after = dt.datetime(2026, 9, 7, tzinfo=dt.UTC)
        assert history.declaration("gate.close", before).verdict == UNDECLARED
        assert history.declaration("gate.close", after).verdict == DECLARED
        assert history.declaration("gate", after).verdict == UNDECLARED
        assert history.declaration("gate", AT_09_14).verdict == DECLARED

    def test_the_boundary_is_inclusive_of_the_flip_instant(self, tmp_path) -> None:
        store = _store(tmp_path)
        assert _history(store).declaration("gate.close", FLIPS[1][0]).verdict == DECLARED


class _Paginator:
    def __init__(self, pages):
        self.pages = pages
        self.calls = []

    def paginate(self, **kwargs):
        self.calls.append(kwargs)
        return iter(self.pages)


class _Client:
    def __init__(self, pages, bodies):
        self.paginator = _Paginator(pages)
        self.bodies = bodies

    def get_paginator(self, name):
        assert name == "list_object_versions"
        return self.paginator

    def get_object(self, *, Bucket, Key, VersionId):  # noqa: N803 - boto3's own keyword names
        return {"Body": io.BytesIO(self.bodies[VersionId])}


class TestTheS3Reader:
    def test_reads_every_page_and_only_the_pointer_key(self) -> None:
        pointer = f"crucible/{POINTER_KEY}"
        pages = [
            {"Versions": [{"Key": pointer, "LastModified": FLIPS[0][0], "VersionId": "v1"}]},
            {
                "Versions": [
                    {"Key": f"{pointer}-sibling", "LastModified": FLIPS[2][0], "VersionId": "x"},
                    {"Key": pointer, "LastModified": FLIPS[1][0], "VersionId": "v2"},
                ],
                "DeleteMarkers": [{"Key": pointer, "LastModified": FLIPS[2][0], "VersionId": "d1"}],
            },
        ]
        client = _Client(pages, {"v2": _pointer(BEFORE_GATE_PUBLISH)})
        store = S3Store("bucket", "crucible", client=client)
        versions = _s3_pointer_versions(store, POINTER_KEY)
        assert [(v.version_id, v.is_delete_marker) for v in versions] == [
            ("v1", False),
            ("v2", False),
            ("d1", True),
        ]
        assert client.paginator.calls == [{"Bucket": "bucket", "Prefix": pointer}]
        history = ReleaseInForce.for_store(store)
        assert history._read_sha(versions[1])[0] == BEFORE_GATE_PUBLISH
