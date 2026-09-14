"""Which release was in force at an instant, and what its registry declared.

`alpha-engine-config-I10718`. :func:`crucible.alerts.evaluate_absence` graded
every past day in its catch-up window against the `components.yaml` in the
grading process, so a row whose starter was declared on day D paged absent for
days before D, days no starter in force was ever going to run it. Measured
2026-09-14 on the live store: `gate.close` paged for 2026-09-03 though its
`dispatch: github-actions` landed 2026-09-06 (caefd0e), and `gate` paged for
2026-09-09 though its deadline and `gate-publish` starter landed 2026-09-11
(4da73b1).

`dispatch: arc` rows already read the arc's own manifest (crucible-PR279).
`scheduler` and `github-actions` rows have no per-day dispatcher record, so
the source of truth here is the same one
:class:`crucible.gate._ArcRegistryHistory` uses (`alpha-engine-config-I10478`):
a release's published wheel carries the `components.yaml` it shipped. The
release in force at an instant is read from the object-version history of
`releases/current`. The bucket is versioned (S3 Object Lock requires it), each
version is a conditional PUT by the deploy identity that CloudTrail attributes,
and noncurrent versions are retained for 365 days.

Why not a hand-written `since:` field per row: the PR that adds the row picks
the date, backdating it is a one-line edit no artifact contradicts, and one
entry per row excusing days is a suppression collection (repo rule 4). Why not
git history at evaluation time: the sweep runs from an installed wheel with no
repository.

**It only ever narrows, and only on proof.** Every shape that is not a
readable declaration (no version history on this backend, no pointer version
at or before the instant, a delete marker, an unreadable version, a release
whose wheel cannot be read) returns ``UNKNOWN``, and the caller grades the row
against the current registry exactly as before. A read that fails on ACCESS is
flagged so the caller can carry it as an access fault rather than dropping it.
"""

from __future__ import annotations

import datetime as dt
import io
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import yaml

from crucible.documents import load_document_bytes
from crucible.store import S3Store, Store

#: Where a published wheel keeps the registry it shipped with. Shared with
#: :mod:`crucible.gate`, which reads the same member for the arc's stages.
RELEASE_REGISTRY_MEMBER = "crucible/components.yaml"

#: The three answers. Deliberately not a bool: "no evidence" must never be
#: confused with "declared nothing".
DECLARED = "declared"
UNDECLARED = "undeclared"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class WheelRead:
    """A release's wheel bytes, or why they could not be read. Never both."""

    raw: bytes | None
    problem: str | None
    access_problem: bool = False


def read_release_wheel(store: Store, sha: str) -> WheelRead:
    """The published wheel for ``sha``, resolved through `crucible.release`.

    Lifted out of `crucible.gate._ArcRegistryHistory` on its second adoption
    (`policy-shared-code`). The problem strings are the ones that class
    already reported, so its callers read exactly as before.
    """
    from crucible.release import published_wheel_key  # noqa: PLC0415 - cycle

    try:
        wheel = published_wheel_key(store, sha)
    except Exception as exc:  # noqa: BLE001 - returned as the problem, never suppressed
        return WheelRead(
            None,
            f"release {sha} could not be resolved to a published wheel: "
            f"{type(exc).__name__}: {exc}",
        )
    try:
        present = store.exists(wheel)
        raw = store.get_bytes(wheel) if present else None
    except Exception as exc:  # noqa: BLE001 - returned as an access problem, never suppressed
        return WheelRead(
            None,
            f"{wheel} could not be read: {type(exc).__name__}: {exc}. That is a statement "
            "about our access, not about the system being measured",
            access_problem=True,
        )
    if raw is None:
        return WheelRead(None, f"{wheel} is absent, so what that release declared cannot be read")
    return WheelRead(raw, None)


def registry_rows(raw: bytes) -> dict[str, dict[str, Any]]:
    """Name -> row mapping of a historical `components.yaml`, parsed narrowly.

    Not `crucible.components.load_registry`: that validates every row against
    TODAY's model, and a release predating a since-required field would raise
    there. Only the fields a declaration test reads are consulted.
    """
    document = yaml.safe_load(raw) or {}
    rows = document.get("components") or {}
    items = rows.items() if isinstance(rows, dict) else ((r.get("name"), r) for r in rows)
    return {str(name): (row or {}) for name, row in items if name}


def row_has_a_starter(row: dict[str, Any] | None) -> bool:
    """Whether a registry row, as a release shipped it, would have been started
    and graded: ACTIVE, with a declared starter and a deadline."""
    if not row:
        return False
    return (
        row.get("lifecycle") == "ACTIVE"
        and row.get("dispatch") in ("arc", "scheduler", "github-actions")
        and bool(row.get("deadline"))
    )


@dataclass(frozen=True)
class PointerVersion:
    """One object version of `releases/current`."""

    at: dt.datetime
    version_id: str
    is_delete_marker: bool = False


@dataclass(frozen=True)
class Declaration:
    """What the release in force at an instant declared about one row."""

    verdict: str
    detail: str
    access_problem: bool = False


def _s3_pointer_versions(store: S3Store, key: str) -> list[PointerVersion]:
    s3_key = store._s3_key(key)
    paginator = store.client.get_paginator("list_object_versions")
    found: list[PointerVersion] = []
    for page in paginator.paginate(Bucket=store.bucket, Prefix=s3_key):
        for entry in page.get("Versions", []):
            if entry.get("Key") == s3_key:
                found.append(PointerVersion(_as_utc(entry["LastModified"]), entry["VersionId"]))
        for entry in page.get("DeleteMarkers", []):
            if entry.get("Key") == s3_key:
                found.append(
                    PointerVersion(_as_utc(entry["LastModified"]), entry["VersionId"], True)
                )
    return found


def _s3_read_version(store: S3Store, key: str, version_id: str) -> bytes:
    response = store.client.get_object(
        Bucket=store.bucket, Key=store._s3_key(key), VersionId=version_id
    )
    return response["Body"].read()


def _as_utc(value: dt.datetime) -> dt.datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=dt.UTC)
    return value.astimezone(dt.UTC)


@dataclass
class ReleaseInForce:
    """Answers "did the release in force at ``instant`` declare ``row``?".

    ``list_versions`` returns every version of the pointer, or ``None`` when
    the backend keeps no version history (a `LocalStore`), which is no
    evidence. ``read_version`` returns one version's body. Both are injected
    so the decision logic is tested without a network; :meth:`for_store`
    wires the S3 implementations.

    Consulted lazily and cached: a pass whose rows all delivered makes no
    reads, and each release's wheel is read at most once.
    """

    store: Store
    list_versions: Callable[[], Sequence[PointerVersion] | None]
    read_version: Callable[[str], bytes]
    _versions: tuple[PointerVersion, ...] | None = field(default=None, init=False)
    _versions_problem: str | None = field(default=None, init=False)
    _versions_read: bool = field(default=False, init=False)
    _sha_by_version: dict[str, tuple[str | None, str, bool]] = field(
        default_factory=dict, init=False
    )
    _rows_by_sha: dict[str, tuple[dict[str, dict[str, Any]] | None, str, bool]] = field(
        default_factory=dict, init=False
    )

    @classmethod
    def for_store(cls, store: Store) -> ReleaseInForce:
        from crucible.release import POINTER_KEY  # noqa: PLC0415 - cycle

        if isinstance(store, S3Store):
            return cls(
                store,
                lambda: _s3_pointer_versions(store, POINTER_KEY),
                lambda version_id: _s3_read_version(store, POINTER_KEY, version_id),
            )
        return cls(store, lambda: None, _no_version_history)

    def declaration(self, name: str, instant: dt.datetime) -> Declaration:
        versions, problem = self._all_versions()
        if problem is not None:
            return Declaration(UNKNOWN, problem, access_problem=True)
        if versions is None:
            return Declaration(
                UNKNOWN,
                f"{type(self.store).__name__} keeps no version history of the release pointer",
            )
        at = instant.astimezone(dt.UTC)
        in_force = [v for v in versions if v.at <= at]
        if not in_force:
            return Declaration(
                UNKNOWN,
                f"no version of the release pointer at or before {at.isoformat()} is retained",
            )
        version = max(in_force, key=lambda v: v.at)
        if version.is_delete_marker:
            return Declaration(
                UNKNOWN, f"the release pointer was deleted at {version.at.isoformat()}"
            )
        sha, detail, access = self._sha_of(version)
        if sha is None:
            return Declaration(UNKNOWN, detail, access_problem=access)
        rows, detail, access = self._rows_of(sha)
        if rows is None:
            return Declaration(UNKNOWN, detail, access_problem=access)
        if row_has_a_starter(rows.get(name)):
            return Declaration(
                DECLARED, f"release {sha} (in force at {at.isoformat()}) declared it"
            )
        return Declaration(
            UNDECLARED,
            f"release {sha}, in force at {at.isoformat()} since {version.at.isoformat()}, "
            f"declared no started, ACTIVE, deadlined `{name}` row",
        )

    def _all_versions(self) -> tuple[tuple[PointerVersion, ...] | None, str | None]:
        if not self._versions_read:
            self._versions_read = True
            try:
                listed = self.list_versions()
            except Exception as exc:  # noqa: BLE001 - returned as an access problem, never suppressed
                self._versions_problem = (
                    f"the release pointer's version history could not be listed: "
                    f"{type(exc).__name__}: {exc}. That is a statement about our access, not "
                    "about the system being measured"
                )
            else:
                self._versions = None if listed is None else tuple(listed)
        return self._versions, self._versions_problem

    def _sha_of(self, version: PointerVersion) -> tuple[str | None, str, bool]:
        if version.version_id not in self._sha_by_version:
            self._sha_by_version[version.version_id] = self._read_sha(version)
        return self._sha_by_version[version.version_id]

    def _read_sha(self, version: PointerVersion) -> tuple[str | None, str, bool]:
        from crucible.release import POINTER_KEY, parse_release_pointer  # noqa: PLC0415 - cycle

        source = f"{POINTER_KEY}?versionId={version.version_id}"
        try:
            raw = self.read_version(version.version_id)
        except Exception as exc:  # noqa: BLE001 - returned as an access problem, never suppressed
            return None, f"{source} could not be read: {type(exc).__name__}: {exc}", True
        try:
            return parse_release_pointer(source, load_document_bytes(source, raw)).sha, "", False
        except Exception as exc:  # noqa: BLE001 - returned as the problem, never suppressed
            return None, f"{source} is not a readable release pointer: {exc}", False

    def _rows_of(self, sha: str) -> tuple[dict[str, dict[str, Any]] | None, str, bool]:
        if sha not in self._rows_by_sha:
            wheel = read_release_wheel(self.store, sha)
            if wheel.raw is None:
                self._rows_by_sha[sha] = (None, str(wheel.problem), wheel.access_problem)
            else:
                try:
                    with zipfile.ZipFile(io.BytesIO(wheel.raw)) as archive:
                        rows = registry_rows(archive.read(RELEASE_REGISTRY_MEMBER))
                    self._rows_by_sha[sha] = (rows, "", False)
                except Exception as exc:  # noqa: BLE001 - returned as the problem, never suppressed
                    self._rows_by_sha[sha] = (
                        None,
                        f"release {sha}'s wheel carries no readable {RELEASE_REGISTRY_MEMBER}: "
                        f"{type(exc).__name__}: {exc}",
                        False,
                    )
        return self._rows_by_sha[sha]


def _no_version_history(version_id: str) -> bytes:
    raise RuntimeError(
        f"asked for pointer version {version_id} on a backend with no version history; "
        "list_versions returned None, so nothing should ask"
    )
