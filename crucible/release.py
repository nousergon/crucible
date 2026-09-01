"""One artifact, one pointer, one verified flip.

Normative source: plan §4.11.

    releases/{sha}/crucible-{sha}-py3-none-any.whl
    releases/{sha}/release.json
    releases/current            -> {"sha": ...}   (conditional PUT)
    runs/deploy/{trading_day}/run.json            (run-manifest schema)

**Nothing is installed anywhere at deploy time.** A job pulls `releases/current`
at start and installs that wheel. There is no box to SSH into, no `git pull` on
boot, and therefore none of the `fix-boot-pull-netrc` bug class.

**The pointer flip is a compare-and-swap, not a write.** Two deploys racing —
two merges inside one smoke window — must not give the pointer to whichever
finished last: that is the last-writer-wins shape that once handed a cycle's
verdict to its worst-informed author. The loser re-reads and fails the deploy
rather than overwriting.

**The flip is gated on a REAL run.** `crucible smoke --release {sha}` executes
against live S3 read paths and writes `runs/smoke/{trading_day}/run.json`
through the ordinary runner, so the gate produces the same telemetry every
other job does. `status: ok` flips the pointer; anything else leaves it exactly
where it was and fails the deploy. A smoke that "ran and had nothing to do" is
not a thing the manifest can express (§4.2), which is what makes this gate
meaningful rather than ceremonial.

**Rollback is the same primitive backwards.** `crucible release.pin <sha>`
repoints `current` in about two seconds, with no rebuild. Every release stays
in S3 (retain-archives).

**The trader never follows `current`.** It pins, at `trader/release_pin`, and
promotion to the trader is an explicit off-market-hours action.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from crucible.store import ETAG_ABSENT, PointerConflictError, Store, sha256_hex

__all__ = [
    "POINTER_KEY",
    "PointerConflictError",
    "flip_on_smoke",
    "RELEASE_SCHEMA_VERSION",
    "ReleaseRecord",
    "StaleReleasePointerError",
    "TRADER_PIN_KEY",
    "current_release",
    "pin",
    "publish_release",
    "read_pointer",
    "release_json_key",
    "release_prefix",
    "resolve_release",
    "wheel_key",
    "write_deploy_manifest",
]

RELEASE_SCHEMA_VERSION = "release.v1"

#: The single mutable object in the whole release layout. Everything else is
#: immutable and content-addressed by the sha in its own prefix.
POINTER_KEY = "releases/current"

#: The trader's separate pin. Two pointers, never one: a trader that followed
#: `current` would be promoted by every merge, and release promotion to the
#: trader is an explicit off-market-hours action (§4.11).
TRADER_PIN_KEY = "trader/release_pin"

#: Exhaustive. `current` is what jobs follow; `trader` is what the trader
#: follows. A third target is a design change.
PIN_TARGETS: tuple[str, ...] = ("current", "trader")

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class StaleReleasePointerError(RuntimeError):
    """`releases/current` names a sha whose artifacts are not there.

    A distinct type because it is a distinct operator action: the release
    prefix has to be restored or the pointer moved back, and neither is what
    you do about a transient S3 error. It is also one of the four scripted
    faults (§10.7) — the failure path is proven by inducing it, not by
    reading this docstring.
    """


def _assert_sha(sha: str) -> str:
    if not _SHA_RE.match(sha):
        raise ValueError(
            f"{sha!r} is not a 40-character lowercase git sha. The release layout is "
            "content-addressed by the commit; an abbreviated or uppercase sha would "
            "produce a second prefix for the same build."
        )
    return sha


def release_prefix(sha: str) -> str:
    return f"releases/{_assert_sha(sha)}"


def wheel_key(sha: str) -> str:
    """The wheel's key. Named by sha, not by version.

    `pyproject`'s version moves rarely and a build is identified by its
    commit; two builds sharing a version string in one prefix would be
    indistinguishable in a rollback.
    """
    return f"{release_prefix(sha)}/crucible-{sha}-py3-none-any.whl"


def release_json_key(sha: str) -> str:
    return f"{release_prefix(sha)}/release.json"


@dataclass(frozen=True)
class ReleaseRecord:
    """What `release.json` carries. Everything needed to answer "what is this".

    `lockfile_sha256` is here because the wheel does not pin its own
    transitive tree: two wheels built from one commit against two resolved
    dependency sets are two different artifacts, and only the lockfile hash
    says which one this is.
    """

    schema_version: str
    sha: str
    built_at: str
    lockfile_sha256: str
    wheel_sha256: str
    test_summary: str
    workflow_run_url: str
    python_requires: str = ">=3.12,<3.13"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> bytes:
        return json.dumps(asdict(self), indent=2, sort_keys=True).encode("utf-8")


def publish_release(
    store: Store,
    *,
    sha: str,
    wheel: bytes,
    lockfile: bytes,
    test_summary: str,
    workflow_run_url: str,
    now: dt.datetime | None = None,
) -> ReleaseRecord:
    """Write the immutable half of a release. Does NOT touch the pointer.

    Separated from the flip on purpose: uploading the artifact is safe and
    repeatable, and moving the pointer is the act with consequences. A single
    function doing both would make "publish it but do not promote it yet"
    impossible to express, and the smoke gate sits precisely in between.
    """
    _assert_sha(sha)
    if not wheel:
        raise ValueError(f"refusing to publish an empty wheel for {sha}")
    stamp = (now or dt.datetime.now(dt.UTC)).astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    record = ReleaseRecord(
        schema_version=RELEASE_SCHEMA_VERSION,
        sha=sha,
        built_at=stamp,
        lockfile_sha256=sha256_hex(lockfile),
        wheel_sha256=sha256_hex(wheel),
        test_summary=test_summary,
        workflow_run_url=workflow_run_url,
    )
    store.put_bytes(wheel_key(sha), wheel)
    store.put_bytes(release_json_key(sha), record.to_json())
    return record


def read_pointer(store: Store, key: str = POINTER_KEY) -> tuple[str | None, str]:
    """``(sha, version_token)``. ``sha`` is None when the pointer is unset.

    The version token comes back with the value so the caller can swap
    against exactly what it read. Returning only the sha would leave the
    caller to fetch the version separately, and the gap between those two
    reads is the race this whole module is built to close.
    """
    version = store.etag(key)
    if version == ETAG_ABSENT:
        return None, ETAG_ABSENT
    payload = json.loads(store.get_bytes(key).decode("utf-8"))
    return payload["sha"], version


def current_release(store: Store) -> str | None:
    return read_pointer(store)[0]


def resolve_release(store: Store, key: str = POINTER_KEY) -> str:
    """The sha a job should install, verified to actually be there.

    Raises :class:`StaleReleasePointerError` when the pointer names a sha
    whose wheel is absent. A job that installed nothing and carried on would
    run whatever was already on the box — which is the silent version of
    every deploy bug, and the reason this is checked at read time rather than
    trusted from the pointer.
    """
    sha, _ = read_pointer(store, key)
    if sha is None:
        raise StaleReleasePointerError(
            f"{key} is unset: no release has ever been promoted. A job cannot choose "
            "a release for itself."
        )
    if not store.exists(wheel_key(sha)):
        raise StaleReleasePointerError(
            f"{key} names {sha}, whose wheel is not at {wheel_key(sha)}. Restore the "
            f"release prefix or `crucible release.pin <prior-sha>`; a job must never "
            "fall back to whatever is already installed."
        )
    return sha


def pin(
    store: Store,
    sha: str,
    *,
    target: str = "current",
    expect: str | None = None,
    now: dt.datetime | None = None,
) -> str:
    """Point ``target`` at ``sha`` by compare-and-swap. Returns the new version.

    ``expect`` is the version token the caller read. Omitting it means "swap
    against whatever is there right now", which is correct for an operator
    rollback — the operator has just looked — and wrong for an automated
    deploy, which passes the token it read before running the smoke. The
    deploy path always passes it.
    """
    _assert_sha(sha)
    if target not in PIN_TARGETS:
        raise ValueError(f"pin target {target!r} not in {PIN_TARGETS}")
    if not store.exists(wheel_key(sha)):
        raise StaleReleasePointerError(
            f"refusing to pin {target} to {sha}: no wheel at {wheel_key(sha)}. A "
            "pointer to an artifact that is not there is a stale pointer the moment "
            "it is written."
        )
    key = POINTER_KEY if target == "current" else TRADER_PIN_KEY
    version = expect if expect is not None else store.etag(key)
    payload = json.dumps(
        {
            "sha": sha,
            "target": target,
            "pinned_at": (now or dt.datetime.now(dt.UTC))
            .astimezone(dt.UTC)
            .strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    return store.compare_and_swap(key, version, payload)


def write_deploy_manifest(
    store: Store,
    manifest: dict[str, Any],
) -> str:
    """Write the deploy's own run manifest, validated like any other.

    §4.11: "the deploy writes its own `deploy.json` in the run-manifest
    schema, so §4.5's page shows deploys beside runs". It goes to
    `runs/deploy/{trading_day}/run.json` — the same key shape as every job —
    because a deploy that reported itself in a bespoke place is a deploy the
    console has to be taught about separately, and would not be.
    """
    from crucible.manifest import manifest_key, validate  # noqa: PLC0415 - cycle

    validate(manifest)
    key = manifest_key("deploy", manifest["trading_day"])
    store.put_bytes(key, json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"))
    return key


def flip_on_smoke(
    store: Store,
    *,
    sha: str,
    smoke_manifest: dict[str, Any],
    expect: str,
) -> bool:
    """Flip `releases/current` to ``sha`` iff the smoke manifest says `ok`.

    The one place the gate is expressed, and it reads the manifest's status
    rather than a boolean handed in by the caller. A caller-supplied "it
    passed" is the shape that lets a deploy promote a build whose smoke
    quietly did nothing.

    Returns True on a flip. Raises :class:`PointerConflictError` when another
    deploy moved the pointer while this smoke was running — the deploy then
    FAILS rather than overwriting, and the operator sees two deploys raced.
    """
    if smoke_manifest.get("job") != "smoke":
        raise ValueError(
            f"flip_on_smoke was handed a {smoke_manifest.get('job')!r} manifest. The "
            "gate is the smoke run, and reading any other job's status here would "
            "promote a build nothing verified."
        )
    if smoke_manifest.get("release_sha") != sha:
        raise ValueError(
            f"the smoke manifest is for release {smoke_manifest.get('release_sha')!r}, "
            f"not {sha!r}. Promoting on another build's smoke is the gate failing open."
        )
    if smoke_manifest.get("status") != "ok":
        return False
    pin(store, sha, target="current", expect=expect)
    return True
