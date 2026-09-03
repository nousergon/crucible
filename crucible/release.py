"""One artifact, one pointer, one verified flip.

Normative source: plan §4.11.

    releases/{sha}/crucible-{version}-py3-none-any.whl   (PEP-440-legal; see wheel_filename_for)
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
from functools import cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from crucible.documents import load_store_document
from crucible.keys import manifest_key
from crucible.store import ETAG_ABSENT, PointerConflictError, S3Store, Store, sha256_hex

_SCHEMA_DIR = Path(__file__).parent / "schemas"


@cache
def _validator_for(schema_filename: str) -> Draft202012Validator:
    """A cached validator for one of this module's own schema files.

    Mirrors `crucible.manifest.load_schema` / `crucible.champion.load_schema`:
    the schema ships inside the package, so a missing file means a broken
    build, not a degraded write.
    """
    path = _SCHEMA_DIR / schema_filename
    if not path.is_file():
        raise FileNotFoundError(
            f"{schema_filename} missing at {path}. It ships inside the package; a "
            "missing schema means a broken build, not a degraded write."
        )
    schema = json.loads(path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _validate_release_artifact(schema_filename: str, payload: dict[str, Any]) -> None:
    """Raise with every error, never just the first, against one of this
    module's own artifact schemas. A writer that could emit a non-conformant
    document would defeat the schema entirely — validated on the way OUT,
    not only wherever something later reads it back."""
    errors = sorted(
        _validator_for(schema_filename).iter_errors(payload), key=lambda e: list(e.absolute_path)
    )
    if not errors:
        return
    detail = "\n".join(
        f"  - {'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}" for e in errors
    )
    raise ValueError(f"{schema_filename}: document does not conform:\n{detail}")


__all__ = [
    "POINTER_KEY",
    "PointerConflictError",
    "RELEASE_PROVENANCE_SCHEMA_VERSION",
    "ReleaseImmutabilityError",
    "assert_immutable_write",
    "flip_on_smoke",
    "RELEASE_SCHEMA_VERSION",
    "ReleaseProvenance",
    "ReleaseRecord",
    "StaleReleasePointerError",
    "TRADER_PIN_KEY",
    "current_release",
    "parse_release_record",
    "pin",
    "provenance_key",
    "publish_release",
    "published_wheel_key",
    "read_pointer",
    "RETENTION_CLOCK_SKEW_SLACK",
    "release_json_key",
    "release_object_lock_params",
    "release_prefix",
    "resolve_release",
    "retention_meets_target",
    "wheel_filename_for",
    "wheel_key",
    "wheel_key_for",
    "write_deploy_manifest",
]

#: `release.json`'s schema. Bumped to v2 by alpha-engine-config-I9786: v1
#: carried `built_at` / `workflow_run_url` / `test_summary`, three fields
#: that move on every rebuild of the same commit, which made a
#: `workflow_dispatch` re-run of an unchanged `main` an UNCONDITIONAL
#: `ReleaseImmutabilityError` — `release.json` could never be byte-identical
#: across two runs even though the wheel it described was. v2 carries only
#: what is a deterministic function of the commit; the per-run facts moved to
#: :class:`ReleaseProvenance`. A v1 object already published under an older
#: build stays exactly as published — it is never rewritten — so the one
#: live transitional effect is that the FIRST re-run of a sha published
#: before this shipped still raises (the v1 and v2 bytes genuinely differ);
#: every rebuild after that is the clean no-op the issue asks for.
#:
#: Bumped to v3 by alpha-engine-config-I9908: adds `wheel_filename`. Every
#: wheel this pipeline had ever published was unpip-installable — the
#: workflow renamed it to `crucible-{sha}-py3-none-any.whl`, and a 40-hex
#: git sha is not a PEP 440 version, so pip refused the filename under
#: either name it was ever given. v3 publishes the wheel under the filename
#: `uv build` itself produced (PEP-440-legal, carrying the sha as a local
#: version segment — see `wheel_filename_for`) and records that filename so
#: a bash bootstrap on a spot box can download it by name without
#: reimplementing the version derivation in shell.
RELEASE_SCHEMA_VERSION = "release.v3"

#: The base PEP 440 version `pyproject.toml` declares. `deploy.yml` appends
#: `+g<sha12>` at BUILD time (a local version segment, never committed —
#: crucible_v2_rebuild_plan §4.11 / alpha-engine-config-I9908: "no committed
#: pyproject change per release"); this constant is the same literal so a
#: reader that never ran the build can still name the wheel a given sha
#: produced. Asserted equal to `pyproject.toml`'s own `version` by
#: `tests/test_release.py::test_base_version_matches_pyproject`, so the two
#: cannot drift silently.
_BASE_VERSION = "0.1.0"


def wheel_filename_for(sha: str) -> str:
    """The wheel's own PEP 440 filename for ``sha`` — deterministic, no I/O.

    `crucible-{version}-py3-none-any.whl` where `version` is
    `{_BASE_VERSION}+g{sha[:12]}`: a 40-hex git sha is not a PEP 440
    version and pip refuses it outright (`ERROR: crucible.whl is not a
    valid wheel filename`, alpha-engine-config-I9908's console evidence);
    the `+g<sha12>` local-version-segment form is PEP-440-legal while still
    recovering the commit from the filename (`g` for "git", following the
    convention `setuptools_scm`/`hatch-vcs` use for a dev build). `sha[:12]`
    and `_BASE_VERSION` both match `deploy.yml`'s own derivation —
    `tests/test_release.py::test_base_version_matches_pyproject` is the
    assertion that keeps the two from drifting apart.
    """
    return f"crucible-{_BASE_VERSION}+g{_assert_sha(sha)[:12]}-py3-none-any.whl"


#: `releases/{sha}/provenance/{run_id}-{run_attempt}.json`'s schema. One
#: instance per publish ATTEMPT, never immutable-checked (I9786): a second
#: attempt for an already-published sha is expected to differ in exactly
#: these fields, and each attempt is a durable record, not a contender for
#: the one slot `release.json` occupies.
RELEASE_PROVENANCE_SCHEMA_VERSION = "release_provenance.v1"

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

#: How long a published release object (the wheel, `release.json`) is locked
#: under S3 Object Lock once it lands, applied in GOVERNANCE mode. Ten years:
#: chosen to outlast any plausible rollback target for the practical lifetime
#: of this system, not to match a specific compliance horizon — a release a
#: decade old is not a rollback candidate, it is history (`preference:
#: retain ALL archives — data is an asset`). GOVERNANCE rather than
#: COMPLIANCE: an operator holding `s3:BypassGovernanceRetention` can still
#: remove a genuinely bad artifact (a leaked secret baked into a wheel, say),
#: where COMPLIANCE mode cannot be overridden by anyone, including the
#: account root, for the retention's whole duration — turning a legitimate
#: takedown into a decade-long wait. `releases/current` (:data:`POINTER_KEY`)
#: is never locked: it is a pointer, mutable by design, moved by conditional
#: PUT (see the module docstring).
RELEASE_OBJECT_LOCK_RETENTION = dt.timedelta(days=3650)


def release_object_lock_params(
    store: Store, *, now: dt.datetime | None = None
) -> tuple[str | None, dt.datetime | None]:
    """The ``(object_lock_mode, object_lock_retain_until)`` to pass on the
    write itself, or ``(None, None)`` off S3.

    Resolved by the caller of :meth:`crucible.store.Store.put_bytes.
    put_bytes` and passed on the SAME call that writes the bytes
    (alpha-engine-config-I9787) — never as a separate ``PutObjectRetention``
    call afterward, which left a window, bounded only by that one extra
    round trip, in which a published release object existed unlocked. A
    process dying in that window left a published, unprotected artifact and
    nothing detected it.

    A no-op off S3 — :class:`crucible.store.LocalStore` (the laptop and test
    backend) has no Object Lock concept, and the bucket-level
    ``ObjectLockEnabled`` flag this defends against being inert without it is
    an S3-only condition (I9787's "Gotcha").

    Called only for the wheel and ``release.json`` keys, never for
    :data:`POINTER_KEY` — the pointer flip goes through :func:`pin` /
    ``compare_and_swap``, a different code path that never reaches this.
    """
    if not isinstance(store, S3Store):
        return None, None
    stamp = (now or dt.datetime.now(dt.UTC)).astimezone(dt.UTC)
    return "GOVERNANCE", stamp + RELEASE_OBJECT_LOCK_RETENTION


#: Slack allowed when comparing a stored `RetainUntilDate` against a target
#: recomputed from `HeadObject`'s `LastModified` (alpha-engine-config-I9898
#: round-2 adversarial review, finding 1). `release.py` stamps
#: `retain_until` at publish time from `dt.datetime.now(dt.UTC)`, which S3
#: then stores at MILLISECOND precision; `HeadObject`'s `LastModified` comes
#: back at SECOND precision (S3's own truncation, not this codebase's), so
#: recomputing the target from `LastModified` and comparing it exactly
#: against the stored value is always a few hundred milliseconds short for
#: an object that is genuinely, fully compliant. A day is generous against a
#: single request round trip and costs nothing: a retention genuinely short
#: by a day is still short against the plan's ten-year horizon.
#: `crucible.release_lock_sweep` carried this same slack as a private
#: `_CLOCK_SKEW_SLACK` before I9898 round 2 — the two now share one
#: constant and one comparison (:func:`retention_meets_target`) so a sweep
#: reading MET and a repair reading `extended` for the same object can no
#: longer happen.
RETENTION_CLOCK_SKEW_SLACK = dt.timedelta(days=1)


def retention_meets_target(
    current_retain_until: dt.datetime,
    target_retain_until: dt.datetime,
    *,
    slack: dt.timedelta = RETENTION_CLOCK_SKEW_SLACK,
) -> bool:
    """Whether a stored ``current_retain_until`` already satisfies
    ``target_retain_until``, allowing ``slack`` for the millisecond-vs-second
    precision mismatch between what S3 stores and what `HeadObject` reports.

    The ONE comparison both `crucible.release_retention` (the repair) and
    `crucible.release_lock_sweep` (the read-only detector) use — living here
    rather than in either of those two modules because `release_retention`
    already imports from `release_lock_sweep` (`_RELEASE_OBJECT_RE`), so
    putting it in either of those two would make the other import back into
    it, a cycle. `crucible.release` is a dependency both already have.

    Never reversed to "target - slack <= current": written as
    "current + slack >= target" so a caller passing a genuinely-shorter
    retention (not a clock-skew artifact, but a real gap) is still correctly
    read as not meeting the target once the gap exceeds ``slack``.
    """
    return current_retain_until.astimezone(dt.UTC) + slack >= target_retain_until.astimezone(dt.UTC)


class StaleReleasePointerError(RuntimeError):
    """`releases/current` names a sha whose artifacts are not there.

    A distinct type because it is a distinct operator action: the release
    prefix has to be restored or the pointer moved back, and neither is what
    you do about a transient S3 error. It is also one of the four scripted
    faults (§10.7) — the failure path is proven by inducing it, not by
    reading this docstring.
    """


class ReleaseImmutabilityError(RuntimeError):
    """A publish would have changed the bytes of an already-published release.

    §4.11 claims "immutable versioned artifacts", and the whole rollback
    story rests on it: `crucible release.pin <prior-sha>` is only a rollback
    if the wheel under that prefix is still the wheel that was tested and
    installed. The IAM grant is `s3:PutObject` on `crucible/releases/*` with
    no Object Lock, so nothing below this line stops a second
    `workflow_dispatch` for the same sha from rebuilding the wheel and
    overwriting the one a consumer already installed — at which point
    `release.json.wheel_sha256` changes under it and the rollback target is a
    different artifact wearing the same name.

    A distinct type because it is a distinct operator action: re-promoting an
    existing build is `crucible release.pin <sha>`, which needs no rebuild,
    and re-publishing a *changed* build needs a new commit. Neither is
    "retry the deploy".
    """


def assert_immutable_write(store: Store, key: str, payload: bytes) -> bool:
    """Whether ``key`` still needs writing. Refuses a differing overwrite.

    Returns True when the key is absent (write it) and False when it already
    holds exactly ``payload`` (an idempotent re-publish — a re-run of the
    same workflow against the same artifact is a no-op, not a failure).
    Raises :class:`ReleaseImmutabilityError` when the key exists with
    different bytes.

    The comparison is on the bytes rather than on a recorded digest: a digest
    the writer supplies about its own payload is a claim, and the thing being
    protected here is precisely the case where the claim and the object have
    diverged.
    """
    if not store.exists(key):
        return True
    existing = store.get_bytes(key)
    if existing == payload:
        return False
    raise ReleaseImmutabilityError(
        f"{key} already exists with different bytes "
        f"({sha256_hex(existing)[:12]} on the store, {sha256_hex(payload)[:12]} offered). "
        "A published release is immutable: overwriting it would change the artifact a "
        "prior consumer installed and make the rollback target a build nobody tested. "
        "To re-promote this build use `crucible release.pin <sha>` (no rebuild); to "
        "publish different bytes, publish them under their own commit."
    )


def _assert_sha(sha: str) -> str:
    if not _SHA_RE.match(sha):
        raise ValueError(
            f"{sha!r} is not a 40-character lowercase git sha. The release layout is "
            "content-addressed by the commit; an abbreviated or uppercase sha would "
            "produce a second prefix for the same build."
        )
    return sha


def release_prefix(sha: str) -> str:
    """Stays here rather than in `crucible.keys` (alpha-engine-config-I9807
    class sweep): `_assert_sha` is release-domain validation reused directly
    by callers elsewhere in this module (`_assert_sha` at lines ~416, ~523),
    not a key-only helper like `crucible.keys.arm_key_segment`. Moving just
    the three key functions below would either duplicate `_assert_sha` in
    `crucible.keys` — a second source of truth for sha validation — or make
    the generic key module import a release-specific validator, which is
    backwards: nothing in `crucible.keys` depends on a domain module today.
    """
    return f"releases/{_assert_sha(sha)}"


def wheel_key(sha: str) -> str:
    """The key THIS pipeline publishes ``sha``'s wheel under (release.v3+).

    Prefixed by the sha for rollback addressing, per :func:`release_prefix`
    (a build is identified by its commit); the object name under it is
    `wheel_filename_for(sha)` — pip-installable, per alpha-engine-config-I9908.
    Callers holding a `ReleaseRecord` already parsed from a possibly older
    `release.json` should use :func:`wheel_key_for` with `record.wheel_filename`
    instead: a v2 record's wheel is not at this path (see `parse_release_record`).
    """
    return wheel_key_for(sha, wheel_filename_for(sha))


def wheel_key_for(sha: str, wheel_filename: str) -> str:
    """The key a specific wheel FILENAME lives at, for ``sha``.

    Distinct from :func:`wheel_key`, which derives the filename this
    release.v3 pipeline publishes under. This one takes the filename as
    given — from a parsed :class:`ReleaseRecord`, v2 or v3 — because a v2
    record's wheel was published under a different (unpip-installable)
    name than v3 derives, and a reader that already knows the real filename
    must not re-derive a v3-shaped guess for an object that was never
    written under it.
    """
    return f"{release_prefix(sha)}/{wheel_filename}"


def release_json_key(sha: str) -> str:
    return f"{release_prefix(sha)}/release.json"


def provenance_key(sha: str, run_id: str, run_attempt: str = "1") -> str:
    """One key per publish attempt. Never contended for, unlike `release.json`.

    ``run_id`` is the CI run that produced this attempt (GitHub's
    ``GITHUB_RUN_ID``, or a local caller's own identifier); ``run_attempt``
    distinguishes a "re-run failed jobs" retry that reuses the same run id.
    Both are required to be non-empty: an empty component would let two
    unrelated attempts collide on `releases/{sha}/provenance/-1.json` (or
    worse, on the same key), silently discarding one attempt's record.
    """
    if not run_id:
        raise ValueError(
            "run_id must be non-empty: provenance is keyed per attempt, and an empty "
            "run_id would collide across every re-run that also omitted one."
        )
    if not run_attempt:
        raise ValueError("run_attempt must be non-empty, for the same reason as run_id.")
    return f"{release_prefix(sha)}/provenance/{run_id}-{run_attempt}.json"


@dataclass(frozen=True)
class ReleaseRecord:
    """What `release.json` carries: what a rollback needs to know **what**
    this release is — deterministic across every rebuild of the same commit.

    `lockfile_sha256` is here because the wheel does not pin its own
    transitive tree: two wheels built from one commit against two resolved
    dependency sets are two different artifacts, and only the lockfile hash
    says which one this is.

    **Nothing here can differ between two builds of the same commit**
    (alpha-engine-config-I9786) — that is what makes `assert_immutable_write`
    a correctness check rather than a false-alarm generator: a second
    `workflow_dispatch` for an unchanged commit produces these same bytes,
    and the write becomes a no-op instead of a `ReleaseImmutabilityError`.
    Per-run facts — when this build ran, which workflow run produced it,
    what the tests printed — belong to :class:`ReleaseProvenance`, one
    instance per attempt, never here.
    """

    schema_version: str
    sha: str
    lockfile_sha256: str
    wheel_sha256: str
    #: The wheel's own filename, added in release.v3 (alpha-engine-config-
    #: I9908). Required, no default: a record built without it is exactly
    #: the shape that made every prior wheel unpip-installable, so there is
    #: no safe value to default to. `parse_release_record` synthesizes it
    #: for a legacy v2 document being read back; nothing WRITES a v2 record
    #: any more.
    wheel_filename: str
    python_requires: str = ">=3.12,<3.13"
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Validated on CONSTRUCTION, not only inside `publish_release`
        # (alpha-engine-config-I9814): a document that fails the schema
        # cannot exist as a `ReleaseRecord` at all, so every writer —
        # `publish_release`, `crucible.deploy._publish`, or one that does not
        # exist yet — is refused the moment it builds one, with no second
        # call site to remember or forget.
        _validate_release_artifact("release.v3.json", asdict(self))

    def to_json(self) -> bytes:
        return json.dumps(asdict(self), indent=2, sort_keys=True).encode("utf-8")


#: `release.json`'s retired schema. Predates `wheel_sha256` integrity
#: checking; a document at this version carries no verified relationship to
#: any wheel bytes at all, so there is no correct way to synthesize a
#: `ReleaseRecord` from one. `parse_release_record` refuses it by name
#: (alpha-engine-config-I9908) rather than raising a bare `KeyError` three
#: frames into a dataclass construction.
_RETIRED_SCHEMA_VERSION_V1 = "release.v1"

#: The schema `release.json` carried before I9908 — no `wheel_filename`,
#: and every object actually published under it lives at the LEGACY,
#: unpip-installable key `crucible-{sha}-py3-none-any.whl`
#: (`deploy.yml`'s pre-fix rename; alpha-engine-config-I9908's console
#: evidence). `parse_release_record` synthesizes that exact legacy name for
#: a v2 document, never `wheel_filename_for(sha)` — a v2 release was never
#: published at the v3 path, and guessing one would be would send a caller
#: to a key nothing wrote.
_PREDECESSOR_SCHEMA_VERSION_V2 = "release.v2"


def parse_release_record(payload: dict[str, Any]) -> ReleaseRecord:
    """Parse a `release.json` payload into a :class:`ReleaseRecord`.

    Accepts `release.v2` (this schema's predecessor) and `release.v3` (this
    module's current :data:`RELEASE_SCHEMA_VERSION`); refuses `release.v1`
    by name. alpha-engine-config-I9908: every reader of a possibly-old
    `release.json` — `crucible.deploy._publish`'s idempotent-republish
    check, `crucible.track_c._verify_release_artifacts`'s smoke gate — goes
    through here rather than constructing `ReleaseRecord` directly, so a
    release published before this fix landed is still readable instead of
    raising a `TypeError` for a missing `wheel_filename` nothing wrote at
    the time.
    """
    version = payload.get("schema_version")
    if version == RELEASE_SCHEMA_VERSION:
        return ReleaseRecord(**payload)
    if version == _PREDECESSOR_SCHEMA_VERSION_V2:
        normalized = dict(payload)
        sha = normalized.get("sha", "")
        normalized["wheel_filename"] = f"crucible-{sha}-py3-none-any.whl"
        normalized["schema_version"] = RELEASE_SCHEMA_VERSION
        return ReleaseRecord(**normalized)
    if version == _RETIRED_SCHEMA_VERSION_V1:
        # Retired by the issue named in this module's RELEASE_SCHEMA_VERSION
        # docstring above: v1 predates wheel_sha256 integrity checking, so
        # there is no verified relationship between a v1 document and any
        # wheel bytes to read it back into.
        raise ValueError(
            "release.json is release.v1, retired: v1 predates wheel_sha256 integrity "
            "checking, so there is no verified relationship between a v1 document and "
            "any wheel bytes to read it back into. No v1 release is a valid smoke or "
            "install target."
        )
    raise ValueError(
        f"release.json schema_version {version!r} is not recognized; expected "
        f"{_PREDECESSOR_SCHEMA_VERSION_V2!r} or {RELEASE_SCHEMA_VERSION!r}."
    )


@dataclass(frozen=True)
class ReleaseProvenance:
    """One publish ATTEMPT for `sha`, at `provenance_key(sha, run_id, run_attempt)`.

    Exactly the three fields that used to live in `release.json` and made it
    unreproducible (I9786's Gotcha: `built_at` is repository metadata, not
    the build instant — carried here unexamined is still more honest than
    dropping it, since a wrong-but-named field is better than an absent one,
    and it is never used for anything but display). Never immutable-checked:
    a second attempt for an already-published sha is EXPECTED to differ here,
    and each attempt adds a record rather than contending for one slot.
    """

    schema_version: str
    sha: str
    run_id: str
    run_attempt: str
    built_at: str
    workflow_run_url: str
    test_summary: str

    def __post_init__(self) -> None:
        # Same reasoning as `ReleaseRecord.__post_init__`: the schema is
        # enforced by the type, not by whichever function happens to call
        # `_validate_release_artifact` on it afterward.
        _validate_release_artifact("release_provenance.v1.json", asdict(self))

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
    run_id: str = "0",
    run_attempt: str = "1",
    now: dt.datetime | None = None,
) -> ReleaseRecord:
    """Write the immutable half of a release. Does NOT touch the pointer.

    Separated from the flip on purpose: uploading the artifact is safe and
    repeatable, and moving the pointer is the act with consequences. A single
    function doing both would make "publish it but do not promote it yet"
    impossible to express, and the smoke gate sits precisely in between.

    "Repeatable" means **byte-identical**, not "overwrites whatever is
    there": re-publishing a sha whose `release.json`/wheel prefix already
    holds different IDENTITY bytes raises :class:`ReleaseImmutabilityError`.
    A re-publish whose identity matches — including one with a different
    `run_id`, `built_at` or `test_summary`, which the identity record does
    not even carry (alpha-engine-config-I9786) — is a clean no-op: neither
    key is rewritten. See :class:`ReleaseRecord` for why nothing in it can
    differ between two builds of the same commit, and :class:`ReleaseImmutabilityError`
    for why a genuinely differing rebuild still raises.

    A :class:`ReleaseProvenance` record for THIS attempt is always written,
    at `provenance_key(sha, run_id, run_attempt)`, whether or not the
    identity keys needed writing — so a re-run that changed nothing is still
    a reconstructible event, not a silent no-op with no trace.

    When ``store`` is an :class:`~crucible.store.S3Store`, each identity key
    actually written here is locked under S3 Object Lock GOVERNANCE mode for
    :data:`RELEASE_OBJECT_LOCK_RETENTION`, on the SAME `put_bytes` call that
    writes it (alpha-engine-config-I9787) — see
    :func:`release_object_lock_params`. `assert_immutable_write` refuses a
    differing overwrite at THIS writer; the lock is the defense against a
    writer that skips this function entirely (a hand-rolled `aws s3 cp`, a
    second `workflow_dispatch` running code that predates this fix).
    """
    _assert_sha(sha)
    if not wheel:
        raise ValueError(f"refusing to publish an empty wheel for {sha}")
    stamp = (now or dt.datetime.now(dt.UTC)).astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    record = ReleaseRecord(
        schema_version=RELEASE_SCHEMA_VERSION,
        sha=sha,
        lockfile_sha256=sha256_hex(lockfile),
        wheel_sha256=sha256_hex(wheel),
        wheel_filename=wheel_filename_for(sha),
    )
    provenance = ReleaseProvenance(
        schema_version=RELEASE_PROVENANCE_SCHEMA_VERSION,
        sha=sha,
        run_id=run_id,
        run_attempt=run_attempt,
        built_at=stamp,
        workflow_run_url=workflow_run_url,
        test_summary=test_summary,
    )
    # Contract-tested at birth (M0 discipline): both dataclasses validate
    # themselves against their own schema in `__post_init__`, so the
    # `ReleaseRecord(...)` / `ReleaseProvenance(...)` calls above already
    # raised if either document were non-conformant — there is nothing left
    # to check here, on purpose (alpha-engine-config-I9814): a second,
    # separate validation call at this call site is exactly the "two
    # readings of the same contract" shape that let the CLI's own
    # construction skip validation entirely.
    #
    # Immutability is enforced BEFORE the first of the two identity writes,
    # so a refusal cannot leave a prefix half-overwritten: a wheel from one
    # build beside a release.json from another is worse than either.
    payloads = ((wheel_key(sha), wheel), (release_json_key(sha), record.to_json()))
    needed = [
        (key, payload) for key, payload in payloads if assert_immutable_write(store, key, payload)
    ]
    lock_mode, retain_until = release_object_lock_params(store, now=now)
    for key, payload in needed:
        store.put_bytes(
            key, payload, object_lock_mode=lock_mode, object_lock_retain_until=retain_until
        )
    # Unconditional and unlocked: it is per-attempt by construction (the key
    # already carries run_id/run_attempt) so it never contends with itself,
    # and it is the durable trace that this attempt happened even when the
    # identity keys needed no write at all.
    store.put_bytes(provenance_key(sha, run_id, run_attempt), provenance.to_json())
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
    payload = load_store_document(store, key)
    return payload["sha"], version


def current_release(store: Store) -> str | None:
    return read_pointer(store)[0]


def published_wheel_key(store: Store, sha: str) -> str:
    """The key ``sha``'s wheel was ACTUALLY published at.

    Read out of that release's own `release.json`
    (:attr:`ReleaseRecord.wheel_filename`, synthesized for a `release.v2`
    document by :func:`parse_release_record`) rather than derived from the sha
    — which is the same lookup `crucible.track_c._verify_release_artifacts`
    already performs, for the same reason.

    alpha-engine-config-I9917 item 1: :func:`wheel_key` derives the CURRENT
    (v3, PEP 440) name unconditionally, and every release published before
    alpha-engine-config-I9908's fix is stored under the legacy
    `crucible-{sha40}-py3-none-any.whl` name. Deriving the name therefore
    refused every prior release — the exact set a rollback reaches for — with
    a message naming a key nothing ever wrote, which reads as "the object was
    deleted" rather than "this release predates the naming fix".

    Raises :class:`StaleReleasePointerError` for a sha with no `release.json`,
    naming THAT absence: a sha with no release record was never published at
    all, and the two conditions want different operator responses.
    """
    _assert_sha(sha)
    meta_key = release_json_key(sha)
    if not store.exists(meta_key):
        raise StaleReleasePointerError(
            f"{sha} was never published: no release record at {meta_key}. A release's "
            "wheel filename is read from its own release.json, so a sha without one "
            "has no addressable wheel — this is not the same condition as a published "
            "release whose wheel was deleted, which names the wheel key instead."
        )
    record = parse_release_record(load_store_document(store, meta_key))
    return wheel_key_for(sha, record.wheel_filename)


def resolve_release(store: Store, key: str = POINTER_KEY) -> str:
    """The sha a job should install, verified to actually be there.

    Raises :class:`StaleReleasePointerError` when the pointer names a sha
    whose wheel is absent. A job that installed nothing and carried on would
    run whatever was already on the box — which is the silent version of
    every deploy bug, and the reason this is checked at read time rather than
    trusted from the pointer.

    The wheel key comes from :func:`published_wheel_key`, so a pointer at a
    pre-alpha-engine-config-I9908 release resolves instead of being refused
    against a v3-shaped path that release never occupied.
    """
    sha, _ = read_pointer(store, key)
    if sha is None:
        raise StaleReleasePointerError(
            f"{key} is unset: no release has ever been promoted. A job cannot choose "
            "a release for itself."
        )
    wheel = published_wheel_key(store, sha)
    if not store.exists(wheel):
        raise StaleReleasePointerError(
            f"{key} names {sha}, whose wheel is not at {wheel}. Restore the "
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

    The wheel key comes from :func:`published_wheel_key`, so a rollback to a
    pre-alpha-engine-config-I9908 release is accepted rather than refused
    against a v3-shaped path that release never occupied. This is the call
    site the bug actually bit: `release.pin <prior-sha>` IS the rollback
    command, and prior shas are precisely the ones with legacy wheel names.
    """
    _assert_sha(sha)
    if target not in PIN_TARGETS:
        raise ValueError(f"pin target {target!r} not in {PIN_TARGETS}")
    wheel = published_wheel_key(store, sha)
    if not store.exists(wheel):
        raise StaleReleasePointerError(
            f"refusing to pin {target} to {sha}: no wheel at {wheel}. A "
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
    from crucible.manifest import validate  # noqa: PLC0415 - cycle

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
