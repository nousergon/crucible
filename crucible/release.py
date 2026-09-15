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
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ValidationError

from crucible.calendar import is_trading_day as _nyse_is_trading_day
from crucible.documents import load_document_bytes, load_store_document, read_manifests_under
from crucible.keys import POINTER_KEY, TRADER_PIN_KEY, manifest_key, runs_prefix
from crucible.models import (
    ReleasePointerDocument,
    ReleaseProvenanceDocument,
    ReleaseRecordDocument,
    ReleaseRecordV3Document,
    TraderReleasePinDocument,
)
from crucible.store import ETAG_ABSENT, PointerConflictError, S3Store, Store, sha256_hex

#: `alpha-engine-config-I10045` row 5: `_validate_release_artifact` used to
#: hand-roll a `jsonschema.Draft202012Validator` per schema FILE, loaded off
#: disk on every distinct `schema_filename`. Both schemas are now GENERATED
#: from `crucible.models.ReleaseRecordDocument` /
#: `crucible.models.ReleaseProvenanceDocument` — this maps the same
#: filenames straight to the model that generates them, so there is one
#: source of truth for both "does this payload conform" and "what does the
#: published schema say" (`tests/test_typed_boundary_release.py`'s byte-identity
#: test is what keeps the committed `.json` files honest about it).
_RELEASE_ARTIFACT_MODELS: dict[str, type[BaseModel]] = {
    "release.v4.json": ReleaseRecordDocument,
    "release.v3.json": ReleaseRecordV3Document,
    "release_provenance.v1.json": ReleaseProvenanceDocument,
}


def _validate_release_artifact(schema_filename: str, payload: dict[str, Any]) -> None:
    """Raise with every error, never just the first, against one of this
    module's own artifact schemas. A writer that could emit a non-conformant
    document would defeat the schema entirely — validated on the way OUT,
    not only wherever something later reads it back.

    Public signature and message shape (`"{schema_filename}: document does
    not conform:\\n..."`) are UNCHANGED from before this PR: both
    `tests/test_release.py::test_publish_validates_the_identity_record_against_its_own_schema`
    and its provenance counterpart call this function directly and match on
    "does not conform", as does `ReleaseRecord.__post_init__` /
    `ReleaseProvenance.__post_init__`'s own construction-time guard.
    """
    model = _RELEASE_ARTIFACT_MODELS.get(schema_filename)
    if model is None:
        raise FileNotFoundError(
            f"{schema_filename} is not one of this module's generated artifact schemas "
            f"({sorted(_RELEASE_ARTIFACT_MODELS)}). A missing entry means a broken build, "
            "not a degraded write."
        )
    try:
        model.model_validate(payload)
    except ValidationError as exc:
        detail = "\n".join(
            f"  - {'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}"
            for e in exc.errors()
        )
        raise ValueError(f"{schema_filename}: document does not conform:\n{detail}") from exc


__all__ = [
    "POINTER_KEY",
    "PointerConflictError",
    "RELEASE_PROVENANCE_SCHEMA_VERSION",
    "ReleaseImmutabilityError",
    "assert_immutable_write",
    "flip_on_smoke",
    "RELEASE_SCHEMA_VERSION",
    "PublishedWheel",
    "ReleaseProvenance",
    "ReleaseRecord",
    "ReleaseRecordMismatchError",
    "ReleaseHasNoWheelhouseError",
    "require_wheelhouse",
    "wheelhouse_object_key",
    "StaleReleasePointerError",
    "TRADER_PIN_BLACKOUT_END_ET",
    "TRADER_PIN_BLACKOUT_START_ET",
    "TRADER_PIN_KEY",
    "TRADER_SMOKE_JOB",
    "TraderPinRefusedError",
    "TraderSmokeEvidence",
    "passing_trader_smoke",
    "pin_trader",
    "select_trader_smoke",
    "trader_pin_window_refusal",
    "assert_sha",
    "current_release",
    "parse_release_pointer",
    "parse_trader_release_pin",
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
    "resolve_published_wheel",
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
#:
#: Bumped to v4 by alpha-engine-config-I10812: adds `wheelhouse`, every
#: dependency wheel the release installs (built for the box platform) and the
#: hash-locked lock pinning them, published under `releases/{sha}/wheelhouse/`.
#: A box installs from it with no index access; a v3 release, which has none,
#: is refused at boot rather than resolved from PyPI.
RELEASE_SCHEMA_VERSION = "release.v4"

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

#: `POINTER_KEY` (the single mutable object in the whole release layout —
#: everything else is immutable and content-addressed by the sha in its own
#: prefix) and `TRADER_PIN_KEY` (the trader's separate pin: two pointers,
#: never one, since a trader that followed `current` would be promoted by
#: every merge — §4.11) are defined in `crucible.keys` with every other
#: store key shape and imported above; both stay in this module's `__all__`
#: for their existing readers (alpha-engine-config-I9899, round 2).

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

    This `(None, None)` branch is also WHY `LocalStore.put_bytes`'s own
    ``NotImplementedError`` guard never fires from the release path
    (alpha-engine-config-I9817): every caller here resolves its lock params
    through this function, which deliberately never hands a LocalStore a
    mode to refuse. That guard remains real for a caller that bypasses this
    function; it is not what makes a LocalStore-backed release publish
    unable to claim retention — the absence of a mode passed at all is.

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


def assert_sha(sha: str) -> str:
    """Refuse anything but a 40-character lowercase git sha; return it.

    Public so a caller that reads a sha out of a document it does not own —
    `track_c`'s smoke reading `releases/current` — can validate it BEFORE
    entering a block that swallows record-shaped failures. Measured on the
    review of alpha-engine-config-I9932: with the check only inside
    :func:`resolve_published_wheel`, a corrupt pointer (`"sha":
    "NOT-A-VALID-SHA"`) was swallowed with the record conditions and the
    smoke reported `ok` over it.
    """
    return _assert_sha(sha)


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


def wheelhouse_object_key(sha: str, filename: str) -> str:
    """The key one wheelhouse object (a dependency wheel, or the lock) lives at.

    `releases/{sha}/wheelhouse/{filename}` (alpha-engine-config-I10812). The
    directory name is `crucible.wheelhouse.WHEELHOUSE_DIRNAME`, which the box
    bootstrap is lockstep-tested against; a filename carrying a separator is
    refused, because the box interpolates it into a local path.
    """
    from crucible.wheelhouse import WHEELHOUSE_DIRNAME  # noqa: PLC0415 - leaf module

    if not filename or "/" in filename:
        raise ValueError(f"{filename!r} is not a wheelhouse object name")
    return f"{release_prefix(sha)}/{WHEELHOUSE_DIRNAME}/{filename}"


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
    #: release.v4 (alpha-engine-config-I10812): the hash-locked dependency set,
    #: as `crucible.wheelhouse.build_manifest` produced it. `None` only on a
    #: release.v3 record read back (or written by the pre-wheelhouse helper
    #: :func:`publish_release`) — never on a record `crucible.deploy` publishes.
    wheelhouse: dict[str, Any] | None = None

    def _payload(self) -> dict[str, Any]:
        """The document, with `wheelhouse` omitted on a v3 record: a v3 record
        must serialise to the exact bytes it was published as, or re-reading
        and re-publishing one would trip `assert_immutable_write`."""
        payload = asdict(self)
        if payload["wheelhouse"] is None:
            del payload["wheelhouse"]
        return payload

    def __post_init__(self) -> None:
        # Validated on CONSTRUCTION, not only inside `publish_release`
        # (alpha-engine-config-I9814): a document that fails the schema
        # cannot exist as a `ReleaseRecord` at all, so every writer —
        # `publish_release`, `crucible.deploy._publish`, or one that does not
        # exist yet — is refused the moment it builds one, with no second
        # call site to remember or forget.
        # The schema FILE is chosen by the record's own version, never by a
        # default: a v4 record missing its wheelhouse must be refused by the v4
        # schema, not quietly accepted by v3's.
        # An UNKNOWN version is validated against the current schema, which
        # refuses the version AND names every other violation in one message —
        # a refusal that stopped at the version would hide the rest.
        schema_file = _RECORD_SCHEMA_FILES.get(
            self.schema_version, _RECORD_SCHEMA_FILES[RELEASE_SCHEMA_VERSION]
        )
        _validate_release_artifact(schema_file, self._payload())

    def to_json(self) -> bytes:
        return json.dumps(self._payload(), indent=2, sort_keys=True).encode("utf-8")

    @property
    def wheelhouse_keys(self) -> list[tuple[str, str]]:
        """`(key, sha256)` for every wheelhouse object this record names — each
        dependency wheel, then the lock. Empty for a v3 record; callers that
        need one go through :func:`require_wheelhouse` first."""
        if self.wheelhouse is None:
            return []
        from crucible.wheelhouse import LOCK_FILENAME  # noqa: PLC0415 - leaf module

        pairs = [
            (wheelhouse_object_key(self.sha, w["filename"]), w["sha256"])
            for w in self.wheelhouse["wheels"]
        ]
        pairs.append(
            (wheelhouse_object_key(self.sha, LOCK_FILENAME), self.wheelhouse["lock_sha256"])
        )
        return pairs

    @property
    def wheel_key(self) -> str:
        """The store key THIS record's wheel is published at.

        `wheel_key_for(self.sha, self.wheel_filename)` — stated once, on the
        record, so no caller pairs a sha with a `wheel_filename` read from
        some OTHER record. Before alpha-engine-config-I9932 that pairing was
        restated at three call sites (`release.published_wheel_key`,
        `track_c._verify_release_artifacts`, `track_c`'s pointed-release
        branch) and one of them omitted the `record.sha == sha` guard the
        others carried. A property on the record cannot be handed the wrong
        sha: it only knows its own.
        """
        return wheel_key_for(self.sha, self.wheel_filename)


class ReleaseHasNoWheelhouseError(ValueError):
    """A release carries no hash-locked wheelhouse, so nothing can install it
    without an index — and alpha-engine-config-I10812 removed index access from
    the box. Raised by name so the smoke and the flip refuse it, never degrade."""


def require_wheelhouse(record: ReleaseRecord) -> dict[str, Any]:
    """Return ``record``'s wheelhouse, or raise naming the fix.

    The same refusal the box bootstrap prints, in the same words, so an
    operator meets one message whichever surface reached it first.
    """
    if record.wheelhouse is None:
        raise ReleaseHasNoWheelhouseError(
            f"release {record.sha} ({record.schema_version}) publishes no wheelhouse. Boxes "
            "install offline from a hash-locked wheelhouse only and never fall back to "
            "PyPI. Fix: `crucible release.pin` a release that carries one (any release "
            "deployed since the wheelhouse landed), or merge to main so deploy.yml publishes a "
            "new one. Re-deploying THIS sha cannot add one: its release.json is immutable."
        )
    return record.wheelhouse


#: schema_version -> the generated schema file a `ReleaseRecord` of that version
#: validates against. v2 is never constructed directly: `parse_release_record`
#: normalises it to v3 first.
_RECORD_SCHEMA_FILES = {"release.v3": "release.v3.json", "release.v4": "release.v4.json"}


@dataclass(frozen=True)
class PublishedWheel:
    """What :func:`resolve_published_wheel` hands back: the record it read,
    the bytes it read it from (for a caller's lineage), and the wheel key the
    record itself names."""

    record: ReleaseRecord
    record_key: str
    record_bytes: bytes
    wheel_key: str


class ReleaseRecordMismatchError(ValueError):
    """A `release.json` under one sha's prefix describes another sha.

    A record under the wrong prefix yields a wrong wheel key and a misleading
    refusal ("no wheel at ...") for an object that was never supposed to be
    there. Raised by name so a caller gating a promotion (`track_c`'s smoke)
    fails closed on it and a caller merely reporting (`track_c`'s
    pointed-release branch) can record it as the pointed build being broken.
    `ValueError` for the callers that already catch that.
    """


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

#: The schema `release.json` carried before alpha-engine-config-I10812 — no
#: `wheelhouse`. Still READ (rollback addressing, the smoke's pointed-release
#: branch), never published by `crucible.deploy`, and refused by every
#: installer: see :func:`require_wheelhouse`.
_PREDECESSOR_SCHEMA_VERSION_V3 = "release.v3"


def parse_release_record(payload: dict[str, Any]) -> ReleaseRecord:
    """Parse a `release.json` payload into a :class:`ReleaseRecord`.

    Accepts `release.v2`, `release.v3` (both pre-wheelhouse) and `release.v4`
    (this module's current :data:`RELEASE_SCHEMA_VERSION`); refuses `release.v1`
    by name. A v2 or v3 record parses with `wheelhouse=None`: readable, and
    refused by anything that would install it (:func:`require_wheelhouse`).
    alpha-engine-config-I9908: every reader of a possibly-old
    `release.json` — `crucible.deploy._publish`'s idempotent-republish
    check, `crucible.track_c._verify_release_artifacts`'s smoke gate — goes
    through here rather than constructing `ReleaseRecord` directly, so a
    release published before this fix landed is still readable instead of
    raising a `TypeError` for a missing `wheel_filename` nothing wrote at
    the time.
    """
    version = payload.get("schema_version")
    if version in (RELEASE_SCHEMA_VERSION, _PREDECESSOR_SCHEMA_VERSION_V3):
        return ReleaseRecord(**payload)
    if version == _PREDECESSOR_SCHEMA_VERSION_V2:
        normalized = dict(payload)
        sha = normalized.get("sha", "")
        normalized["wheel_filename"] = f"crucible-{sha}-py3-none-any.whl"
        normalized["schema_version"] = _PREDECESSOR_SCHEMA_VERSION_V3
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
        f"{_PREDECESSOR_SCHEMA_VERSION_V2!r}, {_PREDECESSOR_SCHEMA_VERSION_V3!r} or "
        f"{RELEASE_SCHEMA_VERSION!r}."
    )


@dataclass(frozen=True)
class ReleaseProvenance:
    """One publish ATTEMPT for `sha`, at `provenance_key(sha, run_id, run_attempt)`.

    Exactly the three fields that used to live in `release.json` and made it
    unreproducible (I9786's Gotcha: `built_at` is repository metadata, not
    the build instant — carried here unexamined is still more honest than
    dropping it, since a wrong-but-named field is better than an absent one,
    and it is never used for anything but display). Never immutable-checked
    ACROSS attempts: a second, distinct `run_id`/`run_attempt` for an
    already-published sha is EXPECTED to differ here, and each attempt adds a
    record rather than contending for one slot. The write at THIS attempt's
    own key — `provenance_key(sha, run_id, run_attempt)` — IS immutable-
    checked (alpha-engine-config-I9817): two invocations naming the same
    attempt must describe the same attempt, or the second is a defect, not a
    re-run.
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
    """Write the immutable half of a PRE-WHEELHOUSE (release.v3) release.
    Does NOT touch the pointer.

    alpha-engine-config-I10812: this helper has no production caller — the
    publisher is `crucible.deploy._publish`, which requires a release.v4
    record and uploads its wheelhouse. It is kept writing v3 because it is
    what the pointer, rollback and history tests lay releases down with, and a
    v3 release is exactly what those need to exercise: published, pinnable,
    and refused by every installer.

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
        schema_version=_PREDECESSOR_SCHEMA_VERSION_V3,
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
    # Per-attempt by construction (the key already carries run_id/run_attempt)
    # so it never contends with a DIFFERENT attempt, and it is the durable
    # trace that this attempt happened even when the identity keys needed no
    # write at all. Still unlocked (unlike the identity writes above): Object
    # Lock retention is a claim about the wheel/release.json bytes an
    # installer trusts, and the provenance record was never that.
    # Immutability-checked (I9817) though: two invocations naming the SAME
    # run_id and run_attempt describing different bytes means the record of
    # "what happened during this attempt" has been silently replaced, which
    # is exactly the durable-trace guarantee this record exists to provide.
    provenance_bytes = provenance.to_json()
    prov_key = provenance_key(sha, run_id, run_attempt)
    if assert_immutable_write(store, prov_key, provenance_bytes):
        store.put_bytes(prov_key, provenance_bytes)
    return record


def parse_release_pointer(source: str, payload: dict[str, Any]) -> ReleasePointerDocument:
    """Validate ``payload`` (a document already read from ``source``) against
    `crucible.models.ReleasePointerDocument`, raising with every field named.

    `alpha-engine-config-I9847` (wave 2): factored out of :func:`read_pointer`
    so `crucible.track_c`'s own STRICT read of this same document (the
    pointed-release branch of its smoke) raises the identical message shape
    rather than growing a second hand-rolled wrapper — the same "one source
    of truth" argument `_validate_release_artifact` makes for the release
    record and provenance. Both call sites already raise on a document
    `load_store_document`/`load_document_bytes` could not even parse; this
    closes the gap one level up, where the document parses but does not
    conform (a bad `sha`, a missing `target`, an extra key).
    """
    try:
        return ReleasePointerDocument.model_validate(payload)
    except ValidationError as exc:
        detail = "\n".join(
            f"  - {'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}"
            for e in exc.errors()
        )
        raise ValueError(
            f"{source}: document does not conform to a release pointer:\n{detail}"
        ) from exc


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
    # `alpha-engine-config-I9847` (wave 2): validated through
    # `ReleasePointerDocument` rather than indexed straight off the raw
    # dict — a pointer written without `sha` used to reach this function's
    # caller as a bare `KeyError` naming neither the document nor the field.
    # The trader's pin is validated through its own model
    # (`TraderReleasePinDocument`, alpha-engine-config-I10649), which carries
    # the smoke evidence the pin was gated on.
    pointer = _parse_pointer_for(key, payload)
    return pointer.sha, version


def parse_trader_release_pin(source: str, payload: dict[str, Any]) -> TraderReleasePinDocument:
    """Validate ``payload`` against `crucible.models.TraderReleasePinDocument`,
    raising with every field named — the trader-pin sibling of
    :func:`parse_release_pointer`."""
    try:
        return TraderReleasePinDocument.model_validate(payload)
    except ValidationError as exc:
        detail = "\n".join(
            f"  - {'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}"
            for e in exc.errors()
        )
        raise ValueError(
            f"{source}: document does not conform to a trader release pin:\n{detail}"
        ) from exc


def _parse_pointer_for(
    key: str, payload: dict[str, Any]
) -> ReleasePointerDocument | TraderReleasePinDocument:
    """The one key -> model dispatch: `trader/release_pin` is a trader pin, every
    other pointer key is a harness release pointer."""
    if key == TRADER_PIN_KEY:
        return parse_trader_release_pin(key, payload)
    return parse_release_pointer(key, payload)


def current_release(store: Store) -> str | None:
    return read_pointer(store)[0]


def resolve_published_wheel(store: Store, sha: str) -> PublishedWheel:
    """Read ``sha``'s own `release.json`, check it describes ``sha``, and
    return the wheel key the record names.

    THE one read → parse → `wheel_key_for` sequence (alpha-engine-config-
    I9932). It used to exist in three places — here (as
    `published_wheel_key`), `crucible.track_c._verify_release_artifacts`, and
    `track_c`'s pointed-release branch — and this one omitted the
    `record.sha == sha` guard the other two carried, so a `release.json`
    copied under the wrong prefix resolved to a wheel key for a build that
    was never published there, and `pin` / `resolve_release` then refused
    with "no wheel at <wrong key>" — a message about the wrong object.

    The wheel filename is read out of the record
    (:attr:`ReleaseRecord.wheel_filename`, synthesized for a `release.v2`
    document by :func:`parse_release_record`) rather than derived from the
    sha. alpha-engine-config-I9917 item 1: :func:`wheel_key` derives the
    CURRENT (v3, PEP 440) name unconditionally, and every release published
    before alpha-engine-config-I9908's fix is stored under the legacy
    `crucible-{sha40}-py3-none-any.whl` name, so deriving refused every prior
    release — the exact set a rollback reaches for.

    Raises :class:`StaleReleasePointerError` for a sha with no `release.json`
    (never published — a different operator response from "published, wheel
    deleted", which names the wheel key), and
    :class:`ReleaseRecordMismatchError` when the record under ``sha``'s
    prefix describes another sha. Does NOT check the wheel object exists:
    callers do, each with the message their situation needs.
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
    # Bytes held once for lineage (track_c records them); parse through the
    # documents reader so the store-consumer contract holds (I9929/I9931).
    record_bytes = store.get_bytes(meta_key)
    record = parse_release_record(load_document_bytes(meta_key, record_bytes))
    if record.sha != sha:
        raise ReleaseRecordMismatchError(
            f"{meta_key} describes {record.sha}, not {sha}. A release record under another "
            "build's prefix names a wheel that was never published there; trusting it "
            "would resolve, pin or gate on the wrong artifact."
        )
    return PublishedWheel(
        record=record, record_key=meta_key, record_bytes=record_bytes, wheel_key=record.wheel_key
    )


def published_wheel_key(store: Store, sha: str) -> str:
    """The key ``sha``'s wheel was ACTUALLY published at — see
    :func:`resolve_published_wheel`, of which this is the key-only view."""
    return resolve_published_wheel(store, sha).wheel_key


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
    trader_smoke: TraderSmokeEvidence | None = None,
) -> str:
    """Point ``target`` at ``sha`` by compare-and-swap. Returns the new version.

    ``target="trader"`` is refused here unless ``trader_smoke`` is passing
    evidence FOR ``sha`` (alpha-engine-config-I10649). Callers go through
    :func:`pin_trader`, which reads that evidence from the store and enforces
    the off-market-hours window; the check is repeated in this primitive so
    no second caller can move the trader's pin around the gate.

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
    if target == "trader":
        if trader_smoke is None:
            raise TraderPinRefusedError(
                f"refusing to pin the trader to {sha}: no trader smoke evidence was "
                "presented. The trader's pin moves only through `pin_trader`, which reads a "
                f"passing `{TRADER_SMOKE_JOB}` manifest for this sha from the store."
            )
        if trader_smoke.release_sha != sha or trader_smoke.status != "ok":
            raise TraderPinRefusedError(
                f"refusing to pin the trader to {sha}: the smoke evidence presented is "
                f"{trader_smoke.status!r} for release {trader_smoke.release_sha!r}. "
                "Promoting on another build's smoke, or on a smoke that did not pass, is "
                "the gate failing open."
            )
    elif trader_smoke is not None:
        raise ValueError(
            "trader smoke evidence was passed for target 'current'. The harness pointer is "
            "gated by the harness smoke (`flip_on_smoke`), never by the trader's."
        )
    wheel = published_wheel_key(store, sha)
    if not store.exists(wheel):
        raise StaleReleasePointerError(
            f"refusing to pin {target} to {sha}: no wheel at {wheel}. A "
            "pointer to an artifact that is not there is a stale pointer the moment "
            "it is written."
        )
    key = POINTER_KEY if target == "current" else TRADER_PIN_KEY
    version = expect if expect is not None else store.etag(key)
    document: dict[str, Any] = {
        "sha": sha,
        "target": target,
        "pinned_at": (now or dt.datetime.now(dt.UTC))
        .astimezone(dt.UTC)
        .strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if trader_smoke is not None:
        # alpha-engine-config-I10649 deliverable 3: the trader pin names the
        # smoke it was gated on, so the pin alone walks back to its evidence.
        document |= {
            "smoke_run_id": trader_smoke.run_id,
            "smoke_status": trader_smoke.status,
            "smoke_manifest_key": trader_smoke.manifest_key,
        }
    # Validated BEFORE the swap, through the same model its reader uses: a pin
    # its own reader would refuse must never reach the store.
    _parse_pointer_for(key, document)
    payload = json.dumps(document, indent=2, sort_keys=True).encode("utf-8")
    return store.compare_and_swap(key, version, payload)


# ── the trader's pin: smoke-gated, off-market-hours (alpha-engine-config-I10649)

#: The trader's own smoke job, written by `nousergon/crucible-trader`
#: (`crucible_trader.paper_smoke`): install the pinned wheel, connect to the
#: IB Gateway PAPER session, read the book, place no order. The harness's
#: `smoke` job is a different thing — an end-to-end harness run — and reading
#: its status here would promote a trader build nothing connected to a broker.
TRADER_SMOKE_JOB = "trader.smoke"

#: The window in which a trader pin is refused on an NYSE trading day, in
#: America/New_York wall time: from the trader's pre-open work to after the
#: close has settled (plan §4.11: "executed off-market-hours"). Half-open,
#: `[05:00, 16:30)`. Whether a date is a trading day comes from
#: `krepis.trading_calendar` — the one NYSE calendar in the fleet — and is
#: never re-derived here; early-close sessions keep the full window, which is
#: the conservative direction for a refusal.
TRADER_PIN_BLACKOUT_START_ET = dt.time(5, 0)
TRADER_PIN_BLACKOUT_END_ET = dt.time(16, 30)
_ET = ZoneInfo("America/New_York")


class TraderPinRefusedError(RuntimeError):
    """The trader's pin was not moved, for a named reason.

    A distinct type because each reason is a distinct operator action: run the
    trader smoke for this sha, wait for the market to close, or look at why the
    smoke failed. None of them is "retry the pin".
    """


@dataclass(frozen=True)
class TraderSmokeEvidence:
    """The one `trader.smoke` manifest a trader pin is gated on."""

    manifest_key: str
    run_id: str
    release_sha: str
    status: str
    trading_day: str
    finished: str


def trader_pin_window_refusal(now: dt.datetime) -> str | None:
    """The refusal reason when ``now`` is inside the trader-pin blackout, else None.

    ``now`` must be timezone-aware: a naive instant has no answer to "is the
    market open", and guessing UTC or local is how a pin lands at 09:31 ET.
    """
    if now.tzinfo is None:
        raise ValueError(
            f"{now!r} is naive. The trader-pin window is an America/New_York wall-clock "
            "window and a naive instant cannot be placed in it."
        )
    local = now.astimezone(_ET)
    if not _nyse_is_trading_day(local.date()):
        return None
    if TRADER_PIN_BLACKOUT_START_ET <= local.time() < TRADER_PIN_BLACKOUT_END_ET:
        return (
            f"market_hours: {local.strftime('%Y-%m-%d %H:%M')} ET is inside the trader-pin "
            f"blackout [{TRADER_PIN_BLACKOUT_START_ET:%H:%M}, "
            f"{TRADER_PIN_BLACKOUT_END_ET:%H:%M}) ET on an NYSE trading day. A trader pin "
            "is an off-market-hours action (plan §4.11); an in-session incident is the kill "
            "switch's, and the trader installs its wheel at session start, so a mid-session "
            "pin would change nothing the running trader executes."
        )
    return None


def select_trader_smoke(
    documents: tuple[tuple[str, dict[str, Any]], ...] | list[tuple[str, dict[str, Any]]],
    sha: str,
) -> tuple[TraderSmokeEvidence | None, list[str]]:
    """The newest passing trader smoke for ``sha``, plus a line per non-passing one.

    Pure: ``documents`` are ``(key, manifest)`` pairs already validated by the
    caller. A manifest for another sha is not evidence; a failed one for this
    sha is returned as a reason line so a refusal can say WHY there is no pass
    (a broker session that expired is a human action, a defect is not).
    """
    _assert_sha(sha)
    passing: list[tuple[str, dict[str, Any]]] = []
    failures: list[str] = []
    for key, document in documents:
        if document.get("job") != TRADER_SMOKE_JOB or document.get("release_sha") != sha:
            continue
        if document.get("status") == "ok":
            passing.append((key, document))
        else:
            failures.append(
                f"{key} ({document.get('run_id')}): {document.get('status')} — "
                f"{document.get('reason')}"
            )
    if not passing:
        return None, failures
    key, document = max(passing, key=lambda pair: (pair[1]["finished"], pair[0]))
    evidence = TraderSmokeEvidence(
        manifest_key=key,
        run_id=document["run_id"],
        release_sha=sha,
        status="ok",
        trading_day=document["trading_day"],
        finished=document["finished"],
    )
    return evidence, failures


def passing_trader_smoke(store: Store, sha: str) -> TraderSmokeEvidence:
    """Read every `trader.smoke` manifest and return the newest passing one for ``sha``.

    Raises :class:`TraderPinRefusedError` when there is none, naming every
    failed smoke for this sha. Every manifest under the prefix is validated
    against the run-manifest schema before it counts: a document that does not
    conform cannot certify a pin, and an unreadable key or an unlistable prefix
    is a refusal — "we could not read the evidence" never reads as "there is
    evidence".
    """
    from crucible.manifest import validate  # noqa: PLC0415 - cycle, as write_deploy_manifest

    _assert_sha(sha)
    prefix = runs_prefix(TRADER_SMOKE_JOB)
    read = read_manifests_under(store, prefix)
    if read.listing_problem is not None:
        raise TraderPinRefusedError(f"refusing to pin the trader to {sha}: {read.listing_problem}")
    if read.faults:
        raise TraderPinRefusedError(
            f"refusing to pin the trader to {sha}: {len(read.faults)} trader smoke manifest(s) "
            f"under {prefix} could not be read: {sorted(read.faults.items())}"
        )
    for _key, document in read.documents:
        validate(document)
    evidence, failures = select_trader_smoke(read.documents, sha)
    if evidence is None:
        detail = (
            f" Failed smokes for this sha: {' | '.join(failures)}"
            if failures
            else " No trader smoke has ever run for this sha."
        )
        raise TraderPinRefusedError(
            f"no_passing_smoke: refusing to pin the trader to {sha}: no `{TRADER_SMOKE_JOB}` "
            f"manifest under {prefix} reads status ok for this release.{detail} Run the "
            "trader's paper smoke against this sha first."
        )
    return evidence


def pin_trader(
    store: Store,
    sha: str,
    *,
    now: dt.datetime,
    expect: str | None = None,
) -> tuple[str, TraderSmokeEvidence]:
    """Move `trader/release_pin` to ``sha`` iff it is off-market-hours and the
    trader's own smoke passed for ``sha``. Returns ``(version, evidence)``.

    The same rule serves a forward promotion and a rollback: a rollback to a
    sha whose smoke already passed is allowed on that record, with no new
    smoke — the smoke gates a build, not a direction (I10649 gotcha 2).
    """
    refusal = trader_pin_window_refusal(now)
    if refusal is not None:
        raise TraderPinRefusedError(f"refusing to pin the trader to {sha}: {refusal}")
    evidence = passing_trader_smoke(store, sha)
    version = pin(store, sha, target="trader", expect=expect, now=now, trader_smoke=evidence)
    return version, evidence


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
