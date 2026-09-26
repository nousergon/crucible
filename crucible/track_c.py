"""Track C's job handlers: release, smoke, alerting, drift, console.

Normative source: plan §4.6, §4.7, §4.9, §4.11; alpha-engine-config-I9757.

They live here rather than in `cli.py` so three tracks can land handlers in
parallel without editing one another's lines — `cli.py` gains a dispatch entry
per job and nothing else.

**Every one of them runs through `crucible.runner.run_job`.** That is where
the manifest guarantee lives, and a job invoked around it would produce no
telemetry on the path where telemetry matters. The smoke job in particular is
a real run whose manifest is what gates the release flip: if it wrote no
manifest there would be nothing to gate on, and the gate would be a return
code from a process nobody kept.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from typing import Any
from zoneinfo import ZoneInfo

from crucible import alerts, release
from crucible.board import (
    BOARD_SCHEMA_VERSION,
    board_delta,
    board_payload,
    build_board,
    pointer_may_move,
    read_acceptance,
    render_board_html,
)
from crucible.calendar import resolve_trading_day
from crucible.components import load_registry
from crucible.console.public import (
    PUBLIC_JSON_KEY,
    PUBLIC_KEY,
    build_public_page,
    write_public_page,
)
from crucible.console.render import (
    CONSOLE_JSON_KEY,
    CONSOLE_KEY,
    build_page,
    classify_registry,
    write_page,
)
from crucible.documents import load_document_bytes, read_store_document
from crucible.drift import drift_metrics
from crucible.drift_inputs import DRIFT_INPUT_SCHEMA_VERSION, compute_drift_inputs
from crucible.gate import (
    GATES,
    PHASES,
    build_ladder,
    evaluate,
)
from crucible.keys import (
    BOARD_CURRENT_KEY,
    BOARD_HTML_KEY,
    DRIFT_INPUTS,
    board_key,
    drift_input_key,
    drift_metrics_key,
)
from crucible.manifest import RUN_MANIFEST_SCHEMA_VERSION
from crucible.release_lock_sweep import release_lock_findings, release_lock_metric
from crucible.runner import RunContext, run_job, spot_interruption_guard
from crucible.store import Store, open_store, sha256_hex

__all__ = [
    "console_handler",
    "drift_handler",
    "heartbeat_handler",
    "release_pin_apply_handler",
    "release_pin_handler",
    "release_pin_request_handler",
    "smoke_handler",
    "sweep_handler",
]

#: The ambient live read paths the smoke exercises **in addition to**
#: verifying the release under test. A real end-to-end read against the live
#: store, not a ping: §4.11 gates the pointer flip on this, and a gate that
#: only proves the process started would promote a build that cannot reach
#: its own data. Each entry is (label, key-or-prefix).
#:
#: Neither is `required`, and there is deliberately no `required` column any
#: more. A required flag that was False on every row was a column with one
#: value, and it read as if some read somewhere could fail the smoke while
#: none could. What actually gates the flip is
#: :func:`_verify_release_artifacts` below, whose every check raises. These
#: two are *observations*: on a first-ever deploy the pointer is unset and
#: `runs/` is empty, and refusing to bootstrap the system is not a gate. They
#: are counted honestly into `smoke_ok` instead of asserted.
SMOKE_READS: tuple[tuple[str, str], ...] = (
    ("release pointer", release.POINTER_KEY),
    ("run manifests", "runs/"),
)


def _store(args: argparse.Namespace) -> Store:
    # alpha-engine-config-I9922 N1: read-only under `--dry-run` regardless of
    # whether the calling handler's own body checks the flag — this is the
    # single point every one of this module's six handlers resolves a store
    # through.
    return open_store(getattr(args, "store", None), dry_run=bool(getattr(args, "dry_run", False)))


# ── release.pin ───────────────────────────────────────────────────────────


def _now() -> dt.datetime:
    """The pin's wall clock. One seam, so the off-market-hours refusal is
    verified by running the handler with a frozen clock (I10649 closes-when)."""
    return dt.datetime.now(dt.UTC)


def release_pin_handler(args: argparse.Namespace) -> int:
    """Repoint `releases/current`, or pin the trader. §4.11 rollback.

    Runs through the runner like every other job, so a rollback at 3am leaves
    a manifest saying who moved what and when — which is the difference
    between an incident with a timeline and one reconstructed from memory.

    `--target trader` goes through :func:`crucible.release.pin_trader`
    (alpha-engine-config-I10649): refused inside the market-hours blackout and
    refused without a passing `trader.smoke` manifest for this sha. The smoke
    manifest the pin was gated on is recorded as this run's INPUT (content
    hash included) and named in the metric, so `crucible explain` walks from
    the pin to the smoke that certified it.
    """
    dry_run = bool(getattr(args, "dry_run", False))
    store = _store(args)
    target = args.target

    def body(ctx: RunContext) -> None:
        key = release.POINTER_KEY if target == "current" else release.TRADER_PIN_KEY
        # The pointer being MOVED, not `releases/current` for both targets: a
        # trader pin used to report current's sha as its "before".
        before, _ = release.read_pointer(store, key)
        evidence: release.TraderSmokeEvidence | None = None
        if target == "trader":
            now = _now()
            refusal = release.trader_pin_window_refusal(now)
            if refusal is not None:
                raise release.TraderPinRefusedError(
                    f"refusing to pin the trader to {args.sha}: {refusal}"
                )
            evidence = release.passing_trader_smoke(store, args.sha)
            ctx.record_input(
                evidence.manifest_key,
                store.get_bytes(evidence.manifest_key),
                schema_version=RUN_MANIFEST_SCHEMA_VERSION,
            )
        # alpha-engine-config-I9922 R2-1: the store guard is the backstop —
        # `release.pin` MOVES A POINTER, so under `--dry-run` it never calls
        # `release.pin()` at all (rather than calling it and letting the
        # guard refuse mid-write, which is the wrong shape for a pointer
        # move specifically: `release.pin`'s own internals, not just
        # `ctx.record_output`, are what write). The preview is exactly the
        # move it would make — for the trader, after the window and the smoke
        # evidence have both been checked, so a dry run answers "would it move".
        if dry_run:
            print(
                f"release.pin --target {target}: {before or '(unset)'} would move to "
                f"{args.sha}" + (f" (gated on {evidence.run_id})" if evidence is not None else "")
            )
            return
        release.pin(store, args.sha, target=target, trader_smoke=evidence)
        ctx.record_output(key, store.get_bytes(key), schema_version=release.RELEASE_SCHEMA_VERSION)
        gated = (
            f" Gated on trader smoke {evidence.run_id} ({evidence.status}, "
            f"{evidence.manifest_key})."
            if evidence is not None
            else ""
        )
        ctx.record_metric(
            {
                "name": "release_pointer_moved",
                "module": "crucible.release",
                "metric_type": "operational",
                "value": 1.0,
                "unit": "pointer_moves",
                "n_floor": 0,
                "status": "OK",
                "status_reason": (
                    f"{target} moved from {before or '(unset)'} to {args.sha}.{gated}"
                ),
                "source_path": key,
                "last_updated_utc": ctx.started.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )

    run_job(
        "release.pin",
        body,
        store=store,
        trading_day=args.trading_day,
        dry_run=dry_run,
        run_mode=getattr(args, "run_mode", None),
    )
    return 0


# ── the queued trader pin (alpha-engine-config-I11545) ──────────────────────
#
# Brian's ruling: anyone may queue a sha at any time; the next clean
# post-close smokes and pins it; promotion stays explicit. Two jobs, both run
# by `.github/workflows/trader-pin.yml` under one identity:
#
# * `release.pin_request` (its `workflow_dispatch`) writes the request and
#   nothing else. It moves no pointer.
# * `release.pin_apply` (its weekday schedule) reads the request and, when it
#   is still the one pending change and the evening is clean, moves the pin
#   through `release.pin_trader` — the same window and smoke gate a human
#   `release.pin --target trader` goes through, and nothing weaker.

#: The request's calendar day is read in New York time, the timezone the
#: box's smoke timer and the NYSE session are both declared in.
_ET = ZoneInfo("America/New_York")


def _requester_from_environment() -> tuple[str, str | None]:
    """``(requested_by, run_url)`` from the invocation's own environment.

    Read from the environment and never from a flag, so the dispatch input is
    the sha and nothing free-form: in Actions `GITHUB_ACTOR` is the person who
    pressed the button and the run URL is assembled from the run's own ids; on
    a laptop it is `$USER` and there is no run. A request with no requester is
    refused rather than filed anonymously.
    """
    requested_by = os.environ.get("GITHUB_ACTOR") or os.environ.get("USER") or ""
    if not requested_by.strip():
        raise ValueError(
            "release.pin_request: neither GITHUB_ACTOR nor USER is set, so the request "
            "would name no requester. A queued pin is an explicit act by somebody."
        )
    server = os.environ.get("GITHUB_SERVER_URL")
    repository = os.environ.get("GITHUB_REPOSITORY")
    run_id = os.environ.get("GITHUB_RUN_ID")
    run_url = (
        f"{server}/{repository}/actions/runs/{run_id}" if server and repository and run_id else None
    )
    return requested_by.strip(), run_url


def release_pin_request_handler(args: argparse.Namespace) -> int:
    """Queue ``args.sha`` for the trader: write `trader/pin_request.json`.

    Validates before writing (40-hex, published wheel) and records the pin it
    was asked FROM, read live. Moves no pointer; `release.pin_apply` does that
    on the next clean evening, or a human does it with `release.pin --target
    trader`, in which case this request reads `noop` or `stale` afterwards.
    """
    dry_run = bool(getattr(args, "dry_run", False))
    store = _store(args)

    def body(ctx: RunContext) -> None:
        requested_by, run_url = _requester_from_environment()
        document, payload = release.build_trader_pin_request(
            store, args.sha, requested_by=requested_by, requested_at=_now(), run_url=run_url
        )
        ctx.record_output(
            release.TRADER_PIN_REQUEST_KEY,
            payload,
            schema_version=release.TRADER_PIN_REQUEST_SCHEMA_VERSION,
        )
        ctx.record_metric(
            {
                "name": "trader_pin_requested",
                "module": "crucible.release",
                "metric_type": "operational",
                "value": 1.0,
                "unit": "requests",
                "n_floor": 0,
                "status": "OK",
                "status_reason": (
                    f"queued {document.sha} for the trader (from "
                    f"{document.from_sha or '(unset)'}), requested by {document.requested_by}"
                ),
                "source_path": release.TRADER_PIN_REQUEST_KEY,
                "last_updated_utc": ctx.started.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )
        print(
            f"release.pin_request: {release.TRADER_PIN_REQUEST_KEY} -> {document.sha} "
            f"(from {document.from_sha or '(unset)'})"
        )

    run_job(
        "release.pin_request",
        body,
        store=store,
        trading_day=args.trading_day,
        dry_run=dry_run,
        run_mode=getattr(args, "run_mode", None),
    )
    return 0


def _request_day_et(request: Any) -> dt.date:
    instant = dt.datetime.strptime(request.requested_at, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=dt.UTC
    )
    return instant.astimezone(_ET).date()


def release_pin_apply_handler(args: argparse.Namespace) -> int:
    """Apply the queued trader pin, if it is still the one pending change.

    `none`/`noop` are complete, correct results (`ok`). `stale` RAISES: the pin
    was moved after the request was made, and applying it would undo that, so a
    human decides. `fresh` pins through :func:`crucible.release.pin_trader`,
    swapping against the pin version read in the same pass — gated on:

    * ``--postclose clean``: the workflow's guard read today's post-close
      pipeline as finished cleanly. `unclean` with a fresh request RAISES; with
      nothing to apply it is irrelevant and the run is `ok`.
    * a passing `trader.smoke` for the sha. NO smoke yet is a wait (`ok`) on
      the evening of the session the request was made in or after — the box
      smokes a fresh request at midday, so a request queued later that day is
      smoked tomorrow — and a failure on any later evening: the smoke that
      should have run did not. A smoke that FAILED always raises.
    """
    dry_run = bool(getattr(args, "dry_run", False))
    store = _store(args)

    def body(ctx: RunContext) -> None:
        pending = release.pending_trader_pin(store)
        if pending.request_bytes is not None:
            ctx.record_input(
                release.TRADER_PIN_REQUEST_KEY,
                pending.request_bytes,
                schema_version=release.TRADER_PIN_REQUEST_SCHEMA_VERSION,
            )
        moved = 0.0
        outcome = pending.describe()
        if pending.state == "stale":
            raise release.TraderPinRefusedError(f"release.pin_apply: {outcome}")
        if pending.state == "fresh":
            assert pending.request is not None  # `fresh` always carries its request
            sha = pending.request.sha
            if args.postclose != "clean":
                raise release.TraderPinRefusedError(
                    f"release.pin_apply: {outcome}; not applied, because the guard read no "
                    "clean post-close for this session. The request stays queued and is "
                    "applied on the next clean evening."
                )
            evidence, failures = release.read_trader_smoke(store, sha)
            if evidence is None and failures:
                raise release.TraderPinRefusedError(
                    f"release.pin_apply: {outcome}; the trader smoke for {sha} FAILED: "
                    + " | ".join(failures)
                )
            if evidence is None:
                if _request_day_et(pending.request) < ctx.trading_day:
                    raise release.TraderPinRefusedError(
                        f"release.pin_apply: {outcome}; no `{release.TRADER_SMOKE_JOB}` "
                        f"manifest exists for {sha}, and the request predates session "
                        f"{ctx.trading_day.isoformat()}, whose midday smoke should have run it. "
                        "Check the executor box's trader-pin-smoke timer."
                    )
                outcome += (
                    f"; waiting: no `{release.TRADER_SMOKE_JOB}` manifest for {sha} yet (queued "
                    "after this session's smoke); the next session's smoke runs it"
                )
            elif dry_run:
                print(
                    f"release.pin_apply: trader pin {pending.pin_sha or '(unset)'} would move "
                    f"to {sha} (gated on {evidence.run_id})"
                )
                return
            else:
                ctx.record_input(
                    evidence.manifest_key,
                    store.get_bytes(evidence.manifest_key),
                    schema_version=RUN_MANIFEST_SCHEMA_VERSION,
                )
                release.pin_trader(store, sha, now=_now(), expect=pending.pin_version)
                ctx.record_output(
                    release.TRADER_PIN_KEY,
                    store.get_bytes(release.TRADER_PIN_KEY),
                    schema_version=release.RELEASE_SCHEMA_VERSION,
                )
                moved = 1.0
                outcome = (
                    f"moved the trader pin from {pending.pin_sha or '(unset)'} to {sha}, as "
                    f"requested by {pending.request.requested_by} at "
                    f"{pending.request.requested_at}; gated on trader smoke {evidence.run_id} "
                    f"({evidence.manifest_key})"
                )
        print(f"release.pin_apply: {outcome}")
        ctx.record_metric(
            {
                "name": "trader_pin_moved",
                "module": "crucible.release",
                "metric_type": "operational",
                "value": moved,
                "unit": "pointer_moves",
                "n_floor": 0,
                "status": "OK",
                "status_reason": outcome,
                "source_path": release.TRADER_PIN_KEY,
                "last_updated_utc": ctx.started.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )

    run_job(
        "release.pin_apply",
        body,
        store=store,
        trading_day=args.trading_day,
        dry_run=dry_run,
        run_mode=getattr(args, "run_mode", None),
    )
    return 0


# ── smoke ─────────────────────────────────────────────────────────────────


def _verify_release_artifacts(store: Store, sha: str, ctx: RunContext) -> list[str]:
    """Read and verify the artifacts for ``sha``. Every failure here RAISES.

    This is the part of the smoke that gates the flip. It is not a
    reachability probe: it reads the two objects that constitute the release
    `deploy.yml` is about to promote, out of the live store, and checks that
    what came back is what `release.json` says it is.

    Three failures it is built to catch, none of which a ping would:

    * **the wheel for this sha is not there.** `flip_on_smoke` would happily
      promote it — the pointer's own guard (`pin` refuses an unpublished
      sha) fires only on the wheel key, so a prefix missing its
      `release.json` promotes fine and every consumer of the record breaks
      later, away from the deploy that caused it.
    * **the record describes a different build.** A `release.json` for
      another sha under this prefix makes the rollback target a build it does
      not describe.
    * **the bytes in the store are not the bytes that were built.** A
      truncated or clobbered upload leaves a wheel whose sha256 no longer
      matches `wheel_sha256`; nothing downstream re-hashes, so the first
      symptom would be an install failure on a box at 06:30.

    Returns the keys it read, for the manifest's lineage and for the
    `smoke_ok` count. Raising is the whole design: §4.2 has no status between
    ok and failed, so a smoke that could not verify its release cannot report
    anything but `failed`, and the flip refuses on anything but `ok`.
    """
    # `release.json` is read (and parsed) BEFORE the wheel key can even be
    # computed: `wheel_key_for` needs `record.wheel_filename`, and a v2
    # record's wheel is not at the v3-derived `wheel_key(sha)` path
    # (alpha-engine-config-I9908 — see `crucible.release.parse_release_record`).
    # One resolver (alpha-engine-config-I9932): read release.json, check it
    # describes THIS sha, take the wheel key the record itself names. The
    # three restatements of that sequence disagreed on the sha guard.
    try:
        published = release.resolve_published_wheel(store, sha)
    except release.StaleReleasePointerError as exc:
        raise FileNotFoundError(
            f"smoke: the release under test is not published. "
            f"[{release.release_json_key(sha)!r}] absent for {sha}. Promoting a pointer "
            "at a prefix whose artifacts are not there is a stale pointer written "
            "deliberately; the deploy publishes before it smokes, so this means the "
            "publish did not land."
        ) from exc
    except release.ReleaseRecordMismatchError as exc:
        raise ValueError(
            f"smoke: {exc} Gating a promotion on a record belonging to another build is "
            "the gate failing open."
        ) from exc
    meta_k = published.record_key
    meta_bytes = published.record_bytes
    record = published.record
    wheel_k = published.wheel_key
    if not store.exists(wheel_k):
        raise FileNotFoundError(
            f"smoke: the release under test is not published. [{wheel_k!r}] absent "
            f"for {sha}. Promoting a pointer at a prefix whose artifacts are not there "
            "is a stale pointer written deliberately; the deploy publishes before it "
            "smokes, so this means the publish did not land."
        )
    wheel_bytes = store.get_bytes(wheel_k)
    digest = sha256_hex(wheel_bytes)
    if digest != record.wheel_sha256:
        raise ValueError(
            f"smoke: the wheel at {wheel_k} hashes to {digest}, but {meta_k} claims "
            f"{record.wheel_sha256}. The stored artifact is not the one that was built "
            "and tested; promoting it would install unverified bytes on every box that "
            "follows releases/current."
        )
    ctx.record_input(meta_k, meta_bytes, schema_version=release.RELEASE_SCHEMA_VERSION)
    ctx.record_input(wheel_k, wheel_bytes, schema_version=release.RELEASE_SCHEMA_VERSION)
    lock_k = _verify_release_wheelhouse(store, record, ctx)
    return [meta_k, wheel_k, lock_k]


def _verify_release_wheelhouse(store: Store, record: release.ReleaseRecord, ctx: RunContext) -> str:
    """The wheelhouse half of the gate (alpha-engine-config-I10812). Raises.

    A box installs a release offline from `releases/{sha}/wheelhouse/` and
    nothing else, so a release whose wheelhouse is absent, incomplete, or
    disagrees with its lock is a release no box can install — promoting it is
    the outage, not a degraded deploy. Checked here:

    * the record carries a wheelhouse at all;
    * every object it names exists in the store;
    * the lock in the store is the lock the record hashed, and the recorded
      wheels satisfy it (the same check the publisher applies);
    * THIS process was installed from that wheelhouse — `deploy.yml` runs the
      smoke from the offline install and exports its digest, so a smoke run
      from any other dependency set fails here rather than grading one no box
      installs.

    The wheel bytes themselves are not re-hashed here: `deploy.yml`'s install
    proof installs every one with `--require-hashes` before this job starts.
    Returns the lock key, for lineage.
    """
    from crucible.runner import WHEELHOUSE_DIGEST_ENV  # noqa: PLC0415 - one call site
    from crucible.wheelhouse import (  # noqa: PLC0415 - one call site
        WheelhouseLockMismatchError,
        verify_against_lock,
    )

    try:
        manifest = release.require_wheelhouse(record)
    except release.ReleaseHasNoWheelhouseError as exc:
        raise ValueError(f"smoke: {exc}") from exc
    *wheel_keys, (lock_k, lock_sha256) = record.wheelhouse_keys
    missing = [key for key, _ in wheel_keys if not store.exists(key)]
    if missing or not store.exists(lock_k):
        raise FileNotFoundError(
            f"smoke: release {record.sha}'s wheelhouse is incomplete in the store — "
            f"{len(missing)} of {len(wheel_keys)} wheels absent"
            + ("" if store.exists(lock_k) else f", and no lock at {lock_k}")
            + f" (first: {(missing or [lock_k])[0]}). A box installing it would fail offline."
        )
    lock_bytes = store.get_bytes(lock_k)
    if sha256_hex(lock_bytes) != lock_sha256:
        raise ValueError(
            f"smoke: the lock at {lock_k} hashes to {sha256_hex(lock_bytes)}, but the release "
            f"record claims {lock_sha256}. The box's `--require-hashes` install would trust a "
            "lock the release never vouched for."
        )
    try:
        verify_against_lock(manifest["wheels"], lock_bytes.decode("utf-8"))
    except WheelhouseLockMismatchError as exc:
        raise ValueError(f"smoke: {exc}") from exc
    installed = os.environ.get(WHEELHOUSE_DIGEST_ENV, "")
    if installed != manifest["digest"]:
        raise ValueError(
            f"smoke: this process names wheelhouse {installed or '(none)'} in "
            f"${WHEELHOUSE_DIGEST_ENV}, but release {record.sha}'s wheelhouse is "
            f"{manifest['digest']}. The smoke must run from the offline install of the "
            "published wheelhouse, or it grades a dependency set no box installs."
        )
    ctx.record_input(lock_k, lock_bytes, schema_version=release.RELEASE_SCHEMA_VERSION)
    return lock_k


def smoke_handler(args: argparse.Namespace) -> int:
    """A real run against live S3 read paths. Its manifest gates the flip.

    Deliberately read-only against the data it verifies: a smoke that wrote
    production artifacts would make every deploy a data event, and a failed
    smoke would leave half of one behind. The only thing it writes is its own
    manifest, which is the artifact the deploy reads.

    ``--release`` is required and is recorded as the manifest's
    `release_sha`, because :func:`crucible.release.flip_on_smoke` refuses a
    manifest belonging to another build — promoting on another build's smoke
    is the gate failing open.

    **What makes this a gate rather than a ceremony.** It verifies the
    artifacts for the sha it is gating (:func:`_verify_release_artifacts`),
    by reading them out of the live store and re-hashing the wheel against
    the record — so a smoke against an empty store, or against a sha nothing
    published, fails instead of reporting `ok` with an empty `inputs[]`. And
    `smoke_ok` is **counted**, not asserted: its value is the number of read
    paths that actually returned something and its status is derived from
    whether every one of them did. A gate whose metric is the literal `"OK"`
    is a gate that says the same thing on every run it survives, which is
    indistinguishable from one that did nothing.
    """
    store = _store(args)

    def body(ctx: RunContext) -> None:
        read: list[str] = []
        degraded: list[str] = []

        # The gate proper. Raises on every failure; nothing below runs unless
        # the release under test is present and byte-verified.
        read += _verify_release_artifacts(store, args.release, ctx)

        for label, key in SMOKE_READS:
            if key.endswith("/"):
                # A listing, not a get: the paginated list path is the one a
                # truncation bug hides in, so the smoke exercises it.
                found = sum(1 for _ in store.list_keys(key))
                ctx.record_rows(rows_in=ctx.rows_in + found, rows_out=ctx.rows_out)
                if found:
                    read.append(key)
                else:
                    degraded.append(f"{label} ({key}) listed zero keys")
                continue
            if not store.exists(key):
                degraded.append(f"{label} ({key}) is unset")
                continue
            payload = store.get_bytes(key)
            ctx.record_input(key, payload, schema_version=release.RELEASE_SCHEMA_VERSION)
            read.append(key)
            if key == release.POINTER_KEY:
                # The SAME bytes recorded as lineage decide `pointed`. A second
                # fetch here is a TOCTOU on the one object `deploy._flip` moves
                # concurrently: lineage would say one sha and the smoke act on
                # another (crucible-PR81 review, B1).
                # `alpha-engine-config-I9847` (wave 2): validated through
                # `release.parse_release_pointer` — the same wrapper
                # `release.read_pointer` uses — rather than indexed off the
                # raw dict. See `ReleasePointerDocument`'s docstring for why
                # this read and `read_pointer`'s are the two STRICT reads it
                # exists for.
                pointed = release.parse_release_pointer(key, load_document_bytes(key, payload)).sha
                # A v2-or-v3 record for the POINTED sha, not `sha` under
                # test — its wheel may live at the legacy (unpip-installable)
                # v2 path, so `wheel_key(pointed)` alone cannot answer this.
                # A missing or unreadable release.json here means the same
                # thing as a missing wheel: the pointed build is broken.
                # The POINTER's own shape is checked OUTSIDE the swallow
                # below: a `releases/current` that does not even name a sha
                # is a corrupt pointer, not a broken pointed build, and it
                # raises — review of alpha-engine-config-I9932 measured that
                # with the check inside the resolver it was swallowed with
                # the record conditions and the smoke read `ok` over it.
                release.assert_sha(pointed)
                pointed_wheel_present = False
                try:
                    pointed_wheel_present = store.exists(
                        release.resolve_published_wheel(store, pointed).wheel_key
                    )
                except (release.StaleReleasePointerError, ValueError, TypeError):
                    # Swallowed here only: this whole branch is the
                    # documented single non-raise in the job (see below)
                    # — an absent, malformed, or wrong-sha record for the
                    # POINTED release (`ReleaseRecordMismatchError` is a
                    # `ValueError`) is itself evidence the pointed build is
                    # broken, which is exactly what this branch already
                    # records as `degraded`, never as a raise. Exactly those
                    # three record conditions; the pointer itself was
                    # validated above.
                    pointed_wheel_present = False
                if not pointed_wheel_present:
                    # Recorded, NOT raised, and this is the only deliberate
                    # non-raise in the job. Failure mode swallowed: a stale
                    # `releases/current` whose wheel is gone — every job that
                    # resolves the pointer is already broken. It does not fail
                    # the smoke because flipping the pointer to this verified
                    # build is the REMEDY, and refusing to deploy would remove
                    # the only thing that fixes it. Recording surface: the
                    # `smoke_ok` MetricRecord below goes BREACH with this
                    # sentence in `status_reason`, on the manifest §4.5's page
                    # renders — so it is loud on the console rather than
                    # silent in a log.
                    # Human-readable diagnostic text, not a store key — no
                    # store call reads this string (alpha-engine-config-I9852).
                    degraded.append(
                        f"releases/current names {pointed}, whose wheel is absent; every "
                        "job resolving the pointer is broken until this flip lands"
                    )

        # The three release artifacts (release.json, wheel, wheelhouse lock),
        # plus the ambient paths.
        attempted = 3 + len(SMOKE_READS)
        # alpha-engine-config-I10069: the smoke job runs `uv sync --frozen`
        # against this checkout, which never installs an optional extra —
        # this process cannot `import arcticdb` itself and prove anything.
        # What CAN prove it is `deploy.yml`'s install-proof step, which
        # installs the just-published WHEEL with every extra
        # `pyproject.toml` declares and imports each one's module, on the
        # same x86_64 runner the box now uses, BEFORE this job starts. It
        # hands the verified set forward as `$CRUCIBLE_SMOKED_EXTRAS`
        # (comma-separated) via `$GITHUB_ENV`, and it is recorded here on the
        # `smoke_ok` metric so `crucible.deploy._flip` can refuse a manifest
        # that never proved the extras the box actually installs — the same
        # read the flip already does for `release_sha`.
        smoked_extras = sorted(
            extra.strip()
            for extra in os.environ.get("CRUCIBLE_SMOKED_EXTRAS", "").split(",")
            if extra.strip()
        )
        ctx.record_metric(
            {
                "name": "smoke_ok",
                "module": "crucible.track_c",
                "metric_type": "operational",
                "value": float(len(read)),
                "unit": "read_paths",
                "n_floor": 0,
                "status": "OK" if not degraded else "BREACH",
                "status_reason": (
                    f"{len(read)} of {attempted} live read paths returned data for "
                    f"release {args.release}; wheel and release.json verified "
                    f"byte-for-byte, wheelhouse complete and graded."
                    + (f" Degraded: {'; '.join(degraded)}." if degraded else "")
                ),
                "source_path": "runs/smoke/{trading_day}/run.json",
                "last_updated_utc": ctx.started.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "smoked_extras": smoked_extras,
            }
        )

    with spot_interruption_guard():
        run_job(
            "smoke",
            body,
            store=store,
            trading_day=args.trading_day,
            run_mode=getattr(args, "run_mode", None),
            release_sha=args.release,
            # A smoke that needed a retry is a smoke that told us something.
            # Retrying it would promote a build whose first attempt failed,
            # and the deploy would read a green manifest over an amber fact.
            transient_retry=False,
            dry_run=bool(getattr(args, "dry_run", False)),
        )
    return 0


# ── alerts.sweep ──────────────────────────────────────────────────────────


def sweep_handler(args: argparse.Namespace) -> int:
    """Evaluate both page conditions, group by cause, page once per group.

    ``--now`` (`alpha-engine-config-I10125`) is resolved and validated HERE,
    before `run_job` starts — a malformed override, a non-trading day or a
    future day is a usage error the operator sees immediately, not a failed
    manifest the job body raised into. `None` when the flag is absent, which
    is the scheduled/unattended path: it never passes `--now`, so `sweep`
    below gets `now=None` and evaluates the real wall clock exactly as
    before this flag existed (`tests/test_alerts_now_override.py` asserts
    this).
    """
    dry_run = bool(getattr(args, "dry_run", False))
    store = _store(args)
    result: dict[str, Any] = {}
    now_raw = getattr(args, "now", None)
    now_override = None
    if now_raw:
        try:
            now_override = alerts.parse_now_override(now_raw)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    def body(ctx: RunContext) -> None:
        # `alerts.sweep(dry_run=)` evaluates and groups exactly as a real
        # sweep would but never calls `emit` — `outcome["bus_keys"]` is `()`
        # on this path, so the loop below writes nothing without needing its
        # own guard (alpha-engine-config-I9922 R2-1).
        outcome = alerts.sweep(store, sweep_run_id=ctx.run_id, dry_run=dry_run, now=now_override)
        result.update(outcome)
        for key in outcome["bus_keys"]:
            ctx.record_output(
                key, store.get_bytes(key), schema_version=alerts.ALERT_BUS_SCHEMA_VERSION
            )
        ctx.record_metric(outcome["metric"])
        ctx.record_metric(
            {
                "name": "pages_emitted",
                "module": "crucible.alerts",
                "metric_type": "operational",
                "value": float(outcome["pages_emitted"]),
                "unit": "pages",
                "n_floor": 0,
                "status": "OK",
                "status_reason": (
                    f"{outcome['pages_emitted']} page group(s) covering "
                    f"{outcome['members']} condition instance(s)."
                ),
                "source_path": "alerts/{trading_day}/{alert_id}.json",
                "last_updated_utc": ctx.started.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )
        # alpha-engine-config-I9798: I9787 fixed the write-time race — the
        # lock now rides on the same `put_object` that writes the bytes —
        # but nothing detected a release object published before that fix,
        # or one whose lock was lost some other way. Read-only, piggybacked
        # on this sweep rather than a second scheduled job: it is the same
        # "walk the store, find a fact nobody is watching" shape as the two
        # page conditions above, and a release object with no retention is a
        # console row (plan §4.6), not a page — it does not name an
        # operator's next action the way an absent or failed manifest does.
        # alpha-engine-config-I10967. The same read-only, piggybacked shape
        # as the release-lock sweep below, for the class one level in from
        # the two page conditions: a manifest KEY that N runs overwrite. An
        # on-demand job that passes no discriminator records only its last
        # invocation for a trading day, and every earlier run is
        # indistinguishable from a run that never happened - which is what
        # `experiment.new` did seven times on 2026-09-17. Derived over the
        # whole registry, never an enumerated list, and recorded as
        # `unmeasurable` rather than paged: the condition is standing, not
        # eventful, and `unmeasurable` is its own state that never renders as
        # green.
        graded_jobs = alerts.on_demand_graded_jobs()
        # The same window the sweep's own two page conditions are graded
        # over, from the same function - a finding window narrower or wider
        # than the pages' would report a different store than the one the
        # pages were computed from.
        graded_days = alerts.days_to_evaluate(store, now_override or ctx.started)
        invocation_findings = alerts.indistinguishable_invocation_findings(store, days=graded_days)
        result["indistinguishable_invocations"] = [
            {
                "job": f.job,
                "trading_day": f.trading_day,
                "manifest": f.key,
                "run_id": f.run_id,
            }
            for f in invocation_findings
        ]
        ctx.record_metric(
            alerts.indistinguishable_invocation_metric(
                invocation_findings, graded_jobs=graded_jobs, now=ctx.started
            )
        )

        lock_findings = release_lock_findings(store)
        lock_metric = release_lock_metric(lock_findings, now=ctx.started)
        result["release_lock_findings"] = [
            {"key": f.key, "state": f.state, "detail": f.detail} for f in lock_findings
        ]
        ctx.record_metric(lock_metric)

    run_job(
        "alerts.sweep",
        body,
        store=store,
        trading_day=args.trading_day,
        run_mode=getattr(args, "run_mode", None),
        # The sweep fires every calendar day at 21:00 ET, and Friday,
        # Saturday and Sunday all resolve to Friday's trading day — three
        # writers, one key, without this. `calendar_date` is only known once
        # the runner resolves `trading_day`/`started`, so it is a callable
        # rather than a value computed here (alpha-engine-config-I9781).
        discriminator=lambda ctx: ctx.calendar_date.isoformat(),
        dry_run=bool(getattr(args, "dry_run", False)),
        now_override=now_override,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "metric"}, indent=2))
    return 0


def heartbeat_handler(args: argparse.Namespace) -> int:
    """The weekly proof that the alerting path itself is alive."""
    dry_run = bool(getattr(args, "dry_run", False))
    store = _store(args)
    result: dict[str, Any] = {}

    def body(ctx: RunContext) -> None:
        # `alerts.heartbeat(dry_run=)` computes the same summary and message
        # but never calls `emit` and never sends the heartbeat message itself
        # (alpha-engine-config-I9922 R2-1) — a store guard alone cannot catch
        # a real message send, which is not a store write.
        summary = alerts.heartbeat(store, dry_run=dry_run)
        result.update(summary)
        ctx.record_metric(summary["metric"])
        # alpha-engine-config-I10024: who would receive the page. FAIL when
        # no confirmed human leg or no Lambda leg; `unmeasurable` when the
        # list could not be read (and the run then fails on that fault).
        ctx.record_metric(summary["subscribers"])
        # alpha-engine-config-I9986: the week's manifest-vs-cost-sink
        # reconciliation. `heartbeat` has already raised on a FAIL, so a row
        # recorded here is OK or `unmeasurable`.
        ctx.record_metric(summary["cost_reconciliation"])
        ctx.record_metric(
            {
                "name": "pages_last_28_trading_days",
                "module": "crucible.alerts",
                "metric_type": "operational",
                "value": float(summary["pages_in_window"]),
                "unit": "pages",
                "n_floor": 0,
                "status": "OK",
                "status_reason": (
                    f"{summary['runs_ok']} run(s) ok, {summary['runs_failed']} failed, "
                    f"${summary['cost_usd']:.2f} over the trailing trading week."
                ),
                "source_path": "alerts/{trading_day}/{alert_id}.json",
                "last_updated_utc": ctx.started.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "horizon_trading_days": alerts.CEILING_WINDOW_TRADING_DAYS,
            }
        )

    run_job(
        "heartbeat",
        body,
        store=store,
        trading_day=args.trading_day,
        dry_run=dry_run,
        run_mode=getattr(args, "run_mode", None),
    )
    if dry_run:
        print(result["message"])
    return 0


# ── drift ─────────────────────────────────────────────────────────────────


def drift_handler(args: argparse.Namespace) -> int:
    """Three drift MetricRecords per cycle, computed from the feature layer,
    the arms' cross-sections and their settled IC (`crucible.drift_inputs`).

    **The inputs are computed here, then filed as this job's outputs.** Until
    2026-09-05 this handler READ `drift/{day}/input_*.json` "from the
    artifacts track A and track B write" and raised when they were absent —
    and no job had ever written them, so the first arc to reach `drift`
    (weekly@2026-08-07) died at it. The three documents are still written,
    so `crucible explain` walks a drift row back to the frames and
    cross-sections it was derived from, but they are derived by this job
    from artifacts that exist.

    **It still raises when there is nothing to measure** — an absent feature
    layer for the day — rather than emitting three rows of `UNREPORTED` and
    exiting 0. A prediction or IC row whose history has not settled yet is
    `UNREPORTED` WITH its reason on the row, which is the honest reading of
    a young store, and `drift_metrics` refuses a cycle in which all three
    are.
    """
    dry_run = bool(getattr(args, "dry_run", False))
    store = _store(args)

    def body(ctx: RunContext) -> None:
        day = ctx.trading_day.isoformat()
        computed = compute_drift_inputs(store, ctx.trading_day)
        for source in computed.sources:
            ctx.record_input(source, store.get_bytes(source))
        payloads = computed.as_dict()
        inputs = {name: drift_input_key(name, day) for name in DRIFT_INPUTS}
        for name, key in inputs.items():
            body_bytes = json.dumps(payloads[name], indent=2, sort_keys=True).encode("utf-8")
            if not dry_run:
                ctx.record_output(key, body_bytes, schema_version=DRIFT_INPUT_SCHEMA_VERSION)

        records = drift_metrics(
            trading_day=ctx.trading_day,
            feature_psi_by_name=payloads["features"]["psi_by_feature"],
            prediction_psi=payloads["predictions"]["psi"],
            ic_decay_by_horizon={int(h): v for h, v in payloads["ic"]["decay_by_horizon"].items()},
            unmeasured_reasons={
                name: doc["unmeasured_reason"]
                for name, doc in payloads.items()
                if doc.get("unmeasured_reason")
            },
            # `alpha-engine-config-I10071`: names which of the two drift
            # comparisons (cross-sectional vs along-time) produced the worst
            # feature's ratio, on the row itself.
            feature_method_by_name=payloads["features"].get("method_by_feature", {}),
        )
        for record in records:
            ctx.record_metric(record)
        payload = json.dumps(records, indent=2, sort_keys=True).encode("utf-8")
        # alpha-engine-config-I9922 R2-1: the store guard is the backstop —
        # `drift` has a natural report (the three records themselves),
        # printed below under `--dry-run` rather than reached only by dying
        # on the guard.
        if dry_run:
            print(payload.decode("utf-8"))
        else:
            ctx.record_output(drift_metrics_key(day), payload, schema_version="metric_record.v1")

    run_job(
        "drift",
        body,
        store=store,
        trading_day=args.trading_day,
        dry_run=bool(getattr(args, "dry_run", False)),
        run_mode=getattr(args, "run_mode", None),
    )
    return 0


# ── console ───────────────────────────────────────────────────────────────


def board_handler(args: argparse.Namespace) -> int:
    """Render the fully-declared board and file it. Reads; never runs.

    `alpha-engine-config-I9837`. Deliberately its OWN job on its OWN daily
    schedule rather than a stage of `console`: `console` is `dispatch: arc`,
    dispatched by the weekly driver, and the weekly driver is phase-2 work.
    A board that only refreshes once phase 2 opens could not have rendered
    the gap that closed phase 1 — which is the entire reason it exists.

    The board's own red is NOT this job's exit status. Almost every row is
    red on day one and that is the correct reading, so a non-zero exit here
    would be a daily failure alert on a working producer, and a daily failure
    alert nobody can act on is how a channel gets muted. The job fails when
    the MEASUREMENT fails; the reading lives in the artifact.

    `--dry-run` already held the pointer and skipped `board/current.json`
    (I9863's narrow guard, below); passed through to `run_job` as
    `dry_run=True` (alpha-engine-config-I9922) so the run manifest itself is
    skipped too, rather than filing an `ok` firing for a run that touched
    neither the board nor the pointer.
    """
    store = _store(args)
    dry_run = bool(getattr(args, "dry_run", False))

    def body(ctx: RunContext) -> None:
        moment = dt.datetime.now(dt.UTC)
        registry = load_registry()
        # `args.trading_day` when the caller named one — the same day the run
        # manifest is keyed by. Resolving the wall clock here instead would
        # file a replayed board under today's key while its manifest sat
        # under the replayed day (§4.12).
        trading_day = args.trading_day or resolve_trading_day(moment)
        classifications, _ = classify_registry(store, registry, now=moment, trading_day=trading_day)
        # Evaluated ONCE, here, and handed to both consumers. `build_ladder`
        # would otherwise evaluate each gate itself and `build_board` would
        # have to evaluate them a second time to obtain the clause lists the
        # page renders — doubling every store read the clause set makes
        # (`alpha-engine-config-I9826`) and, worse, giving one page two
        # readings of the same gate that are free to disagree.
        readings = {
            phase.gate: evaluate(store, gate=phase.gate, trading_day=trading_day, registry=registry)
            for phase in PHASES
            if phase.gate is not None and phase.gate in GATES
        }
        ladder = build_ladder(
            store,
            trading_day=trading_day,
            registry=registry,
            now=moment,
            readings=readings,
        )
        board = build_board(
            store,
            now=moment,
            trading_day=trading_day,
            registry=registry,
            classifications=classifications,
            ladder=ladder,
            readings=readings,
        )
        acceptance, acceptance_note = read_acceptance(store, trading_day.isoformat())

        previous, previous_unreadable = _read_previous_board(store)

        payload = board_payload(board)
        may_move, pointer_reason = pointer_may_move(previous, board)
        # `--dry-run` must not touch the pointer. The flag is ignored by all
        # five track-C handlers (alpha-engine-config-I9863, which owns the
        # repo-wide fix), and honouring it for `board` alone would normally be
        # the wrong shape — but this is the one job whose `--dry-run` clobbers
        # `board/current.json`, the key the fleet-console adapter reads. One
        # narrow guard here, the class fix in I9863. `dry_run` is the
        # `board_handler`-level variable closed over here, not a fresh read —
        # `run_job` below is passed the same value (I9922).
        if dry_run:
            may_move = False
            pointer_reason = "--dry-run: the pointer and the page are not written"
        # The delta is computed against the incumbent only when this board is
        # the one that supersedes it. When the pointer is HELD -- a replay, or
        # a dry run -- comparing forward in time would publish a set of
        # "regressions" that are an artefact of reading an older board against
        # a newer one, which is a false reading on every replay.
        deltas = board_delta(previous, board) if may_move else []
        # The dated board is ALWAYS written; the pointer is conditional. A
        # replay must not clobber `board/current.json` with an older board —
        # that key is what the fleet console reads.
        written = [] if dry_run else [board_key(board.trading_day)]
        if may_move:
            written.append(BOARD_CURRENT_KEY)
        for key in written:
            store.put_bytes(key, payload)
            ctx.record_output(key, payload, schema_version=BOARD_SCHEMA_VERSION)
        # The page and the JSON, always together: `console-policy` requires
        # every view to serve the JSON an agent reads, because a page whose
        # numbers can only be scraped out of HTML is a page the next automated
        # reader re-derives incorrectly. It moves with the pointer, for the
        # same reason and under the same condition.
        if may_move:
            page = render_board_html(
                board,
                deltas,
                acceptance=acceptance,
                acceptance_note=acceptance_note,
            ).encode("utf-8")
            store.put_bytes(BOARD_HTML_KEY, page)
            ctx.record_output(BOARD_HTML_KEY, page, schema_version=BOARD_SCHEMA_VERSION)

        counts = board.counts()
        ctx.record_metric(
            {
                "name": "board_rows_red",
                "module": "crucible.board",
                "metric_type": "operational",
                "value": float(len(board.red)),
                "unit": "rows",
                "n_floor": 0,
                # OK, not BREACH. Red is the DECLARED starting state of this
                # instrument -- every row exists before the thing it measures
                # -- so a non-zero count is the expected reading and grading it
                # as a breach would page every day of the build. What is
                # graded instead is whether the board could be RENDERED, and
                # that is this job's own status.
                "status": "OK",
                "status_reason": (
                    f"{len(board.red)} of {len(board.rows)} row(s) red; "
                    f"{len(board.grey)} declared-not-built; {len(board.met)} met. "
                    "Red is the declared day-one state of a fully-declared board."
                ),
                "source_path": BOARD_CURRENT_KEY,
                "last_updated_utc": ctx.started.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )
        ctx.record_metric(
            {
                "name": "board_rows_moved",
                "module": "crucible.board",
                "metric_type": "operational",
                "value": float(len(deltas)),
                "unit": "rows",
                "n_floor": 0,
                # BREACH when the previous board could not be read. Reporting
                # "0 rows moved" over a failed comparison is a POSITIVE claim
                # asserted on no evidence — and this is the only surface that
                # reports a VANISHED declaration, so a silent zero here hides
                # exactly the event the board exists to catch.
                # BREACH only when the previous board could not be READ.
                # A held pointer -- a replay, a dry run -- is a deliberate
                # operator action and reporting it as a breach would teach the
                # reader to discount the one status that means something.
                # Either way the reason states plainly that no comparison was
                # made, so the zero is never a claim that nothing moved.
                "status": "OK" if previous_unreadable is None else "BREACH",
                # The digest reports DELTAS, not absolute state. A board
                # reading almost entirely PLANNED for weeks is correct and is
                # also the thing people stop opening; the delta is the part
                # that stays worth reading.
                "status_reason": (
                    f"the previous board could not be read ({previous_unreadable}), so no "
                    "comparison was possible. This is NOT a claim that nothing moved."
                    if previous_unreadable is not None
                    else f"no comparison was made: {pointer_reason}. This is NOT a claim "
                    "that nothing moved."
                    if not may_move
                    else "; ".join(d.describe() for d in deltas)
                    if deltas
                    else "no row changed state since the last board"
                ),
                "source_path": BOARD_CURRENT_KEY,
                "last_updated_utc": ctx.started.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )
        # UNMEASURABLE is visibly distinct from UNMET in the digest as well as
        # on the surface: a row we could not READ is a statement about our
        # access, and folding it into "unmet" reports our own outage as the
        # system's result.
        ctx.record_metric(
            {
                "name": "board_rows_unmeasurable",
                "module": "crucible.board",
                "metric_type": "operational",
                "value": float(counts["UNMEASURABLE"]),
                "unit": "rows",
                "n_floor": 0,
                "status": "OK" if counts["UNMEASURABLE"] == 0 else "BREACH",
                "status_reason": (
                    f"{counts['UNMEASURABLE']} row(s) could not be read at all. Unlike "
                    "UNMET, this is a statement about our access rather than about the "
                    "system, and its objective is zero."
                ),
                "source_path": BOARD_CURRENT_KEY,
                "last_updated_utc": ctx.started.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )
        # `alpha-engine-config-I9914`: the plan §6.1 schedule rows'
        # own outcome signal. Unlike `board_rows_red` -- whose red count IS
        # the declared day-one state of the whole board -- a schedule row
        # reading UNMET is a plan date that has already passed without the
        # phase it names reading MET, which plan §6.1 names as the one clock
        # that "cannot be faked". BREACH, not OK, for that reason: a calm OK
        # over a missed live Saturday would make the unfakeable clock read
        # the same as every other red row on this board.
        schedule_rows = [r for r in board.rows if r.source == "schedule"]
        overdue = [r for r in schedule_rows if r.state == "UNMET"]
        ctx.record_metric(
            {
                "name": "schedule_milestones_overdue",
                "module": "crucible.board",
                "metric_type": "operational",
                "value": float(len(overdue)),
                "unit": "milestones",
                "n_floor": 0,
                "status": "OK" if not overdue else "BREACH",
                "status_reason": (
                    f"{len(overdue)} of {len(schedule_rows)} plan §6.1 milestone(s) are "
                    f"past their date without the phase they name reading MET: "
                    f"{', '.join(r.id for r in overdue)}"
                    if overdue
                    else f"no §6.1 milestone is overdue ({len(schedule_rows)} declared)"
                ),
                "source_path": BOARD_CURRENT_KEY,
                "last_updated_utc": ctx.started.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )

    run_job(
        "board",
        body,
        store=store,
        trading_day=args.trading_day,
        dry_run=dry_run,
        run_mode=getattr(args, "run_mode", None),
    )
    return 0


def _read_previous_board(store: Store) -> tuple[dict[str, Any] | None, str | None]:
    """The last board filed. Returns `(document, unreadable_reason)`.

    **A failed read is not "no change".** The earlier version of this returned
    `None` on any exception, which fed `board_delta(None, …) -> []` and made
    the digest publish `board_rows_moved = 0, "no row changed state"` — a
    POSITIVE claim of no movement, asserted over a read that failed. That is
    the one thing this module's whole argument forbids, and it happened to be
    on the only surface that reports a VANISHED declaration.

    So the two cases are separated and the second is surfaced by the caller:
    "there is no previous board" (the first ever run) and "there is one and I
    could not read it" want opposite responses, and the second is
    `UNMEASURABLE` in this board's own vocabulary.
    """
    # The one guarded reader: absent → (None, None); present-and-unreadable
    # or denied → (None, reason), reported by the caller as UNMEASURABLE.
    read = read_store_document(store, BOARD_CURRENT_KEY)
    if read.absent:
        return None, None
    if read.problem is not None or read.document is None:
        return None, read.problem or f"{BOARD_CURRENT_KEY} read as absent mid-read"
    return read.document, None


def console_handler(args: argparse.Namespace) -> int:
    """Render the static page and its JSON from the manifests."""
    dry_run = bool(getattr(args, "dry_run", False))
    store = _store(args)

    def body(ctx: RunContext) -> None:
        page = build_page(store, now=dt.datetime.now(dt.UTC))
        # Every key `write_page` wrote, unpacked positionally by nobody: the
        # set has changed twice (the ladder joined it, then left it again in
        # `alpha-engine-config-I10575`) and a fixed-arity unpack would have
        # failed the console job on each change.
        #
        # Each key gets ITS OWN schema version rather than one blanket stamp
        # (`alpha-engine-config-I9825`). `gates/ladder.json` is no longer in
        # this map because `console` no longer writes it: the ladder has one
        # producer, `crucible gate --publish` / `crucible gate.close`, and
        # `build_page` READS the published key rather than re-evaluating
        # every gate under this job's environment
        # (`alpha-engine-config-I10575`).
        key_schema_versions = {
            CONSOLE_KEY: "console_page.v1",
            CONSOLE_JSON_KEY: "console_page.v1",
            PUBLIC_KEY: "public_page.v1",
            PUBLIC_JSON_KEY: "public_page.v1",
        }
        # alpha-engine-config-I9922 R2-1: the store guard is the backstop —
        # `console` has a natural report (the page's own JSON), printed below
        # under `--dry-run` rather than reached only by dying on the guard
        # inside `write_page` (which calls `store.put_bytes` directly, not
        # through `ctx`).
        #
        # The PUBLIC surface is rendered by the SAME job, from the same read of
        # the store (`alpha-engine-config-I10223`). Not a second component: a
        # separate job would need its own schedule, its own absence detector
        # and its own registry row, and would put the public page and the
        # operator console on two different readings of one store — which is
        # the drift `board` and `console` already paid for on the ladder. The
        # two pages share no PROJECTION (see `crucible.console.public`'s module
        # docstring); they share a trigger.
        public = build_public_page(store, now=dt.datetime.now(dt.UTC))
        if dry_run:
            print(page.to_json().decode("utf-8"))
            print(public.to_json().decode("utf-8"))
        else:
            for key in (*write_page(store, page), *write_public_page(store, public)):
                ctx.record_output(
                    key, store.get_bytes(key), schema_version=key_schema_versions[key]
                )
        ctx.record_metric(
            {
                "name": "components_unreported",
                "module": "crucible.console",
                "metric_type": "operational",
                "value": float(page.unreported),
                "unit": "components",
                "n_floor": 0,
                # §8.4: the transparency-gap count's objective is ZERO, so a
                # non-zero value is a BREACH and never a soft word. A number
                # over target that renders calmly is a target nobody is held
                # to.
                "status": "OK" if page.unreported == 0 else "BREACH",
                "status_reason": (
                    f"{page.unreported} of {page.population} component(s) render "
                    "UNREPORTED. The objective is zero (observability-policy §8.4)."
                ),
                "source_path": "console/index.json",
                "last_updated_utc": ctx.started.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )

    run_job(
        "console",
        body,
        store=store,
        trading_day=args.trading_day,
        dry_run=bool(getattr(args, "dry_run", False)),
        run_mode=getattr(args, "run_mode", None),
    )
    return 0
