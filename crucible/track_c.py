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
    LADDER_KEY,
    LADDER_SCHEMA_VERSION,
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
from crucible.release_lock_sweep import release_lock_findings, release_lock_metric
from crucible.runner import RunContext, run_job, spot_interruption_guard
from crucible.store import Store, open_store, sha256_hex

__all__ = [
    "console_handler",
    "drift_handler",
    "heartbeat_handler",
    "release_pin_handler",
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


def release_pin_handler(args: argparse.Namespace) -> int:
    """Repoint `releases/current`, or pin the trader. §4.11 rollback.

    Runs through the runner like every other job, so a rollback at 3am leaves
    a manifest saying who moved what and when — which is the difference
    between an incident with a timeline and one reconstructed from memory.
    """
    dry_run = bool(getattr(args, "dry_run", False))
    store = _store(args)

    def body(ctx: RunContext) -> None:
        before = release.current_release(store)
        key = release.POINTER_KEY if args.target == "current" else release.TRADER_PIN_KEY
        # alpha-engine-config-I9922 R2-1: the store guard is the backstop —
        # `release.pin` MOVES A POINTER, so under `--dry-run` it never calls
        # `release.pin()` at all (rather than calling it and letting the
        # guard refuse mid-write, which is the wrong shape for a pointer
        # move specifically: `release.pin`'s own internals, not just
        # `ctx.record_output`, are what write). The preview is exactly the
        # move it would make.
        if dry_run:
            print(
                f"release.pin --target {args.target}: {before or '(unset)'} would move to "
                f"{args.sha}"
            )
            return
        release.pin(store, args.sha, target=args.target)
        ctx.record_output(key, store.get_bytes(key), schema_version=release.RELEASE_SCHEMA_VERSION)
        ctx.record_metric(
            {
                "name": "release_pointer_moved",
                "module": "crucible.release",
                "metric_type": "operational",
                "value": 1.0,
                "unit": "pointer_moves",
                "n_floor": 0,
                "status": "OK",
                "status_reason": (f"{args.target} moved from {before or '(unset)'} to {args.sha}."),
                "source_path": key,
                "last_updated_utc": ctx.started.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )

    run_job(
        "release.pin",
        body,
        store=store,
        trading_day=args.trading_day,
        dry_run=bool(getattr(args, "dry_run", False)),
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
    return [meta_k, wheel_k]


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
                pointed = load_document_bytes(key, payload)["sha"]
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

        attempted = 2 + len(SMOKE_READS)  # the two release artifacts, plus the ambient paths
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
                    f"byte-for-byte." + (f" Degraded: {'; '.join(degraded)}." if degraded else "")
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
    """Evaluate both page conditions, group by cause, page once per group."""
    dry_run = bool(getattr(args, "dry_run", False))
    store = _store(args)
    result: dict[str, Any] = {}

    def body(ctx: RunContext) -> None:
        # `alerts.sweep(dry_run=)` evaluates and groups exactly as a real
        # sweep would but never calls `emit` — `outcome["bus_keys"]` is `()`
        # on this path, so the loop below writes nothing without needing its
        # own guard (alpha-engine-config-I9922 R2-1).
        outcome = alerts.sweep(store, sweep_run_id=ctx.run_id, dry_run=dry_run)
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
        # ladder artifact joined the page and its JSON, and a two-name unpack
        # would have failed the console job the moment it did.
        #
        # Each key gets ITS OWN schema version rather than one blanket stamp —
        # `gates/ladder.json` speaks `phase_ladder.v1`, not `console_page.v1`;
        # stamping every `write_page` key the same version is what let the
        # manifest lineage entry for the ladder disagree with the bytes it
        # described (`alpha-engine-config-I9825`).
        key_schema_versions = {
            CONSOLE_KEY: "console_page.v1",
            CONSOLE_JSON_KEY: "console_page.v1",
            LADDER_KEY: LADDER_SCHEMA_VERSION,
        }
        # alpha-engine-config-I9922 R2-1: the store guard is the backstop —
        # `console` has a natural report (the page's own JSON), printed below
        # under `--dry-run` rather than reached only by dying on the guard
        # inside `write_page` (which calls `store.put_bytes` directly, not
        # through `ctx`).
        if dry_run:
            print(page.to_json().decode("utf-8"))
        else:
            for key in write_page(store, page):
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
