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
from typing import Any

from crucible import alerts, release
from crucible.console.render import build_page, write_page
from crucible.drift import drift_metrics
from crucible.runner import RunContext, run_job, spot_interruption_guard
from crucible.store import Store, open_store

__all__ = [
    "console_handler",
    "drift_handler",
    "heartbeat_handler",
    "release_pin_handler",
    "smoke_handler",
    "sweep_handler",
]

#: What `crucible smoke` actually reads. A real end-to-end read against the
#: live store, not a ping: §4.11 gates the pointer flip on this, and a gate
#: that only proves the process started would promote a build that cannot
#: reach its own data. Each entry is (label, key-or-prefix, required).
SMOKE_READS: tuple[tuple[str, str, bool], ...] = (
    ("release pointer", release.POINTER_KEY, False),
    ("components registry", "runs/", False),
)


def _store(args: argparse.Namespace) -> Store:
    return open_store(getattr(args, "store", None))


# ── release.pin ───────────────────────────────────────────────────────────


def release_pin_handler(args: argparse.Namespace) -> int:
    """Repoint `releases/current`, or pin the trader. §4.11 rollback.

    Runs through the runner like every other job, so a rollback at 3am leaves
    a manifest saying who moved what and when — which is the difference
    between an incident with a timeline and one reconstructed from memory.
    """
    store = _store(args)

    def body(ctx: RunContext) -> None:
        before = release.current_release(store)
        release.pin(store, args.sha, target=args.target)
        key = release.POINTER_KEY if args.target == "current" else release.TRADER_PIN_KEY
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

    run_job("release.pin", body, store=store, trading_day=args.trading_day)
    return 0


# ── smoke ─────────────────────────────────────────────────────────────────


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
    """
    store = _store(args)

    def body(ctx: RunContext) -> None:
        reachable = 0
        for label, key, required in SMOKE_READS:
            if key.endswith("/"):
                # A listing, not a get: the paginated list path is the one a
                # truncation bug hides in, so the smoke exercises it.
                found = sum(1 for _ in store.list_keys(key))
                ctx.record_rows(rows_in=ctx.rows_in + found, rows_out=ctx.rows_out)
                reachable += 1
                continue
            if store.exists(key):
                payload = store.get_bytes(key)
                ctx.record_input(key, payload, schema_version=release.RELEASE_SCHEMA_VERSION)
                reachable += 1
            elif required:
                raise RuntimeError(f"smoke: required artifact {label} missing at {key}")
        ctx.record_metric(
            {
                "name": "smoke_ok",
                "module": "crucible.track_c",
                "metric_type": "operational",
                "value": float(reachable),
                "unit": "read_paths",
                "n_floor": 0,
                "status": "OK",
                "status_reason": (
                    f"{reachable} of {len(SMOKE_READS)} live read paths reachable from "
                    f"release {args.release}."
                ),
                "source_path": "runs/smoke/{trading_day}/run.json",
                "last_updated_utc": ctx.started.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )

    with spot_interruption_guard():
        run_job(
            "smoke",
            body,
            store=store,
            trading_day=args.trading_day,
            release_sha=args.release,
            # A smoke that needed a retry is a smoke that told us something.
            # Retrying it would promote a build whose first attempt failed,
            # and the deploy would read a green manifest over an amber fact.
            transient_retry=False,
        )
    return 0


# ── alerts.sweep ──────────────────────────────────────────────────────────


def sweep_handler(args: argparse.Namespace) -> int:
    """Evaluate both page conditions, group by cause, page once per group."""
    store = _store(args)
    result: dict[str, Any] = {}

    def body(ctx: RunContext) -> None:
        outcome = alerts.sweep(store, sweep_run_id=ctx.run_id)
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

    run_job("alerts.sweep", body, store=store, trading_day=args.trading_day)
    print(json.dumps({k: v for k, v in result.items() if k != "metric"}, indent=2))
    return 0


def heartbeat_handler(args: argparse.Namespace) -> int:
    """The weekly proof that the alerting path itself is alive."""
    store = _store(args)

    def body(ctx: RunContext) -> None:
        summary = alerts.heartbeat(store)
        ctx.record_metric(summary["metric"])
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

    run_job("heartbeat", body, store=store, trading_day=args.trading_day)
    return 0


# ── drift ─────────────────────────────────────────────────────────────────


def drift_handler(args: argparse.Namespace) -> int:
    """Three drift MetricRecords per cycle, from the feature and prediction
    artifacts track A and track B write.

    **It raises when those artifacts are absent**, rather than emitting three
    rows of `UNREPORTED` and exiting 0. A drift job that runs successfully
    against no data is exactly the shape of a monitor that was green for
    months while measuring nothing, and this repository's whole reason for
    existing is that that happened.
    """
    store = _store(args)

    def body(ctx: RunContext) -> None:
        day = ctx.trading_day.isoformat()
        inputs = {
            "features": f"drift/{day}/input_features.json",
            "predictions": f"drift/{day}/input_predictions.json",
            "ic": f"drift/{day}/input_ic.json",
        }
        missing = [k for k, key in inputs.items() if not store.exists(key)]
        if missing:
            raise FileNotFoundError(
                f"drift inputs absent for {day}: {sorted(missing)}. Three rows of "
                "UNREPORTED and an exit code of 0 is what a monitor looks like when it "
                "has been measuring nothing for months."
            )
        payloads = {
            k: json.loads(store.get_bytes(key).decode("utf-8")) for k, key in inputs.items()
        }
        for key in inputs.values():
            ctx.record_input(key, store.get_bytes(key), schema_version="drift_input.v1")

        records = drift_metrics(
            trading_day=ctx.trading_day,
            feature_psi_by_name=payloads["features"]["psi_by_feature"],
            prediction_psi=payloads["predictions"]["psi"],
            ic_decay_by_horizon={int(h): v for h, v in payloads["ic"]["decay_by_horizon"].items()},
        )
        for record in records:
            ctx.record_metric(record)
        ctx.record_output(
            f"drift/{day}/metrics.json",
            json.dumps(records, indent=2, sort_keys=True).encode("utf-8"),
            schema_version="metric_record.v1",
        )

    run_job("drift", body, store=store, trading_day=args.trading_day)
    return 0


# ── console ───────────────────────────────────────────────────────────────


def console_handler(args: argparse.Namespace) -> int:
    """Render the static page and its JSON from the manifests."""
    store = _store(args)

    def body(ctx: RunContext) -> None:
        page = build_page(store, now=dt.datetime.now(dt.UTC))
        html_key, json_key = write_page(store, page)
        for key in (html_key, json_key):
            ctx.record_output(key, store.get_bytes(key), schema_version="console_page.v1")
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

    run_job("console", body, store=store, trading_day=args.trading_day)
    return 0
