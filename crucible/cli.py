"""`crucible <job>` — the one entry point.

Normative source: plan §4.1, §10.8.

    crucible <job> [--date YYYY-MM-DD] [--dry-run] [...job options]

Identical behaviour on a laptop, in a Lambda and on a spot instance: a job is
a Python function, and the scheduler only calls it. Control flow lives in
Python and is tested with pytest, not in a state machine — this is what
replaces the 465 states.

**argparse, not typer.** Typer is a nicer surface and a dependency we would
carry forever for thirteen subcommands with a handful of flags each. Fewer
things to maintain, and one less import on a Lambda cold start (principle 6).
The delta is hand-written `--help` text; the parser below carries it.

**Every job runs through `crucible.runner.run_job`.** That is not a
convention: it is where the manifest guarantee lives, and a job invoked
around it would produce no telemetry on the path where telemetry matters.

The handlers below raise `NotImplementedError` with the owning track named.
The dispatch table, the argument surface and the date resolution are real
today, so three tracks build against a fixed shape.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from crucible import __version__
from crucible.calendar import resolve_trading_day

__all__ = ["JOBS", "JobSpec", "build_parser", "main"]


@dataclass(frozen=True)
class JobSpec:
    """One CLI job: its name, its one-line help, and whether it is scheduled.

    ``scheduled`` is here rather than only in `components.yaml` so the two
    can be checked against each other. A scheduled job whose absence nothing
    watches is the blindness §4.6 exists to remove.
    """

    name: str
    help: str
    scheduled: bool


def _todo(job: str, track: str, note: str) -> Callable[[argparse.Namespace], int]:
    """A handler that refuses loudly, naming who owns it.

    Not a no-op returning 0. A stub that exits cleanly is indistinguishable
    from a job that ran and had nothing to do — which is the shape the whole
    plan exists to make unrepresentable.
    """

    def handler(args: argparse.Namespace) -> int:
        raise NotImplementedError(
            f"`crucible {job}` is not implemented yet — {track}, alpha-engine-config-I9757. {note}"
        )

    return handler


JOBS: dict[str, JobSpec] = {
    "data.daily": JobSpec("data.daily", "Compile one trading day of market data", True),
    "data.weekly": JobSpec("data.weekly", "Weekly data refresh and coverage pass", True),
    "data.heal": JobSpec("data.heal", "Repair a named gap in the store, idempotently", False),
    "experiment.new": JobSpec("experiment.new", "Register an immutable arm from a recipe", False),
    "experiment.run": JobSpec("experiment.run", "Score one arm for one trading day", True),
    "experiment.grade": JobSpec("experiment.grade", "Run one slot's arena cycle", True),
    "promote": JobSpec("promote", "Move a slot's champion pointer, evidence-gated", False),
    "report": JobSpec("report", "Reduce the week's manifests into the attribution table", True),
    "explain": JobSpec("explain", "Walk a run_id or verdict back to what produced it", False),
    "migrate.history": JobSpec(
        "migrate.history", "Import v1 arm history with its provenance", False
    ),
    "release.pin": JobSpec("release.pin", "Repoint a release, or pin the trader to one", False),
    "smoke": JobSpec("smoke", "A real end-to-end run that gates a release flip", False),
}

HANDLERS: dict[str, Callable[[argparse.Namespace], int]] = {
    "data.daily": _todo("data.daily", "track B", "Lifts the ingest core from nousergon-data."),
    "data.weekly": _todo("data.weekly", "track B", "Weekly refresh + coverage MetricRecords."),
    "data.heal": _todo(
        "data.heal",
        "track B",
        "Must record what it repaired in rows_in/rows_out/rows_rejected, not just log it.",
    ),
    "experiment.new": _todo(
        "experiment.new",
        "track A",
        "Arm id is the hash of its spec; an edited recipe is a NEW arm carrying `supersedes`.",
    ),
    "experiment.run": _todo(
        "experiment.run",
        "track A",
        "Seeded by crucible-research/scripts/run_experiment.py (PR784).",
    ),
    "experiment.grade": _todo(
        "experiment.grade",
        "track A",
        "Calls nousergon_lib.arena.engine.run_cycle; re-implementing §§3-6 is a defect.",
    ),
    "promote": _todo(
        "promote",
        "track A",
        "Pointer moves only on a confidence-sequence-supported lead, and never before "
        "promote_min_weeks paired weeks.",
    ),
    "report": _todo(
        "report",
        "track A",
        "Five MetricRecord rows: data, signal IC, prediction IC, portfolio alpha, execution.",
    ),
    "explain": _todo(
        "explain",
        "track A",
        "§10.8: walks the manifest lineage from a verdict to the arms, features, data "
        "snapshot, code sha, cost and LLM calls that produced it.",
    ),
    "migrate.history": _todo(
        "migrate.history",
        "track A",
        "Carries R's `operator_bootstrap` champion flag into v2 so the first "
        "evidence-won promotion is visible as such.",
    ),
    "release.pin": _todo(
        "release.pin",
        "track C",
        "S3 conditional PUT on releases/current; the trader pins separately and never "
        "follows current automatically.",
    ),
    "smoke": _todo(
        "smoke",
        "track C",
        "A REAL run against live S3/ArcticDB. status: ok is what flips releases/current.",
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crucible",
        description=(
            "The experiment harness. One command per job, identical on a laptop, in a "
            "Lambda and on a spot instance. Every job writes a run manifest, including "
            "the ones that fail."
        ),
    )
    parser.add_argument("--version", action="version", version=f"crucible {__version__}")
    subparsers = parser.add_subparsers(dest="job", metavar="<job>", required=True)

    for spec in JOBS.values():
        sub = subparsers.add_parser(spec.name, help=spec.help, description=spec.help)
        sub.add_argument(
            "--date",
            metavar="YYYY-MM-DD",
            help=(
                "Trading day to run for. Defaults to the last completed session — a "
                "Saturday run binds to Friday's close (§4.12). A non-trading day is "
                "REFUSED, not silently resolved."
            ),
        )
        sub.add_argument(
            "--dry-run",
            action="store_true",
            help="Resolve inputs and report what would be written; write nothing.",
        )
        sub.add_argument(
            "--store",
            metavar="URI",
            help="Store root: an s3://bucket/prefix URI or a local directory path.",
        )
        if spec.name in ("experiment.run", "experiment.grade", "promote", "experiment.new"):
            sub.add_argument("--slot", choices=["u", "r", "m", "s"], required=True)
        if spec.name in ("experiment.run", "experiment.new"):
            sub.add_argument("--arm", metavar="ARM_ID", required=True)
        if spec.name == "explain":
            sub.add_argument("target", metavar="RUN_ID|VERDICT_KEY")
        if spec.name == "release.pin":
            sub.add_argument("sha", metavar="RELEASE_SHA")
            sub.add_argument("--target", choices=["current", "trader"], default="current")
        if spec.name == "data.heal":
            sub.add_argument("--gap", required=True, help="The named gap to repair.")

    return parser


def resolve_date(raw: str | None, *, now: dt.datetime | None = None) -> dt.date:
    """`--date` to a trading day.

    An explicit date is parsed and returned as given; it is the runner that
    refuses a non-trading day, so the refusal happens in one place and a
    backfill cannot route around it by calling a different entry point.
    """
    if raw is None:
        return resolve_trading_day(now)
    try:
        return dt.date.fromisoformat(raw)
    except ValueError as exc:
        raise SystemExit(f"--date must be YYYY-MM-DD; got {raw!r} ({exc})") from exc


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code.

    Exceptions are NOT caught here. A job that failed must exit non-zero with
    its traceback intact — the manifest already records the cause, and a
    tidy error message that returns 0 is a degraded-SUCCEEDED by another name.
    """
    args = build_parser().parse_args(argv)
    args.trading_day = resolve_date(getattr(args, "date", None))
    return HANDLERS[args.job](args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
