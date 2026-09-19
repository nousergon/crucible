"""The dispatch's terminal record — written by the box, from its own exit path.

`alpha-engine-config-I11050`. An ABSENCE page could say *no manifest*; it
could not say *why*, and it ended by telling a human to go and look at a box
EC2 had already purged (`-I11049`: the instance is gone from
`DescribeInstances` after ~1h, and the absence horizon is 3h). The evidence
was never missing — every dispatched instance shipped a complete CloudWatch
stream — but nothing joined the three surfaces that each knew the instance
id: the dispatch record, the stream name, and the page text. This module is
the join, and the party writing it is the only one that knows its own exit
code before the process ends.

**It is not a job.** It writes no run manifest, claims no `components.yaml`
row and takes no trading day; `crucible.cli.JOBS` is deliberately untouched.
The box's EXIT trap invokes it as `python -m crucible.dispatch_exit`, which
is why the entry point is `__main__` rather than a CLI subcommand — a
subcommand would have to be either a JOB (it writes no manifest) or a
NON_JOB_HANDLER (which means "a handler the dispatcher may dispatch", and
this is never dispatched).

**Fail loud, with one bounded exception, recorded here** (repo rule 5). The
writer RAISES on every failure — a malformed record is refused by
`DispatchExitDocument` before a byte is written, and a store failure
propagates. The exception is the `__main__` wrapper's exit status: it prints
the failure and exits non-zero **without** re-raising through the trap,
because this runs inside `_finish` after the job's own status has been
decided, and an exception escaping here would replace the job's exit code
with this writer's. The failure mode swallowed is "the exit record could not
be written"; the recording surface is the box console (which the same trap
ships to CloudWatch, and which the absence detector reads as the next rung of
its ladder) plus the absence page itself, which still fires because no exit
record is exactly the state that makes a page louder rather than quieter.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path
from typing import Any

from crucible.dispatch_argv import dispatch_target_trading_day
from crucible.keys import dispatch_exit_key, manifest_key
from crucible.models import (
    DISPATCH_EXIT_CLASS_RECLAIMED,
    DISPATCH_EXIT_CLASSES,
    DISPATCH_EXIT_CONSOLE_TAIL_MAX,
    DispatchExitDocument,
)
from crucible.store import Store, open_store

__all__ = [
    "build_exit_document",
    "classify_exit",
    "last_error_line",
    "main",
    "write_exit_record",
]


#: Substrings that mark a console line as the account of a failure, in TWO
#: tiers. The tiers exist because the box console ends with a wrapper line —
#: `crucible data.heal exited 1` — that names a failure and says nothing
#: about it, and a single-tier scan walking backwards returns that line every
#: time, burying the traceback two lines above it.
#:
#: Substring matching, not a regex over the whole console: these lines are
#: produced by four different layers (the harness, argparse, bash, the AWS
#: CLI) and the only thing they have in common is the vocabulary. A pattern
#: precise enough to parse one of them would silently match none of the
#: others, which is the failure mode that leaves a page saying nothing.
_STRONG_ERROR_MARKERS: tuple[str, ...] = (
    "error:",
    "Traceback (most recent call last)",
    "AccessDenied",
    "denied",
)

#: A raised type's own name, which is what a Python traceback's last line is
#: and what four of the six boxes measured on 2026-09-18 ended with
#: (`SpotInterruptionError`, `ArmPredictionsContractError`, `ArcStageFailed`,
#: `MissingArtifactError`). A substring list cannot express "a dotted
#: identifier ending Error/Exception/Failed" without enumerating every
#: exception the harness can raise, which is a list that goes stale the next
#: time one is added.
_EXCEPTION_NAME = re.compile(r"\b[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Failed)\b")

#: The fallback tier: a line that says something failed without naming what.
#: Reported only when no strong line exists anywhere in the tail.
_WEAK_ERROR_MARKERS: tuple[str, ...] = (
    "exited ",
    "failed",
    "refus",
    "reclaimed",
)


def classify_exit(
    exit_code: int, *, reclaimed: bool, job_started: bool, crucible_installed: bool = True
) -> str:
    """Which of :data:`crucible.models.DISPATCH_EXIT_CLASSES` this exit is.

    The precedence is the box trap's own, restated in Python so it is
    testable: a spot notice wins over everything except a clean exit (an
    attempt that finished before the reclaim took the box DID its work and
    wrote its manifest), then argparse's usage exit 2, then the bootstrap /
    job split.

    ``crucible_installed=False`` is the bootstrap shell's own call: a box
    that died before the wheel existed cannot reach this module at all, and
    the shell writes the same document with this class.
    """
    if exit_code == 0:
        return "ok"
    if not crucible_installed:
        return "bootstrap_failed"
    if reclaimed:
        return DISPATCH_EXIT_CLASS_RECLAIMED
    if exit_code == 2 and job_started:
        # argparse's usage exit: the dispatch was malformed and
        # `crucible.runner.run_job` was never entered. A different incident
        # class from a job that ran and failed — reporting the two the same
        # way cost ten hours once (`alpha-engine-config-I10134`).
        return "refused"
    if not job_started:
        return "bootstrap_failed"
    return "failed"


def last_error_line(console: str | None) -> str | None:
    """The last console line that NAMES a failure, or ``None``.

    Walked backwards because the interesting line is the last thing that went
    wrong, and strong tier first because the last line of a failing box is
    the wrapper's own `exited N`, which is true and useless.
    """
    if not console:
        return None
    lines = [row.strip() for row in console.splitlines() if row.strip()]
    for line in reversed(lines):
        lowered = line.lower()
        if _EXCEPTION_NAME.search(line) or any(
            marker in lowered for marker in _STRONG_ERROR_MARKERS
        ):
            return line[:500]
    for line in reversed(lines):
        lowered = line.lower()
        if any(marker in lowered for marker in _WEAK_ERROR_MARKERS):
            return line[:500]
    return None


def build_exit_document(
    *,
    dispatch_id: str,
    instance_id: str,
    job: str,
    argv: str,
    exit_code: int,
    exit_class: str,
    console: str | None,
    log_group: str | None,
    log_stream: str | None,
    expected_manifest_key: str | None,
    manifest_written: bool,
    attempts: list[dict[str, Any]] | None,
    max_attempts: int,
    finished_at: dt.datetime | None = None,
) -> DispatchExitDocument:
    """The validated `dispatch_exit.v1` document for one ended dispatch.

    ``redispatch_expected`` is DERIVED here rather than passed in, from the
    two facts that decide it — the class is the reclaimed one, and the ladder
    has room (`crucible.runner.MAX_ATTEMPTS`). That is the same predicate
    `crucible.runner` applies when it suppresses the manifest, and deriving
    it in one place is what stops the record and the suppression disagreeing
    about whether an attempt is owed (`alpha-engine-config-I11051`).
    """
    if exit_class not in DISPATCH_EXIT_CLASSES:
        raise ValueError(
            f"exit_class {exit_class!r} is not one of {DISPATCH_EXIT_CLASSES}; the set "
            "is closed and every consumer splits on it."
        )
    ladder_has_room = attempts is None or len(attempts) < max_attempts
    redispatch_expected = exit_class == DISPATCH_EXIT_CLASS_RECLAIMED and ladder_has_room
    next_id: str | None = None
    if redispatch_expected and attempts is not None:
        # The dispatcher's own naming, restated where the promise is made so
        # a detector can check ONE key instead of listing a prefix. A
        # re-dispatch is recorded at `{prior}-r{n+1}`.
        next_id = f"{dispatch_id}-r{len(attempts) + 1}"
    moment = (finished_at or dt.datetime.now(dt.UTC)).astimezone(dt.UTC)
    tail = console[-DISPATCH_EXIT_CONSOLE_TAIL_MAX:] if console else None
    return DispatchExitDocument(
        schema_version="dispatch_exit.v1",
        dispatch_id=dispatch_id,
        instance_id=instance_id,
        job=job,
        argv=argv,
        exit_code=exit_code,
        exit_class=exit_class,  # type: ignore[arg-type]
        last_error_line=last_error_line(console),
        console_tail=tail,
        log_group=log_group,
        log_stream=log_stream,
        expected_manifest_key=expected_manifest_key,
        manifest_written=manifest_written,
        redispatch_expected=redispatch_expected,
        next_attempt_dispatch_id=next_id,
        attempts=attempts,  # type: ignore[arg-type]
        finished_at_utc=moment.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def write_exit_record(store: Store, document: DispatchExitDocument) -> str:
    """Write ``document`` at :func:`crucible.keys.dispatch_exit_key`, return the key.

    The key is computed IN the write call, not bound to a local first, so
    `tests/test_store_writers_are_jobs.py` can see from the syntax tree that
    this module's only write lands in the dispatch namespace — which is the
    property that admits an entry point beside the CLI at all.
    """
    store.put_bytes(
        dispatch_exit_key(document.job, document.dispatch_id),
        json.dumps(document.model_dump(mode="json"), sort_keys=True).encode("utf-8"),
    )
    return dispatch_exit_key(document.job, document.dispatch_id)


def _parse_attempts(raw: str | None) -> list[dict[str, Any]] | None:
    if not raw:
        return None
    rows = json.loads(raw)
    if not isinstance(rows, list):
        raise ValueError(f"attempts must be a JSON list, got {type(rows).__name__}")
    return rows


def _manifest_present(store: Store, key: str) -> bool:
    try:
        store.get_bytes(key)
    except KeyError:
        return False
    return True


def _resolve_expected_manifest(store: Store, args: argparse.Namespace) -> tuple[str | None, bool]:
    """``(expected_manifest_key, manifest_written)`` — a three-rung ladder.

    1. ``--expected-manifest-key``, whatever the caller was told.
    2. ``--expected-manifest-key-file``, the key the RUN itself bound
       (`crucible.runner.EXPECTED_MANIFEST_KEY_FILE`). Authoritative, and it
       wins over the flag: a second derivation is a second answer to "which
       manifest", which is `alpha-engine-config-I11048` exactly.
    3. Failing both, the SAME derivation from argv that
       `crucible.alerts` grades the absence with
       (`crucible.dispatch_argv.dispatch_target_trading_day`) — but claimed
       ONLY when a manifest is actually there under it.

    Rung 3 exists because rung 2 was measured to be a no-op on every live
    dispatch (`alpha-engine-config-I11050`, 2026-09-18): the box's bootstrap
    ASSIGNS ``CRUCIBLE_STATE_DIR`` without exporting it, so
    `crucible.runner._record_expected_manifest_key` — which reads it out of
    the environment — returned early and wrote no breadcrumb, on every job.
    A ladder whose only rung is one the substrate silently disables is not a
    ladder.

    **Rung 3 never ASSERTS a key.** A derived key that is not in the store is
    indistinguishable from a derivation that does not apply — a job writing
    one manifest per slot per day carries a discriminator this function
    cannot know — so an absent derived key yields ``(None, False)``: "the box
    could not say which manifest it owed", which is true, rather than a key
    it never bound, which would page a manifest-shaped hole that does not
    exist. Only rungs 1 and 2 can report a key that is missing, because only
    they were told one.
    """
    for key in (_recorded_manifest_key(args), args.expected_manifest_key or None):
        if key:
            return key, _manifest_present(store, key)
    derived = manifest_key(
        args.job,
        dispatch_target_trading_day(args.job, args.argv, dt.datetime.now(tz=dt.UTC)).isoformat(),
    )
    return (derived, True) if _manifest_present(store, derived) else (None, False)


def _recorded_manifest_key(args: argparse.Namespace) -> str | None:
    if not args.expected_manifest_key_file:
        return None
    recorded = Path(args.expected_manifest_key_file)
    if not recorded.exists():
        return None
    return recorded.read_text(encoding="utf-8").strip() or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m crucible.dispatch_exit",
        description="Write this dispatch's terminal record (dispatch_exit.v1).",
    )
    parser.add_argument("--dispatch-id", required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--argv", default="")
    parser.add_argument("--exit-code", required=True, type=int)
    parser.add_argument(
        "--reclaimed",
        action="store_true",
        help="the spot watcher saw an IMDS interruption notice for this box",
    )
    parser.add_argument(
        "--job-started",
        action="store_true",
        help="the `crucible` process was invoked (as opposed to dying in bootstrap)",
    )
    parser.add_argument(
        "--console",
        default=None,
        help="path to the box console log; its tail and last error line are recorded",
    )
    parser.add_argument("--log-group", default=None)
    parser.add_argument("--log-stream", default=None)
    parser.add_argument("--expected-manifest-key", default=None)
    parser.add_argument(
        "--expected-manifest-key-file",
        default=None,
        help=(
            "a file the RUN left the key it bound in "
            "(`crucible.runner.EXPECTED_MANIFEST_KEY_FILE`). Preferred over "
            "--expected-manifest-key: it is what the run actually bound, not a "
            "re-derivation from argv."
        ),
    )
    parser.add_argument("--attempts", default=None, help="CRUCIBLE_DISPATCH_ATTEMPTS, verbatim")
    parser.add_argument("--store", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    from crucible.runner import MAX_ATTEMPTS  # noqa: PLC0415 - avoids an import cycle

    args = build_parser().parse_args(argv)
    console = None
    if args.console:
        path = Path(args.console)
        if path.exists():
            # Bounded read: only the tail is recorded, and a box console can
            # be megabytes. `errors="replace"` because a killed process can
            # leave a partial UTF-8 sequence at the end of the file, and a
            # decode error here would lose the whole record over one byte.
            with path.open("rb") as handle:
                handle.seek(0, 2)
                size = handle.tell()
                handle.seek(max(0, size - DISPATCH_EXIT_CONSOLE_TAIL_MAX * 4))
                console = handle.read().decode("utf-8", errors="replace")
    store = open_store(args.store)
    expected, manifest_written = _resolve_expected_manifest(store, args)
    document = build_exit_document(
        dispatch_id=args.dispatch_id,
        instance_id=args.instance_id,
        job=args.job,
        argv=args.argv,
        exit_code=args.exit_code,
        exit_class=classify_exit(
            args.exit_code, reclaimed=args.reclaimed, job_started=args.job_started
        ),
        console=console,
        log_group=args.log_group,
        log_stream=args.log_stream,
        expected_manifest_key=expected,
        manifest_written=manifest_written,
        attempts=_parse_attempts(args.attempts),
        max_attempts=MAX_ATTEMPTS,
    )
    print(f"dispatch exit record: {write_exit_record(store, document)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - see the module docstring's rule-5 note
        print(f"dispatch exit record NOT written: {exc!r}", file=sys.stderr)
        raise SystemExit(1) from None
