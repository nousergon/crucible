"""`crucible report.morning` — the 6am PT accountability report.

Normative source: `alpha-engine-config-I9896`; plan §4.6 (exactly two page
conditions), §6 (the three gate rules), §12 rule 3 (the acceptance count is
the only progress figure).

Brian stated on 2026-09-02 that a daily 6am PT accountability report existed.
It did not: the three claude.ai cloud routines were disabled, and the only
crucible schedule was `board.yml`, which renders the board into the store and
delivers it to nobody. A surface that emits nothing is unobserved, not
healthy (principle 7), and an accountability surface that exists only as a
belief is the worst case of that — it is *trusted* while emitting nothing.

**This job READS. It never renders and never runs anything.** It reads the
board `board.yml` already produced and reports it. Rendering here would give
the report a second, disagreeing answer to "what does the board say", and
running anything would break the rule `crucible/gate.py` states for gates and
`board.py` extends to the whole board: a surface that can produce the
artifacts it reports on can satisfy its own reading.

**It is not a page, and it must never become one.** Plan §4.6 admits exactly
two page conditions — absence and failure — and a daily digest is neither.
It goes out on the operator channel with `severity="info"` and `silent=True`,
on Telegram only, so nothing about it touches either SNS pages topic and it
cannot buzz a phone at 6am for a board that is red by design. `crucible.alerts`
is deliberately NOT imported: routing a digest through the page path is how a
page channel becomes the channel someone mutes.

**What it says, exhaustively.** Five things, and no sixth:

1. Every phase gate's reading, quoted with the store it came from, the
   board's `generated_at`, and the crucible commit the board was rendered
   from — plan §6 rule 2, "a reading is always quoted WITH its store and
   commit". The commit comes from the BOARD's own run manifest, not from this
   process's checkout: this job may be running a newer build than the one
   that rendered the board it is reporting.
2. Which rows MOVED since the previous trading day's board, old -> new. Not
   the absolute board: a board reading almost entirely PLANNED for weeks is
   correct and is also the thing people stop opening.
3. The acceptance count — the ONLY progress figure (§12 rule 3) — if any
   artifact carries it, and :data:`ACCEPTANCE_NOT_ON_ANY_ARTIFACT` when
   none does. It is never reconstructed from this process's own checkout:
   that would report the branch the job ran from as though it were `main`'s
   reading, which is a fabricated provenance rather than a missing one.
4. How many rows are UNMEASURED or UNMEASURABLE, so silence is visible on
   the same surface as the readings.
5. The one pending operator action, when the board's own producer exposes
   one, and :data:`NO_OPERATOR_ACTION` when it does not.

**Never a progress narrative.** No PR count, no commit count, no findings
count, no prose about how the build is going. §12 rule 3: those are not
progress, and putting them beside a real figure lends them its authority.

**A stale board is the HEADLINE, not a footnote.** If `board/current.json`
was generated more than one calendar day ago, the first line of the message
says so. Every reading below it is that old, and a reader who learns that at
the bottom has already acted on the readings.

**Delivery failure raises.** The whole deliverable is the delivery; a report
that was rendered and not sent is the silent-swallow shape the plan is a
reaction to. `run_job`'s `finally` still writes the manifest, so the failure
is `status: failed` with its cause and `alerts.sweep`'s failure condition
pages on it — which is the ONE legitimate page this job can produce.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from zoneinfo import ZoneInfo

from crucible.calendar import previous_trading_day
from crucible.keys import BOARD_CURRENT_KEY, board_key, manifest_key, morning_report_key
from crucible.store import LocalStore, S3Store, Store, open_store

__all__ = [
    "ACCEPTANCE_NOT_ON_ANY_ARTIFACT",
    "BOARD_JOB",
    "DELIVERY_TZ",
    "MORNING_JOB",
    "MorningInputs",
    "SILENT_STATES",
    "NO_OPERATOR_ACTION",
    "STALE_AFTER",
    "UndeliveredError",
    "deliver",
    "morning_handler",
    "read_inputs",
    "render_message",
    "run_report",
    "store_uri",
]

#: The CLI job name. One string, used by the handler, the registry row, the
#: schema enum and the manifest key, so a rename is one edit and three test
#: failures rather than four files silently disagreeing.
MORNING_JOB = "report.morning"

#: The job whose artifact this one reports. Named rather than inlined so the
#: manifest lookup and the prose cannot drift apart.
BOARD_JOB = "board"

#: The board states that mean "nobody can read this", as opposed to "this
#: reads no". Reported as their own figure: a board where half the rows
#: cannot be read at all is not the same board as one where half the rows say
#: no, and a single red count cannot tell them apart.
#:
#: Deliberately NOT named `GREY_STATES`. `crucible.board.GREY_STATES` already
#: owns that name for a DIFFERENT set — `PLANNED`/`DECLARED_OFF`, the rows
#: nobody has built yet — and two constants with one name and two meanings is
#: how a reader imports the wrong one and is never told.
SILENT_STATES: tuple[str, ...] = ("UNMEASURED", "UNMEASURABLE")

#: How old `board/current.json` may be before its age becomes the headline.
#: One calendar day, not one trading day: the board renders DAILY on a
#: calendar cron (`board.yml`), so a Saturday board is not late on Sunday for
#: any trading-calendar reason — it is simply a day old, and the reader needs
#: to know that whatever the market did.
STALE_AFTER = dt.timedelta(days=1)

#: Where 06:00 is measured. The cron is UTC (GitHub Actions has no other
#: option), so the LOCAL time the message actually lands at is computed and
#: printed rather than assumed — see `.github/workflows/morning-report.yml`
#: for the DST argument, and `tests/test_morning.py` for the assertion that
#: the committed cron lands at 06:00 PDT and 05:00 PST.
DELIVERY_TZ = ZoneInfo("America/Los_Angeles")

#: The literal line emitted when NO artifact carries the acceptance count.
#: A literal rather than a formatted string: `tests/test_morning.py` asserts
#: this exact text appears, and the PR body names the gap it records
#: (`tests/acceptance/ratchet.json` is a REPOSITORY property, committed to
#: git, and no store artifact republishes it). Saying "not on any artifact"
#: is a measurement; reading the number out of this process's own checkout
#: and printing it as `main`'s reading would be a fabrication that looks
#: exactly like a measurement.
ACCEPTANCE_NOT_ON_ANY_ARTIFACT = "acceptance count: not on any artifact"

#: The literal line emitted when the board's producer exposes no pending
#: operator action. Declared, never omitted: an absent line reads as "there
#: is nothing pending" and is indistinguishable from a line that failed to
#: render.
NO_OPERATOR_ACTION = "pending operator action: none exposed by the board's producer"


class UndeliveredError(RuntimeError):
    """The report was rendered and did not reach the operator.

    A distinct type so the manifest's `reason` names the failure rather than
    a transport's stringified internals, and so a caller cannot confuse it
    with a failure to READ the board — those two want different responses.
    """


def store_uri(store: Store) -> str:
    """The store this reading came from, as a string a human can act on.

    Plan §6 rule 2: a reading is always quoted WITH its store. The value is
    derived from the live store object rather than restated as a literal —
    `crucible/AGENTS.md` forbids a bucket name anywhere in this tree, and a
    literal would in any case name the store this code was WRITTEN against
    rather than the one it ran against, which is the opposite of provenance.
    """
    if isinstance(store, S3Store):
        return f"s3://{store.bucket}/{store.prefix}" if store.prefix else f"s3://{store.bucket}"
    if isinstance(store, LocalStore):
        return str(store.root)
    raise TypeError(
        f"{type(store).__name__} cannot name itself, so a reading taken from it could "
        "not be quoted with its store (plan §6 rule 2). A new Store backend adds a "
        "branch here; falling back to repr() would put an object address on the "
        "operator's phone."
    )


@dataclass(frozen=True)
class MorningInputs:
    """Everything the message is rendered from. Read once, in one place.

    Every field that can be absent is `None` AND carries a sibling reason
    string, because "absent" and "unreadable" want different sentences and
    are indistinguishable from a bare `None`.
    """

    board: dict[str, Any]
    board_uri: str
    previous: dict[str, Any] | None
    previous_reason: str
    previous_day: dt.date
    board_code_sha: str | None
    board_run_note: str
    operator_action: str | None


def _read_json(store: Store, key: str) -> tuple[dict[str, Any] | None, str]:
    """``(document, reason)``. Absent and unreadable are DIFFERENT answers.

    FAILURE MODE SWALLOWED: a malformed document is reported as a reason
    string rather than raised, because a corrupt PREVIOUS board must not stop
    today's report from going out — the report is the thing that would tell
    somebody the board is corrupt. RECORDING SURFACE: the returned reason is
    rendered verbatim into the delivered message and into the run manifest's
    `metrics`, so the swallow is visible on the operator's phone rather than
    only in a log. `board/current.json` itself is NOT read through this path:
    :func:`read_inputs` calls `store.get_bytes` directly for it, so a missing
    or corrupt CURRENT board raises and the manifest reads `failed`.
    """
    try:
        raw = store.get_bytes(key)
    except KeyError:
        return None, f"absent at {key}"
    try:
        return json.loads(raw), ""
    except ValueError as exc:
        return None, f"unreadable at {key}: {exc}"


def read_inputs(store: Store, *, trading_day: dt.date) -> MorningInputs:
    """Read every artifact the message quotes. Reads; never runs.

    `board/current.json` is read WITHOUT a fallback: a report that could not
    read the board has nothing to say, and rendering a message about an
    absent board would be a surface reporting its own outage as the system's
    state.
    """
    board = json.loads(store.get_bytes(BOARD_CURRENT_KEY))
    board_day = str(board.get("trading_day") or trading_day.isoformat())

    previous_day = previous_trading_day(dt.date.fromisoformat(board_day))
    previous, previous_reason = _read_json(store, board_key(previous_day.isoformat()))

    run, run_reason = _read_json(store, manifest_key(BOARD_JOB, board_day))
    code_sha: str | None = None
    operator_action: str | None = None
    if run is None:
        note = f"no board run manifest ({run_reason})"
        operator_action = (
            f"the {BOARD_JOB} render filed no manifest for {board_day} "
            f"({manifest_key(BOARD_JOB, board_day)}) — the daily render is not completing, "
            "so the readings quoted here are from whichever earlier render last succeeded"
        )
    else:
        code_sha = run.get("code_sha") or None
        note = "" if code_sha else "the board run manifest carries no code_sha"
        if run.get("status") == "failed":
            operator_action = (
                f"the {BOARD_JOB} render for {board_day} failed: "
                f"{run.get('reason') or 'no reason recorded'}"
            )

    return MorningInputs(
        board=board,
        board_uri=store_uri(store),
        previous=previous,
        previous_reason=previous_reason,
        previous_day=previous_day,
        board_code_sha=code_sha,
        board_run_note=note,
        operator_action=operator_action,
    )


def _staleness(generated_at: str, now: dt.datetime) -> str | None:
    """The headline, or `None`. A stale board is never a footnote."""
    try:
        generated = dt.datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    except ValueError:
        return (
            f"STALE BOARD: generated_at={generated_at!r} is not a timestamp, so the age "
            "of every reading below is unknown."
        )
    if generated.tzinfo is None:
        generated = generated.replace(tzinfo=dt.UTC)
    age = now - generated
    if age <= STALE_AFTER:
        return None
    hours = age.total_seconds() / 3600
    return (
        f"STALE BOARD: board/current.json was generated {generated_at} — {hours:.0f}h ago. "
        "Every reading below is that old."
    )


def _phase_lines(board: dict[str, Any]) -> list[str]:
    """One line per phase gate, in the board's own order.

    A phase row that vanished from the board is NOT silently absent: the
    section header carries the count, so six phases becoming five is visible
    without the reader knowing there should be six.
    """
    rows = [row for row in board.get("rows", []) if row.get("source") == "phase"]
    if not rows:
        return [
            "  no phase row on the board — the ladder is not being rendered, which is a "
            "defect in the board, not a phase that has no gate"
        ]
    return [f"  {row['id']}  {row['state']}  {row['detail']}" for row in rows]


def _moved_lines(inputs: MorningInputs) -> list[str]:
    """What changed, old -> new. Never an absolute-state retelling.

    A previous board that could not be read yields the READ FAILURE, never
    "nothing moved": reporting no movement over a failed comparison is a
    positive claim asserted on no evidence, and it is exactly the shape
    `crucible.board._read_previous_board` exists to refuse one layer down.
    """
    if inputs.previous is None:
        return [f"  cannot say — the {inputs.previous_day} board is {inputs.previous_reason}"]
    before = {row["id"]: row["state"] for row in inputs.previous.get("rows", [])}
    after = {row["id"]: row["state"] for row in inputs.board.get("rows", [])}
    moved = [
        f"  {row_id}: {before.get(row_id, 'ABSENT')} -> {after.get(row_id, 'VANISHED')}"
        for row_id in sorted(set(before) | set(after))
        if before.get(row_id) != after.get(row_id)
    ]
    return moved or ["  nothing moved"]


def _silence_line(board: dict[str, Any]) -> str:
    counts = board.get("counts", {})
    parts = ", ".join(f"{counts.get(state, 0)} {state}" for state in SILENT_STATES)
    return f"silence: {parts} of {board.get('row_count', 0)} rows"


def render_message(inputs: MorningInputs, *, now: dt.datetime) -> str:
    """The exact bytes delivered. Five sections, and no sixth.

    Deterministic in ``now`` and ``inputs`` alone, so the message a test
    asserts is the message an operator receives — a renderer that reached the
    clock or the store would be tested against something other than what
    ships.
    """
    board = inputs.board
    local = now.astimezone(DELIVERY_TZ)
    commit = inputs.board_code_sha or f"UNKNOWN ({inputs.board_run_note})"
    lines: list[str] = []

    headline = _staleness(str(board.get("generated_at", "")), now)
    if headline:
        lines.append(headline)
        lines.append("")

    lines.append(f"crucible v2 — board for trading day {board.get('trading_day')}")
    lines.append(f"store: {inputs.board_uri}/{BOARD_CURRENT_KEY}")
    lines.append(f"generated: {board.get('generated_at')}  commit: {commit}")
    lines.append(f"delivered: {local:%Y-%m-%d %H:%M %Z}")
    lines.append("")
    lines.append("phase gates")
    lines.extend(_phase_lines(board))
    lines.append("")
    lines.append(f"moved since {inputs.previous_day}")
    lines.extend(_moved_lines(inputs))
    lines.append("")
    lines.append(ACCEPTANCE_NOT_ON_ANY_ARTIFACT)
    lines.append(_silence_line(board))
    lines.append(
        f"pending operator action: {inputs.operator_action}"
        if inputs.operator_action
        else NO_OPERATOR_ACTION
    )
    return "\n".join(lines)


def _krepis_publish(*args: Any, **kwargs: Any) -> Any:
    """The real transport, imported at call time.

    Lazy for the same reason `crucible.alerts` does it: `crucible --help`,
    every unit test and every laptop run import this module, and none of them
    should pull an SNS/HTTP client onto the import path.
    """
    from krepis.alerts import publish  # noqa: PLC0415 - lazy on purpose

    return publish(*args, **kwargs)


def deliver(message: str, *, transport: Callable[..., Any] | None = None) -> str:
    """Send ``message`` on the operator channel. Returns the destination.

    **Telegram only, `severity="info"`, `silent=True`.** Plan §4.6 admits
    exactly two page conditions and this is neither, so it must not reach
    either SNS pages topic: `sns=False` is what keeps a daily digest out of
    the path a page travels, and `silent=True` is what keeps it from buzzing
    a phone at 6am about a board that is red by design. It is the same shape
    `nousergon-lib/.github/workflows/notify-ci-failure.yml` already runs
    fleet-wide, which is also why it needs no AWS credential to deliver.

    **No dedup key.** `krepis.alerts.publish` suppresses a repeat within its
    window, and this message is SUPPOSED to arrive every day even when it is
    byte-identical to yesterday's — a report that stops arriving when nothing
    changed is indistinguishable from a report that stopped arriving.

    **Undelivered raises.** `raise_on_total_failure=True` covers a transport
    that reached nothing; the explicit check below covers the case krepis
    calls a success — `any_ok` is True on a muted or dedup-suppressed publish
    by its documented contract, and neither of those put the report in
    Brian's hands.
    """
    publish = transport if transport is not None else _krepis_publish
    result = publish(
        message,
        severity="info",
        source=f"crucible-v2/{MORNING_JOB}",
        sns=False,
        telegram=True,
        silent=True,
        dedup_key=None,
        raise_on_total_failure=True,
    )
    if not hasattr(result, "any_ok"):
        raise TypeError(
            f"{type(result).__name__} carries no `any_ok`; a transport result that cannot "
            "say whether the report was delivered cannot be recorded as delivered."
        )
    suppressed = bool(getattr(result, "dedup_skipped", False)) or bool(
        getattr(result, "muted", False)
    )
    if not result.any_ok or suppressed:
        raise UndeliveredError(
            f"the {MORNING_JOB} report reached nobody "
            f"(any_ok={result.any_ok}, dedup_skipped={getattr(result, 'dedup_skipped', False)}, "
            f"muted={getattr(result, 'muted', False)}). The delivery IS the deliverable; a "
            "rendered report nobody received is the accountability gap this job closes."
        )
    return str(getattr(result, "telegram_destination", None) or "telegram")


def morning_handler(args: argparse.Namespace) -> int:
    """`crucible report.morning [--date] [--dry-run] [--store]`.

    Runs through `run_job` like every other job (AGENTS.md rule 1), so a
    morning that never reached Brian leaves a `failed` manifest naming why,
    and `alerts.sweep`'s two conditions cover this job on the same terms as
    every other: ABSENCE when the 13:00 UTC cron does not fire — GitHub drops
    scheduled events under load, measured on this fleet — and FAILURE when
    the delivery raises.

    **`--dry-run` renders and files nothing and sends nothing**, and prints
    the message to stdout instead. That is the shape a laptop needs to read
    the production board without writing to it, and it is honoured rather
    than ignored for the same reason `board` honours it: this job's outputs
    land under `runs/`, where a dry run would otherwise fabricate a manifest
    saying a report was delivered.
    """
    from crucible.runner import RunContext, run_job  # noqa: PLC0415 - lazy; see cli.py

    store = open_store(getattr(args, "store", None))
    dry_run = bool(getattr(args, "dry_run", False))
    rendered: list[str] = []

    def body(ctx: RunContext) -> None:
        now = ctx.started
        message = run_report(store, trading_day=ctx.trading_day, now=now)
        rendered.append(message)
        if dry_run:
            # No output, no delivery, and the manifest that `run_job` writes
            # regardless says `outputs: []` -- so a dry run is visibly a dry
            # run rather than one that claims a delivery it did not make.
            print(message)
            return
        destination = deliver(message)
        payload = message.encode("utf-8")
        artifact = morning_report_key(ctx.trading_day.isoformat(), ctx.calendar_date.isoformat())
        ctx.record_output(artifact, payload)
        ctx.record_metric(
            {
                "name": "morning_report_delivered",
                "module": "crucible.morning",
                "metric_type": "operational",
                "value": 1.0,
                "unit": "count",
                "n_floor": 1,
                # OK is the only value this line can ever carry, because the
                # alternative RAISES before it is reached. That is deliberate:
                # a metric that can record its own failure is a second status
                # channel beside the manifest's, and rule 2 admits no third
                # status anywhere.
                "status": "OK",
                "status_reason": f"delivered to {destination}",
                "source_path": artifact,
                "last_updated_utc": now.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )

    run_job(
        MORNING_JOB,
        body,
        store=store,
        trading_day=args.trading_day,
        # The FIRING, not the trading day. A 13:00 UTC cron fires every
        # calendar day and `resolve_trading_day` collapses Saturday, Sunday
        # and Monday onto Friday's close (§4.12), so without this the weekend
        # deliveries overwrite one another and the store's answer to "did the
        # report go out on Sunday" is decided by write ordering. Same shape,
        # same reason, as `alerts.sweep`.
        discriminator=lambda ctx: ctx.calendar_date.isoformat(),
    )
    return 0


def run_report(store: Store, *, trading_day: dt.date, now: dt.datetime) -> str:
    """Read the board and render the message. The whole job, minus delivery.

    Split out so the render is exercisable without argparse, without a
    runner and without a transport — and so `--dry-run` differs from a real
    run in exactly one branch of the handler rather than in a code path the
    tests cannot reach.
    """
    return render_message(read_inputs(store, trading_day=trading_day), now=now)
