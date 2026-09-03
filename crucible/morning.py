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
It goes out on the operator channel with `severity="info"` and `silent=False`,
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
from crucible.keys import (
    BOARD_CURRENT_KEY,
    acceptance_reading_key,
    board_key,
    manifest_key,
    morning_report_key,
)
from crucible.store import LocalStore, S3Store, Store, open_store

__all__ = [
    "ACCEPTANCE_NOT_ON_ANY_ARTIFACT",
    "ACCEPTANCE_UNREADABLE",
    "CLAUSE_SEPARATOR",
    "FORBIDDEN_PROGRESS_TOKENS",
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

#: The line emitted when the acceptance read failed because it could not be
#: REACHED, not because it is absent — measured live: the first
#: `report.morning` run (33766008781, 2026-09-03T14:18Z) died with
#: `AccessDenied` on `s3:ListBucket` reading this OPTIONAL artifact, because
#: S3 returns 403 for a missing key when the caller also lacks
#: `s3:ListBucket` on the prefix. Rendering that as
#: :data:`ACCEPTANCE_NOT_ON_ANY_ARTIFACT` would be the exact conflation plan
#: §6 rule 1 forbids: "absent" and "we are not permitted to look" are
#: different facts, and only one of them is an IAM gap somebody needs to
#: hear about. A `{code}` template rather than a second literal, so the
#: reader sees WHICH failure (`AccessDenied` today; a throttle or a network
#: failure would name itself here too) instead of a generic "broken".
ACCEPTANCE_UNREADABLE = "acceptance count: unreadable ({code})"

#: The literal line emitted when the board's producer exposes no pending
#: operator action. Declared, never omitted: an absent line reads as "there
#: is nothing pending" and is indistinguishable from a line that failed to
#: render.
NO_OPERATOR_ACTION = "pending operator action: none exposed by the board's producer"


#: How the board separates the clauses of one row's `detail`. Read off the
#: live board 2026-09-02: "1/2 clauses met; holding: ...; grades 2 of 5 ...".
#: Declared here rather than inlined so :func:`_withhold_progress` has one
#: definition of what it is splitting, and so a board that changed separator
#: fails the scrubber's own tests rather than silently withholding whole rows.
CLAUSE_SEPARATOR = "; "

#: Phrasings plan §12 rule 3 forbids from a status report: "the only progress
#: number is the acceptance count; PRs merged / findings / commits are NOT
#: progress and do not appear in a status report."
#:
#: **This is a guard over CONTENT, not over the template.** `row["detail"]`
#: is the board's free text and is rendered verbatim, so a board detail
#: reading "1/2 clauses met; 3 of 5 PRs merged" ships a forbidden figure onto
#: the one surface the rule exists for. A test that greps the rendered
#: message over a fixture whose details are placeholders grades the template
#: and passes while that happens.
#:
#: Deliberately a small closed set of PHRASES rather than a growing denylist
#: of everything anyone might write: it is the rule's own vocabulary, and a
#: clause it misses is a clause the rule's own wording does not name either.
#: The failure mode is a false NEGATIVE (a novel phrasing passes through),
#: never a false positive that silently deletes a real reading — every
#: withholding is declared on the line it happened.
FORBIDDEN_PROGRESS_TOKENS: tuple[str, ...] = (
    "pull request",
    " pr ",
    " prs ",
    "merged",
    "commits",
    "findings",
    "progress",
)

#: What replaces a withheld clause. Carries no forbidden token itself — a
#: marker that trips the guard it implements would make the whole-message
#: assertion unfalsifiable.
WITHHELD_MARKER = "[{n} clause{s} withheld — plan §12 rule 3]"


def _withhold_progress(detail: str) -> str:
    """Drop the clauses of ``detail`` that state a forbidden progress figure.

    **Withheld, never silently dropped.** The count of removed clauses is
    rendered in their place, so a reader can see that the board said
    something this surface refused to repeat and go read the board. A clause
    that vanished without a trace is indistinguishable from a board that
    never carried it, which is the defect this whole instrument exists to
    remove one layer down.

    **Clean text is passed through byte for byte.** A scrubber that rewrote
    detail it had no objection to would make the report an unfaithful copy of
    the board, which is worse than the defect it fixes.
    """
    clauses = detail.split(CLAUSE_SEPARATOR)
    kept = [c for c in clauses if not any(t in f" {c.lower()} " for t in FORBIDDEN_PROGRESS_TOKENS)]
    removed = len(clauses) - len(kept)
    if not removed:
        return detail
    marker = WITHHELD_MARKER.format(n=removed, s="" if removed == 1 else "s")
    return CLAUSE_SEPARATOR.join([*kept, marker]) if kept else marker


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
    #: §12 rule 3's one progress figure as FILED, or `None` when it is
    #: absent/corrupt OR denied — see `acceptance_denied_code` for telling
    #: those two apart. Never reconstructed from this process's own
    #: checkout: that would print the branch the job ran from as though it
    #: were `main`'s reading (`alpha-engine-config-I9902` builds the
    #: producer).
    acceptance: dict[str, Any] | None
    #: The botocore failure code (`AccessDenied`, ...) when the acceptance
    #: read failed because it could not be REACHED, or `None` when it was
    #: simply absent/corrupt/present. `alpha-engine-config-I9896`: an
    #: `AccessDenied` rendered through `ACCEPTANCE_NOT_ON_ANY_ARTIFACT`
    #: would report an IAM gap as "nobody has filed this yet".
    acceptance_denied_code: str | None


def _client_error_code(exc: BaseException) -> str | None:
    """The service failure code for a botocore read failure, or `None`.

    `None` means "not a botocore failure at all" (`LocalStore` raises only
    `KeyError`/`ValueError`) — the caller re-raises in that case, so this
    stays a classifier and never widens what :func:`_read_json` swallows.

    Imported lazily, like every botocore reference in this tree
    (`crucible.store`'s own precedent): `crucible --help` and every unit
    test that never touches S3 should not need the SDK on the import path.
    """
    from botocore.exceptions import BotoCoreError, ClientError  # noqa: PLC0415 - lazy on purpose

    if isinstance(exc, ClientError):
        return str(exc.response.get("Error", {}).get("Code", "")) or type(exc).__name__
    if isinstance(exc, BotoCoreError):
        return type(exc).__name__
    return None


@dataclass(frozen=True)
class _Read:
    """One optional-artifact read: present, absent/corrupt, or denied.

    Three outcomes because absent and denied are different facts about an
    artifact this job does not own and cannot write: reporting a denied read
    as absent hides an IAM gap behind a normal-looking report (plan §6 rule
    1's conflation, forbidden by name), and it is exactly what happened live
    on 2026-09-03 before this fix — S3 returns 403 for a missing key when
    the caller also lacks `s3:ListBucket` on the prefix.
    """

    document: dict[str, Any] | None
    reason: str
    denied_code: str | None


def _read_json(store: Store, key: str) -> _Read:
    """Read and parse ``key``. Never raises for absent, corrupt or denied.

    FAILURE MODES SWALLOWED: a missing key, a malformed document, and a
    botocore read failure that is not a not-found (`AccessDenied`, a
    throttle, a network failure before the request reached S3) are all
    reported rather than raised, because a corrupt or unreachable PREVIOUS
    board or acceptance reading must not stop today's report from going
    out — the report is the thing that would tell somebody either is
    broken. RECORDING SURFACE: `.reason` / `.denied_code` are rendered
    verbatim into the delivered message, so every swallow here is visible
    on the operator's phone rather than only in a log.

    A `NoSuchKey`/`404`-coded `ClientError` is still ABSENT, not denied —
    `S3Store.get_bytes` already normalizes that shape to `KeyError` in
    production, but this branch matches it too so a caller that raises the
    `ClientError` directly (a mock, or a future backend) degrades to the
    same honest answer rather than a spurious "denied".

    `board/current.json` itself is NOT read through this path:
    :func:`read_inputs` calls `store.get_bytes` directly for it, so a
    missing, corrupt, or denied CURRENT board raises and the manifest reads
    `failed` — that artifact is not optional.
    """
    try:
        raw = store.get_bytes(key)
    except KeyError:
        return _Read(None, f"absent at {key}", None)
    except Exception as exc:  # reclassified below; re-raised unless it's a named botocore code
        code = _client_error_code(exc)
        if code is None:
            raise
        if code in ("NoSuchKey", "404"):
            return _Read(None, f"absent at {key}", None)
        return _Read(None, f"unreadable at {key}: {code}", code)
    try:
        return _Read(json.loads(raw), "", None)
    except ValueError as exc:
        return _Read(None, f"unreadable at {key}: {exc}", None)


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
    previous_read = _read_json(store, board_key(previous_day.isoformat()))
    previous = previous_read.document
    # A DENIED previous board is still "cannot say", never "nothing moved" —
    # `.reason` already carries the right sentence for the absent/corrupt
    # cases, and the denied case gets its own, naming the code rather than
    # a generic "unreadable at <key>" (plan §6 rule 1).
    previous_reason = (
        previous_read.reason
        if previous_read.denied_code is None
        else f"unreadable ({previous_read.denied_code})"
    )

    run_read = _read_json(store, manifest_key(BOARD_JOB, board_day))
    run = run_read.document
    code_sha: str | None = None
    operator_action: str | None = None
    if run is None:
        run_note_reason = (
            f"unreadable ({run_read.denied_code})" if run_read.denied_code else run_read.reason
        )
        note = f"no board run manifest ({run_note_reason})"
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

    acceptance_read = _read_json(store, acceptance_reading_key(board_day))

    return MorningInputs(
        board=board,
        board_uri=store_uri(store),
        previous=previous,
        previous_reason=previous_reason,
        previous_day=previous_day,
        board_code_sha=code_sha,
        board_run_note=note,
        operator_action=operator_action,
        acceptance=acceptance_read.document,
        acceptance_denied_code=acceptance_read.denied_code,
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
    return [f"  {row['id']}  {row['state']}  {_withhold_progress(row['detail'])}" for row in rows]


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


def _acceptance_line(reading: dict[str, Any] | None, *, denied_code: str | None) -> str:
    """§12 rule 3's one progress figure — read, honestly absent, or denied.

    A reading missing any of the four fields the producer contract declares
    (`crucible.keys.acceptance_reading_key`) is treated as ABSENT, not as a
    partial reading: rendering three of four fields of the only number the
    plan calls progress would be a fabrication that looks exactly like a
    measurement, and the absent line is the honest alternative.

    ``denied_code`` takes priority over both: an access failure is a THIRD
    fact, never routed through the absent literal (`alpha-engine-config-I9896`
    measured `AccessDenied` on the first live run — see `ACCEPTANCE_UNREADABLE`).
    """
    if denied_code is not None:
        return ACCEPTANCE_UNREADABLE.format(code=denied_code)
    if reading is None:
        return ACCEPTANCE_NOT_ON_ANY_ARTIFACT
    try:
        met = int(reading["met"])
        unmet = int(reading["unmet"])
        unmeasurable = int(reading["unmeasurable"])
        commit = str(reading["commit"])
    except (KeyError, TypeError, ValueError):
        return ACCEPTANCE_NOT_ON_ANY_ARTIFACT
    return (
        f"acceptance count: {met} met / {unmet} unmet / {unmeasurable} unmeasurable "
        f"(commit {commit})"
    )


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
    lines.append(_acceptance_line(inputs.acceptance, denied_code=inputs.acceptance_denied_code))
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


def _operator_chat() -> str:
    """krepis' own name for the incident channel. Imported at call time.

    Lazy for the same reason :func:`_krepis_publish` is, and read from krepis
    rather than restated as `"operator_chat"` here: a literal would keep
    passing this module's tests on the day krepis renamed the destination,
    and a routing value that no longer names anything falls back to whatever
    the resolver decides — which is the failure this argument exists to
    prevent.
    """
    from krepis.alerts import DESTINATION_OPERATOR_CHAT  # noqa: PLC0415 - lazy on purpose

    return DESTINATION_OPERATOR_CHAT


def deliver(message: str, *, transport: Callable[..., Any] | None = None) -> str:
    """Send ``message`` on the operator channel. Returns the destination.

    **Telegram only, `severity="info"`, `silent=False`.** Plan §4.6 admits
    exactly two page conditions and this is neither, so it must not reach
    either SNS pages topic: `sns=False` is what keeps a daily digest out of
    the path a page travels. §4.6 governs PAGES; a Telegram notification is
    not one. It is the same shape
    `nousergon-lib/.github/workflows/notify-ci-failure.yml` already runs
    fleet-wide, which is also why it needs no AWS credential to deliver.

    **Why it notifies (Brian ruling 2026-09-03, `alpha-engine-config-I9916`).**
    The first delivery (2026-09-03 07:26 PDT) went out `silent=True` into the
    operator chat, under the day's CI-failure alerts; the manifest read `ok`
    and Brian said he had not received it. A report the recipient does not
    see is the accountability gap this job exists to close, one layer down.
    One push per day at 06:00 PT is the accepted cost.

    **No dedup key.** `krepis.alerts.publish` suppresses a repeat within its
    window, and this message is SUPPOSED to arrive every day even when it is
    byte-identical to yesterday's — a report that stops arriving when nothing
    changed is indistinguishable from a report that stopped arriving.

    **The destination is EXPLICIT, not resolved.**
    `krepis.alerts.resolve_destination` routes a non-`error` severity to the
    LOG chat when `TELEGRAM_LOG_CHAT_ID` is configured, and reaches the
    operator chat only through its documented "no log chat configured"
    fallback. So this report lands on Brian's chat today by the absence of a
    fleet-wide setting, and the day someone configures one it would move off
    his chat with this job's manifest still reading `ok` — a delivery that
    silently changed audience is exactly the class of silent success this
    module refuses everywhere else. `DESTINATION_OPERATOR_CHAT` is passed by
    name, from krepis' own constant rather than a literal here, so the
    routing is a decision in this diff instead of a property of the
    environment.

    **Undelivered raises.** `raise_on_total_failure=True` covers a transport
    that reached nothing; the explicit check below covers the case krepis
    calls a success — `any_ok` is True on a muted or dedup-suppressed publish
    by its documented contract, and neither of those put the report in
    Brian's hands.

    **Known: a manifest write that fails AFTER a successful delivery re-sends
    on the retry**, because the send is not idempotent and `run_job`'s one
    declared transient retry re-runs the whole body — two identical reports
    on the operator's phone, never a missing one. Filed by the parent session;
    the failure direction is deliberate, since a delivery skipped to avoid a
    duplicate is the silence this job exists to end.
    """
    publish = transport if transport is not None else _krepis_publish
    result = publish(
        message,
        severity="info",
        source=f"crucible-v2/{MORNING_JOB}",
        sns=False,
        telegram=True,
        silent=False,
        dedup_key=None,
        destination=_operator_chat(),
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
