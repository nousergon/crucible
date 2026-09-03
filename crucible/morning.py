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

**It is a POINTER, and the board page is the artifact.** Brian, 2026-09-03,
on the first delivered report: *"I don't find the report detailed enough. It
should at a minimum be formatted well, so if we need another way to deliver
this report (email, url, etc) let's do so."* Telegram caps a message at 4096
characters, so a message that tried to carry the detail would be truncated by
the transport from the TAIL — losing the link to everything it had left out.
The six sections below are the summary, each under a heading; the presigned
link in the last one opens `board/index.html`, which carries every row's
store key and stamp, every phase's per-clause state and reason, the
acceptance clause list and the §6.1 schedule (`alpha-engine-config-I9921`).

**What it says, exhaustively.** Six headed sections (:data:`SECTIONS`), and
no seventh:

1. Every phase gate's reading, quoted with the store it came from, the
   board's `generated_at`, and the crucible commit the board was rendered
   from — plan §6 rule 2, "a reading is always quoted WITH its store and
   commit". The commit comes from the BOARD's own run manifest, not from this
   process's checkout: this job may be running a newer build than the one
   that rendered the board it is reporting.
2. The plan §6.1 schedule — one line per milestone, read off the board's own
   `schedule:*` rows (`crucible.schedule.MILESTONES`, `alpha-engine-config-
   I9914`) so drift from the plan's dated milestones is visible on the SAME
   surface phase gates are, every morning, rather than discoverable only by
   holding the plan next to the board by hand. A past-due, not-met milestone
   begins its line with `OVERDUE`. Purely a REFLECTION of the board's own
   reading — this section renders exactly what `board.py` already computed
   and adds no measurement of its own.
3. Which rows MOVED since the previous trading day's board, old -> new. Not
   the absolute board: a board reading almost entirely PLANNED for weeks is
   correct and is also the thing people stop opening.
4. The acceptance count — the ONLY progress figure (§12 rule 3) — if any
   artifact carries it, and :data:`ACCEPTANCE_NOT_ON_ANY_ARTIFACT` when
   none does. It is never reconstructed from this process's own checkout:
   that would report the branch the job ran from as though it were `main`'s
   reading, which is a fabricated provenance rather than a missing one.
5. How many rows are UNMEASURED or UNMEASURABLE, so silence is visible on
   the same surface as the readings.
6. The one pending operator action, when the board's own producer exposes
   one, and :data:`NO_OPERATOR_ACTION` when it does not.

**Never a progress narrative.** No PR count, no commit count, no findings
count, no prose about how the build is going. §12 rule 3: those are not
progress, and putting them beside a real figure lends them its authority.
The schedule section above is exempt from that count-word rule by
construction — it renders dates and gate readings, never a PR/commit/findings
figure — but its lines still pass through `_withhold_progress` like every
other section, since it renders board free text and this module trusts no
free text unchecked.

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
    BOARD_HTML_KEY,
    acceptance_reading_key,
    board_key,
    manifest_key,
    morning_report_key,
)
from crucible.store import PRESIGN_MAX_S, LocalStore, S3Store, Store, open_store

__all__ = [
    "ACCEPTANCE_NOT_ON_ANY_ARTIFACT",
    "ACCEPTANCE_UNREADABLE",
    "BOARD_URL_CAVEAT",
    "BOARD_URL_EXPIRES_S",
    "BOARD_URL_UNAVAILABLE",
    "CLAUSE_SEPARATOR",
    "FORBIDDEN_PROGRESS_TOKENS",
    "BOARD_JOB",
    "DELIVERY_TZ",
    "MESSAGE_MAX_CHARS",
    "MORNING_JOB",
    "MorningInputs",
    "SECTIONS",
    "SILENT_STATES",
    "NO_OPERATOR_ACTION",
    "STALE_AFTER",
    "TRUNCATION_MARKER",
    "UndeliveredError",
    "deliver",
    "morning_handler",
    "read_inputs",
    "render_message",
    "run_report",
    "store_uri",
    "wire_length",
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

#: The six headed sections, in the order `alpha-engine-config-I9921` states
#: them. Declared as a tuple rather than written out once in the renderer so
#: `tests/test_morning.py` can assert the ORDER against this list instead of
#: against a copy of it — a test that restates the order it grades passes
#: whichever way the renderer drifts.
#:
#: **CAPS, not bold.** The issue asks for a bold heading per section, and
#: `krepis.telegram.send_message` (0.124.x) exposes no `parse_mode` argument:
#: it always sends Markdown v1 and escapes the body itself. Emitting `*bold*`
#: would work until a board detail interpolated an odd asterisk, at which
#: point Telegram drops the whole message and krepis retries it as plain text
#: with the asterisks visible. A heading that is legible in BOTH renderings is
#: worth more than one that is bold in the common case, so the headings are
#: capitalised and the formatting request is filed as
#: `alpha-engine-config-I9925` (add `parse_mode` to krepis' transport).
SECTIONS: tuple[str, ...] = (
    "LADDER",
    "SCHEDULE (PLAN §6.1)",
    "ACCEPTANCE",
    "MOVED SINCE {previous_day}",
    "SILENCE",
    "FULL BOARD",
)

#: Telegram's hard limit on one message. The renderer fits the message to
#: this itself rather than letting `krepis.telegram._truncate_for_telegram`
#: tail-trim it: that transport cuts the TAIL, and the tail of this message is
#: the link to everything it had to leave out.
MESSAGE_MAX_CHARS = 4096

#: Appended when anything was dropped to fit. Stated, never silent — a
#: message that quietly shortened itself is indistinguishable from a board
#: that had less to say.
TRUNCATION_MARKER = "(truncated — see full board)"

#: The presigned lifetime asked for: the SigV4 maximum, seven days
#: (`crucible.store.PRESIGN_MAX_S`). A link that outlives the weekend is the
#: whole point — a report read on Monday whose link died on Saturday is a
#: report with no board.
BOARD_URL_EXPIRES_S = PRESIGN_MAX_S

#: Rendered beside the URL, every time. **A presigned URL signed with
#: temporary credentials dies with the credential**, and every crucible job
#: runs as an assumed role, so the seven days above is an upper bound the
#: signing session may not reach. Quoting "expires in 7 days" without this
#: would be a promise the credential cannot keep — and a reader who finds a
#: dead link and was told it had days left concludes the board is broken.
#: The durable fix is a page on the fleet console (`policy-console`), filed
#: as `alpha-engine-config-I9926`; a presigned URL is the surface that needs
#: no new identity, not the one that should exist forever.
BOARD_URL_CAVEAT = (
    "presigned GET, expires {expires} or when the signing role's session ends, whichever is first"
)

#: The line emitted when there is no page to link to. FAILURE MODE SWALLOWED:
#: `board/index.html` absent — the board render never published a page, or
#: published one under a held pointer. RECORDING SURFACE: this line, in the
#: delivered message, on the operator's phone. Raising instead would trade a
#: report with a missing link for no report at all, and the report is the
#: surface that would tell anybody the page is missing.
BOARD_URL_UNAVAILABLE = (
    f"no page at {BOARD_HTML_KEY} — the board render published none, so there is "
    "nothing to link to. That is a defect in the board job, not in this report."
)

#: The characters `krepis.telegram._escape_markdown` doubles on the wire.
#: Counted, not guessed: the renderer's 4096-character budget is spent on the
#: ESCAPED body, and a message that fits before escaping and not after is
#: tail-trimmed by the transport — losing the link this whole change adds.
_ESCAPED_CHARS = "\\_`[]"


def wire_length(text: str) -> int:
    """How many characters ``text`` occupies after krepis escapes it.

    `krepis.telegram.send_message` escapes the body for Markdown v1 AFTER the
    caller hands it over, so `len(message)` is not what Telegram measures.
    Computed from :data:`_ESCAPED_CHARS` rather than by calling krepis'
    private `_escape_markdown`: importing a private function would make this
    module's budget silently wrong the day krepis renames it, whereas a
    disagreement about WHICH characters are escaped shows up as a slightly
    conservative budget rather than a dropped message.
    """
    return len(text) + sum(text.count(c) for c in _ESCAPED_CHARS)


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
    #: A presigned GET on `board/index.html`, or `None` when there is no page
    #: to link to (`BOARD_URL_UNAVAILABLE` says so in the message). The
    #: message is a POINTER; this is what it points at.
    board_url: str | None
    #: When the link above stops working, as an upper bound — see
    #: :data:`BOARD_URL_CAVEAT` for why it is a bound and not a promise.
    board_url_expires: str
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


def _board_url(store: Store, *, now: dt.datetime) -> tuple[str | None, str]:
    """A presigned GET on the board page, and when it expires.

    FAILURE MODE SWALLOWED: `board/index.html` absent — `Store.presigned_url`
    raises `KeyError` rather than handing back a link to nothing, and this
    report must still go out, because it is the surface that would tell
    anybody the page is missing. RECORDING SURFACE:
    :data:`BOARD_URL_UNAVAILABLE`, rendered into the delivered message.

    Nothing else is swallowed. A credential failure, a malformed store URI or
    an out-of-range lifetime all propagate: those are defects in this job's
    own configuration, and a report that silently dropped its link over one
    would hide the defect behind a slightly shorter message.
    """
    expires = (now + dt.timedelta(seconds=BOARD_URL_EXPIRES_S)).astimezone(dt.UTC)
    stamp = expires.strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        return store.presigned_url(BOARD_HTML_KEY, BOARD_URL_EXPIRES_S), stamp
    except KeyError:
        return None, stamp


def read_inputs(
    store: Store, *, trading_day: dt.date, now: dt.datetime | None = None
) -> MorningInputs:
    """Read every artifact the message quotes. Reads; never runs.

    `board/current.json` is read WITHOUT a fallback: a report that could not
    read the board has nothing to say, and rendering a message about an
    absent board would be a surface reporting its own outage as the system's
    state.

    ``now`` dates the presigned link's expiry. Defaulted rather than required
    so an existing caller cannot silently get a WRONG expiry from a forgotten
    argument; `run_report` passes the run's own instant, which is what makes
    the rendered message deterministic in its inputs.
    """
    moment = (now or dt.datetime.now(dt.UTC)).astimezone(dt.UTC)
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
    board_url, board_url_expires = _board_url(store, now=moment)

    return MorningInputs(
        board=board,
        board_uri=store_uri(store),
        board_url=board_url,
        board_url_expires=board_url_expires,
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


#: One rendered line, and whether it may be dropped to fit Telegram's 4096.
#:
#: The tier is the whole truncation rule (`alpha-engine-config-I9921`): the
#: LADDER lines are tier 0 and are dropped last, clause-name detail is tier 2
#: and is dropped first. A single boolean would have made "drop the clause
#: names before the schedule" unexpressible, and the issue states that order.
_Line = tuple[str, int]

#: Tier 0 — never dropped while anything else remains. Headings, the ladder's
#: own lines, the acceptance count, the link.
_KEEP = 0
#: Tier 1 — dropped only after every tier-2 line is gone: schedule rows and
#: the moved-since list. Real readings, but readings the page carries in full.
_ROWS = 1
#: Tier 2 — dropped first: clause NAMES and clause ids. The issue names these
#: as the first thing to go, because the count above them survives and the
#: names are exactly what the full board exists to carry.
_NAMES = 2


def _phase_lines(board: dict[str, Any]) -> list[_Line]:
    """The ladder: one line per phase gate, in the board's own order.

    A phase row that vanished from the board is NOT silently absent: an empty
    ladder is a named defect rather than a blank section, so six phases
    becoming zero is visible without the reader knowing there should be six.

    `alpha-engine-config-I9921` asks for `phase0  UNMET  1/2` with the unmet
    clause NAMES beneath it. The fraction is computed from the row's own
    `clauses` list (`crucible.board.BoardRow.clauses`), never parsed back out
    of the English `detail` — a contract restated as a regex is the bug class
    this repository has already paid for twice. A board rendered by a producer
    that carried no clause list falls back to the detail sentence and says
    nothing it cannot show.
    """
    rows = [row for row in board.get("rows", []) if row.get("source") == "phase"]
    if not rows:
        return [
            (
                "  no phase row on the board — the ladder is not being rendered, which is a "
                "defect in the board, not a phase that has no gate",
                _KEEP,
            )
        ]
    lines: list[_Line] = []
    for row in rows:
        clauses = row.get("clauses")
        if isinstance(clauses, list) and clauses:
            met = sum(1 for c in clauses if c.get("met"))
            lines.append((f"  {row['id']}  {row['state']}  {met}/{len(clauses)}", _KEEP))
            unmet = [str(c.get("name", "?")) for c in clauses if not c.get("met")]
            if unmet:
                lines.append((f"    holding: {_withhold_progress(', '.join(unmet))}", _NAMES))
        elif isinstance(clauses, list):
            # Read, and it declares nothing. `0/0` would read as a measurement.
            lines.append(
                (
                    f"  {row['id']}  {row['state']}  no clauses — this gate measured nothing",
                    _KEEP,
                )
            )
        else:
            # No clause list on this board at all: a producer that took no
            # gate reading. The detail sentence is what there is, and
            # inventing a fraction from it would be the fabrication the
            # structured field exists to remove.
            lines.append(
                (f"  {row['id']}  {row['state']}  {_withhold_progress(row['detail'])}", _KEEP)
            )
        if row["state"] == "OUT_OF_ORDER":
            lines.append((f"    out of order: {_withhold_progress(row['detail'])}", _NAMES))
    return lines


def _schedule_lines(board: dict[str, Any]) -> list[_Line]:
    """One line per plan §6.1 milestone (`alpha-engine-config-I9914`).

    Reads `schedule:*` board rows exactly the way :func:`_phase_lines` reads
    `phase:*` rows — this function computes nothing; it quotes what
    `crucible.board._schedule_rows` already decided. A `state` of `UNMET`
    means the row's own `plan_date` has passed without the phase it names
    reading `MET`, so its line begins with the literal word `OVERDUE` — the
    board's own vocabulary has no `OVERDUE` state (`crucible.board.
    BOARD_STATES`), so this is the one place that word is rendered, and it is
    rendered from `UNMET`, never invented on a second condition.

    OVERDUE FIRST (`alpha-engine-config-I9921`): a milestone that has passed
    its date is the only line in this section anybody has to act on, and a
    section ordered by the board's declaration order buries it under the ones
    that are still ahead. The sort is STABLE, so within each group the
    board's own order survives — the plan's sequence is still readable.
    """
    rows = [row for row in board.get("rows", []) if row.get("source") == "schedule"]
    if not rows:
        return [
            (
                "  no schedule row on the board — the plan §6.1 milestones are not being "
                "rendered, which is a defect in the board, not an absent plan",
                _KEEP,
            )
        ]
    ordered = sorted(rows, key=lambda row: 0 if row["state"] == "UNMET" else 1)
    return [
        (
            f"  {'OVERDUE ' if row['state'] == 'UNMET' else ''}{row['id']}  {row['state']}  "
            f"{_withhold_progress(row['detail'])}",
            _ROWS,
        )
        for row in ordered
    ]


def _moved_lines(inputs: MorningInputs) -> list[_Line]:
    """What changed, old -> new. Never an absolute-state retelling.

    A previous board that could not be read yields the READ FAILURE, never
    "nothing moved": reporting no movement over a failed comparison is a
    positive claim asserted on no evidence, and it is exactly the shape
    `crucible.board._read_previous_board` exists to refuse one layer down.
    """
    if inputs.previous is None:
        return [
            (f"  cannot say — the {inputs.previous_day} board is {inputs.previous_reason}", _KEEP)
        ]
    before = {row["id"]: row["state"] for row in inputs.previous.get("rows", [])}
    after = {row["id"]: row["state"] for row in inputs.board.get("rows", [])}
    moved = [
        (f"  {row_id}: {before.get(row_id, 'ABSENT')} -> {after.get(row_id, 'VANISHED')}", _ROWS)
        for row_id in sorted(set(before) | set(after))
        if before.get(row_id) != after.get(row_id)
    ]
    return moved or [("  nothing moved", _KEEP)]


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
        f"of {met + unmet + unmeasurable} (commit {commit})"
    )


def _acceptance_lines(reading: dict[str, Any] | None, *, denied_code: str | None) -> list[_Line]:
    """The acceptance count, then the clause ids behind it.

    The COUNT is tier 0 — §12 rule 3 makes it the only progress figure, and
    it is the last thing this message gives up. The clause IDS are tier 2 and
    are the first thing dropped, because the full board carries them and the
    count above them stays true either way.

    The ids are read from the artifact's own optional `unmet_clauses` /
    `unmeasurable_clauses` lists (`crucible.keys.acceptance_reading_key`).
    When they are absent, this says so rather than going quiet: "the producer
    files no names" and "there are no unmet clauses" are different facts, and
    an empty section renders them identically.
    """
    lines: list[_Line] = [(f"  {_acceptance_line(reading, denied_code=denied_code)}", _KEEP)]
    if reading is None or denied_code is not None:
        return lines
    for field_name, label in (("unmet_clauses", "unmet"), ("unmeasurable_clauses", "unmeasurable")):
        named = reading.get(field_name)
        if isinstance(named, list) and named:
            joined = ", ".join(str(cid) for cid in named)
            lines.append((f"    {label}: {_withhold_progress(joined)}", _NAMES))
        else:
            lines.append((f"    the artifact names no {label} clause ids", _NAMES))
    return lines


def _silence_line(board: dict[str, Any]) -> str:
    counts = board.get("counts", {})
    parts = ", ".join(f"{counts.get(state, 0)} {state}" for state in SILENT_STATES)
    return f"silence: {parts} of {board.get('row_count', 0)} rows"


def _fit(lines: list[_Line]) -> str:
    """Join ``lines``, dropping the least important until Telegram will take it.

    **Truncation is a stated fact, never a silent shortening.** A message that
    quietly dropped half its content is indistinguishable from a board that
    had half as much to say, which is the same defect as a green row over no
    data one surface down.

    Dropped in tier order — clause names, then row lists, then whatever is
    left — and always from the END of the tier, so the earliest lines of each
    section survive. Nothing is dropped from a tier while a higher-numbered
    tier still has a line in it, which is what makes "never the ladder lines"
    a property of the algorithm rather than of the input.
    """
    if wire_length("\n".join(text for text, _ in lines)) <= MESSAGE_MAX_CHARS:
        return "\n".join(text for text, _ in lines)

    kept = list(lines)
    dropped = 0
    marker = f"\n\n{TRUNCATION_MARKER}"
    budget = MESSAGE_MAX_CHARS - wire_length(marker)
    for tier in (_NAMES, _ROWS, _KEEP):
        for index in range(len(kept) - 1, -1, -1):
            if wire_length("\n".join(text for text, _ in kept)) <= budget:
                break
            if kept[index][1] == tier:
                del kept[index]
                dropped += 1
    body = "\n".join(text for text, _ in kept)
    # The count is stated. "Something was cut" tells a reader to open the
    # board; "17 lines were cut" tells them how much of it they are missing.
    return f"{body}\n\n{TRUNCATION_MARKER} — {dropped} line(s) withheld"


def render_message(inputs: MorningInputs, *, now: dt.datetime) -> str:
    """The exact bytes delivered: a header, then :data:`SECTIONS`, in order.

    Deterministic in ``now`` and ``inputs`` alone, so the message a test
    asserts is the message an operator receives — a renderer that reached the
    clock or the store would be tested against something other than what
    ships.

    **The message is a POINTER; the board page is the artifact**
    (`alpha-engine-config-I9921`). Everything that will not fit in 4096
    characters is on the page, and the last section is the link to it.

    The `pending operator action` line is rendered under SILENCE rather than
    as a seventh section. It is not one of the six the issue lists and it was
    not there to be dropped: `alpha-engine-config-I9896` added it deliberately
    and it is the one line in this message naming something a human must do.
    Silence and an unactioned operator step are the same subject — nobody is
    looking — so it sits with the silence counts.
    """
    board = inputs.board
    local = now.astimezone(DELIVERY_TZ)
    commit = inputs.board_code_sha or f"UNKNOWN ({inputs.board_run_note})"
    lines: list[_Line] = []

    headline = _staleness(str(board.get("generated_at", "")), now)
    if headline:
        lines.append((headline, _KEEP))
        lines.append(("", _KEEP))

    lines.append((f"CRUCIBLE V2 — BOARD FOR TRADING DAY {board.get('trading_day')}", _KEEP))
    lines.append((f"store: {inputs.board_uri}/{BOARD_CURRENT_KEY}", _KEEP))
    lines.append((f"generated: {board.get('generated_at')}  commit: {commit}", _KEEP))
    lines.append((f"delivered: {local:%Y-%m-%d %H:%M %Z}", _KEEP))

    sections: list[list[_Line]] = [
        _phase_lines(board),
        _schedule_lines(board),
        _acceptance_lines(inputs.acceptance, denied_code=inputs.acceptance_denied_code),
        _moved_lines(inputs),
        [
            (f"  {_silence_line(board)}", _KEEP),
            (
                f"  pending operator action: {inputs.operator_action}"
                if inputs.operator_action
                else f"  {NO_OPERATOR_ACTION}",
                _KEEP,
            ),
        ],
        _board_lines(inputs),
    ]
    for heading, body in zip(SECTIONS, sections, strict=True):
        lines.append(("", _KEEP))
        lines.append((heading.format(previous_day=inputs.previous_day), _KEEP))
        lines.extend(body)
    return _fit(lines)


def _board_lines(inputs: MorningInputs) -> list[_Line]:
    """The link, and the honest bound on how long it lives."""
    if inputs.board_url is None:
        return [(f"  {BOARD_URL_UNAVAILABLE}", _KEEP)]
    return [
        (f"  {inputs.board_url}", _KEEP),
        (f"  {BOARD_URL_CAVEAT.format(expires=inputs.board_url_expires)}", _KEEP),
    ]


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
    return render_message(read_inputs(store, trading_day=trading_day, now=now), now=now)
