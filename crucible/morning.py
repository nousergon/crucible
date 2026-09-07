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

**It is a POINTER, twice over now.** Brian, 2026-09-03, on the first
delivered report: *"I don't find the report detailed enough... if we need
another way to deliver this report (email, url, etc) let's do so."* The
six-section message that followed (`alpha-engine-config-I9921`) fixed
"detailed enough" and broke legible: Brian, 2026-09-06, on that message —
*"the telegram message is not legible, too much information and its not
formatted cleanly. can we instead have it link to a url that contains the
full update?"* (`alpha-engine-config-I10123`, superseding I9921's six-section
shape in the MESSAGE, not in what is said).

**So there are now three documents, not one.** :func:`render_full_update`
renders everything the old message rendered — the ladder, the §6.1 schedule,
moved-since, acceptance, silence, the board link, store key and stamps — as
GitHub-flavored Markdown, with no character budget: it is posted as a
comment on the rolling `[v2 board] daily update` issue in the private
`alpha-engine-config` tracker (`crucible.tracker`, App-minted token — an
Actions token cannot reach a different, private repository), one comment per
delivery, and its own filed copy sits at `update.md` beside the manifest.
:func:`render_history_body` regenerates that SAME issue's own BODY into a
newest-first table — one row per trading day, its six phase states, its
acceptance figure and a link to that day's comment (deliverable 7: the
issue is the history page, not only a stack of daily comments). And
:func:`render_message` renders the HEADLINE that actually reaches Telegram —
at most :data:`UPDATE_MESSAGE_MAX_CHARS` characters: a title with the trading
day, one line per phase (state and N/M only), the acceptance count, the
moved-since COUNT, the pending operator action when there is one, and THREE
links — "Full update" (today's comment), "History" (the issue itself) and
"Board" (the console, when configured). Everything the six sections used to
spell out in the message itself is now one click away.

**Ordering is the whole safety property.** The comment is posted BEFORE the
headline is rendered, because the headline's one indispensable line is the
comment's permalink — a headline sent before the comment exists is the
illegible shape again, just shorter. `crucible.tracker.post_comment` and
`crucible.morning._find_or_create_rolling_issue` both RAISE rather than
degrade, so a failed post fails the run loudly (`status: failed`) and the
Telegram message is never sent at all — see `morning_handler`.

**Never a progress narrative.** No PR count, no commit count, no findings
count, no prose about how the build is going. §12 rule 3: those are not
progress, and putting them beside a real figure lends them its authority.
Board free text still passes through the forbidden-token withholding in both
documents — the full update's audience is smaller, not looser.

**A stale board is the HEADLINE, not a footnote.** If `board/current.json`
was generated more than one calendar day ago, the first line of BOTH
documents says so. Every reading below it is that old, and a reader who
learns that at the bottom has already acted on the readings.

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
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from zoneinfo import ZoneInfo

from crucible import tracker
from crucible.calendar import previous_trading_day
from crucible.documents import load_store_document, read_document
from crucible.gate import PHASES, TRACKER_REPO
from crucible.keys import (
    BOARD_CURRENT_KEY,
    BOARD_HTML_KEY,
    TRIGGER_RE,
    TRIGGER_UNKNOWN,
    acceptance_reading_key,
    board_key,
    manifest_key,
    morning_history_row_key,
    morning_report_key,
    morning_trigger_key,
    morning_update_key,
    parse_acceptance_reading,
    runs_prefix,
)
from crucible.store import PRESIGN_MAX_S, LocalStore, S3Store, Store, open_store

__all__ = [
    "ACCEPTANCE_NOT_ON_ANY_ARTIFACT",
    "ACCEPTANCE_UNREADABLE",
    "BOARD_URL_CAVEAT",
    "BOARD_URL_EXPIRES_S",
    "BOARD_URL_UNAVAILABLE",
    "BOARD_URL_UNREADABLE",
    "CLAUSE_SEPARATOR",
    "FORBIDDEN_PROGRESS_TOKENS",
    "BOARD_JOB",
    "DELIVERY_CRON_UTC",
    "DELIVERY_CRON_UTC_HOUR",
    "DELIVERY_SEVERITY",
    "DELIVERY_SOURCE",
    "DELIVERY_TZ",
    "HEADLINE_TARGET_LINES",
    "HISTORY_ROW_BASENAME",
    "MORNING_JOB",
    "MorningInputs",
    "NO_OPERATOR_ACTION",
    "ROLLING_ISSUE_TITLE",
    "SILENT_STATES",
    "STALE_AFTER",
    "UPDATE_MESSAGE_MAX_CHARS",
    "UndeliveredError",
    "TRANSPORT_PREFIX",
    "deliver",
    "morning_handler",
    "read_inputs",
    "render_full_update",
    "render_history_body",
    "render_message",
    "resolve_trigger",
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

#: The severity and source :func:`deliver` publishes under. Named constants
#: rather than literals at the call site because they are not only routing —
#: `krepis.alerts.publish` renders BOTH into the prefix it prepends to the
#: body, so they are also two of the terms in this message's character budget
#: (:func:`transport_prefix`). A source string edited at the call site alone
#: would silently move the budget and the message would start being refused.
DELIVERY_SEVERITY = "info"
DELIVERY_SOURCE = f"crucible-v2/{MORNING_JOB}"

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

#: Where 03:00 is measured. The cron is UTC (GitHub Actions has no other
#: option), so the LOCAL time the message actually lands at is computed and
#: printed rather than assumed — see `.github/workflows/morning-report.yml`
#: for the DST argument, and `tests/test_morning.py` for the assertion that
#: the committed cron lands at 03:00 PDT and 02:00 PST.
DELIVERY_TZ = ZoneInfo("America/Los_Angeles")

#: The single declared instant this job's cron fires at, UTC. This is the
#: ONE source of truth for "when does report.morning start" — every other
#: place that used to carry the literal (the workflow's own `on.schedule`
#: cron, its header comment, `components.yaml`'s `report.morning` schedule
#: and deadline prose, and this module's own ABSENCE-condition docstring)
#: either quotes this constant's value in prose or is asserted equal to it
#: by `tests/test_morning.py`'s workflow-shape test, which parses
#: `.github/workflows/morning-report.yml` and reads its cron back against
#: this string — so the workflow file and this constant cannot silently
#: disagree.
#:
#: `0 10 * * *`, not `0 13 * * *` (alpha-engine-config-I9966, corrected
#: 2026-09-06): every scheduled workflow on this account fires 3-5h after
#: its declared cron (MEASURED — `authority-surface`/`llm-callsite-surface`/
#: `observability-registry`/`cloudwatch-alarm-drift`, see
#: `components.yaml`'s `report.morning.deadline` comment for the table), so
#: a cron declaring 06:00 PT was actually landing ~09:00-09:07 PT. Moving
#: the declared instant three hours earlier — 03:00 PDT / 02:00 PST — is
#: what makes the DELIVERED message land near the intended 06:00 PT; the
#: declared cron itself is no longer "the delivery time", only the input to
#: it.
DELIVERY_CRON_UTC = "0 10 * * *"

#: The UTC hour :data:`DELIVERY_CRON_UTC` names, parsed rather than
#: retyped — a second literal here could drift from the cron string above
#: the moment either one was edited alone.
DELIVERY_CRON_UTC_HOUR = int(DELIVERY_CRON_UTC.split()[1])

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

#: The title of the rolling issue `render_full_update`'s markdown is posted
#: to, once per delivery, in the private `alpha-engine-config` tracker
#: (`alpha-engine-config-I10123`). A literal, not derived from anything —
#: `crucible.tracker.find_issue_by_title` searches for it verbatim, and the
#: same string is the CREATE payload's title the one time a day ever needs
#: to create it. Never closed by automation; a human retires it if the shape
#: of the daily update ever changes enough to want a fresh rolling issue.
ROLLING_ISSUE_TITLE = "[v2 board] daily update"

#: Brian's stated cap on the HEADLINE (`alpha-engine-config-I10123`, ruling
#: 2026-09-06): "the telegram message is not legible, too much information."
#: Unlike the old :data:`MESSAGE_MAX_CHARS` this is not a transport limit to
#: fit — the headline's fixed shape (a title, one line per plan phase, three
#: more fixed lines, one links line) never comes close to Telegram's own
#: 4096-character ceiling, so there is no fitter here: :func:`render_message`
#: RAISES if an unbounded field (an operator action, most likely) would ever
#: cross this, since a headline that silently grew past
#: :data:`HEADLINE_TARGET_LINES` is the illegible shape reappearing quietly.
UPDATE_MESSAGE_MAX_CHARS = 1200

#: Brian's "at most ~12 short lines" restated as a number after the
#: deliverable-7 follow-up dropped the headline to a title, six phase lines,
#: three fixed lines and ONE links line — ten in the common case (no stale
#: headline, no pending operator action). `tests/test_morning.py` asserts
#: the common-case fixture against this exactly; the two optional lines
#: (stale, operator action) may push a genuinely exceptional morning past
#: it, which is a target on the ordinary case, not a second hard cap
#: alongside :data:`UPDATE_MESSAGE_MAX_CHARS`.
HEADLINE_TARGET_LINES = 10

#: Padding width for the phase state column in the headline
#: (`Phase 0  UNMET       4/5`) -- the longest of `crucible.board.
#: BOARD_STATES` (`OUT_OF_ORDER`/`DECLARED_OFF`, both 12). A literal rather
#: than importing `crucible.board` (a heavier module this one otherwise
#: never needs): `tests/test_morning.py` pins it against the live tuple so
#: the two cannot drift apart unnoticed.
_STATE_COLUMN_WIDTH = 12

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

#: The console pane the board's rows render on (`alpha-engine-config-I9926`):
#: every row of `board/current.json` is a Decision entity, so the Decision
#: list is the board, beside the phase ladder's own six rows. A path, not a
#: host — the host is `CRUCIBLE_CONSOLE_URL` (`crucible.config`), which this
#: tree carries no default for.
BOARD_CONSOLE_PATH = "/decision?pipeline=crucible-board"

#: Rendered beside the console link. It says what the presigned caveat could
#: not: the address does not expire. It also says what the page IS — the
#: console's Decision kind filtered to this board's rows, ALL of them — and
#: where those rows come from, so a reader comparing the two surfaces knows
#: they are one document. The filter is a facet the console fragment stamps
#: on every row as a literal (`pipeline: crucible-board`), not a field read
#: off the rows: the board's own `surface` is per-row PROVENANCE (measured
#: 2026-09-03: 10 distinct values across 46 rows), and filtering on it hid
#: 20 rows, 17 of them the UNREPORTED component rows the board exists to
#: show (I9926 review B3).
BOARD_CONSOLE_CAVEAT = (
    "fleet console — the Decision list filtered to this board, every row of "
    "board/current.json and none hidden; stable address, no expiry"
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

#: The line emitted when the page could not be REACHED, as opposed to being
#: absent. `S3Store.presigned_url` reaches absence through `exists()` →
#: `head_object`, which re-raises every non-404 `ClientError` — and
#: `alpha-engine-config-I9896` measured the shape that matters: S3 answers 403
#: for a MISSING key when the caller also lacks `s3:ListBucket` on the prefix,
#: which is precisely the case `board/index.html` sits in, since the page is
#: written only under `if may_move:` in `board_handler`. Rendering that as
#: :data:`BOARD_URL_UNAVAILABLE` would blame the board job for an IAM gap;
#: raising it would trade the whole report for a missing optional link. So it
#: is the third fact, named, exactly as `ACCEPTANCE_UNREADABLE` is.
#: FAILURE MODE SWALLOWED: a botocore read failure on `board/index.html`.
#: RECORDING SURFACE: this line, in the delivered message.
BOARD_URL_UNREADABLE = (
    f"the page at {BOARD_HTML_KEY} could not be read ({{code}}) — this is an access failure, "
    "not a missing page, so it is a gap in this job's permissions rather than a defect in "
    "the board job."
)


def wire_length(text: str) -> int:
    """How many characters ``text`` occupies on the wire, under `parse_mode="HTML"`.

    Under Markdown v1, `krepis.telegram.send_message` escaped the body AFTER
    the caller handed it over, so `len(message)` understated what Telegram
    measured. HTML mode inverts that (`alpha-engine-config-I9925`): the
    caller owns the markup and `send_message` escapes nothing for it — every
    piece of interpolated content already went through :func:`_escape_html`
    by the time it reaches a line, so what this module renders IS the wire
    body, and `len()` is exact rather than a lower bound. Kept as a named
    function, not inlined as `len()` at every call site, so the budget's unit
    of measure stays one declared thing if a future parse mode needs a
    different rule again.
    """
    return len(text)


#: What `krepis.alerts.publish` puts in FRONT of this body, unescaped.
#:
#: **The budget is spent on the wire body, and the body is not what this
#: module hands over.** `krepis.alerts.publish` calls its own
#: `_format_message(message, severity, source, state)` and passes the RESULT
#: to the Telegram transport, which truncates on `len()` and only then
#: escapes. So the message this module renders reaches Telegram with
#: ``[INFO] crucible-v2/report.morning: `` in front of it, and a budget
#: computed on the body alone is short by that much — measured on the
#: adversarial review's 60-phase fixture, a body of wire length 4117 plus a
#: 37-character escaped prefix was POSTed at 4152 and refused 400 *message is
#: too long*, which is not an entity-parse error, so krepis' plain-text retry
#: never fired and the report was not delivered at all.
#:
#: Composed from :data:`DELIVERY_SEVERITY` and :data:`DELIVERY_SOURCE` — the
#: same two values :func:`deliver` publishes under — rather than restated as a
#: 35-character literal, so the two cannot drift. krepis exposes no public
#: formatter (`_format_message` is private, and importing a private symbol
#: would make this budget silently wrong the day it is renamed), so
#: `tests/test_morning.py::TestTheWireBodyFitsTheCap::
#: test_the_prefix_matches_the_one_krepis_actually_prepends` PINS this against
#: krepis' real output by calling that formatter from the test — a
#: disagreement fails a test instead of dropping a report.
TRANSPORT_PREFIX = f"[{DELIVERY_SEVERITY.upper()}] {DELIVERY_SOURCE}: "


def _escape_html(text: str) -> str:
    """Escape ``& < >`` so ``text`` renders literally under `parse_mode="HTML"`.

    `krepis.telegram.escape_html` is the ONE escaper (`alpha-engine-config-
    I9925`): a second implementation here would let this module's idea of
    "escaped" drift from what `send_message` assumes the caller already did.
    Imported lazily, like every krepis reference in this tree
    (`_krepis_publish`'s precedent): `crucible --help` and every unit test
    that renders a message without sending one should not need `requests` —
    `krepis.telegram` pulls it in at module scope — on the import path.

    Every piece of board free text (a clause name, a `detail` sentence, a
    failure `reason`) and every URL this module interpolates goes through
    this before it reaches a line — an unescaped `<` in a board reason must
    not be able to break delivery of the whole report, which is exactly the
    HTML-entity-parse-error shape `send_message`'s plain-text retry exists
    to catch, but a message that never needed the retry is the one that
    reaches Brian on the first try.
    """
    from krepis.telegram import escape_html  # noqa: PLC0415 - lazy on purpose

    return escape_html(text)


def _filter_progress_clauses(detail: str) -> tuple[list[str], int]:
    """The clause-withholding rule's core: what plan §12 rule 3 keeps of
    ``detail``, and how many clauses it dropped.

    Extracted from the withholding functions themselves (`alpha-engine-
    config-I10123`) so the full update's markdown and the Telegram headline's
    HTML share ONE decision about which clauses are forbidden — the two
    callers below differ only in how they escape what is left, and a second
    copy of the filtering logic is how "withheld on the headline, present on
    the full update" becomes possible by accident.
    """
    clauses = detail.split(CLAUSE_SEPARATOR)
    kept = [c for c in clauses if not any(t in f" {c.lower()} " for t in FORBIDDEN_PROGRESS_TOKENS)]
    return kept, len(clauses) - len(kept)


def _withhold_progress(detail: str) -> str:
    """Drop the clauses of ``detail`` that state a forbidden progress figure,
    for the Telegram headline's HTML.

    **Withheld, never silently dropped.** The count of removed clauses is
    rendered in their place, so a reader can see that the board said
    something this surface refused to repeat and go read the board. A clause
    that vanished without a trace is indistinguishable from a board that
    never carried it, which is the defect this whole instrument exists to
    remove one layer down.

    **Clean text is passed through byte for byte, modulo HTML-escaping.** A
    scrubber that rewrote detail it had no objection to would make the report
    an unfaithful copy of the board, which is worse than the defect it fixes
    — :func:`_escape_html` is not that: it changes nothing Telegram would
    render differently, and everything it does change (`<`, `>`, `&`) is
    exactly what would otherwise risk the whole message under
    `parse_mode="HTML"` (`alpha-engine-config-I9925`).
    """
    kept, removed = _filter_progress_clauses(detail)
    if not removed:
        return _escape_html(detail)
    marker = WITHHELD_MARKER.format(n=removed, s="" if removed == 1 else "s")
    result = CLAUSE_SEPARATOR.join([*kept, marker]) if kept else marker
    return _escape_html(result)


def _plain_withhold(detail: str) -> str:
    """The same withholding rule, unescaped, for the full update's markdown.

    Plain text rather than HTML-escaped: the full update is a GitHub comment
    body, not a Telegram HTML payload, and running `detail` through
    :func:`_escape_html` here would print literal `&amp;` where the board
    said `&` — correct on the wire this module used to own, wrong on the one
    it posts to now.
    """
    kept, removed = _filter_progress_clauses(detail)
    if not removed:
        return detail
    marker = WITHHELD_MARKER.format(n=removed, s="" if removed == 1 else "s")
    return CLAUSE_SEPARATOR.join([*kept, marker]) if kept else marker


def _md_cell(text: str) -> str:
    """Make ``text`` safe as one cell of a GitHub Markdown table.

    A literal `|` ends the cell early and a literal newline ends the row —
    both are collapsed to a space/slash rather than escaped: GitHub's own
    renderer does not treat a backslash-escaped pipe reliably inside every
    surface that reads its REST-rendered comment body, and a stray `|` in
    board free text is rare enough that losing it is the smaller defect.
    """
    return text.replace("|", "/").replace("\n", " ")


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
    #: The botocore failure code when the PAGE read failed because it could
    #: not be reached, or `None` when the page was simply absent or present.
    #: The same three-way distinction `acceptance_denied_code` carries, for
    #: the same measured reason (`alpha-engine-config-I9896`).
    board_url_denied_code: str | None
    #: The botocore failure code (`AccessDenied`, ...) when the acceptance
    #: read failed because it could not be REACHED, or `None` when it was
    #: simply absent/corrupt/present. `alpha-engine-config-I9896`: an
    #: `AccessDenied` rendered through `ACCEPTANCE_NOT_ON_ANY_ARTIFACT`
    #: would report an IAM gap as "nobody has filed this yet".
    acceptance_denied_code: str | None
    #: The stable console address for the board (`alpha-engine-config-I9926`),
    #: or `None` when no console is configured — in which case the presigned
    #: page above is the link and its caveat is rendered. Never both: two links
    #: to one board is a reader deciding which to trust.
    board_console_url: str | None = None


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
    # The one guarded parser (`crucible.documents`): an array- or null-bodied
    # previous board is "unreadable", never a document the next line's `.get`
    # raises on (alpha-engine-config-I9931).
    parsed = read_document(key, lambda: raw)
    if parsed.problem is not None:
        return _Read(None, f"unreadable at {key}: {parsed.problem}", None)
    return _Read(parsed.document, "", None)


def _board_url(store: Store, *, now: dt.datetime) -> tuple[str | None, str, str | None]:
    """A presigned GET on the board page, when it expires, and why not.

    Three outcomes, like :func:`_read_json`, and for the same reason: the page
    is an OPTIONAL artifact this job does not write, and "it is not there" and
    "we are not permitted to look" want different responses from a reader.

    FAILURE MODE SWALLOWED: `board/index.html` absent — `Store.presigned_url`
    raises `KeyError` rather than handing back a link to nothing — AND a
    botocore read failure that is not a not-found, which
    `S3Store.presigned_url` reaches through `exists()` → `head_object` and
    re-raises. `alpha-engine-config-I9896` measured the second shape live: S3
    returns 403 for a missing key when the caller also lacks `s3:ListBucket`
    on the prefix. The report must still go out either way, because it is the
    surface that would tell anybody the page is missing. RECORDING SURFACE:
    :data:`BOARD_URL_UNAVAILABLE` and :data:`BOARD_URL_UNREADABLE`, rendered
    into the delivered message — and only ONE of them, so the reader is told
    which of the two facts happened.

    Nothing else is swallowed. A malformed store URI or an out-of-range
    lifetime raise `ValueError`/`TypeError` and propagate: those are defects
    in this job's own configuration, not in its access to somebody else's
    artifact, and a report that silently dropped its link over one would hide
    the defect behind a slightly shorter message.
    """
    expires = (now + dt.timedelta(seconds=BOARD_URL_EXPIRES_S)).astimezone(dt.UTC)
    stamp = expires.strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        return store.presigned_url(BOARD_HTML_KEY, BOARD_URL_EXPIRES_S), stamp, None
    except KeyError:
        return None, stamp, None
    except Exception as exc:  # reclassified below; re-raised unless it's a named botocore code
        code = _client_error_code(exc)
        if code is None:
            raise
        if code in ("NoSuchKey", "404"):
            return None, stamp, None
        return None, stamp, code


def read_inputs(
    store: Store,
    *,
    trading_day: dt.date,
    now: dt.datetime | None = None,
    console_url: str | None = None,
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

    ``console_url`` (`alpha-engine-config-I9926`) is the fleet console's base
    URL when one is configured. When set, the FULL BOARD section links the
    console's Decision list — a stable address — and the presigned page is
    not read at all, so a console-linked report never pays for, or fails on,
    a presign it does not use. When unset, the presigned path is what it was.
    """
    moment = (now or dt.datetime.now(dt.UTC)).astimezone(dt.UTC)
    # STRICT face of the one reader: a corrupt current board raises with the
    # cause named, and the manifest reads `failed` — this artifact is not
    # optional (see `_read_json`).
    board = load_store_document(store, BOARD_CURRENT_KEY)
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
    board_console_url: str | None = None
    if console_url:
        board_console_url = console_url.rstrip("/") + BOARD_CONSOLE_PATH
        # No presign was taken, so there is no expiry to report — an empty
        # string, not a stamp that would read as a bound on a link that has
        # none (review C1).
        board_url, board_url_denied_code, board_url_expires = None, None, ""
    else:
        board_url, board_url_expires, board_url_denied_code = _board_url(store, now=moment)

    return MorningInputs(
        board=board,
        board_uri=store_uri(store),
        board_url=board_url,
        board_url_expires=board_url_expires,
        board_url_denied_code=board_url_denied_code,
        board_console_url=board_console_url,
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


def _phase_table_rows(board: dict[str, Any]) -> list[str]:
    """The ladder as Markdown table rows: phase, state, clauses met, holding.

    A phase row that vanished from the board is NOT silently absent: an empty
    ladder is a named defect rather than a blank table, so six phases
    becoming zero is visible without the reader knowing there should be six.
    The fraction is computed from the row's own `clauses` list
    (`crucible.board.BoardRow.clauses`), never parsed back out of the English
    `detail` — a contract restated as a regex is the bug class this
    repository has already paid for twice.
    """
    rows = [row for row in board.get("rows", []) if row.get("source") == "phase"]
    if not rows:
        return [
            "| _no phase row on the board_ | | | "
            "_the ladder is not being rendered — a defect in the board_ |"
        ]
    out: list[str] = []
    for row in rows:
        clauses = row.get("clauses")
        if isinstance(clauses, list) and clauses:
            met = sum(1 for c in clauses if c.get("met"))
            unmet = [str(c.get("name", "?")) for c in clauses if not c.get("met")]
            holding = _md_cell(_plain_withhold(", ".join(unmet))) if unmet else ""
            out.append(f"| {row['id']} | {row['state']} | {met}/{len(clauses)} | {holding} |")
        elif isinstance(clauses, list):
            # Read, and it declares nothing. `0/0` would read as a measurement.
            out.append(
                f"| {row['id']} | {row['state']} | no clauses | this gate measured nothing |"
            )
        else:
            # No clause list on this board at all: a producer that took no
            # gate reading. The detail sentence is what there is, and
            # inventing a fraction from it would be the fabrication the
            # structured field exists to remove.
            out.append(
                f"| {row['id']} | {row['state']} | — | {_md_cell(_plain_withhold(row['detail']))} |"
            )
        if row["state"] == "OUT_OF_ORDER":
            out.append(f"| | | | out of order: {_md_cell(_plain_withhold(row['detail']))} |")
    return out


def _schedule_table_rows(board: dict[str, Any]) -> list[str]:
    """One Markdown table row per plan §6.1 milestone
    (`alpha-engine-config-I9914`).

    Reads `schedule:*` board rows exactly the way :func:`_phase_table_rows`
    reads `phase:*` rows — this function computes nothing; it quotes what
    `crucible.board._schedule_rows` already decided. `OVERDUE FIRST`: a
    milestone that has passed its date is the only row in this table anybody
    has to act on, and a table ordered by the board's declaration order
    buries it under the ones still ahead. The sort is STABLE, so within each
    group the board's own order survives.
    """
    rows = [row for row in board.get("rows", []) if row.get("source") == "schedule"]
    if not rows:
        return [
            "| _no schedule row on the board_ | | "
            "_the plan §6.1 milestones are not being rendered — a defect in the board_ |"
        ]
    ordered = sorted(rows, key=lambda row: 0 if row["state"] == "UNMET" else 1)
    return [
        f"| {'OVERDUE ' if row['state'] == 'UNMET' else ''}{row['id']} | {row['state']} | "
        f"{_md_cell(_plain_withhold(row['detail']))} |"
        for row in ordered
    ]


def _moved_lines_plain(inputs: MorningInputs) -> list[str]:
    """What changed, old -> new, as Markdown list items. Never an absolute-
    state retelling.

    A previous board that could not be read yields the READ FAILURE, never
    "nothing moved": reporting no movement over a failed comparison is a
    positive claim asserted on no evidence, and it is exactly the shape
    `crucible.board._read_previous_board` exists to refuse one layer down.
    """
    if inputs.previous is None:
        return [f"cannot say — the {inputs.previous_day} board is {inputs.previous_reason}"]
    before = {row["id"]: row["state"] for row in inputs.previous.get("rows", [])}
    after = {row["id"]: row["state"] for row in inputs.board.get("rows", [])}
    moved = [
        f"- {row_id}: {before.get(row_id, 'ABSENT')} -> {after.get(row_id, 'VANISHED')}"
        for row_id in sorted(set(before) | set(after))
        if before.get(row_id) != after.get(row_id)
    ]
    return moved or ["nothing moved"]


def _moved_count(inputs: MorningInputs) -> str:
    """The headline's version of :func:`_moved_lines_plain`: a bare count,
    since the headline states none of the row ids that moved — the full
    update carries those."""
    if inputs.previous is None:
        return f"cannot say ({_escape_html(inputs.previous_reason)})"
    before = {row["id"]: row["state"] for row in inputs.previous.get("rows", [])}
    after = {row["id"]: row["state"] for row in inputs.board.get("rows", [])}
    moved = sum(1 for row_id in set(before) | set(after) if before.get(row_id) != after.get(row_id))
    return str(moved)


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

    Returns PLAIN text — the one caller that needs it HTML-escaped
    (`render_message`, the headline) escapes the whole return value itself,
    since none of it is markup this function owns.

    The completeness rule itself is NOT stated here. It lives in
    `crucible.keys.parse_acceptance_reading`, which the board page reads the
    same artifact through, so the two surfaces cannot disagree about whether
    a document is a reading — they did, on one document, before this review.
    """
    if denied_code is not None:
        return ACCEPTANCE_UNREADABLE.format(code=denied_code)
    parsed = parse_acceptance_reading(reading)
    if parsed is None:
        return ACCEPTANCE_NOT_ON_ANY_ARTIFACT
    return (
        f"acceptance count: {parsed.met} met / {parsed.unmet} unmet / "
        f"{parsed.unmeasurable} unmeasurable of {parsed.total} (commit {parsed.commit})"
    )


def _acceptance_lines_plain(
    reading: dict[str, Any] | None, *, denied_code: str | None
) -> list[str]:
    """The acceptance count, then the clause ids behind it, for the full update.

    The ids come from the SAME parse the count does
    (`crucible.keys.parse_acceptance_reading`), so a document complete enough
    to state a figure is the only kind that gets ids listed under it. When the
    producer filed no list this says so rather than going quiet: "the producer
    files no names" and "there are no unmet clauses" are different facts, and
    an empty section renders them identically.
    """
    lines: list[str] = [_acceptance_line(reading, denied_code=denied_code)]
    if denied_code is not None:
        return lines
    parsed = parse_acceptance_reading(reading)
    if parsed is None:
        return lines
    for named, label in (
        (parsed.unmet_clauses, "unmet"),
        (parsed.unmeasurable_clauses, "unmeasurable"),
    ):
        if named:
            lines.append(f"  {label}: {_plain_withhold(', '.join(named))}")
        else:
            lines.append(f"  the artifact names no {label} clause ids")
    return lines


def _silence_line(board: dict[str, Any]) -> str:
    counts = board.get("counts", {})
    parts = ", ".join(f"{counts.get(state, 0)} {state}" for state in SILENT_STATES)
    return f"silence: {parts} of {board.get('row_count', 0)} rows"


def _board_link_lines_plain(inputs: MorningInputs) -> list[str]:
    """The board link, plain, for the full update.

    A read that FAILED is distinguished from a page that is ABSENT, the same
    way :func:`_read_json` does it for the acceptance artifact: an
    `AccessDenied` rendered as "the board render published none" would report
    an IAM gap as a defect in another job (`alpha-engine-config-I9896`
    measured exactly that shape — S3 returns 403 for a missing key when the
    caller also lacks `s3:ListBucket` on the prefix).
    """
    if inputs.board_console_url is not None:
        # `alpha-engine-config-I9926`: the console address is the link, and
        # the only link. The presigned page is not offered as a second one —
        # it was never read (see `read_inputs`), and a reader handed two URLs
        # to one board is a reader deciding which to trust.
        return [inputs.board_console_url, BOARD_CONSOLE_CAVEAT]
    if inputs.board_url_denied_code is not None:
        return [BOARD_URL_UNREADABLE.format(code=inputs.board_url_denied_code)]
    if inputs.board_url is None:
        return [BOARD_URL_UNAVAILABLE]
    return [inputs.board_url, BOARD_URL_CAVEAT.format(expires=inputs.board_url_expires)]


def render_full_update(inputs: MorningInputs, *, now: dt.datetime) -> str:
    """The full daily update, as GitHub-flavored Markdown
    (`alpha-engine-config-I10123`) — everything the six-section Telegram
    message used to carry directly, now posted as a tracker comment instead.

    Deterministic in ``now`` and ``inputs`` alone, so the markdown a test
    asserts is the markdown an operator can open — a renderer that reached
    the clock or the store would be tested against something other than what
    is posted.

    No character budget: a GitHub comment's limit is 65536 bytes, three
    orders of magnitude past what six sections of board free text has ever
    reached, so nothing here is ever withheld for SPACE — only plan §12
    rule 3's forbidden-token withholding still applies, via
    :func:`_plain_withhold`, because that withholding is about CONTENT, not
    space.
    """
    board = inputs.board
    local = now.astimezone(DELIVERY_TZ)
    commit = inputs.board_code_sha or f"UNKNOWN ({inputs.board_run_note})"
    lines: list[str] = []

    headline = _staleness(str(board.get("generated_at", "")), now)
    if headline:
        lines += [f"**{headline}**", ""]

    lines.append(f"# Crucible v2 — board for trading day {board.get('trading_day')}")
    lines.append(f"store: `{inputs.board_uri}/{BOARD_CURRENT_KEY}`  ")
    lines.append(f"generated: {board.get('generated_at')}  commit: `{commit}`  ")
    lines.append(f"delivered: {local:%Y-%m-%d %H:%M %Z}")
    lines.append("")

    lines.append("## Ladder")
    lines.append("| phase | state | clauses met | holding |")
    lines.append("|---|---|---|---|")
    lines += _phase_table_rows(board)
    lines.append("")

    lines.append("## Schedule (plan §6.1)")
    lines.append("| milestone | state | detail |")
    lines.append("|---|---|---|")
    lines += _schedule_table_rows(board)
    lines.append("")

    lines.append("## Acceptance")
    lines += _acceptance_lines_plain(inputs.acceptance, denied_code=inputs.acceptance_denied_code)
    lines.append("")

    lines.append(f"## Moved since {inputs.previous_day}")
    lines += _moved_lines_plain(inputs)
    lines.append("")

    lines.append("## Silence")
    lines.append(_silence_line(board))
    lines.append(
        f"pending operator action: {inputs.operator_action}"
        if inputs.operator_action
        else NO_OPERATOR_ACTION
    )
    lines.append("")

    lines.append("## Board")
    lines += _board_link_lines_plain(inputs)

    return "\n".join(lines)


def _headline_board_link(inputs: MorningInputs) -> str | None:
    """The headline's "Board" link target, or `None` when there is nothing
    to link — the full update still names why (see
    :func:`_board_link_lines_plain`), so the headline simply omits the line
    rather than repeating the explanation in twelve lines instead of one."""
    if inputs.board_console_url is not None:
        return inputs.board_console_url
    if inputs.board_url_denied_code is not None or inputs.board_url is None:
        return None
    return inputs.board_url


def render_message(
    inputs: MorningInputs, *, now: dt.datetime, update_url: str, history_url: str
) -> str:
    """The Telegram headline — Brian's 2026-09-06 ruling
    (`alpha-engine-config-I10123`): *"the telegram message is not legible,
    too much information and its not formatted cleanly. can we instead have
    it link to a url that contains the full update?"* Supersedes the
    six-section message this function used to render directly
    (`alpha-engine-config-I9921`). Extended the same day (deliverable 7,
    Brian) with a THIRD link once the rolling issue became a history index.

    Target at most :data:`HEADLINE_TARGET_LINES` lines, and hard-capped at
    :data:`UPDATE_MESSAGE_MAX_CHARS` characters: a title with the trading
    day, exactly one line per plan §6 phase (`Phase N  STATE  N/M` — no row
    id, no holding lists, no "out of order" prose, and never the six
    `...:closing` rows the board carries alongside the real gate readings —
    `alpha-engine-config-I10123` follow-up, measured live 2026-09-06: those
    six extra PLANNED lines were the second thing making the first delivery
    too long), the acceptance line, the moved-since COUNT, the pending
    operator action when there is one, and ONE line carrying all three links
    — "Full update" (``update_url``, today's comment permalink), "History"
    (``history_url``, the rolling issue itself), and "Board" (the console,
    when configured) — separated by " · " rather than one link per line.

    ``update_url`` and ``history_url`` are REQUIRED, not optional, because
    both the comment and the issue they point at exist BEFORE this is ever
    called (`morning_handler`) — a headline rendered with a link to give
    that does not yet resolve would be the illegible shape reappearing,
    just shorter.

    Deterministic in ``now``/``inputs``/the two URLs alone, so the message a
    test asserts is the message an operator receives.
    """
    board = inputs.board
    lines: list[str] = []

    headline = _staleness(str(board.get("generated_at", "")), now)
    if headline:
        lines.append(f"<b>{_escape_html(headline)}</b>")

    lines.append(f"<b>CRUCIBLE V2 — {_escape_html(str(board.get('trading_day')))}</b>")

    rows_by_id = {row.get("id"): row for row in board.get("rows", [])}
    for phase in PHASES:
        row = rows_by_id.get(_PHASE_ROW_ID[phase.id])
        if row is None:
            lines.append(f"Phase {phase.number}  ABSENT — no reading on this board")
            continue
        clauses = row.get("clauses")
        fraction = (
            f"{sum(1 for c in clauses if c.get('met'))}/{len(clauses)}"
            if isinstance(clauses, list) and clauses
            else "—"
        )
        state = _escape_html(str(row.get("state")))
        lines.append(f"Phase {phase.number}  {state:<{_STATE_COLUMN_WIDTH}} {fraction}")

    lines.append(
        _escape_html(_acceptance_line(inputs.acceptance, denied_code=inputs.acceptance_denied_code))
    )
    lines.append(f"moved since {inputs.previous_day}: {_moved_count(inputs)}")
    if inputs.operator_action:
        lines.append(f"pending operator action: {_escape_html(inputs.operator_action)}")

    link_parts = [
        f'<a href="{_escape_html(update_url)}">Full update</a>',
        f'<a href="{_escape_html(history_url)}">History</a>',
    ]
    board_link = _headline_board_link(inputs)
    if board_link:
        link_parts.append(f'<a href="{_escape_html(board_link)}">Board</a>')
    lines.append(" · ".join(link_parts))

    message = "\n".join(lines)
    budget = UPDATE_MESSAGE_MAX_CHARS - wire_length(TRANSPORT_PREFIX)
    if wire_length(message) > budget:
        raise ValueError(
            f"the headline is {wire_length(message)} wire characters, over the "
            f"{budget}-character budget left after the transport prefix. The headline's "
            "shape is fixed (a title, one line per plan phase, three more fixed lines, "
            "one links line) and should never reach this — a board with more phases than "
            "the plan declares, or an operator action of unbounded length, is the "
            "likeliest cause, and the fix is at the source of that field, not a "
            "truncation here: a headline that silently shortened itself would be the "
            "illegible message this ruling exists to end, in a new shape."
        )
    return message


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

    **`parse_mode="HTML"`** (`alpha-engine-config-I9925`, krepis 0.59.50):
    `message` already carries `<b>` headings and every interpolated board
    string already went through :func:`_escape_html` — this call site is
    where that contract is discharged, not where escaping happens. Passed
    explicitly rather than left to krepis' Markdown-v1 default so a heading
    written as `<b>…</b>` is not sent to a transport that would render the
    literal angle brackets.
    """
    publish = transport if transport is not None else _krepis_publish
    result = publish(
        message,
        severity=DELIVERY_SEVERITY,
        source=DELIVERY_SOURCE,
        sns=False,
        telegram=True,
        silent=False,
        dedup_key=None,
        destination=_operator_chat(),
        raise_on_total_failure=True,
        parse_mode="HTML",
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


#: Where the trigger is read from, in priority order.
#:
#: `GITHUB_EVENT_NAME` FIRST, and that ordering is the whole provenance
#: argument: GitHub Actions sets it itself on every run, and `GITHUB_*` is a
#: reserved prefix a workflow's own `env:` block cannot override. A human who
#: dispatches the report therefore cannot produce evidence saying a schedule
#: did. `CRUCIBLE_TRIGGER` is second and exists for a non-GitHub dispatcher,
#: where it is the only thing that can say.
TRIGGER_VARS: tuple[str, ...] = ("GITHUB_EVENT_NAME", "CRUCIBLE_TRIGGER")


def resolve_trigger(environ: dict[str, str] | None = None) -> str:
    """What started this run, as a key-safe name.

    Never inferred from the clock or the calendar. `run_mode` records
    live-vs-replay and a dispatched run is just as live as a scheduled one,
    so nothing already in the manifest can answer "did a human start this" —
    which is why `alpha-engine-config-I9896`, `-I9914` and `-I9921` all sat
    open on a `closes-when` no artifact could satisfy.

    Returns `TRIGGER_UNKNOWN` when the invocation said nothing. That is a
    declared value, not a swallow: it is written to the store like any other
    trigger, so a run whose starter is unknown is VISIBLE as unknown rather
    than missing, and it satisfies no predicate asking for a schedule.
    """
    env = environ if environ is not None else os.environ
    for var in TRIGGER_VARS:
        value = (env.get(var) or "").strip()
        if not value:
            continue
        if not TRIGGER_RE.match(value):
            raise ValueError(
                f"{var}={value!r} is not a usable trigger name (must match "
                f"{TRIGGER_RE.pattern}). It becomes an S3 key segment; refusing is the "
                "only reading that is not a guess about what started this run."
            )
        return value
    return TRIGGER_UNKNOWN


#: The rolling issue's initial body, the one time a day ever creates it. It
#: is never re-posted or edited afterward — every delivery after the first
#: is one more comment on the same issue. The tracker reference in the body
#: text below is deliberately WITHOUT the `alpha-engine-config-` prefix
#: (spelled with a zero-width joiner between "I" and the digits would be
#: unreadable; this is the accepted alternative) so `tests/test_no_stale_
#: tracker_literals.py` does not read it as a hardcoded pointer this module
#: could derive from `crucible.gate.PHASES` — the rolling issue is not a
#: phase issue, so there is nothing to derive it from, and the citation is
#: static prose that never reaches a raised message or a test failure.
_ROLLING_ISSUE_BODY = (
    "Rolling daily update for the crucible v2 board. One comment per "
    "delivery, oldest first; this issue is never closed or edited by "
    "automation. (alpha-engine-config, issue 10123)"
)


def _find_or_create_rolling_issue() -> int:
    """The rolling `[v2 board] daily update` issue's number, creating it
    once if it does not already exist.

    Never called under `--dry-run` — see `morning_handler`. A SECOND open
    issue carrying this exact title is a loud `TrackerError`, not a pick
    (`crucible.tracker.find_issue_by_title`'s own contract): posting to
    whichever one a race or a manual duplicate left behind is a full update
    nobody can find from the headline that links it.
    """
    number = tracker.find_issue_by_title(TRACKER_REPO, ROLLING_ISSUE_TITLE)
    if number is not None:
        return number
    number, _url = tracker.create_issue(TRACKER_REPO, ROLLING_ISSUE_TITLE, _ROLLING_ISSUE_BODY)
    return number


#: The basename of one delivery's compact history facts
#: (`crucible.keys.morning_history_row_key`). Declared here, once, so the
#: writer below and the index reader cannot disagree about what they are
#: listing for.
HISTORY_ROW_BASENAME = "history_row.json"

#: The board's own row shape for a phase GATE (`crucible.board.py`:
#: `id=f"phase:{phase.id}"`, `source="phase"`) -- DERIVED from
#: `crucible.gate.PHASES` rather than restated, because a board row with
#: `source == "phase"` is NOT always a gate reading: the same source also
#: carries one CLOSING row per phase (`id=f"phase:{phase.id}:closing"`,
#: state `PLANNED` until a live gate reads MET), and a filter on `source`
#: alone catches both. Measured live 2026-09-06 on the first delivered
#: history index (`alpha-engine-config-I10123` follow-up): every phase
#: column read `—` because the reader matched bare `phase0`..`phase5`
#: against the board's actual `phase:phase0`..`phase:phase5` ids, and the
#: headline printed twelve lines -- six real gate lines plus six
#: `phase:phaseN:closing: PLANNED —` lines it never meant to include.
_PHASE_ROW_ID: dict[str, str] = {p.id: f"phase:{p.id}" for p in PHASES}

#: Plan §6: phases 0 through 5, six rungs, no more and no fewer. The history
#: table's phase columns are fixed at this width rather than however many a
#: given day's board happened to carry, so a day that renders five phases
#: (a board mid-incident, say) still lines up under the same header as a day
#: that rendered six. Header labels are the bare `Phase.id` ("phase0", ...);
#: matching against a board row uses :data:`_PHASE_ROW_ID`.
_HISTORY_PHASE_IDS: tuple[str, ...] = tuple(p.id for p in PHASES)


def _history_row_payload(
    inputs: MorningInputs, *, now: dt.datetime, comment_url: str
) -> dict[str, Any]:
    """The compact facts one delivery contributes to the history index.

    Built from the SAME `inputs` the full update and the headline render
    from — never a second read of the board — so the index cannot disagree
    with the comment it links to about what that day's board said.
    """
    rows_by_id = {row.get("id"): row for row in inputs.board.get("rows", [])}
    phases: list[dict[str, Any]] = []
    for phase in PHASES:
        row = rows_by_id.get(_PHASE_ROW_ID[phase.id])
        if row is None:
            continue
        clauses = row.get("clauses")
        has_clauses = isinstance(clauses, list) and bool(clauses)
        phases.append(
            {
                # The BARE phase id ("phase0"), never the board's own
                # `phase:phase0` row id -- a stable contract between this
                # writer and the index reader that does not travel with
                # whatever board.py happens to spell its row ids as.
                "id": phase.id,
                "state": row.get("state"),
                "met": sum(1 for c in clauses if c.get("met")) if has_clauses else None,
                "total": len(clauses) if has_clauses else None,
            }
        )
    parsed = (
        None
        if inputs.acceptance_denied_code is not None
        else parse_acceptance_reading(inputs.acceptance)
    )
    return {
        "trading_day": str(inputs.board.get("trading_day")),
        "delivered_pt": now.astimezone(DELIVERY_TZ).strftime("%Y-%m-%d %H:%M %Z"),
        "phases": phases,
        "acceptance_met": parsed.met if parsed else None,
        "acceptance_total": parsed.total if parsed else None,
        "comment_url": comment_url,
    }


def _read_history_rows(store: Store) -> list[tuple[str, dict[str, Any] | None, str]]:
    """Every `history_row.json` filed under `report.morning`'s manifest
    root, read through the guarded parser (`crucible.documents.read_document`
    — the one module allowed to parse store bytes as JSON, AGENTS.md rule 1).

    SURFACE consumer, like `_read_json` above: a corrupt or vanished row is
    named in its own returned triple (`document=None`, ``problem`` set),
    never raised and never silently dropped. The index this feeds is a VIEW
    over the day's own comment, which is the durable record — a fault in one
    day's row must not blank the rest of the history, and must not be
    invisible either.

    Returns `(key, document_or_none, problem)` in ASCENDING key order —
    `Store.list_keys` sorts, and a key is `runs/report.morning/{trading_day}/
    {calendar_date}/history_row.json`, so ascending order groups by trading
    day and, within a day, by firing: the LAST entry for a given trading day
    is that day's most recent delivery, which is what "a re-delivery for the
    same day replaces its row" means in practice.
    """
    prefix = runs_prefix(MORNING_JOB)
    rows: list[tuple[str, dict[str, Any] | None, str]] = []
    for key in sorted(store.list_keys(prefix)):
        if not key.endswith(f"/{HISTORY_ROW_BASENAME}"):
            continue
        try:
            raw = store.get_bytes(key)
        except KeyError:
            rows.append((key, None, "vanished between listing and read"))
            continue
        read = read_document(key, lambda raw=raw: raw)
        if read.problem is not None:
            rows.append((key, None, read.problem))
        else:
            rows.append((key, read.document, ""))
    return rows


def _phase_cell(phases: dict[str, dict[str, Any]], phase_id: str) -> str:
    phase = phases.get(phase_id)
    if phase is None:
        return "—"
    if phase.get("met") is None or phase.get("total") is None:
        return str(phase.get("state", "—"))
    return f"{phase.get('state', '—')} {phase['met']}/{phase['total']}"


def render_history_body(store: Store, *, console_url: str | None = None) -> str:
    """The rolling issue's regenerated BODY: a newest-first history index
    (`alpha-engine-config-I10123` deliverable 7).

    One row per trading day — the LATEST delivery for that day, so a rerun
    never grows the table — carrying that day's six phase states, its
    acceptance figure, and a link to the comment holding the full update.
    Read errors are named in their own row rather than dropped, for the same
    reason every other surface in this module never goes silent on a fault.
    """
    latest: dict[str, dict[str, Any]] = {}
    faulted: dict[str, str] = {}
    for key, document, problem in _read_history_rows(store):
        # `key` segments: "runs", "report.morning", "{trading_day}", ...
        trading_day = key.split("/")[2]
        if document is not None:
            latest[trading_day] = document
            faulted.pop(trading_day, None)
        else:
            faulted[trading_day] = problem
            latest.pop(trading_day, None)

    lines: list[str] = ["# Crucible v2 — daily update history", ""]
    if console_url:
        lines.append(f"Board: {console_url.rstrip('/')}{BOARD_CONSOLE_PATH}")
        lines.append("")
    header = ["trading day", "delivered (PT)", *_HISTORY_PHASE_IDS, "acceptance", "full update"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))

    days = sorted(set(latest) | set(faulted), reverse=True)
    legacy_days: list[str] = []
    for day in days:
        if day in faulted:
            lines.append(
                f"| {day} | unreadable: {faulted[day]} | "
                + " | ".join(["—"] * (len(header) - 2))
                + " |"
            )
            continue
        row = latest[day]
        raw_phases = row.get("phases", [])
        phases = {p["id"]: p for p in raw_phases if isinstance(p, dict)}
        # A day whose delivery recorded SOME phase facts, none of which match
        # a known phase id, is a LEGACY row -- written before this reader's
        # id-matching bug was fixed (`alpha-engine-config-I10123` follow-up:
        # the first live delivery stored the board's own `phase:phase0`
        # shape, which nothing here ever matched). Its dashes are not "the
        # manifest genuinely lacks the field"; they are a known gap, and get
        # a footnote rather than passing for the honest case silently.
        if raw_phases and not (phases.keys() & set(_HISTORY_PHASE_IDS)):
            legacy_days.append(day)
            cells = ["—*"] * len(_HISTORY_PHASE_IDS)
        else:
            cells = [_phase_cell(phases, phase_id) for phase_id in _HISTORY_PHASE_IDS]
        met, total = row.get("acceptance_met"), row.get("acceptance_total")
        acceptance = f"{met}/{total}" if met is not None and total is not None else "—"
        link = f"[comment]({row.get('comment_url', '')})" if row.get("comment_url") else "—"
        lines.append(
            f"| {day} | {row.get('delivered_pt', '—')} | "
            + " | ".join(cells)
            + f" | {acceptance} | {link} |"
        )
    if not days:
        lines.append(
            "| _no delivery has filed a history row yet_ | "
            + " | ".join(["—"] * (len(header) - 1))
            + " |"
        )
    if legacy_days:
        lines.append("")
        lines.append(
            "\\* phase state unavailable: this delivery's history row predates a fix to "
            "how phase ids are matched ("
            + ", ".join(legacy_days)
            + "). Re-running `report.morning` for that trading day backfills it."
        )
    return "\n".join(lines)


def morning_handler(args: argparse.Namespace) -> int:
    """`crucible report.morning [--date] [--dry-run] [--store]`.

    Runs through `run_job` like every other job (AGENTS.md rule 1), so a
    morning that never reached Brian leaves a `failed` manifest naming why,
    and `alerts.sweep`'s two conditions cover this job on the same terms as
    every other: ABSENCE when the `DELIVERY_CRON_UTC` cron does not fire —
    GitHub drops scheduled events under load, measured on this fleet — and
    FAILURE when either the tracker post or the delivery raises.

    **Ordering is the whole safety property** (`alpha-engine-config-I10123`):
    the full update is posted to the tracker BEFORE the headline is ever
    rendered, because the headline's one indispensable line is the comment's
    own permalink. `_find_or_create_rolling_issue` and `tracker.post_comment`
    both raise on any fault, so a failed post fails the run loudly — the
    manifest reads `failed`, and the headline is never sent at all.

    **`--dry-run` renders and files nothing, sends nothing, and touches the
    tracker not at all** — no search, no create, no comment. It prints the
    full update to stdout instead. That is the shape a laptop needs to read
    the production board without writing to it OR to the private tracker,
    and it is honoured for the same reason `board` honours it: this job's
    outputs land under `runs/`, where a dry run would otherwise fabricate a
    manifest saying a report was delivered — and a dry run that posted a
    real tracker comment would fabricate the OTHER document this job now
    produces. Passed through to `run_job` as `dry_run=True`
    (alpha-engine-config-I9922) so the manifest itself is never written
    either.
    """
    from crucible.runner import RunContext, run_job  # noqa: PLC0415 - lazy; see cli.py

    dry_run = bool(getattr(args, "dry_run", False))
    # Wrapped read-only under `--dry-run` (alpha-engine-config-I9922 N1),
    # defense-in-depth: `body` below already skips `ctx.record_output` when
    # `dry_run`, so nothing here should ever reach a MUTATOR, but a store that
    # refuses on its own is what makes that true structurally rather than by
    # this function remembering to check the flag correctly forever.
    store = open_store(getattr(args, "store", None), dry_run=dry_run)
    # `alpha-engine-config-I9926`: the console's base URL is configuration
    # (`CRUCIBLE_CONSOLE_URL`), resolved through `crucible.config.settings` so
    # its provenance is recorded like every other value; empty means "no
    # console" and the presigned page is linked instead.
    from crucible.config import settings as _settings  # noqa: PLC0415 - lazy; see cli.py

    console_url = _settings().console_url or None

    def body(ctx: RunContext) -> None:
        now = ctx.started
        inputs = read_inputs(store, trading_day=ctx.trading_day, now=now, console_url=console_url)
        update = render_full_update(inputs, now=now)
        if dry_run:
            # No output, no manifest, no delivery, and no tracker call at
            # all (as of alpha-engine-config-I10123): `run_job(dry_run=True)`
            # below skips its own manifest write, and `store` above is
            # read-only regardless -- a dry run is visibly a dry run because
            # nothing under `runs/` changes and no comment is ever posted.
            print(update)
            return
        issue_number = _find_or_create_rolling_issue()
        history_url = f"https://github.com/{TRACKER_REPO}/issues/{issue_number}"
        update_url = tracker.post_comment(TRACKER_REPO, issue_number, update)
        message = render_message(inputs, now=now, update_url=update_url, history_url=history_url)
        destination = deliver(message)
        payload = message.encode("utf-8")
        artifact = morning_report_key(ctx.trading_day.isoformat(), ctx.calendar_date.isoformat())
        ctx.record_output(artifact, payload)
        update_artifact = morning_update_key(
            ctx.trading_day.isoformat(), ctx.calendar_date.isoformat()
        )
        ctx.record_output(update_artifact, update.encode("utf-8"))
        history_row = _history_row_payload(inputs, now=now, comment_url=update_url)
        ctx.record_output(
            morning_history_row_key(ctx.trading_day.isoformat(), ctx.calendar_date.isoformat()),
            json.dumps(history_row, sort_keys=True).encode("utf-8"),
        )
        # WHAT started this delivery, filed as its own object beside the
        # message (alpha-engine-config-I9960). The trigger is the KEY, so an
        # `exists` predicate over
        # `runs/report.morning/*/*/trigger.schedule` answers "has a
        # delivery ever happened that no human dispatched" with no body
        # parse — which is exactly what I9896/I9914/I9921 close on. It goes
        # under the job's own manifest prefix, so the delivering identity
        # needs no second IAM grant, and it is not a manifest
        # (`is_manifest_key`), so it satisfies no absence check.
        trigger = resolve_trigger()
        ctx.record_output(
            morning_trigger_key(
                ctx.trading_day.isoformat(), ctx.calendar_date.isoformat(), trigger
            ),
            f"{trigger}\n".encode(),
        )
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
        # The comment URL and issue number, as `outputs`/`metrics`
        # (`alpha-engine-config-I10123` deliverable 4). `run_manifest.v2`'s
        # `metrics` array is `additionalProperties: true` and needs no
        # schema change: `value` carries the issue number (with its `unit`,
        # per the schema's own dependency), and `status_reason`/`source_path`
        # carry the URL and the repo#issue reference a human or a sweep can
        # act on directly.
        ctx.record_metric(
            {
                "name": "morning_report_tracker_comment",
                "module": "crucible.morning",
                "metric_type": "operational",
                "value": float(issue_number),
                "unit": "issue_number",
                "n_floor": 1,
                "status": "OK",
                "status_reason": (
                    f"posted the full update to {TRACKER_REPO}#{issue_number}: {update_url}"
                ),
                "source_path": update_url,
                "last_updated_utc": now.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )
        # The index rewrite runs LAST, and its failure fails the run
        # (`alpha-engine-config-I10123` deliverable 7): the comment posted
        # above is the durable RECORD of today's update, and this is a VIEW
        # over every day's comment -- so a failure here must not roll back or
        # skip anything already delivered, and must still be loud (the
        # manifest's `reason` names it, and `alerts.sweep`'s failure
        # condition pages on it) rather than silently leaving a stale index.
        history_body = render_history_body(store, console_url=console_url)
        tracker.update_issue_body(TRACKER_REPO, issue_number, history_body)

    run_job(
        MORNING_JOB,
        body,
        store=store,
        trading_day=args.trading_day,
        run_mode=getattr(args, "run_mode", None),
        # The FIRING, not the trading day. `DELIVERY_CRON_UTC` fires every
        # calendar day and `resolve_trading_day` collapses Saturday, Sunday
        # and Monday onto Friday's close (§4.12), so without this the weekend
        # deliveries overwrite one another and the store's answer to "did the
        # report go out on Sunday" is decided by write ordering. Same shape,
        # same reason, as `alerts.sweep`.
        discriminator=lambda ctx: ctx.calendar_date.isoformat(),
        dry_run=dry_run,
    )
    return 0


def run_report(
    store: Store,
    *,
    trading_day: dt.date,
    now: dt.datetime,
    console_url: str | None = None,
) -> str:
    """Read the board and render the FULL UPDATE. Reading and rendering only
    — no tracker call, no delivery.

    Split out so the render is exercisable without argparse, without a
    runner, without the tracker and without a transport.
    ``console_url`` is `crucible.config.Settings.console_url` when the
    handler resolved one (`alpha-engine-config-I9926`).

    Returns the full update's Markdown (`render_full_update`), not the
    Telegram headline — the headline needs a comment URL that only exists
    once this content has actually been posted, so it has no laptop-only
    equivalent; `morning_handler`'s `--dry-run` path prints exactly this.
    """
    return render_full_update(
        read_inputs(store, trading_day=trading_day, now=now, console_url=console_url),
        now=now,
    )
