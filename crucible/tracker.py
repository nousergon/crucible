"""The phase tracker, as the ONE adapter that reads and writes its issues.

Normative source: `alpha-engine-config-I9967` deliverables 2 and 3.

Every phase of the plan §6 ladder has two instruments answering "has this
phase exited": the gate, which measures, and the tracker issue, which a human
opens and closes. On 2026-09-02 they disagreed — `alpha-engine-config-I9757`
was closed the moment its build PRs merged while its own gate read 1 of 6
clauses met — and the one a human reads on a backlog board was the wrong one.
This module is the wire between them, and it is deliberately narrow.

**It may comment. It may never close.** Closing or reopening a phase issue is
Brian's authority (`principles.md` §3.2), and the issue's own deliverable 1
says so in as many words. A machine comment is a RECORD, not a closure. The
only mutating request this module can construct is a `POST` to an issue's
`/comments`; `tests/test_phase_closing_record.py` asserts that as a property
of the source rather than as a convention, because a convention about
authority is the thing that failed here already.

**Every read is guarded and none of them raises.** The board renders this
adapter's answer as a row, and a surface that dies on a 403 publishes
nothing at all — the shape `crucible.documents` exists to remove for the
store. `IssueRead` carries the same three outcomes: read, and the reason it
could not be read, with `access_problem` separating "the tracker says the
issue is open" from "we could not ask".

**The credential is operator-granted and its absence is never green.** The
crucible repository holds an AWS identity via OIDC and NO GitHub credential
for `nousergon/alpha-engine-config`, which is a private repository in the
same org: an Actions `GITHUB_TOKEN` is scoped to the repository it runs in
and cannot read another one's issues at all. Minting a fine-grained token
with `Issues: read and write` on the tracker is a human action — see
:func:`grant_command`, which emits the exact command. Until it is granted
every closing row on the board reads `UNMEASURABLE` with that command in its
detail (`pull-request-policy` §4.2 form 3: the merge emits the command and a
detector stays red until it runs).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

__all__ = [
    "API_ROOT",
    "HTTP_TIMEOUT_S",
    "ISSUE_STATES",
    "TRACKER_TOKEN_VAR",
    "IssueRead",
    "TrackerError",
    "comment_bodies",
    "credential",
    "grant_command",
    "post_comment",
    "read_issue",
]

#: GitHub's REST root. A module-level constant so a test can point the adapter
#: at nothing at all and so the one hostname this package talks to outside AWS
#: is greppable in one line.
API_ROOT = "https://api.github.com"

#: The variable carrying the tracker credential. Named for crucible rather
#: than for GitHub because it is scoped to ONE repository's issues and is not
#: interchangeable with `GH_TOKEN`/`GITHUB_TOKEN`, which in this repository's
#: workflows means "the Actions token for `nousergon/crucible`" and cannot
#: read the tracker at all.
TRACKER_TOKEN_VAR = "CRUCIBLE_TRACKER_TOKEN"

#: Seconds. A daily render blocked forever on a hung socket is an absence
#: page on a working producer.
HTTP_TIMEOUT_S = 20

#: GitHub's closed vocabulary for an issue. Anything else is refused rather
#: than passed through: a state this package has no rendering for would reach
#: the board as a row it cannot classify.
ISSUE_STATES: tuple[str, ...] = ("open", "closed")

#: A GitHub API request, as the caller may substitute it. `(status, body)`.
#: Injected so every test in this repository exercises the real request
#: construction — headers, method, URL and payload — without a socket.
Opener = Callable[[urllib.request.Request], tuple[int, bytes]]


class TrackerError(RuntimeError):
    """A tracker WRITE that could not be performed.

    Raised, never swallowed: the only writer is the one filing a phase's
    closing record, and a record filed to the store while the tracker was
    never told is precisely the two-instruments-disagreeing defect this
    mechanism exists to remove. Reads do not use this — see :class:`IssueRead`.
    """


def grant_command(repo: str) -> str:
    """The exact operator command that grants this adapter its credential.

    One sentence of shell, emitted verbatim onto every surface that is red for
    want of it. An operator step recorded in an issue and nowhere else is
    `alpha-engine-config-I1906`, closed as *fixed* on a PR whose command was
    never run.
    """
    return (
        f"gh secret set {TRACKER_TOKEN_VAR} --repo nousergon/crucible --body '<token>', "
        f"where <token> is a fine-grained personal access token scoped to {repo} "
        "with Issues: read and write and no other permission"
    )


def credential(explicit: str | None = None) -> str | None:
    """The tracker credential, or ``None`` when none is granted.

    ``None`` is a first-class answer and every caller renders it as a named
    absence. It is never defaulted to the Actions token: `GITHUB_TOKEN` is
    scoped to the repository the workflow runs in, so falling back to it would
    turn "no grant" into a 404 that reads like a deleted issue.
    """
    value = (explicit if explicit is not None else os.environ.get(TRACKER_TOKEN_VAR)) or ""
    return value.strip() or None


@dataclass(frozen=True)
class IssueRead:
    """One tracker issue's state, or the reason it could not be read.

    ``state`` is ``'open'`` or ``'closed'`` when the read succeeded and
    ``None`` otherwise, and exactly one of ``state``/``problem`` is set. The
    ``access_problem`` flag mirrors `crucible.documents.DocumentRead`'s field
    of the same name and carries the same distinction: a fault in OUR access
    renders `UNMEASURABLE`, never `UNMET`.
    """

    state: str | None
    problem: str | None
    access_problem: bool = False

    @property
    def closed(self) -> bool:
        """True only when the tracker was READ and says closed."""
        return self.state == "closed"


def _default_opener(request: urllib.request.Request) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_S) as response:
            return int(response.status), response.read()
    except urllib.error.HTTPError as exc:
        # A 403/404/422 is an ANSWER, not a transport failure: it carries a
        # body naming what GitHub refused, and the caller renders that. Only a
        # URLError (an OSError) propagates, and each caller catches it by name.
        return int(exc.code), exc.read()


def _request(
    repo: str,
    path: str,
    *,
    token: str,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    opener: Opener | None = None,
) -> tuple[int, bytes]:
    url = f"{API_ROOT}/repos/{repo}{path}"
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Authorization", f"Bearer {token}")
    request.add_header("Accept", "application/vnd.github+json")
    request.add_header("X-GitHub-Api-Version", "2022-11-28")
    if data is not None:
        request.add_header("Content-Type", "application/json")
    return (opener or _default_opener)(request)


def _decode(body: bytes) -> Any:
    return json.loads(body.decode("utf-8"))


def read_issue(
    repo: str,
    issue: int,
    *,
    token: str | None = None,
    opener: Opener | None = None,
) -> IssueRead:
    """Whether tracker issue ``issue`` is open or closed. Never raises."""
    granted = credential(token)
    if granted is None:
        return IssueRead(
            None,
            f"no tracker credential: ${TRACKER_TOKEN_VAR} is unset, so this render could "
            f"not ask {repo} whether the issue is open or closed. Grant it with: "
            f"{grant_command(repo)}",
            access_problem=True,
        )
    try:
        status, body = _request(repo, f"/issues/{issue}", token=granted, opener=opener)
    except OSError as exc:
        return IssueRead(
            None, f"reading {repo}#{issue} failed at the transport: {exc}", access_problem=True
        )
    if status != 200:
        return IssueRead(
            None,
            f"GitHub answered {status} for {repo}#{issue}: {body.decode('utf-8', 'replace')[:200]}",
            access_problem=True,
        )
    try:
        document = _decode(body)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return IssueRead(
            None, f"{repo}#{issue} did not answer with JSON: {exc}", access_problem=True
        )
    state = document.get("state") if isinstance(document, dict) else None
    if state not in ISSUE_STATES:
        return IssueRead(
            None,
            f"{repo}#{issue} reports state {state!r}, which is not one of {ISSUE_STATES}",
            access_problem=True,
        )
    return IssueRead(str(state), None)


def comment_bodies(
    repo: str,
    issue: int,
    *,
    token: str | None = None,
    opener: Opener | None = None,
) -> list[str]:
    """Every comment body on ``issue``, oldest first. RAISES on any fault.

    The strict face, because its one caller is the writer: it exists so a
    closing record is not posted twice, and a listing that silently came back
    short would post a duplicate rather than skip one.

    Pagination is followed to the declared ceiling and a listing that would
    need more pages raises rather than truncating — a phase issue with over a
    thousand comments is a fact worth failing on, not one worth guessing past.
    """
    granted = credential(token)
    if granted is None:
        raise TrackerError(
            f"no tracker credential: ${TRACKER_TOKEN_VAR} is unset, so this run cannot read "
            f"the comments already on {repo}#{issue} and cannot tell whether the closing "
            f"reading has been posted. Grant it with: {grant_command(repo)}"
        )
    bodies: list[str] = []
    for page in range(1, 11):
        try:
            status, body = _request(
                repo,
                f"/issues/{issue}/comments?per_page=100&page={page}",
                token=granted,
                opener=opener,
            )
        except OSError as exc:
            raise TrackerError(f"listing comments on {repo}#{issue} failed: {exc}") from exc
        if status != 200:
            raise TrackerError(
                f"GitHub answered {status} listing comments on {repo}#{issue}: "
                f"{body.decode('utf-8', 'replace')[:200]}"
            )
        try:
            document = _decode(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise TrackerError(f"{repo}#{issue} comments did not answer with JSON: {exc}") from exc
        if not isinstance(document, list):
            raise TrackerError(
                f"{repo}#{issue} comments answered with a {type(document).__name__}, not a list"
            )
        bodies.extend(str(item.get("body") or "") for item in document if isinstance(item, dict))
        if len(document) < 100:
            return bodies
    raise TrackerError(
        f"{repo}#{issue} carries more than 1000 comments; this reader stops rather than "
        "deciding from a truncated listing whether the closing reading is already posted"
    )


def post_comment(
    repo: str,
    issue: int,
    body: str,
    *,
    token: str | None = None,
    opener: Opener | None = None,
) -> str:
    """Post ``body`` as a comment on ``issue`` and return its URL. RAISES on any fault.

    The ONLY mutating call this package makes against the tracker. It does not
    close, reopen, label, assign or edit: a machine comment is a record, and
    the state of a phase issue is Brian's to set.
    """
    granted = credential(token)
    if granted is None:
        raise TrackerError(
            f"no tracker credential: ${TRACKER_TOKEN_VAR} is unset, so the closing reading "
            f"for {repo}#{issue} cannot be posted. It is not filed to the store either — "
            "a record in one place and not the other is the two-instruments-disagreeing "
            f"defect this mechanism removes. Grant it with: {grant_command(repo)}"
        )
    if not body.strip():
        raise TrackerError(
            f"refusing to post an empty comment to {repo}#{issue}: an empty record is "
            "indistinguishable from no record and harder to notice."
        )
    try:
        status, raw = _request(
            repo,
            f"/issues/{issue}/comments",
            token=granted,
            method="POST",
            payload={"body": body},
            opener=opener,
        )
    except OSError as exc:
        raise TrackerError(f"posting to {repo}#{issue} failed at the transport: {exc}") from exc
    if status != 201:
        raise TrackerError(
            f"GitHub answered {status} posting a comment to {repo}#{issue}: "
            f"{raw.decode('utf-8', 'replace')[:200]}"
        )
    try:
        document = _decode(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise TrackerError(
            f"the comment on {repo}#{issue} was accepted but GitHub's answer was not JSON, "
            f"so its URL cannot be recorded: {exc}"
        ) from exc
    url = document.get("html_url") if isinstance(document, dict) else None
    if not isinstance(url, str) or not url:
        raise TrackerError(
            f"the comment on {repo}#{issue} was accepted but carries no html_url, so the "
            "record cannot name where it landed"
        )
    return url
