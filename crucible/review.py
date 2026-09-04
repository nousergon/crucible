"""Record one independent adversarial review as a durable store artifact.

Plan §11 risk 1: "every track's exit is reviewed by an independent adversarial
agent against §2's acceptance tests, not by its author." The practice is
vindicated — on 2026-09-02, 5 of 8 crucible PRs failed independent adversarial
review round one and 3 failed round two, and every defect found was real.

**The review is a practice that is RUN, not a check that BLOCKS** (Brian
ruling 2026-09-03). `adversarial-review-gate` is never re-armed as a GitHub
required status check on this repository, and this module deliberately posts
no check-run, no commit status and nothing else a pull request can read as
red. Two measured properties decided it, and both made a required check worse
than no check at all:

  * an `if:`-skipped job still posts a check-run under its own name, and a
    `skipped` conclusion counts as successful toward a required check — so any
    workflow carrying both the gate job and a second trigger let the graded
    party green the gate by pressing a dispatch button;
  * the author side of the comparison is read out of a commit message the
    author writes (see :func:`author_identities`), so a required check resting
    on it converts an unexamined PR into a green one.

What remains is the thing that was actually missing: a durable artifact a
PHASE EXIT can read. `crucible.gate._clause_independently_reviewed` reads it
(`alpha-engine-config-I9794`); this module writes it; nothing else does.

**An adverse verdict is structurally durable, not durable by good behaviour.**
The verdict is a segment of the KEY (:func:`crucible.keys.review_key`), so a
`pass` and a `fail` from one reviewer on one session are different objects and
`put_bytes` cannot make the second replace the first. That is the whole fix:
a guard in this writer would only bind callers that go through this writer,
while the key shape binds `aws s3 cp` too, and the review-writer IAM role
carries `PutObject`/`GetObject` with no `DeleteObject`, so a recorded finding
cannot be removed by the party it was recorded against. How a fail is
legitimately superseded is the gate clause's rule, stated there: an
independent `pass` naming a DIFFERENT head sha, filed no earlier than the
fail. Findings are answered by changing the code, and changing the code
changes the sha.

**What this control does NOT prove — read before relying on it.** There is no
cryptographic session attestation anywhere in this fleet, so BOTH sides of the
independence comparison rest on self-declaration:

  * the REVIEWER names itself; an agent dispatching under a session id that is
    not its own defeats the check;
  * the AUTHOR side is derived from the `Claude-Session:` trailer, which the
    author WRITES — omitting it, mistyping it, or dispatching under a session
    id that appears in no commit all pass. Emails and logins can never collide
    with `session_*`, so the trailer is the only real overlap source and an
    author that writes none has an empty overlap set;
  * a required approving GitHub review is not the control and cannot be: this
    fleet has exactly one collaborator and it has authored every PR ever
    opened in this repository, so a login comparison is equal on every
    dispatch and structurally unsatisfiable.

What it buys is real and worth having: the honest path is the cheap one, a
deliberate bypass is legible afterwards in the commit trailers and the review
artifacts, and a CARELESS self-review — overwhelmingly the common failure — is
refused outright. The residual is accepted. Claiming it away would be worse
than the residual, because a control whose documentation overstates it is how
the next reader stops checking.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path
from typing import Any

from crucible.calendar import resolve_trading_day
from crucible.gate import REVIEW_SCHEMA_VERSION
from crucible.keys import REVIEWER_PATTERN, review_key
from crucible.store import Store, open_store

__all__ = [
    "COMMITS_ENDPOINT_CAP",
    "ReviewError",
    "author_identities",
    "check_reviewer",
    "record",
    "review_document",
]

#: The reviewer identity shape, imported from `crucible.keys` rather than
#: restated. The first version of this control had two copies with different
#: flags and a comment claiming they matched — `SESSION_01AB...` passed the
#: workflow and would have raised in `review_key` — which is the drift
#: `crucible/keys.py`'s whole premise exists to prevent. One regex, one place.
_REVIEWER_RE = re.compile(REVIEWER_PATTERN, re.IGNORECASE)

#: The `Claude-Session:` TRAILER, anchored to the start of its own line — not
#: a scan of the whole commit message. A review round routinely quotes the
#: session id it is answering, so a free scan let a commit BODY mentioning a
#: reviewer's session id put that reviewer into the author set and block it on
#: that change forever, with a message saying it had written the code.
SESSION_TRAILER_RE = re.compile(
    r"^[ \t]*Claude-Session:[ \t]*\S*?(session_[A-Za-z0-9]{8,})[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)

#: `GET /pulls/{n}/commits` is capped at 250 commits by GitHub and pages no
#: further. Past that the author set is SILENTLY partial, which fails open in
#: the one direction that matters — a missing author makes a reviewer look
#: independent. Refused rather than truncated.
COMMITS_ENDPOINT_CAP = 250


class ReviewError(RuntimeError):
    """A condition under which a review may not be recorded.

    Always fatal, never degraded (AGENTS.md rule 5): "we could not establish
    independence" and "it was independently reviewed" are different answers,
    and only one of them may end up in the store.
    """


def author_identities(commits: list[dict[str, Any]]) -> list[str]:
    """Every identity that authored ``commits``, lower-cased and sorted.

    Read out of the commits, never out of the dispatch. Three sources, unioned
    rather than ranked, because each covers a case the others miss: the
    `Claude-Session:` trailer names the authoring AGENT (the only identity
    that differs between two agents in this fleet); the git author/committer
    emails cover a commit written by a human or a tool that writes no trailer;
    the GitHub logins cover a bot (`dependabot[bot]`) whose commits carry
    neither.

    The trailer is author-written, so this is better-evidenced than a dispatch
    input, not proof — see the module docstring. Raises when the union is
    empty and when the commit list hit GitHub's page cap: both mean "we cannot
    establish independence", and neither is a pass.
    """
    if len(commits) >= COMMITS_ENDPOINT_CAP:
        raise ReviewError(
            f"the change under review has {len(commits)} commits, at or past GitHub's "
            f"{COMMITS_ENDPOINT_CAP}-commit cap on /pulls/{{n}}/commits. Beyond that the "
            "author set is silently partial, and a missing author makes a reviewer look "
            "independent. Split the change, or derive the author set from an endpoint "
            "that pages past the cap"
        )
    identities: set[str] = set()
    for commit in commits:
        payload = commit.get("commit") or {}
        message = payload.get("message") or ""
        identities.update(m.lower() for m in SESSION_TRAILER_RE.findall(message))
        for holder in (payload.get("author"), payload.get("committer")):
            email = ((holder or {}).get("email") or "").strip()
            if email:
                identities.add(email.lower())
        for holder in (commit.get("author"), commit.get("committer")):
            login = ((holder or {}).get("login") or "").strip()
            if login:
                identities.add(login.lower())
    if not identities:
        raise ReviewError(
            f"none of the {len(commits)} commits under review names an author — no "
            "Claude-Session trailer, no git author or committer email, no GitHub login. "
            "Independence cannot be established against an empty author set, so this is "
            "refused rather than recorded"
        )
    return sorted(identities)


def check_reviewer(reviewer: str, authors: list[str]) -> str:
    """``reviewer``, validated, lower-cased, and not one of ``authors``.

    Raises otherwise. The ONE independence comparison in this repository.
    """
    if not _REVIEWER_RE.match(reviewer):
        raise ReviewError(
            f"reviewer {reviewer!r} is not a Claude session id (`session_<alnum>`, at "
            "least 8 characters). Independence is measured between session identities; a "
            "free-text field the dispatcher fills with anything measures nothing"
        )
    lowered = reviewer.lower()
    if lowered in authors:
        raise ReviewError(
            f"reviewer `{reviewer}` is one of the identities read out of the commits "
            f"under review ({', '.join(authors)}). An adversarial review must be "
            "independent of the author (plan §11 risk 1) — record it from a session that "
            "did not write this change"
        )
    return lowered


def review_document(
    *,
    phase: str,
    verdict: str,
    reviewer: str,
    commits: list[dict[str, Any]],
    head_sha: str,
    pr_number: int,
    summary: str,
    reviewed_at: dt.datetime,
) -> dict[str, Any]:
    """The artifact `crucible.gate._clause_independently_reviewed` reads.

    Every field it validates is produced here, so the producer and the consumer
    cannot disagree about the shape: a document this function builds is a
    document that clause will read, and `tests/test_gate_independent_review.py`
    asserts exactly that rather than restating the field list.
    """
    if verdict not in ("pass", "fail"):
        raise ReviewError(f"verdict {verdict!r} is not 'pass' or 'fail'")
    if not re.match(r"^[0-9a-f]{40}$", head_sha):
        raise ReviewError(
            f"head_sha {head_sha!r} is not a full 40-hex commit sha. A review that does "
            "not name what it reviewed grades nothing"
        )
    authors = author_identities(commits)
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "phase": phase,
        "verdict": verdict,
        "reviewer": check_reviewer(reviewer, authors),
        "authors": authors,
        "pr_number": pr_number,
        "head_sha": head_sha,
        "summary": summary,
        "reviewed_at": reviewed_at.isoformat(),
    }


def record(
    store: Store,
    *,
    document: dict[str, Any],
    trading_day: dt.date,
) -> str:
    """File ``document`` and return its key.

    No read-modify-write and no overwrite guard, deliberately: the verdict is
    a key segment, so a `pass` cannot land on a `fail`'s object under any
    caller, including one that never imports this module. See the module
    docstring.
    """
    key = review_key(
        document["phase"],
        trading_day.isoformat(),
        document["reviewer"],
        document["verdict"],
    )
    store.put_bytes(key, json.dumps(document, indent=2, sort_keys=True).encode("utf-8"))
    return key


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="record an independent adversarial review")
    sub = parser.add_subparsers(dest="command", required=True)

    authors = sub.add_parser("authors", help="identities read out of the commits under review")
    authors.add_argument("--commits", required=True, help="GET /pulls/{n}/commits, as JSON")

    check = sub.add_parser(
        "check",
        help="validate the reviewer and prove independence, writing nothing",
    )
    check.add_argument("--commits", required=True)
    check.add_argument("--reviewer", required=True)

    file_it = sub.add_parser("record", help="file the review artifact in the store")
    file_it.add_argument("--commits", required=True)
    file_it.add_argument("--reviewer", required=True)
    file_it.add_argument("--phase", required=True)
    file_it.add_argument("--verdict", required=True, choices=("pass", "fail"))
    file_it.add_argument("--head-sha", required=True)
    file_it.add_argument("--pr-number", required=True, type=int)
    file_it.add_argument("--summary", required=True)
    file_it.add_argument("--store", default=None)

    args = parser.parse_args(argv)
    try:
        commits = json.loads(Path(args.commits).read_text(encoding="utf-8"))
        if args.command == "authors":
            print("\n".join(author_identities(commits)))
            return 0
        if args.command == "check":
            # A separate verb so the recording workflow can refuse a
            # self-review BEFORE it asks for AWS credentials: a session that
            # may not record a verdict has no business holding a token that
            # could write one.
            print(check_reviewer(args.reviewer, author_identities(commits)))
            return 0
        now = dt.datetime.now(dt.UTC)
        document = review_document(
            phase=args.phase,
            verdict=args.verdict,
            reviewer=args.reviewer,
            commits=commits,
            head_sha=args.head_sha,
            pr_number=args.pr_number,
            summary=args.summary,
            reviewed_at=now,
        )
        # A run launched on a Saturday binds to Friday's close (rule 3). The
        # calendar resolves it; a raw `date.today()` would file a key the §4.12
        # store walk refuses.
        key = record(
            open_store(args.store),
            document=document,
            trading_day=resolve_trading_day(now.replace(tzinfo=None)),
        )
        print(key)
        return 0
    except ReviewError as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
