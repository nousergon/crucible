"""Independent adversarial review, MEASURED — the half of it GitHub can see.

Plan §11 risk 1: "every track's exit is reviewed by an independent adversarial
agent against §2's acceptance tests, not by its author." That practice is
vindicated — on 2026-09-02, 5 of 8 crucible PRs failed round one and 3 failed
round two, and every defect found was real. What was broken was the gate, in
two independent ways, and this module is one half of the repair
(`alpha-engine-config-I9873`; the other half is `crucible.gate`'s
`independently_reviewed` clause, `alpha-engine-config-I9794`).

**Defect 1 — the required check could never be satisfied.** A GitHub
repository-ruleset required status check is NOT satisfied by a check-run
created through `POST /check-runs`. Measured 2026-09-02 by removal: dropping
`adversarial-review-gate` from `required_status_checks` and changing nothing
else merged instantly; restoring it re-blocked, on a head whose check-run read
`completed/success` under both `filter=all` and `filter=latest`. The check
that DOES satisfy the rule (`lint + foundation tests`) is emitted by a real
job; the one that did not was POSTed by a job through the REST API, and that
is the only structural difference. So the required context is now a real job
that READS a verdict and exits 0 or non-zero. Nothing here posts a check-run.

**Defect 2 — the gate graded an assertion, not a fact.** `reviewer` and
`author` were free-text `workflow_dispatch` inputs, typed by the very session
asking to be passed. A gate whose input is supplied by the thing it grades
measures nothing. So:

  * the AUTHOR side is DERIVED from the commits under review — their
    `Claude-Session:` trailers (this fleet writes one on every commit) plus
    the commit author/committer identities GitHub itself records. A dispatcher
    cannot name a false author, because it does not name the author at all.
  * the REVIEWER side is constrained to the same namespace: a Claude session
    id, `session_<alnum>`. Free text is refused, so the comparison is between
    like things rather than between two strings someone chose.

**What is honestly NOT measured.** This fleet has exactly one GitHub
collaborator and it has authored every PR ever opened in this repository, so
no GitHub identity can separate a reviewer from an author — a required
approving review is structurally unsatisfiable here and is not the control.
The reviewing session names itself, and that self-naming is the residual: an
agent that dispatched under another session's id would defeat this. What it
CANNOT do is claim the code was written by someone else, which is the half
that was defeated by construction before this change.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

#: Commit statuses under this prefix carry a review verdict. The reviewer is
#: IN THE CONTEXT, never parsed out of prose: a `description` is 140
#: characters of free text, and a control that has to regex an identity out of
#: an English sentence is a control whose failure mode is a wording change.
STATUS_CONTEXT_PREFIX = "adversarial-review/"

#: A reviewer identity. Deliberately the same shape `crucible.keys` enforces
#: on the store artifact, so the GitHub-side record and the store-side record
#: cannot disagree about what an identity is.
#: Case-insensitive on purpose: every comparison downstream is lowercased, so
#: accepting `SESSION_...` here and refusing it there would refuse a
#: self-review for the wrong reason and report the wrong remedy.
REVIEWER_RE = re.compile(r"^session_[A-Za-z0-9]{8,}$", re.IGNORECASE)

#: A Claude session id as it appears in a commit trailer:
#: `Claude-Session: https://claude.ai/code/session_01Vk1...`.
SESSION_IN_MESSAGE_RE = re.compile(r"session_[A-Za-z0-9]{8,}")


class ReviewError(RuntimeError):
    """A condition under which independence cannot be established.

    Always fatal, never degraded (AGENTS.md rule 5): "we could not tell" and
    "it was reviewed" are different answers, and only one of them may let a
    merge happen.
    """


def author_identities(commits: list[dict[str, Any]]) -> list[str]:
    """Every identity that authored the commits under review, lowercased.

    Derived, never supplied. Three sources, unioned rather than ranked,
    because each covers a case the others miss: the `Claude-Session:` trailer
    names the authoring AGENT (the only identity that differs between two
    agents in this fleet); the git author/committer emails cover a commit
    written by a human or by a tool that writes no trailer; the GitHub logins
    cover a bot (`dependabot[bot]`) whose commits carry neither.

    Raises when the union is empty: a PR whose commits name nobody makes every
    reviewer independent by construction, which is a pass by accident.
    """
    identities: set[str] = set()
    for commit in commits:
        payload = commit.get("commit") or {}
        message = payload.get("message") or ""
        identities.update(m.lower() for m in SESSION_IN_MESSAGE_RE.findall(message))
        for holder in (payload.get("author"), payload.get("committer")):
            email = (holder or {}).get("email") or ""
            if email.strip():
                identities.add(email.strip().lower())
        for holder in (commit.get("author"), commit.get("committer")):
            login = (holder or {}).get("login") or ""
            if login.strip():
                identities.add(login.strip().lower())
    if not identities:
        raise ReviewError(
            f"none of the {len(commits)} commits under review names an author — no "
            "Claude-Session trailer, no git author or committer email, no GitHub "
            "login. Independence cannot be established against an empty author set, "
            "so this is refused rather than passed"
        )
    return sorted(identities)


def check_reviewer(reviewer: str, authors: list[str]) -> str:
    """``reviewer``, validated and proven independent of ``authors``.

    Raises otherwise. This is the ONE independence comparison; both the
    recording job and the gate job reach it, so the two cannot drift into
    disagreeing about who counts as independent.
    """
    if not REVIEWER_RE.match(reviewer):
        raise ReviewError(
            f"reviewer {reviewer!r} is not a Claude session id (`session_<alnum>`, at "
            "least 8 characters). Independence is measured between session identities; "
            "a free-text field the dispatcher fills with anything measures nothing"
        )
    lowered = reviewer.lower()
    if lowered in authors:
        raise ReviewError(
            f"reviewer `{reviewer}` is one of the identities derived from the commits "
            f"under review ({', '.join(authors)}). An adversarial review must be "
            "independent of the author (plan §11 risk 1) — dispatch it from a session "
            "that did not write this change"
        )
    return lowered


def evaluate(
    commits: list[dict[str, Any]], statuses: list[dict[str, Any]], *, dispatch_hint: str
) -> tuple[bool, str]:
    """Whether this head sha carries an independent passing review, and why.

    A `failure` from an independent reviewer is NOT cleared by a later `pass`
    from a different one. A reviewer who found blocking defects is evidence
    about the code; a second reviewer who did not look at the same thing is
    not a rebuttal. The answer is to fix the findings, which produces a new
    head sha carrying no statuses at all.
    """
    authors = author_identities(commits)
    passed: list[str] = []
    failed: list[str] = []
    unfinished: list[str] = []
    self_reviews: list[str] = []
    for status in statuses:
        context = status.get("context") or ""
        if not context.startswith(STATUS_CONTEXT_PREFIX):
            continue
        reviewer = context[len(STATUS_CONTEXT_PREFIX) :].lower()
        state = status.get("state")
        if reviewer in authors:
            self_reviews.append(f"{reviewer} (an author of these commits)")
            continue
        if state == "success":
            passed.append(reviewer)
        elif state == "failure":
            failed.append(f"{reviewer}: {status.get('description') or '(no summary)'}")
        else:
            # `pending` and `error` are not verdicts. Named rather than
            # dropped: "a review is mid-flight" and "no review exists" call
            # for different responses, and a dropped state reads as the second.
            unfinished.append(f"{reviewer} ({state})")

    if failed:
        return False, (
            "an independent adversarial review recorded FAIL on this head sha: "
            + "; ".join(failed)
            + ". Fix the findings and push — a new head sha is reviewed afresh. A "
            "second reviewer passing it does not clear the first reviewer's findings."
        )
    if passed:
        detail = f"independent adversarial review passed: {', '.join(sorted(passed))}"
        if self_reviews:
            detail += f" (ignored, not independent: {', '.join(sorted(self_reviews))})"
        return True, f"{detail}. Authors derived from the commits: {', '.join(authors)}"

    reasons = []
    if unfinished:
        reasons.append(f"reviews recorded but not concluded: {', '.join(sorted(unfinished))}")
    if self_reviews:
        reasons.append(f"only self-reviews found: {', '.join(sorted(self_reviews))}")
    if not reasons:
        reasons.append("no adversarial-review verdict has been recorded on this head sha")
    return False, (
        "; ".join(reasons)
        + f". Authors derived from the commits: {', '.join(authors)}. An independent "
        f"session records a verdict with: {dispatch_hint}"
    )


def _load(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="independent adversarial review, measured")
    sub = parser.add_subparsers(dest="command", required=True)

    authors = sub.add_parser("authors", help="identities derived from the commits under review")
    authors.add_argument("--commits", required=True)

    context = sub.add_parser("context", help="the status context one reviewer may write")
    context.add_argument("--commits", required=True)
    context.add_argument("--reviewer", required=True)

    gate = sub.add_parser("gate", help="does this head sha carry an independent passing review")
    gate.add_argument("--commits", required=True)
    gate.add_argument("--statuses", required=True)
    gate.add_argument("--dispatch-hint", default="gh workflow run 'Adversarial Review Gate'")

    args = parser.parse_args(argv)
    try:
        if args.command == "authors":
            print("\n".join(author_identities(_load(args.commits))))
            return 0
        if args.command == "context":
            reviewer = check_reviewer(args.reviewer, author_identities(_load(args.commits)))
            print(f"{STATUS_CONTEXT_PREFIX}{reviewer}")
            return 0
        ok, detail = evaluate(
            _load(args.commits),
            _load(args.statuses),
            dispatch_hint=args.dispatch_hint,
        )
        print(detail)
        return 0 if ok else 1
    except ReviewError as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
