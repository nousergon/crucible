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

from pydantic import ValidationError

from crucible.gate import REVIEW_SCHEMA_VERSION
from crucible.keys import REVIEWER_PATTERN, review_key
from crucible.models import GitHubCommit, ReviewDocument

__all__ = [
    "COMMITS_ENDPOINT_CAP",
    "REVIEW_RECORD_JOB",
    "ReviewError",
    "author_identities",
    "check_reviewer",
    "record",
    "review_document",
    "review_record_handler",
]

#: The job name registered in `crucible.cli.JOBS`, `crucible/components.yaml`
#: and `crucible.models.JOB_VALUES` — a constant imported by `cli.py` rather
#: than a literal restated at each of those sites, the same shape
#: `crucible.integration_summary.INTEGRATION_TEST_JOB` uses.
REVIEW_RECORD_JOB = "review.record"

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
    for raw in commits:
        # `alpha-engine-config-I10045` row 13: validated through
        # `crucible.models.GitHubCommit` instead of `.get(..., {}) or {}`
        # walked by hand at three levels — a typo'd key at any level used
        # to resolve silently to "no identity" instead of surfacing.
        parsed = GitHubCommit.model_validate(raw)
        identities.update(m.lower() for m in SESSION_TRAILER_RE.findall(parsed.commit.message))
        for git_identity in (parsed.commit.author, parsed.commit.committer):
            email = ((git_identity.email if git_identity else None) or "").strip()
            if email:
                identities.add(email.lower())
        for account in (parsed.author, parsed.committer):
            login = ((account.login if account else None) or "").strip()
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
    document = {
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
    # `alpha-engine-config-I10045` row 13: a final, defense-in-depth check
    # against `crucible.models.ReviewDocument`, ADDED AFTER the two checks
    # above rather than replacing them — both already raise `ReviewError`
    # with their own tested message text
    # (`tests/test_review.py::test_a_third_verdict_is_refused`/
    # `test_a_review_that_names_no_commit_is_refused`) and fire first for
    # the cases they cover. This catches what those two do not: an empty
    # `phase`, a malformed `reviewer`, an empty `authors` list, a
    # wrong-typed `pr_number`.
    try:
        ReviewDocument.model_validate(document)
    except ValidationError as exc:
        raise ReviewError(f"the constructed review document does not conform: {exc}") from exc
    return document


def record(ctx: Any, *, document: dict[str, Any]) -> str:
    """File ``document`` through ``ctx`` and return its key.

    Takes a :class:`~crucible.runner.RunContext`, not a bare
    :class:`~crucible.store.Store` (`alpha-engine-config-I10968`). This was a
    raw `store.put_bytes` reached from this module's own `__main__`, so the
    one producer of the artifact `crucible.gate._clause_independently_reviewed`
    grades wrote no run manifest on either path: no lineage, no spend, no code
    sha, and nothing that said it had run. `ctx.record_output` performs the
    same single write AND enters it into `outputs[]`, so the write is the
    manifest's own evidence rather than a side effect beside it.

    No read-modify-write and no overwrite guard, deliberately: the verdict is
    a key segment, so a `pass` cannot land on a `fail`'s object under any
    caller, including one that never imports this module. See the module
    docstring.

    The trading day comes from ``ctx``, never from a second resolution here: a
    run launched on a Saturday binds to Friday's close (rule 3), and the
    manifest and the artifact it records must name the same day.
    """
    key = review_key(
        document["phase"],
        ctx.trading_day.isoformat(),
        document["reviewer"],
        document["verdict"],
    )
    ctx.record_output(
        key,
        json.dumps(document, indent=2, sort_keys=True).encode("utf-8"),
        schema_version=REVIEW_SCHEMA_VERSION,
    )
    return key


def review_record_handler(args: argparse.Namespace) -> int:
    """`crucible review.record --commits F --reviewer S --phase P ...`.

    The independence comparison happens INSIDE the job body, so a self-review
    produces a `failed` manifest naming the refusal rather than no record at
    all: "a review was attempted and refused" and "no review was attempted"
    are different facts, and only the manifest can tell them apart. The
    workflow still runs `python -m crucible.review check` BEFORE it asks for
    AWS credentials — that verb writes nothing and is what keeps a session
    that may not record a verdict from ever holding a token that could write
    one.
    """
    from crucible.cli import _resolve_store
    from crucible.runner import run_job

    store = _resolve_store(args)
    commits = json.loads(Path(args.commits).read_text(encoding="utf-8"))
    printed: list[str] = []

    def job(ctx: Any) -> None:
        document = review_document(
            phase=args.phase,
            verdict=args.verdict,
            reviewer=args.reviewer,
            commits=commits,
            head_sha=args.head_sha,
            pr_number=args.pr_number,
            summary=args.summary,
            reviewed_at=dt.datetime.now(dt.UTC),
        )
        printed.append(record(ctx, document=document))

    try:
        run_job(
            REVIEW_RECORD_JOB,
            job,
            store=store,
            trading_day=args.trading_day,
            dry_run=bool(getattr(args, "dry_run", False)),
            run_mode=getattr(args, "run_mode", None),
        )
    except ReviewError as exc:
        # NOT a swallow: `run_job` has already written `status: failed` with
        # this cause in its `try/finally`, and the refusal is what the exit
        # code and the `::error::` annotation report. Translating it here is
        # what keeps the workflow's own contract — exit 2, one annotated line,
        # no traceback — while the manifest carries the full record. Any other
        # exception propagates untouched.
        print(f"::error::{exc}", file=sys.stderr)
        return 2
    print(printed[0])
    return 0


def main(argv: list[str] | None = None) -> int:
    """The two verbs that WRITE NOTHING, and only those.

    `record` used to live here too, reached as `python -m crucible.review
    record` with its own `open_store` call — a store writer outside
    `crucible.cli.JOBS` that filed no run manifest
    (`alpha-engine-config-I10968`). It is now the registered job
    :data:`REVIEW_RECORD_JOB`, so the one path that writes the review artifact
    is the one path that writes a manifest, and there is no second entry point
    that could drift from it.

    These two stay here rather than becoming jobs of their own for the reason
    the recording workflow's own step comment states: they must be runnable
    BEFORE the workflow asks for AWS credentials, and a `crucible` job
    resolves a store on the way in.
    """
    parser = argparse.ArgumentParser(
        description="read the identities under review, and prove independence. Writes nothing"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    authors = sub.add_parser("authors", help="identities read out of the commits under review")
    authors.add_argument("--commits", required=True, help="GET /pulls/{n}/commits, as JSON")

    check = sub.add_parser(
        "check",
        help="validate the reviewer and prove independence, writing nothing",
    )
    check.add_argument("--commits", required=True)
    check.add_argument("--reviewer", required=True)

    args = parser.parse_args(argv)
    try:
        commits = json.loads(Path(args.commits).read_text(encoding="utf-8"))
        if args.command == "authors":
            print("\n".join(author_identities(commits)))
            return 0
        # A separate verb so the recording workflow can refuse a self-review
        # BEFORE it asks for AWS credentials: a session that may not record a
        # verdict has no business holding a token that could write one.
        print(check_reviewer(args.reviewer, author_identities(commits)))
        return 0
    except ReviewError as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
