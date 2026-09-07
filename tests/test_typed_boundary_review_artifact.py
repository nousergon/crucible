"""The review artifact's two boundaries (`alpha-engine-config-I10045` row 13):
the GitHub commits API payload `crucible.review.author_identities` reads,
and the `review.v1` document `crucible.review.review_document` writes.

Normative source: `alpha-engine-config-I10045`; `crucible.models.
GitHubCommit`/`ReviewDocument`'s docstrings.

`crucible.gate._review_problem` (the READ side of `review.v1`) is
deliberately UNCHANGED by this PR — see `ReviewDocument`'s docstring for
why — so this file does not touch it.
"""

from __future__ import annotations

import datetime as dt

import pytest
from pydantic import ValidationError

from crucible.models import GitHubCommit, ReviewDocument
from crucible.review import ReviewError, author_identities, review_document

AUTHOR = "session_01AuthorAAAAAAAA"
REVIEWER = "session_01ReviewerBBBBB"
HEAD_SHA = "b" * 40
REVIEWED_AT = dt.datetime(2026, 8, 28, 18, 0, tzinfo=dt.UTC)


def _commit(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "commit": {
            "message": f"fix: a thing\n\nClaude-Session: https://claude.ai/code/{AUTHOR}\n",
            "author": {"email": "someone@example.invalid"},
            "committer": {"email": "someone@example.invalid"},
        },
        "author": {"login": "cipher813"},
        "committer": {"login": "cipher813"},
    }
    payload.update(overrides)
    return payload


class TestGitHubCommitTypesTheRealShape:
    def test_a_real_commit_still_yields_the_session_trailer(self) -> None:
        assert AUTHOR.lower() in author_identities([_commit()])

    def test_an_unknown_field_a_real_commits_payload_always_carries_is_tolerated(self) -> None:
        """A real `GET /pulls/{n}/commits` row carries dozens of fields
        (`sha`, `url`, `stats`, `files`...) this reader never looks at."""
        parsed = GitHubCommit.model_validate(
            _commit(sha="abc123", url="https://api.github.com/...", stats={"total": 4})
        )
        assert parsed.model_extra["sha"] == "abc123"

    def test_a_commit_with_no_github_account_at_all_still_yields_an_identity(self) -> None:
        """`author`/`committer` can be explicitly `null` when GitHub cannot
        associate the commit with an account — a real, common shape."""
        identities = author_identities(
            [_commit(author=None, committer=None)],
        )
        assert AUTHOR.lower() in identities


class TestASilentTypoInTheCommitShapeIsNowVisible:
    def test_a_typo_d_nested_key_does_not_silently_supply_an_email(self) -> None:
        parsed = GitHubCommit.model_validate(
            _commit(commit={"message": "x", "authr": {"email": "typo@example.invalid"}})
        )
        # The typo'd key ("authr") is simply absent from the declared
        # `author` field, not silently populating it -- `commit.author` is
        # None, matching main's `.get("author") or {}` returning `{}` for
        # exactly this input, but now VISIBLE as an absent identity rather
        # than a defaulted one.
        assert parsed.commit.author is None


class TestReviewDocumentCatchesWhatThePreExistingChecksDoNot:
    def test_an_empty_phase_is_refused(self) -> None:
        with pytest.raises(ReviewError, match="does not conform"):
            review_document(
                phase="",
                verdict="pass",
                reviewer=REVIEWER,
                commits=[_commit()],
                head_sha=HEAD_SHA,
                pr_number=48,
                summary="",
                reviewed_at=REVIEWED_AT,
            )

    def test_a_wrongly_typed_pr_number_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="pr_number"):
            ReviewDocument.model_validate(
                {
                    "schema_version": "review.v1",
                    "phase": "phase1",
                    "verdict": "pass",
                    "reviewer": REVIEWER,
                    "authors": ["session_01someoneAAAA"],
                    "pr_number": "not a number",
                    "head_sha": HEAD_SHA,
                    "summary": "",
                    "reviewed_at": REVIEWED_AT.isoformat(),
                }
            )

    def test_pre_existing_checks_still_fire_first_with_their_own_text(self) -> None:
        """The two checks `review_document` already had keep their exact
        tested message text -- this row does not touch them."""
        with pytest.raises(ReviewError, match="not 'pass' or 'fail'"):
            review_document(
                phase="phase1",
                verdict="maybe",
                reviewer=REVIEWER,
                commits=[_commit()],
                head_sha=HEAD_SHA,
                pr_number=48,
                summary="",
                reviewed_at=REVIEWED_AT,
            )
        with pytest.raises(ReviewError, match="40-hex commit sha"):
            review_document(
                phase="phase1",
                verdict="pass",
                reviewer=REVIEWER,
                commits=[_commit()],
                head_sha="not-a-sha",
                pr_number=48,
                summary="",
                reviewed_at=REVIEWED_AT,
            )

    def test_a_well_formed_call_still_builds_the_same_document(self) -> None:
        document = review_document(
            phase="phase1",
            verdict="pass",
            reviewer=REVIEWER,
            commits=[_commit()],
            head_sha=HEAD_SHA,
            pr_number=48,
            summary="no findings",
            reviewed_at=REVIEWED_AT,
        )
        assert document["reviewer"] == REVIEWER.lower()
        assert AUTHOR.lower() in document["authors"]
