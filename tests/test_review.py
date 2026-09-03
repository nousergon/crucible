"""The independence control, made to fire.

`AGENTS.md` test discipline: "a detector nobody has made fail is a detector
nobody knows works." Almost every test here is a REFUSAL, because the control
this module carries could previously refuse nothing at all: `reviewer` and
`author` were free-text `workflow_dispatch` inputs supplied by the session
asking to be passed (`alpha-engine-config-I9873`). The author set is now read
out of the commits under review, and these tests show that derivation
rejecting a self-review it had no way to see.

What is deliberately NOT tested here, because it is not claimed: that a
dishonest recorder can be stopped. `crucible/review.py`'s docstring states the
residual — both sides rest on self-declaration, there is no session
attestation in this fleet, and a determined bypass passes. These tests grade
the careless case, which is the one that actually happens.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from crucible.gate import REVIEW_SCHEMA_VERSION
from crucible.keys import review_key
from crucible.review import (
    COMMITS_ENDPOINT_CAP,
    ReviewError,
    author_identities,
    check_reviewer,
    record,
    review_document,
)
from crucible.store import LocalStore

AUTHOR = "session_01AuthorAAAAAAAA"
REVIEWER = "session_01ReviewerBBBBB"
HEAD_SHA = "b" * 40
FRIDAY = dt.date(2026, 8, 28)
REVIEWED_AT = dt.datetime(2026, 8, 28, 18, 0, tzinfo=dt.UTC)


def commit(
    message: str = f"fix: a thing\n\nClaude-Session: https://claude.ai/code/{AUTHOR}\n",
    *,
    email: str = "someone@example.invalid",
    login: str | None = "cipher813",
) -> dict:
    return {
        "commit": {
            "message": message,
            "author": {"email": email},
            "committer": {"email": email},
        },
        "author": {"login": login} if login else None,
        "committer": {"login": login} if login else None,
    }


def document(**overrides) -> dict:
    kwargs = {
        "phase": "phase1",
        "verdict": "pass",
        "reviewer": REVIEWER,
        "commits": [commit()],
        "head_sha": HEAD_SHA,
        "pr_number": 48,
        "summary": "no findings against plan section 2",
        "reviewed_at": REVIEWED_AT,
    }
    kwargs.update(overrides)
    return review_document(**kwargs)


class TestAuthorIdentitiesAreReadOutOfTheCommits:
    def test_the_session_trailer_is_read_from_the_commit(self) -> None:
        """The whole point: the author is a property of the COMMIT, so a
        recorder cannot name someone else and pass its own review."""
        assert AUTHOR.lower() in author_identities([commit()])

    def test_only_the_trailer_counts_not_the_commit_body(self) -> None:
        """A review round routinely QUOTES the session id it is answering. A
        free scan of the message put that reviewer into the author set and
        blocked it on that change forever, with a message saying it had
        written the code."""
        quoting = commit(
            "fix: address round 1\n\n"
            f"Addresses the review by {REVIEWER}, which found three things.\n\n"
            f"Claude-Session: https://claude.ai/code/{AUTHOR}\n"
        )
        identities = author_identities([quoting])
        assert AUTHOR.lower() in identities
        assert REVIEWER.lower() not in identities

    def test_a_commit_with_no_trailer_still_yields_an_identity(self) -> None:
        """A Dependabot commit carries no session trailer. Falling back to the
        login and the email keeps such a change reviewable."""
        identities = author_identities(
            [
                commit(
                    "chore(deps): bump numpy",
                    email="49699333+dependabot[bot]@users.noreply.invalid",
                    login="dependabot[bot]",
                )
            ]
        )
        assert "dependabot[bot]" in identities

    def test_a_commit_naming_nobody_is_refused_not_passed(self) -> None:
        """An empty author set makes every reviewer independent by
        construction — a pass by accident."""
        with pytest.raises(ReviewError, match="names an author"):
            author_identities([{"commit": {"message": "x", "author": {}, "committer": {}}}])

    def test_a_commit_list_at_githubs_page_cap_is_refused_not_truncated(self) -> None:
        """`/pulls/{n}/commits` caps at 250 and pages no further, so beyond it
        the author set is silently PARTIAL — and a missing author makes a
        reviewer look independent. That fails open in the only direction that
        matters."""
        with pytest.raises(ReviewError, match="silently partial"):
            author_identities([commit()] * COMMITS_ENDPOINT_CAP)


class TestTheReviewerIsRefused:
    def test_free_text_is_not_a_reviewer_identity(self) -> None:
        with pytest.raises(ReviewError, match="not a Claude session id"):
            check_reviewer("the other agent", [AUTHOR.lower()])

    def test_a_github_login_is_not_a_reviewer_identity(self) -> None:
        """Every agent in this fleet dispatches as the same collaborator, so a
        login can never separate a reviewer from an author. Accepting one
        reintroduces a comparison that is equal on every dispatch."""
        with pytest.raises(ReviewError, match="not a Claude session id"):
            check_reviewer("cipher813", [AUTHOR.lower()])

    def test_the_authoring_session_may_not_review_its_own_change(self) -> None:
        with pytest.raises(ReviewError, match="independent of the author"):
            check_reviewer(AUTHOR, [AUTHOR.lower()])

    def test_case_does_not_launder_a_self_review(self) -> None:
        with pytest.raises(ReviewError, match="independent of the author"):
            check_reviewer(AUTHOR.upper(), [AUTHOR.lower()])

    def test_an_independent_session_is_accepted_and_normalised(self) -> None:
        assert check_reviewer(REVIEWER.upper(), [AUTHOR.lower()]) == REVIEWER.lower()


class TestTheReviewerContractHasNotDrifted:
    def test_the_key_module_and_this_module_share_one_pattern(self) -> None:
        """`crucible/keys.py`'s entire premise. The first version of this
        control had two copies of the reviewer regex with DIFFERENT flags and
        a comment claiming they matched, so `SESSION_01AB...` passed the
        workflow and would have raised in `review_key` the moment the writer
        landed — a contract restated twice, already drifted inside the change
        that introduced it."""
        from crucible import keys, review

        assert review._REVIEWER_RE.pattern == keys._REVIEWER_RE.pattern
        assert review._REVIEWER_RE.flags == keys._REVIEWER_RE.flags

    def test_one_spelling_of_a_session_reaches_one_key(self) -> None:
        assert review_key("phase1", "2026-08-28", REVIEWER.upper(), "pass") == review_key(
            "phase1", "2026-08-28", REVIEWER.lower(), "pass"
        )

    def test_a_reviewer_that_is_not_a_session_id_never_becomes_a_key(self) -> None:
        with pytest.raises(ValueError, match="not a Claude session id"):
            review_key("phase1", "2026-08-28", "../../etc/passwd", "pass")


class TestTheDocument:
    def test_a_document_names_what_it_reviewed(self) -> None:
        assert document()["head_sha"] == HEAD_SHA

    def test_a_review_that_names_no_commit_is_refused(self) -> None:
        with pytest.raises(ReviewError, match="40-hex commit sha"):
            document(head_sha="not-a-sha")

    def test_a_third_verdict_is_refused(self) -> None:
        """Rule 2's shape: `pass` or `fail`, and no API for a third. A
        `partial` here would be a key the gate clause never agreed to read."""
        with pytest.raises(ReviewError, match="not 'pass' or 'fail'"):
            document(verdict="partial")

    def test_a_self_review_never_becomes_a_document(self) -> None:
        with pytest.raises(ReviewError, match="independent of the author"):
            document(reviewer=AUTHOR)

    def test_the_authors_are_the_derived_set_not_an_input(self) -> None:
        assert document()["authors"] == author_identities([commit()])


class TestAnAdverseVerdictIsDurable:
    def test_a_pass_cannot_overwrite_a_fail_from_the_same_reviewer(self, tmp_path) -> None:
        """The defect: with the verdict only in the BODY, a reviewer that
        recorded `fail` could re-record `success` on the same key and erase its
        own finding — one dispatch, zero code change, and the phase gate goes
        green on a change nobody re-reviewed. The verdict is a key segment, so
        the two are different objects."""
        store = LocalStore(tmp_path)
        failed = record(store, document=document(verdict="fail"), trading_day=FRIDAY)
        passed = record(store, document=document(verdict="pass"), trading_day=FRIDAY)
        assert failed != passed
        assert store.exists(failed)
        assert json.loads(store.get_bytes(failed))["verdict"] == "fail"

    def test_two_reviewers_on_one_session_do_not_overwrite_each_other(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        first = record(store, document=document(), trading_day=FRIDAY)
        second = record(
            store,
            document=document(reviewer="session_01ThirdCCCCCCCC"),
            trading_day=FRIDAY,
        )
        assert first != second
        assert store.exists(first) and store.exists(second)

    def test_the_filed_document_carries_the_schema_the_gate_reads(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        key = record(store, document=document(), trading_day=FRIDAY)
        assert json.loads(store.get_bytes(key))["schema_version"] == REVIEW_SCHEMA_VERSION


class TestTheCommandLine:
    def _commits(self, tmp_path, payload) -> str:
        path = tmp_path / "commits.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return str(path)

    def test_check_exits_two_on_a_self_review_and_writes_nothing(self, tmp_path, capsys) -> None:
        """Exit 2, not 1: the recording workflow runs `check` BEFORE it asks
        for AWS credentials, so a session that may not record a verdict never
        holds a token that could write one."""
        from crucible.review import main

        code = main(
            ["check", "--commits", self._commits(tmp_path, [commit()]), "--reviewer", AUTHOR]
        )
        assert code == 2
        assert "independent of the author" in capsys.readouterr().err

    def test_check_prints_the_normalised_reviewer(self, tmp_path, capsys) -> None:
        from crucible.review import main

        code = main(
            ["check", "--commits", self._commits(tmp_path, [commit()]), "--reviewer", REVIEWER]
        )
        assert code == 0
        assert capsys.readouterr().out.strip() == REVIEWER.lower()

    def test_record_files_the_artifact_and_prints_its_key(self, tmp_path, capsys) -> None:
        from crucible.review import main

        store_dir = tmp_path / "store"
        code = main(
            [
                "record",
                "--commits",
                self._commits(tmp_path, [commit()]),
                "--reviewer",
                REVIEWER,
                "--phase",
                "phase1",
                "--verdict",
                "pass",
                "--head-sha",
                HEAD_SHA,
                "--pr-number",
                "48",
                "--summary",
                "no findings",
                "--store",
                str(store_dir),
            ]
        )
        assert code == 0
        key = capsys.readouterr().out.strip()
        assert key.startswith("reviews/phase1/")
        assert key.endswith(f"/{REVIEWER.lower()}/pass.json")
        assert LocalStore(store_dir).exists(key)

    def test_record_refuses_a_self_review_before_touching_the_store(self, tmp_path, capsys) -> None:
        from crucible.review import main

        store_dir = tmp_path / "store"
        code = main(
            [
                "record",
                "--commits",
                self._commits(tmp_path, [commit()]),
                "--reviewer",
                AUTHOR,
                "--phase",
                "phase1",
                "--verdict",
                "pass",
                "--head-sha",
                HEAD_SHA,
                "--pr-number",
                "48",
                "--summary",
                "no findings",
                "--store",
                str(store_dir),
            ]
        )
        assert code == 2
        assert "independent of the author" in capsys.readouterr().err
        assert not list(LocalStore(store_dir).list_keys("reviews/"))
