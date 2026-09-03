"""The independence control, made to fire.

`AGENTS.md` test discipline: "a detector nobody has made fail is a detector
nobody knows works." The control this file grades had two defects
(`alpha-engine-config-I9873`), and the second one is the reason every test
here is a REFUSAL: `reviewer` and `author` were free-text `workflow_dispatch`
inputs supplied by the session asking to be passed, so the gate could not
refuse anything. The author set is now derived from the commits under review,
and these tests show that derivation rejecting a self-review it previously had
no way to see.

The module under test lives in `.github/scripts/` rather than in `crucible/`
on purpose: it reads GitHub's commits and statuses APIs, which is CI plumbing
and not part of the harness's public surface. It is loaded here by path so it
still gets a real self-test rather than being the one piece of the control
nobody ever ran.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / ".github" / "scripts" / "adversarial_review.py"
_SPEC = importlib.util.spec_from_file_location("adversarial_review", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
adversarial_review = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(adversarial_review)

AUTHOR_SESSION = "session_01AuthorAAAAAAAA"
REVIEWER_SESSION = "session_01ReviewerBBBBB"
HINT = "gh workflow run ..."


def _commit(
    message: str = f"fix: a thing\n\nClaude-Session: https://claude.ai/code/{AUTHOR_SESSION}\n",
    *,
    email: str = "cipher813@example.invalid",
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


def _status(reviewer: str, state: str, description: str = "") -> dict:
    return {
        "context": f"{adversarial_review.STATUS_CONTEXT_PREFIX}{reviewer.lower()}",
        "state": state,
        "description": description,
    }


class TestAuthorIdentitiesAreDerived:
    def test_the_session_trailer_is_read_out_of_the_commit_message(self) -> None:
        """The whole point: the author is a property of the COMMIT, so a
        dispatcher cannot name someone else and pass its own review."""
        assert AUTHOR_SESSION.lower() in adversarial_review.author_identities([_commit()])

    def test_a_commit_with_no_trailer_still_yields_an_identity(self) -> None:
        """A Dependabot commit carries no session trailer. Falling back to the
        login and the email is what keeps such a PR reviewable instead of
        permanently unmergeable."""
        identities = adversarial_review.author_identities(
            [
                _commit(
                    "chore(deps): bump numpy",
                    email="49699333+dependabot[bot]@users.noreply.invalid",
                    login="dependabot[bot]",
                )
            ]
        )
        assert "dependabot[bot]" in identities

    def test_a_commit_naming_nobody_is_refused_not_passed(self) -> None:
        """An empty author set makes every reviewer independent by
        construction — a pass by accident, which is the failure mode this
        whole change exists to remove."""
        with pytest.raises(adversarial_review.ReviewError, match="names an author"):
            adversarial_review.author_identities(
                [{"commit": {"message": "x", "author": {}, "committer": {}}}]
            )


class TestTheReviewerIsRefused:
    def test_free_text_is_not_a_reviewer_identity(self) -> None:
        with pytest.raises(adversarial_review.ReviewError, match="not a Claude session id"):
            adversarial_review.check_reviewer("the other agent", [AUTHOR_SESSION.lower()])

    def test_a_github_login_is_not_a_reviewer_identity(self) -> None:
        """Every agent in this fleet dispatches as the same collaborator, so a
        login can never separate a reviewer from an author. Accepting one
        would reintroduce a comparison that is equal on every dispatch."""
        with pytest.raises(adversarial_review.ReviewError, match="not a Claude session id"):
            adversarial_review.check_reviewer("cipher813", [AUTHOR_SESSION.lower()])

    def test_the_authoring_session_may_not_review_its_own_change(self) -> None:
        with pytest.raises(adversarial_review.ReviewError, match="independent of the author"):
            adversarial_review.check_reviewer(AUTHOR_SESSION, [AUTHOR_SESSION.lower()])

    def test_case_does_not_launder_a_self_review(self) -> None:
        with pytest.raises(adversarial_review.ReviewError, match="independent of the author"):
            adversarial_review.check_reviewer(AUTHOR_SESSION.upper(), [AUTHOR_SESSION.lower()])

    def test_an_independent_session_is_accepted(self) -> None:
        assert (
            adversarial_review.check_reviewer(REVIEWER_SESSION, [AUTHOR_SESSION.lower()])
            == REVIEWER_SESSION.lower()
        )


class TestTheGateReading:
    def test_no_verdict_at_all_is_not_a_pass(self) -> None:
        ok, detail = adversarial_review.evaluate([_commit()], [], dispatch_hint=HINT)
        assert not ok
        assert "no adversarial-review verdict" in detail
        assert HINT in detail

    def test_a_self_review_does_not_satisfy_the_gate(self) -> None:
        """The exact bypass the free-text inputs allowed: the authoring session
        recording its own success."""
        ok, detail = adversarial_review.evaluate(
            [_commit()], [_status(AUTHOR_SESSION, "success")], dispatch_hint=HINT
        )
        assert not ok
        assert "only self-reviews found" in detail

    def test_an_independent_fail_is_not_cleared_by_another_reviewers_pass(self) -> None:
        """A reviewer who found blocking defects is evidence about the code; a
        second reviewer who did not look at the same thing is not a rebuttal."""
        ok, detail = adversarial_review.evaluate(
            [_commit()],
            [
                _status(REVIEWER_SESSION, "failure", "3 findings"),
                _status("session_01ThirdCCCCCCCC", "success"),
            ],
            dispatch_hint=HINT,
        )
        assert not ok
        assert "recorded FAIL" in detail
        assert "3 findings" in detail

    def test_a_pending_review_is_not_a_pass(self) -> None:
        ok, detail = adversarial_review.evaluate(
            [_commit()], [_status(REVIEWER_SESSION, "pending")], dispatch_hint=HINT
        )
        assert not ok
        assert "not concluded" in detail

    def test_an_unrelated_status_context_is_ignored(self) -> None:
        """A status posted by anything else on the same sha must not be read as
        a review — the context prefix is the whole selector."""
        ok, _ = adversarial_review.evaluate(
            [_commit()],
            [{"context": "continuous-integration/whatever", "state": "success"}],
            dispatch_hint=HINT,
        )
        assert not ok

    def test_an_independent_pass_satisfies_the_gate(self) -> None:
        ok, detail = adversarial_review.evaluate(
            [_commit()], [_status(REVIEWER_SESSION, "success")], dispatch_hint=HINT
        )
        assert ok
        assert REVIEWER_SESSION.lower() in detail


class TestTheCommandLine:
    def _write(self, tmp_path: Path, name: str, payload: object) -> str:
        import json

        path = tmp_path / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return str(path)

    def test_gate_exits_non_zero_with_no_review(self, tmp_path, capsys) -> None:
        code = adversarial_review.main(
            [
                "gate",
                "--commits",
                self._write(tmp_path, "c.json", [_commit()]),
                "--statuses",
                self._write(tmp_path, "s.json", []),
            ]
        )
        assert code == 1
        assert "no adversarial-review verdict" in capsys.readouterr().out

    def test_gate_exits_zero_on_an_independent_pass(self, tmp_path) -> None:
        code = adversarial_review.main(
            [
                "gate",
                "--commits",
                self._write(tmp_path, "c.json", [_commit()]),
                "--statuses",
                self._write(tmp_path, "s.json", [_status(REVIEWER_SESSION, "success")]),
            ]
        )
        assert code == 0

    def test_context_exits_two_on_a_self_review(self, tmp_path, capsys) -> None:
        """Exit 2, not 1: "the reviewer is not allowed" and "the gate is not
        satisfied yet" are different answers, and the recording job must not
        record anything on the first."""
        code = adversarial_review.main(
            [
                "context",
                "--commits",
                self._write(tmp_path, "c.json", [_commit()]),
                "--reviewer",
                AUTHOR_SESSION,
            ]
        )
        assert code == 2
        assert "independent of the author" in capsys.readouterr().err

    def test_context_prints_the_status_context_for_an_independent_reviewer(
        self, tmp_path, capsys
    ) -> None:
        code = adversarial_review.main(
            [
                "context",
                "--commits",
                self._write(tmp_path, "c.json", [_commit()]),
                "--reviewer",
                REVIEWER_SESSION,
            ]
        )
        assert code == 0
        assert capsys.readouterr().out.strip() == (
            f"{adversarial_review.STATUS_CONTEXT_PREFIX}{REVIEWER_SESSION.lower()}"
        )
