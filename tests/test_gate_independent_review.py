"""The phase gate can read the independent adversarial review — and refuse it.

`alpha-engine-config-I9794`. Until this clause existed the review lived only
as a GitHub check-run, so `crucible gate` — the thing that decides whether a
phase may exit — had no way to see it, and a phase could exit with no review
having happened (plan §11 risk 1: "v2 passes its gates because its tests were
written to pass").

Every test here is a REFUSAL except the last one in each class. A clause that
has only been shown accepting a valid artifact has not been shown to
constrain anything.

Fixed date literals throughout, never `today` arithmetic: FRIDAY is the same
2026-08-28 session `tests/test_gate.py` anchors on, so the two files grade the
same window.
"""

from __future__ import annotations

import datetime as dt
import json

from crucible.gate import REVIEW_SCHEMA_VERSION, evaluate
from crucible.keys import review_key, review_prefix
from crucible.store import LocalStore

FRIDAY = dt.date(2026, 8, 28)
WINDOW = [FRIDAY - dt.timedelta(weeks=n) for n in reversed(range(5))]
BEFORE_THE_WINDOW = FRIDAY - dt.timedelta(weeks=9)
HEAD_SHA = "b" * 40
AUTHOR = "session_01AuthorAAAAAAAA"
REVIEWER = "session_01ReviewerBBBBB"


def review_document(
    *,
    verdict: str = "pass",
    reviewer: str = REVIEWER,
    authors: list[str] | None = None,
    head_sha: str = HEAD_SHA,
    schema_version: str = REVIEW_SCHEMA_VERSION,
) -> dict:
    return {
        "schema_version": schema_version,
        "phase": "phase1",
        "verdict": verdict,
        "reviewer": reviewer,
        "authors": authors if authors is not None else [AUTHOR, "cipher813"],
        "pr_number": 46,
        "head_sha": head_sha,
        "summary": "checked against plan §2",
        "reviewed_at": "2026-08-28T18:00:00Z",
    }


def file_review(store: LocalStore, day: dt.date, document: dict) -> str:
    key = review_key("phase1", day.isoformat(), document["reviewer"])
    store.put_bytes(key, json.dumps(document).encode("utf-8"))
    return key


def clause(store: LocalStore):
    result = evaluate(store, gate="phase1", trading_day=FRIDAY)
    return next(c for c in result.clauses if c.name == "independently_reviewed")


class TestTheClauseIsPartOfThePhaseGate:
    def test_phase1_grades_the_independent_review(self, tmp_path) -> None:
        result = evaluate(LocalStore(tmp_path), gate="phase1", trading_day=FRIDAY)
        assert "independently_reviewed" in [c.name for c in result.clauses]


class TestTheClauseRefuses:
    def test_no_review_artifact_at_all_is_unmet_with_the_prefix_named(self, tmp_path) -> None:
        """`no data` is never a pass (principle 7), and the operator's next
        action — which prefix nothing has been filed under — is in the
        output."""
        found = clause(LocalStore(tmp_path))
        assert not found.met
        assert review_prefix("phase1") in found.detail

    def test_a_self_review_does_not_satisfy_the_clause(self, tmp_path) -> None:
        """The defect the whole redesign is about: the reviewer being one of
        the identities derived from the commits under review."""
        store = LocalStore(tmp_path)
        file_review(store, FRIDAY, review_document(reviewer=AUTHOR, authors=[AUTHOR]))
        found = clause(store)
        assert not found.met
        assert "self-review" in found.detail

    def test_case_does_not_launder_a_self_review(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        file_review(store, FRIDAY, review_document(reviewer=AUTHOR, authors=[AUTHOR.upper()]))
        assert not clause(store).met

    def test_a_review_from_before_the_window_does_not_satisfy_a_later_exit(self, tmp_path) -> None:
        """A review of a superseded state of the code is not a review of this
        one — the stale-artifact half of `_clause_pointer_flipped_on_smoke`'s
        shape."""
        store = LocalStore(tmp_path)
        file_review(store, BEFORE_THE_WINDOW, review_document())
        found = clause(store)
        assert not found.met
        assert "outside this gate's window" in found.detail

    def test_an_independent_fail_makes_the_clause_unmet(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        file_review(store, FRIDAY, review_document(verdict="fail"))
        found = clause(store)
        assert not found.met
        assert "recorded `fail`" in found.detail

    def test_a_second_reviewers_pass_does_not_clear_a_fail(self, tmp_path) -> None:
        """Findings are answered by fixing them and filing a new round, never
        by finding a reviewer who did not look."""
        store = LocalStore(tmp_path)
        file_review(store, FRIDAY, review_document(verdict="fail"))
        file_review(
            store, FRIDAY, review_document(reviewer="session_01ThirdCCCCCCCC", verdict="pass")
        )
        assert not clause(store).met

    def test_an_empty_author_list_is_refused_not_treated_as_independent(self, tmp_path) -> None:
        """An empty author set makes every reviewer independent by
        construction — a pass by accident."""
        store = LocalStore(tmp_path)
        file_review(store, FRIDAY, review_document(authors=[]))
        found = clause(store)
        assert not found.met
        assert "non-empty list of identities" in found.detail

    def test_a_review_naming_no_commit_is_refused(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        file_review(store, FRIDAY, review_document(head_sha="not-a-sha"))
        found = clause(store)
        assert not found.met
        assert "40-hex commit sha" in found.detail

    def test_an_unknown_schema_version_is_refused_rather_than_guessed_at(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        file_review(store, FRIDAY, review_document(schema_version="review.v0"))
        found = clause(store)
        assert not found.met
        assert "schema_version" in found.detail

    def test_an_unreadable_document_is_red_not_absent(self, tmp_path) -> None:
        """ "No review was filed" and "the review we have is corrupt" name
        different remedies, and collapsing them reports the wrong one."""
        store = LocalStore(tmp_path)
        store.put_bytes(review_key("phase1", FRIDAY.isoformat(), REVIEWER), b"{not json")
        found = clause(store)
        assert not found.met
        assert "not readable JSON" in found.detail


class TestTheClauseAccepts:
    def test_an_independent_pass_in_the_window_meets_the_clause(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        key = file_review(store, FRIDAY, review_document())
        found = clause(store)
        assert found.met, found.detail
        assert key in found.evidence
        assert REVIEWER.lower() in found.detail

    def test_a_pass_on_any_session_in_the_window_counts(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        file_review(store, WINDOW[1], review_document())
        assert clause(store).met


class TestTheKeyShape:
    def test_two_reviewers_on_one_session_do_not_overwrite_each_other(self) -> None:
        """One key per `(phase, trading_day)` would make the surviving verdict
        whichever execution finished second."""
        first = review_key("phase1", FRIDAY.isoformat(), REVIEWER)
        second = review_key("phase1", FRIDAY.isoformat(), "session_01ThirdCCCCCCCC")
        assert first != second
        assert first.startswith(review_prefix("phase1"))

    def test_a_reviewer_that_is_not_a_session_id_is_refused(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="not a Claude session id"):
            review_key("phase1", FRIDAY.isoformat(), "../../etc/passwd")
