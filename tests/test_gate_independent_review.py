"""The phase gate can read the independent adversarial review — and refuse it.

`alpha-engine-config-I9794`. Until this clause existed the review lived only
as a GitHub check-run, so `crucible gate` — the thing that decides whether a
phase may exit — had no way to see it, and a phase could exit with no review
having happened (plan §11 risk 1: "v2 passes its gates because its tests were
written to pass").

Since Brian's 2026-09-03 ruling this clause is the ONLY place the review is
enforced: `adversarial-review-gate` is never armed as a required status check,
because a `skipped` check-run under a required name counts as success and the
graded party could green it by pressing a dispatch button. A phase gate cannot
be pressed.

Every test here is a refusal except the accepting class at the end. A clause
that has only been shown accepting a valid artifact has not been shown to
constrain anything.

The documents come from `crucible.review.review_document` — the real producer,
not a hand-built dict — so a producer/consumer drift fails these tests rather
than surviving as two field lists that agree today.

Fixed date literals throughout, never `today` arithmetic: FRIDAY is the same
2026-08-28 session `tests/test_gate.py` anchors on, so the two files grade the
same window.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from crucible.gate import REVIEW_SCHEMA_VERSION, evaluate
from crucible.keys import review_key, review_prefix
from crucible.review import review_document
from crucible.store import LocalStore

FRIDAY = dt.date(2026, 8, 28)
WINDOW = [FRIDAY - dt.timedelta(weeks=n) for n in reversed(range(5))]
BEFORE_THE_WINDOW = FRIDAY - dt.timedelta(weeks=9)
HEAD_SHA = "b" * 40
OTHER_SHA = "d" * 40
AUTHOR = "session_01AuthorAAAAAAAA"
REVIEWER = "session_01ReviewerBBBBB"
REVIEWED_AT = dt.datetime(2026, 8, 28, 18, 0, tzinfo=dt.UTC)

COMMITS = [
    {
        "commit": {
            "message": f"fix: a thing\n\nClaude-Session: https://claude.ai/code/{AUTHOR}\n",
            "author": {"email": "someone@example.invalid"},
            "committer": {"email": "someone@example.invalid"},
        },
        "author": {"login": "cipher813"},
        "committer": {"login": "cipher813"},
    }
]


def review(**overrides) -> dict:
    kwargs = {
        "phase": "phase1",
        "verdict": "pass",
        "reviewer": REVIEWER,
        "commits": COMMITS,
        "head_sha": HEAD_SHA,
        "pr_number": 48,
        "summary": "checked against plan section 2",
        "reviewed_at": REVIEWED_AT,
    }
    kwargs.update(overrides)
    return review_document(**kwargs)


def file_review(store: LocalStore, day: dt.date, document: dict) -> str:
    key = review_key("phase1", day.isoformat(), document["reviewer"], document["verdict"])
    store.put_bytes(key, json.dumps(document).encode("utf-8"))
    return key


def file_raw(store: LocalStore, day: dt.date, reviewer: str, verdict: str, body: dict) -> str:
    """File a document the real producer would never emit.

    Used only to grade the clause's refusals of a MALFORMED artifact — a shape
    `review_document` cannot produce, which is exactly why the clause must
    still refuse it rather than assume its own producer wrote everything in
    the prefix.
    """
    key = review_key("phase1", day.isoformat(), reviewer, verdict)
    store.put_bytes(key, json.dumps(body).encode("utf-8"))
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
        """The producer refuses to build one, so this files the shape directly
        to prove the READER refuses it too. A clause that trusted its producer
        would be satisfied by anything anyone put in the prefix."""
        store = LocalStore(tmp_path)
        file_raw(
            store,
            FRIDAY,
            AUTHOR,
            "pass",
            {
                "schema_version": REVIEW_SCHEMA_VERSION,
                "phase": "phase1",
                "verdict": "pass",
                "reviewer": AUTHOR.lower(),
                "authors": [AUTHOR.lower(), "cipher813"],
                "pr_number": 48,
                "head_sha": HEAD_SHA,
                "summary": "looks fine to me",
                "reviewed_at": REVIEWED_AT.isoformat(),
            },
        )
        found = clause(store)
        assert not found.met
        assert "self-review" in found.detail

    def test_a_review_from_before_the_window_does_not_satisfy_a_later_exit(self, tmp_path) -> None:
        """A review of a superseded state of the code is not a review of this
        one."""
        store = LocalStore(tmp_path)
        file_review(store, BEFORE_THE_WINDOW, review())
        found = clause(store)
        assert not found.met
        assert "outside this gate's window" in found.detail

    def test_an_independent_fail_makes_the_clause_unmet(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        file_review(store, FRIDAY, review(verdict="fail", summary="3 findings"))
        found = clause(store)
        assert not found.met
        assert "recorded `fail`" in found.detail
        assert "3 findings" in found.detail

    def test_a_pass_on_the_same_sha_does_not_clear_a_fail(self, tmp_path) -> None:
        """F2, at the gate. The verdict is a key segment so the `pass` cannot
        overwrite the `fail` object — and the clause must not treat the two as
        cancelling either. Findings are answered by changing the code, which
        changes the sha."""
        store = LocalStore(tmp_path)
        file_review(store, FRIDAY, review(verdict="fail", summary="3 findings"))
        file_review(store, FRIDAY, review(verdict="pass"))
        assert not clause(store).met

    def test_a_second_reviewers_pass_on_the_same_sha_does_not_clear_a_fail(self, tmp_path) -> None:
        """A reviewer who found blocking defects is evidence about the code; a
        second reviewer who did not look at the same thing is not a rebuttal."""
        store = LocalStore(tmp_path)
        file_review(store, FRIDAY, review(verdict="fail", summary="3 findings"))
        file_review(store, FRIDAY, review(reviewer="session_01ThirdCCCCCCCC"))
        assert not clause(store).met

    def test_an_earlier_pass_does_not_clear_a_later_fail(self, tmp_path) -> None:
        """Ordering, not merely sha difference: a pass filed BEFORE the fail
        reviewed a state the fail then found defects in."""
        store = LocalStore(tmp_path)
        file_review(store, WINDOW[0], review(head_sha=OTHER_SHA))
        file_review(store, FRIDAY, review(verdict="fail", summary="3 findings"))
        assert not clause(store).met

    def test_an_empty_author_list_is_refused_not_treated_as_independent(self, tmp_path) -> None:
        """An empty author set makes every reviewer independent by
        construction — a pass by accident."""
        store = LocalStore(tmp_path)
        file_raw(
            store,
            FRIDAY,
            REVIEWER,
            "pass",
            {
                "schema_version": REVIEW_SCHEMA_VERSION,
                "phase": "phase1",
                "verdict": "pass",
                "reviewer": REVIEWER.lower(),
                "authors": [],
                "pr_number": 48,
                "head_sha": HEAD_SHA,
                "summary": "",
                "reviewed_at": REVIEWED_AT.isoformat(),
            },
        )
        found = clause(store)
        assert not found.met
        assert "non-empty list of identities" in found.detail

    def test_a_body_that_disagrees_with_its_own_key_is_refused(self, tmp_path) -> None:
        """The durability of an adverse verdict rests on the verdict being IN
        the key. A `fail` body filed under a `pass` key would be counted as a
        pass by one and a fail by the other, and the guarantee would mean
        nothing."""
        store = LocalStore(tmp_path)
        body = review(verdict="pass")
        body["verdict"] = "fail"
        store.put_bytes(
            review_key("phase1", FRIDAY.isoformat(), REVIEWER, "pass"),
            json.dumps(body).encode("utf-8"),
        )
        found = clause(store)
        assert not found.met
        assert "disagrees with the key" in found.detail

    def test_a_review_naming_no_commit_is_refused(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        body = review()
        body["head_sha"] = "not-a-sha"
        store.put_bytes(
            review_key("phase1", FRIDAY.isoformat(), REVIEWER, "pass"),
            json.dumps(body).encode("utf-8"),
        )
        found = clause(store)
        assert not found.met
        assert "40-hex commit sha" in found.detail

    def test_an_unknown_schema_version_is_refused_rather_than_guessed_at(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        body = review()
        body["schema_version"] = "review.v0"
        store.put_bytes(
            review_key("phase1", FRIDAY.isoformat(), REVIEWER, "pass"),
            json.dumps(body).encode("utf-8"),
        )
        found = clause(store)
        assert not found.met
        assert "schema_version" in found.detail

    def test_a_key_of_an_unknown_shape_is_refused(self, tmp_path) -> None:
        """A document filed under a shape this clause cannot parse is
        unreadable, not absent, and reporting it as absent names the wrong
        remedy."""
        store = LocalStore(tmp_path)
        store.put_bytes(f"{review_prefix('phase1')}stray.json", json.dumps(review()).encode())
        found = clause(store)
        assert not found.met
        assert "cannot parse" in found.detail

    def test_an_unreadable_document_is_red_not_absent(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        store.put_bytes(review_key("phase1", FRIDAY.isoformat(), REVIEWER, "pass"), b"{not json")
        found = clause(store)
        assert not found.met
        assert "not readable JSON" in found.detail


class TestTheClauseAccepts:
    def test_an_independent_pass_in_the_window_meets_the_clause(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        key = file_review(store, FRIDAY, review())
        found = clause(store)
        assert found.met, found.detail
        assert key in found.evidence
        assert REVIEWER.lower() in found.detail

    def test_a_pass_on_any_session_in_the_window_counts(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        file_review(store, WINDOW[1], review())
        assert clause(store).met

    def test_a_later_pass_on_a_different_sha_supersedes_a_fail(self, tmp_path) -> None:
        """The legitimate way out of an adverse verdict, and the only one:
        change the code, which changes the sha, and be reviewed again."""
        store = LocalStore(tmp_path)
        file_review(store, WINDOW[0], review(verdict="fail", summary="3 findings"))
        file_review(store, FRIDAY, review(head_sha=OTHER_SHA))
        assert clause(store).met

    def test_a_self_review_beside_a_genuine_one_does_not_spoil_it(self, tmp_path) -> None:
        """Guards against a refusal that over-corrects: a stray self-review in
        the prefix must not veto a real independent pass."""
        store = LocalStore(tmp_path)
        file_raw(
            store,
            FRIDAY,
            AUTHOR,
            "pass",
            {
                "schema_version": REVIEW_SCHEMA_VERSION,
                "phase": "phase1",
                "verdict": "pass",
                "reviewer": AUTHOR.lower(),
                "authors": [AUTHOR.lower()],
                "pr_number": 48,
                "head_sha": HEAD_SHA,
                "summary": "",
                "reviewed_at": REVIEWED_AT.isoformat(),
            },
        )
        file_review(store, FRIDAY, review())
        assert clause(store).met


class TestTheProducerAndTheConsumerAgree:
    def test_every_field_the_clause_validates_is_one_the_producer_writes(self, tmp_path) -> None:
        """The M0 contract test for this artifact: the clause reads exactly
        what `crucible.review` writes, so a field renamed on one side fails
        here instead of surviving as two lists that happen to agree today."""
        store = LocalStore(tmp_path)
        file_review(store, FRIDAY, review())
        assert clause(store).met

    def test_the_producer_refuses_the_verdict_the_key_would_refuse(self) -> None:
        with pytest.raises(ValueError, match="not 'pass' or 'fail'"):
            review_key("phase1", FRIDAY.isoformat(), REVIEWER, "partial")
