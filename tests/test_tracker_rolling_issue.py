"""`crucible.tracker.find_issue_by_title` / `create_issue` —
`alpha-engine-config-I10123`.

The morning report's full update now lives on a rolling `[v2 board] daily
update` issue in the private tracker, found by title search and created once
if absent. Two properties matter more than the happy path:

* a second OPEN issue carrying the exact title is a loud failure, never a
  pick — posting the day's update to whichever one a race or a manual
  duplicate left behind is a full update nobody can find from the headline
  that links it;
* the search API does PHRASE matching, not exact-field matching, so a
  candidate whose title merely CONTAINS the search phrase must not count.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

import pytest

import crucible.tracker as tracker_module
from crucible.tracker import TrackerError, create_issue, find_issue_by_title

REPO = "nousergon/alpha-engine-config"
TITLE = "[v2 board] daily update"


class _Opener:
    """Records every request it was handed — see `test_phase_closing_record.py`
    for the same shape, reused here rather than re-invented."""

    def __init__(self, *responses: tuple[int, bytes]) -> None:
        self.responses = list(responses)
        self.seen: list[urllib.request.Request] = []

    def __call__(self, request: urllib.request.Request) -> tuple[int, bytes]:
        self.seen.append(request)
        if not self.responses:
            raise AssertionError(f"unexpected extra request to {request.full_url}")
        return self.responses.pop(0)


def _search_result(*items: dict) -> bytes:
    return json.dumps({"items": list(items)}).encode()


def _granted(monkeypatch) -> None:
    monkeypatch.setenv(tracker_module.TRACKER_TOKEN_VAR, "a-token")


class TestFindIssueByTitle:
    def test_no_match_returns_none(self, monkeypatch) -> None:
        _granted(monkeypatch)
        opener = _Opener((200, _search_result()))
        assert find_issue_by_title(REPO, TITLE, opener=opener) is None

    def test_exactly_one_match_returns_its_number(self, monkeypatch) -> None:
        _granted(monkeypatch)
        opener = _Opener((200, _search_result({"number": 42, "title": TITLE, "state": "open"})))
        assert find_issue_by_title(REPO, TITLE, opener=opener) == 42

    def test_two_open_matches_raises_rather_than_picking(self, monkeypatch) -> None:
        _granted(monkeypatch)
        opener = _Opener(
            (
                200,
                _search_result(
                    {"number": 42, "title": TITLE, "state": "open"},
                    {"number": 43, "title": TITLE, "state": "open"},
                ),
            )
        )
        with pytest.raises(TrackerError, match="2 open issues"):
            find_issue_by_title(REPO, TITLE, opener=opener)

    def test_a_phrase_matched_but_differently_titled_item_does_not_count(self, monkeypatch) -> None:
        """Search does phrase matching over the whole document, not an
        exact-field match. A candidate titled with extra text around the
        phrase must be filtered out by this adapter, not by GitHub."""
        _granted(monkeypatch)
        opener = _Opener(
            (
                200,
                _search_result({"number": 7, "title": f"re: {TITLE} (archived)", "state": "open"}),
            )
        )
        assert find_issue_by_title(REPO, TITLE, opener=opener) is None

    def test_a_closed_match_does_not_count(self, monkeypatch) -> None:
        _granted(monkeypatch)
        opener = _Opener((200, _search_result({"number": 9, "title": TITLE, "state": "closed"})))
        assert find_issue_by_title(REPO, TITLE, opener=opener) is None

    def test_no_credential_raises(self, monkeypatch) -> None:
        monkeypatch.delenv(tracker_module.TRACKER_TOKEN_VAR, raising=False)
        monkeypatch.delenv(tracker_module.TRACKER_APP_SSM_PREFIX_VAR, raising=False)
        with pytest.raises(TrackerError, match="no tracker credential"):
            find_issue_by_title(REPO, TITLE, opener=_Opener())

    def test_a_non_200_raises(self, monkeypatch) -> None:
        _granted(monkeypatch)
        opener = _Opener((403, b'{"message": "denied"}'))
        with pytest.raises(TrackerError, match="403"):
            find_issue_by_title(REPO, TITLE, opener=opener)

    def test_the_query_is_scoped_to_the_repo_and_the_title(self, monkeypatch) -> None:
        _granted(monkeypatch)
        opener = _Opener((200, _search_result()))
        find_issue_by_title(REPO, TITLE, opener=opener)
        request = opener.seen[0]
        decoded = urllib.parse.unquote_plus(request.full_url)
        assert decoded.startswith(tracker_module.SEARCH_API_ROOT)
        assert f"repo:{REPO}" in decoded
        assert "is:open" in decoded
        assert "is:issue" in decoded
        assert f'"{TITLE}"' in decoded


class TestCreateIssue:
    def test_creates_and_returns_number_and_url(self, monkeypatch) -> None:
        _granted(monkeypatch)
        opener = _Opener(
            (201, json.dumps({"number": 99, "html_url": "https://x/issues/99"}).encode())
        )
        number, url = create_issue(REPO, TITLE, "body text", opener=opener)
        assert number == 99
        assert url == "https://x/issues/99"
        request = opener.seen[0]
        assert request.method == "POST"
        assert request.full_url == f"{tracker_module.API_ROOT}/repos/{REPO}/issues"
        payload = json.loads(request.data.decode())
        assert payload == {"title": TITLE, "body": "body text"}

    def test_an_empty_title_is_refused(self, monkeypatch) -> None:
        _granted(monkeypatch)
        with pytest.raises(TrackerError, match="empty title"):
            create_issue(REPO, "   ", "body", opener=_Opener())

    def test_no_credential_raises(self, monkeypatch) -> None:
        monkeypatch.delenv(tracker_module.TRACKER_TOKEN_VAR, raising=False)
        monkeypatch.delenv(tracker_module.TRACKER_APP_SSM_PREFIX_VAR, raising=False)
        with pytest.raises(TrackerError, match="no tracker credential"):
            create_issue(REPO, TITLE, "body", opener=_Opener())

    def test_a_non_201_raises(self, monkeypatch) -> None:
        _granted(monkeypatch)
        opener = _Opener((422, b'{"message": "validation failed"}'))
        with pytest.raises(TrackerError, match="422"):
            create_issue(REPO, TITLE, "body", opener=opener)

    def test_an_answer_with_no_number_raises(self, monkeypatch) -> None:
        _granted(monkeypatch)
        opener = _Opener((201, json.dumps({"html_url": "https://x/1"}).encode()))
        with pytest.raises(TrackerError, match="number.*html_url|html_url.*number"):
            create_issue(REPO, TITLE, "body", opener=opener)
