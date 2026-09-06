"""The phase's exit is a RECORD derived from the gate, not a convention.

Normative source: `alpha-engine-config-I9967` deliverables 2 and 3, on the
2026-09-02 state where `alpha-engine-config-I9757` sat CLOSED while its own
exit gate read 1 of 6 clauses met. The convention that was supposed to prevent
that — paste the reading before you close the issue — had already failed
twice, so what is tested here is the machinery that replaces it:

* `crucible.keys.closing_record_key` — one key per phase, never day-keyed;
* `crucible.track_f.file_closing_record` — written once, from a MET reading on
  a LIVE run, by compare-and-swap against an absent key, and only after the
  same reading has been posted to the tracker;
* `crucible.tracker` — the one adapter, which may comment and may never close;
* `crucible.board._closing_rows` — the detector: a phase issue CLOSED with no
  closing record beside it renders UNMET on every daily board.

Every guard below is shown FIRING, not merely accepting valid input.
"""

from __future__ import annotations

import ast
import datetime as dt
import io
import json
import pathlib
import urllib.error
import urllib.request

import pytest

import crucible.tracker as tracker_module
from crucible.board import build_board
from crucible.documents import UnreadableDocumentError, read_store_document
from crucible.gate import (
    CLOSING_READING_SCHEMA_VERSION,
    PHASES,
    TRACKER_REPO,
    Clause,
    GateResult,
    closing_reading,
    gate_prefix,
    last_read,
)
from crucible.keys import closing_record_key, gate_key
from crucible.runner import RunContext, run_job
from crucible.store import ETAG_ABSENT, LocalStore, PointerConflictError
from crucible.track_f import closing_record_line, file_closing_record, post_closing_comment
from crucible.tracker import IssueRead, TrackerError

COMMIT = "0123456789abcdef0123456789abcdef01234567"
DAY = dt.date(2026, 8, 28)
STORE_URI = "s3://a-bucket/a-prefix"


# ── Fixtures: real objects, never mocks of the thing under test ────────────


def _reading(*, met: bool, gate: str = "phase1", unmeasurable: bool = False) -> GateResult:
    clause = Clause(
        "arc_runs_ok",
        "the weekly arc ran and its manifest reads ok",
        met,
        "measured" if met else "no arc manifest for this week",
        (),
    )
    if unmeasurable:
        clause = Clause("arc_runs_ok", "the arc ran", False, "denied", (), unmeasurable=True)
    return GateResult(gate=gate, trading_day=DAY, window=[DAY], clauses=[clause])


class _Opener:
    """A GitHub API stand-in that records every request it was handed.

    The whole request — method, URL, headers, body — is asserted against,
    because `crucible.tracker`'s central claim is about which requests it can
    construct at all, and a stub that only returned canned bodies would leave
    exactly that claim untested.
    """

    def __init__(self, *responses: tuple[int, bytes]) -> None:
        self.responses = list(responses)
        self.seen: list[urllib.request.Request] = []

    def __call__(self, request: urllib.request.Request) -> tuple[int, bytes]:
        self.seen.append(request)
        if not self.responses:
            raise AssertionError(f"unexpected extra request to {request.full_url}")
        return self.responses.pop(0)


def _issue(state: str) -> bytes:
    return json.dumps({"state": state}).encode()


def _comment(body: str) -> dict[str, str]:
    return {"body": body}


def _granted(monkeypatch) -> None:
    monkeypatch.setenv(tracker_module.TRACKER_TOKEN_VAR, "a-token")


def _revoked(monkeypatch) -> None:
    monkeypatch.delenv(tracker_module.TRACKER_TOKEN_VAR, raising=False)


# ── The key ───────────────────────────────────────────────────────────────


class TestTheClosingRecordKey:
    def test_it_is_one_key_per_phase_and_carries_no_trading_day(self) -> None:
        assert closing_record_key("phase1") == "gates/phase1/closing.json"
        assert closing_record_key("phase1") == closing_record_key("phase1")

    def test_a_blank_phase_is_refused_rather_than_collapsing_every_phase_onto_one_key(
        self,
    ) -> None:
        with pytest.raises(ValueError, match="phase must be non-empty"):
            closing_record_key("")

    def test_it_sits_under_the_gate_prefix_without_being_read_as_a_dated_reading(
        self, tmp_path
    ) -> None:
        """`crucible.gate.last_read` lists this exact prefix to find the most
        recent trading day a gate was read for. A closing record mistaken for
        a dated reading would report the phase last read on a day that does
        not exist — so the guard is shown answering correctly with the record
        present and no dated reading at all."""
        store = LocalStore(tmp_path)
        assert closing_record_key("phase1").startswith(gate_prefix("phase1"))
        store.put_bytes(closing_record_key("phase1"), b"{}")
        assert last_read(store, "phase1") == (None, False)
        store.put_bytes(gate_key("phase1", DAY.isoformat()), b"{}")
        assert last_read(store, "phase1") == ("2026-08-28", False)


# ── The tracker adapter ───────────────────────────────────────────────────


class TestTheTrackerMayCommentAndMayNeverClose:
    def test_the_module_constructs_no_request_that_could_close_an_issue(self) -> None:
        """A property of the SOURCE, not a convention. Closing or reopening a
        phase issue is Brian's authority; an agent that took it would take the
        one protection reserved for him (`gate-taxonomy-policy` §6 makes a
        human's own act permanent, and a machine claiming it is indelible).

        Two facts are asserted over the module's syntax tree: the only
        non-default HTTP method it can name is `POST`, and every path it can
        hand `_request` is one of three literals. `PATCH /issues/{n}` — the
        request that closes an issue — is unreachable from either.
        """
        tree = ast.parse(pathlib.Path(tracker_module.__file__).read_text(encoding="utf-8"))
        methods = {
            node.value.value
            for node in ast.walk(tree)
            if isinstance(node, ast.keyword)
            and node.arg == "method"
            and isinstance(node.value, ast.Constant)
        }
        assert methods == {"POST"}, methods
        paths = {
            ast.unparse(node.args[1])
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_request"
            and len(node.args) > 1
        }
        assert paths == {
            "f'/issues/{issue}'",
            "f'/issues/{issue}/comments?per_page=100&page={page}'",
            "f'/issues/{issue}/comments'",
        }, paths

    def test_the_post_targets_the_comments_endpoint_with_the_declared_headers(
        self, monkeypatch
    ) -> None:
        _granted(monkeypatch)
        opener = _Opener((201, json.dumps({"html_url": "https://x/1"}).encode()))
        url = tracker_module.post_comment(TRACKER_REPO, 9757, "hello", opener=opener)
        assert url == "https://x/1"
        (request,) = opener.seen
        assert request.get_method() == "POST"
        assert request.full_url == (
            f"{tracker_module.API_ROOT}/repos/{TRACKER_REPO}/issues/9757/comments"
        )
        assert request.get_header("Authorization") == "Bearer a-token"
        assert request.get_header("X-github-api-version") == "2022-11-28"
        assert json.loads(request.data) == {"body": "hello"}


class TestTheTrackerReadNeverRaises:
    def test_a_missing_credential_is_a_named_absence_carrying_the_grant_command(
        self, monkeypatch
    ) -> None:
        _revoked(monkeypatch)
        read = tracker_module.read_issue(TRACKER_REPO, 9757)
        assert read.state is None
        assert read.access_problem is True
        assert tracker_module.TRACKER_APP_SSM_PREFIX_VAR in read.problem
        assert TRACKER_REPO in read.problem

    @pytest.mark.parametrize(
        "response,fragment",
        [
            ((404, b'{"message": "Not Found"}'), "GitHub answered 404"),
            ((200, b"not json at all"), "did not answer with JSON"),
            ((200, b'{"state": "merged"}'), "which is not one of"),
            ((200, b"[]"), "reports state None"),
        ],
    )
    def test_every_fault_is_returned_as_an_access_problem(
        self, monkeypatch, response, fragment
    ) -> None:
        _granted(monkeypatch)
        read = tracker_module.read_issue(TRACKER_REPO, 9757, opener=_Opener(response))
        assert read.state is None
        assert read.access_problem is True
        assert fragment in read.problem

    def test_a_transport_failure_is_reported_rather_than_propagated(self, monkeypatch) -> None:
        _granted(monkeypatch)

        def boom(_request):
            raise urllib.error.URLError("no route to host")

        read = tracker_module.read_issue(TRACKER_REPO, 9757, opener=boom)
        assert read.access_problem is True
        assert "failed at the transport" in read.problem

    def test_an_http_error_is_an_answer_not_a_transport_failure(self, monkeypatch) -> None:
        """`_default_opener` turns a 403 into `(403, body)` so the caller can
        render what GitHub refused. Asserted through the real opener, since
        every other test here substitutes it."""
        _granted(monkeypatch)

        def raise_403(*_args, **_kwargs):
            raise urllib.error.HTTPError("u", 403, "Forbidden", {}, io.BytesIO(b"denied"))

        monkeypatch.setattr(urllib.request, "urlopen", raise_403)
        read = tracker_module.read_issue(TRACKER_REPO, 9757)
        assert "GitHub answered 403" in read.problem

    def test_both_states_read_back_and_only_closed_reads_closed(self, monkeypatch) -> None:
        _granted(monkeypatch)
        assert tracker_module.read_issue(
            TRACKER_REPO, 1, opener=_Opener((200, _issue("closed")))
        ).closed
        assert not tracker_module.read_issue(
            TRACKER_REPO, 1, opener=_Opener((200, _issue("open")))
        ).closed


class TestTheCommentListingIsStrict:
    def test_it_pages_until_a_short_page(self, monkeypatch) -> None:
        _granted(monkeypatch)
        full = json.dumps([_comment(f"c{i}") for i in range(100)]).encode()
        opener = _Opener((200, full), (200, json.dumps([_comment("last")]).encode()))
        bodies = tracker_module.comment_bodies(TRACKER_REPO, 9757, opener=opener)
        assert len(bodies) == 101
        assert bodies[-1] == "last"
        assert "page=2" in opener.seen[1].full_url

    def test_it_refuses_rather_than_truncating_a_listing_it_cannot_finish(
        self, monkeypatch
    ) -> None:
        _granted(monkeypatch)
        full = json.dumps([_comment("c") for _ in range(100)]).encode()
        opener = _Opener(*[(200, full)] * 10)
        with pytest.raises(TrackerError, match="more than 1000 comments"):
            tracker_module.comment_bodies(TRACKER_REPO, 9757, opener=opener)

    @pytest.mark.parametrize(
        "response,fragment",
        [
            ((403, b"denied"), "GitHub answered 403"),
            ((200, b"{"), "did not answer with JSON"),
            ((200, b'{"not": "a list"}'), "not a list"),
        ],
    )
    def test_every_fault_raises_because_a_short_listing_would_post_a_duplicate(
        self, monkeypatch, response, fragment
    ) -> None:
        _granted(monkeypatch)
        with pytest.raises(TrackerError, match=fragment):
            tracker_module.comment_bodies(TRACKER_REPO, 9757, opener=_Opener(response))

    def test_a_missing_credential_raises_on_the_write_path(self, monkeypatch) -> None:
        _revoked(monkeypatch)
        with pytest.raises(TrackerError, match="CRUCIBLE_TRACKER_APP_SSM_PREFIX"):
            tracker_module.comment_bodies(TRACKER_REPO, 9757)


class TestThePostRefusesRatherThanRecordingNothing:
    @pytest.mark.parametrize(
        "responses,fragment",
        [
            (((422, b"unprocessable"),), "GitHub answered 422"),
            (((201, b"not json"),), "was accepted but GitHub's answer was not JSON"),
            (((201, b"{}"),), "carries no html_url"),
        ],
    )
    def test_it_raises_naming_what_happened(self, monkeypatch, responses, fragment) -> None:
        _granted(monkeypatch)
        with pytest.raises(TrackerError, match=fragment):
            tracker_module.post_comment(TRACKER_REPO, 1, "b", opener=_Opener(*responses))

    def test_an_empty_body_is_refused(self, monkeypatch) -> None:
        _granted(monkeypatch)
        with pytest.raises(TrackerError, match="empty comment"):
            tracker_module.post_comment(TRACKER_REPO, 1, "   ")

    def test_a_missing_credential_names_the_grant_and_says_nothing_was_filed(
        self, monkeypatch
    ) -> None:
        _revoked(monkeypatch)
        with pytest.raises(TrackerError, match="not filed to the store either"):
            tracker_module.post_comment(TRACKER_REPO, 1, "b")

    def test_a_transport_failure_raises(self, monkeypatch) -> None:
        _granted(monkeypatch)

        def boom(_request):
            raise urllib.error.URLError("down")

        with pytest.raises(TrackerError, match="failed at the transport"):
            tracker_module.post_comment(TRACKER_REPO, 1, "b", opener=boom)

    def test_an_explicit_credential_beats_the_environment(self, monkeypatch) -> None:
        _revoked(monkeypatch)
        assert tracker_module.credential("explicit") == "explicit"
        assert tracker_module.credential(" ") is None


class TestPostingIsIdempotentAtTheTracker:
    def test_a_reading_already_on_the_issue_is_not_posted_twice(self, monkeypatch) -> None:
        """The half that makes the pair of writes safe in either order: the
        store record can be lost, a race can be lost, the job can be re-run by
        hand, and the issue still ends up with exactly one closing reading."""
        _granted(monkeypatch)
        phase = next(p for p in PHASES if p.id == "phase1")
        document = closing_reading(_reading(met=True), store_uri=STORE_URI, commit=COMMIT)
        from crucible.gate import render_closing_comment

        body = render_closing_comment(document)
        opener = _Opener((200, json.dumps([_comment(body)]).encode()))
        monkeypatch.setattr(tracker_module, "_default_opener", opener)
        assert post_closing_comment(phase, body) == phase.tracker_url
        # ONE request: the listing. No POST was constructed at all.
        assert [r.get_method() for r in opener.seen] == ["GET"]

    def test_a_block_for_a_DIFFERENT_phase_does_not_count_as_this_phase_posted(
        self, monkeypatch
    ) -> None:
        _granted(monkeypatch)
        phase = next(p for p in PHASES if p.id == "phase1")
        other = closing_reading(
            _reading(met=True, gate="phase0"), store_uri=STORE_URI, commit=COMMIT
        )
        from crucible.gate import render_closing_comment

        opener = _Opener(
            (200, json.dumps([_comment(render_closing_comment(other))]).encode()),
            (201, json.dumps({"html_url": "https://x/2"}).encode()),
        )
        monkeypatch.setattr(tracker_module, "_default_opener", opener)
        assert post_closing_comment(phase, "body") == "https://x/2"
        assert len(opener.seen) == 2


# ── Filing the record ─────────────────────────────────────────────────────


def _file(store, reading, monkeypatch, *, responses=((200, b"[]"), (201, b'{"html_url": "u"}'))):
    opener = _Opener(*responses)
    monkeypatch.setattr(tracker_module, "_default_opener", opener)
    filed: dict[str, object] = {}

    def body(ctx: RunContext) -> None:
        filed["key"] = file_closing_record(ctx, store, reading, store_uri=STORE_URI)
        filed["outputs"] = list(ctx.outputs)

    run_job("gate", body, store=store, trading_day=DAY, run_mode="live")
    filed["requests"] = opener.seen
    return filed


class TestTheRecordIsWrittenOnceFromAMetLiveReading:
    def test_a_met_live_reading_posts_then_files_and_the_record_is_the_transcript(
        self, tmp_path, monkeypatch
    ) -> None:
        _granted(monkeypatch)
        store = LocalStore(tmp_path)
        filed = _file(store, _reading(met=True), monkeypatch)
        key = closing_record_key("phase1")
        assert filed["key"] == key
        document = read_store_document(store, key).document
        assert document["schema_version"] == CLOSING_READING_SCHEMA_VERSION
        assert document["gate_state"] == "MET"
        assert document["phase"] == "phase1"
        # The record entered the job's lineage, so `crucible explain` can name
        # the run that filed it.
        assert any(out["key"] == key for out in filed["outputs"])
        # Listed the comments, then posted one. In that order.
        assert [r.get_method() for r in filed["requests"]] == ["GET", "POST"]

    def test_a_second_reading_neither_reposts_nor_rewrites(self, tmp_path, monkeypatch) -> None:
        """A phase exits once. The second run must make NO tracker request at
        all and must leave the bytes byte-identical — the property a
        last-writer-wins `record_output` would quietly break."""
        _granted(monkeypatch)
        store = LocalStore(tmp_path)
        _file(store, _reading(met=True), monkeypatch)
        key = closing_record_key("phase1")
        first = store.get_bytes(key)
        again = _file(store, _reading(met=True), monkeypatch, responses=())
        assert again["key"] == key
        assert again["requests"] == []
        assert store.get_bytes(key) == first

    def test_the_write_is_a_compare_and_swap_against_an_absent_key(
        self, tmp_path, monkeypatch
    ) -> None:
        """Shown by making it LOSE: a record that appeared between the read
        and the write is not overwritten, it is a conflict the caller sees."""
        _granted(monkeypatch)
        store = LocalStore(tmp_path)
        real_cas = LocalStore.compare_and_swap

        def racing_cas(self, key, expected, payload):
            if expected == ETAG_ABSENT and not self.exists(key):
                self.put_bytes(key, b'{"someone": "else"}')
            return real_cas(self, key, expected, payload)

        monkeypatch.setattr(LocalStore, "compare_and_swap", racing_cas)
        with pytest.raises(PointerConflictError):
            _file(store, _reading(met=True), monkeypatch)

    def test_an_unmet_reading_files_nothing_and_tells_the_tracker_nothing(
        self, tmp_path, monkeypatch
    ) -> None:
        _granted(monkeypatch)
        store = LocalStore(tmp_path)
        filed = _file(store, _reading(met=False), monkeypatch, responses=())
        assert filed["key"] is None
        assert not store.exists(closing_record_key("phase1"))

    def test_an_unmeasurable_reading_files_nothing(self, tmp_path, monkeypatch) -> None:
        """UNMEASURABLE is never MET: it is a fact about our reading, not
        about the phase, and a record filed off one would claim an exit
        nobody measured."""
        _granted(monkeypatch)
        store = LocalStore(tmp_path)
        filed = _file(store, _reading(met=False, unmeasurable=True), monkeypatch, responses=())
        assert filed["key"] is None

    def test_a_replay_files_nothing_however_met_it_reads(self, tmp_path, monkeypatch) -> None:
        """§6 row 2: the replay schedule replays real past Saturdays, and a
        phase does not exit on one."""
        _granted(monkeypatch)
        store = LocalStore(tmp_path)
        opener = _Opener()
        monkeypatch.setattr(tracker_module, "_default_opener", opener)
        result: dict[str, object] = {}

        def body(ctx: RunContext) -> None:
            result["key"] = file_closing_record(ctx, store, _reading(met=True), store_uri=STORE_URI)

        run_job("gate", body, store=store, trading_day=DAY, run_mode="replay")
        assert result["key"] is None
        assert not store.exists(closing_record_key("phase1"))

    def test_a_present_but_unreadable_record_raises_rather_than_being_overwritten(
        self, tmp_path, monkeypatch
    ) -> None:
        _granted(monkeypatch)
        store = LocalStore(tmp_path)
        store.put_bytes(closing_record_key("phase1"), b"{ truncated")
        with pytest.raises(UnreadableDocumentError, match="will not overwrite it"):
            _file(store, _reading(met=True), monkeypatch, responses=())
        assert store.get_bytes(closing_record_key("phase1")) == b"{ truncated"

    def test_a_failed_post_leaves_no_record_so_a_re_run_files_both(
        self, tmp_path, monkeypatch
    ) -> None:
        """The ordering argument, tested: comment first, record second. The
        state this prevents — recorded but never announced — is the one
        nothing would ever correct."""
        _revoked(monkeypatch)
        store = LocalStore(tmp_path)
        with pytest.raises(TrackerError, match="CRUCIBLE_TRACKER_APP_SSM_PREFIX"):
            _file(store, _reading(met=True), monkeypatch, responses=())
        assert not store.exists(closing_record_key("phase1"))


class TestTheGateRendersTheRecordItFound:
    def test_it_names_the_key_once_the_record_exists(self, tmp_path, monkeypatch) -> None:
        _granted(monkeypatch)
        store = LocalStore(tmp_path)
        _file(store, _reading(met=True), monkeypatch)
        line = closing_record_line(store, _reading(met=True))
        assert line.startswith(f"closing record filed at {closing_record_key('phase1')}")
        assert "2026-08-28" in line

    def test_a_met_gate_with_no_record_says_how_to_file_it(self, tmp_path) -> None:
        line = closing_record_line(LocalStore(tmp_path), _reading(met=True))
        assert "no closing record" in line
        assert "--run-mode live" in line

    def test_an_unmet_gate_says_the_phase_has_not_exited(self, tmp_path) -> None:
        line = closing_record_line(LocalStore(tmp_path), _reading(met=False))
        assert "has not exited" in line

    def test_an_unreadable_record_says_so_rather_than_reading_as_absent(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        store.put_bytes(closing_record_key("phase1"), b"[]")
        assert "could not be read" in closing_record_line(store, _reading(met=False))


# ── The board detector ────────────────────────────────────────────────────


def _closing_rows(store, tracker_reader):
    board = build_board(
        store,
        now=dt.datetime(2026, 8, 28, 21, 30, tzinfo=dt.UTC),
        trading_day=DAY,
        registry={},
        declarations=None,
        tracker_reader=tracker_reader,
    )
    return {row.id: row for row in board.rows if row.id.endswith(":closing")}


class TestTheBoardRendersAPhaseClosedWithNoRecord:
    def test_a_closed_issue_with_no_record_is_the_finding(self, tmp_path) -> None:
        """`alpha-engine-config-I9757` exactly. Red on every daily board until
        somebody either files the record or reopens the issue."""
        rows = _closing_rows(LocalStore(tmp_path), lambda _repo, _issue: IssueRead("closed", None))
        row = rows["phase:phase1:closing"]
        assert row.state == "UNMET"
        assert row.red is True
        assert "CLOSED and there is no closing record" in row.detail
        assert row.artifact == closing_record_key("phase1")

    def test_an_open_issue_with_no_record_is_planned_not_red(self, tmp_path) -> None:
        rows = _closing_rows(LocalStore(tmp_path), lambda _repo, _issue: IssueRead("open", None))
        row = rows["phase:phase0:closing"]
        assert row.state == "PLANNED"
        assert row.red is False

    def test_an_unreadable_tracker_is_unmeasurable_and_carries_the_grant_command(
        self, tmp_path, monkeypatch
    ) -> None:
        """The detector standing behind the operator-gated credential: until
        the token is granted these rows are red and say what grants them."""
        _revoked(monkeypatch)
        rows = _closing_rows(LocalStore(tmp_path), None)
        row = rows["phase:phase1:closing"]
        assert row.state == "UNMEASURABLE"
        assert row.red is True
        assert tracker_module.TRACKER_APP_SSM_PREFIX_VAR in row.detail

    def test_a_filed_record_on_a_closed_issue_is_met(self, tmp_path, monkeypatch) -> None:
        _granted(monkeypatch)
        store = LocalStore(tmp_path)
        _file(store, _reading(met=True), monkeypatch)
        rows = _closing_rows(store, lambda _repo, _issue: IssueRead("closed", None))
        assert rows["phase:phase1:closing"].state == "MET"

    def test_a_filed_record_on_an_open_issue_is_met_and_says_who_may_close_it(
        self, tmp_path, monkeypatch
    ) -> None:
        _granted(monkeypatch)
        store = LocalStore(tmp_path)
        _file(store, _reading(met=True), monkeypatch)
        row = _closing_rows(store, lambda _repo, _issue: IssueRead("open", None))[
            "phase:phase1:closing"
        ]
        assert row.state == "MET"
        assert "Brian's authority" in row.detail

    def test_a_record_that_does_not_justify_an_exit_is_refused_by_name(self, tmp_path) -> None:
        """The mutation guard, end to end: a block that merely SAYS MET while
        its own transcript disagrees is not a justification, and the board
        names which rule it broke."""
        store = LocalStore(tmp_path)
        document = closing_reading(_reading(met=True), store_uri=STORE_URI, commit=COMMIT)
        document["gate_state"] = "UNMET"
        store.put_bytes(closing_record_key("phase1"), json.dumps(document).encode())
        row = _closing_rows(store, lambda _repo, _issue: IssueRead("closed", None))[
            "phase:phase1:closing"
        ]
        assert row.state == "UNMET"
        assert "does not justify an exit" in row.detail

    def test_an_unreadable_record_is_unmeasurable_not_unmet(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        store.put_bytes(closing_record_key("phase1"), b"[]")
        row = _closing_rows(store, lambda _repo, _issue: IssueRead("closed", None))[
            "phase:phase1:closing"
        ]
        assert row.state == "UNMEASURABLE"

    def test_every_declared_phase_gets_exactly_one_closing_row(self, tmp_path) -> None:
        """Derived from `gate.PHASES`, never hand-listed — the property that
        makes a seventh phase impossible to forget."""
        rows = _closing_rows(LocalStore(tmp_path), lambda _repo, _issue: IssueRead("open", None))
        assert sorted(rows) == sorted(f"phase:{p.id}:closing" for p in PHASES)
