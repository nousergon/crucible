"""The morning report's guards — `alpha-engine-config-I9896`, `-I9921`,
`-I10123`.

What is asserted here is what would make the report WORSE THAN NOTHING,
which is the state it replaces: Brian believed a 6am PT report existed for
weeks while nothing sent one (I9896), then that the six-section message it
grew into was legible (I9921 — it was not, per Brian 2026-09-06: "the
telegram message is not legible, too much information and its not formatted
cleanly"). Since I10123 there are two documents:

* `render_full_update` — everything the old six sections carried, now as
  Markdown posted to the rolling `[v2 board] daily update` tracker issue;
* `render_message` — the short Telegram headline that links to it.

Failure modes asserted:

* **A stale board reported as current**, in EITHER document.
* **"Nothing moved" over a comparison that failed.**
* **A delivery that did not happen, recorded as one.**
* **An invented acceptance count.**
* **A progress narrative.**
* **The headline sent before its link exists**, or sent at all when the
  tracker post failed.
* **A second rolling issue picked silently**, or the wrong one commented on.

Fixed date literals throughout (`crucible/AGENTS.md`, "a test whose subject
moves with the clock stops testing the same thing").
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
from dataclasses import dataclass
from typing import Any

import pytest
import yaml
from botocore.exceptions import ClientError

from crucible.cli import HANDLERS, JOBS
from crucible.components import load_registry
from crucible.keys import (
    BOARD_CURRENT_KEY,
    BOARD_HTML_KEY,
    TRIGGER_UNKNOWN,
    board_key,
    manifest_key,
    morning_report_key,
    morning_trigger_key,
    morning_update_key,
)
from crucible.morning import (
    ACCEPTANCE_NOT_ON_ANY_ARTIFACT,
    ACCEPTANCE_UNREADABLE,
    BOARD_URL_EXPIRES_S,
    BOARD_URL_UNAVAILABLE,
    DELIVERY_TZ,
    MORNING_JOB,
    NO_OPERATOR_ACTION,
    ROLLING_ISSUE_TITLE,
    STALE_AFTER,
    TRANSPORT_PREFIX,
    UPDATE_MESSAGE_MAX_CHARS,
    MorningInputs,
    UndeliveredError,
    deliver,
    morning_handler,
    read_inputs,
    render_message,
    resolve_trigger,
    run_report,
    store_uri,
    wire_length,
)
from crucible.store import LocalStore
from crucible.tracker import TrackerError

TRACKER_REPO = "nousergon/alpha-engine-config"


def _message_output(manifest: dict) -> dict:
    """The delivered HEADLINE's output row (`message.txt`), by suffix rather
    than by position so a third artifact (`update.md`, the trigger evidence)
    cannot silently change which one a test is asserting about."""
    (row,) = [o for o in manifest["outputs"] if o["key"].endswith("message.txt")]
    return row


def _update_output(manifest: dict) -> dict:
    """The filed copy of the full update (`update.md`)."""
    (row,) = [o for o in manifest["outputs"] if o["key"].endswith("update.md")]
    return row


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "morning-report.yml"

#: A Wednesday session, and the Tuesday before it. Both real NYSE sessions.
DAY = dt.date(2026, 9, 2)
PREVIOUS = dt.date(2026, 9, 1)
#: 13:00 UTC on the calendar day after DAY — the cron's own firing instant.
#: Every board `generated_at` fixture and every handler path that reads wall
#: clock is locked to this instant (see `_freeze_morning_now`) so the
#: 24h staleness check cannot go red on a later calendar day (I9948).
FIRED_AT = dt.datetime(2026, 9, 3, 13, 0, tzinfo=dt.UTC)
#: Render time relative to FIRED_AT, inside STALE_AFTER — never a naked
#: absolute pin against wall clock.
GENERATED_AT = FIRED_AT - dt.timedelta(hours=15, minutes=24, seconds=56)
GENERATED = GENERATED_AT.strftime("%Y-%m-%dT%H:%M:%SZ")
#: Explicitly older than STALE_AFTER relative to FIRED_AT — the intentional
#: stale-board case. Derived, not a second absolute calendar pin.
STALE_GENERATED_AT = FIRED_AT - STALE_AFTER - dt.timedelta(hours=1)
STALE_GENERATED = STALE_GENERATED_AT.strftime("%Y-%m-%dT%H:%M:%SZ")
#: Acceptance reading stamp — also relative to the frozen board clock.
MEASURED_AT = (GENERATED_AT + dt.timedelta(minutes=24, seconds=56)).strftime("%Y-%m-%dT%H:%M:%SZ")
SHA = "8fc58b6c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a"
UPDATE_URL = "https://github.com/nousergon/alpha-engine-config/issues/1#issuecomment-42"


@pytest.fixture(autouse=True)
def _freeze_morning_now(monkeypatch: pytest.MonkeyPatch) -> None:
    """Handler paths take `now` from wall clock via `run_job`.

    Lock that clock to FIRED_AT so fixtures whose `generated_at` is relative
    to FIRED_AT stay fresh without depending on the day CI runs. Tests that
    pass an explicit `now=` to `run_report`/`render_message` are unaffected.
    """
    import crucible.runner as runner

    real_run_job = runner.run_job

    def _run_job_at_fired_at(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("now", FIRED_AT)
        return real_run_job(*args, **kwargs)

    monkeypatch.setattr(runner, "run_job", _run_job_at_fired_at)


@pytest.fixture(autouse=True)
def _refuse_real_tracker_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test in this module may reach the network. Every test that drives
    `morning_handler` down the live path stubs `crucible.morning.tracker`
    explicitly (`_stub_tracker`); anything that reaches the real HTTP opener
    without doing so fails loudly rather than trying a real socket."""

    def _refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a test reached crucible.tracker's real HTTP opener")

    import crucible.tracker as tracker_module

    monkeypatch.setattr(tracker_module, "_default_opener", _refuse)


def _clause(name: str, met: bool) -> dict[str, Any]:
    return {"name": name, "met": met, "requirement": f"{name} holds", "detail": f"{name} detail"}


def _row(
    row_id: str,
    source: str,
    state: str,
    detail: str,
    clauses: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "id": row_id,
        "source": source,
        "section": "",
        "title": row_id,
        "state": state,
        "console_state": "FAILED",
        "detail": detail,
        "surface": "crucible/board",
        "artifact": f"{row_id}.json",
        "means_when_red": "x",
        "last_read": None,
        # `None` — no reading was taken — is the default, because that is what
        # a board rendered before `alpha-engine-config-I9921` carries, and the
        # report has to stay honest against one.
        "clauses": clauses,
    }


def _board(
    *,
    generated_at: str = GENERATED,
    rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    rows = (
        rows
        if rows is not None
        else [
            _row(
                "phase0",
                "phase",
                "UNMET",
                "1 of 2 clauses — old_weekly_within_cadence",
                [_clause("a_met", True), _clause("b_unmet", False)],
            ),
            _row(
                "phase1",
                "phase",
                "OUT_OF_ORDER",
                "1 of 5 clauses — phase0's gate is not met",
                [_clause("c_unmet", False)],
            ),
            _row("obj:cost", "objective", "UNMEASURED", "no artifact"),
            _row("obj:alpha", "objective", "MET", "read"),
        ]
    )
    counts = {
        state: sum(1 for r in rows if r["state"] == state)
        for state in ("MET", "UNMET", "UNMEASURED", "UNMEASURABLE", "OUT_OF_ORDER", "PLANNED")
    }
    return {
        "schema_version": "v1",
        "trading_day": DAY.isoformat(),
        "generated_at": generated_at,
        "counts": counts,
        "red_count": 3,
        "row_count": len(rows),
        "rows": rows,
    }


def _seed(
    tmp_path: pathlib.Path,
    *,
    board: dict[str, Any] | None = None,
    previous: dict[str, Any] | None = None,
    board_run: dict[str, Any] | None | str = "default",
    page: bool = True,
) -> LocalStore:
    store = LocalStore(tmp_path)
    store.put_bytes(BOARD_CURRENT_KEY, json.dumps(board or _board()).encode())
    if page:
        # The page the update links to. Seeded by default because the real
        # `board` job writes it in the same run as `board/current.json`; the
        # `page=False` case is a board render that published one and not the
        # other, which the report must survive rather than die on.
        store.put_bytes(BOARD_HTML_KEY, b"<html></html>")
    if previous is not None:
        store.put_bytes(board_key(PREVIOUS.isoformat()), json.dumps(previous).encode())
    if board_run == "default":
        board_run = {"status": "ok", "code_sha": SHA, "reason": ""}
    if board_run is not None:
        store.put_bytes(manifest_key("board", DAY.isoformat()), json.dumps(board_run).encode())
    return store


class _DeniedKeyStore(LocalStore):
    """A `LocalStore` that raises like an S3 caller denied `s3:ListBucket`
    on specific keys — reproducing run 33766008781 (2026-09-03T14:18Z),
    where `report.morning`'s first live run died on exactly this shape
    reading `report/acceptance/{day}.json`. `ClientError` codes are asserted
    by name (`AccessDenied`, `NoSuchKey`) rather than by constructing a real
    `S3Store`, since the fact under test is how `crucible.morning` reacts to
    the *code*, not how boto3 raises it.
    """

    def __init__(self, root: pathlib.Path, *, denied: dict[str, str]) -> None:
        super().__init__(root)
        self._denied = denied

    def get_bytes(self, key: str) -> bytes:
        if key in self._denied:
            raise ClientError(
                {
                    "Error": {
                        "Code": self._denied[key],
                        "Message": f"not authorized to perform: s3:ListBucket on resource: {key!r}",
                    }
                },
                "GetObject",
            )
        return super().get_bytes(key)


@dataclass
class _Result:
    any_ok: bool = True
    dedup_skipped: bool = False
    muted: bool = False
    telegram_destination: str = "telegram:nous_ergon_alerts_bot"


class _Transport:
    """A transport that records what it was asked to send."""

    def __init__(self, result: _Result | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.result = result or _Result()

    def __call__(self, message: str, **kwargs: Any) -> _Result:
        self.calls.append({"message": message, **kwargs})
        return self.result


def _stub_tracker(
    monkeypatch: pytest.MonkeyPatch,
    *,
    existing_issue: int | None = None,
    comment_url: str = UPDATE_URL,
    post_raises: Exception | None = None,
    find_raises: Exception | None = None,
    create_raises: Exception | None = None,
) -> dict[str, list[Any]]:
    """Stub every tracker call `morning_handler`'s live path can make, and
    record what each was called with. Returns the call log so a test can
    assert on ORDER and ARGUMENTS without touching the network."""
    calls: dict[str, list[Any]] = {"find": [], "create": [], "post": []}

    def _find(repo: str, title: str, **kwargs: Any) -> int | None:
        calls["find"].append((repo, title))
        if find_raises is not None:
            raise find_raises
        return existing_issue

    def _create(repo: str, title: str, body: str, **kwargs: Any) -> tuple[int, str]:
        calls["create"].append((repo, title, body))
        if create_raises is not None:
            raise create_raises
        return 1, f"https://github.com/{repo}/issues/1"

    def _post(repo: str, issue: int, body: str, **kwargs: Any) -> str:
        calls["post"].append((repo, issue, body))
        if post_raises is not None:
            raise post_raises
        return comment_url

    monkeypatch.setattr("crucible.morning.tracker.find_issue_by_title", _find)
    monkeypatch.setattr("crucible.morning.tracker.create_issue", _create)
    monkeypatch.setattr("crucible.morning.tracker.post_comment", _post)
    return calls


# ── the full update (the rolling issue comment) ────────────────────────────


class TestTheFullUpdate:
    def test_it_carries_every_section_the_old_message_did(self, tmp_path):
        """`alpha-engine-config-I10123` deliverable 6: the update markdown
        contains every section the old six-headed message did."""
        rows = [
            *_board()["rows"],
            _row("schedule:day5", "schedule", "MET", "met — due 2026-09-04"),
        ]
        store = _seed(tmp_path, board=_board(rows=rows), previous=_board(rows=rows))
        update = run_report(store, trading_day=DAY, now=FIRED_AT)
        for heading in (
            "## Ladder",
            "## Schedule (plan §6.1)",
            "## Acceptance",
            f"## Moved since {PREVIOUS}",
            "## Silence",
            "## Board",
        ):
            assert heading in update, heading
        at = [
            update.index(h)
            for h in (
                "## Ladder",
                "## Schedule (plan §6.1)",
                "## Acceptance",
                f"## Moved since {PREVIOUS}",
                "## Silence",
                "## Board",
            )
        ]
        assert at == sorted(at)
        assert f"# Crucible v2 — board for trading day {DAY.isoformat()}" in update
        assert f"store: `{tmp_path}/{BOARD_CURRENT_KEY}`" in update
        assert f"generated: {GENERATED}  commit: `{SHA}`" in update
        assert "delivered: 2026-09-03 06:00 PDT" in update

    def test_the_ladder_is_a_markdown_table_with_clause_names_in_holding(self, tmp_path):
        store = _seed(tmp_path, previous=_board())
        update = run_report(store, trading_day=DAY, now=FIRED_AT)
        assert "| phase | state | clauses met | holding |" in update
        assert "| phase0 | UNMET | 1/2 | b_unmet |" in update
        assert "a_met" not in update, "a MET clause is not what is holding the phase"
        assert "| | | | out of order: 1 of 5 clauses — phase0's gate is not met |" in update

    def test_a_row_whose_board_carried_no_clause_list_invents_no_fraction(self, tmp_path):
        rows = [_row("phase0", "phase", "UNMET", "1 of 2 clauses — a sentence, not a list")]
        store = _seed(tmp_path, board=_board(rows=rows), previous=_board(rows=rows))
        update = run_report(store, trading_day=DAY, now=FIRED_AT)
        assert "| phase0 | UNMET | — | 1 of 2 clauses — a sentence, not a list |" in update
        assert "0/0" not in update

    def test_a_gate_with_no_clauses_is_not_zero_of_zero(self, tmp_path):
        rows = [_row("phase0", "phase", "UNMEASURED", "gate phase0 has no clauses", [])]
        store = _seed(tmp_path, board=_board(rows=rows), previous=_board(rows=rows))
        update = run_report(store, trading_day=DAY, now=FIRED_AT)
        assert "no clauses | this gate measured nothing" in update
        assert "0/0" not in update

    def test_a_board_with_no_phase_row_is_a_finding_not_an_empty_table(self, tmp_path):
        rows = [_row("obj:alpha", "objective", "MET", "d")]
        store = _seed(tmp_path, board=_board(rows=rows), previous=_board(rows=rows))
        update = run_report(store, trading_day=DAY, now=FIRED_AT)
        assert "no phase row on the board" in update

    def test_a_met_milestone_carries_no_overdue_word(self, tmp_path):
        rows = [_row("schedule:day5_replays", "schedule", "MET", "met — due 2026-09-04")]
        store = _seed(tmp_path, board=_board(rows=rows), previous=_board(rows=rows))
        update = run_report(store, trading_day=DAY, now=FIRED_AT)
        assert "| schedule:day5_replays | MET | met — due 2026-09-04 |" in update
        assert "OVERDUE" not in update

    def test_an_unmet_past_due_milestone_row_begins_with_overdue(self, tmp_path):
        rows = [
            _row(
                "schedule:day5_replays",
                "schedule",
                "UNMET",
                "reads: phase1 1/6",
            )
        ]
        store = _seed(tmp_path, board=_board(rows=rows), previous=_board(rows=rows))
        update = run_report(store, trading_day=DAY, now=FIRED_AT)
        assert "| OVERDUE schedule:day5_replays | UNMET |" in update

    def test_overdue_schedule_rows_sort_ahead_of_the_rest(self, tmp_path):
        rows = [
            *_board()["rows"],
            _row("schedule:ahead", "schedule", "PLANNED", "due 2026-10-01, waiting"),
            _row("schedule:late", "schedule", "UNMET", "since 2026-08-01"),
        ]
        store = _seed(tmp_path, board=_board(rows=rows), previous=_board(rows=rows))
        update = run_report(store, trading_day=DAY, now=FIRED_AT)
        late = update.index("schedule:late")
        ahead = update.index("schedule:ahead")
        assert late < ahead

    def test_no_schedule_rows_is_a_named_defect_not_an_empty_table(self, tmp_path):
        store = _seed(tmp_path, previous=_board())  # default fixture carries no schedule rows
        update = run_report(store, trading_day=DAY, now=FIRED_AT)
        assert "no schedule row on the board" in update

    def test_a_filed_acceptance_reading_is_quoted_with_its_commit(self, tmp_path):
        from crucible.keys import acceptance_reading_key

        store = _seed(tmp_path, previous=_board())
        store.put_bytes(
            acceptance_reading_key(DAY.isoformat()),
            json.dumps(
                {
                    "met": 21,
                    "unmet": 3,
                    "unmeasurable": 0,
                    "commit": SHA,
                    "measured_at": MEASURED_AT,
                }
            ).encode(),
        )
        update = run_report(store, trading_day=DAY, now=FIRED_AT)
        assert f"acceptance count: 21 met / 3 unmet / 0 unmeasurable of 24 (commit {SHA})" in update
        assert ACCEPTANCE_NOT_ON_ANY_ARTIFACT not in update

    def test_the_acceptance_count_is_never_invented(self, tmp_path):
        store = _seed(tmp_path, previous=_board())
        assert ACCEPTANCE_NOT_ON_ANY_ARTIFACT in run_report(store, trading_day=DAY, now=FIRED_AT)

    def test_the_acceptance_section_names_the_unmet_clause_ids(self, tmp_path):
        from crucible.keys import acceptance_reading_key

        store = _seed(tmp_path, previous=_board())
        store.put_bytes(
            acceptance_reading_key(DAY.isoformat()),
            json.dumps(
                {
                    "met": 21,
                    "unmet": 2,
                    "unmeasurable": 1,
                    "commit": SHA,
                    "measured_at": MEASURED_AT,
                    "unmet_clauses": ["c_replays", "c_cost"],
                    "unmeasurable_clauses": ["c_spend"],
                }
            ).encode(),
        )
        update = run_report(store, trading_day=DAY, now=FIRED_AT)
        assert "unmet: c_replays, c_cost" in update
        assert "unmeasurable: c_spend" in update

    def test_an_access_denied_acceptance_read_is_unreadable_not_absent(self, tmp_path):
        from crucible.keys import acceptance_reading_key

        denied = _DeniedKeyStore(
            tmp_path, denied={acceptance_reading_key(DAY.isoformat()): "AccessDenied"}
        )
        denied.put_bytes(BOARD_CURRENT_KEY, json.dumps(_board()).encode())
        denied.put_bytes(board_key(PREVIOUS.isoformat()), json.dumps(_board()).encode())
        denied.put_bytes(
            manifest_key("board", DAY.isoformat()),
            json.dumps({"status": "ok", "code_sha": SHA, "reason": ""}).encode(),
        )
        update = run_report(denied, trading_day=DAY, now=FIRED_AT)
        assert ACCEPTANCE_UNREADABLE.format(code="AccessDenied") in update
        assert ACCEPTANCE_NOT_ON_ANY_ARTIFACT not in update

    def test_a_not_found_acceptance_read_is_still_the_absent_literal(self, tmp_path):
        from crucible.keys import acceptance_reading_key

        store = _DeniedKeyStore(
            tmp_path, denied={acceptance_reading_key(DAY.isoformat()): "NoSuchKey"}
        )
        store.put_bytes(BOARD_CURRENT_KEY, json.dumps(_board()).encode())
        store.put_bytes(board_key(PREVIOUS.isoformat()), json.dumps(_board()).encode())
        store.put_bytes(
            manifest_key("board", DAY.isoformat()),
            json.dumps({"status": "ok", "code_sha": SHA, "reason": ""}).encode(),
        )
        update = run_report(store, trading_day=DAY, now=FIRED_AT)
        assert ACCEPTANCE_NOT_ON_ANY_ARTIFACT in update
        assert "unreadable" not in update.lower()

    def test_an_identical_board_reports_nothing_moved(self, tmp_path):
        store = _seed(tmp_path, previous=_board())
        assert "nothing moved" in run_report(store, trading_day=DAY, now=FIRED_AT)

    def test_a_vanished_row_is_reported_rather_than_silently_dropped(self, tmp_path):
        previous = _board(rows=[*_board()["rows"], _row("obj:gone", "objective", "MET", "was")])
        store = _seed(tmp_path, previous=previous)
        assert "obj:gone: MET -> VANISHED" in run_report(store, trading_day=DAY, now=FIRED_AT)

    def test_an_unreadable_previous_board_is_never_reported_as_nothing_moved(self, tmp_path):
        store = _seed(tmp_path)  # no previous board at all
        update = run_report(store, trading_day=DAY, now=FIRED_AT)
        assert "nothing moved" not in update
        assert "cannot say" in update
        assert board_key(PREVIOUS.isoformat()) in update

    def test_an_access_denied_previous_board_is_unreadable_not_absent_or_a_crash(self, tmp_path):
        store = _DeniedKeyStore(tmp_path, denied={board_key(PREVIOUS.isoformat()): "AccessDenied"})
        store.put_bytes(BOARD_CURRENT_KEY, json.dumps(_board()).encode())
        store.put_bytes(
            manifest_key("board", DAY.isoformat()),
            json.dumps({"status": "ok", "code_sha": SHA, "reason": ""}).encode(),
        )
        update = run_report(store, trading_day=DAY, now=FIRED_AT)
        assert "nothing moved" not in update
        assert "cannot say" in update
        assert "AccessDenied" in update

    def test_an_access_denied_current_board_still_raises(self, tmp_path):
        store = _DeniedKeyStore(tmp_path, denied={BOARD_CURRENT_KEY: "AccessDenied"})
        with pytest.raises(ClientError):
            run_report(store, trading_day=DAY, now=FIRED_AT)

    def test_an_absent_current_board_raises_rather_than_reporting_on_nothing(self, tmp_path):
        store = LocalStore(tmp_path)
        with pytest.raises(KeyError):
            run_report(store, trading_day=DAY, now=FIRED_AT)

    def test_silence_is_reported_as_its_own_figure(self, tmp_path):
        rows = [
            _row("phase0", "phase", "UNMET", "d"),
            _row("a", "objective", "UNMEASURED", "d"),
            _row("b", "objective", "UNMEASURABLE", "d"),
        ]
        store = _seed(tmp_path, board=_board(rows=rows), previous=_board(rows=rows))
        assert "silence: 1 UNMEASURED, 1 UNMEASURABLE of 3 rows" in run_report(
            store, trading_day=DAY, now=FIRED_AT
        )

    def test_a_failed_board_render_becomes_the_pending_operator_action(self, tmp_path):
        store = _seed(
            tmp_path,
            previous=_board(),
            board_run={"status": "failed", "code_sha": SHA, "reason": "AccessDenied on PutObject"},
        )
        update = run_report(store, trading_day=DAY, now=FIRED_AT)
        assert "pending operator action: the board render for 2026-09-02 failed" in update
        assert "AccessDenied on PutObject" in update

    def test_a_missing_board_manifest_is_named_rather_than_a_blank_commit(self, tmp_path):
        store = _seed(tmp_path, previous=_board(), board_run=None)
        update = run_report(store, trading_day=DAY, now=FIRED_AT)
        assert "commit: `UNKNOWN" in update
        assert "filed no manifest" in update

    def test_a_stale_board_is_the_first_line_and_not_a_footnote(self, tmp_path):
        store = _seed(tmp_path, board=_board(generated_at=STALE_GENERATED), previous=_board())
        update = run_report(store, trading_day=DAY, now=FIRED_AT)
        assert update.splitlines()[0].startswith("**STALE BOARD:")
        assert STALE_GENERATED in update.splitlines()[0]

    def test_a_board_generated_within_the_day_carries_no_stale_headline(self, tmp_path):
        store = _seed(tmp_path, previous=_board())
        assert "STALE BOARD" not in run_report(store, trading_day=DAY, now=FIRED_AT)

    def test_an_unparseable_generated_at_is_stale_rather_than_assumed_fresh(self, tmp_path):
        store = _seed(tmp_path, board=_board(generated_at="yesterday"), previous=_board())
        assert run_report(store, trading_day=DAY, now=FIRED_AT).startswith("**STALE BOARD:")

    def test_the_reading_is_quoted_with_its_store(self, tmp_path):
        store = _seed(tmp_path, previous=_board())
        assert f"store: `{tmp_path}/{BOARD_CURRENT_KEY}`" in run_report(
            store, trading_day=DAY, now=FIRED_AT
        )

    def test_the_message_carries_no_progress_narrative(self, tmp_path):
        store = _seed(tmp_path, previous=_board())
        update = run_report(store, trading_day=DAY, now=FIRED_AT).lower()
        for forbidden in ("pull request", " prs ", "merged", "progress", "findings", "commits"):
            assert forbidden not in update

    def test_a_board_detail_carrying_a_progress_figure_is_withheld(self, tmp_path):
        rows = [
            _row(
                "phase0",
                "phase",
                "UNMET",
                "1/2 clauses met; 3 of 5 PRs merged this week; holding: old_weekly",
            )
        ]
        store = _seed(tmp_path, board=_board(rows=rows), previous=_board(rows=rows))
        update = run_report(store, trading_day=DAY, now=FIRED_AT)
        assert "PRs merged" not in update
        assert "1/2 clauses met" in update
        assert "holding: old_weekly" in update
        assert "[1 clause withheld — plan §12 rule 3]" in update

    def test_a_detail_with_no_progress_figure_is_rendered_verbatim(self, tmp_path):
        detail = "1/2 clauses met; holding: old_weekly_within_cadence; grades 2 of 5 deliverables"
        rows = [_row("phase0", "phase", "UNMET", detail)]
        store = _seed(tmp_path, board=_board(rows=rows), previous=_board(rows=rows))
        update = run_report(store, trading_day=DAY, now=FIRED_AT)
        assert f"| phase0 | UNMET | — | {detail} |" in update
        assert "withheld" not in update

    def test_hostile_content_is_passed_through_unescaped_because_this_is_markdown_not_html(
        self, tmp_path
    ):
        """The full update is a GitHub comment body, not a Telegram HTML
        payload — running board free text through `_escape_html` here would
        print literal `&amp;` where the board said `&`."""
        hostile = "Tom & Jerry <ok> if 2<3"
        rows = [_row("phase0", "phase", "UNMET", hostile)]
        store = _seed(tmp_path, board=_board(rows=rows), previous=_board(rows=rows))
        update = run_report(store, trading_day=DAY, now=FIRED_AT)
        assert hostile in update
        assert "&amp;" not in update

    def test_a_pipe_in_board_free_text_does_not_break_the_table(self, tmp_path):
        rows = [_row("phase0", "phase", "UNMET", "reads a|b sentinel")]
        store = _seed(tmp_path, board=_board(rows=rows), previous=_board(rows=rows))
        update = run_report(store, trading_day=DAY, now=FIRED_AT)
        assert "a/b sentinel" in update
        assert "a|b" not in update

    def test_no_character_budget_a_large_board_is_never_truncated(self, tmp_path):
        rows = [
            _row(
                f"phase{i}",
                "phase",
                "UNMET",
                "d",
                [_clause(f"clause_number_{i}_{j}_with_a_long_name", False) for j in range(12)],
            )
            for i in range(60)
        ]
        store = _seed(tmp_path, board=_board(rows=rows), previous=_board(rows=rows))
        update = run_report(store, trading_day=DAY, now=FIRED_AT)
        assert "truncated" not in update
        for i in range(60):
            assert f"phase{i}" in update


class TestTheBoardLinkInTheFullUpdate:
    def test_the_full_update_carries_a_url_for_the_board_page(self, tmp_path):
        store = _seed(tmp_path, previous=_board())
        update = run_report(store, trading_day=DAY, now=FIRED_AT)
        assert (tmp_path / "board" / "index.html").resolve().as_uri() in update

    def test_the_expiry_is_quoted_as_a_bound_and_not_as_a_promise(self, tmp_path):
        store = _seed(tmp_path, previous=_board())
        update = run_report(store, trading_day=DAY, now=FIRED_AT)
        assert "when the signing role's session ends, whichever is first" in update
        assert BOARD_URL_EXPIRES_S == 7 * 24 * 3600

    def test_a_missing_page_is_stated_rather_than_killing_the_whole_report(self, tmp_path):
        store = _seed(tmp_path, previous=_board(), page=False)
        update = run_report(store, trading_day=DAY, now=FIRED_AT)
        assert BOARD_URL_UNAVAILABLE in update
        assert "Ladder" in update, "the rest of the report must still be there"

    def test_a_configuration_failure_still_raises(self, tmp_path):
        class _Broken(LocalStore):
            def presigned_url(self, key: str, expires_s: int) -> str:
                raise ValueError("expires_s out of range")

        _seed(tmp_path, previous=_board())
        with pytest.raises(ValueError):
            run_report(_Broken(tmp_path), trading_day=DAY, now=FIRED_AT)


class TestTheConsoleLinkReplacesThePresignedOneInTheFullUpdate:
    """`alpha-engine-config-I9926`: when a console is configured, the Board
    section carries its stable Decision-list address and nothing presigned."""

    CONSOLE = "https://console.example.test"

    def test_the_console_url_is_the_link(self, tmp_path):
        store = _seed(tmp_path, previous=_board())
        update = run_report(store, trading_day=DAY, now=FIRED_AT, console_url=self.CONSOLE)
        assert f"{self.CONSOLE}/decision?pipeline=crucible-board" in update

    def test_no_expiry_is_reported_when_nothing_was_presigned(self, tmp_path):
        store = _seed(tmp_path, previous=_board())
        inputs = read_inputs(store, trading_day=DAY, now=FIRED_AT, console_url=self.CONSOLE)
        assert inputs.board_url is None
        assert inputs.board_url_expires == ""
        assert inputs.board_console_url == f"{self.CONSOLE}/decision?pipeline=crucible-board"

    def test_the_presigned_link_and_its_caveat_are_absent(self, tmp_path):
        store = _seed(tmp_path, previous=_board())
        update = run_report(store, trading_day=DAY, now=FIRED_AT, console_url=self.CONSOLE)
        assert (tmp_path / "board" / "index.html").resolve().as_uri() not in update
        assert "presigned GET" not in update

    def test_a_trailing_slash_on_the_base_url_does_not_double_up(self, tmp_path):
        store = _seed(tmp_path, previous=_board())
        update = run_report(store, trading_day=DAY, now=FIRED_AT, console_url=self.CONSOLE + "/")
        assert f"{self.CONSOLE}/decision?pipeline=crucible-board" in update
        assert f"{self.CONSOLE}//decision" not in update

    def test_no_console_means_the_presigned_path_is_unchanged(self, tmp_path):
        store = _seed(tmp_path, previous=_board())
        with_none = run_report(store, trading_day=DAY, now=FIRED_AT, console_url=None)
        default = run_report(store, trading_day=DAY, now=FIRED_AT)
        assert with_none == default
        assert "presigned GET" in default


class TestAReadThatFailedIsNotAPageThatIsAbsent:
    """`_board_url` swallowed only `KeyError`, but `S3Store.presigned_url`
    reaches absence through `exists()` → `head_object`, which re-raises every
    non-404 `ClientError`. `alpha-engine-config-I9896` measured the shape live:
    S3 answers **403 for a missing key** when the caller also lacks
    `s3:ListBucket` on the prefix — and `board/index.html` is written only
    under `if may_move:`, so its absence is a live possibility.
    """

    class _Store(LocalStore):
        def __init__(self, root, *, code: str) -> None:
            super().__init__(root)
            self._code = code

        def presigned_url(self, key: str, expires_s: int) -> str:
            if key == BOARD_HTML_KEY:
                raise ClientError(
                    {"Error": {"Code": self._code, "Message": "s3:ListBucket denied"}},
                    "HeadObject",
                )
            return super().presigned_url(key, expires_s)

    def _update(self, tmp_path, code: str) -> str:
        _seed(tmp_path, previous=_board())
        store = self._Store(tmp_path, code=code)
        return run_report(store, trading_day=DAY, now=FIRED_AT)

    def test_an_access_denied_page_read_does_not_kill_the_report(self, tmp_path):
        update = self._update(tmp_path, "AccessDenied")
        assert update.startswith("# Crucible v2 —")

    def test_an_access_denied_page_read_is_named_as_an_access_failure(self, tmp_path):
        update = self._update(tmp_path, "AccessDenied")
        assert "AccessDenied" in update
        assert "access failure" in update
        assert BOARD_URL_UNAVAILABLE not in update

    def test_a_not_found_page_read_is_still_absent_not_denied(self, tmp_path):
        update = self._update(tmp_path, "NoSuchKey")
        assert BOARD_URL_UNAVAILABLE in update
        assert "access failure" not in update

    def test_an_absent_page_is_still_the_unavailable_line(self, tmp_path):
        store = _seed(tmp_path, previous=_board(), page=False)
        assert BOARD_URL_UNAVAILABLE in run_report(store, trading_day=DAY, now=FIRED_AT)


# ── the store it quotes ───────────────────────────────────────────────────


class TestStoreUri:
    def test_a_local_store_names_its_root(self, tmp_path):
        assert store_uri(LocalStore(tmp_path)) == str(tmp_path)

    def test_an_s3_store_names_bucket_and_prefix(self):
        from crucible.store import S3Store

        assert store_uri(S3Store("b", "p/q")) == "s3://b/p/q"
        assert store_uri(S3Store("b")) == "s3://b"

    def test_a_backend_that_cannot_name_itself_raises(self):
        class Nameless:
            pass

        with pytest.raises(TypeError, match="cannot name itself"):
            store_uri(Nameless())  # type: ignore[arg-type]


# ── the headline (the Telegram message) ────────────────────────────────────


class TestTheHeadline:
    def _inputs(self, tmp_path, **seed_kwargs) -> MorningInputs:
        store = _seed(tmp_path, **seed_kwargs)
        return read_inputs(store, trading_day=DAY, now=FIRED_AT)

    def test_it_is_short_titled_and_links_the_update_and_the_board(self, tmp_path):
        inputs = self._inputs(tmp_path, previous=_board())
        message = render_message(inputs, now=FIRED_AT, update_url=UPDATE_URL)
        lines = message.splitlines()
        assert len(lines) <= 12
        assert lines[0] == "<b>CRUCIBLE V2 — 2026-09-02</b>"
        assert f'<a href="{UPDATE_URL}">Full update</a>' in message
        board_uri = (tmp_path / "board" / "index.html").resolve().as_uri()
        assert f'<a href="{board_uri}">Board</a>' in message

    def test_no_holding_lists_and_no_out_of_order_prose(self, tmp_path):
        inputs = self._inputs(tmp_path, previous=_board())
        message = render_message(inputs, now=FIRED_AT, update_url=UPDATE_URL)
        assert "holding:" not in message
        assert "out of order:" not in message
        assert "b_unmet" not in message
        assert "phase0: UNMET 1/2" in message
        assert "phase1: OUT_OF_ORDER 0/1" in message

    def test_it_carries_the_acceptance_line_and_moved_count(self, tmp_path):
        previous = _board(
            rows=[
                _row("phase0", "phase", "UNMET", "old"),
                _row("phase1", "phase", "OUT_OF_ORDER", "old"),
                _row("obj:cost", "objective", "PLANNED", "old"),
                _row("obj:alpha", "objective", "MET", "old"),
            ]
        )
        inputs = self._inputs(tmp_path, previous=previous)
        message = render_message(inputs, now=FIRED_AT, update_url=UPDATE_URL)
        assert ACCEPTANCE_NOT_ON_ANY_ARTIFACT in message
        assert f"moved since {PREVIOUS}: 1" in message

    def test_pending_operator_action_appears_when_present(self, tmp_path):
        inputs = self._inputs(
            tmp_path,
            previous=_board(),
            board_run={"status": "failed", "code_sha": SHA, "reason": "AccessDenied"},
        )
        message = render_message(inputs, now=FIRED_AT, update_url=UPDATE_URL)
        assert "pending operator action:" in message
        assert NO_OPERATOR_ACTION not in message

    def test_no_operator_action_line_when_none_is_pending(self, tmp_path):
        inputs = self._inputs(tmp_path, previous=_board())
        message = render_message(inputs, now=FIRED_AT, update_url=UPDATE_URL)
        assert "pending operator action:" not in message

    def test_a_stale_board_is_the_headline_first_line(self, tmp_path):
        inputs = self._inputs(
            tmp_path, board=_board(generated_at=STALE_GENERATED), previous=_board()
        )
        message = render_message(inputs, now=FIRED_AT, update_url=UPDATE_URL)
        assert message.splitlines()[0].startswith("<b>STALE BOARD:")

    def test_a_missing_board_page_omits_the_board_link_rather_than_failing(self, tmp_path):
        inputs = self._inputs(tmp_path, previous=_board(), page=False)
        message = render_message(inputs, now=FIRED_AT, update_url=UPDATE_URL)
        assert "Board</a>" not in message
        assert f'<a href="{UPDATE_URL}">Full update</a>' in message

    def test_the_console_link_is_used_when_configured(self, tmp_path):
        console = "https://console.example.test"
        store = _seed(tmp_path, previous=_board())
        inputs = read_inputs(store, trading_day=DAY, now=FIRED_AT, console_url=console)
        message = render_message(inputs, now=FIRED_AT, update_url=UPDATE_URL)
        assert f'<a href="{console}/decision?pipeline=crucible-board">Board</a>' in message

    def test_hostile_content_is_html_escaped(self, tmp_path):
        inputs = self._inputs(
            tmp_path,
            previous=_board(),
            board_run={
                "status": "failed",
                "code_sha": SHA,
                "reason": "PutObject denied for role <arn> & retried",
            },
        )
        message = render_message(inputs, now=FIRED_AT, update_url=UPDATE_URL)
        assert "denied for role <arn>" not in message
        assert "denied for role &lt;arn&gt; &amp; retried" in message

    #: Real clause identifiers off `main`'s `crucible/gate.py` — the longest
    #: names any phase gate actually emits, so the worst case measured here
    #: is one this repository can actually produce.
    _REAL_CLAUSE_NAMES: tuple[str, ...] = (
        "old_sf_execution_count_zero",
        "aws_total_within_ceiling",
        "aws_cost_within_ceiling",
        "attribution_renders",
        "explain_walks_a_verdict",
        "dead_lambdas_deleted",
        "old_alerts_muted",
        "acceptance_suite_committed",
    )

    def test_the_wire_fits_the_cap_with_six_red_phases_of_real_clause_names(self, tmp_path):
        """`alpha-engine-config-I10123` deliverable 2's hard budget test: six
        phase rows (`phase0`..`phase5`, one per binding-plan phase), every
        one UNMET/red, holding every real long clause name off `main`'s
        `crucible/gate.py`, plus a long hostile operator action — the
        worst-case fixture the old six-section message was proven against,
        now proving the HEADLINE stays inside its own, much smaller budget.
        """
        rows = [
            _row(
                f"phase{i}",
                "phase",
                "UNMET",
                "blocked by <deploy> & <IAM> gaps",
                [_clause(name, met=False) for name in self._REAL_CLAUSE_NAMES],
            )
            for i in range(6)
        ]
        store = _seed(
            tmp_path,
            board=_board(rows=rows),
            previous=_board(rows=rows),
            board_run={
                "status": "failed",
                "code_sha": SHA,
                "reason": "AccessDenied on PutObject for role arn:aws:iam::111111111111:role/x",
            },
        )
        inputs = read_inputs(store, trading_day=DAY, now=FIRED_AT)
        message = render_message(inputs, now=FIRED_AT, update_url=UPDATE_URL)
        wire = wire_length(TRANSPORT_PREFIX + message)
        assert wire <= UPDATE_MESSAGE_MAX_CHARS, (
            f"the six-red-phase worst case POSTs at {wire} chars, over the "
            f"{UPDATE_MESSAGE_MAX_CHARS}-char budget"
        )
        for i in range(6):
            assert f"phase{i}: UNMET 0/{len(self._REAL_CLAUSE_NAMES)}" in message

    def test_an_over_budget_headline_raises_rather_than_truncating_silently(self, tmp_path):
        """A headline that silently shortened itself would be the illegible
        message reappearing in a new shape — it must fail loudly instead."""
        rows = [_row(f"phase{i}", "phase", "UNMET", "d", [_clause("x", False)]) for i in range(400)]
        store = _seed(tmp_path, board=_board(rows=rows), previous=_board(rows=rows))
        inputs = read_inputs(store, trading_day=DAY, now=FIRED_AT)
        with pytest.raises(ValueError, match="over the"):
            render_message(inputs, now=FIRED_AT, update_url=UPDATE_URL)


# ── delivery ──────────────────────────────────────────────────────────────


class TestDelivery:
    def test_it_never_touches_the_page_path(self):
        transport = _Transport()
        deliver("hello", transport=transport)
        (call,) = transport.calls
        assert call["sns"] is False
        assert call["telegram"] is True
        assert call["severity"] == "info"
        assert call["silent"] is False
        assert call["raise_on_total_failure"] is True
        assert call["destination"] == "operator_chat"

    def test_it_carries_no_dedup_key(self):
        transport = _Transport()
        deliver("hello", transport=transport)
        assert transport.calls[0]["dedup_key"] is None

    def test_a_transport_that_reached_nobody_raises(self):
        with pytest.raises(UndeliveredError, match="reached nobody"):
            deliver("hello", transport=_Transport(_Result(any_ok=False)))

    def test_a_dedup_suppressed_publish_is_not_a_delivery(self):
        with pytest.raises(UndeliveredError):
            deliver("hello", transport=_Transport(_Result(dedup_skipped=True)))

    def test_a_muted_publish_is_not_a_delivery(self):
        with pytest.raises(UndeliveredError):
            deliver("hello", transport=_Transport(_Result(muted=True)))

    def test_a_transport_result_that_cannot_answer_raises(self):
        with pytest.raises(TypeError, match="any_ok"):
            deliver("hello", transport=lambda *a, **k: object())

    def test_deliver_sends_with_parse_mode_html(self):
        transport = _Transport()
        deliver("<b>CRUCIBLE V2</b>\nhello", transport=transport)
        (call,) = transport.calls
        assert call["parse_mode"] == "HTML"


# ── the job ───────────────────────────────────────────────────────────────


def _args(tmp_path: pathlib.Path, *, dry_run: bool) -> argparse.Namespace:
    return argparse.Namespace(
        job=MORNING_JOB,
        store=str(tmp_path),
        dry_run=dry_run,
        date=DAY.isoformat(),
        trading_day=DAY,
    )


class TestTheJob:
    def test_a_delivered_report_posts_the_comment_before_sending_the_message(
        self, tmp_path, monkeypatch
    ):
        """`alpha-engine-config-I10123`: ordering is the whole safety
        property — the comment exists before the headline that links it is
        ever rendered."""
        _seed(tmp_path, previous=_board())
        calls = _stub_tracker(monkeypatch, existing_issue=7)
        transport = _Transport()
        monkeypatch.setattr("crucible.morning._krepis_publish", transport)
        monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)
        monkeypatch.delenv("CRUCIBLE_TRIGGER", raising=False)

        assert morning_handler(_args(tmp_path, dry_run=False)) == 0

        assert calls["find"] == [(TRACKER_REPO, ROLLING_ISSUE_TITLE)]
        assert calls["create"] == []
        (post_call,) = calls["post"]
        assert post_call[0] == TRACKER_REPO
        assert post_call[1] == 7
        # The comment carries the full update, not the headline.
        assert "## Ladder" in post_call[2]
        # The message actually sent carries the comment's OWN permalink.
        (sent,) = transport.calls
        assert UPDATE_URL in sent["message"]

    def test_the_rolling_issue_is_created_when_absent(self, tmp_path, monkeypatch):
        _seed(tmp_path, previous=_board())
        calls = _stub_tracker(monkeypatch, existing_issue=None)
        monkeypatch.setattr("crucible.morning._krepis_publish", lambda *a, **k: _Result())

        assert morning_handler(_args(tmp_path, dry_run=False)) == 0

        assert calls["find"] == [(TRACKER_REPO, ROLLING_ISSUE_TITLE)]
        (create_call,) = calls["create"]
        assert create_call[0] == TRACKER_REPO
        assert create_call[1] == ROLLING_ISSUE_TITLE
        (post_call,) = calls["post"]
        assert post_call[1] == 1  # the number `_create` returned above

    def test_a_second_open_rolling_issue_is_a_loud_failure_not_a_pick(self, tmp_path, monkeypatch):
        store = _seed(tmp_path, previous=_board())
        _stub_tracker(
            monkeypatch,
            find_raises=TrackerError("2 open issues titled '[v2 board] daily update'"),
        )
        monkeypatch.setattr("crucible.morning._krepis_publish", lambda *a, **k: _Result())

        with pytest.raises(TrackerError, match="2 open issues"):
            morning_handler(_args(tmp_path, dry_run=False))

        keys = [k for k in store.list_keys(f"runs/{MORNING_JOB}/") if k.endswith("run.json")]
        manifest = json.loads(store.get_bytes(keys[0]))
        assert manifest["status"] == "failed"

    def test_a_failed_post_fails_the_run_and_never_sends_the_message(self, tmp_path, monkeypatch):
        """A failed post → run fails, no message sent
        (`alpha-engine-config-I10123` ordering rule)."""
        store = _seed(tmp_path, previous=_board())
        _stub_tracker(monkeypatch, existing_issue=7, post_raises=TrackerError("posting failed"))
        transport = _Transport()
        monkeypatch.setattr("crucible.morning._krepis_publish", transport)

        with pytest.raises(TrackerError, match="posting failed"):
            morning_handler(_args(tmp_path, dry_run=False))

        assert transport.calls == [], "an unlinked headline must never be sent"
        keys = [k for k in store.list_keys(f"runs/{MORNING_JOB}/") if k.endswith("run.json")]
        manifest = json.loads(store.get_bytes(keys[0]))
        assert manifest["status"] == "failed"
        assert manifest["outputs"] == []

    def test_a_failed_delivery_raises_and_the_manifest_reads_failed(self, tmp_path, monkeypatch):
        store = _seed(tmp_path, previous=_board())
        _stub_tracker(monkeypatch, existing_issue=7)
        monkeypatch.setattr(
            "crucible.morning._krepis_publish", lambda *a, **k: _Result(any_ok=False)
        )
        with pytest.raises(UndeliveredError):
            morning_handler(_args(tmp_path, dry_run=False))
        keys = [k for k in store.list_keys(f"runs/{MORNING_JOB}/") if k.endswith("run.json")]
        assert len(keys) == 1
        manifest = json.loads(store.get_bytes(keys[0]))
        assert manifest["status"] == "failed"
        assert "reached nobody" in manifest["reason"]
        assert manifest["outputs"] == []

    def test_a_delivered_report_files_the_message_the_update_and_the_trigger(
        self, tmp_path, monkeypatch
    ):
        store = _seed(tmp_path, previous=_board())
        _stub_tracker(monkeypatch, existing_issue=7)
        monkeypatch.setattr("crucible.morning._krepis_publish", lambda *a, **k: _Result())
        monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)
        monkeypatch.delenv("CRUCIBLE_TRIGGER", raising=False)

        assert morning_handler(_args(tmp_path, dry_run=False)) == 0

        keys = [k for k in store.list_keys(f"runs/{MORNING_JOB}/") if k.endswith("run.json")]
        manifest = json.loads(store.get_bytes(keys[0]))
        assert manifest["status"] == "ok"

        message_row = _message_output(manifest)
        message = store.get_bytes(message_row["key"]).decode()
        assert message.startswith("<b>CRUCIBLE V2 —")
        assert message_row["key"] == morning_report_key(DAY.isoformat(), manifest["calendar_date"])

        update_row = _update_output(manifest)
        update = store.get_bytes(update_row["key"]).decode()
        assert "## Ladder" in update
        assert update_row["key"] == morning_update_key(DAY.isoformat(), manifest["calendar_date"])

        assert {o["key"] for o in manifest["outputs"]} == {
            message_row["key"],
            update_row["key"],
            morning_trigger_key(DAY.isoformat(), manifest["calendar_date"], TRIGGER_UNKNOWN),
        }

    def test_the_manifest_records_the_comment_url_and_issue_number_as_a_metric(
        self, tmp_path, monkeypatch
    ):
        """`alpha-engine-config-I10123` deliverable 4: the comment URL and
        issue number are recorded on the manifest's `metrics`, not `outputs`
        (`run_manifest.v2.json`'s `outputs` array is content-hashed file
        references — a metric is the right slot for a fact that is not a
        file this job wrote)."""
        store = _seed(tmp_path, previous=_board())
        _stub_tracker(monkeypatch, existing_issue=42, comment_url=UPDATE_URL)
        monkeypatch.setattr("crucible.morning._krepis_publish", lambda *a, **k: _Result())

        assert morning_handler(_args(tmp_path, dry_run=False)) == 0

        keys = [k for k in store.list_keys(f"runs/{MORNING_JOB}/") if k.endswith("run.json")]
        manifest = json.loads(store.get_bytes(keys[0]))
        (metric,) = [
            m for m in manifest["metrics"] if m["name"] == "morning_report_tracker_comment"
        ]
        assert metric["value"] == 42.0
        assert metric["source_path"] == UPDATE_URL
        assert TRACKER_REPO in metric["status_reason"]
        assert "42" in metric["status_reason"]
        assert UPDATE_URL in metric["status_reason"]

    def test_a_scheduled_firing_files_evidence_that_no_human_started_it(
        self, tmp_path, monkeypatch
    ):
        store = _seed(tmp_path, previous=_board())
        _stub_tracker(monkeypatch, existing_issue=7)
        monkeypatch.setattr("crucible.morning._krepis_publish", lambda *a, **k: _Result())
        monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
        assert morning_handler(_args(tmp_path, dry_run=False)) == 0
        keys = [k for k in store.list_keys(f"runs/{MORNING_JOB}/") if k.endswith("run.json")]
        manifest = json.loads(store.get_bytes(keys[0]))
        expected = morning_trigger_key(DAY.isoformat(), manifest["calendar_date"], "schedule")
        assert expected in {o["key"] for o in manifest["outputs"]}
        assert store.get_bytes(expected).decode().strip() == "schedule"

    def test_a_dispatched_firing_cannot_file_evidence_of_a_schedule(self, tmp_path, monkeypatch):
        store = _seed(tmp_path, previous=_board())
        _stub_tracker(monkeypatch, existing_issue=7)
        monkeypatch.setattr("crucible.morning._krepis_publish", lambda *a, **k: _Result())
        monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
        monkeypatch.setenv("CRUCIBLE_TRIGGER", "schedule")
        assert morning_handler(_args(tmp_path, dry_run=False)) == 0
        assert not [k for k in store.list_keys(f"runs/{MORNING_JOB}/") if k.endswith(".schedule")]

    def test_the_manifest_key_is_discriminated_by_the_firing(self, tmp_path, monkeypatch):
        store = _seed(tmp_path, previous=_board())
        _stub_tracker(monkeypatch, existing_issue=7)
        monkeypatch.setattr("crucible.morning._krepis_publish", lambda *a, **k: _Result())
        morning_handler(_args(tmp_path, dry_run=False))
        keys = [k for k in store.list_keys(f"runs/{MORNING_JOB}/") if k.endswith("run.json")]
        assert keys[0] != manifest_key(MORNING_JOB, DAY.isoformat())
        assert keys[0].startswith(f"runs/{MORNING_JOB}/{DAY.isoformat()}/")

    def test_a_dry_run_delivers_nothing_touches_no_tracker_and_files_no_manifest(
        self, tmp_path, monkeypatch
    ):
        """`alpha-engine-config-I9922`/`-I10123`: `--dry-run` renders and
        files nothing, sends nothing, and touches the private tracker not at
        all — no search, no create, no comment."""
        store = _seed(tmp_path, previous=_board())

        def refuse(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("--dry-run must not reach this")

        monkeypatch.setattr("crucible.morning._krepis_publish", refuse)
        monkeypatch.setattr("crucible.morning.tracker.find_issue_by_title", refuse)
        monkeypatch.setattr("crucible.morning.tracker.create_issue", refuse)
        monkeypatch.setattr("crucible.morning.tracker.post_comment", refuse)

        assert morning_handler(_args(tmp_path, dry_run=True)) == 0
        keys = [k for k in store.list_keys(f"runs/{MORNING_JOB}/") if k.endswith("run.json")]
        assert keys == []


# ── declarations that must agree ────────────────────────────────────────


class TestDeclarations:
    def test_the_job_is_registered_in_the_cli_table_and_has_a_real_handler(self):
        assert JOBS[MORNING_JOB].scheduled is True
        from crucible.cli import is_stub

        assert not is_stub(HANDLERS[MORNING_JOB])

    def test_the_registry_row_declares_the_workflow_that_crons_it(self):
        row = load_registry()[MORNING_JOB]
        assert row.dispatch == "github-actions"
        assert row.dispatch_workflow == WORKFLOW.name
        assert row.absence_watched_by == "alerts.sweep"

    def test_the_workflow_cron_is_0600_pdt_and_0500_pst(self):
        spec = yaml.safe_load(WORKFLOW.read_text())
        crons = [entry["cron"] for entry in spec[True]["schedule"]]
        assert crons == ["0 13 * * *"]
        summer = dt.datetime(2026, 9, 3, 13, 0, tzinfo=dt.UTC).astimezone(DELIVERY_TZ)
        winter = dt.datetime(2026, 11, 4, 13, 0, tzinfo=dt.UTC).astimezone(DELIVERY_TZ)
        assert (summer.hour, summer.tzname()) == (6, "PDT")
        assert (winter.hour, winter.tzname()) == (5, "PST")

    def test_a_failed_run_notifies_because_alerts_sweep_has_never_produced(self):
        spec = yaml.safe_load(WORKFLOW.read_text())
        assert list(spec["jobs"]) == ["report", "notify-failure"]
        notify = spec["jobs"]["notify-failure"]
        assert notify["needs"] == ["report"]
        assert "failure()" in notify["if"]
        assert notify["uses"] == (
            "nousergon/nousergon-lib/.github/workflows/notify-ci-failure.yml"
            "@619de3f32aef2e63329e7270afb87ec6b1e381e3"
        )
        assert notify["secrets"] == "inherit"

    def test_the_workflow_passes_the_tracker_app_ssm_prefix_like_board_yml(self):
        """`alpha-engine-config-I10123`: `morning-report.yml` passes
        `CRUCIBLE_TRACKER_APP_SSM_PREFIX` the same way `board.yml` does."""
        spec = yaml.safe_load(WORKFLOW.read_text())
        assert spec["env"]["CRUCIBLE_TRACKER_APP_SSM_PREFIX"] == (
            "${{ vars.CRUCIBLE_TRACKER_APP_SSM_PREFIX }}"
        )

    def test_the_workflow_is_not_reachable_on_a_pull_request(self):
        spec = yaml.safe_load(WORKFLOW.read_text())
        assert set(spec[True]) == {"schedule", "workflow_dispatch"}

    def test_the_report_writes_only_under_its_own_manifest_prefix(self):
        assert morning_report_key(DAY.isoformat(), "2026-09-03").startswith(
            f"runs/{MORNING_JOB}/{DAY.isoformat()}/"
        )
        assert morning_update_key(DAY.isoformat(), "2026-09-03").startswith(
            f"runs/{MORNING_JOB}/{DAY.isoformat()}/"
        )

    def test_the_silent_states_do_not_shadow_the_boards_grey_states(self):
        from crucible.board import GREY_STATES as BOARD_GREY
        from crucible.morning import SILENT_STATES

        assert set(SILENT_STATES).isdisjoint(BOARD_GREY)


# ── one artifact, one parse, two surfaces (review F2) ─────────────────────


class TestOneAcceptanceReaderForBothSurfaces:
    """The full update and the board page it links to read the SAME document
    with the SAME rule, or one of them is lying about the other."""

    COMPLETE: dict[str, Any] = {
        "met": 21,
        "unmet": 2,
        "unmeasurable": 1,
        "commit": SHA,
        "measured_at": MEASURED_AT,
    }
    #: The review's own document: three counts, no commit.
    NO_COMMIT: dict[str, Any] = {"met": 21, "unmet": 2, "unmeasurable": 1}

    @staticmethod
    def _both(tmp_path, document: dict[str, Any]) -> tuple[str, str]:
        import argparse

        from crucible.keys import acceptance_reading_key
        from crucible.track_c import board_handler

        store = _seed(tmp_path, previous=_board())
        store.put_bytes(acceptance_reading_key(DAY.isoformat()), json.dumps(document).encode())
        board_handler(argparse.Namespace(trading_day=DAY, store=str(tmp_path)))
        update = run_report(store, trading_day=DAY, now=FIRED_AT)
        return update, store.get_bytes(BOARD_HTML_KEY).decode()

    def test_a_document_with_no_commit_is_refused_by_both(self, tmp_path):
        update, page = self._both(tmp_path, self.NO_COMMIT)
        assert ACCEPTANCE_NOT_ON_ANY_ARTIFACT in update
        assert "answers a different question" in page
        assert "21 met" not in page
        assert "UNKNOWN" not in page.split("<h2>objective")[0]

    def test_a_complete_document_renders_the_same_figures_on_both(self, tmp_path):
        update, page = self._both(tmp_path, self.COMPLETE)
        assert "acceptance count: 21 met / 2 unmet / 1 unmeasurable of 24" in update
        assert "21 met / 2 unmet / 1 unmeasurable</strong> of 24 clauses" in page
        assert SHA in update
        assert SHA in page
        assert ACCEPTANCE_NOT_ON_ANY_ARTIFACT not in update

    @pytest.mark.parametrize(
        "document",
        [
            {"met": 21, "unmet": 2, "unmeasurable": 1},
            {"met": 21, "unmet": 2, "commit": SHA},
            {"met": "21", "unmet": 2, "unmeasurable": 1, "commit": SHA},
            {"met": True, "unmet": 2, "unmeasurable": 1, "commit": SHA},
            {"met": 21, "unmet": 2, "unmeasurable": 1, "commit": ""},
        ],
    )
    def test_every_incomplete_shape_is_refused_by_both(self, tmp_path, document):
        update, page = self._both(tmp_path, document)
        assert ACCEPTANCE_NOT_ON_ANY_ARTIFACT in update
        assert "answers a different question" in page

    def test_the_shared_parser_is_the_only_completeness_rule(self):
        from crucible.keys import ACCEPTANCE_REQUIRED_FIELDS, parse_acceptance_reading

        assert ACCEPTANCE_REQUIRED_FIELDS == ("met", "unmet", "unmeasurable", "commit")
        assert parse_acceptance_reading(self.NO_COMMIT) is None
        parsed = parse_acceptance_reading(self.COMPLETE)
        assert parsed is not None
        assert parsed.total == 24
        assert parsed.unmet_clauses is None, "an absent list is 'not filed', never 'there are none'"


def test_read_inputs_returns_the_declared_shape(tmp_path):
    store = _seed(tmp_path, previous=_board())
    inputs = read_inputs(store, trading_day=DAY, now=FIRED_AT)
    assert isinstance(inputs, MorningInputs)
    assert inputs.board["trading_day"] == DAY.isoformat()
    assert inputs.previous is not None
    assert inputs.previous_day == PREVIOUS


def test_resolve_trigger_is_unaffected_by_the_rewrite(monkeypatch):
    monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
    assert resolve_trigger() == "schedule"
    monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)
    monkeypatch.delenv("CRUCIBLE_TRIGGER", raising=False)
    assert resolve_trigger({}) == TRIGGER_UNKNOWN
