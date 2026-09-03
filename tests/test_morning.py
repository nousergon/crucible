"""The morning report's guards — `alpha-engine-config-I9896`.

What is asserted here is what would make the report WORSE THAN NOTHING,
which is the state it replaces: Brian believed a 6am PT report existed for
weeks while nothing sent one.

* **A stale board reported as current.** The readings are only as good as the
  render they came from, and a reader who learns the board is three days old
  after acting on it has been misled by a surface built to prevent exactly
  that.
* **"Nothing moved" over a comparison that failed.** A positive claim of no
  movement asserted on no evidence is the shape `crucible.board` already
  refuses one layer down.
* **A delivery that did not happen, recorded as one.** The delivery IS the
  deliverable; a rendered report nobody received is the silent swallow the
  whole plan is a reaction to.
* **An invented acceptance count.** §12 rule 3 makes it the only progress
  figure, which is exactly why reading it out of the running checkout and
  printing it as `main`'s reading would be worse than saying nothing.
* **A progress narrative.** No PR count, no commit count, no findings count.

Fixed date literals throughout (`crucible/AGENTS.md`, "a test whose subject
moves with the clock stops testing the same thing").
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import re
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
    board_key,
    manifest_key,
    morning_report_key,
)
from crucible.morning import (
    ACCEPTANCE_NOT_ON_ANY_ARTIFACT,
    ACCEPTANCE_UNREADABLE,
    BOARD_URL_EXPIRES_S,
    BOARD_URL_UNAVAILABLE,
    DELIVERY_SEVERITY,
    DELIVERY_SOURCE,
    DELIVERY_TZ,
    MESSAGE_MAX_CHARS,
    MORNING_JOB,
    NO_OPERATOR_ACTION,
    SECTIONS,
    STALE_AFTER,
    TRANSPORT_PREFIX,
    TRUNCATION_MARKER,
    MorningInputs,
    UndeliveredError,
    deliver,
    morning_handler,
    read_inputs,
    render_message,
    run_report,
    store_uri,
    wire_length,
)
from crucible.store import LocalStore

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


@pytest.fixture(autouse=True)
def _freeze_morning_now(monkeypatch: pytest.MonkeyPatch) -> None:
    """Handler paths take `now` from wall clock via `run_job`.

    Lock that clock to FIRED_AT so fixtures whose `generated_at` is relative
    to FIRED_AT stay fresh without depending on the day CI runs. Tests that
    pass an explicit `now=` to `run_report` are unaffected.
    """
    import crucible.runner as runner

    real_run_job = runner.run_job

    def _run_job_at_fired_at(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("now", FIRED_AT)
        return real_run_job(*args, **kwargs)

    monkeypatch.setattr(runner, "run_job", _run_job_at_fired_at)


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
        # The page the message links to. Seeded by default because the real
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


# ── the message ───────────────────────────────────────────────────────────


class TestTheMessage:
    def test_the_exact_message_for_a_fresh_board_with_one_row_moved(self, tmp_path):
        previous = _board(
            rows=[
                _row("phase0", "phase", "UNMET", "old"),
                _row("phase1", "phase", "OUT_OF_ORDER", "old"),
                _row("obj:cost", "objective", "PLANNED", "old"),
                _row("obj:alpha", "objective", "MET", "old"),
            ]
        )
        store = _seed(tmp_path, previous=previous)
        message = run_report(store, trading_day=DAY, now=FIRED_AT)
        assert message == "\n".join(
            [
                "CRUCIBLE V2 — BOARD FOR TRADING DAY 2026-09-02",
                f"store: {tmp_path}/board/current.json",
                f"generated: {GENERATED}  commit: {SHA}",
                "delivered: 2026-09-03 06:00 PDT",
                "",
                "LADDER",
                "  phase0  UNMET  1/2",
                "    holding: b_unmet",
                "  phase1  OUT_OF_ORDER  0/1",
                "    holding: c_unmet",
                "    out of order: 1 of 5 clauses — phase0's gate is not met",
                "",
                "SCHEDULE (PLAN §6.1)",
                "  no schedule row on the board — the plan §6.1 milestones are not being "
                "rendered, which is a defect in the board, not an absent plan",
                "",
                "ACCEPTANCE",
                f"  {ACCEPTANCE_NOT_ON_ANY_ARTIFACT}",
                "",
                "MOVED SINCE 2026-09-01",
                "  obj:cost: PLANNED -> UNMEASURED",
                "",
                "SILENCE",
                "  silence: 1 UNMEASURED, 0 UNMEASURABLE of 4 rows",
                f"  {NO_OPERATOR_ACTION}",
                "",
                "FULL BOARD",
                f"  {(tmp_path / 'board' / 'index.html').resolve().as_uri()}",
                "  presigned GET, expires 2026-09-10T13:00:00Z or when the signing role's "
                "session ends, whichever is first",
            ]
        )

    def test_a_stale_board_is_the_first_line_and_not_a_footnote(self, tmp_path):
        store = _seed(
            tmp_path,
            board=_board(generated_at=STALE_GENERATED),
            previous=_board(),
        )
        message = run_report(store, trading_day=DAY, now=FIRED_AT)
        assert message.splitlines()[0].startswith("STALE BOARD:")
        assert STALE_GENERATED in message.splitlines()[0]

    def test_a_board_generated_within_the_day_carries_no_stale_headline(self, tmp_path):
        store = _seed(tmp_path, previous=_board())
        message = run_report(store, trading_day=DAY, now=FIRED_AT)
        assert "STALE BOARD" not in message

    def test_an_unparseable_generated_at_is_stale_rather_than_assumed_fresh(self, tmp_path):
        store = _seed(tmp_path, board=_board(generated_at="yesterday"), previous=_board())
        message = run_report(store, trading_day=DAY, now=FIRED_AT)
        assert message.splitlines()[0].startswith("STALE BOARD:")

    def test_an_unreadable_previous_board_is_never_reported_as_nothing_moved(self, tmp_path):
        store = _seed(tmp_path)  # no previous board at all
        message = run_report(store, trading_day=DAY, now=FIRED_AT)
        assert "nothing moved" not in message
        assert "cannot say" in message
        assert board_key(PREVIOUS.isoformat()) in message

    def test_a_corrupt_previous_board_names_the_corruption(self, tmp_path):
        store = _seed(tmp_path)
        store.put_bytes(board_key(PREVIOUS.isoformat()), b"{not json")
        message = run_report(store, trading_day=DAY, now=FIRED_AT)
        assert "unreadable at" in message

    def test_an_identical_board_reports_nothing_moved(self, tmp_path):
        store = _seed(tmp_path, previous=_board())
        assert "  nothing moved" in run_report(store, trading_day=DAY, now=FIRED_AT)

    def test_a_vanished_row_is_reported_rather_than_silently_dropped(self, tmp_path):
        previous = _board(rows=[*_board()["rows"], _row("obj:gone", "objective", "MET", "was")])
        store = _seed(tmp_path, previous=previous)
        assert "obj:gone: MET -> VANISHED" in run_report(store, trading_day=DAY, now=FIRED_AT)

    def test_a_filed_acceptance_reading_is_quoted_with_its_commit(self, tmp_path):
        """§12 rule 3's one progress figure, READ rather than asserted. The
        producer contract is `crucible.keys.acceptance_reading_key`'s
        docstring; `alpha-engine-config-I9902` implements it."""
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
        message = run_report(store, trading_day=DAY, now=FIRED_AT)
        assert (
            f"acceptance count: 21 met / 3 unmet / 0 unmeasurable of 24 (commit {SHA})" in message
        )
        assert ACCEPTANCE_NOT_ON_ANY_ARTIFACT not in message

    def test_a_corrupt_acceptance_reading_is_absent_rather_than_guessed(self, tmp_path):
        """A malformed reading is not a number. Rendering a partial parse of
        the only progress figure is exactly the fabrication the literal
        exists to avoid."""
        from crucible.keys import acceptance_reading_key

        store = _seed(tmp_path, previous=_board())
        store.put_bytes(acceptance_reading_key(DAY.isoformat()), b"{not json")
        assert ACCEPTANCE_NOT_ON_ANY_ARTIFACT in run_report(store, trading_day=DAY, now=FIRED_AT)

    def test_an_access_denied_acceptance_read_is_unreadable_not_absent(self, tmp_path, monkeypatch):
        """§6 rule 1: an ACCESS FAILURE reported as absence conflates a
        permissions gap with the artifact never having been filed —
        `alpha-engine-config-I9896` measured this exact `AccessDenied` on the
        first live `report.morning` run. The run manifest must still read
        `ok`: this artifact is OPTIONAL, and a report that cannot read it
        still has four other things to say."""
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
        # `dry_run=` accepted and ignored: this test exercises the real
        # (`dry_run=False`) path — `morning.py`'s `open_store` call now
        # passes the kwarg unconditionally (alpha-engine-config-I9922 N1).
        monkeypatch.setattr("crucible.morning.open_store", lambda uri, dry_run=False: denied)
        monkeypatch.setattr("crucible.morning._krepis_publish", lambda *a, **k: _Result())

        assert morning_handler(_args(tmp_path, dry_run=False)) == 0

        keys = [k for k in denied.list_keys(f"runs/{MORNING_JOB}/") if k.endswith("run.json")]
        manifest = json.loads(denied.get_bytes(keys[0]))
        assert manifest["status"] == "ok"
        (output,) = manifest["outputs"]
        message = denied.get_bytes(output["key"]).decode()
        assert ACCEPTANCE_UNREADABLE.format(code="AccessDenied") in message
        assert ACCEPTANCE_NOT_ON_ANY_ARTIFACT not in message

    def test_a_not_found_acceptance_read_is_still_the_absent_literal(self, tmp_path):
        """The same `_DeniedKeyStore` shape, a `NoSuchKey` code instead of
        `AccessDenied` — proving the new access-failure branch does not
        swallow the ordinary not-found case it sits beside."""
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
        message = run_report(store, trading_day=DAY, now=FIRED_AT)
        assert ACCEPTANCE_NOT_ON_ANY_ARTIFACT in message
        assert "unreadable" not in message.lower()

    def test_an_access_denied_previous_board_is_unreadable_not_absent_or_a_crash(self, tmp_path):
        """Same three-way handling on the previous-day board read (it goes
        through the same `_read_json` helper as acceptance): denied is named,
        never rendered as `nothing moved` and never crashes the report."""
        store = _DeniedKeyStore(tmp_path, denied={board_key(PREVIOUS.isoformat()): "AccessDenied"})
        store.put_bytes(BOARD_CURRENT_KEY, json.dumps(_board()).encode())
        store.put_bytes(
            manifest_key("board", DAY.isoformat()),
            json.dumps({"status": "ok", "code_sha": SHA, "reason": ""}).encode(),
        )
        message = run_report(store, trading_day=DAY, now=FIRED_AT)
        assert "nothing moved" not in message
        assert "cannot say" in message
        assert "AccessDenied" in message

    def test_an_access_denied_current_board_still_raises(self, tmp_path):
        """`board/current.json` is read directly, not through `_read_json` —
        the current board is not optional, so a denied read must still fail
        the manifest rather than being absorbed like the optional reads."""
        store = _DeniedKeyStore(tmp_path, denied={BOARD_CURRENT_KEY: "AccessDenied"})
        with pytest.raises(ClientError):
            run_report(store, trading_day=DAY, now=FIRED_AT)

    def test_the_acceptance_count_is_never_invented(self, tmp_path):
        """§12 rule 3 makes it the only progress figure — which is exactly why
        a fabricated one is worse than an absent one. `tests/acceptance/
        ratchet.json` is committed to git and no store artifact republishes it,
        so this line is the honest reading until a producer files one."""
        store = _seed(tmp_path, previous=_board())
        assert ACCEPTANCE_NOT_ON_ANY_ARTIFACT in run_report(store, trading_day=DAY, now=FIRED_AT)

    def test_silence_is_reported_as_its_own_figure(self, tmp_path):
        rows = [
            _row("phase0", "phase", "UNMET", "d"),
            _row("a", "objective", "UNMEASURED", "d"),
            _row("b", "objective", "UNMEASURABLE", "d"),
        ]
        store = _seed(tmp_path, board=_board(rows=rows), previous=_board(rows=rows))
        line = [
            ln
            for ln in run_report(store, trading_day=DAY, now=FIRED_AT).splitlines()
            if ln.strip().startswith("silence:")
        ]
        assert line == ["  silence: 1 UNMEASURED, 1 UNMEASURABLE of 3 rows"]

    def test_a_failed_board_render_becomes_the_pending_operator_action(self, tmp_path):
        store = _seed(
            tmp_path,
            previous=_board(),
            board_run={"status": "failed", "code_sha": SHA, "reason": "AccessDenied on PutObject"},
        )
        message = run_report(store, trading_day=DAY, now=FIRED_AT)
        assert "pending operator action: the board render for 2026-09-02 failed" in message
        assert "AccessDenied on PutObject" in message

    def test_a_missing_board_manifest_is_named_rather_than_a_blank_commit(self, tmp_path):
        store = _seed(tmp_path, previous=_board(), board_run=None)
        message = run_report(store, trading_day=DAY, now=FIRED_AT)
        assert "commit: UNKNOWN" in message
        assert "filed no manifest" in message

    def test_a_board_with_no_phase_row_is_a_finding_not_an_empty_section(self, tmp_path):
        rows = [_row("obj:alpha", "objective", "MET", "d")]
        store = _seed(tmp_path, board=_board(rows=rows), previous=_board(rows=rows))
        assert "no phase row on the board" in run_report(store, trading_day=DAY, now=FIRED_AT)

    def test_the_message_carries_no_progress_narrative(self, tmp_path):
        """Plan §12 rule 3: PRs merged, findings and commits are NOT progress,
        and putting them beside a real figure lends them its authority."""
        store = _seed(tmp_path, previous=_board())
        message = run_report(store, trading_day=DAY, now=FIRED_AT).lower()
        for forbidden in ("pull request", " prs ", "merged", "progress", "findings", "commits"):
            assert forbidden not in message

    def test_a_board_detail_carrying_a_progress_figure_is_withheld(self, tmp_path):
        """The guard above grades the TEMPLATE unless the fixture's details
        contain the phrasing. `row['detail']` is the board's free text and is
        rendered verbatim, so a detail reading "3 of 5 PRs merged" ships
        straight onto the one surface §12 rule 3 exists for."""
        rows = [
            _row(
                "phase0",
                "phase",
                "UNMET",
                "1/2 clauses met; 3 of 5 PRs merged this week; holding: old_weekly",
            )
        ]
        store = _seed(tmp_path, board=_board(rows=rows), previous=_board(rows=rows))
        message = run_report(store, trading_day=DAY, now=FIRED_AT)
        assert "PRs merged" not in message
        # The surviving clauses are kept, and the withholding is DECLARED --
        # a clause silently dropped is indistinguishable from a board that
        # never carried it.
        assert "1/2 clauses met" in message
        assert "holding: old_weekly" in message
        assert "[1 clause withheld — plan §12 rule 3]" in message

    def test_a_detail_with_no_progress_figure_is_rendered_verbatim(self, tmp_path):
        """A scrubber that rewrites clean text would make the report an
        unfaithful copy of the board, which is worse than the defect."""
        detail = "1/2 clauses met; holding: old_weekly_within_cadence; grades 2 of 5 deliverables"
        rows = [_row("phase0", "phase", "UNMET", detail)]
        store = _seed(tmp_path, board=_board(rows=rows), previous=_board(rows=rows))
        message = run_report(store, trading_day=DAY, now=FIRED_AT)
        assert f"  phase0  UNMET  {detail}" in message
        assert "withheld" not in message

    def test_a_detail_that_is_entirely_a_progress_figure_leaves_the_marker(self, tmp_path):
        """The row never vanishes. A phase row missing from the report is the
        one thing the board's own guards refuse one layer down."""
        rows = [_row("phase0", "phase", "UNMET", "6 findings this week")]
        store = _seed(tmp_path, board=_board(rows=rows), previous=_board(rows=rows))
        message = run_report(store, trading_day=DAY, now=FIRED_AT)
        assert "  phase0  UNMET  [1 clause withheld — plan §12 rule 3]" in message

    def test_the_reading_is_quoted_with_its_store(self, tmp_path):
        """Plan §6 rule 2. Derived from the live store, never a literal."""
        store = _seed(tmp_path, previous=_board())
        assert f"store: {tmp_path}/{BOARD_CURRENT_KEY}" in run_report(
            store, trading_day=DAY, now=FIRED_AT
        )

    def test_an_absent_current_board_raises_rather_than_reporting_on_nothing(self, tmp_path):
        store = LocalStore(tmp_path)
        with pytest.raises(KeyError):
            run_report(store, trading_day=DAY, now=FIRED_AT)


# ── the schedule section (alpha-engine-config-I9914) ──────────────────────


class TestSchedule:
    """Plan §6.1's milestone table, as its own report section.

    This section renders `schedule:*` board rows exactly the way `phase
    gates` renders `phase:*` rows -- it computes nothing new, it quotes what
    `crucible.board._schedule_rows` already decided (`test_board.py` grades
    that decision). What is graded here is that the section exists, sits
    between `phase gates` and `moved since` (the exact-message test above),
    and that a past-due row's line begins with the word `OVERDUE`.
    """

    def test_a_met_milestone_carries_no_overdue_word(self, tmp_path):
        rows = [_row("schedule:day5_replays", "schedule", "MET", "met — due 2026-09-04")]
        store = _seed(tmp_path, board=_board(rows=rows), previous=_board(rows=rows))
        message = run_report(store, trading_day=DAY, now=FIRED_AT)
        assert "  schedule:day5_replays  MET  met — due 2026-09-04" in message
        assert "OVERDUE" not in message

    def test_an_unmet_past_due_milestone_line_begins_with_overdue(self, tmp_path):
        rows = [
            _row(
                "schedule:day5_replays",
                "schedule",
                "UNMET",
                "OVERDUE since 2026-09-04 — reads: phase1 1/6",
            )
        ]
        store = _seed(tmp_path, board=_board(rows=rows), previous=_board(rows=rows))
        message = run_report(store, trading_day=DAY, now=FIRED_AT)
        line = next(ln for ln in message.splitlines() if "schedule:day5_replays" in ln)
        assert line.strip().split()[0] == "OVERDUE", (
            f"the first word of a past-due schedule line must be OVERDUE, got: {line!r}"
        )

    def test_a_planned_milestone_carries_no_overdue_word(self, tmp_path):
        rows = [
            _row(
                "schedule:live1",
                "schedule",
                "PLANNED",
                "due 2026-09-12, waiting on phase2 UNMEASURED",
            )
        ]
        store = _seed(tmp_path, board=_board(rows=rows), previous=_board(rows=rows))
        message = run_report(store, trading_day=DAY, now=FIRED_AT)
        assert "  schedule:live1  PLANNED  due 2026-09-12" in message
        assert "OVERDUE" not in message

    def test_a_schedule_detail_carrying_a_progress_figure_is_withheld(self, tmp_path):
        """Schedule details are ours to write, but this section still passes
        through `_withhold_progress` like every other section — this module
        trusts no board free text unchecked."""
        rows = [
            _row(
                "schedule:live1",
                "schedule",
                "UNMET",
                "OVERDUE since 2026-09-04; 3 PRs merged this week",
            )
        ]
        store = _seed(tmp_path, board=_board(rows=rows), previous=_board(rows=rows))
        message = run_report(store, trading_day=DAY, now=FIRED_AT)
        assert "PRs merged" not in message
        assert "[1 clause withheld — plan §12 rule 3]" in message

    def test_no_schedule_rows_is_a_named_defect_not_an_empty_section(self, tmp_path):
        store = _seed(tmp_path, previous=_board())  # default fixture carries no schedule rows
        message = run_report(store, trading_day=DAY, now=FIRED_AT)
        assert "no schedule row on the board" in message

    def test_the_schedule_section_sits_between_phase_gates_and_moved_since(self, tmp_path):
        rows = [
            *_board()["rows"],
            _row("schedule:day5_replays", "schedule", "MET", "met — due 2026-09-04"),
        ]
        store = _seed(tmp_path, board=_board(rows=rows), previous=_board(rows=rows))
        message = run_report(store, trading_day=DAY, now=FIRED_AT)
        lines = message.splitlines()
        phase_at = lines.index("LADDER")
        schedule_at = lines.index("SCHEDULE (PLAN §6.1)")
        moved_at = next(i for i, ln in enumerate(lines) if ln.startswith("MOVED SINCE"))
        assert phase_at < schedule_at < moved_at


# ── the store it quotes ───────────────────────────────────────────────────


class TestStoreUri:
    def test_a_local_store_names_its_root(self, tmp_path):
        assert store_uri(LocalStore(tmp_path)) == str(tmp_path)

    def test_an_s3_store_names_bucket_and_prefix(self):
        from crucible.store import S3Store

        assert store_uri(S3Store("b", "p/q")) == "s3://b/p/q"
        assert store_uri(S3Store("b")) == "s3://b"

    def test_a_backend_that_cannot_name_itself_raises(self):
        """A repr() fallback would put an object address on the operator's
        phone in the place plan §6 rule 2 requires the store."""

        class Nameless:
            pass

        with pytest.raises(TypeError, match="cannot name itself"):
            store_uri(Nameless())  # type: ignore[arg-type]


# ── delivery ──────────────────────────────────────────────────────────────


class TestDelivery:
    def test_it_never_touches_the_page_path(self):
        transport = _Transport()
        deliver("hello", transport=transport)
        (call,) = transport.calls
        assert call["sns"] is False
        assert call["telegram"] is True
        assert call["severity"] == "info"
        # NOT silent: the 2026-09-03 delivery went out silent and was not
        # seen (alpha-engine-config-I9916). A notification is not a page.
        assert call["silent"] is False
        assert call["raise_on_total_failure"] is True
        # Explicit, never left to `krepis.alerts.resolve_destination`'s
        # fallback: an `info` severity reaches the operator chat TODAY only
        # because no log chat is configured, so configuring
        # TELEGRAM_LOG_CHAT_ID fleet-wide would silently move this report off
        # Brian's chat with the manifest still reading `ok`.
        assert call["destination"] == "operator_chat"

    def test_it_carries_no_dedup_key(self):
        """An identical report must still arrive tomorrow: a digest that stops
        arriving when nothing changed is indistinguishable from one that
        stopped arriving."""
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
    def test_a_failed_delivery_raises_and_the_manifest_reads_failed(self, tmp_path, monkeypatch):
        store = _seed(tmp_path, previous=_board())
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

    def test_a_delivered_report_is_filed_beside_its_manifest(self, tmp_path, monkeypatch):
        store = _seed(tmp_path, previous=_board())
        monkeypatch.setattr("crucible.morning._krepis_publish", lambda *a, **k: _Result())
        assert morning_handler(_args(tmp_path, dry_run=False)) == 0
        keys = [k for k in store.list_keys(f"runs/{MORNING_JOB}/") if k.endswith("run.json")]
        manifest = json.loads(store.get_bytes(keys[0]))
        assert manifest["status"] == "ok"
        (output,) = manifest["outputs"]
        message = store.get_bytes(output["key"]).decode()
        assert message.startswith("CRUCIBLE V2 — BOARD FOR TRADING DAY")
        assert output["key"] == morning_report_key(DAY.isoformat(), manifest["calendar_date"])

    def test_the_manifest_key_is_discriminated_by_the_firing(self, tmp_path, monkeypatch):
        """A 13:00 UTC cron fires every calendar day while three of them
        resolve to Friday's close. Without the discriminator the weekend
        deliveries overwrite one another."""
        store = _seed(tmp_path, previous=_board())
        monkeypatch.setattr("crucible.morning._krepis_publish", lambda *a, **k: _Result())
        morning_handler(_args(tmp_path, dry_run=False))
        keys = [k for k in store.list_keys(f"runs/{MORNING_JOB}/") if k.endswith("run.json")]
        assert keys[0] != manifest_key(MORNING_JOB, DAY.isoformat())
        assert keys[0].startswith(f"runs/{MORNING_JOB}/{DAY.isoformat()}/")

    def test_a_dry_run_delivers_nothing_and_files_no_manifest(self, tmp_path, monkeypatch):
        """alpha-engine-config-I9922. Before this fix, `run_job` wrote a
        manifest regardless of `--dry-run` — this test itself asserted that
        as correct (`manifest["outputs"] == []`), which is exactly the
        documented-but-false claim the issue names: `--dry-run` "renders and
        files nothing", and a manifest at `runs/report.morning/{day}/{firing}/
        run.json` is a real firing that `alerts.sweep` and the board read as
        genuine. A dry run against production must leave the store
        completely untouched under this job's own `runs/` prefix."""
        store = _seed(tmp_path, previous=_board())

        def refuse(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("--dry-run must not reach the transport")

        monkeypatch.setattr("crucible.morning._krepis_publish", refuse)
        assert morning_handler(_args(tmp_path, dry_run=True)) == 0
        keys = [k for k in store.list_keys(f"runs/{MORNING_JOB}/") if k.endswith("run.json")]
        assert keys == []


# ── the three declarations that must agree ────────────────────────────────


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
        """The DST fact, committed. GitHub crons are UTC and have no timezone,
        so November's shift is a fact a reader can look up here rather than a
        surprise on the morning it happens."""
        spec = yaml.safe_load(WORKFLOW.read_text())
        crons = [entry["cron"] for entry in spec[True]["schedule"]]
        assert crons == ["0 13 * * *"]
        summer = dt.datetime(2026, 9, 3, 13, 0, tzinfo=dt.UTC).astimezone(DELIVERY_TZ)
        winter = dt.datetime(2026, 11, 4, 13, 0, tzinfo=dt.UTC).astimezone(DELIVERY_TZ)
        assert (summer.hour, summer.tzname()) == (6, "PDT")
        assert (winter.hour, winter.tzname()) == (5, "PST")

    def test_a_failed_run_notifies_because_alerts_sweep_has_never_produced(self):
        """MEASURED 2026-09-03 (`alpha-engine-config-I9905`): `alerts.sweep`
        has NEVER written a manifest — its dispatcher fails `RunInstances` on
        every invocation since the stack existed. The declared page path is
        therefore dark, so a 06:00 failure would tell nobody at all. This job
        carries `board.yml`'s notify block until that is measurably false."""
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

    def test_the_notify_block_names_what_would_let_it_be_removed(self):
        """A stopgap with no stated removal condition is permanent. The
        comment must name the artifact whose existence retires it, so the
        next reader can check rather than guess."""
        text = WORKFLOW.read_text()
        assert "REMOVE THIS JOB WHEN" in text
        # The artifact whose existence retires the stopgap, named so the next
        # reader can CHECK the condition rather than re-derive it.
        assert "runs/alerts.sweep/" in text
        # A tracker reference, matched by SHAPE rather than by number:
        # `tests/test_no_stale_tracker_literals.py` forbids an issue literal
        # in code, and a guard that pinned one would go stale the way the
        # thing it guards does.
        assert re.search(r"alpha-engine-config-I\d+", text)

    def test_the_workflow_is_not_reachable_on_a_pull_request(self):
        """A live-state job on a PR reports the author's own change as drift,
        or someone else's change the author cannot clear."""
        spec = yaml.safe_load(WORKFLOW.read_text())
        assert set(spec[True]) == {"schedule", "workflow_dispatch"}

    def test_the_report_writes_only_under_its_own_manifest_prefix(self):
        """The identity that writes the manifest needs no second grant to
        write the message. A report whose evidence needed a wider IAM scope
        than its own manifest would be a reason to widen the scope."""
        assert morning_report_key(DAY.isoformat(), "2026-09-03").startswith(
            f"runs/{MORNING_JOB}/{DAY.isoformat()}/"
        )

    def test_the_silent_states_do_not_shadow_the_boards_grey_states(self):
        """Two constants with one name and two meanings is how a reader
        imports the wrong one and is never told."""
        from crucible.board import GREY_STATES as BOARD_GREY
        from crucible.morning import SILENT_STATES

        assert set(SILENT_STATES).isdisjoint(BOARD_GREY)


class TestTheSixHeadedSections:
    """`alpha-engine-config-I9921` — Brian: *"I don't find the report detailed
    enough. It should at a minimum be formatted well."*

    Asserted here: the six headings exist and are in the issue's order, the
    ladder carries the clause NAMES rather than a sentence about them, and
    the message points at a page rather than trying to be one.
    """

    def test_the_six_headings_appear_in_the_order_the_issue_states(self, tmp_path):
        rows = [
            *_board()["rows"],
            _row("schedule:day5", "schedule", "MET", "met — due 2026-09-04"),
        ]
        store = _seed(tmp_path, board=_board(rows=rows), previous=_board(rows=rows))
        lines = run_report(store, trading_day=DAY, now=FIRED_AT).splitlines()
        expected = [s.format(previous_day=PREVIOUS) for s in SECTIONS]
        at = [lines.index(heading) for heading in expected]
        assert at == sorted(at), (
            f"the sections are out of order: {list(zip(expected, at, strict=True))}"
        )

    def test_the_ladder_names_the_unmet_clauses_rather_than_summarising_them(self, tmp_path):
        """A fraction with no names is a number the reader cannot act on, and
        the names are the thing the previous report left out entirely."""
        store = _seed(tmp_path, previous=_board())
        message = run_report(store, trading_day=DAY, now=FIRED_AT)
        assert "  phase0  UNMET  1/2" in message
        assert "    holding: b_unmet" in message
        assert "a_met" not in message, "a MET clause is not what is holding the phase"

    def test_a_row_whose_board_carried_no_clause_list_invents_no_fraction(self, tmp_path):
        """A board rendered before this change carries `clauses: null`.
        Printing `0/0` over it would be a measurement nobody took."""
        rows = [_row("phase0", "phase", "UNMET", "1 of 2 clauses — a sentence, not a list")]
        store = _seed(tmp_path, board=_board(rows=rows), previous=_board(rows=rows))
        message = run_report(store, trading_day=DAY, now=FIRED_AT)
        assert "  phase0  UNMET  1 of 2 clauses — a sentence, not a list" in message
        assert "0/0" not in message

    def test_a_gate_that_declares_no_clauses_is_not_rendered_as_zero_of_zero(self, tmp_path):
        rows = [_row("phase0", "phase", "UNMEASURED", "gate phase0 has no clauses", [])]
        store = _seed(tmp_path, board=_board(rows=rows), previous=_board(rows=rows))
        message = run_report(store, trading_day=DAY, now=FIRED_AT)
        assert "no clauses — this gate measured nothing" in message
        assert "0/0" not in message

    def test_the_out_of_order_reason_is_rendered_when_set(self, tmp_path):
        store = _seed(tmp_path, previous=_board())
        message = run_report(store, trading_day=DAY, now=FIRED_AT)
        assert "    out of order: 1 of 5 clauses — phase0's gate is not met" in message

    def test_overdue_schedule_rows_sort_ahead_of_the_rest(self, tmp_path):
        """The only line in that section anybody has to act on."""
        rows = [
            *_board()["rows"],
            _row("schedule:ahead", "schedule", "PLANNED", "due 2026-10-01, waiting"),
            _row("schedule:late", "schedule", "UNMET", "OVERDUE since 2026-08-01"),
        ]
        store = _seed(tmp_path, board=_board(rows=rows), previous=_board(rows=rows))
        lines = run_report(store, trading_day=DAY, now=FIRED_AT).splitlines()
        late = lines.index("  OVERDUE schedule:late  UNMET  OVERDUE since 2026-08-01")
        ahead = lines.index("  schedule:ahead  PLANNED  due 2026-10-01, waiting")
        assert late < ahead

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
        message = run_report(store, trading_day=DAY, now=FIRED_AT)
        assert "    unmet: c_replays, c_cost" in message
        assert "    unmeasurable: c_spend" in message

    def test_an_acceptance_artifact_with_no_clause_ids_says_so_rather_than_nothing(self, tmp_path):
        """ "the producer files no names" and "there are no unmet clauses" are
        different facts; an omitted line renders them identically."""
        from crucible.keys import acceptance_reading_key

        store = _seed(tmp_path, previous=_board())
        store.put_bytes(
            acceptance_reading_key(DAY.isoformat()),
            json.dumps(
                {"met": 21, "unmet": 2, "unmeasurable": 1, "commit": SHA, "measured_at": "x"}
            ).encode(),
        )
        message = run_report(store, trading_day=DAY, now=FIRED_AT)
        assert "    the artifact names no unmet clause ids" in message


class TestTheLinkToTheFullBoard:
    def test_the_message_carries_a_url_for_the_board_page(self, tmp_path):
        store = _seed(tmp_path, previous=_board())
        message = run_report(store, trading_day=DAY, now=FIRED_AT)
        assert (tmp_path / "board" / "index.html").resolve().as_uri() in message

    def test_the_expiry_is_quoted_as_a_bound_and_not_as_a_promise(self, tmp_path):
        """A presigned URL signed with temporary credentials dies with the
        credential. Quoting seven days flat would be a promise the signing
        role cannot keep, and a reader who finds a dead link having been told
        it had days left concludes the board is broken."""
        store = _seed(tmp_path, previous=_board())
        message = run_report(store, trading_day=DAY, now=FIRED_AT)
        assert "when the signing role's session ends, whichever is first" in message
        assert BOARD_URL_EXPIRES_S == 7 * 24 * 3600

    def test_a_missing_page_is_stated_rather_than_killing_the_whole_report(self, tmp_path):
        """The report is the surface that would tell anybody the page is
        missing, so it must survive the page being missing."""
        store = _seed(tmp_path, previous=_board(), page=False)
        message = run_report(store, trading_day=DAY, now=FIRED_AT)
        assert BOARD_URL_UNAVAILABLE in message
        assert "LADDER" in message, "the rest of the report must still be there"


class TestTheConsoleLinkReplacesThePresignedOne:
    """`alpha-engine-config-I9926`: when a console is configured, the FULL
    BOARD section carries its stable Decision-list address and nothing
    presigned — one link to one board, with a caveat that says it does not
    expire. When none is configured, the presigned path is untouched.
    """

    CONSOLE = "https://console.example.test"

    def test_the_console_url_is_the_link(self, tmp_path):
        store = _seed(tmp_path, previous=_board())
        message = run_report(store, trading_day=DAY, now=FIRED_AT, console_url=self.CONSOLE)
        assert f"  {self.CONSOLE}/decision?pipeline=crucible/board\n" in message

    def test_the_link_is_the_decision_list_filtered_to_this_board(self):
        # Every board row carries `surface: crucible/board`, which the console
        # fragment lifts onto the `pipeline` facet; the filter is what makes
        # the address THIS board's rather than every Decision in the fleet.
        from crucible.morning import BOARD_CONSOLE_PATH

        assert BOARD_CONSOLE_PATH == "/decision?pipeline=crucible/board"

    def test_no_expiry_is_reported_when_nothing_was_presigned(self, tmp_path):
        store = _seed(tmp_path, previous=_board())
        inputs = read_inputs(store, trading_day=DAY, now=FIRED_AT, console_url=self.CONSOLE)
        assert inputs.board_url is None
        assert inputs.board_url_expires == ""
        assert inputs.board_console_url == f"{self.CONSOLE}/decision?pipeline=crucible/board"

    def test_the_presigned_link_and_its_caveat_are_absent(self, tmp_path):
        store = _seed(tmp_path, previous=_board())
        message = run_report(store, trading_day=DAY, now=FIRED_AT, console_url=self.CONSOLE)
        assert (tmp_path / "board" / "index.html").resolve().as_uri() not in message
        assert "presigned GET" not in message

    def test_the_caveat_says_the_address_does_not_expire(self, tmp_path):
        from crucible.morning import BOARD_CONSOLE_CAVEAT

        store = _seed(tmp_path, previous=_board())
        message = run_report(store, trading_day=DAY, now=FIRED_AT, console_url=self.CONSOLE)
        assert BOARD_CONSOLE_CAVEAT in message
        assert "no expiry" in BOARD_CONSOLE_CAVEAT

    def test_a_trailing_slash_on_the_base_url_does_not_double_up(self, tmp_path):
        store = _seed(tmp_path, previous=_board())
        message = run_report(store, trading_day=DAY, now=FIRED_AT, console_url=self.CONSOLE + "/")
        assert f"{self.CONSOLE}/decision?pipeline=crucible/board" in message
        assert f"{self.CONSOLE}//decision" not in message

    def test_the_page_is_not_read_when_a_console_is_configured(self, tmp_path):
        # No presign, no `exists()` on the page: a console-linked report must
        # not fail on — or pay for — a page it does not link.
        store = _seed(tmp_path, previous=_board(), page=False)
        message = run_report(store, trading_day=DAY, now=FIRED_AT, console_url=self.CONSOLE)
        assert BOARD_URL_UNAVAILABLE not in message
        assert f"{self.CONSOLE}/decision" in message

    def test_no_console_means_the_presigned_path_is_unchanged(self, tmp_path):
        store = _seed(tmp_path, previous=_board())
        with_none = run_report(store, trading_day=DAY, now=FIRED_AT, console_url=None)
        default = run_report(store, trading_day=DAY, now=FIRED_AT)
        assert with_none == default
        assert "presigned GET" in default

    def test_the_console_link_rides_the_never_dropped_tier(self, tmp_path):
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
        message = run_report(store, trading_day=DAY, now=FIRED_AT, console_url=self.CONSOLE)
        assert f"{self.CONSOLE}/decision" in message
        assert "(truncated" in message


class TestTheTruncationRule:
    """Telegram takes 4096 characters. What gets dropped is a decision, and
    dropping it silently is the same defect as a green row over no data.
    """

    @staticmethod
    def _huge(tmp_path) -> str:
        # 60 phases, each holding 12 clauses with long names — far past the cap.
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
        return run_report(store, trading_day=DAY, now=FIRED_AT)

    def test_a_message_over_the_cap_is_cut_to_fit(self, tmp_path):
        assert wire_length(self._huge(tmp_path)) <= MESSAGE_MAX_CHARS

    def test_the_cut_is_declared_with_a_count(self, tmp_path):
        message = self._huge(tmp_path)
        assert TRUNCATION_MARKER in message
        assert "line(s) withheld" in message

    def test_every_ladder_line_survives_while_clause_names_are_dropped(self, tmp_path):
        """The issue's rule, and the one that decides what a reader is left
        with: every phase's fraction survives; the names go to the page.

        Dropped from the END of the tier, so the earliest phases keep their
        names — a truncation that cut from the front would leave a reader
        with the tail of a ladder and no idea where it started.
        """
        message = self._huge(tmp_path)
        for i in range(60):
            assert f"  phase{i}  UNMET  0/12" in message, f"phase{i}'s ladder line was dropped"
        holding = [ln for ln in message.splitlines() if ln.startswith("    holding:")]
        assert holding, "dropping every name when only some were needed is over-truncation"
        assert len(holding) < 60, "no clause-name line was dropped, so nothing was truncated"
        assert message.splitlines().index(f"    holding: {holding[0].split(': ')[1]}") < 10

    def test_a_message_under_the_cap_is_untouched(self, tmp_path):
        store = _seed(tmp_path, previous=_board())
        message = run_report(store, trading_day=DAY, now=FIRED_AT)
        assert TRUNCATION_MARKER not in message
        assert "    holding: b_unmet" in message

    def test_the_budget_is_measured_on_the_escaped_body(self, tmp_path):
        """krepis escapes the body AFTER this module hands it over, so a
        message that fits before escaping and not after is tail-trimmed by the
        transport — losing the link, which is the last thing in it."""
        assert wire_length("a_b") == 4
        assert wire_length("[x]") == 5
        assert wire_length("plain") == 5


# ── the POSTed body, not the rendered one, is what has to fit (review F1) ──


class TestTheWireBodyFitsTheCap:
    """`_fit`'s ONE postcondition, asserted on what Telegram actually receives.

    The adversarial review on `alpha-engine-config-I9921` measured the gap:
    the 60-phase fixture rendered a body of wire length **4117** against a
    budget that reserved only `"\\n\\n(truncated — see full board)"` — not the
    ` — N line(s) withheld` it also emits, and not the 35-character
    `[INFO] crucible-v2/report.morning: ` prefix `krepis.alerts.publish`
    prepends. 4117 + 35 = 4152 was POSTed; Telegram returned 400 *message is
    too long*, which is not an entity-parse error, so krepis' plain-text retry
    never fired, `send_message` returned False, `deliver` raised
    `UndeliveredError`, the manifest read `failed` and `alerts.sweep` paged.

    `TestTheTruncationRule` above asserted the ladder survived and that
    `wire_length` was the unit of measure. It never asserted the result was
    under the cap — the one property the whole function exists to deliver.
    This class asserts exactly that, at the cap and over it.
    """

    @staticmethod
    def _wire(message: str) -> int:
        """What krepis escapes and POSTs: the prefix plus this body."""
        return wire_length(TRANSPORT_PREFIX + message)

    def test_the_prefix_matches_the_one_krepis_actually_prepends(self):
        """PINNED against krepis' own formatter, not restated from its source.

        `TRANSPORT_PREFIX` is composed here from `DELIVERY_SEVERITY` and
        `DELIVERY_SOURCE` because krepis exposes no public formatter. That is
        a budget term derived from another package's behaviour, so it is
        pinned by CALLING that behaviour: the day krepis changes its envelope
        this fails, instead of the report silently going over the cap.
        """
        from krepis.alerts import _format_message

        formatted = _format_message("BODY", DELIVERY_SEVERITY, DELIVERY_SOURCE)
        assert formatted == f"{TRANSPORT_PREFIX}BODY"
        assert TRANSPORT_PREFIX.endswith(": ")

    def test_the_reviews_own_fixture_now_fits_the_wire(self, tmp_path):
        """The exact fixture that measured 4117 and was refused."""
        message = TestTheTruncationRule._huge(tmp_path)
        assert self._wire(message) <= MESSAGE_MAX_CHARS
        assert TRUNCATION_MARKER in message

    def test_a_body_at_exactly_the_cap_is_not_truncated(self):
        """The boundary is inclusive: 4096 is a message Telegram takes."""
        from crucible.morning import _KEEP, _fit

        lines = self._lines_of_wire_length(MESSAGE_MAX_CHARS - wire_length(TRANSPORT_PREFIX))
        rendered = _fit(lines)
        assert self._wire(rendered) == MESSAGE_MAX_CHARS
        assert TRUNCATION_MARKER not in rendered
        assert rendered == "\n".join(text for text, _ in lines)
        assert all(tier == _KEEP for _, tier in lines)

    def test_a_body_one_character_over_the_cap_is_truncated_and_fits(self):
        """One character over, and the result is under — including the suffix
        the truncation itself adds, which is what F1 was."""
        from crucible.morning import _fit

        lines = self._lines_of_wire_length(MESSAGE_MAX_CHARS - wire_length(TRANSPORT_PREFIX) + 1)
        rendered = _fit(lines)
        assert TRUNCATION_MARKER in rendered
        assert self._wire(rendered) <= MESSAGE_MAX_CHARS

    def test_the_suffix_the_budget_reserved_is_the_suffix_emitted(self):
        """F1 restated as a property: the reserved string and the emitted
        string are one function, so they cannot drift again."""
        from crucible.morning import _fit, _truncation_suffix

        lines = self._lines_of_wire_length(MESSAGE_MAX_CHARS * 2)
        rendered = _fit(lines)
        dropped = len(lines) - len(rendered.split(TRUNCATION_MARKER)[0].rstrip("\n").splitlines())
        assert rendered.endswith(_truncation_suffix(dropped))
        assert self._wire(rendered) <= MESSAGE_MAX_CHARS

    @pytest.mark.parametrize("phases", [1, 6, 60, 400])
    def test_the_wire_fits_at_every_board_size(self, tmp_path, phases):
        """The postcondition is not a property of one fixture."""
        rows = [
            _row(
                f"phase{i}",
                "phase",
                "UNMET",
                "d",
                [_clause(f"clause_number_{i}_{j}_with_a_long_name", False) for j in range(12)],
            )
            for i in range(phases)
        ]
        store = _seed(tmp_path, board=_board(rows=rows), previous=_board(rows=rows))
        message = run_report(store, trading_day=DAY, now=FIRED_AT)
        assert self._wire(message) <= MESSAGE_MAX_CHARS

    def test_the_console_url_block_alone_fits_the_cap(self, tmp_path):
        """The console variant of the residual below (`alpha-engine-config-
        I9926`): one heading, the console address and its fixed caveat. The
        caveat is the longer of the two, so this is the bound that matters
        once `CRUCIBLE_CONSOLE_URL` is set."""
        from crucible.morning import _URL, _board_lines, wire_length

        inputs = read_inputs(
            _seed(tmp_path, previous=_board()),
            trading_day=DAY,
            now=FIRED_AT,
            console_url="https://console.example.test",
        )
        block = _board_lines(inputs)
        assert block and all(tier == _URL for _, tier in block)
        pinned = "\n".join(["", SECTIONS[-1], *(text for text, _ in block)])
        assert wire_length(pinned) < MESSAGE_MAX_CHARS // 4

    def test_the_url_block_alone_fits_the_cap(self, tmp_path):
        """`_fit`'s stated residual, measured rather than assumed away.

        The `_URL` tier is never dropped, so if it alone exceeded the budget
        nothing could bring the message under the cap. It is one heading, one
        URL (presigned here; the console variant is the test above) and one
        fixed caveat — this asserts that bound holds on a real render rather
        than trusting that it is obviously small.
        """
        from crucible.morning import _URL, _board_lines

        inputs = read_inputs(_seed(tmp_path, previous=_board()), trading_day=DAY, now=FIRED_AT)
        block = _board_lines(inputs)
        assert block and all(tier == _URL for _, tier in block)
        heading = SECTIONS[-1]
        pinned = "\n".join(["", heading, *(text for text, _ in block)])
        assert wire_length(TRANSPORT_PREFIX + pinned) < MESSAGE_MAX_CHARS

    @staticmethod
    def _lines_of_wire_length(target: int) -> list[tuple[str, int]]:
        """`_KEEP` lines of plain text joining to exactly ``target`` wire chars.

        Plain `a`s, so `wire_length` is `len` and the arithmetic under test is
        the fitter's rather than the escaper's.
        """
        from crucible.morning import _KEEP

        lines = [("a" * 100, _KEEP) for _ in range(target // 101)]
        remainder = target - (len(lines) * 101 - 1) - 1
        if remainder > 0:
            lines.append(("a" * remainder, _KEEP))
        built = "\n".join(text for text, _ in lines)
        assert wire_length(built) == target, "the fixture does not hit its own target length"
        return lines


# ── the link is the pointer to everything cut, so it is never cut (F3) ─────


class TestTheLinkOutlivesEveryTruncation:
    """`alpha-engine-config-I9921`: "never the ladder lines", and the review's
    corollary — never the URL either.

    Measured before this fix, on a 200-line `_KEEP` ladder: the FULL BOARD
    heading, the URL and the caveat were the first three lines dropped, purely
    because the inner loop deletes from the END of the list and they are last.
    The message then said "(truncated — see full board)" and carried no board,
    which is the outcome `MESSAGE_MAX_CHARS`'s own docstring names as the
    reason not to let the transport tail-trim.

    The fixture is a board whose phase rows carry NO clause list, so
    `_phase_lines` falls back to the board's unbounded free text at `_KEEP` —
    the path the review named as unbounded, reached without inventing a
    private call.
    """

    @staticmethod
    def _message(tmp_path, *, phases: int) -> str:
        rows = [
            _row(f"phase{i}", "phase", "UNMET", "unbounded board free text " * 12)
            for i in range(phases)
        ]
        store = _seed(tmp_path, board=_board(rows=rows), previous=_board(rows=rows))
        return run_report(store, trading_day=DAY, now=FIRED_AT)

    @pytest.mark.parametrize("phases", [200, 2000])
    def test_the_url_survives_a_keep_tier_body_far_over_budget(self, tmp_path, phases):
        message = self._message(tmp_path, phases=phases)
        assert TRUNCATION_MARKER in message, "the fixture must actually truncate"
        assert "file://" in message, "the link to the full board was dropped"
        assert SECTIONS[-1] in message, "the FULL BOARD heading was dropped"

    def test_the_caveat_stays_with_the_url(self, tmp_path):
        """A presigned link with no expiry beside it is a promise the
        credential cannot keep — the caveat is part of the pointer."""
        message = self._message(tmp_path, phases=200)
        assert "presigned GET, expires" in message

    def test_the_ladder_is_sacrificed_before_the_link(self, tmp_path):
        """The inverse of the defect: at extreme sizes ladder lines go and the
        link stays, never the other way round."""
        message = self._message(tmp_path, phases=2000)
        ladder = [ln for ln in message.splitlines() if ln.startswith("  phase")]
        assert len(ladder) < 2000, "nothing was dropped, so the fixture proves nothing"
        assert "file://" in message

    def test_the_order_of_sacrifice_is_the_one_the_issue_states(self):
        """Clause names, schedule, moved-since, silence, acceptance ids, then
        whatever is left — and the URL is not in the order at all."""
        from crucible.morning import (
            _ACCEPTANCE_IDS,
            _DROP_ORDER,
            _KEEP,
            _MOVED,
            _NAMES,
            _SCHEDULE,
            _SILENCE,
            _URL,
        )

        assert _DROP_ORDER == (_NAMES, _SCHEDULE, _MOVED, _SILENCE, _ACCEPTANCE_IDS, _KEEP)
        assert _URL not in _DROP_ORDER


# ── one artifact, one parse, two surfaces (review F2) ─────────────────────


class TestOneAcceptanceReaderForBothSurfaces:
    """The message and the page it links to read the SAME document with the
    SAME rule, or one of them is lying about the other.

    Measured before this fix, on one document missing only `commit`: the
    message rendered `acceptance count: not on any artifact` while the page it
    pointed at rendered `21 met / 2 unmet / 1 unmeasurable ... at commit
    UNKNOWN`. `board.py`'s own docstring claimed the distinction was "made
    once here so the page and the message cannot disagree". It was made twice.
    """

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
        """One document, rendered by both surfaces from one store."""
        import argparse

        from crucible.keys import acceptance_reading_key
        from crucible.track_c import board_handler

        store = _seed(tmp_path, previous=_board())
        store.put_bytes(acceptance_reading_key(DAY.isoformat()), json.dumps(document).encode())
        board_handler(argparse.Namespace(trading_day=DAY, store=str(tmp_path)))
        message = run_report(store, trading_day=DAY, now=FIRED_AT)
        return message, store.get_bytes(BOARD_HTML_KEY).decode()

    def test_a_document_with_no_commit_is_refused_by_both(self, tmp_path):
        """The exact disagreement the review measured, as one assertion."""
        message, page = self._both(tmp_path, self.NO_COMMIT)
        assert ACCEPTANCE_NOT_ON_ANY_ARTIFACT in message
        assert "answers a different question" in page
        assert "21 met" not in page, "the page quotes a figure the message says does not exist"
        assert "UNKNOWN" not in page.split("<h2>objective")[0]

    def test_a_complete_document_renders_the_same_figures_on_both(self, tmp_path):
        message, page = self._both(tmp_path, self.COMPLETE)
        assert "acceptance count: 21 met / 2 unmet / 1 unmeasurable of 24" in message
        assert "21 met / 2 unmet / 1 unmeasurable</strong> of 24 clauses" in page
        assert SHA in message
        assert SHA in page
        assert ACCEPTANCE_NOT_ON_ANY_ARTIFACT not in message

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
        """One completeness rule, exercised on every way a document can miss
        it — including `True`, which Python calls an `int` and no reader
        should call a count."""
        message, page = self._both(tmp_path, document)
        assert ACCEPTANCE_NOT_ON_ANY_ARTIFACT in message
        assert "answers a different question" in page

    def test_the_shared_parser_is_the_only_completeness_rule(self):
        """Neither renderer restates it. A rule stated in two places has
        already drifted; this one had."""
        from crucible.keys import ACCEPTANCE_REQUIRED_FIELDS, parse_acceptance_reading

        assert ACCEPTANCE_REQUIRED_FIELDS == ("met", "unmet", "unmeasurable", "commit")
        assert parse_acceptance_reading(self.NO_COMMIT) is None
        parsed = parse_acceptance_reading(self.COMPLETE)
        assert parsed is not None
        assert parsed.total == 24
        assert parsed.unmet_clauses is None, "an absent list is 'not filed', never 'there are none'"


# ── the page link: denied is not absent (review F4) ───────────────────────


class TestAReadThatFailedIsNotAPageThatIsAbsent:
    """`_board_url` swallowed only `KeyError`, but `S3Store.presigned_url`
    reaches absence through `exists()` → `head_object`, which re-raises every
    non-404 `ClientError`. `alpha-engine-config-I9896` measured the shape live:
    S3 answers **403 for a missing key** when the caller also lacks
    `s3:ListBucket` on the prefix — and `board/index.html` is written only
    under `if may_move:`, so its absence is a live possibility. The whole
    report would have died over a missing optional link.
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

    def _message(self, tmp_path, code: str) -> str:
        _seed(tmp_path, previous=_board())
        store = self._Store(tmp_path, code=code)
        return run_report(store, trading_day=DAY, now=FIRED_AT)

    def test_an_access_denied_page_read_does_not_kill_the_report(self, tmp_path):
        message = self._message(tmp_path, "AccessDenied")
        assert message.startswith("CRUCIBLE V2 —")

    def test_an_access_denied_page_read_is_named_as_an_access_failure(self, tmp_path):
        message = self._message(tmp_path, "AccessDenied")
        assert "AccessDenied" in message
        assert "access failure" in message
        assert BOARD_URL_UNAVAILABLE not in message, (
            "a denied read reported as 'the board render published none' blames the wrong job"
        )

    def test_a_not_found_page_read_is_still_absent_not_denied(self, tmp_path):
        """A `ClientError` carrying a 404 code is the ABSENT case, matched for
        the same reason `_read_json` matches it: a mock or a future backend
        that raises it directly must degrade to the honest answer."""
        message = self._message(tmp_path, "NoSuchKey")
        assert BOARD_URL_UNAVAILABLE in message
        assert "access failure" not in message

    def test_an_absent_page_is_still_the_unavailable_line(self, tmp_path):
        store = _seed(tmp_path, previous=_board(), page=False)
        assert BOARD_URL_UNAVAILABLE in run_report(store, trading_day=DAY, now=FIRED_AT)

    def test_a_configuration_failure_still_raises(self, tmp_path):
        """Nothing widened. A `ValueError` from an out-of-range lifetime is a
        defect in this job's own configuration and must not be swallowed into
        a slightly shorter message."""

        class _Broken(LocalStore):
            def presigned_url(self, key: str, expires_s: int) -> str:
                raise ValueError("expires_s out of range")

        _seed(tmp_path, previous=_board())
        with pytest.raises(ValueError):
            run_report(_Broken(tmp_path), trading_day=DAY, now=FIRED_AT)


def test_read_inputs_returns_the_declared_shape(tmp_path):
    store = _seed(tmp_path, previous=_board())
    inputs = read_inputs(store, trading_day=DAY)
    assert isinstance(inputs, MorningInputs)
    assert inputs.previous_day == PREVIOUS
    assert inputs.board_code_sha == SHA
    assert render_message(inputs, now=FIRED_AT).startswith("CRUCIBLE V2 —")
