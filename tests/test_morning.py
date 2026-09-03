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

from crucible.cli import HANDLERS, JOBS
from crucible.components import load_registry
from crucible.keys import BOARD_CURRENT_KEY, board_key, manifest_key, morning_report_key
from crucible.morning import (
    ACCEPTANCE_NOT_ON_ANY_ARTIFACT,
    DELIVERY_TZ,
    MORNING_JOB,
    NO_OPERATOR_ACTION,
    MorningInputs,
    UndeliveredError,
    deliver,
    morning_handler,
    read_inputs,
    render_message,
    run_report,
    store_uri,
)
from crucible.store import LocalStore

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "morning-report.yml"

#: A Wednesday session, and the Tuesday before it. Both real NYSE sessions.
DAY = dt.date(2026, 9, 2)
PREVIOUS = dt.date(2026, 9, 1)
#: 13:00 UTC on the calendar day after DAY — the cron's own firing instant.
FIRED_AT = dt.datetime(2026, 9, 3, 13, 0, tzinfo=dt.UTC)
GENERATED = "2026-09-02T21:35:04Z"
SHA = "8fc58b6c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a"


def _row(row_id: str, source: str, state: str, detail: str) -> dict[str, Any]:
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
            _row("phase0", "phase", "UNMET", "1 of 2 clauses — old_weekly_within_cadence"),
            _row("phase1", "phase", "OUT_OF_ORDER", "1 of 5 clauses — phase0's gate is not met"),
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
) -> LocalStore:
    store = LocalStore(tmp_path)
    store.put_bytes(BOARD_CURRENT_KEY, json.dumps(board or _board()).encode())
    if previous is not None:
        store.put_bytes(board_key(PREVIOUS.isoformat()), json.dumps(previous).encode())
    if board_run == "default":
        board_run = {"status": "ok", "code_sha": SHA, "reason": ""}
    if board_run is not None:
        store.put_bytes(manifest_key("board", DAY.isoformat()), json.dumps(board_run).encode())
    return store


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
                "crucible v2 — board for trading day 2026-09-02",
                f"store: {tmp_path}/board/current.json",
                f"generated: {GENERATED}  commit: {SHA}",
                "delivered: 2026-09-03 06:00 PDT",
                "",
                "phase gates",
                "  phase0  UNMET  1 of 2 clauses — old_weekly_within_cadence",
                "  phase1  OUT_OF_ORDER  1 of 5 clauses — phase0's gate is not met",
                "",
                "moved since 2026-09-01",
                "  obj:cost: PLANNED -> UNMEASURED",
                "",
                ACCEPTANCE_NOT_ON_ANY_ARTIFACT,
                "silence: 1 UNMEASURED, 0 UNMEASURABLE of 4 rows",
                NO_OPERATOR_ACTION,
            ]
        )

    def test_a_stale_board_is_the_first_line_and_not_a_footnote(self, tmp_path):
        store = _seed(tmp_path, previous=_board())
        # Three calendar days after the render.
        late = dt.datetime(2026, 9, 5, 13, 0, tzinfo=dt.UTC)
        message = run_report(store, trading_day=DAY, now=late)
        assert message.splitlines()[0].startswith("STALE BOARD:")
        assert GENERATED in message.splitlines()[0]

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
                    "measured_at": "2026-09-02T22:00:00Z",
                }
            ).encode(),
        )
        message = run_report(store, trading_day=DAY, now=FIRED_AT)
        assert f"acceptance count: 21 met / 3 unmet / 0 unmeasurable (commit {SHA})" in message
        assert ACCEPTANCE_NOT_ON_ANY_ARTIFACT not in message

    def test_a_corrupt_acceptance_reading_is_absent_rather_than_guessed(self, tmp_path):
        """A malformed reading is not a number. Rendering a partial parse of
        the only progress figure is exactly the fabrication the literal
        exists to avoid."""
        from crucible.keys import acceptance_reading_key

        store = _seed(tmp_path, previous=_board())
        store.put_bytes(acceptance_reading_key(DAY.isoformat()), b"{not json")
        assert ACCEPTANCE_NOT_ON_ANY_ARTIFACT in run_report(store, trading_day=DAY, now=FIRED_AT)

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
            if ln.startswith("silence:")
        ]
        assert line == ["silence: 1 UNMEASURED, 1 UNMEASURABLE of 3 rows"]

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
        assert call["silent"] is True
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
        assert message.startswith("crucible v2 — board for trading day")
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

    def test_a_dry_run_delivers_nothing_and_claims_nothing(self, tmp_path, monkeypatch):
        store = _seed(tmp_path, previous=_board())

        def refuse(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("--dry-run must not reach the transport")

        monkeypatch.setattr("crucible.morning._krepis_publish", refuse)
        assert morning_handler(_args(tmp_path, dry_run=True)) == 0
        keys = [k for k in store.list_keys(f"runs/{MORNING_JOB}/") if k.endswith("run.json")]
        manifest = json.loads(store.get_bytes(keys[0]))
        assert manifest["status"] == "ok"
        assert manifest["outputs"] == []


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


def test_read_inputs_returns_the_declared_shape(tmp_path):
    store = _seed(tmp_path, previous=_board())
    inputs = read_inputs(store, trading_day=DAY)
    assert isinstance(inputs, MorningInputs)
    assert inputs.previous_day == PREVIOUS
    assert inputs.board_code_sha == SHA
    assert render_message(inputs, now=FIRED_AT).startswith("crucible v2 —")
