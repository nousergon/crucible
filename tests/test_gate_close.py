"""`crucible gate.close` — the FILING half of the phase-exit loop, on a cadence.

Normative source: `alpha-engine-config-I10095`. `crucible-PR121` made a phase's
exit a durable record and nothing ran `crucible gate` on a schedule, so the
record existed only when somebody happened to take a reading. The board
DETECTS that gap and deliberately cannot close it — `crucible-v2-github-board`
is read-only over everything it grades — so the filing is its own daily job
under its own writer identity.

Every guard below is shown FIRING, not merely accepting valid input:

* a phase that has just turned MET gets exactly one record, and a SECOND run
  over the same state files nothing and says so;
* a replay files nothing and posts nothing, however MET the reading is;
* an UNMET gate and an UNMEASURABLE gate file nothing, and are told apart;
* `--dry-run` reaches neither the store nor the tracker;
* the workflow's identity step REFUSES the board's role — driven under real
  bash with a stubbed `aws`, because a `case` branch nobody has made fire is a
  branch nobody knows works.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import subprocess
import urllib.request

import pytest
import yaml

import crucible.tracker as tracker_module
from crucible.cli import HANDLERS, JOBS
from crucible.components import load_registry
from crucible.documents import UnreadableDocumentError, read_store_document
from crucible.gate import PHASES, Clause, GateResult, Phase
from crucible.keys import closing_record_key, gate_key, manifest_key
from crucible.store import LocalStore
from crucible.track_f import CLOSE_OUTCOMES, GATE_CLOSE_JOB, gate_close_handler

DAY = dt.date(2026, 8, 28)
COMMIT = "0123456789abcdef0123456789abcdef01234567"
BOARD_ROLE = "crucible-v2-github-board"
GATE_CLOSE_ROLE = "crucible-v2-github-gate-close"

WORKFLOW = pathlib.Path(__file__).resolve().parents[1] / ".github" / "workflows" / "gate-close.yml"


# ── Fixtures: real objects, never mocks of the thing under test ────────────


def _clause(met: bool, *, unmeasurable: bool = False) -> Clause:
    return Clause(
        "arc_runs_ok",
        "the weekly arc ran and its manifest reads ok",
        met and not unmeasurable,
        "measured" if met else "no arc manifest for this week",
        (),
        unmeasurable=unmeasurable,
    )


def _reading(gate: str, *, met: bool, unmeasurable: bool = False) -> GateResult:
    return GateResult(
        gate=gate,
        trading_day=DAY,
        window=[DAY],
        clauses=[_clause(met, unmeasurable=unmeasurable)],
    )


class _Opener:
    """A GitHub API stand-in that records every request it was handed.

    The requests are asserted against rather than merely counted: this job's
    central claim is about which tracker writes it can make at all, and a stub
    that only returned canned bodies would leave exactly that untested.
    """

    def __init__(self, *responses: tuple[int, bytes]) -> None:
        self.responses = list(responses)
        self.seen: list[urllib.request.Request] = []

    def __call__(self, request: urllib.request.Request) -> tuple[int, bytes]:
        self.seen.append(request)
        if not self.responses:
            raise AssertionError(f"unexpected extra request to {request.full_url}")
        return self.responses.pop(0)


def _run(
    tmp_path,
    monkeypatch,
    readings: dict[str, GateResult],
    *,
    run_mode: str = "live",
    dry_run: bool = False,
    store: LocalStore | None = None,
    opener: _Opener | None = None,
    granted: bool = True,
) -> dict[str, object]:
    """Drive the real handler over a real `LocalStore`.

    `crucible.track_f.evaluate` is the ONE thing replaced: building six live
    gate readings out of real artifacts would test `crucible.gate`, which has
    its own suite, and would say nothing about what this job does with them.
    Everything else here — the store, the manifest, the compare-and-swap, the
    tracker adapter's request construction — is the real object.
    """
    store = store if store is not None else LocalStore(tmp_path / "store")
    opener = opener if opener is not None else _Opener()
    monkeypatch.setattr(tracker_module, "_default_opener", opener)
    # `reading_commit` prefers $GITHUB_SHA, which Actions sets on every run --
    # so a test that only sets CRUCIBLE_COMMIT reads green on a laptop and
    # fails in CI against the real checkout sha (measured, run 34057597154).
    # The override is removed, not worked around.
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    monkeypatch.setenv("CRUCIBLE_COMMIT", COMMIT)
    if granted:
        monkeypatch.setenv(tracker_module.TRACKER_TOKEN_VAR, "a-token")
    else:
        monkeypatch.delenv(tracker_module.TRACKER_TOKEN_VAR, raising=False)

    def _evaluate(_store, *, gate: str, trading_day, weeks=None) -> GateResult:
        assert trading_day == DAY
        return readings[gate]

    monkeypatch.setattr("crucible.track_f.evaluate", _evaluate)
    args = argparse.Namespace(
        trading_day=DAY,
        store=str(store.root) if hasattr(store, "root") else None,
        dry_run=dry_run,
        run_mode=run_mode,
    )
    code = gate_close_handler(args)
    return {"code": code, "store": store, "opener": opener}


def _all(met_gate: str | None = None, *, unmeasurable_gate: str | None = None) -> dict:
    """Every registered gate UNMET, except the one named."""
    out = {}
    for phase in PHASES:
        assert phase.gate is not None, phase.id
        out[phase.gate] = _reading(
            phase.gate,
            met=phase.gate == met_gate,
            unmeasurable=phase.gate == unmeasurable_gate,
        )
    return out


def _manifest(store: LocalStore) -> dict:
    return json.loads(store.get_bytes(manifest_key(GATE_CLOSE_JOB, DAY.isoformat())))


def _outcomes(store: LocalStore) -> dict[str, str]:
    """Per-phase outcome, read back off the manifest the job actually wrote."""
    metrics = {m["name"]: m for m in _manifest(store)["metrics"]}
    out = {}
    for phase in PHASES:
        reason = metrics[f"closing_record_state_{phase.id}"]["status_reason"]
        out[phase.id] = reason.split(": ", 1)[1].split(" — ", 1)[0]
    return out


# ── The record is filed, once, from a MET live reading ────────────────────


class TestAPhaseThatHasJustTurnedMetGetsExactlyOneRecord:
    def test_it_files_the_record_and_posts_the_reading(self, tmp_path, monkeypatch) -> None:
        opener = _Opener((200, b"[]"), (201, json.dumps({"html_url": "https://x/1"}).encode()))
        result = _run(tmp_path, monkeypatch, _all("phase2"), opener=opener)
        store = result["store"]
        assert result["code"] == 0
        document = read_store_document(store, closing_record_key("phase2")).document
        assert document is not None
        assert document["phase"] == "phase2"
        assert document["gate_state"] == "MET"
        assert document["commit"] == COMMIT
        # Posted BEFORE it was filed, and posted exactly once.
        assert [r.get_method() for r in opener.seen] == ["GET", "POST"]

    def test_it_files_nothing_for_the_other_five_phases(self, tmp_path, monkeypatch) -> None:
        opener = _Opener((200, b"[]"), (201, b'{"html_url": "u"}'))
        result = _run(tmp_path, monkeypatch, _all("phase2"), opener=opener)
        store = result["store"]
        for phase in PHASES:
            present = not read_store_document(store, closing_record_key(phase.id)).absent
            assert present is (phase.id == "phase2"), phase.id

    def test_the_record_and_the_manifest_are_the_only_things_it_wrote(
        self, tmp_path, monkeypatch
    ) -> None:
        """It writes the CONSEQUENCE of a reading and never a reading. A job
        that also published `gates/{gate}/{day}/gate.json` would be authoring
        the artifact its own clause is graded from — and the IAM grant
        (`GateCloseRole`) is scoped to `gates/*/closing.json` for that reason,
        so this assertion is the code half of a property the role enforces."""
        opener = _Opener((200, b"[]"), (201, b'{"html_url": "u"}'))
        result = _run(tmp_path, monkeypatch, _all("phase2"), opener=opener)
        store = result["store"]
        written = sorted(store.list_keys(""))
        assert written == sorted(
            [closing_record_key("phase2"), manifest_key(GATE_CLOSE_JOB, DAY.isoformat())]
        ), written
        assert not store.exists(gate_key("phase2", DAY.isoformat()))
        assert not store.exists("gates/ladder.json")

    def test_a_second_run_files_nothing_and_says_so(self, tmp_path, monkeypatch) -> None:
        """Compare-and-swap once. The second run is the one that proves a phase
        exits ONCE — without it, `filed` on every run would look identical."""
        store = LocalStore(tmp_path / "store")
        first = _run(
            tmp_path,
            monkeypatch,
            _all("phase2"),
            store=store,
            opener=_Opener((200, b"[]"), (201, b'{"html_url": "u"}')),
        )
        assert _outcomes(first["store"])["phase2"] == "filed"
        before = store.get_bytes(closing_record_key("phase2"))
        # No tracker responses queued at all: a second POST would raise inside
        # `_Opener`, so "posts nothing" is proven rather than asserted about a
        # counter. The GET of the issue's existing comments is not made either
        # — the record is already present, so `file_closing_record` returns
        # before `post_closing_comment` is reached.
        second = _run(tmp_path, monkeypatch, _all("phase2"), store=store, opener=_Opener())
        assert _outcomes(second["store"])["phase2"] == "already_filed"
        assert store.get_bytes(closing_record_key("phase2")) == before
        assert second["opener"].seen == []

    def test_the_filed_record_enters_the_manifest_lineage(self, tmp_path, monkeypatch) -> None:
        """`crucible explain` must be able to name the run that filed it."""
        result = _run(
            tmp_path,
            monkeypatch,
            _all("phase2"),
            opener=_Opener((200, b"[]"), (201, b'{"html_url": "u"}')),
        )
        outputs = [o["key"] for o in _manifest(result["store"])["outputs"]]
        assert closing_record_key("phase2") in outputs


# ── What it refuses to file ───────────────────────────────────────────────


class TestItFilesNothingOnAReplay:
    def test_a_met_reading_on_a_replay_files_nothing_and_posts_nothing(
        self, tmp_path, monkeypatch
    ) -> None:
        """A replay of a historical day may legitimately read MET and does not
        exit a phase (plan §6 row 2). The `_Opener` is empty, so any tracker
        request at all raises rather than being counted."""
        result = _run(tmp_path, monkeypatch, _all("phase2"), run_mode="replay")
        store = result["store"]
        assert read_store_document(store, closing_record_key("phase2")).absent
        assert result["opener"].seen == []
        assert _outcomes(store)["phase2"] == "replay"

    def test_the_replay_is_counted_as_a_phase_met_without_a_record(
        self, tmp_path, monkeypatch
    ) -> None:
        """Principle 7: the figure that says this job is working. Zero on a
        live run by construction; the replay is the case that makes it
        non-zero, so a metric that read zero unconditionally would be
        indistinguishable from a working one."""
        result = _run(tmp_path, monkeypatch, _all("phase2"), run_mode="replay")
        metric = next(
            m
            for m in _manifest(result["store"])["metrics"]
            if m["name"] == "phases_met_without_a_record"
        )
        assert metric["value"] == 1.0
        assert metric["status"] == "FAIL"
        assert "phase2" in metric["status_reason"]


class TestItFilesNothingOnAGateThatIsNotMet:
    def test_every_phase_unmet_files_nothing_at_all(self, tmp_path, monkeypatch) -> None:
        result = _run(tmp_path, monkeypatch, _all())
        store = result["store"]
        for phase in PHASES:
            assert read_store_document(store, closing_record_key(phase.id)).absent, phase.id
        assert result["opener"].seen == []
        assert set(_outcomes(store).values()) == {"not_met"}

    def test_the_run_still_succeeds_and_reports_zero_filed(self, tmp_path, monkeypatch) -> None:
        """`gate.close` exits 0 whenever the MEASUREMENT succeeded. A non-zero
        exit on "no phase exited today" would be a daily failure alert on a
        working producer, which is how a channel gets muted."""
        result = _run(tmp_path, monkeypatch, _all())
        assert result["code"] == 0
        manifest = _manifest(result["store"])
        assert manifest["status"] == "ok"
        assert manifest["rows_out"] == 0
        assert manifest["rows_in"] == len(PHASES)
        filed = next(m for m in manifest["metrics"] if m["name"] == "closing_records_filed")
        assert filed["value"] == 0.0

    def test_an_unmeasurable_gate_is_told_apart_from_an_unmet_one(
        self, tmp_path, monkeypatch
    ) -> None:
        """ "we could not check" and "we checked and it fell short" call for
        opposite actions, and collapsing them is `alpha-engine-config-I9869`
        round 3 one layer along."""
        result = _run(tmp_path, monkeypatch, _all(unmeasurable_gate="phase2"))
        outcomes = _outcomes(result["store"])
        assert outcomes["phase2"] == "unmeasurable"
        assert outcomes["phase1"] == "not_met"
        assert read_store_document(result["store"], closing_record_key("phase2")).absent


class TestDryRunReachesNeitherTheStoreNorTheTracker:
    def test_it_neither_writes_nor_comments_on_a_met_reading(self, tmp_path, monkeypatch) -> None:
        """The store's read-only guard would refuse the write, but the tracker
        comment is posted FIRST and travels over a different wire — so the
        guard alone would have let a run asked to change nothing leave a real
        comment on a real issue."""
        store = LocalStore(tmp_path / "store")
        result = _run(tmp_path, monkeypatch, _all("phase2"), dry_run=True, store=store)
        assert result["code"] == 0
        assert result["opener"].seen == []
        assert sorted(store.list_keys("")) == []


class TestAPresentButUnreadableRecordIsNeverOverwritten:
    def test_it_raises_rather_than_replacing_the_evidence(self, tmp_path, monkeypatch) -> None:
        """ "the record is corrupt" and "there is no record" call for opposite
        actions. The job fails loud — `run_job` writes a `failed` manifest
        naming it — rather than quietly filing a fresh reading over the only
        evidence of what was written when the phase exited."""
        store = LocalStore(tmp_path / "store")
        store.put_bytes(closing_record_key("phase2"), b"{ truncated")
        with pytest.raises(UnreadableDocumentError, match="will not overwrite it"):
            _run(tmp_path, monkeypatch, _all("phase2"), store=store)
        assert store.get_bytes(closing_record_key("phase2")) == b"{ truncated"
        assert _manifest(store)["status"] == "failed"


class TestAPhaseWithNoRegisteredGateIsStatedRatherThanSkipped:
    def test_it_files_nothing_and_names_the_phase(self, tmp_path, monkeypatch) -> None:
        """`Phase.gate` is `None`-able so a SIXTH phase added to the plan before
        its clause list is written renders UNMEASURED rather than failing an
        import. Every registered phase carries a gate today, and a structural
        test in `tests/test_gate.py` keeps it that way — so this branch is
        driven against a registry that has one, which is the only way to know
        it does something other than raise `KeyError` on `readings[None]`."""
        gateless = Phase("phase6", 6, "A phase with no gate yet", 9762, None)
        monkeypatch.setattr("crucible.track_f.PHASES", (*PHASES, gateless))
        result = _run(tmp_path, monkeypatch, _all())
        metrics = {m["name"]: m for m in _manifest(result["store"])["metrics"]}
        reason = metrics["closing_record_state_phase6"]["status_reason"]
        assert "no_gate_registered" in reason
        assert metrics["closing_record_state_phase6"]["value"] == 0.0
        assert result["opener"].seen == []


# ── The registry, the CLI table and the schema ────────────────────────────


class TestTheJobIsDeclaredEverywhereAJobMustBe:
    def test_it_is_a_scheduled_cli_job_with_a_real_handler(self) -> None:
        assert JOBS[GATE_CLOSE_JOB].scheduled is True
        assert HANDLERS[GATE_CLOSE_JOB] is gate_close_handler

    def test_its_registry_row_declares_the_github_actions_cron_that_starts_it(self) -> None:
        row = load_registry()[GATE_CLOSE_JOB]
        assert row.dispatch == "github-actions"
        assert row.dispatch_workflow == WORKFLOW.name
        # A row naming a workflow that does not exist is a declared cadence
        # with nothing triggering it — the alpha-engine-config-I9878 defect,
        # which `board` carried from 2026-09-02T21:47Z until it was found.
        assert WORKFLOW.is_file(), (
            f"{WORKFLOW} does not exist, so this row names a starter that is not there"
        )

    def test_the_declared_deadline_clears_this_accounts_measured_cron_delay(self) -> None:
        """23:00 UTC + this account's worst MEASURED GitHub Actions delivery
        delay (+304 min, alpha-engine-config-I9960/-I9966) is 04:04 UTC. A
        deadline earlier than that pages on a run that was merely late, and a
        false page most days trains the reader to ignore the true one."""
        deadline = load_registry()[GATE_CLOSE_JOB].deadline
        assert deadline is not None
        due = deadline.due_at(DAY)
        assert due == dt.datetime(2026, 8, 29, 6, 0, tzinfo=dt.UTC)
        assert due > dt.datetime(2026, 8, 29, 4, 4, tzinfo=dt.UTC)

    def test_the_outcome_signal_names_metrics_the_job_actually_emits(
        self, tmp_path, monkeypatch
    ) -> None:
        """A row naming a metric nothing writes is a component that reads as
        graded while being unobserved — `no data` rendered as green."""
        declared = load_registry()[GATE_CLOSE_JOB].signals["outcome"]
        result = _run(tmp_path, monkeypatch, _all())
        emitted = {m["name"] for m in _manifest(result["store"])["metrics"]}
        for name in ("closing_records_filed", "phases_met_without_a_record"):
            assert name in declared, name
            assert name in emitted, name

    def test_every_phase_gets_exactly_one_outcome_from_the_closed_vocabulary(
        self, tmp_path, monkeypatch
    ) -> None:
        """ "this phase was not touched" is a STATED outcome carrying its
        reason, never a line nobody wrote."""
        result = _run(tmp_path, monkeypatch, _all("phase2"), run_mode="replay")
        outcomes = _outcomes(result["store"])
        assert set(outcomes) == {p.id for p in PHASES}
        assert set(outcomes.values()) <= set(CLOSE_OUTCOMES)


# ── The workflow, and the identity it must NOT be ─────────────────────────


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _identity_step() -> str:
    steps = _workflow()["jobs"]["gate-close"]["steps"]
    matches = [s["run"] for s in steps if s.get("name", "").startswith("Confirm the identity")]
    assert len(matches) == 1, "expected exactly one identity-confirmation step"
    return matches[0]


class TestTheWorkflowRunsAsAWriterThatIsNotTheBoard:
    def test_the_declared_role_is_not_the_boards(self) -> None:
        """The board renders the grading surface and is read-only over
        everything it grades. `alpha-engine-config-I10095`'s closes-when says
        it in as many words: the identity that files the record is not
        `crucible-v2-github-board`."""
        arn = _workflow()["env"]["GATE_CLOSE_ROLE_ARN"]
        assert arn.endswith(f"role/{GATE_CLOSE_ROLE}")
        assert BOARD_ROLE not in arn
        steps = _workflow()["jobs"]["gate-close"]["steps"]
        credentials = [s for s in steps if "configure-aws-credentials" in s.get("uses", "")]
        assert len(credentials) == 1
        assert credentials[0]["with"]["role-to-assume"] == "${{ env.GATE_CLOSE_ROLE_ARN }}"

    def test_it_declares_a_cron_and_holds_no_write_permission_beyond_oidc(self) -> None:
        workflow = _workflow()
        triggers = workflow.get(True, workflow.get("on"))
        assert [entry["cron"] for entry in triggers["schedule"]] == ["0 23 * * *"]
        assert workflow["jobs"]["gate-close"]["permissions"] == {
            "contents": "read",
            "id-token": "write",
        }

    def test_the_run_step_passes_run_mode_live(self) -> None:
        """`file_closing_record` files nothing on a replay, so a scheduled run
        that did not say `live` would file nothing, every day, silently."""
        steps = _workflow()["jobs"]["gate-close"]["steps"]
        run = next(s["run"] for s in steps if f"crucible {GATE_CLOSE_JOB}" in s.get("run", ""))
        assert "--run-mode live" in run


def _identity_result(tmp_path: pathlib.Path, assumed: str) -> subprocess.CompletedProcess:
    """Execute the workflow's identity step under real bash with a stubbed
    `aws`, so the `case` branches are shown FIRING rather than merely read."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    shim = bin_dir / "aws"
    shim.write_text(f'#!/usr/bin/env bash\nprintf "%s\\n" "{assumed}"\n', encoding="utf-8")
    shim.chmod(0o755)
    return subprocess.run(
        ["bash", "-c", _identity_step()],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={
            "PATH": f"{bin_dir}{os.pathsep}/usr/bin:/bin",
            "GATE_CLOSE_ROLE_ARN": f"arn:aws:iam::111111111111:role/{GATE_CLOSE_ROLE}",
        },
    )


class TestTheIdentityGuardActuallyFires:
    def test_it_refuses_the_board_role_by_name(self, tmp_path: pathlib.Path) -> None:
        """The negative demonstration. A workflow "simplified" onto the board's
        ARN would run right up until the board's read-only grant refused the
        write — and the board would then be an identity that writes into the
        prefix it grades."""
        result = _identity_result(
            tmp_path, f"arn:aws:sts::111111111111:assumed-role/{BOARD_ROLE}/session"
        )
        assert result.returncode != 0, result.stdout
        assert "BOARD role" in result.stdout

    def test_it_refuses_any_other_identity(self, tmp_path: pathlib.Path) -> None:
        result = _identity_result(
            tmp_path, "arn:aws:sts::111111111111:assumed-role/crucible-v2-github-deploy/session"
        )
        assert result.returncode != 0, result.stdout
        assert "cloudformation deploy" in result.stdout

    def test_it_accepts_the_gate_close_role(self, tmp_path: pathlib.Path) -> None:
        result = _identity_result(
            tmp_path, f"arn:aws:sts::111111111111:assumed-role/{GATE_CLOSE_ROLE}/session"
        )
        assert result.returncode == 0, f"{result.stdout!r} {result.stderr!r}"
        assert GATE_CLOSE_ROLE in result.stdout
