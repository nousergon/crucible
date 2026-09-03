"""Producer/consumer contract for the run manifest's live-or-replay field.

Normative source: plan §6 row 2 and §6.1; alpha-engine-config-I9918.

Three parties have to agree, and each half is asserted against the OTHER
half's real artifact rather than against a restatement of it:

* the CONTRACT — `run_manifest.v2` requires `run_mode`, with a closed
  vocabulary and no default;
* the PRODUCER — `crucible.runner.run_job` writes the mode the INVOCATION
  declared, and refuses to run when no invocation declared one;
* the CONSUMER — `crucible.gate._clause_live_saturdays_first_attempt_ok`
  counts a live Saturday and refuses to count a replay.

Every assertion below is one the system must REFUSE something, or one where
the two halves are compared to each other. A test that only showed a valid
manifest validating would not have caught the defect this field exists for:
the old contract accepted every manifest ever written and could not express
the fact phase 2's gate needs.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest
from jsonschema import Draft202012Validator

import crucible.gate as gate_module
from crucible.keys import manifest_key
from crucible.manifest import (
    PREDECESSOR_SCHEMA_VERSION,
    RUN_MANIFEST_SCHEMA_VERSION,
    ManifestValidationError,
    load_schema,
    load_schema_for,
    validate,
)
from crucible.runmode import (
    RUN_MODE_ENV,
    RUN_MODE_LIVE,
    RUN_MODE_REPLAY,
    RUN_MODES,
    RunModeError,
    resolve_run_mode,
)
from crucible.runner import run_job
from crucible.store import LocalStore

#: A real NYSE session, fixed. Never `today` arithmetic: a test whose subject
#: moves with the clock stops testing the same thing.
FRIDAY = dt.date(2026, 8, 28)


# ---------------------------------------------------------------------------
# the contract
# ---------------------------------------------------------------------------


class TestTheSchemaCanExpressLiveOrReplayAndRefusesSilence:
    def test_run_mode_is_required(self) -> None:
        assert "run_mode" in load_schema()["required"]

    def test_run_mode_has_no_default(self) -> None:
        """The defect the field exists for, one level up: a default is what
        makes a replay indistinguishable from a live run the first time a
        producer forgets to set it."""
        assert "default" not in load_schema()["properties"]["run_mode"]

    def test_the_vocabulary_is_closed_and_is_the_producers_own(self) -> None:
        """Compared against `crucible.runmode`, not restated: two spellings of
        one vocabulary is how they drift apart."""
        assert tuple(load_schema()["properties"]["run_mode"]["enum"]) == RUN_MODES

    def test_a_manifest_with_no_run_mode_is_refused(self) -> None:
        document = _manifest()
        del document["run_mode"]
        with pytest.raises(ManifestValidationError, match="run_mode"):
            validate(document)

    @pytest.mark.parametrize("mode", ["", "LIVE", "saturday", "backfill", "true"])
    def test_a_mode_outside_the_vocabulary_is_refused(self, mode: str) -> None:
        with pytest.raises(ManifestValidationError):
            validate(_manifest(run_mode=mode))

    def test_the_field_is_not_derivable_from_either_date(self) -> None:
        """The gotcha, asserted rather than trusted: the SAME trading day and
        the SAME calendar date carry both modes, so nothing downstream can
        recover the mode from a date."""
        live = _manifest(run_mode=RUN_MODE_LIVE)
        replay = _manifest(run_mode=RUN_MODE_REPLAY)
        validate(live)
        validate(replay)
        assert live["trading_day"] == replay["trading_day"]
        assert live["calendar_date"] == replay["calendar_date"]


class TestManifestsWrittenBeforeTheFieldStayReadable:
    """The grandfathering half of the design note.

    A required field added to `run_manifest.v1` would have retroactively
    invalidated every object already in the store, and `read_manifest`
    validates on READ — so the board, the morning report and the alert sweep
    would all have started refusing documents that were correct when they were
    written. Each document is instead checked against the version it declares.
    Nothing is backfilled: inventing a `run_mode` for a run nobody observed is
    the false liveness claim the field exists to prevent.
    """

    def test_a_v1_document_still_validates(self) -> None:
        document = _manifest()
        del document["run_mode"]
        document["schema_version"] = PREDECESSOR_SCHEMA_VERSION
        validate(document)

    def test_the_frozen_predecessor_never_gained_the_field(self) -> None:
        assert "run_mode" not in load_schema_for(PREDECESSOR_SCHEMA_VERSION)["properties"]

    def test_a_v1_document_is_still_held_to_v1s_own_rules(self) -> None:
        """Grandfathered by VERSION, not waved through: v1 is checked
        strictly, so an old document that was always invalid stays invalid."""
        document = _manifest()
        del document["run_mode"]
        document["schema_version"] = PREDECESSOR_SCHEMA_VERSION
        document["status"] = "degraded"
        with pytest.raises(ManifestValidationError):
            validate(document)

    def test_an_unknown_version_is_refused_not_checked_against_the_newest(self) -> None:
        with pytest.raises(ManifestValidationError, match="run_manifest.v9"):
            validate(_manifest(schema_version="run_manifest.v9"))

    def test_a_document_declaring_no_version_is_refused(self) -> None:
        document = _manifest()
        del document["schema_version"]
        with pytest.raises(ManifestValidationError, match="schema_version"):
            validate(document)


# ---------------------------------------------------------------------------
# the producer
# ---------------------------------------------------------------------------


class TestTheProducerWritesWhatTheInvocationDeclared:
    @pytest.mark.parametrize("mode", RUN_MODES)
    def test_run_job_records_the_declared_mode(self, store: LocalStore, mode: str) -> None:
        run_job("smoke", lambda ctx: None, store=store, trading_day=FRIDAY, run_mode=mode)
        assert _written(store, "smoke")["run_mode"] == mode

    def test_the_same_day_produces_both_modes(self, store: LocalStore) -> None:
        """The gotcha again, at the producer: one trading day, two
        invocations, two different recorded modes. Nothing about the date
        moved, so nothing about the date can be what decided."""
        run_job("smoke", lambda ctx: None, store=store, trading_day=FRIDAY, run_mode=RUN_MODE_LIVE)
        first = _written(store, "smoke")["run_mode"]
        run_job(
            "smoke", lambda ctx: None, store=store, trading_day=FRIDAY, run_mode=RUN_MODE_REPLAY
        )
        second = _written(store, "smoke")["run_mode"]
        assert (first, second) == (RUN_MODE_LIVE, RUN_MODE_REPLAY)

    def test_a_failed_run_still_records_its_mode(self, store: LocalStore) -> None:
        """The manifest guarantee holds on the failure path, and so does this
        field — a replay that failed must not read as a failed live Saturday.
        """

        def boom(ctx: object) -> None:
            raise ValueError("deliberate")

        with pytest.raises(ValueError):
            run_job(
                "smoke",
                boom,
                store=store,
                trading_day=FRIDAY,
                run_mode=RUN_MODE_REPLAY,
                transient_retry=False,
            )
        document = _written(store, "smoke")
        assert document["status"] == "failed"
        assert document["run_mode"] == RUN_MODE_REPLAY

    def test_an_invocation_that_declares_nothing_is_refused(
        self, store: LocalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The refusal that makes the requirement real, shown firing.

        The suite's own `declared_run_mode` fixture is cleared here on
        purpose: without this case, every test above would pass against a
        `resolve_run_mode` that had quietly grown a default.
        """
        monkeypatch.delenv(RUN_MODE_ENV, raising=False)
        with pytest.raises(RunModeError):
            run_job("smoke", lambda ctx: None, store=store, trading_day=FRIDAY)

    def test_the_refusal_happens_before_the_job_body_runs(
        self, store: LocalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(RUN_MODE_ENV, raising=False)
        ran: list[bool] = []
        with pytest.raises(RunModeError):
            run_job("smoke", lambda ctx: ran.append(True), store=store, trading_day=FRIDAY)
        assert not ran, "the job body ran before the invocation had declared its mode"

    def test_an_undeclared_run_writes_no_manifest(
        self, store: LocalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(RUN_MODE_ENV, raising=False)
        with pytest.raises(RunModeError):
            run_job("smoke", lambda ctx: None, store=store, trading_day=FRIDAY)
        with pytest.raises(KeyError):
            store.get_bytes(manifest_key("smoke", FRIDAY.isoformat()))


class TestResolutionOrderIsExplicitThenEnvironmentThenRefusal:
    def test_the_explicit_value_wins_over_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(RUN_MODE_ENV, RUN_MODE_LIVE)
        assert resolve_run_mode(RUN_MODE_REPLAY) == RUN_MODE_REPLAY

    def test_the_environment_answers_when_nothing_explicit_is_passed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(RUN_MODE_ENV, RUN_MODE_REPLAY)
        assert resolve_run_mode() == RUN_MODE_REPLAY

    @pytest.mark.parametrize("value", ["", "LIVE", "yes", "replay "])
    def test_a_junk_environment_value_is_refused_not_ignored(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        """Refused, never silently treated as unset: falling back to a
        refusal would be right, but falling THROUGH to a default would make a
        typo in a workflow read as a live Saturday."""
        monkeypatch.setenv(RUN_MODE_ENV, value)
        with pytest.raises(RunModeError):
            resolve_run_mode()

    def test_an_explicit_value_outside_the_vocabulary_is_refused(self) -> None:
        with pytest.raises(RunModeError):
            resolve_run_mode("saturday")

    def test_the_refusal_names_how_to_fix_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(RUN_MODE_ENV, raising=False)
        with pytest.raises(RunModeError) as caught:
            resolve_run_mode()
        message = str(caught.value)
        assert "--run-mode" in message
        assert RUN_MODE_ENV in message


# ---------------------------------------------------------------------------
# the consumer
# ---------------------------------------------------------------------------


class TestTheGateReadsWhatTheProducerWrote:
    """End to end, over the REAL producer output — not a hand-built fixture.

    The two halves were previously only ever tested against each other's
    restatements, which is how a field can exist in a schema, be written by
    nobody, and read as met by a clause.
    """

    def test_a_live_run_is_counted_and_a_replay_is_not(self, store: LocalStore) -> None:
        anchors = [gate_module.weekly_anchor(day) for day in _render_window()]
        for day in anchors:
            run_job(
                "weekly", lambda ctx: None, store=store, trading_day=day, run_mode=RUN_MODE_LIVE
            )
        live = gate_module._clause_live_saturdays_first_attempt_ok(store, _render_window())
        assert live.met and not live.unmeasurable

        for day in anchors:
            run_job(
                "weekly", lambda ctx: None, store=store, trading_day=day, run_mode=RUN_MODE_REPLAY
            )
        replayed = gate_module._clause_live_saturdays_first_attempt_ok(store, _render_window())
        assert not replayed.met and not replayed.unmeasurable
        assert "not live" in replayed.detail


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _render_window() -> list[dt.date]:
    """Two consecutive weeks ending on a WEDNESDAY — the ladder renders daily,
    and the weekly manifests are filed at the Friday anchors."""
    wednesday = dt.date(2026, 9, 9)
    return [wednesday - dt.timedelta(weeks=n) for n in reversed(range(2))]


def _written(store: LocalStore, job: str) -> dict:
    return json.loads(store.get_bytes(manifest_key(job, FRIDAY.isoformat())).decode("utf-8"))


def _manifest(**overrides: object) -> dict:
    """A conformant `run_manifest.v2`, at the floor: every required field and
    nothing optional, so a test that removes one is testing a requirement."""
    document = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": "01JG0000000000000000000000",
        "job": "weekly",
        "run_mode": RUN_MODE_LIVE,
        "trading_day": "2026-08-28",
        "calendar_date": "2026-08-29",
        "status": "ok",
        "reason": "",
        "started": "2026-08-29T13:00:00Z",
        "finished": "2026-08-29T13:04:11Z",
        "code_sha": "0" * 40,
        "release_sha": "1" * 40,
        "seed": 20260828,
        "inputs": [],
        "outputs": [],
        "rows_in": 0,
        "rows_out": 0,
        "rows_rejected": [],
        "cost_usd": 0.0,
        "llm_calls": [],
        "resource": {
            "instance_type": "local",
            "spot": False,
            "escalated_to_on_demand": False,
            "interruptions": 0,
            "mem_peak_mb": 0.0,
            "disk_free_mb": 0.0,
        },
        "metrics": [],
        "attempts": [{"n": 1, "reason": "initial"}],
    }
    document.update(overrides)
    return document


def test_the_floor_fixture_is_actually_conformant() -> None:
    """The precondition every refusal case above rests on. If this fixture
    stopped validating, every `pytest.raises` here would pass for the wrong
    reason."""
    Draft202012Validator.check_schema(load_schema())
    validate(_manifest())
