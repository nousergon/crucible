"""`dispatch_exit.v1`'s typed boundary (`alpha-engine-config-I11050`).

M0 contract discipline: a new cross-repo artifact gets a versioned JSON
Schema plus a producer/consumer contract test at birth. The producer is
`crucible.dispatch_exit` (invoked from the box's EXIT trap, and by the
bootstrap shell in `nous-ergon-ops/infrastructure/cloudformation/
crucible-v2.yaml` when the wheel was never installed); the consumer is
`crucible.alerts._classify_dispatch_absence`.

Mirrors `tests/test_typed_boundary_fault_record.py`'s shape exactly: the
schema is GENERATED from the model, never hand-edited, and this file is the
drift guard between the committed `.json` and the model that generates it.
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib

import pytest
from pydantic import ValidationError

from crucible.dispatch_exit import build_exit_document, classify_exit, write_exit_record
from crucible.keys import DISPATCH_EXIT_SUFFIX, dispatch_exit_key, parse_dispatch_key
from crucible.models import DISPATCH_EXIT_CLASSES, DispatchExitDocument
from crucible.runner import MAX_ATTEMPTS
from crucible.store import LocalStore

SCHEMA_PATH = (
    pathlib.Path(__file__).resolve().parents[1] / "crucible" / "schemas" / "dispatch_exit.v1.json"
)

VALID = {
    "schema_version": "dispatch_exit.v1",
    "dispatch_id": "441e55ef6f6d6478f42e84919b81a960",
    "instance_id": "i-02d1de194ea8420d5",
    "job": "data.heal",
    "argv": "--date 2026-01-30 --from 2025-07-01 --to 2026-01-30 --run-mode live",
    "exit_code": 1,
    "exit_class": "spot_reclaimed",
    "last_error_line": (
        "crucible.runner.SpotInterruptionError: spot_interruption: received signal 15"
    ),
    "console_tail": "reclaimed by a spot interruption (exit 1): no page",
    "log_group": "/crucible/data.heal",
    "log_stream": "i-02d1de194ea8420d5",
    "expected_manifest_key": "runs/data.heal/2026-01-30/run.json",
    "manifest_written": False,
    "redispatch_expected": True,
    "next_attempt_dispatch_id": "441e55ef6f6d6478f42e84919b81a960-r2",
    "attempts": [{"n": 1, "reason": "initial"}],
    "finished_at_utc": "2026-09-15T01:26:00Z",
}


def _valid(**overrides: object) -> dict:
    document = dict(VALID)
    document.update(overrides)
    return document


class TestTheCommittedSchemaIsGeneratedFromTheModel:
    def test_the_committed_schema_is_byte_identical_to_the_generated_one(self) -> None:
        generated = (
            json.dumps(DispatchExitDocument.model_json_schema(), indent=2, sort_keys=True) + "\n"
        )
        assert SCHEMA_PATH.read_text(encoding="utf-8") == generated, (
            "dispatch_exit.v1.json has drifted from DispatchExitDocument. The schema is "
            "GENERATED, never hand-edited: regenerate it in the same commit."
        )

    def test_the_real_2026_09_15_record_validates(self) -> None:
        DispatchExitDocument.model_validate(_valid())


class TestTheClassAndTheCodeCannotContradictEachOther:
    """Every consumer splits on `exit_class`; a record whose class and code
    disagree would silence a page for a run that failed."""

    def test_ok_with_a_nonzero_code_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="contradicts its code"):
            DispatchExitDocument.model_validate(
                _valid(exit_class="ok", exit_code=1, redispatch_expected=False)
            )

    def test_a_failure_class_with_code_zero_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="the other way round"):
            DispatchExitDocument.model_validate(_valid(exit_code=0))

    def test_refused_is_argparses_exit_two(self) -> None:
        with pytest.raises(ValidationError, match="is exit_code 2"):
            DispatchExitDocument.model_validate(
                _valid(exit_class="refused", exit_code=1, redispatch_expected=False)
            )

    def test_a_redispatch_promise_outside_the_reclaimed_path_is_refused(self) -> None:
        """`alpha-engine-config-I11051`: the promise is what the manifest
        suppression is made on. Declared anywhere else it would owe an
        attempt that nothing launches."""
        with pytest.raises(ValidationError, match="only on the reclaimed path"):
            DispatchExitDocument.model_validate(_valid(exit_class="failed"))

    def test_ok_without_a_manifest_is_refused(self) -> None:
        """Repo rule 1 — an exit record must not excuse a missing manifest."""
        with pytest.raises(ValidationError, match="manifest or it did not happen"):
            DispatchExitDocument.model_validate(
                _valid(
                    exit_class="ok",
                    exit_code=0,
                    manifest_written=False,
                    redispatch_expected=False,
                )
            )

    def test_an_unknown_class_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            DispatchExitDocument.model_validate(_valid(exit_class="reclaimed"))

    def test_a_field_omitted_entirely_is_refused_not_defaulted(self) -> None:
        document = _valid()
        del document["console_tail"]
        with pytest.raises(ValidationError, match="console_tail"):
            DispatchExitDocument.model_validate(document)


class TestTheKeyShapeCannotBeMistakenForADispatchRecord:
    def test_the_exit_key_lives_beside_the_request_record(self) -> None:
        assert dispatch_exit_key("data.heal", "abc") == (
            f"runs/_dispatch/data.heal/abc{DISPATCH_EXIT_SUFFIX}"
        )

    def test_parse_dispatch_key_refuses_an_exit_record(self) -> None:
        """Both end `.json` under the same prefix. Parsed as a dispatch
        record, an exit record is a dispatch with no manifest anywhere near
        it — the detector would manufacture an absence per box."""
        assert parse_dispatch_key(dispatch_exit_key("data.heal", "abc")) is None


class TestTheClassifier:
    def test_a_clean_exit_is_ok_even_after_a_reclaim_notice(self) -> None:
        """A run that finished before the reclaim took the box DID its work."""
        assert classify_exit(0, reclaimed=True, job_started=True) == "ok"

    def test_a_reclaim_outranks_the_job_split(self) -> None:
        assert classify_exit(143, reclaimed=True, job_started=True) == "spot_reclaimed"

    def test_argparses_usage_exit_is_refused(self) -> None:
        assert classify_exit(2, reclaimed=False, job_started=True) == "refused"

    def test_dying_before_the_job_started_is_a_bootstrap_failure(self) -> None:
        assert classify_exit(1, reclaimed=False, job_started=False) == "bootstrap_failed"

    def test_the_shell_fallback_class(self) -> None:
        assert (
            classify_exit(1, reclaimed=False, job_started=False, crucible_installed=False)
            == "bootstrap_failed"
        )

    def test_every_class_it_can_return_is_declared(self) -> None:
        produced = {
            classify_exit(code, reclaimed=reclaimed, job_started=started)
            for code in (0, 1, 2, 143)
            for reclaimed in (True, False)
            for started in (True, False)
        }
        assert produced <= set(DISPATCH_EXIT_CLASSES)


class TestTheWriter:
    def test_the_promise_is_derived_from_the_ladder_not_passed_in(self) -> None:
        """One predicate, in one place: the same "spot and room on the
        ladder" the runner applies when it suppresses the manifest."""
        exhausted = build_exit_document(
            dispatch_id="abc",
            instance_id="i-0x",
            job="data.heal",
            argv="--to 2026-01-30",
            exit_code=1,
            exit_class="spot_reclaimed",
            console=None,
            log_group=None,
            log_stream=None,
            expected_manifest_key=None,
            manifest_written=False,
            attempts=[{"n": n, "reason": "initial"} for n in range(1, MAX_ATTEMPTS + 1)],
            max_attempts=MAX_ATTEMPTS,
        )
        assert exhausted.redispatch_expected is False
        assert exhausted.next_attempt_dispatch_id is None

    def test_a_first_attempt_reclaim_names_its_successor(self) -> None:
        document = build_exit_document(
            dispatch_id="abc",
            instance_id="i-0x",
            job="data.heal",
            argv="--to 2026-01-30",
            exit_code=1,
            exit_class="spot_reclaimed",
            console="crucible.runner.SpotInterruptionError: received signal 15\ndata.heal exited 1",
            log_group="/crucible/data.heal",
            log_stream="i-0x",
            expected_manifest_key="runs/data.heal/2026-01-30/run.json",
            manifest_written=False,
            attempts=[{"n": 1, "reason": "initial"}],
            max_attempts=MAX_ATTEMPTS,
            finished_at=dt.datetime(2026, 9, 15, 1, 26, tzinfo=dt.UTC),
        )
        assert document.redispatch_expected is True
        assert document.next_attempt_dispatch_id == "abc-r2"
        assert document.last_error_line is not None
        assert "SpotInterruptionError" in document.last_error_line

    def test_it_writes_where_the_reader_looks(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        document = DispatchExitDocument.model_validate(_valid())
        key = write_exit_record(store, document)
        assert key == dispatch_exit_key(document.job, document.dispatch_id)
        assert json.loads(store.get_bytes(key).decode()) == json.loads(
            json.dumps(document.model_dump(mode="json"))
        )


class TestTheRunLeavesTheKeyItActuallyBound:
    """`alpha-engine-config-I11050`/`-I11048`: the box's exit trap runs after
    the process is over — including on the reclaimed path, where no manifest
    is written — so the only party that can state WHICH manifest was owed is
    the run itself, before its body starts."""

    def test_it_writes_the_key_into_the_box_state_dir(self, tmp_path, monkeypatch) -> None:
        from crucible.runner import EXPECTED_MANIFEST_KEY_FILE, STATE_DIR_ENV, run_job

        state = tmp_path / "state"
        monkeypatch.setenv(STATE_DIR_ENV, str(state))
        store = LocalStore(tmp_path / "store")
        run_job(
            "data.heal",
            lambda ctx: None,
            store=store,
            trading_day=dt.date(2026, 1, 30),
            run_mode="replay",
        )
        assert (state / EXPECTED_MANIFEST_KEY_FILE).read_text(encoding="utf-8") == (
            "runs/data.heal/2026-01-30/run.json"
        )

    def test_it_is_a_no_op_off_a_box(self, tmp_path, monkeypatch) -> None:
        from crucible.runner import STATE_DIR_ENV, run_job

        monkeypatch.delenv(STATE_DIR_ENV, raising=False)
        run_job(
            "data.heal",
            lambda ctx: None,
            store=LocalStore(tmp_path / "store"),
            trading_day=dt.date(2026, 1, 30),
            run_mode="replay",
        )
        assert list(tmp_path.glob("**/manifest-key")) == []

    def test_an_unwritable_state_dir_does_not_fail_the_run(self, tmp_path, monkeypatch) -> None:
        """The recorded swallow: a diagnostic must never fail a run."""
        blocked = tmp_path / "blocked"
        blocked.write_text("not a directory", encoding="utf-8")
        from crucible.runner import STATE_DIR_ENV, run_job

        monkeypatch.setenv(STATE_DIR_ENV, str(blocked))
        ctx = run_job(
            "data.heal",
            lambda ctx: None,
            store=LocalStore(tmp_path / "store"),
            trading_day=dt.date(2026, 1, 30),
            run_mode="replay",
        )
        assert ctx.run_id
