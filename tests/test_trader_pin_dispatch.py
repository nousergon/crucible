"""A dispatch is a hint; AWS remains the authority for a clean reconcile."""

import datetime as dt
import importlib.util
import pathlib
import sys

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "pin_dispatch_guard", pathlib.Path(__file__).parents[1] / "scripts/trader_pin_guard.py"
)
guard = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = guard
_SPEC.loader.exec_module(guard)

ARN = "arn:aws:states:us-east-1:123456789012:stateMachine:reconcile"
EXEC = ARN.replace(":stateMachine:", ":execution:") + ":run"
NOW = dt.datetime(2026, 11, 18, 3, tzinfo=dt.UTC)


def payload(**changes):
    data = {
        "schema_version": 1,
        "state_machine_arn": ARN,
        "execution_arn": EXEC,
        "status": "SUCCEEDED",
    }
    data.update(changes)
    return {"action": "postclose-reconciled", "client_payload": data}


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_version": 2},
        {"schema_version": True},
        {"unexpected": "field"},
        {"state_machine_arn": ARN + "other"},
        {"execution_arn": "forged"},
        {"status": "FAILED"},
    ],
)
def test_bad_dispatch_refused(changes):
    with pytest.raises(ValueError):
        guard.dispatch_execution(payload(**changes), ARN)


def test_dispatch_contract():
    assert guard.dispatch_execution(payload(), ARN) == EXEC


@pytest.mark.parametrize("matched", [False, True])
def test_dispatched_execution_must_have_succeeded_this_session(matched):
    row = {
        "startDate": NOW - dt.timedelta(hours=2),
        "status": "SUCCEEDED",
        "executionArn": EXEC if matched else EXEC + "other",
    }
    reading = guard.read(NOW, lambda: [{"executions": [row]}], execution_arn=EXEC)
    assert reading.postclose == ("clean" if matched else "unclean")


def test_receipt_is_bound_to_event_and_never_written_by_cron(tmp_path, monkeypatch):
    import json

    monkeypatch.syspath_prepend(str(pathlib.Path(__file__).parents[1] / "scripts"))
    import trader_pin_receipt

    event = tmp_path / "event.json"
    event.write_text(json.dumps(payload()))
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))
    monkeypatch.setenv("POSTCLOSE_STATE_MACHINE_ARN", ARN)
    monkeypatch.setenv("STORE_URI", "s3://fixture/prefix")
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    written = []

    class Client:
        def put_object(self, **kwargs):
            written.append(kwargs)

    monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
    with pytest.raises(ValueError):
        trader_pin_receipt.main(Client())
    assert written == []
    monkeypatch.setenv("GITHUB_EVENT_NAME", "repository_dispatch")
    assert trader_pin_receipt.main(Client()) == 0
    assert written[0]["Key"].startswith("prefix/runs/release.pin_apply/dispatch/")
    assert json.loads(written[0]["Body"])["execution_arn"] == EXEC
    from importlib.resources import files

    from jsonschema import validate

    validate(
        json.loads(written[0]["Body"]),
        json.loads(files("crucible.schemas").joinpath("postclose_receipt.v1.json").read_text()),
    )


def test_receipt_versioned_contract(tmp_path, monkeypatch):
    import json
    from importlib.resources import files

    from jsonschema import Draft202012Validator

    schema = json.loads(files("crucible.schemas").joinpath("postclose_receipt.v1.json").read_text())
    validator = Draft202012Validator(schema)
    valid = {
        "schema_version": 1,
        "execution_arn": EXEC,
        "workflow_run_id": "123",
        "outcome": "workflow_completed",
    }
    validator.validate(valid)
    for invalid in (
        dict(valid, schema_version=True),
        dict(valid, outcome="cron"),
        dict(valid, workflow_run_id=""),
        dict(valid, extra=1),
    ):
        assert list(validator.iter_errors(invalid))
