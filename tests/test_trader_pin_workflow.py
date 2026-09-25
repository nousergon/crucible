"""`.github/workflows/trader-pin.yml`'s contract (alpha-engine-config-I11545).

Read as YAML, never as a shell parse: what is asserted is the structure a
reviewer would otherwise have to re-read on every edit — the triggers, the
main-only first step, the one dispatch input and how it reaches the command,
the identity, and the order guard -> rehearsal -> apply.
"""

from __future__ import annotations

import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "trader-pin.yml"
COMPONENTS = ROOT / "crucible" / "components.yaml"


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _triggers(workflow: dict) -> dict:
    # YAML 1.1 reads a bare `on` as the boolean True.
    return workflow.get(True, workflow.get("on"))


def _runs(job: dict) -> list[str]:
    return [step.get("run", "") for step in job["steps"]]


def test_the_triggers_are_a_weekday_evening_cron_and_a_sha_only_dispatch() -> None:
    triggers = _triggers(_workflow())
    assert set(triggers) == {"schedule", "workflow_dispatch"}
    assert triggers["schedule"] == [{"cron": "15 23 * * 1-5"}]
    inputs = triggers["workflow_dispatch"]["inputs"]
    assert set(inputs) == {"sha"}, "the dispatch takes the sha and nothing free-form"
    assert inputs["sha"]["required"] is True


def test_each_event_reaches_exactly_one_job() -> None:
    jobs = _workflow()["jobs"]
    assert jobs["request"]["if"] == "${{ github.event_name == 'workflow_dispatch' }}"
    assert jobs["apply"]["if"] == "${{ github.event_name == 'schedule' }}"


def test_both_jobs_refuse_a_non_main_ref_first() -> None:
    for name in ("request", "apply"):
        first = _workflow()["jobs"][name]["steps"][0]
        assert first["if"] == "github.ref != 'refs/heads/main'", name
        assert "exit 1" in first["run"], name


def test_the_sha_is_validated_and_only_ever_passed_through_env() -> None:
    steps = _workflow()["jobs"]["request"]["steps"]
    validate = next(s for s in steps if s.get("name") == "The sha input is a 40-hex release sha")
    assert validate["env"] == {"SHA": "${{ inputs.sha }}"}
    assert "^[0-9a-f]{40}$" in validate["run"]
    for step in steps:
        assert "inputs.sha" not in step.get("run", ""), (
            f"{step.get('name')!r} interpolates the input into shell text; pass it via env"
        )
    queue = steps[-1]
    assert queue["env"] == {"SHA": "${{ inputs.sha }}"}
    assert queue["run"] == (
        'uv run crucible release.pin_request "$SHA" --run-mode live --store "$STORE_URI"'
    )


def test_the_identity_is_the_trader_pin_role_built_from_variables() -> None:
    workflow = _workflow()
    arn = workflow["env"]["TRADER_PIN_ROLE_ARN"]
    assert arn == (
        "arn:aws:iam::${{ vars.AWS_ACCOUNT_ID }}:role/${{ vars.CRUCIBLE_ROLE_PREFIX }}-trader-pin"
    )
    for name in ("request", "apply"):
        steps = workflow["jobs"][name]["steps"]
        creds = [s for s in steps if "configure-aws-credentials" in s.get("uses", "")]
        assert [s["with"]["role-to-assume"] for s in creds] == ["${{ env.TRADER_PIN_ROLE_ARN }}"]
        masked = [i for i, s in enumerate(steps) if "::add-mask::" in s.get("run", "")]
        assert masked and masked[0] < steps.index(creds[0]), f"{name}: mask before credentials"
        assert workflow["jobs"][name]["permissions"] == {"contents": "read", "id-token": "write"}
    assert workflow["permissions"] == {}


def test_the_apply_guards_then_rehearses_then_applies() -> None:
    steps = _workflow()["jobs"]["apply"]["steps"]
    guard = next(i for i, s in enumerate(steps) if s.get("id") == "guard")
    assert steps[guard]["run"] == "uv run python scripts/trader_pin_guard.py"
    rehearse, apply = steps[guard + 1], steps[guard + 2]
    for step in (rehearse, apply):
        assert step["if"] == "${{ steps.guard.outputs.postclose != 'skip' }}"
        assert step["env"] == {
            "SESSION": "${{ steps.guard.outputs.session }}",
            "POSTCLOSE": "${{ steps.guard.outputs.postclose }}",
        }
    base = (
        'uv run crucible release.pin_apply --date "$SESSION" --postclose "$POSTCLOSE" '
        '--run-mode live --store "$STORE_URI"'
    )
    assert rehearse["run"] == base + " --dry-run"
    assert apply["run"] == base
    assert len(steps) == guard + 3, "nothing runs after the apply"


def test_it_installs_frozen_and_never_cancels_a_run_in_flight() -> None:
    workflow = _workflow()
    assert workflow["concurrency"] == {"group": "trader-pin", "cancel-in-progress": False}
    for name in ("request", "apply"):
        assert "uv sync --frozen" in _runs(workflow["jobs"][name])


def test_a_failure_notifies() -> None:
    notify = _workflow()["jobs"]["notify-failure"]
    assert notify["needs"] == ["request", "apply"]
    assert notify["uses"].startswith(
        "nousergon/nousergon-lib/.github/workflows/notify-ci-failure.yml@"
    )


def test_the_scheduled_apply_is_the_registry_row_that_names_this_workflow() -> None:
    """The dispatch lockstep (nous-ergon-ops) requires a cron workflow to be
    named by a row; the row is `release.pin_apply`, and it carries a deadline
    so a dropped cron pages as an absence."""
    rows = yaml.safe_load(COMPONENTS.read_text(encoding="utf-8"))["components"]
    naming = {name for name, row in rows.items() if row.get("dispatch_workflow") == WORKFLOW.name}
    assert naming == {"release.pin_apply"}
    row = rows["release.pin_apply"]
    assert row["dispatch"] == "github-actions" and row["deadline"]
    assert rows["release.pin_request"]["dispatch"] is None
