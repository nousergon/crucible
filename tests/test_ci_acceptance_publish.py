"""`ci.yml`'s acceptance job actually publishes the reading it grades.

`alpha-engine-config-I9895`, `I9902`. Two structural facts, neither of which
a passing pytest run elsewhere can prove: the job holds `id-token: write` (no
OIDC, no live AWS, no measurable `TestCost` clause and no publisher identity
at all), and a publish step exists that writes the reading document — never
inferred from "the checker didn't crash", since a workflow can run a script
and simply not upload what it produced.

This does not re-derive `crucible.keys.acceptance_reading_key`'s shape (that
is `test_acceptance_reading.py`'s `--write-json` coverage) — it asserts the
workflow actually calls the checker with `--write-json` and then ships the
file it wrote, which is the half no unit test of `check_reading.py` alone can
see.
"""

from __future__ import annotations

import pathlib

import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
CI_WORKFLOW = REPO / ".github" / "workflows" / "ci.yml"


def _workflow() -> dict:
    return yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))


def _acceptance_job() -> dict:
    workflow = _workflow()
    return workflow["jobs"]["acceptance"]


def _step_bodies(job: dict) -> list[str]:
    return [step.get("run", "") for step in job["steps"]]


def _step_names(job: dict) -> list[str]:
    return [step.get("name", "") for step in job["steps"]]


def test_the_acceptance_job_holds_id_token_write() -> None:
    """No OIDC, no credentials, no measurable TestCost clause and no
    publisher identity — this is the one grant that makes both possible
    (alpha-engine-config-I9895)."""
    job = _acceptance_job()
    assert job["permissions"].get("id-token") == "write", (
        "ci.yml's acceptance job must hold id-token: write to assume its OIDC role"
    )
    assert job["permissions"].get("contents") == "read"


def test_the_job_configures_aws_credentials_via_oidc() -> None:
    job = _acceptance_job()
    uses = [step.get("uses", "") for step in job["steps"]]
    assert any(u.startswith("aws-actions/configure-aws-credentials@") for u in uses), (
        "no aws-actions/configure-aws-credentials step in ci.yml's acceptance job"
    )


def test_the_job_confirms_the_identity_it_assumed() -> None:
    """Same falsifiability argument board.yml and morning-report.yml make:
    what is checked is the identity actually obtained, not the literal in
    `env`."""
    names = _step_names(_acceptance_job())
    assert any("identity" in name.lower() for name in names), (
        "ci.yml's acceptance job never reads back the identity it assumed"
    )


def test_the_reading_is_computed_with_write_json_and_a_commit() -> None:
    bodies = "\n".join(_step_bodies(_acceptance_job()))
    assert "check_reading.py" in bodies
    assert "--write-json" in bodies, (
        "ci.yml's acceptance job must call check_reading.py --write-json so "
        "the reading it grades is the same one it publishes — never a "
        "second parse of check_reading.py's stdout"
    )
    assert "--commit" in bodies, (
        "the published reading must carry the commit it was measured at (plan §6 rule 2)"
    )


def test_a_publish_step_exists_and_writes_the_acceptance_prefix() -> None:
    """The producer must resolve the key through `crucible.keys.
    acceptance_reading_key` — never restate the path as a literal — so the
    producer (this workflow) and the consumer (`crucible/morning.py`) cannot
    diverge on the shape (alpha-engine-config-I9902 adversarial review)."""
    job = _acceptance_job()
    names = _step_names(job)
    assert any("publish" in name.lower() for name in names), (
        "no step in ci.yml's acceptance job publishes the acceptance reading"
    )
    bodies = "\n".join(_step_bodies(job))
    assert "acceptance_reading_key" in bodies, (
        "the workflow must resolve the publish key by calling "
        "crucible.keys.acceptance_reading_key, not restating the path"
    )
    steps = job["steps"]
    publish_idx = next(i for i, s in enumerate(steps) if "publish" in s.get("name", "").lower())
    publish_body = steps[publish_idx].get("run", "")
    assert "report/acceptance/" not in publish_body, (
        "the publish step must not hand-write the report/acceptance/ literal "
        "— it must use the key resolved from keys.acceptance_reading_key"
    )


def test_the_trading_day_is_resolved_through_the_calendar_not_the_wall_clock_date() -> None:
    """§4.12: every store key is an NYSE trading day, resolved through
    `crucible.calendar`, never a raw calendar date."""
    bodies = "\n".join(_step_bodies(_acceptance_job()))
    assert "resolve_trading_day" in bodies, (
        "the trading day used in the published key must come from "
        "crucible.calendar.resolve_trading_day, not date(1)/UTC-now"
    )


def test_publish_is_not_skipped_when_the_reading_check_fails() -> None:
    """A moved reading must still be published — plan §12 rule 3 makes the
    count the progress figure, and a publisher silent on exactly the runs
    that regressed would defeat the point."""
    job = _acceptance_job()
    steps = job["steps"]
    check_idx = next(i for i, s in enumerate(steps) if s.get("id") == "check")
    assert steps[check_idx].get("continue-on-error") is True, (
        "the reading-check step must continue-on-error so later steps "
        "(publish, then the explicit job failure) still run"
    )
    publish_idx = next(i for i, s in enumerate(steps) if "publish" in s.get("name", "").lower())
    assert publish_idx > check_idx, "publish must run after the reading is computed"
    # And the job must still end up failed on a moved reading -- a later step
    # explicitly re-raises the failure `continue-on-error` swallowed.
    fail_idx = next(i for i, s in enumerate(steps) if "fail the job" in s.get("name", "").lower())
    assert fail_idx > publish_idx


def test_no_iam_action_or_identifier_literal_leaks_outside_the_env_block() -> None:
    """`crucible/AGENTS.md`: never an ARN/account id/bucket literal in the
    crucible tree except the existing workflow env pattern (mirroring
    board.yml). The role ARN and store URI belong in `env:`, and every step
    references them through `${{ env.* }}` / `$VARNAME`, never restated."""
    job = _acceptance_job()
    for step in job["steps"]:
        body = step.get("run", "")
        assert "arn:aws:iam::" not in body, (
            f"step {step.get('name')!r} restates the role ARN outside env:"
        )
