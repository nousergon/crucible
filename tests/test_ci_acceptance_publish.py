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


def test_a_publish_step_exists_and_runs_the_registered_job() -> None:
    """The publisher is `crucible acceptance.publish`, not a copy.

    `alpha-engine-config-I10968`: this was a raw `aws s3 cp` of the local
    reading file, preceded by two shell steps that resolved the trading day
    and the store key through `python -c` one-liners. None of it went through
    `crucible.store.Store`, so the producer of the one plan §12 rule-3
    progress figure wrote no run manifest — while `crucible report.morning`,
    `crucible board` and phase 0's own gate clause all read that artifact as
    evidence.

    The key shape is still resolved through `crucible.keys.
    acceptance_reading_key` and the day through `crucible.calendar` — but
    INSIDE the job body now, which is the stronger form of the property
    `alpha-engine-config-I9902`'s review asked for: a value resolved in the
    process that performs the write cannot drift from the value the consumer
    reads, and there is no shell step left in which to restate either.
    """
    job = _acceptance_job()
    names = _step_names(job)
    assert any("publish" in name.lower() for name in names), (
        "no step in ci.yml's acceptance job publishes the acceptance reading"
    )
    steps = job["steps"]
    publish_idx = next(i for i, s in enumerate(steps) if "publish" in s.get("name", "").lower())
    publish_body = steps[publish_idx].get("run", "")
    assert "crucible acceptance.publish" in publish_body, (
        "the publish step must run the registered job, so the write goes "
        "through crucible.runner.run_job and files a manifest (rule 1)"
    )
    assert "--run-mode" in publish_body, (
        "a crucible job's invocation declares its run mode; omitted, it falls "
        "back to $CRUCIBLE_RUN_MODE and a replay could file a live key"
    )
    assert "aws s3" not in publish_body, (
        "the reading is published through the store, never copied around it"
    )


def test_no_step_copies_anything_into_the_store_behind_the_cli() -> None:
    """The class, not the instance (`engagement-protocol-policy` §5).

    A second hand-rolled copy anywhere in this job would file no manifest for
    exactly the same reason the first one did. The `aws s3api
    get-bucket-versioning` READ stays legal; a write does not.
    """
    for step in _acceptance_job()["steps"]:
        body = step.get("run", "")
        for line in body.splitlines():
            if "aws s3 cp" in line or "aws s3 sync" in line or "aws s3api put-object" in line:
                assert "STORE_URI" not in line, (
                    f"step {step.get('name')!r} writes to the store with the AWS CLI: "
                    f"{line.strip()!r}. A store write is a job, so that it files a "
                    "manifest (rule 1)."
                )


def test_the_key_and_the_trading_day_are_never_restated_in_the_shell() -> None:
    """§4.12 plus `alpha-engine-config-I9902`, asserted as an ABSENCE now.

    The two `python -c` resolution steps are gone: the job resolves both, so a
    workflow edit has nothing left to get wrong. A step that reintroduced
    either would be resolving a value a second time, in a second place, which
    is the drift both issues were about.
    """
    bodies = "\n".join(_step_bodies(_acceptance_job()))
    assert "acceptance_reading_key" not in bodies, (
        "the publish key is resolved inside `crucible acceptance.publish`, not "
        "in a shell step that hands it to a copy command"
    )
    steps = _acceptance_job()["steps"]
    publish_body = next(s.get("run", "") for s in steps if "publish" in s.get("name", "").lower())
    # Scoped to the publishing step: the identity step's REFUSAL MESSAGE names
    # the prefix its OIDC role is scoped to, which is documentation of a grant,
    # not a key being restated to write to.
    assert "report/acceptance/" not in publish_body, (
        "the publish step must not hand-write the report/acceptance/ literal"
    )
    assert "resolve_trading_day" not in bodies, (
        "the trading day is resolved inside the job, through crucible.calendar"
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
