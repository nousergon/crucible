"""The deploy's three steps, and the workflow that drives them.

Normative source: plan §4.11; `pull-request-policy` §4.2.

The workflow assertions are here rather than in a separate guard workflow on
purpose: §4.11 says every guard the v1 repos carry as a per-repo workflow
becomes a pytest here, or is not carried.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from crucible.deploy import main as deploy_main
from crucible.manifest import manifest_key, validate
from crucible.release import current_release, publish_release, wheel_key
from crucible.store import LocalStore

SHA = "a" * 40
OTHER = "b" * 40
WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"
DEPLOY_YML = WORKFLOWS / "deploy.yml"


def _release_json(sha=SHA) -> str:
    return json.dumps(
        {
            "schema_version": "release.v1",
            "sha": sha,
            "built_at": "2026-08-28T21:00:00Z",
            "lockfile_sha256": "0" * 64,
            "wheel_sha256": "0" * 64,
            "test_summary": "42 passed",
            "workflow_run_url": "https://github.com/nousergon/crucible/actions/runs/1",
            "python_requires": ">=3.12,<3.13",
            "extra": {},
        }
    )


def _write_smoke(store, sha=SHA, status="ok", *, trading_day=None):
    from crucible.calendar import resolve_trading_day

    day = (trading_day or resolve_trading_day()).isoformat()
    store.put_bytes(
        manifest_key("smoke", day),
        json.dumps(
            {
                "job": "smoke",
                "release_sha": sha,
                "status": status,
                "reason": "" if status == "ok" else "RuntimeError: live read failed",
                "trading_day": day,
            }
        ).encode(),
    )


class TestPublish:
    def test_it_uploads_both_artifacts_and_promotes_nothing(self, tmp_path) -> None:
        store_dir = tmp_path / "store"
        wheel = tmp_path / "w.whl"
        wheel.write_bytes(b"PK\x03\x04")
        meta = tmp_path / "release.json"
        meta.write_text(_release_json())
        deploy_main(
            [
                "publish",
                "--sha",
                SHA,
                "--store",
                str(store_dir),
                "--wheel",
                str(wheel),
                "--release-json",
                str(meta),
            ]
        )
        store = LocalStore(store_dir)
        assert store.exists(wheel_key(SHA))
        assert current_release(store) is None

    def test_a_release_json_for_another_build_is_refused(self, tmp_path) -> None:
        """Publishing it under the wrong prefix would make the rollback target
        a build it does not describe."""
        wheel = tmp_path / "w.whl"
        wheel.write_bytes(b"PK")
        meta = tmp_path / "release.json"
        meta.write_text(_release_json(OTHER))
        with pytest.raises(SystemExit, match="not"):
            deploy_main(
                [
                    "publish",
                    "--sha",
                    SHA,
                    "--store",
                    str(tmp_path / "s"),
                    "--wheel",
                    str(wheel),
                    "--release-json",
                    str(meta),
                ]
            )


class TestFlip:
    def _published(self, tmp_path):
        store = LocalStore(tmp_path)
        publish_release(
            store, sha=SHA, wheel=b"w", lockfile=b"l", test_summary="", workflow_run_url=""
        )
        return store

    def test_an_ok_smoke_flips_the_pointer(self, tmp_path) -> None:
        store = self._published(tmp_path)
        _write_smoke(store)
        assert deploy_main(["flip", "--sha", SHA, "--store", str(tmp_path)]) == 0
        assert current_release(store) == SHA

    def test_a_failed_smoke_fails_the_deploy_and_leaves_the_pointer(self, tmp_path) -> None:
        """ "Nothing was promoted" and "nothing needed promoting" must not look
        the same on the surface."""
        store = self._published(tmp_path)
        _write_smoke(store, status="failed")
        with pytest.raises(SystemExit, match="untouched"):
            deploy_main(["flip", "--sha", SHA, "--store", str(tmp_path)])
        assert current_release(store) is None

    def test_a_missing_smoke_manifest_refuses_rather_than_promoting(self, tmp_path) -> None:
        """Promoting without one would flip the pointer on a step that may
        never have executed."""
        self._published(tmp_path)
        with pytest.raises(SystemExit, match="no smoke manifest"):
            deploy_main(["flip", "--sha", SHA, "--store", str(tmp_path)])


class TestRecord:
    def test_the_deploy_manifest_validates_against_the_run_schema(self, tmp_path) -> None:
        """§4.11: the deploy writes its own manifest in the run-manifest
        schema, so §4.5's page shows deploys beside runs."""
        store = LocalStore(tmp_path)
        publish_release(
            store, sha=SHA, wheel=b"w", lockfile=b"l", test_summary="", workflow_run_url=""
        )
        _write_smoke(store)
        deploy_main(["flip", "--sha", SHA, "--store", str(tmp_path)])
        deploy_main(
            [
                "record",
                "--sha",
                SHA,
                "--store",
                str(tmp_path),
                "--outcome",
                "success",
                "--run-url",
                "https://x",
            ]
        )
        from crucible.calendar import resolve_trading_day

        manifest = json.loads(
            store.get_bytes(manifest_key("deploy", resolve_trading_day().isoformat()))
        )
        validate(manifest)
        assert manifest["status"] == "ok"

    def test_a_successful_job_whose_pointer_did_not_move_records_failed(self, tmp_path) -> None:
        """Derived, never declared. A deploy reporting ok while the pointer
        had not moved is the degraded-SUCCEEDED this system refuses."""
        store = LocalStore(tmp_path)
        deploy_main(
            [
                "record",
                "--sha",
                SHA,
                "--store",
                str(tmp_path),
                "--outcome",
                "success",
                "--run-url",
                "https://x",
            ]
        )
        from crucible.calendar import resolve_trading_day

        manifest = json.loads(
            store.get_bytes(manifest_key("deploy", resolve_trading_day().isoformat()))
        )
        validate(manifest)
        assert manifest["status"] == "failed"
        assert "expected" in manifest["reason"]

    def test_recording_a_failed_deploy_still_exits_zero(self, tmp_path) -> None:
        """A recorder that failed the job a second time would mask which step
        actually broke."""
        assert (
            deploy_main(
                [
                    "record",
                    "--sha",
                    SHA,
                    "--store",
                    str(tmp_path),
                    "--outcome",
                    "failure",
                    "--run-url",
                    "https://x",
                ]
            )
            == 0
        )


class TestTheWorkflowItself:
    @pytest.fixture
    def workflow(self) -> dict:
        return yaml.safe_load(DEPLOY_YML.read_text(encoding="utf-8"))

    def test_it_refuses_any_role_but_the_v2_deploy_role(self, workflow: dict) -> None:
        """The failure being designed out is not "the workflow errors" — it is
        quietly assuming the research role, which exists, is assumable from
        this org, and writes a DIFFERENT bucket."""
        guard = workflow["jobs"]["guard"]
        body = json.dumps(guard)
        assert "crucible-v2-github-deploy" in body
        assert "github-actions-lambda-deploy" in body, (
            "the guard must name the role it is refusing to fall back to, or the "
            "message does not tell an operator what nearly happened"
        )
        assert "id-token" not in json.dumps(guard.get("permissions", {}))

    def test_every_job_that_needs_aws_depends_on_the_guard(self, workflow: dict) -> None:
        jobs = workflow["jobs"]
        for name, job in jobs.items():
            if "id-token: write" in json.dumps(job.get("permissions", {})) or "aws" in json.dumps(
                job
            ):
                if name in ("guard", "notify-main-failure"):
                    continue
                chain = set(job.get("needs", []))
                assert chain, f"{name} has no `needs` and could run before the guard"

    def test_every_action_is_pinned_to_a_sha(self, workflow: dict) -> None:
        """A floating tag is a supply-chain hole: the tag moves, the workflow
        does not, and nothing announces the change."""
        uses = re.findall(r"uses:\s*(\S+)", DEPLOY_YML.read_text(encoding="utf-8"))
        for ref in uses:
            if ref.startswith("./"):
                continue
            _, _, version = ref.partition("@")
            assert re.fullmatch(r"[0-9a-f]{40}", version), f"{ref} is not SHA-pinned"

    def test_permissions_are_default_deny_at_the_workflow_level(self, workflow: dict) -> None:
        assert workflow["permissions"] == {}

    def test_only_the_release_job_may_mint_an_oidc_token(self, workflow: dict) -> None:
        minting = {
            name
            for name, job in workflow["jobs"].items()
            if (job.get("permissions") or {}).get("id-token") == "write"
        }
        assert minting == {"release"}

    def test_a_superseded_deploy_is_not_cancelled_mid_smoke(self, workflow: dict) -> None:
        """Cancelling mid-smoke leaves a release published with no verdict,
        indistinguishable from one never attempted."""
        assert workflow["concurrency"]["cancel-in-progress"] is False

    def test_the_deploy_carries_no_run_this_after_merging_instruction(self) -> None:
        """A PR must be deployable by the merge button alone. Hitting merge is
        Brian's LAST action."""
        text = DEPLOY_YML.read_text(encoding="utf-8").lower()
        assert "after merging" not in text.replace("after merging is", "")
        assert "form 1" in DEPLOY_YML.read_text(encoding="utf-8")

    def test_every_job_declares_a_timeout(self, workflow: dict) -> None:
        for name, job in workflow["jobs"].items():
            if "uses" in job:  # a reusable workflow carries its own
                continue
            assert job.get("timeout-minutes"), f"{name} has no timeout"

    def test_the_pointer_flip_reads_a_manifest_rather_than_an_exit_code(self) -> None:
        """A caller-supplied "it passed" promotes a build whose smoke quietly
        did nothing."""
        text = DEPLOY_YML.read_text(encoding="utf-8")
        assert "crucible.deploy flip" in text
        assert "Flip releases/current on an ok smoke" in text
