"""The deploy's three steps, and the workflow that drives them.

Normative source: plan §4.11; `pull-request-policy` §4.2.

The workflow assertions are here rather than in a separate guard workflow on
purpose: §4.11 says every guard the v1 repos carry as a per-repo workflow
becomes a pytest here, or is not carried.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pytest
import yaml

from crucible.deploy import _publish
from crucible.deploy import main as deploy_main
from crucible.manifest import manifest_key, validate
from crucible.release import (
    POINTER_KEY,
    ReleaseImmutabilityError,
    current_release,
    pin,
    provenance_key,
    publish_release,
    read_pointer,
    release_json_key,
    wheel_key,
)
from crucible.store import LocalStore, S3Store, sha256_hex


class _FakeS3Client:
    """Just enough of the boto3 S3 surface to prove `deploy._publish` asks
    S3 to lock what it writes, per I9782/I9787. See `test_release._FakeS3Client`
    for why this is not `tests/conftest.py`'s shared `FakeS3`."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.retentions: dict[str, dict] = {}

    def put_object(self, **kw) -> dict:
        self.objects[kw["Key"]] = kw["Body"]
        # I9787: the lock rides ON the PUT. No `put_object_retention` method
        # exists on this fake — a source path that still called it separately
        # would fail with an AttributeError, not silently no-op.
        if "ObjectLockMode" in kw or "ObjectLockRetainUntilDate" in kw:
            self.retentions[kw["Key"]] = {
                "Mode": kw.get("ObjectLockMode"),
                "RetainUntilDate": kw.get("ObjectLockRetainUntilDate"),
            }
        return {"ETag": '"fake"'}

    def head_object(self, **kw) -> dict:
        if kw["Key"] not in self.objects:
            from botocore.exceptions import ClientError

            raise ClientError({"Error": {"Code": "404"}}, "head_object")
        return {"ETag": '"fake"'}


SHA = "a" * 40
OTHER = "b" * 40
PRIOR = "c" * 40
WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"
DEPLOY_YML = WORKFLOWS / "deploy.yml"


WHEEL_BYTES = b"PK\x03\x04 a wheel"


def _release_json(sha=SHA, *, wheel: bytes = WHEEL_BYTES, wheel_sha256: str | None = None) -> str:
    """The DETERMINISTIC identity record (alpha-engine-config-I9786) that
    DESCRIBES its wheel.

    `wheel_sha256` is derived from the bytes rather than stubbed, because a
    fixture that pre-supplies a placeholder digest is a fixture in which the
    publish step's integrity check cannot fail — which is how the check came
    to be absent and nothing noticed. `wheel_sha256=` is here only so a test
    can deliberately break the correspondence.
    """
    return json.dumps(
        {
            "schema_version": "release.v2",
            "sha": sha,
            "lockfile_sha256": "0" * 64,
            "wheel_sha256": wheel_sha256 or sha256_hex(wheel),
            "python_requires": ">=3.12,<3.13",
            "extra": {},
        }
    )


def _provenance_json(sha=SHA, *, run_id: str = "1", run_attempt: str = "1") -> str:
    """This attempt's provenance — the three fields that moved out of
    `release.json` because they change on every rebuild of the same commit."""
    return json.dumps(
        {
            "schema_version": "release_provenance.v1",
            "sha": sha,
            "run_id": run_id,
            "run_attempt": run_attempt,
            "built_at": "2026-08-28T21:00:00Z",
            "workflow_run_url": f"https://github.com/nousergon/crucible/actions/runs/{run_id}",
            "test_summary": "42 passed",
        }
    )


def _token(tmp_path, store, name="pointer.token") -> str:
    """The pointer version token as `capture` writes it, i.e. read BEFORE the
    smoke. Every flip test goes through this, so a flip that stopped
    requiring it would fail here rather than pass quietly."""
    path = tmp_path / name
    path.write_bytes(read_pointer(store, POINTER_KEY)[1].encode("utf-8"))
    return str(path)


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
    def _publish(
        self,
        tmp_path,
        store_dir,
        *,
        sha=SHA,
        wheel=WHEEL_BYTES,
        release_json=None,
        provenance_json=None,
        run_id="1",
        suffix="",
    ):
        wheel_path = tmp_path / f"{sha[:6]}{suffix}.whl"
        wheel_path.write_bytes(wheel)
        meta = tmp_path / f"{sha[:6]}{suffix}-release.json"
        meta.write_text(
            release_json if release_json is not None else _release_json(sha, wheel=wheel)
        )
        prov = tmp_path / f"{sha[:6]}{suffix}-provenance.json"
        prov.write_text(
            provenance_json if provenance_json is not None else _provenance_json(sha, run_id=run_id)
        )
        return deploy_main(
            [
                "publish",
                "--sha",
                sha,
                "--store",
                str(store_dir),
                "--wheel",
                str(wheel_path),
                "--release-json",
                str(meta),
                "--provenance-json",
                str(prov),
            ]
        )

    def test_it_uploads_both_artifacts_and_promotes_nothing(self, tmp_path) -> None:
        store_dir = tmp_path / "store"
        self._publish(tmp_path, store_dir)
        store = LocalStore(store_dir)
        assert store.exists(wheel_key(SHA))
        assert current_release(store) is None

    def test_a_release_json_for_another_build_is_refused(self, tmp_path) -> None:
        """Publishing it under the wrong prefix would make the rollback target
        a build it does not describe."""
        with pytest.raises(SystemExit, match="not"):
            self._publish(tmp_path, tmp_path / "s", sha=SHA, release_json=_release_json(OTHER))

    def test_a_record_that_does_not_hash_its_own_wheel_is_refused(self, tmp_path) -> None:
        """`wheel_sha256` is the only statement anyone downstream has about
        what these bytes are. Until it is checked against the bytes it is a
        claim the publisher made about its own artifact, and every later
        verification against it is vacuous."""
        store_dir = tmp_path / "store"
        with pytest.raises(SystemExit, match="hashes to"):
            self._publish(
                tmp_path,
                store_dir,
                release_json=_release_json(SHA, wheel_sha256="0" * 64),
            )
        assert not LocalStore(store_dir).exists(wheel_key(SHA))

    def test_republishing_different_bytes_for_a_published_sha_is_refused(self, tmp_path) -> None:
        """§4.11 claims immutable versioned artifacts, and the rollback story
        rests on it: a `workflow_dispatch` re-run that rebuilt the wheel would
        overwrite the artifact a prior consumer installed."""
        store_dir = tmp_path / "store"
        self._publish(tmp_path, store_dir)
        original = LocalStore(store_dir).get_bytes(wheel_key(SHA))
        with pytest.raises(ReleaseImmutabilityError, match="already exists with different bytes"):
            self._publish(tmp_path, store_dir, wheel=b"PK\x03\x04 a DIFFERENT wheel")
        assert LocalStore(store_dir).get_bytes(wheel_key(SHA)) == original

    def test_republishing_identical_bytes_is_a_no_op_not_a_failure(self, tmp_path) -> None:
        """A re-run against the same artifact must stay idempotent: refusing
        it would make a retried deploy indistinguishable from a corrupted one."""
        store_dir = tmp_path / "store"
        self._publish(tmp_path, store_dir)
        assert self._publish(tmp_path, store_dir) == 0

    def test_a_workflow_dispatch_rerun_with_a_new_run_id_is_still_a_no_op(self, tmp_path) -> None:
        """alpha-engine-config-I9786, at the `deploy._publish` layer: a second
        `workflow_dispatch` for an unchanged commit gets a NEW `run_id` (and a
        new `built_at`, a new `workflow_run_url`) but the same identity
        record — that must stay a no-op, and both attempts must be
        reconstructible from their own provenance record."""
        store_dir = tmp_path / "store"
        assert self._publish(tmp_path, store_dir, run_id="33572214728") == 0
        assert self._publish(tmp_path, store_dir, run_id="33572299999") == 0
        store = LocalStore(store_dir)
        assert store.exists(provenance_key(SHA, "33572214728", "1"))
        assert store.exists(provenance_key(SHA, "33572299999", "1"))

    def test_publish_locks_the_wheel_and_release_json_on_s3_but_not_the_pointer(
        self, tmp_path
    ) -> None:
        """The `deploy._publish` half of I9782/I9787: the same Object Lock
        request `crucible.release.publish_release` makes, requested ON THE
        PUT itself, made here too, since this is the code path `deploy.yml`
        actually drives in CI."""
        wheel_path = tmp_path / "a.whl"
        wheel_path.write_bytes(WHEEL_BYTES)
        meta = tmp_path / "a-release.json"
        meta.write_text(_release_json(SHA, wheel=WHEEL_BYTES))
        prov = tmp_path / "a-provenance.json"
        prov.write_text(_provenance_json(SHA))
        client = _FakeS3Client()
        store = S3Store("bucket", "crucible", client=client)
        args = argparse.Namespace(
            sha=SHA, wheel=str(wheel_path), release_json=str(meta), provenance_json=str(prov)
        )
        assert _publish(args, store) == 0
        for key in (wheel_key(SHA), release_json_key(SHA)):
            s3_key = f"crucible/{key}"
            assert s3_key in client.retentions
            assert client.retentions[s3_key]["Mode"] == "GOVERNANCE"
        assert f"crucible/{POINTER_KEY}" not in client.retentions
        # The provenance object was written too, but never asked to be locked.
        assert f"crucible/{provenance_key(SHA, '1', '1')}" not in client.retentions

    def test_a_refused_overwrite_leaves_neither_key_half_written(self, tmp_path) -> None:
        """A wheel from one build beside a release.json from another is worse
        than either, so both keys are checked before either is written."""
        store_dir = tmp_path / "store"
        self._publish(tmp_path, store_dir)
        before = LocalStore(store_dir).get_bytes(release_json_key(SHA))
        with pytest.raises(ReleaseImmutabilityError):
            self._publish(tmp_path, store_dir, wheel=b"PK different")
        assert LocalStore(store_dir).get_bytes(release_json_key(SHA)) == before


class TestFlip:
    def _published(self, tmp_path, sha=SHA):
        store = LocalStore(tmp_path)
        publish_release(
            store,
            sha=sha,
            wheel=b"w" + sha[:1].encode(),
            lockfile=b"l",
            test_summary="",
            workflow_run_url="",
        )
        return store

    def test_an_ok_smoke_flips_the_pointer(self, tmp_path) -> None:
        store = self._published(tmp_path)
        _write_smoke(store)
        token = _token(tmp_path, store)
        assert (
            deploy_main(
                ["flip", "--sha", SHA, "--store", str(tmp_path), "--expect-pointer-file", token]
            )
            == 0
        )
        assert current_release(store) == SHA

    def test_a_failed_smoke_fails_the_deploy_and_leaves_the_pointer(self, tmp_path) -> None:
        """ "Nothing was promoted" and "nothing needed promoting" must not look
        the same on the surface."""
        store = self._published(tmp_path)
        _write_smoke(store, status="failed")
        token = _token(tmp_path, store)
        with pytest.raises(SystemExit, match="untouched"):
            deploy_main(
                ["flip", "--sha", SHA, "--store", str(tmp_path), "--expect-pointer-file", token]
            )
        assert current_release(store) is None

    def test_a_missing_smoke_manifest_refuses_rather_than_promoting(self, tmp_path) -> None:
        """Promoting without one would flip the pointer on a step that may
        never have executed."""
        store = self._published(tmp_path)
        token = _token(tmp_path, store)
        with pytest.raises(SystemExit, match="no smoke manifest"):
            deploy_main(
                ["flip", "--sha", SHA, "--store", str(tmp_path), "--expect-pointer-file", token]
            )

    def test_an_operator_rollback_during_the_smoke_fails_the_deploy(self, tmp_path) -> None:
        """The interval the compare-and-swap must cover is THE WHOLE SMOKE.

        `crucible release.pin` swaps against whatever is there right now,
        which is correct for an operator who has just looked — so an operator
        rolling back mid-deploy leaves a pointer this deploy never read. A
        flip that read its token immediately before the swap could not see
        that and silently undid the rollback; here the deploy FAILS and the
        rollback stands.
        """
        store = self._published(tmp_path)
        self._published(tmp_path, OTHER)
        self._published(tmp_path, PRIOR)
        pin(store, OTHER)  # the pointer as this deploy found it
        token = _token(tmp_path, store)  # captured BEFORE the smoke

        _write_smoke(store)  # the smoke for SHA runs and passes
        pin(store, PRIOR)  # ...and the operator rolls back mid-smoke, expect=None

        with pytest.raises(SystemExit, match="moved while the smoke"):
            deploy_main(
                ["flip", "--sha", SHA, "--store", str(tmp_path), "--expect-pointer-file", token]
            )
        assert current_release(store) == PRIOR, "the operator's rollback must stand"

    def test_a_flip_without_a_captured_token_is_refused(self, tmp_path) -> None:
        """No fallback to reading the pointer here: that IS the microsecond
        window. A workflow that lost its capture step must fail the deploy,
        not degrade to the unprotected swap with nothing saying so."""
        store = self._published(tmp_path)
        _write_smoke(store)
        with pytest.raises(SystemExit, match="no pointer token"):
            deploy_main(
                [
                    "flip",
                    "--sha",
                    SHA,
                    "--store",
                    str(tmp_path),
                    "--expect-pointer-file",
                    str(tmp_path / "never-written.token"),
                ]
            )
        assert current_release(store) is None


class TestCapture:
    def test_it_writes_the_token_the_flip_will_swap_against(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        publish_release(
            store, sha=SHA, wheel=b"w", lockfile=b"l", test_summary="", workflow_run_url=""
        )
        out = tmp_path / "t" / "pointer.token"
        assert (
            deploy_main(["capture", "--sha", SHA, "--store", str(tmp_path), "--out", str(out)]) == 0
        )
        assert out.read_bytes().decode("utf-8") == read_pointer(store, POINTER_KEY)[1]

    def test_the_unset_pointer_token_survives_the_round_trip(self, tmp_path) -> None:
        """ETAG_ABSENT contains a NUL byte. A token that had to survive shell
        quoting would one day arrive mangled and compare unequal to
        everything, failing a deploy that should have passed."""
        from crucible.store import ETAG_ABSENT

        store = LocalStore(tmp_path)
        publish_release(
            store, sha=SHA, wheel=b"w", lockfile=b"l", test_summary="", workflow_run_url=""
        )
        out = tmp_path / "pointer.token"
        deploy_main(["capture", "--sha", SHA, "--store", str(tmp_path), "--out", str(out)])
        assert out.read_bytes().decode("utf-8") == ETAG_ABSENT
        _write_smoke(store)
        deploy_main(
            ["flip", "--sha", SHA, "--store", str(tmp_path), "--expect-pointer-file", str(out)]
        )
        assert current_release(store) == SHA


class TestRecord:
    def test_the_deploy_manifest_validates_against_the_run_schema(self, tmp_path) -> None:
        """§4.11: the deploy writes its own manifest in the run-manifest
        schema, so §4.5's page shows deploys beside runs."""
        store = LocalStore(tmp_path)
        publish_release(
            store, sha=SHA, wheel=b"w", lockfile=b"l", test_summary="", workflow_run_url=""
        )
        _write_smoke(store)
        token = tmp_path / "pointer.token"
        deploy_main(["capture", "--sha", SHA, "--store", str(tmp_path), "--out", str(token)])
        deploy_main(
            ["flip", "--sha", SHA, "--store", str(tmp_path), "--expect-pointer-file", str(token)]
        )
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

    def _release_steps(self, workflow: dict) -> list[dict]:
        return workflow["jobs"]["release"]["steps"]

    def test_the_identity_check_reads_it_back_from_aws(self, workflow: dict) -> None:
        """A guard must be falsifiable by something other than editing itself.

        The predicate this replaced compared ${DEPLOY_ROLE_ARN} against the
        literal `env` sets it eleven lines above, in the same file, with no
        input, override or dispatch path that could make them differ — so it
        could only be false if someone edited one of two adjacent lines, and
        the "silent fallback" it claimed to prevent was not something this
        workflow could do (`role-to-assume` is that same variable). What the
        assumed identity actually is IS falsifiable, and only AWS can say.
        """
        steps = self._release_steps(workflow)
        check = [s for s in steps if "get-caller-identity" in json.dumps(s)]
        assert check, (
            "the role guard must read the identity back from AWS; a comparison of an "
            "env var against the literal that sets it checks nothing"
        )
        body = json.dumps(check[0])
        assert "crucible-v2-github-deploy" in body
        assert "github-actions-lambda-deploy" in body, (
            "the check must name the role it is refusing to be, or the message does "
            "not tell an operator what nearly happened"
        )
        assert "crucible-v2.yaml" in body, (
            "and it must name the stack that creates the role, because 'the v2 stack "
            "is not applied yet' is the failure an operator will actually hit"
        )

    def test_no_step_asserts_an_env_var_against_the_literal_that_sets_it(
        self, workflow: dict
    ) -> None:
        """The tautology, refused by name so it cannot come back.

        `DEPLOY_ROLE_ARN` is declared in exactly one place. Any `run:` block
        comparing `${DEPLOY_ROLE_ARN}` to that same literal is restating it,
        and burns a runner to do so.
        """
        arn = workflow["env"]["DEPLOY_ROLE_ARN"]
        for name, job in workflow["jobs"].items():
            for step in job.get("steps", []):
                script = step.get("run") or ""
                if "DEPLOY_ROLE_ARN" in script and arn in script:
                    raise AssertionError(
                        f"{name}/{step.get('name')} compares ${{DEPLOY_ROLE_ARN}} against "
                        f"{arn}, the literal that sets it eleven lines above. That "
                        "predicate can only be false if someone edits one of two "
                        "adjacent lines."
                    )

    def test_the_identity_is_confirmed_before_anything_is_written(self, workflow: dict) -> None:
        """A wrong identity must fail the deploy having touched no bucket."""
        names = [json.dumps(s) for s in self._release_steps(workflow)]
        confirm = next(i for i, s in enumerate(names) if "get-caller-identity" in s)
        credentials = next(i for i, s in enumerate(names) if "configure-aws-credentials" in s)
        publish = next(i for i, s in enumerate(names) if "crucible.deploy publish" in s)
        assert credentials < confirm < publish

    def test_the_pointer_token_is_captured_before_the_smoke(self, workflow: dict) -> None:
        """C6: the compare-and-swap is only worth the name over the interval
        it covers, and that interval is the whole smoke — the minutes-long
        window in which an operator rollback can land."""
        names = [json.dumps(s) for s in self._release_steps(workflow)]
        capture = next(i for i, s in enumerate(names) if "crucible.deploy capture" in s)
        smoke = next(i for i, s in enumerate(names) if "crucible smoke" in s)
        flip = next(i for i, s in enumerate(names) if "crucible.deploy flip" in s)
        assert capture < smoke < flip, (
            "capturing the token after the smoke leaves a race window microseconds "
            "wide, which is the one race the concurrency group already serialises away"
        )
        assert "--expect-pointer-file" in names[flip], (
            "the flip must swap against the captured token, not one it reads itself"
        )

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


class TestTheSmokeGate:
    """The smoke is what the pointer flip is gated on, so it is tested here
    beside the flip rather than beside the other jobs.

    §4.11's own stated failure mode: "a gate that only proves the process
    started would promote a build that cannot reach its own data". Every test
    in this class is a state in which the smoke used to report `ok` with
    `smoke_ok` value 1.0 and an empty `inputs[]`.
    """

    def _args(self, tmp_path, sha=SHA):
        import argparse
        import datetime as dt

        return argparse.Namespace(
            store=str(tmp_path), release=sha, trading_day=dt.date(2026, 8, 28)
        )

    def _run(self, tmp_path, sha=SHA):
        from crucible.track_c import smoke_handler

        return smoke_handler(self._args(tmp_path, sha))

    def _manifest(self, tmp_path):
        return json.loads(LocalStore(tmp_path).get_bytes(manifest_key("smoke", "2026-08-28")))

    def _metric(self, manifest, name="smoke_ok"):
        return next(m for m in manifest["metrics"] if m["name"] == name)

    def test_an_empty_store_fails_the_smoke(self, tmp_path) -> None:
        """The reproduction: an empty store, no wheel for the sha, no
        releases/current — and the smoke exited 0, wrote `status: ok`, an
        empty `inputs[]` and `smoke_ok` value 1.0. deploy.yml then flipped
        the pointer on that manifest."""
        with pytest.raises(FileNotFoundError, match="not published"):
            self._run(tmp_path)
        assert self._manifest(tmp_path)["status"] == "failed"

    def test_a_sha_with_no_wheel_fails_even_when_other_releases_exist(self, tmp_path) -> None:
        """The gate is about the build being promoted, not about the store
        being non-empty."""
        store = LocalStore(tmp_path)
        publish_release(
            store, sha=OTHER, wheel=b"w", lockfile=b"l", test_summary="", workflow_run_url=""
        )
        with pytest.raises(FileNotFoundError, match=SHA):
            self._run(tmp_path)

    def test_a_wheel_whose_bytes_do_not_match_the_record_fails(self, tmp_path) -> None:
        """A truncated or clobbered upload. Nothing downstream re-hashes, so
        the first symptom would be an install failure on a box at 06:30."""
        store = LocalStore(tmp_path)
        publish_release(
            store,
            sha=SHA,
            wheel=b"the tested wheel",
            lockfile=b"l",
            test_summary="",
            workflow_run_url="",
        )
        store.put_bytes(wheel_key(SHA), b"something else entirely")
        with pytest.raises(ValueError, match="hashes to"):
            self._run(tmp_path)

    def test_a_release_json_describing_another_build_fails(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        publish_release(
            store, sha=SHA, wheel=b"w", lockfile=b"l", test_summary="", workflow_run_url=""
        )
        publish_release(
            store, sha=OTHER, wheel=b"w2", lockfile=b"l", test_summary="", workflow_run_url=""
        )
        store.put_bytes(release_json_key(SHA), store.get_bytes(release_json_key(OTHER)))
        with pytest.raises(ValueError, match="describes"):
            self._run(tmp_path)

    def test_a_passing_smoke_records_the_artifacts_it_actually_read(self, tmp_path) -> None:
        """`inputs: []` on a gate is the signature of a gate that read
        nothing."""
        store = LocalStore(tmp_path)
        publish_release(
            store, sha=SHA, wheel=b"w", lockfile=b"l", test_summary="", workflow_run_url=""
        )
        assert self._run(tmp_path) == 0
        manifest = self._manifest(tmp_path)
        assert manifest["status"] == "ok"
        keys = {i["key"] for i in manifest["inputs"]}
        assert wheel_key(SHA) in keys and release_json_key(SHA) in keys

    def test_smoke_ok_is_counted_not_asserted(self, tmp_path) -> None:
        """It was the literal string "OK" and the literal 1.0 on every run it
        survived, which is indistinguishable from a metric that did nothing.

        Here the same passing smoke reports BREACH because a live read found
        a real defect — `releases/current` naming a sha whose wheel is gone —
        which is a fact about the system a constant cannot express.
        """
        store = LocalStore(tmp_path)
        publish_release(
            store, sha=SHA, wheel=b"w", lockfile=b"l", test_summary="", workflow_run_url=""
        )
        publish_release(
            store, sha=OTHER, wheel=b"w2", lockfile=b"l", test_summary="", workflow_run_url=""
        )
        pin(store, OTHER)
        (tmp_path / wheel_key(OTHER)).unlink()

        assert self._run(tmp_path) == 0  # the flip is the REMEDY; it must still deploy
        metric = self._metric(self._manifest(tmp_path))
        assert metric["status"] == "BREACH"
        assert OTHER in metric["status_reason"]

    def test_smoke_ok_value_moves_with_what_was_read(self, tmp_path) -> None:
        """A value that is the same number on a bootstrap store and on a
        populated one is not a measurement."""
        store = LocalStore(tmp_path)
        publish_release(
            store, sha=SHA, wheel=b"w", lockfile=b"l", test_summary="", workflow_run_url=""
        )
        self._run(tmp_path)
        bootstrap = self._metric(self._manifest(tmp_path))["value"]

        pin(store, SHA)  # now the pointer is set and runs/ is non-empty
        self._run(tmp_path)
        populated = self._metric(self._manifest(tmp_path))["value"]
        assert populated > bootstrap, (bootstrap, populated)

    def test_no_read_path_is_declared_required_and_then_never_enforced(self) -> None:
        """`SMOKE_READS` carried a `required` column that was False on every
        row: a column with one value, reading as if some read somewhere could
        fail the smoke while none could."""
        from crucible.track_c import SMOKE_READS

        for entry in SMOKE_READS:
            assert len(entry) == 2, (
                f"{entry} still carries a required flag. What gates the flip is the "
                "release verification, which raises; these rows are observations."
            )
