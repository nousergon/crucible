"""The hash-locked, per-release wheelhouse — producer, gate and workflow.

Normative source: plan §4.11; `alpha-engine-config-I10812`.

A dispatched box used to resolve every dependency from PyPI at boot. A PyPI
read timeout killed a heal chunk before its job started, and every box ran a
dependency set the deploy smoke never graded. These tests hold the fix at each
surface that could quietly reintroduce either defect: the lock/manifest
agreement, the release.v4 record, the publisher, the smoke gate, the flip, the
run manifest, the lock sweep, and `deploy.yml` itself.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

from crucible.deploy import main as deploy_main
from crucible.manifest import manifest_key
from crucible.release import (
    RELEASE_SCHEMA_VERSION,
    ReleaseHasNoWheelhouseError,
    ReleaseRecord,
    current_release,
    publish_release,
    read_pointer,
    release_json_key,
    require_wheelhouse,
    wheel_filename_for,
    wheelhouse_object_key,
)
from crucible.release_lock_sweep import _RELEASE_OBJECT_RE
from crucible.runner import WHEELHOUSE_DIGEST_ENV, run_job
from crucible.store import LocalStore, sha256_hex
from crucible.wheelhouse import (
    LOCK_FILENAME,
    PIP_OFFLINE_FLAGS,
    WHEELHOUSE_DIRNAME,
    WheelhouseLockMismatchError,
    build_manifest,
    parse_lock,
    verify_against_lock,
    wheelhouse_digest,
)
from tests.support.releases import (
    SYNTHETIC_DIGEST,
    SYNTHETIC_WHEELS,
    WHEEL_BYTES,
    publish_v4,
    release_json_v4,
    synthetic_lock,
    synthetic_manifest,
    write_wheelhouse,
)

SHA = "a" * 40
ROOT = Path(__file__).resolve().parents[1]
DEPLOY_YML = ROOT / ".github" / "workflows" / "deploy.yml"
SCHEMAS = ROOT / "crucible" / "schemas"
TRADING_DAY = dt.date(2026, 8, 28)


def _wheels() -> list[dict[str, str]]:
    return [{"filename": n, "sha256": sha256_hex(b)} for n, b in sorted(SYNTHETIC_WHEELS.items())]


class TestTheLockAndTheWheelsAgree:
    def test_a_real_uv_export_shape_parses(self) -> None:
        lock = parse_lock(
            "annotated-types==0.8.0 \\\n"
            f"    --hash=sha256:{'1' * 64} \\\n"
            f"    --hash=sha256:{'2' * 64}\n"
            "    # via pydantic\n"
            "h11==0.16.0 ; sys_platform != 'emscripten' \\\n"
            f"    --hash=sha256:{'3' * 64}\n"
            "    # via\n"
            "    #   httpcore2\n"
        )
        assert lock["annotated-types"].hashes == {"1" * 64, "2" * 64}
        assert lock["h11"].marker == "sys_platform != 'emscripten'"

    def test_the_synthetic_wheelhouse_satisfies_its_lock(self) -> None:
        verify_against_lock(_wheels(), synthetic_lock())

    def test_a_wheel_the_lock_does_not_vouch_for_is_refused(self) -> None:
        """The box's `--require-hashes` would refuse it at boot; refusing it
        here fails the deploy instead of the 06:30 box."""
        wheels = _wheels()
        wheels[0] = {**wheels[0], "sha256": "f" * 64}
        with pytest.raises(WheelhouseLockMismatchError, match="not one of"):
            verify_against_lock(wheels, synthetic_lock())

    def test_an_unmarked_requirement_with_no_wheel_is_refused(self) -> None:
        with pytest.raises(WheelhouseLockMismatchError, match="no wheel carries it"):
            verify_against_lock(_wheels()[:1], synthetic_lock())

    def test_a_marker_excluded_requirement_may_be_absent(self) -> None:
        assert "jsfetch-only" in parse_lock(synthetic_lock())
        verify_against_lock(_wheels(), synthetic_lock())  # jsfetch-only has no wheel

    def test_a_wheel_of_another_version_is_refused(self) -> None:
        wheels = _wheels() + [{"filename": "pure_dep-9.9-py3-none-any.whl", "sha256": "a" * 64}]
        with pytest.raises(WheelhouseLockMismatchError, match="two wheels"):
            verify_against_lock(wheels, synthetic_lock())

    def test_a_lock_line_that_is_not_a_hashed_pin_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not a `name==version"):
            parse_lock("somepkg>=1.0 \\\n    --hash=sha256:" + "1" * 64 + "\n")

    def test_the_digest_is_reproducible_with_sha256sum_lines(self) -> None:
        lock_sha = "9" * 64
        text = (
            "".join(
                f"{w['sha256']}  {w['filename']}\n"
                for w in sorted(_wheels(), key=lambda w: w["filename"])
            )
            + f"{lock_sha}  {LOCK_FILENAME}\n"
        )
        assert (
            wheelhouse_digest(lock_sha, reversed(_wheels()))
            == hashlib.sha256(text.encode()).hexdigest()
        )

    def test_build_manifest_refuses_a_stray_file(self, tmp_path) -> None:
        write_wheelhouse(tmp_path)
        (tmp_path / "notes.txt").write_text("x")
        with pytest.raises(ValueError, match="neither wheels nor the lock"):
            build_manifest(tmp_path, extras=["arcticdb"])


class TestTheReleaseV4Contract:
    """Producer half of the cross-repo contract the box bootstrap consumes
    (`nous-ergon-ops/tests/crossrepo/test_crucible_box_wheelhouse_lockstep.py`)."""

    def _record(self, **overrides) -> ReleaseRecord:
        fields = {
            "schema_version": RELEASE_SCHEMA_VERSION,
            "sha": SHA,
            "lockfile_sha256": "0" * 64,
            "wheel_sha256": "1" * 64,
            "wheel_filename": wheel_filename_for(SHA),
            "wheelhouse": synthetic_manifest(),
        }
        fields.update(overrides)
        return ReleaseRecord(**fields)

    def test_a_built_record_validates_against_the_committed_v4_schema(self) -> None:
        schema = json.loads((SCHEMAS / "release.v4.json").read_text(encoding="utf-8"))
        Draft202012Validator(schema).validate(json.loads(self._record().to_json()))

    def test_the_schema_pins_the_fields_the_box_reads(self) -> None:
        schema = json.loads((SCHEMAS / "release.v4.json").read_text(encoding="utf-8"))
        assert "wheelhouse" in schema["required"]
        wheelhouse = schema["$defs"]["WheelhouseManifestDocument"]
        assert set(wheelhouse["required"]) >= {
            "lock_filename",
            "lock_sha256",
            "extras",
            "wheels",
            "digest",
        }
        assert wheelhouse["properties"]["lock_filename"]["const"] == LOCK_FILENAME
        assert set(schema["$defs"]["WheelhouseWheel"]["required"]) == {"filename", "sha256"}

    def test_a_digest_that_does_not_describe_its_manifest_is_refused(self) -> None:
        with pytest.raises(ValueError, match="digest"):
            self._record(wheelhouse={**synthetic_manifest(), "digest": "0" * 64})

    def test_unsorted_wheels_are_refused(self) -> None:
        manifest = synthetic_manifest()
        manifest["wheels"] = list(reversed(manifest["wheels"]))
        with pytest.raises(ValueError, match="sorted"):
            self._record(wheelhouse=manifest)

    def test_a_wheel_filename_with_a_separator_is_refused(self) -> None:
        """The box interpolates it into a local path."""
        manifest = synthetic_manifest()
        manifest["wheels"][0] = {**manifest["wheels"][0], "filename": "../x-1-py3-none-any.whl"}
        with pytest.raises(ValueError, match="does not conform"):
            self._record(wheelhouse=manifest)

    def test_a_v3_release_is_readable_and_refused_by_every_installer(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        record = publish_release(
            store, sha=SHA, wheel=b"w", lockfile=b"l", test_summary="", workflow_run_url=""
        )
        assert record.schema_version == "release.v3"
        with pytest.raises(ReleaseHasNoWheelhouseError, match="publishes no wheelhouse"):
            require_wheelhouse(record)

    def test_the_wheelhouse_keys_live_one_level_under_the_release(self) -> None:
        record = self._record()
        keys = [k for k, _ in record.wheelhouse_keys]
        assert keys[-1] == f"releases/{SHA}/{WHEELHOUSE_DIRNAME}/{LOCK_FILENAME}"
        assert all(k.startswith(f"releases/{SHA}/{WHEELHOUSE_DIRNAME}/") for k in keys)
        with pytest.raises(ValueError):
            wheelhouse_object_key(SHA, "nested/x.whl")


class TestThePublisherShipsTheWheelhouse:
    def _publish(self, tmp_path, *, wheelhouse_dir, release_json):
        wheel = tmp_path / wheel_filename_for(SHA)
        wheel.write_bytes(WHEEL_BYTES)
        meta = tmp_path / "release.json"
        meta.write_text(release_json)
        prov = tmp_path / "provenance.json"
        prov.write_text(
            json.dumps(
                {
                    "schema_version": "release_provenance.v1",
                    "sha": SHA,
                    "run_id": "1",
                    "run_attempt": "1",
                    "built_at": "2026-09-14T00:00:00Z",
                    "workflow_run_url": "https://example/runs/1",
                    "test_summary": "1 passed",
                }
            )
        )
        return deploy_main(
            [
                "publish",
                "--sha",
                SHA,
                "--store",
                str(tmp_path / "store"),
                "--wheel",
                str(wheel),
                "--release-json",
                str(meta),
                "--provenance-json",
                str(prov),
                "--wheelhouse",
                str(wheelhouse_dir),
            ]
        )

    def test_every_wheelhouse_object_is_published_before_release_json(self, tmp_path) -> None:
        manifest = publish_v4(tmp_path, SHA)
        store = LocalStore(tmp_path)
        for filename in [w["filename"] for w in manifest["wheels"]] + [LOCK_FILENAME]:
            assert store.exists(wheelhouse_object_key(SHA, filename))
        record = json.loads(store.get_bytes(release_json_key(SHA)))
        assert record["schema_version"] == "release.v4"
        assert record["wheelhouse"]["digest"] == SYNTHETIC_DIGEST

    def test_a_wheel_whose_bytes_differ_from_the_record_is_refused(self, tmp_path) -> None:
        wheelhouse = tmp_path / "wh"
        manifest = write_wheelhouse(wheelhouse)
        (wheelhouse / manifest["wheels"][0]["filename"]).write_bytes(b"tampered")
        with pytest.raises(SystemExit, match="hashes to"):
            self._publish(
                tmp_path,
                wheelhouse_dir=wheelhouse,
                release_json=release_json_v4(SHA, wheelhouse=manifest),
            )
        assert not LocalStore(tmp_path / "store").exists(release_json_key(SHA))

    def test_a_wheel_the_record_does_not_name_is_refused(self, tmp_path) -> None:
        wheelhouse = tmp_path / "wh"
        manifest = write_wheelhouse(wheelhouse)
        (wheelhouse / "extra_dep-1.0-py3-none-any.whl").write_bytes(b"unvouched")
        with pytest.raises(SystemExit, match="does not name"):
            self._publish(
                tmp_path,
                wheelhouse_dir=wheelhouse,
                release_json=release_json_v4(SHA, wheelhouse=manifest),
            )

    def test_a_lock_the_wheels_do_not_satisfy_is_refused(self, tmp_path) -> None:
        """The lock/manifest hash-set agreement, at the publisher."""
        wheelhouse = tmp_path / "wh"
        manifest = write_wheelhouse(wheelhouse)
        bad_lock = synthetic_lock().replace(sha256_hex(b"PK\x03\x04 pure_dep"), "d" * 64)
        (wheelhouse / LOCK_FILENAME).write_text(bad_lock)
        manifest = {**manifest, "lock_sha256": sha256_hex(bad_lock.encode())}
        manifest["digest"] = wheelhouse_digest(manifest["lock_sha256"], manifest["wheels"])
        with pytest.raises(SystemExit, match="does not match its lock"):
            self._publish(
                tmp_path,
                wheelhouse_dir=wheelhouse,
                release_json=release_json_v4(SHA, wheelhouse=manifest),
            )

    def test_a_release_json_without_a_wheelhouse_is_never_published(self, tmp_path) -> None:
        wheelhouse = tmp_path / "wh"
        write_wheelhouse(wheelhouse)
        v3 = json.loads(release_json_v4(SHA))
        del v3["wheelhouse"]
        v3["schema_version"] = "release.v3"
        with pytest.raises(SystemExit, match="release.v4"):
            self._publish(tmp_path, wheelhouse_dir=wheelhouse, release_json=json.dumps(v3))
        assert not LocalStore(tmp_path / "store").exists(release_json_key(SHA))


class TestTheSmokeGradesTheWheelhouse:
    def _run(self, tmp_path):
        import argparse

        from crucible.track_c import smoke_handler

        return smoke_handler(
            argparse.Namespace(store=str(tmp_path), release=SHA, trading_day=TRADING_DAY)
        )

    def _manifest(self, tmp_path):
        return json.loads(LocalStore(tmp_path).get_bytes(manifest_key("smoke", "2026-08-28")))

    def test_a_smoke_from_the_published_wheelhouse_passes_and_names_it(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setenv(WHEELHOUSE_DIGEST_ENV, SYNTHETIC_DIGEST)
        publish_v4(tmp_path, SHA)
        assert self._run(tmp_path) == 0
        manifest = self._manifest(tmp_path)
        assert manifest["status"] == "ok"
        assert manifest["wheelhouse_digest"] == SYNTHETIC_DIGEST
        assert wheelhouse_object_key(SHA, LOCK_FILENAME) in {i["key"] for i in manifest["inputs"]}

    def test_a_release_without_a_wheelhouse_fails_the_smoke(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv(WHEELHOUSE_DIGEST_ENV, SYNTHETIC_DIGEST)
        publish_release(
            LocalStore(tmp_path),
            sha=SHA,
            wheel=b"w",
            lockfile=b"l",
            test_summary="",
            workflow_run_url="",
        )
        with pytest.raises(ValueError, match="publishes no wheelhouse"):
            self._run(tmp_path)
        assert self._manifest(tmp_path)["status"] == "failed"

    def test_a_missing_wheel_in_the_store_fails_the_smoke(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv(WHEELHOUSE_DIGEST_ENV, SYNTHETIC_DIGEST)
        manifest = publish_v4(tmp_path, SHA)
        (tmp_path / wheelhouse_object_key(SHA, manifest["wheels"][0]["filename"])).unlink()
        with pytest.raises(FileNotFoundError, match="incomplete"):
            self._run(tmp_path)

    def test_a_tampered_lock_in_the_store_fails_the_smoke(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv(WHEELHOUSE_DIGEST_ENV, SYNTHETIC_DIGEST)
        publish_v4(tmp_path, SHA)
        LocalStore(tmp_path).put_bytes(
            wheelhouse_object_key(SHA, LOCK_FILENAME), b"x==1 --hash=sha256:" + b"1" * 64
        )
        with pytest.raises(ValueError, match="never vouched for"):
            self._run(tmp_path)

    @pytest.mark.parametrize("installed", ["", "e" * 64])
    def test_a_smoke_not_run_from_that_wheelhouse_fails(
        self, tmp_path, monkeypatch, installed
    ) -> None:
        """The reproducibility half of the defect: a smoke graded against any
        other dependency set proves nothing about what a box installs."""
        if installed:
            monkeypatch.setenv(WHEELHOUSE_DIGEST_ENV, installed)
        else:
            monkeypatch.delenv(WHEELHOUSE_DIGEST_ENV, raising=False)
        publish_v4(tmp_path, SHA)
        with pytest.raises(ValueError, match="dependency set no box installs"):
            self._run(tmp_path)


class TestTheFlipRefusesAnotherDependencySet:
    def _flip(self, tmp_path, store, digest):
        from tests.test_deploy import _write_smoke

        _write_smoke(store, wheelhouse_digest=digest)
        token = tmp_path / "pointer.token"
        token.write_bytes(read_pointer(store)[1].encode("utf-8"))
        return deploy_main(
            ["flip", "--sha", SHA, "--store", str(tmp_path), "--expect-pointer-file", str(token)]
        )

    @pytest.mark.parametrize("digest", [None, "e" * 64])
    def test_a_smoke_that_names_another_wheelhouse_cannot_flip(self, tmp_path, digest) -> None:
        store = LocalStore(tmp_path)
        publish_v4(tmp_path, SHA)
        with pytest.raises(SystemExit, match="wheelhouse_digest"):
            self._flip(tmp_path, store, digest)
        assert current_release(store) is None

    def test_a_v3_release_cannot_flip_even_on_an_ok_smoke(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        publish_release(
            store, sha=SHA, wheel=b"w", lockfile=b"l", test_summary="", workflow_run_url=""
        )
        with pytest.raises(SystemExit, match="publishes no wheelhouse"):
            self._flip(tmp_path, store, SYNTHETIC_DIGEST)
        assert current_release(store) is None


class TestTheRunManifestNamesItsDependencySet:
    def test_the_digest_the_box_exports_is_stamped(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv(WHEELHOUSE_DIGEST_ENV, SYNTHETIC_DIGEST)
        store = LocalStore(tmp_path)
        run_job("smoke", lambda ctx: None, store=store, trading_day=TRADING_DAY)
        manifest = json.loads(store.get_bytes(manifest_key("smoke", "2026-08-28")))
        assert manifest["wheelhouse_digest"] == SYNTHETIC_DIGEST

    def test_a_run_installed_from_no_wheelhouse_names_none(self, tmp_path, monkeypatch) -> None:
        monkeypatch.delenv(WHEELHOUSE_DIGEST_ENV, raising=False)
        store = LocalStore(tmp_path)
        run_job("smoke", lambda ctx: None, store=store, trading_day=TRADING_DAY)
        manifest = json.loads(store.get_bytes(manifest_key("smoke", "2026-08-28")))
        assert "wheelhouse_digest" not in manifest


class TestTheLockSweepGradesTheWheelhouse:
    def test_wheelhouse_objects_are_release_identity_objects(self) -> None:
        assert _RELEASE_OBJECT_RE.match(f"releases/{SHA}/wheelhouse/numpy-2-cp312-cp312-x.whl")
        assert _RELEASE_OBJECT_RE.match(f"releases/{SHA}/wheelhouse/{LOCK_FILENAME}")
        assert not _RELEASE_OBJECT_RE.match(f"releases/{SHA}/wheelhouse/nested/x.whl")


class TestTheWorkflowInstallsOffline:
    """`deploy.yml` is where both halves are proven before the pointer moves.
    Each assertion here is a line whose removal reintroduces the defect."""

    @pytest.fixture(scope="class")
    def workflow(self) -> dict:
        return yaml.safe_load(DEPLOY_YML.read_text(encoding="utf-8"))

    def _step(self, workflow, job, needle):
        return next(s for s in workflow["jobs"][job]["steps"] if needle in s.get("run", ""))

    def test_every_pip_install_in_the_release_job_is_offline_and_hash_checked(
        self, workflow
    ) -> None:
        lines = [
            line
            for step in workflow["jobs"]["release"]["steps"]
            for line in step.get("run", "").splitlines()
            if "pip install" in line
        ]
        assert len(lines) >= 2, "the container proof AND the smoke venv must both install"
        for line in lines:
            for flag in (*PIP_OFFLINE_FLAGS, "--find-links"):
                assert flag in line, f"{flag} missing from: {line.strip()}"

    def test_the_proof_cuts_the_network_before_pip_runs(self, workflow) -> None:
        script = self._step(workflow, "release", "pip install")["run"]
        assert script.index("docker network disconnect") < script.index(
            "/opt/crucible/bin/pip install"
        )
        assert "import crucible, nousergon_lib.arcticdb, arcticdb, lightgbm" in script

    def test_the_wheelhouse_is_built_for_the_box_platform_from_the_lock(self, workflow) -> None:
        step = self._step(workflow, "build", "crucible.wheelhouse")
        script = step["run"]
        assert "uv export --frozen --no-dev" in script and "--all-extras" in script
        assert "--require-hashes" in script and "--only-binary=:all:" in script
        assert step["env"]["BOX_IMAGE"].startswith("amazonlinux:2023@sha256:")

    def test_the_smoke_runs_from_the_offline_install_and_names_its_digest(self, workflow) -> None:
        steps = workflow["jobs"]["release"]["steps"]
        proof = next(i for i, s in enumerate(steps) if "pip install" in s.get("run", ""))
        smoke = next(i for i, s in enumerate(steps) if "crucible smoke" in s.get("run", ""))
        assert proof < smoke
        assert "CRUCIBLE_WHEELHOUSE_DIGEST=" in steps[proof]["run"]
        assert (
            steps[smoke]["run"].strip().startswith("/tmp/crucible-release-venv/bin/crucible smoke")
        )

    def test_the_publisher_is_handed_the_wheelhouse(self, workflow) -> None:
        assert (
            "--wheelhouse out/wheelhouse"
            in self._step(workflow, "release", "crucible.deploy publish")["run"]
        )
