"""A release.v4 release laid down the way `deploy.yml` lays one down.

alpha-engine-config-I10812 made the wheelhouse part of what a release IS: the
smoke refuses a release without one and the flip refuses a smoke that did not
run from it. Tests that need a publishable release therefore need a real
wheelhouse — a lock whose hashes vouch for the wheels beside it — and they get
one here, published through `python -m crucible.deploy publish`, the code path
the workflow drives. Nothing here fabricates a record by hand: the manifest
comes from `crucible.wheelhouse.build_manifest` over bytes on disk.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from crucible.release import wheel_filename_for
from crucible.store import sha256_hex
from crucible.wheelhouse import LOCK_FILENAME, build_manifest

#: The synthetic dependency set: one project with a platform-specific-looking
#: tag, one pure wheel, and one marker-excluded requirement with no wheel —
#: the three shapes `verify_against_lock` distinguishes.
SYNTHETIC_WHEELS: dict[str, bytes] = {
    "demo_dep-1.0.0-cp312-cp312-manylinux_2_28_x86_64.whl": b"PK\x03\x04 demo_dep",
    "pure_dep-2.1-py3-none-any.whl": b"PK\x03\x04 pure_dep",
}
SYNTHETIC_EXTRAS = ("arcticdb",)
WHEEL_BYTES = b"PK\x03\x04 a wheel"


def synthetic_lock(wheels: dict[str, bytes] = SYNTHETIC_WHEELS) -> str:
    """A `uv export`-shaped lock pinning ``wheels`` by hash, plus one
    marker-excluded requirement and a second allowed hash per pin (the lock
    lists every platform's hash; a wheelhouse carries one)."""
    lines = ["# via uv export, synthetic"]
    for filename, payload in sorted(wheels.items()):
        name, version = filename.split("-")[:2]
        lines.append(f"{name.replace('_', '-')}=={version} \\")
        lines.append(
            f"    --hash=sha256:{hashlib.sha256(b'other platform ' + payload).hexdigest()} \\"
        )
        lines.append(f"    --hash=sha256:{sha256_hex(payload)}")
        lines.append("    # via crucible")
    lines.append("jsfetch-only==1.0 ; sys_platform == 'emscripten' \\")
    lines.append(f"    --hash=sha256:{'e' * 64}")
    return "\n".join(lines) + "\n"


def write_wheelhouse(
    directory: Path,
    *,
    wheels: dict[str, bytes] = SYNTHETIC_WHEELS,
    lock: str | None = None,
) -> dict:
    """Write a wheelhouse into ``directory`` and return its manifest."""
    directory.mkdir(parents=True, exist_ok=True)
    for filename, payload in wheels.items():
        (directory / filename).write_bytes(payload)
    (directory / LOCK_FILENAME).write_text(lock if lock is not None else synthetic_lock(wheels))
    return build_manifest(directory, extras=SYNTHETIC_EXTRAS)


def synthetic_manifest() -> dict:
    with tempfile.TemporaryDirectory() as work:
        return write_wheelhouse(Path(work))


#: The digest every synthetic release carries — what `deploy.yml` exports as
#: `$CRUCIBLE_WHEELHOUSE_DIGEST` for the smoke, and what a smoke manifest
#: records.
SYNTHETIC_DIGEST: str = synthetic_manifest()["digest"]


def release_json_v4(
    sha: str,
    *,
    wheel: bytes = WHEEL_BYTES,
    wheel_sha256: str | None = None,
    wheelhouse: dict | None = None,
) -> str:
    return json.dumps(
        {
            "schema_version": "release.v4",
            "sha": sha,
            "lockfile_sha256": "0" * 64,
            "wheel_sha256": wheel_sha256 or sha256_hex(wheel),
            "wheel_filename": wheel_filename_for(sha),
            "python_requires": ">=3.12,<3.13",
            "extra": {},
            "wheelhouse": wheelhouse if wheelhouse is not None else synthetic_manifest(),
        }
    )


def publish_v4(store_root, sha: str, *, wheel: bytes = WHEEL_BYTES, run_id: str = "1") -> dict:
    """Publish a release.v4 release for ``sha`` into the LocalStore at
    ``store_root`` through `crucible.deploy publish`. Returns the manifest."""
    from crucible.deploy import main as deploy_main  # noqa: PLC0415 - avoid import at collection
    from crucible.store import LocalStore  # noqa: PLC0415 - same reason as above

    # A LocalStore carries its root; a Path or str IS the root. Never
    # `getattr(x, "root", x)`: a PosixPath has `.root == "/"`.
    root = store_root.root if isinstance(store_root, LocalStore) else Path(store_root)
    with tempfile.TemporaryDirectory() as work:
        work_dir = Path(work)
        manifest = write_wheelhouse(work_dir / "wheelhouse")
        wheel_path = work_dir / wheel_filename_for(sha)
        wheel_path.write_bytes(wheel)
        meta = work_dir / "release.json"
        meta.write_text(release_json_v4(sha, wheel=wheel, wheelhouse=manifest))
        prov = work_dir / "provenance.json"
        prov.write_text(
            json.dumps(
                {
                    "schema_version": "release_provenance.v1",
                    "sha": sha,
                    "run_id": run_id,
                    "run_attempt": "1",
                    "built_at": "2026-09-14T00:00:00Z",
                    "workflow_run_url": f"https://github.com/nousergon/crucible/actions/runs/{run_id}",
                    "test_summary": "1 passed",
                }
            )
        )
        code = deploy_main(
            [
                "publish",
                "--sha",
                sha,
                "--store",
                str(root),
                "--wheel",
                str(wheel_path),
                "--release-json",
                str(meta),
                "--provenance-json",
                str(prov),
                "--wheelhouse",
                str(work_dir / "wheelhouse"),
            ]
        )
    assert code == 0
    return manifest
