"""The hash-locked, per-release wheelhouse a dispatched box installs from.

Normative source: plan §4.11; `alpha-engine-config-I10812`.

    releases/{sha}/wheelhouse/requirements.lock.txt   `uv export` of uv.lock, with hashes
    releases/{sha}/wheelhouse/{wheel}.whl             every dependency, built for the BOX

**Why this exists.** Until I10812 a box resolved its whole dependency tree from
PyPI at boot: only the crucible wheel came from the release bucket. That was
two defects at once. A PyPI read timeout killed a heal chunk before its job
started (2026-09-14), and — worse, because nothing ever reported it — every
box re-resolved dependencies the deploy smoke had never graded, so `release_sha`
named the code a run executed but not the dependency set it executed against.

A release now carries every wheel its install needs, for the box platform
(Amazon Linux 2023, CPython 3.12, x86_64), plus the lock that pins them by
hash. The box installs with `--no-index --require-hashes` and so cannot reach
PyPI at all: a missing or altered wheel fails the install loud, it never
resolves around it.

**What lives here, and what does not.** The layout constants the box's
bootstrap (`nous-ergon-ops/infrastructure/cloudformation/crucible-v2.yaml`)
must agree with, the lock parser, the manifest builder `deploy.yml` runs, and
the lock/manifest agreement check the publisher and the smoke both apply. No
store I/O and no crucible imports: `crucible.models` imports
:func:`wheelhouse_digest` from here to validate a record on construction, and
this module must not import it back.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "LOCK_FILENAME",
    "PIP_OFFLINE_FLAGS",
    "WHEELHOUSE_DIRNAME",
    "WHEELHOUSE_PLATFORM",
    "LockRequirement",
    "WheelhouseLockMismatchError",
    "build_manifest",
    "parse_lock",
    "verify_against_lock",
    "wheel_project_and_version",
    "wheelhouse_digest",
]

#: The directory under `releases/{sha}/` every wheel and the lock live in.
#: Read by `nous-ergon-ops`' box bootstrap through
#: `tests/crossrepo/test_crucible_box_wheelhouse_lockstep.py`, never restated.
WHEELHOUSE_DIRNAME = "wheelhouse"

#: The lock's filename inside the wheelhouse. Fixed, not recorded-and-trusted,
#: so the schema can pin it as a literal and a record cannot point the box's
#: `-r` at some other object under the prefix.
LOCK_FILENAME = "requirements.lock.txt"

#: The flags that make an install unable to reach an index. BOTH are required
#: and neither is enough alone: `--no-index` without `--require-hashes` installs
#: whatever wheel of the right name is in the directory; `--require-hashes`
#: without `--no-index` still resolves over the network, which is the
#: availability defect this module closes.
PIP_OFFLINE_FLAGS = ("--no-index", "--require-hashes")

#: The platform the wheelhouse is resolved on, recorded on the release so a
#: reader can see what the wheels were chosen for. `deploy.yml` builds inside
#: this image rather than cross-downloading with `--platform`: pip evaluates
#: environment markers against the RUNNING interpreter, so only an interpreter
#: on the box's own OS resolves the box's marker set and platform tags.
WHEELHOUSE_PLATFORM = "amazonlinux:2023/x86_64/cp312"

_HASH_RE = re.compile(r"--hash=sha256:([0-9a-f]{64})")
_REQ_RE = re.compile(r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)==(?P<version>[^\s;\\]+)")


class WheelhouseLockMismatchError(ValueError):
    """The wheels on a record and the lock they claim to satisfy disagree."""


@dataclass(frozen=True)
class LockRequirement:
    """One pinned requirement of `requirements.lock.txt`."""

    name: str
    version: str
    hashes: frozenset[str]
    marker: str


def _normalize(name: str) -> str:
    """PEP 503 name normalisation — `Foo_Bar` and `foo-bar` are one project."""
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_lock(text: str) -> dict[str, LockRequirement]:
    """Parse a `uv export --format requirements-txt` lock, keyed by normalised name.

    Every requirement must be `name==version` with at least one sha256 hash —
    the only form `pip --require-hashes` accepts. Anything else raises: a lock
    this parser half-understood is a lock whose agreement check would pass over
    exactly the lines it skipped.
    """
    logical: list[str] = []
    buffer = ""
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip() if not raw.lstrip().startswith("--hash") else raw
        line = line.rstrip()
        if not line.strip():
            if buffer:
                logical.append(buffer)
                buffer = ""
            continue
        if line.endswith("\\"):
            buffer += line[:-1] + " "
            continue
        buffer += line
        logical.append(buffer)
        buffer = ""
    if buffer:
        logical.append(buffer)

    requirements: dict[str, LockRequirement] = {}
    for entry in logical:
        entry = entry.strip()
        match = _REQ_RE.match(entry)
        if match is None:
            raise ValueError(
                f"lock line {entry[:120]!r} is not a `name==version --hash=sha256:...` "
                "requirement. `pip --require-hashes` accepts nothing else, so the box would "
                "refuse this lock; refusing it here names the line instead."
            )
        hashes = frozenset(_HASH_RE.findall(entry))
        if not hashes:
            raise ValueError(f"lock requirement {match['name']}=={match['version']} has no hash")
        head = entry.split("--hash", 1)[0]
        marker = head.split(";", 1)[1].strip() if ";" in head else ""
        key = _normalize(match["name"])
        if key in requirements:
            raise ValueError(f"lock pins {match['name']} twice")
        requirements[key] = LockRequirement(
            name=key, version=match["version"], hashes=hashes, marker=marker
        )
    if not requirements:
        raise ValueError("the lock pins no requirements")
    return requirements


def wheel_project_and_version(filename: str) -> tuple[str, str]:
    """`numpy-2.5.3-cp312-cp312-manylinux_2_28_x86_64.whl` -> (`numpy`, `2.5.3`)."""
    if not filename.endswith(".whl") or "/" in filename:
        raise ValueError(f"{filename!r} is not a wheel filename")
    parts = filename[: -len(".whl")].split("-")
    if len(parts) not in (5, 6):
        raise ValueError(f"{filename!r} is not a PEP 427 wheel filename")
    return _normalize(parts[0]), parts[1]


def wheelhouse_digest(lock_sha256: str, wheels: Iterable[Mapping[str, str]]) -> str:
    """The one value naming a release's whole installed dependency set.

    sha256 over `sha256sum`-format lines — every wheel, sorted by filename,
    then the lock — so it is reproducible with coreutils alone from a
    downloaded wheelhouse. Recorded on the release and stamped on every run
    manifest as `wheelhouse_digest` (I10812 deliverable 4).
    """
    lines = [f"{w['sha256']}  {w['filename']}" for w in sorted(wheels, key=lambda w: w["filename"])]
    lines.append(f"{lock_sha256}  {LOCK_FILENAME}")
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def verify_against_lock(wheels: Iterable[Mapping[str, str]], lock_text: str) -> None:
    """Raise unless the wheels are exactly one installable set for the lock.

    * every wheel is a project and version the lock pins, and its sha256 is
      one of the hashes the lock allows for that pin — a wheel the lock does
      not vouch for is a wheel `--require-hashes` refuses on the box;
    * no project appears twice;
    * every lock requirement WITHOUT an environment marker has a wheel. A
      requirement with a marker may be absent: it can be excluded on the box
      (`sys_platform == 'emscripten'`), and whether it is excluded is decided
      by the interpreter that built the wheelhouse on the box's own OS, not
      re-derived here. One that should have applied and is missing still fails
      the offline install, loud.
    """
    lock = parse_lock(lock_text)
    seen: dict[str, str] = {}
    problems: list[str] = []
    for wheel in wheels:
        name, version = wheel_project_and_version(wheel["filename"])
        if name in seen:
            problems.append(f"{name} has two wheels: {seen[name]} and {wheel['filename']}")
            continue
        seen[name] = wheel["filename"]
        pinned = lock.get(name)
        if pinned is None:
            problems.append(f"{wheel['filename']} is not a project the lock pins")
        elif pinned.version != version:
            problems.append(f"{wheel['filename']} is {version}; the lock pins {pinned.version}")
        elif wheel["sha256"] not in pinned.hashes:
            problems.append(
                f"{wheel['filename']} hashes to {wheel['sha256']}, which is not one of the "
                f"{len(pinned.hashes)} hashes the lock allows for {name}=={version}"
            )
    for name, requirement in sorted(lock.items()):
        if name not in seen and not requirement.marker:
            problems.append(f"the lock pins {name}=={requirement.version} but no wheel carries it")
    if problems:
        raise WheelhouseLockMismatchError(
            "the wheelhouse does not match its lock, so a --require-hashes install refuses it: "
            + "; ".join(problems)
        )


def build_manifest(directory: Path, *, extras: Iterable[str]) -> dict[str, object]:
    """The `wheelhouse` object `release.json` (release.v4) carries.

    Hashes the bytes on disk; never trusts a filename or a download log. Runs
    the lock agreement check before returning, so `deploy.yml` cannot build a
    record over a wheelhouse the box would refuse.
    """
    lock_path = directory / LOCK_FILENAME
    lock_bytes = lock_path.read_bytes()
    wheels = [
        {"filename": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in sorted(directory.glob("*.whl"))
    ]
    if not wheels:
        raise ValueError(f"{directory} holds no wheels")
    unexpected = sorted(
        p.name for p in directory.iterdir() if p.name != LOCK_FILENAME and p.suffix != ".whl"
    )
    if unexpected:
        raise ValueError(
            f"{directory} holds files that are neither wheels nor the lock: {unexpected}"
        )
    verify_against_lock(wheels, lock_bytes.decode("utf-8"))
    lock_sha256 = hashlib.sha256(lock_bytes).hexdigest()
    return {
        "lock_filename": LOCK_FILENAME,
        "lock_sha256": lock_sha256,
        "extras": sorted(set(extras)),
        "platform": WHEELHOUSE_PLATFORM,
        "wheels": wheels,
        "digest": wheelhouse_digest(lock_sha256, wheels),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m crucible.wheelhouse",
        description="Print the release.v4 `wheelhouse` manifest for a built wheelhouse.",
    )
    parser.add_argument("--dir", required=True, help="the wheelhouse directory")
    parser.add_argument(
        "--extras",
        required=True,
        help="comma-separated extras the lock was exported with (from pyproject.toml)",
    )
    args = parser.parse_args(argv)
    extras = [e.strip() for e in args.extras.split(",") if e.strip()]
    print(json.dumps(build_manifest(Path(args.dir), extras=extras), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
