"""Every action and reusable workflow this repository calls is pinned by SHA.

A moving ref — `@main`, `@v4`, a branch — is a supply-chain hole: whoever can
push to that ref runs code in this repository's CI, and in the jobs that carry
`secrets: inherit` or `id-token: write`, as this repository's identities.

**Why this file exists separately from the existing workflow guard.**
`tests/test_workflow_triggers.py` asserts a 40-hex pin, but only for
`ci.yml:notify-main-failure`, through a mechanism (`REUSABLE_WORKFLOW_JOBS`)
that engages only for jobs on a PR-REACHABLE workflow. Measured 2026-09-02 on
this branch: changing both `actions/checkout@<sha>` and
`notify-ci-failure.yml@<sha>` to `@main` in `.github/workflows/board.yml` left
all 1029 tests passing. `board.yml` declares no pull_request event, so its
`notify-failure` job — which calls a reusable workflow with `secrets: inherit`
— escaped the mechanism entirely.

The guard was therefore a guard on ONE job of one workflow while reading as a
guard on the repository. That is the "one call site fixed while its siblings
stay broken" shape, and the repository's own words for the narrower version of
it are already on record: *"a moving ref like `@main` is a supply-chain hole,
and a prefix check accepted one."*

This file grades EVERY `uses:` in EVERY workflow, at both step level and job
level, and derives its population from the directory rather than a list.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

WORKFLOW_DIR = Path(__file__).resolve().parents[1] / ".github" / "workflows"

#: A full 40-character commit sha. Not a prefix: GitHub resolves a short sha,
#: so a prefix check accepts `@v4` for any action whose name happens to be
#: four hex characters, and more importantly accepts an abbreviated sha that a
#: future collision could re-point. Forty characters or it is not a pin.
_PINNED = re.compile(r"^[^@]+@[0-9a-f]{40}$")

#: `uses:` values that name a LOCAL path rather than a remote ref. `./…` is
#: this repository's own checked-out tree, which is the diff under review and
#: needs no pin.
_LOCAL = re.compile(r"^\./")


def _workflows() -> list[Path]:
    """Every workflow file, by extension, from the directory itself.

    Both extensions: GitHub reads `.yml` and `.yaml` alike, and a guard that
    globs only one leaves a whole file class unscanned — a hole this
    repository has already had once.
    """
    return sorted([*WORKFLOW_DIR.glob("*.yml"), *WORKFLOW_DIR.glob("*.yaml")])


def _uses_in(document: dict[str, Any]) -> list[tuple[str, str]]:
    """Every `uses:` in the document, as `(where, value)`.

    Job level and step level both. The job-level form is the one that calls a
    reusable workflow, and it is the form that carries `secrets: inherit` —
    so a guard that scanned only steps would miss precisely the highest-
    privilege call in the file.
    """
    found: list[tuple[str, str]] = []
    for job_name, job in (document.get("jobs") or {}).items():
        if not isinstance(job, dict):
            continue
        job_uses = job.get("uses")
        if isinstance(job_uses, str):
            found.append((f"{job_name} (job-level `uses:`)", job_uses))
        for index, step in enumerate(job.get("steps") or []):
            if isinstance(step, dict) and isinstance(step.get("uses"), str):
                found.append((f"{job_name} step {index}", step["uses"]))
    return found


@pytest.mark.parametrize("path", _workflows(), ids=lambda p: p.name)
def test_every_uses_in_every_workflow_is_pinned_by_full_sha(path: Path) -> None:
    document = yaml.safe_load(path.read_text())
    assert isinstance(document, dict), f"{path.name} did not parse to a mapping"
    for where, value in _uses_in(document):
        if _LOCAL.match(value):
            continue
        assert _PINNED.match(value), (
            f"{path.name}: {where} uses {value!r}, which is not pinned to a full "
            "40-character commit sha. A moving ref is a supply-chain hole: whoever can "
            "push to it runs code here as this repository's identities."
        )


def test_the_scan_found_workflows_and_uses_at_all() -> None:
    """Fails CLOSED, in both dimensions.

    An empty file list, or files that parse to no `uses:` at all, would make
    every parametrised case above pass over nothing — green while checking
    nothing, which is the exact failure this repository grades other systems
    on and the reason this guard was written in the first place.
    """
    paths = _workflows()
    assert len(paths) >= 3, f"scanned only {len(paths)} workflow file(s)"
    total = 0
    for path in paths:
        document = yaml.safe_load(path.read_text())
        if isinstance(document, dict):
            total += len(_uses_in(document))
    assert total >= 5, f"found only {total} `uses:` across {len(paths)} workflow(s)"


def test_a_job_level_uses_is_seen_by_the_scanner() -> None:
    """The specific blindness this file was written for.

    The pre-existing guard reached job-level `uses:` only for jobs on a
    PR-reachable workflow, so a reusable-workflow call on a schedule-only
    workflow was unscanned. Asserted directly rather than trusted.
    """
    document = {
        "jobs": {
            "notify": {"uses": "owner/repo/.github/workflows/w.yml@main", "secrets": "inherit"}
        }
    }
    assert _uses_in(document) == [
        ("notify (job-level `uses:`)", "owner/repo/.github/workflows/w.yml@main")
    ]


def test_a_moving_ref_is_rejected_and_a_full_sha_is_accepted() -> None:
    """The predicate itself, in both directions.

    A pin check that accepted everything would pass every case above.
    """
    assert not _PINNED.match("actions/checkout@main")
    assert not _PINNED.match("actions/checkout@v4")
    assert not _PINNED.match("actions/checkout@3d3c42e")  # abbreviated
    assert _PINNED.match("actions/checkout@" + "3" * 40)
