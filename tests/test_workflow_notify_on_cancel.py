"""A job killed at its `timeout-minutes` still reaches the failure notification.

`alpha-engine-config-I11448`, measured 2026-09-23: the Board render was
cancelled at its 25-minute timeout on two consecutive scheduled runs
(35798554590, 35806564700), and `notify-failure` was SKIPPED both times. A job
GitHub kills at `timeout-minutes` concludes `cancelled`, not `failure`, and the
notify job's `if: ${{ failure() }}` is false for a `cancelled` need, so the
one signal that the board had gone stale fired for nobody. Four more workflows
carried the same condition.

This file evaluates every notify job's `if:` against the need results a
timeout, a failure, a green run and a superseded run produce. It is a small
evaluator for the expression subset those conditions use, not a GitHub
Actions emulator: a token it does not know RAISES, so a condition rewritten
into a shape nobody modelled fails here rather than being read as whatever
the evaluator guessed.

**The modelled semantics, stated so they can be checked:** `always()` is
true; `failure()` is true when any need concluded `failure`; `cancelled()` is
true only when the RUN was cancelled — a concurrency supersede or a human
pressing Cancel — and false for a single job killed by its own
`timeout-minutes`, which fails that job without cancelling the run.
"""

from __future__ import annotations

import pathlib
import re
from typing import Any

import pytest
import yaml

WORKFLOW_DIR = pathlib.Path(__file__).resolve().parents[1] / ".github" / "workflows"
WORKFLOWS = sorted(p for p in WORKFLOW_DIR.glob("*.y*ml") if p.suffix in {".yml", ".yaml"})
NOTIFY_WORKFLOW = "nousergon/nousergon-lib/.github/workflows/notify-ci-failure.yml@"

_TOKEN = re.compile(
    r"""\s*(?:
        (?P<contains>contains\(\s*needs\.\*\.result\s*,\s*'(?P<cval>[^']*)'\s*\))
      | (?P<need>needs\.(?P<nname>[A-Za-z0-9_-]+)\.result)
      | (?P<fn>(?:always|failure|cancelled|success)\(\))
      | (?P<ctx>github\.(?:event_name|ref))
      | (?P<str>'[^']*')
      | (?P<op>&&|\|\||==|!=|!|\(|\))
    )""",
    re.VERBOSE,
)


def _to_python(condition: str) -> str:
    """Translate the expression subset into a Python expression over the
    evaluation namespace. Anything else raises."""
    expr = condition.strip()
    if expr.startswith("${{") and expr.endswith("}}"):
        expr = expr[3:-2]
    out: list[str] = []
    pos = 0
    expr = expr.rstrip()
    while pos < len(expr):
        match = _TOKEN.match(expr, pos)
        if match is None or match.end() == pos:
            raise ValueError(f"unmodelled expression at {expr[pos:]!r} in {condition!r}")
        pos = match.end()
        if match["contains"]:
            out.append(f"_contains_need({match['cval']!r})")
        elif match["need"]:
            out.append(f"_need({match['nname']!r})")
        elif match["fn"]:
            out.append(f"_fn({match['fn'][:-2]!r})")
        elif match["ctx"]:
            out.append(f"_ctx({match['ctx']!r})")
        elif match["str"]:
            out.append(repr(match["str"][1:-1]))
        else:
            out.append({"&&": " and ", "||": " or ", "!": " not "}.get(match["op"], match["op"]))
    return "".join(out)


def _evaluate(
    condition: str,
    *,
    needs: dict[str, str],
    run_cancelled: bool,
    event_name: str,
    ref: str = "refs/heads/main",
) -> bool:
    if not condition:
        # GitHub's implicit `success()`.
        return all(result == "success" for result in needs.values()) and not run_cancelled
    functions = {
        "always": True,
        "failure": any(result == "failure" for result in needs.values()),
        "cancelled": run_cancelled,
        "success": all(result == "success" for result in needs.values()) and not run_cancelled,
    }
    namespace: dict[str, Any] = {
        "_fn": functions.__getitem__,
        "_need": needs.__getitem__,
        "_contains_need": lambda value: value in needs.values(),
        "_ctx": {"github.event_name": event_name, "github.ref": ref}.__getitem__,
        "__builtins__": {},
    }
    return bool(eval(_to_python(condition), namespace))  # noqa: S307 - translated, closed namespace


def _load(path: pathlib.Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _notify_jobs() -> list[tuple[str, str, dict[str, Any], bool]]:
    """Every job calling the fleet notification workflow, with whether its
    workflow's concurrency cancels a superseded run."""
    found = []
    for path in WORKFLOWS:
        workflow = _load(path)
        concurrency = workflow.get("concurrency") or {}
        supersedes = isinstance(concurrency, dict) and bool(concurrency.get("cancel-in-progress"))
        for name, job in (workflow.get("jobs") or {}).items():
            if str(job.get("uses", "")).startswith(NOTIFY_WORKFLOW):
                found.append((path.name, name, job, supersedes))
    return found


NOTIFY_JOBS = _notify_jobs()
_IDS = [f"{workflow}:{job}" for workflow, job, _, _ in NOTIFY_JOBS]
#: The notify jobs whose workflow's concurrency cancels a superseded run.
SUPERSEDING = [entry for entry in NOTIFY_JOBS if entry[3]]
#: The notify jobs that carry their own pull_request exclusion.
PR_EXCLUDED = [entry for entry in NOTIFY_JOBS if "pull_request" in entry[2].get("if", "")]


def _ids(entries: list[tuple[str, str, dict[str, Any], bool]]) -> list[str]:
    return [f"{workflow}:{job}" for workflow, job, _, _ in entries]


def _needs(job: dict[str, Any]) -> list[str]:
    needs = job.get("needs") or []
    return [needs] if isinstance(needs, str) else list(needs)


def test_the_notify_jobs_are_found() -> None:
    """Guard the guard: an empty parametrisation passes vacuously."""
    names = {workflow for workflow, _, _, _ in NOTIFY_JOBS}
    assert {"board.yml", "gate-close.yml", "morning-report.yml"} <= names
    assert {"ci.yml", "cost-gate.yml"} <= {workflow for workflow, _, _, _ in SUPERSEDING}
    assert {"ci.yml", "cost-gate.yml"} <= {workflow for workflow, _, _, _ in PR_EXCLUDED}


@pytest.mark.parametrize(("workflow", "name", "job", "supersedes"), NOTIFY_JOBS, ids=_IDS)
def test_a_timed_out_need_reaches_the_notification(
    workflow: str, name: str, job: dict[str, Any], supersedes: bool
) -> None:
    """The Board case: each need in turn killed at its timeout, the jobs
    after it skipped or green, and the run itself NOT cancelled."""
    needs = _needs(job)
    assert needs, f"{workflow}:{name} needs nothing, so it can notify on nothing"
    for index, timed_out in enumerate(needs):
        for after in ("skipped", "success"):
            results = {
                need: ("success" if i < index else "cancelled" if i == index else after)
                for i, need in enumerate(needs)
            }
            assert _evaluate(
                job.get("if", ""), needs=results, run_cancelled=False, event_name="schedule"
            ), (
                f"{workflow}:{name} does not run when `{timed_out}` is killed at its "
                f"timeout-minutes (needs {results}). A timeout concludes `cancelled`, "
                "not `failure`."
            )


@pytest.mark.parametrize(("workflow", "name", "job", "supersedes"), NOTIFY_JOBS, ids=_IDS)
def test_a_failed_need_still_reaches_the_notification(
    workflow: str, name: str, job: dict[str, Any], supersedes: bool
) -> None:
    needs = _needs(job)
    results = {need: ("failure" if i == 0 else "skipped") for i, need in enumerate(needs)}
    assert _evaluate(job.get("if", ""), needs=results, run_cancelled=False, event_name="push")


@pytest.mark.parametrize(("workflow", "name", "job", "supersedes"), NOTIFY_JOBS, ids=_IDS)
def test_a_green_run_notifies_nobody(
    workflow: str, name: str, job: dict[str, Any], supersedes: bool
) -> None:
    results = dict.fromkeys(_needs(job), "success")
    assert not _evaluate(job.get("if", ""), needs=results, run_cancelled=False, event_name="push")


@pytest.mark.parametrize(
    ("workflow", "name", "job", "supersedes"), SUPERSEDING, ids=_ids(SUPERSEDING)
)
def test_a_superseded_run_does_not_page(
    workflow: str, name: str, job: dict[str, Any], supersedes: bool
) -> None:
    """Where concurrency cancels a superseded run, every cancelled need is
    routine — a burst of merges would otherwise page once per merge. Only
    the workflows whose concurrency does that are held to it."""
    results = dict.fromkeys(_needs(job), "cancelled")
    assert not _evaluate(job.get("if", ""), needs=results, run_cancelled=True, event_name="push")


def test_the_evaluator_refuses_an_unmodelled_token() -> None:
    with pytest.raises(ValueError, match="unmodelled"):
        _to_python("${{ startsWith(github.ref, 'refs/tags/') }}")


def test_the_pre_fix_condition_is_what_skipped_the_board() -> None:
    """The measured defect, reproduced through the same evaluator: the
    condition board.yml carried before I11448 does not run on a timeout."""
    assert not _evaluate(
        "${{ failure() }}", needs={"board": "cancelled"}, run_cancelled=False, event_name="schedule"
    )


@pytest.mark.parametrize(
    ("workflow", "name", "job", "supersedes"), PR_EXCLUDED, ids=_ids(PR_EXCLUDED)
)
def test_a_notify_job_kept_off_pull_requests_stays_off(
    workflow: str, name: str, job: dict[str, Any], supersedes: bool
) -> None:
    """`always()` must not have widened a PR exclusion the job already had
    (`tests/test_workflow_triggers.py` allowlists these jobs on it)."""
    for result in ("failure", "cancelled"):
        results = dict.fromkeys(_needs(job), result)
        assert not _evaluate(
            job.get("if", ""), needs=results, run_cancelled=False, event_name="pull_request"
        )
