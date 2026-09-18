"""Every path that writes the production store is a registered job.

Normative source: `AGENTS.md` rule 1 ("manifest or it did not happen") and
`alpha-engine-config-I10968`.

## The class, and why the first fix did not survive it

`alpha-engine-config-I10459` promoted ONE hand-rolled store write — the
nightly integration summary, a `store.put_bytes` inside
`.github/workflows/integration-nightly.yml` — into a registered job. It fixed
that call site. Two others of the identical shape were still there months
later, and both fed surfaces that GRADE the system:

* `python -m crucible.review record` wrote the verdict
  `crucible.gate._clause_independently_reviewed` reads, from a
  `store.put_bytes` reached out of that module's own `__main__`;
* `.github/workflows/ci.yml` copied the plan §12 rule-3 acceptance reading
  into the store with `aws s3 cp` — never through `crucible.store.Store` at
  all.

Neither filed a manifest, so neither had lineage, spend, a code sha, or
anything that said it had run. `engagement-protocol-policy` §5: fixing one
call site of a systemic defect is not a fix. This module is what makes a
THIRD one fail on the commit that adds it.

## What is checked, precisely — two ENTRY-POINT surfaces, not every write

There are ~30 `store.put_bytes` call sites under `crucible/`, and every one
of them is legitimate: they sit inside job bodies that `run_job` has already
wrapped. A predicate over write CALL SITES would therefore have to prove
reachability from a `run_job` body, which no static scan can do honestly.

What both instances of this class actually share is narrower and decidable:
**a way to reach a store write that is not `crucible <job>`**. There are
exactly two such surfaces in this repository, and each is scanned here:

1. **A workflow step that writes to the store with the AWS CLI.** The store
   is reachable from CI by URI, so a `cp`/`sync`/`put-object` naming
   `STORE_URI` bypasses the package entirely. Reads stay legal — `deploy.yml`
   pulls release proof, `board.yml` pulls the board — because a read files
   nothing and has no lineage to record (`AGENTS.md`, "Reading a gate from
   the laptop").
2. **A `crucible/` module with its own `__main__` that reaches a write
   primitive.** That is precisely what `crucible/review.py` was: an entry
   point beside the CLI, with a store write behind it. `crucible/cli.py` is
   the job table itself, and a module named for a declared workflow job
   (`crucible.models.WORKFLOW_JOB_VALUES` — `crucible/deploy.py`) writes its
   own `run_manifest.v2` document under its own identity by design. Both are
   admitted by DERIVATION, never by a file list: `AGENTS.md` rule 4 forbids a
   suppression collection, and a one-member allowlist is the case that reads
   as harmless.

Residual, named rather than claimed away: a module with no `__main__` whose
function is imported and called by something outside the CLI would not be
caught here. Nothing in this repository does that today, and the two surfaces
above are the two that were actually used.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from crucible.models import WORKFLOW_JOB_VALUES

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = REPO_ROOT / "crucible"
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

#: The RAW store-mutating primitives — `crucible.store.Store`'s own, the ones
#: that write without a `RunContext` in the picture.
#: `RunContext.record_output`/`record_output_cas` are deliberately NOT here:
#: they only exist inside a `run_job` body, which is the correct shape, and
#: `crucible/review.py` reaches one from the module that also carries the
#: read-only `authors`/`check` entry point. Listing them would fail the fixed
#: file for having been fixed. Mirrors
#: `tests/test_champion_pointer_writer_contract.py::_WRITE_ATTRS`, minus the
#: two context methods, for the same reason that test names the module rather
#: than the call.
WRITE_ATTRS = ("put_bytes", "compare_and_swap")

#: An AWS CLI form that WRITES. `aws s3 cp` is directional, so the store URI's
#: position decides it: a destination naming the store is a write, a source
#: naming it is a read.
_AWS_WRITE_FORMS = ("aws s3 cp", "aws s3 sync", "aws s3api put-object")

#: How a workflow names the store.
_STORE_REFERENCES = ("STORE_URI", "CRUCIBLE_STORE_URI")


def _module_entry_points() -> list[Path]:
    """Every `crucible/**.py` that can be run as a program."""
    return sorted(p for p in PACKAGE.rglob("*.py") if '__name__ == "__main__"' in p.read_text())


def _admitted_entry_points() -> set[Path]:
    """The entry points that MAY reach a write, derived from the job tables.

    `cli.py` is the job table. A module named for a declared workflow job is
    run by a workflow under its own identity and writes its own manifest in
    the same schema (`crucible.models.WORKFLOW_JOB_VALUES`).
    """
    admitted = {PACKAGE / "cli.py"}
    admitted |= {PACKAGE / f"{job.split('.')[0]}.py" for job in WORKFLOW_JOB_VALUES}
    return admitted


def _writes_only_into_the_dispatch_namespace(source: str) -> bool:
    """Does every write in ``source`` key off `crucible.keys.dispatch_exit_key`?

    **The third admission, and it is a PROPERTY rather than a name**
    (`alpha-engine-config-I11050`). `runs/_dispatch/` is the one store
    namespace whose documents are written by the SUBSTRATE about a dispatch,
    not by a job about its work: the request record is written by the
    dispatcher Lambda in `nous-ergon-ops` (which this guard cannot see at
    all), and the exit record is written by the box's EXIT trap — which runs
    after the job's process is over, including on the argparse-refusal and
    reclaimed paths where there IS no job and no manifest by construction.

    A module admitted here is therefore not an ungraded write beside the CLI;
    it is the evidence that makes rule 1 honest for the one path that
    deliberately files no manifest. The admission stays narrow because it is
    keyed on the key: a module that also wrote anywhere else fails, and
    `crucible.keys.dispatch_exit_key` refuses any key outside that prefix.
    """
    tree = ast.parse(source)
    key_arguments = [
        node.args[0] if node.args else None
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in WRITE_ATTRS
    ]
    if not key_arguments:
        return False
    return all(
        isinstance(argument, ast.Call)
        and isinstance(argument.func, ast.Name)
        and argument.func.id == "dispatch_exit_key"
        for argument in key_arguments
    )


def _writes_in(source: str) -> set[str]:
    return {
        node.func.attr
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in WRITE_ATTRS
    }


def _write_lines(body: str) -> list[str]:
    """Lines of a workflow `run:` body that write to the store with the CLI."""
    hits = []
    for raw in body.splitlines():
        line = raw.strip()
        if line.startswith("#"):
            continue
        if not any(form in line for form in _AWS_WRITE_FORMS):
            continue
        if not any(name in line for name in _STORE_REFERENCES):
            continue
        if "aws s3 cp" in line:
            # Directional: `cp SOURCE DEST`. A read names the store first.
            after = line.split("aws s3 cp", 1)[1]
            arguments = [a for a in after.split() if not a.startswith("-")]
            if arguments and any(name in arguments[0] for name in _STORE_REFERENCES):
                continue
        hits.append(line)
    return hits


def _run_bodies(text: str) -> list[str]:
    """Every `run:` block body, read with a regex rather than a YAML load.

    The point is what the FILE contains: a YAML load of a workflow with a
    templated expression is lossy in ways that would silently drop a step,
    and a dropped step reads as a pass.
    """
    return re.findall(r"run:\s*\|[^\n]*\n((?:[ \t]+.*\n|\n)+)", text)


def test_the_scan_finds_something_to_scan() -> None:
    """An empty derivation would pass every assertion below vacuously."""
    assert _module_entry_points(), "no runnable module under crucible/ — the scan is inert"
    assert list(WORKFLOWS.glob("*.yml")), "no workflows — the scan is inert"


@pytest.mark.parametrize("workflow", sorted(WORKFLOWS.glob("*.yml")), ids=lambda p: p.name)
def test_no_workflow_writes_to_the_store_behind_the_cli(workflow: Path) -> None:
    text = workflow.read_text(encoding="utf-8")
    offenders = [line for body in _run_bodies(text) for line in _write_lines(body)]
    assert not offenders, (
        f"{workflow.name} writes to the store with the AWS CLI: {offenders}. A store "
        "write is a job — it goes through `crucible <job>` so that `crucible.runner."
        "run_job` files its manifest, its lineage, its spend and its code sha. Reads "
        "are unaffected."
    )


@pytest.mark.parametrize(
    "module", _module_entry_points(), ids=lambda p: str(p.relative_to(REPO_ROOT))
)
def test_no_second_entry_point_reaches_a_store_write(module: Path) -> None:
    if module in _admitted_entry_points():
        return
    source = module.read_text(encoding="utf-8")
    if _writes_only_into_the_dispatch_namespace(source):
        return
    writes = _writes_in(source)
    assert not writes, (
        f"{module.relative_to(REPO_ROOT)} can be run as a program and reaches "
        f"{sorted(writes)}. That is a store writer beside the CLI, and it files no run "
        "manifest: register it in `crucible.cli.JOBS` with a `crucible/components.yaml` "
        "row, or move the write behind one."
    )


def test_the_guard_fires_on_the_shape_it_was_built_for() -> None:
    """The self-test, both halves — a detector nobody has made fail is a
    detector nobody knows works. The two sources are the two real instances:
    `crucible/review.py`'s old `__main__` write and `ci.yml`'s old copy."""
    old_review_main = (
        "def record(store, *, document):\n"
        "    store.put_bytes('reviews/x.json', b'{}')\n"
        'if __name__ == "__main__":\n'
        "    raise SystemExit(0)\n"
    )
    assert _writes_in(old_review_main) == {"put_bytes"}

    old_publish_step = (
        '          aws s3 cp acceptance-reading.json "${STORE_URI}/${KEY}" --only-show-errors\n'
    )
    assert _write_lines(old_publish_step), "the workflow half of the guard measures nothing"


def test_the_dispatch_namespace_admission_does_not_admit_a_second_write() -> None:
    """The admission's own self-test. An entry point that writes the exit
    record AND anything else is exactly the shape this guard exists for, and
    it must still fail — otherwise the admission is a file list wearing a
    predicate's clothes."""
    exit_record_only = (
        "def write(store, document):\n"
        "    store.put_bytes(dispatch_exit_key(document.job, document.dispatch_id), b'{}')\n"
        'if __name__ == "__main__":\n'
        "    raise SystemExit(0)\n"
    )
    assert _writes_only_into_the_dispatch_namespace(exit_record_only)

    also_writes_a_verdict = exit_record_only.replace(
        "    raise SystemExit(0)\n",
        "    store.put_bytes('reviews/x.json', b'{}')\n",
    )
    assert not _writes_only_into_the_dispatch_namespace(also_writes_a_verdict)

    writes_nothing = 'if __name__ == "__main__":\n    raise SystemExit(0)\n'
    assert not _writes_only_into_the_dispatch_namespace(writes_nothing), (
        "a module with no writes must not be ADMITTED by this predicate — it has "
        "nothing to admit, and a vacuous true here would admit the next one that does"
    )

    # ...and the reads that must stay legal.
    legal_read = '          aws s3 cp "${STORE_URI}/board/current.json" board.json\n'
    assert not _write_lines(legal_read)
