"""No `crucible` job description claims component-1 (vendor/ingest) work.

Normative source: `architecture.d/146` rule 1 (Brian's ruling, 2026-09-14):
"Collection lives only in component 1. New vendor ingest, a new
`market_data/*` key or an ArcticDB write is placed in `nousergon-data`,
never inside `crucible`, a trader or Metron. v2's `data.daily` / `data.weekly`
read component 1's outputs (rebuild plan §4.3 amendment)."

**The defect this closes** (`alpha-engine-config-I10979`). `crucible/cli.py`'s
`HANDLERS` dict carried three DEAD `_todo(...)` placeholder entries for
`data.daily`/`data.weekly`/`data.heal` — dead because
`HANDLERS.update(TRACK_A_HANDLERS)`, a few lines further down the same
module, unconditionally overwrites them with the real implementations
(`crucible.track_a.handle_data_daily` and friends) at import time. Nothing
ever reached the `_todo` stub; `crucible.cli.is_stub` never saw it either.
That made the stale text inside them free to drift for a full release cycle
with nothing to catch it: `"data.daily": _todo(..., "Lifts the ingest core
from nousergon-data.")` described the PRE-ruling-146 architecture, a year
after rule 1 replaced it — and a reader who trusted the dead literal instead
of the live handler would build a second ingest inside this repo, which is
exactly the failure rule 1 exists to prevent.

**Why this scan and not just deleting the three entries.** Removing the dead
stubs (done in the same change as this test) fixes the one instance; nothing
stops the same class of claim from reappearing in a live, reachable place —
a `JobSpec.help` string (real `--help` text, read every day), a docstring, a
new stub for a job not yet built. This scans the whole package's string
literals for the CLAIM, not just the three now-deleted entries, so a future
regression anywhere in `crucible/` is caught before merge rather than found
by a human a year later.

**AST-based, not a raw-text grep**, for the same reason
`tests/test_no_stale_tracker_literals.py` chose AST: a `#` comment narrating
this history (this file's own docstring above, and the removal comment left
in `crucible/cli.py`) is never part of the AST at all, so a scan built on
`ast.walk` cannot false-positive on prose that correctly cites the phrase it
forbids elsewhere. A module/class/function docstring is exempted the same
way `test_no_stale_tracker_literals.py` exempts it — it is narration, not a
job description a caller reads to decide what a job does.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SCAN_ROOT = REPO_ROOT / "crucible"

SELF = Path(__file__).resolve()

_IGNORED_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache"}

#: Narrow phrases, not bare words. "compile" and "refresh" are legitimate,
#: ACCURATE descriptions of what `data.daily`/`data.weekly` do to their own
#: read-side panel (`crucible.data.daily`, `crucible.keys.data_panel_key`) —
#: a scan on those words alone would false-positive on the correct
#: description. What is actually forbidden is a claim that crucible ingests,
#: lifts or compiles FROM a vendor or from `nousergon-data` itself, which is
#: component 1's job by rule 1.
_FORBIDDEN = re.compile(
    r"lifts?\s+the\s+ingest\s+core"
    r"|vendor\s+ingest"
    r"|ingest[^.]{0,40}nousergon-data"
    r"|nousergon-data[^.]{0,40}ingest"
    r"|compil(?:e|es|ing)[^.]{0,40}from\s+nousergon-data",
    re.IGNORECASE,
)


def _scanned_py_files() -> list[Path]:
    files: list[Path] = []
    for path in SCAN_ROOT.rglob("*.py"):
        if any(part in _IGNORED_DIRS for part in path.parts):
            continue
        if path.resolve() == SELF:
            continue
        files.append(path)
    return sorted(set(files))


def test_the_scan_actually_reads_files() -> None:
    files = _scanned_py_files()
    assert len(files) >= 20, (
        f"the component-boundary scan walked only {len(files)} files under "
        f"{SCAN_ROOT} — it is not reading the package, so a clean result "
        "means nothing."
    )


def _docstring_node_ids(tree: ast.AST) -> set[int]:
    """Same shape as `test_no_stale_tracker_literals.py::_docstring_node_ids`."""
    ids: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        body = getattr(node, "body", [])
        if not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
            if isinstance(first.value.value, str):
                ids.add(id(first.value))
    return ids


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _findings_in_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        return [f"{_display(path)}: does not parse ({exc})"]

    exempt = _docstring_node_ids(tree)
    findings: list[str] = []
    for node in ast.walk(tree):
        if id(node) in exempt:
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if _FORBIDDEN.search(node.value):
                findings.append(
                    f"{_display(path)}:{node.lineno}: claims component-1 "
                    f"(vendor/ingest) work crucible does not own "
                    f"(architecture.d/146 rule 1): {node.value!r}"
                )
    return findings


def test_no_component1_ingest_claim_anywhere_in_the_package() -> None:
    findings: list[str] = []
    for path in _scanned_py_files():
        findings.extend(_findings_in_file(path))
    assert not findings, (
        "the package claims vendor-ingest/component-1 work crucible does not "
        "own (architecture.d/146 rule 1 — collection lives only in "
        f"nousergon-data). {len(findings)} finding(s):\n" + "\n".join(f"  - {f}" for f in findings)
    )


def test_readme_carries_no_component1_ingest_claim() -> None:
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    hits = _FORBIDDEN.findall(text)
    assert not hits, f"README.md claims component-1 ingest work: {hits}"


def test_the_data_jobs_are_not_stubs() -> None:
    """The other half of the fix this test's own module docstring narrates:
    `data.daily`/`data.weekly`/`data.heal` must dispatch to their real
    `crucible.track_a` handlers, never to a `_todo` placeholder — a
    regression here is what let the dead, stale text sit unreachable and
    uncorrected for a release cycle in the first place."""
    from crucible.cli import HANDLERS, is_stub

    for job in ("data.daily", "data.weekly", "data.heal"):
        assert not is_stub(HANDLERS[job]), (
            f"`{job}` dispatches to a `_todo` stub, not a real handler"
        )


def test_data_job_source_module_is_not_the_dead_todo_literal() -> None:
    """Mutation guard, same shape as
    `test_migrate_history_cli.py::test_the_source_no_longer_carries_the_todo`:
    the dead literal this issue closes must not reappear in `cli.py`."""
    source = (REPO_ROOT / "crucible" / "cli.py").read_text(encoding="utf-8")
    assert '"data.daily": _todo(' not in source
    assert '"data.weekly": _todo(' not in source
    assert '"data.heal": _todo(' not in source


# ---------------------------------------------------------------------------
# The detector is shown firing, and shown permitting the correct description
# — a guard nobody has made fail is a guard nobody knows works.
# ---------------------------------------------------------------------------


def test_the_scan_fires_on_the_actual_historical_defect() -> None:
    assert _FORBIDDEN.search("Lifts the ingest core from nousergon-data.")


def test_the_scan_fires_on_vendor_ingest_phrasing() -> None:
    assert _FORBIDDEN.search("data.daily performs vendor ingest before compiling the panel.")


def test_the_scan_permits_the_correct_read_side_description() -> None:
    assert not _FORBIDDEN.search(
        "data.daily reads nousergon-data's published market_data/* contracts and "
        "compiles crucible's own daily panel; data.heal repairs a range of that "
        "panel, in region, idempotently."
    )


def test_the_scan_permits_the_live_jobspec_help_text() -> None:
    """The actual, current `JOBS[...].help` strings — proven not to trip
    the guard, so this test doubles as the positive-case check on real
    production text rather than only a synthetic sample."""
    from crucible.cli import JOBS

    for job in ("data.daily", "data.weekly", "data.heal"):
        assert not _FORBIDDEN.search(JOBS[job].help), (
            f"JOBS[{job!r}].help unexpectedly trips the component-1 guard: {JOBS[job].help!r}"
        )
