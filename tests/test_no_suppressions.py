"""No suppression collections in this repository. At all.

Normative source: plan §11.1 (risk 1, "Execution, not design"):

    "no `_KNOWN_*` / `_GRANDFATHERED_*` / xfail collections in
    `nousergon/crucible` at all — a failing test is fixed or the feature is
    cut"

The measured background: 277 live suppression entries across the fleet with
zero expiries, and 13 of 20 sampled test suppressions outliving an issue that
was already CLOSED. Each was added for a good local reason and none was ever
removed, because nothing counted them. This test counts them, and the count
must be zero.

**Why the whole tree, not just `tests/`.** A grandfathered-field list in
production code is the same defect wearing different clothes: it is a rule
the system declines to apply to the cases that already violate it, and the
list only ever grows.

**Two narrow exemptions, and they are not a list of files.** This module
itself, because it must contain the patterns in order to search for them; and
*docstrings*, because a rule has to be able to state what it forbids. Neither
is a per-file waiver: a docstring is not executable, and a second module
cannot be added to the exemption without editing the identity check below.
Comments are NOT exempt — `# noqa` is a comment, and it is exactly the thing
being counted.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: This file, and nothing else. Deliberately a single Path compared by
#: equality rather than a collection: an exemption LIST is itself the bug
#: class, and the first entry added beside this one is how it starts.
SELF = Path(__file__).resolve()

# Directories that are not source: caches, the virtualenv, git internals.
_IGNORED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "dist",
    "node_modules",
}

# Markdown is excluded: it is prose, and this repository's own CONTRIBUTING
# and acceptance README have to be able to NAME what is forbidden. Everything
# executable or configuration-bearing is scanned.
_SCANNED_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini"}

#: What a suppression collection looks like, and why each one is refused.
FORBIDDEN: dict[str, str] = {
    r"_KNOWN_": (
        "a `_KNOWN_*` list is a rule the code declines to apply to the cases that "
        "already violate it; it only ever grows"
    ),
    r"_GRANDFATHERED_": (
        "a `_GRANDFATHERED_*` list freezes today's violations into the contract; "
        "fix the violation or change the contract"
    ),
    r"\bxfail\b": (
        "an xfail is a failing test that reports green; a failing test is fixed or "
        "the feature is cut (plan §11.1)"
    ),
    r"pytest\.skip": (
        "a skipped test measures nothing and looks like a passing one on the summary line"
    ),
    r"pytest\.mark\.skip": ("same as pytest.skip, spelled as a marker"),
    r"# *type: *ignore\[.*\] *# *TODO": (
        "a type-ignore with a TODO is a suppression with an intention attached and no expiry"
    ),
    r"# *noqa *$": (
        "a bare `# noqa` suppresses every rule, present and future, on that line; name the code"
    ),
}

_PATTERNS = {p: re.compile(p) for p in FORBIDDEN}


def _docstring_lines(source: str) -> set[int]:
    """Line numbers occupied by docstrings in ``source``.

    Excluded from the scan because a policy has to be able to state what it
    forbids, and this repository's modules document their own rules. Parsed
    with `ast` rather than matched with a regex: a heuristic for "is this
    inside a string" is the kind of approximation that silently exempts real
    code the first time someone writes an unusual literal.

    A file that does not parse is NOT silently skipped — see the caller.
    """
    tree = ast.parse(source)
    lines: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        body = getattr(node, "body", [])
        if not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
            if isinstance(first.value.value, str):
                lines.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))
    return lines


def _scanned_files() -> list[Path]:
    files: list[Path] = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in _IGNORED_DIRS for part in path.parts):
            continue
        if path.suffix not in _SCANNED_SUFFIXES:
            continue
        if path.resolve() == SELF:
            continue
        files.append(path)
    return files


def test_the_scan_actually_reads_files() -> None:
    """A guard that walks nothing reports clean.

    This is the failure mode of every detector in the audit: the sweep ran,
    found nothing, and was dark rather than green. Asserting a floor on the
    file count means a broken walk fails here instead of silently blessing
    the tree.
    """
    files = _scanned_files()
    assert len(files) >= 12, (
        f"the suppression scan walked only {len(files)} files; it is not reading the "
        "tree, so its clean result means nothing."
    )
    assert any(f.suffix == ".py" and f.parent.name == "crucible" for f in files), (
        "the scan did not reach the package source"
    )


def test_no_suppression_collections_anywhere_in_the_tree() -> None:
    findings: list[str] = []
    for path in _scanned_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        exempt: set[int] = set()
        if path.suffix == ".py":
            try:
                exempt = _docstring_lines(text)
            except SyntaxError as exc:
                # A source file that does not parse is a finding in itself,
                # never a quiet skip: an unparseable file scanned as clean is
                # how a guard reports green over code it never read.
                findings.append(f"{path.relative_to(REPO_ROOT)}: does not parse ({exc})")
                continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if lineno in exempt:
                continue
            for pattern, compiled in _PATTERNS.items():
                if compiled.search(line):
                    rel = path.relative_to(REPO_ROOT)
                    findings.append(
                        f"{rel}:{lineno}: matches {pattern!r} — {FORBIDDEN[pattern]}\n"
                        f"      {line.strip()[:120]}"
                    )
    assert not findings, (
        "this repository carries no suppression collections (plan §11.1). "
        f"{len(findings)} finding(s):\n" + "\n".join(f"  - {f}" for f in findings)
    )


def test_the_scan_can_actually_find_something(tmp_path: Path) -> None:
    """The detector is shown firing.

    A guard nobody has made fail is a guard nobody knows works — the audit's
    own repeated finding. This constructs each forbidden pattern and asserts
    the matcher catches it, so a regex broken by an edit fails here rather
    than reporting a clean tree forever.
    """
    samples = {
        r"_KNOWN_": "_KNOWN_FAILURES = ['a']",
        r"_GRANDFATHERED_": "_GRANDFATHERED_BARE_FIELDS = {'x'}",
        r"\bxfail\b": "@pytest.mark.xfail(reason='later')",
        r"pytest\.skip": "pytest.skip('not ready')",
        r"pytest\.mark\.skip": "@pytest.mark.skip",
        r"# *type: *ignore\[.*\] *# *TODO": "x = y  # type: ignore[arg-type]  # TODO",
        r"# *noqa *$": "import os  # noqa",
    }
    assert set(samples) == set(FORBIDDEN), (
        "every forbidden pattern needs a sample proving the matcher fires on it; "
        f"missing: {sorted(set(FORBIDDEN) - set(samples))}"
    )
    for pattern, sample in samples.items():
        assert _PATTERNS[pattern].search(sample), (
            f"pattern {pattern!r} no longer matches its own sample {sample!r} — the "
            "detector is broken and would report a clean tree."
        )
