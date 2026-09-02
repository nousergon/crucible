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

#: A second, deliberate exemption — reviewed 2026-09-02 on
#: alpha-engine-config-I9807's PR, and it is exactly one entry, not a list:
#: `tests/test_key_construction_placement.py` names its own
#: `_KNOWN_ARCHITECTURAL_EXCEPTIONS` registry with a `_KNOWN_` prefix ON
#: PURPOSE, so THIS scanner sees it rather than being blind to an unnamed
#: collection — the exact defect a hidden allowlist (originally just
#: `_ALLOWED`, matching no `FORBIDDEN` pattern here) turned out to BE.
#: It is a reviewed, architectural registry: every entry is a case where a
#: function correctly lives outside `crucible/keys.py` today, never a "not
#: moved yet" debt list — a debt list would itself be a suppression
#: collection, and shipping one is a rule change (`AGENTS.md` rule 4)
#: outside a single review's authority, so that file carries none (round 3
#: of I9807's review: `_KNOWN_TRACKED_DEBT` was removed by moving its one
#: entry, `gate_key`, into `crucible/keys.py` instead of listing it here).
#:
#: **Gated on the exact IDENTIFIER, not the file.** An earlier version of
#: this exemption skipped the whole `_KNOWN_` PATTERN for the sanctioned
#: file, which admitted any number of arbitrarily-named `_KNOWN_*`
#: collections there undetected (measured: appending
#: `_KNOWN_ANYTHING_GOES: dict[str, str] = {"a": "b"}` to that file still
#: passed clean) — a hole in the scanner, and the file-path exemption's own
#: comment asserted the opposite. `_SANCTIONED_KNOWN_IDENTIFIERS` names
#: exactly the identifier(s) this exemption covers; `_line_is_exempt`
#: below only skips a `_KNOWN_` match when removing every occurrence of a
#: sanctioned identifier from the line leaves no `_KNOWN_` text behind — a
#: third, unsanctioned `_KNOWN_*` collection in the SAME file, on the SAME
#: or a different line, still fails. Only the `_KNOWN_` pattern is exempted
#: this way, and only for the one file — `xfail`, `pytest.skip` and the
#: rest are still scanned there like everywhere else.
_SANCTIONED_KNOWN_REGISTRY_FILE = REPO_ROOT / "tests" / "test_key_construction_placement.py"
_SANCTIONED_KNOWN_IDENTIFIERS = ("_KNOWN_ARCHITECTURAL_EXCEPTIONS",)


def _line_is_exempt_known_pattern(line: str) -> bool:
    """True only if every `_KNOWN_` occurrence on ``line`` is accounted for
    by a sanctioned identifier substring. An unsanctioned `_KNOWN_*` name on
    the same line — even one that also contains a sanctioned identifier —
    still leaves `_KNOWN_` text behind and is reported."""
    stripped = line
    for name in _SANCTIONED_KNOWN_IDENTIFIERS:
        stripped = stripped.replace(name, "")
    return "_KNOWN_" not in stripped


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


def _findings_for_file(path: Path, text: str) -> list[str]:
    """Every FORBIDDEN-pattern match in ``text``, as if read from ``path``.

    Extracted from the tree-wide scan so the sanctioned-exemption's exact
    boundary can be exercised directly, against a real path comparison,
    rather than only observed indirectly through the whole-tree result.
    """
    findings: list[str] = []
    exempt: set[int] = set()
    if path.suffix == ".py":
        try:
            exempt = _docstring_lines(text)
        except SyntaxError as exc:
            # A source file that does not parse is a finding in itself,
            # never a quiet skip: an unparseable file scanned as clean is
            # how a guard reports green over code it never read.
            findings.append(f"{path}: does not parse ({exc})")
            return findings
    is_sanctioned_registry_file = path.resolve() == _SANCTIONED_KNOWN_REGISTRY_FILE
    for lineno, line in enumerate(text.splitlines(), start=1):
        if lineno in exempt:
            continue
        for pattern, compiled in _PATTERNS.items():
            if (
                pattern == r"_KNOWN_"
                and is_sanctioned_registry_file
                and _line_is_exempt_known_pattern(line)
            ):
                # The one deliberate exemption declared beside
                # `_SANCTIONED_KNOWN_REGISTRY_FILE` above — gated on the
                # exact identifier via `_line_is_exempt_known_pattern`,
                # not the whole `_KNOWN_` pattern for the file. Every
                # other FORBIDDEN pattern is still scanned here.
                continue
            if compiled.search(line):
                findings.append(
                    f"{path}:{lineno}: matches {pattern!r} — {FORBIDDEN[pattern]}\n"
                    f"      {line.strip()[:120]}"
                )
    return findings


def test_no_suppression_collections_anywhere_in_the_tree() -> None:
    findings: list[str] = []
    for path in _scanned_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        findings.extend(
            f.replace(str(path), str(path.relative_to(REPO_ROOT)), 1)
            for f in _findings_for_file(path, text)
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


class TestTheSanctionedKnownRegistryExemptionIsExactlyAsNarrowAsClaimed:
    """Self-tests for `_SANCTIONED_KNOWN_REGISTRY_FILE` / `_line_is_exempt_known_pattern`.

    A guard that weakens `test_no_suppression_collections_anywhere_in_the_tree`
    is the one change in alpha-engine-config-I9807 with no self-test proving
    what it does and does not admit — caught on that PR's round-3 review,
    after a file-path-only exemption was measured to admit an arbitrarily
    named `_KNOWN_*` collection undetected. These four cases are exactly the
    axes that mattered: the sanctioned identifier itself, an unsanctioned
    identifier in the same file, a different FORBIDDEN pattern in the same
    file, and the sanctioned identifier's own pattern in a different file.
    """

    def test_the_sanctioned_identifier_itself_is_exempt(self) -> None:
        text = '_KNOWN_ARCHITECTURAL_EXCEPTIONS: dict[str, dict[str, str]] = {"x": {}}'
        assert _findings_for_file(_SANCTIONED_KNOWN_REGISTRY_FILE, text) == []

    def test_an_unsanctioned_known_collection_in_the_sanctioned_file_is_still_reported(
        self,
    ) -> None:
        """The exact mutation the review measured passing clean: appending an
        arbitrarily named `_KNOWN_*` collection to the sanctioned file."""
        text = '_KNOWN_ANYTHING_GOES: dict[str, str] = {"a": "b"}'
        findings = _findings_for_file(_SANCTIONED_KNOWN_REGISTRY_FILE, text)
        assert findings, (
            "an unsanctioned _KNOWN_* collection in the sanctioned file must still be "
            "reported — the exemption is gated on the identifier, not the file."
        )

    def test_a_line_naming_both_a_sanctioned_and_unsanctioned_identifier_is_reported(
        self,
    ) -> None:
        """A sanctioned identifier's presence on a line must not launder an
        unsanctioned one riding along on the same line."""
        text = "_KNOWN_ANYTHING_GOES = _KNOWN_ARCHITECTURAL_EXCEPTIONS"
        findings = _findings_for_file(_SANCTIONED_KNOWN_REGISTRY_FILE, text)
        assert findings, (
            "a line naming an unsanctioned _KNOWN_* identifier is reported even when a "
            "sanctioned identifier also appears on it."
        )

    def test_a_different_forbidden_pattern_in_the_sanctioned_file_is_still_reported(
        self,
    ) -> None:
        text = "_GRANDFATHERED_THING = 1"
        findings = _findings_for_file(_SANCTIONED_KNOWN_REGISTRY_FILE, text)
        assert findings, (
            "the sanctioned file's exemption covers only the _KNOWN_ pattern; "
            "_GRANDFATHERED_ (and every other FORBIDDEN pattern) is still scanned there."
        )

    def test_a_known_collection_in_an_unsanctioned_file_is_still_reported(self) -> None:
        text = "_KNOWN_ARCHITECTURAL_EXCEPTIONS = {}"
        other_path = _SANCTIONED_KNOWN_REGISTRY_FILE.parent / "test_some_other_module.py"
        assert other_path.resolve() != _SANCTIONED_KNOWN_REGISTRY_FILE
        findings = _findings_for_file(other_path, text)
        assert findings, (
            "the exemption is scoped to one specific file; the same sanctioned "
            "identifier text in any other file is still reported."
        )
