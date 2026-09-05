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
#: **Gated on the exact IDENTIFIER, not the file, not a substring.** Two
#: prior holes, both measured, both closed:
#:
#: Round 2: an earlier version skipped the whole `_KNOWN_` PATTERN for the
#: sanctioned file, admitting any arbitrarily-named `_KNOWN_*` collection
#: there undetected (`_KNOWN_ANYTHING_GOES: dict[str, str] = {"a": "b"}`
#: passed clean).
#:
#: Round 3: the fix for that used `str.replace` to strip the sanctioned
#: name as a SUBSTRING, so any identifier merely CONTAINING the sanctioned
#: name was exempt too — `_KNOWN_ARCHITECTURAL_EXCEPTIONS_2` (the
#: sanctioned name plus a suffix) and
#: `X_KNOWN_ARCHITECTURAL_EXCEPTIONS_EXTRA` (the sanctioned name inside a
#: longer identifier) both passed clean under that version: a debt list —
#: the exact thing this file must never carry — landing under a
#: two-character rename.
#:
#: Fixed by matching WHOLE IDENTIFIER TOKENS: `_line_is_exempt_known_pattern`
#: extracts every identifier-shaped token on the line and requires every
#: token that CONTAINS `_KNOWN_` to EXACTLY equal a name in
#: `_SANCTIONED_KNOWN_IDENTIFIERS` — not merely contain one, and not merely
#: be contained by one. A third, unsanctioned `_KNOWN_*` identifier in the
#: SAME file, on the SAME or a different line, still fails, whether it
#: extends the sanctioned name, is extended BY it, or shares none of it.
#: Only the `_KNOWN_` pattern is exempted this way, and only for the one
#: file — `xfail`, `pytest.skip` and the rest are still scanned there like
#: everywhere else.
_SANCTIONED_KNOWN_REGISTRY_FILE = REPO_ROOT / "tests" / "test_key_construction_placement.py"
_SANCTIONED_KNOWN_IDENTIFIERS = frozenset({"_KNOWN_ARCHITECTURAL_EXCEPTIONS"})

_IDENTIFIER_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _line_is_exempt_known_pattern(line: str) -> bool:
    """True only if every identifier TOKEN on ``line`` that contains
    `_KNOWN_` is EXACTLY one of `_SANCTIONED_KNOWN_IDENTIFIERS` — not a
    token that merely contains a sanctioned name as a substring
    (`_KNOWN_ARCHITECTURAL_EXCEPTIONS_2`) or that a sanctioned name is a
    substring of (`X_KNOWN_ARCHITECTURAL_EXCEPTIONS_EXTRA`, one token,
    since `X` and `_` are both word characters with no boundary between
    them). A line with no `_KNOWN_`-containing token at all is vacuously
    exempt from THIS pattern (there is nothing on it to sanction), which is
    safe: the caller only reaches this function when deciding whether to
    skip a `_KNOWN_` match, and a line with no such token cannot produce
    one.
    """
    tokens_containing_known = [t for t in _IDENTIFIER_TOKEN_RE.findall(line) if "_KNOWN_" in t]
    if not tokens_containing_known:
        return True
    return all(token in _SANCTIONED_KNOWN_IDENTIFIERS for token in tokens_containing_known)


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
#: Suffix-less files that are configuration and must be scanned by NAME —
#: `.coveragerc` has no suffix, so a `pragma: no cover` inside it was
#: invisible to the suffix filter (independent review, 2026-09-05).
_SCANNED_NAMES = {".coveragerc"}

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
    r"(?i)pragma[:\s]?\s*no\s*cover": (
        "a `pragma: no cover` narrows the coverage ratchet one line at a time, by the "
        "author, with no reviewer and no expiry — the 93% floor `pyproject.toml` "
        "declares is only a floor if nothing can carve lines out from under it "
        "(independent adversarial review of crucible-PR89, 2026-09-04: eight live "
        "sites, none scanned). A line that cannot be reached by a test is restructured "
        "or covered; a STRUCTURAL exclusion (`if TYPE_CHECKING:`, the `__main__` "
        "guard) is declared once in `[tool.coverage.report].exclude_lines` and that "
        "list is closed by `test_coverage_exclusions_are_structural_only` below"
    ),
}

#: The only coverage exclusions permitted, each structural — a pattern that
#: names a construct whose body cannot execute under pytest BY CONSTRUCTION,
#: never a free-text marker an author can attach to an arbitrary line. Adding
#: an entry is a rule change reviewed as one, not a way to make a PR pass.
#: ANCHORED to the whole line (`^\s*...\s*$`). coverage.py applies each
#: entry as `re.search` over the raw source line, comments included, so an
#: unanchored `if TYPE_CHECKING:` was attachable as a trailing comment to any
#: `def` and excluded its whole body — the same per-line narrowing as the
#: pragma, and nothing scanned for it (independent review, 2026-09-05).
_STRUCTURAL_COVERAGE_EXCLUSIONS = frozenset(
    {
        "^\\s*if __name__ == [\"']__main__[\"']:\\s*$",
        r"^\s*if TYPE_CHECKING:\s*$",
    }
)

#: The ONLY keys `[tool.coverage.report]` and `[tool.coverage.run]` may carry,
#: and the values the two scope-defining ones must hold. `exclude_also`,
#: `partial_branches`, a non-empty `omit` or a narrowed `source` each narrow
#: the ratchet file-by-file or line-by-line with no reviewer — the class the
#: pragma belonged to (independent review, 2026-09-05, findings 3-5).
_COVERAGE_REPORT_KEYS = frozenset({"fail_under", "show_missing", "exclude_lines"})
_COVERAGE_RUN_KEYS = frozenset({"source", "omit"})
_COVERAGE_SOURCE = ["crucible"]
_COVERAGE_FLOOR = 93

#: Files coverage.py reads INSTEAD of pyproject.toml when present (its
#: discovery order: .coveragerc, setup.cfg, tox.ini, pyproject.toml). A
#: `.coveragerc` carrying `exclude_lines = pragma: no cover` displaced the
#: whole pinned table, fail_under included, in the reviewer's reproduction.
_DISPLACING_COVERAGE_FILES = (".coveragerc", "setup.cfg", "tox.ini")

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
        if path.suffix not in _SCANNED_SUFFIXES and path.name not in _SCANNED_NAMES:
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
        r"(?i)pragma[:\s]?\s*no\s*cover": "def _client():  # pragma: no cover",
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
    what it does and does not admit. Two rounds of review found two holes,
    both measured, both closed, both pinned here:

    Round 3: a file-path-only exemption admitted an arbitrarily named
    `_KNOWN_*` collection anywhere in the sanctioned file, undetected.

    Round 4: the round-3 fix matched the sanctioned name as a SUBSTRING,
    so `_KNOWN_ARCHITECTURAL_EXCEPTIONS_2` (sanctioned name plus a suffix)
    and `X_KNOWN_ARCHITECTURAL_EXCEPTIONS_EXTRA` (sanctioned name inside a
    longer identifier) both passed clean — a debt list landing under a
    two-character rename. Fixed by requiring a WHOLE-TOKEN match.
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

    def test_a_sanctioned_name_plus_a_suffix_is_still_reported(self) -> None:
        """Round-4 finding: the round-3 fix stripped the sanctioned name as a
        SUBSTRING, so an identifier merely CONTAINING it (the sanctioned name
        plus a two-character suffix) was exempt too — a debt list landing
        under a rename. `_KNOWN_ARCHITECTURAL_EXCEPTIONS_2` is a distinct
        identifier from `_KNOWN_ARCHITECTURAL_EXCEPTIONS` and must not be
        exempt merely because it starts with it."""
        text = '_KNOWN_ARCHITECTURAL_EXCEPTIONS_2: dict[str, str] = {"track_a": "not moved yet"}'
        findings = _findings_for_file(_SANCTIONED_KNOWN_REGISTRY_FILE, text)
        assert findings, (
            "_KNOWN_ARCHITECTURAL_EXCEPTIONS_2 CONTAINS the sanctioned identifier as a "
            "prefix but is not equal to it, and must still be reported."
        )

    def test_a_sanctioned_name_embedded_in_a_longer_identifier_is_still_reported(self) -> None:
        """Round-4 finding, the other direction: the sanctioned name as a
        substring INSIDE a longer identifier, rather than the sanctioned
        name extended by a suffix."""
        text = "X_KNOWN_ARCHITECTURAL_EXCEPTIONS_EXTRA = {}"
        findings = _findings_for_file(_SANCTIONED_KNOWN_REGISTRY_FILE, text)
        assert findings, (
            "X_KNOWN_ARCHITECTURAL_EXCEPTIONS_EXTRA is one identifier token containing "
            "the sanctioned name as a substring, not equal to it, and must still be "
            "reported."
        )


def test_coverage_exclusions_are_structural_only() -> None:
    """The coverage floor is only a floor if nothing can carve lines out from
    under it. `[tool.coverage.report].exclude_lines` must be EXACTLY the
    structural set above — no `pragma: no cover` (a per-line, author-applied,
    unreviewed narrowing; the 2026-09-04 independent review's finding against
    crucible-PR89), and no new pattern that was not reviewed as a rule change.

    Read with `tomllib` from the file, not from coverage's loaded config, so
    the assertion is about what the repository declares rather than about
    whichever configuration happened to be active in this process.
    """
    import tomllib

    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = set(config["tool"]["coverage"]["report"]["exclude_lines"])
    assert declared == set(_STRUCTURAL_COVERAGE_EXCLUSIONS), (
        f"pyproject.toml exclude_lines is {sorted(declared)}; the permitted structural "
        f"set is {sorted(_STRUCTURAL_COVERAGE_EXCLUSIONS)}. An exclusion is a construct "
        "whose body cannot run under pytest by construction, never a marker an author "
        "attaches to a line; a line a test cannot reach is restructured or covered."
    )


def test_the_structural_set_names_no_free_text_marker() -> None:
    """The closed set itself must not smuggle the thing it replaces: every
    entry is a Python construct (an `if` header), not a comment marker."""
    for entry in _STRUCTURAL_COVERAGE_EXCLUSIONS:
        construct = entry.removeprefix("^\\s*")
        assert construct.startswith("if "), f"{entry!r} is not a structural construct"
        assert "pragma" not in entry and "#" not in entry, (
            f"{entry!r} is a comment marker, which is exactly the per-line narrowing the "
            "closed set exists to forbid"
        )


def test_each_structural_exclusion_is_anchored_and_cannot_ride_a_comment() -> None:
    """Finding 1 of the 2026-09-05 independent review: coverage.py matches
    `exclude_lines` with `re.search` on the raw line, so an unanchored entry
    is attachable as a trailing comment to any `def` and excludes its whole
    body. Each entry must match the real construct and NOTHING that carries
    the construct's text after code."""
    real = {
        r"^\s*if TYPE_CHECKING:\s*$": ["if TYPE_CHECKING:", "    if TYPE_CHECKING:  "],
        "^\\s*if __name__ == [\"']__main__[\"']:\\s*$": [
            'if __name__ == "__main__":',
            "if __name__ == '__main__':",
        ],
    }
    assert set(real) == set(_STRUCTURAL_COVERAGE_EXCLUSIONS)
    smuggled = [
        "def unused():  # if TYPE_CHECKING:",
        "x = compute()  # if TYPE_CHECKING:",
        'def unused2():  # if __name__ == "__main__":',
        "def unused3():  # comment says if __name__ == q__main__q:",
    ]
    for pattern in _STRUCTURAL_COVERAGE_EXCLUSIONS:
        assert pattern.startswith("^") and pattern.endswith("$"), f"{pattern!r} is not anchored"
        compiled = re.compile(pattern)
        for line in real[pattern]:
            assert compiled.search(line), f"{pattern!r} must match the real construct {line!r}"
        for line in smuggled:
            assert not compiled.search(line), (
                f"{pattern!r} matches {line!r} — a trailing comment would exclude that "
                "line's whole block, the per-line narrowing this set exists to forbid"
            )


def test_no_other_coverage_narrowing_knob_is_set() -> None:
    """Findings 3-5: `exclude_also`, `omit`, `source` (and `partial_branches`,
    `exclude_also`'s sibling) narrow the ratchet with no reviewer. The two
    coverage tables are CLOSED: exactly these keys, and the two scope keys
    hold exactly the whole-tree values."""
    import tomllib

    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    report = config["tool"]["coverage"]["report"]
    run = config["tool"]["coverage"]["run"]
    assert set(report) == set(_COVERAGE_REPORT_KEYS), (
        f"[tool.coverage.report] carries {sorted(set(report) - _COVERAGE_REPORT_KEYS)}; "
        "exclude_also / partial_branches / any new key narrows the ratchet without review"
    )
    assert set(run) == set(_COVERAGE_RUN_KEYS), (
        f"[tool.coverage.run] carries {sorted(set(run) ^ _COVERAGE_RUN_KEYS)}"
    )
    assert run["source"] == _COVERAGE_SOURCE, "the denominator is the whole package"
    assert run["omit"] == [], "omit stays empty so the scope cannot be narrowed file-by-file"
    assert report["fail_under"] >= _COVERAGE_FLOOR, (
        "lowering the floor is a policy amendment, visible here as well as in the diff"
    )


def test_no_file_displaces_the_pinned_coverage_config() -> None:
    """Finding 2: coverage.py reads `.coveragerc`, then `setup.cfg`, then
    `tox.ini`, and only then `pyproject.toml`. A `.coveragerc` with its own
    `exclude_lines` replaced the WHOLE pinned table, `fail_under` included,
    and the suffix-filtered scanner never opened it. None may exist carrying
    coverage config, and the scanner now opens `.coveragerc` by name."""
    assert not (REPO_ROOT / ".coveragerc").exists(), ".coveragerc displaces pyproject.toml"
    for name in ("setup.cfg", "tox.ini"):
        path = REPO_ROOT / name
        if path.exists():
            assert "[coverage:" not in path.read_text(encoding="utf-8"), (
                f"{name} carries a [coverage:*] section, which displaces pyproject.toml"
            )
    assert ".coveragerc" in _SCANNED_NAMES


def test_ci_pins_the_coverage_config_file() -> None:
    """Belt to the test above's braces: the CI invocation names the config
    file, so even a `.coveragerc` that slipped past review is not what CI
    measures against."""
    text = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    cov_lines = [ln for ln in text.splitlines() if "--cov=crucible" in ln and "pytest" in ln]
    assert cov_lines, "ci.yml no longer runs the coverage step"
    for line in cov_lines:
        assert "--cov-config=pyproject.toml" in line, line


def test_the_pragma_pattern_catches_every_spelling_coverage_honours() -> None:
    """coverage.py's own default is `#\s*(pragma|PRAGMA)[:\s]?\s*(no|NO)\s*(cover|COVER)`;
    the scanner must fire on at least everything that default would honour."""
    pattern = next(p for p in FORBIDDEN if "pragma" in p)
    compiled = _PATTERNS[pattern]
    for line in (
        "x = 1  # pragma: no cover",
        "x = 1  # pragma:NO COVER",
        "x = 1  # PRAGMA: no cover",
        "x = 1  # pragma: no\tcover",
        "x = 1  # pragma no cover",
        "x = 1  #pragma:nocover",
    ):
        assert compiled.search(line), line
