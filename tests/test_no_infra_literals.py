r"""No infrastructure identifier literals in `.github/` or `crucible/`. At all.

Normative source: `crucible/AGENTS.md` (Visibility) — "Never here: ...
infrastructure identifiers, bucket names, account numbers, ARNs." This repo
goes PUBLIC at phase-1 exit (`alpha-engine-config-I9775`), and until
`alpha-engine-config-I9906`, four workflows carried the AWS account id and
the store bucket as literals: `arn:aws:iam::711398986525:role/...` and
`s3://alpha-engine-crucible-v2/crucible`, in `board.yml`, `deploy.yml`,
`ci.yml`, `morning-report.yml` and `adversarial-review-record.yml`. Measured
2026-09-03: all FIVE workflow files carried the account id, not the four
`alpha-engine-config-I9906` named — `adversarial-review-record.yml` is a
fifth instance the issue's own survey missed.

**Scope is `.github/` and `crucible/` only, mirroring `test_no_suppressions.py`'s
shape but not its tree-wide reach.** `tests/` deliberately carries the real
account id as literal test fixtures (`test_autonomy.py`, `test_tags.py`)
asserting behaviour against the account this fleet actually runs in — that is
a different concern (a test fixture, never shipped, never read by a public
clone of this repo) from a literal baked into a workflow or a package module
that ships with the tree.

**What is forbidden, and why the boundary sits where it does.** Two patterns:
a bare 12-digit AWS account id, and the `s3://alpha-engine-` bucket-name
prefix. Both are the SENSITIVE half of an infrastructure identifier — the
account id names the AWS account this fleet writes to, and the bucket name
narrows an attacker straight to the one store worth reading. Neither can be
reconstructed from public information.

**Deliberately NOT forbidden: the literal `arn:aws:` prefix on its own.**
`alpha-engine-config-I9906`'s own dispatch decided — and this file documents
the decision — that a role NAME (`crucible-v2-github-deploy`, ...) may stay a
literal, because a role name alone grants no access and is not a secret; only
the account id is. Constructing a role ARN from a var-substituted account id
still requires writing the literal syntax `arn:aws:iam::` — banning that
string outright would make the very form this PR ships (`arn:aws:iam::${{
vars.AWS_ACCOUNT_ID }}:role/crucible-v2-github-deploy`) fail its own guard.
The 12-digit pattern already catches a FULLY literal ARN (one with the
account id inlined); a prefix with no digit run inlined carries nothing this
guard needs to refuse.

**Why a bare 12-digit run and not a scoped ARN regex.** A 12-digit account id
can appear inside an ARN, an IAM policy `Principal`, a CloudTrail log path, or
plain prose — the SHAPE that matters is the twelve digits, not the syntax
around them. `\b\d{12}\b` (word-boundary anchored) does not false-positive on
a longer digit run (a run id, a float literal in `crucible/slots/strategy.py`
carries 15-digit mantissas) because a word boundary cannot occur in the
MIDDLE of a contiguous digit sequence — verified directly in
`TestTheAccountIdPatternDoesNotFalsePositiveOnLongerDigitRuns` below.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: `.github/` and `crucible/` only — the two directories `crucible/AGENTS.md`
#: binds, and the two the deliverable named. Not `tests/`, which carries the
#: real account id as fixture data on purpose (see module docstring).
SCAN_ROOTS = (REPO_ROOT / ".github", REPO_ROOT / "crucible")

#: This file, and nothing else — it must contain the patterns in order to
#: search for and document them. Same exemption shape as
#: `tests/test_no_suppressions.py::SELF`: a single path compared by equality,
#: never a collection.
SELF = Path(__file__).resolve()

_IGNORED_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache"}

_SCANNED_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini"}

#: What is forbidden, and why. See the module docstring for why `arn:aws:` on
#: its own is deliberately absent from this table.
FORBIDDEN: dict[str, str] = {
    r"\b\d{12}\b": (
        "a bare 12-digit AWS account id — the sensitive half of an "
        "infrastructure identifier per crucible/AGENTS.md; resolve it "
        "through ${{ vars.AWS_ACCOUNT_ID }} instead"
    ),
    r"s3://alpha-engine-": (
        "a literal alpha-engine bucket name — resolve it through "
        "${{ vars.CRUCIBLE_STORE_URI }} instead"
    ),
}

_PATTERNS = {p: re.compile(p) for p in FORBIDDEN}


def _scanned_files() -> list[Path]:
    files: list[Path] = []
    for root in SCAN_ROOTS:
        for path in root.rglob("*"):
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
    """A guard that walks nothing reports clean. Assert it walked something,
    and that it reached both scanned roots — the failure this repository's
    other scanners have all guarded against once already."""
    files = _scanned_files()
    assert len(files) >= 5, (
        f"the infra-literal scan walked only {len(files)} files under "
        f"{SCAN_ROOTS} — it is not reading the tree, so its clean result means nothing."
    )
    assert any(f.parent.name == "workflows" for f in files), (
        "the scan did not reach .github/workflows/"
    )
    assert any(f.suffix == ".py" and f.parent.name == "crucible" for f in files), (
        "the scan did not reach the crucible/ package source"
    )


def _findings_for_file(path: Path, text: str) -> list[str]:
    findings: list[str] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for pattern, compiled in _PATTERNS.items():
            if compiled.search(line):
                findings.append(
                    f"{path}:{lineno}: matches {pattern!r} — {FORBIDDEN[pattern]}\n"
                    f"      {line.strip()[:160]}"
                )
    return findings


def test_no_infra_identifier_literal_in_github_or_crucible() -> None:
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
    # Tracker reference lives only above, in the module docstring — never in
    # this runtime string — per `tests/test_no_stale_tracker_literals.py`.
    assert not findings, (
        "no infrastructure identifier literal (bare account id, alpha-engine bucket "
        "name) may appear under .github/ or crucible/ (see crucible/AGENTS.md). "
        f"{len(findings)} finding(s):\n" + "\n".join(f"  - {f}" for f in findings)
    )


def test_the_scan_can_actually_find_something() -> None:
    """The detector is shown firing. A guard nobody has made fail is a guard
    nobody knows works."""
    samples = {
        r"\b\d{12}\b": "role/x  # arn:aws:iam::711398986525:role/x",
        r"s3://alpha-engine-": "STORE_URI: s3://alpha-engine-crucible-v2/crucible",
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


class TestTheAccountIdPatternDoesNotFalsePositiveOnLongerDigitRuns:
    r"""`\b\d{12}\b` is word-boundary anchored, so it cannot match a 12-digit
    SUBSTRING of a longer contiguous digit run — a word boundary can only
    occur at the edge of the whole run, never in its middle. Verified against
    the real shape this repository carries: `crucible/slots/strategy.py`'s
    15-digit float mantissas (e.g. `1.959963984540054`), and a GitHub Actions
    run id (11 digits, one short of the account-id pattern).
    """

    def test_a_15_digit_float_mantissa_does_not_match(self) -> None:
        assert not _PATTERNS[r"\b\d{12}\b"].search("z = 1.959963984540054 if abs(...)")

    def test_an_11_digit_run_id_does_not_match(self) -> None:
        assert not _PATTERNS[r"\b\d{12}\b"].search("run 33778251060")

    def test_a_genuine_12_digit_account_id_does_match(self) -> None:
        assert _PATTERNS[r"\b\d{12}\b"].search("arn:aws:iam::711398986525:role/x")

    def test_a_var_interpolated_account_id_carries_no_bare_digit_run(self) -> None:
        """The exact form this PR ships: no 12-digit literal appears at all,
        so this line must not be flagged — the whole point of the fix."""
        line = (
            "DEPLOY_ROLE_ARN: arn:aws:iam::${{ vars.AWS_ACCOUNT_ID }}"
            ":role/crucible-v2-github-deploy"
        )
        assert not _PATTERNS[r"\b\d{12}\b"].search(line)
        assert not _PATTERNS[r"s3://alpha-engine-"].search(line)
