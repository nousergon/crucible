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

**Why the account-id pattern requires an adjacent marker, and not a bare
12-digit run (corrected 2026-09-03 — Finding 4, adversarial review of
`alpha-engine-config-I9906`).** An earlier version of this pattern was
`\b\d{12}\b` alone, on the argument that a 12-digit account id can appear
inside an ARN, an IAM policy `Principal`, a CloudTrail log path, or plain
prose, and the SHAPE that matters is the twelve digits, not the syntax around
them. That argument is still correct for what it covers, but it is
**unsound against the clock**: GitHub Actions run ids are monotonically
increasing and 11 digits today (`TestTheAccountIdPatternDoesNotFalsePositiveOnLongerDigitRuns`
already carries an 11-digit exemption test for exactly this reason) — the
next order of magnitude makes a run id 12 digits, and a bare `\b\d{12}\b`
would then fail this repo's CI on every comment that cites one (this file's
own docstrings already cite run ids by number, e.g. `deploy.yml:283`).

The pattern now requires the 12 digits to sit immediately after
`arn:aws:iam::`, a bare `::`, or an `--account`/`account` token — the three
shapes an account id actually appears in across `.github/` and `crucible/`
today (an ARN literal, or a CloudFormation/CLI `--account` flag). **The
accepted delta:** a bare account id typed into unstructured prose with no
adjacent marker — the "plain prose" case the original argument named — no
longer matches. That gap is accepted deliberately rather than closed by a
context-sniffing exclusion (a denylist of "this looks like a run id", the
exact partial-parser shape this file's sibling guards have already been
beaten by five times over): the account-id leak surface in this repo's
`.github/` and `crucible/` is ARNs and CLI flags, not free-form prose, and a
scanner that goes silently red on a legitimate run-id citation gets its
whole pattern deleted by someone in a hurry — which is a bigger, permanent
hole than the narrower one this trades for.
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

#: A 12-digit account id, but ONLY when it sits immediately after
#: `arn:aws:iam::`, a bare `::`, or an `--account`/`account` token — see the
#: module docstring ("Why the account-id pattern requires an adjacent
#: marker...") for why a bare `\b\d{12}\b` was tightened (Finding 4,
#: adversarial review of `alpha-engine-config-I9906`). A module constant
#: rather than a repeated literal: every test below that needs to reason
#: about this specific pattern imports it, so the pattern cannot drift
#: between the table and its own tests.
ACCOUNT_ID_PATTERN = r"(?:arn:aws:iam::|::|--account[ =]|\baccount[ :=])\d{12}\b"

#: A bare `alpha-engine-*` BUCKET stem, with no `s3://` scheme. A module
#: constant for the same reason :data:`ACCOUNT_ID_PATTERN` is one: it was
#: written out four times below, and the correction of 2026-09-04 had to be
#: made in all five places or the tests would have kept asserting the old
#: pattern's behaviour while the scan ran the new one — a self-test that
#: grades a string nothing uses.
BUCKET_STEM_PATTERN = r"\balpha-engine-(data|research|crucible-v2)(?![\w-])"

#: What is forbidden, and why. See the module docstring for why `arn:aws:` on
#: its own is deliberately absent from this table.
FORBIDDEN: dict[str, str] = {
    ACCOUNT_ID_PATTERN: (
        "a bare 12-digit AWS account id — the sensitive half of an "
        "infrastructure identifier per crucible/AGENTS.md; resolve it "
        "through ${{ vars.AWS_ACCOUNT_ID }} instead"
    ),
    r"s3://alpha-engine-": (
        "a literal alpha-engine bucket name — resolve it through "
        "${{ vars.CRUCIBLE_STORE_URI }} instead"
    ),
    # Finding 3 (adversarial review, `alpha-engine-config-I9906`): the two
    # patterns above catch the `s3://` URI form and a fully-inlined account
    # id, but a bucket name written bare — no scheme, just the identifier —
    # matched neither. `crucible/config.py:71 DEFAULT_ARCTIC_BUCKET =
    # "alpha-engine-data"` and `ci.yml:251 --s3-bucket alpha-engine-research`
    # were both live instances the tree-clean reading missed. Scoped to the
    # KNOWN bucket stems this fleet actually has, not a bare `alpha-engine-`
    # prefix: that broader prefix also matches non-bucket identifiers this
    # repo legitimately carries as literals (the `alpha-engine-alerts` /
    # `alpha-engine-alerts-muted` SNS topic names in `crucible/alerts.py`,
    # and every `alpha-engine-config-I####` tracker reference), and a
    # detector that forbids its own class of false positive gets deleted
    # rather than fixed the first time it fires on one.
    # `(?![\w-])`, not `\b` (corrected 2026-09-04, `alpha-engine-config-I9964`).
    # `\b` matches before a hyphen, so `\balpha-engine-research\b` fired on
    # `alpha-engine-research-eval-judge` — a v1 LAMBDA FUNCTION name, not a
    # bucket, and one of the six `alpha-engine-config-I9756` deletes. The
    # guard's own note above says this pattern is scoped to "the KNOWN bucket
    # stems this fleet actually has" precisely so it does not fire on
    # non-bucket identifiers, and `-research-` collided with that intent the
    # first time a function name landed in `crucible/`. Requiring the stem to
    # END there keeps every bucket form it was written for — bare
    # `alpha-engine-research`, `--s3-bucket alpha-engine-research`,
    # `alpha-engine-crucible-v2/crucible` (a `/` is not `[\w-]`) — and stops
    # matching a longer identifier that merely starts with one.
    BUCKET_STEM_PATTERN: (
        "a literal alpha-engine-* bucket name (no s3:// scheme) — resolve it "
        "through a repository variable instead, per crucible/AGENTS.md"
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
        ACCOUNT_ID_PATTERN: "role/x  # arn:aws:iam::711398986525:role/x",
        r"s3://alpha-engine-": "STORE_URI: s3://alpha-engine-crucible-v2/crucible",
        BUCKET_STEM_PATTERN: 'BUCKET = "alpha-engine-data"',
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
    r"""The account-id pattern (`ACCOUNT_ID_PATTERN`) is word-boundary
    anchored on its digit run, so it cannot match a 12-digit SUBSTRING of a
    longer contiguous digit run — a word boundary can only occur at the edge
    of the whole run, never in its middle. Verified against the real shape
    this repository carries: `crucible/slots/strategy.py`'s 15-digit float
    mantissas (e.g. `1.959963984540054`), and a GitHub Actions run id (11
    digits today, one short of the account-id pattern).

    Since Finding 4 (adversarial review, `alpha-engine-config-I9906`) the
    pattern ALSO requires an adjacent marker (`arn:aws:iam::`, `::`,
    `--account`/`account`) — `test_a_bare_12_digit_run_id_does_not_match_with_no_adjacent_marker`
    proves the accepted delta: a future 12-digit run id with no such marker
    nearby does not trip this detector, which a bare `\b\d{12}\b` would have.
    """

    def test_a_15_digit_float_mantissa_does_not_match(self) -> None:
        assert not _PATTERNS[ACCOUNT_ID_PATTERN].search("z = 1.959963984540054 if abs(...)")

    def test_an_11_digit_run_id_does_not_match(self) -> None:
        assert not _PATTERNS[ACCOUNT_ID_PATTERN].search("run 33778251060")

    def test_a_bare_12_digit_run_id_does_not_match_with_no_adjacent_marker(self) -> None:
        """The exact seeded proof from Finding 4: a 12-digit GHA run id, cited
        the way `deploy.yml` already cites run ids, with no `arn:aws:iam::`,
        `::` or `account` token nearby."""
        assert not _PATTERNS[ACCOUNT_ID_PATTERN].search("# see run 337782510601")

    def test_a_genuine_12_digit_account_id_does_match(self) -> None:
        assert _PATTERNS[ACCOUNT_ID_PATTERN].search("arn:aws:iam::711398986525:role/x")

    def test_a_bare_account_id_after_a_double_colon_does_match(self) -> None:
        assert _PATTERNS[ACCOUNT_ID_PATTERN].search("Principal: arn::711398986525")

    def test_an_account_flag_literal_does_match(self) -> None:
        assert _PATTERNS[ACCOUNT_ID_PATTERN].search("--account 711398986525")

    def test_a_bare_alpha_engine_bucket_name_does_match(self) -> None:
        pattern = BUCKET_STEM_PATTERN
        assert _PATTERNS[pattern].search('DEFAULT_ARCTIC_BUCKET = "alpha-engine-data"')
        assert _PATTERNS[pattern].search("--s3-bucket alpha-engine-research")

    def test_the_bare_bucket_pattern_does_not_match_the_alerts_topic_names(self) -> None:
        """`crucible/alerts.py` carries `alpha-engine-alerts` and
        `alpha-engine-alerts-muted` as SNS topic names, deliberately literal —
        a topic name grants no access and is not a bucket. The scoped
        alternation (not a bare `alpha-engine-` prefix) is what keeps this
        pattern from also catching those."""
        pattern = BUCKET_STEM_PATTERN
        assert not _PATTERNS[pattern].search('MUTED_TOPIC = "alpha-engine-alerts-muted"')
        assert not _PATTERNS[pattern].search("default is `alpha-engine-alerts`")

    def test_the_bare_bucket_pattern_does_not_match_a_longer_lambda_name(self) -> None:
        """`alpha-engine-config-I9964`. `\\b` matches before a hyphen, so the
        pre-correction pattern fired on `alpha-engine-research-eval-judge` —
        a v1 LAMBDA function name that `crucible/gate.py` now carries as a
        literal because the phase-0 clause probes those six by EXACT name.

        The old and new patterns are compared here as READINGS THAT DIFFER,
        not as one assertion about the new one: a self-test that only shows
        the current pattern accepting the sample cannot show the correction
        changed anything."""
        old = re.compile(r"\balpha-engine-(data|research|crucible-v2)\b")
        for name in (
            "alpha-engine-research-eval-judge",
            "alpha-engine-research-perturbation-battery",
            "alpha-engine-research-thinktank",
        ):
            assert old.search(name), f"{name} did not reproduce the old false positive"
            assert not _PATTERNS[BUCKET_STEM_PATTERN].search(name), name

    def test_the_correction_did_not_stop_catching_the_bucket_forms(self) -> None:
        """The other half of the same comparison: every form the pattern was
        written for still matches, so the fix narrowed the false positive and
        not the detector."""
        for line in (
            'DEFAULT_ARCTIC_BUCKET = "alpha-engine-data"',
            "--s3-bucket alpha-engine-research",
            "STORE_URI: s3://alpha-engine-crucible-v2/crucible",
            "bucket: alpha-engine-crucible-v2,",
        ):
            assert _PATTERNS[BUCKET_STEM_PATTERN].search(line), line

    def test_the_bare_bucket_pattern_does_not_match_a_tracker_reference(self) -> None:
        # No literal tracker number here on purpose — `test_no_stale_tracker_
        # literals.py` forbids exactly that shape in a non-docstring string,
        # in this very package. `config-` never appears in the bucket
        # alternation, so any `alpha-engine-config-I<N>` reference is already
        # excluded by construction; this proves it without citing one.
        pattern = BUCKET_STEM_PATTERN
        assert not _PATTERNS[pattern].search("alpha-engine-config-" + "I" + "9906")

    def test_a_var_interpolated_account_id_carries_no_bare_digit_run(self) -> None:
        """The exact form this PR ships: no 12-digit literal appears at all,
        so this line must not be flagged — the whole point of the fix."""
        line = (
            "DEPLOY_ROLE_ARN: arn:aws:iam::${{ vars.AWS_ACCOUNT_ID }}"
            ":role/crucible-v2-github-deploy"
        )
        assert not _PATTERNS[ACCOUNT_ID_PATTERN].search(line)
        assert not _PATTERNS[r"s3://alpha-engine-"].search(line)
