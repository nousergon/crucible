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

**Scope is the whole tree: `.github/`, `crucible/` AND `tests/`
(corrected 2026-09-07, `alpha-engine-config-I10156`).** This file previously
exempted `tests/` on the written argument that a fixture is "never shipped,
never read by a public clone of this repo". That argument is false on its
face and was load-bearing: `tests/` IS in the tree, a public clone reads it,
and it carried the real account id fifteen times plus the real runtime,
dispatcher, scheduler and deploy role names. The exemption was written while
the repo was private, when the claim was harmless and untestable; the flip
made it wrong without anything editing it. A carve-out whose justification
stops being true is the failure mode this file exists to catch.

**What is forbidden, and why the boundary sits where it does.** Two patterns:
a bare 12-digit AWS account id, and the `s3://alpha-engine-` bucket-name
prefix. Both are the SENSITIVE half of an infrastructure identifier — the
account id names the AWS account this fleet writes to, and the bucket name
narrows an attacker straight to the one store worth reading. Neither can be
reconstructed from public information.

**Role, function and topic NAMES are now forbidden too
(reversed 2026-09-07 by Brian's ruling, `alpha-engine-config-I10156`).**
`alpha-engine-config-I9906` had decided the opposite — that a role name may
stay a literal "because a role name alone grants no access and is not a
secret". Brian's ruling replaced the secrecy test with the tiering policy's
actual test, which is PURPOSE: "if someone else picks up the repo they won't
be able to use the account so it doesn't belong there." An outside reader of
this AGPL tree can use none of these names; publishing them only removes the
guessing step from an `AssumeRole` enumeration against a named account.
`repository-tiering-policy.md` says the same thing directly — "it isn't
harmful" is explicitly NOT a public job.

**Still deliberately NOT forbidden: the literal `arn:aws:` prefix on its
own.** Constructing an ARN from var-substituted parts requires writing the
syntax `arn:aws:iam::`, so banning that string would make the very form this
tree ships (`arn:aws:iam::${{ vars.AWS_ACCOUNT_ID }}:role/${{
vars.CRUCIBLE_ROLE_PREFIX }}-deploy`) fail its own guard. The 12-digit
pattern already catches a fully-literal ARN; a prefix with nothing inlined
carries nothing to refuse.

**And the account id is masked in the logs as well as absent from the tree.**
Seven workflows print an assumed-role ARN (`echo "identity: ${assumed}"`),
and on a public repository run logs are public. GitHub masks SECRETS but
never VARIABLES, so `${{ vars.AWS_ACCOUNT_ID }}` alone would have put the
account id in a public log on every run — the tree-clean reading would have
been true and useless. Each such job now runs `::add-mask::` on the account
id before it can be printed.

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
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The whole shipped tree, and it MEANS the whole tree (corrected 2026-09-07,
#: second pass). `tests/` was exempt until earlier today on an argument that
#: was false the moment this repo went public. The replacement listed three
#: directories — `.github/`, `crucible/`, `tests/` — while the docstring above
#: it said "the whole tree", and `README.md` was carrying a live Lambda
#: function name the entire time, in the runbook, which is the single most-read
#: file a public repository has. A scan whose scope is an enumerated list of
#: directories grows a hole every time the repo grows a directory, so the
#: scope is now REPO_ROOT with an explicit skip list — the same inversion the
#: `tests/` exemption needed, applied one level up.
SCAN_ROOTS = (REPO_ROOT,)

#: This file, and nothing else — it must contain the patterns in order to
#: search for and document them. Same exemption shape as
#: `tests/test_no_suppressions.py::SELF`: a single path compared by equality,
#: never a collection.
SELF = Path(__file__).resolve()

#: The agent-instruction files, which are SYMLINKS into the private
#: `nous-ergon-ops` repo and are gitignored here — they are never committed and
#: never ship, so a literal in them is not published by this repository.
#: `test_the_agent_instruction_files_are_not_committed` below is what makes
#: that claim checkable rather than assumed; without it this skip would be the
#: same shape as the `tests/` exemption this file spent the morning removing —
#: a carve-out resting on an unverified sentence.
NOT_SHIPPED = {"AGENTS.md", "CLAUDE.md"}

_IGNORED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".ropeproject",
    "node_modules",
    "htmlcov",
    ".mypy_cache",
}

#: `.md` is in the set and is the reason this correction exists: the leak that
#: survived two passes of this guard was in `README.md`, not in code.
_SCANNED_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini", ".md", ".sh"}

#: A 12-digit account id, but ONLY when it sits immediately after
#: `arn:aws:iam::`, a bare `::`, or an `--account`/`account` token — see the
#: module docstring ("Why the account-id pattern requires an adjacent
#: marker...") for why a bare `\b\d{12}\b` was tightened (Finding 4,
#: adversarial review of `alpha-engine-config-I9906`). A module constant
#: rather than a repeated literal: every test below that needs to reason
#: about this specific pattern imports it, so the pattern cannot drift
#: between the table and its own tests.
ACCOUNT_ID_PATTERN = r"(?:arn:aws:iam::|::|--account[ =]|\baccount[ :=])\d{12}\b"

#: The canonical AWS-documentation placeholder account ids, and the all-zero
#: one. Exempt because they name NO account: AWS publishes `123456789012` in
#: its own examples precisely so a fixture can carry an ARN-shaped string
#: without carrying an account. Needed only since 2026-09-07, when the scan
#: was extended over `tests/` (`alpha-engine-config-I10156`) and started
#: reading a suite that has always built ARNs out of these.
#:
#: An ENUMERATED set, never a heuristic. "Looks like a placeholder" is the
#: shape that lets a real id through the day someone picks a memorable one;
#: three literals that can be checked by eye cannot.
PLACEHOLDER_ACCOUNT_IDS = ("123456789012", "111111111111", "000000000000")

#: A bare `alpha-engine-*` BUCKET stem, with no `s3://` scheme. A module
#: constant for the same reason :data:`ACCOUNT_ID_PATTERN` is one: it was
#: written out four times below, and the correction of 2026-09-04 had to be
#: made in all five places or the tests would have kept asserting the old
#: pattern's behaviour while the scan ran the new one — a self-test that
#: grades a string nothing uses.
BUCKET_STEM_PATTERN = r"\balpha-engine-(data|research|crucible-v2)(?![\w-])"

#: A live IDENTITY, FUNCTION or TOPIC name. Added 2026-09-07 with Brian's
#: ruling (`alpha-engine-config-I10156`), which reversed I9906's decision that
#: role names may stay literals.
#:
#: The alternation is enumerated rather than a bare `crucible-v2-` prefix, and
#: deliberately so: `crucible/config.py`'s `DEFAULT_STACK_NAME = "crucible-v2"`
#: is a CloudFormation stack name that legitimately stays, and a pattern broad
#: enough to catch it would be deleted by the first person it blocked. `\b`
#: after each alternative, never `(?![\w-])`: `\b` matches before a hyphen, so
#: `crucible-v2-github` catches `crucible-v2-github-deploy` and its six
#: siblings, which is the whole point — the PREFIX is the identity, the
#: suffix (`-deploy`, `-board`) is a purpose word that may stay.
#: The liquidity floor's VALUE, in any spelling Python or a doc would use.
#: Added 2026-09-07, and found the hard way: after `alpha-engine-config-I10156`
#: moved `LIQUIDITY_FLOOR_USD` behind an accessor, this scan reported the tree
#: clean while `registry.py`'s `liquidity_pass_raw` entry still carried
#: `expression="dollar_volume_20d_raw >= 5_000_000"` and a docstring still said
#: what the constant "was". Both survived because every pattern here matched a
#: NAME and the leak was a NUMBER — the scan graded the identifier and the
#: value walked past it. A tuned threshold is exactly as published in an
#: expression string as it is in an assignment.
LIQUIDITY_FLOOR_VALUE_PATTERN = r"\b5[_,]?000[_,]?000(?:\.0+)?\b|\b5e6\b"

IDENTITY_NAME_PATTERN = (
    r"\b(?:crucible-v2-(?:github|runtime|dispatcher|scheduler|stack-check|pages)"
    r"|alpha-engine-alerts(?:-muted)?)\b"
)

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
    LIQUIDITY_FLOOR_VALUE_PATTERN: (
        "the liquidity floor's literal VALUE — a tuned threshold, and as "
        "published in an expression string or a docstring as in an "
        "assignment. Read it through crucible.features.compute."
        "liquidity_floor_usd(); the value is recorded in "
        "alpha-engine-config/strategy/UNIVERSE_GATES.md"
    ),
    # The machine-principal allowlist used to be a fourth repository-variable
    # example here (`CRUCIBLE_MACHINE_PRINCIPALS`); it is now a live stack
    # derivation instead (`crucible.autonomy.machine_principals`), not a
    # variable to set — see alpha-engine-config-I10307.
    IDENTITY_NAME_PATTERN: (
        "a literal IAM role, Lambda function or SNS topic name — an outside "
        "reader of this public tree can use none of them, and publishing one "
        "removes the guessing step from an AssumeRole enumeration. Resolve it "
        "through a repository variable (${{ vars.CRUCIBLE_ROLE_PREFIX }}, "
        "CRUCIBLE_PAGES_TOPIC, CRUCIBLE_MUTED_TOPIC), a live derivation off "
        "the crucible-v2 stack, or a synthetic name in a fixture"
    ),
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
            if path.name in NOT_SHIPPED and path.is_symlink():
                continue
            files.append(path)
    return files


def test_the_agent_instruction_files_are_not_committed() -> None:
    """The skip above is only sound while these files are genuinely unshipped.

    They are symlinks into the private `nous-ergon-ops` repo and are gitignored
    here, so a literal inside them is not published BY THIS REPOSITORY. That is
    a claim about git state, not about the filesystem, so it is checked rather
    than asserted in a comment — the `tests/` exemption removed earlier today
    was exactly a plausible sentence nobody had tested.
    """
    tracked = subprocess.run(
        ["git", "ls-files", "--", *sorted(NOT_SHIPPED)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert not tracked, (
        f"{tracked} is COMMITTED to this public repository. The infra-literal scan "
        "skips these files on the grounds that they never ship; that is now false, "
        "so either gitignore them again (bash nous-ergon-ops/scripts/"
        "link_agent_instructions.sh) or remove them from NOT_SHIPPED and scan them."
    )


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
    # The tracker reference for this widening lives in the module docstring,
    # never in this runtime string — `tests/test_no_stale_tracker_literals.py`.
    assert any(f.name == "README.md" for f in files), (
        "the scan did not reach README.md — the repository's most-read file, "
        "and where a live Lambda function name survived two passes of this "
        "guard because the scope was an enumerated list of directories"
    )
    assert any(f.suffix == ".py" and "tests" in f.parts for f in files), (
        "the scan did not reach tests/ — the directory whose exemption was "
        "removed once this repo went public, and the one that carried the "
        "real account id fifteen times while reporting clean"
    )


def _findings_for_file(path: Path, text: str) -> list[str]:
    findings: list[str] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for pattern, compiled in _PATTERNS.items():
            match = compiled.search(line)
            if match and pattern is ACCOUNT_ID_PATTERN:
                if any(p in match.group(0) for p in PLACEHOLDER_ACCOUNT_IDS):
                    continue
            if match:
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
        "no infrastructure identifier literal (account id, alpha-engine bucket name, "
        "IAM role / Lambda function / SNS topic name) may appear anywhere in the "
        "tree (see crucible/AGENTS.md). "
        f"{len(findings)} finding(s):\n" + "\n".join(f"  - {f}" for f in findings)
    )


def test_the_scan_can_actually_find_something() -> None:
    """The detector is shown firing. A guard nobody has made fail is a guard
    nobody knows works."""
    samples = {
        ACCOUNT_ID_PATTERN: "role/x  # arn:aws:iam::711398986525:role/x",
        r"s3://alpha-engine-": "STORE_URI: s3://alpha-engine-crucible-v2/crucible",
        IDENTITY_NAME_PATTERN: "role-to-assume: crucible-v2-github-deploy",
        LIQUIDITY_FLOOR_VALUE_PATTERN: "LIQUIDITY_FLOOR_USD = 5_000_000.0",
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
