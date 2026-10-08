"""Nothing in this repository reaches AWS Cost Explorer.

alpha-engine-config-I12168 (Brian, 2026-10-08): the fleet makes ZERO Cost
Explorer calls, permanently. Every request bills $0.01, and this package's own
reader shape issued the 44,167 calls / $441.67 of I10389. Since I11707 the gate
reads spend from the expense collector's `expenses/latest.json`
(:class:`crucible.cost.CollectorSpendClient`), and since I12168 the collector
itself reads the CUR billing export, so no Cost Explorer call remains anywhere.

`CostExplorerCache` and `CRUCIBLE_CE_CALL_BUDGET` keep bounding the reader as a
second line. This is the first: it fails on

  * a boto3 Cost Explorer client constructed in any non-test module;
  * an `aws ce ...` CLI invocation in a script, workflow or printed hint
    (a remediation string an agent follows is a call by proxy);
  * an IAM Allow of a `ce:` action in a JSON policy document.

Test modules are skipped for the first two: their fakes stand in for a client
by design. Comments are skipped: prose about the retired call is not a call.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".pytest_cache", "build"}

_CLIENT = re.compile(r"""(?:client|resource)\s*\(\s*(?:service_name\s*=\s*)?["']ce["']""")
_CLI = re.compile(r"(?:^|[\s;|&(`$\"'])aws\s+ce\s+[a-z]")


def _files(*suffixes: str, include_tests: bool = False) -> Iterator[Path]:
    for path in ROOT.rglob("*"):
        if path.suffix not in suffixes or not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if SKIP_DIRS & set(rel.parts):
            continue
        if not include_tests and (rel.parts[0] == "tests" or path.name.startswith("test_")):
            continue
        yield path


def _code_lines(path: Path) -> Iterator[tuple[int, str]]:
    for n, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        if line.lstrip().startswith("#"):
            continue
        yield n, line


def test_no_cost_explorer_client_is_constructed() -> None:
    offenders = [
        f"{p.relative_to(ROOT)}:{n}"
        for p in _files(".py")
        for n, line in _code_lines(p)
        if _CLIENT.search(line)
    ]
    assert offenders == [], offenders


def test_no_script_workflow_or_hint_runs_the_cost_explorer_cli() -> None:
    offenders = [
        f"{p.relative_to(ROOT)}:{n}"
        for p in _files(".py", ".sh", ".yml", ".yaml")
        for n, line in _code_lines(p)
        if _CLI.search(line)
    ]
    assert offenders == [], offenders


def _statements(doc: object) -> Iterator[dict]:
    if isinstance(doc, dict):
        if "Effect" in doc and ("Action" in doc or "NotAction" in doc):
            yield doc
        for v in doc.values():
            yield from _statements(v)
    elif isinstance(doc, list):
        for v in doc:
            yield from _statements(v)


def ce_allows(doc: object) -> list[str]:
    out: list[str] = []
    for st in _statements(doc):
        if st.get("Effect") != "Allow":
            continue
        actions = st.get("Action") or []
        actions = [actions] if isinstance(actions, str) else actions
        out += [a for a in actions if a.lower().startswith("ce:")]
    return out


def test_no_policy_document_allows_a_cost_explorer_action() -> None:
    offenders: list[str] = []
    for path in _files(".json", include_tests=True):
        try:
            doc = json.loads(path.read_text())
        except (ValueError, UnicodeDecodeError):
            continue
        offenders += [f"{path.relative_to(ROOT)}: {a}" for a in ce_allows(doc)]
    assert offenders == [], offenders


def test_the_default_spend_client_is_the_collector_not_cost_explorer(monkeypatch) -> None:
    from crucible import cost

    monkeypatch.setenv(cost.EXPENSES_URI_VAR, "s3://bucket/expenses/latest.json")
    assert isinstance(cost.default_client(), cost.CollectorSpendClient)


def test_the_guard_matches_what_it_claims_to() -> None:
    """A guard never shown to fire is a comment."""
    assert _CLIENT.search('ce = boto3.client("ce", region_name="us-east-1")')
    assert _CLIENT.search("boto3.client(service_name='ce')")
    assert not _CLIENT.search('boto3.client("ecs")')
    assert _CLI.search("`aws ce update-cost-allocation-tags-status --cost-allocation-tags-status")
    assert not _CLI.search("aws cloudwatch get-metric-data")
    assert ce_allows(
        {"Statement": [{"Effect": "Allow", "Action": "ce:GetCostAndUsage", "Resource": "*"}]}
    ) == ["ce:GetCostAndUsage"]
    assert ce_allows({"Statement": [{"Effect": "Deny", "Action": "ce:*", "Resource": "*"}]}) == []
