"""The runbook is checked against the code it documents, clause by clause.

`docs/NEW_EXPERIMENT.md` tells an operator which fields a recipe declares,
which rankers exist, and which commands to run. Every one of those is a
LIVE registry in this package, and a doc that names a stale one is worse
than no doc: it is a procedure that fails halfway, after the recipe is
already filed.

This repository has already paid for that class twice — `AGENTS.md` pointed
at a phase issue that had closed two hours earlier
(`alpha-engine-config-I9757`), and `README.md` published a live function
name through two passes of a scanner whose own docstring claimed it read the
whole tree (`alpha-engine-config-I10156`). Both were prose nothing executed.
These tests execute it: the field table is compared to the pydantic model,
the ranker names to `RANKERS`, and every command block to the real argument
parser, so the runbook cannot drift from the CLI without CI going red.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from crucible.cli import JOBS, build_parser
from crucible.models import ARM_RECIPE_REQUIRED_FIELDS, ArmRecipeDocument
from crucible.slots.arms import LLM_CALLSITE_PARAM, REQUIRED_ARM_FIELDS
from crucible.slots.rankers import RANKERS

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = REPO_ROOT / "docs" / "NEW_EXPERIMENT.md"
README = REPO_ROOT / "README.md"

#: One row of the runbook's field table: `| `field` | required | notes |`.
_FIELD_ROW = re.compile(r"^\|\s*`([^`]+)`(?:,\s*`([^`]+)`)?\s*\|\s*([^|]+)\|", re.MULTILINE)

#: A command line inside a fenced block, with or without the `uv run` prefix.
_COMMAND = re.compile(r"^(?:uv run )?crucible ([a-z.]+) (.*)$", re.MULTILINE)


def _text() -> str:
    return RUNBOOK.read_text(encoding="utf-8")


def _field_table_rows() -> list[tuple[str, str]]:
    """(field name, required cell) for every row of the field table."""
    body = _text().split("| Field | Required | Notes |", 1)[1].split("\n\n", 1)[0]
    rows: list[tuple[str, str]] = []
    for first, second, required in _FIELD_ROW.findall(body):
        rows.append((first, required.strip()))
        if second:
            rows.append((second, required.strip()))
    return rows


def test_the_runbook_exists_and_the_readme_links_it() -> None:
    """A runbook nothing points at is a file, not a procedure."""
    assert RUNBOOK.is_file()
    assert "docs/NEW_EXPERIMENT.md" in README.read_text(encoding="utf-8")


def test_every_documented_field_is_a_field_of_the_recipe_model() -> None:
    documented = {name for name, _ in _field_table_rows()}
    assert documented, "the field table parsed to nothing — the heading moved"
    unknown = documented - set(ArmRecipeDocument.model_fields)
    assert not unknown, (
        f"the runbook documents {sorted(unknown)}, which `ArmRecipeDocument` does not "
        'declare. `extra="forbid"`: a recipe carrying one of these is REFUSED, so the '
        "runbook is telling an operator to write a recipe that cannot load."
    )


def test_every_field_of_the_recipe_model_is_documented() -> None:
    """The inverse, and the one that actually goes stale: a field added to
    the model with no row here is a field an operator never learns to set."""
    documented = {name for name, _ in _field_table_rows()}
    missing = set(ArmRecipeDocument.model_fields) - documented
    assert not missing, f"recipe fields with no row in the runbook's table: {sorted(missing)}"


def test_the_required_column_matches_the_pre_registration_contract() -> None:
    required_in_doc = {name for name, cell in _field_table_rows() if cell.startswith("yes")}
    assert required_in_doc == set(ARM_RECIPE_REQUIRED_FIELDS)
    # The reader's own tuple and the model's must agree, or the doc is right
    # about one of them and wrong about the other.
    assert set(REQUIRED_ARM_FIELDS) == set(ARM_RECIPE_REQUIRED_FIELDS)


def test_every_ranker_the_runbook_names_is_registered() -> None:
    """The example recipes name rankers. A renamed ranker must not leave a
    copy-pasteable example that raises `unknown ranker` on first load."""
    named = set(re.findall(r"^ranker: ([\w.]+)$", _text(), re.MULTILINE))
    assert named, "no example recipe declares a ranker — the examples moved"
    unknown = named - set(RANKERS)
    assert not unknown, (
        f"the runbook's examples name {sorted(unknown)}, which is not in RANKERS. "
        f"Registered: {sorted(RANKERS)}"
    )


def test_the_llm_callsite_key_is_named_by_its_constant_value() -> None:
    assert f"params.{LLM_CALLSITE_PARAM}" in _text()


def test_every_command_in_the_runbook_parses_against_the_real_cli() -> None:
    """The strongest clause here: the parser is the oracle. A renamed flag,
    a removed job or a newly required argument fails this test rather than
    an operator's terminal."""
    parser = build_parser()
    commands = _COMMAND.findall(_text())
    assert len(commands) >= 4, f"only {len(commands)} command(s) parsed out of the runbook"
    for job, rest in commands:
        assert job in JOBS, f"the runbook names job {job!r}, which the CLI does not carry"
        # A trailing `# comment` on an example line is prose, not an argument.
        argv = [job, *rest.split("#", 1)[0].split()]
        parser.parse_args(argv)


def test_the_command_extractor_fails_on_a_flag_the_cli_does_not_have() -> None:
    """A guard nobody has made fail is a guard nobody knows works. This shows
    the clause above refuses, rather than merely accepting what is there."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["experiment.new", "--slot", "r", "--not-a-real-flag"])
