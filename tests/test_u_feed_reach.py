"""What the U champion's feed reaches, held to what the slot docs say it reaches.

`alpha-engine-config-I12691`. v1 documented its universe slot as deciding
"which names reach the predictor" while the predictor read an uncontested
literal (`predictor_universe_cut = attractiveness_top_20`) the pointer never
touched (`alpha-engine-config-I11381`). v2 carried the same sentence over. In
v2 the U cut is read by S's alpha resolution as the book's eligibility mask; M
scores the whole feature-layer cross-section and is served before U.

These tests pin that wiring by source, so the two cannot drift apart again in
silence: wiring the U feed into M (or into anything else) fails here, and the
fix is to change the slot docs in the same PR, not this test's allow-list.
"""

from __future__ import annotations

import ast
from pathlib import Path

import crucible.keys
import crucible.slots
import crucible.slots.universe
from crucible.daily_chain import DAILY_CHAIN_SERVE_ORDER

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "crucible"

#: Where the U feed's key builder may appear: its definition, the U slot that
#: WRITES the feed, and the one consumer that READS it.
U_FEED_KEY_SITES = frozenset(
    {
        "crucible/keys.py",
        "crucible/slots/universe.py",
        "crucible/slots/strategy.py",
    }
)

#: Modules that resolve the U champion pointer by its literal slot name.
U_POINTER_READERS = frozenset({"crucible/slots/strategy.py"})


def _modules() -> list[tuple[str, ast.Module]]:
    return [
        (path.relative_to(ROOT).as_posix(), ast.parse(path.read_text(encoding="utf-8")))
        for path in sorted(PACKAGE.rglob("*.py"))
    ]


def _names_u_feed_key(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "universe_members_key":
            return True
        if isinstance(node, ast.Attribute) and node.attr == "universe_members_key":
            return True
        if isinstance(node, ast.FunctionDef) and node.name == "universe_members_key":
            return True
        if isinstance(node, ast.ImportFrom) and any(
            alias.name == "universe_members_key" for alias in node.names
        ):
            return True
    return False


def _reads_u_pointer_by_literal(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "champion_key"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "u"
        ):
            return True
    return False


def test_the_u_feed_key_appears_only_at_its_writer_and_its_one_reader() -> None:
    sites = {rel for rel, tree in _modules() if _names_u_feed_key(tree)}
    assert sites == U_FEED_KEY_SITES, (
        f"`universe_members_key` is named in {sorted(sites)}; expected exactly "
        f"{sorted(U_FEED_KEY_SITES)}. A new reader of the U feed changes what a U "
        "promotion reaches: update the U row in `crucible.slots`' slot table and "
        "`crucible.slots.universe`'s docstring in the same change."
    )


def test_only_s_resolves_the_u_champion_pointer() -> None:
    readers = {rel for rel, tree in _modules() if _reads_u_pointer_by_literal(tree)}
    assert readers == U_POINTER_READERS, (
        f"`champion_key('u')` is resolved in {sorted(readers)}; expected only "
        f"{sorted(U_POINTER_READERS)}. The U champion gates S's eligibility mask "
        "and nothing else."
    )


def test_m_is_served_before_u_and_s_after_both() -> None:
    order = list(DAILY_CHAIN_SERVE_ORDER)
    assert order.index("m") < order.index("u") < order.index("s"), (
        f"DAILY_CHAIN_SERVE_ORDER is {order}. M served after U could start reading "
        "the session's cut; S must follow both, since it reads M's predictions and "
        "U's cut."
    )


def test_the_slot_docs_do_not_claim_u_reaches_the_predictor() -> None:
    docs = {
        "crucible.slots": crucible.slots.__doc__ or "",
        "crucible.slots.universe": crucible.slots.universe.__doc__ or "",
        "crucible.keys.universe_members_key": crucible.keys.universe_members_key.__doc__ or "",
    }
    claiming = sorted(name for name, doc in docs.items() if "reach the predictor" in doc)
    assert not claiming, (
        f"{claiming} say the U feed reaches the predictor; the wiring pinned above "
        "says it reaches S's eligibility mask only."
    )
