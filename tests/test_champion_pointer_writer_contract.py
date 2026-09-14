"""Every writer of `champion_key(slot)` writes a document
`ChampionPointer.from_dict` accepts.

`alpha-engine-config-I10691`: `crucible.migrate.run_migrate_history` hand-
rolled `{"schema_version": "champion.v1", "champion": ...}` at
`champions/{slot}/current.json` — a shape `ChampionPointer.from_dict`
refuses outright (wrong `schema_version`; the serving-arm field is `arm_id`,
not `champion`). `crucible.champion.write_champion` was always safe by
construction: it takes a `ChampionPointer` and validates `to_dict()` against
the same `ChampionPointerDocument` `from_dict` validates against, so a
caller cannot go through it and produce a bad shape.

This module pins two things:

1. **The inventory is exhaustive.** A grep/AST scan of every `crucible/*.py`
   module for a function that both derives a key from `champion_key(...)`
   and hands that key to a raw store-write primitive
   (`put_bytes`/`compare_and_swap`/`record_output_cas`/`record_output`).
   `crucible/champion.py` (the sanctioned `write_champion` path) and
   `crucible/migrate.py` (this issue's writer) are the only two today. A
   THIRD writer appearing anywhere else and bypassing `write_champion` fails
   this test rather than being discovered by the trader.
2. **Each known writer's REAL output round-trips.** `run_migrate_history`
   run end to end through `crucible.runner.run_job`, and `write_champion`
   called directly, both produce bytes that
   `ChampionPointer.from_dict(json.loads(...))` accepts without raising.
"""

from __future__ import annotations

import ast
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from crucible.champion import ChampionPointer, read_champion_etag, write_champion
from crucible.keys import champion_key
from crucible.migrate import run_migrate_history
from crucible.runner import run_job
from crucible.store import LocalStore

REPO_ROOT = Path(__file__).resolve().parent.parent
CRUCIBLE_SRC = REPO_ROOT / "crucible"

#: `champion.py` is the sanctioned writer (`write_champion`, validated on the
#: way out); `migrate.py` is this issue's writer, dynamically exercised
#: below. A file the scan finds that is not in this set is an undocumented
#: writer of `champions/{slot}/current.json`.
KNOWN_WRITER_MODULES = frozenset({"champion.py", "migrate.py"})

_WRITE_ATTRS = ("put_bytes", "compare_and_swap", "record_output_cas", "record_output")


def _champion_key_writers(source: str) -> set[str]:
    """Attribute names of raw-write calls, in any function of ``source``,
    whose first argument is either `champion_key(...)` directly or a local
    name that function assigned from `champion_key(...)` earlier in the same
    function body.

    Deliberately per-function and value-traced rather than "the file
    contains both substrings somewhere" — several read-only modules
    (`crucible/slots/cycle.py`, `crucible/slots/model.py`, `crucible/cli.py`,
    `crucible/console/*.py`, `crucible/gate.py`) call `champion_key(...)` to
    *read* a pointer in one function and call an unrelated write primitive
    for a different key in another; a same-file substring scan would flag
    all of them as writers.
    """
    tree = ast.parse(source)
    hits: set[str] = set()

    def scan_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        champion_vars: set[str] = set()
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Assign)
                and len(child.targets) == 1
                and isinstance(child.targets[0], ast.Name)
                and isinstance(child.value, ast.Call)
                and isinstance(child.value.func, ast.Name)
                and child.value.func.id == "champion_key"
            ):
                champion_vars.add(child.targets[0].id)
        for child in ast.walk(node):
            if not (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr in _WRITE_ATTRS
                and child.args
            ):
                continue
            arg0 = child.args[0]
            is_champion_arg = (isinstance(arg0, ast.Name) and arg0.id in champion_vars) or (
                isinstance(arg0, ast.Call)
                and isinstance(arg0.func, ast.Name)
                and arg0.func.id == "champion_key"
            )
            if is_champion_arg:
                hits.add(child.func.attr)

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            scan_function(node)
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            scan_function(node)
            self.generic_visit(node)

    Visitor().visit(tree)
    return hits


class TestWriterInventoryIsExhaustive:
    def test_the_known_writer_set_matches_the_tree(self) -> None:
        found = {
            p.name
            for p in sorted(CRUCIBLE_SRC.rglob("*.py"))
            if _champion_key_writers(p.read_text())
        }
        assert found == set(KNOWN_WRITER_MODULES), (
            f"champion_key(...) is written from {sorted(found)}, not "
            f"{sorted(KNOWN_WRITER_MODULES)}. A new writer must go through "
            "`crucible.champion.write_champion` (safe by construction) or be added "
            "here with its own dynamic round-trip check below — never left as a bare "
            "put_bytes/compare_and_swap/record_output(_cas) on a champion_key()."
        )


def _from_dict_accepts(store: LocalStore, slot: str) -> ChampionPointer:
    payload: dict[str, Any] = json.loads(store.get_bytes(champion_key(slot)))
    assert payload.get("schema_version") == "champion_pointer.v1", payload.get("schema_version")
    assert "champion" not in payload, "the refused v1 shape must not reappear"
    assert "arm_id" in payload
    return ChampionPointer.from_dict(payload)


class TestKnownWritersRoundTrip:
    def test_write_champion_produces_a_document_from_dict_accepts(self, tmp_path) -> None:
        store = LocalStore(tmp_path / "store")
        pointer = ChampionPointer(
            slot="u",
            arm_id="u:momentum_sleeve:1",
            as_of="2026-08-28",
            decided_at="2026-08-28T21:00:00Z",
            run_id="01JG0000000000000000000000",
            code_sha="a" * 40,
            promotion_source="bootstrap",
            manifest_key="runs/experiment.grade/2026-08-28/run.json",
            evidence={"status": "bootstrap"},
        )
        write_champion(store, pointer, expected=read_champion_etag(store, "u"))
        round_tripped = _from_dict_accepts(store, "u")
        assert round_tripped.arm_id == "u:momentum_sleeve:1"

    def test_migrate_history_produces_a_document_from_dict_accepts(
        self, tmp_path, strategy_dir, cycle_date
    ) -> None:
        from crucible.slots.arms import load_arm_specs

        v1 = LocalStore(tmp_path / "v1")
        v1.put_bytes(
            "config/producer_champion.json",
            json.dumps(
                {
                    "champion": "momentum_sleeve",
                    "promoted_at": "2026-07-13T22:07:09Z",
                    "promotion_source": "operator_bootstrap",
                }
            ).encode(),
        )
        store = LocalStore(tmp_path / "store")
        # The fixture tree holds U recipes and the v1 source here is R's pointer;
        # a recipe for another slot is refused, so the same recipe is declared R.
        recipes = {
            s.name: replace(s, slot="r") for s in load_arm_specs("u", strategy_dir=strategy_dir)
        }

        def job(ctx: Any) -> None:
            run_migrate_history(
                ctx,
                v1_store=v1,
                slots=("r",),
                arm_recipes=recipes,
                allow_missing=True,
            )

        run_job("migrate.history", job, store=store, trading_day=cycle_date)
        pointer = _from_dict_accepts(store, "r")
        assert pointer.promotion_source == "operator_bootstrap"
        assert pointer.attestation is not None and pointer.attestation["status"] == "UNKNOWN", (
            "an import carries no contamination attestation; UNKNOWN is the refusal "
            "the S slot would apply to this same shape, not a fabricated PASS"
        )

    def test_migrate_history_maps_an_unrecognized_v1_promotion_source(
        self, tmp_path, strategy_dir, cycle_date
    ) -> None:
        """A v1 value outside the closed set (`gate_engine`, seen in
        production) must not reach `ChampionPointer.__post_init__` unmapped —
        that would raise `ValueError` and fail the whole migration run rather
        than importing the pointer as an operator-placed decision."""
        from crucible.slots.arms import load_arm_specs

        v1 = LocalStore(tmp_path / "v1")
        v1.put_bytes(
            "config/producer_champion.json",
            json.dumps(
                {
                    "champion": "momentum_sleeve",
                    "promoted_at": "2026-07-13T22:07:09Z",
                    "promotion_source": "gate_engine",
                }
            ).encode(),
        )
        store = LocalStore(tmp_path / "store")
        # The fixture tree holds U recipes and the v1 source here is R's pointer;
        # a recipe for another slot is refused, so the same recipe is declared R.
        recipes = {
            s.name: replace(s, slot="r") for s in load_arm_specs("u", strategy_dir=strategy_dir)
        }

        def job(ctx: Any) -> None:
            run_migrate_history(
                ctx,
                v1_store=v1,
                slots=("r",),
                arm_recipes=recipes,
                allow_missing=True,
            )

        run_job("migrate.history", job, store=store, trading_day=cycle_date)
        pointer = _from_dict_accepts(store, "r")
        assert pointer.promotion_source == "operator_bootstrap", (
            "an unrecognized v1 value must never be mistaken for `evidence` — a "
            "migration did not re-run the arena"
        )
        assert "gate_engine" in pointer.evidence["reason"]
