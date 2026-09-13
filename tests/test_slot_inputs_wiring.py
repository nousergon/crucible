"""No input machinery is declared and left unwired.

`alpha-engine-config-I9777` was filed against "an arm registers fine and can
never be graded". The first fix declared `stack_prediction_columns` and
`write_arm_predictions`, wired neither to anything outside a test, and so
reproduced the defect one frame later — the 2026-09-01 adversarial review of
`crucible-PR28` demonstrated a well-formed stacked arm dying in `train_arm`.

A test asserting today's call sites would not have caught that and will not
catch the next one: the defect is not *which* function is called, it is that
a producer was DECLARED and never reached from production code. So the two
tests here assert properties instead.

* Every public callable this module exports is reachable from `crucible/`
  outside its own module. A new writer nobody wires fails here on the day it
  is written, whatever it is called.
* Every kind the grammar admits has a resolver, and every resolver is
  actually exercised by the one panel-building seam. A kind that is
  declarable and unresolvable is `I9777` with a new name.

`alpha-engine-config-I9957` extends the first property to
`crucible/slots/model.py`, one module over, where it had never been applied:
`crucible-PR92` gave a refused M arm an `unservable` metric whose declared
home is the manifest of the job that loaded the slot, and no job loaded it.
The M cycle job now does, so the reading there is no longer "the whole module
is unreachable" — see :class:`TestTheModelPathIsNotDeclaredAndLeftUnwired`,
which asserts the loader IS reached and pins what is still orphaned.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from crucible.slots import inputs as inputs_module
from crucible.slots.inputs import (
    INPUT_KINDS,
    INPUT_RESOLVERS,
    InputRef,
    UnproducibleInputError,
    resolve_declared_inputs,
    write_arm_predictions,
)
from crucible.slots.model import FeaturePanel, load_model_recipes
from crucible.store import LocalStore

PACKAGE = Path(inputs_module.__file__).resolve().parent.parent


def _referenced_names(tree: ast.AST) -> set[str]:
    """Every bare name and attribute used anywhere under ``tree``.

    Parsed rather than grepped: a name that appears only in a docstring or a
    comment is not wiring, and mistaking prose for wiring is precisely how
    `stack_prediction_columns` came to be documented as the panel stacker
    while no production code stacked anything.
    """
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            used.add(node.attr)
    return used


def _module_paths(root: Path, module: Path) -> tuple[str, str, str]:
    """`crucible/slots/model.py` -> (`crucible.slots.model`, `crucible.slots`, `model`).

    All three spellings are needed because all three reach the module: the
    fully qualified import, `from <package> import <stem>` (which binds the
    MODULE object), and a root-relative import inside a flat package.
    """
    parts = (root.name, *module.resolve().relative_to(root.resolve()).with_suffix("").parts)
    return ".".join(parts), ".".join(parts[:-1]), parts[-1]


def _names_bound_from(tree: ast.AST, paths: tuple[str, str, str]) -> set[str]:
    """The names a file binds FROM ``dotted``, and the aliases it binds it under.

    `from crucible.slots.model import train_arm` binds `train_arm`;
    `import crucible.slots.model as m` binds the alias `m`, through which any
    attribute reference reaches the module. Everything else in the file — a
    same-named function of its own, a name imported from somewhere else — is
    not a reference to this module and must not be counted as one.
    """
    dotted, parent, stem = paths
    bound: set[str] = set()
    aliased = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in {dotted, stem}:
            bound |= {alias.asname or alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module == parent:
            aliased = aliased or any(alias.name == stem for alias in node.names)
        elif isinstance(node, ast.Import):
            aliased = aliased or any(alias.name == dotted for alias in node.names)
    if aliased:
        # An alias reaches every public name; the caller intersects with what
        # the module actually exports, so this cannot over-report.
        bound |= {"*"}
    return bound


def _entry_points(root: Path, *, module: Path) -> set[str]:
    """Names of ``module`` that production code outside it actually references.

    **Resolved per file against what that file IMPORTS from this module**, not
    by bare name. Two modules may legitimately define the same name — the S
    slot and the M slot each have a `grade_arm`, because each grades its own
    recipe type — and a scan that matched on the name alone reported the M
    slot as WIRED the day the S cycle job referenced its own `grade_arm`
    (`alpha-engine-config-I10512`, measured on this test). That is this file's
    own failure mode inverted: a detector that mistakes a name collision for
    wiring reports a gap closed that is still open, which is worse than not
    detecting it, because the pin below then reads as re-stated.
    """
    paths = _module_paths(root, module)
    used: set[str] = set()
    for path in sorted(root.rglob("*.py")):
        if path.resolve() == module.resolve():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        bound = _names_bound_from(tree, paths)
        if not bound:
            continue
        referenced = _referenced_names(tree)
        used |= referenced if "*" in bound else (referenced & bound)
    return used


def _definition_graph(module: Path) -> dict[str, set[str]]:
    """`{top-level definition: the names its body references}`.

    Classes and module-level assignments are nodes too, because wiring is not
    only a call: :data:`INPUT_RESOLVERS` reaches a resolver by holding it, and
    `InputRef.column` reaches :func:`prediction_column` from a method body.
    """
    tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
    graph: dict[str, set[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            graph.setdefault(node.name, set()).update(_referenced_names(node) - {node.name})
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    graph.setdefault(target.id, set()).update(_referenced_names(node.value))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is not None:
                graph.setdefault(node.target.id, set()).update(_referenced_names(node.value))
    return graph


def _reachable(graph: dict[str, set[str]], seeds: set[str]) -> set[str]:
    """Transitive closure of ``seeds`` over ``graph``."""
    seen: set[str] = set()
    stack = [s for s in seeds if s in graph]
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        stack.extend(n for n in graph.get(name, ()) if n not in seen)
    return seen


def _exported_callables(module_obj: object) -> list[str]:
    return sorted(
        name
        for name in module_obj.__all__  # type: ignore[attr-defined]
        if inspect.isfunction(getattr(module_obj, name))
    )


def _orphan_callables(module_obj: object) -> list[str]:
    """Every callable ``module_obj`` exports that no production code reaches.

    The whole check, for one module, in the terms
    :class:`TestNothingIsDeclaredAndLeftUnwired` established: seeds are the
    names `crucible/` references OUTSIDE the module (a re-export in
    `crucible/slots/__init__.py` is an `ast.ImportFrom` alias and an `__all__`
    string, neither of which is an `ast.Name`, so re-exporting does NOT count
    as wiring — which is exactly the distinction `alpha-engine-config-I9957`
    turns on).
    """
    module = Path(module_obj.__file__)  # type: ignore[attr-defined]
    graph = _definition_graph(module)
    seeds = _entry_points(PACKAGE, module=module)
    reachable = _reachable(graph, seeds)
    exported = _exported_callables(module_obj)
    assert exported, f"{module_obj!r} exports no callables; the check would be vacuous"
    return sorted(n for n in exported if n not in reachable)


class TestNothingIsDeclaredAndLeftUnwired:
    def test_every_exported_callable_is_reachable_from_production_code(self) -> None:
        """The measured defect, as a property of the call graph.

        At review time `grep -rn stack_prediction_columns crucible/ | grep -v
        inputs.py` returned nothing: both halves of the artifact contract
        were reachable only from `tests/`. Reachability, not a call-site
        list, so a future writer wired through a table, a decorator or a
        third function still counts — and one wired through nothing does not.
        """
        module = Path(inputs_module.__file__)
        graph = _definition_graph(module)
        seeds = _entry_points(PACKAGE, module=module)
        assert seeds & set(graph), "no production code references this module at all"
        orphans = _orphan_callables(inputs_module)
        assert orphans == [], (
            f"{orphans} are exported by `crucible.slots.inputs` and are not reachable "
            "from anything `crucible/` references. A producer with no production caller "
            "is the shape of the stacked-arm defect: an arm registers against "
            "machinery that never runs, and the failure surfaces at training as somebody "
            "else's KeyError."
        )

    def test_the_reachability_check_fails_on_an_orphan(self) -> None:
        """The detector, made to fire. A guard nobody has failed is unproven."""
        graph = {"wired": {"helper"}, "helper": set(), "orphan": set()}
        reachable = _reachable(graph, {"wired"})
        assert "helper" in reachable
        assert "orphan" not in reachable

    def test_the_producer_half_is_reached_from_the_module_that_fits_arms(self) -> None:
        """Named explicitly, because it is the half a stacked arm depends on."""
        source = Path(inputs_module.__file__).parent / "model.py"
        text = source.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(source))
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "write_arm_predictions" in called
        assert "resolve_declared_inputs" in called


class TestEveryDeclarableKindResolves:
    def test_the_resolver_table_is_total_over_the_grammar(self) -> None:
        assert set(INPUT_RESOLVERS) == set(INPUT_KINDS), (
            "a kind that can be declared and cannot be resolved registers an arm that "
            "dies when its design matrix is built"
        )

    def test_an_unwired_kind_is_refused_at_registration_naming_the_missing_producer(
        self, tmp_path, monkeypatch
    ) -> None:
        """The guard firing. A detector nobody has made fail is unproven."""
        monkeypatch.setattr(inputs_module, "INPUT_KINDS", (*INPUT_KINDS, "signals"))
        arms = tmp_path / "m"
        arms.mkdir()
        (arms / "meta.yaml").write_text(
            "\n".join(
                [
                    "slot: m",
                    "name: meta",
                    "spec:",
                    "  features: [mom_21d_ratio]",
                    "  inputs:",
                    "    - signals[thinktank_ratio]",
                    "  estimator: {kind: ridge, alpha: 1.0}",
                    "  label_horizon_trading_days: 21",
                    "  refit_cadence_trading_days: 5",
                    "  training_window: {kind: expanding, min_trading_days: 504}",
                    "  cpcv: {n_groups: 6, k_test: 2, embargo_trading_days: 2}",
                    "registered_at: '2026-06-01'",
                ]
            ),
            encoding="utf-8",
        )
        with pytest.raises(UnproducibleInputError, match="no producer wired for kind"):
            load_model_recipes(arms, feature_columns=("mom_21d_ratio",))

    def test_every_resolver_materialises_its_kind_through_the_one_seam(self, tmp_path) -> None:
        """Behavioural, and derived from the table rather than from a list.

        A fourth kind added to `INPUT_RESOLVERS` is exercised here without
        this file being edited; a resolver that silently drops its columns
        fails here rather than at somebody's training run.
        """
        store = LocalStore(tmp_path)
        base_id = "m:base:abc123"
        dates = ("2026-08-26", "2026-08-27")
        names = ("AAA", "BBB")
        for day in dates:
            write_arm_predictions(
                _Ctx(store),
                arm_id=base_id,
                trading_day=day,
                feature_version="v1",
                predicted_alpha=dict.fromkeys(names, 0.1),
            )
        panel = FeaturePanel(
            dates=dates,
            names=names,
            features={"mom_21d_ratio": np.zeros((len(dates), len(names)))},
            forward_returns=np.zeros((len(dates), len(names))),
            feature_version="v1",
        )
        refs = {
            "features": InputRef(kind="features", ref="mom_21d_ratio"),
            "predictions": InputRef(kind="predictions", ref="base"),
        }
        assert set(refs) == set(INPUT_RESOLVERS), (
            "this test enumerates one reference per declarable kind; a kind with no "
            "reference here is a kind whose resolver is never exercised"
        )
        resolved = resolve_declared_inputs(
            panel,
            store=store,
            recipe=_StubRecipe(tuple(refs.values())),
            base_arm_ids={"base": base_id},
        )
        for ref in refs.values():
            assert ref.column in resolved.features, ref.text
        assert set(resolved.resolved_inputs) == {r.text for r in refs.values()}


#: The M-path callables `crucible/` still reaches from nothing, RE-STATED
#: (`alpha-engine-config-I9957`).
#:
#: Measured 2026-09-04 on `main` (32933bf) the orphan set was `crucible.slots.
#: model.__all__`'s callables ENTIRELY — all eleven. Deliverable 1 of that
#: tracker built the M cycle job (`crucible.slots.model.produce` / `.grade`,
#: dispatched through `crucible.slots.dispatchable_slots`), so the loader now
#: HAS a production caller and `SlotRecipes.refusal_metrics` lands on a real
#: `runs/experiment.run/{day}/m/run.json`. The pin was re-stated rather than
#: deleted, exactly as the assertion it replaced demanded: the class it
#: detects — a producer declared and reached from nothing — outlives its
#: first instance.
#:
#: What is still orphaned, measured 2026-09-06 on this branch: exactly one
#: callable, `evaluate_input_completeness`. It is policy §5.3's SECOND
#: serving precondition (per-input rows and freshness), and unlike the
#: behavioural veto beside it there is no producer anywhere in this
#: repository for the `observed` mapping it grades — that is an upstream
#: data-coverage feed, not something the fitting stack can derive. It is
#: pinned here so the day one exists, this test goes red and the pin is
#: re-stated deliberately rather than the gap widening in silence.
#:
#: The tracker id is cited in prose only, never as a code literal:
#: `tests/test_no_stale_tracker_literals.py` refuses a hardcoded issue number
#: in the package, and it caught this constant on the first run.

#: The measured orphan set, as an EQUALITY. A new orphan fails this test on
#: the day it is written; a wired one fails it too, and must be removed here
#: in the same change that wires it.
#: **Corrected 2026-09-13** (`alpha-engine-config-I10512`, measured while
#: merging the S cycle job onto the M cycle job): the pin used to read
#: `["evaluate_input_completeness"]`, on a detector that matched a bound name
#: BARE — so a file importing `crucible.slots.strategy.grade_arm` under the
#: name `grade_arm` made `crucible.slots.model.grade_arm` read as wired too,
#: a name COLLISION rather than a caller. `_entry_points` (this file, above)
#: now resolves per file against what that file imports FROM THIS module, and
#: under the fixed detector every one of these reads as orphaned: none is
#: imported BY NAME from `crucible.slots.model` anywhere in `crucible/`. They
#: ARE reached in production — `crucible.track_a.handle_experiment_run` calls
#: `module.produce(...)`/`module.grade(...)` on the module
#: `crucible.slots.dispatchable_slots()` resolves at runtime — but that
#: dispatch is invisible to a static import-graph scan BY DESIGN:
#: `dispatchable_slots()`'s whole point (see its own docstring) is that a
#: slot's entry points are read off the module rather than off a call-site
#: list, so the arc grows with "no list to update" the moment a slot lands.
#: Every dispatchable slot — U, R, M, S alike — is wired exactly this way,
#: and `tests/test_weekly.py::TestDerivation` pins the property a static scan
#: cannot: that `dispatchable_slots()` finds `produce`/`grade` on all four
#: slot modules and the arc actually dispatches to each.
STILL_ORPHANED_M_CALLABLES: list[str] = [
    "cpcv_oos_ic",
    "design_panel",
    "evaluate_behavioural_veto",
    "evaluate_input_completeness",
    "grade",
    "grade_arm",
    "predict_cross_section",
    "produce",
    "produce_arm_predictions",
    "settled_training_days",
    "train_arm",
]


class TestTheModelPathIsNotDeclaredAndLeftUnwired:
    """The same property as above, applied to `crucible/slots/model.py`.

    `alpha-engine-config-I9957`: `crucible-PR92` gave a refused M arm a
    per-arm `unservable` metric (`SlotRecipes.refusal_metrics`) whose declared
    home is "the manifest of whatever job loaded the slot", and no job loaded
    the slot. The M cycle job closes that; this class keeps the guard on the
    module rather than retiring it with its first instance.

    See :data:`STILL_ORPHANED_M_CALLABLES` for why `produce`/`grade` and
    everything they call are pinned as orphans rather than asserted
    reachable: they are wired, but only through `dispatchable_slots()`'s
    dynamic dispatch, which this static scan cannot and must not try to see.
    """

    def test_the_m_loader_is_reached_from_production_code(self) -> None:
        """Deliverable 1, as a property of the call graph rather than a claim.

        `load_model_recipes` reachable from `crucible/` is what makes
        `SlotRecipes.refusal_metrics` land on a manifest a scheduled run
        writes; unreachable, the rows are well-formed and seen by nobody. It
        is the one M-path callable this STATIC scan can attest to — reached
        through `track_a._recipes_for_registration`, which needs no dynamic
        dispatch. `produce`/`grade` are reached only through
        `dispatchable_slots()`'s dynamic lookup (see the class docstring)
        and are pinned as orphans in :data:`STILL_ORPHANED_M_CALLABLES`
        rather than asserted reachable here.
        """
        from crucible.slots import model as model_module  # noqa: PLC0415 - local to this class

        orphans = _orphan_callables(model_module)
        assert "load_model_recipes" not in orphans, (
            "`load_model_recipes` is unreachable from production code again, so the M "
            "slot has no caller and a refused arm's `unservable` row reaches no "
            "manifest — the defect this module's tracker was filed for, restored."
        )

    def test_the_orphan_set_is_exactly_the_pin(self) -> None:
        """An equality, so the gap cannot rot in either direction.

        Wire something and forget to say so: red. Declare a new producer and
        wire it to nothing: red. Either way the pin is re-stated deliberately.
        """
        from crucible.slots import model as model_module  # noqa: PLC0415 - local to this class

        assert _orphan_callables(model_module) == STILL_ORPHANED_M_CALLABLES, (
            "the set of M-path callables with no production caller is no longer the "
            "pinned one. Re-state `STILL_ORPHANED_M_CALLABLES` to the measured set, in "
            "the same change that moved it."
        )

    def test_the_reachability_check_distinguishes_a_wired_loader_from_an_unwired_one(
        self, tmp_path
    ) -> None:
        """Policy §7.4: the detector, run on both sides, asserting the outputs DIFFER.

        The real functions (`_entry_points`, `_definition_graph`,
        `_reachable`) over a real two-file package, because the claim under
        test is about what this algorithm reports — not about what a mocked
        graph would. Same shape as the live condition: a module defining a
        loader, and a sibling that either calls it or only re-exports it.
        """

        def orphans_for(name: str, job_body: str) -> list[str]:
            root = tmp_path / name
            root.mkdir()
            module = root / "slot.py"
            module.write_text(
                "__all__ = ['load_recipes']\n\n\ndef load_recipes():\n    return ()\n",
                encoding="utf-8",
            )
            (root / "job.py").write_text(job_body, encoding="utf-8")
            graph = _definition_graph(module)
            reachable = _reachable(graph, _entry_points(root, module=module))
            return sorted(n for n in ["load_recipes"] if n not in reachable)

        # The live condition: `crucible/slots/__init__.py` names the loader in
        # an `ImportFrom` and an `__all__` string, and nothing calls it.
        reexport_only = orphans_for(
            "unwired", "from slot import load_recipes\n\n__all__ = ['load_recipes']\n"
        )
        # Deliverable 1: a job that loads the slot and records the refusals.
        wired = orphans_for(
            "wired",
            "from slot import load_recipes\n\n\ndef run(ctx):\n"
            "    loaded = load_recipes()\n    return loaded\n",
        )

        assert reexport_only == ["load_recipes"], (
            "a re-export was counted as production wiring — the guard would pass "
            "vacuously on exactly the condition it exists to catch"
        )
        assert wired == []
        assert reexport_only != wired, (
            "the detector reports the same verdict for a loader with a caller and a "
            "loader with only a re-export, so passing it proves nothing"
        )


class _Ctx:
    def __init__(self, store):
        self.store = store
        self.inputs: list[dict] = []
        self.outputs: list[dict] = []

    def record_input(self, key, payload, schema_version="v1") -> None:
        self.inputs.append({"key": key})

    def record_output(self, key, payload, schema_version="v1") -> None:
        self.store.put_bytes(key, payload)
        self.outputs.append({"key": key})


class _StubRecipe:
    def __init__(self, refs):
        self.name = "stub"
        self.inputs = refs


class TestTheImportTimeWiringGuardIsShownFiring:
    """`_check_every_kind_is_wired` runs at import; it used to be a bare `if`
    under `pragma: no cover` that no test could reach."""

    def test_a_kind_with_no_resolver_is_refused_by_name(self) -> None:
        with pytest.raises(RuntimeError, match=r"input kind\(s\) \['ghost'\]"):
            inputs_module._check_every_kind_is_wired(("features", "ghost"), {"features": object()})

    def test_the_committed_table_passes_its_own_guard(self) -> None:
        inputs_module._check_every_kind_is_wired(
            inputs_module.INPUT_KINDS, inputs_module.INPUT_RESOLVERS
        )
