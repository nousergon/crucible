"""Registered is not producing (`alpha-engine-config-I11030`).

Measured 2026-09-24, read-only against the v2 store. Three R arms are ACTIVE
in `arms/r/register.jsonl` and have never written under `experiments/` or
`arm_predictions/`: `scanner_predictor_direct` (filed 2026-07-13),
`scanner_top20_predictor` (2026-07-20) and `thinktank_coverage`
(2026-06-22). Every `experiment.run[r]` manifest refuses them BY NAME: each
ranker reads a column the feature catalogue declares no producer for
(`predicted_alpha_ratio`, `thinktank_rating_ratio`). Meanwhile
`experiment.register[r]` read the same four recipes and refused none.

Two halves, tested separately:

* **the class.** A U/R recipe load applies the catalogue partition
  `experiment.run` applies, so nothing registers that the run will refuse;
* **the detector.** `every_registered_arm_produces` reads UNMET while an
  ACTIVE, non-control registered arm has produced nothing past its slot's
  settle window. It covers every arm, not only those a v1 ledger row names.
"""

from __future__ import annotations

import datetime as dt

import pytest

from crucible import gate as gate_module
from crucible.keys import arm_register_key, experiments_prefix, strategy_arm_key
from crucible.registration import load_registrable_recipes
from crucible.slots import dispatchable_slots
from crucible.slots.arms import control_specs, read_register, register_arms, write_register
from crucible.slots.cycle import ARM_REFUSED_METRIC
from crucible.slots.inputs import SlotUnservableError
from crucible.store import LocalStore

#: The read day of every clause below, and the day the arms are FILED on in
#: the "mute" fixtures: 2026-07-13, 51 sessions earlier, is the live filing
#: date of `scanner_predictor_direct`.
AS_OF = dt.date(2026, 9, 23)
FILED_LONG_AGO = "2026-07-13"
FILED_LAST_WEEK = "2026-09-18"

RECIPE = """\
name: {name}
slot: r
ranker: {ranker}
registered_at: '2026-06-15'
params:
  top_n: 10
notes: fixture recipe for {name}
"""

PRODUCIBLE = ("no_agent_quant", "quant_composite")
UNDECLARED = ("scanner_predictor_direct", "predicted_alpha_direct")


def _seed(store: LocalStore, *recipes: tuple[str, str]) -> None:
    for name, ranker in recipes:
        store.put_bytes(
            strategy_arm_key("r", name),
            RECIPE.format(name=name, ranker=ranker).encode("utf-8"),
        )


@pytest.fixture
def store(tmp_path) -> LocalStore:
    return LocalStore(tmp_path)


class TestRegistrationAppliesTheRunsPartition:
    def test_an_arm_the_run_would_refuse_is_refused_at_load(self, store) -> None:
        """The disagreement, as an assertion: `experiment.register[r]` read
        4 recipes and refused 0 while `experiment.run[r]` refused 3."""
        _seed(store, PRODUCIBLE, UNDECLARED)
        load = load_registrable_recipes("r", store=store)
        assert load.names == ("no_agent_quant",)
        assert load.refused_names == ("scanner_predictor_direct",)
        assert load.refusals[0].unresolvable == ("predicted_alpha_ratio",)

    def test_the_refusal_reaches_the_manifest_as_the_runs_own_metric(self, store) -> None:
        _seed(store, PRODUCIBLE, UNDECLARED)
        (metric,) = load_registrable_recipes("r", store=store).refusal_metrics
        assert metric["name"] == ARM_REFUSED_METRIC
        assert metric["status"] == "unservable"
        assert "scanner_predictor_direct" in metric["status_reason"]

    def test_a_slot_whose_every_recipe_is_refused_raises_never_loads_empty(self, store) -> None:
        """The module's rule 5: an empty load reads like a slot whose arms
        were all already registered."""
        _seed(store, UNDECLARED)
        with pytest.raises(SlotUnservableError):
            load_registrable_recipes("r", store=store)

    def test_a_fully_producible_slot_is_unchanged(self, store) -> None:
        _seed(store, PRODUCIBLE)
        load = load_registrable_recipes("r", store=store)
        assert load.names == ("no_agent_quant",)
        assert load.refusals == () and load.refusal_metrics == ()


def _register(store: LocalStore, *, filed_on: str, names: tuple[str, ...]) -> list[str]:
    """Register ``names`` (plus the slot's controls) the way the arms were
    registered live: straight into the register, whether or not the run can
    produce them. Loaded with the partition bypassed on purpose, because the
    live rows predate it."""
    from crucible.slots import get_slot
    from crucible.slots.arms import load_arm_specs

    specs = [s for s in load_arm_specs("r", store=store) if s.name in names]
    register, _ = register_arms(
        read_register(store, "r"), specs + control_specs(get_slot("r")), filed_on=filed_on
    )
    write_register(store, "r", register)
    return [s.arm_id for s in specs]


def _produce(store: LocalStore, arm_id: str) -> None:
    store.put_bytes(f"{experiments_prefix(arm_id)}2026-09-18/shadow.json", b"{}")


def _clause(store: LocalStore, monkeypatch):
    monkeypatch.setattr(
        "crucible.slots.dispatchable_slots", lambda: {"r": dispatchable_slots()["r"]}
    )
    return gate_module._clause_every_registered_arm_produces(store, [AS_OF])


class TestTheDetector:
    def test_a_registered_arm_mute_past_its_window_reads_unmet(self, store, monkeypatch) -> None:
        """The live state. Controls are skipped: they are scored, not
        produced, and a clause that demanded their output would be red over
        the harness's own design."""
        _seed(store, PRODUCIBLE, UNDECLARED)
        quant, scanner = _register(
            store, filed_on=FILED_LONG_AGO, names=("no_agent_quant", "scanner_predictor_direct")
        )
        _produce(store, quant)
        clause = _clause(store, monkeypatch)
        assert not clause.met and not clause.unmeasurable
        assert "1 registered arm(s) MUTE" in clause.detail
        assert scanner in clause.detail and "51 sessions ago" in clause.detail
        assert quant not in clause.detail
        assert "control" not in clause.detail
        assert arm_register_key("r") in clause.evidence

    def test_met_once_every_arm_has_produced(self, store, monkeypatch) -> None:
        _seed(store, PRODUCIBLE, UNDECLARED)
        for arm_id in _register(
            store, filed_on=FILED_LONG_AGO, names=("no_agent_quant", "scanner_predictor_direct")
        ):
            _produce(store, arm_id)
        clause = _clause(store, monkeypatch)
        assert clause.met, clause.detail
        assert "all 2 active non-control" in clause.detail

    def test_an_arm_inside_its_settle_window_has_not_been_asked(self, store, monkeypatch) -> None:
        _seed(store, PRODUCIBLE)
        (quant,) = _register(store, filed_on=FILED_LAST_WEEK, names=("no_agent_quant",))
        clause = _clause(store, monkeypatch)
        assert clause.met, clause.detail
        assert f"{quant} (3/21)" in clause.detail

    def test_a_retired_arm_is_not_held_to_it(self, store, monkeypatch) -> None:
        """Deliverable 4: an arm that will never produce says so with a
        `retired` event, and then it is no longer mute."""
        _seed(store, PRODUCIBLE, UNDECLARED)
        quant, scanner = _register(
            store, filed_on=FILED_LONG_AGO, names=("no_agent_quant", "scanner_predictor_direct")
        )
        _produce(store, quant)
        register = read_register(store, "r").retire(
            scanner, "2026-09-23", "ranks on predicted_alpha_ratio, which nothing produces"
        )
        write_register(store, "r", register)
        assert _clause(store, monkeypatch).met

    def test_an_unlistable_production_prefix_is_unmeasurable(self, store, monkeypatch) -> None:
        """Unknown output is never a pass."""
        _seed(store, PRODUCIBLE)
        _register(store, filed_on=FILED_LONG_AGO, names=("no_agent_quant",))
        real = LocalStore.list_keys

        def denied(self, prefix):
            if prefix.startswith(("experiments/", "arm_predictions/")):
                raise PermissionError("AccessDenied on ListObjectsV2")
            return real(self, prefix)

        monkeypatch.setattr(LocalStore, "list_keys", denied)
        clause = _clause(store, monkeypatch)
        assert clause.unmeasurable and not clause.met

    def test_no_arm_registered_anywhere_is_unmeasurable_never_met(self, store, monkeypatch) -> None:
        """Every register absent: a clause over zero arms read nothing, and
        MET over an empty denominator is the vacuous pass."""
        clause = _clause(store, monkeypatch)
        assert clause.unmeasurable and not clause.met
        assert "zero arms" in clause.detail

    def test_phase_3_assembles_it(self) -> None:
        assert "_clause_every_registered_arm_produces" in gate_module._phase3.__code__.co_names
