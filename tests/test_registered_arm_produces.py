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
  `experiment.run` applies. A recipe reading a column nothing will produce
  never registers. A recipe WAITING on a declared pending producer
  (`crucible.features.PENDING_COLUMNS`) registers, which is what its ranker
  says ("Registered now, servable when ..."), and the run still refuses it;
* **the detector.** `every_registered_arm_produces` reads UNMET while an
  ACTIVE, non-control registered arm has produced nothing past its slot's
  settle window and waits on no declared producer. A waiting arm is named
  with its producer and not counted mute (the 2026-09-24 ruling: the three
  arms stay, parked on purpose). The wait lapses on its own the day the
  catalogue produces the column.
"""

from __future__ import annotations

import datetime as dt

import pytest

from crucible import gate as gate_module
from crucible.features import PENDING_COLUMNS
from crucible.keys import arm_register_key, experiments_prefix, strategy_arm_key
from crucible.registration import load_registrable_recipes
from crucible.slots import dispatchable_slots
from crucible.slots.arms import control_specs, read_register, register_arms, write_register
from crucible.slots.cycle import ARM_REFUSED_METRIC, partition_by_catalog
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


def _seed(store: LocalStore, *recipes: tuple[str, str]) -> None:
    for name, ranker in recipes:
        store.put_bytes(
            strategy_arm_key("r", name),
            RECIPE.format(name=name, ranker=ranker).encode("utf-8"),
        )


@pytest.fixture
def store(tmp_path) -> LocalStore:
    return LocalStore(tmp_path)


PENDING = ("scanner_predictor_direct", "predicted_alpha_direct")
THINKTANK = ("thinktank_coverage", "thinktank_rating_direct")


@pytest.fixture
def no_pending_producer(monkeypatch) -> None:
    """`predicted_alpha_ratio` with no declared producer: the column is then
    one nothing will ever materialise, which is the defect class."""
    monkeypatch.delitem(PENDING_COLUMNS, "predicted_alpha_ratio")


@pytest.fixture
def producer_landed(monkeypatch) -> None:
    """The catalogue now produces `predicted_alpha_ratio` (track B landed)."""
    from crucible.slots import producibility

    real = producibility._catalog_columns
    monkeypatch.setattr(
        producibility, "_catalog_columns", lambda: [*real(), "predicted_alpha_ratio"]
    )


class TestRegistrationAppliesTheRunsPartition:
    def test_a_recipe_waiting_on_a_declared_producer_registers(self, store) -> None:
        """Its ranker reads "Registered now, servable when the M slot
        materializes predicted_alpha_ratio (track B)". The release declares
        it, so it registers, and the load says what it waits on."""
        _seed(store, PRODUCIBLE, PENDING, THINKTANK)
        load = load_registrable_recipes("r", store=store)
        assert set(load.names) == {
            "no_agent_quant",
            "scanner_predictor_direct",
            "thinktank_coverage",
        }
        assert load.refusals == () and load.refusal_metrics == ()
        waits = {w.arm: w.waits_on() for w in load.waiting}
        assert "predicted_alpha_ratio" in waits["scanner_predictor_direct"]
        assert "track B" in waits["scanner_predictor_direct"]
        assert "phase 5" in waits["thinktank_coverage"]

    def test_the_run_still_refuses_a_waiting_arm_and_says_what_it_waits_on(self, store) -> None:
        from crucible.features import CATALOG
        from crucible.slots.arms import load_arm_specs

        _seed(store, PRODUCIBLE, PENDING)
        _, refused = partition_by_catalog(
            load_arm_specs("r", store=store), catalog_columns=[c.name for c in CATALOG]
        )
        (refusal,) = refused
        assert refusal.arm == "scanner_predictor_direct"
        assert "track B" in refusal.reason and "waits BY DESIGN" in refusal.reason

    def test_an_arm_reading_a_column_nothing_will_produce_is_refused_at_load(
        self, store, no_pending_producer
    ) -> None:
        _seed(store, PRODUCIBLE, PENDING)
        load = load_registrable_recipes("r", store=store)
        assert load.names == ("no_agent_quant",)
        assert load.refused_names == ("scanner_predictor_direct",)
        assert load.refusals[0].unresolvable == ("predicted_alpha_ratio",)
        assert load.waiting == ()

    def test_the_refusal_reaches_the_manifest_as_the_runs_own_metric(
        self, store, no_pending_producer
    ) -> None:
        _seed(store, PRODUCIBLE, PENDING)
        (metric,) = load_registrable_recipes("r", store=store).refusal_metrics
        assert metric["name"] == ARM_REFUSED_METRIC
        assert metric["status"] == "unservable"
        assert "scanner_predictor_direct" in metric["status_reason"]
        assert "nothing will ever materialise" in metric["status_reason"]

    def test_a_slot_whose_every_recipe_is_refused_raises_never_loads_empty(
        self, store, no_pending_producer
    ) -> None:
        """The module's rule 5: an empty load reads like a slot whose arms
        were all already registered."""
        _seed(store, PENDING)
        with pytest.raises(SlotUnservableError):
            load_registrable_recipes("r", store=store)

    def test_a_slot_of_only_waiting_recipes_registers_them(self, store) -> None:
        _seed(store, PENDING)
        assert load_registrable_recipes("r", store=store).names == ("scanner_predictor_direct",)

    def test_a_fully_producible_slot_is_unchanged(self, store) -> None:
        _seed(store, PRODUCIBLE)
        load = load_registrable_recipes("r", store=store)
        assert load.names == ("no_agent_quant",)
        assert load.refusals == () and load.refusal_metrics == () and load.waiting == ()


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
        """A recipe the run CAN produce, registered and silent. Controls are
        skipped: they are scored, not produced, and a clause that demanded
        their output would be red over the harness's own design."""
        _seed(store, PRODUCIBLE, PENDING)
        quant, scanner = _register(
            store, filed_on=FILED_LONG_AGO, names=("no_agent_quant", "scanner_predictor_direct")
        )
        clause = _clause(store, monkeypatch)
        assert not clause.met and not clause.unmeasurable
        assert "1 registered arm(s) MUTE" in clause.detail
        assert quant in clause.detail and "51 sessions ago" in clause.detail
        assert "control" not in clause.detail
        assert arm_register_key("r") in clause.evidence

    def test_an_arm_waiting_on_a_declared_producer_is_named_not_mute(
        self, store, monkeypatch
    ) -> None:
        """The live state on 2026-09-24: three R arms parked on purpose."""
        _seed(store, PRODUCIBLE, PENDING, THINKTANK)
        quant, scanner, thinktank = _register(
            store,
            filed_on=FILED_LONG_AGO,
            names=("no_agent_quant", "scanner_predictor_direct", "thinktank_coverage"),
        )
        _produce(store, quant)
        clause = _clause(store, monkeypatch)
        assert clause.met, clause.detail
        assert "0 mute" in clause.detail and "MUTE" not in clause.detail
        assert "2 registered arm(s) WAITING" in clause.detail
        assert f"{scanner} waits on predicted_alpha_ratio" in clause.detail
        assert "track B" in clause.detail
        assert f"{thinktank} waits on thinktank_rating_ratio" in clause.detail
        assert "phase 5" in clause.detail
        assert "all 1 active non-control" in clause.detail

    def test_a_waiting_arm_is_still_named_when_another_is_mute(self, store, monkeypatch) -> None:
        _seed(store, PRODUCIBLE, PENDING)
        quant, scanner = _register(
            store, filed_on=FILED_LONG_AGO, names=("no_agent_quant", "scanner_predictor_direct")
        )
        clause = _clause(store, monkeypatch)
        assert not clause.met
        assert "1 registered arm(s) MUTE" in clause.detail
        assert f"{scanner} waits on predicted_alpha_ratio" in clause.detail

    def test_the_wait_lapses_once_the_catalogue_produces_the_column(
        self, store, monkeypatch, producer_landed
    ) -> None:
        """The other direction: the waiver is tied to a column the catalogue
        LACKS, so it lapses by itself, and a still-silent arm is mute."""
        _seed(store, PRODUCIBLE, PENDING)
        quant, scanner = _register(
            store, filed_on=FILED_LONG_AGO, names=("no_agent_quant", "scanner_predictor_direct")
        )
        _produce(store, quant)
        clause = _clause(store, monkeypatch)
        assert not clause.met and not clause.unmeasurable
        assert "1 registered arm(s) MUTE" in clause.detail and scanner in clause.detail
        assert "WAITING" not in clause.detail

    def test_a_column_with_no_declared_producer_earns_no_wait(
        self, store, monkeypatch, no_pending_producer
    ) -> None:
        """No arm can declare itself waiting: without a declared producer for
        the column it lacks, a silent arm is mute."""
        _seed(store, PRODUCIBLE, PENDING)
        quant, scanner = _register(
            store, filed_on=FILED_LONG_AGO, names=("no_agent_quant", "scanner_predictor_direct")
        )
        _produce(store, quant)
        clause = _clause(store, monkeypatch)
        assert not clause.met
        assert "1 registered arm(s) MUTE" in clause.detail and scanner in clause.detail

    def test_an_arm_no_recipe_in_force_hashes_to_earns_no_wait(self, store, monkeypatch) -> None:
        """The wait is read off the recipe the arm id is the hash of. An arm
        whose recipe was since edited cannot borrow its successor's wait."""
        _seed(store, PRODUCIBLE, PENDING)
        quant, scanner = _register(
            store, filed_on=FILED_LONG_AGO, names=("no_agent_quant", "scanner_predictor_direct")
        )
        _produce(store, quant)
        store.put_bytes(
            strategy_arm_key("r", "scanner_predictor_direct"),
            RECIPE.format(name="scanner_predictor_direct", ranker="predicted_alpha_direct")
            .replace("top_n: 10", "top_n: 20")
            .encode("utf-8"),
        )
        clause = _clause(store, monkeypatch)
        assert not clause.met
        assert "1 registered arm(s) MUTE" in clause.detail and scanner in clause.detail

    def test_an_unreadable_recipe_tree_is_unmeasurable_never_a_wait(
        self, store, monkeypatch
    ) -> None:
        _seed(store, PRODUCIBLE, PENDING)
        quant, _ = _register(
            store, filed_on=FILED_LONG_AGO, names=("no_agent_quant", "scanner_predictor_direct")
        )
        _produce(store, quant)

        def denied(*_args, **_kwargs):
            raise PermissionError("AccessDenied on ListObjectsV2")

        monkeypatch.setattr("crucible.slots.arms.load_arm_specs", denied)
        clause = _clause(store, monkeypatch)
        assert clause.unmeasurable and not clause.met
        assert "PermissionError" in clause.detail

    def test_only_waiting_arms_is_not_a_reading(self, store, monkeypatch) -> None:
        """Zero arms held to production is the vacuous pass, whatever waits."""
        _seed(store, PENDING)
        (scanner,) = _register(store, filed_on=FILED_LONG_AGO, names=("scanner_predictor_direct",))
        clause = _clause(store, monkeypatch)
        assert clause.unmeasurable and not clause.met
        assert f"{scanner} waits on predicted_alpha_ratio" in clause.detail

    def test_met_once_every_arm_has_produced(self, store, monkeypatch) -> None:
        _seed(store, PRODUCIBLE, PENDING)
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
        _seed(store, PRODUCIBLE, PENDING)
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
