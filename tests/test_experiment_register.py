"""`experiment.register` — the derived registration stage.

Normative source: `alpha-engine-config-I10927`. The defect, measured
2026-09-16: `experiment.new` is not an arc stage and is scheduled nowhere, so
the path from "a recipe is merged" to "an arm is scored" ran only when an
operator typed it — ten recipes merged into `alpha-engine-config`'s strategy
tree had never been registered, the U slot had graded the same three arms
every cycle since 2026-09-12, and no surface anywhere said so.

Every test below was seen failing before the code that makes it pass.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json

import pytest

import crucible.track_a as track_a
from crucible.cli import HANDLERS, JOBS
from crucible.keys import arm_register_key, strategy_arm_key
from crucible.manifest import manifest_key
from crucible.registration import RecipeLoad, load_registrable_recipes
from crucible.slots.arms import read_register
from crucible.slots.inputs import InputRefusal
from crucible.store import LocalStore
from crucible.weekly import ARC_SLOT_JOBS, arc_stages

FRIDAY = dt.date(2026, 8, 28)

RECIPE = """\
name: {name}
slot: u
ranker: {ranker}
registered_at: '2026-06-01'
params:
  top_n: {top_n}
"""


def _args(store_uri: str, *, slot: str = "u", dry_run: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        job="experiment.register",
        slot=slot,
        dry_run=dry_run,
        store=store_uri,
        strategy_dir=None,
        trading_day=FRIDAY,
        date=FRIDAY.isoformat(),
        run_mode="replay",
    )


def _seed_tree(store: LocalStore, *recipes: tuple[str, str, int]) -> None:
    """The strategy tree as the release publisher syncs it into the store."""
    for name, ranker, top_n in recipes:
        store.put_bytes(
            strategy_arm_key("u", name),
            RECIPE.format(name=name, ranker=ranker, top_n=top_n).encode("utf-8"),
        )


def _run(tmp_path, **kwargs) -> LocalStore:
    store = LocalStore(tmp_path)
    HANDLERS["experiment.register"](_args(str(tmp_path), **kwargs))
    return store


def _manifest(store: LocalStore, slot: str = "u") -> dict:
    return json.loads(
        store.get_bytes(manifest_key("experiment.register", FRIDAY.isoformat(), discriminator=slot))
    )


@pytest.fixture
def seeded(tmp_path) -> LocalStore:
    store = LocalStore(tmp_path)
    _seed_tree(
        store,
        ("momentum_sleeve", "momentum_sleeve", 8),
        ("attractiveness", "attractiveness_blend", 5),
    )
    return store


class TestItRegistersWhatTheReleaseDeclares:
    def test_a_recipe_the_register_lacks_gains_a_row(self, tmp_path, seeded) -> None:
        """The whole defect, as an assertion: two recipes are in the release's
        tree and neither is in the register, so both must be in the register
        after the stage runs — with no `--arm` and no operator."""
        assert not seeded.exists(arm_register_key("u"))
        _run(tmp_path)
        registered = {
            track_a.name_component(arm_id) for arm_id in read_register(seeded, "u").all_arms()
        }
        assert registered == {"momentum_sleeve", "attractiveness"}

    def test_the_set_is_read_from_the_release_tree_and_is_never_a_list(self, tmp_path) -> None:
        """A third recipe published into the tree needs no code change to be
        registered. A hand-written list of arms would fail this."""
        store = LocalStore(tmp_path)
        _seed_tree(store, ("momentum_sleeve", "momentum_sleeve", 8))
        _run(tmp_path)
        _seed_tree(store, ("attractiveness_mom121", "attractiveness_mom121_blend", 5))
        _run(tmp_path)
        registered = {
            track_a.name_component(arm_id) for arm_id in read_register(store, "u").all_arms()
        }
        assert registered == {"momentum_sleeve", "attractiveness_mom121"}

    def test_an_absent_recipe_tree_raises_rather_than_registering_nothing_quietly(
        self, tmp_path
    ) -> None:
        """Rule 5. "Registered 0" and "read 0" are opposite facts, and a slot
        whose release declares no arms at all is a broken release."""
        with pytest.raises(FileNotFoundError):
            _run(tmp_path)


class TestIdempotent:
    def test_a_second_run_appends_nothing_and_says_so(self, tmp_path, seeded, capsys) -> None:
        _run(tmp_path)
        first = seeded.get_bytes(arm_register_key("u"))
        capsys.readouterr()
        _run(tmp_path)
        report = json.loads(capsys.readouterr().out)
        assert seeded.get_bytes(arm_register_key("u")) == first
        assert report["registered"] == []
        assert len(report["already_present"]) == 2
        assert _manifest(seeded)["rows_out"] == 0

    def test_the_second_run_still_files_a_manifest(self, tmp_path, seeded) -> None:
        """A quiet week is `ok` with a manifest, never an absent one — an
        absent manifest past the deadline is the §4.6 absence page."""
        _run(tmp_path)
        _run(tmp_path)
        assert _manifest(seeded)["status"] == "ok"


class TestARegistrationIsNeverAPromotion:
    def test_no_pointer_or_champion_key_is_written(self, tmp_path, seeded) -> None:
        """It makes an arm SCORED, never SERVED. The arena decides the
        pointer, later and on evidence."""
        _run(tmp_path)
        assert not [k for k in seeded.list_keys() if k.startswith("champions/")]
        assert [o["key"] for o in _manifest(seeded)["outputs"]] == [arm_register_key("u")]


class TestTheManifestSaysWhatHappened:
    def test_rows_and_the_coverage_metric_are_recorded(self, tmp_path, seeded) -> None:
        _run(tmp_path)
        manifest = _manifest(seeded)
        assert manifest["rows_in"] == 2
        assert manifest["rows_out"] == 2
        metric = next(
            m for m in manifest["metrics"] if m["name"] == track_a.REGISTER_COVERAGE_METRIC
        )
        assert metric["value"] == 2.0
        assert metric["unit"] == "arms"
        assert "2 recipe(s) read" in metric["status_reason"]
        assert "0 refused" in metric["status_reason"]


class TestARefusalIsRecordedAndDoesNotFailTheStage:
    """`sota_directional_combine` is refused at registration BY RULING
    (`alpha-engine-config-I10695`), and its three producible M siblings must
    still register. The refusal reaches the manifest — as a rejection with a
    reason and as an `unservable` MetricRecord — rather than being swallowed
    or taking the slot down with it."""

    @staticmethod
    def _load_with_a_refusal(real_specs) -> RecipeLoad:
        refusal = InputRefusal(
            arm="sota_directional_combine",
            unresolvable=("directional_combine_residual_stack",),
            reason="no producer declares this input",
        )
        return RecipeLoad(
            slot="u",
            specs=real_specs,
            refusals=(refusal,),
            refusal_metrics=(
                {
                    "name": "arm_refused_at_registration",
                    "module": "crucible.slots.u",
                    "metric_type": "count",
                    "value": 1.0,
                    "unit": "inputs",
                    "n_floor": 1,
                    "status": "unservable",
                    "status_reason": "refused at registration",
                    "source_path": "strategy/arms/u/sota_directional_combine.yaml",
                    "last_updated_utc": "2026-08-28T00:00:00Z",
                },
            ),
        )

    def test_the_siblings_register_and_the_refusal_is_on_the_manifest(
        self, tmp_path, seeded, monkeypatch
    ) -> None:
        real = load_registrable_recipes("u", store=seeded).specs
        monkeypatch.setattr(
            track_a, "load_registrable_recipes", lambda slot, **_: self._load_with_a_refusal(real)
        )
        _run(tmp_path)
        manifest = _manifest(seeded)
        assert manifest["status"] == "ok", "a ruled refusal is not a failed stage"
        assert manifest["rows_in"] == 3, "the refused recipe was READ"
        assert manifest["rows_out"] == 2, "and did not register"
        assert manifest["rows_rejected"] == [
            {
                "reason": "u:sota_directional_combine unresolvable inputs "
                "['directional_combine_residual_stack']",
                "count": 1,
            }
        ]
        assert any(m["status"] == "unservable" for m in manifest["metrics"])

    def test_the_rejection_reason_is_fitted_before_the_schema_refuses_it(self) -> None:
        """`record_rejected` RAISES over 200 characters rather than
        truncating, and the overflow costs the manifest every row count. The
        job authors the string, so the job fits it."""
        refusal = InputRefusal(
            arm="x" * 400, unresolvable=("y" * 400,), reason="unresolvable input"
        )
        assert len(track_a._refusal_reason("u", refusal)) <= 200


class TestDryRun:
    def test_a_dry_run_reports_the_diff_and_writes_nothing(self, tmp_path, seeded, capsys) -> None:
        before = sorted(seeded.list_keys())
        HANDLERS["experiment.register"](_args(str(tmp_path), dry_run=True))
        report = json.loads(capsys.readouterr().out)
        assert len(report["would_register"]) == 2
        assert report["registered"] == []
        assert sorted(seeded.list_keys()) == before, (
            "a dry run writes nothing at all — not the register, not a manifest"
        )


class TestItIsAnArcStageBeforeTheRun:
    def test_it_is_slot_scoped_and_runs_before_experiment_run(self) -> None:
        """Derived from the deadline table, never asserted as a literal time:
        the arm this stage registers is the arm `experiment.run` is supposed
        to score, so it must be earlier or the first cycle scores nothing."""
        assert "experiment.register" in ARC_SLOT_JOBS
        stages = arc_stages(FRIDAY)
        latest_register = max(s.due_at for s in stages if s.job == "experiment.register")
        earliest_run = min(s.due_at for s in stages if s.job == "experiment.run")
        assert latest_register < earliest_run

    def test_it_expands_over_every_dispatchable_slot(self) -> None:
        slots = [s.slot for s in arc_stages(FRIDAY) if s.job == "experiment.register"]
        assert slots == [s.slot for s in arc_stages(FRIDAY) if s.job == "experiment.run"]

    def test_experiment_new_still_requires_its_arm(self) -> None:
        """This stage is NOT a relaxation of `experiment.new --arm`
        (`alpha-engine-config-I10696`): that flag stays required and that
        job's meaning is unchanged. The deliberate act moved up a layer, to
        the release pin — it was not removed."""
        from crucible.cli import build_parser

        with pytest.raises(SystemExit):
            build_parser().parse_args(
                ["experiment.new", "--slot", "u", "--date", FRIDAY.isoformat()]
            )
        assert JOBS["experiment.register"].name == "experiment.register"


class TestTheGapIsGradeable:
    """Deliverable 3. Until this clause existed, every reading that looked at
    arms looked at the REGISTER, so an arm that never reached it was
    invisible to all of them — the ten unregistered recipes were found by
    diffing two listings by hand."""

    @staticmethod
    def _clause(store: LocalStore, monkeypatch):
        from crucible import gate as gate_module
        from crucible.slots import dispatchable_slots

        # One slot, so the reading is about the recipe/register join rather
        # than about which slots happen to have entry points today.
        monkeypatch.setattr(
            "crucible.slots.dispatchable_slots", lambda: {"u": dispatchable_slots()["u"]}
        )
        return gate_module._clause_every_recipe_registered(store, [FRIDAY])

    def test_red_while_a_declared_recipe_has_no_register_row(
        self, tmp_path, seeded, monkeypatch
    ) -> None:
        """The detector, firing. A detector nobody has made fail is a
        detector nobody knows works."""
        clause = self._clause(seeded, monkeypatch)
        assert not clause.met and not clause.unmeasurable
        assert "u:attractiveness" in clause.detail
        assert "is scored by nothing" in clause.detail
        assert arm_register_key("u") in clause.evidence

    def test_met_once_the_stage_has_run(self, tmp_path, seeded, monkeypatch) -> None:
        _run(tmp_path)
        clause = self._clause(seeded, monkeypatch)
        assert clause.met, clause.detail
        assert "2 registrable recipe(s)" in clause.detail

    def test_an_edited_recipe_reads_red_against_its_superseded_row(
        self, tmp_path, seeded, monkeypatch
    ) -> None:
        """The join is on ARM ID — the hash of the spec — because an edited
        recipe is a NEW arm. Joining on the file name would read MET over a
        register holding only the superseded version."""
        _run(tmp_path)
        _seed_tree(seeded, ("attractiveness", "attractiveness_blend", 11))
        clause = self._clause(seeded, monkeypatch)
        assert not clause.met
        assert "u:attractiveness" in clause.detail

    def test_a_refused_recipe_is_named_but_is_not_a_gap(
        self, tmp_path, seeded, monkeypatch
    ) -> None:
        """`sota_directional_combine` does not register BY RULING. A clause
        that stayed red over a ruled outcome would be muted within a week,
        and would then be red over nothing."""
        _run(tmp_path)
        real = load_registrable_recipes("u", store=seeded)
        monkeypatch.setattr(
            "crucible.registration.load_registrable_recipes",
            lambda slot, **_: RecipeLoad(
                slot=slot,
                specs=real.specs,
                refusals=(
                    InputRefusal(
                        arm="sota_directional_combine",
                        unresolvable=("directional_combine_residual_stack",),
                        reason="no producer declares this input",
                    ),
                ),
                refusal_metrics=(),
            ),
        )
        clause = self._clause(seeded, monkeypatch)
        assert clause.met, clause.detail
        assert "1 refused at registration: u:sota_directional_combine" in clause.detail

    def test_an_unreadable_register_is_unmeasurable_never_met(self, tmp_path, monkeypatch) -> None:
        """An unknown set never grades MET."""
        from crucible import gate as gate_module

        store = LocalStore(tmp_path)
        _seed_tree(store, ("momentum_sleeve", "momentum_sleeve", 8))

        def _denied(_store, slot):
            return set(), arm_register_key(slot), "AccessDenied reading the register", True, None

        monkeypatch.setattr(gate_module, "_register_arms", _denied)
        clause = self._clause(store, monkeypatch)
        assert clause.unmeasurable and not clause.met


S_RECIPE = """\
slot: s
name: {name}
notes: fixture recipe for {name}
spec:
  benchmark: SPY
  cost_model:
    name: flat_bps_v0
    placeholder: true
    params:
      half_spread_bps: 2.5
      commission_bps: 0.5
      slippage_bps: 10.0
  rules:
    - rule_id: position_loss_floor
      params:
        position_loss_floor_pct: -0.15
"""


class TestTheSSlotIsReadAsOfTheReadDay:
    """`alpha-engine-config-I11512`. An S recipe declares no `registered_at`,
    and the loader stamps it — and consults the register at all — only when
    it is handed a trading day. The clause handed it none, so every S recipe
    refused and S read UNMET "unservable" whether or not its register
    existed: a reading that could never turn MET. And while the M champion
    pointer is absent S registers nothing BY RULING
    (`alpha-engine-config-I11452`), which the clause graded as a gap.

    Measured 2026-09-24 on `gates/phase3/2026-09-23/gate.json`: UNMET naming
    `stock_registry` and `stock_registry_sqrt_impact` as refused for want of
    a `registered_at`, with `champions/m/current.json` absent.
    """

    ARMS = ("stock_registry", "stock_registry_sqrt_impact")

    @staticmethod
    def _clause(store: LocalStore, monkeypatch):
        from crucible import gate as gate_module
        from crucible.slots import dispatchable_slots

        monkeypatch.setattr(
            "crucible.slots.dispatchable_slots", lambda: {"s": dispatchable_slots()["s"]}
        )
        return gate_module._clause_every_recipe_registered(store, [FRIDAY])

    @pytest.fixture
    def s_tree(self, tmp_path) -> LocalStore:
        store = LocalStore(tmp_path)
        for name in self.ARMS:
            store.put_bytes(strategy_arm_key("s", name), S_RECIPE.format(name=name).encode("utf-8"))
        return store

    @staticmethod
    def _seat_m_champion(store: LocalStore) -> None:
        from crucible.keys import champion_key

        store.put_bytes(champion_key("m"), b'{"arm_id": "m:fixture_model:aaaaaaaaaaaa"}')

    @staticmethod
    def _register_s(store: LocalStore, *names: str) -> None:
        """What `experiment.run[s]` does on its first cycle: load AS OF the
        day, stamp the clock, append the register rows."""
        from crucible.slots.arms import register_arms, write_register

        load = load_registrable_recipes("s", store=store, today=FRIDAY.isoformat())
        specs = [spec for spec in load.specs if spec.name in names]
        register, _ = register_arms(read_register(store, "s"), specs, filed_on=FRIDAY.isoformat())
        write_register(store, "s", register)

    def test_no_m_champion_is_a_ruled_wait_named_in_the_detail_not_a_gap(
        self, s_tree, monkeypatch
    ) -> None:
        """The live 2026-09-23 state. Red over a ruled outcome is a clause
        that gets muted, and is then red over nothing."""
        from crucible.keys import champion_key

        clause = self._clause(s_tree, monkeypatch)
        assert clause.met and not clause.unmeasurable, clause.detail
        assert "unservable" not in clause.detail
        assert "2 awaiting an upstream champion" in clause.detail
        assert "s:stock_registry, s:stock_registry_sqrt_impact" in clause.detail
        assert champion_key("m") in clause.detail
        assert champion_key("m") in clause.evidence

    def test_the_wait_ends_with_the_absence(self, s_tree, monkeypatch) -> None:
        """Once the M pointer exists, an unregistered S recipe is a gap like
        any other — the exclusion is not a standing pass for S."""
        self._seat_m_champion(s_tree)
        clause = self._clause(s_tree, monkeypatch)
        assert not clause.met and not clause.unmeasurable
        assert "s:stock_registry is declared by the release in force" in clause.detail
        assert "awaiting" not in clause.detail

    def test_a_registered_s_arm_reads_registered(self, s_tree, monkeypatch) -> None:
        """The reading that could never turn MET. Loaded with no day, both
        recipes refused and the clause read UNMET over a register holding
        both of them."""
        self._seat_m_champion(s_tree)
        self._register_s(s_tree, *self.ARMS)
        clause = self._clause(s_tree, monkeypatch)
        assert clause.met, clause.detail
        assert "all 2 registrable recipe(s)" in clause.detail

    def test_one_registered_one_not_names_only_the_gap(self, s_tree, monkeypatch) -> None:
        self._seat_m_champion(s_tree)
        self._register_s(s_tree, "stock_registry")
        clause = self._clause(s_tree, monkeypatch)
        assert not clause.met
        assert "s:stock_registry_sqrt_impact is declared" in clause.detail
        assert "s:stock_registry is declared" not in clause.detail

    def test_an_unreadable_pointer_is_unmeasurable_never_a_wait(self, s_tree, monkeypatch) -> None:
        """Whether S's unregistered recipes are a gap or a ruled wait is
        unknown when the pointer cannot be read, and an unknown never grades
        MET."""
        real_exists = LocalStore.exists

        def denied(self, key):
            if key.startswith("champions/"):
                raise PermissionError("AccessDenied on HeadObject")
            return real_exists(self, key)

        monkeypatch.setattr(LocalStore, "exists", denied)
        clause = self._clause(s_tree, monkeypatch)
        assert clause.unmeasurable and not clause.met
        assert "PermissionError" in clause.detail


class TestASlotWhereNothingIsRegistrableDoesNotStopTheArc:
    """`crucible.weekly.run_arc` stops at the FIRST stage that raises, and
    `experiment.register` runs at 11:00 — ahead of `experiment.run`,
    `experiment.grade`, `promote`, `report`, `console` and `explain`.

    S refuses every recipe outside a cycle BY DESIGN: an S arm declares no
    `registered_at`, and `experiment.run --slot s` stamps it from the first
    run that registers the arm. Measured against the live store on
    2026-09-17, `experiment.register --slot s` raised `SlotUnservableError`
    uncaught — which on 2026-09-19 would have taken the whole Saturday arc
    down over a slot behaving exactly as designed, including the producer
    phase 1's `explain_walks_a_verdict` depends on.
    """

    def test_it_files_a_reading_instead_of_raising(self, tmp_path, monkeypatch) -> None:
        from crucible import track_a
        from crucible.slots.inputs import InputRefusal, SlotUnservableError

        def refuse(*_args, **_kwargs):
            raise SlotUnservableError(
                (
                    InputRefusal(
                        arm="stock_registry",
                        unresolvable=("registered_at",),
                        reason=(
                            "declares no `registered_at`, so it has no out-of-sample clock "
                            "and no cycle trading day was supplied to stamp one from"
                        ),
                    ),
                )
            )

        monkeypatch.setattr(track_a, "load_registrable_recipes", refuse)
        store = LocalStore(tmp_path)

        assert HANDLERS["experiment.register"](_args(str(tmp_path), slot="s")) == 0

        manifest = _manifest(store, "s")
        assert manifest["status"] == "ok"
        assert manifest["rows_out"] == 0
        # RECORDED, not swallowed.
        assert len(manifest["rows_rejected"]) == 1
        coverage = [m for m in manifest["metrics"] if m["name"] == track_a.REGISTER_COVERAGE_METRIC]
        assert coverage, manifest["metrics"]
        assert coverage[-1]["status"] == "unservable"
        assert "did not fail the arc" in coverage[-1]["status_reason"]

    def test_the_refusals_are_read_structured_not_off_the_message(self) -> None:
        """`SlotUnservableError.args[0]` is the FORMATTED message, and
        iterating a str yields characters — one rejection per character, a
        reading-shaped artifact containing nothing that `record_rejected`
        raises over at 200. The structured `.refusals` field is what carries
        the arms.
        """
        from crucible.slots.inputs import InputRefusal, SlotUnservableError
        from crucible.track_a import _refusals_from

        error = SlotUnservableError(
            (
                InputRefusal(
                    arm="stock_registry", unresolvable=("registered_at",), reason="no registered_at"
                ),
                InputRefusal(
                    arm="stock_registry_sqrt_impact",
                    unresolvable=("registered_at",),
                    reason="ditto",
                ),
            )
        )
        assert _refusals_from(error) == [
            "stock_registry: no registered_at",
            "stock_registry_sqrt_impact: ditto",
        ]
