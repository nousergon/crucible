"""`crucible migrate.history` end to end, through `crucible.cli.main`.

`alpha-engine-config-I10713`. The job used to reach no working code: the CLI
table still carried a `_todo` for it, and the `crucible.track_a` handler that
silently replaced that entry defaulted the v1 store to the v2 store itself
(so every v1 source read as absent) and swallowed a missing recipe
directory. These tests drive the real CLI against a local v2 store holding a
published strategy tree and a local v1 store holding documents shaped like
the production ones measured 2026-09-14:

* `config/producer_champion.json` — `champion`, `promoted_at`,
  `promotion_source: arena_held`;
* `config/scanner_spec_champion.json` — a top-level `champion`, `decided_on`,
  `last_promoted_on: null` and NO `promoted_at`;
* dated series under `research/producer_leaderboard/` and
  `predictor/model_zoo/promotions/`.
"""

from __future__ import annotations

import datetime as dt
import inspect
import json
import re
from pathlib import Path

import pytest

from crucible import cli, track_a
from crucible.cli import HANDLERS, is_stub, main
from crucible.keys import champion_key, manifest_key, shadow_key
from crucible.migrate import MigrationPointerConflict, MigrationSourceMissing
from crucible.slots.arms import load_arm_specs, read_register
from crucible.slots.producibility import UnproducibleChampionError
from crucible.store import LocalStore

TRADING_DAY = "2026-08-28"

U_RECIPE = """name: momentum_sleeve
slot: u
ranker: momentum_sleeve
registered_at: '2026-06-01'
promotion_source: operator_bootstrap
params:
  top_n: 8
notes: fixture recipe carrying v1 provenance
"""

U_CHALLENGER = """name: tech_score_gate
slot: u
ranker: tech_score_gate
registered_at: '2026-06-01'
params:
  top_n: 8
notes: fixture challenger
"""

# The v1 R champion here is `no_agent_quant`, an arm the phase-1 feature layer
# can produce. Production's v1 pointer names `scanner_predictor_direct`, whose
# recipe the release REFUSES until track B (`R_REFUSED_RECIPE` below): this
# fixture mirrored that until `alpha-engine-config-I11085`, and so encoded the
# defect - a v1 import seating R on an arm that cannot produce - as the
# expected outcome. That shape now has its own refusal tests
# (`TestAnUnproducibleChampionIsRefusedAtImport`).
R_RECIPE = """name: no_agent_quant
slot: r
ranker: quant_composite
registered_at: '2026-07-13'
bootstrap: true
promotion_source: operator_bootstrap
params:
  top_n: 15
  momentum_weight: 0.5
  trend_weight: 0.3
  reversal_weight: 0.2
notes: fixture recipe carrying v1 provenance
"""

# Production's R recipe, verbatim in the part that matters: it ranks on
# `predicted_alpha_ratio`, a column the phase-1 feature catalogue declares no
# producer for, so `experiment.run` refuses it BY NAME at registration.
R_REFUSED_RECIPE = """name: scanner_predictor_direct
slot: r
ranker: predicted_alpha_direct
registered_at: '2026-07-13'
bootstrap: true
promotion_source: operator_bootstrap
params:
  top_n: 10
notes: fixture recipe that refuses in phase 1
"""


def _files(root: Path) -> dict[str, bytes]:
    if not root.exists():
        return {}
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.fixture
def v2_root(tmp_path: Path) -> Path:
    """A v2 store with the strategy tree published under `strategy/current/`,
    the tree the box reads (there is no checkout on a spot instance)."""
    root = tmp_path / "v2"
    store = LocalStore(root)
    store.put_bytes("strategy/current/arms/u/momentum_sleeve.yaml", U_RECIPE.encode())
    store.put_bytes("strategy/current/arms/u/tech_score_gate.yaml", U_CHALLENGER.encode())
    store.put_bytes("strategy/current/arms/r/no_agent_quant.yaml", R_RECIPE.encode())
    # The migration ADMITS a slot only once the arm the v1 champion resolves to
    # has PRODUCED (`crucible.migrate.admission_refusal`,
    # `alpha-engine-config-I10961`): a pointer seated on an arm that emits
    # nothing points the trader's contract at silence. Seeded, not waived —
    # production is what the predicate reads, so a fixture without it is a
    # fixture of a slot that is genuinely not admissible.
    for slot, name in (("u", "momentum_sleeve"), ("r", "no_agent_quant")):
        arm_id = next(
            spec.arm_id for spec in load_arm_specs(slot, store=store) if spec.name == name
        )
        store.put_bytes(shadow_key(arm_id, "2026-09-11"), json.dumps({"names": []}).encode())
    return root


@pytest.fixture
def v1_root(tmp_path: Path) -> Path:
    root = tmp_path / "v1"
    store = LocalStore(root)
    store.put_bytes(
        "config/producer_champion.json",
        json.dumps(
            {
                "schema_version": 1,
                "champion": "no_agent_quant",
                "promoted_at": "2026-07-13T22:07:09.292909+00:00",
                "promotion_source": "arena_held",
            }
        ).encode(),
    )
    store.put_bytes(
        "config/scanner_spec_champion.json",
        json.dumps(
            {
                "schema_version": 1,
                "slot_id": "scanner_spec",
                "champion": "momentum_sleeve",
                "champion_before": "momentum_sleeve",
                "decided_on": "2026-09-11",
                "decision": "hold",
                "last_promoted_on": None,
                "arms": {"momentum_sleeve": {"is_champion": True}},
            }
        ).encode(),
    )
    store.put_bytes(
        "research/producer_leaderboard/2026-09-11.json",
        json.dumps({"champion": "no_agent_quant", "date": "2026-09-11"}).encode(),
    )
    store.put_bytes(
        "predictor/model_zoo/leaderboard/latest.json",
        json.dumps(
            {
                "schema_version": 1,
                "champion_arch": {"version_id": "v3.0-meta-2026-09-11-a214ae0a"},
                "serving_champion": {"served_version": "v3.0-meta-2026-08-14-119e069b"},
                # v1's leaderboard `champion` is a METRICS block, not a name —
                # the source declares how it names its champion for exactly
                # this reason (`alpha-engine-config-I10961`).
                "champion": {"forward_days": 21, "cpcv_mean_ic": 0.105001},
            }
        ).encode(),
    )
    store.put_bytes(
        "predictor/model_zoo/promotions/2026-09-11.json",
        json.dumps(
            {
                "schema_version": 1,
                "run_date": "2026-09-11",
                "promoted_kind": "arena-pointer",
                "promoted": "spec-sota-combine-2026-09-11-753cfbed",
            }
        ).encode(),
    )
    return root


def _argv(v2_root: Path, v1_root: Path, *extra: str) -> list[str]:
    return [
        "migrate.history",
        "--store",
        str(v2_root),
        "--v1-store",
        str(v1_root),
        "--run-mode",
        "live",
        "--date",
        TRADING_DAY,
        *extra,
    ]


def _printed_result(capsys: pytest.CaptureFixture[str]) -> dict:
    """The job's printed result: the first JSON document on stdout. The runner
    prints its own dry-run summary line after it."""
    document, _ = json.JSONDecoder().raw_decode(capsys.readouterr().out)
    return document


def _run_asserted(v2_root: Path, v1_root: Path, slots: tuple[str, ...]) -> None:
    """`run_migrate_history` with slots NAMED, which is the operator path.

    The CLI names none (`slots=None`), so every refusal it meets is recorded
    as a `deferred` reason and the run exits `ok` - `migrate.history` is an arc
    stage and `run_arc` stops at the first raise
    (`alpha-engine-config-I10961`). The refusals are still refusals, and this
    is where they are shown RAISING: a caller that named its slots asserted
    they were importable.
    """
    from crucible.migrate import run_migrate_history
    from crucible.runner import run_job
    from crucible.store import read_only

    store = LocalStore(v2_root)
    recipes = {
        recipe.name: recipe for slot in ("u", "r") for recipe in load_arm_specs(slot, store=store)
    }

    def job(ctx: object) -> None:
        run_migrate_history(
            ctx,
            v1_store=read_only(LocalStore(v1_root), reason="test reads v1 read-only"),
            slots=slots,
            arm_recipes=recipes,
        )

    run_job(
        "migrate.history",
        job,
        store=store,
        trading_day=dt.date.fromisoformat(TRADING_DAY),
    )


def _recipe_id(v2_root: Path, slot: str, name: str) -> str:
    specs = load_arm_specs(slot, store=LocalStore(v2_root))
    return next(spec.arm_id for spec in specs if spec.name == name)


class TestEndToEnd:
    def test_both_pointers_land_on_the_published_recipes(
        self, v2_root: Path, v1_root: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        v1_before = _files(v1_root)
        assert main(_argv(v2_root, v1_root)) == 0
        store = LocalStore(v2_root)

        r_pointer = json.loads(store.get_bytes(champion_key("r")))
        assert r_pointer["arm_id"] == _recipe_id(v2_root, "r", "no_agent_quant"), (
            "the R pointer must name the arm the published recipe produces, or no "
            "cycle ever scores the champion"
        )
        assert r_pointer["promotion_source"] == "operator_bootstrap", (
            "v1's `arena_held` is not an evidence win in v2's closed set"
        )
        assert "arena_held" in r_pointer["evidence"]["reason"]
        assert r_pointer["as_of"] == "2026-07-13"

        u_pointer = json.loads(store.get_bytes(champion_key("u")))
        assert u_pointer["arm_id"] == _recipe_id(v2_root, "u", "momentum_sleeve")
        assert u_pointer["as_of"] == "2026-06-01", (
            "U's v1 pointer carries no installation date; the clock starts at the "
            "recipe's registered_at, never at `decided_on` (a hold) or today"
        )
        assert "registered_at" in u_pointer["evidence"]["reason"]

        for slot, name in (("u", "momentum_sleeve"), ("r", "no_agent_quant")):
            register = read_register(store, slot)
            assert _recipe_id(v2_root, slot, name) in set(register.all_arms())

        manifest = json.loads(store.get_bytes(manifest_key("migrate.history", TRADING_DAY)))
        assert manifest["status"] == "ok"
        outputs = {o["key"] for o in manifest["outputs"]}
        assert {champion_key("u"), champion_key("r")} <= outputs
        assert any(k.startswith(f"migrations/{TRADING_DAY}/") for k in outputs)

        result = _printed_result(capsys)
        # M is CONSIDERED and DEFERRED rather than absent from the map
        # (`alpha-engine-config-I10961`): its v1 champion source is declared
        # (`predictor/model_zoo/leaderboard/latest.json`) and the mapping from
        # v1's `champion_arch` onto an M `ModelRecipe` is not built, so the
        # reason is on the manifest every run instead of the slot being
        # invisible, which is what the hardcoded `("u", "r")` made it.
        assert result["pointers"] == {"u": "written", "r": "written", "m": "deferred"}
        assert "no v2 recipe was supplied" in result["deferred"]["m"]
        assert result["sources_missing"] == [], (
            "the dated M promotions series must count as found, not fail the run"
        )
        assert _files(v1_root) == v1_before, "the v1 store is read-only"

    def test_a_rerun_is_idempotent(
        self, v2_root: Path, v1_root: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(_argv(v2_root, v1_root)) == 0
        store = LocalStore(v2_root)
        watched = [
            champion_key("u"),
            champion_key("r"),
            "arms/u/register.jsonl",
            "arms/r/register.jsonl",
        ]
        before = {key: store.get_bytes(key) for key in watched}
        capsys.readouterr()

        assert main(_argv(v2_root, v1_root)) == 0
        after = {key: store.get_bytes(key) for key in watched}
        assert after == before, "a rerun must not rewrite a pointer or the register"
        assert _printed_result(capsys)["pointers"] == {
            "u": "unchanged",
            "r": "unchanged",
            "m": "deferred",
        }

    def test_a_dry_run_writes_nothing(
        self, v2_root: Path, v1_root: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        v2_before, v1_before = _files(v2_root), _files(v1_root)
        assert main(_argv(v2_root, v1_root, "--dry-run")) == 0
        assert _files(v2_root) == v2_before
        assert _files(v1_root) == v1_before
        assert _printed_result(capsys)["pointers"] == {
            "u": "would_write",
            "r": "would_write",
            "m": "deferred",
        }


class TestRefusals:
    def _drop_the_r_champions_recipe(self, v2_root: Path) -> None:
        # The slot still has a recipe; just not the one v1 names as champion.
        LocalStore(v2_root).put_bytes(
            "strategy/current/arms/r/scanner_top20_predictor.yaml",
            R_RECIPE.replace("no_agent_quant", "scanner_top20_predictor").encode(),
        )
        (v2_root / "strategy/current/arms/r/no_agent_quant.yaml").unlink()

    def test_a_missing_recipe_raises_when_the_slot_was_NAMED(
        self, v2_root: Path, v1_root: Path
    ) -> None:
        self._drop_the_r_champions_recipe(v2_root)
        store = LocalStore(v2_root)
        with pytest.raises(MigrationSourceMissing, match="no_agent_quant"):
            _run_asserted(v2_root, v1_root, ("u", "r"))
        assert not store.exists(champion_key("r"))
        assert not store.exists(champion_key("u")), "nothing is written before the refusal"

    def test_a_missing_recipe_DEFERS_on_the_scheduled_path(
        self, v2_root: Path, v1_root: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The same refusal, recorded rather than raised. The CLI names no
        slots, and `migrate.history` is a weekly arc stage: a raise here would
        kill every stage after it (`crucible-PR317` / `-I10927`). Nothing is
        written either way - that is the guarantee, and it is unchanged."""
        self._drop_the_r_champions_recipe(v2_root)
        store = LocalStore(v2_root)
        assert main(_argv(v2_root, v1_root)) == 0
        result = _printed_result(capsys)
        assert result["pointers"]["r"] == "deferred"
        assert "no_agent_quant" in result["deferred"]["r"]
        assert not store.exists(champion_key("r"))

    def test_an_unpublished_slot_is_not_swallowed(self, v2_root: Path, v1_root: Path) -> None:
        """The previous handler caught `FileNotFoundError` per slot and carried
        on, so a missing tree surfaced later as the wrong cause."""
        (v2_root / "strategy/current/arms/r/no_agent_quant.yaml").unlink()
        with pytest.raises(FileNotFoundError, match="slot 'r'"):
            main(_argv(v2_root, v1_root))

    def _seat_a_foreign_r_pointer(self, v2_root: Path) -> bytes:
        evidence_pointer = json.dumps(
            {"arm_id": "r:scanner_top20_predictor:abcdefabcdef", "promotion_source": "evidence"}
        ).encode()
        LocalStore(v2_root).put_bytes(champion_key("r"), evidence_pointer)
        return evidence_pointer

    def test_an_existing_pointer_raises_when_the_slot_was_NAMED(
        self, v2_root: Path, v1_root: Path
    ) -> None:
        evidence_pointer = self._seat_a_foreign_r_pointer(v2_root)
        with pytest.raises(MigrationPointerConflict, match="scanner_top20_predictor"):
            _run_asserted(v2_root, v1_root, ("u", "r"))
        assert LocalStore(v2_root).get_bytes(champion_key("r")) == evidence_pointer

    def test_an_existing_pointer_DEFERS_on_the_scheduled_path_and_is_never_overwritten(
        self, v2_root: Path, v1_root: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """This is the state an ORDINARY Saturday reaches: `promote` runs an
        hour after this stage and moves the same pointer on the arena's own
        evidence, so from the following week every run of this stage would
        have raised - and taken `promote`, `report`, `console`, `explain`,
        `iac.conformance` and `drift` down with it."""
        evidence_pointer = self._seat_a_foreign_r_pointer(v2_root)
        assert main(_argv(v2_root, v1_root)) == 0
        result = _printed_result(capsys)
        assert result["pointers"]["r"] == "deferred"
        assert "scanner_top20_predictor" in result["deferred"]["r"]
        assert LocalStore(v2_root).get_bytes(champion_key("r")) == evidence_pointer

    def test_a_v1_store_equal_to_the_v2_store_is_refused(self, v2_root: Path) -> None:
        argv = _argv(v2_root, v2_root)
        assert main(argv) == cli.USAGE_EXIT_CODE
        assert not LocalStore(v2_root).exists(champion_key("r"))

    def test_no_v1_store_and_no_data_bucket_is_refused(
        self, v2_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("CRUCIBLE_ARCTIC_BUCKET", raising=False)
        argv = [
            "migrate.history",
            "--store",
            str(v2_root),
            "--run-mode",
            "live",
            "--date",
            TRADING_DAY,
        ]
        assert main(argv) == cli.USAGE_EXIT_CODE


class TestAnUnproducibleChampionIsRefusedAtImport:
    """`alpha-engine-config-I11085`. The 2026-09-14 import seated R on
    `scanner_predictor_direct`, whose own recipe refuses BY NAME in phase 1,
    and the 2026-09-19 arc died at `experiment.run[r]` resolving it. The
    import must refuse at import, naming the arm and why.

    The fixture SEEDS a shadow for the refused arm, deliberately: the
    runtime admission predicate (`admission_refusal`, "has it produced?") is
    then satisfied, so these tests fail unless the recipe-level refusal is
    what stops the import. A stale or backfilled shadow is exactly how a
    has-it-produced check can be satisfied by an arm that will never produce
    again.
    """

    @pytest.fixture
    def production_shaped(self, v2_root: Path, v1_root: Path) -> tuple[Path, Path, str]:
        store = LocalStore(v2_root)
        store.put_bytes(
            "strategy/current/arms/r/scanner_predictor_direct.yaml", R_REFUSED_RECIPE.encode()
        )
        refused_id = _recipe_id(v2_root, "r", "scanner_predictor_direct")
        store.put_bytes(shadow_key(refused_id, "2026-09-11"), json.dumps({"names": []}).encode())
        v1 = LocalStore(v1_root)
        pointer = json.loads(v1.get_bytes("config/producer_champion.json"))
        v1.put_bytes(
            "config/producer_champion.json",
            json.dumps({**pointer, "champion": "scanner_predictor_direct"}).encode(),
        )
        return v2_root, v1_root, refused_id

    def test_the_scheduled_path_writes_no_pointer_and_names_the_arm_and_why(
        self, production_shaped: tuple[Path, Path, str], capsys: pytest.CaptureFixture[str]
    ) -> None:
        v2_root, v1_root, refused_id = production_shaped
        assert main(_argv(v2_root, v1_root)) == 0
        store = LocalStore(v2_root)
        assert not store.exists(champion_key("r")), (
            "a v1 pointer must never be imported onto an arm the release cannot produce"
        )
        assert refused_id not in set(read_register(store, "r").all_arms())
        result = _printed_result(capsys)
        assert result["pointers"]["r"] == "deferred"
        why = result["deferred"]["r"]
        assert refused_id in why
        assert "predicted_alpha_ratio" in why, "the reason must name the missing column"
        assert "cannot produce" in why
        # U is unaffected: a refusal on one slot defers that slot only.
        assert result["pointers"]["u"] == "written"
        manifest = json.loads(store.get_bytes(manifest_key("migrate.history", TRADING_DAY)))
        metric = next(m for m in manifest["metrics"] if m["name"] == "arms_migrated")
        assert refused_id in metric["status_reason"]

    def test_the_asserted_path_raises_naming_the_arm_and_writes_nothing(
        self, production_shaped: tuple[Path, Path, str]
    ) -> None:
        v2_root, v1_root, refused_id = production_shaped
        store = LocalStore(v2_root)
        with pytest.raises(UnproducibleChampionError, match=re.escape(refused_id)) as excinfo:
            _run_asserted(v2_root, v1_root, ("u", "r"))
        assert "predicted_alpha_ratio" in str(excinfo.value)
        assert not store.exists(champion_key("r"))
        assert not store.exists(champion_key("u")), "nothing is written before the refusal"

    def test_a_rerun_over_the_pointer_it_already_seated_reports_the_refusal(
        self, production_shaped: tuple[Path, Path, str], capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Today's production state: the 2026-09-14 import already seated R on
        the refused arm. A rerun must not report that pointer `unchanged` as
        though it were sound - and must not overwrite it either (a v1 import
        never moves an existing pointer)."""
        v2_root, v1_root, refused_id = production_shaped
        store = LocalStore(v2_root)
        seated = json.dumps(
            {"arm_id": refused_id, "evidence": {"status": "migrated"}}, sort_keys=True
        ).encode()
        store.put_bytes(champion_key("r"), seated)
        assert main(_argv(v2_root, v1_root)) == 0
        result = _printed_result(capsys)
        assert result["pointers"]["r"] == "deferred"
        assert refused_id in result["deferred"]["r"]
        assert store.get_bytes(champion_key("r")) == seated


class TestTheStubIsGone:
    """Mutation guard: the `_todo` path, and the second handler that shadowed
    it, must not come back."""

    def test_the_dispatched_handler_is_the_real_one(self) -> None:
        handler = HANDLERS["migrate.history"]
        assert not is_stub(handler)
        assert handler is cli._migrate_history

    def test_no_other_module_registers_a_second_handler(self) -> None:
        assert "migrate.history" not in track_a.HANDLERS, (
            "`HANDLERS.update(TRACK_A_HANDLERS)` runs after the CLI table is built, so "
            "a track-A entry would silently replace the CLI's handler again"
        )

    def test_the_cli_source_no_longer_carries_a_stub_for_it(self) -> None:
        source = inspect.getsource(cli)
        assert '_todo(\n        "migrate.history"' not in source
        assert '_todo("migrate.history"' not in source
