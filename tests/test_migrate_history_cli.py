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

import inspect
import json
from pathlib import Path

import pytest

from crucible import cli, track_a
from crucible.cli import HANDLERS, is_stub, main
from crucible.keys import champion_key, manifest_key, shadow_key
from crucible.migrate import MigrationPointerConflict, MigrationSourceMissing
from crucible.slots.arms import load_arm_specs, read_register
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

R_RECIPE = """name: scanner_predictor_direct
slot: r
ranker: predicted_alpha_direct
registered_at: '2026-07-13'
bootstrap: true
promotion_source: operator_bootstrap
params:
  top_n: 10
notes: fixture recipe carrying v1 provenance
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
    store.put_bytes("strategy/current/arms/r/scanner_predictor_direct.yaml", R_RECIPE.encode())
    # The migration ADMITS a slot only once the arm the v1 champion resolves to
    # has PRODUCED (`crucible.migrate.admission_refusal`,
    # `alpha-engine-config-I10961`): a pointer seated on an arm that emits
    # nothing points the trader's contract at silence. Seeded, not waived —
    # production is what the predicate reads, so a fixture without it is a
    # fixture of a slot that is genuinely not admissible.
    for slot, name in (("u", "momentum_sleeve"), ("r", "scanner_predictor_direct")):
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
                "champion": "scanner_predictor_direct",
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
        json.dumps({"champion": "scanner_predictor_direct", "date": "2026-09-11"}).encode(),
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
        assert r_pointer["arm_id"] == _recipe_id(v2_root, "r", "scanner_predictor_direct"), (
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

        for slot, name in (("u", "momentum_sleeve"), ("r", "scanner_predictor_direct")):
            register = read_register(store, slot)
            assert _recipe_id(v2_root, slot, name) in set(register.all_arms())

        manifest = json.loads(store.get_bytes(manifest_key("migrate.history", TRADING_DAY)))
        assert manifest["status"] == "ok"
        outputs = {o["key"] for o in manifest["outputs"]}
        assert {champion_key("u"), champion_key("r")} <= outputs
        assert any(k.startswith(f"migrations/{TRADING_DAY}/") for k in outputs)

        result = _printed_result(capsys)
        assert result["pointers"] == {"u": "written", "r": "written"}
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
        }


class TestRefusals:
    def test_a_missing_recipe_raises(self, v2_root: Path, v1_root: Path) -> None:
        store = LocalStore(v2_root)
        # The slot still has a recipe; just not the one v1 names as champion.
        store.put_bytes(
            "strategy/current/arms/r/scanner_top20_predictor.yaml",
            R_RECIPE.replace("scanner_predictor_direct", "scanner_top20_predictor").encode(),
        )
        (v2_root / "strategy/current/arms/r/scanner_predictor_direct.yaml").unlink()
        with pytest.raises(MigrationSourceMissing, match="scanner_predictor_direct"):
            main(_argv(v2_root, v1_root))
        assert not store.exists(champion_key("r"))
        assert not store.exists(champion_key("u")), "nothing is written before the refusal"

    def test_an_unpublished_slot_is_not_swallowed(self, v2_root: Path, v1_root: Path) -> None:
        """The previous handler caught `FileNotFoundError` per slot and carried
        on, so a missing tree surfaced later as the wrong cause."""
        (v2_root / "strategy/current/arms/r/scanner_predictor_direct.yaml").unlink()
        with pytest.raises(FileNotFoundError, match="slot 'r'"):
            main(_argv(v2_root, v1_root))

    def test_an_existing_pointer_is_never_overwritten(self, v2_root: Path, v1_root: Path) -> None:
        store = LocalStore(v2_root)
        evidence_pointer = json.dumps(
            {"arm_id": "r:scanner_top20_predictor:abcdefabcdef", "promotion_source": "evidence"}
        ).encode()
        store.put_bytes(champion_key("r"), evidence_pointer)
        with pytest.raises(MigrationPointerConflict, match="scanner_top20_predictor"):
            main(_argv(v2_root, v1_root))
        assert store.get_bytes(champion_key("r")) == evidence_pointer

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
