"""Every recipe under `strategy/arms/` loads through EXACTLY ONE loader.

`alpha-engine-config-I9961`. `strategy/arms/` holds four sibling directories
and three recipe schemas, and until this module existed nothing asserted
which loader owned which directory. The consequence was measured, not
hypothetical: `crucible.slots.arms.REQUIRED_ARM_FIELDS` demands `ranker` and
`params`, no M recipe on `alpha-engine-config` `origin/main` carries either,
so `load_arm_specs("m", …)` raised against the real tree and
`crucible experiment.new --slot m` was broken for the whole life of the
command. Nothing surfaced it, because the M path has no production caller at
all (`alpha-engine-config-I9957`) and the only thing that would have
exercised the wrong loader is a command nobody runs.

**The assertion is a PARTITION, not a list of acceptances.** For each slot the
recipe shape that slot declares is offered to ALL THREE loaders, and exactly
one must accept it. A test that only showed the right loader accepting would
pass just as happily if a second loader also accepted — and two readers
disagreeing about one directory, with nothing saying which is authoritative,
is precisely the defect. Zero acceptances is the other half, and is what a
new slot directory with no declared reader looks like.

**These are FIXTURES, not the live tree.** `strategy/arms/` is private
strategy content (`repository-tiering-policy` test 2) and this repository is
public; it holds the shape, never the values. The fixtures below mirror the
structure of the real recipes — an `ArmSpec` for U and R, a `ModelRecipe` for
M, a `StrategyRecipe` for S. The same property is asserted against the REAL
files by `alpha-engine-config`'s
`cross-repo-checks/test_strategy_arms_crucible_contract.py`, which installs
this package and runs these very loaders over the private tree.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from crucible.slots.arms import (
    FOREIGN_RECIPE_LOADERS,
    ForeignRecipeSchemaError,
    load_arm_specs,
)
from crucible.slots.model import load_model_recipes
from crucible.slots.strategy import load_strategy_recipes

#: `slot -> the ONE loader that reads `strategy/arms/{slot}/`.
#:
#: The mapping lives here rather than in the package on purpose: a production
#: dict holding these three callables would make `crucible.slots.model` and
#: `crucible.slots.strategy` reachable from production code as an `ast.Name`,
#: and `tests/test_slot_inputs_wiring.py` pins the M path's total
#: unreachability as a measured phase-3 gap. Wiring a registry to satisfy a
#: test would report that gap closed while nothing had changed about it.
SLOT_LOADERS: dict[str, str] = {
    "u": "load_arm_specs",
    "r": "load_arm_specs",
    "m": "load_model_recipes",
    "s": "load_strategy_recipes",
}

#: The one feature column the M fixtures declare. Stated so `load_model_recipes`
#: is handed an explicit catalogue rather than the live one — the loader takes
#: the parameter for exactly this reason, and never so a caller can opt out of
#: the producibility check.
_LAYER = ("mom_21d_ratio",)


def _arm_spec_recipe(*, slot: str, name: str, ranker: str) -> dict:
    """The U/R shape: `ranker`, `params`, `registered_at`, no `spec` block."""
    return {
        "name": name,
        "slot": slot,
        "ranker": ranker,
        "registered_at": "2026-06-01",
        "params": {"top_n": 8},
    }


def _model_recipe(name: str = "residual_momentum") -> dict:
    """The M shape: a hashed `spec` block, no `ranker` and no `params`."""
    return {
        "name": name,
        "slot": "m",
        "registered_at": "2026-06-01",
        "spec": {
            "features": list(_LAYER),
            "estimator": {"kind": "ridge", "alpha": 1.0},
            "label_horizon_trading_days": 21,
            "refit_cadence_trading_days": 5,
            "training_window": {"kind": "expanding", "min_trading_days": 504},
            "cpcv": {"n_groups": 6, "k_test": 2, "embargo_trading_days": 2},
        },
    }


def _strategy_recipe(name: str = "stock_registry") -> dict:
    """The S shape: an exit chain and a named cost model inside `spec`."""
    return {
        "name": name,
        "slot": "s",
        "spec": {
            "benchmark": "SPY",
            "cost_model": {
                "name": "flat_bps_v0",
                "placeholder": True,
                "params": {
                    "half_spread_bps": 2.5,
                    "commission_bps": 0.5,
                    "slippage_bps": 10.0,
                },
            },
            "walk_forward": {
                "test_window": 21,
                "min_train": 504,
                "purge": 21,
                "embargo": 2,
                "train_mode": "expanding",
            },
            "rules": [
                {
                    "rule_id": "position_loss_floor",
                    "params": {"position_loss_floor_pct": -0.15},
                }
            ],
        },
    }


def _recipe_for(slot: str) -> dict:
    if slot in ("u", "r"):
        return _arm_spec_recipe(slot=slot, name="momentum_sleeve", ranker="momentum_sleeve")
    if slot == "m":
        return _model_recipe()
    return _strategy_recipe()


def _write(root: Path, slot: str, payload: dict) -> Path:
    """`{root}/strategy/arms/{slot}/{name}.yaml`, returning the strategy root."""
    strategy = root / "strategy"
    directory = strategy / "arms" / slot
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{payload['name']}.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )
    return strategy


def _accepting_loaders(strategy: Path, slot: str) -> list[str]:
    """Which of the three loaders read `{strategy}/arms/{slot}/` without refusing.

    Every loader is offered every directory. Asking only the declared one
    would make this a test of acceptance, and the defect under repair is a
    directory two loaders both believed they owned.

    An EMPTY result is not an acceptance. `load_strategy_recipes` globs and
    returns a tuple, so a directory of documents it cannot recognise could
    come back as `()` rather than as a raise — and "walked away quietly" must
    not read as "reads this slot", which is the same class of blindness the
    whole partition exists to remove.
    """
    arms = strategy / "arms" / slot
    accepted: list[str] = []
    candidates = (
        ("load_arm_specs", lambda: load_arm_specs(slot, strategy_dir=strategy)),
        ("load_model_recipes", lambda: load_model_recipes(arms, feature_columns=_LAYER)),
        ("load_strategy_recipes", lambda: load_strategy_recipes(arms)),
    )
    for label, call in candidates:
        try:
            result = call()
        except Exception:  # noqa: BLE001 - any refusal is a non-acceptance, whatever its class
            continue
        registered = getattr(result, "registered", result)
        if len(registered):
            accepted.append(label)
    return accepted


def _assert_exactly_one_loader(strategy: Path, slot: str, expected: str | None) -> None:
    """The contract itself, as one callable both the real rows and the §7.4
    demonstrations below run — so what the demonstrations prove is what the
    rows assert, and not a second implementation that resembles it."""
    accepted = _accepting_loaders(strategy, slot)
    assert accepted == ([expected] if expected else []), (
        f"slot {slot!r}: `strategy/arms/{slot}/` is read by "
        f"{accepted or 'NO loader'}, not by exactly {[expected] if expected else []}. "
        "Two loaders accepting one directory is the I9961 defect — the recipe's "
        "meaning then depends on which caller got there first; zero means a directory "
        "with no declared reader, which registers nothing and reports nothing."
    )


@pytest.mark.parametrize("slot", sorted(SLOT_LOADERS))
def test_every_slots_recipe_is_read_by_exactly_one_loader(slot: str, tmp_path: Path) -> None:
    strategy = _write(tmp_path, slot, _recipe_for(slot))
    _assert_exactly_one_loader(strategy, slot, SLOT_LOADERS[slot])


def test_the_contract_goes_red_when_a_recipe_matches_no_loader(tmp_path: Path) -> None:
    """Policy §7.4, half one: the same assertion, against a recipe nothing reads.

    A file carrying only the fields every schema shares — `name` and `slot` —
    is what a new slot directory looks like before anything exists to read
    it. The check must report zero rather than shrug.
    """
    strategy = _write(tmp_path, "m", {"name": "orphan", "slot": "m"})
    assert _accepting_loaders(strategy, "m") == []
    with pytest.raises(AssertionError, match="NO loader"):
        _assert_exactly_one_loader(strategy, "m", SLOT_LOADERS["m"])


def test_the_contract_goes_red_when_two_loaders_accept_one_recipe(tmp_path: Path) -> None:
    """Policy §7.4, half two: a document deliberately well-formed under TWO schemas.

    A file carrying an `ArmSpec`'s `ranker`/`params`/`registered_at` AND a
    `ModelRecipe`'s whole `spec` block satisfies both, so both loaders read
    it. Filed under a slot `load_arm_specs` serves, that ambiguity is real
    and the assertion catches it. Filed under M it cannot arise at all,
    because the slot is refused BY NAME before a byte is read — which is what
    the I9961 fix buys, stated here as a measurement of the two readings
    DIFFERING rather than as a belief about the refusal.
    """
    hybrid = {
        **_model_recipe(name="momentum_sleeve"),
        **_arm_spec_recipe(slot="u", name="momentum_sleeve", ranker="momentum_sleeve"),
    }
    hybrid["spec"] = _model_recipe()["spec"]

    ambiguous = _write(tmp_path / "u-tree", "u", {**hybrid, "slot": "u"})
    accepted_u = _accepting_loaders(ambiguous, "u")
    assert sorted(accepted_u) == ["load_arm_specs", "load_model_recipes"], (
        "a document satisfying BOTH schemas was not seen as ambiguous, so the "
        f"partition assertion proves nothing; got {accepted_u}"
    )
    with pytest.raises(AssertionError, match="Two loaders accepting one directory"):
        _assert_exactly_one_loader(ambiguous, "u", SLOT_LOADERS["u"])

    unambiguous = _write(tmp_path / "m-tree", "m", {**hybrid, "slot": "m"})
    accepted_m = _accepting_loaders(unambiguous, "m")
    assert accepted_m == ["load_model_recipes"]
    _assert_exactly_one_loader(unambiguous, "m", SLOT_LOADERS["m"])

    assert accepted_u != accepted_m, (
        "the check reports the same verdict for an unambiguous recipe and for one two "
        "loaders both read, so passing it proves nothing"
    )


class TestTheForeignSlotRefusalNamesItsLoader:
    """Deliverable 1: `load_arm_specs` refuses M and S BY NAME."""

    def test_the_m_slot_refusal_names_the_loader_that_does_read_it(self, tmp_path: Path) -> None:
        strategy = _write(tmp_path, "m", _model_recipe())
        with pytest.raises(ForeignRecipeSchemaError) as exc:
            load_arm_specs("m", strategy_dir=strategy)
        assert "crucible.slots.model.load_model_recipes" in str(exc.value)

    def test_the_s_slot_refusal_names_the_loader_that_does_read_it(self, tmp_path: Path) -> None:
        strategy = _write(tmp_path, "s", _strategy_recipe())
        with pytest.raises(ForeignRecipeSchemaError) as exc:
            load_arm_specs("s", strategy_dir=strategy)
        assert "crucible.slots.strategy.load_strategy_recipes" in str(exc.value)

    def test_the_refusal_is_about_the_slot_not_about_what_is_on_disk(self, tmp_path: Path) -> None:
        """Before the fix the failure depended on the files: an absent
        directory raised `FileNotFoundError` and a present one raised on
        `ranker`. Neither said "wrong loader", and an operator reading either
        went looking at the recipe."""
        with pytest.raises(ForeignRecipeSchemaError):
            load_arm_specs("m", strategy_dir=tmp_path / "does-not-exist")

    def test_the_refusal_is_not_extended_to_the_slots_this_loader_serves(
        self, tmp_path: Path
    ) -> None:
        """U and R are unaffected — a refusal that swallowed them would be a
        far larger outage than the defect it repairs."""
        assert set(FOREIGN_RECIPE_LOADERS) == {"m", "s"}
        strategy = _write(tmp_path, "u", _recipe_for("u"))
        (spec,) = load_arm_specs("u", strategy_dir=strategy)
        assert spec.slot == "u"
