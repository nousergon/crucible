"""`--arm` takes the form the operator is actually holding.

`experiment.new` PRINTS registered arm ids — `{slot}:{name}:{spec_hash}` —
and both selectors then compared that string to `spec.name`, so pasting the
id the previous command had just printed produced `no arm named 'u:…:…'`.
The CLI's own metavar said `ARM_ID` while its help text said "by name": the
two documented forms disagreed, and the one the code accepted was the one
neither the terminal nor the runbook hands you.

Nothing tested this at all — `grep -rn "arm_name=" tests/` was empty before
this file — which is the whole reason it survived. Both call sites now
resolve the selector through `crucible.slots.arm_name`, the one parser that
knows both shapes and RAISES on a string that is neither.
"""

from __future__ import annotations

import json

import pytest
from conftest import sessions_ending

from crucible.cli import main
from crucible.config import Settings
from crucible.data import run_daily
from crucible.keys import shadow_key
from crucible.runner import run_job
from crucible.slots import universe
from crucible.slots.arms import load_arm_specs
from crucible.slots.cycle import MissingArtifactError


def _one_arm(strategy_dir):
    return load_arm_specs("u", strategy_dir=strategy_dir)[0]


def _registered_ids(capsys) -> list[str]:
    """The `registered` list from the handler's own JSON line.

    Located by its key rather than by position: `run_job` prints its own
    lines around it, and a positional read would grade whichever line the
    runner happens to print last.
    """
    out = capsys.readouterr().out
    decoder = json.JSONDecoder()
    for start in (i for i, ch in enumerate(out) if ch == "{"):
        try:
            document, _ = decoder.raw_decode(out[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(document, dict) and "registered" in document:
            return document["registered"]
    raise AssertionError(f"experiment.new printed no `registered` document; stdout was:\n{out}")


def _register(strategy_dir, store_root, selector: str) -> int:
    return main(
        [
            "experiment.new",
            "--slot",
            "u",
            "--arm",
            selector,
            "--run-mode",
            "live",
            "--store",
            str(store_root),
            "--strategy-dir",
            str(strategy_dir),
            "--date",
            "2026-08-28",
        ]
    )


class TestExperimentNewTakesEitherForm:
    def test_a_bare_name_registers_the_arm(self, strategy_dir, tmp_path, capsys) -> None:
        arm = _one_arm(strategy_dir)
        assert _register(strategy_dir, tmp_path / "store-name", arm.name) == 0
        assert _registered_ids(capsys) == [arm.arm_id]

    def test_the_printed_id_registers_the_same_arm(self, strategy_dir, tmp_path, capsys) -> None:
        """The regression: this is the string the previous run printed."""
        arm = _one_arm(strategy_dir)
        assert _register(strategy_dir, tmp_path / "store-id", arm.arm_id) == 0
        assert _registered_ids(capsys) == [arm.arm_id]

    def test_an_unknown_name_still_refuses(self, strategy_dir, tmp_path) -> None:
        """A selector matching nothing must not register everything, and must
        not exit 0 — registering nothing looks exactly like registering it."""
        with pytest.raises(KeyError, match="no arm named"):
            _register(strategy_dir, tmp_path / "store-missing", "not_a_filed_arm")

    def test_a_selector_that_is_neither_shape_is_refused_by_the_parser(
        self, strategy_dir, tmp_path
    ) -> None:
        """`u:momentum_sleeve` has a colon and two parts: neither a bare name
        nor a derived id. Guessing which half is the name is how an
        unparseable id gets reported as a non-control and becomes eligible to
        serve, so it raises here instead."""
        with pytest.raises(ValueError, match="neither a bare name nor a"):
            _register(strategy_dir, tmp_path / "store-malformed", "u:momentum_sleeve")


class TestExperimentRunTakesEitherForm:
    def _settings(self, strategy_dir, store_root) -> Settings:
        return Settings(
            store_uri=str(store_root),
            arctic_bucket="unused-in-this-test",
            strategy_dir=strategy_dir,
            origins={"store_uri": "test", "strategy_dir": "test"},
        )

    def _seed(self, store, source, cycle_date) -> None:
        for day in sessions_ending(cycle_date, 1):
            run_job(
                "data.daily",
                lambda c: run_daily(c, source=source, expected_symbols=source.symbols()),
                store=store,
                trading_day=day,
            )

    def test_the_printed_id_narrows_the_run_to_that_arm(
        self, store, source, strategy_dir, cycle_date, tmp_path
    ) -> None:
        self._seed(store, source, cycle_date)
        settings = self._settings(strategy_dir, tmp_path / "store")
        arm = _one_arm(strategy_dir)
        others = [s for s in load_arm_specs("u", strategy_dir=strategy_dir) if s.name != arm.name]
        run_job(
            "experiment.run",
            lambda c: universe.produce(c, settings=settings, arm_name=arm.arm_id),
            store=store,
            trading_day=cycle_date,
        )
        assert store.exists(shadow_key(arm.arm_id, cycle_date.isoformat()))
        for other in others:
            assert not store.exists(shadow_key(other.arm_id, cycle_date.isoformat())), (
                f"{other.name} produced a shadow for a run narrowed to {arm.name} — the "
                "selector widened instead of narrowing"
            )

    def test_an_unknown_selector_raises_rather_than_producing_nothing(
        self, store, source, strategy_dir, cycle_date, tmp_path
    ) -> None:
        self._seed(store, source, cycle_date)
        settings = self._settings(strategy_dir, tmp_path / "store")
        with pytest.raises(MissingArtifactError, match="no arm named"):
            run_job(
                "experiment.run",
                lambda c: universe.produce(c, settings=settings, arm_name="not_a_filed_arm"),
                store=store,
                trading_day=cycle_date,
            )
