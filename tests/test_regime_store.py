"""Contracts for the regime parameter loader and write-once store helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from crucible.keys import regime_key, strategy_slot_key
from crucible.regime import RegimeParamsError
from crucible.regime_store import load_regime_params_from_store, select_window
from crucible.store import LocalStore

PARAMS = b"regime:\n  normal_at: -0.5\n  stressed_at: 0.5\n  crisis_at: 1.5\n"


def test_params_load_from_checkout_and_store_alike(tmp_path: Path) -> None:
    strategy = tmp_path / "strategy"
    (strategy / "slots").mkdir(parents=True)
    (strategy / "slots" / "regime.yaml").write_bytes(PARAMS)
    store = LocalStore(tmp_path / "store")
    store.put_bytes(strategy_slot_key("regime"), PARAMS)
    assert load_regime_params_from_store(strategy_dir=strategy) == load_regime_params_from_store(
        store=store
    )


def test_absent_regime_file_is_refused_naming_the_key(tmp_path: Path) -> None:
    with pytest.raises(RegimeParamsError, match="strategy/current/slots/regime.yaml"):
        load_regime_params_from_store(store=LocalStore(tmp_path))


def test_a_file_without_a_regime_block_is_refused(tmp_path: Path) -> None:
    store = LocalStore(tmp_path)
    store.put_bytes(strategy_slot_key("regime"), b"portfolio: {}\n")
    with pytest.raises(RegimeParamsError, match="regime.*block"):
        load_regime_params_from_store(store=store)


@pytest.mark.parametrize("missing", ["normal_at", "stressed_at", "crisis_at"])
def test_a_missing_band_key_is_refused_by_name(tmp_path: Path, missing: str) -> None:
    values = {"normal_at": -0.5, "stressed_at": 0.5, "crisis_at": 1.5}
    del values[missing]
    body = "regime:\n" + "".join(f"  {key}: {value}\n" for key, value in values.items())
    store = LocalStore(tmp_path)
    store.put_bytes(strategy_slot_key("regime"), body.encode())
    with pytest.raises(RegimeParamsError, match=missing):
        load_regime_params_from_store(store=store)


def test_regime_key_shape_and_refusal() -> None:
    assert regime_key("2026-11-20") == "regime/2026-11-20.json"
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        regime_key("20-11-2026")


def test_select_window_is_bounded_and_requires_the_day() -> None:
    sessions = [f"2026-11-{day:02d}" for day in range(1, 11)]
    assert select_window(sessions, sessions[-1], size=3) == sessions[-3:]
    with pytest.raises(ValueError, match="2026-11-20"):
        select_window(sessions, "2026-11-20")
