"""Store-facing loaders and write-once persistence for market-regime readings."""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, cast

import pandas as pd  # pyright: ignore[reportMissingTypeStubs]
import yaml

from crucible.calendar import is_trading_day
from crucible.features.compute import read_features
from crucible.keys import features_key, features_prefix, regime_key, strategy_slot_key
from crucible.regime import (
    REGIME_HISTORY_SESSIONS,
    REGIME_SCHEMA,
    REQUIRED_COLUMNS,
    RegimeParams,
    RegimeParamsError,
)
from crucible.runner import RunContext
from crucible.store import ETAG_ABSENT, Store


class RegimeConflictError(RuntimeError):
    """An existing write-once regime document differs from the proposed body."""


def load_regime_params(path: Path | str) -> RegimeParams:
    """Load the required ``regime`` block from a strategy checkout."""
    resolved = Path(path)
    if not resolved.exists():
        raise RegimeParamsError(f"no regime parameter set at {resolved}; there is no default")
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or "regime" not in payload:
        raise RegimeParamsError(
            f"{resolved}: expected a mapping carrying a 'regime' block; the file declares none."
        )
    block = cast(Mapping[str, object], payload)["regime"]
    return RegimeParams.from_mapping(cast(Mapping[str, object], block), source=str(resolved))


def load_regime_params_from_store(
    *, store: Store | None = None, strategy_dir: Path | str | None = None
) -> RegimeParams:
    """Load regime bands from a checkout when configured, otherwise the synced store."""
    if strategy_dir is not None:
        return load_regime_params(Path(strategy_dir) / "slots" / "regime.yaml")
    if store is None:
        raise ValueError("load_regime_params_from_store needs either a store or a strategy_dir")
    key = strategy_slot_key("regime")
    if not store.exists(key):
        raise RegimeParamsError(
            f"no regime parameter set at store key {key!r}; there is no default"
        )
    payload = yaml.safe_load(store.get_bytes(key).decode("utf-8"))
    if not isinstance(payload, Mapping) or "regime" not in payload:
        raise RegimeParamsError(
            f"{key}: expected a mapping carrying a 'regime' block; the file declares none."
        )
    block = cast(Mapping[str, object], payload)["regime"]
    return RegimeParams.from_mapping(cast(Mapping[str, object], block), source=key)


def compiled_sessions(store: Store, version: str) -> list[str]:
    """Return the sorted ISO session names materialized for a feature version."""
    prefix = features_prefix(version)
    return sorted(
        key[len(prefix) : -len(".parquet")]
        for key in store.list_keys(prefix)
        if key.endswith(".parquet")
    )


def select_window(
    sessions: Sequence[str], day: str, size: int = REGIME_HISTORY_SESSIONS
) -> list[str]:
    """Select at most ``size`` compiled sessions ending at ``day``."""
    if day not in sessions:
        raise ValueError(f"no compiled feature session for {day}")
    return [session for session in sessions if session <= day][-size:]


def assert_no_gaps(window: Sequence[str]) -> None:
    """Refuse the first missing exchange session inside a compiled window."""
    if not window:
        return
    present = set(window)
    day = dt.date.fromisoformat(window[0])
    end = dt.date.fromisoformat(window[-1])
    while day <= end:
        if is_trading_day(day) and day.isoformat() not in present:
            raise ValueError(f"compiled feature window is missing trading day {day.isoformat()}")
        day += dt.timedelta(days=1)


def read_regime_frame(
    store: Store, version: str, days: Sequence[str], ctx: RunContext | None = None
) -> pd.DataFrame:
    """Read and concatenate each day's independent feature cross-section."""
    frames: list[pd.DataFrame] = []
    columns = ["ticker", *REQUIRED_COLUMNS]
    for day in days:
        key = features_key(version, day)
        payload = store.get_bytes(key)
        if ctx is not None:
            ctx.record_input(key, payload, schema_version="features.v1")
        frame = read_features(payload)
        frames.append(cast(pd.DataFrame, frame.loc[:, list(dict.fromkeys(columns))]))
    if not frames:
        return pd.DataFrame({name: pd.Series(dtype="object") for name in dict.fromkeys(columns)})
    return pd.concat(frames, ignore_index=True)


def write_regime_document(
    ctx: RunContext, store: Store, day: str, payload: bytes
) -> Literal["written", "unchanged"]:
    """Create a regime document, accepting only a byte-identical rerun."""
    key = regime_key(day)
    if store.exists(key):
        if store.get_bytes(key) == payload:
            return "unchanged"
        raise RegimeConflictError(f"write-once regime document differs at {key}")
    ctx.record_output_cas(key, ETAG_ABSENT, payload, schema_version=REGIME_SCHEMA)
    return "written"
