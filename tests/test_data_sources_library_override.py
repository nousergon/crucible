"""`ArcticPriceSource(library=...)` — a dedicated-library override.

Normative source: `alpha-engine-config-I10457`. `ArcticPriceSource` was
hard-wired to `nousergon_lib.arcticdb`'s single production `universe`
library (via `load_universe_ohlcv`/`open_universe_lib`), with no
library-selection parameter anywhere in the call path — blocking 7 of the
integration tier's 24 CLI jobs (`data.daily`, `data.weekly`, `data.heal`
directly; `experiment.run`, `experiment.grade`, `promote`, `weekly`
indirectly) from ever running against a dedicated, isolated ArcticDB
library instead of the real production one.

`library=None` (the default) is production, byte-for-byte unchanged: it
still goes through `load_universe_ohlcv`/`open_universe_lib`, never the
generic path below. `library="some-name"` routes through the generic
`open_arctic(bucket).get_library(name, create_if_missing=True)` +
`nousergon_lib.arcticdb._load_arctic_frames` path instead — the same
low-level path `tests/integration/test_arctic_connectivity.py` and
`tests/integration/conftest.py::arctic_library` already use to avoid ever
touching `open_universe_lib`'s hard-wired production name.
"""

from __future__ import annotations

import datetime as dt

import pytest

from crucible.data.sources import ArcticPriceSource, MissingSourceError


class _FakeArctic:
    """Stands in for `nousergon_lib.arcticdb.open_arctic`'s return value.

    Records the library name it was asked to open, and whether
    `create_if_missing` was honoured, so a test can assert the override
    actually reached the right library rather than merely not crashing.
    """

    def __init__(self, lib: object) -> None:
        self._lib = lib
        self.get_library_calls: list[tuple[str, bool]] = []

    def get_library(self, name: str, create_if_missing: bool = False):
        self.get_library_calls.append((name, create_if_missing))
        return self._lib


class _FakeLib:
    def __init__(self, symbols: list[str] | None = None) -> None:
        self._symbols = symbols or []

    def list_symbols(self) -> list[str]:
        return list(self._symbols)


def test_library_none_never_reaches_the_generic_open_arctic_path(
    frames, cycle_date, monkeypatch
) -> None:
    """Default behaviour (production) is untouched: `open_arctic` is not
    even imported/called when no override is given."""
    requested = sorted(frames)

    def _boom_open_arctic(bucket, region=None):
        raise AssertionError("open_arctic must not be reached when library=None")

    monkeypatch.setattr("nousergon_lib.arcticdb.load_universe_ohlcv", lambda *a, **k: frames)
    monkeypatch.setattr("nousergon_lib.arcticdb.open_arctic", _boom_open_arctic)

    source = ArcticPriceSource("test-bucket")
    assert source.library is None
    panel = source.load_panel(end=cycle_date, lookback_days=400, symbols=requested)
    assert set(panel["ticker"]) == set(requested)


def test_library_override_never_reaches_load_universe_ohlcv_or_open_universe_lib(
    frames, cycle_date, monkeypatch
) -> None:
    """The dedicated-library path must never fall through to the
    production-hard-wired helpers — a fallback there would silently read
    the production `universe` library from a call that named a different
    one."""
    requested = sorted(frames)
    fake_lib = _FakeLib(requested)
    fake_arctic = _FakeArctic(fake_lib)

    def _boom(*a, **k):
        raise AssertionError("production-hard-wired helper must not be reached")

    monkeypatch.setattr("nousergon_lib.arcticdb.load_universe_ohlcv", _boom)
    monkeypatch.setattr("nousergon_lib.arcticdb.open_universe_lib", _boom)
    monkeypatch.setattr(
        "nousergon_lib.arcticdb.open_arctic", lambda bucket, region=None: fake_arctic
    )
    monkeypatch.setattr(
        "nousergon_lib.arcticdb._load_arctic_frames",
        lambda lib, symbols, **kwargs: {s: frames[s] for s in symbols},
    )

    source = ArcticPriceSource("test-bucket", library="crucible-integration")
    panel = source.load_panel(end=cycle_date, lookback_days=400, symbols=requested)
    assert set(panel["ticker"]) == set(requested)
    assert fake_arctic.get_library_calls == [("crucible-integration", True)]


def test_library_override_opens_with_create_if_missing_true(
    frames, cycle_date, monkeypatch
) -> None:
    """Mirrors `tests/integration/conftest.py::arctic_library`'s own
    cold-start shape — the dedicated library may not exist yet on a fresh
    bucket."""
    requested = sorted(frames)
    fake_lib = _FakeLib(requested)
    fake_arctic = _FakeArctic(fake_lib)
    monkeypatch.setattr(
        "nousergon_lib.arcticdb.open_arctic", lambda bucket, region=None: fake_arctic
    )
    monkeypatch.setattr(
        "nousergon_lib.arcticdb._load_arctic_frames",
        lambda lib, symbols, **kwargs: {s: frames[s] for s in symbols},
    )

    source = ArcticPriceSource("test-bucket", library="crucible-integration")
    source.load_panel(end=cycle_date, lookback_days=400, symbols=requested)
    assert fake_arctic.get_library_calls[0][1] is True


def test_library_override_with_no_symbols_lists_the_dedicated_library(
    frames, cycle_date, monkeypatch
) -> None:
    """`symbols=None` against a dedicated library cannot ask the production
    `get_universe_symbols` helper (it is hard-wired to `universe`), so this
    path falls back to the library's own `list_symbols()` instead."""
    requested = sorted(frames)
    fake_lib = _FakeLib(requested)
    fake_arctic = _FakeArctic(fake_lib)
    monkeypatch.setattr(
        "nousergon_lib.arcticdb.open_arctic", lambda bucket, region=None: fake_arctic
    )

    captured: dict[str, object] = {}

    def _fake_load(lib, symbols, **kwargs):
        captured["symbols"] = symbols
        return {s: frames[s] for s in symbols}

    monkeypatch.setattr("nousergon_lib.arcticdb._load_arctic_frames", _fake_load)

    source = ArcticPriceSource("test-bucket", library="crucible-integration")
    panel = source.load_panel(end=cycle_date, lookback_days=400, symbols=None)
    assert sorted(captured["symbols"]) == requested
    assert set(panel["ticker"]) == set(requested)


def test_library_override_empty_result_raises_missing_source_error(
    frames, cycle_date, monkeypatch
) -> None:
    fake_lib = _FakeLib([])
    fake_arctic = _FakeArctic(fake_lib)
    monkeypatch.setattr(
        "nousergon_lib.arcticdb.open_arctic", lambda bucket, region=None: fake_arctic
    )
    monkeypatch.setattr("nousergon_lib.arcticdb._load_arctic_frames", lambda lib, symbols, **k: {})

    source = ArcticPriceSource("test-bucket", library="crucible-integration")
    with pytest.raises(MissingSourceError, match="crucible-integration"):
        source.load_panel(end=cycle_date, lookback_days=400, symbols=sorted(frames))


def test_library_override_reaching_failure_wraps_as_missing_source_error(
    frames, cycle_date, monkeypatch
) -> None:
    def _boom(bucket, region=None):
        raise RuntimeError("library open failed")

    monkeypatch.setattr("nousergon_lib.arcticdb.open_arctic", _boom)

    source = ArcticPriceSource("test-bucket", library="crucible-integration")
    with pytest.raises(MissingSourceError, match="crucible-integration"):
        source.load_panel(end=cycle_date, lookback_days=400, symbols=sorted(frames))


def test_snapshot_id_records_the_library_override(cycle_date) -> None:
    default_source = ArcticPriceSource("test-bucket")
    dedicated_source = ArcticPriceSource("test-bucket", library="crucible-integration")
    assert default_source.snapshot_id() == "arcticdb:test-bucket"
    assert dedicated_source.snapshot_id() == "arcticdb:test-bucket:crucible-integration"
    assert default_source.snapshot_id() != dedicated_source.snapshot_id()


def test_library_override_classifies_unlisted_symbols_through_the_dedicated_library(
    frames, cycle_date, monkeypatch
) -> None:
    """The absent-symbol classification (`_classify_absent_symbols`) must
    still consult the DEDICATED library's own `has_symbol`/`get_description`
    — never the production `open_universe_lib` — or a symbol legitimately
    unlisted in the dedicated library would be misread against production
    bounds it was never resolved from."""
    requested = sorted(frames)
    newco = requested[0]
    thinned = {t: f for t, f in frames.items() if t != newco}

    class _BoundedFakeLib(_FakeLib):
        def has_symbol(self, sym: str) -> bool:
            return sym == newco

        def get_description(self, sym: str):
            class _Desc:
                date_range = (
                    cycle_date + dt.timedelta(days=30),
                    cycle_date + dt.timedelta(days=60),
                )

            return _Desc()

    fake_lib = _BoundedFakeLib(list(thinned))
    fake_arctic = _FakeArctic(fake_lib)
    monkeypatch.setattr(
        "nousergon_lib.arcticdb.open_arctic", lambda bucket, region=None: fake_arctic
    )

    def _boom_production(*a, **k):
        raise AssertionError("must not reach the production open_universe_lib")

    monkeypatch.setattr("nousergon_lib.arcticdb.open_universe_lib", _boom_production)
    monkeypatch.setattr(
        "nousergon_lib.arcticdb._load_arctic_frames", lambda lib, symbols, **k: thinned
    )

    source = ArcticPriceSource("test-bucket", library="crucible-integration")
    panel = source.load_panel(end=cycle_date, lookback_days=400, symbols=requested)
    assert panel.attrs["unlisted_in_window"] == [newco]
    assert newco not in set(panel["ticker"])
