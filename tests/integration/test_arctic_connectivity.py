"""The real-ArcticDB half of this tier's hard constraint.

`data.daily`/`data.weekly`/`data.heal` cannot be pointed at a dedicated
library today (see `README.md`, "Not exercised, and why" —
`alpha-engine-config-I10420`). This module is what proves "a real S3 prefix
and a real ArcticDB library" generically instead: a write, a read-back and a
symbol-list against the dedicated library this tier owns, using the
low-level `open_arctic(...).get_library(...)` path that carries no
production library-name constant.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

from tests.integration.conftest import INTEGRATION_TRADING_DAY


def test_the_dedicated_library_is_really_arcticdb_and_really_writable(
    arctic_library: Any, integration_arctic_library: str
) -> None:
    """A real `Library.write`/`Library.read` round trip.

    Not a `PriceSource` shape (OHLCV columns, a `DatetimeIndex`) — this test
    proves ArcticDB connectivity and write/read integrity on the dedicated
    library, which is a property of the CONNECTION, not of what a future
    `ArcticPriceSource(library=...)` override would store there.
    """
    import pandas as pd

    symbol = f"integration-probe-{INTEGRATION_TRADING_DAY}"
    frame = pd.DataFrame(
        {"probe_value": [1.0, 2.0, 3.0]},
        index=pd.to_datetime(["2026-09-04", "2026-09-05", "2026-09-08"]),
    )
    try:
        arctic_library.write(symbol, frame)
        read_back = arctic_library.read(symbol).data
        assert list(read_back["probe_value"]) == [1.0, 2.0, 3.0], (
            "wrote 3 rows to the dedicated ArcticDB library and read back something else — "
            f"the real dependency is reachable but not returning what was written: {read_back}"
        )
        assert symbol in arctic_library.list_symbols(), (
            f"{symbol!r} was written and read back but does not appear in "
            f"{integration_arctic_library!r}'s own symbol list"
        )
    finally:
        # Idempotent teardown — a probe symbol left behind after a failed
        # assertion above must not accumulate across nightly runs, and a
        # second run's `write` would silently version over it either way.
        # Never `except: pass` (rule 5): only the ONE outcome a missing
        # symbol produces is swallowed, and only here, after the write this
        # function itself performed.
        if arctic_library.has_symbol(symbol):
            arctic_library.delete(symbol)


def test_the_seed_covers_every_symbol_data_daily_reads_beyond_the_universe(
    integration_arctic_bucket: str,
    integration_arctic_library: str,
    integration_arctic_symbols: list[str],
    strategy_dir: Path,
) -> None:
    """Reads back what `integration_arctic_symbols` wrote — through the EXACT
    path `crucible.data.daily.run_daily` reads it through
    (`ArcticPriceSource(library=...).load_panel`) — BEFORE any CLI job case
    runs (this module collects and runs before `test_cli_jobs.py`,
    alphabetically and by this suite's own docstring).

    `alpha-engine-config-I10701`, third measurement: run 34797908861 failed
    `test_data_daily`/`weekly`/`heal`/`experiment_run`/`weekly`/
    `experiment_grade` with `MissingSourceError: ... returned zero symbols`.
    The dedicated library WAS written (`test_the_dedicated_library_is_really_
    arcticdb_and_really_writable` above proved the connection is real and
    read-after-write works), but `run_daily` also reads a panel row for
    every slot's declared `benchmark` (`crucible.slots.
    declared_benchmark_symbols`, S's `"SPY"`) and every factor-attribution
    proxy (`crucible.slots.attribution_factor_symbols`) — BEYOND the
    declared-universe tickers `integration_arctic_symbols` used to seed —
    and `integration_arctic_symbols` predated both producer clauses, so it
    never wrote them. This test is the fixture-ordering-and-completeness
    guard `alpha-engine-config-I10701` asks for: a future extra-symbol
    source added to `run_daily` without a matching addition here fails HERE,
    naming the missing symbol, rather than surfacing three jobs downstream
    as an opaque zero-symbols outage.
    """
    from crucible.data.sources import ArcticPriceSource
    from crucible.slots import attribution_factor_symbols, declared_benchmark_symbols

    extra_symbols = sorted(
        (declared_benchmark_symbols() | attribution_factor_symbols(strategy_dir=strategy_dir))
        - set(integration_arctic_symbols)
    )
    if not extra_symbols:
        return
    source = ArcticPriceSource(integration_arctic_bucket, library=integration_arctic_library)
    panel = source.load_panel(
        end=dt.date.fromisoformat(INTEGRATION_TRADING_DAY),
        lookback_days=30,
        symbols=extra_symbols,
    )
    # `load_panel` returns a long-format DataFrame (`trading_day`, `ticker`,
    # `*_raw` columns) — never a `{symbol: frame}` mapping — the same shape
    # `crucible.data.daily.run_daily` reads its own `extra_panel` as
    # (`extra_observed_today = {str(t) for t in extra_day_rows["ticker"]
    # .unique()}`). `set(panel)` would read a DataFrame's COLUMN names, not
    # its tickers, and pass this assertion vacuously regardless of what was
    # actually seeded — mirror the producer's own read here rather than
    # inventing a second, wrong one.
    observed = {str(t) for t in panel["ticker"].unique()}
    missing = sorted(set(extra_symbols) - observed)
    assert not missing, (
        f"the dedicated ArcticDB library {integration_arctic_library!r} carries no row for "
        f"{missing} — every slot's declared benchmark and every factor-attribution proxy "
        "must be seeded, or `data.daily`'s own extra-panel fetch fails `MissingSourceError` "
        "for every case in `test_cli_jobs.py` that reaches it. Add the missing symbol(s) to "
        "`tests/integration/conftest.py::integration_arctic_symbols`."
    )
