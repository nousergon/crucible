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
