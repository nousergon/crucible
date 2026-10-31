"""Think Tank ratings documents shaped like `thinktank/ratings/{trading_day}.json`.

The production document (measured 2026-10-28): `schema_version` 1, the
`trading_day` it rates, the `updated_at` it was written (the next business
day), and `rows` keyed by ticker, each carrying an integer `rating` on 0-100.
Only the fields `crucible.data.point_in_time` reads are modelled.
"""

from __future__ import annotations

import datetime as dt
import json
import random

from crucible.keys import thinktank_ratings_key

#: Labelled 2026-09-09, written the morning of 2026-09-10 New York time, so it
#: is admissible from 2026-09-11 — the `END` session the point-in-time
#: fixtures resolve on.
DEFAULT_LABEL = dt.date(2026, 9, 9)
DEFAULT_UPDATED_AT = "2026-09-10T14:35:00+00:00"


def ratings_document(
    tickers: list[str],
    label: dt.date = DEFAULT_LABEL,
    updated_at: str = DEFAULT_UPDATED_AT,
    *,
    seed: int = 7,
    ratings: dict[str, object] | None = None,
) -> bytes:
    rng = random.Random(seed)
    values = ratings if ratings is not None else {t: rng.randint(10, 95) for t in tickers}
    return json.dumps(
        {
            "schema_version": 1,
            "trading_day": label.isoformat(),
            "updated_at": updated_at,
            "rows": {t: {"rating": v, "stance": "neutral"} for t, v in values.items()},
        }
    ).encode()


def thinktank_objects(
    tickers: list[str],
    label: dt.date = DEFAULT_LABEL,
    updated_at: str = DEFAULT_UPDATED_AT,
    *,
    seed: int = 7,
) -> dict[str, bytes]:
    """One ratings document covering a third of ``tickers`` (the Think Tank rates a
    subset of the universe by design)."""
    covered = tickers[: max(1, len(tickers) // 3)]
    return {thinktank_ratings_key(label): ratings_document(covered, label, updated_at, seed=seed)}
