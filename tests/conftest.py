"""Shared fixtures: a synthetic market that is a real panel, not a mock.

Every fixture here produces the SAME shapes production produces — NYSE
sessions from `krepis.trading_calendar`, the canonical panel columns, the
real `LocalStore` — so a test against them exercises the contract rather
than a parallel one built to pass.

The prices are seeded and deterministic. That is load-bearing twice over:
the replay gate asserts that grading the same date twice reproduces the same
verdict, and the control arms assert that a planted edge is visible. Neither
claim means anything against a market that moves between runs.
"""

from __future__ import annotations

import datetime as dt
import math
import random

import pytest

from crucible.calendar import is_trading_day
from crucible.store import LocalStore

#: Enough sessions for the 252-session feature window plus a 21-session
#: horizon plus the dates a ladder needs. Shorter panels make `mom_12_1`
#: null everywhere, which silently disarms the arms that rank on it.
SESSIONS = 320


def sessions_ending(end: dt.date, count: int) -> list[dt.date]:
    """``count`` NYSE sessions ending at ``end``, ascending."""
    out: list[dt.date] = []
    day = end
    while len(out) < count:
        if is_trading_day(day):
            out.append(day)
        day -= dt.timedelta(days=1)
    return sorted(out)


def synthetic_frames(
    *,
    end: dt.date,
    n_tickers: int = 40,
    sessions: int = SESSIONS,
    seed: int = 20260901,
) -> dict[str, object]:
    """A ``{ticker: OHLCV frame}`` mapping with a real, persistent cross-section.

    Each ticker gets its own drift, drawn once. That is what makes momentum
    a signal rather than noise: without persistent per-name drift, a
    momentum ranker is a random selector and the whole grading path would be
    tested against a market in which nothing is measurable.
    """
    import pandas as pd

    days = sessions_ending(end, sessions)
    rng = random.Random(seed)
    frames: dict[str, object] = {}
    for index in range(n_tickers):
        ticker = f"T{index:03d}"
        drift = rng.gauss(0.0004, 0.0009)
        vol = rng.uniform(0.008, 0.02)
        price = rng.uniform(20.0, 300.0)
        rows = []
        for day in days:
            step = drift + rng.gauss(0.0, vol)
            price = max(1.0, price * math.exp(step))
            volume = rng.uniform(3e5, 4e6)
            rows.append(
                {
                    "trading_day": day,
                    "ticker": ticker,
                    "open_raw": price * 0.998,
                    "high_raw": price * 1.006,
                    "low_raw": price * 0.994,
                    "close_raw": price,
                    "volume_raw": volume,
                }
            )
        frames[ticker] = pd.DataFrame(rows).set_index("trading_day")
    return frames


@pytest.fixture
def store(tmp_path):
    return LocalStore(tmp_path / "store")


@pytest.fixture
def cycle_date() -> dt.date:
    """A real NYSE session, well clear of a holiday, used as the cycle date."""
    return dt.date(2026, 8, 28)


@pytest.fixture
def frames(cycle_date):
    return synthetic_frames(end=cycle_date)


@pytest.fixture
def source(frames):
    from crucible.data import FramePriceSource

    return FramePriceSource(frames, snapshot="frames:conftest-seed-20260901")


@pytest.fixture
def strategy_dir(tmp_path):
    """A minimal strategy tree: three U arms that do not share a callable.

    Three because `min_active_arms` is 3 — a slot with fewer produces zero
    or one comparison, which is the `no_promotable_challenger` defect the
    engine's own config refuses.
    """
    root = tmp_path / "strategy"
    arms = root / "arms" / "u"
    arms.mkdir(parents=True)
    recipes = {
        "momentum_sleeve": ("momentum_sleeve", {"top_n": 8}),
        "tech_score_gate": ("tech_score_gate", {"top_n": 8}),
        "mom_12_1_sleeve": ("mom_12_1_sleeve", {"top_n": 8}),
    }
    for name, (ranker, params) in recipes.items():
        body = [
            f"name: {name}",
            "slot: u",
            f"ranker: {ranker}",
            "registered_at: '2026-06-01'",
            "params:",
        ]
        body += [f"  {k}: {v}" for k, v in params.items()]
        body.append(f"notes: fixture recipe for {name}")
        (arms / f"{name}.yaml").write_text("\n".join(body) + "\n", encoding="utf-8")
    return root
