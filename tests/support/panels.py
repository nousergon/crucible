"""Synthetic panels and books, on REAL trading days.

Not fixtures of convenience: every date these produce is an NYSE session, so
a store written from them survives
:meth:`crucible.store.Store.assert_keys_bind_to_trading_days` (§4.12). A
helper that emitted `end - timedelta(i)` would generate Saturdays and would
quietly exempt every test using it from the trading-day contract.

The signal is planted, not random: `mom_21d_ratio` carries a known positive
relationship to the forward label and `vol_21d_ratio` carries none, so a
grader that ranks them the wrong way round is detectably broken. That is the
same idea as the §10.1 control arms, applied to the test fixtures.
"""

from __future__ import annotations

import datetime as dt
import random

import numpy as np

from crucible.calendar import is_trading_day
from crucible.slots.model import FeaturePanel
from crucible.slots.strategy import Book

END = dt.date(2026, 8, 28)


def trading_days(n: int, end: dt.date = END) -> list[str]:
    out: list[str] = []
    day = end
    while len(out) < n:
        if is_trading_day(day):
            out.append(day.isoformat())
        day -= dt.timedelta(days=1)
    return list(reversed(out))


def synthetic_panel(*, n_days: int, n_names: int, seed: int) -> FeaturePanel:
    rng = np.random.default_rng(seed)
    dates = tuple(trading_days(n_days))
    names = tuple(f"T{i:03d}" for i in range(n_names))
    momentum = rng.normal(0.0, 1.0, size=(n_days, n_names))
    volatility = rng.normal(0.0, 1.0, size=(n_days, n_names))
    noise = rng.normal(0.0, 1.0, size=(n_days, n_names))
    # A planted edge: momentum explains a third of the label's variance,
    # volatility explains none of it.
    forward = 0.6 * momentum + 0.8 * noise
    return FeaturePanel(
        dates=dates,
        names=names,
        features={"mom_21d_ratio": momentum, "vol_21d_ratio": volatility},
        forward_returns=forward,
        feature_version="v1",
    )


def synthetic_book(*, n_days: int, seed: int) -> Book:
    rng = random.Random(seed)
    dates = tuple(trading_days(n_days))
    portfolio = tuple(rng.gauss(0.0009, 0.011) for _ in dates)
    benchmark = tuple(rng.gauss(0.0004, 0.009) for _ in dates)
    turnover = tuple(abs(rng.gauss(0.08, 0.02)) for _ in dates)
    return Book(
        dates=dates,
        portfolio_returns=portfolio,
        benchmark_returns=benchmark,
        turnover=turnover,
        benchmark_symbol="SPY",
    )
