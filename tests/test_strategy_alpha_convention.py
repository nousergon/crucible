"""The solver's alpha convention, enforced where the M feed meets it.

alpha-engine-config-I12121. `crucible.portfolio.solve_target_weights` reads
``alpha_hat`` as DEVIATION from the benchmark: the benchmark entry is 0.0, the
null hypothesis. An M arm predicts a raw forward return. Before this fix,
`_build_sessions` copied the feed straight in, so (a) the benchmark entry
carried the M model's own forecast for the benchmark ticker, which the feed
prices like any name (SPY sat at the 2nd percentile of the 2026-11-03 feed),
and (b) every name carried the market's drift as conviction against a
benchmark that carried none.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crucible.slots.strategy import (
    CASH_TICKER,
    ResolvedSession,
    _build_sessions,
    _universe_for,
)

DAY, SETTLE = "2026-11-02", "2026-11-03"


def _session(alpha: dict[str, float]) -> ResolvedSession:
    tickers = tuple(sorted(alpha))
    return ResolvedSession(
        trading_day=DAY,
        tickers=tickers,
        alpha_hat=tuple(alpha[t] for t in tickers),
        eligibility=tuple(True for _ in tickers),
        stance_caps=tuple(1.0 for _ in tickers),
        alpha_source="test",
        eligibility_source="test",
        sectors_source="test",
        stance_caps_source="test",
    )


def _inputs(alpha: dict[str, float]):
    resolved = [_session(alpha)]
    universe = _universe_for(resolved, benchmark="SPY")
    cols = [t for t in universe.tickers if t != CASH_TICKER]
    returns = pd.DataFrame(0.001, index=[DAY, SETTLE], columns=cols)
    (inputs,) = _build_sessions(
        resolved,
        returns=returns,
        benchmark="SPY",
        universe=universe,
        next_session={DAY: SETTLE},
        adv_by_day={},
    )
    return universe, inputs


# A feed shaped like the served one: names on a positive level, the benchmark
# ticker priced low.
FEED = {"AAA": 0.030, "BBB": 0.012, "CCC": 0.006, "DDD": -0.004, "SPY": -0.003}


def test_the_benchmark_entry_is_the_null_hypothesis_not_the_models_forecast():
    universe, inputs = _inputs(FEED)
    assert inputs.alpha_hat[universe.benchmark_idx] == 0.0
    assert inputs.alpha_hat[universe.cash_idx] == 0.0


def test_names_enter_as_excess_over_the_priced_cross_section():
    universe, inputs = _inputs(FEED)
    names = [t for t in FEED if t != "SPY"]
    raw = np.array([FEED[t] for t in names])
    got = np.array([inputs.alpha_hat[universe.tickers.index(t)] for t in names])
    assert got == pytest.approx(raw - raw.mean())
    assert got.mean() == pytest.approx(0.0, abs=1e-15)
    # A shift, not a scale: ranks and spreads are the feed's.
    assert list(np.argsort(got)) == list(np.argsort(raw))
    assert np.ptp(got) == pytest.approx(np.ptp(raw))


def test_a_uniform_level_carries_no_conviction():
    """The I12121 symptom: a feed whose every name sits ~7% above zero. After
    the transform no name looks better than the benchmark on level alone."""
    flat = {t: 0.073 for t in ("AAA", "BBB", "CCC")} | {"SPY": 0.073}
    universe, inputs = _inputs(flat)
    assert np.allclose(inputs.alpha_hat, 0.0)


def test_a_benched_all_zero_session_stays_all_zero():
    universe, inputs = _inputs(dict.fromkeys(FEED, 0.0))
    assert not np.any(inputs.alpha_hat)


def test_the_benchmark_and_cash_stay_eligible_and_uncapped():
    universe, inputs = _inputs(FEED)
    for i in (universe.benchmark_idx, universe.cash_idx):
        assert inputs.eligibility[i]
        assert inputs.stance_caps[i] == 1.0
