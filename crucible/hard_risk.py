"""The hard-risk layer: loss floor, drawdown ladder, re-entry lockout -- as pure rules.

v1 → v2 trading-rule parity audit (2026-11-09), rows 3, 4, 7 and 8. v1 ran these
in the executor (`exit_manager.check_position_loss_floor`,
`risk_guard.compute_drawdown_multiplier`, `main._reentry_lockouts`); v2 had
none of them. This module states each rule once, as a function of facts the
CALLER supplies, so the trader (which alone can read average cost and NAV from
the broker) and any replay of its sessions decide identically.

**Precedence, highest first: hard risk > eligibility > optimizer > timing.** A
name this layer cuts is handed to `crucible.portfolio.solve_target_weights` as
``hard_exit``, which pins it to zero ABOVE the eligibility cut, ABOVE
``ineligible_held_exit: discretionary`` and inside the mandatory turnover
floor -- so the sale is never budget-limited -- and a name it locks out cannot
be bought by the optimizer the same day or for the lockout window after.

**No value lives here.** Every threshold is a `PortfolioParams` field declared
in the private strategy tree; a ``null`` field switches its rule off, and this
module never substitutes a number for one.

The negative-alpha exit (row 6) is NOT here: it needs only the alpha the
solver already has, so `crucible.portfolio.forced_exit_mask` applies it
inside every solve, grading included.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import math
from collections.abc import Iterable, Mapping

from crucible.calendar import is_trading_day
from crucible.portfolio import PortfolioParams, declared_value

__all__ = [
    "DRAWDOWN_EXIT",
    "HARD_EXIT_REASONS",
    "LOCKOUT",
    "LOSS_FLOOR_EXIT",
    "DrawdownState",
    "Holding",
    "drawdown_state",
    "lockout_through",
    "loss_floor_breaches",
    "select_drawdown_exits",
]

#: The reasons a hard-risk exit is recorded under. A future intraday unit
#: (audit row 2, the gap stop) adds its own reason to this vocabulary.
LOSS_FLOOR_EXIT = "position_loss_floor"
DRAWDOWN_EXIT = "drawdown_forced_exit"
LOCKOUT = "hard_risk_lockout"
HARD_EXIT_REASONS: frozenset[str] = frozenset({LOSS_FLOOR_EXIT, DRAWDOWN_EXIT})

#: The ladder's sizing multipliers. Structure, not tuning: the thresholds that
#: select a rung are `PortfolioParams` fields; the rungs are what the fields
#: are named for.
FULL, HALF, QUARTER, HALT = 1.0, 0.5, 0.25, 0.0

#: Float slack in "at or below a threshold" (one part in 10^12 of NAV).
_AT_THRESHOLD = 1e-12


@dataclasses.dataclass(frozen=True)
class Holding:
    """One held name, as the broker reports it at decision time."""

    shares: int
    avg_cost: float | None
    mark: float | None

    @property
    def loss_from_cost(self) -> float | None:
        """``mark / avg_cost - 1``, or None when either price is unusable."""
        if self.avg_cost is None or self.mark is None:
            return None
        if not (math.isfinite(self.avg_cost) and self.avg_cost > 0):
            return None
        if not (math.isfinite(self.mark) and self.mark > 0):
            return None
        return self.mark / self.avg_cost - 1.0


def loss_floor_breaches(
    holdings: Mapping[str, Holding], params: PortfolioParams
) -> dict[str, float]:
    """``{ticker: loss}`` for every held name at or below the loss floor.

    Stance-agnostic and unconditional, as in v1 (COIN bled -19% while ranked
    #1): the model still liking a name is not a reason to keep it past the
    floor. A holding with no usable average cost or mark RAISES: a floor that
    silently skips the names it cannot price is a floor with holes in exactly
    the cases a broker read went wrong.
    """
    floor = params.position_loss_floor_pct
    if floor is None:
        return {}
    breaches: dict[str, float] = {}
    for ticker, holding in sorted(holdings.items()):
        if holding.shares <= 0:
            continue
        loss = holding.loss_from_cost
        if loss is None:
            raise ValueError(
                f"{ticker}: no usable average cost ({holding.avg_cost!r}) or mark "
                f"({holding.mark!r}); the loss floor cannot be evaluated on a partial book"
            )
        if loss <= float(floor):
            breaches[ticker] = loss
    return breaches


@dataclasses.dataclass(frozen=True)
class DrawdownState:
    """Where NAV sits on the ladder this session."""

    nav_usd: float
    peak_nav_usd: float
    drawdown: float
    multiplier: float
    circuit_breaker: bool
    forced_exit_count: int

    def to_document(self) -> dict[str, object]:
        return dataclasses.asdict(self)


def drawdown_state(
    nav_usd: float, peak_nav_usd: float | None, params: PortfolioParams
) -> DrawdownState | None:
    """The ladder rung for ``nav_usd`` against the running peak, or None when off.

    The peak this session is ``max(previous peak, nav)``, so the first session
    seeds it and a new high resets the ladder to full size. Thresholds are
    compared with ``<=``, as in v1: a drawdown exactly at a threshold is in
    that tier.
    """
    if params.drawdown_circuit_breaker_at is None:
        return None
    if not (math.isfinite(nav_usd) and nav_usd > 0):
        raise ValueError(f"NAV {nav_usd!r} is not a positive number; the ladder cannot be read")
    peak = nav_usd if peak_nav_usd is None else max(float(peak_nav_usd), nav_usd)
    drawdown = nav_usd / peak - 1.0
    half = declared_value(params.drawdown_half_size_at)
    quarter = declared_value(params.drawdown_quarter_size_at)
    breaker = float(params.drawdown_circuit_breaker_at)
    exits_half = int(declared_value(params.drawdown_forced_exits_at_half))
    exits_quarter = int(declared_value(params.drawdown_forced_exits_at_quarter))
    # A drawdown AT a threshold is in that tier; the tolerance keeps a NAV that
    # lands on it exactly from reading as a hair above it in floating point.
    if drawdown <= breaker + _AT_THRESHOLD:
        multiplier, count = HALT, exits_quarter
    elif drawdown <= quarter + _AT_THRESHOLD:
        multiplier, count = QUARTER, exits_quarter
    elif drawdown <= half + _AT_THRESHOLD:
        multiplier, count = HALF, exits_half
    else:
        multiplier, count = FULL, 0
    return DrawdownState(
        nav_usd=float(nav_usd),
        peak_nav_usd=float(peak),
        drawdown=float(drawdown),
        multiplier=multiplier,
        circuit_breaker=multiplier == HALT,
        forced_exit_count=count,
    )


def select_drawdown_exits(
    holdings: Mapping[str, Holding],
    alpha: Mapping[str, float],
    count: int,
    *,
    already_exiting: Iterable[str] = (),
) -> list[str]:
    """The ``count`` lowest-conviction held names to force out, beyond those already exiting.

    v1 §2g ranked by score, then by market value. v2's conviction is the
    alpha the session was constructed on; ties (an M bench serves a zero
    vector) break on the worse loss from cost, then on the ticker, so the
    choice is deterministic and leans toward the name already hurting the
    book. A name already exiting for another hard-risk reason counts toward
    the tier's total, as v1's idempotent `select_forced_exits` did.
    """
    exiting = set(already_exiting)
    remaining = count - len(exiting & set(holdings))
    if remaining <= 0:
        return []
    candidates = [t for t, h in holdings.items() if h.shares > 0 and t not in exiting]

    def rank(ticker: str) -> tuple[float, float, str]:
        loss = holdings[ticker].loss_from_cost
        return (
            float(alpha.get(ticker, 0.0)),
            0.0 if loss is None else loss,
            ticker,
        )

    return sorted(candidates, key=rank)[:remaining]


def lockout_through(exit_day: dt.date, sessions: int) -> dt.date:
    """The last session a name exited on ``exit_day`` may not be bought.

    The exit session itself is blocked (a name cut today is not bought today)
    and so are the ``sessions`` trading days after it: an exit on D blocks D
    through D+``sessions``, matching v1's "an exit on day D blocks D+1 .. D+5"
    plus the same-day precedence rule.
    """
    if sessions < 1:
        raise ValueError(f"a lockout of {sessions} sessions is not a lockout; null disables it")
    day = exit_day
    counted = 0
    while counted < sessions:
        day += dt.timedelta(days=1)
        if is_trading_day(day):
            counted += 1
    return day
