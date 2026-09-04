"""Arm recipes as test fixtures.

The real recipes live in `alpha-engine-config/strategy/arms/{m,s}/` — tuned
parameters are strategy edge and are private (`repository-tiering-policy`
test 2). What is public is the *shape*: which rule ids exist, in what order a
chain may run, and which fields a recipe must declare to register at all.

These fixtures therefore mirror the private files' STRUCTURE with values that
are deliberately round numbers rather than the live ones. A test asserting a
tuned constant would either leak it or drift from it; a test asserting the
structure does neither.
"""

from __future__ import annotations

from pathlib import Path

STOCK_REGISTRY_YAML = """\
slot: s
name: stock_registry
spec:
  benchmark: SPY
  cost_model:
    name: flat_bps_v0
    placeholder: true
    params:
      half_spread_bps: 2.5
      commission_bps: 0.5
      slippage_bps: 10.0
  walk_forward:
    test_window: 21
    min_train: 100
    purge: 21
    embargo: 2
    train_mode: expanding
  rules:
    - rule_id: position_loss_floor
      params: {position_loss_floor_pct: -0.15}
    - rule_id: catalyst_hard_exit
      params: {catalyst_followthrough_days: 3}
    - rule_id: atr_trailing_stop
      params: {atr_period: 14, atr_multiplier: 3.0,
               sector_relative_outperform_threshold: 0.05}
    - rule_id: fallback_stop
      params: {fallback_stop_pct: 0.10}
    - rule_id: profit_take
      params: {profit_take_pct: 0.25}
    - rule_id: momentum_exit
      params: {momentum_exit_threshold: -15.0, momentum_exit_rsi: 30}
    - rule_id: time_decay
      params: {time_decay_reduce_days: 5, time_decay_exit_days: 10}
"""

TIGHT_STOP_YAML = """\
slot: s
name: tight_stop
spec:
  benchmark: SPY
  cost_model:
    name: flat_bps_v0
    placeholder: true
    params:
      half_spread_bps: 2.5
      commission_bps: 0.5
      slippage_bps: 10.0
  walk_forward:
    test_window: 21
    min_train: 100
    purge: 21
    embargo: 2
    train_mode: expanding
  rules:
    - rule_id: position_loss_floor
      params: {position_loss_floor_pct: -0.15}
    - rule_id: fallback_stop
      params: {fallback_stop_pct: 0.06}
    - rule_id: profit_take
      params: {profit_take_pct: 0.25}
"""


def write_strategy_arms(root: Path) -> Path:
    """Write the S-slot fixture recipes into ``root`` and return it."""
    (root / "stock_registry.yaml").write_text(STOCK_REGISTRY_YAML, encoding="utf-8")
    (root / "tight_stop.yaml").write_text(TIGHT_STOP_YAML, encoding="utf-8")
    return root
