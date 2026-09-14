# The feature catalogue

Every column the feature layer produces, with its units and its lineage.
Normative source: the plan's §10 component 4, and the fleet feature-store
rule in `AGENTS.md` — **every column name carries an explicit units suffix**
(`_raw`, `_ratio`, `_pct`, `_zscore`, `_log_return`).

The table below is **generated from `crucible.features.registry.CATALOG`**
and asserted equal to it by
`tests/test_feature_registry_contract.py::test_the_documentation_table_is_the_catalogue`.
A column added without its documentation row is therefore a red test, not a
document that quietly stopped describing the layer. Regenerate with:

    uv run python -m crucible.features.docgen

## Reading this table

- **Unit** is the concrete unit. For every normalized suffix the suffix
  *pins* the unit — `_ratio` means `unit: ratio` and nothing else. `_raw` is
  the one suffix with no single unit, because "raw" means "not normalized by
  this layer", and it may never claim a normalized unit word. Both directions
  are refused at construction and by
  `crucible/schemas/feature_registry.v1.json`. The defect this exists to
  close: `avg_volume_20d` was emitted as a normalized ratio and consumed as
  raw shares, and 901 of 903 tickers silently failed the scanner liquidity
  gate for months.
- **Window** is a count of **sessions**, never calendar days (§4.12). `—`
  means a point-in-time column reading only the current row.
- **Inputs** is lineage, and it is a field rather than a comment: it is what
  `explain` answers "which data produced this column" from.

## The version is derived, never declared

`crucible.features.DEFAULT_FEATURE_VERSION` is a 12-hex digest of this whole
catalogue, so editing any row below writes to a different
`features/{version}/` prefix and **cannot overwrite the layer an earlier
verdict was computed from**. No consumer names a version:
`FeatureLayerSource(store=..., registry=CATALOG)` resolves it. A hand-written
`"v1"` is refused by the registry schema's `feature_version` pattern
(`alpha-engine-config-I9772`, disagreement 2).

At the time of writing this file the catalogue's version is
`vf795db2b5049`; that string is expected to move whenever a row
below changes, and is recorded here as an illustration rather than a pin.

## Columns

<!-- BEGIN GENERATED CATALOG TABLE -->
| Column | Unit | Window (sessions) | Cross-sectional | Expression | Inputs |
|---|---|---|---|---|---|
| `close_raw` | USD | — | no | `close` | `close_raw` |
| `dollar_volume_20d_raw` | USD | 20 | no | `mean(close * volume, 20)` | `close_raw`, `volume_raw` |
| `return_1d_log_return` | log_return | 1 | no | `log(close / close.shift(1))` | `close_raw` |
| `momentum_20d_log_return` | log_return | 20 | no | `log(close / close.shift(20))` | `close_raw` |
| `return_60d_log_return` | log_return | 60 | no | `log(close / close.shift(60))` | `close_raw` |
| `mom_12_1_log_return` | log_return | 252 | no | `log(close.shift(21) / close.shift(252))` | `close_raw` |
| `volatility_20d_ratio` | ratio | 20 | no | `std(log_return_1d, 20)` | `close_raw` |
| `close_to_sma50_ratio` | ratio | 50 | no | `close / mean(close, 50)` | `close_raw` |
| `close_to_sma200_ratio` | ratio | 200 | no | `close / mean(close, 200)` | `close_raw` |
| `rsi_14_ratio` | ratio | 14 | no | `wilder_rsi(close, 14) / 100` | `close_raw` |
| `liquidity_pass_raw` | indicator | 20 | no | `dollar_volume_20d_raw >= liquidity_floor_usd()` | `dollar_volume_20d_raw` |
| `tech_score_ratio` | ratio | — | yes | `mean(rank01(rsi_14_ratio), rank01(close_to_sma50_ratio), rank01(close_to_sma200_ratio), rank01(momentum_20d_log_return))` | `rsi_14_ratio`, `close_to_sma50_ratio`, `close_to_sma200_ratio`, `momentum_20d_log_return` |
| `momentum_20d_zscore` | zscore | — | yes | `zscore(momentum_20d_log_return)` | `momentum_20d_log_return`, `liquidity_pass_raw` |
| `return_60d_zscore` | zscore | — | yes | `zscore(return_60d_log_return)` | `return_60d_log_return`, `liquidity_pass_raw` |
| `mom_12_1_zscore` | zscore | — | yes | `zscore(mom_12_1_log_return)` | `mom_12_1_log_return`, `liquidity_pass_raw` |
| `market_return_1d_log_return` | log_return | 1 | yes | `mean(return_1d_log_return) over the day's cross-section` | `return_1d_log_return` |
| `beta_60d_raw` | beta | 60 | no | `cov(return_1d_log_return, market_return_1d_log_return, 60) / var(market_return_1d_log_return, 60), shifted one session` | `return_1d_log_return`, `market_return_1d_log_return` |
| `residual_return_1d_log_return` | log_return | 1 | no | `return_1d_log_return - beta_60d_raw * market_return_1d_log_return` | `return_1d_log_return`, `market_return_1d_log_return`, `beta_60d_raw` |
| `residual_vol_20d_ratio` | ratio | 20 | no | `std(residual_return_1d_log_return, 20)` | `residual_return_1d_log_return` |
| `residual_momentum_252d_skip21d_ratio` | ratio | 252 | no | `sum(residual_return_1d_log_return, 231).shift(21) / (residual_vol_20d_ratio * sqrt(231))` | `residual_return_1d_log_return`, `residual_vol_20d_ratio` |
| `residual_momentum_252d_skip21d_zscore` | zscore | — | yes | `zscore(residual_momentum_252d_skip21d_ratio)` | `residual_momentum_252d_skip21d_ratio`, `liquidity_pass_raw` |
| `momentum_change_21d_log_return` | log_return | 42 | no | `sum(return_1d_log_return, 21) - sum(return_1d_log_return, 21).shift(21)` | `return_1d_log_return` |
| `momentum_change_21d_zscore` | zscore | — | yes | `zscore(momentum_change_21d_log_return)` | `momentum_change_21d_log_return`, `liquidity_pass_raw` |
| `momentum_5d_log_return` | log_return | 5 | no | `log(close) - log(close).shift(5)` | `close_raw` |
| `atr_14_ratio` | ratio | 14 | no | `mean(max(high - low, |high - close.shift(1)|, |low - close.shift(1)|), 14) / close` | `high_raw`, `low_raw`, `close_raw` |
| `vol_ratio_10_60_ratio` | ratio | 60 | no | `std(return_1d_log_return, 10) / std(return_1d_log_return, 60)` | `return_1d_log_return` |
| `dist_from_52w_high_ratio` | ratio | 252 | no | `close / max(close, 252) - 1` | `close_raw` |
| `dist_from_52w_low_ratio` | ratio | 252 | no | `close / min(close, 252) - 1` | `close_raw` |
| `sector_raw` | gics_sector_label | — | no | `constituents.sector_map[ticker], fetched before the session` | `point_in_time.sector.sector_map` |
| `roe_ratio` | ratio | — | no | `fundamental.roe (TTM return on equity, decimal, clipped [-1, 1])` | `point_in_time.fundamental.roe` |
| `debt_to_equity_div2_ratio` | ratio | — | no | `fundamental.debt_to_equity (total debt / equity / 2, clipped [-3, 3])` | `point_in_time.fundamental.debt_to_equity` |
| `gross_margin_ratio` | ratio | — | no | `fundamental.gross_margin (TTM, 0-1 fraction)` | `point_in_time.fundamental.gross_margin` |
| `current_ratio_div3_ratio` | ratio | — | no | `fundamental.current_ratio (current assets / liabilities / 3, clipped [0, 3])` | `point_in_time.fundamental.current_ratio` |
| `pe_div30_ratio` | ratio | — | no | `fundamental.pe_ratio (trailing P/E / 30, clipped [-3, 3])` | `point_in_time.fundamental.pe_ratio` |
| `pb_div5_ratio` | ratio | — | no | `fundamental.pb_ratio (price / book / 5, clipped [-3, 3])` | `point_in_time.fundamental.pb_ratio` |
| `fcf_yield_ratio` | ratio | — | no | `fundamental.fcf_yield (TTM free cash flow / market cap, clipped [-0.5, 0.5])` | `point_in_time.fundamental.fcf_yield` |
| `revenue_growth_3y_ratio` | ratio | — | no | `fundamental.revenue_growth_3y (3-year revenue CAGR, decimal)` | `point_in_time.fundamental.revenue_growth_3y` |
| `eps_growth_3y_ratio` | ratio | — | no | `fundamental.eps_growth_3y (3-year EPS CAGR, decimal)` | `point_in_time.fundamental.eps_growth_3y` |
| `capex_growth_5y_ratio` | ratio | — | no | `fundamental.capex_growth_5y (5-year capex growth, decimal)` | `point_in_time.fundamental.capex_growth_5y` |
| `payout_ratio` | ratio | — | no | `fundamental.payout_ratio (TTM dividends / net income, clipped [0, 2])` | `point_in_time.fundamental.payout_ratio` |
| `sustainable_growth_rate_ratio` | ratio | — | no | `roe_ratio * (1 - payout_ratio)` | `roe_ratio`, `payout_ratio` |
| `institutional_accumulation_raw` | funds | — | no | `n_funds_increasing - n_funds_decreasing, 0 where fewer than 3 funds moved, from the newest 13F quarter whose filing deadline precedes the session` | `point_in_time.institutional.n_funds_increasing`, `point_in_time.institutional.n_funds_decreasing` |
| `return_120d_log_return` | log_return | 120 | no | `log(close) - log(close).shift(120)` | `close_raw` |
| `quality_pillar_pct` | pct | — | yes | `wmean(sector_pct(roe_ratio) .30, 100 - sector_pct(debt_to_equity_div2_ratio) .25, sector_pct(gross_margin_ratio) .25, sector_pct(current_ratio_div3_ratio) .20)` | `sector_raw`, `roe_ratio`, `debt_to_equity_div2_ratio`, `gross_margin_ratio`, `current_ratio_div3_ratio` |
| `value_pillar_pct` | pct | — | yes | `wmean(100 - sector_pct(pe_div30_ratio) .40, 100 - sector_pct(pb_div5_ratio) .30, sector_pct(fcf_yield_ratio) .30)` | `sector_raw`, `pe_div30_ratio`, `pb_div5_ratio`, `fcf_yield_ratio` |
| `momentum_pillar_pct` | pct | — | yes | `wmean(sector_pct(momentum_20d_log_return) .30, sector_pct(return_60d_log_return) .25, sector_pct(return_120d_log_return) .20, sector_pct(dist_from_52w_high_ratio) .15, sector_pct(momentum_5d_log_return) .10)` | `sector_raw`, `momentum_20d_log_return`, `return_60d_log_return`, `return_120d_log_return`, `dist_from_52w_high_ratio`, `momentum_5d_log_return` |
| `growth_pillar_pct` | pct | — | yes | `wmean(sector_pct(revenue_growth_3y_ratio) .30, sector_pct(eps_growth_3y_ratio) .30, sector_pct(sustainable_growth_rate_ratio) .25, sector_pct(capex_growth_5y_ratio) .15)` | `sector_raw`, `revenue_growth_3y_ratio`, `eps_growth_3y_ratio`, `sustainable_growth_rate_ratio`, `capex_growth_5y_ratio` |
| `stewardship_pillar_pct` | pct | — | yes | `wmean(100 - sector_pct(payout_ratio) .35, sector_pct(capex_growth_5y_ratio) .35, sector_pct(institutional_accumulation_raw) .30)` | `sector_raw`, `payout_ratio`, `capex_growth_5y_ratio`, `institutional_accumulation_raw` |
| `defensiveness_pillar_pct` | pct | — | yes | `wmean(100 - sector_pct(volatility_20d_ratio) .50, 100 - sector_pct(vol_ratio_10_60_ratio) .30, 100 - sector_pct(atr_14_ratio) .20)` | `sector_raw`, `volatility_20d_ratio`, `vol_ratio_10_60_ratio`, `atr_14_ratio` |
| `momentum_12_1_pillar_pct` | pct | — | yes | `wmean(sector_pct(mom_12_1_log_return) .40, sector_pct(return_120d_log_return) .25, sector_pct(dist_from_52w_high_ratio) .20, sector_pct(return_60d_log_return) .15)` | `sector_raw`, `mom_12_1_log_return`, `return_120d_log_return`, `dist_from_52w_high_ratio`, `return_60d_log_return` |
<!-- END GENERATED CATALOG TABLE -->

## Adding a column

1. Add a `FeatureSpec` to `CATALOG` in `crucible/features/registry.py`,
   carrying its unit, expression, description, inputs and session window.
2. Implement it in `crucible/features/compute.py`. The layer never fills and
   never looks ahead: a ticker without enough history gets a null, and the
   arm that consumes it drops the ticker with a recorded reason.
3. Regenerate this table (`uv run python -m crucible.features.docgen`).
4. The version changes. That is the mechanism working: a consumer reading the
   old prefix keeps reading the artifact its verdict was computed from.

A column no registered arm consumes is not added. A feature nothing reads is
a column that rots without anyone noticing it stopped being computed
correctly.
