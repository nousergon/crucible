# Crucible v2 — board for trading day 2026-09-24
store: `s3://store/crucible/board/current.json`  
generated: 2026-09-25T01:29:28Z  commit: `9aeb864be8eb2de0ae70026b9fa60af298aca708`  
delivered: 2026-09-25 03:00 PDT

## Ladder
| phase | state | clauses met | holding |
|---|---|---|---|
| phase:phase0 | MET | 5/5 |  |
| phase:phase1 | UNMET | 2/6 | arc_runs_ok, arms_all_scored, attribution_renders, explain_walks_a_verdict |
| phase:phase2 | OUT_OF_ORDER | 5/7 | replays_ok, aws_cost_within_ceiling |
| | | | out of order: 5/7 clauses met; unmeasurable: aws_cost_within_ceiling; holding: replays_ok; grades all 6 of 6 alpha-engine-config-I9758 deliverables; graded while phase1 (alpha-engine-config-I9757) is UNMET — a later phase may not be exited ahead of an earlier one |
| phase:phase3 | OUT_OF_ORDER | 4/12 | m_promotion_or_verdict_backed_non_promotion, r_promotion_or_verdict_backed_non_promotion, s_promotion_or_verdict_backed_non_promotion, u_promotion_or_verdict_backed_non_promotion, portfolio_engine_used_by_s_slot, named_transaction_cost_model, factor_neutral_attribution, v1_arms_carried_or_excluded |
| | | | out of order: 4/12 clauses met; unmeasurable: m_promotion_or_verdict_backed_non_promotion, s_promotion_or_verdict_backed_non_promotion, portfolio_engine_used_by_s_slot, named_transaction_cost_model, factor_neutral_attribution; holding: r_promotion_or_verdict_backed_non_promotion, u_promotion_or_verdict_backed_non_promotion, v1_arms_carried_or_excluded; grades 4 of 5 alpha-engine-config-I9759 deliverables; not gate-readable: benchmark_per_slot; graded while phase1 (alpha-engine-config-I9757) is UNMET — a later phase may not be exited ahead of an earlier one |
| phase:phase4 | OUT_OF_ORDER | 2/11 | trader_one_week_on_v2_champion, broker_reconciliation_control_arm_passed, execution_shortfall_row_graded, shadow_books_cover_every_active_arm, kill_switch_fire_drill_passed, aws_total_within_ceiling, old_sf_execution_count_zero, data_cutover_ready, data_collection_reliability |
| | | | out of order: 2/11 clauses met; unmeasurable: aws_total_within_ceiling; holding: trader_one_week_on_v2_champion, broker_reconciliation_control_arm_passed, execution_shortfall_row_graded, shadow_books_cover_every_active_arm, kill_switch_fire_drill_passed, old_sf_execution_count_zero, data_cutover_ready, data_collection_reliability; grades 9 of 15 alpha-engine-config-I9760 deliverables; not gate-readable: trader_release_pin_ib_paper_smoke, portfolio_adopted_by_trader, lambdas_and_alarms_removed, artifact_registry_tombstoned, codebuild_consumers_retired, system_optimized_doc_rewritten; graded while phase1 (alpha-engine-config-I9757) is UNMET — a later phase may not be exited ahead of an earlier one |
| phase:phase5 | OUT_OF_ORDER | 0/1 | every_llm_arm_has_a_verdict_within_one_cycle |
| | | | out of order: 0/1 clauses met; unmeasurable: every_llm_arm_has_a_verdict_within_one_cycle; grades 1 of 4 alpha-engine-config-I9761 deliverables; not gate-readable: judge_calibration, pit_constituent_lists, leave_one_out_ablation_replay; graded while phase1 (alpha-engine-config-I9757) is UNMET — a later phase may not be exited ahead of an earlier one |
| phase:phase0:earliest-satisfiable | MET | — | phase 0 already meets its exit gate |
| phase:phase1:earliest-satisfiable | PLANNED | — | phase 1 is unmet, but no unmet clause could derive an earliest-satisfiable date this render |
| phase:phase2:earliest-satisfiable | PLANNED | — | phase 2 is unmet, but no unmet clause could derive an earliest-satisfiable date this render |
| phase:phase3:earliest-satisfiable | PLANNED | — | phase 3 is unmet, but no unmet clause could derive an earliest-satisfiable date this render |
| phase:phase4:earliest-satisfiable | PLANNED | — | phase 4 is unmet, but no unmet clause could derive an earliest-satisfiable date this render |
| phase:phase5:earliest-satisfiable | PLANNED | — | phase 5 is unmet, but no unmet clause could derive an earliest-satisfiable date this render |
| phase:phase0:closing | MET | — | alpha-engine-config-I9756 is closed and gates/phase0/closing.json records the reading that justified it: 5/5 clauses met on trading day 2026-09-08, at commit c06e4efb90c31b6b84e5e4d4925688ee2ea99054. |
| phase:phase1:closing | MET | — | alpha-engine-config-I9757 is closed and gates/phase1/closing.json records the reading that justified it: 6/6 clauses met on trading day 2026-09-08, at commit c06e4efb90c31b6b84e5e4d4925688ee2ea99054. |
| phase:phase2:closing | MET | — | alpha-engine-config-I9758 is closed and gates/phase2/closing.json records the reading that justified it: 7/7 clauses met on trading day 2026-09-11, at commit ef324652454d1252ca1fbe6ee338172ce9dd0480. |
| phase:phase3:closing | PLANNED | — | alpha-engine-config-I9759 is open and no closing record has been filed. That is the expected state of a phase that has not exited; the record is written by the first live `crucible gate` reading that reads MET. |
| phase:phase4:closing | PLANNED | — | alpha-engine-config-I9760 is open and no closing record has been filed. That is the expected state of a phase that has not exited; the record is written by the first live `crucible gate` reading that reads MET. |
| phase:phase5:closing | PLANNED | — | alpha-engine-config-I9761 is open and no closing record has been filed. That is the expected state of a phase that has not exited; the record is written by the first live `crucible gate` reading that reads MET. |

board generated 2026-09-25T01:29:28Z; gates/ladder.json generated 2026-09-25T01:24:26Z

## Schedule (plan §6.1)
| milestone | state | detail |
|---|---|---|
| OVERDUE schedule:day5_replays | UNMET | OVERDUE since 2026-09-04 (plan 2026-09-06) — reads: phase1 2/6 |
| OVERDUE schedule:days6_7_scheduler | UNMET | OVERDUE since 2026-09-08 (plan 2026-09-08) — reads: phase2 5/7 |
| OVERDUE schedule:live1 | UNMET | OVERDUE since 2026-09-11 (plan 2026-09-12) — reads: phase2 5/7 |
| OVERDUE schedule:live2_cutover | UNMET | OVERDUE since 2026-09-18 (plan 2026-09-19) — reads: phase2 5/7 |
| schedule:live3 | PLANNED | due 2026-09-25 (plan 2026-09-26), waiting on phase2 5/7 |

## Acceptance
acceptance count: 22 met / 2 unmet / 0 unmeasurable of 24 (commit 9efbd4fcd4b7348b85b22b64494c23c8bf38ff88)
  unmet: TestAttribution::test_alpha_is_factor_neutral_not_raw_excess_return, TestAutonomy::test_the_ruled_live_gate_two_live_saturdays_plus_five_replayed
  the artifact names no unmeasurable clause ids

## Moved since 2026-09-23
nothing moved

not yet due (4) — RUNNING or ARMED in either board, so the change is schedule phase, not a move:
- component:acceptance.publish: UNMEASURED -> MET
- component:deploy: UNMEASURED -> MET
- component:smoke: UNMEASURED -> MET
- component:trader.session: ABSENT -> UNMEASURED

## Silence
silence: 30 UNMEASURED, 0 UNMEASURABLE of 81 rows
pending operator action: none exposed by the board's producer

## Board
https://console.example/decision?pipeline=crucible-board
fleet console — the Decision list filtered to this board, every row of board/current.json and none hidden; stable address, no expiry