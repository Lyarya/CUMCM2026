# Q3 rolling optimization audit

## Locked contracts

- Q2 physical baseline: `e86d18d785a6dda28b03d28167c1894b81a100b1`.
- Arya Q3 forecast milestone: `bca6cbfc1c275180023095576954a2e5bfee0bfd` (integrated as cherry-pick `135f7006d359dd7dc2b53442ce319937148480e9`).
- Risk setting: expected cost, lambda = 0.
- Main adjustment settlement: previous commitment.
- Sensitivity settlement: original 00:00 commitment.
- Actual execution calls the unchanged Q2 `settle_realized_day` function.

## Annual economic results

| schedule | release_hours | risk_setting | settlement_mode | formal_days | solver_success | planned_purchase_energy_kwh | planned_purchase_cost_yuan | model_expected_operating_cost_yuan | adjustment_energy_kwh | adjustment_increase_energy_kwh | adjustment_decrease_energy_kwh | adjustment_cost_yuan | realized_emergency_energy_kwh | realized_emergency_cost_yuan | realized_total_cost_yuan | emergency_interval_count | emergency_interval_rate | emergency_days | adjusted_interval_count | battery_throughput_kwh | mean_soc_kwh | minimum_soc_kwh | maximum_soc_kwh | final_soc_kwh | solver_runtime_seconds | maximum_scenario_power_balance_residual_kw |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S0 | 0 | lambda=0 expected-cost | SEQUENTIAL_PREVIOUS_COMMITMENT | 334 | 334 | 21813861.402375 | 13335360.457343 | 14387554.210064 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 366345.312534 | 1392610.139087 | 14727970.596431 | 10391 | 0.216047 | 332 | 0 | 11961878.716120 | 7352.828334 | 1200.000000 | 10800.000000 | 7834.328770 | 1413.722161 | 0.000000 |
| S1 | 0,6 | lambda=0 expected-cost | SEQUENTIAL_PREVIOUS_COMMITMENT | 334 | 334 | 21833071.455750 | 13343577.266057 | 14395802.348290 | 58627.605313 | 33542.095373 | 25085.509940 | 22736.378664 | 332290.996176 | 1266536.150011 | 14632849.794731 | 9951 | 0.206899 | 332 | 1787 | 11993554.284404 | 7341.923497 | 1200.000000 | 10800.000000 | 7787.660968 | 2723.183615 | 0.000000 |
| S2 | 0,6,12 | lambda=0 expected-cost | SEQUENTIAL_PREVIOUS_COMMITMENT | 334 | 334 | 21833872.361234 | 13343919.364468 | 14396144.446701 | 293363.901712 | 65098.441746 | 228265.459966 | 1433.571729 | 302254.535714 | 1160147.795416 | 14505500.731613 | 9640 | 0.200432 | 331 | 3241 | 12023747.937996 | 7312.747046 | 1200.000000 | 10800.000000 | 7781.221397 | 3373.833085 | 0.000000 |
| S3 | 0,6,12,18 | lambda=0 expected-cost | SEQUENTIAL_PREVIOUS_COMMITMENT | 334 | 334 | 21836446.807630 | 13345030.618356 | 14397255.700590 | 295374.391499 | 65319.670259 | 230054.721240 | 879.451543 | 302356.180086 | 1161808.074550 | 14507718.144450 | 9652 | 0.200682 | 331 | 3347 | 12027025.678836 | 7311.174120 | 1200.000000 | 10800.000000 | 7781.221397 | 3517.822307 | 0.000000 |

## Value of information

| schedule | realized_total_cost_yuan | voi_vs_s0_yuan | incremental_from_schedule | incremental_voi_yuan |
| --- | --- | --- | --- | --- |
| S0 | 14727970.596431 | 0.000000 | S0 | 0.000000 |
| S1 | 14632849.794731 | 95120.801699 | S0 | 95120.801699 |
| S2 | 14505500.731613 | 222469.864818 | S1 | 127349.063118 |
| S3 | 14507718.144450 | 220252.451981 | S2 | -2217.412836 |

## Physical and causal checks

### S0

- `realized_power_balance` = PASS
- `maximum_power_balance_residual_kw` = 1.8189894035458565e-12
- `realized_soc_recurrence` = PASS
- `maximum_soc_recurrence_residual_kwh` = 2.2737367544323206e-13
- `cross_day_realized_soc` = PASS
- `maximum_cross_day_soc_residual_kwh` = 0.0
- `soc_bounds` = PASS
- `actual_action_limits` = PASS
- `phantom_discharge` = PASS
- `unabsorbed_discharge_energy_kwh` = 0.0
- `hidden_export` = PASS
- `emergency_nonnegative` = PASS
- `executed_interval_freeze` = PASS
- `maximum_executed_freeze_residual_kw` = 0.0
- `causality` = PASS
- `settlement_reconciliation` = PASS
- `adjustment_objective_reconciliation` = PASS
- `maximum_adjustment_objective_residual_yuan` = 0.0
- `time_alignment` = PASS

### S1

- `realized_power_balance` = PASS
- `maximum_power_balance_residual_kw` = 1.8189894035458565e-12
- `realized_soc_recurrence` = PASS
- `maximum_soc_recurrence_residual_kwh` = 2.2737367544323206e-13
- `cross_day_realized_soc` = PASS
- `maximum_cross_day_soc_residual_kwh` = 0.0
- `soc_bounds` = PASS
- `actual_action_limits` = PASS
- `phantom_discharge` = PASS
- `unabsorbed_discharge_energy_kwh` = 0.0
- `hidden_export` = PASS
- `emergency_nonnegative` = PASS
- `executed_interval_freeze` = PASS
- `maximum_executed_freeze_residual_kw` = 0.0
- `causality` = PASS
- `settlement_reconciliation` = PASS
- `adjustment_objective_reconciliation` = PASS
- `maximum_adjustment_objective_residual_yuan` = 5.647143552778289e-13
- `time_alignment` = PASS

### S2

- `realized_power_balance` = PASS
- `maximum_power_balance_residual_kw` = 1.8189894035458565e-12
- `realized_soc_recurrence` = PASS
- `maximum_soc_recurrence_residual_kwh` = 2.2737367544323206e-13
- `cross_day_realized_soc` = PASS
- `maximum_cross_day_soc_residual_kwh` = 0.0
- `soc_bounds` = PASS
- `actual_action_limits` = PASS
- `phantom_discharge` = PASS
- `unabsorbed_discharge_energy_kwh` = 0.0
- `hidden_export` = PASS
- `emergency_nonnegative` = PASS
- `executed_interval_freeze` = PASS
- `maximum_executed_freeze_residual_kw` = 0.0
- `causality` = PASS
- `settlement_reconciliation` = PASS
- `adjustment_objective_reconciliation` = PASS
- `maximum_adjustment_objective_residual_yuan` = 9.094947017729282e-13
- `time_alignment` = PASS

### S3

- `realized_power_balance` = PASS
- `maximum_power_balance_residual_kw` = 1.8189894035458565e-12
- `realized_soc_recurrence` = PASS
- `maximum_soc_recurrence_residual_kwh` = 2.2737367544323206e-13
- `cross_day_realized_soc` = PASS
- `maximum_cross_day_soc_residual_kwh` = 0.0
- `soc_bounds` = PASS
- `actual_action_limits` = PASS
- `phantom_discharge` = PASS
- `unabsorbed_discharge_energy_kwh` = 0.0
- `hidden_export` = PASS
- `emergency_nonnegative` = PASS
- `executed_interval_freeze` = PASS
- `maximum_executed_freeze_residual_kw` = 0.0
- `causality` = PASS
- `settlement_reconciliation` = PASS
- `adjustment_objective_reconciliation` = PASS
- `maximum_adjustment_objective_residual_yuan` = 5.684341886080801e-13
- `time_alignment` = PASS

## Artifact protection

- Q2 and Arya forecast artifact hashes unchanged: `True`.
- Future Appendix 2 actual load/PV values are used only by physical execution and ex-post evaluation.
- Scenario count remains 50 and every scenario source date precedes its target date.
- No unabsorbed-discharge or hidden-export accounting channel exists.

## Final validation

- Annual solve status: `S0=S1=S2=S3=334/334 Optimal`.
- Result workbook: four official sheets; both purchase sheets are `335 x 147`, the storage sheet contains `334 x 6` data rows, and the emergency sheet contains all `334 x 144` ten-minute rows including zeros.
- Time labels: `0:00-0:10` through `23:50-0:00+1` on planned, adjusted, and emergency records.
- Workbook reconciliation: planned energy, planned cost, adjustment cost, emergency energy, five-times-tariff emergency cost, and realized total cost agree with the audited S3 CSV within `1e-4`.
- `pytest -q`: `80 passed`.
- `python src/smoke_test.py`: `PASS`.
- `paper/main.pdf`: `PASS`, XeLaTeX output is 20 pages; Q3 pages were rendered and visually inspected.
- `git diff --check`: `PASS`.
- Commit/push: `NO`; manual review remains required.

## Final close review

- Main-settlement ranking: `S2 < S3 < S1 < S0`; S2 = `{0,6,12}` has the lowest annual realized cost, `14,505,500.731613 yuan`.
- Main-settlement `VOI_S2 = 222,469.864818 yuan`.
- Main-settlement incremental `VOI_18 = -2,217.412836 yuan`; the negative value is retained without truncation.
- Original-00 sensitivity ranking is also `S2 < S3 < S1 < S0`; its incremental `VOI_18 = -2,216.935540 yuan`, so the economic conclusion is stable.
- Negative 18:00 VOI means the ex-post reduction in operating cost did not cover the associated adjustment/dispatch effects. It does not imply that the 18:00 forecast has worse statistical accuracy.
- `adjustment_energy_kwh` is gross adjustment: the sum of all increase and decrease magnitudes. The maximum identity residual for `gross = increase + decrease` is `1.17e-10 kWh`.
- `adjustment_cost_yuan` is net settlement after charging increases at `1.5p` and crediting decreases at `0.5p`; it is not gross energy multiplied by one average price.
- Maximum daily cost reconciliation residual is `1.10e-10 yuan`; maximum annual reconciliation residual is `5.83e-9 yuan`.
- S0 is not the Q2 main result. S0 uses the validated Q3 00:00 Attachment-3/fusion information layer, whereas Q2 uses its locked Q2 forecast interface; the physical execution contract and lambda-zero risk setting remain the same.
- Workbook review: official sheet order and role columns are preserved; the previously audited one-cell interval-label correction gives a complete `0:00-0:10` through `23:50-0:00+1` grid.
