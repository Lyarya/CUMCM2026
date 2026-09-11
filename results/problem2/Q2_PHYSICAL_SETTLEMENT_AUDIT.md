# Q2 physical settlement audit

## Scope

This audit covers the corrected expected-cost Q2 base model only. Forecasting, the 50 paired scenarios, the continuation-value formulation, CVaR, Q3 and Q4 were not changed.

## Physical correction

The previous result contained 138 intervals and 2688.603684605 kWh of `realized_unabsorbed_discharge`. That field was an accounting-only residual. It has been removed from the realized dispatch output.

Planned charge and discharge are now upper bounds on physical execution. Actual discharge is capped by the actual load-plus-charge deficit after PV and by available battery energy. Actual charge is capped by SOC headroom. The realized balance is

`grid_used + emergency + actual_PV + actual_discharge = actual_load + actual_charge + PV_spill`.

Actual SOC uses the actual actions:

`E_next = E_start + 0.9 * actual_charge * dt - actual_discharge * dt / 0.9`.

The next day starts from the preceding day's actual terminal SOC. There is no daily reset, daily energy-neutral constraint, export, battery spill, or unexplained physical sink.

## Full-year result

- Formal days: 334; Optimal: 334.
- Planned purchase energy: 21,793,145.182619 kWh.
- Planned purchase cost: 13,323,201.223427 yuan.
- Expected scenario emergency energy: 277,524.659704 kWh.
- Expected scenario emergency cost: 1,051,144.022760 yuan.
- Expected operating cost: 14,374,345.246186 yuan.
- Realized emergency energy: 398,161.900954 kWh.
- Realized emergency cost: 1,516,260.778930 yuan.
- Realized total cost: 14,839,462.002357 yuan.
- `old_reported_total_cost`: 14,907,862.919985 yuan, from commit `b642c0e` under the fixed Q2 tariff.
- Physical-settlement cost change versus `b642c0e`: -68,400.917628 yuan (-0.458824%).
- Actual charge/discharge energy: 6,601,473.526737 / 5,345,542.659214 kWh.
- Battery action clipping: 986,800.262152 kWh over 8,249 intervals.

The cost changed because clipped physical actions change the realized SOC passed to later days. It is not a forecasting, scenario, continuation-value, or risk-parameter change.

## Final cost consistency audit

Every one of the 48,096 emergency-purchase rows was recomputed as

`emergency_energy_kwh * matching Appendix 4 price_yuan_per_kwh * 5`.

Dates run from 2025-02-01 through 2025-12-31, with exactly 144 rows per day. The old comparison workbook is the version at commit `c9e4d08`, because that is the version whose emergency energy is 2,341,013.444 kWh. Rows were matched to the 144 Appendix 4 price columns by date and chronological slot, so the old display-label shift does not change the price join.

| Metric | New worktree | Old `c9e4d08` |
|---|---:|---:|
| Planned energy (kWh) | 21,793,145.182619 | 19,085,009.878810 |
| Planned cost from workbook (yuan) | 13,323,201.223427 | 11,402,778.052807 |
| Realized emergency energy (kWh) | 398,161.900954 | 2,341,013.444223 |
| Realized emergency cost using Appendix 4 (yuan) | 1,630,312.507253 | 8,580,363.261331 |
| Realized total using Appendix 4 emergency prices (yuan) | 14,953,513.730680 | 19,983,141.314138 |
| Average 5x emergency price (yuan/kWh) | 4.094596955 | 3.665234509 |
| Weighted ordinary Appendix 4 price (yuan/kWh) | 0.818919391 | 0.733046902 |

For both versions, `realized total = planned cost from workbook + recomputed emergency cost`. The numerical identity residuals are below `2e-9` yuan. Planned-energy totals also reconcile to the 144 interval cells within `2e-5` kWh.

The value 14,907,862.919985 yuan must not be paired with 2,341,013.444 kWh. The total comes from commit `b642c0e`, whose own emergency energy is 415,156.600521 kWh and whose fixed-Q2-tariff emergency cost is 1,581,277.506305 yuan. Pairing that total with the planned cost and emergency energy from `c9e4d08` would imply an effective emergency price of only 1.497251063 yuan/kWh, or an ordinary price of 0.299450213 yuan/kWh before the 5x multiplier. The correctly recomputed Appendix 4 weighted prices are 3.665234509 and 0.733046902 yuan/kWh respectively. The low implied price is therefore caused by mixing outputs from two different code versions, not by any valid row-level price calculation.

The Q2 implementation itself uses the fixed 144-slot tariff in `data/processed/C题/problem1_day.csv`; under that model tariff the new realized emergency cost and total are 1,516,260.778930 and 14,839,462.002357 yuan. Appendix 4 is the dynamic-price input reserved by the current project contract for Q4. The Appendix 4 totals above are an explicit same-basis audit requested for comparison and are not silently substituted into the Q2 model result.

## Invariants and reconciliation

- Maximum actual power-balance residual: 9.999999e-7 kW.
- Total absolute unexplained balance residual: 0.000642671 kWh, numerical round-off only.
- Maximum actual SOC recursion residual: 9.840001e-6 kWh.
- Actual SOC range: 1200.0 to 10800.0 kWh.
- Maximum cross-day actual SOC mismatch: 0.0 kWh.
- Simultaneous actual charge/discharge intervals: 0.
- Actual charge or discharge above plan: 0 intervals.
- Planned-grid-use bound violations: 0.
- PV-spill bound violations: 0.
- Unabsorbed-discharge output columns: 0.

`result2.xlsx` reconciles to the generated dispatch within CSV precision: planned purchase 21,793,145.182619 kWh, emergency purchase 398,161.900954 kWh, actual charge 6,601,473.526737 kWh and actual discharge 5,345,542.659214 kWh. Its sheets have shapes 335x147, 2005x6 and 48097x3. The plan and emergency sheets both start at `0:10-0:20` and end the first day at `0:00-0:10+1`.

## Former failure point

At 2025-09-06 19:10, planned discharge is 3613.603027 kW while actual discharge is limited to 2924.192900 kW. Planned grid use is zero, unused paid grid is 614.303454 kW, actual load is 2924.486000 kW, actual PV is 0.293100 kW, emergency purchase and PV spill are zero, and actual SOC moves from 7192.462115 to 6650.944911 kWh. The realized power-balance residual is zero.
