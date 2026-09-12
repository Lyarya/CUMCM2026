# Q4 Design Checkpoint

## Scope and locked predecessors

- Working branch: `chongwen`.
- Q3 final milestone imported from `99797c4bfe098c47cb469766a5d920ff8cf478a0` and represented on this branch by cherry-pick `5e5ad2b4eaafca6d1eb3b172cb157ecbdec8ea07`.
- The Q2 physical baseline and Q3 rolling model remain the authoritative implementations. Q4 adds a price provider and accounting adapter; it does not copy the optimizer, battery equations, SOC semantics, forecasts, or paired scenario generator.
- This checkpoint covers only 2025-02-01 and 2025-02-02. No 334-day Q4 run was started.

## Q4 design summary

### Q4-2

Q4-2 reuses the locked Q2 chain unchanged: causal load/PV point forecasts, 50 paired whole-day residual scenarios, expected-cost two-stage stochastic MILP, the existing decreasing-marginal continuation value, realized physical execution, and cross-day actual SOC. The only optimizer input replaced is the 144-element price vector. Planned purchase uses `p_t`; scenario emergency purchase uses `5 p_t`.

### Q4-3

Q4-3 reuses the locked Q3 chain unchanged: causal forecast fusion, S0--S3 releases, frozen executed intervals, `SEQUENTIAL_PREVIOUS_COMMITMENT` as the main settlement, the original-00 rule as sensitivity, and the Q2 realized physical execution. The price vector is the only model input replaced. Initial purchase uses `p_t`, an increase uses `1.5 p_t`, a reduction is settled at `0.5 p_t`, and emergency purchase uses `5 p_t`.

Every coefficient is the price of its own 10-minute interval. No daily average or manually constructed peak/flat/valley tariff is used. Peak-valley battery behavior remains an outcome of the MILP.

## Price information interpretation

Classification: `AMBIGUOUS`.

The official wording states that external-grid prices fluctuate in real time and asks Q4 to recalculate Q2 and Q3 using the prices in Attachment 4. Attachment 4 supplies one price for every 10-minute interval of 2025. The statement does not say whether a decision maker at 00:00, 06:00, 12:00, or 18:00 knows the remaining realized prices of that day.

Two explicit interfaces are therefore retained pending a human choice:

- `GIVEN_PRICE`: maps to the reviewed `oracle_perfect_information` interface and treats Attachment 4's full future path as given at the decision time. This is a perfect-information/oracle interpretation.
- `CAUSAL_PRICE`: maps to `causal_forecast`. A ridge price model selected only on 2025-01-17 through 2025-01-30, using strictly historical lag/calendar features, is frozen before the formal 2025-02-01 through 2025-12-31 period. Its 00:00 forecast is reused at later Q3 releases without intraday price assimilation.

The prior solution outline recommended a causal main case plus an oracle information-value sensitivity, but that recommendation is not promoted to a locked main model because the statement is ambiguous. The same outline also mentioned CVaR, alternative forecast experts, seasonally stratified scenario sampling, and alternate terminal treatments. Those suggestions conflict with, or extend beyond, the latest audited Q2/Q3 implementations and are not introduced here.

## Thin implementation

- `src/problem4/price_*` is the existing reviewed Arya price-information layer from commit `4c5ddfcdbf2f2eb2ebf6e14987ad17568c6f2e90`.
- `src/problem4/dispatch_adapter.py` passes either full-day dynamic price vector into the locked Q2/Q3 functions and uses Attachment 4's realized price for ex-post settlement.
- `src/problem3/rolling_dispatch.py` accepts an optional price provider. Its default remains the fixed Q2 price, so the locked Q3 execution path is preserved.
- Checkpoints bind the selected mode and SHA-256 signature of the persisted price forecast and selection artifacts, preventing resume under a different price input.
- Q4-3 retains Q3 checkpoint/resume, freeze, physical, causality, and settlement audits. Q4-2 has a compatible sample checkpoint and repeats realized power, SOC, cross-day, phantom-discharge, price-alignment, and cost-reconciliation audits.

## Two-day smoke result

Dates: 2025-02-01 and 2025-02-02.

Both modes produced `2/2 Optimal` for Q4-2 and for every Q4-3 schedule S0, S1, S2, and S3. All tested cases passed:

- exact 144-slot decision and realized-price alignment;
- 5-times emergency-price accounting;
- realized power balance;
- actual SOC recurrence and bounds;
- cross-day actual SOC transfer;
- phantom discharge and unabsorbed discharge checks;
- settlement reconciliation;
- S0--S3 executed-interval freeze;
- load/PV causality and, for `CAUSAL_PRICE`, price causality.

Maximum observed physical residuals were `9.094947017729282e-13 kW` for power balance and `2.2737367544323206e-13 kWh` for SOC recurrence. Unabsorbed discharge was exactly `0 kWh`; freeze and dynamic-price alignment residuals were zero.

Two-day realized costs are smoke diagnostics only:

| Mode | Q4-2 | S0 | S1 | S2 | S3 |
|---|---:|---:|---:|---:|---:|
| GIVEN_PRICE | 86,693.359 | 86,539.722 | 86,501.806 | 85,629.711 | 85,639.862 |
| CAUSAL_PRICE | 87,533.930 | 87,295.433 | 87,255.472 | 86,349.847 | 86,352.671 |

The uncached complete smoke took 241.39 seconds. Linear scaling gives an engineering estimate of about 4.5 hours for all Q4-2/Q4-3 schedules under `GIVEN_PRICE`, 6.7 hours under `CAUSAL_PRICE`, or roughly 11.2 hours for both modes on this machine. Solver time may vary; this estimate is not an annual result.

## Decision gate

The implementation is technically ready for an annual run, but the annual run is deliberately blocked until a human selects the main price-information interpretation. The recommended reporting design is `CAUSAL_PRICE` as the main causal analysis and `GIVEN_PRICE` as a perfect-information sensitivity/oracle bound, subject to that decision.

Machine-readable smoke evidence is in `results/problem4/tables/q4_dispatch_smoke.json`.
