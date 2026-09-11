# Stage 2B Q2 base MILP milestone

This milestone freezes the completed expected-cost stochastic dispatch model before any CVaR or other risk extension.

## Validated scope

- Formal operating period: 2025-02-01 through 2025-12-31, 334 days.
- Solver outcome: 334/334 days `Optimal` with PuLP--HiGHS.
- Uncertainty representation: 50 paired load--PV scenarios per day.
- Official workbook: `results/problem2/result2.xlsx`.
- Full test suite: 55 passed.
- `git diff --check`: passed.

## Terminal battery treatment

The final battery energy is exactly `6000.0 kWh`; all 334 daily results have `final_energy_kwh = 6000.0`.

The model uses a soft horizon-end treatment. For each day, the reference is the energy carried into that day, and nonnegative deviation variables satisfy

```text
E[144] - reference = terminal_above - terminal_below.
```

The objective includes the linear penalty

```text
penalty_rate * (terminal_above + terminal_below),
```

where the default penalty rate is `eta_d * max(price) = 1.25568 yuan/kWh`. The optimized terminal deviation and its penalty are zero on every formal day. The first day starts from the official `6000 kWh`; after each solve, `result.final_energy_kwh` is explicitly passed as the next day's initial energy. Because each optimal daily terminal value equals its carried reference, the sequence remains at `6000 kWh`.

This is not a hidden daily hard constraint. The only initial-state equality is `E[0] = inputs.initial_energy`. There is no constraint `E[144] = E[0]`, no constraint `E[144] = 6000`, and no daily assignment that resets the initial state to `6000`. The cross-day continuity test also starts from `7321.5 kWh` and verifies that the next day receives the preceding optimized final state.

No CVaR, Q3, or Q4 work is included in this milestone.
