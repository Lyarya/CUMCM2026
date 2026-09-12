# Perfect-information offline benchmark audit

## Positioning

This is a non-deployable full-horizon benchmark using future actual load and PV. It does not replace the causal Q2/Q3 strategies or their official workbooks.

## Formulation

- Period: 2025-02-01 through 2025-12-31; 334 days and 48096 ten-minute intervals.
- Fixed Q2/Q3 tariff; no Attachment-4 dynamic price.
- One continuous SOC trajectory, initial energy 6000 kWh, no daily reset.
- No terminal equality; the resulting finite-horizon optimism is explicitly retained.
- Grid purchase is nonnegative; export is absent; PV curtailment is bounded by actual PV.
- Only charge/discharge mutual exclusion is relaxed.

## Result and audit

- Solver status: Optimal.
- Objective: 12228166.562875 yuan.
- Simultaneous charge/discharge count: 0.
- Maximum power-balance residual: 1.819e-12 kW.
- Maximum SOC-recurrence residual: 1.023e-12 kWh.
- Classification: exact physical perfect-information optimum for this instance.

## Boundary

The benchmark has noncausal access to future actuals and is not an implementable policy. Its unconstrained terminal SOC can make the lower bound optimistic near the end of the horizon.
