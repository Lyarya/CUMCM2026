# Stage 2B Q2 physical-settlement milestone

This milestone records the completed expected-cost stochastic dispatch and physical realized-settlement model before any CVaR or other risk extension.

## Validated scope

- Formal operating period: 2025-02-01 through 2025-12-31, 334 days.
- Solver outcome: 334/334 days `Optimal` with PuLP--HiGHS.
- Uncertainty representation: 50 paired load--PV scenarios per day.
- Official workbook: `results/problem2/result2.xlsx`.
- Full test suite: 58 passed.
- `git diff --check`: passed.

## Terminal and realized battery treatment

The planning model uses a decreasing piecewise-linear continuation value and has no daily terminal target or reset. The first day starts from the official `6000 kWh`.

Planned charge and discharge are upper bounds on actual execution. The realized settlement caps discharge by the actual absorbable deficit and available SOC, caps charge by SOC headroom, and advances SOC using the actual executed actions. Each next day receives the preceding day's actual terminal SOC. Actual SOC remains between `1200` and `10800 kWh`, the maximum cross-day mismatch is zero, and the final battery energy is `7834.330492 kWh`.

## Cost scope

Q2 uses the fixed 144-slot price profile from Appendix 1. Appendix 2 actual load and PV are used for backtesting, and emergency purchases are charged at five times the corresponding Appendix 1 price. The model expected total is `14,374,345.246186 yuan`; the realized backtest total is `14,839,462.002357 yuan`. Appendix 4 dynamic-price calculations are diagnostic only and are not the formal Q2 result.

No CVaR, Q3, or Q4 work is included in this milestone.
