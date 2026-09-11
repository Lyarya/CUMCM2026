# Q2 CVaR risk-sensitivity audit

Base commit: `e86d18d785a6dda28b03d28167c1894b81a100b1`. The locked physical settlement, forecast, 50 paired scenarios, and continuation value are unchanged.

The scenario loss is emergency-purchase cost only. For each day, $\mathrm{CVaR}_{\alpha}=\zeta+[(1-\alpha)S]^{-1}\sum_s\xi_s$ with $\xi_s\ge C_s^{emg}-\zeta$ and $\xi_s\ge0$. The objective adds $\lambda\mathrm{CVaR}_{\alpha}$ once; planned cost and continuation value are excluded from the scenario loss.

The sweep is a sensitivity analysis over stated risk preferences. Realized Appendix 2 outcomes were not used to select lambda.

| lambda | planned_purchase_cost_yuan | expected_scenario_emergency_cost_yuan | cvar_cost_yuan | realized_emergency_cost_yuan | realized_total_cost_yuan | solver_success |
| --- | --- | --- | --- | --- | --- | --- |
| 0.000000 | 13323201.223400 | 1051144.022760 | 5394358.762730 | 1516260.778930 | 14839462.002400 | 334 |
| 0.050000 | 13516409.597604 | 871370.440043 | 4671883.999910 | 1305446.650900 | 14821856.248504 | 334 |
| 0.100000 | 13688981.518054 | 742476.194425 | 4036914.448698 | 1163373.215129 | 14852354.733182 | 334 |
| 0.200000 | 14062893.890574 | 527325.158140 | 2874801.676619 | 918453.007617 | 14981346.898191 | 334 |
| 0.500000 | 14907937.261196 | 235972.889090 | 1114982.578431 | 577883.614165 | 15485820.875361 | 334 |

Causality audit: PASS. All recorded physical and accounting audits: PASS.
