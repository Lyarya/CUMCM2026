# Q1–Q3 manuscript and figure redesign

Base: `25ed90cf38fa47e74149f88def968223e7bfbf99`, branch `arya`.
Scope: presentation only; no model, result, workbook, forecast, or raw-data change.

## Evidence and writing contract

One framework extends deterministic storage dispatch to uncertain day-ahead planning and sequential information updates. Forecast accuracy is intermediate; realized procurement and information value are the decision criteria. No alternative-forecast dispatch comparison is claimed. Q4 remains unfinished.

Canonical notation: power g/c/d/w in kW; E is energy in kWh; p is price in yuan/kWh; actual PV P versus forecast hat P; risk weight lambda_r; scenario s; fusion weight theta_b. LP equality is an instance-level observation, not a theorem. Statistical significance does not remove serial-dependence limitations.

Read: official C statement, full local modeling plan, README, local development rules, Q1 implementation and LP audit, Q2 physical/CVaR audits and summary, Q3 rolling audit, current 32-page PDF, TeX, saved result tables, existing plotting sources. The master plan contains superseded proposals: use final audited implementations, not unselected low-rank experts, total-cost CVaR, or unimplemented greedy/aging layers.

## Skills and adaptation

Python is the saved Nature Figure backend. Use Nature Figure integrity/layout/export checks and Nature Writing evidence-first structure; Chinese competition prose overrides English/journal defaults. No literal `nature.skill` is needed: installed nature-figure/nature-writing provide the relevant rules.

ModelViz catalog and trend templates were inspected. Its automatic service cannot run in this environment because langchain_core is absent. Fallback is manual style-only adaptation: time-series marker conventions, shared legends and small multiples. Do not inherit simulated data, spline smoothing, remote-sensing lag semantics, automatic deletion code, or font changes. Existing project palette and Songti SC are authoritative.

## FIGURE QA INVENTORY (original PDF numbering)

| Original | Decision | Evidence / action |
|---|---|---|
| 1 representative day | REDESIGN | Keep saved representative date; separate labels from curves; omit dynamic-price trace from Q1–Q3 EDA |
| 2 seasonal PV | MERGE | Merge with 3; distinct markers/linestyles, extra top margin |
| 3 PV lag correlation | MERGE | Show full [-1,1] range, not truncated negative correlations |
| 4 official lead error | MERGE | Same evidence repeated by 12; consolidate in Q3 |
| 5 Hankel spectrum | REMOVE | Summarize saved diagnostic in text; not the selected predictor |
| 6 lowrank versus forecastability | REMOVE | Preserve limitation in text; no causal interpretation |
| 7 Q1 dispatch | MERGE | Shared-time-axis panels with 8; all 144 intervals retained |
| 8 Q1 SOC/price | MERGE | Separate units; detail view of final 3 hours |
| 9 sensitivity | REDESIGN | 12 saved deterministic cases; no fabricated uncertainty |
| 10 forecast comparison | REDESIGN | Keep full 8-model comparison table; figure shows paired daily differences instead of repeating it |
| 11 risk tradeoff | REDESIGN | Single cost-risk curve with lambda labels; not annual-total CVaR |
| 12 official lead error | MERGE | Two panels: full-year official metrics and matched OOS official/fusion comparison |
| 13 schedule cost | MERGE | Merge with 14; zero-origin cost overview and explicitly labelled local detail |
| 14 incremental VOI | MERGE | Retain negative 18:00 value, zero baseline, separated annotations |
| 15–18 Q4 price figures | KEEP | Existing independent work preserved; no Q4 solver/results |

All new assets go to paper/figures/redesign; generated presentation tables to paper/contents/generated. Original scripts and results remain unchanged. Descriptive IQR bars/bands refer to actual days, not confidence intervals. No smoothing or fabricated replications. Source hashes and regression checks are reported separately.

## Known pending issues

AI-use dates/model identities and human-verification declarations require author completion. Q2 workbook has a previously audited legacy interval-label shift; this task must not edit result2.xlsx. This is a delivery-format limitation, not a change in locked monetary results. Official requested-date tables must be checked against saved results rather than filled with invented entries.
