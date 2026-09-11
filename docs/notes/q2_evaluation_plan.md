# Q2 经济、可靠性与风险评价交接方案

本评价层直接读取 Chongwen 已锁定的 Q2 物理结算结果。评价器不训练预测模型、不重新生成
场景、不建立或复制 MILP，也不提前实现 CVaR；最终 Q2 求解与物理校验仍由
`src/problem2/model.py`、`run.py` 和 `evaluate.py` 负责。

## 分工边界

Arya 负责：

- 预测与成对场景的完整性复核；
- 日度及 2025-02-01 至 2025-12-31 全周期经济指标；
- 紧急购电、弃光和求解可靠性评价；
- 无储能、确定性、期望成本随机优化及后续风险策略的统一比较；
- 风险参数扫描汇总、成本—风险权衡点识别和论文图表；
- Q2 结果解释，强调预测精度不等于下游决策价值。

Chongwen 负责：

- Q2 基础期望成本随机 MILP；
- 购电、储能充放电、SOC、紧急购电和弃光决策；
- 334 天跨日连续的顺序求解与求解器状态记录；
- 真实负荷/光伏发生后的 realized dispatch 核算；
- `result2.xlsx` 的最终生成与模板验证。

## 优化器输出契约

Arya 的入口位于：

```text
src/problem2/evaluation_analysis.py::load_final_q2_evaluation()
src/problem2/evaluation_analysis.py::DailyOptimizerResult
src/problem2/evaluation_analysis.py::summarize_q2_backtest(results)
```

每个正式日必须提供：

| 字段 | 形状 | 单位 | 解释 |
|---|---:|---|---|
| `date` | 标量 | 日期 | 运营日 |
| `planned_grid_purchase_kw` | `[144]` | kW | 日前计划购电功率 |
| `charge_kw` | `[144]` | kW | 计划充电功率 |
| `discharge_kw` | `[144]` | kW | 计划放电功率 |
| `soc_kwh` | `[145]` | kWh | 区间边界储能电量 |
| `emergency_purchase_kw` | `[144]` 或 `[...,144]` | kW | 真实或场景紧急购电功率 |
| `spill_kw` | `[144]` 或 `[...,144]` | kW | 真实或场景弃光功率 |
| `planned_cost_yuan` | 标量 | 元 | 已聚合的计划购电成本 |
| `emergency_cost_yuan` | 标量 | 元 | 已聚合的紧急购电成本 |
| `total_cost_yuan` | 标量 | 元 | 前两项之和 |
| `initial_energy_kwh` | 标量 | kWh | 当日初始储能电量 |
| `final_energy_kwh` | 标量 | kWh | 当日最终储能电量 |
| `solver_status` | 标量 | -- | 求解器状态 |
| `solver_runtime_seconds` | 标量 | s | 求解时间 |
| `price_yuan_per_kwh` | `[144]`，建议提供 | 元/kWh | 用于独立复核成本 |
| `scenario_weights` | `[S]`，可选 | -- | 缺省时各场景等权 |
| `evaluation_basis` | 标量 | -- | `realized` 或 `scenario_expected` |

功率轨迹由评价器统一乘以 `dt=1/6 h` 转为电量。三个日成本字段是优化器已经聚合的
金额，评价器直接使用；提供电价时只做 `power * price * dt` 一致性复核，绝不再次乘时间步。
紧急购电成本按 `5 * price` 复核。

正式适配器从 `table_p2_dispatch.csv` 读取 `[144]` 实际充放电、紧急购电、弃光及
`[145]` 实际 SOC，从 `table_p2_daily_summary.csv` 读取实际首末 SOC 和三类成本。
计划充放电仅作为求解器动作上限，不会替代实际动作进入评价；评价层也不会利用实测数据
自行重建、裁剪或改变物理结算。

当前权威实测口径为：计划购电成本 13,323,201.223427 元、实际紧急购电成本
1,516,260.778930 元、实际总成本 14,839,462.002357 元。旧临时总成本
14,907,862.919985 元不得再作为 Q2 基准输入。

## 日度与全周期指标

`daily_metrics(...)` 输出计划、紧急与总成本，计划与紧急购电量，紧急购电占总购电量比例，
弃光量，充放电量和电池吞吐量，首末及最小/最大储能电量，紧急购电日标记、最大紧急
功率、求解状态和运行时间。

`summarize_q2_backtest(...)` 要求恰好覆盖 334 个正式日，并检查：

```text
final_energy(day d) ~= initial_energy(day d+1)
```

Q2 不继承 Q1 的每日 `6000 -> 6000 kWh` 约束。首日按题设从 6000 kWh 开始，之后必须
跨日连续；12 月 31 日末电量和 horizon-end 处理由优化模型明确给出。

全周期汇总包括累计三类成本、累计计划/紧急购电量、紧急购电占总购电量比例、弃光量、
电池吞吐量、紧急购电日数及比例、最大单日紧急购电量、最大单日总成本、求解成功率、
平均/累计运行时间和最终电量。

## 基线与决策价值比较

`compare_methods(...)` 接收同一日期集合下的结果，支持：

1. 无储能、直接购电基线；
2. 点预测确定性 MILP；
3. 基础期望成本随机 MILP；
4. 未来风险感知/CVaR 或安全裕度变体。

统一输出计划成本、紧急成本、总成本、紧急购电量、弃光量、电池吞吐量及求解时间。节省
定义为：

```text
cost_saving = baseline_total_cost - strategy_total_cost
```

因此正值表示策略节省费用。比较必须基于 realized downstream dispatch，而不能用 RMSE
变化代替经济收益。

## 未来风险扫描（当前不实现优化）

`summarize_risk_sweep(results_by_risk_parameter)` 不限定风险参数是 `lambda_CVaR` 还是
`safety_margin_alpha`，仅按参数值排序并汇总成本、紧急购电量、紧急购电日、弃光、电池
吞吐量、期末电量与求解时间。辅助函数可返回最低总成本、最低紧急购电量以及简单的成本—
风险非支配点。

风险扩展的执行顺序必须是：

```text
基础期望成本随机 MILP
  -> 验证 334 天跨日连续结果
  -> 接入 Arya 评价器
  -> 再考虑 CVaR/安全裕度扩展
  -> 风险—成本敏感性分析
  -> 写入论文结果
```

## 图表与输出表

绘图入口位于 `src/problem2/visualize.py`：

- `plot_daily_economic_performance`：上图为相对基线的日成本节省，下图为累计节省；
- `plot_daily_cost_decomposition`：计划、紧急和总成本的逐日分解；
- `plot_risk_cost_tradeoff`：成本与紧急购电量分面展示，避免混合单位双轴；
- `plot_soc_emergency_diagnostic`：代表日计划购电、紧急购电与 SOC 诊断。

所有绘图函数只在传入真实优化结果后才保存 SVG、PDF、JPG、PNG，不会生成虚假结果。

`write_evaluation_tables(...)` 只写入调用方实际提供且非空的表，预定义输出为：

```text
results/problem2/q2_daily_evaluation.csv
results/problem2/q2_period_summary.csv
results/problem2/q2_method_comparison.csv
results/problem2/q2_risk_sweep.csv
```

当前只提供 schema 和写出函数，不创建任何占位数值文件。后续论文结果结构依次为：预测与
不确定性摘要、随机调度模型、334 天经济/可靠性回测、基线比较、风险—成本敏感性以及代表
日调度/SOC 图。Stage 2A++ 已验证的预测正文保持不变。
