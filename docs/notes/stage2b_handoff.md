# Stage 2B 预测与不确定性输入交接

本说明冻结 Stage 2A++ 已完成的 Q2 正式预测与不确定性产物，仅定义 Chongwen 在
Stage 2B 优化中应读取的输入。本文档不改变预测模型、不重新生成场景，也不包含 MILP
实现。

## 正式预测设定

- 正式负荷预测器：`Last Week`（上周同刻）。
- 正式光伏预测器：`7-day same-slot mean`（近 7 日同刻均值）。
- 正式日期：2025-02-01 至 2025-12-31，共 334 天。
- 每日预测时域：144 个 10 分钟区间。
- 时间步长：`dt = 1/6 h`。
- 点预测文件：`results/tables/forecasting/q2_forecast_predictions.csv`。
- 场景文件：`results/tables/forecasting/q2_scenarios.npz`。
- 场景清单：`results/tables/forecasting/q2_scenario_manifest.csv`。

## 场景构造与配对约束

每个目标日包含 50 个场景。Stage 2A 使用当时已经完整观测的历史预测残差，按整日
144 点轨迹进行有放回重采样。一次抽样同时选取同一个历史日期的负荷残差和光伏残差，
因此

```text
load_scenarios[s, :]
pv_scenarios[s, :]
```

必须以同一个 `s` 联合使用。`scenario_source_dates[s]` 是二者共享的残差来源日期。
不得分别打乱、排序或重新抽取负荷与光伏场景。全部来源日期严格早于目标日；光伏场景
仅施加 `>= 0 kW` 的物理下界，因为题目没有提供额定光伏容量上界。Stage 2B 可将 50
个自助场景视为等概率样本，除非之后另行明确场景权重方案。

## Chongwen 应调用的接口

```python
from src.problem2.forecast_interface import get_q2_day_inputs

day = get_q2_day_inputs("2025-02-01", initial_energy=6000.0)
```

函数：

```text
src/problem2/forecast_interface.py::get_q2_day_inputs(date, initial_energy)
```

返回只读 `Q2DayInputs`：

| 字段 | 形状 | 单位 | 含义 |
|---|---:|---|---|
| `date` | 标量 | 日期 | 运营日 |
| `load_forecast` | `[144]` | kW | 正式负荷点预测 |
| `pv_forecast` | `[144]` | kW | 正式光伏点预测 |
| `load_scenarios` | `[50, 144]` | kW | 成对负荷场景 |
| `pv_scenarios` | `[50, 144]` | kW | 成对光伏场景 |
| `price` | `[144]` | 元/kWh | Q2 使用的附件 1 固定日内购电价 |
| `initial_energy` | 标量 | kWh | 由优化器调用方传入的当日初始电量 |
| `timestamps` | `[144]` | 区间结束时刻 | 当日 00:10 至次日 00:00 |
| `scenario_source_dates` | `[50]` | 日期 | 负荷/PV 共享的场景来源索引 |
| `dt_hours` | 标量 | h | `1/6` |

完整性审计函数：

```text
src/problem2/forecast_interface.py::audit_q2_handoff_integrity()
```

该函数复核正式产物哈希、原始附件哈希、场景配对、正式日期与严格因果条件。

## 优化器输入契约

1. `actual_load` 和 `actual_generation` 仅用于 Stage 2A 样本外评价，不属于优化器输入，
   接口不会返回这两列。
2. 对场景 `s` 建立约束时，必须同时使用
   `load_scenarios[s, :]` 与 `pv_scenarios[s, :]`。
3. `price` 是 Q2 的确定性固定日内电价；附件 4 的动态电价属于 Q4，不能在 Q2 中混用。
4. 首个正式日可按题设传入 `initial_energy=6000.0`。从第二天起，应传入上一运营日求解
   得到的日末电量，满足
   `E_(d+1,0) = E_(d,24)`。
5. Q2--Q4 不得自动继承 Q1 的每日约束 `E_start = E_end = 6000`。规划期末电量、终端
   价值或扩展视界应由 Stage 2B 明确建模，不能由本接口暗中重置。
6. 功率变量使用 kW，进入能量平衡或费用计算时乘以 `dt_hours = 1/6 h`；储能电量使用
   kWh，电价使用元/kWh。
7. 本接口只负责不可变输入契约，不创建购电、充放电、SOC、紧急购电或 CVaR 决策变量。
