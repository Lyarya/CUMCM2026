# Q1 LP 松弛验证审计

## 实验边界

正式 MILP 直接调用 `src/problem1/model.py`；LP 松弛在独立模块中复现同一目标函数、
144 个时段、功率平衡、SOC 递推、功率边界、SOC 首末条件、弃光边界和非负购电约束，
只删除二元充放电模式变量及其互斥整数约束。未修改正式模型与 `result1.xlsx`。

## 同口径结果

| formulation   | solver_status   |   objective_cost_yuan |   total_grid_purchase_kwh |   total_charge_kwh |   total_discharge_kwh |   total_curtailment_kwh |   minimum_soc_energy_kwh |   maximum_soc_energy_kwh |   maximum_simultaneous_charge_discharge_kw |   maximum_power_balance_residual_kw |   maximum_soc_recurrence_residual_kwh |   solver_runtime_seconds |
|:--------------|:----------------|----------------------:|--------------------------:|-------------------:|----------------------:|------------------------:|-------------------------:|-------------------------:|-------------------------------------------:|------------------------------------:|--------------------------------------:|-------------------------:|
| MILP          | Optimal         |           35126.94859 |                 59482.699 |        20740.66613 |           16799.93957 |                       0 |                     1200 |                    10800 |                                          0 |                     4.000000001e-05 |                       0.0004700000009 |            0.210641417   |
| LP relaxation | Optimal         |           35126.94859 |                 59482.699 |        20740.66613 |           16799.93957 |                       0 |                     1200 |                    10800 |                                          0 |                     4.000000001e-05 |                       0.0004700000009 |            0.06687845901 |

松弛间隙定义为 `MILP objective - LP objective`，本次为
`0` 元；零间隙容差为 `1e-05` 元，
自然互斥容差为 `1e-06` kW。

## 调度差异

```json
{
  "maximum_absolute_grid_purchase_kw_difference": 1999.9999999999995,
  "maximum_absolute_charge_kw_difference": 2000.0,
  "maximum_absolute_discharge_kw_difference": 0.0,
  "maximum_absolute_spill_kw_difference": 0.0,
  "maximum_absolute_storage_start_kwh_difference": 300.0,
  "maximum_absolute_storage_end_kwh_difference": 300.0,
  "intervals_with_any_schedule_difference_above_1e_6": 5,
  "materially_different_intervals_above_1e_3": 3,
  "differing_intervals": [
    {
      "slot": 79,
      "interval_start": "13:00",
      "milp_grid_kw": 0.0,
      "lp_grid_kw": 0.0,
      "milp_charge_kw": 1455.0453,
      "lp_charge_kw": 1455.0453,
      "milp_discharge_kw": 0.0,
      "lp_discharge_kw": 0.0,
      "milp_storage_end_kwh": 9874.202,
      "lp_storage_end_kwh": 9874.2019
    },
    {
      "slot": 80,
      "interval_start": "13:10",
      "milp_grid_kw": 0.0,
      "lp_grid_kw": 0.0,
      "milp_charge_kw": 1338.4535,
      "lp_charge_kw": 1338.4535,
      "milp_discharge_kw": 0.0,
      "lp_discharge_kw": 0.0,
      "milp_storage_end_kwh": 10074.97,
      "lp_storage_end_kwh": 10074.97
    },
    {
      "slot": 140,
      "interval_start": "23:10",
      "milp_grid_kw": 5416.1518,
      "lp_grid_kw": 3416.1518,
      "milp_charge_kw": 2000.0,
      "lp_charge_kw": 0.0,
      "milp_discharge_kw": 0.0,
      "lp_discharge_kw": 0.0,
      "milp_storage_end_kwh": 5250.0,
      "lp_storage_end_kwh": 4950.0
    },
    {
      "slot": 141,
      "interval_start": "23:20",
      "milp_grid_kw": 8421.9811,
      "lp_grid_kw": 8421.9811,
      "milp_charge_kw": 5000.0,
      "lp_charge_kw": 5000.0,
      "milp_discharge_kw": 0.0,
      "lp_discharge_kw": 0.0,
      "milp_storage_end_kwh": 6000.0,
      "lp_storage_end_kwh": 5700.0
    },
    {
      "slot": 142,
      "interval_start": "23:30",
      "milp_grid_kw": 3429.6923,
      "lp_grid_kw": 5429.6923,
      "milp_charge_kw": 0.0,
      "lp_charge_kw": 2000.0,
      "milp_discharge_kw": 0.0,
      "lp_discharge_kw": 0.0,
      "milp_storage_end_kwh": 6000.0,
      "lp_storage_end_kwh": 6000.0
    }
  ]
}
```

## 文件完整性

- 受保护文件运行前后哈希一致：`True`
- 受保护文件：`data/processed/C题/problem1_day.csv`, `src/problem1/model.py`, `results/problem1/result1.xlsx`, `data/raw/C题/附件/附件1.xlsx`, `data/raw/C题/附件/**`, `src/problem2/**`, `src/problem3/**`, `src/problem4/**`, `results/problem2/**`, `results/problem3/**`, `results/problem4/**`, `paper/contents/sections/06_problem2.tex`, `paper/contents/sections/07_problem3.tex`, `paper/contents/sections/08_problem4.tex`

## 审计结论

Q1 的 LP 松弛在该算例上取得零松弛间隙，且最优解自然满足充放电互斥。这一结论仅针对附件1给定的确定性单日算例，不外推至 Q2、Q3 或 Q4。
