# Q4 PRICE / INFORMATION CHECKPOINT

本检查点只完成动态电价的数据、预测和信息接口，不运行任何调度。

## 可复现入口

```bash
python src/problem4/run.py
```

该入口生成规范价格表、审计 JSON、统计与预测 CSV、LaTeX 表格以及价格类图形。正式价格接口为：

- `src/problem4/price_interface.py::get_q4_price_inputs`
- `src/problem4/price_interface.py::get_q4_2_price_inputs`
- `src/problem4/price_interface.py::get_q4_3_price_inputs`

## 已锁定判断

- 题面未来价格可用性：`AMBIGUOUS`。
- 同时支持完美信息与严格因果价格预测模式，调用方必须显式选择。
- 轻量价格模型只按 1 月验证集选择，2—12 月结果不参与选模。
- 价格与系统状态的相关性用于描述关联，不作因果解释。

## 明确未完成

- 未实现 Q4-2 调度。
- 未实现 Q4-3 滚动调度。
- 未修改电池、SOC、应急购电或结算逻辑。
- 未生成任何 `result4*.xlsx`。
- 未生成经济成本、SOC 或购电动作图。
