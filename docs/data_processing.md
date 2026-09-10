# C 题 Stage 1A 规范数据管道

## 1. 数据边界

数据管道只读取以下四个官方工作簿，任何处理均不回写原文件：

- `data/raw/C题/附件/附件1.xlsx`
- `data/raw/C题/附件/附件2.xlsx`
- `data/raw/C题/附件/附件3.xlsx`
- `data/raw/C题/附件/附件4.xlsx`

入口为：

```bash
python src/run_preprocess.py
```

程序在处理前后计算四个文件的 SHA-256，并在
`data/processed/C题/data_quality_report.json` 中记录路径、工作表、原始形状和哈希一致性。

## 2. 时间与单位定义

- 附件 1、2、4 的时刻均表示 10 分钟区间结束时刻。
- `00:10` 对应区间 `[00:00, 00:10]`，`0:00+1` 对应运营日次日 `00:00`。
- `slot` 从 1 到 144；`interval_start = interval_end - 10 min`。
- 功率单位为 kW，10 分钟电量按 `kWh = kW / 6` 转换，`DT_HOURS = 1/6`。
- `net_load_kw = load_kw - pv_actual_kw`，光伏富余时允许为负，不截断。

## 3. 规范数据表

### `problem1_day.csv`

附件 1 的字段改为统一英文名，并增加 `slot`、区间起止、kWh 和净负荷字段。这里的“标准化”
指字段、时段和单位规范化，不进行 Z-score 或尺度缩放。输出固定为 144 行，时段从 `00:10`
到 `24:00`。

### `actual_10min.csv`

附件 2 的负荷、实际光伏与附件 4 的电价分别由宽表转成长表。各表先检查日期和联合键唯一性，
再按 `operating_date + slot + interval_start + interval_end` 做外连接，所有连接均使用
`validate="one_to_one"`。连接保留任一来源的缺失值，不执行静默删除、重复平均或全局前后填充。

联合键确认无重复后，才按 10 分钟完整时间轴重新索引。缺失时间点保留 NaN。当前官方数据输出
52560 行，`interval_end` 从 `2025-01-01 00:10:00` 连续到
`2026-01-01 00:00:00`。

因果历史特征为：

- `pv_lag_1d`：实际光伏滞后 144 个时段；
- `pv_lag_7d`：实际光伏滞后 1008 个时段；
- `pv_same_slot_7d_mean`：同一时段此前完整 7 天的光伏均值；
- `load_lag_1d`：负荷滞后 144 个时段；
- `load_lag_7d`：负荷滞后 1008 个时段。

所有特征均由 `shift` 后的数据构造，当前时刻及未来观测不会进入历史特征。历史不足时保留 NaN。

### `pv_forecast_hourly.csv`

附件 3 的 24 个预报列转换为长表。日期列的空白只按原表版式向下继承到同一运营日的
6:00、12:00 和 18:00 发布行；预测值不填补。每行以
`release_time + horizon_hour` 为唯一键，并按

```text
target_time = release_time + horizon_hour
```

计算目标时刻。当前数据包含 365 天、每天 4 次发布、每次 24 个步长，共 35040 行。

### `forecast_actual_alignment.csv`

每条官方预测按精确 `target_time` 查询实际光伏。由于多个发布时间可以预测同一目标时刻，
`target_time` 在预测表中不是唯一键，不能伪装成 one-to-one merge；程序先验证实际数据
`interval_end` 唯一，再用唯一索引逐条查询，从而避免 many-to-many 或静默扩行。

文件保留全部 35040 条预测，并记录 `alignment_status`、`unmatched_reason` 和绝对误差。超出
实际数据末端的年末预测标为 `out_of_actual_range`，不强行匹配。报告同时在共同有效样本上
比较正确对齐和目标时刻前后各偏移 1 小时的 MAE。

### `pv_forecast_10min.csv`

每个 `release_time` 独立将 24 个小时节点插值为 144 个 10 分钟节点，不跨发布批次。主方法为
PCHIP，节点不足或计算失败时退化为线性插值。发布时刻已有的实际光伏可作为零时刻锚点；程序
只查询与 `release_time` 完全相同的观测，未来实际值不参与插值。第一个发布批次缺少零时刻实际
观测时，不外推早于第一个小时节点的值，对应位置保留 NaN。预测值可截断到 0，实际值不截断；
所有原始小时节点在输出中显式恢复为原值。

## 4. 窗口与掩码

`src/common/windowing.py` 在生成窗口前检查时间戳唯一性，并逐窗口核验相邻时刻是否均为 10 分钟。
输入和目标的 NaN 掩码在任何填充值之前生成，随后只为数组计算将缺失位置填为 0。有效窗口要求：
时间连续、输入完整、评分目标完整。

`masked_mae`、`masked_rmse`、`masked_nmae` 和 `masked_nrmse` 只在目标观测掩码内计算；若有效
目标位置的预测为非有限值则直接报错。NMAE 与 NRMSE 默认以有效目标的平均绝对值归一化，也可
显式传入正的尺度参数。

## 5. 质量报告与验收

`data_quality_report.json` 记录：原始文件路径、SHA-256、工作表与形状；处理后形状和时间范围；
缺失、重复和非 10 分钟间隔数量；每日记录数；预测 raw/aligned/unmatched 数量及原因；正确对齐
和 ±1 小时偏移 MAE；插值节点保持情况；窗口有效/无效数量与评分覆盖率。

验收命令：

```bash
python src/run_preprocess.py
pytest -v
python src/smoke_test.py
git diff --check
git diff -- data/raw/C题/附件
```
