# Experiment Log

## 2026-09-11 — Stage 2A causal Q2 forecasting

### Experiment

January rolling validation followed by fixed-rule February--December walk-forward forecasting for load and renewable generation.

### Input

`data/processed/C题/actual_10min.csv` from the Stage 1 canonical pipeline. No raw workbook or official forecast is read.

### Protocol and parameters

- January 2025: chronological calibration and model selection only.
- February 1--December 31, 2025: 334 formal forecast origins, 144 ten-minute targets per origin.
- Load candidates: Yesterday, Last Week and phase-aware Ridge.
- Generation candidates: Yesterday, Last Week, seven-day same-slot mean, fixed-alpha Seasonal, polynomial-Vandermonde Hankel low-rank ranks 1/3/5, and convex Seasonal/Low-Rank fusion.
- Selected calibration values: Ridge alpha 0.01, Seasonal alpha 0.55, Hankel rank 5, fusion weights 1.00/0.00.
- No model coefficient, rank, alpha or fusion-weight update is performed in the formal evaluation period.

### Result

- Selected load forecaster: Last Week. Formal MAE/RMSE/Bias = 176.796/244.287/1.949 kW.
- Selected generation forecaster: seven-day same-slot mean. Formal full-series MAE/RMSE/Bias = 153.400/303.858/2.769 kW; active-generation MAE/RMSE/Bias = 274.384/406.450/4.863 kW.
- The pure polynomial-Vandermonde Hankel expert performed poorly. January selected rank 5 among ranks 1/3/5, but its formal RMSE was 2968.762 kW.
- Convex fusion selected a zero Low-Rank weight and exactly reduced to the Seasonal expert; no independent low-rank gain was observed.

### Observation

The strong Hankel singular-energy concentration found in Stage 1B does not by itself provide a useful polynomial analytical extrapolator after the earlier research-specific residual learner is removed. The seven-day same-slot mean is the strongest January-selected generation forecast and also remains best among the evaluated candidates in the formal period.

### Decision

Use Last Week for Q2 load, the seven-day same-slot mean for Q2 generation, and paired complete-day load/generation residual resampling for uncertainty scenarios. Retain the low-rank results as a negative but informative ablation; do not tune the evaluation rule to make it win.

## 2026-09-11 — Stage 2A+ 标准 SSA 结构预测检查点

### Experiment

采用独立的标准 SSA recurrent forecasting，检验 Stage 1B 的 Hankel 低秩能量集中能否转化为稳定的 144 步新能源发电预测能力。该实现不含 polynomial/modal Vandermonde、神经残差、门控、缩放或在线更新。

### Protocol and parameters

- 仅以 2025 年 1 月 15--31 日的 17 个共同滚动起点选择参数，确保 3、7、14 日窗口使用完全相同的验证日。
- 候选历史窗为 3、7、14 日，候选秩为 1、3、5；嵌入维数固定为 144。
- 各预测日只使用该日前已经观测的完整历史；2 月 1 日至 12 月 31 日参数固定。
- 每个历史窗只作均值中心化，不作标准化；递推完成后恢复历史均值，并仅施加新能源功率非负约束。

### Result

- 一月选择的最优 SSA 配置为 14 日历史窗、秩 5，验证 RMSE 为 327.180 kW；同一验证样本上近 7 日同刻均值为 203.871 kW。
- 固定配置在 2--12 月的全时段 MAE/RMSE/Bias 为 260.834/393.833/76.451 kW，CV(RMSE) 为 0.1656；有效发电时段为 367.660/484.846/37.746 kW，CV(RMSE) 为 0.1139。
- 近 7 日同刻均值同期全时段 MAE/RMSE/Bias 为 153.400/303.858/2.769 kW，CV(RMSE) 为 0.1277；有效发电时段 RMSE 为 406.450 kW。
- SSA 的每日 RMSE 胜率为 11.38%；每日 RMSE 差（SSA 减近 7 日均值）的均值与中位数分别为 99.585 和 98.325 kW。基于 334 个预测日的双侧 Wilcoxon 配对检验为 W=3931，p=3.3453e-42。
- 153 个一月候选日配置与 3006 个正式期候选日配置均无数值失效；最小递推分母为 0.9441，最大递推系数范数为 0.2433。

### Observation

随秩由 1 增至 5，累计奇异值能量和预测精度同步提高，说明保留主要谱分量对 SSA 内部配置有解释力；但秩 5 约 98% 的能量保留仍未形成相对近 7 日同刻均值的预测优势。标准 SSA 明显避免了 polynomial-Vandermonde 外推的严重失配，却仍不足以替代简单周期基线。

### Decision

保持新能源最终预测器为近 7 日同刻均值，不重生成既有残差池或场景。结论限定为：低秩谱集中反映可压缩结构，但不自动转化为更优的多步预测能力；不得据此扩展为“低秩预测没有价值”。

## 2026-09-11 — Stage 2A++ 最终预测检查点

### Experiment

在进入调度优化前，对九类候选方法执行统一的严格因果比较。公开 DLinear 的结构依据为 cure-lab/LTSF-Linear 官方实现：移动平均分解后，季节项与趋势项分别经过线性投影并相加；本项目仅作最小、独立的单变量实现。结构—残差方法由通用秩一 Hankel—二次多项式解析教师和一个 `Linear--GELU--Linear` 小型残差网络构成。

### Protocol and parameters

- 所有方法共用 2025 年 1 月 18--31 日的 14 个滚动预测起点。每个起点只使用目标日以前已经结束的样本；缩放器最晚数据时刻和训练目标最晚时刻均严格早于目标首时刻。
- DLinear 网格为 3/7/14 日输入窗与 25/49/73 移动平均核，分别审查原始 kW 和因果 z-score；随机种子固定为 2026，损失为 MSE，优化器为 AdamW，并使用按时间排序的内部早停。
- 解析教师固定为局部均值中心化、Hankel 分解、秩一重构、对角平均和二次多项式时间 Vandermonde 外推。几何审查覆盖 3/7/14 日输入窗，以及 144 行和 $\lfloor N/2\rfloor$ 行 Hankel 矩阵。
- 残差网络仅学习实测序列相对解析教师的残差；残差缩放统计量同样只来自合法历史。相位—结构融合权重按 0.05 间隔仅由一月决定。
- 2 月 1 日以后，DLinear、残差网络、缩放器、结构几何和融合权重全部冻结；新实测值只进入输入窗，不进行重训练、缩放器更新或梯度计算。

### Result

- 一月统一样本上，近 7 日同刻均值的 MAE/RMSE 为 101.382/213.551 kW，在九类候选中最优，因此继续作为正式新能源预测器。
- DLinear 的一月最优配置为 7 日输入窗、73 点移动平均核和因果 z-score，参数量 290592，MAE/RMSE 为 151.795/261.481 kW；同配置原始 kW 的 RMSE 为 917.367 kW，因果标准化没有造成实质性损害。
- 结构—残差方法由一月选定 7 日输入窗、144 行 Hankel 矩阵、秩 1 和 16 个隐藏单元，参数量 18592；一月 RMSE 为 1397.551 kW。相位融合的最优相位权重为 1.00，因而严格退化为相位方法，未获得独立结构增益。
- 冻结评估期中，近 7 日同刻均值的全时段 MAE/RMSE/Bias 为 153.400/303.858/2.769 kW，有效发电时段 RMSE 为 406.450 kW。DLinear、结构—残差方法和相位融合的全时段 RMSE 分别为 692.692、1993.198 和 341.373 kW。
- 以预测日为配对单位，DLinear、结构—残差方法、相位融合相对近 7 日均值的日 RMSE 胜率分别为 0.60%、0.00% 和 26.05%；三者差值中位数分别为 345.468、1699.390 和 30.019 kW。
- 事后仅作报告的冻结期最优方法仍为近 7 日同刻均值，与一月选定结果一致；事后结果未参与参数或模型选择。

### Integrity and decision

原始四个 Excel 文件的 SHA-256 均未变化；既有 Q2 点预测、场景归档、场景清单、指标、决策和泄露审计文件的运行前后哈希完全一致。由于一月选定方法没有变化，未重生成残差池或场景。

保持负荷预测器为 Last Week，新能源预测器为近 7 日同刻均值，准备进入 Stage 2B。这里的 `StructuralResidualHybrid` 是为竞赛审计建立的通用、固定、匿名混合基线，不复现、也不声称复现任何未公开研究架构。
