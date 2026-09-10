"""Reproduce Stage 1B structural diagnostics, tables, figures and findings."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.data_diagnostic_plots import (
    plot_hankel_spectrum,
    plot_lowrank_forecastability,
    plot_official_lead_error,
    plot_pv_acf,
    plot_pv_seasonal_profile,
    plot_typical_day,
)
from src.common.data_diagnostics import (
    DiagnosticsConfig,
    add_calendar_fields,
    anomaly_candidates,
    correlation_matrices,
    descriptive_statistics,
    hankel_diagnostics,
    lag_correlations,
    lowrank_forecastability,
    official_forecast_diagnostics,
    pv_acf_curve,
    seasonal_naive_metrics,
)
from src.common.data_utils import save_table
from src.common.paths import DOCS_DIR, PAPER_FIGURES_DIR, PROCESSED_DATA_DIR, RESULTS_DIR


INPUT_DIR = PROCESSED_DATA_DIR / "C题"
TABLE_DIR = RESULTS_DIR / "tables" / "data_diagnostics"
FIGURE_DIR = RESULTS_DIR / "figures" / "data_diagnostics"
PAPER_FIGURE_DIR = PAPER_FIGURES_DIR / "common"


def _value(frame: pd.DataFrame, rank: int, column: str) -> float:
    return float(frame.loc[frame["rank"] == rank, column].iloc[0])


def _baseline_row(
    metrics: pd.DataFrame,
    baseline: str,
    subset: str,
    variable: str = "PV",
) -> pd.Series:
    return metrics.loc[
        (metrics["variable"] == variable)
        & (metrics["baseline"] == baseline)
        & (metrics["subset"] == subset)
    ].iloc[0]


def _lag_value(lags: pd.DataFrame, subset: str, lag: int) -> float:
    return float(
        lags.loc[
            (lags["variable"] == "PV")
            & (lags["subset"] == subset)
            & (lags["lag_steps"] == lag),
            "pearson_correlation",
        ].iloc[0]
    )


def _corr_value(
    correlations: pd.DataFrame,
    energy: str,
    error: str,
    method: str,
) -> float:
    return float(
        correlations.loc[
            (correlations["energy_metric"] == energy)
            & (correlations["forecastability_metric"] == error)
            & (correlations["correlation_method"] == method),
            "correlation",
        ].iloc[0]
    )


def _write_findings(
    *,
    representative_date: str,
    lag_table: pd.DataFrame,
    baselines: pd.DataFrame,
    official_overall: pd.DataFrame,
    official_lead: pd.DataFrame,
    singular_energy: pd.DataFrame,
    lowrank_correlations: pd.DataFrame,
    anomalies: pd.DataFrame,
) -> None:
    yesterday = _baseline_row(baselines, "Yesterday", "full_series")
    last_week = _baseline_row(baselines, "Last Week", "full_series")
    yesterday_day = _baseline_row(baselines, "Yesterday", "daylight_only")
    last_week_day = _baseline_row(baselines, "Last Week", "daylight_only")
    overall = official_overall.iloc[0]
    top1 = _value(singular_energy, 1, "cumulative_energy_median")
    top3 = _value(singular_energy, 3, "cumulative_energy_median")
    top5 = _value(singular_energy, 5, "cumulative_energy_median")
    pearson = _corr_value(
        lowrank_correlations,
        "top3_energy",
        "negative_yesterday_rmse",
        "pearson",
    )
    spearman = _corr_value(
        lowrank_correlations,
        "top3_energy",
        "negative_yesterday_rmse",
        "spearman",
    )
    mae_increasing = bool(np.all(np.diff(official_lead["mae"].to_numpy(dtype=float)) >= 0))
    anomaly_counts = anomalies.groupby("variable").size().to_dict()

    content = f"""# C 题数据结构诊断结论

## A. Observed Findings

- 以日光伏电量最接近全年日中位数的日期作为代表日，得到代表日 `{representative_date}`。该日期仅用于展示典型日内结构，不参与参数拟合或样本筛选。
- PV 的 1 天与 7 天 Pearson 时滞相关系数分别为 `{_lag_value(lag_table, 'full_series', 144):.4f}` 和 `{_lag_value(lag_table, 'full_series', 1008):.4f}`。仅保留当前时刻与滞后时刻均有光伏出力的样本后，对应相关系数为 `{_lag_value(lag_table, 'daylight_only', 144):.4f}` 和 `{_lag_value(lag_table, 'daylight_only', 1008):.4f}`。
- PV Yesterday 基线的全序列 MAE、RMSE 与 Bias 分别为 `{yesterday.mae:.3f}`、`{yesterday.rmse:.3f}` 和 `{yesterday.bias:.3f}` kW；Last Week 基线分别为 `{last_week.mae:.3f}`、`{last_week.rmse:.3f}` 和 `{last_week.bias:.3f}` kW。日照时段 Yesterday 的 MAE/RMSE/Bias 为 `{yesterday_day.mae:.3f}`、`{yesterday_day.rmse:.3f}`、`{yesterday_day.bias:.3f}` kW，Last Week 为 `{last_week_day.mae:.3f}`、`{last_week_day.rmse:.3f}`、`{last_week_day.bias:.3f}` kW。
- 官方光伏预报在 `{int(overall['count'])}` 个有效对齐样本上的 MAE、RMSE 与 Bias 分别为 `{overall.mae:.3f}`、`{overall.rmse:.3f}` 和 `{overall.bias:.3f}` kW。五个提前量区间的 MAE 为 `{', '.join(f'{value:.3f}' for value in official_lead['mae'])}` kW；按区间顺序判断，MAE{'呈单调非减变化' if mae_increasing else '并非严格单调增加'}。
- 对固定 7 天历史窗构造 Hankel 矩阵并去除窗口均值后，前 1、3、5 阶累计能量中位数分别为 `{100 * top1:.2f}%`、`{100 * top3:.2f}%` 和 `{100 * top5:.2f}%`。
- 前三阶累计能量与负 Yesterday 次日 RMSE 的 Pearson、Spearman 相关系数分别为 `{pearson:.4f}` 和 `{spearman:.4f}`。该结果使用全部固定日步长窗口，未按相关方向或显著性删选窗口。
- 异常筛查仅形成候选标记：PV `{int(anomaly_counts.get('PV', 0))}` 条、Load `{int(anomaly_counts.get('Load', 0))}` 条、Price `{int(anomaly_counts.get('Price', 0))}` 条。所有观测均保留在规范数据中。

## B. Modeling Implications

- 日、周时滞相关与因果 Seasonal Naive 误差共同给出相位基线的直接依据，因此 Stage 2 应保留 Yesterday/Last Week 或其相位组合，作为所有复杂模型必须超过的基准。
- Hankel 能量集中说明 PV 历史窗可由少量主导谱分量近似表达，因此值得测试 Hankel-based Low-Rank Expert；该结论只支持“值得测试”，不等同于证明其预测性能最优。
- 官方预报应继续作为独立专家输入。其不同提前量误差存在差异，Stage 2 的融合权重不宜默认与 horizon 无关。
- Low-rank energy 与基线误差的相关性只描述数据结构与可预测性之间的统计联系。无论相关性强弱，模型选择仍须通过严格的时序外推比较确认。

## C. Not Yet Established

- 尚未证明 Lyra 或任何 Hankel-based 模型最优。
- 尚未证明大型深度模型是必要的。
- 尚未证明 Official、Phase 与 Low-Rank 的融合形式或权重最优。
- 尚未完成概率预测、区间校准或场景生成。
- 尚未完成问题二至问题四的决策优化、滚动调度或经济性评价。
"""
    (DOCS_DIR / "data_diagnostics_findings.md").write_text(content, encoding="utf-8")


def _write_figure_qa(config: DiagnosticsConfig, representative_date: str) -> None:
    content = f"""# Stage 1B figure QA

- Backend: Python / matplotlib only.
- Archetype: quantitative evidence figures.
- Source data: canonical processed CSV files and generated diagnostic tables.
- Exclusions: none. Daylight-only panels use the declared physical predicate `PV > {config.daylight_threshold_kw}` at both members of each lag pair.
- Representative day: {representative_date}, selected as the day whose PV energy is closest to the annual daily median.
- Hankel contract: history={config.history_steps}, rows={config.hankel_rows}, spectrum stride={config.spectrum_stride}, forecastability stride={config.forecastability_stride}.
- Forecastability target: next-day Yesterday baseline error, with no future data in the predictor.
- Exports: SVG and PDF with editable text; JPG at 450 dpi.
- Typography: Songti SC first; minus-sign rendering enabled.
"""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    (FIGURE_DIR / "figure_qa.md").write_text(content, encoding="utf-8")


def main() -> None:
    config = DiagnosticsConfig()
    actual_path = INPUT_DIR / "actual_10min.csv"
    alignment_path = INPUT_DIR / "forecast_actual_alignment.csv"
    if not actual_path.exists() or not alignment_path.exists():
        raise FileNotFoundError("Run python src/run_preprocess.py before Stage 1B")

    actual = pd.read_csv(
        actual_path,
        parse_dates=["operating_date", "interval_start", "interval_end"],
    )
    alignment = pd.read_csv(
        alignment_path,
        parse_dates=["release_time", "target_time"],
    )
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    descriptive = descriptive_statistics(actual)
    pearson, spearman = correlation_matrices(actual)
    lag_table = lag_correlations(actual, config)
    baselines = seasonal_naive_metrics(actual, config)
    official_overall, official_lead, official_issue, official_month = (
        official_forecast_diagnostics(alignment)
    )
    singular = hankel_diagnostics(actual, config)
    singular_energy = singular.summary.assign(
        lookback=config.history_steps,
        hankel_rows=config.hankel_rows,
        stride=config.spectrum_stride,
    )
    lowrank_windows, lowrank_correlations = lowrank_forecastability(actual, config)
    anomalies = anomaly_candidates(actual, config)
    acf = pv_acf_curve(actual, config)

    save_table(descriptive, TABLE_DIR / "descriptive_statistics.csv", float_format="%.10g")
    save_table(pearson, TABLE_DIR / "correlation_pearson.csv", float_format="%.10g")
    save_table(spearman, TABLE_DIR / "correlation_spearman.csv", float_format="%.10g")
    save_table(lag_table, TABLE_DIR / "lag_correlations.csv", float_format="%.10g")
    save_table(baselines, TABLE_DIR / "seasonal_naive_metrics.csv", float_format="%.10g")
    save_table(official_overall, TABLE_DIR / "official_forecast_overall.csv", float_format="%.10g")
    save_table(official_lead, TABLE_DIR / "official_forecast_by_lead.csv", float_format="%.10g")
    save_table(official_issue, TABLE_DIR / "official_forecast_by_issue_hour.csv", float_format="%.10g")
    save_table(official_month, TABLE_DIR / "official_forecast_by_month.csv", float_format="%.10g")
    save_table(singular_energy, TABLE_DIR / "hankel_singular_energy.csv", float_format="%.10g")
    save_table(lowrank_windows, TABLE_DIR / "lowrank_forecastability.csv", float_format="%.10g")
    save_table(
        lowrank_correlations,
        TABLE_DIR / "lowrank_forecastability_correlations.csv",
        float_format="%.10g",
    )
    save_table(anomalies, TABLE_DIR / "anomaly_candidates.csv", float_format="%.10g")

    calendar = add_calendar_fields(actual)
    daily_energy = calendar.groupby("operating_date")["pv_actual_kwh"].sum()
    median_energy = float(daily_energy.median())
    representative_date = str((daily_energy - median_energy).abs().idxmin().date())
    typical_day = calendar.loc[
        calendar["operating_date"] == pd.Timestamp(representative_date)
    ]

    plot_typical_day(typical_day, date_label=representative_date, output_dir=FIGURE_DIR)
    plot_pv_seasonal_profile(calendar, output_dir=FIGURE_DIR)
    plot_pv_acf(acf, output_dir=FIGURE_DIR)
    plot_hankel_spectrum(
        singular,
        maximum_rank=config.maximum_spectrum_rank,
        output_dir=FIGURE_DIR,
    )
    plot_lowrank_forecastability(lowrank_windows, lowrank_correlations, output_dir=FIGURE_DIR)
    plot_official_lead_error(official_lead, output_dir=FIGURE_DIR)

    for figure_name in [
        "fig_data_typical_day",
        "fig_pv_seasonal_profile",
        "fig_pv_acf",
        "fig_hankel_spectrum",
        "fig_lowrank_forecastability",
        "fig_official_lead_error",
    ]:
        shutil.copy2(FIGURE_DIR / f"{figure_name}.pdf", PAPER_FIGURE_DIR / f"{figure_name}.pdf")

    _write_findings(
        representative_date=representative_date,
        lag_table=lag_table,
        baselines=baselines,
        official_overall=official_overall,
        official_lead=official_lead,
        singular_energy=singular_energy,
        lowrank_correlations=lowrank_correlations,
        anomalies=anomalies,
    )
    _write_figure_qa(config, representative_date)

    summary = {
        "configuration": config.__dict__,
        "representative_date": representative_date,
        "hankel_audit": singular.audit,
        "hankel_energy_median": {
            "top1": _value(singular_energy, 1, "cumulative_energy_median"),
            "top3": _value(singular_energy, 3, "cumulative_energy_median"),
            "top5": _value(singular_energy, 5, "cumulative_energy_median"),
        },
        "official_overall": official_overall.iloc[0].to_dict(),
        "lowrank_forecastability_correlations": lowrank_correlations.to_dict(orient="records"),
        "anomaly_candidate_count": anomalies.groupby("variable").size().to_dict(),
    }
    (TABLE_DIR / "diagnostics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Stage 1B diagnostics complete.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
