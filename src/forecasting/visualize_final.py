"""Chinese formal figures for the final forecasting checkpoint."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.plotting import CUMCM_PALETTE, configure_plots, style_academic_axes


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Songti SC", "SimSun", "STSong", "Arial", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})


LABELS = {
    "Yesterday": "昨日同刻", "Last Week": "上周同刻", "7-day same-slot mean": "近7日同刻均值",
    "Seasonal/Phase": "相位组合", "SSA best config": "SSA结构模型",
    "Polynomial analytical-only teacher": "解析结构教师", "DLinear": "DLinear",
    "StructuralResidualHybrid": "结构—残差混合", "PhaseStructuralFusion": "相位—结构融合",
}


def _save(fig: plt.Figure, output_dir: Path, name: str) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    common = {"bbox_inches": "tight", "facecolor": "white", "transparent": False}
    fig.savefig(output_dir / f"{name}.svg", **common)
    fig.savefig(output_dir / f"{name}.pdf", **common)
    fig.savefig(output_dir / f"{name}.jpg", dpi=600, **common)
    fig.savefig(output_dir / f"{name}.png", dpi=600, **common)
    return {suffix: output_dir / f"{name}.{suffix}" for suffix in ("svg", "pdf", "jpg", "png")}


def plot_final_comparison(table: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    """Place January selection evidence beside untouched OOS evidence."""
    configure_plots()
    ordered = table.sort_values("validation_rmse_kw", kind="stable")
    y = np.arange(len(ordered))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.5), sharey=True, facecolor="white")
    for axis, column, title in [
        (axes[0], "validation_rmse_kw", "a  一月因果验证"),
        (axes[1], "oos_rmse_kw", "b  二月至十二月冻结评估"),
    ]:
        colors = [CUMCM_PALETTE["primary"] if selected else CUMCM_PALETTE["neutral"] for selected in ordered["selected_by_january"]]
        axis.barh(y, ordered[column], color=colors, height=0.65)
        axis.set_xlabel("均方根误差（kW）")
        axis.set_title(title, fontsize=10)
        style_academic_axes(axis, grid_axis="x")
    axes[0].set_yticks(y, [LABELS[name] for name in ordered["model"]])
    axes[0].invert_yaxis()
    fig.text(0.01, 0.005, "深蓝色仅表示由一月验证选定的正式模型；右图不参与模型选择。", fontsize=7.5, color=CUMCM_PALETTE["observed"])
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    saved = _save(fig, output_dir, "fig_p2_final_forecast_comparison")
    plt.close(fig)
    return saved


def plot_typical_day(
    dates: pd.DatetimeIndex,
    actual: np.ndarray,
    predictions: dict[str, np.ndarray],
    output_dir: Path,
) -> tuple[dict[str, Path], str]:
    """Show the median-energy day using only predeclared representative logic."""
    energy = actual.sum(axis=1)
    position = int(np.argmin(np.abs(energy - np.median(energy))))
    hours = np.arange(actual.shape[1]) / 6
    fig, axis = plt.subplots(figsize=(7.2, 3.6), facecolor="white")
    configure_plots()
    axis.plot(hours, actual[position], color=CUMCM_PALETTE["observed"], linewidth=2, label="实际发电")
    specs = [
        ("7-day same-slot mean", CUMCM_PALETTE["primary"], "--"),
        ("DLinear", CUMCM_PALETTE["price"], "-"),
        ("StructuralResidualHybrid", CUMCM_PALETTE["pv"], "-"),
        ("PhaseStructuralFusion", CUMCM_PALETTE["soc"], "-"),
    ]
    for model, color, linestyle in specs:
        axis.plot(hours, predictions[model][position], color=color, linestyle=linestyle, linewidth=1.4, label=LABELS[model])
    axis.set(xlabel="运营日内时刻", ylabel="新能源发电功率（kW）", title=f"典型日预测轨迹（{dates[position].date()}）")
    axis.set_xticks(np.arange(0, 25, 4), [f"{hour:02d}:00" for hour in range(0, 25, 4)])
    axis.legend(ncol=3, fontsize=7.5, loc="upper right")
    style_academic_axes(axis)
    fig.tight_layout()
    saved = _save(fig, output_dir, "fig_p2_final_typical_day")
    plt.close(fig)
    return saved, dates[position].date().isoformat()


def plot_daily_differences(daily: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    """Show same-day RMSE differences against the seven-day baseline."""
    configure_plots()
    fig, axis = plt.subplots(figsize=(7.2, 3.6), facecolor="white")
    colors = [CUMCM_PALETTE["price"], CUMCM_PALETTE["pv"], CUMCM_PALETTE["soc"]]
    models = ["DLinear", "StructuralResidualHybrid", "PhaseStructuralFusion"]
    data = [daily.loc[daily.model == model, "rmse_difference_model_minus_7day_kw"].to_numpy() for model in models]
    parts = axis.violinplot(data, showmeans=False, showmedians=True, widths=0.75)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color); body.set_edgecolor(color); body.set_alpha(0.45)
    for key in ("cmedians", "cbars", "cmins", "cmaxes"):
        parts[key].set_color(CUMCM_PALETTE["observed"]); parts[key].set_linewidth(0.9)
    axis.axhline(0, color=CUMCM_PALETTE["observed"], linestyle="--", linewidth=1)
    axis.set_xticks(range(1, 4), [LABELS[m] for m in models])
    axis.set(ylabel="相对近7日均值的日RMSE差（kW）", title="冻结评估期的每日成对误差差值")
    style_academic_axes(axis)
    fig.text(0.01, 0.005, "差值小于0表示对应模型在同一预测日优于近7日同刻均值。", fontsize=7.5, color=CUMCM_PALETTE["observed"])
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    saved = _save(fig, output_dir, "fig_p2_final_daily_differences")
    plt.close(fig)
    return saved
