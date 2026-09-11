"""Chinese scientific figures for the Q4 price/information layer."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

from src.common.paths import problem_figures_dir
from src.common.plotting import CUMCM_PALETTE, configure_plots, style_academic_axes


FIGURE_DIR = problem_figures_dir(4)


def _save_q4_figure(figure, name: str) -> dict[str, Path]:
    """Export editable vectors and 600-dpi raster files."""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(
        {
            "font.sans-serif": ["Songti SC", "SimSun", "STSong", "Arial Unicode MS"],
            "font.size": 10,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )
    paths = {
        "svg": FIGURE_DIR / f"{name}.svg",
        "pdf": FIGURE_DIR / f"{name}.pdf",
        "jpg": FIGURE_DIR / f"{name}.jpg",
        "png": FIGURE_DIR / f"{name}.png",
        "tiff": FIGURE_DIR / f"{name}.tiff",
    }
    common = {"bbox_inches": "tight", "facecolor": "white", "transparent": False}
    figure.savefig(paths["svg"], **common)
    figure.savefig(paths["pdf"], **common)
    figure.savefig(paths["jpg"], dpi=600, **common)
    figure.savefig(paths["png"], dpi=600, **common)
    figure.savefig(paths["tiff"], dpi=600, pil_kwargs={"compression": "tiff_lzw"}, **common)
    return paths


def plot_annual_monthly_price(frame: pd.DataFrame) -> dict[str, Path]:
    """Annual daily envelope and monthly boxplots."""

    daily = frame.groupby("operating_date", sort=True)["price_yuan_per_kwh"].agg(
        ["min", "mean", "max"]
    )
    monthly = [
        frame.loc[frame["month"].eq(month), "price_yuan_per_kwh"].to_numpy(float)
        for month in range(1, 13)
    ]
    figure, axes = plt.subplots(2, 1, figsize=(7.2, 7.0), constrained_layout=True)
    axis = axes[0]
    axis.fill_between(
        daily.index,
        daily["min"],
        daily["max"],
        color=CUMCM_PALETTE["price"],
        alpha=0.18,
        label="日内最小—最大范围",
    )
    axis.plot(daily.index, daily["mean"], color=CUMCM_PALETTE["primary"], linewidth=1.15, label="日均电价")
    axis.set_ylabel("电价（元/kWh）")
    axis.set_xlabel("日期")
    axis.set_ylim(0, float(daily["max"].max()) * 1.16)
    axis.legend(frameon=False, ncol=2, loc="upper right")
    axis.text(0.01, 0.95, "a", transform=axis.transAxes, va="top", fontweight="bold")
    style_academic_axes(axis)

    axis = axes[1]
    boxes = axis.boxplot(
        monthly,
        positions=np.arange(1, 13),
        widths=0.62,
        patch_artist=True,
        showfliers=True,
        flierprops={"marker": ".", "markersize": 1.2, "alpha": 0.18, "markeredgecolor": CUMCM_PALETTE["neutral"]},
        medianprops={"color": CUMCM_PALETTE["observed"], "linewidth": 1.2},
        whiskerprops={"color": CUMCM_PALETTE["neutral"]},
        capprops={"color": CUMCM_PALETTE["neutral"]},
    )
    for box in boxes["boxes"]:
        box.set_facecolor(CUMCM_PALETTE["price"])
        box.set_alpha(0.48)
        box.set_edgecolor(CUMCM_PALETTE["price"])
    axis.set_xticks(range(1, 13), [f"{month}月" for month in range(1, 13)])
    axis.set_ylabel("电价（元/kWh）")
    axis.set_xlabel("月份")
    axis.text(0.01, 0.95, "b", transform=axis.transAxes, va="top", fontweight="bold")
    style_academic_axes(axis)
    return _save_q4_figure(figure, "fig_p4_price_annual_monthly")


def plot_intraday_distribution(frame: pd.DataFrame, profile: pd.DataFrame) -> dict[str, Path]:
    """Intraday mean/quantiles and full-sample histogram."""

    hours = (profile["slot"].to_numpy(float) - 0.5) / 6
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 4.0), constrained_layout=True)
    axis = axes[0]
    axis.fill_between(
        hours,
        profile["q1_yuan_per_kwh"],
        profile["q3_yuan_per_kwh"],
        color=CUMCM_PALETTE["price"],
        alpha=0.22,
        label="四分位区间",
    )
    axis.plot(hours, profile["mean_yuan_per_kwh"], color=CUMCM_PALETTE["primary"], linewidth=1.6, label="时段均值")
    axis.set_xlim(0, 24)
    axis.set_xticks([0, 4, 8, 12, 16, 20, 24])
    axis.set_xlabel("日内时刻")
    axis.set_ylabel("电价（元/kWh）")
    axis.legend(frameon=False)
    axis.text(0.02, 0.96, "a", transform=axis.transAxes, va="top", fontweight="bold")
    style_academic_axes(axis)

    axis = axes[1]
    axis.hist(
        frame["price_yuan_per_kwh"].to_numpy(float),
        bins=50,
        color=CUMCM_PALETTE["price"],
        alpha=0.78,
        edgecolor="white",
        linewidth=0.35,
    )
    axis.axvline(frame["price_yuan_per_kwh"].median(), color=CUMCM_PALETTE["primary"], linewidth=1.4, linestyle="--", label="中位数")
    axis.set_xlabel("电价（元/kWh）")
    axis.set_ylabel("区间数")
    axis.legend(frameon=False)
    axis.text(0.02, 0.96, "b", transform=axis.transAxes, va="top", fontweight="bold")
    style_academic_axes(axis)
    return _save_q4_figure(figure, "fig_p4_price_intraday_distribution")


def plot_price_system_relation(frame: pd.DataFrame, correlations: pd.DataFrame) -> dict[str, Path]:
    """Correlation matrix and full-sample price/net-load density."""

    matrix = correlations.pivot(index="method", columns="variable", values="coefficient").loc[
        ["Pearson", "Spearman"], ["实际负荷", "实际光伏", "净负荷"]
    ]
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 4.0), constrained_layout=True)
    axis = axes[0]
    image = axis.imshow(matrix.to_numpy(float), vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    axis.set_xticks(np.arange(3), matrix.columns)
    axis.set_yticks(np.arange(2), matrix.index)
    for row in range(2):
        for column in range(3):
            value = float(matrix.iloc[row, column])
            axis.text(column, row, f"{value:.3f}", ha="center", va="center", color="white" if abs(value) > 0.45 else CUMCM_PALETTE["observed"])
    axis.set_xlabel("系统状态变量")
    axis.set_ylabel("相关系数")
    axis.text(-0.14, 1.04, "a", transform=axis.transAxes, va="top", fontweight="bold")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="相关系数")

    axis = axes[1]
    density_cmap = LinearSegmentedColormap.from_list(
        "cumcm_density",
        ["#F7F8FA", CUMCM_PALETTE["primary_light"], CUMCM_PALETTE["primary"]],
    )
    density = axis.hexbin(
        frame["net_load_kw"],
        frame["price_yuan_per_kwh"],
        gridsize=55,
        mincnt=1,
        cmap=density_cmap,
        linewidths=0,
    )
    axis.set_xlabel("净负荷（kW）")
    axis.set_ylabel("电价（元/kWh）")
    axis.text(0.02, 0.96, "b", transform=axis.transAxes, va="top", fontweight="bold")
    style_academic_axes(axis, grid_axis=None)
    figure.colorbar(density, ax=axis, fraction=0.046, pad=0.04, label="区间数量")
    return _save_q4_figure(figure, "fig_p4_price_system_relation")


def plot_price_forecast_comparison(comparison: pd.DataFrame) -> dict[str, Path]:
    """Validation and evaluation RMSE for lightweight causal candidates."""

    methods = comparison.loc[comparison["period"].eq("validation"), "method"].tolist()
    validation = comparison.loc[comparison["period"].eq("validation")].set_index("method").loc[methods]
    evaluation = comparison.loc[comparison["period"].eq("evaluation")].set_index("method").loc[methods]
    x = np.arange(len(methods), dtype=float)
    width = 0.35
    figure, axis = plt.subplots(figsize=(7.2, 4.3))
    axis.bar(x - width / 2, validation["rmse_yuan_per_kwh"], width, color=CUMCM_PALETTE["primary_light"], label="1月验证集")
    axis.bar(x + width / 2, evaluation["rmse_yuan_per_kwh"], width, color=CUMCM_PALETTE["price"], label="2—12月样本外")
    axis.set_xticks(x, methods)
    axis.set_ylabel("RMSE（元/kWh）")
    axis.set_xlabel("价格预测方法")
    axis.legend(frameon=False, ncol=2)
    style_academic_axes(axis)
    figure.tight_layout()
    return _save_q4_figure(figure, "fig_p4_price_forecast_comparison")


def generate_price_figures(
    frame: pd.DataFrame,
    profile: pd.DataFrame,
    correlations: pd.DataFrame,
    comparison: pd.DataFrame,
) -> list[Path]:
    configure_plots()
    bundles = [
        plot_annual_monthly_price(frame),
        plot_intraday_distribution(frame, profile),
        plot_price_system_relation(frame, correlations),
        plot_price_forecast_comparison(comparison),
    ]
    return [path for bundle in bundles for path in bundle.values()]


__all__ = [
    "generate_price_figures",
    "plot_annual_monthly_price",
    "plot_intraday_distribution",
    "plot_price_forecast_comparison",
    "plot_price_system_relation",
]
