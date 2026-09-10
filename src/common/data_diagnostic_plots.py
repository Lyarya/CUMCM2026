"""Chinese publication figures for Stage 1B data diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.plotting import (
    CUMCM_PALETTE,
    configure_plots,
    save_figure,
    style_academic_axes,
)
from src.common.spectral import SingularSpectrumAnalysis


SEASON_LABELS = {
    "spring": "春季",
    "summer": "夏季",
    "autumn": "秋季",
    "winter": "冬季",
}
SEASON_COLORS = {
    "spring": "#69A87D",
    "summer": "#E39B34",
    "autumn": "#B86B4B",
    "winter": "#5DA8C9",
}


def _time_ticks(ax: plt.Axes) -> None:
    positions = np.arange(0, 144, 24)
    ax.set_xticks(positions, [f"{hour:02d}:00" for hour in range(0, 24, 4)])


def plot_typical_day(
    day: pd.DataFrame,
    *,
    date_label: str,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Show the load, PV, net-load and price structure of a representative day."""
    configure_plots()
    x = np.arange(len(day))
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.2, 5.6),
        sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1.0]},
    )
    axes[0].plot(x, day["load_kw"], color=CUMCM_PALETTE["load"], lw=1.8, label="小区负荷")
    axes[0].plot(x, day["pv_actual_kw"], color=CUMCM_PALETTE["pv"], lw=1.8, label="光伏出力")
    axes[0].plot(
        x,
        day["net_load_kw"],
        color=CUMCM_PALETTE["observed"],
        lw=1.6,
        label="净负荷",
    )
    axes[0].axhline(0, color=CUMCM_PALETTE["neutral"], lw=0.8)
    axes[0].set_ylabel("功率（kW）")
    axes[0].set_title(f"代表日负荷—光伏—净负荷结构（{date_label}）")
    axes[0].legend(ncol=3, frameon=False, loc="upper center")

    axes[1].plot(
        x,
        day["price_yuan_per_kwh"],
        color=CUMCM_PALETTE["price"],
        lw=1.7,
    )
    axes[1].set_ylabel("电价（元/kWh）")
    axes[1].set_xlabel("区间结束时刻")
    _time_ticks(axes[1])
    for ax in axes:
        style_academic_axes(ax)
    fig.tight_layout()
    saved = save_figure(fig, output_dir, "fig_data_typical_day", dpi=450)
    plt.close(fig)
    return saved


def plot_pv_seasonal_profile(
    frame: pd.DataFrame,
    *,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Plot mean intraday PV profiles by season."""
    configure_plots()
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for season in ["spring", "summer", "autumn", "winter"]:
        subset = frame.loc[frame["season"] == season]
        profile = subset.groupby("slot", sort=True)["pv_actual_kw"].mean()
        ax.plot(
            np.arange(len(profile)),
            profile,
            lw=1.9,
            color=SEASON_COLORS[season],
            label=SEASON_LABELS[season],
        )
    ax.set_title("光伏出力的季节性日内轮廓")
    ax.set_ylabel("平均光伏功率（kW）")
    ax.set_xlabel("区间结束时刻")
    _time_ticks(ax)
    ax.legend(ncol=4, frameon=False, loc="upper center")
    style_academic_axes(ax)
    fig.tight_layout()
    saved = save_figure(fig, output_dir, "fig_pv_seasonal_profile", dpi=450)
    plt.close(fig)
    return saved


def plot_pv_acf(acf: pd.DataFrame, *, output_dir: str | Path) -> dict[str, Path]:
    """Compare full-series and daylight-only PV lag correlations."""
    configure_plots()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for subset, label, color in [
        ("full_series", "全序列", CUMCM_PALETTE["primary"]),
        ("daylight_only", "仅日照时段", CUMCM_PALETTE["pv"]),
    ]:
        shown = acf.loc[acf["subset"] == subset]
        ax.plot(
            shown["lag_steps"] / 144,
            shown["pearson_correlation"],
            lw=1.35,
            color=color,
            label=label,
        )
    for day, label in [(1, "1天"), (7, "7天")]:
        ax.axvline(day, color=CUMCM_PALETTE["neutral"], ls="--", lw=0.9)
        ax.text(day, 0.02, label, rotation=90, va="bottom", ha="right", fontsize=8)
    ax.set_xlim(0, 7)
    ax.set_ylim(-0.15, 1.02)
    ax.set_title("光伏序列的时滞相关结构")
    ax.set_xlabel("滞后天数")
    ax.set_ylabel("Pearson相关系数")
    ax.legend(frameon=False, loc="upper right")
    style_academic_axes(ax)
    fig.tight_layout()
    saved = save_figure(fig, output_dir, "fig_pv_acf", dpi=450)
    plt.close(fig)
    return saved


def plot_hankel_spectrum(
    analysis: SingularSpectrumAnalysis,
    *,
    maximum_rank: int,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Plot the PV singular-value decay and cumulative squared energy."""
    configure_plots()
    shown = analysis.summary.loc[analysis.summary["rank"] <= maximum_rank]
    ranks = shown["rank"].to_numpy(dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.7))
    axes[0].fill_between(
        ranks,
        shown["normalized_singular_value_q25"],
        shown["normalized_singular_value_q75"],
        color=CUMCM_PALETTE["neutral_light"],
        alpha=0.8,
    )
    axes[0].plot(
        ranks,
        shown["normalized_singular_value_median"],
        color=CUMCM_PALETTE["primary"],
        lw=1.8,
    )
    axes[0].set_yscale("log")
    axes[0].set_title("奇异值衰减")
    axes[0].set_ylabel(r"归一化奇异值 $\sigma_i/\sigma_1$")

    axes[1].fill_between(
        ranks,
        100 * shown["cumulative_energy_q25"],
        100 * shown["cumulative_energy_q75"],
        color=CUMCM_PALETTE["neutral_light"],
        alpha=0.8,
    )
    axes[1].plot(
        ranks,
        100 * shown["cumulative_energy_median"],
        color=CUMCM_PALETTE["pv"],
        lw=1.8,
    )
    axes[1].set_ylim(0, 101)
    axes[1].set_title("累计谱能量")
    axes[1].set_ylabel("累计能量占比（%）")
    for ax in axes:
        ax.set_xlabel("奇异值序号")
        style_academic_axes(ax)
    fig.suptitle("光伏历史窗口的Hankel奇异谱", y=1.01)
    fig.tight_layout()
    saved = save_figure(fig, output_dir, "fig_hankel_spectrum", dpi=450)
    plt.close(fig)
    return saved


def plot_lowrank_forecastability(
    windows: pd.DataFrame,
    correlations: pd.DataFrame,
    *,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Plot top-3 energy against the causal next-day Yesterday error."""
    configure_plots()
    x = 100 * windows["top3_energy"].to_numpy(dtype=float)
    y = windows["yesterday_rmse"].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    ax.scatter(
        x[valid],
        y[valid],
        s=21,
        alpha=0.65,
        color=CUMCM_PALETTE["primary_light"],
        edgecolor="white",
        linewidth=0.35,
    )
    if valid.sum() >= 2:
        coefficient = np.polyfit(x[valid], y[valid], 1)
        line_x = np.linspace(x[valid].min(), x[valid].max(), 100)
        ax.plot(
            line_x,
            np.polyval(coefficient, line_x),
            color=CUMCM_PALETTE["bad"],
            lw=1.4,
            label="线性趋势",
        )
    selected = correlations.query(
        "energy_metric == 'top3_energy' and "
        "forecastability_metric == 'negative_yesterday_rmse'"
    ).set_index("correlation_method")
    pearson = float(selected.loc["pearson", "correlation"])
    spearman = float(selected.loc["spearman", "correlation"])
    ax.text(
        0.03,
        0.97,
        f"与负RMSE的相关系数\nPearson = {pearson:.3f}\nSpearman = {spearman:.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": CUMCM_PALETTE["neutral_light"], "pad": 4},
    )
    ax.set_title("低秩能量集中度与次日可预测性")
    ax.set_xlabel("前三阶累计Hankel能量（%）")
    ax.set_ylabel("Yesterday基线次日RMSE（kW）")
    if valid.sum() >= 2:
        ax.legend(frameon=False)
    style_academic_axes(ax)
    fig.tight_layout()
    saved = save_figure(fig, output_dir, "fig_lowrank_forecastability", dpi=450)
    plt.close(fig)
    return saved


def plot_official_lead_error(
    by_lead: pd.DataFrame,
    *,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Compare official forecast MAE and RMSE across lead-time bins."""
    configure_plots()
    x = np.arange(len(by_lead))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    ax.bar(
        x - width / 2,
        by_lead["mae"],
        width,
        color=CUMCM_PALETTE["primary"],
        label="MAE",
    )
    ax.bar(
        x + width / 2,
        by_lead["rmse"],
        width,
        color=CUMCM_PALETTE["price"],
        label="RMSE",
    )
    ax.set_xticks(x, by_lead["lead_bin"])
    ax.set_title("官方光伏预报误差随提前量的变化")
    ax.set_xlabel("预测提前量")
    ax.set_ylabel("预测误差（kW）")
    ax.legend(frameon=False, ncol=2)
    style_academic_axes(ax)
    fig.tight_layout()
    saved = save_figure(fig, output_dir, "fig_official_lead_error", dpi=450)
    plt.close(fig)
    return saved
