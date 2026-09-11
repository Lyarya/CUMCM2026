"""Chinese publication figures for the Stage 2A+ SSA checkpoint."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.plotting import CUMCM_PALETTE, configure_plots, style_academic_axes


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Songti SC", "SimSun", "STSong", "Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def _save_ssa_figure(fig: plt.Figure, output_dir: str | Path, name: str) -> dict[str, Path]:
    """Export editable vectors and high-resolution Chinese previews."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "svg": output / f"{name}.svg",
        "pdf": output / f"{name}.pdf",
        "jpg": output / f"{name}.jpg",
        "png": output / f"{name}.png",
    }
    common = {"bbox_inches": "tight", "facecolor": "white", "transparent": False}
    fig.savefig(paths["svg"], **common)
    fig.savefig(paths["pdf"], **common)
    fig.savefig(paths["jpg"], dpi=600, **common)
    fig.savefig(paths["png"], dpi=600, **common)
    return paths


def _bootstrap_mean_interval(
    values: np.ndarray, *, seed: int, repetitions: int = 2000
) -> tuple[float, float, float]:
    sample = np.asarray(values, dtype=float)
    sample = sample[np.isfinite(sample)]
    if sample.size < 2:
        raise ValueError("bootstrap interval requires at least two finite daily values")
    generator = np.random.default_rng(seed)
    draw = generator.integers(0, sample.size, size=(repetitions, sample.size))
    means = sample[draw].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(sample.mean()), float(lower), float(upper)


def plot_ssa_validation(
    daily: pd.DataFrame,
    output_dir: str | Path,
    *,
    selected_window: int,
    selected_rank: int,
    seed: int = 2026,
) -> dict[str, Path]:
    """Show January daily RMSE and retained energy over the declared SSA grid."""
    configure_plots()
    colors = [CUMCM_PALETTE["primary"], CUMCM_PALETTE["pv"], CUMCM_PALETTE["price"]]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), facecolor="white")
    for color, (window, group) in zip(colors, daily.groupby("window_days", sort=True)):
        points = []
        lower = []
        upper = []
        energy = []
        ranks = sorted(group["rank"].unique())
        for rank_index, rank in enumerate(ranks):
            subset = group.loc[(group["rank"] == rank) & (group["status"] == "ok")]
            mean, lo, hi = _bootstrap_mean_interval(
                subset["full_rmse_kw"].to_numpy(),
                seed=seed + int(window) * 10 + rank_index,
            )
            points.append(mean)
            lower.append(mean - lo)
            upper.append(hi - mean)
            energy.append(float(subset["cumulative_singular_energy"].mean()) * 100)
        axes[0].errorbar(
            ranks,
            points,
            yerr=np.vstack([lower, upper]),
            marker="o",
            markersize=4.5,
            linewidth=1.5,
            capsize=2.5,
            color=color,
            label=f"{window}日历史窗",
        )
        axes[1].plot(
            ranks,
            energy,
            marker="o",
            markersize=4.5,
            linewidth=1.5,
            color=color,
            label=f"{window}日历史窗",
        )
    selected = daily.loc[
        (daily["window_days"] == selected_window)
        & (daily["rank"] == selected_rank)
        & (daily["status"] == "ok")
    ]
    axes[0].scatter(
        [selected_rank],
        [selected["full_rmse_kw"].mean()],
        marker="*",
        s=90,
        color=CUMCM_PALETTE["bad"],
        edgecolor="white",
        linewidth=0.6,
        zorder=5,
        label="一月选定配置",
    )
    axes[0].set(
        xlabel="SSA 秩",
        ylabel="日均方根误差（kW）",
        title="a  一月滚动验证误差",
        xticks=sorted(daily["rank"].unique()),
    )
    axes[1].set(
        xlabel="SSA 秩",
        ylabel="累计奇异值能量（%）",
        title="b  低秩能量保留率",
        xticks=sorted(daily["rank"].unique()),
    )
    for axis in axes:
        style_academic_axes(axis)
    axes[0].legend(loc="best", fontsize=7.5)
    fig.text(
        0.01,
        0.005,
        "误差线为17个共同验证日的日RMSE均值95% bootstrap区间；星号表示仅由一月确定的配置。",
        ha="left",
        va="bottom",
        fontsize=7.2,
        color=CUMCM_PALETTE["observed"],
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    paths = _save_ssa_figure(fig, output_dir, "fig_p2_ssa_validation")
    plt.close(fig)
    return paths


def plot_ssa_daily_comparison(
    daily: pd.DataFrame,
    output_dir: str | Path,
    *,
    seed: int = 2026,
) -> dict[str, Path]:
    """Compare monthly daily RMSE and the paired daily difference distribution."""
    configure_plots()
    frame = daily.copy()
    frame["month"] = pd.to_datetime(frame["date"]).dt.month
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), facecolor="white")
    model_specs = [
        ("seven_day_rmse_kw", "近7日同刻均值", CUMCM_PALETTE["primary"]),
        ("ssa_rmse_kw", "SSA结构专家", CUMCM_PALETTE["pv"]),
    ]
    for offset, (column, label, color) in enumerate(model_specs):
        means, lower, upper = [], [], []
        months = sorted(frame["month"].unique())
        for month in months:
            values = frame.loc[frame["month"] == month, column].to_numpy()
            mean, lo, hi = _bootstrap_mean_interval(
                values, seed=seed + offset * 100 + int(month)
            )
            means.append(mean)
            lower.append(mean - lo)
            upper.append(hi - mean)
        axes[0].errorbar(
            months,
            means,
            yerr=np.vstack([lower, upper]),
            marker="o",
            markersize=4,
            linewidth=1.4,
            capsize=2.2,
            label=label,
            color=color,
        )
    differences = frame["rmse_difference_ssa_minus_7day_kw"].to_numpy(dtype=float)
    axes[1].hist(
        differences,
        bins=24,
        color=CUMCM_PALETTE["neutral_light"],
        edgecolor=CUMCM_PALETTE["neutral"],
        linewidth=0.6,
    )
    axes[1].axvline(0, color=CUMCM_PALETTE["observed"], linewidth=1.1, linestyle="--")
    axes[1].axvline(
        np.median(differences),
        color=CUMCM_PALETTE["bad"] if np.median(differences) > 0 else CUMCM_PALETTE["good"],
        linewidth=1.5,
        label="差值中位数",
    )
    axes[0].set(
        xlabel="月份",
        ylabel="日均方根误差（kW）",
        title="a  正式期月度误差",
        xticks=sorted(frame["month"].unique()),
    )
    axes[1].set(
        xlabel="SSA与近7日均值的日RMSE差（kW）",
        ylabel="预测日数",
        title="b  每日成对误差差值",
    )
    axes[0].legend(loc="best", fontsize=8)
    axes[1].legend(loc="best", fontsize=8)
    style_academic_axes(axes[0])
    style_academic_axes(axes[1])
    fig.text(
        0.01,
        0.005,
        "月度误差线为预测日均值的95% bootstrap区间；差值大于0表示SSA误差更大。",
        ha="left",
        va="bottom",
        fontsize=7.2,
        color=CUMCM_PALETTE["observed"],
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    paths = _save_ssa_figure(fig, output_dir, "fig_p2_ssa_daily_comparison")
    plt.close(fig)
    return paths


def plot_ssa_representative_day(
    predictions: pd.DataFrame,
    daily: pd.DataFrame,
    output_dir: str | Path,
) -> tuple[dict[str, Path], str]:
    """Show the objectively median paired-difference day without cherry-picking."""
    median_difference = float(daily["rmse_difference_ssa_minus_7day_kw"].median())
    representative = str(
        daily.loc[
            (daily["rmse_difference_ssa_minus_7day_kw"] - median_difference).abs().idxmin(),
            "date",
        ]
    )
    day = predictions.loc[predictions["operating_date"] == representative]
    hours = np.arange(len(day)) / 6
    configure_plots()
    fig, axis = plt.subplots(figsize=(7.2, 3.5), facecolor="white")
    axis.plot(
        hours,
        day["actual_generation"],
        color=CUMCM_PALETTE["observed"],
        linewidth=2.0,
        label="实际值",
        zorder=5,
    )
    axis.plot(
        hours,
        day["seven_day_mean_pred"],
        color=CUMCM_PALETTE["primary"],
        linewidth=1.5,
        label="近7日同刻均值",
    )
    axis.plot(
        hours,
        day["ssa_pred"],
        color=CUMCM_PALETTE["pv"],
        linewidth=1.5,
        label="SSA结构专家",
    )
    axis.plot(
        hours,
        day["vandermonde_lowrank_pred"],
        color=CUMCM_PALETTE["price"],
        linewidth=1.15,
        linestyle="--",
        label="多项式Vandermonde低秩专家",
    )
    axis.set(
        xlabel="运营日内时刻",
        ylabel="新能源发电功率（kW）",
        title=f"代表日结构预测对比（{representative}）",
    )
    axis.set_xticks(np.arange(0, 25, 4), [f"{hour:02d}:00" for hour in range(0, 25, 4)])
    axis.legend(loc="upper right", ncol=2, fontsize=8)
    style_academic_axes(axis)
    fig.tight_layout()
    paths = _save_ssa_figure(fig, output_dir, "fig_p2_ssa_representative_day")
    plt.close(fig)
    return paths, representative
