"""Forecast diagnostics for C-problem load and photovoltaic power."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from src.common.paths import problem_figures_dir
from src.common.plotting import CUMCM_PALETTE, configure_plots, style_academic_axes
from src.common.spectral import SingularSpectrumAnalysis
from src.problem2.evaluation_analysis import DailyOptimizerResult, daily_metrics


FIGURE_WIDTH_MM = 183.0


def _configure_q2_plots() -> None:
    """Apply the shared competition style and keep vector text editable."""
    configure_plots()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Songti SC", "SimSun", "STSong", "Arial Unicode MS"],
            "font.size": 10,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def plot_hankel_singular_spectrum(
    analysis: SingularSpectrumAnalysis,
    *,
    signal_label: str = "光伏功率",
    max_rank: int = 20,
    output_dir: str | Path | None = None,
    name: str = "fig_p2_hankel_singular_spectrum",
) -> dict[str, Path]:
    """Plot normalized singular values and cumulative spectral energy."""
    _configure_q2_plots()
    shown = analysis.summary.loc[analysis.summary["rank"] <= max_rank].copy()
    if shown.empty:
        raise ValueError("No spectrum ranks are available for plotting.")
    spectrum_columns = [
        "normalized_singular_value_q25",
        "normalized_singular_value_q75",
        "normalized_singular_value_median",
    ]
    if np.any(shown[spectrum_columns].to_numpy(dtype=float) <= 0):
        raise ValueError("Singular values must be strictly positive on a logarithmic axis.")
    ranks = shown["rank"].to_numpy(dtype=float)
    median_color = CUMCM_PALETTE["primary"]
    band_color = CUMCM_PALETTE["neutral_light"]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), facecolor="white")
    axes[0].fill_between(
        ranks,
        shown["normalized_singular_value_q25"].to_numpy(dtype=float),
        shown["normalized_singular_value_q75"].to_numpy(dtype=float),
        color=band_color,
        alpha=0.85,
        linewidth=0,
    )
    axes[0].plot(
        ranks,
        shown["normalized_singular_value_median"],
        color=median_color,
        linewidth=1.9,
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"归一化奇异值 $\sigma_i/\sigma_1$")

    axes[1].fill_between(
        ranks,
        shown["cumulative_energy_q25"].to_numpy(dtype=float) * 100,
        shown["cumulative_energy_q75"].to_numpy(dtype=float) * 100,
        color=band_color,
        alpha=0.85,
        linewidth=0,
    )
    axes[1].plot(
        ranks,
        shown["cumulative_energy_median"].to_numpy(dtype=float) * 100,
        color=median_color,
        linewidth=1.9,
    )
    axes[1].set_ylim(0, 100.5)
    axes[1].set_ylabel("累计谱能量（%）")

    for axis in axes:
        axis.set_xlabel("奇异值序号")
        axis.set_xlim(1, float(shown["rank"].max()))
        style_academic_axes(axis)
    axes[0].set_title(f"{signal_label}：奇异值衰减")
    axes[1].set_title(f"{signal_label}：累计谱能量")
    fig.legend(
        handles=[
            Line2D([0], [0], color=median_color, linewidth=1.9, label="窗口中位数"),
            Patch(facecolor=band_color, edgecolor="none", alpha=0.85, label="四分位区间"),
        ],
        loc="upper center",
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    return _save_and_close(fig, output_dir, name)


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")
    numeric = frame[list(columns - {"date"})].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{label} contains NaN or infinite plotting values")


def _save_and_close(
    fig: plt.Figure,
    output_dir: str | Path | None,
    name: str,
) -> dict[str, Path]:
    target = Path(output_dir or problem_figures_dir(2))
    target.mkdir(parents=True, exist_ok=True)
    paths = {
        "svg": target / f"{name}.svg",
        "pdf": target / f"{name}.pdf",
        "jpg": target / f"{name}.jpg",
        "png": target / f"{name}.png",
    }
    common = {"bbox_inches": "tight", "facecolor": "white", "transparent": False}
    fig.savefig(paths["svg"], **common)
    fig.savefig(paths["pdf"], **common)
    fig.savefig(paths["jpg"], dpi=400, **common)
    fig.savefig(paths["png"], dpi=600, **common)
    plt.close(fig)
    return paths


def plot_daily_economic_performance(
    strategy_daily: pd.DataFrame,
    baseline_daily: pd.DataFrame,
    *,
    strategy_label: str = "随机优化策略",
    baseline_label: str = "基准策略",
    output_dir: str | Path | None = None,
    name: str = "fig_p2_daily_economic_performance",
) -> dict[str, Path]:
    """Plot daily and cumulative savings; positive values mean cost savings."""
    required = {"date", "total_cost_yuan"}
    _require_columns(strategy_daily, required, "strategy_daily")
    _require_columns(baseline_daily, required, "baseline_daily")
    strategy = strategy_daily[list(required)].rename(
        columns={"total_cost_yuan": "strategy_cost"}
    )
    baseline = baseline_daily[list(required)].rename(
        columns={"total_cost_yuan": "baseline_cost"}
    )
    merged = baseline.merge(strategy, on="date", validate="one_to_one").sort_values(
        "date", kind="stable"
    )
    if len(merged) != len(strategy_daily) or len(merged) != len(baseline_daily):
        raise AssertionError("strategy and baseline daily dates must match exactly")
    dates = pd.to_datetime(merged["date"])
    saving = merged["baseline_cost"].to_numpy() - merged["strategy_cost"].to_numpy()
    cumulative = np.cumsum(saving)

    _configure_q2_plots()
    fig, axes = plt.subplots(
        2, 1, figsize=(11.2, 6.6), sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1.0]},
        layout="constrained",
        facecolor="white",
    )
    colors = np.where(saving >= 0, CUMCM_PALETTE["good"], CUMCM_PALETTE["bad"])
    axes[0].bar(dates, saving, width=1.0, color=colors, linewidth=0)
    axes[0].axhline(0, color=CUMCM_PALETTE["neutral"], linewidth=0.8)
    axes[0].set_ylabel("日成本节省（元）")
    axes[0].set_title(f"{strategy_label}相对{baseline_label}的逐日经济收益")
    axes[1].plot(dates, cumulative, color=CUMCM_PALETTE["primary"], linewidth=2.0)
    axes[1].set(xlabel="运营日期", ylabel="累计成本节省（元）")
    for axis in axes:
        style_academic_axes(axis)
    fig.autofmt_xdate(rotation=0)
    return _save_and_close(fig, output_dir, name)


def plot_daily_cost_decomposition(
    daily: pd.DataFrame,
    *,
    output_dir: str | Path | None = None,
    name: str = "fig_p2_daily_cost_decomposition",
) -> dict[str, Path]:
    """Plot planned, emergency, and total realized/expected cost over time."""
    required = {
        "date", "planned_purchase_cost_yuan", "emergency_purchase_cost_yuan",
        "total_cost_yuan",
    }
    _require_columns(daily, required, "daily")
    frame = daily.sort_values("date", kind="stable")
    dates = pd.to_datetime(frame["date"])
    _configure_q2_plots()
    fig, ax = plt.subplots(figsize=(11.2, 4.8), facecolor="white")
    ax.plot(
        dates, frame["planned_purchase_cost_yuan"],
        color=CUMCM_PALETTE["grid_purchase"], linewidth=1.3, label="计划购电成本",
    )
    ax.plot(
        dates, frame["emergency_purchase_cost_yuan"],
        color=CUMCM_PALETTE["bad"], linewidth=1.2, label="紧急购电成本",
    )
    ax.plot(
        dates, frame["total_cost_yuan"],
        color=CUMCM_PALETTE["primary"], linewidth=1.8, label="总成本",
    )
    ax.set(xlabel="运营日期", ylabel="日成本（元）", title="问题二逐日成本分解")
    ax.legend(ncol=3, loc="upper center", frameon=False)
    style_academic_axes(ax)
    fig.autofmt_xdate(rotation=0)
    fig.tight_layout()
    return _save_and_close(fig, output_dir, name)


def plot_risk_cost_tradeoff(
    risk_sweep: pd.DataFrame,
    *,
    parameter_label: str = "风险参数",
    output_dir: str | Path | None = None,
    name: str = "fig_p2_risk_cost_tradeoff",
) -> dict[str, Path]:
    """Plot cost and emergency-energy responses for a future risk sweep."""
    required = {
        "risk_parameter", "planned_cost", "emergency_cost", "total_cost",
        "emergency_energy",
    }
    _require_columns(risk_sweep, required, "risk_sweep")
    frame = risk_sweep.sort_values("risk_parameter", kind="stable")
    x = frame["risk_parameter"].to_numpy(dtype=float)
    _configure_q2_plots()
    fig, axes = plt.subplots(
        2, 1, figsize=(8.2, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [1.3, 1.0]},
        layout="constrained",
        facecolor="white",
    )
    axes[0].plot(
        x, frame["planned_cost"], marker="o", color=CUMCM_PALETTE["grid_purchase"],
        linewidth=1.6, label="计划购电成本",
    )
    axes[0].plot(
        x, frame["emergency_cost"], marker="s", color=CUMCM_PALETTE["bad"],
        linewidth=1.6, label="紧急购电成本",
    )
    axes[0].plot(
        x, frame["total_cost"], marker="D", color=CUMCM_PALETTE["primary"],
        linewidth=2.0, label="总成本",
    )
    axes[0].set_ylabel("累计成本（元）")
    axes[0].set_title("风险设置下的成本—可靠性权衡")
    axes[0].legend(ncol=3, loc="best", frameon=False)
    axes[1].plot(
        x, frame["emergency_energy"], marker="o", color=CUMCM_PALETTE["bad"],
        linewidth=1.8,
    )
    axes[1].set(xlabel=parameter_label, ylabel="累计紧急购电量（kWh）")
    for axis in axes:
        style_academic_axes(axis)
    return _save_and_close(fig, output_dir, name)


def plot_cvar_risk_tradeoff(
    risk_sweep: pd.DataFrame,
    *,
    output_dir: str | Path | None = None,
    name: str = "fig_p2_cvar_risk_tradeoff",
) -> dict[str, Path]:
    """Plot the finalized CVaR sweep without conflating ex-ante and ex-post metrics."""
    required = {
        "lambda", "expected_operating_cost_yuan", "cvar_cost_yuan",
        "realized_emergency_energy_kwh",
    }
    _require_columns(risk_sweep, required, "risk_sweep")
    frame = risk_sweep.sort_values("lambda", kind="stable")
    x = frame["lambda"].to_numpy(dtype=float)
    if x.tolist() != [0.0, 0.05, 0.1, 0.2, 0.5]:
        raise ValueError("the finalized CVaR figure requires the five predeclared lambda values")

    _configure_q2_plots()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), facecolor="white")
    panels = (
        ("expected_operating_cost_yuan", 1e6, "期望运行成本（百万元）", CUMCM_PALETTE["primary"]),
        ("cvar_cost_yuan", 1e6, "紧急购电CVaR（百万元）", CUMCM_PALETTE["bad"]),
        ("realized_emergency_energy_kwh", 1e4, "事后紧急购电量（万kWh）", CUMCM_PALETTE["price"]),
    )
    for label, (column, scale, ylabel, color), axis in zip("abc", panels, axes):
        y = frame[column].to_numpy(dtype=float) / scale
        axis.plot(x, y, marker="o", color=color, linewidth=1.7, markersize=4.2)
        mild = int(np.flatnonzero(np.isclose(x, 0.05))[0])
        axis.scatter(
            [x[mild]], [y[mild]], s=42, facecolor="white", edgecolor=color,
            linewidth=1.4, zorder=5,
        )
        axis.set_xticks(x, [f"{value:g}" for value in x])
        axis.tick_params(axis="x", labelrotation=35, labelsize=7.5)
        for tick in axis.get_xticklabels():
            tick.set_horizontalalignment("right")
        axis.set_xlabel(r"风险权重 $\lambda_{\rm r}$")
        axis.set_ylabel(ylabel)
        axis.set_title(f"{label}  {ylabel.split('（')[0]}", loc="left", fontsize=9.5)
        style_academic_axes(axis)
    fig.text(
        0.5, 0.01,
        r"空心点为 $\lambda_{\rm r}=0.05$；事后指标仅用于评价，不参与风险权重选择。",
        ha="center", fontsize=7.5, color=CUMCM_PALETTE["observed"],
    )
    fig.tight_layout(rect=(0, 0.055, 1, 1), w_pad=1.2)

    target = Path(output_dir or problem_figures_dir(2))
    target.mkdir(parents=True, exist_ok=True)
    paths = {suffix: target / f"{name}.{suffix}" for suffix in ("svg", "pdf", "jpg", "png", "tiff")}
    common = {"bbox_inches": "tight", "facecolor": "white", "transparent": False}
    fig.savefig(paths["svg"], **common)
    fig.savefig(paths["pdf"], **common)
    fig.savefig(paths["jpg"], dpi=600, **common)
    fig.savefig(paths["png"], dpi=600, **common)
    fig.savefig(paths["tiff"], dpi=600, **common)
    plt.close(fig)
    # Matplotlib writes harmless spaces before SVG line breaks; remove them so
    # repository whitespace checks remain meaningful.
    svg_text = paths["svg"].read_text(encoding="utf-8")
    paths["svg"].write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    return paths


def plot_soc_emergency_diagnostic(
    result: DailyOptimizerResult,
    *,
    output_dir: str | Path | None = None,
    name: str = "fig_p2_soc_emergency_diagnostic",
) -> dict[str, Path]:
    """Plot one real optimizer day's grid/emergency power and SOC trajectory."""
    daily_metrics(result, validate_costs_from_price=result.price_yuan_per_kwh is not None)
    grid = np.asarray(result.planned_grid_purchase_kw, dtype=float)
    emergency_rows = np.asarray(result.emergency_purchase_kw, dtype=float).reshape(-1, 144)
    weights = None
    if result.scenario_weights is not None and len(emergency_rows) > 1:
        weights = np.asarray(result.scenario_weights, dtype=float).reshape(-1)
        weights = weights / weights.sum()
    emergency = np.average(emergency_rows, axis=0, weights=weights)
    soc = np.asarray(result.soc_kwh, dtype=float)
    interval_hours = np.arange(144) / 6.0
    state_hours = np.arange(145) / 6.0

    _configure_q2_plots()
    fig, axes = plt.subplots(
        2, 1, figsize=(10.4, 6.0), sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1.0]},
        layout="constrained",
        facecolor="white",
    )
    axes[0].step(
        interval_hours, grid, where="post", color=CUMCM_PALETTE["grid_purchase"],
        linewidth=1.6, label="计划购电功率",
    )
    axes[0].step(
        interval_hours, emergency, where="post", color=CUMCM_PALETTE["bad"],
        linewidth=1.5, label="紧急购电功率",
    )
    axes[0].set_ylabel("功率（kW）")
    axes[0].set_title(f"代表日调度诊断：{pd.Timestamp(result.date).date()}")
    axes[0].legend(ncol=2, loc="upper center", frameon=False)
    axes[1].plot(state_hours, soc, color=CUMCM_PALETTE["soc"], linewidth=2.0)
    axes[1].set(xlabel="时刻", ylabel="储能电量（kWh）", xlim=(0, 24))
    for axis in axes:
        style_academic_axes(axis)
    return _save_and_close(fig, output_dir, name)


__all__ = [
    "plot_daily_cost_decomposition",
    "plot_daily_economic_performance",
    "plot_hankel_singular_spectrum",
    "plot_cvar_risk_tradeoff",
    "plot_risk_cost_tradeoff",
    "plot_soc_emergency_diagnostic",
]
