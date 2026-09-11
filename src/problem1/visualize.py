"""Publication figures for Problem 1 deterministic dispatch."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.paths import paper_problem_figures_dir, problem_figures_dir
from src.common.plotting import CUMCM_PALETTE, configure_plots, save_figure, style_academic_axes


def _hour_ticks() -> tuple[np.ndarray, list[str]]:
    ticks = np.arange(0, 25, 4)
    return ticks, [f"{hour}:00" for hour in ticks]


def create_q1_figures(dispatch: pd.DataFrame) -> dict[str, dict[str, Path]]:
    """Generate the two required Q1 figures and copy their PDFs into the paper."""
    configure_plots()
    output_dir = problem_figures_dir(1)
    paper_dir = paper_problem_figures_dir(1)
    paper_dir.mkdir(parents=True, exist_ok=True)
    hours = np.arange(len(dispatch), dtype=float) / 6.0
    ticks, tick_labels = _hour_ticks()

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(11.2, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.7, 1.0], "hspace": 0.12},
        facecolor="white",
    )
    ax_top.plot(hours, dispatch["load_kw"], color=CUMCM_PALETTE["load"], linewidth=1.8, label="小区负荷")
    ax_top.plot(hours, dispatch["pv_forecast_kw"], color=CUMCM_PALETTE["pv"], linewidth=1.8, label="光伏预测功率")
    ax_top.step(hours, dispatch["grid_purchase_kw"], where="post", color=CUMCM_PALETTE["grid_purchase"], linewidth=1.5, label="外网购电功率")
    ax_top.set_ylabel("功率（kW）")
    ax_top.set_title("问题一确定性调度的供需功率")
    ax_top.legend(ncol=3, loc="upper center", frameon=False)
    style_academic_axes(ax_top)

    ax_bottom.step(hours, dispatch["charge_kw"], where="post", color=CUMCM_PALETTE["charge"], linewidth=1.5, label="充电功率")
    ax_bottom.step(hours, -dispatch["discharge_kw"], where="post", color=CUMCM_PALETTE["discharge"], linewidth=1.5, label="放电功率（负向展示）")
    ax_bottom.axhline(0, color=CUMCM_PALETTE["neutral"], linewidth=0.8)
    ax_bottom.set(xlabel="时刻", ylabel="储能功率（kW）", xticks=ticks, xticklabels=tick_labels, xlim=(0, 24))
    ax_bottom.legend(ncol=2, loc="lower center", frameon=False)
    style_academic_axes(ax_bottom)
    fig.align_ylabels()
    saved_power = save_figure(fig, output_dir, "fig_p1_dispatch_power", dpi=400)
    plt.close(fig)

    energy_hours = np.arange(len(dispatch) + 1, dtype=float) / 6.0
    energy = np.concatenate(
        ([dispatch["storage_start_kwh"].iloc[0]], dispatch["storage_end_kwh"].to_numpy())
    )
    fig, ax_energy = plt.subplots(figsize=(11.2, 4.9), facecolor="white")
    ax_energy.plot(energy_hours, energy, color=CUMCM_PALETTE["soc"], linewidth=2.0, label="储能电量")
    ax_energy.axhline(1_200, color=CUMCM_PALETTE["bad"], linestyle="--", linewidth=1.0, label="运行下限")
    ax_energy.axhline(10_800, color=CUMCM_PALETTE["good"], linestyle="--", linewidth=1.0, label="运行上限")
    ax_energy.set(xlabel="时刻", ylabel="储能电量（kWh）", title="储能电量轨迹与分时电价", xticks=ticks, xticklabels=tick_labels, xlim=(0, 24))
    style_academic_axes(ax_energy)
    ax_price = ax_energy.twinx()
    ax_price.step(hours, dispatch["price_yuan_per_kwh"], where="post", color=CUMCM_PALETTE["price"], linewidth=1.35, alpha=0.9, label="分时电价")
    ax_price.set_ylabel("电价（元/kWh）")
    ax_price.spines["top"].set_visible(False)
    lines_a, labels_a = ax_energy.get_legend_handles_labels()
    lines_b, labels_b = ax_price.get_legend_handles_labels()
    ax_energy.legend(lines_a + lines_b, labels_a + labels_b, ncol=4, loc="upper center", frameon=False)
    fig.tight_layout()
    saved_storage = save_figure(fig, output_dir, "fig_p1_storage_price", dpi=400)
    plt.close(fig)

    for saved in (saved_power, saved_storage):
        shutil.copy2(saved["pdf"], paper_dir / saved["pdf"].name)
    return {"dispatch_power": saved_power, "storage_price": saved_storage}
