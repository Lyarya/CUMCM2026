"""Chinese publication figures for Stage 2A forecasting evidence."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.plotting import CUMCM_PALETTE, configure_plots, save_figure, style_academic_axes


MODEL_LABELS = {
    "Yesterday": "昨日同刻",
    "Last Week": "上周同刻",
    "7-day mean": "近7日同刻均值",
    "Seasonal": "相位组合",
    "Low-Rank rank 1": "低秩专家（秩1）",
    "Low-Rank rank 3": "低秩专家（秩3）",
    "Low-Rank rank 5": "低秩专家（秩5）",
    "Seasonal + Low-Rank": "相位—低秩融合",
}


def _bootstrap_mean_interval(
    values: np.ndarray, *, seed: int, repetitions: int = 2000
) -> tuple[float, float, float]:
    sample = np.asarray(values, dtype=float)
    sample = sample[np.isfinite(sample)]
    if sample.size < 2:
        raise ValueError("bootstrap interval requires at least two finite daily metrics")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, sample.size, size=(repetitions, sample.size))
    means = sample[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(sample.mean()), float(lower), float(upper)


def plot_generation_model_comparison(
    daily_metrics: pd.DataFrame,
    output_dir: str | Path,
    *,
    seed: int = 2026,
    selected_model: str,
) -> tuple[dict[str, Path], pd.DataFrame]:
    """Compare daily generation RMSE with paired-day bootstrap uncertainty."""
    configure_plots()
    subset = daily_metrics.loc[daily_metrics["variable"] == "generation"].copy()
    rows = []
    model_order = [model for model in MODEL_LABELS if model in set(subset["model"])]
    for model_index, model in enumerate(model_order):
        mean, lower, upper = _bootstrap_mean_interval(
            subset.loc[subset["model"] == model, "rmse_kw"].to_numpy(),
            seed=seed + model_index,
        )
        rows.append(
            {
                "model": model,
                "daily_rmse_mean_kw": mean,
                "bootstrap_95_lower_kw": lower,
                "bootstrap_95_upper_kw": upper,
                "daily_samples": int((subset["model"] == model).sum()),
            }
        )
    summary = pd.DataFrame(rows).sort_values("daily_rmse_mean_kw", kind="stable")
    phase_summary = summary.loc[~summary["model"].str.startswith("Low-Rank")]
    lowrank_summary = summary.loc[summary["model"].str.startswith("Low-Rank")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.5), facecolor="white")
    for axis, shown, title in [
        (axes[0], phase_summary, "相位类方法与融合"),
        (axes[1], lowrank_summary, "Hankel低秩解析外推"),
    ]:
        y = np.arange(len(shown))
        means = shown["daily_rmse_mean_kw"].to_numpy()
        errors = np.vstack(
            [
                means - shown["bootstrap_95_lower_kw"].to_numpy(),
                shown["bootstrap_95_upper_kw"].to_numpy() - means,
            ]
        )
        colors = [
            CUMCM_PALETTE["primary"] if model == selected_model else CUMCM_PALETTE["neutral"]
            for model in shown["model"]
        ]
        axis.errorbar(
            means,
            y,
            xerr=errors,
            fmt="none",
            ecolor=CUMCM_PALETTE["neutral"],
            elinewidth=1.2,
            capsize=3,
            zorder=1,
        )
        axis.scatter(means, y, s=42, c=colors, edgecolor="white", linewidth=0.6, zorder=3)
        axis.set_yticks(y, [MODEL_LABELS[model] for model in shown["model"]])
        axis.invert_yaxis()
        axis.set_xlabel("日均方根误差（kW）")
        axis.set_title(title, fontsize=10)
        style_academic_axes(axis, grid_axis="x")
    axes[0].set_ylabel("新能源发电预测方法")
    fig.text(
        0.99,
        0.005,
        "误差线：334个评价日的95% bootstrap置信区间",
        ha="right",
        va="bottom",
        fontsize=7.5,
        color=CUMCM_PALETTE["observed"],
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    saved = save_figure(fig, output_dir, "fig_p2_generation_model_comparison", dpi=450)
    plt.close(fig)
    return saved, summary


def plot_joint_scenario_example(
    predictions: pd.DataFrame,
    scenario_archive: dict[str, np.ndarray],
    output_dir: str | Path,
) -> tuple[dict[str, Path], str]:
    """Show one representative day's paired residual scenario envelope."""
    configure_plots()
    daily_energy = predictions.groupby("operating_date", sort=True)["actual_generation"].sum()
    representative = str((daily_energy - daily_energy.median()).abs().idxmin())
    target_dates = pd.to_datetime(scenario_archive["target_dates"])
    archive_position = int(np.flatnonzero(target_dates == pd.Timestamp(representative))[0])
    day = predictions.loc[predictions["operating_date"] == representative].copy()
    load_scenarios = scenario_archive["load_kw"][archive_position]
    generation_scenarios = scenario_archive["generation_kw"][archive_position]
    hours = np.arange(len(day)) / 6

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.3), sharex=True, facecolor="white")
    panels = [
        (
            axes[0],
            load_scenarios,
            day["forecast_load"].to_numpy(),
            day["actual_load"].to_numpy(),
            "负荷功率（kW）",
            CUMCM_PALETTE["load"],
        ),
        (
            axes[1],
            generation_scenarios,
            day["forecast_generation"].to_numpy(),
            day["actual_generation"].to_numpy(),
            "新能源发电功率（kW）",
            CUMCM_PALETTE["pv"],
        ),
    ]
    for axis, scenarios, forecast, actual, ylabel, color in panels:
        q10, q25, q75, q90 = np.quantile(scenarios, [0.10, 0.25, 0.75, 0.90], axis=0)
        axis.fill_between(hours, q10, q90, color=color, alpha=0.15, label="10%—90%场景区间")
        axis.fill_between(hours, q25, q75, color=color, alpha=0.28, label="25%—75%场景区间")
        axis.plot(hours, forecast, color=color, linewidth=1.8, label="点预测")
        axis.plot(
            hours,
            actual,
            color=CUMCM_PALETTE["observed"],
            linewidth=1.5,
            label="实际值",
        )
        axis.set_ylabel(ylabel)
        style_academic_axes(axis)
    axes[0].legend(loc="upper right", ncol=2, fontsize=8.5)
    axes[1].set_xlabel("运营日内时刻")
    axes[1].set_xticks(np.arange(0, 25, 4), [f"{hour:02d}:00" for hour in range(0, 25, 4)])
    fig.tight_layout()
    saved = save_figure(fig, output_dir, "fig_p2_joint_residual_scenarios", dpi=450)
    plt.close(fig)
    return saved, representative
