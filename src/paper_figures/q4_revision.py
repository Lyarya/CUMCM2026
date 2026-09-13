"""Redraw the compact Q4 relationship and causal price-forecast panels.

The script reads the audited canonical series and frozen comparison table.  It
does not rerun forecasting or dispatch and leaves the original figure assets
unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cumcm-q4-paper-mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

from src.common.paths import DATA_DIR, PAPER_FIGURES_DIR, RESULTS_DIR
from src.common.plotting import configure_plots, style_academic_axes


ACTUAL_PATH = DATA_DIR / "processed" / "C题" / "actual_10min.csv"
FORECAST_METRICS_PATH = RESULTS_DIR / "problem4" / "tables" / "q4_price_forecast_comparison.csv"
OUTPUT_DIR = PAPER_FIGURES_DIR / "redesign"


def _configure() -> None:
    configure_plots()
    plt.rcParams.update(
        {
            "font.sans-serif": ["Songti SC", "SimSun", "STSong", "Arial Unicode MS"],
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, name: str) -> dict[str, Path]:
    """Export editable vectors and 600-dpi review rasters."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base = OUTPUT_DIR / name
    outputs = {
        "svg": base.with_suffix(".svg"),
        "pdf": base.with_suffix(".pdf"),
        "jpg": base.with_suffix(".jpg"),
        "png": base.with_suffix(".png"),
        "tiff": base.with_suffix(".tiff"),
    }
    common = {"bbox_inches": "tight", "facecolor": "white"}
    fig.savefig(outputs["svg"], **common)
    fig.savefig(outputs["pdf"], **common)
    fig.savefig(outputs["jpg"], dpi=600, **common)
    fig.savefig(outputs["png"], dpi=600, **common)
    fig.savefig(outputs["tiff"], dpi=600, pil_kwargs={"compression": "tiff_lzw"}, **common)
    return outputs


def plot_price_netload_relation(frame: pd.DataFrame) -> dict[str, Path]:
    """Show the full-sample joint density and its descriptive linear trend."""

    x = frame["net_load_kw"].to_numpy(float)
    y = frame["price_yuan_per_kwh"].to_numpy(float)
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("The canonical price/net-load series must be finite")

    slope, intercept = np.polyfit(x, y, deg=1)
    pearson = float(pd.Series(x).corr(pd.Series(y), method="pearson"))
    spearman = float(pd.Series(x).corr(pd.Series(y), method="spearman"))
    x_line = np.linspace(-2500.0, 7000.0, 300)

    fig, ax = plt.subplots(figsize=(7.1, 3.15), layout="constrained")
    cmap = LinearSegmentedColormap.from_list(
        "muted_blue_density", ["#F7F8FA", "#C8D7E1", "#779CB5", "#355F7A"]
    )
    density = ax.hexbin(
        x,
        y,
        gridsize=68,
        mincnt=1,
        cmap=cmap,
        linewidths=0,
        rasterized=True,
    )
    ax.plot(
        x_line,
        intercept + slope * x_line,
        color="#B65F54",
        linestyle="--",
        linewidth=1.8,
        label="线性回归趋势",
        zorder=5,
    )
    ax.set_xlim(-2500, 7000)
    ax.set_ylim(0, max(1.82, float(y.max()) * 1.02))
    ax.set_xlabel("净负荷（kW）")
    ax.set_ylabel("电价（元/kWh）")
    ax.legend(loc="upper left", frameon=False)
    ax.text(
        0.985,
        0.96,
        f"Pearson $r={pearson:.3f}$\nSpearman $\\rho={spearman:.3f}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color="#4D5963",
        linespacing=1.25,
    )
    style_academic_axes(ax, grid_axis=None)
    colorbar = fig.colorbar(density, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label("区间数量")
    return _save(fig, "fig_q4_price_netload_relation")


def plot_causal_price_forecast(metrics: pd.DataFrame) -> dict[str, Path]:
    """Compare January validation and frozen Feb--Dec OOS RMSE."""

    periods = ("validation", "evaluation")
    if set(periods) - set(metrics["period"]):
        raise ValueError("Both validation and evaluation rows are required")
    methods = metrics.loc[metrics["period"].eq("validation"), "method"].tolist()
    validation = metrics.loc[metrics["period"].eq("validation")].set_index("method").loc[methods]
    evaluation = metrics.loc[metrics["period"].eq("evaluation")].set_index("method").loc[methods]

    centers = np.arange(len(methods), dtype=float) * 1.25
    width = 0.27
    fig, ax = plt.subplots(figsize=(7.1, 3.45), layout="constrained")
    bars_validation = ax.bar(
        centers - width / 1.7,
        validation["rmse_yuan_per_kwh"],
        width,
        color="#AFC4D0",
        edgecolor="#56758A",
        linewidth=0.65,
        label="1月验证集",
    )
    bars_evaluation = ax.bar(
        centers + width / 1.7,
        evaluation["rmse_yuan_per_kwh"],
        width,
        color="#B9786A",
        edgecolor="#7D493F",
        linewidth=0.65,
        hatch="//",
        label="2—12月冻结样本外",
    )
    for bars in (bars_validation, bars_evaluation):
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=7, color="#333333")

    upper = max(
        float(validation["rmse_yuan_per_kwh"].max()),
        float(evaluation["rmse_yuan_per_kwh"].max()),
    )
    ax.set_ylim(0, upper * 1.24)
    ax.set_xlim(centers[0] - 0.62, centers[-1] + 0.62)
    ax.set_xticks(centers, methods)
    ax.set_ylabel("RMSE（元/kWh）")
    ax.set_xlabel("价格预测方法")
    ax.legend(loc="upper center", ncol=2, frameon=False)
    style_academic_axes(ax)
    return _save(fig, "fig_q4_causal_price_forecast")


def main() -> None:
    _configure()
    frame = pd.read_csv(ACTUAL_PATH)
    metrics = pd.read_csv(FORECAST_METRICS_PATH)
    outputs = {
        **{f"relation_{key}": value for key, value in plot_price_netload_relation(frame).items()},
        **{f"forecast_{key}": value for key, value in plot_causal_price_forecast(metrics).items()},
    }
    plt.close("all")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
