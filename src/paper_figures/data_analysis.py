"""Regenerate the data-analysis lead-time figure from audited result tables.

Figure contract
---------------
Core conclusion: official PV forecast error increases with forecast lead time.
Archetype: single-panel quantitative trend comparison.
Evidence: MAE and RMSE for all five audited lead-time bins.
Statistics: deterministic grouped metrics; no uncertainty bars are implied.

Run with::

    python -m src.paper_figures.data_analysis
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cumcm-paper-data-analysis-mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.plotting import CUMCM_PALETTE as PAL
from src.common.plotting import configure_plots, style_academic_axes


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/tables/data_diagnostics/official_forecast_by_lead.csv"
OUTPUT_DIR = ROOT / "paper/figures/redesign"
OUTPUT_NAME = "fig_data_official_lead_error"
LEAD_ORDER = ["1-3h", "4-6h", "7-12h", "13-18h", "19-24h"]


def load_metrics() -> pd.DataFrame:
    """Load and validate the immutable lead-time summary."""
    data = pd.read_csv(SOURCE)
    required = {"lead_bin", "count", "mae", "rmse", "bias"}
    if not required.issubset(data.columns):
        missing = sorted(required.difference(data.columns))
        raise ValueError(f"Missing required columns: {missing}")
    data = data.set_index("lead_bin").loc[LEAD_ORDER].reset_index()
    if len(data) != 5 or not np.isfinite(data[["mae", "rmse"]]).all().all():
        raise ValueError("Lead-time metrics must contain five finite groups.")
    if not (data[["mae", "rmse"]] >= 0).all().all():
        raise ValueError("Forecast error metrics must be non-negative.")
    if int(data["count"].sum()) != 35004:
        raise ValueError("Aligned forecast count differs from the audited value 35004.")
    return data


def make_figure(data: pd.DataFrame) -> plt.Figure:
    """Create a compact trend figure consistent with the paper palette."""
    configure_plots()
    plt.rcParams.update(
        {
            "font.sans-serif": ["Songti SC", "SimSun", "STSong", "Arial Unicode MS"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    x = np.arange(len(data), dtype=float)
    fig, ax = plt.subplots(figsize=(7.1, 2.65), layout="constrained", facecolor="white")

    ax.plot(
        x,
        data["mae"],
        color=PAL["load"],
        linestyle="-",
        linewidth=1.6,
        marker="o",
        markersize=5.0,
        markerfacecolor=PAL["load"],
        markeredgecolor="white",
        markeredgewidth=0.7,
        label="MAE",
        zorder=3,
    )
    ax.plot(
        x,
        data["rmse"],
        color=PAL["price"],
        linestyle="--",
        linewidth=1.6,
        marker="s",
        markersize=5.2,
        markerfacecolor="white",
        markeredgecolor=PAL["price"],
        markeredgewidth=1.2,
        label="RMSE",
        zorder=3,
    )

    ax.set_xticks(x, [label.replace("h", " h") for label in LEAD_ORDER])
    ax.set_xlim(-0.22, 4.22)
    ax.set_ylim(0, 720)
    ax.set_yticks(np.arange(0, 701, 100))
    ax.set_xlabel("预测提前期")
    ax.set_ylabel("预测误差（kW）")
    ax.set_title("官方光伏预报误差的提前期效应", loc="left", pad=9)
    ax.legend(loc="upper left", ncol=2, frameon=False, handlelength=2.5)
    style_academic_axes(ax)

    for xpos, value in zip(x, data["mae"], strict=True):
        ax.annotate(
            f"{value:.1f}",
            (xpos, value),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=PAL["load"],
            fontsize=7,
        )
    for xpos, value in zip(x, data["rmse"], strict=True):
        ax.annotate(
            f"{value:.1f}",
            (xpos, value),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color="#9A621D",
            fontsize=7,
        )

    return fig


def main() -> None:
    data = load_metrics()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = make_figure(data)
    common = {"bbox_inches": "tight", "facecolor": "white", "transparent": False}
    fig.savefig(OUTPUT_DIR / "fig_data_official_lead_error.svg", **common)
    fig.savefig(OUTPUT_DIR / "fig_data_official_lead_error.pdf", **common)
    fig.savefig(OUTPUT_DIR / "fig_data_official_lead_error.jpg", dpi=600, **common)
    fig.savefig(OUTPUT_DIR / "fig_data_official_lead_error.png", dpi=600, **common)
    fig.savefig(OUTPUT_DIR / "fig_data_official_lead_error.tiff", dpi=600, **common)
    plt.close(fig)


if __name__ == "__main__":
    main()
