"""Consistent, publication-oriented plotting and export helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


FIGURE_FORMATS = ("svg", "pdf", "jpg", "png")

# Palette distilled from the user's energy-competition and Lyra figure code.
# Energy variables retain the competition colors; model-focused panels use the
# quieter Lyra blue/teal/orange/purple-grey family.
CUMCM_PALETTE = {
    "primary": "#0B4F8A",
    "primary_light": "#5DA8C9",
    "load": "#0D47A1",
    "pv": "#42A6A4",
    "price": "#E39B34",
    "grid_purchase": "#6BAED6",
    "charge": "#6C9BC4",
    "discharge": "#FF7F0E",
    "soc": "#7567A5",
    "observed": "#37474F",
    "neutral": "#9099A6",
    "neutral_light": "#D8DDE5",
    "grid": "#E8EBF0",
    "good": "#3A9D5D",
    "bad": "#C85A63",
}


def configure_plots() -> None:
    """Apply compact publication-oriented defaults."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Songti SC", "SimSun", "STSong", "Arial Unicode MS"],
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            "font.size": 10,
        }
    )


def style_academic_axes(ax: Axes, *, grid_axis: str | None = "y") -> None:
    """Apply the clean axes style reused from the earlier energy project."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis is not None:
        ax.grid(
            axis=grid_axis,
            color=CUMCM_PALETTE["grid"],
            linestyle="--",
            linewidth=0.6,
            alpha=0.8,
        )
        ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="major", labelsize=9)


def save_figure(
    fig: Figure,
    output_dir: str | Path,
    name: str | None = None,
    *,
    dpi: int = 400,
) -> dict[str, Path]:
    """Save SVG, PDF and JPG versions of a figure.

    ``name`` should follow ``fig_p{problem}_{description}``. For compatibility
    with the original helper, a suffix on ``output_dir`` is treated as a base
    filename and ignored in favour of all three required formats.
    """
    output = Path(output_dir)
    if name is None:
        if not output.suffix:
            raise ValueError("name is required when output_dir is a directory.")
        name = output.stem
        output = output.parent
    if not name.startswith("fig_"):
        raise ValueError("Figure name must start with 'fig_'.")

    output.mkdir(parents=True, exist_ok=True)
    saved = {
        "svg": output / f"{name}.svg",
        "pdf": output / f"{name}.pdf",
        "jpg": output / f"{name}.jpg",
        "png": output / f"{name}.png",
    }
    common = {"bbox_inches": "tight", "facecolor": "white", "transparent": False}
    fig.savefig(saved["svg"], **common)
    fig.savefig(saved["pdf"], **common)
    fig.savefig(saved["jpg"], dpi=max(dpi, 400), **common)
    fig.savefig(saved["png"], dpi=max(dpi, 600), **common)
    return saved


def plot_prediction_comparison(
    actual: Sequence[float],
    predictions: Mapping[str, Sequence[float]],
    *,
    x: Sequence[float] | None = None,
    xlabel: str = "时间步",
    ylabel: str = "数值",
    title: str = "预测结果对比",
) -> tuple[Figure, Axes]:
    """Plot ground truth and any number of forecast series on one axis."""
    actual_values = np.asarray(actual, dtype=float).reshape(-1)
    x_values = np.arange(actual_values.size) if x is None else np.asarray(x)
    if x_values.size != actual_values.size:
        raise ValueError("x and actual must have the same length.")

    fig, ax = plt.subplots(figsize=(11, 4.8), facecolor="white")
    for label, values in predictions.items():
        pred = np.asarray(values, dtype=float).reshape(-1)
        if pred.size != actual_values.size:
            raise ValueError(f"Prediction {label!r} has a different length from actual.")
        ax.plot(x_values, pred, linewidth=1.3, alpha=0.85, label=label, zorder=2)
    ax.plot(
        x_values,
        actual_values,
        color=CUMCM_PALETTE["observed"],
        linewidth=2.1,
        label="实际值",
        zorder=4,
    )
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.legend(loc="best", framealpha=1.0, edgecolor="black")
    style_academic_axes(ax)
    fig.tight_layout()
    return fig, ax


def plot_metric_comparison(
    metrics: Mapping[str, Mapping[str, float]],
    *,
    title: str = "模型评价指标对比",
    ylabel: str = "指标值",
) -> tuple[Figure, Axes]:
    """Create grouped bars for a mapping of model names to metric values."""
    if not metrics:
        raise ValueError("metrics cannot be empty.")
    model_names = list(metrics)
    metric_names = list(next(iter(metrics.values())))
    if not metric_names:
        raise ValueError("Each model must contain at least one metric.")
    if any(set(values) != set(metric_names) for values in metrics.values()):
        raise ValueError("All models must provide the same metric names.")

    x = np.arange(len(model_names), dtype=float)
    width = 0.8 / len(metric_names)
    fig, ax = plt.subplots(figsize=(max(7.0, 1.5 * len(model_names)), 4.8), facecolor="white")
    for index, metric in enumerate(metric_names):
        values = [float(metrics[model][metric]) for model in model_names]
        positions = x + (index - (len(metric_names) - 1) / 2) * width
        bars = ax.bar(positions, values, width, label=metric.upper(), edgecolor="black", linewidth=0.6)
        ax.bar_label(bars, fmt="%.3g", padding=2, fontsize=8)
    ax.set_xticks(x, model_names)
    ax.set(ylabel=ylabel, title=title)
    ax.legend(framealpha=1.0, edgecolor="black")
    style_academic_axes(ax)
    fig.tight_layout()
    return fig, ax
