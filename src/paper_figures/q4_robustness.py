"""Generate Q4 robustness figures from audited compact result tables only."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/arya-q4-robustness-mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.plotting import CUMCM_PALETTE as PAL, configure_plots


ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = ROOT / "results/problem4/robustness/tables"
FIGURE_DIR = ROOT / "results/problem4/robustness/figures"
AUDIT_PATH = FIGURE_DIR / "q4_robustness_figure_audit.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _setup() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    configure_plots()
    plt.rcParams.update(
        {
            "font.sans-serif": ["Songti SC", "SimSun", "STSong", "Arial Unicode MS"],
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.titlesize": 9,
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def _style(axis, title: str, xlabel: str, ylabel: str) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color=PAL["grid"], linewidth=0.55, linestyle="--")
    axis.set_axisbelow(True)
    axis.set_title(title, loc="left", pad=8)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.margins(x=0.06, y=0.13)


def _plot_pair(axis, x, q42, s2) -> None:
    axis.plot(
        x,
        q42,
        color=PAL["load"],
        linestyle="-",
        marker="o",
        linewidth=1.35,
        markersize=4.2,
        label="Q4-2 日前随机调度",
    )
    axis.plot(
        x,
        s2,
        color=PAL["price"],
        linestyle="--",
        marker="s",
        markerfacecolor="white",
        markeredgewidth=1.0,
        linewidth=1.35,
        markersize=4.5,
        label="Q4-3/S2 滚动调度",
    )


def _save(figure, name: str, sources: list[Path], contract: str) -> dict[str, object]:
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    outside: list[str] = []
    for axis in figure.axes:
        for item in [
            *axis.texts,
            axis.title,
            axis.xaxis.label,
            axis.yaxis.label,
            *axis.get_xticklabels(),
            *axis.get_yticklabels(),
        ]:
            if not item.get_visible() or not item.get_text():
                continue
            box = item.get_window_extent(renderer)
            if (
                box.x0 < -2
                or box.y0 < -2
                or box.x1 > figure.bbox.width + 2
                or box.y1 > figure.bbox.height + 2
            ):
                outside.append(item.get_text())
    if outside:
        raise AssertionError(f"figure text is outside the canvas: {name}: {outside}")
    pdf_path = FIGURE_DIR / f"{name}.pdf"
    svg_path = FIGURE_DIR / f"{name}.svg"
    jpg_path = FIGURE_DIR / f"{name}.jpg"
    png_path = FIGURE_DIR / f"{name}.png"
    figure.savefig(pdf_path, facecolor="white", bbox_inches="tight")
    figure.savefig(svg_path, facecolor="white", bbox_inches="tight")
    # Matplotlib writes harmless trailing spaces in multiline SVG path data.
    # Normalize the text export so repository whitespace validation remains clean.
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    figure.savefig(jpg_path, dpi=600, facecolor="white", bbox_inches="tight")
    figure.savefig(png_path, dpi=600, facecolor="white", bbox_inches="tight")
    paths = {
        "pdf": str(pdf_path.relative_to(ROOT)),
        "svg": str(svg_path.relative_to(ROOT)),
        "jpg": str(jpg_path.relative_to(ROOT)),
        "png": str(png_path.relative_to(ROOT)),
    }
    plt.close(figure)
    return {
        "name": name,
        "paths": paths,
        "source_hashes": {
            str(path.relative_to(ROOT)): _sha256(path) for path in sources
        },
        "contract": contract,
        "error_bars": False,
        "fabricated_data": False,
        "text_outside_canvas": outside,
        "visual_encodings": "color + marker + line style",
    }


def plot_price_volatility(table: pd.DataFrame, source: Path) -> dict[str, object]:
    expected = np.array([0.00, 0.25, 0.50, 0.75, 1.00])
    table = table.sort_values("gamma")
    np.testing.assert_allclose(table["gamma"], expected)
    if not table["audit_status"].eq("PASS").all():
        raise AssertionError("price-volatility figure received a failed case")
    figure, axes = plt.subplots(1, 2, figsize=(7.1, 2.65), layout="constrained")
    x = table["gamma"].to_numpy(float)
    _plot_pair(
        axes[0],
        x,
        table["q42_realized_total_cost_yuan"] / 1e4,
        table["s2_realized_total_cost_yuan"] / 1e4,
    )
    _style(axes[0], "a  年度实际总成本", r"价格波动系数 $\gamma$", "费用（万元）")
    _plot_pair(
        axes[1],
        x,
        table["q42_emergency_energy_kwh"] / 1e4,
        table["s2_emergency_energy_kwh"] / 1e4,
    )
    _style(axes[1], "b  紧急购电量", r"价格波动系数 $\gamma$", "电量（万kWh）")
    for axis in axes:
        axis.set_xticks(x)
        formal = table.loc[np.isclose(table["gamma"], 1.0)].iloc[0]
        value = (
            formal["q42_realized_total_cost_yuan"] / 1e4
            if axis is axes[0]
            else formal["q42_emergency_energy_kwh"] / 1e4
        )
        axis.scatter([1.0], [value], s=48, facecolors="none", edgecolors="black", zorder=5)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False)
    return _save(
        figure,
        "fig_p4_price_volatility_robustness",
        [source],
        "Five controlled gamma cases; lines are deterministic annual outcomes, not means; no error bars.",
    )


def plot_forecast_uncertainty(table: pd.DataFrame, source: Path) -> dict[str, object]:
    expected = np.array([0.75, 1.00, 1.25, 1.50])
    table = table.sort_values("kappa")
    np.testing.assert_allclose(table["kappa"], expected)
    if not table["audit_status"].eq("PASS").all():
        raise AssertionError("uncertainty figure received a failed case")
    figure, axes = plt.subplots(1, 2, figsize=(7.1, 2.65), layout="constrained")
    x = table["kappa"].to_numpy(float)
    _plot_pair(
        axes[0],
        x,
        table["q42_realized_total_cost_yuan"] / 1e4,
        table["s2_realized_total_cost_yuan"] / 1e4,
    )
    _style(axes[0], "a  年度实际总成本", r"残差幅度系数 $\kappa$", "费用（万元）")
    _plot_pair(
        axes[1],
        x,
        table["q42_emergency_energy_kwh"] / 1e4,
        table["s2_emergency_energy_kwh"] / 1e4,
    )
    _style(axes[1], "b  紧急购电量", r"残差幅度系数 $\kappa$", "电量（万kWh）")
    for axis in axes:
        axis.set_xticks(x)
        formal = table.loc[np.isclose(table["kappa"], 1.0)].iloc[0]
        value = (
            formal["q42_realized_total_cost_yuan"] / 1e4
            if axis is axes[0]
            else formal["q42_emergency_energy_kwh"] / 1e4
        )
        axis.scatter([1.0], [value], s=48, facecolors="none", edgecolors="black", zorder=5)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False)
    return _save(
        figure,
        "fig_p4_forecast_uncertainty_robustness",
        [source],
        "Four controlled kappa cases with fixed paired scenario indices; deterministic annual outcomes; no error bars.",
    )


def main() -> None:
    _setup()
    price_path = TABLE_DIR / "q4_price_volatility_sensitivity.csv"
    uncertainty_path = TABLE_DIR / "q4_uncertainty_sensitivity.csv"
    price = pd.read_csv(price_path)
    uncertainty = pd.read_csv(uncertainty_path)
    audits = [
        plot_price_volatility(price, price_path),
        plot_forecast_uncertainty(uncertainty, uncertainty_path),
    ]
    AUDIT_PATH.write_text(
        json.dumps(
            {
                "visual_qa": "PASS",
                "figures": audits,
                "error_bars_used": False,
                "reason_no_error_bars": (
                    "Each parameter point is one controlled full-year trajectory, not a repeated-sample distribution."
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
