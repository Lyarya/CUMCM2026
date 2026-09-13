"""Create the compact Q2--Q3 perfect-information gap figure."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/arya-theory-mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.common.paths import PAPER_FIGURES_DIR, RESULTS_DIR
from src.common.plotting import CUMCM_PALETTE as PAL, configure_plots


ROOT = Path(__file__).resolve().parents[2]
SUMMARY_PATH = RESULTS_DIR / "analysis" / "theory_strengthening_summary.json"
QA_PATH = RESULTS_DIR / "analysis" / "information_gap_figure_qa.json"
OUTPUT_DIR = PAPER_FIGURES_DIR / "redesign"
OUTPUT_STEM = OUTPUT_DIR / "fig_information_gap"


def save_figure(fig: plt.Figure) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_STEM.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(
        OUTPUT_STEM.with_suffix(".jpg"), dpi=600, bbox_inches="tight", facecolor="white"
    )
    fig.savefig(
        OUTPUT_STEM.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white"
    )
    fig.savefig(
        OUTPUT_STEM.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white"
    )
    svg = OUTPUT_STEM.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")


def main() -> None:
    payload = json.loads(SUMMARY_PATH.read_text())
    costs = payload["cost_comparison"]
    values = np.array(
        [
            costs["perfect_information_cost_yuan"],
            costs["q3_s2_realized_total_cost_yuan"],
            costs["q2_realized_total_cost_yuan"],
        ]
    ) / 1e4
    labels = ["离线全信息基准", r"Q3：$S_2$滚动策略", "Q2：因果日前策略"]
    colors = [PAL["neutral"], PAL["pv"], PAL["load"]]
    markers = ["D", "s", "o"]

    configure_plots()
    plt.rcParams.update(
        {
            "font.sans-serif": ["Songti SC"],
            "axes.unicode_minus": False,
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(7.1, 2.35), layout="constrained")
    y = np.arange(3)
    for yi, value, color, marker in zip(y, values, colors, markers):
        ax.hlines(yi, 0, value, color=color, lw=1.6, alpha=0.75)
        ax.plot(value, yi, marker=marker, color=color, ms=6)
        ax.annotate(
            f"{value:.2f}",
            (value, yi),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=7,
        )
    q2_midpoint = (values[1] + values[2]) / 2
    ax.plot([values[1], values[2]], [2.28, 2.28], color=PAL["primary_light"], lw=1.0)
    ax.vlines([values[1], values[2]], 2.22, 2.34, color=PAL["primary_light"], lw=1.0)
    ax.text(
        q2_midpoint,
        2.40,
        f"Q2→Q3 降低 {costs['q2_to_q3_cost_reduction_yuan']/1e4:.2f} 万元",
        ha="center",
        va="bottom",
        fontsize=7,
        color=PAL["load"],
    )
    remaining_midpoint = (values[0] + values[1]) / 2
    ax.annotate(
        "",
        xy=(values[0], 0.35),
        xytext=(values[1], 0.35),
        arrowprops={"arrowstyle": "<->", "color": PAL["primary_light"], "lw": 1.0},
    )
    ax.text(
        remaining_midpoint,
        0.48,
        f"剩余差距 {costs['q3_remaining_gap_yuan']/1e4:.2f} 万元",
        ha="center",
        va="bottom",
        fontsize=7,
        color=PAL["neutral"],
    )
    upper = float(np.ceil(values.max() / 100.0) * 100.0 + 100.0)
    ax.set(xlim=(0, upper), ylim=(-0.45, 2.65), yticks=y, yticklabels=labels)
    ax.set_xlabel("实际或基准购电费用（万元）")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#E2E6EA", lw=0.5)
    ax.set_axisbelow(True)
    save_figure(fig)
    plt.close(fig)
    QA_PATH.write_text(
        json.dumps(
            {
                "core_conclusion": "Q3 narrows but does not eliminate the observable cost gap to the offline perfect-information benchmark.",
                "archetype": "single-panel horizontal lollipop gap plot",
                "backend": "Python/matplotlib",
                "source_data": str(SUMMARY_PATH.relative_to(ROOT)),
                "annual_error_bars": "none; each point is one audited annual total",
                "axis_origin": 0,
                "outputs": [str(OUTPUT_STEM.with_suffix(ext).relative_to(ROOT)) for ext in (".pdf", ".svg", ".jpg", ".png", ".tiff")],
                "reviewer_risk": "offline benchmark is non-deployable and has finite-horizon terminal optimism",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
