"""Verify Python figure generation and the paper handoff path."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.plotting import CUMCM_PALETTE, configure_plots

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    configure_plots()
    results_dir = ROOT / "results" / "figures"
    paper_dir = ROOT / "paper" / "figures" / "common"
    results_dir.mkdir(parents=True, exist_ok=True)
    paper_dir.mkdir(parents=True, exist_ok=True)

    x = np.linspace(0, 10, 200)
    y = np.sin(x)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(x, y, color=CUMCM_PALETTE["primary"], linewidth=1.8)
    ax.set(xlabel="横坐标 $x$", ylabel=r"$\sin(x)$", title="Python 与 LaTeX 绘图流程测试")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    for suffix, options in (("pdf", {}), ("png", {"dpi": 300})):
        output = results_dir / f"test_pipeline.{suffix}"
        fig.savefig(output, bbox_inches="tight", **options)
        shutil.copy2(output, paper_dir / output.name)
    plt.close(fig)
    print("Figures generated in results/ and copied to paper/figures/common/.")


if __name__ == "__main__":
    main()
