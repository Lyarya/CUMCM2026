"""Chinese publication figures for the Q3 forecast/information layer only."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

from src.common.paths import problem_figures_dir
from src.common.plotting import CUMCM_PALETTE, configure_plots, style_academic_axes
from src.problem3.forecast_data import LEAD_BIN_LABELS, RELEASE_HOURS


FIGURE_DIR = problem_figures_dir(3)


def _save_q3_figure(figure, name: str) -> dict[str, Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(
        {
            "font.sans-serif": ["Songti SC", "SimSun", "STSong", "Arial Unicode MS"],
            "font.size": 10,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    paths = {
        "svg": FIGURE_DIR / f"{name}.svg",
        "pdf": FIGURE_DIR / f"{name}.pdf",
        "jpg": FIGURE_DIR / f"{name}.jpg",
        "png": FIGURE_DIR / f"{name}.png",
        "tiff": FIGURE_DIR / f"{name}.tiff",
    }
    figure.savefig(paths["svg"], bbox_inches="tight")
    figure.savefig(paths["pdf"], bbox_inches="tight")
    figure.savefig(paths["jpg"], dpi=600, bbox_inches="tight")
    figure.savefig(paths["png"], dpi=600, bbox_inches="tight")
    figure.savefig(
        paths["tiff"],
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    return paths


def _grouped_metric_bars(table: pd.DataFrame, group_column: str, groups: list[object], name: str, xlabel: str) -> dict[str, Path]:
    figure, axis = plt.subplots(figsize=(7.2, 4.3))
    x = np.arange(len(groups), dtype=float)
    width = 0.34
    mae = [float(table.loc[table[group_column].astype(str).eq(str(group)), "mae_kw"].iloc[0]) for group in groups]
    rmse = [float(table.loc[table[group_column].astype(str).eq(str(group)), "rmse_kw"].iloc[0]) for group in groups]
    axis.bar(x - width / 2, mae, width, color=CUMCM_PALETTE["pv"], label="MAE")
    axis.bar(x + width / 2, rmse, width, color=CUMCM_PALETTE["primary"], label="RMSE")
    axis.set_xticks(x, [str(group) for group in groups])
    axis.set_xlabel(xlabel)
    axis.set_ylabel("预测误差（kW）")
    axis.legend(ncol=2, frameon=False)
    style_academic_axes(axis)
    figure.tight_layout()
    paths = _save_q3_figure(figure, name)
    plt.close(figure)
    return paths


def plot_official_error_by_lead(table: pd.DataFrame) -> dict[str, Path]:
    rows = table.loc[table["subset"].eq("全部时段")]
    return _grouped_metric_bars(rows, "lead_bin", list(LEAD_BIN_LABELS), "fig_p3_official_error_by_lead", "预测提前期")


def plot_official_error_by_release(table: pd.DataFrame) -> dict[str, Path]:
    rows = table.loc[table["subset"].eq("全部时段")].copy()
    rows["release_label"] = rows["release_hour"].map(lambda value: f"{int(value):02d}:00")
    return _grouped_metric_bars(rows, "release_label", [f"{hour:02d}:00" for hour in RELEASE_HOURS], "fig_p3_official_error_by_release", "预报发布时间")


def plot_method_comparison(table: pd.DataFrame) -> dict[str, Path]:
    rows = table.loc[
        table["period"].eq("evaluation")
        & table["scope_type"].eq("overall")
    ].sort_values("rmse_kw")
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    y = np.arange(len(rows))
    axis.barh(y, rows["rmse_kw"], color=CUMCM_PALETTE["primary_light"], label="RMSE")
    axis.scatter(rows["mae_kw"], y, color=CUMCM_PALETTE["discharge"], s=28, zorder=3, label="MAE")
    axis.set_yticks(y, rows["method"])
    axis.invert_yaxis()
    axis.set_xlabel("预测误差（kW）")
    axis.legend(ncol=2, frameon=False, loc="upper right")
    style_academic_axes(axis, grid_axis="x")
    figure.tight_layout()
    paths = _save_q3_figure(figure, "fig_p3_forecast_method_comparison")
    plt.close(figure)
    return paths


def plot_official_vs_fusion_by_lead(table: pd.DataFrame) -> dict[str, Path]:
    rows = table.loc[
        table["period"].eq("evaluation")
        & table["scope_type"].eq("lead_bin")
        & table["method"].isin(["官方预测", "提前期分箱融合"])
    ]
    figure, axis = plt.subplots(figsize=(7.2, 4.3))
    x = np.arange(len(LEAD_BIN_LABELS), dtype=float)
    width = 0.36
    for offset, (method, color) in enumerate(
        (("官方预测", CUMCM_PALETTE["neutral"]), ("提前期分箱融合", CUMCM_PALETTE["primary"]))
    ):
        values = [
            float(rows.loc[rows["method"].eq(method) & rows["scope_value"].eq(label), "rmse_kw"].iloc[0])
            for label in LEAD_BIN_LABELS
        ]
        axis.bar(x + (offset - 0.5) * width, values, width, color=color, label=method)
    axis.set_xticks(x, LEAD_BIN_LABELS)
    axis.set_xlabel("预测提前期")
    axis.set_ylabel("RMSE（kW）")
    axis.legend(ncol=2, frameon=False)
    style_academic_axes(axis)
    figure.tight_layout()
    paths = _save_q3_figure(figure, "fig_p3_official_vs_fusion_by_lead")
    plt.close(figure)
    return paths


def generate_forecast_figures(
    by_lead: pd.DataFrame,
    by_release: pd.DataFrame,
    comparison: pd.DataFrame,
) -> list[Path]:
    configure_plots()
    bundles = [
        plot_official_error_by_lead(by_lead),
        plot_official_error_by_release(by_release),
        plot_method_comparison(comparison),
        plot_official_vs_fusion_by_lead(comparison),
    ]
    return [path for bundle in bundles for path in bundle.values()]


__all__ = [
    "generate_forecast_figures",
    "plot_method_comparison",
    "plot_official_error_by_lead",
    "plot_official_error_by_release",
    "plot_official_vs_fusion_by_lead",
]
