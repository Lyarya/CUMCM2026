"""Chinese publication figures for the Q3 forecast and economic layers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

from src.common.paths import problem_figures_dir, problem_results_dir
from src.common.plotting import (
    CUMCM_PALETTE,
    configure_plots,
    save_figure,
    style_academic_axes,
)
from src.problem3.forecast_data import LEAD_BIN_LABELS, RELEASE_HOURS


FIGURE_DIR = problem_figures_dir(3)
TABLE_DIR = problem_results_dir(3) / "tables"
SCHEDULE_ORDER = ("S0", "S1", "S2", "S3")


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


def _clean_svg_whitespace(paths: dict[str, Path]) -> dict[str, Path]:
    """Remove Matplotlib's path-line trailing spaces for clean Git diffs."""
    svg_path = paths.get("svg")
    if svg_path is not None:
        text = svg_path.read_text(encoding="utf-8")
        svg_path.write_text(
            "\n".join(line.rstrip() for line in text.splitlines()) + "\n",
            encoding="utf-8",
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


def summarize_economic_results(
    schedule_table: pd.DataFrame,
    voi_table: pd.DataFrame,
    settlement_table: pd.DataFrame,
) -> dict[str, object]:
    """Validate canonical Q3 tables and derive paper-facing quantities."""
    required_schedule = {
        "schedule",
        "release_hours",
        "formal_days",
        "solver_success",
        "settlement_mode",
        "realized_emergency_energy_kwh",
        "realized_total_cost_yuan",
    }
    required_voi = {
        "schedule",
        "realized_total_cost_yuan",
        "voi_vs_s0_yuan",
        "incremental_from_schedule",
        "incremental_voi_yuan",
    }
    required_settlement = {"schedule", "settlement_mode", "realized_total_cost_yuan"}
    for name, table, required in (
        ("schedule comparison", schedule_table, required_schedule),
        ("VOI", voi_table, required_voi),
        ("settlement sensitivity", settlement_table, required_settlement),
    ):
        missing = required.difference(table.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")

    schedules = schedule_table.set_index("schedule").loc[list(SCHEDULE_ORDER)].copy()
    voi = voi_table.set_index("schedule").loc[list(SCHEDULE_ORDER)].copy()
    if schedules.index.has_duplicates or voi.index.has_duplicates:
        raise ValueError("Q3 reporting tables must have one row per schedule")
    if not np.isfinite(
        schedules[["realized_emergency_energy_kwh", "realized_total_cost_yuan"]].to_numpy(float)
    ).all():
        raise ValueError("Q3 schedule metrics contain non-finite values")
    if not np.isfinite(
        voi[["realized_total_cost_yuan", "voi_vs_s0_yuan", "incremental_voi_yuan"]].to_numpy(float)
    ).all():
        raise ValueError("Q3 VOI metrics contain non-finite values")
    if not (schedules["formal_days"].astype(int).eq(334)).all() or not (
        schedules["solver_success"].astype(int).eq(334)
    ).all():
        raise ValueError("Q3 formal reporting requires 334/334 successful days")
    if not schedules["settlement_mode"].eq("SEQUENTIAL_PREVIOUS_COMMITMENT").all():
        raise ValueError("Q3 main schedule table does not use the locked settlement mode")
    np.testing.assert_allclose(
        schedules["realized_total_cost_yuan"].to_numpy(float),
        voi["realized_total_cost_yuan"].to_numpy(float),
        rtol=0.0,
        atol=1e-6,
    )

    release_hours: dict[str, int] = {}
    for schedule, raw_hours in schedules["release_hours"].items():
        release_hours[schedule] = int(str(raw_hours).split(",")[-1])
    if release_hours != {"S0": 0, "S1": 6, "S2": 12, "S3": 18}:
        raise ValueError(f"Unexpected Q3 release schedule: {release_hours}")
    expected_incremental_sources = {"S1": "S0", "S2": "S1", "S3": "S2"}
    for schedule, previous in expected_incremental_sources.items():
        if str(voi.loc[schedule, "incremental_from_schedule"]) != previous:
            raise ValueError(f"{schedule} has an unexpected incremental VOI source")

    costs = schedules["realized_total_cost_yuan"].astype(float)
    emergency = schedules["realized_emergency_energy_kwh"].astype(float)
    best_schedule = str(costs.idxmin())
    savings = float(costs.loc["S0"] - costs.loc[best_schedule])
    savings_pct = 100.0 * savings / float(costs.loc["S0"])
    emergency_reduction_pct = 100.0 * (
        float(emergency.loc["S0"] - emergency.loc[best_schedule])
        / float(emergency.loc["S0"])
    )

    sensitivity_rankings: dict[str, list[str]] = {}
    for mode, rows in settlement_table.groupby("settlement_mode", sort=True):
        sensitivity_rankings[str(mode)] = rows.sort_values("realized_total_cost_yuan")[
            "schedule"
        ].astype(str).tolist()
    ranking_stable = bool(sensitivity_rankings) and len(
        {tuple(order) for order in sensitivity_rankings.values()}
    ) == 1
    main_mode = "SEQUENTIAL_PREVIOUS_COMMITMENT"
    alternate_mode = "ORIGINAL_00_COMMITMENT"
    if {main_mode, alternate_mode}.difference(sensitivity_rankings):
        raise ValueError("Q3 settlement sensitivity is missing a required mode")
    indexed_settlement = settlement_table.set_index(["settlement_mode", "schedule"])
    settlement_differences = {
        schedule: abs(
            float(indexed_settlement.loc[(main_mode, schedule), "realized_total_cost_yuan"])
            - float(indexed_settlement.loc[(alternate_mode, schedule), "realized_total_cost_yuan"])
        )
        for schedule in SCHEDULE_ORDER
    }

    return {
        "costs_yuan": costs.to_dict(),
        "emergency_energy_kwh": emergency.to_dict(),
        "incremental_voi_yuan": voi["incremental_voi_yuan"].astype(float).to_dict(),
        "release_hours": release_hours,
        "best_schedule": best_schedule,
        "savings_vs_s0_yuan": savings,
        "savings_vs_s0_pct": savings_pct,
        "emergency_reduction_vs_s0_pct": emergency_reduction_pct,
        "settlement_rankings": sensitivity_rankings,
        "settlement_ranking_stable": ranking_stable,
        "settlement_max_difference_yuan": max(settlement_differences.values()),
    }


def _latex_number(value: float, decimals: int, *, signed: bool = False) -> str:
    prefix = "+" if signed and value > 0 else ""
    return prefix + f"{value:,.{decimals}f}".replace(",", r"\,")


def write_economic_numbers(summary: dict[str, object]) -> Path:
    """Write data-derived LaTeX macros used by the Q3 narrative."""
    costs = summary["costs_yuan"]
    emergency = summary["emergency_energy_kwh"]
    incremental = summary["incremental_voi_yuan"]
    if not isinstance(costs, dict) or not isinstance(emergency, dict) or not isinstance(incremental, dict):
        raise TypeError("Q3 summary dictionaries are malformed")
    macros = {
        "QThreeSZeroCostYuan": _latex_number(float(costs["S0"]), 6),
        "QThreeSOneCostYuan": _latex_number(float(costs["S1"]), 6),
        "QThreeSTwoCostYuan": _latex_number(float(costs["S2"]), 6),
        "QThreeSThreeCostYuan": _latex_number(float(costs["S3"]), 6),
        "QThreeSavingsYuan": _latex_number(float(summary["savings_vs_s0_yuan"]), 6),
        "QThreeSavingsPercent": _latex_number(float(summary["savings_vs_s0_pct"]), 2),
        "QThreeSZeroEmergencyKWh": _latex_number(float(emergency["S0"]), 6),
        "QThreeSTwoEmergencyKWh": _latex_number(float(emergency["S2"]), 6),
        "QThreeEmergencyReductionPercent": _latex_number(
            float(summary["emergency_reduction_vs_s0_pct"]), 2
        ),
        "QThreeIncrementSixYuan": _latex_number(float(incremental["S1"]), 6, signed=True),
        "QThreeIncrementTwelveYuan": _latex_number(float(incremental["S2"]), 6, signed=True),
        "QThreeIncrementEighteenYuan": _latex_number(float(incremental["S3"]), 6, signed=True),
        "QThreeSettlementMaxDifferenceYuan": _latex_number(
            float(summary["settlement_max_difference_yuan"]), 6
        ),
    }
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    output = TABLE_DIR / "q3_economic_numbers.tex"
    lines = ["% Generated from the canonical Q3 economic result tables; do not edit manually."]
    lines.extend(f"\\providecommand{{\\{name}}}{{{value}}}" for name, value in macros.items())
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def plot_schedule_total_cost(
    schedule_table: pd.DataFrame,
    summary: dict[str, object],
) -> dict[str, Path]:
    """Plot annual realized cost from a true zero baseline."""
    rows = schedule_table.set_index("schedule").loc[list(SCHEDULE_ORDER)]
    values = rows["realized_total_cost_yuan"].to_numpy(float) / 10_000.0
    best_schedule = str(summary["best_schedule"])
    best_index = SCHEDULE_ORDER.index(best_schedule)
    colors = [
        CUMCM_PALETTE["neutral_light"],
        CUMCM_PALETTE["primary_light"],
        CUMCM_PALETTE["good"],
        CUMCM_PALETTE["primary"],
    ]
    figure, axis = plt.subplots(figsize=(7.2, 4.5), facecolor="white")
    x = np.arange(len(SCHEDULE_ORDER), dtype=float)
    bars = axis.bar(x, values, width=0.62, color=colors, edgecolor="white", linewidth=0.8)
    axis.set_xticks(x, SCHEDULE_ORDER)
    axis.set_xlabel("信息更新时间表")
    axis.set_ylabel("年度实际总成本（万元）")
    axis.set_ylim(0.0, float(values.max()) * 1.13)
    axis.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=3, fontsize=9)
    axis.annotate(
        "最低",
        xy=(best_index, values[best_index]),
        xytext=(best_index, values.max() * 1.075),
        ha="center",
        va="bottom",
        color=CUMCM_PALETTE["good"],
        fontsize=10,
        fontweight="bold",
        arrowprops={"arrowstyle": "-|>", "color": CUMCM_PALETTE["good"], "lw": 1.0},
    )
    axis.text(
        0.02,
        0.96,
        (
            f"S0→{best_schedule} 节省 "
            f"{float(summary['savings_vs_s0_yuan']) / 10_000.0:.2f} 万元"
            f"（{float(summary['savings_vs_s0_pct']):.2f}%）"
        ),
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        color=CUMCM_PALETTE["observed"],
    )
    style_academic_axes(axis, grid_axis="y")
    figure.tight_layout()
    paths = _clean_svg_whitespace(
        save_figure(figure, FIGURE_DIR, "fig_p3_schedule_total_cost", dpi=600)
    )
    plt.close(figure)
    return paths


def plot_incremental_voi(
    voi_table: pd.DataFrame,
    summary: dict[str, object],
) -> dict[str, Path]:
    """Plot the marginal value of each added intraday information release."""
    voi = voi_table.set_index("schedule").loc[["S1", "S2", "S3"]]
    values = voi["incremental_voi_yuan"].to_numpy(float) / 10_000.0
    release_hours = summary["release_hours"]
    if not isinstance(release_hours, dict):
        raise TypeError("Q3 release-hour mapping is malformed")
    labels = [f"{int(release_hours[schedule]):02d}:00" for schedule in ("S1", "S2", "S3")]
    colors = [CUMCM_PALETTE["primary_light"], CUMCM_PALETTE["good"], CUMCM_PALETTE["bad"]]
    figure, axis = plt.subplots(figsize=(7.2, 4.4), facecolor="white")
    x = np.arange(3, dtype=float)
    bars = axis.bar(x, values, width=0.58, color=colors, edgecolor="white", linewidth=0.8)
    axis.axhline(0.0, color=CUMCM_PALETTE["observed"], linewidth=1.0, zorder=0)
    axis.set_xticks(x, labels)
    axis.set_xlabel("新增预测发布时间")
    axis.set_ylabel("增量信息价值（万元）")
    lower = min(-1.5, float(values.min()) * 4.0)
    upper = float(values.max()) * 1.25
    axis.set_ylim(lower, upper)
    for bar, value in zip(bars, values, strict=True):
        offset = 4 if value >= 0 else -5
        vertical_alignment = "bottom" if value >= 0 else "top"
        axis.annotate(
            f"{value:+.2f}",
            xy=(bar.get_x() + bar.get_width() / 2.0, value),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va=vertical_alignment,
            fontsize=9.5,
            fontweight="bold" if value < 0 else "normal",
            color=CUMCM_PALETTE["observed"],
        )
    axis.text(
        2.0,
        lower * 0.72,
        "边际价值略为负",
        ha="center",
        va="top",
        fontsize=9,
        color=CUMCM_PALETTE["bad"],
    )
    style_academic_axes(axis, grid_axis="y")
    figure.tight_layout()
    paths = _clean_svg_whitespace(
        save_figure(figure, FIGURE_DIR, "fig_p3_incremental_voi", dpi=600)
    )
    plt.close(figure)
    return paths


def generate_economic_figures(
    schedule_table: pd.DataFrame,
    voi_table: pd.DataFrame,
    settlement_table: pd.DataFrame,
) -> tuple[list[Path], dict[str, object], Path]:
    """Generate the two final Q3 economic figures and paper macros."""
    configure_plots()
    summary = summarize_economic_results(schedule_table, voi_table, settlement_table)
    bundles = [
        plot_schedule_total_cost(schedule_table, summary),
        plot_incremental_voi(voi_table, summary),
    ]
    number_path = write_economic_numbers(summary)
    return [path for bundle in bundles for path in bundle.values()], summary, number_path


def main() -> None:
    schedule_table = pd.read_csv(TABLE_DIR / "q3_schedule_comparison.csv")
    voi_table = pd.read_csv(TABLE_DIR / "q3_voi.csv")
    settlement_table = pd.read_csv(TABLE_DIR / "q3_settlement_mode_sensitivity.csv")
    paths, summary, number_path = generate_economic_figures(
        schedule_table,
        voi_table,
        settlement_table,
    )
    print(f"best_schedule={summary['best_schedule']}")
    print(f"savings_vs_s0_pct={float(summary['savings_vs_s0_pct']):.8f}")
    print(
        "emergency_reduction_vs_s0_pct="
        f"{float(summary['emergency_reduction_vs_s0_pct']):.8f}"
    )
    print(f"numbers={number_path}")
    for path in paths:
        print(path)


__all__ = [
    "generate_economic_figures",
    "generate_forecast_figures",
    "plot_incremental_voi",
    "plot_method_comparison",
    "plot_official_error_by_lead",
    "plot_official_error_by_release",
    "plot_official_vs_fusion_by_lead",
    "plot_schedule_total_cost",
    "summarize_economic_results",
    "write_economic_numbers",
]


if __name__ == "__main__":
    main()
