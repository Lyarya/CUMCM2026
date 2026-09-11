"""Daily paired significance audit for the frozen Q2 PV forecasts.

This module consumes the saved Stage 2A++ out-of-sample predictions. It does
not fit, update, or call any forecasting or dispatch model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from src.common.plotting import CUMCM_PALETTE, configure_plots, style_academic_axes


ROOT = Path(__file__).resolve().parents[2]
PREDICTION_PATH = ROOT / "results/tables/forecasting/forecast_model_final_predictions.csv"
SELECTION_PATH = ROOT / "results/tables/forecasting/forecast_final_leakage_audit.json"
TABLE_DIR = ROOT / "results/problem2/tables"
FIGURE_DIR = ROOT / "results/problem2/figures"

REFERENCE_MODEL = "7-day same-slot mean"
COMPARATORS = (
    "Yesterday",
    "Last Week",
    "SSA best config",
    "DLinear",
    "PhaseStructuralFusion",
)
OOS_START = pd.Timestamp("2025-02-01")
OOS_END = pd.Timestamp("2025-12-31")
ALPHA = 0.05

MODEL_LABELS = {
    REFERENCE_MODEL: "近7日同刻均值",
    "Yesterday": "昨日同刻",
    "Last Week": "上周同刻",
    "SSA best config": "SSA",
    "DLinear": "DLinear",
    "PhaseStructuralFusion": "相位--结构融合",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_predictions(path: Path = PREDICTION_PATH) -> pd.DataFrame:
    """Load and validate the common frozen February--December OOS artifact."""
    frame = pd.read_csv(path, parse_dates=["operating_date", "datetime"])
    ordered = ["operating_date", "datetime", "actual_generation", REFERENCE_MODEL, *COMPARATORS]
    missing = [column for column in ordered if column not in frame]
    if missing:
        raise ValueError(f"saved OOS prediction columns are missing: {missing}")
    frame = frame.loc[frame["operating_date"].between(OOS_START, OOS_END), ordered].copy()
    frame = frame.sort_values(["operating_date", "datetime"], kind="stable").reset_index(drop=True)
    if frame["operating_date"].nunique() != 334:
        raise AssertionError("formal OOS period must contain 334 operating days")
    if not (frame.groupby("operating_date", sort=True).size() == 144).all():
        raise AssertionError("every OOS operating day must contain 144 intervals")
    if frame.duplicated(["operating_date", "datetime"]).any():
        raise AssertionError("saved OOS predictions contain duplicate timestamps")
    numeric = frame[["actual_generation", REFERENCE_MODEL, *COMPARATORS]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise AssertionError("saved OOS predictions contain NaN or Inf")
    expected_dates = pd.Series(pd.date_range(OOS_START, OOS_END, freq="D"))
    observed_dates = frame["operating_date"].drop_duplicates().reset_index(drop=True)
    if not observed_dates.equals(expected_dates):
        raise AssertionError("saved OOS operating dates are not the complete formal period")
    return frame


def daily_mae_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute one MAE value per operating day and model."""
    rows: list[dict[str, object]] = []
    for model in (REFERENCE_MODEL, *COMPARATORS):
        error = (frame[model] - frame["actual_generation"]).abs()
        values = error.groupby(frame["operating_date"], sort=True).mean()
        rows.extend(
            {"date": date.date().isoformat(), "model": model, "daily_mae": float(value)}
            for date, value in values.items()
        )
    return pd.DataFrame(rows).sort_values(["date", "model"], kind="stable").reset_index(drop=True)


def paired_significance(daily: pd.DataFrame) -> pd.DataFrame:
    """Run predeclared two-sided Wilcoxon tests and one BH-FDR correction."""
    wide = daily.pivot(index="date", columns="model", values="daily_mae")
    rows: list[dict[str, object]] = []
    for comparator in COMPARATORS:
        paired = wide[[REFERENCE_MODEL, comparator]].dropna()
        reference = paired[REFERENCE_MODEL].to_numpy(float)
        comparison = paired[comparator].to_numpy(float)
        difference = reference - comparison
        ties = np.isclose(difference, 0.0, rtol=0.0, atol=1e-12)
        if np.all(ties):
            statistic, p_value = 0.0, 1.0
        else:
            result = wilcoxon(
                difference,
                alternative="two-sided",
                zero_method="wilcox",
                correction=False,
                method="auto",
            )
            statistic, p_value = float(result.statistic), float(result.pvalue)
        reference_mean = float(reference.mean())
        comparator_mean = float(comparison.mean())
        rows.append(
            {
                "reference_model": REFERENCE_MODEL,
                "comparator": comparator,
                "n_days": len(paired),
                "reference_mean_mae": reference_mean,
                "comparator_mean_mae": comparator_mean,
                "reference_median_mae": float(np.median(reference)),
                "comparator_median_mae": float(np.median(comparison)),
                "mean_paired_difference": float(difference.mean()),
                "median_paired_difference": float(np.median(difference)),
                "relative_mae_improvement": (reference_mean - comparator_mean) / comparator_mean,
                "reference_daily_win_rate": float(np.mean(difference < -1e-12)),
                "tie_rate": float(np.mean(ties)),
                "wilcoxon_statistic": statistic,
                "raw_p": p_value,
            }
        )
    output = pd.DataFrame(rows)
    rejected, adjusted, _, _ = multipletests(output["raw_p"], alpha=ALPHA, method="fdr_bh")
    output["fdr_adjusted_p"] = adjusted
    output["significant_after_fdr"] = rejected
    return output


def _write_latex_table(results: pd.DataFrame, path: Path) -> None:
    rows = []
    for row in results.itertuples(index=False):
        p_text = f"{row.fdr_adjusted_p:.2e}" if row.fdr_adjusted_p < 0.001 else f"{row.fdr_adjusted_p:.3f}"
        rows.append(
            f"    {MODEL_LABELS[row.comparator]} & {row.n_days:d} & "
            f"{row.reference_mean_mae:.2f} & {row.comparator_mean_mae:.2f} & "
            f"{row.median_paired_difference:.2f} & {100 * row.reference_daily_win_rate:.1f}\\% & "
            f"{p_text} & {'是' if row.significant_after_fdr else '否'} \\\\"
        )
    content = "\n".join(
        [
            r"\begin{table}[H]",
            r"  \centering",
            r"  \caption{正式评价期新能源预测的日级配对比较}",
            r"  \label{tab:q2-forecast-significance}",
            r"  \scriptsize",
            r"  \setlength{\tabcolsep}{3.2pt}",
            r"  \begin{tabular}{lrrrrrrc}",
            r"    \toprule",
            r"    比较模型 & 日数 & 参考MAE & 比较MAE & 差值中位数 & 参考胜率 & FDR $p$ & 显著 \\",
            r"    \midrule",
            *rows,
            r"    \bottomrule",
            r"  \end{tabular}",
            r"  \vspace{2pt}",
            r"  \begin{minipage}{0.97\textwidth}",
            r"    \footnotesize 注：参考模型为近7日同刻均值，MAE 与差值单位均为 kW。差值定义为",
            r"    参考模型日MAE减比较模型日MAE，负值表示参考模型误差更小；显著性为双侧",
            r"    Wilcoxon符号秩检验经Benjamini--Hochberg校正后的结果。",
            r"  \end{minipage}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")


def plot_paired_differences(daily: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    """Plot the predeclared daily MAE differences with zero as the reference."""
    configure_plots()
    mpl.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
    wide = daily.pivot(index="date", columns="model", values="daily_mae")
    data = [(wide[REFERENCE_MODEL] - wide[model]).to_numpy(float) for model in COMPARATORS]
    colors = [
        CUMCM_PALETTE["primary_light"], CUMCM_PALETTE["neutral"], CUMCM_PALETTE["pv"],
        CUMCM_PALETTE["price"], CUMCM_PALETTE["soc"],
    ]
    fig, axis = plt.subplots(figsize=(7.2, 3.7), facecolor="white")
    parts = axis.violinplot(data, showmeans=False, showmedians=True, widths=0.72)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.45)
    for key in ("cmedians", "cbars", "cmins", "cmaxes"):
        parts[key].set_color(CUMCM_PALETTE["observed"])
        parts[key].set_linewidth(0.8)
    axis.axhline(0.0, color=CUMCM_PALETTE["bad"], linestyle="--", linewidth=1.0)
    labels = [MODEL_LABELS[model].replace("--", "—") for model in COMPARATORS]
    axis.set_xticks(np.arange(1, len(labels) + 1), labels, rotation=12, ha="right")
    axis.set_ylabel("参考模型日MAE－比较模型日MAE（kW）")
    axis.set_title("2—12月冻结评价期的日级配对误差差值")
    style_academic_axes(axis)
    axis.text(
        0.01, 0.98, "零线以下表示近7日同刻均值在同一日误差更小",
        transform=axis.transAxes, va="top", fontsize=8, color=CUMCM_PALETTE["observed"],
    )
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    common = {"bbox_inches": "tight", "facecolor": "white", "transparent": False}
    for suffix in ("svg", "pdf"):
        paths[suffix] = output_dir / f"fig_p2_forecast_significance.{suffix}"
        fig.savefig(paths[suffix], **common)
    for suffix in ("png", "jpg", "tiff"):
        paths[suffix] = output_dir / f"fig_p2_forecast_significance.{suffix}"
        fig.savefig(paths[suffix], dpi=600, **common)
    plt.close(fig)
    return paths


def run_significance_analysis() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Generate all frozen-prediction significance artifacts."""
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    if selection.get("selected_generation_forecaster") != REFERENCE_MODEL:
        raise AssertionError("the formal Stage 2A++ PV forecast selection changed")
    if not selection.get("strict_causal_protocol") or not selection.get("formal_model_and_scaler_frozen"):
        raise AssertionError("the saved OOS predictions do not carry the required causal audit")

    frame = load_frozen_predictions()
    daily = daily_mae_table(frame)
    results = paired_significance(daily)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    daily_path = TABLE_DIR / "q2_forecast_daily_mae.csv"
    significance_path = TABLE_DIR / "q2_forecast_significance.csv"
    latex_path = TABLE_DIR / "q2_forecast_significance.tex"
    metadata_path = TABLE_DIR / "q2_forecast_significance_audit.json"
    daily.to_csv(daily_path, index=False, float_format="%.10f")
    results.to_csv(significance_path, index=False, float_format="%.12g")
    _write_latex_table(results, latex_path)
    figure_paths = plot_paired_differences(daily, FIGURE_DIR)

    exclusions = {
        "Polynomial analytical-only teacher": "仅作为解析教师诊断，不是预先指定的可选正式预测器。",
        "StructuralResidualHybrid": "同一复杂候选族中验证表现弱于已纳入的 PhaseStructuralFusion。",
        "Seasonal/Phase": "PhaseStructuralFusion 的已保存预测与其完全一致，避免重复检验同一序列。",
    }
    metadata: dict[str, object] = {
        "source_prediction_file": str(PREDICTION_PATH.relative_to(ROOT)),
        "source_prediction_sha256": _sha256(PREDICTION_PATH),
        "selection_audit_file": str(SELECTION_PATH.relative_to(ROOT)),
        "selection_audit_sha256": _sha256(SELECTION_PATH),
        "reference_model": REFERENCE_MODEL,
        "comparators": list(COMPARATORS),
        "candidate_exclusions": exclusions,
        "oos_date_range": [OOS_START.date().isoformat(), OOS_END.date().isoformat()],
        "formal_days": 334,
        "intervals_per_day": 144,
        "paired_unit": "operating day",
        "metric": "daily MAE over 144 ten-minute intervals",
        "paired_difference": "reference daily MAE minus comparator daily MAE",
        "wilcoxon": {
            "alternative": "two-sided",
            "zero_method": "wilcox (discard exact zero differences)",
            "continuity_correction": False,
            "scipy_method": "auto",
        },
        "multiple_testing": {"method": "Benjamini-Hochberg FDR", "alpha": ALPHA, "family_size": len(results)},
        "forecast_retuning_on_feb_dec_actuals": False,
        "saved_forecasts_only": True,
        "causal_audit_passed": True,
        "outputs": {
            "daily_mae": str(daily_path.relative_to(ROOT)),
            "significance": str(significance_path.relative_to(ROOT)),
            "latex_table": str(latex_path.relative_to(ROOT)),
            "figure": {key: str(value.relative_to(ROOT)) for key, value in figure_paths.items()},
        },
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return daily, results, metadata


if __name__ == "__main__":
    run_significance_analysis()
