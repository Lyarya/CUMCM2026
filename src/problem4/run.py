"""Reproduce the Q4 price/information checkpoint without dispatch optimization."""

from __future__ import annotations

import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.common.data_validation import ATTACHMENTS, file_sha256
from src.common.paths import PROJECT_ROOT, RESULTS_DIR
from src.problem2.forecast_interface import SCENARIO_PATH, audit_q2_handoff_integrity
from src.problem4.price_analysis import (
    correlation_statistics,
    intraday_profile,
    monthly_statistics,
    peak_offpeak_statistics,
    price_summary,
    volatility_statistics,
)
from src.problem4.price_data import CANONICAL_PATH, RAW_ATTACHMENT4, audit_attachment4, build_price_canonical
from src.problem4.price_forecast import (
    COMPARISON_PATH,
    PREDICTIONS_PATH,
    SELECTION_PATH,
    build_price_forecasts,
)
from src.problem4.visualize import generate_price_figures


OUTPUT_DIR = RESULTS_DIR / "problem4"
TABLE_DIR = OUTPUT_DIR / "tables"
INFORMATION_CLASSIFICATION = "AMBIGUOUS"
OFFICIAL_EVIDENCE = (
    "问题4：外网的电价也是实时波动的；要求根据附件2、附件3和附件4的电价数据，"
    "在波动电价下重新计算问题2和问题3。题面未说明日初是否可知全天未来电价。"
)


def _protected_hashes() -> dict[str, str]:
    """Protect existing Q2/Q3 source and outputs during this price-only run."""
    roots = [
        PROJECT_ROOT / "src" / "problem2", PROJECT_ROOT / "src" / "problem3",
        RESULTS_DIR / "problem2", RESULTS_DIR / "problem3",
    ]
    paths = [path for root in roots for path in root.rglob("*")
             if path.is_file() and "__pycache__" not in path.parts]
    return {str(path.relative_to(PROJECT_ROOT)): file_sha256(path) for path in sorted(paths)}


def _write_paper_tables(
    summary: pd.DataFrame, correlations: pd.DataFrame,
    volatility: pd.DataFrame, selection: dict[str, object],
) -> None:
    row = summary.iloc[0]
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\caption{附件4动态电价的描述性统计}",
        r"\label{tab:q4-price-summary}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"指标 & 均值 & 标准差 & 最小值 & 第一四分位数 & 中位数 & 第三四分位数 & 最大值 & 样本数 \\",
        r"\midrule",
        (
            f"电价/(元/kWh) & {row.mean_yuan_per_kwh:.3f} & {row.std_yuan_per_kwh:.3f} & "
            f"{row.min_yuan_per_kwh:.3f} & {row.q1_yuan_per_kwh:.3f} & "
            f"{row.median_yuan_per_kwh:.3f} & {row.q3_yuan_per_kwh:.3f} & "
            f"{row.max_yuan_per_kwh:.3f} & {int(row.n)}" + r" \\"
        ),
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table}",
        "",
    ]
    (TABLE_DIR / "q4_price_summary.tex").write_text("\n".join(lines), encoding="utf-8")

    pivot = correlations.pivot(index="variable", columns="method", values="coefficient")
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\caption{动态电价与系统状态的相关性}",
        r"\label{tab:q4-price-correlation}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"系统状态 & Pearson系数 & Spearman系数 \\",
        r"\midrule",
    ]
    for variable in ("实际负荷", "实际光伏", "净负荷"):
        lines.append(
            f"{variable} & {pivot.loc[variable, 'Pearson']:.3f} & {pivot.loc[variable, 'Spearman']:.3f}" + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (TABLE_DIR / "q4_price_correlations.tex").write_text("\n".join(lines), encoding="utf-8")
    overall_volatility = volatility.loc[volatility["period"].eq("全年")].iloc[0]
    macros = {
        "QFourPriceMean": f"{row.mean_yuan_per_kwh:.3f}",
        "QFourPriceStd": f"{row.std_yuan_per_kwh:.3f}",
        "QFourPriceMin": f"{row.min_yuan_per_kwh:.4f}",
        "QFourPriceMax": f"{row.max_yuan_per_kwh:.4f}",
        "QFourDailyRange": f"{overall_volatility.mean_daily_range:.3f}",
        "QFourMaxChange": f"{overall_volatility.maximum_absolute_10min_change:.4f}",
        "QFourLoadPearson": f"{pivot.loc['实际负荷', 'Pearson']:.3f}",
        "QFourLoadSpearman": f"{pivot.loc['实际负荷', 'Spearman']:.3f}",
        "QFourNetPearson": f"{pivot.loc['净负荷', 'Pearson']:.3f}",
        "QFourNetSpearman": f"{pivot.loc['净负荷', 'Spearman']:.3f}",
        "QFourValMAE": f"{selection['validation_mae_yuan_per_kwh']:.4f}",
        "QFourValRMSE": f"{selection['validation_rmse_yuan_per_kwh']:.4f}",
        "QFourOOSMAE": f"{selection['evaluation_mae_yuan_per_kwh']:.4f}",
        "QFourOOSRMSE": f"{selection['evaluation_rmse_yuan_per_kwh']:.4f}",
    }
    (TABLE_DIR / "q4_price_numbers.tex").write_text(
        "\n".join(r"\newcommand{" + "\\" + name + "}{" + value + "}"
                  for name, value in macros.items()) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    protected_before = _protected_hashes()
    stage2a_before = audit_q2_handoff_integrity()
    raw_hashes_before = {path.name: file_sha256(path) for path in ATTACHMENTS.values()}
    q2_scenario_hash_before = file_sha256(SCENARIO_PATH)
    raw4_hash_before = file_sha256(RAW_ATTACHMENT4)

    canonical = build_price_canonical()
    audit = audit_attachment4(canonical)
    summary = price_summary(canonical)
    monthly = monthly_statistics(canonical)
    profile = intraday_profile(canonical)
    peak_offpeak = peak_offpeak_statistics(canonical)
    volatility = volatility_statistics(canonical)
    correlations = correlation_statistics(canonical)
    predictions, comparison, selection = build_price_forecasts(canonical)

    CANONICAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    canonical.to_csv(CANONICAL_PATH, index=False, float_format="%.10f")
    audit_path = TABLE_DIR / "q4_attachment4_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    summary.to_csv(TABLE_DIR / "q4_price_summary.csv", index=False, float_format="%.10f")
    monthly.to_csv(TABLE_DIR / "q4_price_monthly_statistics.csv", index=False, float_format="%.10f")
    profile.to_csv(TABLE_DIR / "q4_price_intraday_profile.csv", index=False, float_format="%.10f")
    peak_offpeak.to_csv(TABLE_DIR / "q4_price_peak_offpeak.csv", index=False, float_format="%.10f")
    volatility.to_csv(TABLE_DIR / "q4_price_volatility.csv", index=False, float_format="%.10f")
    correlations.to_csv(TABLE_DIR / "q4_price_correlations.csv", index=False, float_format="%.12g")
    comparison.to_csv(COMPARISON_PATH, index=False, float_format="%.10f")
    prediction_columns = [
        "operating_date",
        "slot",
        "interval_start",
        "interval_end",
        "price_yuan_per_kwh",
        "previous_day_price_yuan_per_kwh",
        "previous_week_price_yuan_per_kwh",
        "same_slot_7d_mean_price_yuan_per_kwh",
        "causal_ridge_price_yuan_per_kwh",
        "selected_price_forecast_yuan_per_kwh",
        "period",
    ]
    predictions[prediction_columns].to_csv(PREDICTIONS_PATH, index=False, float_format="%.10f")
    SELECTION_PATH.write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    information_audit = {
        "classification": INFORMATION_CLASSIFICATION,
        "official_evidence": OFFICIAL_EVIDENCE,
        "explicit_future_price_availability_statement_found": False,
        "oracle_mode_supported": True,
        "causal_forecast_mode_supported": True,
        "dispatch_executed": False,
    }
    (TABLE_DIR / "q4_price_information_audit.json").write_text(
        json.dumps(information_audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_paper_tables(summary, correlations, volatility, selection)
    figures = generate_price_figures(canonical, profile, correlations, comparison)

    raw_hashes_after = {path.name: file_sha256(path) for path in ATTACHMENTS.values()}
    q2_scenario_hash_after = file_sha256(SCENARIO_PATH)
    protected_after = _protected_hashes()
    stage2a_after = audit_q2_handoff_integrity()
    checkpoint = {
        "attachment4_raw_sha256_before": raw4_hash_before,
        "attachment4_raw_sha256_after": file_sha256(RAW_ATTACHMENT4),
        "attachment4_raw_unchanged": raw4_hash_before == file_sha256(RAW_ATTACHMENT4),
        "all_raw_hashes_before": raw_hashes_before,
        "all_raw_hashes_after": raw_hashes_after,
        "all_raw_unchanged": raw_hashes_before == raw_hashes_after,
        "q2_scenario_sha256_before": q2_scenario_hash_before,
        "q2_scenario_sha256_after": q2_scenario_hash_after,
        "q2_scenario_unchanged": q2_scenario_hash_before == q2_scenario_hash_after,
        "q2_q3_protected_hashes_before": protected_before,
        "q2_q3_protected_hashes_after": protected_after,
        "q2_q3_files_unchanged": protected_before == protected_after,
        "stage2a_unchanged": stage2a_before == stage2a_after
        and bool(stage2a_after["protected_artifact_hashes_unchanged"]),
        "price_information_classification": INFORMATION_CLASSIFICATION,
        "selected_causal_price_forecaster": selection["selected_method"],
        "canonical_sha256": file_sha256(CANONICAL_PATH),
        "forecast_comparison_sha256": file_sha256(COMPARISON_PATH),
        "forecast_predictions_sha256": file_sha256(PREDICTIONS_PATH),
        "figure_files": [str(path) for path in figures],
        "q2_dispatch_modified": False,
        "q3_rolling_dispatch_implemented": False,
        "result4_generated": False,
    }
    (TABLE_DIR / "q4_price_checkpoint.json").write_text(
        json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not all(checkpoint[key] for key in (
        "all_raw_unchanged", "q2_scenario_unchanged",
        "q2_q3_files_unchanged", "stage2a_unchanged",
    )):
        raise AssertionError("An existing protected artifact changed during the Q4 run")
    print("Q4 price/information checkpoint generated; no dispatch optimization was run.")
    print(json.dumps(selection, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
