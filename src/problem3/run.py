"""Reproduce the Q3 forecast/information checkpoint without dispatch optimization."""

from __future__ import annotations

import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.paths import RESULTS_DIR
from src.problem2.forecast_interface import SCENARIO_PATH, audit_q2_handoff_integrity
from src.problem3.forecast_analysis import official_metric_tables
from src.problem3.forecast_data import (
    CANONICAL_PATH,
    RAW_ATTACHMENT3,
    audit_attachment3,
    audit_hourly_to_ten_minute_mapping,
    build_official_canonical,
    file_sha256,
)
from src.problem3.forecast_fusion import METHOD_COLUMNS, build_fusion_analysis
from src.problem3.visualize import generate_forecast_figures


OUTPUT_DIR = RESULTS_DIR / "problem3"
TABLE_DIR = OUTPUT_DIR / "tables"


def _write_paper_tables(by_lead, comparison) -> None:
    lead = by_lead.loc[by_lead["subset"].eq("全部时段")]
    lines = [
        r"\begin{table}[H]", r"\centering", r"\small",
        r"\caption{附件3官方光伏预测的提前期误差}", r"\label{tab:q3-official-lead}",
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"提前期 & 样本数 & MAE/kW & RMSE/kW & 偏差/kW \\", r"\midrule",
    ]
    for row in lead.itertuples(index=False):
        lines.append(
            f"{row.lead_bin} & {int(row.n)} & {row.mae_kw:.2f} & {row.rmse_kw:.2f} & {row.bias_kw:.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (TABLE_DIR / "q3_official_forecast_by_lead.tex").write_text("\n".join(lines), encoding="utf-8")

    selected_methods = ["官方预测", "七日同刻均值", "静态凸融合", "提前期分箱融合", "发布时间分组融合", "在线因果融合"]
    rows = comparison.loc[
        comparison["period"].eq("evaluation")
        & comparison["scope_type"].eq("overall")
        & comparison["method"].isin(selected_methods)
    ].set_index("method").loc[selected_methods].reset_index()
    lines = [
        r"\begin{table}[H]", r"\centering", r"\small",
        r"\caption{问题三候选光伏预测方法的样本外比较}", r"\label{tab:q3-method-comparison}",
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"方法 & 样本数 & MAE/kW & RMSE/kW & 偏差/kW \\", r"\midrule",
    ]
    for row in rows.itertuples(index=False):
        lines.append(
            f"{row.method} & {int(row.n)} & {row.mae_kw:.2f} & {row.rmse_kw:.2f} & {row.bias_kw:.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (TABLE_DIR / "q3_forecast_method_comparison.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    raw_hash_before = file_sha256(RAW_ATTACHMENT3)
    q2_scenario_hash_before = file_sha256(SCENARIO_PATH)
    q2_audit_before = audit_q2_handoff_integrity()

    canonical = build_official_canonical()
    attachment_audit = audit_attachment3(canonical)
    mapping_audit = audit_hourly_to_ten_minute_mapping()
    overall, by_lead, by_release = official_metric_tables(canonical)
    predictions, comparison, significance, weights, selection = build_fusion_analysis(canonical)

    CANONICAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    canonical.to_csv(CANONICAL_PATH, index=False, float_format="%.10f")
    overall.to_csv(TABLE_DIR / "q3_official_forecast_overall.csv", index=False, float_format="%.10f")
    by_lead.to_csv(TABLE_DIR / "q3_official_forecast_by_lead.csv", index=False, float_format="%.10f")
    by_release.to_csv(TABLE_DIR / "q3_official_forecast_by_release.csv", index=False, float_format="%.10f")
    comparison.to_csv(TABLE_DIR / "q3_forecast_method_comparison.csv", index=False, float_format="%.10f")
    significance.to_csv(TABLE_DIR / "q3_forecast_significance.csv", index=False, float_format="%.10g")
    weights.to_csv(TABLE_DIR / "q3_forecast_fusion_weights.csv", index=False, float_format="%.10f")
    prediction_columns = [
        "forecast_id", "issue_time", "target_time", "release_hour", "lead_hours", "lead_bin",
        "actual_pv_kw", "daylight_flag", "period", *dict.fromkeys(METHOD_COLUMNS.values()),
    ]
    predictions[prediction_columns].to_csv(
        TABLE_DIR / "q3_forecast_predictions.csv", index=False, float_format="%.10f"
    )
    (TABLE_DIR / "q3_forecast_selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_paper_tables(by_lead, comparison)

    figures = generate_forecast_figures(by_lead, by_release, comparison)
    raw_hash_after = file_sha256(RAW_ATTACHMENT3)
    q2_scenario_hash_after = file_sha256(SCENARIO_PATH)
    q2_audit_after = audit_q2_handoff_integrity()
    audit = {
        "attachment3": attachment_audit,
        "hourly_to_ten_minute_mapping": mapping_audit,
        "attachment3_raw_sha256_before": raw_hash_before,
        "attachment3_raw_sha256_after": raw_hash_after,
        "attachment3_raw_unchanged": raw_hash_before == raw_hash_after,
        "canonical_sha256": file_sha256(CANONICAL_PATH),
        "method_comparison_sha256": file_sha256(TABLE_DIR / "q3_forecast_method_comparison.csv"),
        "q2_scenario_sha256_before": q2_scenario_hash_before,
        "q2_scenario_sha256_after": q2_scenario_hash_after,
        "q2_scenario_unchanged": q2_scenario_hash_before == q2_scenario_hash_after,
        "stage2a_unchanged": q2_audit_before == q2_audit_after
        and bool(q2_audit_after["protected_artifact_hashes_unchanged"]),
        "strict_causal_protocol": True,
        "economic_dispatch_executed": False,
        "figure_files": [str(path) for path in figures],
    }
    (TABLE_DIR / "q3_forecast_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Q3 forecast/information checkpoint generated; no dispatch optimization was run.")
    print(json.dumps(selection, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
