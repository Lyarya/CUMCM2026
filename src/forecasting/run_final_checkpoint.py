"""Artifact-producing entry point for the final forecasting checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.paths import PROCESSED_DATA_DIR, RESULTS_DIR, problem_figures_dir

from .data import load_daily_panel
from .final_checkpoint import FinalCheckpointConfig, run_final_checkpoint, sha256_file
from .visualize_final import plot_daily_differences, plot_final_comparison, plot_typical_day


TABLE_DIR = RESULTS_DIR / "tables" / "forecasting"
FIGURE_DIR = problem_figures_dir(2)
CANONICAL = PROCESSED_DATA_DIR / "C题" / "actual_10min.csv"
PROTECTED = (
    "q2_forecast_predictions.csv", "q2_scenarios.npz", "q2_scenario_manifest.csv",
    "q2_forecast_metrics.csv", "q2_model_decision.json", "q2_leakage_audit.json",
)


def main() -> None:
    before = {name: sha256_file(TABLE_DIR / name) for name in PROTECTED}
    raw_dir = CANONICAL.parents[1] / "raw" / "C题" / "附件"
    # The project paths place raw beside processed under data/.
    raw_dir = CANONICAL.parents[2] / "raw" / "C题" / "附件"
    raw_before = {path.name: sha256_file(path) for path in sorted(raw_dir.glob("*.xlsx"))}

    panel = load_daily_panel(CANONICAL)
    stage2 = pd.read_csv(TABLE_DIR / "q2_forecast_predictions.csv")
    ssa = pd.read_csv(TABLE_DIR / "q2_ssa_predictions.csv")
    run = run_final_checkpoint(panel, stage2, ssa, FinalCheckpointConfig())
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    for name, table in run.tables.items():
        table.to_csv(TABLE_DIR / name, index=False, float_format="%.8f")

    oos = run.predictions["oos"]
    eval_dates = run.metadata["evaluation_dates"]
    prediction_table = pd.DataFrame({
        "operating_date": np.repeat(eval_dates.strftime("%Y-%m-%d"), 144),
        "datetime": panel.datetimes[run.evaluation_indices].reshape(-1),
        "actual_generation": panel.generation_kw[run.evaluation_indices].reshape(-1),
        **{name: values.reshape(-1) for name, values in oos.items()},
    })
    prediction_table.to_csv(TABLE_DIR / "forecast_model_final_predictions.csv", index=False, float_format="%.8f")

    plot_final_comparison(run.tables["forecast_model_final_comparison.csv"], FIGURE_DIR)
    _, representative = plot_typical_day(
        eval_dates, panel.generation_kw[run.evaluation_indices], oos, FIGURE_DIR
    )
    plot_daily_differences(run.tables["forecast_daily_errors.csv"], FIGURE_DIR)

    after = {name: sha256_file(TABLE_DIR / name) for name in PROTECTED}
    raw_after = {path.name: sha256_file(path) for path in sorted(raw_dir.glob("*.xlsx"))}
    scenario_preserved = before == after
    if run.selected_model == "7-day same-slot mean" and not scenario_preserved:
        raise AssertionError("selected baseline is unchanged but protected Stage 2A artifacts changed")
    if run.selected_model != "7-day same-slot mean":
        raise RuntimeError(
            "January selected a new formal model; scenario regeneration is required before this checkpoint can be declared ready"
        )
    audit = {
        "strict_causal_protocol": True,
        "common_validation_period": "2025-01-18/2025-01-31",
        "formal_evaluation_period": "2025-02-01/2025-12-31",
        "formal_model_and_scaler_frozen": True,
        "gradients_disabled_during_formal_evaluation": True,
        "official_attachment3_used": False,
        "protected_artifact_hashes_before": before,
        "protected_artifact_hashes_after": after,
        "protected_artifacts_unchanged": scenario_preserved,
        "raw_excel_hashes_before": raw_before,
        "raw_excel_hashes_after": raw_after,
        "raw_excel_hashes_unchanged": raw_before == raw_after,
        "representative_figure_date": representative,
        "selected_generation_forecaster": run.selected_model,
        "best_hindsight_oos_model": run.best_hindsight_model,
        "generic_fixed_hybrid_reproduces_unpublished_architecture": False,
        "ready_for_stage2b": bool(scenario_preserved and raw_before == raw_after),
    }
    (TABLE_DIR / "forecast_final_leakage_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = run.tables["forecast_selection_summary.csv"].iloc[0].to_dict()
    print(json.dumps({"selection": summary, "audit": audit}, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
