"""Reproducible Stage 2A entry point; no dispatch optimization is performed."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.common.paths import PROCESSED_DATA_DIR, RESULTS_DIR, problem_figures_dir

from .config import Q2ForecastConfig
from .data import load_daily_panel
from .rolling import run_q2_forecasting
from .scenarios import build_walk_forward_scenarios
from .visualize import plot_generation_model_comparison, plot_joint_scenario_example


TABLE_DIR = RESULTS_DIR / "tables" / "forecasting"
FIGURE_DIR = problem_figures_dir(2)
CANONICAL_PATH = PROCESSED_DATA_DIR / "C题" / "actual_10min.csv"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    config = Q2ForecastConfig()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    panel = load_daily_panel(CANONICAL_PATH, day_steps=config.day_steps)
    run = run_q2_forecasting(panel, config)

    run.predictions.to_csv(TABLE_DIR / "q2_forecast_predictions.csv", index=False, float_format="%.8f")
    run.load_comparison.to_csv(
        TABLE_DIR / "q2_load_model_comparison.csv", index=False, float_format="%.8f"
    )
    run.generation_comparison.to_csv(
        TABLE_DIR / "q2_generation_model_comparison.csv", index=False, float_format="%.8f"
    )
    run.hankel_comparison.to_csv(
        TABLE_DIR / "hankel_rank_comparison.csv", index=False, float_format="%.8f"
    )
    run.fusion_weights.to_csv(
        TABLE_DIR / "q2_fusion_weights.csv", index=False, float_format="%.8f"
    )
    run.final_metrics.to_csv(
        TABLE_DIR / "q2_forecast_metrics.csv", index=False, float_format="%.8f"
    )
    run.daily_metrics.to_csv(
        TABLE_DIR / "q2_daily_metrics.csv", index=False, float_format="%.8f"
    )

    scenario_archive, scenario_manifest = build_walk_forward_scenarios(
        calibration_dates=run.calibration_dates,
        calibration_load_residuals=run.calibration_load_residuals,
        calibration_generation_residuals=run.calibration_generation_residuals,
        evaluation_dates=run.evaluation_dates,
        forecast_load=run.evaluation_load_forecast,
        forecast_generation=run.evaluation_generation_forecast,
        actual_load=run.evaluation_actual_load,
        actual_generation=run.evaluation_actual_generation,
        scenario_count=config.scenario_count,
        seed=config.seed,
    )
    np.savez_compressed(TABLE_DIR / "q2_scenarios.npz", **scenario_archive)
    scenario_manifest.to_csv(TABLE_DIR / "q2_scenario_manifest.csv", index=False)

    _, figure_summary = plot_generation_model_comparison(
        run.daily_metrics,
        FIGURE_DIR,
        seed=config.seed,
        selected_model=str(run.decision["selected_generation_forecaster"]),
    )
    figure_summary.to_csv(
        TABLE_DIR / "q2_generation_daily_rmse_ci.csv", index=False, float_format="%.8f"
    )
    _, representative_date = plot_joint_scenario_example(
        run.predictions, scenario_archive, FIGURE_DIR
    )

    source_dates = scenario_archive["source_residual_dates"]
    target_dates = scenario_archive["target_dates"][:, None]
    leakage_audit = {
        "canonical_source": str(CANONICAL_PATH.relative_to(CANONICAL_PATH.parents[2])),
        "calibration_period": "2025-01-01/2025-01-31",
        "formal_evaluation_period": "2025-02-01/2025-12-31",
        "formal_prediction_rows": int(len(run.predictions)),
        "formal_evaluation_days": int(len(run.evaluation_dates)),
        "timestamps_per_day": int(
            run.predictions.groupby("operating_date").size().min()
        ),
        "formal_parameters_fixed": bool(run.decision["formal_parameters_fixed"]),
        "online_parameter_learning": bool(run.decision["online_parameter_learning"]),
        "official_forecast_used": bool(run.decision["official_forecast_used"]),
        "scenario_source_dates_strictly_before_target": bool((source_dates < target_dates).all()),
        "scenario_method": "paired_whole_day_residual_resampling",
        "scenario_count_per_day": config.scenario_count,
        "generation_physical_upper_bound": None,
        "generation_upper_bound_note": "No rated capacity is supplied; only the physical lower bound zero is enforced.",
        "representative_scenario_figure_date": representative_date,
        "ready_for_stage2b": True,
    }
    _write_json(TABLE_DIR / "q2_model_decision.json", run.decision)
    _write_json(TABLE_DIR / "q2_leakage_audit.json", leakage_audit)
    print(json.dumps({"decision": run.decision, "leakage_audit": leakage_audit}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
