"""Reproducible entry point for the Stage 2A+ standard-SSA checkpoint."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.common.paths import PROCESSED_DATA_DIR, RESULTS_DIR, problem_figures_dir

from .config import Q2ForecastConfig
from .data import load_daily_panel
from .ssa_checkpoint import SSACheckpointConfig, run_ssa_checkpoint
from .visualize_ssa import (
    plot_ssa_daily_comparison,
    plot_ssa_representative_day,
    plot_ssa_validation,
)


TABLE_DIR = RESULTS_DIR / "tables" / "forecasting"
FIGURE_DIR = problem_figures_dir(2)
CANONICAL_PATH = PROCESSED_DATA_DIR / "C题" / "actual_10min.csv"
PROTECTED_STAGE2A = (
    "q2_forecast_predictions.csv",
    "q2_forecast_metrics.csv",
    "q2_scenario_manifest.csv",
    "q2_scenarios.npz",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    base_config = Q2ForecastConfig()
    ssa_config = SSACheckpointConfig()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    protected_before = {
        name: _sha256(TABLE_DIR / name) for name in PROTECTED_STAGE2A
    }
    panel = load_daily_panel(CANONICAL_PATH, day_steps=base_config.day_steps)
    stage2_predictions = pd.read_csv(
        TABLE_DIR / "q2_forecast_predictions.csv", parse_dates=["datetime"]
    )
    stage2_decision = json.loads(
        (TABLE_DIR / "q2_model_decision.json").read_text(encoding="utf-8")
    )
    run = run_ssa_checkpoint(
        panel,
        stage2_predictions,
        stage2_decision,
        ssa_config,
        base_config,
    )

    run.validation_results.to_csv(
        TABLE_DIR / "ssa_validation_results.csv", index=False, float_format="%.10f"
    )
    run.rank_window_comparison.to_csv(
        TABLE_DIR / "ssa_rank_window_comparison.csv", index=False, float_format="%.10f"
    )
    run.out_of_sample_metrics.to_csv(
        TABLE_DIR / "ssa_out_of_sample_metrics.csv", index=False, float_format="%.10f"
    )
    run.daily_comparison.to_csv(
        TABLE_DIR / "ssa_daily_comparison.csv", index=False, float_format="%.10f"
    )
    run.model_selection.to_csv(
        TABLE_DIR / "ssa_model_selection.csv", index=False, float_format="%.12g"
    )
    run.predictions.to_csv(
        TABLE_DIR / "q2_ssa_predictions.csv", index=False, float_format="%.10f"
    )
    _write_json(TABLE_DIR / "ssa_leakage_audit.json", run.leakage_audit)

    selection = run.model_selection.iloc[0]
    best_window = int(selection["best_ssa_window_days"])
    best_rank = int(selection["best_ssa_rank"])
    plot_ssa_validation(
        run.validation_results,
        FIGURE_DIR,
        selected_window=best_window,
        selected_rank=best_rank,
        seed=base_config.seed,
    )
    plot_ssa_daily_comparison(run.daily_comparison, FIGURE_DIR, seed=base_config.seed)
    _, representative = plot_ssa_representative_day(
        run.predictions, run.daily_comparison, FIGURE_DIR
    )

    scenarios_regenerated = bool(selection["regenerate_q2_scenarios"])
    if scenarios_regenerated:
        raise RuntimeError(
            "SSA met the replacement rule; regenerate the Stage 2A residual scenarios "
            "before accepting this checkpoint. Existing Stage 2A artifacts remain protected."
        )

    protected_after = {
        name: _sha256(TABLE_DIR / name) for name in PROTECTED_STAGE2A
    }
    if protected_before != protected_after:
        raise AssertionError("Stage 2A artifacts changed although SSA was not selected")
    qa = {
        "core_conclusion": (
            "比较奇异谱能量、标准SSA递推误差与周期基线误差，判断低秩压缩是否转化为多步预测优势。"
        ),
        "evidence_chain": [
            "一月共同起点的秩—窗口误差与能量",
            "334个正式预测日的成对误差分布与月度汇总",
            "按成对误差中位数客观选取的代表日曲线",
        ],
        "archetype": "quantitative grid",
        "backend": "python",
        "source_rows": {
            "validation_daily_rows": int(len(run.validation_results)),
            "formal_prediction_rows": int(len(run.predictions)),
            "paired_days": int(len(run.daily_comparison)),
        },
        "exclusions": "none",
        "transformations": "mean centering only inside SSA; no standardization; no plotted downsampling",
        "uncertainty": "daily-unit nonparametric bootstrap, 2000 resamples, seed 2026",
        "representative_day_rule": "closest daily RMSE difference to the median paired difference",
        "representative_day": representative,
        "formats": ["svg", "pdf", "jpg", "png"],
        "language": "Chinese labels throughout",
        "stage2a_artifacts_unchanged": protected_before == protected_after,
    }
    _write_json(FIGURE_DIR / "ssa_figure_qa.json", qa)
    print(
        json.dumps(
            {
                "model_selection": run.model_selection.to_dict(orient="records")[0],
                "leakage_audit": run.leakage_audit,
                "stage2a_artifacts_unchanged": protected_before == protected_after,
                "representative_day": representative,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
