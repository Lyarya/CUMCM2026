"""Causal Stage 2A+ checkpoint comparing standard SSA with fixed baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .config import Q2ForecastConfig
from .data import DailyPanel, date_indices
from .evaluation import forecast_metrics
from .model import same_slot_mean, seasonal_phase
from .ssa import SSARecurrentForecaster


@dataclass(frozen=True)
class SSACheckpointConfig:
    """Small, interpretable search fixed before formal evaluation."""

    window_days: tuple[int, ...] = (3, 7, 14)
    ranks: tuple[int, ...] = (1, 3, 5)
    embedding_dimension: int = 144
    horizon: int = 144
    meaningful_rmse_reduction: float = 0.01

    def __post_init__(self) -> None:
        if not self.window_days or any(days <= 0 for days in self.window_days):
            raise ValueError("window_days must be positive")
        if not self.ranks or any(rank <= 0 for rank in self.ranks):
            raise ValueError("SSA ranks must be positive")
        if self.embedding_dimension < 2 or self.horizon <= 0:
            raise ValueError("invalid SSA dimensions")
        if not 0 <= self.meaningful_rmse_reduction < 1:
            raise ValueError("meaningful_rmse_reduction must lie in [0, 1)")


@dataclass(frozen=True)
class SSACheckpointRun:
    validation_results: pd.DataFrame
    rank_window_comparison: pd.DataFrame
    out_of_sample_metrics: pd.DataFrame
    daily_comparison: pd.DataFrame
    model_selection: pd.DataFrame
    predictions: pd.DataFrame
    leakage_audit: dict[str, object]
    validation_prediction_map: dict[tuple[int, int], np.ndarray]
    evaluation_prediction_map: dict[tuple[int, int], np.ndarray]


def causal_history(
    daily_values: np.ndarray, day_index: int, *, window_days: int
) -> np.ndarray:
    """Return complete days strictly before the forecast origin."""
    values = np.asarray(daily_values, dtype=float)
    if values.ndim != 2 or day_index < window_days or day_index > len(values):
        raise ValueError("invalid day index or insufficient causal history")
    return values[day_index - window_days : day_index].reshape(-1).copy()


def _metric_bundle(actual: np.ndarray, forecast: np.ndarray) -> dict[str, float | int | str]:
    truth = np.asarray(actual, dtype=float).reshape(-1)
    pred = np.asarray(forecast, dtype=float).reshape(-1)
    full = forecast_metrics(truth, pred)
    active_mask = truth > 0
    active = forecast_metrics(truth[active_mask], pred[active_mask])

    def cv_rmse(metrics: dict[str, float | int], values: np.ndarray) -> tuple[float, str, float]:
        finite = values[np.isfinite(values)]
        denominator = float(finite.mean()) if finite.size else np.nan
        if not np.isfinite(denominator) or abs(denominator) <= 1e-12:
            return np.nan, "mean_actual_near_zero", denominator
        return float(metrics["rmse_kw"]) / denominator, "ok", denominator

    full_cv, full_status, full_mean = cv_rmse(full, truth)
    active_cv, active_status, active_mean = cv_rmse(active, truth[active_mask])
    return {
        "full_count": int(full["count"]),
        "full_mae_kw": float(full["mae_kw"]),
        "full_rmse_kw": float(full["rmse_kw"]),
        "full_bias_kw": float(full["bias_kw"]),
        "full_mean_actual_kw": full_mean,
        "full_cv_rmse": full_cv,
        "full_cv_status": full_status,
        "effective_count": int(active["count"]),
        "effective_mae_kw": float(active["mae_kw"]),
        "effective_rmse_kw": float(active["rmse_kw"]),
        "effective_bias_kw": float(active["bias_kw"]),
        "effective_mean_actual_kw": active_mean,
        "effective_cv_rmse": active_cv,
        "effective_cv_status": active_status,
    }


def _run_grid(
    panel: DailyPanel,
    indices: np.ndarray,
    config: SSACheckpointConfig,
    *,
    scope: str,
) -> tuple[pd.DataFrame, dict[tuple[int, int], np.ndarray]]:
    rows: list[dict[str, object]] = []
    prediction_lists = {
        (window, rank): [] for window in config.window_days for rank in config.ranks
    }
    actual = panel.generation_kw[indices]
    for position, day_index in enumerate(indices):
        for window in config.window_days:
            history = causal_history(panel.generation_kw, int(day_index), window_days=window)
            forecaster = SSARecurrentForecaster(
                horizon=config.horizon,
                embedding_dimension=config.embedding_dimension,
            )
            results = forecaster.forecast_ranks(history, config.ranks)
            for rank, result in results.items():
                prediction_lists[(window, rank)].append(result.forecast)
                row: dict[str, object] = {
                    "scope": scope,
                    "forecast_date": panel.dates[int(day_index)].date().isoformat(),
                    "history_start_date": panel.dates[int(day_index) - window].date().isoformat(),
                    "history_end_date": panel.dates[int(day_index) - 1].date().isoformat(),
                    "history_points": int(window * config.horizon),
                    "window_days": window,
                    "embedding_dimension": config.embedding_dimension,
                    "trajectory_columns": int(window * config.horizon - config.embedding_dimension + 1),
                    "rank": rank,
                    "cumulative_singular_energy": result.cumulative_energy,
                    "mean_centering": True,
                    "standardization": False,
                    "inverse_centering": True,
                    "recurrence_denominator": result.recurrence_denominator,
                    "recurrence_coefficient_norm": result.recurrence_coefficient_norm,
                    "status": result.status,
                    "failure_reason": result.failure_reason,
                }
                if result.status == "ok":
                    row.update(_metric_bundle(actual[position], result.forecast))
                rows.append(row)
    predictions = {
        key: np.stack(values) for key, values in prediction_lists.items()
    }
    return pd.DataFrame(rows), predictions


def _aggregate_validation(
    daily: pd.DataFrame,
    actual: np.ndarray,
    predictions: dict[tuple[int, int], np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    expected_days = int(daily["forecast_date"].nunique())
    for (window, rank), forecast in predictions.items():
        subset = daily.loc[(daily["window_days"] == window) & (daily["rank"] == rank)]
        valid = subset["status"].eq("ok").to_numpy()
        pooled = _metric_bundle(actual[valid], forecast[valid]) if valid.any() else {}
        rows.append(
            {
                "scope": "january_common_origin_validation",
                "window_days": window,
                "rank": rank,
                "expected_days": expected_days,
                "valid_days": int(valid.sum()),
                "invalid_days": int((~valid).sum()),
                "failure_rate": float((~valid).mean()),
                "selection_eligible": bool(valid.all()),
                "mean_cumulative_singular_energy": float(
                    subset["cumulative_singular_energy"].mean()
                ),
                "median_cumulative_singular_energy": float(
                    subset["cumulative_singular_energy"].median()
                ),
                "mean_daily_rmse_kw": float(subset.loc[valid, "full_rmse_kw"].mean()),
                "median_daily_rmse_kw": float(subset.loc[valid, "full_rmse_kw"].median()),
                **pooled,
            }
        )
    return pd.DataFrame(rows).sort_values(["window_days", "rank"], kind="stable")


def select_best_ssa(validation_summary: pd.DataFrame) -> tuple[int, int]:
    """Select only from January common-origin rows with zero numerical failures."""
    required_scope = "january_common_origin_validation"
    if set(validation_summary["scope"]) != {required_scope}:
        raise ValueError("SSA selection may use January validation rows only")
    eligible = validation_summary.loc[validation_summary["selection_eligible"]].dropna(
        subset=["full_rmse_kw"]
    )
    if eligible.empty:
        raise ValueError("no numerically valid SSA candidate is available")
    best = eligible.sort_values(
        ["full_rmse_kw", "window_days", "rank"], kind="stable"
    ).iloc[0]
    return int(best["window_days"]), int(best["rank"])


def paired_daily_comparison(
    dates: pd.DatetimeIndex,
    actual: np.ndarray,
    seven_day_forecast: np.ndarray,
    ssa_forecast: np.ndarray,
) -> pd.DataFrame:
    """Build same-date paired daily errors; time points are not treated as IID."""
    truth = np.asarray(actual, dtype=float)
    baseline = np.asarray(seven_day_forecast, dtype=float)
    structural = np.asarray(ssa_forecast, dtype=float)
    if truth.shape != baseline.shape or truth.shape != structural.shape:
        raise ValueError("paired daily forecasts must have identical shapes")
    if truth.ndim != 2 or len(dates) != truth.shape[0]:
        raise ValueError("dates must align one-to-one with forecast days")
    rows = []
    for index, day in enumerate(dates):
        base = _metric_bundle(truth[index], baseline[index])
        ssa = _metric_bundle(truth[index], structural[index])
        rows.append(
            {
                "date": day.date().isoformat(),
                "seven_day_mae_kw": base["full_mae_kw"],
                "seven_day_rmse_kw": base["full_rmse_kw"],
                "ssa_mae_kw": ssa["full_mae_kw"],
                "ssa_rmse_kw": ssa["full_rmse_kw"],
                "rmse_difference_ssa_minus_7day_kw": ssa["full_rmse_kw"]
                - base["full_rmse_kw"],
                "ssa_wins": bool(ssa["full_rmse_kw"] < base["full_rmse_kw"]),
                "effective_seven_day_mae_kw": base["effective_mae_kw"],
                "effective_seven_day_rmse_kw": base["effective_rmse_kw"],
                "effective_ssa_mae_kw": ssa["effective_mae_kw"],
                "effective_ssa_rmse_kw": ssa["effective_rmse_kw"],
            }
        )
    return pd.DataFrame(rows)


def _oos_metric_table(
    actual: np.ndarray,
    prediction_map: dict[str, np.ndarray],
    metadata: dict[str, dict[str, object]],
) -> pd.DataFrame:
    rows = []
    for model, values in prediction_map.items():
        bundle = _metric_bundle(actual, values)
        for subset, prefix in [("full_series", "full"), ("effective_generation", "effective")]:
            rows.append(
                {
                    "scope": "february_december_walk_forward",
                    "subset": subset,
                    "model": model,
                    **metadata[model],
                    "count": bundle[f"{prefix}_count"],
                    "mae_kw": bundle[f"{prefix}_mae_kw"],
                    "rmse_kw": bundle[f"{prefix}_rmse_kw"],
                    "bias_kw": bundle[f"{prefix}_bias_kw"],
                    "mean_actual_kw": bundle[f"{prefix}_mean_actual_kw"],
                    "cv_rmse": bundle[f"{prefix}_cv_rmse"],
                    "cv_status": bundle[f"{prefix}_cv_status"],
                }
            )
    return pd.DataFrame(rows)


def run_ssa_checkpoint(
    panel: DailyPanel,
    stage2_predictions: pd.DataFrame,
    stage2_decision: dict[str, object],
    config: SSACheckpointConfig,
    base_config: Q2ForecastConfig | None = None,
) -> SSACheckpointRun:
    """Run January-only SSA selection and fixed February--December evaluation."""
    base = Q2ForecastConfig() if base_config is None else base_config
    january = date_indices(panel, base.calibration_start, base.calibration_end)
    common_start = max(config.window_days)
    validation_indices = january[january >= common_start]
    evaluation_indices = date_indices(panel, base.evaluation_start, base.evaluation_end)
    if len(validation_indices) != 31 - common_start or len(evaluation_indices) != 334:
        raise ValueError("unexpected validation or evaluation calendar")

    validation_daily, validation_predictions = _run_grid(
        panel,
        validation_indices,
        config,
        scope="january_common_origin_validation",
    )
    validation_actual = panel.generation_kw[validation_indices]
    validation_summary = _aggregate_validation(
        validation_daily, validation_actual, validation_predictions
    )
    best_window, best_rank = select_best_ssa(validation_summary)

    evaluation_daily, evaluation_predictions = _run_grid(
        panel,
        evaluation_indices,
        config,
        scope="february_december_walk_forward",
    )
    best_ssa = evaluation_predictions[(best_window, best_rank)]
    expected_rows = len(evaluation_indices) * config.horizon
    if len(stage2_predictions) != expected_rows:
        raise ValueError("Stage 2A prediction table does not match the fixed OOS period")
    expected_dates = np.repeat(panel.dates[evaluation_indices].strftime("%Y-%m-%d"), config.horizon)
    if not np.array_equal(stage2_predictions["operating_date"].astype(str), expected_dates):
        raise ValueError("Stage 2A rows do not align with SSA evaluation dates")

    shape = (len(evaluation_indices), config.horizon)
    actual = panel.generation_kw[evaluation_indices]
    stage2_arrays = {
        "Yesterday": stage2_predictions["yesterday_generation"].to_numpy().reshape(shape),
        "Last Week": stage2_predictions["last_week_generation"].to_numpy().reshape(shape),
        "7-day same-slot mean": stage2_predictions["same_slot_7d_mean"].to_numpy().reshape(shape),
        "Seasonal": stage2_predictions["seasonal_pred"].to_numpy().reshape(shape),
        "Polynomial-Vandermonde rank 1": stage2_predictions["lowrank_rank1_pred"].to_numpy().reshape(shape),
        "Polynomial-Vandermonde rank 3": stage2_predictions["lowrank_rank3_pred"].to_numpy().reshape(shape),
        "Polynomial-Vandermonde rank 5": stage2_predictions["lowrank_rank5_pred"].to_numpy().reshape(shape),
        "Current final forecaster": stage2_predictions["forecast_generation"].to_numpy().reshape(shape),
    }
    prediction_map = dict(stage2_arrays)
    metadata: dict[str, dict[str, object]] = {}
    for model in stage2_arrays:
        family = "baseline"
        if model.startswith("Polynomial"):
            family = "polynomial_vandermonde"
        elif model == "Current final forecaster":
            family = "current_final"
        metadata[model] = {
            "model_family": family,
            "window_days": np.nan,
            "rank": np.nan,
            "selected_ssa": False,
        }
    for (window, rank), values in evaluation_predictions.items():
        model = f"SSA window {window}d rank {rank}"
        prediction_map[model] = values
        metadata[model] = {
            "model_family": "ssa_recurrent",
            "window_days": window,
            "rank": rank,
            "selected_ssa": bool((window, rank) == (best_window, best_rank)),
        }
    oos_metrics = _oos_metric_table(actual, prediction_map, metadata)

    seven_day = stage2_arrays["7-day same-slot mean"]
    daily = paired_daily_comparison(
        panel.dates[evaluation_indices], actual, seven_day, best_ssa
    )
    differences = daily["rmse_difference_ssa_minus_7day_kw"].to_numpy(dtype=float)
    statistic, p_value = wilcoxon(
        daily["ssa_rmse_kw"].to_numpy(),
        daily["seven_day_rmse_kw"].to_numpy(),
        alternative="two-sided",
        method="auto",
    )

    best_validation = validation_summary.loc[
        (validation_summary["window_days"] == best_window)
        & (validation_summary["rank"] == best_rank)
    ].iloc[0]
    validation_seven = np.stack(
        [same_slot_mean(panel.generation_kw, int(index)) for index in validation_indices]
    )
    validation_seven_metrics = _metric_bundle(validation_actual, validation_seven)
    best_validation_rmse = float(best_validation["full_rmse_kw"])
    seven_validation_rmse = float(validation_seven_metrics["full_rmse_kw"])
    best_model_name = f"SSA window {best_window}d rank {best_rank}"
    full_oos = oos_metrics.loc[oos_metrics["subset"] == "full_series"]
    best_oos = full_oos.loc[full_oos["model"] == best_model_name].iloc[0]
    seven_oos = full_oos.loc[full_oos["model"] == "7-day same-slot mean"].iloc[0]
    validation_reduction = 1 - best_validation_rmse / seven_validation_rmse
    oos_reduction = 1 - float(best_oos["rmse_kw"]) / float(seven_oos["rmse_kw"])
    best_oos_daily_status = evaluation_daily.loc[
        (evaluation_daily["window_days"] == best_window)
        & (evaluation_daily["rank"] == best_rank),
        "status",
    ]
    stable_validation = bool(best_validation["invalid_days"] == 0)
    stable_oos = bool(best_oos_daily_status.eq("ok").all())
    daily_win_rate = float(daily["ssa_wins"].mean())
    ssa_beats = bool(
        stable_validation
        and stable_oos
        and validation_reduction >= config.meaningful_rmse_reduction
        and oos_reduction >= config.meaningful_rmse_reduction
        and daily_win_rate > 0.5
    )
    selected_forecaster = (
        "Hankel-SSA Structural Expert" if ssa_beats else str(stage2_decision["selected_generation_forecaster"])
    )

    model_selection = pd.DataFrame(
        [
            {
                "selection_period": "2025-01-15/2025-01-31",
                "formal_evaluation_period": "2025-02-01/2025-12-31",
                "selection_rule": "minimum January common-origin pooled full-period RMSE among zero-failure SSA candidates",
                "best_ssa_window_days": best_window,
                "best_ssa_rank": best_rank,
                "best_ssa_validation_rmse_kw": best_validation_rmse,
                "seven_day_validation_rmse_kw": seven_validation_rmse,
                "validation_relative_rmse_reduction": validation_reduction,
                "best_ssa_oos_rmse_kw": float(best_oos["rmse_kw"]),
                "seven_day_oos_rmse_kw": float(seven_oos["rmse_kw"]),
                "oos_relative_rmse_reduction": oos_reduction,
                "meaningful_reduction_threshold": config.meaningful_rmse_reduction,
                "ssa_validation_stable": stable_validation,
                "ssa_oos_stable": stable_oos,
                "mean_daily_rmse_ssa_kw": float(daily["ssa_rmse_kw"].mean()),
                "median_daily_rmse_ssa_kw": float(daily["ssa_rmse_kw"].median()),
                "mean_daily_rmse_7day_kw": float(daily["seven_day_rmse_kw"].mean()),
                "median_daily_rmse_7day_kw": float(daily["seven_day_rmse_kw"].median()),
                "ssa_daily_win_rate": daily_win_rate,
                "daily_rmse_difference_mean_kw": float(differences.mean()),
                "daily_rmse_difference_median_kw": float(np.median(differences)),
                "daily_rmse_difference_q025_kw": float(np.quantile(differences, 0.025)),
                "daily_rmse_difference_q975_kw": float(np.quantile(differences, 0.975)),
                "wilcoxon_statistic": float(statistic),
                "wilcoxon_p_value_two_sided": float(p_value),
                "paired_unit": "forecast_day",
                "ssa_beats_7day": ssa_beats,
                "selected_generation_forecaster": selected_forecaster,
                "regenerate_q2_scenarios": ssa_beats,
                "formal_parameters_fixed": True,
                "online_learning": False,
                "official_forecast_used": False,
            }
        ]
    )

    flat = (-1,)
    prediction_output = pd.DataFrame(
        {
            "operating_date": expected_dates,
            "datetime": panel.datetimes[evaluation_indices].reshape(flat),
            "actual_generation": actual.reshape(flat),
            "seven_day_mean_pred": seven_day.reshape(flat),
            "seasonal_pred": stage2_arrays["Seasonal"].reshape(flat),
            "ssa_pred": best_ssa.reshape(flat),
            "vandermonde_lowrank_pred": stage2_predictions["lowrank_pred"].to_numpy(),
        }
    )
    leakage_audit: dict[str, object] = {
        "calibration_scope": "2025-01-15/2025-01-31 common origins",
        "formal_evaluation_scope": "2025-02-01/2025-12-31",
        "selection_uses_january_only": True,
        "formal_parameters_fixed": True,
        "best_ssa_window_days": best_window,
        "best_ssa_rank": best_rank,
        "history_end_strictly_before_target_date": bool((
            pd.to_datetime(validation_daily["history_end_date"])
            < pd.to_datetime(validation_daily["forecast_date"])
        ).all()) and bool((
            pd.to_datetime(evaluation_daily["history_end_date"])
            < pd.to_datetime(evaluation_daily["forecast_date"])
        ).all()),
        "mean_centering": True,
        "standardization": False,
        "full_year_mean_or_std_used": False,
        "inverse_centering_applied": True,
        "official_forecast_used": False,
        "online_learning": False,
        "paired_comparison_unit": "forecast_day",
        "paired_day_count": int(len(daily)),
        "validation_candidate_day_count": int(len(validation_daily)),
        "validation_invalid_forecast_count": int(validation_daily["status"].ne("ok").sum()),
        "validation_failure_rate": float(validation_daily["status"].ne("ok").mean()),
        "oos_candidate_day_count": int(len(evaluation_daily)),
        "oos_invalid_forecast_count": int(evaluation_daily["status"].ne("ok").sum()),
        "oos_failure_rate": float(evaluation_daily["status"].ne("ok").mean()),
        "minimum_recurrence_denominator": float(
            min(
                validation_daily["recurrence_denominator"].min(),
                evaluation_daily["recurrence_denominator"].min(),
            )
        ),
        "maximum_recurrence_coefficient_norm": float(
            max(
                validation_daily["recurrence_coefficient_norm"].max(),
                evaluation_daily["recurrence_coefficient_norm"].max(),
            )
        ),
        "all_ssa_oos_forecasts_finite": bool(
            all(np.isfinite(values).all() for values in evaluation_predictions.values())
        ),
        "scenarios_regenerated": ssa_beats,
        "current_lowrank_form": "polynomial Vandermonde: V[t,j] = tau_t ** j",
        "inference_boundary": (
            "Poor polynomial-Vandermonde forecasts do not establish that low-rank forecasting has no value."
        ),
    }
    return SSACheckpointRun(
        validation_results=validation_daily,
        rank_window_comparison=validation_summary,
        out_of_sample_metrics=oos_metrics,
        daily_comparison=daily,
        model_selection=model_selection,
        predictions=prediction_output,
        leakage_audit=leakage_audit,
        validation_prediction_map=validation_predictions,
        evaluation_prediction_map=evaluation_predictions,
    )
