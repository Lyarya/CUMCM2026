"""January calibration and frozen-rule February--December walk-forward evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Q2ForecastConfig
from .data import DailyPanel, date_indices
from .evaluation import choose_by_rmse, comparison_table, daily_metric_table, forecast_metrics
from .model import (
    HankelLowRankExpert,
    convex_fusion,
    fit_load_ridge,
    last_week,
    predict_load_ridge,
    same_slot_mean,
    seasonal_phase,
    yesterday,
)


@dataclass(frozen=True)
class ForecastRun:
    """All tabular outputs plus residuals required by the scenario layer."""

    predictions: pd.DataFrame
    load_comparison: pd.DataFrame
    generation_comparison: pd.DataFrame
    hankel_comparison: pd.DataFrame
    fusion_weights: pd.DataFrame
    final_metrics: pd.DataFrame
    daily_metrics: pd.DataFrame
    decision: dict[str, object]
    calibration_dates: pd.DatetimeIndex
    calibration_load_residuals: np.ndarray
    calibration_generation_residuals: np.ndarray
    evaluation_dates: pd.DatetimeIndex
    evaluation_load_forecast: np.ndarray
    evaluation_generation_forecast: np.ndarray
    evaluation_actual_load: np.ndarray
    evaluation_actual_generation: np.ndarray


def _stack(mapping: dict[int, np.ndarray], indices: np.ndarray) -> np.ndarray:
    return np.stack([mapping[int(index)] for index in indices])


def _phase_predictions(values: np.ndarray, indices: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "Yesterday": np.stack([yesterday(values, int(index)) for index in indices]),
        "Last Week": np.stack([last_week(values, int(index)) for index in indices]),
        "7-day mean": np.stack([same_slot_mean(values, int(index)) for index in indices]),
    }


def _calibrate_ridge(
    panel: DailyPanel,
    validation_indices: np.ndarray,
    config: Q2ForecastConfig,
) -> tuple[float, np.ndarray, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    predictions: dict[float, list[np.ndarray]] = {alpha: [] for alpha in config.ridge_alphas}
    for day_index in validation_indices:
        train_indices = np.arange(config.history_days, int(day_index), dtype=int)
        for alpha in config.ridge_alphas:
            estimator = fit_load_ridge(panel, train_indices, alpha)
            predictions[alpha].append(predict_load_ridge(estimator, panel, int(day_index)))
    actual = panel.load_kw[validation_indices]
    for alpha in config.ridge_alphas:
        metrics = forecast_metrics(actual, np.stack(predictions[alpha]))
        rows.append({"ridge_alpha": alpha, **metrics})
    table = pd.DataFrame(rows)
    selected_alpha = float(table.loc[table["rmse_kw"].idxmin(), "ridge_alpha"])
    return selected_alpha, np.stack(predictions[selected_alpha]), table


def _generation_predictions(
    panel: DailyPanel,
    indices: np.ndarray,
    config: Q2ForecastConfig,
) -> tuple[dict[str, np.ndarray], dict[int, np.ndarray]]:
    phase = _phase_predictions(panel.generation_kw, indices)
    expert = HankelLowRankExpert(
        lookback=config.history_steps,
        horizon=config.day_steps,
        hankel_rows=config.hankel_rows,
        polynomial_degree=config.polynomial_degree,
        polynomial_ridge=config.polynomial_ridge,
    )
    rank_lists: dict[int, list[np.ndarray]] = {rank: [] for rank in config.hankel_ranks}
    for day_index in indices:
        history = panel.generation_kw[
            int(day_index) - config.history_days : int(day_index)
        ].reshape(-1)
        forecasts = expert.forecast_ranks(history, config.hankel_ranks)
        for rank, values in forecasts.items():
            rank_lists[rank].append(values)
    rank_predictions = {rank: np.stack(values) for rank, values in rank_lists.items()}
    return phase, rank_predictions


def _select_seasonal_alpha(
    actual: np.ndarray,
    yesterday_pred: np.ndarray,
    last_week_pred: np.ndarray,
    alphas: tuple[float, ...],
) -> tuple[float, pd.DataFrame]:
    rows = []
    for alpha in alphas:
        forecast = alpha * yesterday_pred + (1 - alpha) * last_week_pred
        rows.append({"seasonal_alpha": alpha, **forecast_metrics(actual, forecast)})
    table = pd.DataFrame(rows)
    selected = float(table.loc[table["rmse_kw"].idxmin(), "seasonal_alpha"])
    return selected, table


def _select_fusion_weight(
    actual: np.ndarray,
    seasonal: np.ndarray,
    low_rank: np.ndarray,
    weights: tuple[float, ...],
) -> tuple[float, pd.DataFrame]:
    rows = []
    for weight in weights:
        forecast = convex_fusion(seasonal, low_rank, weight)
        rows.append(
            {
                "seasonal_weight": weight,
                "low_rank_weight": 1 - weight,
                **forecast_metrics(actual, forecast),
            }
        )
    table = pd.DataFrame(rows)
    selected = float(table.loc[table["rmse_kw"].idxmin(), "seasonal_weight"])
    return selected, table


def run_q2_forecasting(panel: DailyPanel, config: Q2ForecastConfig) -> ForecastRun:
    """Run the complete causal protocol without online parameter learning."""
    calibration_all = date_indices(panel, config.calibration_start, config.calibration_end)
    evaluation_indices = date_indices(panel, config.evaluation_start, config.evaluation_end)
    if calibration_all.size != 31 or evaluation_indices.size != 334:
        raise ValueError("expected 31 calibration days and 334 formal evaluation days")
    generation_calibration = calibration_all[calibration_all >= config.history_days]
    load_calibration = calibration_all[calibration_all >= config.history_days + 1]
    if not np.all(panel.dates[evaluation_indices] >= pd.Timestamp(config.evaluation_start)):
        raise AssertionError("formal evaluation begins before February 1")

    selected_ridge_alpha, ridge_calibration, ridge_search = _calibrate_ridge(
        panel, load_calibration, config
    )
    load_calibration_predictions = _phase_predictions(panel.load_kw, load_calibration)
    load_calibration_predictions = {
        "Yesterday": load_calibration_predictions["Yesterday"],
        "Last Week": load_calibration_predictions["Last Week"],
        "Ridge phase-aware": ridge_calibration,
    }
    load_calibration_table = comparison_table(
        panel.load_kw[load_calibration],
        load_calibration_predictions,
        variable="load",
        scope="january_rolling_validation",
    )
    selected_load_model = choose_by_rmse(load_calibration_table)
    ridge_final = fit_load_ridge(panel, calibration_all[calibration_all >= config.history_days], selected_ridge_alpha)

    generation_phase_cal, generation_ranks_cal = _generation_predictions(
        panel, generation_calibration, config
    )
    generation_actual_cal = panel.generation_kw[generation_calibration]
    selected_seasonal_alpha, seasonal_search = _select_seasonal_alpha(
        generation_actual_cal,
        generation_phase_cal["Yesterday"],
        generation_phase_cal["Last Week"],
        config.seasonal_alphas,
    )
    seasonal_calibration = (
        selected_seasonal_alpha * generation_phase_cal["Yesterday"]
        + (1 - selected_seasonal_alpha) * generation_phase_cal["Last Week"]
    )
    rank_rows = []
    for rank, values in generation_ranks_cal.items():
        rank_rows.append(
            {
                "scope": "january_rolling_validation",
                "rank": rank,
                **forecast_metrics(generation_actual_cal, values),
            }
        )
    rank_calibration_table = pd.DataFrame(rank_rows)
    selected_rank = int(rank_calibration_table.loc[rank_calibration_table["rmse_kw"].idxmin(), "rank"])
    selected_lowrank_calibration = generation_ranks_cal[selected_rank]
    selected_seasonal_weight, fusion_search = _select_fusion_weight(
        generation_actual_cal,
        seasonal_calibration,
        selected_lowrank_calibration,
        config.fusion_seasonal_weights,
    )
    fusion_calibration = convex_fusion(
        seasonal_calibration, selected_lowrank_calibration, selected_seasonal_weight
    )
    generation_calibration_predictions = {
        **generation_phase_cal,
        "Seasonal": seasonal_calibration,
        **{
            f"Low-Rank rank {rank}": values
            for rank, values in generation_ranks_cal.items()
        },
        "Seasonal + Low-Rank": fusion_calibration,
    }
    generation_calibration_table = pd.concat(
        [
            comparison_table(
                generation_actual_cal,
                generation_calibration_predictions,
                variable="generation",
                scope="january_rolling_validation",
            ),
            comparison_table(
                generation_actual_cal,
                generation_calibration_predictions,
                variable="generation",
                scope="january_rolling_validation",
                active_only=True,
            ),
        ],
        ignore_index=True,
    )
    generation_full_validation = generation_calibration_table.loc[
        generation_calibration_table["subset"] == "full_series"
    ]
    selected_generation_model = choose_by_rmse(generation_full_validation)

    load_evaluation_phase = _phase_predictions(panel.load_kw, evaluation_indices)
    ridge_evaluation = np.stack(
        [predict_load_ridge(ridge_final, panel, int(index)) for index in evaluation_indices]
    )
    load_evaluation_predictions = {
        "Yesterday": load_evaluation_phase["Yesterday"],
        "Last Week": load_evaluation_phase["Last Week"],
        "Ridge phase-aware": ridge_evaluation,
    }
    selected_load_evaluation = load_evaluation_predictions[selected_load_model]

    generation_phase_eval, generation_ranks_eval = _generation_predictions(
        panel, evaluation_indices, config
    )
    seasonal_evaluation = (
        selected_seasonal_alpha * generation_phase_eval["Yesterday"]
        + (1 - selected_seasonal_alpha) * generation_phase_eval["Last Week"]
    )
    selected_lowrank_evaluation = generation_ranks_eval[selected_rank]
    fusion_evaluation = convex_fusion(
        seasonal_evaluation, selected_lowrank_evaluation, selected_seasonal_weight
    )
    generation_evaluation_predictions = {
        **generation_phase_eval,
        "Seasonal": seasonal_evaluation,
        **{
            f"Low-Rank rank {rank}": values
            for rank, values in generation_ranks_eval.items()
        },
        "Seasonal + Low-Rank": fusion_evaluation,
    }
    selected_generation_evaluation = generation_evaluation_predictions[
        selected_generation_model
    ]

    load_evaluation_table = comparison_table(
        panel.load_kw[evaluation_indices],
        load_evaluation_predictions,
        variable="load",
        scope="february_december_walk_forward",
    )
    generation_evaluation_table = pd.concat(
        [
            comparison_table(
                panel.generation_kw[evaluation_indices],
                generation_evaluation_predictions,
                variable="generation",
                scope="february_december_walk_forward",
            ),
            comparison_table(
                panel.generation_kw[evaluation_indices],
                generation_evaluation_predictions,
                variable="generation",
                scope="february_december_walk_forward",
                active_only=True,
            ),
        ],
        ignore_index=True,
    )
    load_comparison = pd.concat([load_calibration_table, load_evaluation_table], ignore_index=True)
    generation_comparison = pd.concat(
        [generation_calibration_table, generation_evaluation_table], ignore_index=True
    )

    hankel_evaluation_table = generation_evaluation_table.loc[
        (generation_evaluation_table["subset"] == "full_series")
        & generation_evaluation_table["model"].str.startswith("Low-Rank rank")
    ].copy()
    hankel_evaluation_table["rank"] = hankel_evaluation_table["model"].str.extract(r"(\d+)$").astype(int)
    hankel_comparison = pd.concat(
        [rank_calibration_table, hankel_evaluation_table.drop(columns="model")], ignore_index=True
    )

    final_metrics = pd.concat(
        [
            comparison_table(
                panel.load_kw[evaluation_indices],
                {selected_load_model: selected_load_evaluation},
                variable="load",
                scope="february_december_walk_forward",
            ),
            comparison_table(
                panel.generation_kw[evaluation_indices],
                {selected_generation_model: selected_generation_evaluation},
                variable="generation",
                scope="february_december_walk_forward",
            ),
            comparison_table(
                panel.generation_kw[evaluation_indices],
                {selected_generation_model: selected_generation_evaluation},
                variable="generation",
                scope="february_december_walk_forward",
                active_only=True,
            ),
        ],
        ignore_index=True,
    )
    daily_metrics = pd.concat(
        [
            daily_metric_table(
                panel.dates[evaluation_indices],
                panel.load_kw[evaluation_indices],
                load_evaluation_predictions,
                variable="load",
            ),
            daily_metric_table(
                panel.dates[evaluation_indices],
                panel.generation_kw[evaluation_indices],
                generation_evaluation_predictions,
                variable="generation",
            ),
        ],
        ignore_index=True,
    )

    common_calibration = load_calibration
    load_calibration_selected = load_calibration_predictions[selected_load_model]
    generation_position = {int(index): position for position, index in enumerate(generation_calibration)}
    selected_generation_calibration_all = generation_calibration_predictions[
        selected_generation_model
    ]
    generation_calibration_selected = np.stack(
        [
            selected_generation_calibration_all[generation_position[int(index)]]
            for index in common_calibration
        ]
    )
    calibration_load_residuals = panel.load_kw[common_calibration] - load_calibration_selected
    calibration_generation_residuals = (
        panel.generation_kw[common_calibration] - generation_calibration_selected
    )

    flat_shape = (-1,)
    prediction_columns: dict[str, np.ndarray] = {
        "datetime": panel.datetimes[evaluation_indices].reshape(flat_shape),
        "actual_load": panel.load_kw[evaluation_indices].reshape(flat_shape),
        "load_yesterday": load_evaluation_predictions["Yesterday"].reshape(flat_shape),
        "load_last_week": load_evaluation_predictions["Last Week"].reshape(flat_shape),
        "load_ridge": ridge_evaluation.reshape(flat_shape),
        "forecast_load": selected_load_evaluation.reshape(flat_shape),
        "actual_generation": panel.generation_kw[evaluation_indices].reshape(flat_shape),
        "yesterday_generation": generation_phase_eval["Yesterday"].reshape(flat_shape),
        "last_week_generation": generation_phase_eval["Last Week"].reshape(flat_shape),
        "same_slot_7d_mean": generation_phase_eval["7-day mean"].reshape(flat_shape),
        "seasonal_pred": seasonal_evaluation.reshape(flat_shape),
    }
    for rank, values in generation_ranks_eval.items():
        prediction_columns[f"lowrank_rank{rank}_pred"] = values.reshape(flat_shape)
    prediction_columns["lowrank_pred"] = selected_lowrank_evaluation.reshape(flat_shape)
    prediction_columns["forecast_generation"] = selected_generation_evaluation.reshape(flat_shape)
    predictions = pd.DataFrame(prediction_columns)
    predictions.insert(
        0,
        "operating_date",
        np.repeat(panel.dates[evaluation_indices].strftime("%Y-%m-%d"), config.day_steps),
    )

    fusion_weights = pd.DataFrame(
        [
            {
                "calibration_period": "2025-01-01/2025-01-31",
                "formal_evaluation_period": "2025-02-01/2025-12-31",
                "seasonal_alpha": selected_seasonal_alpha,
                "selected_hankel_rank": selected_rank,
                "seasonal_weight": selected_seasonal_weight,
                "low_rank_weight": 1 - selected_seasonal_weight,
                "weights_fixed_during_formal_evaluation": True,
                "online_parameter_learning": False,
            }
        ]
    )
    decision: dict[str, object] = {
        "selected_load_forecaster": selected_load_model,
        "selected_ridge_alpha": selected_ridge_alpha,
        "selected_generation_forecaster": selected_generation_model,
        "selected_seasonal_alpha": selected_seasonal_alpha,
        "selected_hankel_rank": selected_rank,
        "selected_fusion_rule": {
            "seasonal_weight": selected_seasonal_weight,
            "low_rank_weight": 1 - selected_seasonal_weight,
        },
        "vandermonde_form": "polynomial: V[t,j] = tau_t ** j",
        "calibration_load_days": int(len(load_calibration)),
        "calibration_generation_days": int(len(generation_calibration)),
        "formal_evaluation_days": int(len(evaluation_indices)),
        "formal_parameters_fixed": True,
        "online_parameter_learning": False,
        "official_forecast_used": False,
        "ridge_search": ridge_search.to_dict(orient="records"),
        "seasonal_search": seasonal_search.to_dict(orient="records"),
        "fusion_search": fusion_search.to_dict(orient="records"),
    }
    return ForecastRun(
        predictions=predictions,
        load_comparison=load_comparison,
        generation_comparison=generation_comparison,
        hankel_comparison=hankel_comparison,
        fusion_weights=fusion_weights,
        final_metrics=final_metrics,
        daily_metrics=daily_metrics,
        decision=decision,
        calibration_dates=panel.dates[common_calibration],
        calibration_load_residuals=calibration_load_residuals,
        calibration_generation_residuals=calibration_generation_residuals,
        evaluation_dates=panel.dates[evaluation_indices],
        evaluation_load_forecast=selected_load_evaluation,
        evaluation_generation_forecast=selected_generation_evaluation,
        evaluation_actual_load=panel.load_kw[evaluation_indices],
        evaluation_actual_generation=panel.generation_kw[evaluation_indices],
    )
