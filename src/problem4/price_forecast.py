"""Lightweight, strictly chronological price-forecast comparison."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.common.paths import problem_results_dir
from src.common.time_utils import interval_metadata, validate_timestamps


CALIBRATION_START = pd.Timestamp("2025-01-08")
CALIBRATION_END = pd.Timestamp("2025-01-16")
VALIDATION_START = pd.Timestamp("2025-01-17")
VALIDATION_END = pd.Timestamp("2025-01-30")
EVALUATION_START = pd.Timestamp("2025-02-01")
EVALUATION_END = pd.Timestamp("2025-12-31")
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)

TABLE_DIR = problem_results_dir(4) / "tables"
PREDICTIONS_PATH = TABLE_DIR / "q4_price_forecast_predictions.csv"
COMPARISON_PATH = TABLE_DIR / "q4_price_forecast_comparison.csv"
SELECTION_PATH = TABLE_DIR / "q4_price_forecast_selection.json"

METHOD_COLUMNS = {
    "前一日同刻": "previous_day_price_yuan_per_kwh",
    "前一周同刻": "previous_week_price_yuan_per_kwh",
    "近7日同刻均值": "same_slot_7d_mean_price_yuan_per_kwh",
    "因果岭回归": "causal_ridge_price_yuan_per_kwh",
}
SELECTED_OUTPUT_COLUMN = "selected_price_forecast_yuan_per_kwh"
FEATURE_COLUMNS = (
    "previous_day_price_yuan_per_kwh",
    "previous_week_price_yuan_per_kwh",
    "same_slot_7d_mean_price_yuan_per_kwh",
    "slot_sin",
    "slot_cos",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
)


def _add_causal_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.sort_values("interval_end", kind="stable").reset_index(drop=True).copy()
    audit = validate_timestamps(data["interval_end"], frequency="10min")
    if any(audit[key] for key in ("invalid_count", "duplicate_count", "non_frequency_count")):
        raise ValueError("Price lag construction requires a complete, unique 10min grid")
    metadata = interval_metadata(data["interval_end"])
    if not metadata["operating_date"].equals(data["operating_date"]) or not metadata["slot"].equals(data["slot"]):
        raise ValueError("Price operating date/slot metadata does not match timestamps")
    price = data["price_yuan_per_kwh"].astype(float)
    data["previous_day_price_yuan_per_kwh"] = price.shift(144)
    data["previous_week_price_yuan_per_kwh"] = price.shift(1008)
    lagged = np.stack([price.shift(144 * day).to_numpy(float) for day in range(1, 8)])
    complete = np.isfinite(lagged).all(axis=0)
    data["same_slot_7d_mean_price_yuan_per_kwh"] = np.nan
    data.loc[complete, "same_slot_7d_mean_price_yuan_per_kwh"] = lagged[:, complete].mean(axis=0)
    phase = 2 * np.pi * (data["slot"].to_numpy(float) - 1) / 144
    weekday_phase = 2 * np.pi * data["weekday"].to_numpy(float) / 7
    month_phase = 2 * np.pi * (data["month"].to_numpy(float) - 1) / 12
    data["slot_sin"] = np.sin(phase)
    data["slot_cos"] = np.cos(phase)
    data["weekday_sin"] = np.sin(weekday_phase)
    data["weekday_cos"] = np.cos(weekday_phase)
    data["month_sin"] = np.sin(month_phase)
    data["month_cos"] = np.cos(month_phase)
    return data


def _period_mask(data: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
    dates = data["operating_date"].dt.normalize()
    return dates.between(start, end, inclusive="both").to_numpy()


def _metric_row(
    data: pd.DataFrame,
    prediction_column: str,
    *,
    method: str,
    period: str,
    mask: np.ndarray,
) -> dict[str, object]:
    actual = data["price_yuan_per_kwh"].to_numpy(float)
    predicted = data[prediction_column].to_numpy(float)
    valid = mask & np.isfinite(actual) & np.isfinite(predicted)
    error = predicted[valid] - actual[valid]
    return {
        "period": period,
        "method": method,
        "n": int(valid.sum()),
        "mae_yuan_per_kwh": float(np.mean(np.abs(error))),
        "rmse_yuan_per_kwh": float(np.sqrt(np.mean(np.square(error)))),
        "bias_yuan_per_kwh": float(np.mean(error)),
    }


def build_price_forecasts(
    canonical: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Select a lightweight model on January only, then freeze it for Feb--Dec."""

    data = _add_causal_features(canonical)
    calibration = _period_mask(data, CALIBRATION_START, CALIBRATION_END)
    validation = _period_mask(data, VALIDATION_START, VALIDATION_END)
    evaluation = _period_mask(data, EVALUATION_START, EVALUATION_END)
    feature_valid = data[list(FEATURE_COLUMNS)].notna().all(axis=1).to_numpy()
    target_valid = data["price_yuan_per_kwh"].notna().to_numpy()
    train_mask = calibration & feature_valid & target_valid
    validation_mask = validation & feature_valid & target_valid
    if train_mask.sum() != 9 * 144 or validation_mask.sum() != 14 * 144:
        raise AssertionError("Price forecast calibration/validation windows are incomplete")

    x = data.loc[:, FEATURE_COLUMNS].to_numpy(float)
    y = data["price_yuan_per_kwh"].to_numpy(float)
    ridge_trials: list[tuple[float, Pipeline, dict[str, object]]] = []
    for alpha in RIDGE_ALPHAS:
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=float(alpha))),
            ]
        )
        model.fit(x[train_mask], y[train_mask])
        trial_prediction = np.full(len(data), np.nan)
        trial_prediction[feature_valid] = model.predict(x[feature_valid])
        data["_ridge_trial"] = trial_prediction
        metric = _metric_row(
            data,
            "_ridge_trial",
            method=f"因果岭回归(alpha={alpha:g})",
            period="validation",
            mask=validation,
        )
        ridge_trials.append((float(alpha), model, metric))
    selected_alpha, selected_model, _ = min(
        ridge_trials,
        key=lambda item: (
            float(item[2]["rmse_yuan_per_kwh"]),
            float(item[2]["mae_yuan_per_kwh"]),
        ),
    )
    ridge_prediction = np.full(len(data), np.nan)
    ridge_prediction[feature_valid] = selected_model.predict(x[feature_valid])
    data["causal_ridge_price_yuan_per_kwh"] = ridge_prediction
    data = data.drop(columns=["_ridge_trial"])

    comparison_rows: list[dict[str, object]] = []
    for period, mask in (("validation", validation), ("evaluation", evaluation)):
        for method, column in METHOD_COLUMNS.items():
            comparison_rows.append(
                _metric_row(data, column, method=method, period=period, mask=mask)
            )
    comparison = pd.DataFrame(comparison_rows)
    validation_rows = comparison.loc[comparison["period"].eq("validation")]
    winner = validation_rows.sort_values(
        ["rmse_yuan_per_kwh", "mae_yuan_per_kwh"], kind="stable"
    ).iloc[0]
    selected_method = str(winner["method"])
    selected_column = METHOD_COLUMNS[selected_method]
    data[SELECTED_OUTPUT_COLUMN] = data[selected_column]
    evaluation_row = comparison.loc[
        comparison["period"].eq("evaluation") & comparison["method"].eq(selected_method)
    ].iloc[0]
    data["period"] = np.select(
        [calibration, validation, evaluation],
        ["calibration", "validation", "evaluation"],
        default="unused",
    )
    selection: dict[str, object] = {
        "selection_rule": "lowest January validation RMSE; MAE tie-break",
        "calibration_period": [str(CALIBRATION_START.date()), str(CALIBRATION_END.date())],
        "validation_period": [str(VALIDATION_START.date()), str(VALIDATION_END.date())],
        "evaluation_period": [str(EVALUATION_START.date()), str(EVALUATION_END.date())],
        "selected_method": selected_method,
        "selected_output_column": SELECTED_OUTPUT_COLUMN,
        "selected_ridge_alpha": selected_alpha if selected_method == "因果岭回归" else None,
        "feature_columns": list(FEATURE_COLUMNS),
        "all_price_lags_strictly_historical": True,
        "model_frozen_before_evaluation": True,
        "forecast_issue_rule": "operating_date 00:00; completed interval-end observations available at issue",
        "latest_possible_feature_observation": "target_time - 1 day <= forecast issue",
        "intraday_price_assimilation": False,
        "validation_mae_yuan_per_kwh": float(winner["mae_yuan_per_kwh"]),
        "validation_rmse_yuan_per_kwh": float(winner["rmse_yuan_per_kwh"]),
        "evaluation_mae_yuan_per_kwh": float(evaluation_row["mae_yuan_per_kwh"]),
        "evaluation_rmse_yuan_per_kwh": float(evaluation_row["rmse_yuan_per_kwh"]),
    }
    return data, comparison, selection


def load_selection(path: str | Path = SELECTION_PATH) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "CALIBRATION_END",
    "CALIBRATION_START",
    "COMPARISON_PATH",
    "EVALUATION_END",
    "EVALUATION_START",
    "FEATURE_COLUMNS",
    "METHOD_COLUMNS",
    "PREDICTIONS_PATH",
    "RIDGE_ALPHAS",
    "SELECTED_OUTPUT_COLUMN",
    "SELECTION_PATH",
    "VALIDATION_END",
    "VALIDATION_START",
    "build_price_forecasts",
    "load_selection",
]
