"""Forecast metrics and deterministic model selection helpers."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def forecast_metrics(actual: np.ndarray, forecast: np.ndarray) -> dict[str, float | int]:
    """Return count, MAE, RMSE and forecast-minus-actual bias."""
    truth = np.asarray(actual, dtype=float).reshape(-1)
    pred = np.asarray(forecast, dtype=float).reshape(-1)
    valid = np.isfinite(truth) & np.isfinite(pred)
    if not valid.any():
        return {"count": 0, "mae_kw": np.nan, "rmse_kw": np.nan, "bias_kw": np.nan}
    error = pred[valid] - truth[valid]
    return {
        "count": int(valid.sum()),
        "mae_kw": float(np.mean(np.abs(error))),
        "rmse_kw": float(np.sqrt(np.mean(np.square(error)))),
        "bias_kw": float(np.mean(error)),
    }


def comparison_table(
    actual: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    *,
    variable: str,
    scope: str,
    active_only: bool = False,
) -> pd.DataFrame:
    """Evaluate multiple forecasts on one common, explicitly defined sample."""
    truth = np.asarray(actual, dtype=float)
    mask = truth > 0 if active_only else np.ones_like(truth, dtype=bool)
    rows = []
    for model, values in predictions.items():
        metrics = forecast_metrics(truth[mask], np.asarray(values, dtype=float)[mask])
        rows.append(
            {
                "scope": scope,
                "variable": variable,
                "subset": "active_generation" if active_only else "full_series",
                "model": model,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def choose_by_rmse(table: pd.DataFrame) -> str:
    """Select the smallest-RMSE model with stable row-order tie breaking."""
    valid = table.dropna(subset=["rmse_kw"])
    if valid.empty:
        raise ValueError("no finite validation RMSE is available")
    return str(valid.loc[valid["rmse_kw"].idxmin(), "model"])


def daily_metric_table(
    dates: pd.DatetimeIndex,
    actual: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    *,
    variable: str,
) -> pd.DataFrame:
    """Return one full-day metric row per date and model."""
    rows: list[dict[str, object]] = []
    for day_index, day in enumerate(dates):
        for model, values in predictions.items():
            rows.append(
                {
                    "date": day.date().isoformat(),
                    "variable": variable,
                    "model": model,
                    **forecast_metrics(actual[day_index], values[day_index]),
                }
            )
    return pd.DataFrame(rows)
