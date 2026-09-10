"""Common regression metrics used by baseline models."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _masked_values(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    mask: ArrayLike | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create the observation mask before replacing missing targets by zero."""
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    observed = np.isfinite(true)
    if mask is not None:
        supplied = np.asarray(mask, dtype=bool)
        if supplied.shape != true.shape:
            raise ValueError("mask must have the same shape as y_true")
        observed &= supplied
    if np.any(observed & ~np.isfinite(pred)):
        raise ValueError("y_pred is non-finite at an observed target")
    true_filled = np.where(observed, true, 0.0)
    pred_filled = np.where(observed, pred, 0.0)
    return true_filled[observed], pred_filled[observed]


def masked_mae(y_true: ArrayLike, y_pred: ArrayLike, mask: ArrayLike | None = None) -> float:
    """Mean absolute error over observed targets only."""
    true, pred = _masked_values(y_true, y_pred, mask)
    return float(np.mean(np.abs(true - pred))) if true.size else float("nan")


def masked_rmse(y_true: ArrayLike, y_pred: ArrayLike, mask: ArrayLike | None = None) -> float:
    """Root mean squared error over observed targets only."""
    true, pred = _masked_values(y_true, y_pred, mask)
    return float(np.sqrt(np.mean(np.square(true - pred)))) if true.size else float("nan")


def _normalization_scale(values: np.ndarray, scale: float | None) -> float:
    denominator = float(np.mean(np.abs(values))) if scale is None else float(scale)
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("Normalization scale must be positive and finite")
    return denominator


def masked_nmae(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    mask: ArrayLike | None = None,
    *,
    scale: float | None = None,
) -> float:
    """MAE normalized by ``scale`` or the observed mean absolute target."""
    true, pred = _masked_values(y_true, y_pred, mask)
    if not true.size:
        return float("nan")
    return float(np.mean(np.abs(true - pred)) / _normalization_scale(true, scale))


def masked_nrmse(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    mask: ArrayLike | None = None,
    *,
    scale: float | None = None,
) -> float:
    """RMSE normalized by ``scale`` or the observed mean absolute target."""
    true, pred = _masked_values(y_true, y_pred, mask)
    if not true.size:
        return float("nan")
    rmse = np.sqrt(np.mean(np.square(true - pred)))
    return float(rmse / _normalization_scale(true, scale))


def regression_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """Return MAE, RMSE and R-squared in a serializable mapping."""
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    return {
        "mae": float(mean_absolute_error(true, pred)),
        "rmse": float(mean_squared_error(true, pred) ** 0.5),
        "r2": float(r2_score(true, pred)),
    }
