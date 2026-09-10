"""Common regression metrics used by baseline models."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """Return MAE, RMSE and R-squared in a serializable mapping."""
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    return {
        "mae": float(mean_absolute_error(true, pred)),
        "rmse": float(mean_squared_error(true, pred) ** 0.5),
        "r2": float(r2_score(true, pred)),
    }
