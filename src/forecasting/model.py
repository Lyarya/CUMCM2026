"""Interpretable phase and Hankel low-rank forecasting experts for Q2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import DailyPanel


def yesterday(values: np.ndarray, day_index: int) -> np.ndarray:
    """Forecast a day from the immediately preceding operating day."""
    if day_index < 1:
        raise ValueError("Yesterday requires one complete preceding day")
    return np.asarray(values[day_index - 1], dtype=float).copy()


def last_week(values: np.ndarray, day_index: int) -> np.ndarray:
    """Forecast a day from the same slots seven operating days earlier."""
    if day_index < 7:
        raise ValueError("Last Week requires seven complete preceding days")
    return np.asarray(values[day_index - 7], dtype=float).copy()


def same_slot_mean(values: np.ndarray, day_index: int, *, days: int = 7) -> np.ndarray:
    """Forecast each slot by its mean over the immediately preceding complete days."""
    if day_index < days:
        raise ValueError("same-slot mean has insufficient history")
    return np.asarray(values[day_index - days : day_index], dtype=float).mean(axis=0)


def seasonal_phase(values: np.ndarray, day_index: int, alpha: float) -> np.ndarray:
    """Convexly combine the preceding day and the preceding week."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must lie in [0, 1]")
    return alpha * yesterday(values, day_index) + (1 - alpha) * last_week(values, day_index)


def convex_fusion(seasonal: np.ndarray, low_rank: np.ndarray, seasonal_weight: float) -> np.ndarray:
    """Fuse two experts with fixed non-negative weights that sum to one."""
    if not 0 <= seasonal_weight <= 1:
        raise ValueError("seasonal_weight must lie in [0, 1]")
    return seasonal_weight * np.asarray(seasonal) + (1 - seasonal_weight) * np.asarray(low_rank)


def polynomial_vandermonde(time: np.ndarray, degree: int) -> np.ndarray:
    """Return the increasing polynomial Vandermonde basis V[t,j] = time[t]**j."""
    points = np.asarray(time, dtype=float).reshape(-1)
    if degree < 0:
        raise ValueError("degree cannot be negative")
    return np.vander(points, N=degree + 1, increasing=True)


def diagonal_average(matrix: np.ndarray) -> np.ndarray:
    """Map a Hankel matrix back to a series by anti-diagonal averaging."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    rows, columns = values.shape
    indices = np.arange(rows)[:, None] + np.arange(columns)[None, :]
    return np.bincount(indices.ravel(), weights=values.ravel()) / np.bincount(indices.ravel())


@dataclass(frozen=True)
class HankelLowRankExpert:
    """Hankel-SVD reconstruction followed by polynomial Vandermonde extrapolation.

    The basis matches the inspected reference implementation: powers of a
    normalized time coordinate, not modal powers of fitted eigenvalues.
    """

    lookback: int
    horizon: int
    hankel_rows: int
    polynomial_degree: int = 2
    polynomial_ridge: float = 5.0

    def __post_init__(self) -> None:
        if self.lookback < 4 or self.horizon <= 0:
            raise ValueError("invalid lookback or horizon")
        if not 2 <= self.hankel_rows < self.lookback:
            raise ValueError("hankel_rows must lie in [2, lookback)")
        if self.polynomial_degree < 0 or self.polynomial_ridge < 0:
            raise ValueError("invalid polynomial configuration")

    def _projection_weight(self) -> np.ndarray:
        input_time = np.linspace(-1.0, 1.0, self.lookback)
        output_time = np.linspace(
            1.0 + 2.0 / self.lookback,
            1.0 + 2.0 * self.horizon / self.lookback,
            self.horizon,
        )
        input_basis = polynomial_vandermonde(input_time, self.polynomial_degree)
        output_basis = polynomial_vandermonde(output_time, self.polynomial_degree)
        gram = input_basis.T @ input_basis
        regularized = gram + self.polynomial_ridge * np.eye(self.polynomial_degree + 1)
        coefficients = np.linalg.solve(regularized, input_basis.T)
        return output_basis @ coefficients

    def forecast_ranks(self, history: np.ndarray, ranks: tuple[int, ...]) -> dict[int, np.ndarray]:
        """Forecast all requested ranks from one shared causal history SVD."""
        values = np.asarray(history, dtype=float).reshape(-1)
        if values.size != self.lookback or not np.isfinite(values).all():
            raise ValueError(f"history must contain {self.lookback} finite observations")
        if any(rank <= 0 or rank > self.hankel_rows for rank in ranks):
            raise ValueError("requested rank is outside the supported range")
        mean = float(values.mean())
        centered = values - mean
        trajectory = np.lib.stride_tricks.sliding_window_view(centered, self.hankel_rows).T
        gram = trajectory @ trajectory.T
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        order = np.argsort(eigenvalues)[::-1]
        left = eigenvectors[:, order]
        weight = self._projection_weight()
        forecasts: dict[int, np.ndarray] = {}
        for rank in ranks:
            basis = left[:, :rank]
            reconstructed_matrix = basis @ (basis.T @ trajectory)
            reconstructed = diagonal_average(reconstructed_matrix)
            forecasts[rank] = np.maximum(weight @ reconstructed + mean, 0.0)
        return forecasts


def load_feature_matrix(panel: DailyPanel, day_index: int) -> np.ndarray:
    """Construct phase-aware load features using only d-1 and d-7 observations."""
    if day_index < 7:
        raise ValueError("load features require seven complete preceding days")
    steps = panel.load_kw.shape[1]
    phase = np.arange(steps, dtype=float) / steps
    calendar_day = panel.dates[day_index]
    weekday = np.zeros((steps, 7), dtype=float)
    weekday[:, calendar_day.weekday()] = 1.0
    month = np.zeros((steps, 12), dtype=float)
    month[:, calendar_day.month - 1] = 1.0
    return np.column_stack(
        [
            panel.load_kw[day_index - 1],
            panel.load_kw[day_index - 7],
            phase,
            np.sin(2 * np.pi * phase),
            np.cos(2 * np.pi * phase),
            weekday,
            month,
        ]
    )


def fit_load_ridge(panel: DailyPanel, train_day_indices: np.ndarray, alpha: float) -> Pipeline:
    """Fit one Ridge model; callers control the chronological training cutoff."""
    indices = np.asarray(train_day_indices, dtype=int)
    if indices.size == 0 or indices.min() < 7:
        raise ValueError("Ridge requires at least one train day with seven-day history")
    features = np.vstack([load_feature_matrix(panel, index) for index in indices])
    targets = np.concatenate([panel.load_kw[index] for index in indices])
    estimator = Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=float(alpha))),
        ]
    )
    estimator.fit(features, targets)
    return estimator


def predict_load_ridge(estimator: Pipeline, panel: DailyPanel, day_index: int) -> np.ndarray:
    """Apply a fitted, frozen Ridge model to one forecast origin."""
    return np.maximum(estimator.predict(load_feature_matrix(panel, day_index)), 0.0)
