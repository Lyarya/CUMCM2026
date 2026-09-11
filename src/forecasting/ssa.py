"""Standard singular-spectrum-analysis recurrent forecasting.

This module deliberately contains no polynomial or modal Vandermonde basis,
neural residual learner, gating, online adaptation, or dataset-specific scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import svd
from scipy.sparse.linalg import svds


class SSANumericalError(RuntimeError):
    """Raised when a fitted SSA recurrence is numerically unsafe."""


@dataclass(frozen=True)
class SSAForecastResult:
    """One rank-specific forecast and its stability diagnostics."""

    forecast: np.ndarray
    cumulative_energy: float
    recurrence_denominator: float
    recurrence_coefficient_norm: float
    status: str
    failure_reason: str


def trajectory_matrix(series: np.ndarray, embedding_dimension: int) -> np.ndarray:
    """Return the SSA trajectory matrix with lagged segments as columns."""
    values = np.asarray(series, dtype=float).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("SSA history contains NaN or Inf")
    if not 2 <= embedding_dimension < values.size:
        raise ValueError("embedding_dimension must lie in [2, history length)")
    return np.lib.stride_tricks.sliding_window_view(values, embedding_dimension).T.copy()


def diagonal_average(matrix: np.ndarray) -> np.ndarray:
    """Hankelize a two-dimensional matrix by anti-diagonal averaging."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or min(values.shape) < 1:
        raise ValueError("matrix must be a non-empty two-dimensional array")
    row, column = np.indices(values.shape)
    anti_diagonal = (row + column).reshape(-1)
    totals = np.bincount(anti_diagonal, weights=values.reshape(-1))
    counts = np.bincount(anti_diagonal)
    return totals / counts


@dataclass(frozen=True)
class SSARecurrentForecaster:
    """Mean-centered standard SSA with recurrent multi-step forecasting."""

    horizon: int = 144
    embedding_dimension: int = 144
    denominator_tolerance: float = 1e-8
    explosion_factor: float = 100.0

    def __post_init__(self) -> None:
        if self.horizon <= 0 or self.embedding_dimension < 2:
            raise ValueError("horizon must be positive and embedding_dimension at least two")
        if self.denominator_tolerance <= 0 or self.explosion_factor <= 1:
            raise ValueError("invalid numerical-stability thresholds")

    def _forecast_one_rank(
        self,
        *,
        centered_history: np.ndarray,
        trajectory: np.ndarray,
        left_vectors: np.ndarray,
        eigenvalues: np.ndarray,
        total_energy: float,
        mean: float,
        rank: int,
    ) -> SSAForecastResult:
        signal_subspace = left_vectors[:, :rank]
        retained = float(np.maximum(eigenvalues[:rank], 0.0).sum())
        cumulative_energy = retained / total_energy if total_energy > 0 else 1.0

        last_components = signal_subspace[-1, :]
        denominator = float(1.0 - np.dot(last_components, last_components))
        if not np.isfinite(denominator) or denominator <= self.denominator_tolerance:
            raise SSANumericalError(
                f"recurrence denominator {denominator!r} is not safely positive"
            )
        coefficients = signal_subspace[:-1, :] @ last_components / denominator
        coefficient_norm = float(np.linalg.norm(coefficients))
        if not np.isfinite(coefficients).all() or not np.isfinite(coefficient_norm):
            raise SSANumericalError("recurrence coefficients contain NaN or Inf")

        reconstructed_matrix = signal_subspace @ (signal_subspace.T @ trajectory)
        reconstructed = diagonal_average(reconstructed_matrix)
        if not np.isfinite(reconstructed).all():
            raise SSANumericalError("SSA reconstruction contains NaN or Inf")

        state = reconstructed[-(self.embedding_dimension - 1) :].astype(float).tolist()
        reference_scale = max(float(np.max(np.abs(centered_history))), 1.0)
        explosion_limit = self.explosion_factor * reference_scale
        future: list[float] = []
        for _ in range(self.horizon):
            next_value = float(coefficients @ np.asarray(state[-len(coefficients) :]))
            if not np.isfinite(next_value):
                raise SSANumericalError("SSA recurrence produced NaN or Inf")
            if abs(next_value) > explosion_limit:
                raise SSANumericalError(
                    f"SSA recurrence exceeded the declared explosion limit {explosion_limit:.6g}"
                )
            state.append(next_value)
            future.append(next_value)

        raw_forecast = np.asarray(future, dtype=float) + mean
        if not np.isfinite(raw_forecast).all():
            raise SSANumericalError("inverse centering produced NaN or Inf")
        # Generation cannot be negative; no unsupported upper capacity is imposed.
        forecast = np.maximum(raw_forecast, 0.0)
        return SSAForecastResult(
            forecast=forecast,
            cumulative_energy=float(cumulative_energy),
            recurrence_denominator=denominator,
            recurrence_coefficient_norm=coefficient_norm,
            status="ok",
            failure_reason="",
        )

    def forecast_ranks(
        self, history: np.ndarray, ranks: tuple[int, ...]
    ) -> dict[int, SSAForecastResult]:
        """Fit one causal SSA decomposition and forecast every requested rank."""
        values = np.asarray(history, dtype=float).reshape(-1)
        if not np.isfinite(values).all():
            raise ValueError("SSA history contains NaN or Inf")
        if values.size <= self.embedding_dimension:
            raise ValueError("SSA history is too short for the embedding dimension")
        if not ranks or len(set(ranks)) != len(ranks):
            raise ValueError("ranks must be a non-empty tuple without duplicates")

        trajectory_columns = values.size - self.embedding_dimension + 1
        legal_rank = min(self.embedding_dimension, trajectory_columns)
        if any(rank <= 0 or rank > legal_rank for rank in ranks):
            raise ValueError(f"SSA rank must lie in [1, {legal_rank}]")

        mean = float(values.mean())
        centered = values - mean
        trajectory = trajectory_matrix(centered, self.embedding_dimension)
        max_rank = max(ranks)
        if max_rank < min(trajectory.shape):
            left_vectors, singular_values, _ = svds(
                trajectory,
                k=max_rank,
                which="LM",
                solver="arpack",
                v0=np.ones(min(trajectory.shape), dtype=float),
                return_singular_vectors=True,
            )
        else:
            left_vectors, singular_values, _ = svd(
                trajectory, full_matrices=False, check_finite=False, lapack_driver="gesdd"
            )
            left_vectors = left_vectors[:, :max_rank]
            singular_values = singular_values[:max_rank]
        order = np.argsort(singular_values)[::-1]
        singular_values = np.maximum(singular_values[order], 0.0)
        left_vectors = left_vectors[:, order]
        eigenvalues = np.square(singular_values)
        total_energy = float(np.square(trajectory).sum())

        results: dict[int, SSAForecastResult] = {}
        for rank in ranks:
            try:
                results[rank] = self._forecast_one_rank(
                    centered_history=centered,
                    trajectory=trajectory,
                    left_vectors=left_vectors,
                    eigenvalues=eigenvalues,
                    total_energy=total_energy,
                    mean=mean,
                    rank=rank,
                )
            except SSANumericalError as exc:
                retained = float(eigenvalues[:rank].sum())
                results[rank] = SSAForecastResult(
                    forecast=np.full(self.horizon, np.nan),
                    cumulative_energy=retained / total_energy if total_energy > 0 else 1.0,
                    recurrence_denominator=np.nan,
                    recurrence_coefficient_norm=np.nan,
                    status="invalid",
                    failure_reason=str(exc),
                )
        return results
