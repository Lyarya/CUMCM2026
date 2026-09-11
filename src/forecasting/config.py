"""Central configuration for the Stage 2A forecasting protocol."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Q2ForecastConfig:
    """Immutable settings fixed before the February--December evaluation."""

    day_steps: int = 144
    history_days: int = 7
    hankel_rows: int = 144
    hankel_ranks: tuple[int, ...] = (1, 3, 5)
    polynomial_degree: int = 2
    polynomial_ridge: float = 5.0
    ridge_alphas: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
    seasonal_alphas: tuple[float, ...] = tuple(index / 20 for index in range(21))
    fusion_seasonal_weights: tuple[float, ...] = tuple(index / 20 for index in range(21))
    calibration_start: date = date(2025, 1, 1)
    calibration_end: date = date(2025, 1, 31)
    evaluation_start: date = date(2025, 2, 1)
    evaluation_end: date = date(2025, 12, 31)
    scenario_count: int = 50
    seed: int = 2026

    @property
    def history_steps(self) -> int:
        return self.day_steps * self.history_days

    def __post_init__(self) -> None:
        if self.day_steps <= 0 or self.history_days < 7:
            raise ValueError("day_steps must be positive and history_days must be at least seven")
        if not 2 <= self.hankel_rows < self.history_steps:
            raise ValueError("hankel_rows must lie in [2, history_steps)")
        if any(rank <= 0 or rank > self.hankel_rows for rank in self.hankel_ranks):
            raise ValueError("all Hankel ranks must be positive and no larger than hankel_rows")
        if self.polynomial_degree < 0 or self.polynomial_ridge < 0:
            raise ValueError("invalid polynomial configuration")
        if any(alpha < 0 for alpha in self.ridge_alphas):
            raise ValueError("ridge alphas cannot be negative")
        if any(not 0 <= weight <= 1 for weight in self.seasonal_alphas):
            raise ValueError("seasonal weights must lie in [0, 1]")
        if any(not 0 <= weight <= 1 for weight in self.fusion_seasonal_weights):
            raise ValueError("fusion weights must lie in [0, 1]")
        if self.calibration_end >= self.evaluation_start:
            raise ValueError("calibration must end before evaluation starts")
        if self.scenario_count <= 0:
            raise ValueError("scenario_count must be positive")
