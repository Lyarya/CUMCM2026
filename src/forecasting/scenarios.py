"""Leakage-safe paired whole-day residual scenarios for stochastic dispatch."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScenarioBatch:
    """Joint load/generation scenarios and their paired residual provenance."""

    target_date: pd.Timestamp
    load_kw: np.ndarray
    generation_kw: np.ndarray
    source_residual_dates: pd.DatetimeIndex


class PairedDailyResidualGenerator:
    """Sample complete paired residual days without splitting time or variables."""

    def __init__(self, *, day_steps: int = 144, seed: int = 2026) -> None:
        if day_steps <= 0:
            raise ValueError("day_steps must be positive")
        self.day_steps = int(day_steps)
        self._generator = np.random.default_rng(seed)
        self._dates: list[pd.Timestamp] = []
        self._load: list[np.ndarray] = []
        self._generation: list[np.ndarray] = []

    @property
    def pool_size(self) -> int:
        return len(self._dates)

    def add_residual_day(
        self,
        residual_date: object,
        load_residual: np.ndarray,
        generation_residual: np.ndarray,
    ) -> None:
        """Add one completed residual pair after its full target day is observed."""
        day = pd.Timestamp(residual_date).normalize()
        load = np.asarray(load_residual, dtype=float).reshape(-1)
        generation = np.asarray(generation_residual, dtype=float).reshape(-1)
        if load.size != self.day_steps or generation.size != self.day_steps:
            raise ValueError("each residual day must preserve the complete daily vector")
        if not np.isfinite(load).all() or not np.isfinite(generation).all():
            raise ValueError("residual days must be finite")
        if self._dates and day <= self._dates[-1]:
            raise ValueError("residual days must be added once in strictly chronological order")
        self._dates.append(day)
        self._load.append(load.copy())
        self._generation.append(generation.copy())

    def sample(
        self,
        target_date: object,
        forecast_load: np.ndarray,
        forecast_generation: np.ndarray,
        *,
        scenario_count: int,
        generation_upper_bound: float | None = None,
    ) -> ScenarioBatch:
        """Sample prior complete days and apply each paired trajectory as a unit."""
        target = pd.Timestamp(target_date).normalize()
        if scenario_count <= 0:
            raise ValueError("scenario_count must be positive")
        if not self._dates:
            raise ValueError("residual pool is empty")
        eligible = np.array([day < target for day in self._dates], dtype=bool)
        eligible_indices = np.flatnonzero(eligible)
        if eligible_indices.size == 0:
            raise ValueError("residual pool contains no date strictly before the target")
        load_base = np.asarray(forecast_load, dtype=float).reshape(-1)
        generation_base = np.asarray(forecast_generation, dtype=float).reshape(-1)
        if load_base.size != self.day_steps or generation_base.size != self.day_steps:
            raise ValueError("forecasts must contain one complete day")
        selected = self._generator.choice(eligible_indices, size=scenario_count, replace=True)
        load_residuals = np.stack([self._load[index] for index in selected])
        generation_residuals = np.stack([self._generation[index] for index in selected])
        load_scenarios = np.maximum(load_base[None, :] + load_residuals, 0.0)
        generation_scenarios = np.maximum(
            generation_base[None, :] + generation_residuals, 0.0
        )
        if generation_upper_bound is not None:
            if generation_upper_bound <= 0:
                raise ValueError("generation_upper_bound must be positive")
            generation_scenarios = np.minimum(generation_scenarios, generation_upper_bound)
        return ScenarioBatch(
            target_date=target,
            load_kw=load_scenarios,
            generation_kw=generation_scenarios,
            source_residual_dates=pd.DatetimeIndex([self._dates[index] for index in selected]),
        )


def build_walk_forward_scenarios(
    *,
    calibration_dates: pd.DatetimeIndex,
    calibration_load_residuals: np.ndarray,
    calibration_generation_residuals: np.ndarray,
    evaluation_dates: pd.DatetimeIndex,
    forecast_load: np.ndarray,
    forecast_generation: np.ndarray,
    actual_load: np.ndarray,
    actual_generation: np.ndarray,
    scenario_count: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Generate every day before adding that day's newly observed residual pair."""
    day_steps = forecast_load.shape[1]
    generator = PairedDailyResidualGenerator(day_steps=day_steps, seed=seed)
    for position, day in enumerate(calibration_dates):
        generator.add_residual_day(
            day,
            calibration_load_residuals[position],
            calibration_generation_residuals[position],
        )

    load_batches: list[np.ndarray] = []
    generation_batches: list[np.ndarray] = []
    source_dates: list[np.ndarray] = []
    manifest_rows: list[dict[str, object]] = []
    for position, target_date in enumerate(evaluation_dates):
        pool_size_before = generator.pool_size
        batch = generator.sample(
            target_date,
            forecast_load[position],
            forecast_generation[position],
            scenario_count=scenario_count,
            generation_upper_bound=None,
        )
        load_batches.append(batch.load_kw.astype(np.float32))
        generation_batches.append(batch.generation_kw.astype(np.float32))
        source_dates.append(batch.source_residual_dates.to_numpy(dtype="datetime64[D]"))
        manifest_rows.append(
            {
                "target_date": target_date.date().isoformat(),
                "residual_pool_size": pool_size_before,
                "scenario_count": scenario_count,
                "method": "paired_whole_day_residual_resampling",
                "day_steps": day_steps,
                "generation_lower_bound_kw": 0.0,
                "generation_upper_bound_kw": np.nan,
                "load_lower_bound_kw": 0.0,
                "all_sources_strictly_before_target": bool(
                    (batch.source_residual_dates < target_date).all()
                ),
            }
        )
        generator.add_residual_day(
            target_date,
            actual_load[position] - forecast_load[position],
            actual_generation[position] - forecast_generation[position],
        )
    archive = {
        "target_dates": evaluation_dates.to_numpy(dtype="datetime64[D]"),
        "source_residual_dates": np.stack(source_dates),
        "load_kw": np.stack(load_batches),
        "generation_kw": np.stack(generation_batches),
    }
    return archive, pd.DataFrame(manifest_rows)
