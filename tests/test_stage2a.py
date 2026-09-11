"""Leakage and mathematical-contract tests for Stage 2A."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.forecasting.config import Q2ForecastConfig
from src.forecasting.data import DailyPanel, date_indices
from src.forecasting.model import (
    HankelLowRankExpert,
    convex_fusion,
    last_week,
    load_feature_matrix,
    polynomial_vandermonde,
    seasonal_phase,
    yesterday,
)
from src.forecasting.scenarios import PairedDailyResidualGenerator


RESULT_DIR = Path(__file__).resolve().parents[1] / "results" / "tables" / "forecasting"


def _panel(days: int = 12, steps: int = 144) -> DailyPanel:
    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    values = np.arange(days * steps, dtype=float).reshape(days, steps)
    datetimes = np.stack(
        [
            pd.date_range(day + pd.Timedelta(minutes=10), periods=steps, freq="10min")
            .to_numpy(dtype="datetime64[ns]")
            for day in dates
        ]
    )
    return DailyPanel(dates, datetimes, values * 2 + 100, values)


def test_lag144_and_lag1008_are_exact() -> None:
    panel = _panel()
    np.testing.assert_array_equal(yesterday(panel.generation_kw, 8), panel.generation_kw[7])
    np.testing.assert_array_equal(last_week(panel.generation_kw, 8), panel.generation_kw[1])


def test_daily_origin_cannot_access_current_or_future_actuals() -> None:
    panel = _panel()
    day_index = 8
    phase_before = seasonal_phase(panel.generation_kw, day_index, 0.6)
    features_before = load_feature_matrix(panel, day_index)
    altered_generation = panel.generation_kw.copy()
    altered_load = panel.load_kw.copy()
    altered_generation[day_index:] = 999_999
    altered_load[day_index:] = 999_999
    altered = DailyPanel(panel.dates, panel.datetimes, altered_load, altered_generation)
    np.testing.assert_array_equal(
        phase_before, seasonal_phase(altered.generation_kw, day_index, 0.6)
    )
    np.testing.assert_array_equal(features_before, load_feature_matrix(altered, day_index))


def test_january_is_calibration_and_february_starts_formal_evaluation() -> None:
    config = Q2ForecastConfig()
    panel = _panel(days=365)
    calibration = date_indices(panel, config.calibration_start, config.calibration_end)
    evaluation = date_indices(panel, config.evaluation_start, config.evaluation_end)
    assert len(calibration) == 31
    assert panel.dates[evaluation[0]] == pd.Timestamp("2025-02-01")
    assert panel.dates[evaluation[-1]] == pd.Timestamp("2025-12-31")
    with pytest.raises(FrozenInstanceError):
        config.scenario_count = 99  # type: ignore[misc]


def test_hankel_rank_is_configurable_and_vandermonde_is_polynomial() -> None:
    time = np.array([-1.0, 0.0, 2.0])
    expected = np.column_stack([np.ones(3), time, time**2])
    np.testing.assert_allclose(polynomial_vandermonde(time, 2), expected)
    history = 20 + np.sin(np.arange(1008) * 2 * np.pi / 144)
    expert = HankelLowRankExpert(1008, 144, 144, polynomial_degree=2)
    forecasts = expert.forecast_ranks(history, (1, 3, 5))
    assert set(forecasts) == {1, 3, 5}
    assert all(values.shape == (144,) for values in forecasts.values())
    assert all((values >= 0).all() for values in forecasts.values())


def test_fusion_weights_are_nonnegative_and_sum_to_one() -> None:
    seasonal = np.array([1.0, 2.0])
    low_rank = np.array([3.0, 4.0])
    weight = 0.35
    fused = convex_fusion(seasonal, low_rank, weight)
    np.testing.assert_allclose(fused, weight * seasonal + (1 - weight) * low_rank)
    assert weight >= 0 and 1 - weight >= 0
    assert weight + (1 - weight) == pytest.approx(1.0)


def test_residual_pool_excludes_future_and_preserves_paired_whole_days() -> None:
    steps = 144
    generator = PairedDailyResidualGenerator(day_steps=steps, seed=2026)
    generator.add_residual_day("2025-01-01", np.ones(steps), np.ones(steps) * 10)
    generator.add_residual_day("2025-01-03", np.ones(steps) * 3, np.ones(steps) * 30)
    load_forecast = np.ones(steps) * 100
    generation_forecast = np.ones(steps) * 50
    batch = generator.sample(
        "2025-01-02", load_forecast, generation_forecast, scenario_count=8
    )
    assert (batch.source_residual_dates == pd.Timestamp("2025-01-01")).all()
    load_residual = batch.load_kw - load_forecast
    generation_residual = batch.generation_kw - generation_forecast
    np.testing.assert_allclose(generation_residual, 10 * load_residual)
    assert np.all(load_residual == load_residual[:, :1])


def test_all_daily_timestamps_are_strictly_aligned() -> None:
    panel = _panel()
    assert panel.datetimes.shape[1] == 144
    flattened = panel.datetimes.reshape(-1).astype("datetime64[ns]").astype("int64")
    assert np.all(np.diff(flattened) == pd.Timedelta(minutes=10).value)


def test_formal_artifacts_freeze_january_choices_and_start_february() -> None:
    weights = pd.read_csv(RESULT_DIR / "q2_fusion_weights.csv")
    assert len(weights) == 1
    assert weights.loc[0, "calibration_period"] == "2025-01-01/2025-01-31"
    assert weights.loc[0, "formal_evaluation_period"] == "2025-02-01/2025-12-31"
    assert bool(weights.loc[0, "weights_fixed_during_formal_evaluation"])
    assert not bool(weights.loc[0, "online_parameter_learning"])
    assert weights.loc[0, "seasonal_weight"] >= 0
    assert weights.loc[0, "low_rank_weight"] >= 0
    assert weights.loc[0, "seasonal_weight"] + weights.loc[0, "low_rank_weight"] == pytest.approx(1)

    predictions = pd.read_csv(
        RESULT_DIR / "q2_forecast_predictions.csv", usecols=["operating_date", "datetime"]
    )
    assert predictions["operating_date"].min() == "2025-02-01"
    assert predictions["operating_date"].max() == "2025-12-31"
    assert predictions.groupby("operating_date").size().eq(144).all()
    assert pd.to_datetime(predictions["datetime"]).is_monotonic_increasing

    audit = json.loads((RESULT_DIR / "q2_leakage_audit.json").read_text(encoding="utf-8"))
    assert audit["scenario_source_dates_strictly_before_target"] is True
    assert audit["online_parameter_learning"] is False
    assert audit["official_forecast_used"] is False
