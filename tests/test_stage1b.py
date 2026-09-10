"""Unit tests for Stage 1B structural diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.data_diagnostics import (
    DiagnosticsConfig,
    anomaly_candidates,
    lag_correlations,
    lowrank_forecastability,
    official_forecast_diagnostics,
    seasonal_naive_metrics,
)


def _synthetic_actual(values: np.ndarray) -> pd.DataFrame:
    n = len(values)
    return pd.DataFrame(
        {
            "interval_end": pd.date_range("2025-01-01 00:10", periods=n, freq="10min"),
            "pv_actual_kw": values,
            "load_kw": 2 * values + 10,
            "net_load_kw": values + 10,
            "price_yuan_per_kwh": np.linspace(0.4, 1.0, n),
        }
    )


def test_lag_table_contains_full_and_daylight_pairs() -> None:
    values = np.tile(np.r_[np.zeros(72), np.arange(1, 73)], 8).astype(float)
    frame = _synthetic_actual(values)
    table = lag_correlations(frame, DiagnosticsConfig(lags=(144,)))
    pv = table.loc[table["variable"] == "PV"]
    assert set(pv["subset"]) == {"full_series", "daylight_only"}
    assert int(pv.loc[pv["subset"] == "daylight_only", "count"].iloc[0]) < int(
        pv.loc[pv["subset"] == "full_series", "count"].iloc[0]
    )


def test_seasonal_naive_uses_exact_historical_lags() -> None:
    values = np.arange(1_200, dtype=float)
    table = seasonal_naive_metrics(_synthetic_actual(values), DiagnosticsConfig())
    yesterday = table.query(
        "variable == 'PV' and baseline == 'Yesterday' and subset == 'full_series'"
    ).iloc[0]
    assert yesterday["count"] == 1_056
    assert yesterday["mae"] == 144.0
    assert yesterday["bias"] == -144.0


def test_official_diagnostics_excludes_unmatched_rows() -> None:
    frame = pd.DataFrame(
        {
            "alignment_status": ["matched", "matched", "unmatched"],
            "release_time": pd.to_datetime(
                ["2025-01-01 00:00", "2025-01-01 06:00", "2025-01-01 00:00"]
            ),
            "target_time": pd.to_datetime(
                ["2025-01-01 01:00", "2025-01-01 10:00", "2026-01-01 01:00"]
            ),
            "horizon_hour": [1, 4, 1],
            "pv_forecast_kw": [2.0, 8.0, 999.0],
            "pv_actual_kw": [1.0, 10.0, np.nan],
        }
    )
    overall, by_lead, _, _ = official_forecast_diagnostics(frame)
    assert int(overall.loc[0, "count"]) == 2
    assert overall.loc[0, "mae"] == 1.5
    assert by_lead.loc[by_lead["lead_bin"] == "1-3h", "count"].iloc[0] == 1


def test_lowrank_energy_uses_history_only() -> None:
    config = DiagnosticsConfig(
        history_steps=144,
        forecast_horizon_steps=24,
        hankel_rows=24,
        forecastability_stride=24,
    )
    values = 100 + 20 * np.sin(np.arange(240) * 2 * np.pi / 24)
    frame_a = _synthetic_actual(values)
    frame_b = frame_a.copy()
    frame_b.loc[144:, "pv_actual_kw"] += 5_000
    windows_a, _ = lowrank_forecastability(frame_a, config)
    windows_b, _ = lowrank_forecastability(frame_b, config)
    assert windows_a.loc[0, "top1_energy"] == windows_b.loc[0, "top1_energy"]
    assert windows_a.loc[0, "top3_energy"] == windows_b.loc[0, "top3_energy"]


def test_anomaly_detection_only_returns_flags() -> None:
    values = np.r_[np.ones(30), 100.0]
    candidates = anomaly_candidates(_synthetic_actual(values), DiagnosticsConfig())
    assert set(candidates["action"]) == {"flag_only_do_not_delete"}
    assert len(candidates) >= 1
