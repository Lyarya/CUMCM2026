"""Stage 1A acceptance tests for the canonical C-problem pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.common.data_validation import RAW_C_DIR, file_sha256
from src.common.metrics import masked_mae, masked_nmae, masked_nrmse, masked_rmse
from src.common.preprocessing import (
    DT_HOURS,
    add_causal_history_features,
    build_canonical_data,
    interpolate_forecast_batches,
)
from src.common.time_utils import parse_interval_end_offset, validate_timestamps
from src.common.windowing import build_windows


OUTPUT_DIR = Path("data/processed/C题")


@pytest.fixture(scope="session")
def canonical():
    return build_canonical_data(RAW_C_DIR)


def test_next_midnight_marker() -> None:
    assert parse_interval_end_offset("00:10:00") == pd.Timedelta(minutes=10)
    assert parse_interval_end_offset("0:00+1") == pd.Timedelta(days=1)


def test_problem1_contract(canonical) -> None:
    frame = canonical.problem1_day
    assert len(frame) == 144
    assert frame["slot"].tolist() == list(range(1, 145))
    assert frame["interval_end"].iloc[-1] == "24:00"
    np.testing.assert_allclose(frame["load_kwh"], frame["load_kw"] / 6)


def test_actual_contract_and_continuity(canonical) -> None:
    frame = canonical.actual_10min
    assert len(frame) == 52_560
    assert frame["interval_end"].iloc[0] == pd.Timestamp("2025-01-01 00:10")
    assert frame["interval_end"].iloc[-1] == pd.Timestamp("2026-01-01 00:00")
    assert not frame["interval_end"].duplicated().any()
    check = validate_timestamps(frame["interval_end"], frequency="10min")
    assert check["invalid_count"] == 0
    assert check["duplicate_count"] == 0
    assert check["non_frequency_count"] == 0
    assert frame.groupby("operating_date").size().eq(144).all()
    np.testing.assert_allclose(frame["pv_actual_kwh"], frame["pv_actual_kw"] * DT_HOURS)
    np.testing.assert_allclose(
        frame["net_load_kw"], frame["load_kw"] - frame["pv_actual_kw"]
    )


def test_forecast_hourly_contract(canonical) -> None:
    frame = canonical.pv_forecast_hourly
    assert len(frame) == 35_040
    assert not frame.duplicated(["release_time", "horizon_hour"]).any()
    expected_target = frame["release_time"] + pd.to_timedelta(frame["horizon_hour"], unit="h")
    pd.testing.assert_series_equal(frame["target_time"], expected_target, check_names=False)
    releases = frame[["release_time"]].drop_duplicates()
    assert releases.groupby(releases["release_time"].dt.normalize()).size().eq(4).all()
    assert frame.groupby("release_time").size().eq(24).all()


def test_alignment_counts(canonical) -> None:
    report = canonical.alignment_report
    assert report["raw_forecast_count"] == 35_040
    assert report["aligned_count"] == 35_004
    assert report["unmatched_count"] == 36
    assert report["unmatched_reasons"] == {"out_of_actual_range": 36}
    assert report["minimum_mae_alignment"] == "target_time"


def test_interpolation_preserves_hourly_nodes(canonical) -> None:
    assert canonical.interpolation_report["hourly_nodes_preserved"] is True
    assert canonical.interpolation_report["future_actual_used"] is False
    assert canonical.pv_forecast_10min.groupby("release_time").size().eq(144).all()


def test_interpolation_cannot_use_future_actual() -> None:
    release = pd.Timestamp("2025-01-01 06:00")
    forecast = pd.DataFrame(
        {
            "forecast_id": np.arange(1, 25),
            "release_time": release,
            "horizon_hour": np.arange(1, 25),
            "target_time": release + pd.to_timedelta(np.arange(1, 25), unit="h"),
            "pv_forecast_kw": np.arange(1, 25, dtype=float),
        }
    )
    times = pd.date_range(release, periods=145, freq="10min")
    actual_a = pd.DataFrame({"interval_end": times, "pv_actual_kw": 1.0})
    actual_b = actual_a.copy()
    actual_b.loc[actual_b["interval_end"] > release, "pv_actual_kw"] = 999_999.0
    out_a, _ = interpolate_forecast_batches(forecast, actual_a)
    out_b, _ = interpolate_forecast_batches(forecast, actual_b)
    np.testing.assert_allclose(
        out_a["pv_forecast_10min_kw"], out_b["pv_forecast_10min_kw"], equal_nan=True
    )


def test_lag_features_are_exact_and_causal() -> None:
    n = 1_200
    times = pd.date_range("2025-01-01 00:10", periods=n, freq="10min")
    values = np.arange(n, dtype=float)
    frame = pd.DataFrame(
        {
            "interval_end": times,
            "slot": np.tile(np.arange(1, 145), int(np.ceil(n / 144)))[:n],
            "pv_actual_kw": values,
            "load_kw": values * 2,
        }
    )
    featured = add_causal_history_features(frame)
    assert featured.loc[144, "pv_lag_1d"] == values[0]
    assert featured.loc[1008, "pv_lag_7d"] == values[0]
    assert featured.loc[1008, "load_lag_7d"] == values[0] * 2

    altered = frame.copy()
    altered.loc[1100:, ["pv_actual_kw", "load_kw"]] = -999.0
    featured_altered = add_causal_history_features(altered)
    pd.testing.assert_frame_equal(featured.iloc[:1100], featured_altered.iloc[:1100])


def test_timestamp_window_and_masks() -> None:
    frame = pd.DataFrame(
        {
            "interval_end": pd.date_range("2025-01-01 00:10", periods=6, freq="10min"),
            "feature": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0],
            "target": [2.0, 3.0, 4.0, np.nan, 6.0, 7.0],
        }
    )
    windows = build_windows(
        frame,
        feature_columns=["feature"],
        target_columns=["target"],
        history_steps=2,
        horizon_steps=1,
    )
    assert windows.input_mask[0, 1, 0] == 0
    assert windows.inputs[0, 1, 0] == 0
    assert windows.valid_count == 2
    assert windows.invalid_count == 2

    gapped = frame.copy()
    gapped.loc[3:, "interval_end"] += pd.Timedelta(minutes=10)
    gap_windows = build_windows(
        gapped,
        feature_columns=["feature"],
        target_columns=["target"],
        history_steps=2,
        horizon_steps=1,
    )
    assert gap_windows.valid_count < windows.valid_count


def test_masked_metrics_create_mask_before_zero_fill() -> None:
    true = np.array([1.0, np.nan, 3.0])
    pred = np.array([2.0, 1_000_000.0, 1.0])
    assert masked_mae(true, pred) == pytest.approx(1.5)
    assert masked_rmse(true, pred) == pytest.approx(np.sqrt(2.5))
    assert masked_nmae(true, pred) == pytest.approx(0.75)
    assert masked_nrmse(true, pred) == pytest.approx(np.sqrt(2.5) / 2)


def test_raw_workbook_hashes_unchanged(canonical) -> None:
    before = {path.name: file_sha256(path) for path in sorted(RAW_C_DIR.glob("*.xlsx"))}
    _ = canonical.actual_10min.shape
    after = {path.name: file_sha256(path) for path in sorted(RAW_C_DIR.glob("*.xlsx"))}
    assert before == after
