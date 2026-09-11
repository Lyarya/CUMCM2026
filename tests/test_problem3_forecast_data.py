"""Attachment-3 mapping and causality tests for the Q3 forecast layer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.problem3.forecast_data import (
    RAW_ATTACHMENT3,
    audit_attachment3,
    audit_hourly_to_ten_minute_mapping,
    build_official_canonical,
    file_sha256,
    lead_bin,
)
from src.problem3.information_schedule import _seasonal_forecast


def test_attachment3_issue_target_mapping_and_raw_hash() -> None:
    frame = build_official_canonical()
    audit = audit_attachment3(frame)
    expected = frame["issue_time"] + pd.to_timedelta(frame["lead_hours"], unit="h")
    assert expected.equals(frame["target_time"])
    assert audit["raw_rows"] == 35_040
    assert audit["release_hours"] == [0, 6, 12, 18]
    assert audit["missing_release_count"] == 0
    assert audit["duplicate_issue_horizon_count"] == 0
    assert audit["negative_pv_forecast_count"] == 0
    final_audit = json.loads(
        Path("results/tables/forecasting/forecast_final_leakage_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert file_sha256(RAW_ATTACHMENT3) == final_audit["raw_excel_hashes_before"]["附件3.xlsx"]


def test_lead_bin_boundaries() -> None:
    observed = lead_bin(np.array([1, 3, 4, 6, 7, 12, 13, 18, 19, 24])).astype(str).tolist()
    assert observed == [
        "1-3h", "1-3h", "4-6h", "4-6h", "7-12h", "7-12h",
        "13-18h", "13-18h", "19-24h", "19-24h",
    ]


def test_hourly_to_ten_minute_mapping_is_release_local_and_exact_at_nodes() -> None:
    audit = audit_hourly_to_ten_minute_mapping()
    assert audit["mapping_pass"] is True
    assert audit["batch_count"] == 365 * 4
    assert audit["batches_with_144_steps"] == 365 * 4
    assert audit["maximum_hourly_node_error_kw"] <= 1e-9
    assert audit["cross_batch_interpolation"] is False


def test_future_actual_changes_cannot_change_issued_seasonal_forecast() -> None:
    actual_frame = pd.read_csv("data/processed/C题/actual_10min.csv", parse_dates=["interval_end"])
    actual = actual_frame.set_index("interval_end")["pv_actual_kw"].astype(float)
    issue = pd.Timestamp("2025-06-20 06:00")
    targets = pd.date_range(issue + pd.Timedelta(minutes=10), periods=108, freq="10min")
    first = _seasonal_forecast(targets, issue, actual)
    altered = actual.copy()
    altered.loc[altered.index > issue] = 999_999.0
    second = _seasonal_forecast(targets, issue, altered)
    np.testing.assert_array_equal(first, second)
