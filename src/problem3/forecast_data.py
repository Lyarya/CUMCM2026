"""Canonical Attachment-3 forecast data and causal seasonal experts for Q3."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.paths import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR


RAW_ATTACHMENT3 = RAW_DATA_DIR / "C题" / "附件" / "附件3.xlsx"
ACTUAL_PATH = PROCESSED_DATA_DIR / "C题" / "actual_10min.csv"
HOURLY_PATH = PROCESSED_DATA_DIR / "C题" / "pv_forecast_hourly.csv"
TEN_MINUTE_PATH = PROCESSED_DATA_DIR / "C题" / "pv_forecast_10min.csv"
CANONICAL_PATH = INTERIM_DATA_DIR / "C题" / "q3_official_forecast_canonical.csv"

RELEASE_HOURS = (0, 6, 12, 18)
LEAD_BIN_LABELS = ("1-3h", "4-6h", "7-12h", "13-18h", "19-24h")
LEAD_BIN_EDGES = (0.0, 3.0, 6.0, 12.0, 18.0, 24.0)


def file_sha256(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def lead_bin(values: pd.Series | np.ndarray) -> pd.Categorical:
    """Map positive lead hours to the five competition reporting bins."""

    return pd.cut(
        np.asarray(values, dtype=float),
        bins=LEAD_BIN_EDGES,
        labels=LEAD_BIN_LABELS,
        include_lowest=False,
        right=True,
    )


def _actual_series() -> pd.Series:
    actual = pd.read_csv(ACTUAL_PATH, parse_dates=["interval_end"])
    if actual["interval_end"].duplicated().any():
        raise AssertionError("actual interval-end timestamps must be unique")
    return actual.set_index("interval_end")["pv_actual_kw"].astype(float)


def build_official_canonical() -> pd.DataFrame:
    """Return all 35,040 hourly Attachment-3 forecasts with exact provenance."""

    forecast = pd.read_csv(HOURLY_PATH, parse_dates=["release_time", "target_time"])
    expected_target = forecast["release_time"] + pd.to_timedelta(
        forecast["horizon_hour"], unit="h"
    )
    if not expected_target.equals(forecast["target_time"]):
        raise AssertionError("Attachment-3 target_time != release_time + horizon_hour")
    actual = _actual_series()
    frame = pd.DataFrame(
        {
            "forecast_id": forecast["forecast_id"].astype(int),
            "date": forecast["release_time"].dt.normalize(),
            "issue_time": forecast["release_time"],
            "target_time": forecast["target_time"],
            "lead_minutes": forecast["horizon_hour"].astype(int) * 60,
            "lead_hours": forecast["horizon_hour"].astype(int),
            "lead_bin": lead_bin(forecast["horizon_hour"]).astype(str),
            "release_hour": forecast["release_time"].dt.hour.astype(int),
            "forecast_pv_kw": forecast["pv_forecast_kw"].astype(float),
            "actual_pv_kw": forecast["target_time"].map(actual),
        }
    )
    frame["alignment_status"] = np.where(
        frame["actual_pv_kw"].notna(), "matched", "target_outside_actual_range"
    )
    frame["daylight_flag"] = frame["actual_pv_kw"].fillna(0.0).gt(0.0)
    for lag, name in ((1, "yesterday_pv_kw"), (7, "last_week_pv_kw")):
        frame[name] = (frame["target_time"] - pd.Timedelta(days=lag)).map(actual)
    lagged = [
        (frame["target_time"] - pd.Timedelta(days=lag)).map(actual).to_numpy(float)
        for lag in range(1, 8)
    ]
    lagged_array = np.stack(lagged)
    complete = np.isfinite(lagged_array).all(axis=0)
    frame["same_slot_7d_mean_kw"] = np.nan
    frame.loc[complete, "same_slot_7d_mean_kw"] = lagged_array[:, complete].mean(axis=0)
    return frame


def audit_attachment3(frame: pd.DataFrame | None = None) -> dict[str, object]:
    """Audit the raw issue grid, target grid, values and overlap structure."""

    frame = build_official_canonical() if frame is None else frame.copy()
    counts = frame.groupby(["date", "release_hour"], sort=True).size()
    releases_per_day = frame[["date", "release_hour"]].drop_duplicates().groupby("date").size()
    target_version_count = frame.groupby("target_time").size()
    expected_pairs = pd.MultiIndex.from_product(
        [pd.date_range("2025-01-01", "2025-12-31", freq="D"), RELEASE_HOURS],
        names=["date", "release_hour"],
    )
    missing_release_count = int(len(expected_pairs.difference(counts.index)))
    non_hourly_targets = int(
        ((frame["target_time"].dt.minute != 0) | (frame["target_time"].dt.second != 0)).sum()
    )
    return {
        "source_path": str(RAW_ATTACHMENT3),
        "source_sha256": file_sha256(RAW_ATTACHMENT3),
        "raw_rows": int(len(frame)),
        "date_start": frame["date"].min().date().isoformat(),
        "date_end": frame["date"].max().date().isoformat(),
        "release_hours": list(RELEASE_HOURS),
        "release_rows": int(len(counts)),
        "missing_release_count": missing_release_count,
        "days_with_four_releases": int(releases_per_day.eq(4).sum()),
        "batches_with_24_horizons": int(counts.eq(24).sum()),
        "duplicate_forecast_id_count": int(frame["forecast_id"].duplicated().sum()),
        "duplicate_issue_horizon_count": int(
            frame.duplicated(["issue_time", "lead_hours"]).sum()
        ),
        "non_hourly_target_count": non_hourly_targets,
        "negative_pv_forecast_count": int(frame["forecast_pv_kw"].lt(0).sum()),
        "missing_forecast_count": int(frame["forecast_pv_kw"].isna().sum()),
        "matched_actual_count": int(frame["actual_pv_kw"].notna().sum()),
        "unmatched_actual_count": int(frame["actual_pv_kw"].isna().sum()),
        "forecast_min_kw": float(frame["forecast_pv_kw"].min()),
        "forecast_max_kw": float(frame["forecast_pv_kw"].max()),
        "target_resolution": "hourly nodes",
        "issue_target_rule": "target_time = issue_time + lead_hours",
        "issue_target_alignment_pass": bool(
            missing_release_count == 0
            and counts.eq(24).all()
            and non_hourly_targets == 0
        ),
        "targets_with_multiple_versions": int(target_version_count.gt(1).sum()),
        "maximum_versions_per_target": int(target_version_count.max()),
    }


def audit_hourly_to_ten_minute_mapping() -> dict[str, object]:
    """Verify Stage-1 PCHIP batches and preservation of hourly source nodes."""

    hourly = pd.read_csv(HOURLY_PATH, parse_dates=["release_time", "target_time"])
    fine = pd.read_csv(TEN_MINUTE_PATH, parse_dates=["release_time", "target_time"])
    counts = fine.groupby("release_time").size()
    nodes = fine.loc[fine["is_original_hourly_node"].astype(bool)].copy()
    nodes["horizon_hour"] = (nodes["horizon_step_10min"] // 6).astype(int)
    checked = nodes.merge(
        hourly[["release_time", "horizon_hour", "pv_forecast_kw"]],
        on=["release_time", "horizon_hour"],
        how="left",
        validate="one_to_one",
    )
    max_node_error = float(
        np.nanmax(np.abs(checked["pv_forecast_10min_kw"] - checked["pv_forecast_kw"]))
    )
    return {
        "method": "release-local PCHIP with linear fallback",
        "ten_minute_rows": int(len(fine)),
        "batch_count": int(fine["release_time"].nunique()),
        "batches_with_144_steps": int(counts.eq(144).sum()),
        "hourly_node_count": int(len(nodes)),
        "maximum_hourly_node_error_kw": max_node_error,
        "cross_batch_interpolation": False,
        "mapping_pass": bool(counts.eq(144).all() and max_node_error <= 1e-9),
    }


__all__ = [
    "ACTUAL_PATH",
    "CANONICAL_PATH",
    "HOURLY_PATH",
    "LEAD_BIN_LABELS",
    "RAW_ATTACHMENT3",
    "RELEASE_HOURS",
    "TEN_MINUTE_PATH",
    "audit_attachment3",
    "audit_hourly_to_ten_minute_mapping",
    "build_official_canonical",
    "file_sha256",
    "lead_bin",
]
