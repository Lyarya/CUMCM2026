"""Step-1 raw-data quality audit for all four C-problem attachments.

This module checks source integrity and time structure without cleaning,
imputing, normalizing, interpolating, or writing transformed model inputs.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.common.paths import PROJECT_ROOT, RAW_DATA_DIR
from src.common.time_utils import parse_interval_end_offset, validate_timestamps


RAW_C_DIR = RAW_DATA_DIR / "C题" / "附件"
ATTACHMENTS = {index: RAW_C_DIR / f"附件{index}.xlsx" for index in range(1, 5)}
TEN_MINUTES = pd.Timedelta(minutes=10)
EXPECTED_SLOTS = 144
EXPECTED_DAYS = 365


@dataclass(frozen=True)
class DataQualityAudit:
    """Machine-readable report plus source tables used by the paper figure."""

    report: dict[str, Any]
    summary: pd.DataFrame
    attachment1_comparison: pd.DataFrame


def file_sha256(path: str | Path) -> str:
    """Return a source-file checksum without modifying the file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_slot_offsets(labels: list[object], source: str) -> pd.TimedeltaIndex:
    offsets = pd.TimedeltaIndex([parse_interval_end_offset(label) for label in labels])
    expected = pd.timedelta_range(TEN_MINUTES, periods=EXPECTED_SLOTS, freq=TEN_MINUTES)
    if len(offsets) != EXPECTED_SLOTS or not offsets.equals(expected):
        raise ValueError(
            f"{source} does not contain the required ordered 10-minute interval ends "
            "from 00:10 through 0:00+1."
        )
    return offsets


def _numeric_matrix(frame: pd.DataFrame, columns: list[object], source: str) -> np.ndarray:
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
    non_numeric = int(numeric.isna().sum().sum() - frame[columns].isna().sum().sum())
    if non_numeric:
        raise ValueError(f"{source} contains {non_numeric} non-numeric measurement cells.")
    return numeric.to_numpy(dtype=float)


def _audit_wide_sheet(
    path: Path,
    sheet_name: str,
    value_name: str,
) -> tuple[dict[str, Any], np.ndarray]:
    frame = pd.read_excel(path, sheet_name=sheet_name)
    date_column = frame.columns[0]
    value_columns = list(frame.columns[1:])
    offsets = _validate_slot_offsets(value_columns, f"{path.name}/{sheet_name}")
    dates = pd.to_datetime(frame[date_column], errors="coerce").dt.normalize()
    if dates.isna().any():
        raise ValueError(f"{path.name}/{sheet_name} contains invalid operating dates.")
    matrix = _numeric_matrix(frame, value_columns, f"{path.name}/{sheet_name}")

    timestamps = (
        dates.to_numpy(dtype="datetime64[ns]")[:, None]
        + offsets.to_numpy(dtype="timedelta64[ns]")[None, :]
    ).reshape(-1)
    ordered = np.sort(timestamps)
    duplicate_count = int(pd.Index(timestamps).duplicated().sum())
    non_ten_minute_gaps = int(np.sum(np.diff(ordered) != np.timedelta64(10, "m")))
    day_counts = pd.Series(timestamps).groupby(dates.repeat(EXPECTED_SLOTS).to_numpy()).size()

    audit = {
        "source_file": str(path.relative_to(PROJECT_ROOT)),
        "sheet": sheet_name,
        "value_name": value_name,
        "raw_rows": int(frame.shape[0]),
        "raw_columns": int(frame.shape[1]),
        "observed_points": int(matrix.size),
        "expected_points": EXPECTED_DAYS * EXPECTED_SLOTS,
        "measurement_missing": int(np.isnan(matrix).sum()),
        "negative_values": int(np.sum(matrix < 0)),
        "zero_values": int(np.sum(matrix == 0)),
        "duplicate_timestamps": duplicate_count,
        "non_ten_minute_gaps": non_ten_minute_gaps,
        "operating_date_start": str(dates.min().date()),
        "operating_date_end": str(dates.max().date()),
        "interval_end_start": str(pd.Timestamp(ordered[0])),
        "interval_end_end": str(pd.Timestamp(ordered[-1])),
        "days": int(dates.nunique()),
        "minimum_records_per_day": int(day_counts.min()),
        "maximum_records_per_day": int(day_counts.max()),
        "minimum": float(np.nanmin(matrix)),
        "maximum": float(np.nanmax(matrix)),
        "mean": float(np.nanmean(matrix)),
    }
    return audit, matrix


def _audit_attachment1(path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    frame = pd.read_excel(path, sheet_name="Sheet1")
    required = ["时间", "电价", "小区负载", "光伏发电预测功率"]
    if list(frame.columns) != required:
        raise ValueError(f"附件1字段不符合题面定义：{list(frame.columns)}")
    _validate_slot_offsets(frame["时间"].tolist(), "附件1/Sheet1")
    numeric = frame[required[1:]].apply(pd.to_numeric, errors="coerce")
    audit = {
        "source_file": str(path.relative_to(PROJECT_ROOT)),
        "sheet": "Sheet1",
        "raw_rows": int(frame.shape[0]),
        "raw_columns": int(frame.shape[1]),
        "observed_points": int(len(frame)),
        "expected_points": EXPECTED_SLOTS,
        "measurement_missing": int(numeric.isna().sum().sum()),
        "duplicate_slots": int(frame["时间"].duplicated().sum()),
        "non_ten_minute_gaps": 0,
        "negative_values": int((numeric < 0).sum().sum()),
        "zero_values": int((numeric == 0).sum().sum()),
        "interval_end_start": "00:10:00",
        "interval_end_end": "0:00+1",
        "columns": required,
    }
    return audit, frame


def _audit_attachment3(path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    frame = pd.read_excel(path, sheet_name="Sheet1")
    horizon_columns = [f"预报{hour}小时" for hour in range(1, 25)]
    required = ["日期", "预报时刻", *horizon_columns]
    if list(frame.columns) != required:
        raise ValueError("附件3字段或预报步长不符合题面定义。")
    raw_date_blanks = int(frame["日期"].isna().sum())
    dates = pd.to_datetime(frame["日期"].ffill(), errors="coerce").dt.normalize()
    if dates.isna().any():
        raise ValueError("附件3日期向下填充后仍有无效值。")
    release_offsets = pd.TimedeltaIndex(
        [parse_interval_end_offset(value) for value in frame["预报时刻"]]
    )
    release_times = dates + release_offsets
    forecasts = _numeric_matrix(frame, horizon_columns, "附件3/Sheet1")
    per_day = pd.DataFrame({"date": dates, "release": release_offsets}).groupby("date").size()
    expected_releases = {
        pd.Timedelta(hours=0),
        pd.Timedelta(hours=6),
        pd.Timedelta(hours=12),
        pd.Timedelta(hours=18),
    }
    actual_releases = set(release_offsets.unique())
    if actual_releases != expected_releases:
        raise ValueError(f"附件3发布时间异常：{sorted(map(str, actual_releases))}")
    ordered = np.sort(release_times.to_numpy(dtype="datetime64[ns]"))
    audit = {
        "source_file": str(path.relative_to(PROJECT_ROOT)),
        "sheet": "Sheet1",
        "raw_rows": int(frame.shape[0]),
        "raw_columns": int(frame.shape[1]),
        "forecast_values": int(forecasts.size),
        "expected_forecast_values": EXPECTED_DAYS * 4 * 24,
        "measurement_missing": int(np.isnan(forecasts).sum()),
        "structural_blank_date_cells": raw_date_blanks,
        "negative_values": int(np.sum(forecasts < 0)),
        "zero_values": int(np.sum(forecasts == 0)),
        "duplicate_release_times": int(release_times.duplicated().sum()),
        "non_six_hour_release_gaps": int(np.sum(np.diff(ordered) != np.timedelta64(6, "h"))),
        "operating_date_start": str(dates.min().date()),
        "operating_date_end": str(dates.max().date()),
        "release_time_start": str(release_times.min()),
        "release_time_end": str(release_times.max()),
        "days": int(dates.nunique()),
        "minimum_releases_per_day": int(per_day.min()),
        "maximum_releases_per_day": int(per_day.max()),
        "forecast_horizons": 24,
        "minimum": float(np.nanmin(forecasts)),
        "maximum": float(np.nanmax(forecasts)),
        "mean": float(np.nanmean(forecasts)),
    }
    return audit, frame


def _comparison_row(name: str, official: np.ndarray, annual_mean: np.ndarray) -> dict[str, Any]:
    difference = official - annual_mean
    scale = float(np.mean(np.abs(annual_mean)))
    return {
        "variable": name,
        "mae": float(np.mean(np.abs(difference))),
        "maximum_absolute_error": float(np.max(np.abs(difference))),
        "mean_absolute_reference": scale,
        "normalized_mae_percent": float(100 * np.mean(np.abs(difference)) / scale),
    }


def audit_all_attachments(raw_dir: str | Path = RAW_C_DIR) -> DataQualityAudit:
    """Audit all official sources and compare Attachment 1 with slot means."""
    raw = Path(raw_dir)
    paths = {index: raw / f"附件{index}.xlsx" for index in range(1, 5)}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing official attachments:\n" + "\n".join(missing))

    expected_sheets = {
        1: ["Sheet1"],
        2: ["小区负载", "光伏发电实际功率"],
        3: ["Sheet1"],
        4: ["Sheet1"],
    }
    workbook_sheets = {
        path.name: pd.ExcelFile(path).sheet_names for path in paths.values()
    }
    for index, path in paths.items():
        if workbook_sheets[path.name] != expected_sheets[index]:
            raise ValueError(
                f"{path.name} sheets {workbook_sheets[path.name]} do not match "
                f"the expected structure {expected_sheets[index]}."
            )

    source_hashes_before = {path.name: file_sha256(path) for path in paths.values()}
    attachment1_audit, attachment1 = _audit_attachment1(paths[1])
    load_audit, load_matrix = _audit_wide_sheet(paths[2], "小区负载", "load_kw")
    pv_audit, pv_matrix = _audit_wide_sheet(paths[2], "光伏发电实际功率", "pv_actual_kw")
    forecast_audit, _ = _audit_attachment3(paths[3])
    price_audit, price_matrix = _audit_wide_sheet(paths[4], "Sheet1", "price_yuan_per_kwh")
    source_hashes_after = {path.name: file_sha256(path) for path in paths.values()}
    if source_hashes_before != source_hashes_after:
        raise RuntimeError("An official raw attachment changed during the read-only audit.")

    comparison = pd.DataFrame(
        [
            _comparison_row(
                "load_kw",
                attachment1["小区负载"].to_numpy(dtype=float),
                load_matrix.mean(axis=0),
            ),
            _comparison_row(
                "pv_kw",
                attachment1["光伏发电预测功率"].to_numpy(dtype=float),
                pv_matrix.mean(axis=0),
            ),
            _comparison_row(
                "price_yuan_per_kwh",
                attachment1["电价"].to_numpy(dtype=float),
                price_matrix.mean(axis=0),
            ),
        ]
    )
    datasets = {
        "attachment1": attachment1_audit,
        "attachment2_load": load_audit,
        "attachment2_pv": pv_audit,
        "attachment3_pv_forecast": forecast_audit,
        "attachment4_price": price_audit,
    }
    summary_rows = []
    for name, item in datasets.items():
        observed = int(item.get("observed_points", item.get("forecast_values", 0)))
        expected = int(item.get("expected_points", item.get("expected_forecast_values", 0)))
        summary_rows.append(
            {
                "dataset": name,
                "observed_points": observed,
                "expected_points": expected,
                "coverage_percent": 100.0 * observed / expected,
                "measurement_missing": int(item["measurement_missing"]),
                "negative_values": int(item["negative_values"]),
                "duplicate_timestamps": int(
                    item.get("duplicate_timestamps", item.get("duplicate_release_times", item.get("duplicate_slots", 0)))
                ),
                "unexpected_time_gaps": int(
                    item.get("non_ten_minute_gaps", item.get("non_six_hour_release_gaps", 0))
                ),
            }
        )

    report = {
        "stage": "step1_raw_data_quality_audit",
        "scope": "No cleaning, imputation, interpolation, standardization, windowing, or modeling.",
        "raw_directory": str(raw.relative_to(PROJECT_ROOT)),
        "workbook_sheets": workbook_sheets,
        "source_sha256": source_hashes_after,
        "raw_files_unchanged": True,
        "datasets": datasets,
        "attachment1_vs_annual_slot_mean": comparison.to_dict(orient="records"),
        "units": {
            "load": "kW",
            "pv_power": "kW",
            "electricity_price": "yuan/kWh",
            "dispatch_energy_output": "kWh",
        },
        "time_semantics": {
            "attachments_1_2_4": "10-minute interval-end timestamps",
            "end_marker": "0:00+1 is next-day 00:00 (24:00 of operating_date)",
            "attachment_3": "forecast releases every 6 hours with 24 hourly horizons",
        },
        "downstream_status": "pending_step2_transformation_and_alignment",
    }
    return DataQualityAudit(report, pd.DataFrame(summary_rows), comparison)
