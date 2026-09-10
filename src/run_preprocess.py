"""Run the complete Stage 1A canonical data pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.data_utils import save_table
from src.common.data_validation import RAW_C_DIR, audit_all_attachments, file_sha256
from src.common.paths import PROCESSED_DATA_DIR, PROJECT_ROOT
from src.common.preprocessing import build_canonical_data
from src.common.time_utils import validate_timestamps
from src.common.windowing import summarize_windows


OUTPUT_DIR = PROCESSED_DATA_DIR / "C题"


def _frame_shape(frame: pd.DataFrame) -> list[int]:
    return [int(frame.shape[0]), int(frame.shape[1])]


def _measurement_summary(frame: pd.DataFrame, columns: list[str]) -> dict[str, int]:
    return {column: int(frame[column].isna().sum()) for column in columns}


def _build_report(data: Any, raw_audit: Any, hashes_before: dict[str, str]) -> dict[str, Any]:
    actual_time = validate_timestamps(
        data.actual_10min["interval_end"],
        frequency="10min",
        expected_start="2025-01-01 00:10:00",
        expected_end="2026-01-01 00:00:00",
    )
    daily = data.actual_10min.groupby("operating_date", sort=True).size()
    release_horizons = data.pv_forecast_hourly.groupby("release_time", sort=True).size()
    releases_per_day = (
        data.pv_forecast_hourly[["release_time"]]
        .drop_duplicates()
        .assign(release_date=lambda x: x["release_time"].dt.normalize())
        .groupby("release_date")
        .size()
    )
    window_summary = summarize_windows(
        data.actual_10min,
        feature_columns=[
            "load_kw",
            "pv_actual_kw",
            "price_yuan_per_kwh",
            "pv_lag_1d",
            "pv_lag_7d",
            "pv_same_slot_7d_mean",
            "load_lag_1d",
            "load_lag_7d",
        ],
        target_columns=["pv_actual_kw"],
        history_steps=1008,
        horizon_steps=144,
    )
    hashes_after = {
        path.name: file_sha256(path)
        for path in sorted(RAW_C_DIR.glob("附件[1-4].xlsx"))
    }
    raw_files = {
        str((RAW_C_DIR / name).relative_to(PROJECT_ROOT)): {
            "sha256": digest,
            "unchanged": hashes_before.get(name) == digest,
        }
        for name, digest in hashes_after.items()
    }
    return {
        "stage": "Stage 1A canonical data pipeline",
        "raw_directory": str(RAW_C_DIR.relative_to(PROJECT_ROOT)),
        "raw_files": raw_files,
        "raw_files_unchanged": hashes_before == hashes_after,
        "workbook_sheets": raw_audit.report["workbook_sheets"],
        "raw_sheet_profiles": raw_audit.report["datasets"],
        "processed": {
            "problem1_day": {
                "path": "data/processed/C题/problem1_day.csv",
                "shape": _frame_shape(data.problem1_day),
                "slot_start": int(data.problem1_day["slot"].min()),
                "slot_end": int(data.problem1_day["slot"].max()),
                "interval_end_start": str(data.problem1_day["interval_end"].iloc[0]),
                "interval_end_end": str(data.problem1_day["interval_end"].iloc[-1]),
                "missing": _measurement_summary(
                    data.problem1_day,
                    ["price_yuan_per_kwh", "load_kw", "pv_forecast_kw"],
                ),
                "duplicate_slot_count": int(data.problem1_day["slot"].duplicated().sum()),
            },
            "actual_10min": {
                "path": "data/processed/C题/actual_10min.csv",
                "shape": _frame_shape(data.actual_10min),
                "time_range": [
                    str(data.actual_10min["interval_end"].min()),
                    str(data.actual_10min["interval_end"].max()),
                ],
                "timestamp_validation": actual_time,
                "missing_measurements": _measurement_summary(
                    data.actual_10min,
                    ["load_kw", "pv_actual_kw", "price_yuan_per_kwh"],
                ),
                "duplicate_interval_end_count": int(
                    data.actual_10min["interval_end"].duplicated().sum()
                ),
                "daily_records": {
                    "days": int(daily.size),
                    "minimum": int(daily.min()),
                    "maximum": int(daily.max()),
                },
                "negative_net_load_count": int((data.actual_10min["net_load_kw"] < 0).sum()),
            },
            "pv_forecast_hourly": {
                "path": "data/processed/C题/pv_forecast_hourly.csv",
                "shape": _frame_shape(data.pv_forecast_hourly),
                "release_time_range": [
                    str(data.pv_forecast_hourly["release_time"].min()),
                    str(data.pv_forecast_hourly["release_time"].max()),
                ],
                "target_time_range": [
                    str(data.pv_forecast_hourly["target_time"].min()),
                    str(data.pv_forecast_hourly["target_time"].max()),
                ],
                "missing_forecasts": int(data.pv_forecast_hourly["pv_forecast_kw"].isna().sum()),
                "duplicate_release_horizon_count": int(
                    data.pv_forecast_hourly.duplicated(
                        ["release_time", "horizon_hour"]
                    ).sum()
                ),
                "releases_per_day": {
                    "minimum": int(releases_per_day.min()),
                    "maximum": int(releases_per_day.max()),
                },
                "horizons_per_release": {
                    "minimum": int(release_horizons.min()),
                    "maximum": int(release_horizons.max()),
                },
            },
            "forecast_actual_alignment": {
                "path": "data/processed/C题/forecast_actual_alignment.csv",
                "shape": _frame_shape(data.forecast_actual_alignment),
                **data.alignment_report,
            },
            "pv_forecast_10min": {
                "path": "data/processed/C题/pv_forecast_10min.csv",
                "shape": _frame_shape(data.pv_forecast_10min),
                **data.interpolation_report,
            },
        },
        "window_validation": {
            "history_steps": 1008,
            "horizon_steps": 144,
            **window_summary,
        },
        "units_and_time_rules": {
            "timestamp_semantics": "10-minute interval end",
            "slot_range": [1, 144],
            "dt_hours": 1 / 6,
            "energy_conversion": "kWh = kW / 6",
            "net_load": "load_kw - pv_actual_kw; negative values retained",
        },
    }


def main() -> None:
    raw_paths = sorted(RAW_C_DIR.glob("附件[1-4].xlsx"))
    if len(raw_paths) != 4:
        raise FileNotFoundError("Exactly four official raw workbooks are required")
    hashes_before = {path.name: file_sha256(path) for path in raw_paths}
    raw_audit = audit_all_attachments(RAW_C_DIR)
    data = build_canonical_data(RAW_C_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_table(data.problem1_day, OUTPUT_DIR / "problem1_day.csv", float_format="%.10g")
    save_table(data.actual_10min, OUTPUT_DIR / "actual_10min.csv", float_format="%.10g")
    save_table(
        data.pv_forecast_hourly,
        OUTPUT_DIR / "pv_forecast_hourly.csv",
        float_format="%.10g",
    )
    save_table(
        data.forecast_actual_alignment,
        OUTPUT_DIR / "forecast_actual_alignment.csv",
        float_format="%.10g",
    )
    save_table(
        data.pv_forecast_10min,
        OUTPUT_DIR / "pv_forecast_10min.csv",
        float_format="%.10g",
    )

    report = _build_report(data, raw_audit, hashes_before)
    report_path = OUTPUT_DIR / "data_quality_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Stage 1A complete.")
    print(json.dumps(report["processed"], ensure_ascii=False, indent=2))
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
