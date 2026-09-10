"""Canonical, leakage-safe preprocessing for CUMCM 2026 Problem C."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from src.common.data_validation import file_sha256
from src.common.time_utils import (
    SLOTS_PER_DAY,
    TEN_MINUTES,
    interval_metadata,
    parse_interval_end_offset,
    validate_slot_labels,
    validate_timestamps,
)


DT_HOURS = 1.0 / 6.0
ONE_DAY_SLOTS = 144
SEVEN_DAY_SLOTS = 1008


@dataclass(frozen=True)
class CanonicalData:
    """Frames produced by the Stage 1A data contract."""

    problem1_day: pd.DataFrame
    actual_10min: pd.DataFrame
    pv_forecast_hourly: pd.DataFrame
    forecast_actual_alignment: pd.DataFrame
    pv_forecast_10min: pd.DataFrame
    alignment_report: dict[str, Any]
    interpolation_report: dict[str, Any]


def _duration_label(offset: pd.Timedelta) -> str:
    minutes = int(offset / pd.Timedelta(minutes=1))
    if minutes == 24 * 60:
        return "24:00"
    hour, minute = divmod(minutes, 60)
    return f"{hour:02d}:{minute:02d}"


def load_problem1_day(path: str | Path) -> pd.DataFrame:
    """Normalize Attachment 1 without statistical scaling."""
    source = Path(path)
    raw = pd.read_excel(source, sheet_name="Sheet1")
    expected = ["时间", "电价", "小区负载", "光伏发电预测功率"]
    if list(raw.columns) != expected:
        raise ValueError(f"Unexpected Attachment 1 columns: {list(raw.columns)}")
    offsets = validate_slot_labels(raw["时间"].tolist())
    numeric = raw[expected[1:]].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("Attachment 1 contains missing or non-numeric measurements")

    output = pd.DataFrame(
        {
            "slot": np.arange(1, SLOTS_PER_DAY + 1, dtype=int),
            "interval_start": [_duration_label(value - TEN_MINUTES) for value in offsets],
            "interval_end": [_duration_label(value) for value in offsets],
            "price_yuan_per_kwh": numeric["电价"].to_numpy(dtype=float),
            "load_kw": numeric["小区负载"].to_numpy(dtype=float),
            "pv_forecast_kw": numeric["光伏发电预测功率"].to_numpy(dtype=float),
        }
    )
    output["load_kwh"] = output["load_kw"] * DT_HOURS
    output["pv_forecast_kwh"] = output["pv_forecast_kw"] * DT_HOURS
    output["net_load_kw"] = output["load_kw"] - output["pv_forecast_kw"]
    output["net_load_kwh"] = output["net_load_kw"] * DT_HOURS
    return output


def wide_sheet_to_long(
    path: str | Path,
    *,
    sheet_name: str,
    value_name: str,
) -> pd.DataFrame:
    """Convert one date-by-slot sheet to a timestamped long table."""
    raw = pd.read_excel(path, sheet_name=sheet_name)
    if raw.shape[1] != SLOTS_PER_DAY + 1:
        raise ValueError(f"{Path(path).name}/{sheet_name} must have 144 slot columns")
    date_column = raw.columns[0]
    value_columns = list(raw.columns[1:])
    offsets = validate_slot_labels(value_columns)
    dates = pd.to_datetime(raw[date_column], errors="coerce").dt.normalize()
    if dates.isna().any() or dates.duplicated().any():
        raise ValueError(f"{Path(path).name}/{sheet_name} has invalid or duplicate dates")
    values = raw[value_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    operating_date = pd.DatetimeIndex(np.repeat(dates.to_numpy(), SLOTS_PER_DAY))
    slot = np.tile(np.arange(1, SLOTS_PER_DAY + 1, dtype=int), len(dates))
    interval_end = pd.DatetimeIndex(
        operating_date.to_numpy(dtype="datetime64[ns]")
        + np.tile(offsets.to_numpy(dtype="timedelta64[ns]"), len(dates))
    )
    output = pd.DataFrame(
        {
            "operating_date": operating_date,
            "slot": slot,
            "interval_start": interval_end - TEN_MINUTES,
            "interval_end": interval_end,
            value_name: values.reshape(-1),
        }
    )
    if output[["operating_date", "slot", "interval_end"]].duplicated().any():
        raise ValueError(f"Duplicate long-table keys in {Path(path).name}/{sheet_name}")
    return output


def _merge_one_to_one(left: pd.DataFrame, right: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Outer merge that retains missing values and rejects duplicate keys."""
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        raise ValueError(f"Duplicate merge keys: {keys}")
    return left.merge(right, on=keys, how="outer", validate="one_to_one", sort=True)


def add_causal_history_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add fixed historical lags using only rows strictly earlier than each row."""
    output = frame.sort_values("interval_end", kind="stable").reset_index(drop=True).copy()
    output["pv_lag_1d"] = output["pv_actual_kw"].shift(ONE_DAY_SLOTS)
    output["pv_lag_7d"] = output["pv_actual_kw"].shift(SEVEN_DAY_SLOTS)
    output["load_lag_1d"] = output["load_kw"].shift(ONE_DAY_SLOTS)
    output["load_lag_7d"] = output["load_kw"].shift(SEVEN_DAY_SLOTS)
    output["pv_same_slot_7d_mean"] = output.groupby("slot", sort=False)[
        "pv_actual_kw"
    ].transform(lambda values: values.shift(1).rolling(7, min_periods=7).mean())
    return output


def load_actual_10min(attachment2: str | Path, attachment4: str | Path) -> pd.DataFrame:
    """Build the complete 10-minute actual load, PV and price table."""
    load = wide_sheet_to_long(attachment2, sheet_name="小区负载", value_name="load_kw")
    pv = wide_sheet_to_long(
        attachment2,
        sheet_name="光伏发电实际功率",
        value_name="pv_actual_kw",
    )
    price = wide_sheet_to_long(
        attachment4,
        sheet_name="Sheet1",
        value_name="price_yuan_per_kwh",
    )
    keys = ["operating_date", "slot", "interval_start", "interval_end"]
    merged = _merge_one_to_one(load, pv, keys)
    merged = _merge_one_to_one(merged, price, keys)
    merged = merged.sort_values("interval_end", kind="stable").reset_index(drop=True)
    initial_check = validate_timestamps(merged["interval_end"], frequency="10min")
    if initial_check["invalid_count"] or initial_check["duplicate_count"]:
        raise ValueError("Invalid or duplicate interval_end before frequency completion")

    complete_index = pd.date_range(
        merged["interval_end"].min(), merged["interval_end"].max(), freq="10min"
    )
    values = (
        merged.set_index("interval_end")[["load_kw", "pv_actual_kw", "price_yuan_per_kwh"]]
        .reindex(complete_index)
        .rename_axis("interval_end")
        .reset_index()
    )
    metadata = interval_metadata(values["interval_end"])
    output = metadata.merge(values, on="interval_end", validate="one_to_one")
    output["load_kwh"] = output["load_kw"] * DT_HOURS
    output["pv_actual_kwh"] = output["pv_actual_kw"] * DT_HOURS
    output["net_load_kw"] = output["load_kw"] - output["pv_actual_kw"]
    output["net_load_kwh"] = output["net_load_kw"] * DT_HOURS
    return add_causal_history_features(output)


def load_pv_forecast_hourly(path: str | Path) -> pd.DataFrame:
    """Convert Attachment 3 to one row per release and forecast horizon."""
    raw = pd.read_excel(path, sheet_name="Sheet1")
    horizon_columns = [f"预报{hour}小时" for hour in range(1, 25)]
    expected = ["日期", "预报时刻", *horizon_columns]
    if list(raw.columns) != expected:
        raise ValueError("Unexpected Attachment 3 columns")
    release_dates = pd.to_datetime(raw["日期"].ffill(), errors="coerce").dt.normalize()
    if release_dates.isna().any():
        raise ValueError("Attachment 3 contains an invalid release date")
    release_offset = raw["预报时刻"].map(parse_interval_end_offset)
    release_time = release_dates + pd.to_timedelta(release_offset)
    releases = raw.assign(release_time=release_time)
    long = releases.melt(
        id_vars=["release_time"],
        value_vars=horizon_columns,
        var_name="horizon_label",
        value_name="pv_forecast_kw",
    )
    long["horizon_hour"] = long["horizon_label"].str.extract(r"(\d+)").astype(int)
    long["pv_forecast_kw"] = pd.to_numeric(long["pv_forecast_kw"], errors="coerce")
    long["target_time"] = long["release_time"] + pd.to_timedelta(
        long["horizon_hour"], unit="h"
    )
    long = long.sort_values(["release_time", "horizon_hour"], kind="stable").reset_index(drop=True)
    if long.duplicated(["release_time", "horizon_hour"]).any():
        raise ValueError("Attachment 3 has duplicate release/horizon keys")
    long.insert(0, "forecast_id", np.arange(1, len(long) + 1, dtype=int))
    return long[
        ["forecast_id", "release_time", "horizon_hour", "target_time", "pv_forecast_kw"]
    ]


def _lookup_actual(
    actual_lookup: pd.Series,
    timestamps: pd.Series | pd.DatetimeIndex,
) -> np.ndarray:
    """Use a verified unique index instead of a many-to-one merge."""
    return actual_lookup.reindex(pd.DatetimeIndex(timestamps)).to_numpy(dtype=float)


def align_forecast_to_actual(
    forecast: pd.DataFrame,
    actual: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Align every forecast record to actual PV at its exact target time."""
    if actual["interval_end"].duplicated().any():
        raise ValueError("Actual PV lookup timestamps must be unique")
    actual_lookup = actual.set_index("interval_end")["pv_actual_kw"].sort_index()
    aligned = forecast.copy()
    aligned["pv_actual_kw"] = _lookup_actual(actual_lookup, aligned["target_time"])
    below = aligned["target_time"] < actual_lookup.index.min()
    above = aligned["target_time"] > actual_lookup.index.max()
    missing = aligned["pv_actual_kw"].isna()
    aligned["alignment_status"] = np.where(missing, "unmatched", "matched")
    aligned["unmatched_reason"] = ""
    aligned.loc[missing & (below | above), "unmatched_reason"] = "out_of_actual_range"
    aligned.loc[missing & ~(below | above), "unmatched_reason"] = "missing_actual_value"
    aligned["absolute_error_kw"] = (
        aligned["pv_forecast_kw"] - aligned["pv_actual_kw"]
    ).abs()

    candidate_actual: dict[str, np.ndarray] = {}
    for label, offset in {
        "target_minus_1h": pd.Timedelta(hours=-1),
        "target_time": pd.Timedelta(0),
        "target_plus_1h": pd.Timedelta(hours=1),
    }.items():
        candidate_actual[label] = _lookup_actual(
            actual_lookup, aligned["target_time"] + offset
        )
    common = np.isfinite(aligned["pv_forecast_kw"].to_numpy(dtype=float))
    for values in candidate_actual.values():
        common &= np.isfinite(values)
    forecast_values = aligned["pv_forecast_kw"].to_numpy(dtype=float)
    mae = {
        label: float(np.mean(np.abs(forecast_values[common] - values[common])))
        for label, values in candidate_actual.items()
    }
    minimum_mae_alignment = min(mae, key=mae.get)
    report = {
        "raw_forecast_count": int(len(aligned)),
        "aligned_count": int((aligned["alignment_status"] == "matched").sum()),
        "unmatched_count": int((aligned["alignment_status"] == "unmatched").sum()),
        "unmatched_reasons": {
            str(key): int(value)
            for key, value in aligned.loc[
                aligned["alignment_status"] == "unmatched", "unmatched_reason"
            ]
            .value_counts()
            .items()
        },
        "sanity_common_count": int(common.sum()),
        "mae_kw": mae,
        "minimum_mae_alignment": minimum_mae_alignment,
        "selected_alignment": "target_time = release_time + horizon_hour",
    }
    return aligned, report


def interpolate_forecast_batches(
    forecast: pd.DataFrame,
    actual: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Interpolate each release independently to 10 minutes with causal anchors."""
    if actual["interval_end"].duplicated().any():
        raise ValueError("Actual PV lookup timestamps must be unique")
    actual_lookup = actual.set_index("interval_end")["pv_actual_kw"].sort_index()
    target_minutes = np.arange(10, 24 * 60 + 1, 10, dtype=int)
    target_hours = target_minutes / 60.0
    batches: list[pd.DataFrame] = []
    method_counts: dict[str, int] = {}
    anchor_count = 0

    for release_time, batch in forecast.groupby("release_time", sort=True):
        ordered = batch.sort_values("horizon_hour", kind="stable")
        hours = ordered["horizon_hour"].to_numpy(dtype=float)
        hourly = ordered["pv_forecast_kw"].to_numpy(dtype=float)
        anchor = actual_lookup.get(pd.Timestamp(release_time), np.nan)
        anchor_used = bool(np.isfinite(anchor))
        if anchor_used:
            anchor_count += 1
            x = np.concatenate([[0.0], hours])
            y = np.concatenate([[float(anchor)], hourly])
        else:
            x, y = hours.copy(), hourly.copy()
        valid = np.isfinite(y)
        x_valid, y_valid = x[valid], y[valid]

        method = "pchip"
        values = np.full(len(target_hours), np.nan, dtype=float)
        try:
            if len(x_valid) < 2:
                raise ValueError("fewer than two interpolation nodes")
            values = PchipInterpolator(x_valid, y_valid, extrapolate=False)(target_hours)
        except (ValueError, FloatingPointError):
            method = "linear"
            if len(x_valid) >= 2:
                values = np.interp(
                    target_hours,
                    x_valid,
                    y_valid,
                    left=np.nan,
                    right=np.nan,
                )
        values = np.where(np.isfinite(values), np.maximum(values, 0.0), np.nan)

        hourly_node = target_minutes % 60 == 0
        hourly_index = target_minutes[hourly_node] // 60 - 1
        raw_nodes = hourly[hourly_index]
        values[hourly_node] = np.where(
            np.isfinite(raw_nodes), np.maximum(raw_nodes, 0.0), np.nan
        )
        method_counts[method] = method_counts.get(method, 0) + 1
        batches.append(
            pd.DataFrame(
                {
                    "release_time": pd.Timestamp(release_time),
                    "horizon_step_10min": np.arange(1, SLOTS_PER_DAY + 1, dtype=int),
                    "target_time": pd.Timestamp(release_time)
                    + pd.to_timedelta(target_minutes, unit="m"),
                    "pv_forecast_10min_kw": values,
                    "is_original_hourly_node": hourly_node,
                    "actual_anchor_used": anchor_used,
                    "interpolation_method": method,
                }
            )
        )
    output = pd.concat(batches, ignore_index=True)
    node_rows = output[output["is_original_hourly_node"]]
    hourly_lookup = forecast.set_index(["release_time", "target_time"])["pv_forecast_kw"]
    original = hourly_lookup.reindex(
        pd.MultiIndex.from_frame(node_rows[["release_time", "target_time"]])
    ).to_numpy(dtype=float)
    preserved = np.isclose(
        node_rows["pv_forecast_10min_kw"].to_numpy(dtype=float),
        np.maximum(original, 0.0),
        equal_nan=True,
    )
    report = {
        "rows": int(len(output)),
        "release_batches": int(output["release_time"].nunique()),
        "rows_per_batch": SLOTS_PER_DAY,
        "missing_values": int(output["pv_forecast_10min_kw"].isna().sum()),
        "anchor_batches": int(anchor_count),
        "method_counts": method_counts,
        "hourly_nodes_checked": int(len(node_rows)),
        "hourly_nodes_preserved": bool(preserved.all()),
        "future_actual_used": False,
    }
    return output, report


def build_canonical_data(raw_dir: str | Path) -> CanonicalData:
    """Read only the four official workbooks and build all Stage 1A frames."""
    raw = Path(raw_dir)
    paths = {index: raw / f"附件{index}.xlsx" for index in range(1, 5)}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing official attachments:\n" + "\n".join(missing))
    before = {path.name: file_sha256(path) for path in paths.values()}
    problem1 = load_problem1_day(paths[1])
    actual = load_actual_10min(paths[2], paths[4])
    forecast = load_pv_forecast_hourly(paths[3])
    alignment, alignment_report = align_forecast_to_actual(forecast, actual)
    interpolated, interpolation_report = interpolate_forecast_batches(forecast, actual)
    after = {path.name: file_sha256(path) for path in paths.values()}
    if before != after:
        raise RuntimeError("An official raw workbook changed during preprocessing")
    return CanonicalData(
        problem1_day=problem1,
        actual_10min=actual,
        pv_forecast_hourly=forecast,
        forecast_actual_alignment=alignment,
        pv_forecast_10min=interpolated,
        alignment_report=alignment_report,
        interpolation_report=interpolation_report,
    )
