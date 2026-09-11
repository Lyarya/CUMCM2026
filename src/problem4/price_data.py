"""Canonical Attachment-4 price data and source-integrity audit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.common.data_validation import file_sha256
from src.common.paths import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.common.preprocessing import wide_sheet_to_long
from src.common.time_utils import validate_timestamps


RAW_ATTACHMENT4 = RAW_DATA_DIR / "C题" / "附件" / "附件4.xlsx"
ACTUAL_PATH = PROCESSED_DATA_DIR / "C题" / "actual_10min.csv"
CANONICAL_PATH = INTERIM_DATA_DIR / "C题" / "q4_price_canonical.csv"
EXPECTED_ROWS = 365 * 144
PRICE_UNIT = "yuan/kWh"


def build_price_canonical() -> pd.DataFrame:
    """Read Attachment 4 without mutation and align it to the operational grid."""

    raw = wide_sheet_to_long(
        RAW_ATTACHMENT4,
        sheet_name="Sheet1",
        value_name="attachment4_price_yuan_per_kwh",
    )
    actual = pd.read_csv(
        ACTUAL_PATH,
        parse_dates=["operating_date", "interval_start", "interval_end"],
    )
    columns = [
        "operating_date",
        "slot",
        "interval_start",
        "interval_end",
        "load_kw",
        "pv_actual_kw",
        "net_load_kw",
        "price_yuan_per_kwh",
    ]
    if actual["interval_end"].duplicated().any():
        raise AssertionError("Canonical operational timestamps must be unique")
    merged = raw.merge(
        actual[columns],
        on=["operating_date", "slot", "interval_start", "interval_end"],
        how="outer",
        validate="one_to_one",
        indicator=True,
        sort=True,
    )
    if not merged["_merge"].eq("both").all():
        raise AssertionError("Attachment 4 does not align one-to-one with actual_10min")
    price_gap = (
        merged["attachment4_price_yuan_per_kwh"] - merged["price_yuan_per_kwh"]
    ).abs()
    if not np.allclose(
        merged["attachment4_price_yuan_per_kwh"],
        merged["price_yuan_per_kwh"], rtol=0, atol=1e-12, equal_nan=True,
    ):
        raise AssertionError("Attachment 4 price differs from the Stage-1 canonical price")

    output = merged.drop(columns=["price_yuan_per_kwh", "_merge"]).rename(
        columns={"attachment4_price_yuan_per_kwh": "price_yuan_per_kwh"}
    )
    output = output.sort_values("interval_end", kind="stable").reset_index(drop=True)
    output["month"] = output["operating_date"].dt.month.astype(int)
    output["weekday"] = output["operating_date"].dt.weekday.astype(int)
    output["hour"] = output["interval_start"].dt.hour.astype(int)
    output["minute"] = output["interval_start"].dt.minute.astype(int)
    output.attrs["raw_to_stage1_max_absolute_price_gap"] = float(price_gap.max())
    return output


def audit_attachment4(frame: pd.DataFrame | None = None) -> dict[str, object]:
    """Return a complete audit without deleting or repairing any price value."""

    data = build_price_canonical() if frame is None else frame.copy()
    with pd.ExcelFile(RAW_ATTACHMENT4) as workbook:
        sheet_names = list(workbook.sheet_names)
        raw = pd.read_excel(workbook, sheet_name="Sheet1")
    timestamp_audit = validate_timestamps(data["interval_end"], frequency="10min")
    prices = data["price_yuan_per_kwh"].astype(float)
    q1, q3 = prices.quantile([0.25, 0.75])
    iqr = q3 - q1
    z_score = (prices - prices.mean()) / prices.std(ddof=1)
    day_counts = data.groupby("operating_date", sort=True).size()
    exact_start = pd.Timestamp("2025-01-01 00:10:00")
    exact_end = pd.Timestamp("2026-01-01 00:00:00")
    alignment_pass = bool(
        len(data) == EXPECTED_ROWS
        and data["interval_end"].min() == exact_start
        and data["interval_end"].max() == exact_end
        and timestamp_audit["duplicate_count"] == 0
        and timestamp_audit["invalid_count"] == 0
        and timestamp_audit["is_monotonic_increasing"]
        and timestamp_audit["non_frequency_count"] == 0
        and day_counts.eq(144).all()
    )
    return {
        "source_path": str(RAW_ATTACHMENT4),
        "source_sha256": file_sha256(RAW_ATTACHMENT4),
        "sheet": "Sheet1",
        "sheet_names": sheet_names,
        "raw_shape": list(raw.shape),
        "raw_missing_cells": int(raw.iloc[:, 1:].isna().sum().sum()),
        "price_unit": PRICE_UNIT,
        "observed_points": int(len(data)),
        "date_start": data["operating_date"].min().date().isoformat(),
        "date_end": data["operating_date"].max().date().isoformat(),
        "interval_end_start": str(data["interval_end"].min()),
        "interval_end_end": str(data["interval_end"].max()),
        "temporal_resolution": "10min interval-end timestamps",
        "missing_price_count": int(prices.isna().sum()),
        "duplicate_timestamp_count": int(data["interval_end"].duplicated().sum()),
        "non_10min_gap_count": int(timestamp_audit["non_frequency_count"]),
        "invalid_timestamp_count": int(timestamp_audit["invalid_count"]),
        "missing_timestamp_count": int(len(pd.date_range(exact_start, exact_end, freq="10min").difference(data["interval_end"]))),
        "minimum_records_per_day": int(day_counts.min()),
        "maximum_records_per_day": int(day_counts.max()),
        "negative_price_count": int(prices.lt(0).sum()),
        "zero_price_count": int(prices.eq(0).sum()),
        "iqr_upper_extreme_count": int(prices.gt(q3 + 1.5 * iqr).sum()),
        "iqr_lower_extreme_count": int(prices.lt(q1 - 1.5 * iqr).sum()),
        "absolute_z_gt_3_count": int(z_score.abs().gt(3).sum()),
        "minimum_price": float(prices.min()),
        "maximum_price": float(prices.max()),
        "timestamp_alignment_pass": alignment_pass,
        "raw_to_stage1_max_absolute_price_gap": data.attrs.get("raw_to_stage1_max_absolute_price_gap"),
    }


def load_price_canonical(path: str | Path = CANONICAL_PATH) -> pd.DataFrame:
    """Load the persisted optimizer-independent price table."""

    return pd.read_csv(
        path,
        parse_dates=["operating_date", "interval_start", "interval_end"],
    )


__all__ = [
    "ACTUAL_PATH",
    "CANONICAL_PATH",
    "EXPECTED_ROWS",
    "PRICE_UNIT",
    "RAW_ATTACHMENT4",
    "audit_attachment4",
    "build_price_canonical",
    "load_price_canonical",
]
