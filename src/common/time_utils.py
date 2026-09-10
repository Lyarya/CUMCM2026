"""Time semantics shared by the C-problem data pipeline."""

from __future__ import annotations

from datetime import time
from typing import Iterable

import numpy as np
import pandas as pd


TEN_MINUTES = pd.Timedelta(minutes=10)
SLOTS_PER_DAY = 144


def parse_interval_end_offset(value: object) -> pd.Timedelta:
    """Parse a slot label while preserving ``0:00+1`` as next-day midnight."""
    if isinstance(value, time):
        return pd.Timedelta(hours=value.hour, minutes=value.minute, seconds=value.second)

    text = str(value).strip()
    next_day = text.endswith("+1")
    if next_day:
        text = text[:-2]
    parts = text.split(":")
    if len(parts) not in {2, 3}:
        raise ValueError(f"Unsupported time label: {value!r}")
    hour, minute = int(parts[0]), int(parts[1])
    second = int(parts[2]) if len(parts) == 3 else 0
    if not 0 <= hour <= 23 or not 0 <= minute <= 59 or not 0 <= second <= 59:
        raise ValueError(f"Invalid clock time: {value!r}")
    if next_day and (hour, minute, second) != (0, 0, 0):
        raise ValueError(f"Only midnight may use the +1 marker: {value!r}")
    return pd.Timedelta(days=int(next_day), hours=hour, minutes=minute, seconds=second)


def validate_slot_labels(labels: Iterable[object]) -> pd.TimedeltaIndex:
    """Validate the ordered 144 interval ends from 00:10 through next midnight."""
    offsets = pd.TimedeltaIndex([parse_interval_end_offset(label) for label in labels])
    expected = pd.timedelta_range(TEN_MINUTES, periods=SLOTS_PER_DAY, freq=TEN_MINUTES)
    if not offsets.equals(expected):
        raise ValueError(
            "Slot labels must be the ordered 10-minute interval ends from "
            "00:10 through 0:00+1."
        )
    return offsets


def validate_timestamps(
    timestamps: Iterable[object],
    *,
    frequency: str | pd.Timedelta = "10min",
    expected_start: object | None = None,
    expected_end: object | None = None,
) -> dict[str, object]:
    """Validate uniqueness, order, frequency and optional endpoints.

    The function never drops or repairs timestamps.  It returns counts so callers
    can decide whether a complete-frequency reindex is appropriate.
    """
    index = pd.DatetimeIndex(pd.to_datetime(list(timestamps), errors="coerce"))
    invalid_count = int(index.isna().sum())
    duplicate_count = int(index.duplicated().sum()) if not invalid_count else 0
    valid = index[~index.isna()]
    monotonic = bool(valid.is_monotonic_increasing)
    delta = pd.Timedelta(frequency)
    diffs = valid.to_series(index=np.arange(len(valid))).diff().iloc[1:]
    non_frequency_count = int((diffs != delta).sum())

    if expected_start is not None and len(valid):
        if valid[0] != pd.Timestamp(expected_start):
            raise ValueError(f"Unexpected first timestamp: {valid[0]}")
    if expected_end is not None and len(valid):
        if valid[-1] != pd.Timestamp(expected_end):
            raise ValueError(f"Unexpected last timestamp: {valid[-1]}")
    return {
        "count": int(len(index)),
        "invalid_count": invalid_count,
        "duplicate_count": duplicate_count,
        "non_frequency_count": non_frequency_count,
        "is_monotonic_increasing": monotonic,
        "start": str(valid[0]) if len(valid) else None,
        "end": str(valid[-1]) if len(valid) else None,
        "frequency": str(delta),
    }


def interval_metadata(interval_end: pd.Series | pd.DatetimeIndex) -> pd.DataFrame:
    """Derive operating date, slot and interval start from interval-end times."""
    ends = pd.DatetimeIndex(pd.to_datetime(interval_end))
    starts = ends - TEN_MINUTES
    operating_dates = starts.normalize()
    slot = ((ends - operating_dates) / TEN_MINUTES).astype(int)
    return pd.DataFrame(
        {
            "operating_date": operating_dates,
            "slot": slot,
            "interval_start": starts,
            "interval_end": ends,
        }
    )
