"""Daily view of the Stage 1 canonical ten-minute data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DailyPanel:
    """Aligned daily arrays with exactly 144 interval-end observations per day."""

    dates: pd.DatetimeIndex
    datetimes: np.ndarray
    load_kw: np.ndarray
    generation_kw: np.ndarray

    @property
    def n_days(self) -> int:
        return len(self.dates)


def load_daily_panel(path: str | Path, *, day_steps: int = 144) -> DailyPanel:
    """Read canonical data only and validate its daily/timestamp contract."""
    frame = pd.read_csv(path, parse_dates=["operating_date", "interval_end"])
    required = {"operating_date", "slot", "interval_end", "load_kw", "pv_actual_kw"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"canonical data is missing columns: {sorted(missing)}")
    frame = frame.sort_values(["operating_date", "slot"], kind="stable").reset_index(drop=True)
    if frame.duplicated(["operating_date", "slot"]).any():
        raise ValueError("operating_date/slot keys must be unique")
    counts = frame.groupby("operating_date", sort=True).size()
    if counts.empty or not counts.eq(day_steps).all():
        raise ValueError(f"every operating day must contain exactly {day_steps} rows")
    expected_slots = np.tile(np.arange(1, day_steps + 1), len(counts))
    if not np.array_equal(frame["slot"].to_numpy(dtype=int), expected_slots):
        raise ValueError("slot must run from 1 to day_steps for every day")
    if frame[["load_kw", "pv_actual_kw"]].isna().any().any():
        raise ValueError("Stage 2A requires complete canonical load and generation observations")
    timestamps = pd.DatetimeIndex(frame["interval_end"])
    if timestamps.has_duplicates:
        raise ValueError("interval_end must be unique")
    if not np.all(np.diff(timestamps.asi8) == pd.Timedelta(minutes=10).value):
        raise ValueError("interval_end must remain globally continuous at ten-minute resolution")

    dates = pd.DatetimeIndex(counts.index).normalize()
    expected_end = np.concatenate(
        [
            pd.date_range(day + pd.Timedelta(minutes=10), periods=day_steps, freq="10min")
            .to_numpy(dtype="datetime64[ns]")
            for day in dates
        ]
    )
    if not np.array_equal(timestamps.to_numpy(dtype="datetime64[ns]"), expected_end):
        raise ValueError("daily interval-end timestamps are not aligned with operating_date")
    shape = (len(dates), day_steps)
    return DailyPanel(
        dates=dates,
        datetimes=timestamps.to_numpy(dtype="datetime64[ns]").reshape(shape),
        load_kw=frame["load_kw"].to_numpy(dtype=float).reshape(shape),
        generation_kw=frame["pv_actual_kw"].to_numpy(dtype=float).reshape(shape),
    )


def date_indices(panel: DailyPanel, start: object, end: object) -> np.ndarray:
    """Return chronologically ordered daily indices inside an inclusive date range."""
    lower = pd.Timestamp(start)
    upper = pd.Timestamp(end)
    return np.flatnonzero((panel.dates >= lower) & (panel.dates <= upper))
