"""Leakage-safe timestamp-aware window construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WindowBatch:
    """Window arrays with masks created before NaNs are filled by zero."""

    inputs: np.ndarray
    targets: np.ndarray
    input_mask: np.ndarray
    target_mask: np.ndarray
    valid_windows: np.ndarray
    input_start: pd.DatetimeIndex
    target_end: pd.DatetimeIndex

    @property
    def valid_count(self) -> int:
        return int(self.valid_windows.sum())

    @property
    def invalid_count(self) -> int:
        return int((~self.valid_windows).sum())


def build_windows(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    target_columns: Sequence[str],
    history_steps: int,
    horizon_steps: int,
    timestamp_column: str = "interval_end",
    frequency: str | pd.Timedelta = "10min",
) -> WindowBatch:
    """Build chronological windows without crossing missing timestamp intervals."""
    if history_steps <= 0 or horizon_steps <= 0:
        raise ValueError("history_steps and horizon_steps must be positive")
    ordered = frame.sort_values(timestamp_column, kind="stable").reset_index(drop=True)
    timestamps = pd.DatetimeIndex(pd.to_datetime(ordered[timestamp_column]))
    if timestamps.has_duplicates:
        raise ValueError("Window timestamps must be unique")
    n_windows = len(ordered) - history_steps - horizon_steps + 1
    if n_windows <= 0:
        raise ValueError("Not enough rows for the requested window")

    features = ordered[list(feature_columns)].to_numpy(dtype=float)
    targets = ordered[list(target_columns)].to_numpy(dtype=float)
    input_mask_all = np.isfinite(features)
    target_mask_all = np.isfinite(targets)
    feature_filled = np.where(input_mask_all, features, 0.0)
    target_filled = np.where(target_mask_all, targets, 0.0)

    inputs = np.stack(
        [feature_filled[start : start + history_steps] for start in range(n_windows)]
    )
    outputs = np.stack(
        [
            target_filled[
                start + history_steps : start + history_steps + horizon_steps
            ]
            for start in range(n_windows)
        ]
    )
    input_masks = np.stack(
        [input_mask_all[start : start + history_steps] for start in range(n_windows)]
    )
    target_masks = np.stack(
        [
            target_mask_all[
                start + history_steps : start + history_steps + horizon_steps
            ]
            for start in range(n_windows)
        ]
    )

    expected = pd.Timedelta(frequency).value
    # Pandas 3 may retain datetime64[us], while Timedelta.value is always ns.
    # Normalize explicitly so continuity checks do not depend on array resolution.
    time_ns = timestamps.to_numpy(dtype="datetime64[ns]").astype("int64")
    continuity = np.array(
        [
            np.all(
                np.diff(time_ns[start : start + history_steps + horizon_steps])
                == expected
            )
            for start in range(n_windows)
        ],
        dtype=bool,
    )
    valid = continuity & input_masks.all(axis=(1, 2)) & target_masks.all(axis=(1, 2))
    return WindowBatch(
        inputs=inputs,
        targets=outputs,
        input_mask=input_masks,
        target_mask=target_masks,
        valid_windows=valid,
        input_start=timestamps[:n_windows],
        target_end=timestamps[history_steps + horizon_steps - 1 :],
    )


def summarize_windows(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    target_columns: Sequence[str],
    history_steps: int,
    horizon_steps: int,
    timestamp_column: str = "interval_end",
    frequency: str | pd.Timedelta = "10min",
) -> dict[str, int | float]:
    """Count valid windows without materializing a year of overlapping arrays."""
    ordered = frame.sort_values(timestamp_column, kind="stable").reset_index(drop=True)
    n_windows = len(ordered) - history_steps - horizon_steps + 1
    if n_windows <= 0:
        raise ValueError("Not enough rows for the requested window")
    timestamps = pd.DatetimeIndex(pd.to_datetime(ordered[timestamp_column]))
    if timestamps.has_duplicates:
        raise ValueError("Window timestamps must be unique")

    feature_rows = np.isfinite(ordered[list(feature_columns)].to_numpy(dtype=float)).all(axis=1)
    target_rows = np.isfinite(ordered[list(target_columns)].to_numpy(dtype=float)).all(axis=1)
    history_valid_all = (
        np.convolve(feature_rows.astype(int), np.ones(history_steps, dtype=int), mode="valid")
        == history_steps
    )
    target_valid_all = (
        np.convolve(target_rows.astype(int), np.ones(horizon_steps, dtype=int), mode="valid")
        == horizon_steps
    )
    history_valid = history_valid_all[:n_windows]
    target_valid = target_valid_all[history_steps : history_steps + n_windows]

    time_ns = timestamps.to_numpy(dtype="datetime64[ns]").astype("int64")
    step_ok = np.diff(time_ns) == pd.Timedelta(frequency).value
    bad_prefix = np.concatenate([[0], np.cumsum(~step_ok)])
    continuity = np.array(
        [
            bad_prefix[start + history_steps + horizon_steps - 1] - bad_prefix[start] == 0
            for start in range(n_windows)
        ]
    )
    valid = history_valid & target_valid & continuity
    scoring_total = n_windows * horizon_steps * len(target_columns)
    scoring_observed = int(
        sum(
            target_rows[start + history_steps : start + history_steps + horizon_steps].sum()
            for start in range(n_windows)
        )
        * len(target_columns)
    )
    return {
        "total_windows": int(n_windows),
        "valid_windows": int(valid.sum()),
        "invalid_windows": int((~valid).sum()),
        "scoring_observed": scoring_observed,
        "scoring_total": int(scoring_total),
        "scoring_coverage": float(scoring_observed / scoring_total),
    }
