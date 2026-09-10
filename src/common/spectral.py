"""Hankel singular-spectrum analysis for C-problem load and PV signals.

Adapted from OnlineLyra-TSTE's PV decomposition analysis. This module has no
file-system assumptions: preprocessing must supply a clean one-dimensional
power series before these functions are called.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class SingularSpectrumAnalysis:
    """Window-level spectra, pooled quantiles and an analysis audit."""

    window_spectra: pd.DataFrame
    summary: pd.DataFrame
    audit: dict[str, int | float | str]


def hankel_matrix(series: ArrayLike, rows: int | None = None) -> np.ndarray:
    """Return a Hankel trajectory matrix whose anti-diagonals are constant."""
    values = np.asarray(series, dtype=float).reshape(-1)
    if values.size < 3:
        raise ValueError("At least three observations are required for Hankelization.")
    hankel_rows = values.size // 2 if rows is None else int(rows)
    if not 2 <= hankel_rows < values.size:
        raise ValueError("rows must satisfy 2 <= rows < len(series).")
    return np.lib.stride_tricks.sliding_window_view(values, hankel_rows).T


def diagonal_average(matrix: ArrayLike) -> np.ndarray:
    """Map a reconstructed Hankel matrix back to a one-dimensional series."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError("matrix must be two-dimensional.")
    rows, columns = values.shape
    indices = np.arange(rows)[:, None] + np.arange(columns)[None, :]
    sums = np.bincount(indices.ravel(), weights=values.ravel())
    counts = np.bincount(indices.ravel())
    return sums / counts


def rank_one_decomposition(series: ArrayLike, rows: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Split a window into an SSA rank-one trend and residual component."""
    values = np.asarray(series, dtype=float).reshape(-1)
    if np.any(~np.isfinite(values)):
        raise ValueError("series must contain only finite values.")
    centered = values - values.mean()
    trajectory = hankel_matrix(centered, rows)
    left, singular_values, right_t = np.linalg.svd(trajectory, full_matrices=False)
    trend_matrix = singular_values[0] * np.outer(left[:, 0], right_t[0])
    trend = diagonal_average(trend_matrix) + values.mean()
    return trend, values - trend


def analyze_singular_spectrum(
    series: ArrayLike,
    *,
    lookback: int,
    hankel_rows: int | None = None,
    stride: int = 1,
    max_windows: int | None = None,
    seed: int = 2026,
    signal_name: str = "power",
) -> SingularSpectrumAnalysis:
    """Summarize normalized singular values and cumulative squared energy.

    This is deliberately data-agnostic. Pass a cleaned C-problem load or PV
    series; this function neither reads nor writes competition files.
    """
    values = np.asarray(series, dtype=float).reshape(-1)
    if np.any(~np.isfinite(values)):
        raise ValueError("series must be preprocessed to remove missing or infinite values.")
    if lookback < 4 or lookback > values.size:
        raise ValueError("lookback must be at least 4 and no longer than the series.")
    if stride <= 0:
        raise ValueError("stride must be positive.")
    rows = lookback // 2 if hankel_rows is None else int(hankel_rows)
    if not 2 <= rows < lookback:
        raise ValueError("hankel_rows must satisfy 2 <= hankel_rows < lookback.")

    all_windows = np.lib.stride_tricks.sliding_window_view(values, lookback)[::stride]
    selected_indices = np.arange(len(all_windows))
    if max_windows is not None and len(all_windows) > max_windows:
        if max_windows <= 0:
            raise ValueError("max_windows must be positive when supplied.")
        generator = np.random.default_rng(seed)
        selected_indices = np.sort(generator.choice(len(all_windows), max_windows, replace=False))
    windows = np.asarray(all_windows[selected_indices], dtype=float)
    centered = windows - windows.mean(axis=1, keepdims=True)
    trajectory = np.lib.stride_tricks.sliding_window_view(centered, rows, axis=1).transpose(0, 2, 1)
    singular_values = np.linalg.svd(trajectory, compute_uv=False)
    squared = np.square(singular_values)
    total_energy = squared.sum(axis=1)
    valid = total_energy > np.finfo(float).eps
    excluded = int((~valid).sum())
    singular_values = singular_values[valid]
    total_energy = total_energy[valid]
    valid_indices = selected_indices[valid]
    if singular_values.size == 0:
        raise ValueError("All selected windows are constant and have zero spectral energy.")

    normalized = singular_values / singular_values[:, :1]
    cumulative = np.cumsum(np.square(singular_values), axis=1) / total_energy[:, None]
    ranks = np.arange(1, singular_values.shape[1] + 1)

    window_spectra = pd.DataFrame(
        {
            "signal": signal_name,
            "window_index": np.repeat(valid_indices, len(ranks)),
            "window_start": np.repeat(valid_indices * stride, len(ranks)),
            "rank": np.tile(ranks, len(valid_indices)),
            "normalized_singular_value": normalized.ravel(),
            "cumulative_squared_energy": cumulative.ravel(),
        }
    )
    summary = pd.DataFrame(
        {
            "rank": ranks,
            "normalized_singular_value_q25": np.quantile(normalized, 0.25, axis=0),
            "normalized_singular_value_median": np.median(normalized, axis=0),
            "normalized_singular_value_q75": np.quantile(normalized, 0.75, axis=0),
            "cumulative_energy_q25": np.quantile(cumulative, 0.25, axis=0),
            "cumulative_energy_median": np.median(cumulative, axis=0),
            "cumulative_energy_q75": np.quantile(cumulative, 0.75, axis=0),
        }
    )
    audit: dict[str, int | float | str] = {
        "signal": signal_name,
        "series_length": int(values.size),
        "lookback": int(lookback),
        "hankel_rows": int(rows),
        "hankel_columns": int(lookback - rows + 1),
        "stride": int(stride),
        "candidate_windows": int(len(all_windows)),
        "selected_windows": int(len(selected_indices)),
        "valid_nonconstant_windows": int(valid.sum()),
        "excluded_zero_energy_windows": excluded,
        "seed": int(seed),
    }
    return SingularSpectrumAnalysis(window_spectra, summary, audit)


def rank_for_energy(summary: pd.DataFrame, threshold: float = 0.90) -> int:
    """Return the first rank whose median cumulative energy reaches a threshold."""
    if not 0 < threshold <= 1:
        raise ValueError("threshold must lie in (0, 1].")
    required = {"rank", "cumulative_energy_median"}
    if not required.issubset(summary.columns):
        raise ValueError(f"summary must contain columns {sorted(required)}.")
    reached = summary.loc[summary["cumulative_energy_median"] >= threshold, "rank"]
    if reached.empty:
        return int(summary["rank"].max())
    return int(reached.iloc[0])
