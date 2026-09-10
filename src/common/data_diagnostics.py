"""Structural diagnostics for the canonical C-problem time series.

This module performs descriptive and structural analysis only.  It does not
fit forecasting models or solve any dispatch problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from src.common.spectral import SingularSpectrumAnalysis, analyze_singular_spectrum, hankel_matrix


VARIABLES = {
    "pv_actual_kw": "PV",
    "load_kw": "Load",
    "net_load_kw": "Net Load",
    "price_yuan_per_kwh": "Price",
}
SIGNALS = {
    "pv_actual_kw": "PV",
    "load_kw": "Load",
    "net_load_kw": "Net Load",
}


@dataclass(frozen=True)
class DiagnosticsConfig:
    """Central configuration for every Stage 1B diagnostic."""

    lags: tuple[int, ...] = (1, 6, 144, 1008)
    daylight_threshold_kw: float = 0.0
    history_steps: int = 1008
    forecast_horizon_steps: int = 144
    hankel_rows: int = 144
    spectrum_stride: int = 1008
    forecastability_stride: int = 144
    maximum_spectrum_rank: int = 30
    zscore_threshold: float = 3.0
    iqr_multiplier: float = 1.5


def add_calendar_fields(frame: pd.DataFrame) -> pd.DataFrame:
    """Add calendar labels without modifying the canonical input."""
    output = frame.copy()
    output["interval_end"] = pd.to_datetime(output["interval_end"])
    output["month"] = output["interval_end"].dt.month.astype(int)
    output["hour_of_day"] = (
        output["interval_end"].dt.hour + output["interval_end"].dt.minute / 60.0
    )
    month_to_season = {
        12: "winter",
        1: "winter",
        2: "winter",
        3: "spring",
        4: "spring",
        5: "spring",
        6: "summer",
        7: "summer",
        8: "summer",
        9: "autumn",
        10: "autumn",
        11: "autumn",
    }
    output["season"] = output["month"].map(month_to_season)
    return output


def descriptive_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    """Return overall, monthly, hourly and seasonal summaries in tidy form."""
    data = add_calendar_fields(frame)
    rows: list[dict[str, object]] = []
    group_specs: list[tuple[str, Iterable[tuple[object, pd.DataFrame]]]] = [
        ("overall", [("all", data)]),
        ("month", data.groupby("month", sort=True)),
        ("hour_of_day", data.groupby("hour_of_day", sort=True)),
        ("season", data.groupby("season", sort=True)),
    ]
    for group_type, groups in group_specs:
        for group_value, subset in groups:
            for column, label in VARIABLES.items():
                values = subset[column].dropna()
                rows.append(
                    {
                        "group_type": group_type,
                        "group_value": group_value,
                        "variable": label,
                        "count": int(values.count()),
                        "mean": float(values.mean()),
                        "std": float(values.std(ddof=1)),
                        "min": float(values.min()),
                        "25%": float(values.quantile(0.25)),
                        "median": float(values.median()),
                        "75%": float(values.quantile(0.75)),
                        "max": float(values.max()),
                    }
                )
    return pd.DataFrame(rows)


def correlation_matrices(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute descriptive Pearson and Spearman correlations."""
    renamed = frame[list(VARIABLES)].rename(columns=VARIABLES)
    pearson = renamed.corr(method="pearson")
    spearman = renamed.corr(method="spearman")
    pearson.index.name = "variable"
    spearman.index.name = "variable"
    return pearson.reset_index(), spearman.reset_index()


def _lag_pair(values: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    if lag <= 0 or lag >= values.size:
        raise ValueError("lag must lie between 1 and len(values)-1")
    return values[lag:], values[:-lag]


def _pearson_pair(current: np.ndarray, lagged: np.ndarray, mask: np.ndarray) -> tuple[int, float]:
    valid = mask & np.isfinite(current) & np.isfinite(lagged)
    if valid.sum() < 2:
        return int(valid.sum()), float("nan")
    return int(valid.sum()), float(np.corrcoef(current[valid], lagged[valid])[0, 1])


def lag_correlations(
    frame: pd.DataFrame,
    config: DiagnosticsConfig,
    *,
    lags: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Compute fixed-lag correlations, including PV daylight-only pairs."""
    requested = tuple(config.lags if lags is None else lags)
    rows: list[dict[str, object]] = []
    for column, label in SIGNALS.items():
        values = frame[column].to_numpy(dtype=float)
        for lag in requested:
            current, lagged = _lag_pair(values, int(lag))
            n, value = _pearson_pair(current, lagged, np.ones(current.shape, dtype=bool))
            rows.append(
                {
                    "variable": label,
                    "subset": "full_series",
                    "lag_steps": int(lag),
                    "lag_hours": float(lag / 6),
                    "count": n,
                    "pearson_correlation": value,
                }
            )
            if column == "pv_actual_kw":
                daylight = (current > config.daylight_threshold_kw) & (
                    lagged > config.daylight_threshold_kw
                )
                n_day, value_day = _pearson_pair(current, lagged, daylight)
                rows.append(
                    {
                        "variable": label,
                        "subset": "daylight_only",
                        "lag_steps": int(lag),
                        "lag_hours": float(lag / 6),
                        "count": n_day,
                        "pearson_correlation": value_day,
                    }
                )
    return pd.DataFrame(rows)


def pv_acf_curve(frame: pd.DataFrame, config: DiagnosticsConfig) -> pd.DataFrame:
    """Compute PV correlations for every lag through one week."""
    values = frame["pv_actual_kw"].to_numpy(dtype=float)
    rows: list[dict[str, object]] = []
    for lag in range(1, max(config.lags) + 1):
        current, lagged = _lag_pair(values, lag)
        for subset, mask in {
            "full_series": np.ones(current.shape, dtype=bool),
            "daylight_only": (current > config.daylight_threshold_kw)
            & (lagged > config.daylight_threshold_kw),
        }.items():
            count, correlation = _pearson_pair(current, lagged, mask)
            rows.append(
                {
                    "variable": "PV",
                    "subset": subset,
                    "lag_steps": lag,
                    "lag_hours": lag / 6,
                    "count": count,
                    "pearson_correlation": correlation,
                }
            )
    return pd.DataFrame(rows)


def _error_metrics(actual: np.ndarray, predicted: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    valid = mask & np.isfinite(actual) & np.isfinite(predicted)
    error = predicted[valid] - actual[valid]
    return {
        "count": int(valid.sum()),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "bias": float(np.mean(error)),
    }


def seasonal_naive_metrics(frame: pd.DataFrame, config: DiagnosticsConfig) -> pd.DataFrame:
    """Evaluate strictly causal yesterday and last-week baselines."""
    rows: list[dict[str, object]] = []
    for column, label in {"pv_actual_kw": "PV", "load_kw": "Load"}.items():
        actual = frame[column].to_numpy(dtype=float)
        for baseline, lag in {"Yesterday": 144, "Last Week": 1008}.items():
            current, predicted = _lag_pair(actual, lag)
            subsets = {"full_series": np.ones(current.shape, dtype=bool)}
            if column == "pv_actual_kw":
                subsets["daylight_only"] = current > config.daylight_threshold_kw
            for subset_name, mask in subsets.items():
                rows.append(
                    {
                        "variable": label,
                        "baseline": baseline,
                        "subset": subset_name,
                        "lag_steps": lag,
                        **_error_metrics(current, predicted, mask),
                    }
                )
    return pd.DataFrame(rows)


def official_forecast_diagnostics(
    alignment: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize aligned official forecast errors without using unmatched rows."""
    valid = alignment.loc[alignment["alignment_status"] == "matched"].copy()
    valid["release_time"] = pd.to_datetime(valid["release_time"])
    valid["target_time"] = pd.to_datetime(valid["target_time"])
    valid["forecast_error_kw"] = valid["pv_forecast_kw"] - valid["pv_actual_kw"]

    def summarize(data: pd.DataFrame, key: str, ordered_values: Iterable[object]) -> pd.DataFrame:
        rows = []
        for value in ordered_values:
            subset = data.loc[data[key] == value]
            error = subset["forecast_error_kw"].to_numpy(dtype=float)
            if error.size == 0:
                rows.append(
                    {key: value, "count": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan}
                )
                continue
            rows.append(
                {
                    key: value,
                    "count": int(len(error)),
                    "mae": float(np.mean(np.abs(error))),
                    "rmse": float(np.sqrt(np.mean(np.square(error)))),
                    "bias": float(np.mean(error)),
                }
            )
        return pd.DataFrame(rows)

    all_error = valid["forecast_error_kw"].to_numpy(dtype=float)
    overall = pd.DataFrame(
        [
            {
                "count": int(len(all_error)),
                "mae": float(np.mean(np.abs(all_error))),
                "rmse": float(np.sqrt(np.mean(np.square(all_error)))),
                "bias": float(np.mean(all_error)),
            }
        ]
    )
    lead_labels = ["1-3h", "4-6h", "7-12h", "13-18h", "19-24h"]
    valid["lead_bin"] = pd.cut(
        valid["horizon_hour"],
        bins=[0, 3, 6, 12, 18, 24],
        labels=lead_labels,
        include_lowest=True,
    ).astype(str)
    valid["issue_hour"] = valid["release_time"].dt.hour.astype(int)
    valid["month"] = valid["target_time"].dt.month.astype(int)
    by_lead = summarize(valid, "lead_bin", lead_labels)
    by_issue = summarize(valid, "issue_hour", [0, 6, 12, 18])
    by_month = summarize(valid, "month", range(1, 13))
    return overall, by_lead, by_issue, by_month


def hankel_diagnostics(
    frame: pd.DataFrame,
    config: DiagnosticsConfig,
) -> SingularSpectrumAnalysis:
    """Compute fixed-configuration PV Hankel spectra on chronological windows."""
    return analyze_singular_spectrum(
        frame["pv_actual_kw"].to_numpy(dtype=float),
        lookback=config.history_steps,
        hankel_rows=config.hankel_rows,
        stride=config.spectrum_stride,
        max_windows=None,
        signal_name="PV",
    )


def _lowrank_energy(values: np.ndarray, rows: int) -> tuple[float, float]:
    centered = values - np.mean(values)
    matrix = hankel_matrix(centered, rows=rows)
    gram = matrix @ matrix.T
    eigenvalues = np.linalg.eigvalsh(gram)
    energy = np.maximum(eigenvalues[::-1], 0.0)
    total = energy.sum()
    if total <= np.finfo(float).eps:
        return float("nan"), float("nan")
    cumulative = np.cumsum(energy) / total
    return float(cumulative[0]), float(cumulative[2])


def _safe_correlation(x: np.ndarray, y: np.ndarray, method: str) -> tuple[float, float, int]:
    valid = np.isfinite(x) & np.isfinite(y)
    x_valid, y_valid = x[valid], y[valid]
    if valid.sum() < 3:
        return float("nan"), float("nan"), int(valid.sum())
    x_near_constant = np.linalg.norm(x_valid - np.mean(x_valid)) < 1e-13 * max(
        abs(float(np.mean(x_valid))), 1.0
    )
    y_near_constant = np.linalg.norm(y_valid - np.mean(y_valid)) < 1e-13 * max(
        abs(float(np.mean(y_valid))), 1.0
    )
    if x_near_constant or y_near_constant:
        return float("nan"), float("nan"), int(valid.sum())
    result = stats.pearsonr(x_valid, y_valid) if method == "pearson" else stats.spearmanr(x_valid, y_valid)
    return float(result.statistic), float(result.pvalue), int(valid.sum())


def lowrank_forecastability(
    frame: pd.DataFrame,
    config: DiagnosticsConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Relate history low-rank energy to causal next-day Yesterday errors."""
    timestamps = pd.DatetimeIndex(pd.to_datetime(frame["interval_end"]))
    pv = frame["pv_actual_kw"].to_numpy(dtype=float)
    last_start = len(pv) - config.history_steps - config.forecast_horizon_steps
    starts = np.arange(0, last_start + 1, config.forecastability_stride, dtype=int)
    rows: list[dict[str, object]] = []
    for window_id, start in enumerate(starts, start=1):
        split = start + config.history_steps
        history = pv[start:split]
        actual = pv[split : split + config.forecast_horizon_steps]
        predicted = pv[
            split - 144 : split - 144 + config.forecast_horizon_steps
        ]
        top1, top3 = _lowrank_energy(history, config.hankel_rows)
        error = predicted - actual
        rows.append(
            {
                "window_id": window_id,
                "history_start": timestamps[start],
                "history_end": timestamps[split - 1],
                "forecast_start": timestamps[split],
                "forecast_end": timestamps[split + config.forecast_horizon_steps - 1],
                "top1_energy": top1,
                "top3_energy": top3,
                "yesterday_mae": float(np.mean(np.abs(error))),
                "yesterday_rmse": float(np.sqrt(np.mean(np.square(error)))),
            }
        )
    windows = pd.DataFrame(rows)
    correlation_rows = []
    for energy_column in ["top1_energy", "top3_energy"]:
        for error_column in ["yesterday_mae", "yesterday_rmse"]:
            for method in ["pearson", "spearman"]:
                coefficient, pvalue, count = _safe_correlation(
                    windows[energy_column].to_numpy(dtype=float),
                    -windows[error_column].to_numpy(dtype=float),
                    method,
                )
                correlation_rows.append(
                    {
                        "energy_metric": energy_column,
                        "forecastability_metric": f"negative_{error_column}",
                        "correlation_method": method,
                        "count": count,
                        "correlation": coefficient,
                        "p_value": pvalue,
                    }
                )
    return windows, pd.DataFrame(correlation_rows)


def anomaly_candidates(frame: pd.DataFrame, config: DiagnosticsConfig) -> pd.DataFrame:
    """Flag global Z-score and IQR candidates without deleting observations."""
    rows: list[pd.DataFrame] = []
    for column, label in {
        "pv_actual_kw": "PV",
        "load_kw": "Load",
        "price_yuan_per_kwh": "Price",
    }.items():
        values = frame[column].astype(float)
        std = values.std(ddof=0)
        zscore = (values - values.mean()) / std
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - config.iqr_multiplier * iqr
        upper = q3 + config.iqr_multiplier * iqr
        z_flag = zscore.abs() > config.zscore_threshold
        iqr_flag = (values < lower) | (values > upper)
        selected = z_flag | iqr_flag
        candidate = pd.DataFrame(
            {
                "interval_end": pd.to_datetime(frame.loc[selected, "interval_end"]),
                "variable": label,
                "value": values[selected],
                "zscore": zscore[selected],
                "zscore_flag": z_flag[selected],
                "iqr_lower": float(lower),
                "iqr_upper": float(upper),
                "iqr_flag": iqr_flag[selected],
                "action": "flag_only_do_not_delete",
            }
        )
        rows.append(candidate)
    return pd.concat(rows, ignore_index=True).sort_values(
        ["interval_end", "variable"], kind="stable"
    )
