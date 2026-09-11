"""Q3 forecast-arrival, settlement and future VOI interfaces without dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.paths import PROCESSED_DATA_DIR, RESULTS_DIR
from src.problem3.forecast_data import ACTUAL_PATH, LEAD_BIN_LABELS, TEN_MINUTE_PATH, lead_bin


SCHEDULES: dict[str, tuple[int, ...]] = {
    "S0": (0,),
    "S1": (0, 6),
    "S2": (0, 6, 12),
    "S3": (0, 6, 12, 18),
}
SELECTION_PATH = RESULTS_DIR / "problem3" / "tables" / "q3_forecast_selection.json"
VOI_COLUMNS = (
    "schedule",
    "total_cost_yuan",
    "base_purchase_cost_yuan",
    "adjustment_cost_yuan",
    "emergency_cost_yuan",
    "emergency_energy_kwh",
    "emergency_interval_count",
    "adjustment_count",
    "adjustment_absolute_energy_kwh",
    "voi_vs_s0_yuan",
    "incremental_voi_yuan",
)


class SettlementMode(str, Enum):
    SEQUENTIAL_PREVIOUS_COMMITMENT = "SEQUENTIAL_PREVIOUS_COMMITMENT"
    ORIGINAL_00_COMMITMENT = "ORIGINAL_00_COMMITMENT"


@dataclass(frozen=True)
class Q3ForecastUpdate:
    operating_date: object
    release_hour: int
    issue_time: np.datetime64
    actual_history_cutoff: np.datetime64
    current_realized_soc: float
    timestamps: np.ndarray
    official_forecast_kw: np.ndarray
    seasonal_forecast_kw: np.ndarray
    fused_forecast_kw: np.ndarray
    lead_bins: np.ndarray
    executed_mask: np.ndarray
    future_mask: np.ndarray


def schedule_releases(name: str) -> tuple[int, ...]:
    try:
        return SCHEDULES[name]
    except KeyError as exc:
        raise KeyError(f"unknown Q3 information schedule {name!r}") from exc


def _load_selection() -> dict[str, object]:
    if not SELECTION_PATH.exists():
        raise FileNotFoundError("run src/problem3/run.py before requesting Q3 update inputs")
    return json.loads(SELECTION_PATH.read_text(encoding="utf-8"))


def _seasonal_forecast(
    target_times: pd.DatetimeIndex,
    issue_time: pd.Timestamp,
    actual: pd.Series,
) -> np.ndarray:
    history_times = np.concatenate(
        [
            (target_times - pd.Timedelta(days=lag)).to_numpy(dtype="datetime64[ns]")[:, None]
            for lag in range(1, 8)
        ],
        axis=1,
    )
    if np.max(history_times) > np.datetime64(issue_time, "ns"):
        raise AssertionError("seasonal Q3 expert requested future actual PV")
    values = np.column_stack(
        [pd.Series(target_times - pd.Timedelta(days=lag)).map(actual).to_numpy(float) for lag in range(1, 8)]
    )
    if not np.isfinite(values).all():
        raise ValueError("seven complete causal historical days are required")
    return values.mean(axis=1)


def get_q3_forecast_update(
    date: object,
    release_hour: int,
    current_realized_soc: float,
) -> Q3ForecastUpdate:
    """Expose only the remaining operating-day forecast at one legal issue time."""

    if int(release_hour) not in (0, 6, 12, 18):
        raise ValueError("release_hour must be one of 0, 6, 12, 18")
    soc = float(current_realized_soc)
    if not np.isfinite(soc):
        raise ValueError("current_realized_soc must be supplied as a finite value")
    day = pd.Timestamp(date).normalize()
    issue = day + pd.Timedelta(hours=int(release_hour))
    timestamps = pd.date_range(day + pd.Timedelta(minutes=10), periods=144, freq="10min")
    executed = timestamps <= issue
    future = ~executed

    fine = pd.read_csv(TEN_MINUTE_PATH, parse_dates=["release_time", "target_time"])
    batch = fine.loc[fine["release_time"].eq(issue)].set_index("target_time")
    if len(batch) != 144 or batch.index.duplicated().any():
        raise AssertionError(f"Attachment-3 10-minute batch is incomplete at {issue}")
    official = np.full(144, np.nan)
    official[future] = batch.reindex(timestamps[future])["pv_forecast_10min_kw"].to_numpy(float)
    if not np.isfinite(official[future]).all():
        raise AssertionError("future official PCHIP forecast contains missing values")

    actual_frame = pd.read_csv(ACTUAL_PATH, parse_dates=["interval_end"])
    actual = actual_frame.set_index("interval_end")["pv_actual_kw"].astype(float)
    seasonal = np.full(144, np.nan)
    seasonal[future] = _seasonal_forecast(timestamps[future], issue, actual)
    selection = _load_selection()
    if selection["selected_method"] != "提前期分箱融合":
        raise AssertionError("Q3 forecast update interface expects the validated lead-aware fusion")
    lead_hours = (timestamps - issue) / pd.Timedelta(hours=1)
    bins = np.full(144, "已执行", dtype=object)
    bins[future] = lead_bin(lead_hours[future]).astype(str)
    weights = {str(k): float(v) for k, v in selection["lead_weights"].items()}
    official_weights = np.array([weights[str(label)] for label in bins[future]], dtype=float)
    fused = np.full(144, np.nan)
    fused[future] = np.maximum(
        official_weights * official[future] + (1.0 - official_weights) * seasonal[future],
        0.0,
    )
    return Q3ForecastUpdate(
        operating_date=day.date(),
        release_hour=int(release_hour),
        issue_time=np.datetime64(issue, "ns"),
        actual_history_cutoff=np.datetime64(issue, "ns"),
        current_realized_soc=soc,
        timestamps=timestamps.to_numpy(dtype="datetime64[ns]"),
        official_forecast_kw=official,
        seasonal_forecast_kw=seasonal,
        fused_forecast_kw=fused,
        lead_bins=bins,
        executed_mask=np.asarray(executed, dtype=bool),
        future_mask=np.asarray(future, dtype=bool),
    )


def adjustment_settlement(
    original_grid_kwh: np.ndarray,
    revisions_kwh: np.ndarray,
    price_yuan_per_kwh: np.ndarray,
    mode: SettlementMode,
) -> dict[str, np.ndarray]:
    """Evaluate adjustment accounting only; this function performs no dispatch."""

    base = np.asarray(original_grid_kwh, dtype=float)
    revisions = np.asarray(revisions_kwh, dtype=float)
    price = np.asarray(price_yuan_per_kwh, dtype=float)
    if revisions.ndim != 2 or revisions.shape[1:] != base.shape or price.shape != base.shape:
        raise ValueError("revisions must be [revision, interval] and match base/price")
    if mode == SettlementMode.SEQUENTIAL_PREVIOUS_COMMITMENT:
        prior = np.vstack([base[None, :], revisions[:-1]])
        deltas = revisions - prior
    elif mode == SettlementMode.ORIGINAL_00_COMMITMENT:
        deltas = (revisions[-1] - base)[None, :]
    else:
        raise ValueError(f"unsupported settlement mode {mode!r}")
    increase = np.maximum(deltas, 0.0)
    decrease = np.maximum(-deltas, 0.0)
    adjustment = 1.5 * price[None, :] * increase - 0.5 * price[None, :] * decrease
    return {
        "delta_plus_kwh": increase,
        "delta_minus_kwh": decrease,
        "adjustment_cost_yuan": adjustment,
        "base_cost_yuan": price * base,
        "final_commitment_kwh": revisions[-1].copy(),
    }


def empty_voi_table() -> pd.DataFrame:
    """Return the future VOI result schema with zero fabricated result rows."""

    return pd.DataFrame(columns=VOI_COLUMNS)


__all__ = [
    "Q3ForecastUpdate",
    "SCHEDULES",
    "SettlementMode",
    "VOI_COLUMNS",
    "adjustment_settlement",
    "empty_voi_table",
    "get_q3_forecast_update",
    "schedule_releases",
]
