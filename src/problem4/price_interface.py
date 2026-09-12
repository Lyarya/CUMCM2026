"""Dispatch-neutral price contract for future Q4-2 and Q4-3 optimizers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from src.problem4.price_forecast import PREDICTIONS_PATH, SELECTED_OUTPUT_COLUMN, load_selection


FORMAL_START = pd.Timestamp("2025-02-01")
FORMAL_END = pd.Timestamp("2025-12-31")
RELEASE_HOURS = (0, 6, 12, 18)
HORIZON = 144
DT_HOURS = 1.0 / 6.0


class PriceInformationMode(StrEnum):
    """Two explicit interpretations of the statement's price availability."""

    ORACLE_PERFECT_INFORMATION = "oracle_perfect_information"
    CAUSAL_FORECAST = "causal_forecast"


@dataclass(frozen=True)
class Q4PriceInputs:
    """Price-only inputs; no battery state or dispatch decision is included."""

    date: np.datetime64
    issue_time: np.datetime64
    actual_history_cutoff: np.datetime64
    interval_end: np.ndarray
    price_yuan_per_kwh: np.ndarray
    executed_mask: np.ndarray
    future_mask: np.ndarray
    mode: str
    source_method: str
    dt_hours: float = DT_HOURS


def _load_predictions() -> pd.DataFrame:
    frame = pd.read_csv(
        PREDICTIONS_PATH,
        parse_dates=["operating_date", "interval_start", "interval_end"],
    )
    if frame["interval_end"].duplicated().any():
        raise AssertionError("Persisted Q4 price timestamps must be unique")
    return frame


def get_q4_price_inputs(
    date: str | pd.Timestamp,
    *,
    release_hour: int = 0,
    mode: PriceInformationMode | str,
) -> Q4PriceInputs:
    """Return a 144-step price path under one explicit information assumption.

    Executed intervals are represented by NaN in ``price_yuan_per_kwh``.  The
    future optimizer must only consume entries where ``future_mask`` is true.
    Prices are yuan/kWh and the time step is 1/6 hour.
    The causal path is issued at midnight and reused at later releases; those
    calls mask executed intervals but do not assimilate intraday price updates.
    """

    requested = pd.Timestamp(date)
    if pd.isna(requested) or requested.tzinfo is not None or requested != requested.normalize():
        raise ValueError("date must be a timezone-naive midnight operating date")
    operating_date = requested.normalize()
    if not FORMAL_START <= operating_date <= FORMAL_END:
        raise ValueError("Q4 formal date must lie between 2025-02-01 and 2025-12-31")
    if isinstance(release_hour, bool) or release_hour not in RELEASE_HOURS:
        raise ValueError(f"release_hour must be one of {RELEASE_HOURS}")
    information_mode = PriceInformationMode(mode)
    frame = _load_predictions()
    day = frame.loc[frame["operating_date"].eq(operating_date)].sort_values("slot")
    if len(day) != HORIZON or not day["slot"].to_numpy().tolist() == list(range(1, 145)):
        raise AssertionError("Requested date must contain exactly slots 1..144")
    issue_time = operating_date + pd.Timedelta(hours=release_hour)
    interval_end = day["interval_end"].to_numpy(dtype="datetime64[ns]")
    expected = pd.date_range(operating_date + pd.Timedelta(minutes=10), periods=144, freq="10min")
    if not np.array_equal(interval_end, expected.to_numpy(dtype="datetime64[ns]")):
        raise AssertionError("Price timestamps do not match the requested operating day")
    executed = interval_end <= np.datetime64(issue_time)
    future = ~executed
    selection = load_selection()
    if information_mode is PriceInformationMode.ORACLE_PERFECT_INFORMATION:
        values = day["price_yuan_per_kwh"].to_numpy(float)
        source_method = "附件4完美信息价格"
    else:
        validation_end = pd.Timestamp(selection["validation_period"][1]) + pd.Timedelta(days=1)
        if validation_end >= operating_date or not selection["model_frozen_before_evaluation"]:
            raise AssertionError("Price model selection must precede the requested day")
        values = day[SELECTED_OUTPUT_COLUMN].to_numpy(float)
        source_method = str(selection["selected_method"])
    output = values.copy()
    output[executed] = np.nan
    if not np.isfinite(output[future]).all():
        raise AssertionError("Future Q4 price inputs must all be finite")
    return Q4PriceInputs(
        date=operating_date.to_datetime64(),
        issue_time=issue_time.to_datetime64(),
        actual_history_cutoff=issue_time.to_datetime64(),
        interval_end=interval_end,
        price_yuan_per_kwh=output,
        executed_mask=executed,
        future_mask=future,
        mode=information_mode.value,
        source_method=source_method,
    )


def get_q4_2_price_inputs(
    date: str | pd.Timestamp,
    *,
    mode: PriceInformationMode | str,
) -> Q4PriceInputs:
    """Price contract for future Q4-2 day-ahead dispatch; dispatch is not run."""

    return get_q4_price_inputs(date, release_hour=0, mode=mode)


def get_q4_3_price_inputs(
    date: str | pd.Timestamp,
    release_hour: int,
    *,
    mode: PriceInformationMode | str,
) -> Q4PriceInputs:
    """Price contract for future Q4-3 rolling dispatch; dispatch is not run."""

    return get_q4_price_inputs(date, release_hour=release_hour, mode=mode)


__all__ = [
    "DT_HOURS",
    "FORMAL_END",
    "FORMAL_START",
    "HORIZON",
    "PriceInformationMode",
    "Q4PriceInputs",
    "RELEASE_HOURS",
    "get_q4_2_price_inputs",
    "get_q4_3_price_inputs",
    "get_q4_price_inputs",
]
