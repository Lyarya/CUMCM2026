"""Thin dynamic-price adapters around the locked Q2/Q3 dispatch pipelines."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from src.problem2.evaluate import validate_q2_day
from src.problem2.forecast_interface import FORMAL_FORECAST_PATH, get_q2_day_inputs
from src.problem2.model import Q2DispatchParameters, solve_expected_cost_dispatch
from src.problem2.run import settle_realized_day
from src.problem3.rolling_dispatch import Q3ScheduleResult, run_schedule
from src.problem4.price_forecast import PREDICTIONS_PATH, SELECTION_PATH
from src.problem4.price_interface import PriceInformationMode, get_q4_2_price_inputs


GIVEN_PRICE = "GIVEN_PRICE"
CAUSAL_PRICE = "CAUSAL_PRICE"
PRICE_MODE_MAP = {
    GIVEN_PRICE: PriceInformationMode.ORACLE_PERFECT_INFORMATION,
    CAUSAL_PRICE: PriceInformationMode.CAUSAL_FORECAST,
}
DT_HOURS = 1.0 / 6.0


@dataclass(frozen=True)
class Q42SampleResult:
    mode: str
    daily: pd.DataFrame
    intervals: pd.DataFrame
    audit: dict[str, object]


def _price_pair(date: object, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Return decision and realized settlement prices on the same 144-slot grid."""

    if mode not in PRICE_MODE_MAP:
        raise ValueError(f"unsupported Q4 price mode: {mode}")
    decision = get_q4_2_price_inputs(date, mode=PRICE_MODE_MAP[mode])
    actual = get_q4_2_price_inputs(
        date, mode=PriceInformationMode.ORACLE_PERFECT_INFORMATION
    )
    if not np.array_equal(decision.interval_end, actual.interval_end):
        raise AssertionError("decision and settlement price timestamps differ")
    if decision.executed_mask.any() or actual.executed_mask.any():
        raise AssertionError("midnight Q4 price input unexpectedly masks an interval")
    decision_price = np.asarray(decision.price_yuan_per_kwh, dtype=float)
    settlement_price = np.asarray(actual.price_yuan_per_kwh, dtype=float)
    if not np.isfinite(decision_price).all() or not np.isfinite(settlement_price).all():
        raise AssertionError("Q4 price provider returned non-finite prices")
    return decision_price, settlement_price


def price_input_signature(mode: str) -> str:
    """Bind resumable checkpoints to the selected persisted price inputs."""

    digest = hashlib.sha256()
    digest.update(mode.encode("utf-8"))
    for path in (PREDICTIONS_PATH, SELECTION_PATH):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def q43_price_provider(mode: str):
    """Build the full-day provider consumed by the minimally extended Q3 runner."""

    if mode not in PRICE_MODE_MAP:
        raise ValueError(f"unsupported Q4 price mode: {mode}")

    def provider(date: object) -> tuple[np.ndarray, np.ndarray]:
        return _price_pair(date, mode)

    return provider


def _actual_frame() -> pd.DataFrame:
    return pd.read_csv(
        FORMAL_FORECAST_PATH,
        usecols=["operating_date", "datetime", "actual_load", "actual_generation"],
        parse_dates=["datetime"],
    ).sort_values(["operating_date", "datetime"], kind="stable")


def _audit_realized_dispatch(daily: pd.DataFrame, intervals: pd.DataFrame) -> dict[str, object]:
    tolerance = 2e-3
    balance = (
        intervals["realized_planned_grid_used_kw"]
        + intervals["realized_emergency_kw"]
        + intervals["actual_pv_kw"]
        + intervals["actual_discharge_kw"]
        - intervals["actual_load_kw"]
        - intervals["actual_charge_kw"]
        - intervals["realized_pv_spill_kw"]
    )
    expected_soc = (
        intervals["actual_storage_start_kwh"]
        + 0.9 * intervals["actual_charge_kw"] * DT_HOURS
        - intervals["actual_discharge_kw"] * DT_HOURS / 0.9
    )
    soc_residual = intervals["actual_storage_end_kwh"] - expected_soc
    cross = (
        daily["initial_energy_kwh"].iloc[1:].to_numpy(float)
        - daily["actual_final_energy_kwh"].iloc[:-1].to_numpy(float)
    )
    absorbable = np.maximum(
        intervals["actual_load_kw"]
        + intervals["actual_charge_kw"]
        - intervals["actual_pv_kw"],
        0.0,
    )
    price_alignment_residual = np.abs(
        intervals["realized_emergency_cost_yuan"]
        - 5.0
        * intervals["price_yuan_per_kwh"]
        * intervals["realized_emergency_kwh"]
    )
    reconciliation = np.abs(
        daily["realized_total_cost_yuan"]
        - daily["planned_purchase_cost_yuan"]
        - daily["realized_emergency_cost_yuan"]
    )
    checks = {
        "optimal": bool(daily["solver_status"].eq("Optimal").all()),
        "dynamic_price_alignment": bool(price_alignment_residual.max() <= tolerance),
        "maximum_price_alignment_residual_yuan": float(price_alignment_residual.max()),
        "realized_power_balance": bool(np.abs(balance).max() <= tolerance),
        "maximum_power_balance_residual_kw": float(np.abs(balance).max()),
        "soc_recurrence": bool(np.abs(soc_residual).max() <= tolerance),
        "maximum_soc_recurrence_residual_kwh": float(np.abs(soc_residual).max()),
        "cross_day_soc": bool(not len(cross) or np.abs(cross).max() <= tolerance),
        "maximum_cross_day_soc_residual_kwh": float(np.abs(cross).max()) if len(cross) else 0.0,
        "soc_bounds": bool(
            intervals[["actual_storage_start_kwh", "actual_storage_end_kwh"]]
            .to_numpy(float)
            .min()
            >= 1200.0 - tolerance
            and intervals[["actual_storage_start_kwh", "actual_storage_end_kwh"]]
            .to_numpy(float)
            .max()
            <= 10800.0 + tolerance
        ),
        "phantom_discharge": bool(
            (intervals["actual_discharge_kw"] <= absorbable + tolerance).all()
        ),
        "unabsorbed_discharge_energy_kwh": 0.0,
        "settlement_reconciliation": bool(reconciliation.max() <= tolerance),
        "maximum_settlement_residual_yuan": float(reconciliation.max()),
        "time_alignment": bool(
            len(intervals) == 144 * len(daily)
            and intervals.groupby("operating_date")["slot"].nunique().eq(144).all()
        ),
    }
    if not all(value for value in checks.values() if isinstance(value, bool)):
        raise AssertionError(f"Q4-2 physical audit failed: {checks}")
    return checks


def run_q42_sample(
    dates: pd.DatetimeIndex,
    *,
    mode: str,
    checkpoint_path: Path | None = None,
) -> Q42SampleResult:
    """Run a short Q4-2 sample without copying or changing the locked Q2 model."""

    actual = _actual_frame()
    signature = price_input_signature(mode)
    carried_soc = 6000.0
    daily_rows: list[dict[str, object]] = []
    interval_frames: list[pd.DataFrame] = []
    start_index = 0
    if checkpoint_path is not None and checkpoint_path.exists():
        with checkpoint_path.open("rb") as stream:
            state = pickle.load(stream)
        expected_dates = [str(pd.Timestamp(value).date()) for value in dates]
        if state["dates"] != expected_dates or state["signature"] != signature:
            raise AssertionError("Q4-2 checkpoint does not match dates or price inputs")
        start_index = int(state["next_index"])
        carried_soc = float(state["carried_soc"])
        daily_rows = state["daily_rows"]
        interval_frames = state["interval_frames"]

    for index, date in enumerate(dates[start_index:], start=start_index + 1):
        decision_price, settlement_price = _price_pair(date, mode)
        base_inputs = get_q2_day_inputs(date, initial_energy=carried_soc)
        inputs = replace(base_inputs, price=decision_price)
        solved = solve_expected_cost_dispatch(inputs, Q2DispatchParameters())
        validation = validate_q2_day(inputs, solved)
        plan = solved.dispatch.copy()
        plan["decision_price_yuan_per_kwh"] = plan["price_yuan_per_kwh"]
        plan["price_yuan_per_kwh"] = settlement_price
        actual_day = actual.loc[
            actual["operating_date"].astype(str).eq(str(pd.Timestamp(date).date()))
        ].sort_values("datetime", kind="stable")
        initial_soc = carried_soc
        settled, carried_soc = settle_realized_day(plan, actual_day, carried_soc)
        settled["operating_date"] = str(pd.Timestamp(date).date())
        planned_cost = float(
            np.sum(settlement_price * settled["planned_grid_kwh"].to_numpy(float))
        )
        emergency_cost = float(settled["realized_emergency_cost_yuan"].sum())
        daily_rows.append(
            {
                "date": str(pd.Timestamp(date).date()),
                "solver_status": solved.status,
                "solver_runtime_seconds": solved.runtime_seconds,
                "initial_energy_kwh": initial_soc,
                "planned_final_energy_kwh": solved.final_energy_kwh,
                "actual_final_energy_kwh": carried_soc,
                "planned_purchase_cost_yuan": planned_cost,
                "decision_price_planned_cost_yuan": solved.planned_purchase_cost_yuan,
                "expected_scenario_emergency_cost_yuan": solved.expected_emergency_cost_yuan,
                "realized_emergency_cost_yuan": emergency_cost,
                "realized_total_cost_yuan": planned_cost + emergency_cost,
                "maximum_scenario_power_balance_residual_kw": validation.maximum_power_balance_residual_kw,
            }
        )
        interval_frames.append(settled)
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint_path.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                pickle.dump(
                    {
                        "dates": [str(pd.Timestamp(value).date()) for value in dates],
                        "signature": signature,
                        "next_index": index,
                        "carried_soc": carried_soc,
                        "daily_rows": daily_rows,
                        "interval_frames": interval_frames,
                    },
                    stream,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            temporary.replace(checkpoint_path)

    daily = pd.DataFrame(daily_rows)
    intervals = pd.concat(interval_frames, ignore_index=True)
    return Q42SampleResult(mode, daily, intervals, _audit_realized_dispatch(daily, intervals))


def run_q43_sample(
    schedule: str,
    dates: pd.DatetimeIndex,
    *,
    mode: str,
    checkpoint_path: Path | None = None,
) -> Q3ScheduleResult:
    """Run the locked Q3 rolling pipeline with only the price provider replaced."""

    signature = price_input_signature(mode)
    return run_schedule(
        schedule,
        dates,
        progress=False,
        checkpoint_path=checkpoint_path,
        price_provider=q43_price_provider(mode),
        price_information_mode=f"{mode}:{signature}",
    )


__all__ = [
    "CAUSAL_PRICE",
    "GIVEN_PRICE",
    "PRICE_MODE_MAP",
    "Q42SampleResult",
    "price_input_signature",
    "q43_price_provider",
    "run_q42_sample",
    "run_q43_sample",
]
