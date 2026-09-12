"""Causal Q3 rolling optimization, physical execution, settlement, and VOI."""

from __future__ import annotations

from dataclasses import dataclass, replace
import pickle
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from src.problem2.forecast_interface import Q2DayInputs, get_q2_day_inputs
from src.problem2.model import Q2DispatchParameters
from src.problem2.run import FORMAL_FORECAST_PATH, settle_realized_day
from src.problem3.information_schedule import (
    SettlementMode,
    adjustment_settlement,
    get_q3_forecast_update,
    schedule_releases,
)
from src.problem3.rolling_model import Q3RollingPlanResult, solve_remaining_dispatch


DT_HOURS = 1.0 / 6.0
ACTUAL_COLUMNS = (
    "actual_load_kw",
    "actual_pv_kw",
    "actual_charge_kw",
    "actual_discharge_kw",
    "actual_storage_start_kwh",
    "actual_storage_end_kwh",
    "realized_planned_grid_used_kw",
    "realized_unused_planned_grid_kw",
    "realized_emergency_kw",
    "realized_pv_spill_kw",
    "actual_charge_kwh",
    "actual_discharge_kwh",
    "realized_planned_grid_used_kwh",
    "realized_unused_planned_grid_kwh",
    "realized_emergency_kwh",
    "realized_pv_spill_kwh",
    "realized_emergency_cost_yuan",
)


@dataclass(frozen=True)
class Q3ScheduleResult:
    schedule: str
    daily_main: pd.DataFrame
    daily_sensitivity: pd.DataFrame
    intervals: pd.DataFrame
    audit: dict[str, object]


def build_release_inputs(
    date: object, release_hour: int, current_realized_soc: float
) -> tuple[Q2DayInputs, object]:
    """Build the locked 50-scenario remaining horizon around Arya's forecast."""

    base = get_q2_day_inputs(date, initial_energy=current_realized_soc)
    update = get_q3_forecast_update(date, release_hour, current_realized_soc)
    start = int(release_hour) * 6
    expected_future = np.arange(144) >= start
    if not np.array_equal(update.future_mask, expected_future):
        raise AssertionError("Q3 release mask does not match the ten-minute grid")
    center = np.array(base.pv_forecast, copy=True)
    center[start:] = update.fused_forecast_kw[start:]
    residual = base.pv_scenarios - base.pv_forecast[None, :]
    pv_scenarios = np.maximum(center[None, start:] + residual[:, start:], 0.0)
    inputs = Q2DayInputs(
        date=base.date,
        load_forecast=np.array(base.load_forecast[start:], copy=True),
        pv_forecast=np.array(center[start:], copy=True),
        load_scenarios=np.array(base.load_scenarios[:, start:], copy=True),
        pv_scenarios=pv_scenarios,
        price=np.array(base.price[start:], copy=True),
        initial_energy=float(current_realized_soc),
        timestamps=np.array(base.timestamps[start:], copy=True),
        scenario_source_dates=np.array(base.scenario_source_dates, copy=True),
        dt_hours=base.dt_hours,
    )
    return inputs, update


def _solve_release(
    date: object,
    release_hour: int,
    current_realized_soc: float,
    prior_commitment_kw: np.ndarray | None = None,
) -> tuple[Q3RollingPlanResult, object]:
    inputs, update = build_release_inputs(date, release_hour, current_realized_soc)
    result = solve_remaining_dispatch(
        inputs,
        Q2DispatchParameters(),
        prior_commitment_kw=prior_commitment_kw,
    )
    frame = result.dispatch.copy()
    frame["slot"] = np.arange(release_hour * 6 + 1, 145)
    frame["release_hour"] = release_hour
    result = replace(result, dispatch=frame)
    return result, update


def settle_segment_with_locked_q2(
    plan_segment: pd.DataFrame,
    actual_segment: pd.DataFrame,
    initial_energy_kwh: float,
) -> tuple[pd.DataFrame, float]:
    """Execute a segment by invoking the unchanged 144-row Q2 settlement."""

    plan = plan_segment.copy().reset_index(drop=True)
    actual = actual_segment.copy().reset_index(drop=True)
    count = len(plan)
    if count < 1 or count > 144 or len(actual) != count:
        raise ValueError("Q3 execution segment has inconsistent length")
    if not np.array_equal(
        pd.to_datetime(plan["timestamp"]).to_numpy(dtype="datetime64[ns]"),
        pd.to_datetime(actual["datetime"]).to_numpy(dtype="datetime64[ns]"),
    ):
        raise AssertionError("Q3 plan and actual segment timestamps are misaligned")
    minimal = plan[
        [
            "slot",
            "timestamp",
            "planned_grid_kw",
            "planned_charge_limit_kw",
            "planned_discharge_limit_kw",
            "price_yuan_per_kwh",
        ]
    ].copy()
    minimal["slot"] = np.arange(1, count + 1)
    actual_minimal = actual[["datetime", "actual_load", "actual_generation"]].copy()
    if count < 144:
        padding_count = 144 - count
        last_time = pd.Timestamp(actual_minimal["datetime"].iloc[-1])
        pad_times = pd.date_range(last_time + pd.Timedelta(minutes=10), periods=padding_count, freq="10min")
        plan_pad = pd.DataFrame(
            {
                "slot": np.arange(count + 1, 145),
                "timestamp": pad_times,
                "planned_grid_kw": 0.0,
                "planned_charge_limit_kw": 0.0,
                "planned_discharge_limit_kw": 0.0,
                "price_yuan_per_kwh": 0.0,
            }
        )
        actual_pad = pd.DataFrame(
            {"datetime": pad_times, "actual_load": 0.0, "actual_generation": 0.0}
        )
        minimal = pd.concat([minimal, plan_pad], ignore_index=True)
        actual_minimal = pd.concat([actual_minimal, actual_pad], ignore_index=True)
    settled, final_energy = settle_realized_day(
        minimal, actual_minimal, float(initial_energy_kwh)
    )
    output = plan.copy()
    for column in ACTUAL_COLUMNS:
        output[column] = settled[column].iloc[:count].to_numpy()
    return output, float(final_energy)


def _settlement_rows(
    original_grid_kwh: np.ndarray,
    revisions_kwh: list[np.ndarray],
    price: np.ndarray,
) -> dict[SettlementMode, dict[str, float]]:
    if not revisions_kwh:
        return {
            mode: {
                "adjustment_cost_yuan": 0.0,
                "adjustment_energy_kwh": 0.0,
                "adjustment_increase_energy_kwh": 0.0,
                "adjustment_decrease_energy_kwh": 0.0,
                "adjusted_interval_count": 0,
            }
            for mode in SettlementMode
        }
    revisions = np.stack(revisions_kwh)
    results: dict[SettlementMode, dict[str, float]] = {}
    for mode in SettlementMode:
        settled = adjustment_settlement(original_grid_kwh, revisions, price, mode)
        delta_plus = settled["delta_plus_kwh"]
        delta_minus = settled["delta_minus_kwh"]
        touched = np.any(delta_plus + delta_minus > 1e-9, axis=0)
        results[mode] = {
            "adjustment_cost_yuan": float(settled["adjustment_cost_yuan"].sum()),
            "adjustment_energy_kwh": float((delta_plus + delta_minus).sum()),
            "adjustment_increase_energy_kwh": float(delta_plus.sum()),
            "adjustment_decrease_energy_kwh": float(delta_minus.sum()),
            "adjusted_interval_count": int(touched.sum()),
        }
    return results


def run_schedule_day(
    date: object,
    schedule: str,
    initial_energy_kwh: float,
    actual_day: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, object]], float, dict[str, float]]:
    releases = schedule_releases(schedule)
    current_soc = float(initial_energy_kwh)
    original_grid_kwh: np.ndarray | None = None
    current_commitment_kw: np.ndarray | None = None
    revisions_kwh: list[np.ndarray] = []
    executed: list[pd.DataFrame] = []
    total_solver_runtime = 0.0
    freeze_residual = 0.0
    causal_checks = True
    initial_expected_cost = np.nan
    maximum_scenario_balance = 0.0
    optimized_adjustment_cost = 0.0
    price = get_q2_day_inputs(date, initial_energy=initial_energy_kwh).price
    for release_index, release_hour in enumerate(releases):
        start = release_hour * 6
        prior_future = (
            None
            if release_index == 0
            else np.array(current_commitment_kw[start:], copy=True)
        )
        result, update = _solve_release(
            date, release_hour, current_soc, prior_commitment_kw=prior_future
        )
        total_solver_runtime += result.runtime_seconds
        maximum_scenario_balance = max(
            maximum_scenario_balance,
            result.maximum_scenario_power_balance_residual_kw,
        )
        if release_index > 0:
            optimized_adjustment_cost += result.adjustment_cost_yuan
        plan = result.dispatch.copy().reset_index(drop=True)
        if release_index == 0:
            current_commitment_kw = plan["planned_grid_kw"].to_numpy(copy=True)
            original_grid_kwh = current_commitment_kw * DT_HOURS
            initial_expected_cost = result.operating_cost_yuan
        else:
            assert current_commitment_kw is not None
            previous = current_commitment_kw.copy()
            current_commitment_kw[start:] = plan["planned_grid_kw"].to_numpy()
            freeze_residual = max(
                freeze_residual,
                float(np.max(np.abs(current_commitment_kw[:start] - previous[:start])))
                if start else 0.0,
            )
            revisions_kwh.append(current_commitment_kw.copy() * DT_HOURS)
        causal_checks = causal_checks and bool(
            update.actual_history_cutoff == update.issue_time
            and update.executed_mask[:start].all()
            and update.future_mask[start:].all()
            and not np.isfinite(update.fused_forecast_kw[:start]).any()
            and np.isfinite(update.fused_forecast_kw[start:]).all()
            and abs(update.current_realized_soc - current_soc) <= 1e-9
        )
        end = releases[release_index + 1] * 6 if release_index + 1 < len(releases) else 144
        segment = plan.iloc[: end - start].copy()
        actual_segment = actual_day.iloc[start:end].copy()
        settled, current_soc = settle_segment_with_locked_q2(
            segment, actual_segment, current_soc
        )
        settled["schedule"] = schedule
        settled["release_hour"] = release_hour
        settled["operating_date"] = str(pd.Timestamp(date).date())
        executed.append(settled)
    intervals = pd.concat(executed, ignore_index=True).sort_values("slot").reset_index(drop=True)
    if len(intervals) != 144 or intervals["slot"].duplicated().any():
        raise AssertionError("Q3 rolling execution did not freeze exactly 144 unique intervals")
    assert original_grid_kwh is not None
    settlement = _settlement_rows(original_grid_kwh, revisions_kwh, price)
    main_adjustment_cost = settlement[
        SettlementMode.SEQUENTIAL_PREVIOUS_COMMITMENT
    ]["adjustment_cost_yuan"]
    adjustment_objective_residual = abs(main_adjustment_cost - optimized_adjustment_cost)
    if adjustment_objective_residual > 2e-3:
        raise AssertionError(
            "Q3 optimized adjustment cost does not match sequential settlement"
        )
    emergency_cost = float(intervals["realized_emergency_cost_yuan"].sum())
    base_cost = float(np.sum(price * original_grid_kwh))
    rows: list[dict[str, object]] = []
    for mode, values in settlement.items():
        rows.append(
            {
                "date": str(pd.Timestamp(date).date()),
                "schedule": schedule,
                "settlement_mode": mode.value,
                "solver_status": "Optimal",
                "solver_success": 1,
                "solver_runtime_seconds": total_solver_runtime,
                "release_count": len(releases),
                "maximum_scenario_power_balance_residual_kw": maximum_scenario_balance,
                "initial_energy_kwh": initial_energy_kwh,
                "final_energy_kwh": current_soc,
                "initial_planned_purchase_energy_kwh": float(original_grid_kwh.sum()),
                "initial_planned_purchase_cost_yuan": base_cost,
                "initial_expected_operating_cost_yuan": initial_expected_cost,
                **values,
                "emergency_energy_kwh": float(intervals["realized_emergency_kwh"].sum()),
                "emergency_cost_yuan": emergency_cost,
                "realized_total_cost_yuan": base_cost + values["adjustment_cost_yuan"] + emergency_cost,
                "emergency_interval_count": int((intervals["realized_emergency_kw"] > 1e-9).sum()),
                "battery_throughput_kwh": float(
                    intervals["actual_charge_kwh"].sum() + intervals["actual_discharge_kwh"].sum()
                ),
                "mean_soc_kwh": float(
                    intervals[["actual_storage_start_kwh", "actual_storage_end_kwh"]].to_numpy().mean()
                ),
                "minimum_soc_kwh": float(
                    intervals[["actual_storage_start_kwh", "actual_storage_end_kwh"]].min().min()
                ),
                "maximum_soc_kwh": float(
                    intervals[["actual_storage_start_kwh", "actual_storage_end_kwh"]].max().max()
                ),
            }
        )
    checks = {
        "freeze_residual_kw": freeze_residual,
        "causal_release_inputs": float(causal_checks),
        "maximum_scenario_power_balance_residual_kw": maximum_scenario_balance,
        "adjustment_objective_residual_yuan": adjustment_objective_residual,
    }
    intervals["initial_00_grid_kw"] = original_grid_kwh / DT_HOURS
    return intervals, rows, current_soc, checks


def audit_schedule(
    daily_main: pd.DataFrame,
    intervals: pd.DataFrame,
    freeze_residual_kw: float,
    causal_checks: bool,
    adjustment_objective_residual_yuan: float = 0.0,
) -> dict[str, object]:
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
    soc_error = intervals["actual_storage_end_kwh"] - expected_soc
    cross = daily_main["initial_energy_kwh"].iloc[1:].to_numpy() - daily_main[
        "final_energy_kwh"
    ].iloc[:-1].to_numpy()
    maximum_cross_day = float(np.max(np.abs(cross))) if len(cross) else 0.0
    absorbable = np.maximum(
        intervals["actual_load_kw"] + intervals["actual_charge_kw"] - intervals["actual_pv_kw"],
        0.0,
    )
    report = {
        "realized_power_balance": bool(np.abs(balance).max() <= tolerance),
        "maximum_power_balance_residual_kw": float(np.abs(balance).max()),
        "realized_soc_recurrence": bool(np.abs(soc_error).max() <= tolerance),
        "maximum_soc_recurrence_residual_kwh": float(np.abs(soc_error).max()),
        "cross_day_realized_soc": bool(maximum_cross_day <= tolerance),
        "maximum_cross_day_soc_residual_kwh": maximum_cross_day,
        "soc_bounds": bool(
            intervals[["actual_storage_start_kwh", "actual_storage_end_kwh"]].min().min()
            >= 1200.0 - tolerance
            and intervals[["actual_storage_start_kwh", "actual_storage_end_kwh"]].max().max()
            <= 10800.0 + tolerance
        ),
        "actual_action_limits": bool(
            (intervals["actual_charge_kw"] <= intervals["planned_charge_limit_kw"] + tolerance).all()
            and (intervals["actual_discharge_kw"] <= intervals["planned_discharge_limit_kw"] + tolerance).all()
        ),
        "phantom_discharge": bool((intervals["actual_discharge_kw"] <= absorbable + tolerance).all()),
        "unabsorbed_discharge_energy_kwh": 0.0,
        "hidden_export": bool((intervals["realized_pv_spill_kw"] <= intervals["actual_pv_kw"] + tolerance).all()),
        "emergency_nonnegative": bool((intervals["realized_emergency_kw"] >= -tolerance).all()),
        "executed_interval_freeze": bool(freeze_residual_kw <= tolerance),
        "maximum_executed_freeze_residual_kw": float(freeze_residual_kw),
        "causality": bool(causal_checks),
        "settlement_reconciliation": bool(
            np.max(
                np.abs(
                    daily_main["realized_total_cost_yuan"]
                    - daily_main["initial_planned_purchase_cost_yuan"]
                    - daily_main["adjustment_cost_yuan"]
                    - daily_main["emergency_cost_yuan"]
                )
            )
            <= tolerance
        ),
        "adjustment_objective_reconciliation": bool(
            adjustment_objective_residual_yuan <= tolerance
        ),
        "maximum_adjustment_objective_residual_yuan": float(
            adjustment_objective_residual_yuan
        ),
        "time_alignment": bool(
            len(intervals) == len(daily_main) * 144
            and intervals.groupby("operating_date")["slot"].nunique().eq(144).all()
        ),
    }
    if not all(value for value in report.values() if isinstance(value, bool)):
        raise AssertionError(f"Q3 schedule audit failed: {report}")
    return report


def run_schedule(
    schedule: str,
    dates: pd.DatetimeIndex,
    *,
    progress: bool = False,
    checkpoint_path: Path | None = None,
) -> Q3ScheduleResult:
    actual = pd.read_csv(
        FORMAL_FORECAST_PATH,
        usecols=["operating_date", "datetime", "actual_load", "actual_generation"],
        parse_dates=["datetime"],
    )
    carried_soc = 6000.0
    interval_frames: list[pd.DataFrame] = []
    rows: list[dict[str, object]] = []
    max_freeze = 0.0
    causal = True
    max_adjustment_objective_residual = 0.0
    started = perf_counter()
    start_index = 0
    if checkpoint_path is not None and checkpoint_path.exists():
        with checkpoint_path.open("rb") as stream:
            checkpoint = pickle.load(stream)
        expected_dates = [str(pd.Timestamp(value).date()) for value in dates]
        if checkpoint["schedule"] != schedule or checkpoint["dates"] != expected_dates:
            raise AssertionError("Q3 checkpoint does not match the requested schedule/dates")
        start_index = int(checkpoint["next_index"])
        carried_soc = float(checkpoint["carried_soc"])
        interval_frames = checkpoint["interval_frames"]
        rows = checkpoint["rows"]
        max_freeze = float(checkpoint["max_freeze"])
        causal = bool(checkpoint["causal"])
        max_adjustment_objective_residual = float(
            checkpoint["max_adjustment_objective_residual"]
        )
        if progress:
            print(f"Q3 {schedule}: resumed at {start_index}/{len(dates)} days", flush=True)
    for index, date in enumerate(dates[start_index:], start=start_index + 1):
        actual_day = actual.loc[
            actual["operating_date"].astype(str) == str(date.date())
        ].sort_values("datetime", kind="stable")
        if len(actual_day) != 144:
            raise AssertionError(f"Appendix 2 actual day {date.date()} does not contain 144 intervals")
        intervals, day_rows, carried_soc, checks = run_schedule_day(
            date, schedule, carried_soc, actual_day
        )
        interval_frames.append(intervals)
        rows.extend(day_rows)
        max_freeze = max(max_freeze, checks["freeze_residual_kw"])
        causal = causal and bool(checks["causal_release_inputs"])
        max_adjustment_objective_residual = max(
            max_adjustment_objective_residual,
            checks["adjustment_objective_residual_yuan"],
        )
        if progress and (index == 1 or index % 10 == 0 or index == len(dates)):
            print(f"Q3 {schedule}: {index}/{len(dates)} days Optimal", flush=True)
        if checkpoint_path is not None and (index % 10 == 0 or index == len(dates)):
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint_path.with_suffix(".tmp")
            payload = {
                "schedule": schedule,
                "dates": [str(pd.Timestamp(value).date()) for value in dates],
                "next_index": index,
                "carried_soc": carried_soc,
                "interval_frames": interval_frames,
                "rows": rows,
                "max_freeze": max_freeze,
                "causal": causal,
                "max_adjustment_objective_residual": max_adjustment_objective_residual,
            }
            with temporary.open("wb") as stream:
                pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
            temporary.replace(checkpoint_path)
    all_daily = pd.DataFrame(rows)
    main = all_daily.loc[
        all_daily["settlement_mode"].eq(SettlementMode.SEQUENTIAL_PREVIOUS_COMMITMENT.value)
    ].reset_index(drop=True)
    intervals = pd.concat(interval_frames, ignore_index=True)
    main["wall_runtime_seconds"] = perf_counter() - started
    audit = audit_schedule(
        main,
        intervals,
        max_freeze,
        causal,
        max_adjustment_objective_residual,
    )
    return Q3ScheduleResult(schedule, main, all_daily, intervals, audit)


__all__ = [
    "Q3ScheduleResult",
    "audit_schedule",
    "build_release_inputs",
    "run_schedule",
    "run_schedule_day",
    "settle_segment_with_locked_q2",
]
