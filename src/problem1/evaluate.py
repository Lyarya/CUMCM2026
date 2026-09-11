"""Validation and baseline comparison for Problem 1."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.problem1.model import DT_HOURS, DispatchParameters, DispatchResult


@dataclass(frozen=True)
class ValidationReport:
    period_count: int
    initial_energy_kwh: float
    terminal_energy_kwh: float
    minimum_energy_kwh: float
    maximum_energy_kwh: float
    maximum_charge_kw: float
    maximum_discharge_kw: float
    simultaneous_charge_discharge_count: int
    negative_grid_purchase_count: int
    maximum_spill_excess_kw: float
    maximum_power_balance_residual_kw: float
    maximum_energy_transition_residual_kwh: float
    objective_recalculation_error_yuan: float


def validate_dispatch(
    result: DispatchResult,
    parameters: DispatchParameters | None = None,
    *,
    tolerance: float = 1e-3,
) -> ValidationReport:
    """Validate every physical and accounting constraint requested for Q1."""
    parameters = parameters or DispatchParameters()
    frame = result.dispatch
    power_residual = (
        frame["grid_purchase_kw"]
        + frame["pv_forecast_kw"]
        + frame["discharge_kw"]
        - frame["load_kw"]
        - frame["charge_kw"]
        - frame["spill_kw"]
    )
    expected_end = (
        frame["storage_start_kwh"]
        + parameters.charge_efficiency * frame["charge_kw"] * DT_HOURS
        - frame["discharge_kw"] * DT_HOURS / parameters.discharge_efficiency
    )
    transition_residual = frame["storage_end_kwh"] - expected_end
    objective = float(
        (frame["price_yuan_per_kwh"] * frame["grid_purchase_kw"] * DT_HOURS).sum()
    )
    report = ValidationReport(
        period_count=len(frame),
        initial_energy_kwh=float(frame["storage_start_kwh"].iloc[0]),
        terminal_energy_kwh=float(frame["storage_end_kwh"].iloc[-1]),
        minimum_energy_kwh=float(
            min(frame["storage_start_kwh"].min(), frame["storage_end_kwh"].min())
        ),
        maximum_energy_kwh=float(
            max(frame["storage_start_kwh"].max(), frame["storage_end_kwh"].max())
        ),
        maximum_charge_kw=float(frame["charge_kw"].max()),
        maximum_discharge_kw=float(frame["discharge_kw"].max()),
        simultaneous_charge_discharge_count=int(
            ((frame["charge_kw"] > tolerance) & (frame["discharge_kw"] > tolerance)).sum()
        ),
        negative_grid_purchase_count=int((frame["grid_purchase_kw"] < -tolerance).sum()),
        maximum_spill_excess_kw=float(
            np.maximum(frame["spill_kw"] - frame["pv_forecast_kw"], 0.0).max()
        ),
        maximum_power_balance_residual_kw=float(np.abs(power_residual).max()),
        maximum_energy_transition_residual_kwh=float(np.abs(transition_residual).max()),
        objective_recalculation_error_yuan=abs(objective - result.objective_yuan),
    )
    checks = {
        "period count": report.period_count == 144,
        "initial energy": abs(report.initial_energy_kwh - parameters.initial_energy_kwh) <= tolerance,
        "terminal energy": abs(report.terminal_energy_kwh - parameters.terminal_energy_kwh) <= tolerance,
        "minimum energy": report.minimum_energy_kwh >= parameters.minimum_energy_kwh - tolerance,
        "maximum energy": report.maximum_energy_kwh <= parameters.maximum_energy_kwh + tolerance,
        "charge power": report.maximum_charge_kw <= parameters.maximum_charge_kw + tolerance,
        "discharge power": report.maximum_discharge_kw <= parameters.maximum_discharge_kw + tolerance,
        "mutual exclusion": report.simultaneous_charge_discharge_count == 0,
        "nonnegative grid purchase": report.negative_grid_purchase_count == 0,
        "spill bounded by photovoltaic generation": report.maximum_spill_excess_kw <= tolerance,
        "power balance": report.maximum_power_balance_residual_kw <= tolerance,
        "energy transition": report.maximum_energy_transition_residual_kwh <= tolerance,
        "objective accounting": report.objective_recalculation_error_yuan <= tolerance,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"Q1 validation failed: {', '.join(failed)}; {asdict(report)}")
    return report


def compute_baseline(data: pd.DataFrame) -> dict[str, float]:
    """Evaluate the deterministic no-storage baseline."""
    grid_kw = np.maximum(data["load_kw"].to_numpy() - data["pv_forecast_kw"].to_numpy(), 0.0)
    spill_kw = np.maximum(data["pv_forecast_kw"].to_numpy() - data["load_kw"].to_numpy(), 0.0)
    return {
        "baseline_grid_energy_kwh": float(grid_kw.sum() * DT_HOURS),
        "baseline_cost_yuan": float((data["price_yuan_per_kwh"].to_numpy() * grid_kw).sum() * DT_HOURS),
        "baseline_spill_energy_kwh": float(spill_kw.sum() * DT_HOURS),
    }


def build_summary(
    result: DispatchResult,
    validation: ValidationReport,
    baseline: dict[str, float],
    parameters: DispatchParameters | None = None,
) -> dict[str, float | str | int]:
    """Build the compact numerical summary saved by the official runner."""
    parameters = parameters or DispatchParameters()
    frame = result.dispatch
    saving = baseline["baseline_cost_yuan"] - result.objective_yuan
    return {
        "solver": result.solver_name,
        "solver_status": result.status,
        "runtime_seconds": result.runtime_seconds,
        "optimal_cost_yuan": result.objective_yuan,
        "total_grid_energy_kwh": float(frame["grid_purchase_kwh"].sum()),
        "total_charge_energy_kwh": float(frame["charge_kwh"].sum()),
        "total_discharge_energy_kwh": float(frame["discharge_kwh"].sum()),
        "total_spill_energy_kwh": float(frame["spill_kwh"].sum()),
        "minimum_storage_energy_kwh": validation.minimum_energy_kwh,
        "maximum_storage_energy_kwh": validation.maximum_energy_kwh,
        "minimum_soc_percent": 100 * validation.minimum_energy_kwh / parameters.capacity_kwh,
        "maximum_soc_percent": 100 * validation.maximum_energy_kwh / parameters.capacity_kwh,
        "terminal_storage_energy_kwh": validation.terminal_energy_kwh,
        "charge_period_count": int((frame["charge_kw"] > 1e-4).sum()),
        "discharge_period_count": int((frame["discharge_kw"] > 1e-4).sum()),
        **baseline,
        "cost_saving_yuan": saving,
        "cost_saving_percent": 100 * saving / baseline["baseline_cost_yuan"],
        "maximum_spill_excess_kw": validation.maximum_spill_excess_kw,
        "maximum_power_balance_residual_kw": validation.maximum_power_balance_residual_kw,
        "maximum_energy_transition_residual_kwh": validation.maximum_energy_transition_residual_kwh,
        "objective_recalculation_error_yuan": validation.objective_recalculation_error_yuan,
    }
