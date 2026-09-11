"""Validation and machine-readable summaries for the Q2 stochastic MILP."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from src.problem2.forecast_interface import Q2DayInputs
from src.problem2.model import Q2DayResult, Q2DispatchParameters


@dataclass(frozen=True)
class Q2ValidationReport:
    period_count: int
    scenario_count: int
    minimum_energy_kwh: float
    maximum_energy_kwh: float
    maximum_charge_kw: float
    maximum_discharge_kw: float
    simultaneous_charge_discharge_count: int
    negative_grid_purchase_count: int
    negative_emergency_purchase_count: int
    maximum_spill_excess_kw: float
    maximum_power_balance_residual_kw: float
    maximum_energy_transition_residual_kwh: float
    planned_cost_recalculation_error_yuan: float
    emergency_cost_recalculation_error_yuan: float
    scenario_pairs_unchanged: bool


def validate_q2_day(
    inputs: Q2DayInputs,
    result: Q2DayResult,
    parameters: Q2DispatchParameters | None = None,
    *,
    tolerance: float = 2e-3,
) -> Q2ValidationReport:
    """Check all physical, scenario, pairing, and price-accounting constraints."""

    parameters = parameters or Q2DispatchParameters()
    frame = result.dispatch
    grid = frame["planned_grid_kw"].to_numpy(dtype=float)
    charge = frame["charge_kw"].to_numpy(dtype=float)
    discharge = frame["discharge_kw"].to_numpy(dtype=float)
    power_residual = (
        grid[None, :]
        + result.emergency_kw
        + inputs.pv_scenarios
        + discharge[None, :]
        - inputs.load_scenarios
        - charge[None, :]
        - result.spill_kw
    )
    expected_end = (
        frame["storage_start_kwh"].to_numpy(dtype=float)
        + parameters.charge_efficiency * charge * inputs.dt_hours
        - discharge * inputs.dt_hours / parameters.discharge_efficiency
    )
    energy_residual = frame["storage_end_kwh"].to_numpy(dtype=float) - expected_end
    energy_values = np.concatenate(
        [
            frame["storage_start_kwh"].to_numpy(dtype=float),
            frame["storage_end_kwh"].tail(1).to_numpy(dtype=float),
        ]
    )
    planned_cost = float(np.sum(inputs.price * grid * inputs.dt_hours))
    emergency_cost = float(
        parameters.emergency_price_multiplier
        * np.mean(np.sum(inputs.price[None, :] * result.emergency_kw * inputs.dt_hours, axis=1))
    )
    report = Q2ValidationReport(
        period_count=len(frame),
        scenario_count=result.emergency_kw.shape[0],
        minimum_energy_kwh=float(np.min(energy_values)),
        maximum_energy_kwh=float(np.max(energy_values)),
        maximum_charge_kw=float(np.max(charge)),
        maximum_discharge_kw=float(np.max(discharge)),
        simultaneous_charge_discharge_count=int(
            np.count_nonzero((charge > tolerance) & (discharge > tolerance))
        ),
        negative_grid_purchase_count=int(np.count_nonzero(grid < -tolerance)),
        negative_emergency_purchase_count=int(
            np.count_nonzero(result.emergency_kw < -tolerance)
        ),
        maximum_spill_excess_kw=float(
            np.max(np.maximum(result.spill_kw - inputs.pv_scenarios, 0.0))
        ),
        maximum_power_balance_residual_kw=float(np.max(np.abs(power_residual))),
        maximum_energy_transition_residual_kwh=float(np.max(np.abs(energy_residual))),
        planned_cost_recalculation_error_yuan=abs(
            planned_cost - result.planned_purchase_cost_yuan
        ),
        emergency_cost_recalculation_error_yuan=abs(
            emergency_cost - result.expected_emergency_cost_yuan
        ),
        scenario_pairs_unchanged=bool(
            np.array_equal(result.scenario_source_dates, inputs.scenario_source_dates)
            and result.emergency_kw.shape == inputs.load_scenarios.shape
            and result.spill_kw.shape == inputs.pv_scenarios.shape
        ),
    )
    checks = {
        "144 intervals": report.period_count == 144,
        "50 paired scenarios": report.scenario_count == 50 and report.scenario_pairs_unchanged,
        "minimum SOC": report.minimum_energy_kwh >= parameters.minimum_energy_kwh - tolerance,
        "maximum SOC": report.maximum_energy_kwh <= parameters.maximum_energy_kwh + tolerance,
        "charge power": report.maximum_charge_kw <= parameters.maximum_charge_kw + tolerance,
        "discharge power": report.maximum_discharge_kw <= parameters.maximum_discharge_kw + tolerance,
        "mutual exclusion": report.simultaneous_charge_discharge_count == 0,
        "nonnegative planned grid": report.negative_grid_purchase_count == 0,
        "nonnegative emergency grid": report.negative_emergency_purchase_count == 0,
        "spill upper bound": report.maximum_spill_excess_kw <= tolerance,
        "scenario power balance": report.maximum_power_balance_residual_kw <= tolerance,
        "SOC recursion": report.maximum_energy_transition_residual_kwh <= tolerance,
        "planned cost": report.planned_cost_recalculation_error_yuan <= tolerance,
        "5x emergency cost": report.emergency_cost_recalculation_error_yuan <= tolerance,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"Q2 validation failed: {', '.join(failed)}; {asdict(report)}")
    return report


def build_daily_summary(
    result: Q2DayResult,
    validation: Q2ValidationReport,
    inputs: Q2DayInputs,
) -> dict[str, object]:
    """Build one row for the chronological Q2 checkpoint summary."""

    frame = result.dispatch
    return {
        "date": str(result.date),
        "solver": result.solver_name,
        "status": result.status,
        "runtime_seconds": result.runtime_seconds,
        "initial_energy_kwh": float(inputs.initial_energy),
        "final_energy_kwh": result.final_energy_kwh,
        "terminal_reference_kwh": result.terminal_reference_kwh,
        "terminal_penalty_rate_yuan_per_kwh": result.terminal_penalty_rate_yuan_per_kwh,
        "terminal_penalty_yuan": result.terminal_penalty_yuan,
        "planned_purchase_cost_yuan": result.planned_purchase_cost_yuan,
        "expected_emergency_cost_yuan": result.expected_emergency_cost_yuan,
        "expected_total_cost_yuan": result.operating_cost_yuan,
        "optimization_objective_yuan": result.optimization_objective_yuan,
        "planned_purchase_energy_kwh": float(frame["planned_grid_kwh"].sum()),
        "expected_emergency_energy_kwh": float(frame["expected_emergency_kwh"].sum()),
        "charge_energy_kwh": float(frame["charge_kwh"].sum()),
        "discharge_energy_kwh": float(frame["discharge_kwh"].sum()),
        "expected_spill_energy_kwh": float(frame["expected_spill_kwh"].sum()),
        "minimum_energy_kwh": validation.minimum_energy_kwh,
        "maximum_energy_kwh": validation.maximum_energy_kwh,
        "maximum_power_balance_residual_kw": validation.maximum_power_balance_residual_kw,
        "maximum_soc_residual_kwh": validation.maximum_energy_transition_residual_kwh,
    }
