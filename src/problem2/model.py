"""Expected-cost stochastic MILP for Problem 2 day-ahead dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd
import pulp

from src.problem2.forecast_interface import Q2DayInputs


@dataclass(frozen=True)
class Q2DispatchParameters:
    """Battery, emergency-pricing, and linear terminal-treatment parameters."""

    capacity_kwh: float = 12_000.0
    minimum_energy_kwh: float = 1_200.0
    maximum_energy_kwh: float = 10_800.0
    maximum_charge_kw: float = 5_000.0
    maximum_discharge_kw: float = 5_000.0
    charge_efficiency: float = 0.9
    discharge_efficiency: float = 0.9
    emergency_price_multiplier: float = 5.0
    terminal_penalty_yuan_per_kwh: float | None = None


@dataclass(frozen=True)
class Q2DayResult:
    """Optimal shared plan, scenario recourse, and cost accounting for one day."""

    date: object
    dispatch: pd.DataFrame
    emergency_kw: np.ndarray
    spill_kw: np.ndarray
    scenario_source_dates: np.ndarray
    status: str
    runtime_seconds: float
    solver_name: str
    planned_purchase_cost_yuan: float
    expected_emergency_cost_yuan: float
    operating_cost_yuan: float
    terminal_penalty_yuan: float
    optimization_objective_yuan: float
    terminal_reference_kwh: float
    terminal_penalty_rate_yuan_per_kwh: float

    @property
    def final_energy_kwh(self) -> float:
        return float(self.dispatch["storage_end_kwh"].iloc[-1])


def _validate_inputs(inputs: Q2DayInputs, parameters: Q2DispatchParameters) -> None:
    expected_scenario_shape = (inputs.scenario_count, inputs.horizon)
    if inputs.horizon != 144:
        raise ValueError(f"Q2 requires 144 ten-minute periods, got {inputs.horizon}")
    if inputs.scenario_count != 50:
        raise ValueError(f"Q2 requires 50 paired scenarios, got {inputs.scenario_count}")
    if inputs.load_scenarios.shape != expected_scenario_shape:
        raise ValueError("load scenario array does not match [scenario, time]")
    if inputs.pv_scenarios.shape != expected_scenario_shape:
        raise ValueError("PV scenario array does not match [scenario, time]")
    numeric = np.concatenate(
        [
            inputs.load_forecast,
            inputs.pv_forecast,
            inputs.load_scenarios.reshape(-1),
            inputs.pv_scenarios.reshape(-1),
            inputs.price,
            np.array([inputs.initial_energy, inputs.dt_hours]),
        ]
    )
    if not np.isfinite(numeric).all():
        raise ValueError("Q2 optimizer input contains NaN or infinite values")
    if (inputs.pv_scenarios < 0).any():
        raise ValueError("PV scenarios must be nonnegative")
    if not (
        parameters.minimum_energy_kwh
        <= inputs.initial_energy
        <= parameters.maximum_energy_kwh
    ):
        raise ValueError("initial energy is outside the permitted battery range")


def solve_expected_cost_dispatch(
    inputs: Q2DayInputs,
    parameters: Q2DispatchParameters | None = None,
    *,
    terminal_reference_kwh: float | None = None,
) -> Q2DayResult:
    """Solve one Q2 day with shared decisions and scenario-specific recourse.

    The terminal treatment is a linear L1 penalty around the energy carried into
    the day. It is deliberately soft: no daily terminal equality or reset to
    6000 kWh is imposed. The caller must pass this result's final energy to the
    next day's ``get_q2_day_inputs`` call.
    """

    parameters = parameters or Q2DispatchParameters()
    _validate_inputs(inputs, parameters)
    periods = range(inputs.horizon)
    scenarios = range(inputs.scenario_count)
    reference = float(
        inputs.initial_energy
        if terminal_reference_kwh is None
        else terminal_reference_kwh
    )
    if not parameters.minimum_energy_kwh <= reference <= parameters.maximum_energy_kwh:
        raise ValueError("terminal reference is outside the permitted battery range")
    penalty_rate = parameters.terminal_penalty_yuan_per_kwh
    if penalty_rate is None:
        penalty_rate = parameters.discharge_efficiency * float(np.max(inputs.price))
    if not np.isfinite(penalty_rate) or penalty_rate < 0:
        raise ValueError("terminal penalty rate must be finite and nonnegative")

    model = pulp.LpProblem(
        f"CUMCM2026_Q2_Expected_Cost_{inputs.date}", pulp.LpMinimize
    )
    grid = pulp.LpVariable.dicts("planned_grid_kw", periods, lowBound=0)
    charge = pulp.LpVariable.dicts("charge_kw", periods, lowBound=0)
    discharge = pulp.LpVariable.dicts("discharge_kw", periods, lowBound=0)
    mode = pulp.LpVariable.dicts("charge_mode", periods, cat=pulp.LpBinary)
    energy = pulp.LpVariable.dicts(
        "energy_kwh",
        range(inputs.horizon + 1),
        lowBound=parameters.minimum_energy_kwh,
        upBound=parameters.maximum_energy_kwh,
    )
    emergency = pulp.LpVariable.dicts(
        "emergency_kw", (scenarios, periods), lowBound=0
    )
    spill = pulp.LpVariable.dicts("spill_kw", (scenarios, periods), lowBound=0)
    terminal_above = pulp.LpVariable("terminal_above_reference_kwh", lowBound=0)
    terminal_below = pulp.LpVariable("terminal_below_reference_kwh", lowBound=0)

    model += energy[0] == inputs.initial_energy, "carried_initial_energy"
    model += (
        energy[inputs.horizon] - reference == terminal_above - terminal_below
    ), "soft_terminal_deviation"
    for t in periods:
        model += (
            energy[t + 1]
            == energy[t]
            + parameters.charge_efficiency * charge[t] * inputs.dt_hours
            - discharge[t] * inputs.dt_hours / parameters.discharge_efficiency
        ), f"energy_transition_{t + 1:03d}"
        model += (
            charge[t] <= parameters.maximum_charge_kw * mode[t]
        ), f"charge_limit_{t + 1:03d}"
        model += (
            discharge[t] <= parameters.maximum_discharge_kw * (1 - mode[t])
        ), f"discharge_limit_{t + 1:03d}"
        for s in scenarios:
            pv_kw = float(inputs.pv_scenarios[s, t])
            load_kw = float(inputs.load_scenarios[s, t])
            model += (
                grid[t]
                + emergency[s][t]
                + pv_kw
                + discharge[t]
                == load_kw + charge[t] + spill[s][t]
            ), f"power_balance_s{s + 1:02d}_t{t + 1:03d}"
            model += spill[s][t] <= pv_kw, f"spill_limit_s{s + 1:02d}_t{t + 1:03d}"

    planned_cost = pulp.lpSum(
        float(inputs.price[t]) * grid[t] * inputs.dt_hours for t in periods
    )
    expected_emergency_cost = (
        pulp.lpSum(
            parameters.emergency_price_multiplier
            * float(inputs.price[t])
            * emergency[s][t]
            * inputs.dt_hours
            for s in scenarios
            for t in periods
        )
        / inputs.scenario_count
    )
    terminal_penalty = penalty_rate * (terminal_above + terminal_below)
    model += planned_cost + expected_emergency_cost + terminal_penalty

    solver = pulp.HiGHS(msg=False, threads=1)
    started = perf_counter()
    model.solve(solver)
    runtime = perf_counter() - started
    status = pulp.LpStatus[model.status]
    if status != "Optimal":
        raise RuntimeError(
            f"HiGHS did not find an optimal Q2 solution for {inputs.date}: {status}"
        )

    emergency_values = np.array(
        [[pulp.value(emergency[s][t]) for t in periods] for s in scenarios],
        dtype=float,
    )
    spill_values = np.array(
        [[pulp.value(spill[s][t]) for t in periods] for s in scenarios],
        dtype=float,
    )
    dispatch = pd.DataFrame(
        {
            "date": str(inputs.date),
            "slot": np.arange(1, inputs.horizon + 1),
            "timestamp": inputs.timestamps,
            "price_yuan_per_kwh": inputs.price,
            "load_forecast_kw": inputs.load_forecast,
            "pv_forecast_kw": inputs.pv_forecast,
            "planned_grid_kw": [pulp.value(grid[t]) for t in periods],
            "charge_kw": [pulp.value(charge[t]) for t in periods],
            "discharge_kw": [pulp.value(discharge[t]) for t in periods],
            "storage_start_kwh": [pulp.value(energy[t]) for t in periods],
            "storage_end_kwh": [pulp.value(energy[t + 1]) for t in periods],
            "expected_emergency_kw": emergency_values.mean(axis=0),
            "expected_spill_kw": spill_values.mean(axis=0),
        }
    )
    for power_column in (
        "planned_grid",
        "charge",
        "discharge",
        "expected_emergency",
        "expected_spill",
    ):
        dispatch[f"{power_column}_kwh"] = (
            dispatch[f"{power_column}_kw"] * inputs.dt_hours
        )

    planned_cost_value = float(pulp.value(planned_cost))
    emergency_cost_value = float(pulp.value(expected_emergency_cost))
    terminal_penalty_value = float(pulp.value(terminal_penalty))
    return Q2DayResult(
        date=inputs.date,
        dispatch=dispatch,
        emergency_kw=emergency_values,
        spill_kw=spill_values,
        scenario_source_dates=np.array(inputs.scenario_source_dates, copy=True),
        status=status,
        runtime_seconds=runtime,
        solver_name="PuLP HiGHS",
        planned_purchase_cost_yuan=planned_cost_value,
        expected_emergency_cost_yuan=emergency_cost_value,
        operating_cost_yuan=planned_cost_value + emergency_cost_value,
        terminal_penalty_yuan=terminal_penalty_value,
        optimization_objective_yuan=float(pulp.value(model.objective)),
        terminal_reference_kwh=reference,
        terminal_penalty_rate_yuan_per_kwh=float(penalty_rate),
    )
