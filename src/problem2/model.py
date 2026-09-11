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
    """Battery, emergency-pricing, and terminal-value parameters."""

    capacity_kwh: float = 12_000.0
    minimum_energy_kwh: float = 1_200.0
    maximum_energy_kwh: float = 10_800.0
    maximum_charge_kw: float = 5_000.0
    maximum_discharge_kw: float = 5_000.0
    charge_efficiency: float = 0.9
    discharge_efficiency: float = 0.9
    emergency_price_multiplier: float = 5.0
    terminal_reserve_quantile: float = 0.80
    terminal_value_price_quantiles: tuple[float, ...] = (0.90, 0.50, 0.10)


@dataclass(frozen=True)
class Q2DayResult:
    """Optimal shared plan, scenario recourse, and cost accounting for one day."""

    date: object
    dispatch: pd.DataFrame
    emergency_kw: np.ndarray
    spill_kw: np.ndarray
    planned_grid_used_kw: np.ndarray
    unused_planned_grid_kw: np.ndarray
    scenario_source_dates: np.ndarray
    status: str
    runtime_seconds: float
    solver_name: str
    planned_purchase_cost_yuan: float
    expected_emergency_cost_yuan: float
    operating_cost_yuan: float
    terminal_value_credit_yuan: float
    optimization_objective_yuan: float
    terminal_value_breakpoints_kwh: tuple[float, ...]
    terminal_value_rates_yuan_per_kwh: tuple[float, ...]

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
) -> Q2DayResult:
    """Solve one Q2 day with shared decisions and scenario-specific recourse.

    Paid day-ahead energy may be unused in each scenario. PV curtailment is kept
    as a separate physical quantity. A concave piecewise-linear continuation
    value rewards useful horizon-end energy with decreasing marginal value; it
    neither targets the day's initial energy nor imposes a daily reset.
    """

    parameters = parameters or Q2DispatchParameters()
    _validate_inputs(inputs, parameters)
    periods = range(inputs.horizon)
    scenarios = range(inputs.scenario_count)
    quantiles = tuple(float(value) for value in parameters.terminal_value_price_quantiles)
    if len(quantiles) != 3:
        raise ValueError("terminal value requires three decreasing marginal-value segments")
    if not 0.0 <= parameters.terminal_reserve_quantile <= 1.0:
        raise ValueError("terminal reserve quantile must lie in [0, 1]")
    if any(not 0.0 <= quantile <= 1.0 for quantile in quantiles):
        raise ValueError("terminal value price quantiles must lie in [0, 1]")
    scenario_net_load = inputs.load_scenarios - inputs.pv_scenarios
    point_net_load = inputs.load_forecast - inputs.pv_forecast
    positive_error_energy = np.maximum(
        scenario_net_load - point_net_load[None, :], 0.0
    ).sum(axis=1) * inputs.dt_hours
    uncertainty_reserve = float(
        np.clip(
            parameters.minimum_energy_kwh
            + np.quantile(positive_error_energy, parameters.terminal_reserve_quantile),
            parameters.minimum_energy_kwh + 1.0,
            parameters.maximum_energy_kwh - 2.0,
        )
    )
    upper_midpoint = uncertainty_reserve + (
        parameters.maximum_energy_kwh - uncertainty_reserve
    ) / 2.0
    breakpoints = (uncertainty_reserve, upper_midpoint, parameters.maximum_energy_kwh)
    terminal_value_rates = tuple(
        parameters.discharge_efficiency * float(np.quantile(inputs.price, quantile))
        for quantile in quantiles
    )
    if any(left < right for left, right in zip(terminal_value_rates, terminal_value_rates[1:])):
        raise ValueError("terminal marginal values must be nonincreasing")

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
    planned_grid_used = pulp.LpVariable.dicts(
        "planned_grid_used_kw", (scenarios, periods), lowBound=0
    )
    terminal_segments = [
        pulp.LpVariable(
            f"terminal_value_segment_{index + 1}_kwh",
            lowBound=0,
            upBound=upper - lower,
        )
        for index, (lower, upper) in enumerate(
            zip((parameters.minimum_energy_kwh,) + breakpoints[:-1], breakpoints)
        )
    ]

    model += energy[0] == inputs.initial_energy, "carried_initial_energy"
    model += energy[inputs.horizon] == parameters.minimum_energy_kwh + pulp.lpSum(
        terminal_segments
    ), "terminal_value_decomposition"
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
                planned_grid_used[s][t]
                + emergency[s][t]
                + pv_kw
                + discharge[t]
                == load_kw + charge[t] + spill[s][t]
            ), f"power_balance_s{s + 1:02d}_t{t + 1:03d}"
            model += (
                planned_grid_used[s][t] <= grid[t]
            ), f"planned_grid_use_limit_s{s + 1:02d}_t{t + 1:03d}"
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
    terminal_value_credit = pulp.lpSum(
        rate * segment for rate, segment in zip(terminal_value_rates, terminal_segments)
    )
    model += planned_cost + expected_emergency_cost - terminal_value_credit

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
    planned_grid_used_values = np.array(
        [[pulp.value(planned_grid_used[s][t]) for t in periods] for s in scenarios],
        dtype=float,
    )
    grid_values = np.array([pulp.value(grid[t]) for t in periods], dtype=float)
    unused_planned_grid_values = grid_values[None, :] - planned_grid_used_values
    dispatch = pd.DataFrame(
        {
            "date": str(inputs.date),
            "slot": np.arange(1, inputs.horizon + 1),
            "timestamp": inputs.timestamps,
            "price_yuan_per_kwh": inputs.price,
            "load_forecast_kw": inputs.load_forecast,
            "pv_forecast_kw": inputs.pv_forecast,
            "planned_grid_kw": grid_values,
            "charge_kw": [pulp.value(charge[t]) for t in periods],
            "discharge_kw": [pulp.value(discharge[t]) for t in periods],
            "storage_start_kwh": [pulp.value(energy[t]) for t in periods],
            "storage_end_kwh": [pulp.value(energy[t + 1]) for t in periods],
            "expected_emergency_kw": emergency_values.mean(axis=0),
            "expected_spill_kw": spill_values.mean(axis=0),
            "expected_planned_grid_used_kw": planned_grid_used_values.mean(axis=0),
            "expected_unused_planned_grid_kw": unused_planned_grid_values.mean(axis=0),
        }
    )
    for power_column in (
        "planned_grid",
        "charge",
        "discharge",
        "expected_emergency",
        "expected_spill",
        "expected_planned_grid_used",
        "expected_unused_planned_grid",
    ):
        dispatch[f"{power_column}_kwh"] = (
            dispatch[f"{power_column}_kw"] * inputs.dt_hours
        )

    planned_cost_value = float(pulp.value(planned_cost))
    emergency_cost_value = float(pulp.value(expected_emergency_cost))
    terminal_value_credit_value = float(pulp.value(terminal_value_credit))
    return Q2DayResult(
        date=inputs.date,
        dispatch=dispatch,
        emergency_kw=emergency_values,
        spill_kw=spill_values,
        planned_grid_used_kw=planned_grid_used_values,
        unused_planned_grid_kw=unused_planned_grid_values,
        scenario_source_dates=np.array(inputs.scenario_source_dates, copy=True),
        status=status,
        runtime_seconds=runtime,
        solver_name="PuLP HiGHS",
        planned_purchase_cost_yuan=planned_cost_value,
        expected_emergency_cost_yuan=emergency_cost_value,
        operating_cost_yuan=planned_cost_value + emergency_cost_value,
        terminal_value_credit_yuan=terminal_value_credit_value,
        optimization_objective_yuan=float(pulp.value(model.objective)),
        terminal_value_breakpoints_kwh=breakpoints,
        terminal_value_rates_yuan_per_kwh=terminal_value_rates,
    )
