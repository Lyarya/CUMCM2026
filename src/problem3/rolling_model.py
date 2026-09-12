"""Remaining-horizon Q3 planning MILP using the locked Q2 equations.

This module owns only the day-ahead re-optimization wrapper.  Actual physical
settlement is deliberately delegated to ``src.problem2.run.settle_realized_day``.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd
import pulp

from src.problem2.model import Q2DispatchParameters
from src.problem2.forecast_interface import Q2DayInputs


@dataclass(frozen=True)
class Q3RollingPlanResult:
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
    adjustment_cost_yuan: float
    expected_emergency_cost_yuan: float
    operating_cost_yuan: float
    terminal_value_credit_yuan: float
    optimization_objective_yuan: float
    terminal_value_breakpoints_kwh: tuple[float, ...]
    terminal_value_rates_yuan_per_kwh: tuple[float, ...]
    maximum_scenario_power_balance_residual_kw: float

    @property
    def final_energy_kwh(self) -> float:
        return float(self.dispatch["storage_end_kwh"].iloc[-1])


def _terminal_segments(
    inputs: Q2DayInputs, parameters: Q2DispatchParameters
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    net = inputs.load_scenarios - inputs.pv_scenarios
    point = inputs.load_forecast - inputs.pv_forecast
    positive_error = np.maximum(net - point[None, :], 0.0).sum(axis=1) * inputs.dt_hours
    reserve = float(
        np.clip(
            parameters.minimum_energy_kwh
            + np.quantile(positive_error, parameters.terminal_reserve_quantile),
            parameters.minimum_energy_kwh + 1.0,
            parameters.maximum_energy_kwh - 2.0,
        )
    )
    midpoint = reserve + (parameters.maximum_energy_kwh - reserve) / 2.0
    return (
        (reserve, midpoint, parameters.maximum_energy_kwh),
        tuple(
            parameters.discharge_efficiency * float(np.quantile(inputs.price, q))
            for q in parameters.terminal_value_price_quantiles
        ),
    )


def solve_remaining_dispatch(
    inputs: Q2DayInputs,
    parameters: Q2DispatchParameters | None = None,
    *,
    prior_commitment_kw: np.ndarray | None = None,
) -> Q3RollingPlanResult:
    """Solve one remaining Q3 horizon with the locked Q2 physical equations.

    At 00:00, grid energy is priced at the ordinary tariff.  At later legal
    releases, the already paid commitment is sunk and the decision objective
    uses the official incremental settlement: 1.5p for an increase and a
    0.5p credit for a decrease, both relative to the previous commitment.
    """

    parameters = parameters or Q2DispatchParameters()
    if inputs.horizon < 1 or inputs.horizon > 144:
        raise ValueError(f"remaining Q3 horizon must lie in [1, 144], got {inputs.horizon}")
    if inputs.scenario_count != 50:
        raise ValueError("Q3 requires the locked 50 paired scenarios")
    expected_shape = (inputs.scenario_count, inputs.horizon)
    if inputs.load_scenarios.shape != expected_shape or inputs.pv_scenarios.shape != expected_shape:
        raise ValueError("Q3 scenario arrays must have shape [50, remaining horizon]")
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
        raise ValueError("Q3 rolling input contains NaN or infinite values")
    if (inputs.pv_scenarios < 0).any():
        raise ValueError("Q3 PV scenarios must be nonnegative")
    if not parameters.minimum_energy_kwh <= inputs.initial_energy <= parameters.maximum_energy_kwh:
        raise ValueError("Q3 initial energy is outside the permitted battery range")
    if prior_commitment_kw is not None:
        prior_commitment_kw = np.asarray(prior_commitment_kw, dtype=float)
        if prior_commitment_kw.shape != (inputs.horizon,):
            raise ValueError("prior Q3 commitment must match the remaining horizon")
        if not np.isfinite(prior_commitment_kw).all() or (prior_commitment_kw < -2e-6).any():
            raise ValueError("prior Q3 commitment must be finite and nonnegative")
        prior_commitment_kw = np.maximum(prior_commitment_kw, 0.0)
    breakpoints, rates = _terminal_segments(inputs, parameters)
    periods = range(inputs.horizon)
    scenarios = range(inputs.scenario_count)
    model = pulp.LpProblem(f"CUMCM2026_Q3_Rolling_{inputs.date}", pulp.LpMinimize)
    grid = pulp.LpVariable.dicts("q3_planned_grid_kw", periods, lowBound=0)
    charge = pulp.LpVariable.dicts("q3_planned_charge_kw", periods, lowBound=0)
    discharge = pulp.LpVariable.dicts("q3_planned_discharge_kw", periods, lowBound=0)
    mode = pulp.LpVariable.dicts("q3_charge_mode", periods, cat=pulp.LpBinary)
    energy = pulp.LpVariable.dicts(
        "q3_energy_kwh",
        range(inputs.horizon + 1),
        lowBound=parameters.minimum_energy_kwh,
        upBound=parameters.maximum_energy_kwh,
    )
    emergency = pulp.LpVariable.dicts("q3_emergency_kw", (scenarios, periods), lowBound=0)
    spill = pulp.LpVariable.dicts("q3_spill_kw", (scenarios, periods), lowBound=0)
    used = pulp.LpVariable.dicts("q3_planned_grid_used_kw", (scenarios, periods), lowBound=0)
    increase = None
    decrease = None
    if prior_commitment_kw is not None:
        increase = pulp.LpVariable.dicts("q3_adjustment_increase_kw", periods, lowBound=0)
        decrease = pulp.LpVariable.dicts("q3_adjustment_decrease_kw", periods, lowBound=0)
    segment_vars = [
        pulp.LpVariable(
            f"q3_terminal_segment_{index + 1}_kwh", lowBound=0, upBound=upper - lower
        )
        for index, (lower, upper) in enumerate(
            zip((parameters.minimum_energy_kwh,) + breakpoints[:-1], breakpoints)
        )
    ]
    model += energy[0] == inputs.initial_energy, "q3_initial_energy"
    model += energy[inputs.horizon] == parameters.minimum_energy_kwh + pulp.lpSum(segment_vars)
    for t in periods:
        model += energy[t + 1] == (
            energy[t]
            + parameters.charge_efficiency * charge[t] * inputs.dt_hours
            - discharge[t] * inputs.dt_hours / parameters.discharge_efficiency
        )
        model += charge[t] <= parameters.maximum_charge_kw * mode[t]
        model += discharge[t] <= parameters.maximum_discharge_kw * (1 - mode[t])
        if prior_commitment_kw is not None:
            assert increase is not None and decrease is not None
            model += (
                grid[t] - float(prior_commitment_kw[t]) == increase[t] - decrease[t]
            )
        for s in scenarios:
            model += (
                used[s][t]
                + emergency[s][t]
                + float(inputs.pv_scenarios[s, t])
                + discharge[t]
                == float(inputs.load_scenarios[s, t]) + charge[t] + spill[s][t]
            )
            model += used[s][t] <= grid[t]
            model += spill[s][t] <= float(inputs.pv_scenarios[s, t])
    planned_cost = pulp.lpSum(
        float(inputs.price[t]) * grid[t] * inputs.dt_hours for t in periods
    )
    if prior_commitment_kw is None:
        decision_energy_cost = planned_cost
    else:
        assert increase is not None and decrease is not None
        decision_energy_cost = pulp.lpSum(
            float(inputs.price[t])
            * inputs.dt_hours
            * (1.5 * increase[t] - 0.5 * decrease[t])
            for t in periods
        )
    scenario_costs = [
        pulp.lpSum(
            parameters.emergency_price_multiplier
            * float(inputs.price[t])
            * emergency[s][t]
            * inputs.dt_hours
            for t in periods
        )
        for s in scenarios
    ]
    expected_emergency = pulp.lpSum(scenario_costs) / inputs.scenario_count
    terminal_credit = pulp.lpSum(rate * variable for rate, variable in zip(rates, segment_vars))
    model += decision_energy_cost + expected_emergency - terminal_credit
    solver = pulp.HiGHS(msg=False, threads=1)
    started = perf_counter()
    model.solve(solver)
    runtime = perf_counter() - started
    status = pulp.LpStatus[model.status]
    if status != "Optimal":
        raise RuntimeError(f"Q3 rolling MILP failed for {inputs.date}: {status}")
    emergency_values = np.array(
        [[pulp.value(emergency[s][t]) for t in periods] for s in scenarios], dtype=float
    )
    spill_values = np.array(
        [[pulp.value(spill[s][t]) for t in periods] for s in scenarios], dtype=float
    )
    used_values = np.array(
        [[pulp.value(used[s][t]) for t in periods] for s in scenarios], dtype=float
    )
    grid_values = np.array([pulp.value(grid[t]) for t in periods], dtype=float)
    if (grid_values < -2e-6).any():
        raise AssertionError("Q3 solver returned materially negative planned purchase")
    grid_values = np.maximum(grid_values, 0.0)
    unused_values = grid_values[None, :] - used_values
    charge_values = np.array([pulp.value(charge[t]) for t in periods], dtype=float)
    discharge_values = np.array([pulp.value(discharge[t]) for t in periods], dtype=float)
    scenario_balance = (
        used_values
        + emergency_values
        + inputs.pv_scenarios
        + discharge_values[None, :]
        - inputs.load_scenarios
        - charge_values[None, :]
        - spill_values
    )
    maximum_scenario_balance = float(np.max(np.abs(scenario_balance)))
    if maximum_scenario_balance > 2e-3:
        raise AssertionError(
            f"Q3 scenario power balance residual is {maximum_scenario_balance:.6g} kW"
        )
    frame = pd.DataFrame(
        {
            "date": str(inputs.date),
            "slot": np.arange(1, inputs.horizon + 1),
            "timestamp": inputs.timestamps,
            "price_yuan_per_kwh": inputs.price,
            "load_forecast_kw": inputs.load_forecast,
            "pv_forecast_kw": inputs.pv_forecast,
            "planned_grid_kw": grid_values,
            "charge_kw": charge_values,
            "discharge_kw": discharge_values,
            "planned_charge_limit_kw": charge_values,
            "planned_discharge_limit_kw": discharge_values,
            "storage_start_kwh": [pulp.value(energy[t]) for t in periods],
            "storage_end_kwh": [pulp.value(energy[t + 1]) for t in periods],
            "expected_emergency_kw": emergency_values.mean(axis=0),
            "expected_spill_kw": spill_values.mean(axis=0),
            "expected_planned_grid_used_kw": used_values.mean(axis=0),
            "expected_unused_planned_grid_kw": unused_values.mean(axis=0),
        }
    )
    for column in (
        "planned_grid", "charge", "discharge", "expected_emergency", "expected_spill",
        "expected_planned_grid_used", "expected_unused_planned_grid",
    ):
        frame[f"{column}_kwh"] = frame[f"{column}_kw"] * inputs.dt_hours
    planned_value = float(pulp.value(planned_cost))
    adjustment_value = 0.0 if prior_commitment_kw is None else float(pulp.value(decision_energy_cost))
    expected_value = float(pulp.value(expected_emergency))
    terminal_value = float(pulp.value(terminal_credit))
    return Q3RollingPlanResult(
        date=inputs.date,
        dispatch=frame,
        emergency_kw=emergency_values,
        spill_kw=spill_values,
        planned_grid_used_kw=used_values,
        unused_planned_grid_kw=unused_values,
        scenario_source_dates=np.array(inputs.scenario_source_dates, copy=True),
        status=status,
        runtime_seconds=runtime,
        solver_name="PuLP HiGHS",
        planned_purchase_cost_yuan=planned_value,
        adjustment_cost_yuan=adjustment_value,
        expected_emergency_cost_yuan=expected_value,
        operating_cost_yuan=planned_value + expected_value,
        terminal_value_credit_yuan=terminal_value,
        optimization_objective_yuan=float(pulp.value(model.objective)),
        terminal_value_breakpoints_kwh=breakpoints,
        terminal_value_rates_yuan_per_kwh=rates,
        maximum_scenario_power_balance_residual_kw=maximum_scenario_balance,
    )


__all__ = ["Q3RollingPlanResult", "solve_remaining_dispatch"]
