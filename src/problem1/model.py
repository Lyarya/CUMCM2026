"""Deterministic microgrid dispatch model for Problem 1."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import pandas as pd
import pulp


DT_HOURS = 1.0 / 6.0


@dataclass(frozen=True)
class DispatchParameters:
    """Physical parameters from Appendix 1 of the problem statement."""

    capacity_kwh: float = 12_000.0
    initial_energy_kwh: float = 6_000.0
    minimum_energy_kwh: float = 1_200.0
    maximum_energy_kwh: float = 10_800.0
    maximum_charge_kw: float = 5_000.0
    maximum_discharge_kw: float = 5_000.0
    charge_efficiency: float = 0.9
    discharge_efficiency: float = 0.9
    terminal_energy_kwh: float = 6_000.0


@dataclass(frozen=True)
class DispatchResult:
    """Optimal dispatch and solver metadata."""

    dispatch: pd.DataFrame
    status: str
    objective_yuan: float
    runtime_seconds: float
    solver_name: str


def solve_deterministic_dispatch(
    data: pd.DataFrame,
    parameters: DispatchParameters | None = None,
) -> DispatchResult:
    """Solve the 144-period deterministic Q1 MILP with PuLP and CBC."""
    parameters = parameters or DispatchParameters()
    required = {
        "slot",
        "interval_start",
        "interval_end",
        "price_yuan_per_kwh",
        "load_kw",
        "pv_forecast_kw",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if len(data) != 144:
        raise ValueError(f"Q1 requires 144 ten-minute periods, got {len(data)}")

    frame = data.sort_values("slot").reset_index(drop=True).copy()
    if frame["slot"].tolist() != list(range(1, 145)):
        raise ValueError("slot must be the consecutive integers 1 through 144")
    if frame[list(required - {"interval_start", "interval_end"})].isna().any().any():
        raise ValueError("Q1 input contains missing numeric values")

    periods = range(len(frame))
    model = pulp.LpProblem("CUMCM2026_Q1_Deterministic_Dispatch", pulp.LpMinimize)
    grid = model.add_variable_dicts("grid_kw", periods, lowBound=0)
    charge = model.add_variable_dicts("charge_kw", periods, lowBound=0)
    discharge = model.add_variable_dicts("discharge_kw", periods, lowBound=0)
    spill = model.add_variable_dicts("spill_kw", periods, lowBound=0)
    mode = model.add_variable_dicts("charge_mode", periods, cat=pulp.LpBinary)
    energy = model.add_variable_dicts(
        "energy_kwh",
        range(len(frame) + 1),
        lowBound=parameters.minimum_energy_kwh,
        upBound=parameters.maximum_energy_kwh,
    )

    model += energy[0] == parameters.initial_energy_kwh, "initial_energy"
    model += energy[len(frame)] == parameters.terminal_energy_kwh, "terminal_energy"

    for t in periods:
        load_kw = float(frame.at[t, "load_kw"])
        pv_kw = float(frame.at[t, "pv_forecast_kw"])
        model += (
            grid[t] + pv_kw + discharge[t] == load_kw + charge[t] + spill[t]
        ), f"power_balance_{t + 1:03d}"
        model += (
            energy[t + 1]
            == energy[t]
            + parameters.charge_efficiency * charge[t] * DT_HOURS
            - discharge[t] * DT_HOURS / parameters.discharge_efficiency
        ), f"energy_transition_{t + 1:03d}"
        model += (
            charge[t] <= parameters.maximum_charge_kw * mode[t]
        ), f"charge_limit_{t + 1:03d}"
        model += (
            discharge[t] <= parameters.maximum_discharge_kw * (1 - mode[t])
        ), f"discharge_limit_{t + 1:03d}"
        model += spill[t] <= pv_kw, f"spill_limit_{t + 1:03d}"

    model += pulp.lpSum(
        float(frame.at[t, "price_yuan_per_kwh"]) * grid[t] * DT_HOURS
        for t in periods
    )

    solver = pulp.COIN_CMD(
        path=pulp.apis.PULP_CBC_CMD.pulp_cbc_path,
        msg=False,
        threads=1,
    )
    started = perf_counter()
    model.solve(solver)
    runtime_seconds = perf_counter() - started
    status = pulp.LpStatus[model.status]
    if status != "Optimal":
        raise RuntimeError(f"CBC did not find an optimal Q1 solution: {status}")

    result = frame[
        [
            "slot",
            "interval_start",
            "interval_end",
            "price_yuan_per_kwh",
            "load_kw",
            "pv_forecast_kw",
        ]
    ].copy()
    result["grid_purchase_kw"] = [pulp.value(grid[t]) for t in periods]
    result["charge_kw"] = [pulp.value(charge[t]) for t in periods]
    result["discharge_kw"] = [pulp.value(discharge[t]) for t in periods]
    result["spill_kw"] = [pulp.value(spill[t]) for t in periods]
    result["storage_start_kwh"] = [pulp.value(energy[t]) for t in periods]
    result["storage_end_kwh"] = [pulp.value(energy[t + 1]) for t in periods]
    result["grid_purchase_kwh"] = result["grid_purchase_kw"] * DT_HOURS
    result["charge_kwh"] = result["charge_kw"] * DT_HOURS
    result["discharge_kwh"] = result["discharge_kw"] * DT_HOURS
    result["spill_kwh"] = result["spill_kw"] * DT_HOURS

    return DispatchResult(
        dispatch=result,
        status=status,
        objective_yuan=float(pulp.value(model.objective)),
        runtime_seconds=runtime_seconds,
        solver_name="PuLP CBC",
    )
