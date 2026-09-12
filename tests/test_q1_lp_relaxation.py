"""Focused tests for the independent Q1 LP-relaxation audit."""

from __future__ import annotations

import pandas as pd

from src.problem1.lp_relaxation import (
    INPUT_PATH,
    OBJECTIVE_GAP_TOLERANCE_YUAN,
    SIMULTANEOUS_ACTION_TOLERANCE_KW,
    solution_metrics,
    solve_lp_relaxation,
)
from src.problem1.model import DispatchParameters, solve_deterministic_dispatch


def test_lp_relaxation_preserves_q1_contract() -> None:
    data = pd.read_csv(INPUT_PATH)
    result = solve_lp_relaxation(data)
    metrics = solution_metrics(result)
    assert result.status == "Optimal"
    assert len(result.dispatch) == 144
    assert metrics["common_physical_invariants_pass"] is True
    assert metrics["minimum_soc_energy_kwh"] >= 1_200 - 1e-3
    assert metrics["maximum_soc_energy_kwh"] <= 10_800 + 1e-3
    assert abs(result.dispatch["storage_start_kwh"].iloc[0] - 6_000) <= 1e-3
    assert abs(result.dispatch["storage_end_kwh"].iloc[-1] - 6_000) <= 1e-3


def test_lp_relaxation_is_exact_and_naturally_exclusive_on_attachment1() -> None:
    data = pd.read_csv(INPUT_PATH)
    parameters = DispatchParameters()
    milp = solve_deterministic_dispatch(data, parameters)
    lp = solve_lp_relaxation(data, parameters)
    lp_metrics = solution_metrics(lp, parameters)
    assert abs(milp.objective_yuan - lp.objective_yuan) <= OBJECTIVE_GAP_TOLERANCE_YUAN
    assert (
        lp_metrics["maximum_simultaneous_charge_discharge_kw"]
        <= SIMULTANEOUS_ACTION_TOLERANCE_KW
    )
