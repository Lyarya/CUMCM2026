"""Physical invariants and one-factor isolation for Q1 sensitivity."""

from dataclasses import asdict, replace
import json

import numpy as np
import pandas as pd
import pytest

from src.problem1.model import DispatchParameters, solve_deterministic_dispatch
from src.problem1.sensitivity import (
    AUDIT_PATH, INPUT_PATH, TABLE_DIR, audit_case,
    sensitivity_cases, summarize_sensitivity,
)


def test_parameter_sweeps_change_only_the_requested_factor():
    base = asdict(DispatchParameters())
    cases = sensitivity_cases()
    assert len(cases) == 12
    assert len(set(parameters for _, _, parameters in cases)) == 10
    for group, value, parameters in cases:
        current = asdict(parameters)
        if group == "efficiency":
            allowed = {"charge_efficiency", "discharge_efficiency"}
            assert parameters.charge_efficiency == parameters.discharge_efficiency == value
        elif group == "capacity":
            allowed = {"capacity_kwh", "minimum_energy_kwh", "maximum_energy_kwh", "initial_energy_kwh", "terminal_energy_kwh"}
            assert parameters.capacity_kwh == 12000 * value
            assert parameters.minimum_energy_kwh == 0.1 * parameters.capacity_kwh
            assert parameters.maximum_energy_kwh == 0.9 * parameters.capacity_kwh
            assert parameters.initial_energy_kwh == parameters.terminal_energy_kwh == 0.5 * parameters.capacity_kwh
        else:
            allowed = {"maximum_charge_kw", "maximum_discharge_kw"}
            assert parameters.maximum_charge_kw == parameters.maximum_discharge_kw == 5000 * value
        assert {key: val for key, val in current.items() if key not in allowed} == {
            key: val for key, val in base.items() if key not in allowed}


@pytest.mark.parametrize("group,value,parameters", sensitivity_cases())
def test_every_case_is_optimal_and_physically_feasible(group, value, parameters):
    result = solve_deterministic_dispatch(pd.read_csv(INPUT_PATH), parameters)
    report = audit_case(result, parameters)
    assert report["physical_invariants_pass"]
    table = pd.read_csv(TABLE_DIR / f"q1_{group}_sensitivity.csv")
    row = table.loc[np.isclose(table["parameter_value"], value)].iloc[0]
    assert result.objective_yuan == pytest.approx(row.optimal_cost_yuan, abs=1e-3)
    assert row.total_discharge_energy_kwh == pytest.approx(
        row.total_charge_energy_kwh * parameters.charge_efficiency * parameters.discharge_efficiency,
        abs=0.01,
    )


@pytest.fixture(scope="module")
def baseline_result():
    return solve_deterministic_dispatch(pd.read_csv(INPUT_PATH))


@pytest.mark.parametrize("corruption", ["nonoptimal", "nan", "negative_spill", "soc_chain", "unit", "simultaneous", "spill_excess", "soc_bounds"])
def test_invalid_runs_are_rejected(baseline_result, corruption):
    frame = baseline_result.dispatch.copy(deep=True)
    result = replace(baseline_result, dispatch=frame)
    if corruption == "nonoptimal":
        result = replace(result, status="Infeasible")
    elif corruption == "nan":
        frame.loc[0, "grid_purchase_kw"] = np.nan
    elif corruption == "negative_spill":
        frame.loc[0, "spill_kw"] = -1
    elif corruption == "soc_chain":
        frame.loc[0, ["storage_end_kwh", "storage_start_kwh"]] += 100
    elif corruption == "unit":
        frame.loc[0, "grid_purchase_kwh"] += 10
    elif corruption == "simultaneous":
        frame.loc[0, ["charge_kw", "discharge_kw"]] = 100
    elif corruption == "spill_excess":
        frame.loc[0, "spill_kw"] = frame.loc[0, "pv_forecast_kw"] + 100
    elif corruption == "soc_bounds":
        frame.loc[5, "storage_end_kwh"] = 12000
    with pytest.raises(AssertionError):
        audit_case(result, DispatchParameters())


def test_saved_baseline_and_other_problem_artifacts_are_unchanged():
    audit = json.loads(AUDIT_PATH.read_text())
    # This is a historical run audit: later validated Q2/Q3/Q4 milestones may
    # legitimately change protected files after the Q1 sweep has completed.
    assert audit["protected_hashes_before"] == audit["protected_hashes_after"]
    assert audit["baseline_q1_unchanged"]
    assert audit["baseline_cost_yuan"] == pytest.approx(35126.948591235, abs=1e-3)


def test_summary_uses_normalized_parameter_changes_and_actual_marginal_gains():
    table = pd.concat([pd.read_csv(TABLE_DIR / f"q1_{group}_sensitivity.csv")
                       for group in ("efficiency", "capacity", "power")])
    summary = summarize_sensitivity(table).set_index("sweep")
    for group in ("capacity", "power"):
        assert summary.loc[group, "gain_step1_yuan"] > summary.loc[group, "gain_step2_yuan"] > summary.loc[group, "gain_step3_yuan"]
        assert summary.loc[group, "diminishing_return_observed"]
    expected = ((35126.948591235 - 33767.032629) / 35126.948591235) / ((0.95 - 0.90) / 0.90)
    assert summary.loc["efficiency", "baseline_to_upper_normalized_cost_sensitivity"] == pytest.approx(expected, abs=1e-7)
