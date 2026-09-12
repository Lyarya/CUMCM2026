"""Focused tests for the analysis-only information-bound layer."""

from __future__ import annotations

import json

import numpy as np

from src.analysis.information_bound import (
    DT_HOURS,
    ETA_C,
    ETA_D,
    INITIAL_ENERGY_KWH,
    LOWER_BOUND_PATH,
    SUMMARY_PATH,
    audit_lp_solution,
    build_perfect_information_lp,
    load_full_horizon_inputs,
    q3_daily_value_analysis,
    protected_hashes,
)


def test_formal_perfect_information_grid_is_continuous() -> None:
    frame = load_full_horizon_inputs()
    assert len(frame) == 334 * 144
    assert frame["operating_date"].nunique() == 334
    assert frame.groupby("operating_date").size().eq(144).all()


def test_lp_builder_retains_q1_q2_physical_equations() -> None:
    load = np.array([1000.0, 1200.0])
    pv = np.array([200.0, 100.0])
    price = np.array([0.5, 1.0])
    objective, matrix, rhs, bounds, slices = build_perfect_information_lp(load, pv, price)
    assert matrix.shape == (5, 11)
    assert rhs.tolist() == [800.0, 1100.0, 0.0, 0.0, INITIAL_ENERGY_KWH]
    assert np.allclose(objective[slices["grid"]], price * DT_HOURS)
    assert bounds[slices["charge"].start] == (0.0, 5000.0)
    assert bounds[slices["discharge"].start] == (0.0, 5000.0)
    dense = matrix.toarray()
    assert np.isclose(dense[2, slices["charge"].start], -ETA_C * DT_HOURS)
    assert np.isclose(dense[2, slices["discharge"].start], DT_HOURS / ETA_D)


def test_q3_daily_value_uses_operating_days_as_paired_units() -> None:
    result = q3_daily_value_analysis()
    assert result["paired_unit"] == "operating day"
    assert result["n_days"] == 334
    assert np.isclose(result["annual_saving_yuan"], 222469.864818, atol=1e-5)
    assert result["positive_saving_day_proportion"] > 0.8


def test_protected_formal_artifacts_are_hashable_and_complete() -> None:
    hashes = protected_hashes()
    assert len(hashes) == 8
    assert all(len(value) == 64 for value in hashes.values())
    assert "results/problem1/result1.xlsx" in hashes
    assert "results/problem2/result2.xlsx" in hashes
    assert "results/problem3/result3.xlsx" in hashes


def test_saved_information_bound_is_physical_and_ordered() -> None:
    assert LOWER_BOUND_PATH.exists()
    assert SUMMARY_PATH.exists()
    summary = json.loads(SUMMARY_PATH.read_text())
    lower = summary["perfect_information"]
    costs = summary["cost_comparison"]
    assert lower["physical_audit_pass"] is True
    assert lower["exact_physical_perfect_information_optimum"] is True
    assert lower["simultaneous_charge_discharge_count"] == 0
    assert lower["maximum_simultaneous_charge_discharge_kw"] <= 1e-6
    assert (
        costs["perfect_information_cost_yuan"]
        <= costs["q3_s2_realized_total_cost_yuan"]
        <= costs["q2_realized_total_cost_yuan"]
    )
