"""CVaR-only regression tests layered on the locked Q2 implementation."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from src.problem2.forecast_interface import (
    FORMAL_FORECAST_PATH,
    SCENARIO_PATH,
    get_q2_day_inputs,
)
from src.problem2.model import (
    Q2DispatchParameters,
    calculate_empirical_cvar,
    solve_expected_cost_dispatch,
)
from src.problem2.run import RESULT2_PATH, run_chronological
from src.problem2.run_cvar import (
    AUDIT_PATH,
    CVAR_DIR,
    METADATA_PATH,
    SWEEP_PATH,
    audit_physical_run,
)
from src.problem2.risk_analysis import (
    ALPHA as REPORT_ALPHA,
    EXPECTED_LAMBDAS,
    load_final_cvar_sweep,
    mild_risk_changes,
)


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def risk_solution():
    inputs = get_q2_day_inputs("2025-07-15", initial_energy=6_000.0)
    parameters = Q2DispatchParameters(cvar_alpha=0.90, risk_weight=0.10)
    return inputs, parameters, solve_expected_cost_dispatch(inputs, parameters)


@pytest.fixture(scope="module")
def risk_two_day():
    return run_chronological(
        pd.date_range("2025-02-01", "2025-02-02", freq="D"),
        parameters=Q2DispatchParameters(cvar_alpha=0.90, risk_weight=0.10),
    )


def test_manual_discrete_cvar_value_and_denominator() -> None:
    costs = np.arange(10.0)
    var_cost, cvar_cost = calculate_empirical_cvar(costs, 0.80)
    assert var_cost == pytest.approx(7.0)
    assert cvar_cost == pytest.approx((8.0 + 9.0) / 2.0)
    assert cvar_cost == pytest.approx(
        var_cost + np.maximum(costs - var_cost, 0.0).sum() / ((1.0 - 0.80) * 10)
    )


def test_alpha_090_uses_exact_tail_denominator() -> None:
    costs = np.arange(50.0)
    var_cost, cvar_cost = calculate_empirical_cvar(costs, 0.90)
    expected = var_cost + np.maximum(costs - var_cost, 0.0).sum() / 5.0
    assert cvar_cost == pytest.approx(expected)
    assert cvar_cost == pytest.approx(np.mean(costs[-5:]))


def test_cvar_excess_is_nonnegative(risk_solution) -> None:
    _, _, result = risk_solution
    assert np.all(result.cvar_excess_cost_yuan >= -1e-7)


def test_cvar_excess_dominates_loss_minus_var(risk_solution) -> None:
    _, _, result = risk_solution
    assert np.all(
        result.cvar_excess_cost_yuan
        >= result.scenario_emergency_cost_yuan - result.var_cost_yuan - 2e-3
    )


def test_cvar_solver_value_matches_linearized_formula(risk_solution) -> None:
    _, _, result = risk_solution
    reconstructed = result.var_cost_yuan + result.cvar_excess_cost_yuan.sum() / (
        (1.0 - result.cvar_alpha) * len(result.cvar_excess_cost_yuan)
    )
    assert result.cvar_cost_yuan == pytest.approx(reconstructed, abs=2e-3)


def test_scenario_loss_is_emergency_cost_only(risk_solution) -> None:
    inputs, parameters, result = risk_solution
    recomputed = parameters.emergency_price_multiplier * np.sum(
        inputs.price[None, :] * result.emergency_kw * inputs.dt_hours, axis=1
    )
    assert result.scenario_emergency_cost_yuan == pytest.approx(recomputed, abs=2e-3)
    assert result.planned_purchase_cost_yuan > 0.0
    assert not np.allclose(
        result.scenario_emergency_cost_yuan,
        recomputed + result.planned_purchase_cost_yuan,
    )


def test_plan_expected_and_cvar_terms_are_counted_once(risk_solution) -> None:
    _, parameters, result = risk_solution
    accounted = (
        result.planned_purchase_cost_yuan
        + result.expected_emergency_cost_yuan
        + parameters.risk_weight * result.cvar_cost_yuan
        - result.terminal_value_credit_yuan
    )
    assert result.optimization_objective_yuan == pytest.approx(accounted, abs=2e-3)


def test_continuation_value_is_outside_operating_cost(risk_solution) -> None:
    _, _, result = risk_solution
    assert result.operating_cost_yuan == pytest.approx(
        result.planned_purchase_cost_yuan + result.expected_emergency_cost_yuan,
        abs=2e-3,
    )


def test_lambda_zero_preserves_expected_cost_model() -> None:
    inputs = get_q2_day_inputs("2025-04-15", initial_energy=6_000.0)
    base = solve_expected_cost_dispatch(inputs)
    zero = solve_expected_cost_dispatch(
        inputs, Q2DispatchParameters(cvar_alpha=0.90, risk_weight=0.0)
    )
    assert zero.planned_purchase_cost_yuan == pytest.approx(
        base.planned_purchase_cost_yuan, abs=1e-7
    )
    assert zero.expected_emergency_cost_yuan == pytest.approx(
        base.expected_emergency_cost_yuan, abs=1e-7
    )
    assert zero.optimization_objective_yuan == pytest.approx(
        base.optimization_objective_yuan, abs=1e-7
    )
    assert zero.dispatch["planned_grid_kw"].to_numpy() == pytest.approx(
        base.dispatch["planned_grid_kw"].to_numpy(), abs=1e-7
    )


def test_actual_actions_remain_bounded_by_planned_actions(risk_two_day) -> None:
    _, intervals, *_ = risk_two_day
    assert (
        intervals["actual_charge_kw"]
        <= intervals["planned_charge_limit_kw"] + 2e-3
    ).all()
    assert (
        intervals["actual_discharge_kw"]
        <= intervals["planned_discharge_limit_kw"] + 2e-3
    ).all()


def test_actual_soc_uses_actual_action_recurrence(risk_two_day) -> None:
    daily, intervals, *_ = risk_two_day
    report = audit_physical_run(daily, intervals)
    assert report["actual_soc_recurrence"] is True


def test_next_day_soc_uses_previous_actual_end(risk_two_day) -> None:
    daily, _, *_ = risk_two_day
    assert daily.loc[1, "actual_initial_energy_kwh"] == pytest.approx(
        daily.loc[0, "actual_final_energy_kwh"], abs=2e-3
    )


def test_unabsorbed_discharge_remains_zero(risk_two_day) -> None:
    daily, intervals, *_ = risk_two_day
    report = audit_physical_run(daily, intervals)
    assert report["unabsorbed_discharge_energy_kwh"] == 0.0


def test_actual_power_balance_remains_valid(risk_two_day) -> None:
    daily, intervals, *_ = risk_two_day
    assert audit_physical_run(daily, intervals)["actual_power_balance"] is True


def test_actual_soc_bounds_remain_valid(risk_two_day) -> None:
    daily, intervals, *_ = risk_two_day
    assert audit_physical_run(daily, intervals)["soc_bounds"] is True


@pytest.fixture(scope="module")
def frozen_hash_check(risk_solution):
    before = {
        "forecast": _sha256(FORMAL_FORECAST_PATH),
        "scenario": _sha256(SCENARIO_PATH),
        "result2": _sha256(RESULT2_PATH),
    }
    inputs = get_q2_day_inputs("2025-10-15", initial_energy=7_000.0)
    solve_expected_cost_dispatch(
        inputs, Q2DispatchParameters(cvar_alpha=0.90, risk_weight=0.05)
    )
    after = {
        "forecast": _sha256(FORMAL_FORECAST_PATH),
        "scenario": _sha256(SCENARIO_PATH),
        "result2": _sha256(RESULT2_PATH),
    }
    return before, after


def test_locked_result2_is_not_modified(frozen_hash_check) -> None:
    before, after = frozen_hash_check
    assert after["result2"] == before["result2"]


def test_scenario_generation_artifact_is_not_modified(frozen_hash_check) -> None:
    before, after = frozen_hash_check
    assert after["scenario"] == before["scenario"]


def test_forecast_artifact_is_not_modified(frozen_hash_check) -> None:
    before, after = frozen_hash_check
    assert after["forecast"] == before["forecast"]


def test_cvar_outputs_are_separate_from_locked_result2() -> None:
    assert RESULT2_PATH not in {SWEEP_PATH, METADATA_PATH, AUDIT_PATH}
    assert CVAR_DIR != RESULT2_PATH.parent
    assert all(path.name != "result2.xlsx" for path in (SWEEP_PATH, METADATA_PATH, AUDIT_PATH))


def test_committed_risk_sweep_matches_final_reporting_contract() -> None:
    sweep = load_final_cvar_sweep()
    assert sweep["lambda"].tolist() == list(EXPECTED_LAMBDAS)
    assert (sweep["alpha"] == REPORT_ALPHA).all()
    assert ((sweep["solver_success"] == 334) & (sweep["formal_days"] == 334)).all()
    base = sweep.loc[sweep["lambda"] == 0.0].iloc[0]
    assert base["expected_operating_cost_yuan"] == pytest.approx(14_374_345.2462, abs=1e-3)
    assert base["cvar_cost_yuan"] == pytest.approx(5_394_358.7627, abs=1e-3)
    assert base["realized_total_cost_yuan"] == pytest.approx(14_839_462.0024, abs=1e-3)


def test_mild_risk_reporting_effects_are_derived_from_saved_sweep() -> None:
    changes = mild_risk_changes(load_final_cvar_sweep())
    assert changes["expected_operating_cost_change_pct"] == pytest.approx(0.0934637)
    assert changes["emergency_energy_reduction_pct"] == pytest.approx(13.7480572)
    assert changes["emergency_cvar_reduction_pct"] == pytest.approx(13.3931537)
    assert changes["realized_total_cost_change_pct"] == pytest.approx(-0.1186415)
