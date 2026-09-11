"""Tests for Arya's Q2 economic/reliability evaluation checkpoint."""

from __future__ import annotations

from dataclasses import fields

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.problem2.evaluation_analysis import (
    DAILY_EVALUATION_COLUMNS,
    DT_HOURS,
    FORMAL_DATES,
    METHOD_COMPARISON_COLUMNS,
    OUTPUT_TABLE_SCHEMAS,
    PERIOD_SUMMARY_COLUMNS,
    RISK_SWEEP_COLUMNS,
    DailyOptimizerResult,
    calculate_cost_from_power,
    compare_methods,
    daily_metrics,
    minimum_emergency_energy_setting,
    minimum_total_cost_setting,
    pareto_cost_risk_points,
    load_final_q2_evaluation,
    summarize_chronological_results,
    summarize_q2_backtest,
    summarize_risk_sweep,
    trajectory_energy_kwh,
    write_evaluation_tables,
)
from src.problem2.forecast_interface import audit_q2_handoff_integrity
from src.problem2.visualize import (
    plot_daily_cost_decomposition,
    plot_daily_economic_performance,
    plot_risk_cost_tradeoff,
    plot_soc_emergency_diagnostic,
)


def _make_result(
    day: object,
    *,
    planned_kw: float = 60.0,
    emergency_kw: float = 0.0,
    spill_kw: float = 0.0,
    initial_energy: float = 6_000.0,
    final_energy: float = 6_000.0,
    runtime: float = 0.2,
    scenario_expected: bool = False,
) -> DailyOptimizerResult:
    price = np.full(144, 0.5)
    planned = np.full(144, planned_kw)
    charge = np.full(144, 2.0)
    discharge = np.full(144, 1.0)
    if scenario_expected:
        emergency = np.stack(
            [np.full(144, emergency_kw), np.full(144, 2 * emergency_kw)]
        )
        spill = np.full(144, spill_kw)
        weights = np.array([0.25, 0.75])
        basis = "scenario_expected"
    else:
        emergency = np.full(144, emergency_kw)
        spill = np.full(144, spill_kw)
        weights = None
        basis = "realized"
    planned_cost = calculate_cost_from_power(planned, price)
    emergency_cost = calculate_cost_from_power(
        emergency, price, price_multiplier=5.0, scenario_weights=weights
    )
    return DailyOptimizerResult(
        date=day,
        planned_grid_purchase_kw=planned,
        charge_kw=charge,
        discharge_kw=discharge,
        soc_kwh=np.linspace(initial_energy, final_energy, 145),
        emergency_purchase_kw=emergency,
        spill_kw=spill,
        planned_cost_yuan=planned_cost,
        emergency_cost_yuan=emergency_cost,
        total_cost_yuan=planned_cost + emergency_cost,
        initial_energy_kwh=initial_energy,
        final_energy_kwh=final_energy,
        solver_status="Optimal",
        solver_runtime_seconds=runtime,
        price_yuan_per_kwh=price,
        scenario_weights=weights,
        evaluation_basis=basis,
    )


def _continuous_results(
    dates: pd.DatetimeIndex,
    *,
    planned_kw: float = 60.0,
    emergency_kw: float = 0.0,
) -> list[DailyOptimizerResult]:
    results = []
    energy = 6_000.0
    for index, day in enumerate(dates):
        final = energy + 0.1
        results.append(
            _make_result(
                day,
                planned_kw=planned_kw,
                emergency_kw=emergency_kw if index % 10 == 0 else 0.0,
                initial_energy=energy,
                final_energy=final,
                runtime=0.1 + index / 10_000,
            )
        )
        energy = final
    return results


def test_optimizer_output_contract_contains_required_fields() -> None:
    names = {field.name for field in fields(DailyOptimizerResult)}
    assert {
        "date", "planned_grid_purchase_kw", "charge_kw", "discharge_kw",
        "soc_kwh", "emergency_purchase_kw", "spill_kw", "planned_cost_yuan",
        "emergency_cost_yuan", "total_cost_yuan", "initial_energy_kwh",
        "final_energy_kwh", "solver_status", "solver_runtime_seconds",
    }.issubset(names)


def test_power_to_energy_and_cost_use_dt_once() -> None:
    power = np.zeros(144)
    power[0] = 6.0
    price = np.full(144, 2.0)
    assert DT_HOURS == pytest.approx(1 / 6)
    assert float(trajectory_energy_kwh(power)) == pytest.approx(1.0)
    assert calculate_cost_from_power(power, price) == pytest.approx(2.0)
    assert calculate_cost_from_power(
        power, price, price_multiplier=5.0
    ) == pytest.approx(10.0)


def test_daily_aggregation_supports_scenario_and_realized_dimensions() -> None:
    result = _make_result(
        "2025-02-01", emergency_kw=6.0, spill_kw=3.0,
        scenario_expected=True,
    )
    metrics = daily_metrics(result)
    assert metrics["planned_grid_energy_kwh"] == pytest.approx(60.0 * 24)
    assert metrics["emergency_purchase_energy_kwh"] == pytest.approx(10.5 * 24)
    assert metrics["emergency_grid_energy_share"] == pytest.approx(10.5 / 70.5)
    assert metrics["pv_curtailment_energy_kwh"] == pytest.approx(3.0 * 24)
    assert metrics["battery_charge_energy_kwh"] == pytest.approx(2.0 * 24)
    assert metrics["battery_discharge_energy_kwh"] == pytest.approx(1.0 * 24)
    assert metrics["battery_throughput_kwh"] == pytest.approx(3.0 * 24)
    assert metrics["total_cost_yuan"] == pytest.approx(
        metrics["planned_purchase_cost_yuan"]
        + metrics["emergency_purchase_cost_yuan"]
    )
    assert metrics["emergency_purchase_day"] == 1


def test_full_334_day_backtest_is_chronological_finite_and_continuous() -> None:
    evaluation = summarize_q2_backtest(
        list(reversed(_continuous_results(FORMAL_DATES, emergency_kw=1.0)))
    )
    daily = evaluation.daily
    period = evaluation.period.iloc[0]
    assert len(daily) == 334
    assert pd.DatetimeIndex(daily["date"]).equals(FORMAL_DATES)
    assert (daily["soc_continuity_error_kwh"].abs() <= 1e-12).all()
    assert np.isfinite(daily.select_dtypes(include=[np.number])).all().all()
    assert period["day_count"] == 334
    assert period["cumulative_total_cost_yuan"] == pytest.approx(
        daily["total_cost_yuan"].sum()
    )
    assert period["cumulative_emergency_energy_kwh"] == pytest.approx(
        daily["emergency_purchase_energy_kwh"].sum()
    )
    assert period["emergency_grid_energy_share"] == pytest.approx(
        period["cumulative_emergency_energy_kwh"]
        / period["cumulative_total_grid_energy_kwh"]
    )
    assert period["emergency_purchase_days"] == 34
    assert period["solver_success_rate"] == pytest.approx(1.0)
    assert period["final_battery_energy_kwh"] == pytest.approx(6_033.4)


def test_cross_day_soc_discontinuity_is_rejected() -> None:
    results = _continuous_results(pd.date_range("2025-02-01", periods=2))
    broken = _make_result(
        "2025-02-02", initial_energy=7_000.0, final_energy=7_000.0
    )
    with pytest.raises(AssertionError, match="cross-day SOC continuity"):
        summarize_chronological_results([results[0], broken])


def test_baseline_comparison_uses_positive_saving_sign() -> None:
    dates = pd.date_range("2025-02-01", periods=3)
    comparison = compare_methods(
        {
            "no_storage": _continuous_results(dates, planned_kw=100.0),
            "stochastic": _continuous_results(dates, planned_kw=80.0),
        },
        baseline_method="no_storage",
    )
    baseline = comparison.loc[comparison["method"] == "no_storage"].iloc[0]
    strategy = comparison.loc[comparison["method"] == "stochastic"].iloc[0]
    assert baseline["cost_saving"] == pytest.approx(0.0)
    assert strategy["cost_saving"] > 0
    assert strategy["cost_saving"] == pytest.approx(
        baseline["total_cost"] - strategy["total_cost"]
    )
    assert strategy["cost_saving_percent"] > 0
    assert list(comparison.columns) == list(METHOD_COMPARISON_COLUMNS)


def test_risk_sweep_sorts_and_identifies_tradeoff_settings() -> None:
    dates = pd.date_range("2025-02-01", periods=3)
    sweep = summarize_risk_sweep(
        {
            1.0: _continuous_results(dates, planned_kw=90.0, emergency_kw=0.0),
            0.0: _continuous_results(dates, planned_kw=60.0, emergency_kw=20.0),
            0.5: _continuous_results(dates, planned_kw=70.0, emergency_kw=1.0),
        }
    )
    assert sweep["risk_parameter"].tolist() == [0.0, 0.5, 1.0]
    assert list(sweep.columns) == list(RISK_SWEEP_COLUMNS)
    assert minimum_total_cost_setting(sweep)["risk_parameter"] == pytest.approx(0.5)
    assert minimum_emergency_energy_setting(sweep)["risk_parameter"] == pytest.approx(1.0)
    pareto = pareto_cost_risk_points(sweep)
    assert set(pareto["risk_parameter"]) == {0.5, 1.0}


def test_table_schemas_write_only_supplied_nonempty_results(tmp_path) -> None:
    evaluation = summarize_chronological_results(
        _continuous_results(pd.date_range("2025-02-01", periods=2))
    )
    written = write_evaluation_tables(
        daily=evaluation.daily, period=evaluation.period, output_dir=tmp_path
    )
    assert set(written) == {"q2_daily_evaluation.csv", "q2_period_summary.csv"}
    assert all(path.exists() for path in written.values())
    assert list(evaluation.daily.columns) == list(DAILY_EVALUATION_COLUMNS)
    assert list(evaluation.period.columns) == list(PERIOD_SUMMARY_COLUMNS)
    assert set(OUTPUT_TABLE_SCHEMAS) == {
        "q2_daily_evaluation.csv", "q2_period_summary.csv",
        "q2_method_comparison.csv", "q2_risk_sweep.csv",
    }
    with pytest.raises(ValueError, match="at least one"):
        write_evaluation_tables(output_dir=tmp_path)


def test_plotting_functions_accept_realistic_shapes_without_repo_outputs(tmp_path) -> None:
    dates = pd.date_range("2025-02-01", periods=5)
    strategy_eval = summarize_chronological_results(
        _continuous_results(dates, planned_kw=70.0, emergency_kw=1.0)
    )
    baseline_eval = summarize_chronological_results(
        _continuous_results(dates, planned_kw=90.0)
    )
    sweep = summarize_risk_sweep(
        {
            0.0: _continuous_results(dates, planned_kw=60.0, emergency_kw=4.0),
            0.5: _continuous_results(dates, planned_kw=70.0, emergency_kw=1.0),
            1.0: _continuous_results(dates, planned_kw=90.0),
        }
    )
    outputs = [
        plot_daily_economic_performance(
            strategy_eval.daily, baseline_eval.daily,
            output_dir=tmp_path / "economic",
        ),
        plot_daily_cost_decomposition(
            strategy_eval.daily, output_dir=tmp_path / "decomposition"
        ),
        plot_risk_cost_tradeoff(sweep, output_dir=tmp_path / "risk"),
        plot_soc_emergency_diagnostic(
            _make_result("2025-02-01", emergency_kw=1.0),
            output_dir=tmp_path / "diagnostic",
        ),
    ]
    for exported in outputs:
        assert set(exported) == {"svg", "pdf", "jpg", "png"}
        assert all(path.exists() and path.stat().st_size > 0 for path in exported.values())
    assert not plt.get_fignums()


def test_stage2a_artifacts_and_raw_data_remain_unchanged() -> None:
    audit = audit_q2_handoff_integrity()
    assert audit["formal_pv_forecaster"] == "7-day same-slot mean"
    assert audit["q2_scenario_hash_unchanged"] is True
    assert audit["protected_artifact_hashes_unchanged"] is True
    assert audit["raw_data_hashes_unchanged"] is True


def test_final_chongwen_outputs_use_realized_physical_settlement() -> None:
    evaluation = load_final_q2_evaluation()
    period = evaluation.period.iloc[0]
    assert len(evaluation.daily) == 334
    assert period["solver_success_rate"] == pytest.approx(1.0)
    assert period["cumulative_planned_purchase_cost_yuan"] == pytest.approx(
        13_323_201.223427, abs=1e-3
    )
    assert period["cumulative_emergency_purchase_cost_yuan"] == pytest.approx(
        1_516_260.778930, abs=1e-3
    )
    assert period["cumulative_total_cost_yuan"] == pytest.approx(
        14_839_462.002357, abs=1e-3
    )
    assert period["cumulative_total_cost_yuan"] != pytest.approx(
        14_907_862.919985, abs=1e-3
    )
    assert period["cumulative_emergency_energy_kwh"] == pytest.approx(
        398_161.900954, abs=1e-3
    )
    assert period["maximum_cross_day_soc_error_kwh"] == pytest.approx(0.0, abs=2e-3)
