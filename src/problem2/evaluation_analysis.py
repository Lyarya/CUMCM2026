"""Read-only economic and reliability analysis for finalized Q2 outputs.

This module deliberately does not import :mod:`src.problem2.model` or execute
the optimizer.  It consumes finalized planned/realized trajectories, converts
power to energy exactly once, and prepares comparable daily, period, method,
and risk-sweep summaries.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.common.paths import problem_results_dir


HORIZON = 144
DT_HOURS = 1.0 / 6.0
EMERGENCY_PRICE_MULTIPLIER = 5.0
FORMAL_DATES = pd.date_range("2025-02-01", "2025-12-31", freq="D")

DAILY_EVALUATION_COLUMNS = (
    "date", "evaluation_basis", "planned_purchase_cost_yuan",
    "emergency_purchase_cost_yuan", "total_cost_yuan",
    "planned_grid_energy_kwh", "emergency_purchase_energy_kwh",
    "total_grid_energy_kwh", "emergency_grid_energy_share",
    "pv_curtailment_energy_kwh", "battery_charge_energy_kwh",
    "battery_discharge_energy_kwh", "battery_throughput_kwh",
    "initial_energy_kwh", "final_energy_kwh", "minimum_soc_kwh",
    "maximum_soc_kwh", "maximum_emergency_power_kw",
    "emergency_purchase_day", "solver_optimal", "solver_status",
    "solver_runtime_seconds", "soc_continuity_error_kwh",
)
PERIOD_SUMMARY_COLUMNS = (
    "period_start", "period_end", "day_count",
    "cumulative_planned_purchase_cost_yuan",
    "cumulative_emergency_purchase_cost_yuan", "cumulative_total_cost_yuan",
    "cumulative_planned_grid_energy_kwh", "cumulative_emergency_energy_kwh",
    "cumulative_total_grid_energy_kwh", "emergency_grid_energy_share",
    "cumulative_pv_curtailment_energy_kwh",
    "cumulative_battery_throughput_kwh", "emergency_purchase_days",
    "emergency_purchase_day_proportion", "maximum_daily_emergency_energy_kwh",
    "maximum_daily_total_cost_yuan", "maximum_cross_day_soc_error_kwh",
    "solver_success_rate", "average_solver_runtime_seconds",
    "total_solver_runtime_seconds", "final_battery_energy_kwh",
)
METHOD_COMPARISON_COLUMNS = (
    "method", "planned_cost", "emergency_cost", "total_cost",
    "emergency_energy", "curtailment_energy", "battery_throughput",
    "solver_runtime", "cost_saving", "cost_saving_percent",
)
RISK_SWEEP_COLUMNS = (
    "risk_parameter", "planned_cost", "emergency_cost", "total_cost",
    "emergency_energy", "emergency_days", "pv_curtailment",
    "battery_throughput", "final_energy", "solver_runtime",
)
OUTPUT_TABLE_SCHEMAS = {
    "q2_daily_evaluation.csv": DAILY_EVALUATION_COLUMNS,
    "q2_period_summary.csv": PERIOD_SUMMARY_COLUMNS,
    "q2_method_comparison.csv": METHOD_COMPARISON_COLUMNS,
    "q2_risk_sweep.csv": RISK_SWEEP_COLUMNS,
}


@dataclass(frozen=True)
class DailyOptimizerResult:
    """Optimizer-independent daily evaluation contract.

    For ``evaluation_basis='realized'``, charge, discharge, SOC, emergency,
    and spill must all be the physically realized trajectories.
    """

    date: object
    planned_grid_purchase_kw: object
    charge_kw: object
    discharge_kw: object
    soc_kwh: object
    emergency_purchase_kw: object
    spill_kw: object
    planned_cost_yuan: float
    emergency_cost_yuan: float
    total_cost_yuan: float
    initial_energy_kwh: float
    final_energy_kwh: float
    solver_status: str
    solver_runtime_seconds: float
    price_yuan_per_kwh: object | None = None
    scenario_weights: object | None = None
    evaluation_basis: Literal["realized", "scenario_expected"] = "realized"


@dataclass(frozen=True)
class BacktestEvaluation:
    daily: pd.DataFrame
    period: pd.DataFrame


def _finite_scalar(name: str, value: object, *, nonnegative: bool = False) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or (nonnegative and scalar < 0):
        raise ValueError(f"{name} must be finite" + (" and nonnegative" if nonnegative else ""))
    return scalar


def _power_profile(name: str, values: object, *, one_dimensional: bool) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0 or array.shape[-1] != HORIZON:
        raise ValueError(f"{name} must end with a {HORIZON}-interval dimension")
    if one_dimensional and array.shape != (HORIZON,):
        raise ValueError(f"{name} must have shape ({HORIZON},)")
    if not np.isfinite(array).all() or (array < -1e-9).any():
        raise ValueError(f"{name} must contain finite nonnegative power")
    return np.maximum(array, 0.0)


def _normalized_weights(weights: object | None, count: int) -> np.ndarray:
    if weights is None:
        return np.full(count, 1.0 / count)
    array = np.asarray(weights, dtype=float).reshape(-1)
    if array.shape != (count,) or not np.isfinite(array).all() or (array < 0).any() or array.sum() <= 0:
        raise ValueError("scenario weights must be finite, nonnegative, and match trajectories")
    return array / array.sum()


def trajectory_energy_kwh(power_kw: object) -> np.ndarray:
    power = np.asarray(power_kw, dtype=float)
    if power.ndim == 0 or power.shape[-1] != HORIZON or not np.isfinite(power).all():
        raise ValueError(f"power must end with {HORIZON} finite intervals")
    return power.sum(axis=-1) * DT_HOURS


def expected_energy_kwh(power_kw: object, scenario_weights: object | None = None) -> float:
    energies = np.asarray(trajectory_energy_kwh(power_kw), dtype=float).reshape(-1)
    return float(np.dot(_normalized_weights(scenario_weights, energies.size), energies))


def _weights_for(values: np.ndarray, weights: object | None) -> object | None:
    return None if values.reshape(-1, HORIZON).shape[0] == 1 else weights


def calculate_cost_from_power(
    power_kw: object,
    price_yuan_per_kwh: object,
    *,
    price_multiplier: float = 1.0,
    scenario_weights: object | None = None,
) -> float:
    power = np.asarray(power_kw, dtype=float)
    price = np.asarray(price_yuan_per_kwh, dtype=float).reshape(-1)
    multiplier = _finite_scalar("price_multiplier", price_multiplier, nonnegative=True)
    if power.ndim == 0 or power.shape[-1] != HORIZON or price.shape != (HORIZON,):
        raise ValueError("power and price must share the 144-interval horizon")
    if not np.isfinite(power).all() or not np.isfinite(price).all() or (power < -1e-9).any() or (price < 0).any():
        raise ValueError("power and price must be finite and nonnegative")
    costs = (power * price * multiplier).sum(axis=-1) * DT_HOURS
    flattened = np.asarray(costs, dtype=float).reshape(-1)
    return float(np.dot(_normalized_weights(scenario_weights, flattened.size), flattened))


def _validated_arrays(result: DailyOptimizerResult) -> dict[str, np.ndarray]:
    arrays = {
        "planned": _power_profile("planned_grid_purchase_kw", result.planned_grid_purchase_kw, one_dimensional=True),
        "charge": _power_profile("charge_kw", result.charge_kw, one_dimensional=True),
        "discharge": _power_profile("discharge_kw", result.discharge_kw, one_dimensional=True),
        "emergency": _power_profile("emergency_purchase_kw", result.emergency_purchase_kw, one_dimensional=False),
        "spill": _power_profile("spill_kw", result.spill_kw, one_dimensional=False),
    }
    soc = np.asarray(result.soc_kwh, dtype=float).reshape(-1)
    if soc.shape != (HORIZON + 1,) or not np.isfinite(soc).all():
        raise ValueError("soc_kwh must contain 145 finite boundary states")
    arrays["soc"] = soc
    if result.evaluation_basis not in {"realized", "scenario_expected"}:
        raise ValueError("unknown evaluation basis")
    if result.evaluation_basis == "realized" and (arrays["emergency"].ndim != 1 or arrays["spill"].ndim != 1):
        raise ValueError("realized evaluation requires one emergency and spill trajectory")
    return arrays


def validate_daily_result(
    result: DailyOptimizerResult,
    *,
    tolerance: float = 2e-3,
    validate_costs_from_price: bool = True,
) -> None:
    arrays = _validated_arrays(result)
    initial = _finite_scalar("initial_energy_kwh", result.initial_energy_kwh)
    final = _finite_scalar("final_energy_kwh", result.final_energy_kwh)
    planned_cost = _finite_scalar("planned_cost_yuan", result.planned_cost_yuan, nonnegative=True)
    emergency_cost = _finite_scalar("emergency_cost_yuan", result.emergency_cost_yuan, nonnegative=True)
    total_cost = _finite_scalar("total_cost_yuan", result.total_cost_yuan, nonnegative=True)
    _finite_scalar("solver_runtime_seconds", result.solver_runtime_seconds, nonnegative=True)
    if abs(arrays["soc"][0] - initial) > tolerance or abs(arrays["soc"][-1] - final) > tolerance:
        raise AssertionError("realized SOC endpoints disagree with daily summary")
    if abs(total_cost - planned_cost - emergency_cost) > tolerance:
        raise AssertionError("total cost is not planned cost plus emergency cost")
    if result.price_yuan_per_kwh is not None and validate_costs_from_price:
        if abs(calculate_cost_from_power(arrays["planned"], result.price_yuan_per_kwh) - planned_cost) > tolerance:
            raise AssertionError("planned cost disagrees with planned power * price * dt")
        expected = calculate_cost_from_power(
            arrays["emergency"], result.price_yuan_per_kwh,
            price_multiplier=EMERGENCY_PRICE_MULTIPLIER,
            scenario_weights=_weights_for(arrays["emergency"], result.scenario_weights),
        )
        if abs(expected - emergency_cost) > tolerance:
            raise AssertionError("emergency cost disagrees with 5 * power * price * dt")


def daily_metrics(result: DailyOptimizerResult, *, tolerance: float = 2e-3, validate_costs_from_price: bool = True) -> dict[str, object]:
    validate_daily_result(result, tolerance=tolerance, validate_costs_from_price=validate_costs_from_price)
    arrays = _validated_arrays(result)
    planned_energy = expected_energy_kwh(arrays["planned"])
    emergency_energy = expected_energy_kwh(arrays["emergency"], _weights_for(arrays["emergency"], result.scenario_weights))
    curtailment = expected_energy_kwh(arrays["spill"], _weights_for(arrays["spill"], result.scenario_weights))
    charge_energy = expected_energy_kwh(arrays["charge"])
    discharge_energy = expected_energy_kwh(arrays["discharge"])
    total_grid_energy = planned_energy + emergency_energy
    status = str(result.solver_status).strip()
    return {
        "date": pd.Timestamp(result.date).normalize(), "evaluation_basis": result.evaluation_basis,
        "planned_purchase_cost_yuan": float(result.planned_cost_yuan),
        "emergency_purchase_cost_yuan": float(result.emergency_cost_yuan),
        "total_cost_yuan": float(result.total_cost_yuan),
        "planned_grid_energy_kwh": planned_energy,
        "emergency_purchase_energy_kwh": emergency_energy,
        "total_grid_energy_kwh": total_grid_energy,
        "emergency_grid_energy_share": emergency_energy / total_grid_energy if total_grid_energy > tolerance else 0.0,
        "pv_curtailment_energy_kwh": curtailment,
        "battery_charge_energy_kwh": charge_energy,
        "battery_discharge_energy_kwh": discharge_energy,
        "battery_throughput_kwh": charge_energy + discharge_energy,
        "initial_energy_kwh": float(result.initial_energy_kwh),
        "final_energy_kwh": float(result.final_energy_kwh),
        "minimum_soc_kwh": float(arrays["soc"].min()), "maximum_soc_kwh": float(arrays["soc"].max()),
        "maximum_emergency_power_kw": float(arrays["emergency"].max()),
        "emergency_purchase_day": int(emergency_energy > tolerance),
        "solver_optimal": int(status.casefold() == "optimal"), "solver_status": status,
        "solver_runtime_seconds": float(result.solver_runtime_seconds), "soc_continuity_error_kwh": 0.0,
    }


def summarize_chronological_results(
    results: Iterable[DailyOptimizerResult],
    *,
    expected_dates: Sequence[object] | pd.DatetimeIndex | None = None,
    continuity_tolerance_kwh: float = 2e-3,
    validate_costs_from_price: bool = True,
) -> BacktestEvaluation:
    rows = [daily_metrics(item, tolerance=continuity_tolerance_kwh, validate_costs_from_price=validate_costs_from_price) for item in results]
    if not rows:
        raise ValueError("at least one daily result is required")
    daily = pd.DataFrame(rows).sort_values("date", kind="stable").reset_index(drop=True)
    if daily["date"].duplicated().any():
        raise ValueError("duplicate daily results")
    if expected_dates is not None and not pd.DatetimeIndex(daily["date"]).equals(pd.DatetimeIndex(expected_dates).normalize()):
        raise AssertionError("daily result dates do not match the expected period")
    continuity = np.zeros(len(daily))
    continuity[1:] = daily["initial_energy_kwh"].to_numpy()[1:] - daily["final_energy_kwh"].to_numpy()[:-1]
    daily["soc_continuity_error_kwh"] = continuity
    maximum_gap = float(np.abs(continuity).max())
    if maximum_gap > continuity_tolerance_kwh:
        raise AssertionError(f"cross-day SOC continuity failed: {maximum_gap:.6g} kWh")
    emergency = float(daily["emergency_purchase_energy_kwh"].sum())
    total_grid = float(daily["total_grid_energy_kwh"].sum())
    period = pd.DataFrame([{
        "period_start": daily["date"].iloc[0], "period_end": daily["date"].iloc[-1], "day_count": len(daily),
        "cumulative_planned_purchase_cost_yuan": daily["planned_purchase_cost_yuan"].sum(),
        "cumulative_emergency_purchase_cost_yuan": daily["emergency_purchase_cost_yuan"].sum(),
        "cumulative_total_cost_yuan": daily["total_cost_yuan"].sum(),
        "cumulative_planned_grid_energy_kwh": daily["planned_grid_energy_kwh"].sum(),
        "cumulative_emergency_energy_kwh": emergency,
        "cumulative_total_grid_energy_kwh": total_grid,
        "emergency_grid_energy_share": emergency / total_grid if total_grid > 0 else 0.0,
        "cumulative_pv_curtailment_energy_kwh": daily["pv_curtailment_energy_kwh"].sum(),
        "cumulative_battery_throughput_kwh": daily["battery_throughput_kwh"].sum(),
        "emergency_purchase_days": int(daily["emergency_purchase_day"].sum()),
        "emergency_purchase_day_proportion": daily["emergency_purchase_day"].mean(),
        "maximum_daily_emergency_energy_kwh": daily["emergency_purchase_energy_kwh"].max(),
        "maximum_daily_total_cost_yuan": daily["total_cost_yuan"].max(),
        "maximum_cross_day_soc_error_kwh": maximum_gap,
        "solver_success_rate": daily["solver_optimal"].mean(),
        "average_solver_runtime_seconds": daily["solver_runtime_seconds"].mean(),
        "total_solver_runtime_seconds": daily["solver_runtime_seconds"].sum(),
        "final_battery_energy_kwh": daily["final_energy_kwh"].iloc[-1],
    }], columns=PERIOD_SUMMARY_COLUMNS)
    if not np.isfinite(daily.select_dtypes(include=[np.number])).all().all() or not np.isfinite(period.select_dtypes(include=[np.number])).all().all():
        raise AssertionError("evaluation contains nonfinite values")
    return BacktestEvaluation(daily=daily[list(DAILY_EVALUATION_COLUMNS)], period=period)


def summarize_q2_backtest(results: Iterable[DailyOptimizerResult], *, continuity_tolerance_kwh: float = 2e-3) -> BacktestEvaluation:
    return summarize_chronological_results(results, expected_dates=FORMAL_DATES, continuity_tolerance_kwh=continuity_tolerance_kwh)


def load_final_q2_evaluation(table_dir: str | Path | None = None) -> BacktestEvaluation:
    """Load Chongwen's finalized physical settlement without rerunning Q2."""
    source = Path(table_dir) if table_dir is not None else problem_results_dir(2) / "tables"
    daily_source = pd.read_csv(source / "table_p2_daily_summary.csv")
    dispatch = pd.read_csv(source / "table_p2_dispatch.csv")
    required = {
        "date", "slot", "price_yuan_per_kwh", "planned_grid_kw",
        "actual_charge_kw", "actual_discharge_kw", "actual_storage_start_kwh",
        "actual_storage_end_kwh", "realized_emergency_kw", "realized_pv_spill_kw",
    }
    if required.difference(dispatch.columns) or len(daily_source) != len(FORMAL_DATES):
        raise ValueError("finalized Q2 files do not satisfy the physical evaluation schema")
    results: list[DailyOptimizerResult] = []
    for summary in daily_source.itertuples(index=False):
        day = dispatch.loc[dispatch["date"].astype(str).eq(str(summary.date))].sort_values("slot")
        if day["slot"].tolist() != list(range(1, HORIZON + 1)):
            raise AssertionError(f"{summary.date} does not contain exactly slots 1..144")
        soc = np.r_[day["actual_storage_start_kwh"].to_numpy(), day["actual_storage_end_kwh"].iloc[-1]]
        results.append(DailyOptimizerResult(
            date=summary.date,
            planned_grid_purchase_kw=day["planned_grid_kw"].to_numpy(),
            charge_kw=day["actual_charge_kw"].to_numpy(),
            discharge_kw=day["actual_discharge_kw"].to_numpy(),
            soc_kwh=soc,
            emergency_purchase_kw=day["realized_emergency_kw"].to_numpy(),
            spill_kw=day["realized_pv_spill_kw"].to_numpy(),
            planned_cost_yuan=summary.planned_purchase_cost_yuan,
            emergency_cost_yuan=summary.realized_emergency_cost_yuan,
            total_cost_yuan=summary.realized_total_cost_yuan,
            initial_energy_kwh=summary.actual_initial_energy_kwh,
            final_energy_kwh=summary.actual_final_energy_kwh,
            solver_status=summary.status,
            solver_runtime_seconds=summary.runtime_seconds,
            price_yuan_per_kwh=day["price_yuan_per_kwh"].to_numpy(),
            evaluation_basis="realized",
        ))
    return summarize_q2_backtest(results)


def compare_methods(results_by_method: Mapping[str, Iterable[DailyOptimizerResult]], *, baseline_method: str) -> pd.DataFrame:
    if baseline_method not in results_by_method:
        raise KeyError(f"missing baseline {baseline_method!r}")
    rows = []
    dates: pd.DatetimeIndex | None = None
    for method, results in results_by_method.items():
        evaluation = summarize_chronological_results(results)
        observed = pd.DatetimeIndex(evaluation.daily["date"])
        if dates is not None and not observed.equals(dates):
            raise AssertionError("method comparisons must use identical dates")
        dates = observed
        summary = evaluation.period.iloc[0]
        rows.append({
            "method": method, "planned_cost": summary.cumulative_planned_purchase_cost_yuan,
            "emergency_cost": summary.cumulative_emergency_purchase_cost_yuan,
            "total_cost": summary.cumulative_total_cost_yuan,
            "emergency_energy": summary.cumulative_emergency_energy_kwh,
            "curtailment_energy": summary.cumulative_pv_curtailment_energy_kwh,
            "battery_throughput": summary.cumulative_battery_throughput_kwh,
            "solver_runtime": summary.total_solver_runtime_seconds,
        })
    table = pd.DataFrame(rows)
    baseline = float(table.loc[table["method"].eq(baseline_method), "total_cost"].iloc[0])
    table["cost_saving"] = baseline - table["total_cost"]
    table["cost_saving_percent"] = 100 * table["cost_saving"] / baseline
    return table[list(METHOD_COMPARISON_COLUMNS)]


def summarize_risk_sweep(results_by_risk_parameter: Mapping[float, Iterable[DailyOptimizerResult]]) -> pd.DataFrame:
    if not results_by_risk_parameter:
        raise ValueError("risk sweep requires results")
    rows = []
    dates: pd.DatetimeIndex | None = None
    for parameter, results in results_by_risk_parameter.items():
        evaluation = summarize_chronological_results(results)
        observed = pd.DatetimeIndex(evaluation.daily["date"])
        if dates is not None and not observed.equals(dates):
            raise AssertionError("risk settings must use identical dates")
        dates = observed
        summary = evaluation.period.iloc[0]
        rows.append({
            "risk_parameter": _finite_scalar("risk_parameter", parameter),
            "planned_cost": summary.cumulative_planned_purchase_cost_yuan,
            "emergency_cost": summary.cumulative_emergency_purchase_cost_yuan,
            "total_cost": summary.cumulative_total_cost_yuan,
            "emergency_energy": summary.cumulative_emergency_energy_kwh,
            "emergency_days": summary.emergency_purchase_days,
            "pv_curtailment": summary.cumulative_pv_curtailment_energy_kwh,
            "battery_throughput": summary.cumulative_battery_throughput_kwh,
            "final_energy": summary.final_battery_energy_kwh,
            "solver_runtime": summary.total_solver_runtime_seconds,
        })
    return pd.DataFrame(rows).sort_values("risk_parameter")[list(RISK_SWEEP_COLUMNS)].reset_index(drop=True)


def minimum_total_cost_setting(risk_sweep: pd.DataFrame) -> pd.Series:
    return risk_sweep.loc[risk_sweep["total_cost"].astype(float).idxmin()].copy()


def minimum_emergency_energy_setting(risk_sweep: pd.DataFrame) -> pd.Series:
    return risk_sweep.loc[risk_sweep["emergency_energy"].astype(float).idxmin()].copy()


def pareto_cost_risk_points(risk_sweep: pd.DataFrame) -> pd.DataFrame:
    values = risk_sweep[["total_cost", "emergency_energy"]].to_numpy(dtype=float)
    dominated = np.array([bool(((values <= point).all(axis=1) & (values < point).any(axis=1)).any()) for point in values])
    return risk_sweep.loc[~dominated].sort_values("risk_parameter").reset_index(drop=True)


def write_evaluation_tables(
    *, daily: pd.DataFrame | None = None, period: pd.DataFrame | None = None,
    method_comparison: pd.DataFrame | None = None, risk_sweep: pd.DataFrame | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    supplied = {"q2_daily_evaluation.csv": daily, "q2_period_summary.csv": period,
                "q2_method_comparison.csv": method_comparison, "q2_risk_sweep.csv": risk_sweep}
    if not any(value is not None for value in supplied.values()):
        raise ValueError("at least one populated evaluation table must be supplied")
    destination = Path(output_dir) if output_dir is not None else problem_results_dir(2)
    destination.mkdir(parents=True, exist_ok=True)
    written = {}
    for filename, table in supplied.items():
        if table is None:
            continue
        if table.empty:
            raise ValueError(f"refusing to write empty placeholder {filename}")
        columns = list(OUTPUT_TABLE_SCHEMAS[filename])
        missing = set(columns).difference(table.columns)
        if missing:
            raise ValueError(f"{filename} is missing {sorted(missing)}")
        path = destination / filename
        table[columns].to_csv(path, index=False)
        written[filename] = path
    return written


__all__ = [
    "BacktestEvaluation", "DAILY_EVALUATION_COLUMNS", "DT_HOURS",
    "DailyOptimizerResult", "FORMAL_DATES", "HORIZON",
    "METHOD_COMPARISON_COLUMNS", "OUTPUT_TABLE_SCHEMAS",
    "PERIOD_SUMMARY_COLUMNS", "RISK_SWEEP_COLUMNS",
    "calculate_cost_from_power", "compare_methods", "daily_metrics",
    "expected_energy_kwh", "load_final_q2_evaluation",
    "minimum_emergency_energy_setting", "minimum_total_cost_setting",
    "pareto_cost_risk_points", "summarize_chronological_results",
    "summarize_q2_backtest", "summarize_risk_sweep", "trajectory_energy_kwh",
    "validate_daily_result", "write_evaluation_tables",
]
