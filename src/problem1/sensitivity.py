"""One-factor Q1 sensitivity using the unchanged deterministic MILP."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.common.data_validation import file_sha256
from src.common.paths import PROJECT_ROOT, problem_results_dir
from src.problem1.evaluate import build_summary, compute_baseline, validate_dispatch
from src.problem1.model import DT_HOURS, DispatchParameters, DispatchResult, solve_deterministic_dispatch


TABLE_DIR = problem_results_dir(1) / "tables"
INPUT_PATH = PROJECT_ROOT / "data/processed/C题/problem1_day.csv"
AUDIT_PATH = TABLE_DIR / "q1_sensitivity_audit.json"
EFFICIENCIES = (0.80, 0.85, 0.90, 0.95)
MULTIPLIERS = (0.50, 0.75, 1.00, 1.25)
TOLERANCE = 1e-3  # CBC text-output precision, in kW/kWh/yuan as appropriate.


def sensitivity_cases() -> list[tuple[str, float, DispatchParameters]]:
    """Vary only the requested parameter, including capacity-linked SOC limits."""
    base = DispatchParameters()
    cases = [("efficiency", eta, replace(base, charge_efficiency=eta, discharge_efficiency=eta))
             for eta in EFFICIENCIES]
    for multiplier in MULTIPLIERS:
        capacity = base.capacity_kwh * multiplier
        cases.append(("capacity", multiplier, replace(
            base, capacity_kwh=capacity, minimum_energy_kwh=0.1 * capacity,
            maximum_energy_kwh=0.9 * capacity, initial_energy_kwh=0.5 * capacity,
            terminal_energy_kwh=0.5 * capacity,
        )))
    cases.extend(("power", multiplier, replace(
        base, maximum_charge_kw=base.maximum_charge_kw * multiplier,
        maximum_discharge_kw=base.maximum_discharge_kw * multiplier,
    )) for multiplier in MULTIPLIERS)
    return cases


def audit_case(result: DispatchResult, parameters: DispatchParameters) -> dict[str, object]:
    """Reuse official validation and add finite, chain and lower-bound checks."""
    frame = result.dispatch
    if result.status != "Optimal" or not np.isfinite(result.objective_yuan):
        raise AssertionError("Sensitivity run must be optimal with finite objective")
    numeric = frame.select_dtypes(include="number").to_numpy(float)
    if not np.isfinite(numeric).all():
        raise AssertionError("Sensitivity dispatch contains nonfinite values")
    if frame["slot"].tolist() != list(range(1, 145)):
        raise AssertionError("Sensitivity slots must be exactly 1..144")
    validation = validate_dispatch(result, parameters, tolerance=TOLERANCE)
    continuity = float(np.max(np.abs(
        frame["storage_start_kwh"].to_numpy()[1:] - frame["storage_end_kwh"].to_numpy()[:-1]
    )))
    minimum_action = float(frame[["grid_purchase_kw", "charge_kw", "discharge_kw", "spill_kw"]].min().min())
    if minimum_action < -TOLERANCE:
        raise AssertionError("Negative grid/charge/discharge/spill action")
    if continuity > TOLERANCE:
        raise AssertionError("SOC trajectory is discontinuous between intervals")
    # Check the energy-unit columns as well as the solver's original powers.
    for power, energy in (("grid_purchase_kw", "grid_purchase_kwh"), ("charge_kw", "charge_kwh"),
                          ("discharge_kw", "discharge_kwh"), ("spill_kw", "spill_kwh")):
        if not np.allclose(frame[energy], frame[power] * DT_HOURS, rtol=0, atol=1e-8):
            raise AssertionError("Power-to-energy conversion must be /6")
    return {**asdict(validation), "soc_chain_residual_kwh": continuity,
            "minimum_action_kw": minimum_action, "physical_invariants_pass": True}


def protected_hashes() -> dict[str, str]:
    """Snapshot raw inputs, baseline Q1, and all Q2/Q3/Q4 work before a run."""
    roots = [PROJECT_ROOT / "data", PROJECT_ROOT / "src/forecasting"]
    for problem in (2, 3, 4):
        roots += [PROJECT_ROOT / f"src/problem{problem}", problem_results_dir(problem),
                  PROJECT_ROOT / f"paper/figures/problem{problem}"]
    paths = {path for root in roots for path in root.rglob("*")
             if path.is_file() and "__pycache__" not in path.parts and path.name != ".DS_Store"}
    paths.update(TABLE_DIR.glob("table_p1_*"))
    paths.update(path for path in (problem_results_dir(1) / "figures").glob("*")
                 if path.is_file() and "sensitivity" not in path.name)
    for relative in ("src/problem1/model.py", "src/problem1/evaluate.py", "src/problem1/run.py",
                     "results/problem1/result1.xlsx", "paper/contents/sections/06_problem2.tex",
                     "paper/contents/sections/07_problem3.tex", "paper/contents/sections/08_problem4.tex"):
        path = PROJECT_ROOT / relative
        if path.exists():
            paths.add(path)
    return {str(path.relative_to(PROJECT_ROOT)): file_sha256(path) for path in sorted(paths)}


def summarize_sensitivity(table: pd.DataFrame) -> pd.DataFrame:
    """Compare marginal gains without treating unequal sweep widths as comparable."""
    rows = []
    for group, part in table.groupby("sweep", sort=False):
        part = part.sort_values("parameter_value")
        x = part["parameter_value"].to_numpy(float)
        cost = part["optimal_cost_yuan"].to_numpy(float)
        base_index = 2
        gains = -np.diff(cost)
        gain_per_unit = gains / np.diff(x)
        # Dimensionless, baseline-to-next-point finite-change elasticity.
        elasticity = ((cost[base_index] - cost[3]) / cost[base_index]) / ((x[3] - x[base_index]) / x[base_index])
        diminishing = bool(np.all(np.diff(gain_per_unit) <= 0.01) and gain_per_unit[0] > gain_per_unit[-1] + 0.01)
        rows.append({
            "sweep": group, "cases": len(part), "baseline_cost_yuan": cost[base_index],
            "lowest_setting_cost_yuan": cost[0], "highest_setting_cost_yuan": cost[-1],
            "full_range_cost_reduction_yuan": cost[0] - cost[-1],
            "full_range_cost_reduction_percent": 100 * (cost[0] - cost[-1]) / cost[0],
            "gain_step1_yuan": gains[0], "gain_step2_yuan": gains[1], "gain_step3_yuan": gains[2],
            "baseline_to_upper_cost_saving_percent": 100 * (cost[base_index] - cost[3]) / cost[base_index],
            "baseline_to_upper_normalized_cost_sensitivity": elasticity,
            "cost_nonincreasing": bool(np.all(np.diff(cost) <= 0.01)),
            "diminishing_return_observed": diminishing,
        })
    return pd.DataFrame(rows)


def run_analysis() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    data = pd.read_csv(INPUT_PATH)
    before = protected_hashes()
    base = DispatchParameters()
    stored = json.loads((TABLE_DIR / "table_p1_summary.json").read_text())
    baseline = compute_baseline(data)
    cache: dict[DispatchParameters, DispatchResult] = {}
    rows, dispatches = [], []
    for group, value, parameters in sensitivity_cases():
        if parameters not in cache:
            result = solve_deterministic_dispatch(data, parameters)
            audit_case(result, parameters)  # No infeasible run is retained or exported.
            cache[parameters] = result
            dispatches.append(result.dispatch.assign(case_id=f"{group}_{value:g}"))
        result = cache[parameters]
        invariants = audit_case(result, parameters)
        validation = validate_dispatch(result, parameters)
        metrics = build_summary(result, validation, baseline, parameters)
        rows.append({"sweep": group, "parameter_value": value,
                     "case_id": f"{group}_{value:g}",
                     "dispatch_case_id": "efficiency_0.9" if parameters == base else f"{group}_{value:g}",
                     **asdict(parameters), "round_trip_efficiency": parameters.charge_efficiency * parameters.discharge_efficiency,
                     **metrics, **invariants})
        print(f"{group}={value:g}: {result.status}; cost={result.objective_yuan:.6f}; physical=PASS", flush=True)
    table = pd.DataFrame(rows)
    actual_base = table.loc[table["case_id"].eq("efficiency_0.9")].iloc[0]
    for metric in ("optimal_cost_yuan", "total_grid_energy_kwh", "total_charge_energy_kwh",
                   "total_discharge_energy_kwh", "total_spill_energy_kwh"):
        if abs(float(actual_base[metric]) - float(stored[metric])) > TOLERANCE:
            raise AssertionError(f"Baseline Q1 metric changed: {metric}")
    stored_dispatch = pd.read_csv(TABLE_DIR / "table_p1_dispatch.csv")
    replay = cache[base].dispatch
    numeric_columns = replay.select_dtypes(include="number").columns
    max_baseline_gap = float(np.max(np.abs(
        stored_dispatch[numeric_columns].to_numpy() - replay[numeric_columns].to_numpy()
    )))
    # A linear objective can have several schedules of identical cost. Verify
    # the saved schedule independently instead of imposing a new tie-break.
    stored_result = DispatchResult(stored_dispatch, "Optimal", float(stored["optimal_cost_yuan"]), 0.0, "stored baseline")
    validate_dispatch(stored_result, base, tolerance=TOLERANCE)
    if not np.allclose(stored_dispatch[["price_yuan_per_kwh", "load_kw", "pv_forecast_kw"]],
                       replay[["price_yuan_per_kwh", "load_kw", "pv_forecast_kw"]], rtol=0, atol=1e-10):
        raise AssertionError("Stored baseline uses different input data")
    after = protected_hashes()
    if before != after:
        raise AssertionError("Protected inputs or Q1/Q2/Q3/Q4 artifacts changed")
    summary = summarize_sensitivity(table)
    # Write only once every case, baseline and protected-file audit has passed.
    for group in ("efficiency", "capacity", "power"):
        table.loc[table["sweep"].eq(group)].to_csv(
            TABLE_DIR / f"q1_{group}_sensitivity.csv", index=False, float_format="%.10g")
    summary.to_csv(TABLE_DIR / "q1_sensitivity_summary.csv", index=False, float_format="%.10g")
    pd.concat(dispatches, ignore_index=True).to_csv(
        TABLE_DIR / "q1_sensitivity_dispatch.csv", index=False, float_format="%.10g")
    audit = {
        "case_count": len(table), "unique_solve_count": len(cache),
        "baseline_q1_unchanged": True, "baseline_max_dispatch_gap": max_baseline_gap,
        "baseline_resolve_schedule_identical": max_baseline_gap <= TOLERANCE,
        "baseline_alternative_optimum": max_baseline_gap > TOLERANCE,
        "baseline_comparison_rule": "unchanged saved files; independently feasible saved schedule; equal objective and aggregate energies within 1e-3; no new tie-break",
        "baseline_cost_yuan": cache[base].objective_yuan,
        "no_storage_cost_yuan": baseline["baseline_cost_yuan"],
        "all_cases_optimal": bool(table["solver_status"].eq("Optimal").all()),
        "all_physical_invariants_pass": bool(table["physical_invariants_pass"].all()),
        "physical_tolerance": TOLERANCE,
        "maximum_power_balance_residual_kw": float(table["maximum_power_balance_residual_kw"].max()),
        "maximum_soc_recurrence_residual_kwh": float(table["maximum_energy_transition_residual_kwh"].max()),
        "protected_hashes_before": before, "protected_hashes_after": after,
        "protected_files_unchanged": before == after,
        "dominance_rule": "baseline-to-upper fractional cost reduction / fractional parameter increase; one-day finite differences, unequal step sizes",
        "dominant_parameter_by_normalized_sensitivity": str(summary.loc[summary["baseline_to_upper_normalized_cost_sensitivity"].idxmax(), "sweep"]),
        "energy_reporting_convention": "AC bus-side charge/discharge energy; kWh=kW/6; SOC includes all 145 boundary states",
    }
    AUDIT_PATH.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    return table, summary, audit


def write_paper_numbers(table: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Generate paper values directly from validated sensitivity results."""
    indexed = table.set_index("case_id")
    effects = summary.set_index("sweep")
    numbers = {
        "QOneEffLowCost": (indexed.loc["efficiency_0.8", "optimal_cost_yuan"], 2),
        "QOneEffHighCost": (indexed.loc["efficiency_0.95", "optimal_cost_yuan"], 2),
        "QOneEffReduction": (effects.loc["efficiency", "full_range_cost_reduction_percent"], 2),
        "QOneCapacityGainFirst": (effects.loc["capacity", "gain_step1_yuan"], 2),
        "QOneCapacityGainSecond": (effects.loc["capacity", "gain_step2_yuan"], 2),
        "QOneCapacityGainThird": (effects.loc["capacity", "gain_step3_yuan"], 2),
        "QOnePowerGainFirst": (effects.loc["power", "gain_step1_yuan"], 2),
        "QOnePowerGainSecond": (effects.loc["power", "gain_step2_yuan"], 2),
        "QOnePowerGainThird": (effects.loc["power", "gain_step3_yuan"], 2),
        "QOneEffElasticity": (effects.loc["efficiency", "baseline_to_upper_normalized_cost_sensitivity"], 3),
        "QOneCapacityElasticity": (effects.loc["capacity", "baseline_to_upper_normalized_cost_sensitivity"], 3),
        "QOnePowerElasticity": (effects.loc["power", "baseline_to_upper_normalized_cost_sensitivity"], 3),
    }
    lines = [r"\newcommand{" + "\\" + name + "}{" + f"{value:.{places}f}" + "}"
             for name, (value, places) in numbers.items()]
    (TABLE_DIR / "q1_sensitivity_numbers.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    table, summary, audit = run_analysis()
    from src.problem1.visualize import create_q1_sensitivity_figure
    create_q1_sensitivity_figure(table)
    write_paper_numbers(table, summary)
    if protected_hashes() != audit["protected_hashes_before"]:
        raise AssertionError("Protected artifact changed during figure export")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
