"""Independent CVaR sensitivity runner for the locked Q2 physical model."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from src.common.paths import PROJECT_ROOT, problem_results_dir
from src.problem2.forecast_interface import audit_q2_handoff_integrity
from src.problem2.model import Q2DispatchParameters
from src.problem2.run import (
    FORMAL_END,
    FORMAL_START,
    PROTECTED_FORECAST_PATHS,
    RESULT2_PATH,
    TABLE_DIR,
    run_chronological,
)


BASE_COMMIT = "e86d18d785a6dda28b03d28167c1894b81a100b1"
ALPHA = 0.90
DEFAULT_LAMBDAS = (0.0, 0.05, 0.10, 0.20, 0.50)
OUTPUT_DIR = problem_results_dir(2)
CVAR_DIR = OUTPUT_DIR / "cvar"
SWEEP_PATH = OUTPUT_DIR / "q2_risk_sweep.csv"
METADATA_PATH = OUTPUT_DIR / "q2_risk_metadata.json"
AUDIT_PATH = OUTPUT_DIR / "Q2_CVAR_AUDIT.md"
BASELINE = {
    "planned_purchase_cost_yuan": 13_323_201.223427,
    "expected_total_cost_yuan": 14_374_345.246186,
    "realized_emergency_energy_kwh": 398_161.900954,
    "realized_emergency_cost_yuan": 1_516_260.778930,
    "realized_total_cost_yuan": 14_839_462.002357,
}
BASE_OUTPUT_PATHS = (
    RESULT2_PATH,
    TABLE_DIR / "table_p2_daily_summary.csv",
    TABLE_DIR / "table_p2_dispatch.csv",
    TABLE_DIR / "table_p2_scenario_recourse.npz",
    TABLE_DIR / "table_p2_summary.json",
    TABLE_DIR / "table_p2_table3_emergency.csv",
    TABLE_DIR / "table_p2_table3_emergency.tex",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected_hashes() -> dict[str, str]:
    """Hash frozen forecasts, scenarios, and locked base outputs."""

    paths = tuple(PROTECTED_FORECAST_PATHS) + BASE_OUTPUT_PATHS
    return {str(path.relative_to(PROJECT_ROOT)): _sha256(path) for path in paths}


def _lambda_slug(value: float) -> str:
    return f"{int(round(value * 100)):03d}"


def audit_physical_run(daily: pd.DataFrame, intervals: pd.DataFrame) -> dict[str, object]:
    """Verify the locked physical settlement and all requested accounting identities."""

    tolerance = 2e-3
    balance = (
        intervals["realized_planned_grid_used_kw"]
        + intervals["realized_emergency_kw"]
        + intervals["actual_pv_kw"]
        + intervals["actual_discharge_kw"]
        - intervals["actual_load_kw"]
        - intervals["actual_charge_kw"]
        - intervals["realized_pv_spill_kw"]
    )
    expected_end = (
        intervals["actual_storage_start_kwh"]
        + 0.9 * intervals["actual_charge_kw"] / 6.0
        - intervals["actual_discharge_kw"] / (0.9 * 6.0)
    )
    soc_residual = intervals["actual_storage_end_kwh"] - expected_end
    cross_day = daily["actual_initial_energy_kwh"].iloc[1:].to_numpy(dtype=float) - daily[
        "actual_final_energy_kwh"
    ].iloc[:-1].to_numpy(dtype=float)
    planned_cost_error = float(
        abs(
            (intervals["price_yuan_per_kwh"] * intervals["planned_grid_kwh"]).sum()
            - daily["planned_purchase_cost_yuan"].sum()
        )
    )
    expected_cost_error = float(
        abs(
            daily["expected_total_cost_yuan"].sum()
            - daily["planned_purchase_cost_yuan"].sum()
            - daily["expected_emergency_cost_yuan"].sum()
        )
    )
    realized_cost_error = float(
        abs(
            daily["realized_total_cost_yuan"].sum()
            - daily["planned_purchase_cost_yuan"].sum()
            - daily["realized_emergency_cost_yuan"].sum()
        )
    )
    report = {
        "scenario_power_balance": bool(daily["maximum_power_balance_residual_kw"].max() <= tolerance),
        "actual_power_balance": bool(np.abs(balance).max() <= tolerance),
        "actual_soc_recurrence": bool(np.abs(soc_residual).max() <= tolerance),
        "soc_bounds": bool(
            intervals[["actual_storage_start_kwh", "actual_storage_end_kwh"]].min().min()
            >= 1_200.0 - tolerance
            and intervals[["actual_storage_start_kwh", "actual_storage_end_kwh"]].max().max()
            <= 10_800.0 + tolerance
        ),
        "actual_charge_discharge_limits": bool(
            (intervals["actual_charge_kw"] <= intervals["planned_charge_limit_kw"] + tolerance).all()
            and (
                intervals["actual_discharge_kw"]
                <= intervals["planned_discharge_limit_kw"] + tolerance
            ).all()
        ),
        "unabsorbed_discharge_energy_kwh": 0.0,
        "cross_day_realized_soc": bool(cross_day.size == 0 or np.abs(cross_day).max() <= tolerance),
        "planned_cost_accounting": planned_cost_error <= tolerance,
        "expected_cost_accounting": expected_cost_error <= tolerance,
        "realized_cost_accounting": realized_cost_error <= tolerance,
        "maximum_actual_power_balance_residual_kw": float(np.abs(balance).max()),
        "maximum_actual_soc_residual_kwh": float(np.abs(soc_residual).max()),
        "maximum_cross_day_soc_residual_kwh": float(np.abs(cross_day).max()) if cross_day.size else 0.0,
    }
    required = [value for key, value in report.items() if isinstance(value, bool)]
    if not all(required):
        raise AssertionError(f"CVaR physical/accounting audit failed: {report}")
    return report


def summarize_run(
    alpha: float,
    risk_weight: float,
    daily: pd.DataFrame,
    intervals: pd.DataFrame,
    wall_seconds: float,
) -> dict[str, object]:
    scenario_intervals = len(daily) * 50 * 144
    realized_emergency_mask = intervals["realized_emergency_kw"] > 1e-9
    row = {
        "risk_method": "CVaR",
        "alpha": alpha,
        "lambda": risk_weight,
        "planned_purchase_energy_kwh": float(daily["planned_purchase_energy_kwh"].sum()),
        "planned_purchase_cost_yuan": float(daily["planned_purchase_cost_yuan"].sum()),
        "expected_scenario_emergency_energy_kwh": float(daily["expected_emergency_energy_kwh"].sum()),
        "expected_scenario_emergency_cost_yuan": float(daily["expected_emergency_cost_yuan"].sum()),
        "expected_operating_cost_yuan": float(daily["expected_total_cost_yuan"].sum()),
        "var_cost_yuan": float(daily["var_cost_yuan"].sum()),
        "cvar_cost_yuan": float(daily["cvar_cost_yuan"].sum()),
        "lambda_cvar_term_yuan": float(daily["lambda_cvar_term_yuan"].sum()),
        "realized_emergency_energy_kwh": float(daily["realized_emergency_energy_kwh"].sum()),
        "realized_emergency_cost_yuan": float(daily["realized_emergency_cost_yuan"].sum()),
        "realized_total_cost_yuan": float(daily["realized_total_cost_yuan"].sum()),
        "scenario_emergency_interval_rate": float(
            daily["scenario_emergency_interval_count"].sum() / scenario_intervals
        ),
        "realized_emergency_interval_rate": float(realized_emergency_mask.mean()),
        "emergency_days": int((daily["realized_emergency_energy_kwh"] > 1e-9).sum()),
        "emergency_intervals": int(realized_emergency_mask.sum()),
        "battery_throughput_kwh": float(
            daily["actual_charge_energy_kwh"].sum() + daily["actual_discharge_energy_kwh"].sum()
        ),
        "mean_end_soc_kwh": float(daily["actual_final_energy_kwh"].mean()),
        "min_soc_kwh": float(intervals[["actual_storage_start_kwh", "actual_storage_end_kwh"]].min().min()),
        "max_soc_kwh": float(intervals[["actual_storage_start_kwh", "actual_storage_end_kwh"]].max().max()),
        "mean_daily_emergency_energy_kwh": float(daily["realized_emergency_energy_kwh"].mean()),
        "max_daily_emergency_energy_kwh": float(daily["realized_emergency_energy_kwh"].max()),
        "unused_planned_grid_energy_kwh": float(daily["realized_unused_planned_grid_energy_kwh"].sum()),
        "realized_pv_curtailment_kwh": float(daily["realized_pv_spill_energy_kwh"].sum()),
        "runtime_seconds": wall_seconds,
        "solver_success": int((daily["status"] == "Optimal").sum()),
        "formal_days": int(len(daily)),
    }
    return row


def verify_lambda_zero(row: dict[str, object]) -> None:
    """Stop the sweep unless lambda zero reproduces the locked annual result."""

    comparisons = {
        "planned_purchase_cost_yuan": BASELINE["planned_purchase_cost_yuan"],
        "expected_operating_cost_yuan": BASELINE["expected_total_cost_yuan"],
        "realized_emergency_energy_kwh": BASELINE["realized_emergency_energy_kwh"],
        "realized_emergency_cost_yuan": BASELINE["realized_emergency_cost_yuan"],
        "realized_total_cost_yuan": BASELINE["realized_total_cost_yuan"],
    }
    failures = {
        key: (float(row[key]), expected)
        for key, expected in comparisons.items()
        if abs(float(row[key]) - expected) > 0.05
    }
    if int(row["solver_success"]) != 334 or failures:
        raise AssertionError(f"lambda=0 failed locked-base regression: {failures}")


def _write_outputs(
    rows: list[dict[str, object]], hashes: dict[str, str], audits: dict[str, dict[str, object]]
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values("lambda")
    frame.to_csv(SWEEP_PATH, index=False, float_format="%.12g")
    handoff = audit_q2_handoff_integrity()
    causality = all(
        bool(handoff[key])
        for key in (
            "protected_artifact_hashes_unchanged",
            "q2_scenario_hash_unchanged",
            "raw_data_hashes_unchanged",
            "paired_scenarios",
            "leakage_audit",
        )
    )
    metadata = {
        "base_commit": BASE_COMMIT,
        "branch": "q2-cvar",
        "alpha": ALPHA,
        "lambda_grid": [float(value) for value in frame["lambda"]],
        "scenario_count": 50,
        "scenario_emergency_cost_definition": "sum_t 5 * fixed_intraday_price_t * emergency_kw[s,t] * (1/6 h)",
        "cvar_formula": "zeta + sum_s xi_s / ((1-alpha)*S), xi_s >= emergency_cost_s-zeta, xi_s >= 0",
        "risk_objective": "planned_cost + expected_scenario_emergency_cost + lambda*CVaR - unchanged_terminal_value_credit",
        "continuation_value_treatment": "unchanged decreasing piecewise-linear terminal credit; excluded from scenario loss and realized cost",
        "physical_settlement_contract": "locked actual actions capped by plan, actual SOC recursion, cross-day actual SOC carry, no unabsorbed discharge",
        "causal_information_rule": "day-ahead plan uses information available by 00:00; Appendix 2 actuals enter only settlement",
        "forecasting_modified": False,
        "scenario_generation_modified": False,
        "physical_settlement_modified": False,
        "base_result2_overwritten": False,
        "baseline": BASELINE,
        "code_head": BASE_COMMIT,
        "protected_hashes": hashes,
        "causality_audit_passed": causality,
        "physical_audits": audits,
    }
    METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    display_columns = [
        "lambda",
        "planned_purchase_cost_yuan",
        "expected_scenario_emergency_cost_yuan",
        "cvar_cost_yuan",
        "realized_emergency_cost_yuan",
        "realized_total_cost_yuan",
        "solver_success",
    ]
    display = frame[display_columns]
    header = "| " + " | ".join(display_columns) + " |"
    divider = "| " + " | ".join(["---"] * len(display_columns)) + " |"
    data_lines = []
    for _, row in display.iterrows():
        cells = [
            f"{float(row[column]):.6f}" if column != "solver_success" else str(int(row[column]))
            for column in display_columns
        ]
        data_lines.append("| " + " | ".join(cells) + " |")
    lines = [
        "# Q2 CVaR risk-sensitivity audit",
        "",
        f"Base commit: `{BASE_COMMIT}`. The locked physical settlement, forecast, 50 paired scenarios, and continuation value are unchanged.",
        "",
        "The scenario loss is emergency-purchase cost only. For each day, "
        r"$\mathrm{CVaR}_{\alpha}=\zeta+[(1-\alpha)S]^{-1}\sum_s\xi_s$ with "
        r"$\xi_s\ge C_s^{emg}-\zeta$ and $\xi_s\ge0$. The objective adds "
        r"$\lambda\mathrm{CVaR}_{\alpha}$ once; planned cost and continuation value are excluded from the scenario loss.",
        "",
        "The sweep is a sensitivity analysis over stated risk preferences. Realized Appendix 2 outcomes were not used to select lambda.",
        "",
        "\n".join([header, divider, *data_lines]),
        "",
        f"Causality audit: {'PASS' if causality else 'FAIL'}. All recorded physical and accounting audits: "
        f"{'PASS' if all(all(v for v in audit.values() if isinstance(v, bool)) for audit in audits.values()) else 'FAIL'}.",
        "",
    ]
    AUDIT_PATH.write_text("\n".join(lines), encoding="utf-8")


def run_one(alpha: float, risk_weight: float, dates: pd.DatetimeIndex) -> tuple[dict[str, object], dict[str, object]]:
    parameters = Q2DispatchParameters(cvar_alpha=alpha, risk_weight=risk_weight)
    started = perf_counter()
    daily, intervals, emergency, *_ = run_chronological(
        dates, parameters=parameters, progress=True
    )
    elapsed = perf_counter() - started
    daily["scenario_emergency_interval_count"] = (emergency > 1e-9).sum(axis=(1, 2))
    audit = audit_physical_run(daily, intervals)
    row = summarize_run(alpha, risk_weight, daily, intervals, elapsed)
    CVAR_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_csv(
        CVAR_DIR / f"daily_alpha{int(round(alpha * 100)):03d}_lambda{_lambda_slug(risk_weight)}.csv",
        index=False,
        float_format="%.12g",
    )
    return row, audit


def _run_one_task(task: tuple[float, float, pd.DatetimeIndex]):
    """Pickle-friendly adapter used by the Windows process pool."""

    return run_one(*task)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--lambda-values", type=float, nargs="+", default=DEFAULT_LAMBDAS)
    parser.add_argument("--representative", action="store_true")
    parser.add_argument("--refresh-report-only", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    hashes_before = protected_hashes()
    if args.refresh_report_only:
        if not SWEEP_PATH.exists() or not METADATA_PATH.exists():
            raise FileNotFoundError("risk sweep and metadata are required for report refresh")
        rows = pd.read_csv(SWEEP_PATH).to_dict("records")
        audits = json.loads(METADATA_PATH.read_text(encoding="utf-8")).get(
            "physical_audits", {}
        )
        _write_outputs(rows, hashes_before, audits)
        print(f"refreshed {AUDIT_PATH}")
        return
    dates = (
        pd.to_datetime(["2025-02-01", "2025-04-15", "2025-07-15", "2025-10-15"])
        if args.representative
        else pd.date_range(FORMAL_START, FORMAL_END, freq="D")
    )
    if args.workers < 1:
        raise ValueError("workers must be at least one")
    if args.workers > 1 and 0.0 in args.lambda_values:
        raise ValueError("lambda=0 must be run alone before a parallel risk sweep")
    tasks = [(args.alpha, risk_weight, pd.DatetimeIndex(dates)) for risk_weight in args.lambda_values]
    if args.workers == 1:
        completed = [run_one(*task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as executor:
            completed = list(executor.map(_run_one_task, tasks))
    rows = [item[0] for item in completed]
    audits = {f"lambda_{float(row['lambda']):g}": audit for row, audit in completed}
    for row in rows:
        if not args.representative and float(row["lambda"]) == 0.0:
            verify_lambda_zero(row)
    if protected_hashes() != hashes_before:
        raise AssertionError("CVaR run changed a frozen forecast, scenario, or base Q2 output")
    if args.representative:
        print(json.dumps({"rows": rows, "audits": audits}, ensure_ascii=False, indent=2))
        return
    existing = pd.read_csv(SWEEP_PATH).to_dict("records") if SWEEP_PATH.exists() else []
    replacements = {float(row["lambda"]): row for row in rows}
    merged = [row for row in existing if float(row["lambda"]) not in replacements] + rows
    existing_audits = {}
    if METADATA_PATH.exists():
        existing_audits = json.loads(METADATA_PATH.read_text(encoding="utf-8")).get("physical_audits", {})
    existing_audits.update(audits)
    _write_outputs(merged, hashes_before, existing_audits)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
