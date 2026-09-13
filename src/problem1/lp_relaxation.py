"""Independent LP-relaxation audit for the validated Q1 MILP.

The formal solver in :mod:`src.problem1.model` is intentionally left unchanged.
This module reproduces the same model after removing only the binary operating
mode and its two mutual-exclusion constraints.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import pulp

from src.common.data_validation import file_sha256
from src.common.paths import PROJECT_ROOT, problem_results_dir
from src.problem1.evaluate import validate_dispatch
from src.problem1.model import (
    DT_HOURS,
    DispatchParameters,
    DispatchResult,
    solve_deterministic_dispatch,
)


INPUT_PATH = PROJECT_ROOT / "data/processed/C题/problem1_day.csv"
RESULT_DIR = problem_results_dir(1)
TABLE_DIR = RESULT_DIR / "tables"
COMPARISON_PATH = TABLE_DIR / "q1_lp_relaxation_comparison.csv"
PAPER_TABLE_PATH = TABLE_DIR / "q1_lp_relaxation_comparison.tex"
AUDIT_PATH = RESULT_DIR / "Q1_LP_RELAXATION_AUDIT.md"
NUMERICAL_TOLERANCE = 1e-3
OBJECTIVE_GAP_TOLERANCE_YUAN = 1e-5
SIMULTANEOUS_ACTION_TOLERANCE_KW = 1e-6


def _validated_input(data: pd.DataFrame) -> pd.DataFrame:
    """Apply the same input contract and ordering as the formal Q1 solver."""
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
    numeric = list(required - {"interval_start", "interval_end"})
    if frame[numeric].isna().any().any():
        raise ValueError("Q1 input contains missing numeric values")
    return frame


def solve_lp_relaxation(
    data: pd.DataFrame,
    parameters: DispatchParameters | None = None,
) -> DispatchResult:
    """Solve the exact continuous relaxation of the formal Q1 MILP.

    Relative to ``solve_deterministic_dispatch``, the only removed objects are
    ``charge_mode[t]`` and the two mode-dependent Big-M inequalities. Direct
    charge/discharge upper bounds retain the same 5000 kW feasible limits.
    """
    parameters = parameters or DispatchParameters()
    frame = _validated_input(data)
    periods = range(len(frame))
    model = pulp.LpProblem("CUMCM2026_Q1_LP_Relaxation", pulp.LpMinimize)
    grid = pulp.LpVariable.dicts("grid_kw", periods, lowBound=0)
    charge = pulp.LpVariable.dicts(
        "charge_kw", periods, lowBound=0, upBound=parameters.maximum_charge_kw
    )
    discharge = pulp.LpVariable.dicts(
        "discharge_kw", periods, lowBound=0, upBound=parameters.maximum_discharge_kw
    )
    spill = pulp.LpVariable.dicts("spill_kw", periods, lowBound=0)
    energy = pulp.LpVariable.dicts(
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
            grid[t] + pv_kw + discharge[t]
            == load_kw + charge[t] + spill[t]
        ), f"power_balance_{t + 1:03d}"
        model += (
            energy[t + 1]
            == energy[t]
            + parameters.charge_efficiency * charge[t] * DT_HOURS
            - discharge[t] * DT_HOURS / parameters.discharge_efficiency
        ), f"energy_transition_{t + 1:03d}"
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
        raise RuntimeError(f"CBC did not find an optimal Q1 LP solution: {status}")

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
    for power, energy_name in (
        ("grid_purchase_kw", "grid_purchase_kwh"),
        ("charge_kw", "charge_kwh"),
        ("discharge_kw", "discharge_kwh"),
        ("spill_kw", "spill_kwh"),
    ):
        result[energy_name] = result[power] * DT_HOURS

    return DispatchResult(
        dispatch=result,
        status=status,
        objective_yuan=float(pulp.value(model.objective)),
        runtime_seconds=runtime_seconds,
        solver_name="PuLP CBC (LP relaxation)",
    )


def solution_metrics(
    result: DispatchResult,
    parameters: DispatchParameters | None = None,
) -> dict[str, float | str | int | bool]:
    """Measure the common physical and numerical properties of either solution."""
    parameters = parameters or DispatchParameters()
    frame = result.dispatch
    power_residual = (
        frame["grid_purchase_kw"]
        + frame["pv_forecast_kw"]
        + frame["discharge_kw"]
        - frame["load_kw"]
        - frame["charge_kw"]
        - frame["spill_kw"]
    )
    expected_end = (
        frame["storage_start_kwh"]
        + parameters.charge_efficiency * frame["charge_kw"] * DT_HOURS
        - frame["discharge_kw"] * DT_HOURS / parameters.discharge_efficiency
    )
    recurrence_residual = frame["storage_end_kwh"] - expected_end
    soc_values = np.r_[
        frame["storage_start_kwh"].to_numpy(float),
        float(frame["storage_end_kwh"].iloc[-1]),
    ]
    simultaneous = np.minimum(
        frame["charge_kw"].to_numpy(float),
        frame["discharge_kw"].to_numpy(float),
    )
    objective_recalculated = float(
        (frame["price_yuan_per_kwh"] * frame["grid_purchase_kw"] * DT_HOURS).sum()
    )
    metrics: dict[str, float | str | int | bool] = {
        "solver_status": result.status,
        "objective_cost_yuan": result.objective_yuan,
        "total_grid_purchase_kwh": float(frame["grid_purchase_kwh"].sum()),
        "total_charge_kwh": float(frame["charge_kwh"].sum()),
        "total_discharge_kwh": float(frame["discharge_kwh"].sum()),
        "total_curtailment_kwh": float(frame["spill_kwh"].sum()),
        "minimum_soc_energy_kwh": float(soc_values.min()),
        "maximum_soc_energy_kwh": float(soc_values.max()),
        "maximum_simultaneous_charge_discharge_kw": float(simultaneous.max()),
        "simultaneous_interval_count_above_1e_6": int(
            np.sum(simultaneous > SIMULTANEOUS_ACTION_TOLERANCE_KW)
        ),
        "maximum_power_balance_residual_kw": float(np.abs(power_residual).max()),
        "maximum_soc_recurrence_residual_kwh": float(np.abs(recurrence_residual).max()),
        "solver_runtime_seconds": result.runtime_seconds,
        "objective_recalculation_error_yuan": abs(
            objective_recalculated - result.objective_yuan
        ),
    }
    finite = np.isfinite(frame.select_dtypes(include="number").to_numpy(float)).all()
    checks = {
        "optimal": result.status == "Optimal",
        "finite": bool(finite and np.isfinite(result.objective_yuan)),
        "144 intervals": len(frame) == 144,
        "nonnegative actions": float(
            frame[["grid_purchase_kw", "charge_kw", "discharge_kw", "spill_kw"]]
            .min()
            .min()
        )
        >= -NUMERICAL_TOLERANCE,
        "charge limit": float(frame["charge_kw"].max())
        <= parameters.maximum_charge_kw + NUMERICAL_TOLERANCE,
        "discharge limit": float(frame["discharge_kw"].max())
        <= parameters.maximum_discharge_kw + NUMERICAL_TOLERANCE,
        "spill limit": float(
            np.maximum(frame["spill_kw"] - frame["pv_forecast_kw"], 0).max()
        )
        <= NUMERICAL_TOLERANCE,
        "initial energy": abs(float(soc_values[0]) - parameters.initial_energy_kwh)
        <= NUMERICAL_TOLERANCE,
        "terminal energy": abs(float(soc_values[-1]) - parameters.terminal_energy_kwh)
        <= NUMERICAL_TOLERANCE,
        "minimum energy": float(soc_values.min())
        >= parameters.minimum_energy_kwh - NUMERICAL_TOLERANCE,
        "maximum energy": float(soc_values.max())
        <= parameters.maximum_energy_kwh + NUMERICAL_TOLERANCE,
        "power balance": float(metrics["maximum_power_balance_residual_kw"])
        <= NUMERICAL_TOLERANCE,
        "SOC recurrence": float(metrics["maximum_soc_recurrence_residual_kwh"])
        <= NUMERICAL_TOLERANCE,
        "objective accounting": float(metrics["objective_recalculation_error_yuan"])
        <= NUMERICAL_TOLERANCE,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"Q1 LP audit failed: {', '.join(failed)}; {metrics}")
    metrics["common_physical_invariants_pass"] = True
    return metrics


def _protected_hashes() -> dict[str, str]:
    def tree_digest(root: Path) -> str:
        digest = hashlib.sha256()
        files = sorted(
            path
            for path in root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.name != ".DS_Store"
        )
        for path in files:
            digest.update(str(path.relative_to(root)).encode("utf-8"))
            digest.update(file_sha256(path).encode("ascii"))
        return digest.hexdigest()

    paths = [
        INPUT_PATH,
        PROJECT_ROOT / "src/problem1/model.py",
        RESULT_DIR / "result1.xlsx",
        PROJECT_ROOT / "data/raw/C题/附件/附件1.xlsx",
    ]
    hashes = {
        str(path.relative_to(PROJECT_ROOT)): file_sha256(path)
        for path in paths
        if path.exists()
    }
    for relative in (
        "data/raw/C题/附件",
        "src/problem2",
        "src/problem3",
        "src/problem4",
        "results/problem2",
        "results/problem3",
        "results/problem4",
    ):
        root = PROJECT_ROOT / relative
        if root.exists():
            hashes[f"{relative}/**"] = tree_digest(root)
    for relative in (
        "paper/contents/sections/06_problem2.tex",
        "paper/contents/sections/07_problem3.tex",
        "paper/contents/sections/08_problem4.tex",
    ):
        path = PROJECT_ROOT / relative
        if path.exists():
            hashes[relative] = file_sha256(path)
    return hashes


def _schedule_differences(
    milp: DispatchResult,
    lp: DispatchResult,
) -> dict[str, object]:
    columns = [
        "grid_purchase_kw",
        "charge_kw",
        "discharge_kw",
        "spill_kw",
        "storage_start_kwh",
        "storage_end_kwh",
    ]
    differences: dict[str, object] = {}
    any_difference = np.zeros(len(milp.dispatch), dtype=bool)
    for column in columns:
        delta = np.abs(
            milp.dispatch[column].to_numpy(float) - lp.dispatch[column].to_numpy(float)
        )
        differences[f"maximum_absolute_{column}_difference"] = float(delta.max())
        any_difference |= delta > SIMULTANEOUS_ACTION_TOLERANCE_KW
    differences["intervals_with_any_schedule_difference_above_1e_6"] = int(
        any_difference.sum()
    )
    material_difference = np.zeros(len(milp.dispatch), dtype=bool)
    for column in columns:
        material_difference |= (
            np.abs(
                milp.dispatch[column].to_numpy(float)
                - lp.dispatch[column].to_numpy(float)
            )
            > NUMERICAL_TOLERANCE
        )
    differences["materially_different_intervals_above_1e_3"] = int(
        material_difference.sum()
    )
    differing_rows = []
    for index in np.flatnonzero(any_difference):
        differing_rows.append(
            {
                "slot": int(milp.dispatch.at[index, "slot"]),
                "interval_start": str(milp.dispatch.at[index, "interval_start"]),
                "milp_grid_kw": float(milp.dispatch.at[index, "grid_purchase_kw"]),
                "lp_grid_kw": float(lp.dispatch.at[index, "grid_purchase_kw"]),
                "milp_charge_kw": float(milp.dispatch.at[index, "charge_kw"]),
                "lp_charge_kw": float(lp.dispatch.at[index, "charge_kw"]),
                "milp_discharge_kw": float(milp.dispatch.at[index, "discharge_kw"]),
                "lp_discharge_kw": float(lp.dispatch.at[index, "discharge_kw"]),
                "milp_storage_end_kwh": float(
                    milp.dispatch.at[index, "storage_end_kwh"]
                ),
                "lp_storage_end_kwh": float(lp.dispatch.at[index, "storage_end_kwh"]),
            }
        )
    differences["differing_intervals"] = differing_rows
    return differences


def _write_paper_table(comparison: pd.DataFrame) -> None:
    rows = []
    labels = {"MILP": "正式MILP", "LP relaxation": "LP松弛"}
    for row in comparison.to_dict("records"):
        rows.append(
            f"    {labels[str(row['formulation'])]} & "
            f"{float(row['objective_cost_yuan']):.3f} & "
            f"{float(row['total_grid_purchase_kwh']):.3f} & "
            f"{float(row['total_charge_kwh']):.3f} & "
            f"{float(row['total_discharge_kwh']):.3f} & "
            f"{float(row['maximum_simultaneous_charge_discharge_kw']):.2e} \\\\"
        )
    content = "\n".join(
        [
            r"\begin{table}[H]",
            r"  \centering",
            r"  \caption{问题一正式MILP与LP松弛的同口径求解对照}",
            r"  \label{tab:p1-lp-relaxation}",
            r"  \small",
            r"  \begin{tabular}{lrrrrr}",
            r"    \toprule",
            r"    模型 & 成本/元 & 购电量/kWh & 充电量/kWh & 放电量/kWh & 最大同时动作/kW \\",
            r"    \midrule",
            *rows,
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    PAPER_TABLE_PATH.write_text(content, encoding="utf-8")


def _write_audit(
    comparison: pd.DataFrame,
    gap_yuan: float,
    exact_on_instance: bool,
    schedule_differences: dict[str, object],
    protected_before: dict[str, str],
    protected_after: dict[str, str],
) -> None:
    table = comparison[
        [
            "formulation",
            "solver_status",
            "objective_cost_yuan",
            "total_grid_purchase_kwh",
            "total_charge_kwh",
            "total_discharge_kwh",
            "total_curtailment_kwh",
            "minimum_soc_energy_kwh",
            "maximum_soc_energy_kwh",
            "maximum_simultaneous_charge_discharge_kw",
            "maximum_power_balance_residual_kw",
            "maximum_soc_recurrence_residual_kwh",
            "solver_runtime_seconds",
        ]
    ].to_markdown(index=False, floatfmt=".10g")
    if exact_on_instance:
        conclusion = (
            "Q1 的 LP 松弛在该算例上取得零松弛间隙，且最优解自然满足充放电互斥。"
            "这一结论仅针对附件1给定的确定性单日算例，不外推至 Q2、Q3 或 Q4。"
        )
    else:
        conclusion = (
            "该算例的 LP 松弛未同时满足零松弛间隙与自然充放电互斥条件，"
            "因此不能将 LP 解表述为正式 MILP 的精确替代。"
        )
    content = f"""# Q1 LP 松弛验证审计

## 实验边界

正式 MILP 直接调用 `src/problem1/model.py`；LP 松弛在独立模块中复现同一目标函数、
144 个时段、功率平衡、SOC 递推、功率边界、SOC 首末条件、弃光边界和非负购电约束，
只删除二元充放电模式变量及其互斥整数约束。未修改正式模型与 `result1.xlsx`。

## 同口径结果

{table}

松弛间隙定义为 `MILP objective - LP objective`，本次为
`{gap_yuan:.12g}` 元；零间隙容差为 `{OBJECTIVE_GAP_TOLERANCE_YUAN:g}` 元，
自然互斥容差为 `{SIMULTANEOUS_ACTION_TOLERANCE_KW:g}` kW。

## 调度差异

```json
{json.dumps(schedule_differences, ensure_ascii=False, indent=2)}
```

## 文件完整性

- 受保护文件运行前后哈希一致：`{protected_before == protected_after}`
- 受保护文件：{', '.join(f'`{name}`' for name in protected_before)}

## 审计结论

{conclusion}
"""
    AUDIT_PATH.write_text(content, encoding="utf-8")


def run_analysis() -> dict[str, object]:
    """Solve both formulations, verify invariants, and write audited outputs."""
    data = pd.read_csv(INPUT_PATH)
    parameters = DispatchParameters()
    protected_before = _protected_hashes()

    milp = solve_deterministic_dispatch(data, parameters)
    validate_dispatch(milp, parameters, tolerance=NUMERICAL_TOLERANCE)
    lp = solve_lp_relaxation(data, parameters)
    rows = []
    for formulation, result in (("MILP", milp), ("LP relaxation", lp)):
        rows.append({"formulation": formulation, **solution_metrics(result, parameters)})
    comparison = pd.DataFrame(rows)
    gap_yuan = milp.objective_yuan - lp.objective_yuan
    if gap_yuan < -OBJECTIVE_GAP_TOLERANCE_YUAN:
        raise AssertionError("LP relaxation objective cannot exceed the MILP objective")
    lp_simultaneous = float(
        comparison.loc[
            comparison["formulation"].eq("LP relaxation"),
            "maximum_simultaneous_charge_discharge_kw",
        ].iloc[0]
    )
    exact_on_instance = bool(
        abs(gap_yuan) <= OBJECTIVE_GAP_TOLERANCE_YUAN
        and lp_simultaneous <= SIMULTANEOUS_ACTION_TOLERANCE_KW
    )
    schedule_differences = _schedule_differences(milp, lp)

    protected_after = _protected_hashes()
    if protected_before != protected_after:
        raise AssertionError("Formal Q1 input/model/result changed during LP audit")

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(COMPARISON_PATH, index=False, float_format="%.12g")
    _write_paper_table(comparison)
    _write_audit(
        comparison,
        gap_yuan,
        exact_on_instance,
        schedule_differences,
        protected_before,
        protected_after,
    )
    result = {
        "milp_objective_yuan": milp.objective_yuan,
        "lp_objective_yuan": lp.objective_yuan,
        "relaxation_gap_yuan": gap_yuan,
        "lp_maximum_simultaneous_charge_discharge_kw": lp_simultaneous,
        "exact_on_this_instance": exact_on_instance,
        "schedule_differences": schedule_differences,
        "protected_hashes_unchanged": protected_before == protected_after,
        "outputs": [str(COMPARISON_PATH), str(PAPER_TABLE_PATH), str(AUDIT_PATH)],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run_analysis()
