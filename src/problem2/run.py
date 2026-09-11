"""Chronological runner and official result2 exporter for Problem 2."""

from __future__ import annotations

import argparse
from copy import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import openpyxl
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.paths import PROJECT_ROOT, problem_results_dir
from src.problem2.evaluate import build_daily_summary, validate_q2_day
from src.problem2.forecast_interface import (
    FINAL_AUDIT_PATH,
    FORECAST_DIR,
    FORMAL_FORECAST_PATH,
    SCENARIO_PATH,
    audit_q2_handoff_integrity,
    get_q2_day_inputs,
)
from src.problem2.model import Q2DayResult, Q2DispatchParameters, solve_expected_cost_dispatch


FORMAL_START = "2025-02-01"
FORMAL_END = "2025-12-31"
OFFICIAL_INITIAL_ENERGY_KWH = 6_000.0
OUTPUT_DIR = problem_results_dir(2)
TABLE_DIR = OUTPUT_DIR / "tables"
TEMPLATE_PATH = PROJECT_ROOT / "data" / "raw" / "C题" / "附件" / "附件5" / "result2.xlsx"
RESULT2_PATH = OUTPUT_DIR / "result2.xlsx"
PROTECTED_FORECAST_PATHS = (
    FORMAL_FORECAST_PATH,
    SCENARIO_PATH,
    FORECAST_DIR / "q2_scenario_manifest.csv",
    FORECAST_DIR / "q2_forecast_metrics.csv",
    FORECAST_DIR / "q2_model_decision.json",
    FORECAST_DIR / "q2_leakage_audit.json",
    FINAL_AUDIT_PATH,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _capture_row_style(sheet, source_row: int, column_count: int) -> tuple[object, list[tuple[object, str, object, object]]]:
    cells = []
    for column in range(1, column_count + 1):
        source = sheet.cell(source_row, column)
        cells.append(
            (
                copy(source._style),
                source.number_format,
                copy(source.alignment),
                copy(source.protection),
            )
        )
    return sheet.row_dimensions[source_row].height, cells


def _apply_row_style(sheet, style, target_row: int) -> None:
    height, cells = style
    sheet.row_dimensions[target_row].height = height
    for column, (cell_style, number_format, alignment, protection) in enumerate(cells, start=1):
        target = sheet.cell(target_row, column)
        target._style = copy(cell_style)
        target.number_format = number_format
        target.alignment = copy(alignment)
        target.protection = copy(protection)


def export_result2(
    daily: pd.DataFrame,
    intervals: pd.DataFrame,
    output_path: Path = RESULT2_PATH,
    template_path: Path = TEMPLATE_PATH,
) -> Path:
    """Fill the three-sheet official result2 workbook without editing its source."""

    dates = pd.to_datetime(daily["date"]).dt.normalize()
    if len(daily) != 334 or dates.iloc[0] != pd.Timestamp(FORMAL_START):
        raise ValueError("official result2 export requires all 334 formal days")
    if dates.iloc[-1] != pd.Timestamp(FORMAL_END):
        raise ValueError("official result2 export must end on 2025-12-31")
    if len(intervals) != 334 * 144:
        raise ValueError("official result2 export requires 144 intervals per formal day")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(template_path, output_path)
    workbook = openpyxl.load_workbook(output_path)
    expected_sheets = ["计划购电量", "充放电量", "紧急购电量"]
    if workbook.sheetnames != expected_sheets:
        raise ValueError(f"unexpected result2 sheets: {workbook.sheetnames}")

    purchase_sheet = workbook["计划购电量"]
    if purchase_sheet.max_row != 335 or purchase_sheet.max_column != 147:
        raise ValueError("official planned-purchase sheet shape changed")
    for day_index, (_, row) in enumerate(daily.iterrows(), start=2):
        day_frame = intervals.loc[intervals["date"] == row["date"]].sort_values("slot")
        if len(day_frame) != 144:
            raise AssertionError(f"{row['date']} does not contain 144 dispatch intervals")
        for slot, value in enumerate(day_frame["planned_grid_kwh"], start=2):
            cell = purchase_sheet.cell(day_index, slot, float(value))
            cell.number_format = "0.000000"
        purchase_sheet.cell(day_index, 146, float(row["planned_purchase_energy_kwh"])).number_format = "0.000000"
        purchase_sheet.cell(day_index, 147, float(row["planned_purchase_cost_yuan"])).number_format = "0.000000"

    storage_sheet = workbook["充放电量"]
    storage_styles = [_capture_row_style(storage_sheet, 2 + block, 6) for block in range(6)]
    storage_sheet.delete_rows(2, storage_sheet.max_row - 1)
    time_blocks = ("0:00-4:00", "4:00-8:00", "8:00-12:00", "12:00-16:00", "16:00-20:00", "20:00-24:00")
    target_row = 2
    for _, day_row in daily.iterrows():
        day_frame = intervals.loc[intervals["date"] == day_row["date"]].sort_values("slot")
        for block, time_label in enumerate(time_blocks):
            _apply_row_style(storage_sheet, storage_styles[block], target_row)
            block_frame = day_frame.iloc[block * 24 : (block + 1) * 24]
            storage_sheet.cell(target_row, 1, pd.Timestamp(day_row["date"]).to_pydatetime() if block == 0 else None)
            storage_sheet.cell(target_row, 2, time_label)
            storage_sheet.cell(target_row, 3, float(block_frame["charge_kwh"].sum())).number_format = "0.000000"
            storage_sheet.cell(target_row, 4, float(block_frame["discharge_kwh"].sum())).number_format = "0.000000"
            if block == 0:
                storage_sheet.cell(target_row, 5, "00:00")
                storage_sheet.cell(target_row, 6, float(day_row["initial_energy_kwh"])).number_format = "0.000000"
            elif block == 1:
                storage_sheet.cell(target_row, 5, "24:00")
                storage_sheet.cell(target_row, 6, float(day_row["final_energy_kwh"])).number_format = "0.000000"
            else:
                storage_sheet.cell(target_row, 5, None)
                storage_sheet.cell(target_row, 6, None)
            target_row += 1

    emergency_sheet = workbook["紧急购电量"]
    emergency_styles = [_capture_row_style(emergency_sheet, 2 + row, 3) for row in range(3)]
    emergency_sheet.delete_rows(2, emergency_sheet.max_row - 1)
    target_row = 2
    for _, day_row in daily.iterrows():
        day_frame = intervals.loc[intervals["date"] == day_row["date"]].sort_values("slot")
        positive = day_frame.loc[day_frame["expected_emergency_kwh"] > 1e-9]
        if positive.empty:
            positive = day_frame.iloc[[0]].assign(expected_emergency_kwh=0.0)
        for item_index, (_, interval_row) in enumerate(positive.iterrows()):
            _apply_row_style(emergency_sheet, emergency_styles[min(item_index, 2)], target_row)
            emergency_sheet.cell(
                target_row,
                1,
                pd.Timestamp(day_row["date"]).to_pydatetime() if item_index == 0 else None,
            )
            emergency_sheet.cell(target_row, 2, purchase_sheet.cell(1, int(interval_row["slot"]) + 1).value)
            emergency_sheet.cell(target_row, 3, float(interval_row["expected_emergency_kwh"])).number_format = "0.000000"
            target_row += 1

    workbook.calculation.fullCalcOnLoad = True
    workbook.calculation.forceFullCalc = True
    workbook.save(output_path)

    check = openpyxl.load_workbook(output_path, read_only=True, data_only=True)
    if check.sheetnames != expected_sheets:
        raise AssertionError("result2 sheet structure changed during export")
    if check["计划购电量"].max_row != 335 or check["计划购电量"].max_column != 147:
        raise AssertionError("result2 planned-purchase sheet shape is incorrect")
    return output_path


def run_chronological(
    dates: pd.DatetimeIndex,
    *,
    initial_energy_kwh: float = OFFICIAL_INITIAL_ENERGY_KWH,
    parameters: Q2DispatchParameters | None = None,
    progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Solve dates in order, carrying each optimal final SOC into the next day."""

    parameters = parameters or Q2DispatchParameters()
    carried_energy = float(initial_energy_kwh)
    summaries: list[dict[str, object]] = []
    interval_frames: list[pd.DataFrame] = []
    emergency_days: list[np.ndarray] = []
    spill_days: list[np.ndarray] = []
    source_days: list[np.ndarray] = []
    for day_index, target in enumerate(dates, start=1):
        inputs = get_q2_day_inputs(target, initial_energy=carried_energy)
        result: Q2DayResult = solve_expected_cost_dispatch(inputs, parameters)
        validation = validate_q2_day(inputs, result, parameters)
        summary = build_daily_summary(result, validation, inputs)
        if summaries and abs(float(summary["initial_energy_kwh"]) - float(summaries[-1]["final_energy_kwh"])) > 2e-3:
            raise AssertionError(f"cross-day SOC discontinuity before {target.date()}")
        summaries.append(summary)
        interval_frames.append(result.dispatch)
        emergency_days.append(result.emergency_kw)
        spill_days.append(result.spill_kw)
        source_days.append(result.scenario_source_dates)
        carried_energy = result.final_energy_kwh
        if progress and (day_index == 1 or day_index % 10 == 0 or day_index == len(dates)):
            print(
                f"Q2 progress: {day_index}/{len(dates)} days; "
                f"{target.date()} {result.status}; runtime={result.runtime_seconds:.3f}s",
                flush=True,
            )
    return (
        pd.DataFrame(summaries),
        pd.concat(interval_frames, ignore_index=True),
        np.stack(emergency_days),
        np.stack(spill_days),
        np.stack(source_days),
    )


def _aggregate_summary(daily: pd.DataFrame, intervals: pd.DataFrame) -> dict[str, object]:
    return {
        "q2_formulation": "expected-cost stochastic MILP",
        "solver": str(daily["solver"].iloc[0]),
        "scenario_count": 50,
        "formal_days": int(len(daily)),
        "optimal_days": int((daily["status"] == "Optimal").sum()),
        "infeasible_days": int((daily["status"] != "Optimal").sum()),
        "planned_purchase_cost_yuan": float(daily["planned_purchase_cost_yuan"].sum()),
        "expected_emergency_cost_yuan": float(daily["expected_emergency_cost_yuan"].sum()),
        "expected_total_cost_yuan": float(daily["expected_total_cost_yuan"].sum()),
        "planned_purchase_energy_kwh": float(daily["planned_purchase_energy_kwh"].sum()),
        "expected_emergency_energy_kwh": float(daily["expected_emergency_energy_kwh"].sum()),
        "charge_energy_kwh": float(daily["charge_energy_kwh"].sum()),
        "discharge_energy_kwh": float(daily["discharge_energy_kwh"].sum()),
        "expected_spill_energy_kwh": float(daily["expected_spill_energy_kwh"].sum()),
        "minimum_energy_kwh": float(intervals[["storage_start_kwh", "storage_end_kwh"]].min().min()),
        "maximum_energy_kwh": float(intervals[["storage_start_kwh", "storage_end_kwh"]].max().max()),
        "final_battery_energy_kwh": float(daily["final_energy_kwh"].iloc[-1]),
        "maximum_power_balance_residual_kw": float(daily["maximum_power_balance_residual_kw"].max()),
        "maximum_soc_residual_kwh": float(daily["maximum_soc_residual_kwh"].max()),
        "total_solver_runtime_seconds": float(daily["runtime_seconds"].sum()),
        "daily_reset_to_6000": False,
        "terminal_treatment": "linear L1 penalty around carried initial energy",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representative", action="store_true", help="solve four representative days only")
    args = parser.parse_args()
    dates = (
        pd.to_datetime(["2025-02-01", "2025-04-15", "2025-07-15", "2025-10-15"])
        if args.representative
        else pd.date_range(FORMAL_START, FORMAL_END, freq="D")
    )
    protected_before = {str(path): _sha256(path) for path in PROTECTED_FORECAST_PATHS}
    daily, intervals, emergency, spill, source_dates = run_chronological(
        pd.DatetimeIndex(dates), progress=True
    )
    protected_after = {str(path): _sha256(path) for path in PROTECTED_FORECAST_PATHS}
    if protected_before != protected_after:
        raise AssertionError("a frozen Stage 2A forecast/scenario artifact changed")
    audit = audit_q2_handoff_integrity()
    if not all(
        bool(audit[key])
        for key in (
            "protected_artifact_hashes_unchanged",
            "q2_scenario_hash_unchanged",
            "raw_data_hashes_unchanged",
            "paired_scenarios",
            "leakage_audit",
        )
    ):
        raise AssertionError(f"Stage 2B handoff audit failed: {audit}")

    if args.representative:
        print(json.dumps(_aggregate_summary(daily, intervals), indent=2, ensure_ascii=False))
        return

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_csv(TABLE_DIR / "table_p2_daily_summary.csv", index=False, float_format="%.10g")
    intervals.to_csv(TABLE_DIR / "table_p2_dispatch.csv", index=False, float_format="%.10g")
    np.savez_compressed(
        TABLE_DIR / "table_p2_scenario_recourse.npz",
        target_dates=daily["date"].to_numpy(dtype="U10"),
        source_residual_dates=source_dates.astype("datetime64[D]"),
        emergency_kw=emergency,
        spill_kw=spill,
    )
    summary = _aggregate_summary(daily, intervals)
    (TABLE_DIR / "table_p2_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    export_result2(daily, intervals)
    print("Q2 expected-cost stochastic MILP complete.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"result2: {RESULT2_PATH}")


if __name__ == "__main__":
    main()
