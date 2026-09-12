"""Perfect-information lower bound and day-level Q3 value analysis.

This module is deliberately independent of the formal Q1--Q3 runners.  It
reads their frozen outputs, solves an information-relaxed full-horizon LP, and
writes analysis-only artifacts.  It never edits official result workbooks.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog
from scipy.stats import wilcoxon

from src.common.paths import PROJECT_ROOT, RESULTS_DIR


DT_HOURS = 1.0 / 6.0
ETA_C = 0.9
ETA_D = 0.9
MIN_ENERGY_KWH = 1_200.0
MAX_ENERGY_KWH = 10_800.0
MAX_POWER_KW = 5_000.0
INITIAL_ENERGY_KWH = 6_000.0
FORMAL_START = "2025-02-01"
FORMAL_END = "2025-12-31"
SIMULTANEOUS_TOLERANCE_KW = 1e-6
BALANCE_TOLERANCE_KW = 2e-5
SOC_TOLERANCE_KWH = 2e-5
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 2026

FORECAST_PATH = RESULTS_DIR / "tables" / "forecasting" / "q2_forecast_predictions.csv"
PRICE_PATH = PROJECT_ROOT / "data" / "processed" / "C题" / "problem1_day.csv"
Q2_SUMMARY_PATH = RESULTS_DIR / "problem2" / "tables" / "table_p2_summary.json"
Q3_DAILY_PATH = RESULTS_DIR / "problem3" / "tables" / "q3_daily_economic_results.csv"
Q3_SCHEDULE_PATH = RESULTS_DIR / "problem3" / "tables" / "q3_schedule_comparison.csv"
OUTPUT_DIR = RESULTS_DIR / "analysis"
LOWER_BOUND_PATH = OUTPUT_DIR / "perfect_information_lower_bound.csv"
DAILY_VALUE_PATH = OUTPUT_DIR / "q3_daily_value_significance.csv"
SUMMARY_PATH = OUTPUT_DIR / "theory_strengthening_summary.json"
AUDIT_PATH = OUTPUT_DIR / "PERFECT_INFORMATION_AUDIT.md"
PAPER_GENERATED_DIR = PROJECT_ROOT / "paper" / "contents" / "generated"
INFORMATION_BOUND_TEX_PATH = PAPER_GENERATED_DIR / "information_bound.tex"
Q3_DAILY_VALUE_TEX_PATH = PAPER_GENERATED_DIR / "q3_daily_value.tex"

PROTECTED_PATHS = (
    PROJECT_ROOT / "src" / "problem1" / "model.py",
    PROJECT_ROOT / "src" / "problem2" / "model.py",
    PROJECT_ROOT / "src" / "problem2" / "run.py",
    PROJECT_ROOT / "src" / "problem3" / "rolling_model.py",
    PROJECT_ROOT / "src" / "problem3" / "rolling_dispatch.py",
    RESULTS_DIR / "problem1" / "result1.xlsx",
    RESULTS_DIR / "problem2" / "result2.xlsx",
    RESULTS_DIR / "problem3" / "result3.xlsx",
)


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one immutable input."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected_hashes() -> dict[str, str]:
    """Hash formal model and result artifacts without modifying them."""

    return {str(path.relative_to(PROJECT_ROOT)): sha256(path) for path in PROTECTED_PATHS}


def load_full_horizon_inputs() -> pd.DataFrame:
    """Load actual load/PV and the fixed Q2 tariff on the formal 334-day grid."""

    actual = pd.read_csv(
        FORECAST_PATH,
        usecols=["operating_date", "datetime", "actual_load", "actual_generation"],
        parse_dates=["operating_date", "datetime"],
    ).sort_values("datetime", kind="stable")
    mask = actual["operating_date"].between(FORMAL_START, FORMAL_END)
    actual = actual.loc[mask].reset_index(drop=True)
    expected_rows = 334 * 144
    if len(actual) != expected_rows:
        raise AssertionError(f"formal actual series must contain {expected_rows} rows")
    if actual["datetime"].duplicated().any():
        raise AssertionError("formal actual timestamps are not unique")
    expected_time = pd.date_range(
        "2025-02-01 00:10:00", "2026-01-01 00:00:00", freq="10min"
    )
    if not np.array_equal(actual["datetime"].to_numpy(), expected_time.to_numpy()):
        raise AssertionError("formal actual series is not a continuous ten-minute grid")
    daily_counts = actual.groupby("operating_date", sort=True).size()
    if not daily_counts.eq(144).all() or len(daily_counts) != 334:
        raise AssertionError("formal actual series must contain 144 intervals per day")

    prices = pd.read_csv(PRICE_PATH).sort_values("slot", kind="stable")
    if prices["slot"].tolist() != list(range(1, 145)):
        raise AssertionError("fixed Q2 price profile must contain slots 1--144")
    price = prices["price_yuan_per_kwh"].to_numpy(dtype=float)
    actual["price_yuan_per_kwh"] = np.tile(price, 334)
    values = actual[["actual_load", "actual_generation", "price_yuan_per_kwh"]].to_numpy(float)
    if not np.isfinite(values).all():
        raise AssertionError("perfect-information inputs contain non-finite values")
    if (actual["actual_generation"] < 0).any() or (actual["price_yuan_per_kwh"] < 0).any():
        raise AssertionError("perfect-information inputs violate PV/price nonnegativity")
    return actual


def _variable_slices(periods: int) -> dict[str, slice]:
    return {
        "grid": slice(0, periods),
        "charge": slice(periods, 2 * periods),
        "discharge": slice(2 * periods, 3 * periods),
        "spill": slice(3 * periods, 4 * periods),
        "energy": slice(4 * periods, 5 * periods + 1),
    }


def build_perfect_information_lp(
    load_kw: np.ndarray,
    pv_kw: np.ndarray,
    price_yuan_per_kwh: np.ndarray,
) -> tuple[np.ndarray, sparse.csr_matrix, np.ndarray, list[tuple[float, float | None]], dict[str, slice]]:
    """Build the full-horizon physical LP with only mutual exclusion relaxed."""

    load = np.asarray(load_kw, dtype=float)
    pv = np.asarray(pv_kw, dtype=float)
    price = np.asarray(price_yuan_per_kwh, dtype=float)
    if load.ndim != 1 or load.shape != pv.shape or load.shape != price.shape:
        raise ValueError("load, PV and price must be aligned one-dimensional arrays")
    if not np.isfinite(np.column_stack([load, pv, price])).all():
        raise ValueError("LP inputs must be finite")
    if (pv < 0).any() or (price < 0).any():
        raise ValueError("PV and price must be nonnegative")

    periods = len(load)
    sl = _variable_slices(periods)
    variable_count = 5 * periods + 1
    objective = np.zeros(variable_count)
    objective[sl["grid"]] = price * DT_HOURS

    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    rhs = np.zeros(2 * periods + 1)
    for t in range(periods):
        # g + PV + d = load + c + spill
        rows.extend([t] * 4)
        columns.extend([t, periods + t, 2 * periods + t, 3 * periods + t])
        data.extend([1.0, -1.0, 1.0, -1.0])
        rhs[t] = load[t] - pv[t]

        row = periods + t
        rows.extend([row] * 4)
        columns.extend([periods + t, 2 * periods + t, 4 * periods + t, 4 * periods + t + 1])
        data.extend([-ETA_C * DT_HOURS, DT_HOURS / ETA_D, -1.0, 1.0])
    rows.append(2 * periods)
    columns.append(4 * periods)
    data.append(1.0)
    rhs[2 * periods] = INITIAL_ENERGY_KWH
    equality = sparse.coo_matrix(
        (data, (rows, columns)), shape=(2 * periods + 1, variable_count)
    ).tocsr()

    bounds: list[tuple[float, float | None]] = []
    bounds.extend([(0.0, None)] * periods)
    bounds.extend([(0.0, MAX_POWER_KW)] * periods)
    bounds.extend([(0.0, MAX_POWER_KW)] * periods)
    bounds.extend([(0.0, float(value)) for value in pv])
    bounds.extend([(MIN_ENERGY_KWH, MAX_ENERGY_KWH)] * (periods + 1))
    return objective, equality, rhs, bounds, sl


def audit_lp_solution(frame: pd.DataFrame) -> dict[str, float | int | bool]:
    """Audit physical equations and whether the relaxed optimum is MILP-feasible."""

    balance = (
        frame["grid_kw"]
        + frame["pv_kw"]
        + frame["discharge_kw"]
        - frame["load_kw"]
        - frame["charge_kw"]
        - frame["spill_kw"]
    )
    soc = (
        frame["energy_end_kwh"]
        - frame["energy_start_kwh"]
        - ETA_C * frame["charge_kw"] * DT_HOURS
        + frame["discharge_kw"] * DT_HOURS / ETA_D
    )
    simultaneous = np.minimum(frame["charge_kw"], frame["discharge_kw"])
    return {
        "maximum_power_balance_residual_kw": float(balance.abs().max()),
        "maximum_soc_recurrence_residual_kwh": float(soc.abs().max()),
        "minimum_energy_kwh": float(frame[["energy_start_kwh", "energy_end_kwh"]].min().min()),
        "maximum_energy_kwh": float(frame[["energy_start_kwh", "energy_end_kwh"]].max().max()),
        "final_energy_kwh": float(frame["energy_end_kwh"].iloc[-1]),
        "maximum_simultaneous_charge_discharge_kw": (
            0.0
            if abs(float(simultaneous.max())) <= SIMULTANEOUS_TOLERANCE_KW
            else float(simultaneous.max())
        ),
        "simultaneous_charge_discharge_count": int((simultaneous > SIMULTANEOUS_TOLERANCE_KW).sum()),
        "grid_nonnegative": bool((frame["grid_kw"] >= -BALANCE_TOLERANCE_KW).all()),
        "spill_bounds": bool(
            (frame["spill_kw"] >= -BALANCE_TOLERANCE_KW).all()
            and (frame["spill_kw"] <= frame["pv_kw"] + BALANCE_TOLERANCE_KW).all()
        ),
        "soc_bounds": bool(
            (frame["energy_start_kwh"] >= MIN_ENERGY_KWH - SOC_TOLERANCE_KWH).all()
            and (frame["energy_end_kwh"] <= MAX_ENERGY_KWH + SOC_TOLERANCE_KWH).all()
        ),
        "cross_day_continuity": bool(
            np.max(
                np.abs(
                    frame["energy_end_kwh"].to_numpy()[:-1]
                    - frame["energy_start_kwh"].to_numpy()[1:]
                )
            )
            <= SOC_TOLERANCE_KWH
        ),
    }


def solve_perfect_information_lp(actual: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Solve and audit the 334-day clairvoyant LP benchmark."""

    load = actual["actual_load"].to_numpy(float)
    pv = actual["actual_generation"].to_numpy(float)
    price = actual["price_yuan_per_kwh"].to_numpy(float)
    objective, equality, rhs, bounds, sl = build_perfect_information_lp(load, pv, price)
    started = perf_counter()
    solved = linprog(
        objective,
        A_eq=equality,
        b_eq=rhs,
        bounds=bounds,
        method="highs",
        options={"presolve": True},
    )
    runtime = perf_counter() - started
    if not solved.success:
        raise RuntimeError(f"perfect-information LP failed: {solved.status}: {solved.message}")
    periods = len(actual)
    vector = solved.x
    frame = actual.copy()
    frame["load_kw"] = load
    frame["pv_kw"] = pv
    frame["grid_kw"] = vector[sl["grid"]]
    frame["charge_kw"] = vector[sl["charge"]]
    frame["discharge_kw"] = vector[sl["discharge"]]
    frame["spill_kw"] = vector[sl["spill"]]
    energy = vector[sl["energy"]]
    frame["energy_start_kwh"] = energy[:-1]
    frame["energy_end_kwh"] = energy[1:]
    audit = audit_lp_solution(frame)
    physical_pass = bool(
        audit["maximum_power_balance_residual_kw"] <= BALANCE_TOLERANCE_KW
        and audit["maximum_soc_recurrence_residual_kwh"] <= SOC_TOLERANCE_KWH
        and audit["grid_nonnegative"]
        and audit["spill_bounds"]
        and audit["soc_bounds"]
        and audit["cross_day_continuity"]
    )
    exact = physical_pass and audit["simultaneous_charge_discharge_count"] == 0
    summary: dict[str, object] = {
        "benchmark": "full-horizon perfect-information LP relaxation",
        "deployable_strategy": False,
        "formal_start": FORMAL_START,
        "formal_end": FORMAL_END,
        "formal_days": 334,
        "intervals": periods,
        "dt_hours": DT_HOURS,
        "solver": "SciPy HiGHS linprog",
        "solver_status": "Optimal",
        "solver_runtime_seconds": runtime,
        "objective_cost_yuan": float(solved.fun),
        "total_grid_purchase_kwh": float(frame["grid_kw"].sum() * DT_HOURS),
        "total_charge_kwh": float(frame["charge_kw"].sum() * DT_HOURS),
        "total_discharge_kwh": float(frame["discharge_kw"].sum() * DT_HOURS),
        "total_curtailment_kwh": float(frame["spill_kw"].sum() * DT_HOURS),
        "initial_energy_kwh": float(energy[0]),
        "terminal_constraint": "none beyond physical SOC bounds",
        "horizon_end_optimism": True,
        "physical_audit_pass": physical_pass,
        "exact_physical_perfect_information_optimum": exact,
        **audit,
    }
    return frame, summary


def q3_daily_value_analysis() -> dict[str, object]:
    """Compute paired daily S0-minus-S2 savings and conservative inference."""

    daily = pd.read_csv(Q3_DAILY_PATH)
    main = daily.loc[daily["main_settlement"].astype(bool)].copy()
    pivot = main.pivot(index="date", columns="schedule", values="realized_total_cost_yuan")
    if set(["S0", "S2"]).difference(pivot.columns) or len(pivot) != 334:
        raise AssertionError("Q3 daily main-settlement table lacks 334 paired S0/S2 days")
    paired = (pivot["S0"] - pivot["S2"]).dropna()
    if len(paired) != 334:
        raise AssertionError("Q3 S0/S2 comparison is not complete and paired")
    test = wilcoxon(paired, alternative="two-sided", zero_method="wilcox", method="auto")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = paired.to_numpy(float)
    bootstrap = np.empty(BOOTSTRAP_RESAMPLES)
    for index in range(BOOTSTRAP_RESAMPLES):
        bootstrap[index] = rng.choice(values, size=len(values), replace=True).mean()
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "comparison": "S0 minus S2 realized daily cost",
        "paired_unit": "operating day",
        "n_days": int(len(values)),
        "annual_saving_yuan": float(values.sum()),
        "mean_daily_saving_yuan": float(values.mean()),
        "median_daily_saving_yuan": float(np.median(values)),
        "first_quartile_daily_saving_yuan": float(np.quantile(values, 0.25)),
        "third_quartile_daily_saving_yuan": float(np.quantile(values, 0.75)),
        "interquartile_range_yuan": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
        "positive_saving_day_proportion": float(np.mean(values > 0.0)),
        "zero_saving_day_count": int(np.sum(values == 0.0)),
        "wilcoxon_alternative": "two-sided",
        "wilcoxon_zero_method": "wilcox",
        "wilcoxon_method": "auto",
        "wilcoxon_statistic": float(test.statistic),
        "wilcoxon_p_value": float(test.pvalue),
        "bootstrap_statistic": "mean paired daily saving",
        "bootstrap_method": "percentile",
        "bootstrap_resampling_unit": "operating day",
        "bootstrap_confidence_level": 0.95,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_95_lower_yuan": float(lower),
        "bootstrap_95_upper_yuan": float(upper),
    }


def write_outputs(lower_bound: dict[str, object], daily: dict[str, object], hashes: dict[str, str]) -> None:
    """Write compact analysis-only tables, metadata, and audit notes."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([lower_bound]).to_csv(LOWER_BOUND_PATH, index=False)
    pd.DataFrame([daily]).to_csv(DAILY_VALUE_PATH, index=False)
    q2 = json.loads(Q2_SUMMARY_PATH.read_text())
    q3 = pd.read_csv(Q3_SCHEDULE_PATH).set_index("schedule")
    c_q2 = float(q2["realized_total_cost_yuan"])
    c_q3 = float(q3.loc["S2", "realized_total_cost_yuan"])
    c_lb = float(lower_bound["objective_cost_yuan"])
    if not c_lb <= c_q3 <= c_q2:
        raise AssertionError("cost ordering C_LB <= C_Q3 <= C_Q2 failed")
    denominator = c_q2 - c_lb
    if denominator <= 0:
        raise AssertionError("perfect-information gap denominator is nonpositive")
    combined = {
        "q1_exact_relaxation_status": "CONDITIONAL",
        "q2_asymmetric_local_regret": "p_t*(e_t)_+ + 4*p_t*(-e_t)_+",
        "q3_theoretical_voi_scope": "ex-ante optimum under nested information and ignorable added information",
        "q3_realized_voi_scope": "finite-sample forecast-driven realized operational value",
        "perfect_information": lower_bound,
        "q3_daily_value": daily,
        "cost_comparison": {
            "q2_realized_total_cost_yuan": c_q2,
            "q3_s2_realized_total_cost_yuan": c_q3,
            "perfect_information_cost_yuan": c_lb,
            "q2_information_gap_yuan": c_q2 - c_lb,
            "q3_remaining_gap_yuan": c_q3 - c_lb,
            "q2_to_q3_cost_reduction_yuan": c_q2 - c_q3,
            "observable_cost_gap_closure_ratio": (c_q2 - c_q3) / denominator,
        },
        "protected_hashes": hashes,
        "protected_hashes_unchanged": True,
        "formal_result_files_modified": False,
    }
    SUMMARY_PATH.write_text(json.dumps(combined, ensure_ascii=False, indent=2) + "\n")
    PAPER_GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    INFORMATION_BOUND_TEX_PATH.write_text(
        "\\begin{table}[htbp]\n"
        "\\centering\\small\n"
        "\\caption{因果策略与离线全信息基准的同口径费用比较}"
        "\\label{tab:information-bound}\n"
        "\\setlength{\\tabcolsep}{5pt}\n"
        "\\begin{tabular}{lrr}\n\\toprule\n"
        "方案 & 费用/元 & 相对离线基准差距/元 \\\\\n"
        "\\midrule\n"
        f"离线全信息基准 & {c_lb:.6f} & 0 \\\\\n"
        f"问题三$S_2$ & {c_q3:.6f} & {c_q3-c_lb:.6f} \\\\\n"
        f"问题二因果日前策略 & {c_q2:.6f} & {c_q2-c_lb:.6f} \\\\\n"
        "\\bottomrule\n\\end{tabular}\n"
        "\\par\\vspace{3pt}\\begin{minipage}{0.94\\textwidth}\\footnotesize "
        "离线基准使用未来真实负荷和光伏，不能实施；其年末电量仅受物理边界约束，"
        "故具有有限视界乐观性。\\end{minipage}\n"
        "\\end{table}\n"
    )
    Q3_DAILY_VALUE_TEX_PATH.write_text(
        "\\begin{table}[htbp]\n"
        "\\centering\\small\n"
        "\\caption{$S_2$相对$S_0$的日级配对经济收益}"
        "\\label{tab:q3-daily-value}\n"
        "\\setlength{\\tabcolsep}{5pt}\n"
        "\\begin{tabular}{lr}\n\\toprule\n"
        "统计量 & 数值 \\\\\n\\midrule\n"
        f"配对运营日数 & {daily['n_days']} \\\\\n"
        f"日节省均值/元 & {daily['mean_daily_saving_yuan']:.3f} \\\\\n"
        f"日节省中位数/元 & {daily['median_daily_saving_yuan']:.3f} \\\\\n"
        f"四分位区间/元 & [{daily['first_quartile_daily_saving_yuan']:.3f}, {daily['third_quartile_daily_saving_yuan']:.3f}] \\\\\n"
        f"正节省日期占比 & {100*daily['positive_saving_day_proportion']:.2f}\\% \\\\\n"
        f"Wilcoxon统计量 & {daily['wilcoxon_statistic']:.0f} \\\\\n"
        f"双侧$p$值 & ${daily['wilcoxon_p_value']/10**np.floor(np.log10(daily['wilcoxon_p_value'])):.3f}\\times10^{{{int(np.floor(np.log10(daily['wilcoxon_p_value'])))}}}$ \\\\\n"
        f"日均节省95\\% bootstrap区间/元 & [{daily['bootstrap_95_lower_yuan']:.3f}, {daily['bootstrap_95_upper_yuan']:.3f}] \\\\\n"
        "\\bottomrule\n\\end{tabular}\n"
        "\\par\\vspace{3pt}\\begin{minipage}{0.94\\textwidth}\\footnotesize "
        "统计单位为运营日；双侧Wilcoxon符号秩检验采用\\texttt{wilcox}零差处理。"
        "Bootstrap对334个配对日差有放回重采样10000次，随机种子为2026。\\end{minipage}\n"
        "\\end{table}\n"
    )
    exact_label = (
        "exact physical perfect-information optimum for this instance"
        if lower_bound["exact_physical_perfect_information_optimum"]
        else "information-relaxed LP lower bound only"
    )
    AUDIT_PATH.write_text(
        "# Perfect-information offline benchmark audit\n\n"
        "## Positioning\n\n"
        "This is a non-deployable full-horizon benchmark using future actual load and PV. "
        "It does not replace the causal Q2/Q3 strategies or their official workbooks.\n\n"
        "## Formulation\n\n"
        f"- Period: {FORMAL_START} through {FORMAL_END}; 334 days and {lower_bound['intervals']} ten-minute intervals.\n"
        "- Fixed Q2/Q3 tariff; no Attachment-4 dynamic price.\n"
        "- One continuous SOC trajectory, initial energy 6000 kWh, no daily reset.\n"
        "- No terminal equality; the resulting finite-horizon optimism is explicitly retained.\n"
        "- Grid purchase is nonnegative; export is absent; PV curtailment is bounded by actual PV.\n"
        "- Only charge/discharge mutual exclusion is relaxed.\n\n"
        "## Result and audit\n\n"
        f"- Solver status: {lower_bound['solver_status']}.\n"
        f"- Objective: {lower_bound['objective_cost_yuan']:.6f} yuan.\n"
        f"- Simultaneous charge/discharge count: {lower_bound['simultaneous_charge_discharge_count']}.\n"
        f"- Maximum power-balance residual: {lower_bound['maximum_power_balance_residual_kw']:.3e} kW.\n"
        f"- Maximum SOC-recurrence residual: {lower_bound['maximum_soc_recurrence_residual_kwh']:.3e} kWh.\n"
        f"- Classification: {exact_label}.\n\n"
        "## Boundary\n\n"
        "The benchmark has noncausal access to future actuals and is not an implementable policy. "
        "Its unconstrained terminal SOC can make the lower bound optimistic near the end of the horizon.\n"
    )


def main() -> None:
    before = protected_hashes()
    actual = load_full_horizon_inputs()
    _, lower_bound = solve_perfect_information_lp(actual)
    daily = q3_daily_value_analysis()
    after = protected_hashes()
    if before != after:
        raise AssertionError("a protected formal model or result file changed during analysis")
    write_outputs(lower_bound, daily, before)
    print(json.dumps({"perfect_information": lower_bound, "q3_daily_value": daily}, indent=2))


if __name__ == "__main__":
    main()
