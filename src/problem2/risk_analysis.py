"""Read-only final reporting for the committed Q2 CVaR sensitivity sweep."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.paths import PROJECT_ROOT, problem_figures_dir, problem_results_dir
from src.problem2.visualize import plot_cvar_risk_tradeoff


SWEEP_PATH = problem_results_dir(2) / "q2_risk_sweep.csv"
METADATA_PATH = problem_results_dir(2) / "q2_risk_metadata.json"
TABLE_DIR = problem_results_dir(2) / "tables"
EXPECTED_LAMBDAS = (0.0, 0.05, 0.10, 0.20, 0.50)
ALPHA = 0.90


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_final_cvar_sweep(path: Path = SWEEP_PATH) -> pd.DataFrame:
    """Load the saved sweep and enforce its reporting contract without rerunning it."""
    frame = pd.read_csv(path).sort_values("lambda", kind="stable").reset_index(drop=True)
    required = {
        "risk_method", "alpha", "lambda", "expected_operating_cost_yuan",
        "cvar_cost_yuan", "realized_emergency_energy_kwh", "realized_total_cost_yuan",
        "solver_success", "formal_days",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"final CVaR sweep is missing columns: {missing}")
    numeric = frame[list(required - {"risk_method"})].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("final CVaR sweep contains NaN or Inf")
    if frame["lambda"].tolist() != list(EXPECTED_LAMBDAS):
        raise AssertionError("final CVaR sweep does not contain the predeclared lambda grid")
    if not (frame["alpha"] == ALPHA).all() or not (frame["risk_method"] == "CVaR").all():
        raise AssertionError("final CVaR sweep has inconsistent risk semantics")
    if not ((frame["solver_success"] == 334) & (frame["formal_days"] == 334)).all():
        raise AssertionError("not every CVaR point contains 334/334 optimal days")
    return frame


def mild_risk_changes(frame: pd.DataFrame) -> dict[str, float]:
    """Return lambda=0.05 changes relative to lambda=0, in percent."""
    base = frame.loc[np.isclose(frame["lambda"], 0.0)].iloc[0]
    mild = frame.loc[np.isclose(frame["lambda"], 0.05)].iloc[0]
    return {
        "expected_operating_cost_change_pct": 100.0
        * (mild.expected_operating_cost_yuan - base.expected_operating_cost_yuan)
        / base.expected_operating_cost_yuan,
        "emergency_energy_reduction_pct": 100.0
        * (base.realized_emergency_energy_kwh - mild.realized_emergency_energy_kwh)
        / base.realized_emergency_energy_kwh,
        "emergency_cvar_reduction_pct": 100.0
        * (base.cvar_cost_yuan - mild.cvar_cost_yuan) / base.cvar_cost_yuan,
        "realized_total_cost_change_pct": 100.0
        * (mild.realized_total_cost_yuan - base.realized_total_cost_yuan)
        / base.realized_total_cost_yuan,
    }


def _write_latex_outputs(frame: pd.DataFrame, changes: dict[str, float]) -> tuple[Path, Path]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    number_path = TABLE_DIR / "q2_cvar_numbers.tex"
    number_path.write_text(
        "\n".join(
            [
                f"\\newcommand{{\\QTwoMildExpectedCostChange}}{{{changes['expected_operating_cost_change_pct']:.3f}}}",
                f"\\newcommand{{\\QTwoMildEmergencyEnergyReduction}}{{{changes['emergency_energy_reduction_pct']:.2f}}}",
                f"\\newcommand{{\\QTwoMildCvarReduction}}{{{changes['emergency_cvar_reduction_pct']:.2f}}}",
                f"\\newcommand{{\\QTwoMildRealizedCostReduction}}{{{-changes['realized_total_cost_change_pct']:.3f}}}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            f"    {float(row['lambda']):g} & {row['expected_operating_cost_yuan'] / 1e4:.3f} & "
            f"{row['cvar_cost_yuan'] / 1e4:.3f} & {row['realized_emergency_energy_kwh'] / 1e4:.3f} & "
            f"{row['realized_total_cost_yuan'] / 1e4:.3f} \\\\"
        )
    table_path = TABLE_DIR / "q2_cvar_sensitivity.tex"
    table_path.write_text(
        "\n".join(
            [
                r"\begin{table}[H]",
                r"  \centering",
                r"  \caption{Q2 的 CVaR 风险权重敏感性结果}",
                r"  \label{tab:p2-cvar-sweep}",
                r"  \small",
                r"  \setlength{\tabcolsep}{5pt}",
                r"  \begin{tabular}{crrrr}",
                r"    \toprule",
                r"    $\lambda_{\rm r}$ & 期望运行成本 & 紧急购电CVaR & 实际紧急电量$^{*}$ & 实际总费用 \\",
                r"    \midrule",
                *rows,
                r"    \bottomrule",
                r"  \end{tabular}",
                r"  \vspace{2pt}",
                r"  \begin{minipage}{0.96\textwidth}",
                r"    \footnotesize 注：费用单位为万元；$^{*}$实际紧急电量单位为万 kWh。CVaR 的场景损失仅为",
                r"    紧急购电费用，计划购电费不在 CVaR 内。风险权重代表事前偏好，实际回测结果不参与权重选择。",
                r"  \end{minipage}",
                r"\end{table}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return number_path, table_path


def finalize_risk_outputs() -> dict[str, object]:
    """Create only reporting artifacts from the committed, already-solved sweep."""
    frame = load_final_cvar_sweep()
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    if metadata["scenario_emergency_cost_definition"] != (
        "sum_t 5 * fixed_intraday_price_t * emergency_kw[s,t] * (1/6 h)"
    ):
        raise AssertionError("CVaR is not defined on emergency procurement cost only")
    if metadata["forecasting_modified"] or metadata["scenario_generation_modified"]:
        raise AssertionError("the committed CVaR sweep changed a locked information artifact")
    if metadata["physical_settlement_modified"] or not metadata["causality_audit_passed"]:
        raise AssertionError("the committed CVaR sweep changed or failed the physical contract")
    physical = metadata["physical_audits"]
    if not all(
        all(value for value in audit.values() if isinstance(value, bool))
        and float(audit["unabsorbed_discharge_energy_kwh"]) == 0.0
        for audit in physical.values()
    ):
        raise AssertionError("one or more committed CVaR physical audits failed")

    changes = mild_risk_changes(frame)
    number_path, table_path = _write_latex_outputs(frame, changes)
    figure_paths = plot_cvar_risk_tradeoff(frame, output_dir=problem_figures_dir(2))
    audit = {
        "source_sweep": str(SWEEP_PATH.relative_to(PROJECT_ROOT)),
        "source_sweep_sha256": _sha256(SWEEP_PATH),
        "source_metadata": str(METADATA_PATH.relative_to(PROJECT_ROOT)),
        "source_metadata_sha256": _sha256(METADATA_PATH),
        "cvar_definition": "emergency procurement cost only",
        "alpha": ALPHA,
        "lambda_values": list(EXPECTED_LAMBDAS),
        "all_solver_points": "334/334 Optimal",
        "all_physical_audits_passed": True,
        "changes_lambda_005_vs_0": changes,
        "interpretation": (
            "lambda=0 remains the main expected-cost solution; lambda=0.05 is an illustrative "
            "mild-risk point, and realized cost is ex-post evidence only"
        ),
        "outputs": {
            "latex_numbers": str(number_path.relative_to(PROJECT_ROOT)),
            "latex_table": str(table_path.relative_to(PROJECT_ROOT)),
            "figure": {key: str(value.relative_to(PROJECT_ROOT)) for key, value in figure_paths.items()},
        },
    }
    audit_path = TABLE_DIR / "q2_cvar_reporting_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit


if __name__ == "__main__":
    finalize_risk_outputs()
