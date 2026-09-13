"""Build the Q4 paper table from frozen audited outputs only.

This presentation-only script never runs a solver and never edits the formal
result workbooks.  It keeps the Q4 numbers in the TeX manuscript traceable to
the committed machine-readable result tables.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
Q42_PATH = ROOT / "results/problem4/tables/q42_summary.json"
Q43_PATH = ROOT / "results/problem4/tables/q43_schedule_comparison.csv"
OUTPUT_PATH = ROOT / "paper/contents/generated/q4_dispatch_results.tex"


def _wan(value: float) -> str:
    return f"{value / 10_000:.3f}"


def main() -> None:
    q42 = json.loads(Q42_PATH.read_text(encoding="utf-8"))
    q43 = pd.read_csv(Q43_PATH).set_index("schedule")
    if q42["optimal_days"] != 334 or not q43["solver_success"].eq(334).all():
        raise AssertionError("Q4 paper table requires 334/334 Optimal results")

    rows = [
        (
            "Q4-2",
            "0",
            _wan(q42["planned_purchase_cost_yuan"]),
            "--",
            _wan(q42["realized_emergency_cost_yuan"]),
            _wan(q42["realized_total_cost_yuan"]),
            f"{q42['realized_emergency_energy_kwh'] / 10_000:.3f}",
        )
    ]
    for schedule in ("S0", "S1", "S2", "S3"):
        row = q43.loc[schedule]
        rows.append(
            (
                schedule,
                str(row["release_hours"]),
                _wan(row["planned_purchase_cost_yuan"]),
                _wan(row["adjustment_cost_yuan"]),
                _wan(row["realized_emergency_cost_yuan"]),
                _wan(row["realized_total_cost_yuan"]),
                f"{row['realized_emergency_energy_kwh'] / 10_000:.3f}",
            )
        )

    body = "\n".join(" & ".join(values) + r" \\" for values in rows)
    OUTPUT_PATH.write_text(
        "\\begin{table}[H]\n"
        "\\centering\\footnotesize\n"
        "\\caption{动态电价下问题四的年度实际结算结果}"
        "\\label{tab:q4-dispatch-results}\n"
        "\\setlength{\\tabcolsep}{3.5pt}\n"
        "\\begin{tabular}{lcrrrrr}\n\\toprule\n"
        "方案 & 发布时间/h & 计划费/万元 & 调整费/万元 & 紧急费/万元 & 总费用/万元 & 紧急电量/万kWh \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n\\end{tabular}\n"
        "\\par\\vspace{3pt}\\begin{minipage}{0.97\\textwidth}\\footnotesize "
        "评价期为2025年2月1日至12月31日；各方案均为334/334日最优。"
        "Q4-2无滚动调整；S0--S3采用相对上一承诺的主结算口径。"
        "所有数值均为单次年度实际回放结果，不附加统计误差棒。"
        "\\end{minipage}\n\\end{table}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
