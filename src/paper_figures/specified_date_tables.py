"""Generate the Q2/Q3 specified-date paper tables from frozen workbooks.

This script is presentation-only.  It reads ``result2.xlsx`` and
``result3.xlsx`` without modifying them and writes compact LaTeX tables in the
paper's generated-content directory.
"""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "paper/contents/generated"
SPECIFIED_DATES = (
    date(2025, 3, 20),
    date(2025, 6, 21),
    date(2025, 9, 23),
    date(2025, 12, 21),
)
PLAN_PERIODS = (
    "10:00-10:10",
    "12:00-12:10",
    "14:00-14:10",
    "16:00-16:10",
    "18:00-18:10",
    "20:00-20:10",
)
STORAGE_PERIODS = (
    "0:00-4:00",
    "4:00-8:00",
    "8:00-12:00",
    "12:00-16:00",
    "16:00-20:00",
    "20:00-24:00",
)
POSITIVE_TOLERANCE_KWH = 1e-9
PROBLEM_NAMES = {2: "二", 3: "三"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fmt(value: float) -> str:
    if not np.isfinite(value):
        raise AssertionError("Specified-date tables require finite values")
    return f"{value:.3f}"


def _date_frame(path: Path, sheet: str) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name=sheet)
    date_column = frame.columns[0]
    frame[date_column] = pd.to_datetime(frame[date_column].ffill()).dt.date
    frame = frame.loc[frame[date_column].isin(SPECIFIED_DATES)].copy()
    if set(frame[date_column]) != set(SPECIFIED_DATES):
        raise AssertionError(f"{path.name}/{sheet} does not cover all specified dates")
    return frame.rename(columns={date_column: "date"})


def _plan_rows(path: Path) -> list[list[str]]:
    frame = _date_frame(path, "计划购电量")
    required = [*PLAN_PERIODS, "全天购电量", "全天购电费"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise AssertionError(f"{path.name} missing plan columns: {missing}")
    if frame["date"].duplicated().any():
        raise AssertionError(f"{path.name} has duplicate plan dates")
    frame = frame.set_index("date")
    rows: list[list[str]] = []
    for day in SPECIFIED_DATES:
        values = [_fmt(float(frame.at[day, column])) for column in required]
        rows.append([day.isoformat(), *values])
    return rows


def _storage_rows(path: Path) -> tuple[list[list[str]], dict[date, dict[str, float]]]:
    frame = _date_frame(path, "充放电量")
    for column in ("时间段", "充电量", "放电量", "时刻", "储电量"):
        if column not in frame.columns:
            raise AssertionError(f"{path.name} missing storage column: {column}")

    rows: list[list[str]] = []
    states: dict[date, dict[str, float]] = {}
    for day in SPECIFIED_DATES:
        day_frame = frame.loc[frame["date"].eq(day)].copy()
        if day_frame["时间段"].tolist() != list(STORAGE_PERIODS):
            raise AssertionError(f"{path.name} has unexpected storage periods on {day}")
        row = []
        for period in STORAGE_PERIODS:
            match = day_frame.loc[day_frame["时间段"].eq(period)].iloc[0]
            row.extend((_fmt(float(match["充电量"])), _fmt(float(match["放电量"]))))
        rows.append(row)
        observed = day_frame.dropna(subset=["时刻", "储电量"])
        states[day] = {
            str(record["时刻"]): float(record["储电量"])
            for _, record in observed.iterrows()
        }
        if set(states[day]) != {"00:00", "24:00"}:
            raise AssertionError(f"{path.name} has incomplete boundary energy on {day}")
    return rows, states


def _split_period(period: str) -> tuple[str, str]:
    parts = str(period).split("-", maxsplit=1)
    if len(parts) != 2:
        raise AssertionError(f"Unexpected interval label: {period}")
    return parts[0], parts[1]


def _emergency_segments(path: Path) -> tuple[dict[date, list[tuple[str, float]]], dict[date, float]]:
    frame = _date_frame(path, "紧急购电量")
    if not {"购电时间段", "购电量"}.issubset(frame.columns):
        raise AssertionError(f"{path.name} has an invalid emergency-purchase sheet")

    result: dict[date, list[tuple[str, float]]] = {}
    totals: dict[date, float] = {}
    for day in SPECIFIED_DATES:
        day_frame = frame.loc[frame["date"].eq(day)].reset_index(drop=True)
        if len(day_frame) != 144:
            raise AssertionError(f"{path.name} must contain 144 emergency rows on {day}")
        values = pd.to_numeric(day_frame["购电量"], errors="raise").astype(float)
        if not np.isfinite(values).all() or (values < -POSITIVE_TOLERANCE_KWH).any():
            raise AssertionError(f"{path.name} has invalid emergency energy on {day}")

        groups: list[dict[str, object]] = []
        for slot, (period, value) in enumerate(zip(day_frame["购电时间段"], values)):
            if value <= POSITIVE_TOLERANCE_KWH:
                continue
            start, end = _split_period(str(period))
            if groups and slot == int(groups[-1]["last_slot"]) + 1:
                groups[-1]["end"] = end
                groups[-1]["energy"] = float(groups[-1]["energy"]) + value
                groups[-1]["last_slot"] = slot
            else:
                groups.append(
                    {"start": start, "end": end, "energy": value, "last_slot": slot}
                )
        result[day] = [
            (f"{group['start']}-{group['end']}", float(group["energy"]))
            for group in groups
        ]
        totals[day] = float(values.sum())
        if not np.isclose(totals[day], sum(value for _, value in result[day])):
            raise AssertionError(f"{path.name} emergency segment sum mismatch on {day}")
    return result, totals


def _plan_table(problem: int, rows: list[list[str]]) -> str:
    body = "\n".join(" & ".join(row) + r" \\" for row in rows)
    headers = [
        "日期",
        "10:00--10:10",
        "12:00--12:10",
        "14:00--14:10",
        "16:00--16:10",
        "18:00--18:10",
        "20:00--20:10",
        "全天电量/kWh",
        "全天费用/元",
    ]
    return (
        "\\begin{table}[H]\n\\centering\\scriptsize\n"
        f"\\caption{{问题{PROBLEM_NAMES[problem]}指定日期的计划购电结果}}"
        f"\\label{{tab:q{problem}-specified-plan}}\n"
        "\\setlength{\\tabcolsep}{2.4pt}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{lrrrrrrrr}\n\\toprule\n"
        + " & ".join(headers)
        + r" \\"
        + "\n\\midrule\n"
        + body
        + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n"
    )


def _storage_table(
    problem: int, rows: list[list[str]], states: dict[date, dict[str, float]]
) -> str:
    date_header = " & ".join(
        f"\\multicolumn{{2}}{{c}}{{{day.isoformat()}}}" for day in SPECIFIED_DATES
    )
    cmidrules = " ".join(
        f"\\cmidrule(lr){{{2 + 2 * index}-{3 + 2 * index}}}"
        for index in range(len(SPECIFIED_DATES))
    )
    subheader = "时段 & " + " & ".join(["充电量 & 放电量"] * len(SPECIFIED_DATES))
    body_rows = []
    for period_index, period in enumerate(STORAGE_PERIODS):
        values = []
        for date_index in range(len(SPECIFIED_DATES)):
            values.extend(rows[date_index][2 * period_index : 2 * period_index + 2])
        body_rows.append(period.replace("-", "--") + " & " + " & ".join(values) + r" \\")
    state_rows = []
    for time in ("00:00", "24:00"):
        values = " & ".join(
            f"\\multicolumn{{2}}{{c}}{{{_fmt(states[day][time])}}}"
            for day in SPECIFIED_DATES
        )
        state_rows.append(f"{time}储电量 & {values}" + r" \\")
    return (
        "\\begin{table}[H]\n\\centering\\scriptsize\n"
        f"\\caption{{问题{PROBLEM_NAMES[problem]}指定日期的储能充放电与边界电量}}"
        f"\\label{{tab:q{problem}-specified-storage}}\n"
        "\\setlength{\\tabcolsep}{2.0pt}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{lrrrrrrrr}\n\\toprule\n"
        f" & {date_header} \\\\\n{cmidrules}\n{subheader} \\\\\n\\midrule\n"
        + "\n".join(body_rows)
        + "\n\\midrule\n"
        + "\n".join(state_rows)
        + "\n\\bottomrule\n\\end{tabular}}\n"
        "\\par\\vspace{2pt}\\footnotesize 充、放电量及边界储电量单位均为kWh。\n"
        "\\end{table}\n"
    )


def _emergency_table(
    problem: int,
    segments: dict[date, list[tuple[str, float]]],
    totals: dict[date, float],
) -> str:
    date_header = " & ".join(
        f"\\multicolumn{{2}}{{c}}{{{day.isoformat()}}}" for day in SPECIFIED_DATES
    )
    cmidrules = " ".join(
        f"\\cmidrule(lr){{{1 + 2 * index}-{2 + 2 * index}}}"
        for index in range(len(SPECIFIED_DATES))
    )
    subheader = " & ".join(["时间段 & 购电量/kWh"] * len(SPECIFIED_DATES))
    row_count = max(len(segments[day]) for day in SPECIFIED_DATES)
    body_rows = []
    for row_index in range(row_count):
        cells = []
        for day in SPECIFIED_DATES:
            if row_index < len(segments[day]):
                period, energy = segments[day][row_index]
                cells.extend((period.replace("-", "--"), _fmt(energy)))
            else:
                cells.extend(("", ""))
        body_rows.append(" & ".join(cells) + r" \\")
    total_cells = []
    for day in SPECIFIED_DATES:
        total_cells.extend(("合计", _fmt(totals[day])))
    body_rows.append("\\midrule\n" + " & ".join(total_cells) + r" \\")
    return (
        "\\begin{table}[H]\n\\centering\\scriptsize\n"
        f"\\caption{{问题{PROBLEM_NAMES[problem]}指定日期的紧急购电结果}}"
        f"\\label{{tab:q{problem}-specified-emergency}}\n"
        "\\setlength{\\tabcolsep}{2.2pt}\\renewcommand{\\arraystretch}{0.90}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{rrrrrrrr}\n\\toprule\n"
        f"{date_header} \\\\\n{cmidrules}\n{subheader} \\\\\n\\midrule\n"
        + "\n".join(body_rows)
        + "\n\\bottomrule\n\\end{tabular}}\n"
        "\\par\\vspace{2pt}\\footnotesize 仅列非零紧急购电；连续10分钟区间合并，电量按区间求和。\n"
        "\\end{table}\n"
    )


def build_problem(problem: int, path: Path) -> Path:
    plan_rows = _plan_rows(path)
    storage_rows, states = _storage_rows(path)
    segments, totals = _emergency_segments(path)
    output = OUTPUT_DIR / f"q{problem}_specified_dates.tex"
    output.write_text(
        f"% Generated from {path.relative_to(ROOT)}\n"
        f"% source_sha256={_sha256(path)}\n"
        + _plan_table(problem, plan_rows)
        + "\n"
        + _storage_table(problem, storage_rows, states)
        + "\n"
        + _emergency_table(problem, segments, totals),
        encoding="utf-8",
    )
    return output


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = (
        build_problem(2, ROOT / "results/problem2/result2.xlsx"),
        build_problem(3, ROOT / "results/problem3/result3.xlsx"),
    )
    for output in outputs:
        print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
