"""Two-day Q4 design smoke test; this module never launches the annual run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import pandas as pd

from src.common.paths import problem_results_dir
from src.problem4.dispatch_adapter import (
    CAUSAL_PRICE,
    GIVEN_PRICE,
    q43_price_provider,
    run_q42_sample,
    run_q43_sample,
)


SCHEDULES = ("S0", "S1", "S2", "S3")
DEFAULT_DATES = pd.date_range("2025-02-01", periods=2, freq="D")
OUTPUT_PATH = problem_results_dir(4) / "tables" / "q4_dispatch_smoke.json"


def _audit_q43_prices(intervals: pd.DataFrame, mode: str) -> dict[str, object]:
    settlement_residuals: list[float] = []
    decision_residuals: list[float] = []
    emergency_cost_residuals: list[float] = []
    for operating_date, day in intervals.groupby("operating_date", sort=True):
        rows = day.sort_values("slot", kind="stable")
        decision, settlement = q43_price_provider(mode)(pd.Timestamp(operating_date))
        settlement_residuals.append(
            float(abs(rows["price_yuan_per_kwh"].to_numpy(float) - settlement).max())
        )
        decision_residuals.append(
            float(
                abs(
                    rows["decision_price_yuan_per_kwh"].to_numpy(float) - decision
                ).max()
            )
        )
        emergency_cost_residuals.append(
            float(
                abs(
                    rows["realized_emergency_cost_yuan"].to_numpy(float)
                    - 5.0
                    * settlement
                    * rows["realized_emergency_kwh"].to_numpy(float)
                ).max()
            )
        )
    maximum_settlement = max(settlement_residuals, default=0.0)
    maximum_decision = max(decision_residuals, default=0.0)
    maximum_emergency = max(emergency_cost_residuals, default=0.0)
    return {
        "dynamic_settlement_price_alignment": maximum_settlement <= 1e-12,
        "maximum_settlement_price_residual_yuan_per_kwh": maximum_settlement,
        "dynamic_decision_price_alignment": maximum_decision <= 1e-12,
        "maximum_decision_price_residual_yuan_per_kwh": maximum_decision,
        "five_times_emergency_price": maximum_emergency <= 2e-3,
        "maximum_emergency_price_residual_yuan": maximum_emergency,
        "price_information_audit": (
            "PASS" if mode == CAUSAL_PRICE else "ASSUMPTION_NOT_CAUSAL"
        ),
        "given_price_assumption": mode == GIVEN_PRICE,
    }


def run_smoke(output_path: Path = OUTPUT_PATH) -> dict[str, object]:
    started = perf_counter()
    cache_dir = output_path.parent / ".q4_smoke_cache"
    payload: dict[str, object] = {
        "scope": "two-day design smoke only; no annual Q4 run",
        "dates": [str(value.date()) for value in DEFAULT_DATES],
        "price_information_classification": "AMBIGUOUS",
        "modes": {},
    }
    for mode in (GIVEN_PRICE, CAUSAL_PRICE):
        mode_started = perf_counter()
        q42 = run_q42_sample(
            DEFAULT_DATES,
            mode=mode,
            checkpoint_path=cache_dir / f"q42_{mode}.pkl",
        )
        schedules: dict[str, object] = {}
        for schedule in SCHEDULES:
            q43 = run_q43_sample(
                schedule,
                DEFAULT_DATES,
                mode=mode,
                checkpoint_path=cache_dir / f"q43_{mode}_{schedule}.pkl",
            )
            schedules[schedule] = {
                "days_optimal": int(
                    q43.daily_main["solver_status"].eq("Optimal").sum()
                ),
                "realized_total_cost_yuan": float(
                    q43.daily_main["realized_total_cost_yuan"].sum()
                ),
                "audit": q43.audit,
                "price_audit": _audit_q43_prices(q43.intervals, mode),
            }
        payload["modes"][mode] = {
            "q4_2": {
                "days_optimal": int(q42.daily["solver_status"].eq("Optimal").sum()),
                "realized_total_cost_yuan": float(q42.daily["realized_total_cost_yuan"].sum()),
                "audit": q42.audit,
            },
            "q4_3": schedules,
            "runtime_seconds": perf_counter() - mode_started,
        }
    payload["runtime_seconds"] = perf_counter() - started
    payload["annual_run_performed"] = False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    arguments = parser.parse_args()
    print(json.dumps(run_smoke(arguments.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
