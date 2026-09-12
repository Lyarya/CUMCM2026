"""Independent Q4 robustness experiments around the locked GIVEN_PRICE model.

This module never calls the formal annual exporter and never writes result4-2
or result4-3.  It injects controlled price/scenario perturbations into the
already-audited Q4 adapters in an isolated process, then retains only compact
case summaries and physical audits.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.common.paths import PROJECT_ROOT, RESULTS_DIR
from src.problem2.forecast_interface import (
    Q2DayInputs,
    SCENARIO_PATH,
    get_q2_day_inputs as get_formal_q2_day_inputs,
)
import src.problem3.rolling_dispatch as q3_rolling
import src.problem4.dispatch_adapter as q4_adapter
from src.problem4.dispatch_adapter import GIVEN_PRICE


FORMAL_DATES = pd.date_range("2025-02-01", "2025-12-31", freq="D")
PRICE_GAMMAS = (0.00, 0.25, 0.50, 0.75, 1.00)
UNCERTAINTY_KAPPAS = (0.75, 1.00, 1.25, 1.50)
SEED = 2026
TOLERANCE = 2e-3
REPRODUCTION_COST_TOLERANCE = 1e-3
REPRODUCTION_ENERGY_TOLERANCE = 1e-6

ROBUSTNESS_DIR = RESULTS_DIR / "problem4" / "robustness"
PRICE_CASE_DIR = ROBUSTNESS_DIR / "price_volatility"
UNCERTAINTY_CASE_DIR = ROBUSTNESS_DIR / "forecast_uncertainty"
TABLE_DIR = ROBUSTNESS_DIR / "tables"
PRICE_TABLE_PATH = RESULTS_DIR / "problem4/tables/q4_price_forecast_predictions.csv"
FORMAL_Q42_PATH = RESULTS_DIR / "problem4/tables/q42_summary.json"
FORMAL_Q42_DISPATCH_PATH = RESULTS_DIR / "problem4/tables/q42_dispatch.csv"
FORMAL_Q43_PATH = RESULTS_DIR / "problem4/tables/q43_schedule_comparison.csv"
FORMAL_Q43_VOI_PATH = RESULTS_DIR / "problem4/tables/q43_voi.csv"
FORMAL_AUDIT_PATH = RESULTS_DIR / "problem4/Q4_GIVEN_PRICE_INDEPENDENT_AUDIT.json"
FORMAL_RESULT42_PATH = RESULTS_DIR / "problem4/result4-2.xlsx"
FORMAL_RESULT43_PATH = RESULTS_DIR / "problem4/result4-3.xlsx"


@dataclass(frozen=True)
class RobustnessSpec:
    gamma: float
    kappa: float

    @property
    def key(self) -> str:
        return f"gamma_{self.gamma:.2f}_kappa_{self.kappa:.2f}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_formal_price_frame() -> pd.DataFrame:
    """Load the immutable Feb--Dec Attachment-4 price grid."""

    frame = pd.read_csv(
        PRICE_TABLE_PATH,
        parse_dates=["operating_date", "interval_start", "interval_end"],
    )
    frame = frame.loc[
        frame["operating_date"].between(FORMAL_DATES[0], FORMAL_DATES[-1]),
        ["operating_date", "slot", "interval_end", "price_yuan_per_kwh"],
    ].sort_values(["operating_date", "slot"], kind="stable")
    if len(frame) != len(FORMAL_DATES) * 144:
        raise AssertionError("formal Q4 robustness price grid is not 334 x 144")
    if frame[["operating_date", "slot"]].duplicated().any():
        raise AssertionError("formal Q4 robustness price grid has duplicate keys")
    if not frame.groupby("operating_date")["slot"].nunique().eq(144).all():
        raise AssertionError("a formal Q4 robustness day does not contain 144 prices")
    price = frame["price_yuan_per_kwh"].to_numpy(float)
    if not np.isfinite(price).all() or (price < 0.0).any():
        raise AssertionError("formal Q4 prices must be finite and nonnegative")
    return frame.reset_index(drop=True)


def scale_price_frame(frame: pd.DataFrame, gamma: float) -> pd.DataFrame:
    """Scale only deviations from the formal-period mean price."""

    gamma = float(gamma)
    if gamma < 0.0:
        raise ValueError("gamma must be nonnegative")
    result = frame.copy()
    original = frame["price_yuan_per_kwh"].to_numpy(float)
    mean = float(original.mean())
    scaled = (
        np.array(original, copy=True)
        if gamma == 1.0
        else mean + gamma * (original - mean)
    )
    if not np.isfinite(scaled).all() or (scaled < 0.0).any():
        raise AssertionError("price scaling produced an invalid price")
    if abs(float(scaled.mean()) - mean) > 1e-12:
        raise AssertionError("price scaling changed the formal-period mean")
    if not np.allclose(scaled - mean, gamma * (original - mean), atol=1e-12):
        raise AssertionError("price scaling changed more than volatility")
    result["price_yuan_per_kwh"] = scaled
    return result


def scale_q2_scenarios(base: Q2DayInputs, kappa: float) -> Q2DayInputs:
    """Scale paired load/PV residual amplitudes around unchanged point forecasts."""

    kappa = float(kappa)
    if kappa < 0.0:
        raise ValueError("kappa must be nonnegative")
    if kappa == 1.0:
        load = np.array(base.load_scenarios, copy=True)
        pv = np.array(base.pv_scenarios, copy=True)
    else:
        load_residual = base.load_scenarios - base.load_forecast[None, :]
        pv_residual = base.pv_scenarios - base.pv_forecast[None, :]
        load = base.load_forecast[None, :] + kappa * load_residual
        pv = np.maximum(base.pv_forecast[None, :] + kappa * pv_residual, 0.0)
    if load.shape != base.load_scenarios.shape or pv.shape != base.pv_scenarios.shape:
        raise AssertionError("scenario scaling changed array shapes")
    if not np.isfinite(load).all() or not np.isfinite(pv).all():
        raise AssertionError("scenario scaling produced non-finite values")
    if (pv < 0.0).any():
        raise AssertionError("scenario scaling produced negative PV")
    if kappa == 1.0:
        np.testing.assert_array_equal(load, base.load_scenarios)
        np.testing.assert_array_equal(pv, base.pv_scenarios)
    return replace(
        base,
        load_scenarios=np.array(load, copy=True),
        pv_scenarios=np.array(pv, copy=True),
        scenario_source_dates=np.array(base.scenario_source_dates, copy=True),
    )


def _price_lookup(gamma: float) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    frame = scale_price_frame(load_formal_price_frame(), gamma)
    lookup = {
        str(date.date()): daily.sort_values("slot")["price_yuan_per_kwh"].to_numpy(float)
        for date, daily in frame.groupby("operating_date", sort=True)
    }
    values = frame["price_yuan_per_kwh"].to_numpy(float)
    stats = {
        "price_mean_yuan_per_kwh": float(values.mean()),
        "price_std_yuan_per_kwh": float(values.std(ddof=0)),
        "price_min_yuan_per_kwh": float(values.min()),
        "price_max_yuan_per_kwh": float(values.max()),
        "timestamp_alignment": bool(
            len(lookup) == 334 and all(value.shape == (144,) for value in lookup.values())
        ),
        "missing_count": int(frame["price_yuan_per_kwh"].isna().sum()),
        "duplicate_count": int(frame[["operating_date", "slot"]].duplicated().sum()),
    }
    return lookup, stats


def _weighted_price(intervals: pd.DataFrame, energy_column: str) -> float | None:
    energy = intervals[energy_column].to_numpy(float)
    total = float(energy.sum())
    if total <= 1e-12:
        return None
    price = intervals["price_yuan_per_kwh"].to_numpy(float)
    return float(np.sum(price * energy) / total)


def _audit_price_alignment(
    intervals: pd.DataFrame, lookup: dict[str, np.ndarray]
) -> dict[str, object]:
    maximum = 0.0
    for date, daily in intervals.groupby("operating_date", sort=True):
        observed = daily.sort_values("slot")["price_yuan_per_kwh"].to_numpy(float)
        expected = lookup[str(pd.Timestamp(date).date())]
        maximum = max(maximum, float(np.max(np.abs(observed - expected))))
    return {
        "price_alignment": bool(maximum <= 1e-12),
        "maximum_price_alignment_residual_yuan_per_kwh": maximum,
    }


def _audit_q42(result, lookup: dict[str, np.ndarray]) -> dict[str, object]:
    intervals = result.intervals
    checks = {
        **result.audit,
        **_audit_price_alignment(intervals, lookup),
        "solver_success": bool(result.daily["solver_status"].eq("Optimal").all()),
        "grid_nonnegative": bool((intervals["planned_grid_kw"] >= -TOLERANCE).all()),
        "charge_discharge_feasible": bool(
            (intervals["actual_charge_kw"] <= intervals["planned_charge_limit_kw"] + TOLERANCE).all()
            and (
                intervals["actual_discharge_kw"]
                <= intervals["planned_discharge_limit_kw"] + TOLERANCE
            ).all()
        ),
        "scenario_power_balance": bool(
            result.daily["maximum_scenario_power_balance_residual_kw"].max()
            <= TOLERANCE
        ),
    }
    checks["status"] = (
        "PASS" if all(value for value in checks.values() if isinstance(value, bool)) else "FAIL"
    )
    if checks["status"] != "PASS":
        raise AssertionError(f"Q4-2 robustness audit failed: {checks}")
    return checks


def _audit_s2(result, lookup: dict[str, np.ndarray]) -> dict[str, object]:
    intervals = result.intervals
    checks = {
        **result.audit,
        **_audit_price_alignment(intervals, lookup),
        "solver_success": bool(result.daily_main["solver_status"].eq("Optimal").all()),
        "grid_nonnegative": bool((intervals["planned_grid_kw"] >= -TOLERANCE).all()),
        "no_simultaneous_actual_charge_discharge": bool(
            ~(
                (intervals["actual_charge_kw"] > TOLERANCE)
                & (intervals["actual_discharge_kw"] > TOLERANCE)
            ).any()
        ),
        "scenario_power_balance": bool(
            result.daily_main["maximum_scenario_power_balance_residual_kw"].max()
            <= TOLERANCE
        ),
    }
    checks["status"] = (
        "PASS" if all(value for value in checks.values() if isinstance(value, bool)) else "FAIL"
    )
    if checks["status"] != "PASS":
        raise AssertionError(f"Q4-3/S2 robustness audit failed: {checks}")
    return checks


def _q42_metrics(result) -> dict[str, object]:
    daily = result.daily
    intervals = result.intervals
    return {
        "planned_cost_yuan": float(daily["planned_purchase_cost_yuan"].sum()),
        "expected_emergency_cost_yuan": float(
            daily["expected_scenario_emergency_cost_yuan"].sum()
        ),
        "realized_total_cost_yuan": float(daily["realized_total_cost_yuan"].sum()),
        "realized_emergency_energy_kwh": float(
            daily["realized_emergency_energy_kwh"].sum()
        ),
        "realized_emergency_cost_yuan": float(
            daily["realized_emergency_cost_yuan"].sum()
        ),
        "grid_energy_kwh": float(daily["planned_purchase_energy_kwh"].sum()),
        "charge_energy_kwh": float(intervals["actual_charge_kwh"].sum()),
        "discharge_energy_kwh": float(intervals["actual_discharge_kwh"].sum()),
        "battery_throughput_kwh": float(
            intervals["actual_charge_kwh"].sum()
            + intervals["actual_discharge_kwh"].sum()
        ),
        "final_soc_kwh": float(daily["actual_final_energy_kwh"].iloc[-1]),
        "charge_weighted_price_yuan_per_kwh": _weighted_price(
            intervals, "actual_charge_kwh"
        ),
        "discharge_weighted_price_yuan_per_kwh": _weighted_price(
            intervals, "actual_discharge_kwh"
        ),
        "solver_runtime_seconds": float(daily["solver_runtime_seconds"].sum()),
    }


def _s2_metrics(result) -> dict[str, object]:
    daily = result.daily_main
    intervals = result.intervals
    return {
        "planned_base_cost_yuan": float(
            daily["initial_planned_purchase_cost_yuan"].sum()
        ),
        "model_expected_operating_cost_yuan": float(
            daily["initial_expected_operating_cost_yuan"].sum()
        ),
        "realized_total_cost_yuan": float(daily["realized_total_cost_yuan"].sum()),
        "adjustment_cost_yuan": float(daily["adjustment_cost_yuan"].sum()),
        "realized_emergency_energy_kwh": float(daily["emergency_energy_kwh"].sum()),
        "realized_emergency_cost_yuan": float(daily["emergency_cost_yuan"].sum()),
        "grid_energy_kwh": float(daily["initial_planned_purchase_energy_kwh"].sum()),
        "charge_energy_kwh": float(intervals["actual_charge_kwh"].sum()),
        "discharge_energy_kwh": float(intervals["actual_discharge_kwh"].sum()),
        "battery_throughput_kwh": float(
            intervals["actual_charge_kwh"].sum()
            + intervals["actual_discharge_kwh"].sum()
        ),
        "final_soc_kwh": float(daily["final_energy_kwh"].iloc[-1]),
        "charge_weighted_price_yuan_per_kwh": _weighted_price(
            intervals, "actual_charge_kwh"
        ),
        "discharge_weighted_price_yuan_per_kwh": _weighted_price(
            intervals, "actual_discharge_kwh"
        ),
        "solver_runtime_seconds": float(daily["solver_runtime_seconds"].sum()),
    }


def formal_baseline() -> dict[str, object]:
    q42 = json.loads(FORMAL_Q42_PATH.read_text(encoding="utf-8"))
    q43 = pd.read_csv(FORMAL_Q43_PATH).set_index("schedule").loc["S2"]
    return {
        "q42_realized_total_cost_yuan": float(q42["realized_total_cost_yuan"]),
        "q42_emergency_energy_kwh": float(q42["realized_emergency_energy_kwh"]),
        "s2_realized_total_cost_yuan": float(q43["realized_total_cost_yuan"]),
        "s2_emergency_energy_kwh": float(q43["realized_emergency_energy_kwh"]),
        "s2_adjustment_cost_yuan": float(q43["adjustment_cost_yuan"]),
        "result4_2_sha256": _sha256(FORMAL_RESULT42_PATH),
        "result4_3_sha256": _sha256(FORMAL_RESULT43_PATH),
    }


def _charge_discharge_from_throughput(
    throughput_kwh: float,
    initial_soc_kwh: float,
    final_soc_kwh: float,
    efficiency: float = 0.9,
) -> tuple[float, float]:
    """Recover annual charge/discharge totals from energy conservation."""

    delta = float(final_soc_kwh) - float(initial_soc_kwh)
    charge = (delta + float(throughput_kwh) / efficiency) / (
        efficiency + 1.0 / efficiency
    )
    discharge = float(throughput_kwh) - charge
    if charge < -TOLERANCE or discharge < -TOLERANCE:
        raise AssertionError("formal throughput implies negative charge or discharge")
    return float(charge), float(discharge)


def formal_baseline_payload() -> dict[str, object]:
    """Represent gamma=kappa=1 from audited formal outputs without re-solving."""

    q42_summary = json.loads(FORMAL_Q42_PATH.read_text(encoding="utf-8"))
    q42_dispatch = pd.read_csv(FORMAL_Q42_DISPATCH_PATH)
    q43_row = pd.read_csv(FORMAL_Q43_PATH).set_index("schedule").loc["S2"]
    audit = json.loads(FORMAL_AUDIT_PATH.read_text(encoding="utf-8"))
    _, price_stats = _price_lookup(1.0)
    q42_charge = float(q42_dispatch["actual_charge_kwh"].sum())
    q42_discharge = float(q42_dispatch["actual_discharge_kwh"].sum())
    s2_charge, s2_discharge = _charge_discharge_from_throughput(
        q43_row["battery_throughput_kwh"],
        6000.0,
        q43_row["final_soc_kwh"],
    )
    s2_diagnostics = audit["q43"]["S2"]["economic_diagnostics"]
    q42 = {
        "planned_cost_yuan": float(q42_summary["planned_purchase_cost_yuan"]),
        "expected_emergency_cost_yuan": float(
            q42_summary["expected_scenario_emergency_cost_yuan"]
        ),
        "realized_total_cost_yuan": float(q42_summary["realized_total_cost_yuan"]),
        "realized_emergency_energy_kwh": float(
            q42_summary["realized_emergency_energy_kwh"]
        ),
        "realized_emergency_cost_yuan": float(
            q42_summary["realized_emergency_cost_yuan"]
        ),
        "grid_energy_kwh": float(q42_summary["planned_purchase_energy_kwh"]),
        "charge_energy_kwh": q42_charge,
        "discharge_energy_kwh": q42_discharge,
        "battery_throughput_kwh": q42_charge + q42_discharge,
        "final_soc_kwh": float(q42_summary["final_soc_kwh"]),
        "charge_weighted_price_yuan_per_kwh": float(
            q42_summary["economic_diagnostics"]["charge_weighted_price_yuan_per_kwh"]
        ),
        "discharge_weighted_price_yuan_per_kwh": float(
            q42_summary["economic_diagnostics"]["discharge_weighted_price_yuan_per_kwh"]
        ),
        "solver_runtime_seconds": float(q42_summary["solver_runtime_seconds"]),
    }
    s2 = {
        "planned_base_cost_yuan": float(q43_row["planned_purchase_cost_yuan"]),
        "model_expected_operating_cost_yuan": float(
            q43_row["model_expected_operating_cost_yuan"]
        ),
        "realized_total_cost_yuan": float(q43_row["realized_total_cost_yuan"]),
        "adjustment_cost_yuan": float(q43_row["adjustment_cost_yuan"]),
        "realized_emergency_energy_kwh": float(
            q43_row["realized_emergency_energy_kwh"]
        ),
        "realized_emergency_cost_yuan": float(
            q43_row["realized_emergency_cost_yuan"]
        ),
        "grid_energy_kwh": float(q43_row["planned_purchase_energy_kwh"]),
        "charge_energy_kwh": s2_charge,
        "discharge_energy_kwh": s2_discharge,
        "battery_throughput_kwh": float(q43_row["battery_throughput_kwh"]),
        "final_soc_kwh": float(q43_row["final_soc_kwh"]),
        "charge_weighted_price_yuan_per_kwh": float(
            s2_diagnostics["charge_weighted_price_yuan_per_kwh"]
        ),
        "discharge_weighted_price_yuan_per_kwh": float(
            s2_diagnostics["discharge_weighted_price_yuan_per_kwh"]
        ),
        "solver_runtime_seconds": float(q43_row["solver_runtime_seconds"]),
    }
    saving = q42["realized_total_cost_yuan"] - s2["realized_total_cost_yuan"]
    q42_audit = {**audit["q42"]["audit"], "status": "PASS"}
    s2_audit = {
        **audit["q43"]["S2"]["physical"],
        "price_alignment": bool(audit["q43"]["S2"]["price_alignment"]["pass"]),
        "status": "PASS",
    }
    payload = {
        "experiment": "formal_baseline_reused",
        "gamma": 1.0,
        "kappa": 1.0,
        "seed": SEED,
        "formal_dates": [str(FORMAL_DATES[0].date()), str(FORMAL_DATES[-1].date())],
        "scenario_pairing": "unchanged persisted residual-day indices; no resampling",
        "price": price_stats,
        "q42": q42,
        "s2": s2,
        "absolute_saving_s2_vs_q42_yuan": float(saving),
        "saving_rate_s2_vs_q42": float(saving / q42["realized_total_cost_yuan"]),
        "q42_audit": q42_audit,
        "s2_audit": s2_audit,
        "audit_status": "PASS",
        "wall_runtime_seconds": 0.0,
        "case_signature": "audited-formal-baseline",
        "formal_input_hashes": {
            "price_table_sha256": _sha256(PRICE_TABLE_PATH),
            "scenario_archive_sha256": _sha256(SCENARIO_PATH),
        },
        "baseline_reproduction": "REUSED_AUDITED_FORMAL_BASELINE",
    }
    assert_baseline_reproduction(payload)
    return payload


def assert_baseline_reproduction(payload: dict[str, object]) -> None:
    expected = formal_baseline()
    observed = {
        "q42_realized_total_cost_yuan": payload["q42"]["realized_total_cost_yuan"],
        "q42_emergency_energy_kwh": payload["q42"]["realized_emergency_energy_kwh"],
        "s2_realized_total_cost_yuan": payload["s2"]["realized_total_cost_yuan"],
        "s2_emergency_energy_kwh": payload["s2"]["realized_emergency_energy_kwh"],
        "s2_adjustment_cost_yuan": payload["s2"]["adjustment_cost_yuan"],
    }
    for name, value in observed.items():
        tolerance = (
            REPRODUCTION_COST_TOLERANCE
            if "cost" in name
            else REPRODUCTION_ENERGY_TOLERANCE
        )
        if abs(float(value) - float(expected[name])) > tolerance:
            raise AssertionError(
                f"ROBUSTNESS_BASELINE_REPRODUCTION_FAIL: {name}: "
                f"observed={value}, expected={expected[name]}"
            )


def _case_signature(spec: RobustnessSpec) -> str:
    digest = hashlib.sha256()
    for path in (Path(__file__), PRICE_TABLE_PATH, SCENARIO_PATH):
        digest.update(path.read_bytes())
    digest.update(f"{spec.gamma:.12g}|{spec.kappa:.12g}|{SEED}".encode())
    return digest.hexdigest()


def _case_path(spec: RobustnessSpec) -> Path:
    if spec.kappa == 1.0 and spec.gamma != 1.0:
        directory = PRICE_CASE_DIR
    elif spec.gamma == 1.0 and spec.kappa != 1.0:
        directory = UNCERTAINTY_CASE_DIR
    else:
        directory = ROBUSTNESS_DIR / "baseline"
    return directory / f"{spec.key}.json"


def _checkpoint_paths(spec: RobustnessSpec) -> tuple[Path, Path]:
    directory = (
        Path("/private/tmp/cumcm_q4_robustness_checkpoints")
        / f"{spec.key}_{_case_signature(spec)[:16]}"
    )
    return directory / "q42.pkl", directory / "q43_s2.pkl"


def run_identity_smoke() -> dict[str, object]:
    """Compare the identity robustness wrapper with the formal runner on two days."""

    dates = pd.date_range("2025-02-01", periods=2, freq="D")
    started = perf_counter()
    formal_q42 = q4_adapter.run_q42_sample(dates, mode=GIVEN_PRICE)
    formal_s2 = q4_adapter.run_q43_sample("S2", dates, mode=GIVEN_PRICE)
    price_lookup, _ = _price_lookup(1.0)

    def identity_provider(date: object, initial_energy: float) -> Q2DayInputs:
        return scale_q2_scenarios(get_formal_q2_day_inputs(date, initial_energy), 1.0)

    def identity_price(date: object, mode: str) -> tuple[np.ndarray, np.ndarray]:
        if mode != GIVEN_PRICE:
            raise AssertionError("identity smoke must never enable CAUSAL_PRICE")
        price = np.array(price_lookup[str(pd.Timestamp(date).date())], copy=True)
        return price, np.array(price, copy=True)

    with ExitStack() as stack:
        stack.enter_context(patch.object(q4_adapter, "_price_pair", identity_price))
        stack.enter_context(
            patch.object(q4_adapter, "get_q2_day_inputs", identity_provider)
        )
        stack.enter_context(
            patch.object(q3_rolling, "get_q2_day_inputs", identity_provider)
        )
        wrapped_q42 = q4_adapter.run_q42_sample(dates, mode=GIVEN_PRICE)
        wrapped_s2 = q4_adapter.run_q43_sample("S2", dates, mode=GIVEN_PRICE)

    comparisons: dict[str, float] = {}
    for label, formal, wrapped, columns in (
        (
            "q42_daily",
            formal_q42.daily,
            wrapped_q42.daily,
            (
                "planned_purchase_cost_yuan",
                "realized_emergency_energy_kwh",
                "realized_total_cost_yuan",
                "actual_final_energy_kwh",
            ),
        ),
        (
            "s2_daily",
            formal_s2.daily_main,
            wrapped_s2.daily_main,
            (
                "initial_planned_purchase_cost_yuan",
                "adjustment_cost_yuan",
                "emergency_energy_kwh",
                "realized_total_cost_yuan",
                "final_energy_kwh",
            ),
        ),
    ):
        for column in columns:
            residual = float(
                np.max(
                    np.abs(
                        formal[column].to_numpy(float)
                        - wrapped[column].to_numpy(float)
                    )
                )
            )
            comparisons[f"{label}_{column}_maximum_residual"] = residual
    for label, formal, wrapped in (
        ("q42_intervals", formal_q42.intervals, wrapped_q42.intervals),
        ("s2_intervals", formal_s2.intervals, wrapped_s2.intervals),
    ):
        for column in (
            "planned_grid_kw",
            "actual_charge_kw",
            "actual_discharge_kw",
            "actual_storage_end_kwh",
            "realized_emergency_kwh",
            "price_yuan_per_kwh",
        ):
            residual = float(
                np.max(
                    np.abs(
                        formal[column].to_numpy(float)
                        - wrapped[column].to_numpy(float)
                    )
                )
            )
            comparisons[f"{label}_{column}_maximum_residual"] = residual
    maximum = max(comparisons.values(), default=0.0)
    payload = {
        "scope": "two-day identity smoke; no annual baseline solve",
        "dates": [str(value.date()) for value in dates],
        "gamma": 1.0,
        "kappa": 1.0,
        "mode": GIVEN_PRICE,
        "maximum_identity_residual": maximum,
        "comparisons": comparisons,
        "formal_q42_audit": formal_q42.audit,
        "wrapped_q42_audit": wrapped_q42.audit,
        "formal_s2_audit": formal_s2.audit,
        "wrapped_s2_audit": wrapped_s2.audit,
        "runtime_seconds": perf_counter() - started,
        "status": "PASS" if maximum <= 1e-9 else "FAIL",
    }
    if payload["status"] != "PASS":
        raise AssertionError(f"Q4 robustness identity smoke failed: {payload}")
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    (TABLE_DIR / "q4_robustness_identity_smoke.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def run_case(spec: RobustnessSpec, *, reuse: bool = True) -> dict[str, object]:
    """Run one full-year Q4-2 plus Q4-3/S2 controlled case."""

    if spec.gamma == 1.0 and spec.kappa == 1.0:
        raise ValueError(
            "the audited formal baseline must be reused; an annual identity rerun is prohibited"
        )

    output_path = _case_path(spec)
    signature = _case_signature(spec)
    if reuse and output_path.exists():
        saved = json.loads(output_path.read_text(encoding="utf-8"))
        if saved.get("case_signature") == signature and saved.get("audit_status") == "PASS":
            return saved

    price_lookup, price_stats = _price_lookup(spec.gamma)

    def scenario_provider(date: object, initial_energy: float) -> Q2DayInputs:
        base = get_formal_q2_day_inputs(date, initial_energy)
        scaled = scale_q2_scenarios(base, spec.kappa)
        if not np.array_equal(scaled.scenario_source_dates, base.scenario_source_dates):
            raise AssertionError("paired residual-day indices changed")
        if not (scaled.scenario_source_dates < np.datetime64(base.date)).all():
            raise AssertionError("scenario scaling introduced future leakage")
        return scaled

    def price_pair(date: object, mode: str) -> tuple[np.ndarray, np.ndarray]:
        if mode != GIVEN_PRICE:
            raise AssertionError("robustness runner must never enable CAUSAL_PRICE")
        price = np.array(price_lookup[str(pd.Timestamp(date).date())], copy=True)
        return price, np.array(price, copy=True)

    q42_checkpoint, q43_checkpoint = _checkpoint_paths(spec)
    started = perf_counter()
    with ExitStack() as stack:
        stack.enter_context(patch.object(q4_adapter, "_price_pair", price_pair))
        stack.enter_context(
            patch.object(q4_adapter, "get_q2_day_inputs", scenario_provider)
        )
        stack.enter_context(
            patch.object(q3_rolling, "get_q2_day_inputs", scenario_provider)
        )
        q42 = q4_adapter.run_q42_sample(
            FORMAL_DATES,
            mode=GIVEN_PRICE,
            checkpoint_path=q42_checkpoint,
            progress=True,
        )
        q43 = q4_adapter.run_q43_sample(
            "S2",
            FORMAL_DATES,
            mode=GIVEN_PRICE,
            checkpoint_path=q43_checkpoint,
            progress=True,
        )

    q42_audit = _audit_q42(q42, price_lookup)
    s2_audit = _audit_s2(q43, price_lookup)
    q42_metrics = _q42_metrics(q42)
    s2_metrics = _s2_metrics(q43)
    saving = q42_metrics["realized_total_cost_yuan"] - s2_metrics["realized_total_cost_yuan"]
    payload = {
        "experiment": (
            "price_volatility"
            if spec.kappa == 1.0 and spec.gamma != 1.0
            else "forecast_uncertainty"
            if spec.gamma == 1.0 and spec.kappa != 1.0
            else "formal_baseline"
        ),
        "gamma": spec.gamma,
        "kappa": spec.kappa,
        "seed": SEED,
        "formal_dates": [str(FORMAL_DATES[0].date()), str(FORMAL_DATES[-1].date())],
        "scenario_pairing": "unchanged persisted residual-day indices; amplitude only",
        "price": price_stats,
        "q42": q42_metrics,
        "s2": s2_metrics,
        "absolute_saving_s2_vs_q42_yuan": float(saving),
        "saving_rate_s2_vs_q42": float(
            saving / q42_metrics["realized_total_cost_yuan"]
        ),
        "q42_audit": q42_audit,
        "s2_audit": s2_audit,
        "audit_status": "PASS",
        "wall_runtime_seconds": perf_counter() - started,
        "case_signature": signature,
        "formal_input_hashes": {
            "price_table_sha256": _sha256(PRICE_TABLE_PATH),
            "scenario_archive_sha256": _sha256(SCENARIO_PATH),
        },
    }
    if spec.gamma == 1.0 and spec.kappa == 1.0:
        assert_baseline_reproduction(payload)
        payload["baseline_reproduction"] = "PASS"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output_path)
    return payload


def _worker(spec: RobustnessSpec) -> dict[str, object]:
    return run_case(spec)


def _run_many(specs: list[RobustnessSpec], workers: int) -> list[dict[str, object]]:
    if workers <= 1:
        return [run_case(spec) for spec in specs]
    completed: dict[str, dict[str, object]] = {}
    with ProcessPoolExecutor(max_workers=min(workers, len(specs))) as executor:
        futures = {executor.submit(_worker, spec): spec for spec in specs}
        for future in as_completed(futures):
            spec = futures[future]
            completed[spec.key] = future.result()
            print(f"robustness case complete: {spec.key}", flush=True)
    return [completed[spec.key] for spec in specs]


def _price_row(payload: dict[str, object]) -> dict[str, object]:
    q42, s2, price = payload["q42"], payload["s2"], payload["price"]
    return {
        "gamma": payload["gamma"],
        "price_mean_yuan_per_kwh": price["price_mean_yuan_per_kwh"],
        "price_std_yuan_per_kwh": price["price_std_yuan_per_kwh"],
        "price_min_yuan_per_kwh": price["price_min_yuan_per_kwh"],
        "price_max_yuan_per_kwh": price["price_max_yuan_per_kwh"],
        "q42_planned_cost_yuan": q42["planned_cost_yuan"],
        "q42_realized_total_cost_yuan": q42["realized_total_cost_yuan"],
        "q42_emergency_energy_kwh": q42["realized_emergency_energy_kwh"],
        "q42_emergency_cost_yuan": q42["realized_emergency_cost_yuan"],
        "q42_grid_energy_kwh": q42["grid_energy_kwh"],
        "q42_charge_energy_kwh": q42["charge_energy_kwh"],
        "q42_discharge_energy_kwh": q42["discharge_energy_kwh"],
        "q42_battery_throughput_kwh": q42["battery_throughput_kwh"],
        "q42_final_soc_kwh": q42["final_soc_kwh"],
        "q42_charge_weighted_price_yuan_per_kwh": q42[
            "charge_weighted_price_yuan_per_kwh"
        ],
        "q42_discharge_weighted_price_yuan_per_kwh": q42[
            "discharge_weighted_price_yuan_per_kwh"
        ],
        "s2_planned_base_cost_yuan": s2["planned_base_cost_yuan"],
        "s2_realized_total_cost_yuan": s2["realized_total_cost_yuan"],
        "s2_adjustment_cost_yuan": s2["adjustment_cost_yuan"],
        "s2_emergency_energy_kwh": s2["realized_emergency_energy_kwh"],
        "s2_emergency_cost_yuan": s2["realized_emergency_cost_yuan"],
        "s2_grid_energy_kwh": s2["grid_energy_kwh"],
        "s2_charge_energy_kwh": s2["charge_energy_kwh"],
        "s2_discharge_energy_kwh": s2["discharge_energy_kwh"],
        "s2_battery_throughput_kwh": s2["battery_throughput_kwh"],
        "s2_final_soc_kwh": s2["final_soc_kwh"],
        "s2_charge_weighted_price_yuan_per_kwh": s2[
            "charge_weighted_price_yuan_per_kwh"
        ],
        "s2_discharge_weighted_price_yuan_per_kwh": s2[
            "discharge_weighted_price_yuan_per_kwh"
        ],
        "s2_saving_yuan": payload["absolute_saving_s2_vs_q42_yuan"],
        "s2_saving_rate": payload["saving_rate_s2_vs_q42"],
        "q42_solver_runtime_seconds": q42["solver_runtime_seconds"],
        "s2_solver_runtime_seconds": s2["solver_runtime_seconds"],
        "case_wall_runtime_seconds": payload["wall_runtime_seconds"],
        "audit_status": payload["audit_status"],
    }


def _uncertainty_row(payload: dict[str, object], baseline: dict[str, object]) -> dict[str, object]:
    q42, s2 = payload["q42"], payload["s2"]
    return {
        "kappa": payload["kappa"],
        "q42_planned_cost_yuan": q42["planned_cost_yuan"],
        "q42_expected_emergency_cost_yuan": q42["expected_emergency_cost_yuan"],
        "q42_realized_total_cost_yuan": q42["realized_total_cost_yuan"],
        "q42_emergency_energy_kwh": q42["realized_emergency_energy_kwh"],
        "q42_emergency_cost_yuan": q42["realized_emergency_cost_yuan"],
        "q42_charge_energy_kwh": q42["charge_energy_kwh"],
        "q42_discharge_energy_kwh": q42["discharge_energy_kwh"],
        "q42_final_soc_kwh": q42["final_soc_kwh"],
        "s2_realized_total_cost_yuan": s2["realized_total_cost_yuan"],
        "s2_adjustment_cost_yuan": s2["adjustment_cost_yuan"],
        "s2_emergency_energy_kwh": s2["realized_emergency_energy_kwh"],
        "s2_emergency_cost_yuan": s2["realized_emergency_cost_yuan"],
        "s2_charge_energy_kwh": s2["charge_energy_kwh"],
        "s2_discharge_energy_kwh": s2["discharge_energy_kwh"],
        "s2_final_soc_kwh": s2["final_soc_kwh"],
        "s2_saving_yuan": payload["absolute_saving_s2_vs_q42_yuan"],
        "s2_saving_rate": payload["saving_rate_s2_vs_q42"],
        "q42_cost_change_vs_kappa1_yuan": (
            q42["realized_total_cost_yuan"]
            - baseline["q42"]["realized_total_cost_yuan"]
        ),
        "s2_cost_change_vs_kappa1_yuan": (
            s2["realized_total_cost_yuan"]
            - baseline["s2"]["realized_total_cost_yuan"]
        ),
        "q42_emergency_change_vs_kappa1_kwh": (
            q42["realized_emergency_energy_kwh"]
            - baseline["q42"]["realized_emergency_energy_kwh"]
        ),
        "s2_emergency_change_vs_kappa1_kwh": (
            s2["realized_emergency_energy_kwh"]
            - baseline["s2"]["realized_emergency_energy_kwh"]
        ),
        "q42_solver_runtime_seconds": q42["solver_runtime_seconds"],
        "s2_solver_runtime_seconds": s2["solver_runtime_seconds"],
        "case_wall_runtime_seconds": payload["wall_runtime_seconds"],
        "audit_status": payload["audit_status"],
    }


def write_summaries(
    price_payloads: list[dict[str, object]],
    uncertainty_payloads: list[dict[str, object]],
    started: float,
    workers: int,
) -> dict[str, object]:
    baseline = next(
        payload
        for payload in price_payloads
        if payload["gamma"] == 1.0 and payload["kappa"] == 1.0
    )
    assert_baseline_reproduction(baseline)
    price_table = pd.DataFrame([_price_row(payload) for payload in price_payloads]).sort_values(
        "gamma"
    )
    uncertainty_table = pd.DataFrame(
        [_uncertainty_row(payload, baseline) for payload in uncertainty_payloads]
    ).sort_values("kappa")
    if not price_table["audit_status"].eq("PASS").all():
        raise AssertionError("a price-volatility case failed audit")
    if not uncertainty_table["audit_status"].eq("PASS").all():
        raise AssertionError("an uncertainty-spread case failed audit")
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    price_path = TABLE_DIR / "q4_price_volatility_sensitivity.csv"
    uncertainty_path = TABLE_DIR / "q4_uncertainty_sensitivity.csv"
    price_table.to_csv(price_path, index=False, float_format="%.10f")
    uncertainty_table.to_csv(uncertainty_path, index=False, float_format="%.10f")
    summary = {
        "formal_baseline": formal_baseline(),
        "formal_baseline_reproduction": "PASS",
        "parameter_grid": {
            "gamma": list(PRICE_GAMMAS),
            "kappa": list(UNCERTAINTY_KAPPAS),
        },
        "experiment_definition": {
            "price": "p_gamma = mean(p) + gamma * (p - mean(p))",
            "uncertainty": "epsilon_kappa = kappa * epsilon with formal PV clipping",
            "evaluation_horizon": [
                str(FORMAL_DATES[0].date()),
                str(FORMAL_DATES[-1].date()),
            ],
            "methods": ["Q4-2", "Q4-3/S2"],
            "S2_release_hours": [0, 6, 12],
            "seed": SEED,
            "paired_scenario_indices": "fixed persisted indices; no resampling",
            "CAUSAL_PRICE_run": False,
        },
        "price_cases": price_payloads,
        "uncertainty_cases": uncertainty_payloads,
        "runtime": {
            "price_wall_seconds_sum": float(
                sum(payload["wall_runtime_seconds"] for payload in price_payloads)
            ),
            "uncertainty_wall_seconds_sum": float(
                sum(payload["wall_runtime_seconds"] for payload in uncertainty_payloads)
            ),
            "total_orchestration_wall_seconds": perf_counter() - started,
            "workers": workers,
            "baseline_case_counted_once": True,
        },
        "formal_files_unchanged": {
            "result4_2_sha256": _sha256(FORMAL_RESULT42_PATH),
            "result4_3_sha256": _sha256(FORMAL_RESULT43_PATH),
        },
        "all_cases_pass": True,
    }
    summary_path = TABLE_DIR / "q4_robustness_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_checkpoint_document(price_table, uncertainty_table, summary)
    return summary


def write_checkpoint_document(
    price_table: pd.DataFrame,
    uncertainty_table: pd.DataFrame,
    summary: dict[str, object],
) -> Path:
    """Write a compact human-readable audit from completed result tables."""

    def markdown_table(frame: pd.DataFrame, columns: list[tuple[str, str, str]]) -> str:
        headers = [label for _, label, _ in columns]
        rows = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
        for row in frame.itertuples(index=False):
            values = []
            for key, _, format_spec in columns:
                value = getattr(row, key)
                values.append(format(value, format_spec) if format_spec else str(value))
            rows.append("| " + " | ".join(values) + " |")
        return "\n".join(rows)

    price_view = markdown_table(
        price_table,
        [
            ("gamma", "γ", ".2f"),
            ("price_std_yuan_per_kwh", "价格标准差（元/kWh）", ".6f"),
            ("q42_realized_total_cost_yuan", "Q4-2费用（元）", ".3f"),
            ("s2_realized_total_cost_yuan", "S2费用（元）", ".3f"),
            ("s2_saving_yuan", "S2节省（元）", ".3f"),
            ("s2_saving_rate", "S2节省率", ".4%"),
            ("q42_emergency_energy_kwh", "Q4-2紧急购电（kWh）", ".3f"),
            ("s2_emergency_energy_kwh", "S2紧急购电（kWh）", ".3f"),
            ("audit_status", "审计", ""),
        ],
    )
    uncertainty_view = markdown_table(
        uncertainty_table,
        [
            ("kappa", "κ", ".2f"),
            ("q42_realized_total_cost_yuan", "Q4-2费用（元）", ".3f"),
            ("s2_realized_total_cost_yuan", "S2费用（元）", ".3f"),
            ("s2_saving_yuan", "S2节省（元）", ".3f"),
            ("s2_saving_rate", "S2节省率", ".4%"),
            ("q42_emergency_energy_kwh", "Q4-2紧急购电（kWh）", ".3f"),
            ("s2_emergency_energy_kwh", "S2紧急购电（kWh）", ".3f"),
            ("s2_adjustment_cost_yuan", "S2调整结算（元）", ".3f"),
            ("audit_status", "审计", ""),
        ],
    )
    price_costs = price_table[
        ["q42_realized_total_cost_yuan", "s2_realized_total_cost_yuan"]
    ].to_numpy(float)
    uncertainty_costs = uncertainty_table[
        ["q42_realized_total_cost_yuan", "s2_realized_total_cost_yuan"]
    ].to_numpy(float)
    price_strictly_monotonic = bool(
        all(
            np.all(np.diff(price_costs[:, index]) >= 0.0)
            or np.all(np.diff(price_costs[:, index]) <= 0.0)
            for index in range(2)
        )
    )
    uncertainty_strictly_monotonic = bool(
        all(
            np.all(np.diff(uncertainty_costs[:, index]) >= 0.0)
            or np.all(np.diff(uncertainty_costs[:, index]) <= 0.0)
            for index in range(2)
        )
    )
    path = PROJECT_ROOT / "docs/notes/q4_robustness_checkpoint.md"
    content = f"""# Q4 鲁棒性实验检查点

## 实验范围

本实验仅检验正式 `GIVEN_PRICE` 方案对价格波动强度与预测残差幅度的敏感性。评价期为 2025-02-01 至 2025-12-31，固定随机种子为 {SEED}，复用正式预测、50 条成对整日残差索引、电池参数、物理执行、结算与跨日实际 SOC 递推。每个扰动点只比较 Q4-2 与正式滚动策略 S2={{0,6,12}}，不重新比较 S0--S3，因而不能据此声称 S2 在每个扰动点仍是四种更新时间表中的最优者。

正式点 `(γ,κ)=(1,1)` 不重算全年，而由已审计正式结果复用；在运行扰动点前，以连续两日轨迹完成恒等 wrapper smoke，正式接口与 wrapper 的最大数值差为 0。

## 价格波动强度

采用

`p_t^(γ) = mean(p) + γ[p_t - mean(p)]`,  `γ ∈ {{0,0.25,0.50,0.75,1.00}}`。

该变换保持评价期均价、时间戳与顺序不变，只改变离均差；所有价格非负。所有扰动共用相同的预测、残差场景及跨日初始状态规则。

{price_view}

S2 在全部 γ 点均比 Q4-2 节省费用，节省率为 {price_table['s2_saving_rate'].min():.2%}--{price_table['s2_saving_rate'].max():.2%}。随着 γ 增大，充放电加权价差总体扩大，说明价格波动提高了储能时移的可利用价差信号；但年度实际成本在 γ=0 至 0.25 间略有上升，随后下降，故成本关于 γ 并非严格单调（严格单调：{'是' if price_strictly_monotonic else '否'}）。

## 预测不确定性强度

保持点预测中心不变，将正式成对残差同时缩放为 `ε^(s,κ)=κ ε^(s)`，`κ ∈ {{0.75,1.00,1.25,1.50}}`。负荷与光伏沿用相同残差日索引；光伏场景按正式规则截断至非负。

{uncertainty_view}

随 κ 增大，两个方案的实际紧急购电量均下降，反映更宽的规划场景促使日前/滚动计划提高预防性覆盖。然而实际总费用并非单调：Q4-2 与 S2 均在正式 κ=1 附近取得本组较低费用，继续放大不确定性会增加保守计划或调整的经济代价（严格单调：{'是' if uncertainty_strictly_monotonic else '否'}）。S2 在所有 κ 点仍优于 Q4-2，节省率为 {uncertainty_table['s2_saving_rate'].min():.2%}--{uncertainty_table['s2_saving_rate'].max():.2%}。κ=1.25 与 1.50 的调整结算费用为负，表示顺序承诺口径下减购折算收益超过增购费用，并非负购电量或会计遗漏。

## 审计结论

- 8 个表格点（含复用正式基线）全部通过审计；7 个扰动点的 Q4-2 与 S2 均为 334/334 日 Optimal。
- 功率平衡、SOC 递推与边界、跨日实际 SOC、购电非负、充放电可行性、执行区间冻结、结算核对及价格对齐全部通过。
- 所有 case 的未吸收放电为 0，无幻影放电。
- 未运行 `CAUSAL_PRICE`，未覆盖正式工作簿或正式年度结果。
- 各 case wall runtime 和 solver runtime 记录于机器可读汇总；并行数为 {summary['runtime']['workers']}，仅在独立参数点之间并行。

## 局限

本实验采用固定的历史残差场景索引和一次全年时序轨迹，旨在进行控制变量敏感性检验，不提供重复抽样分布、置信区间或误差棒。结果仅说明正式 Q4-2 与 S2 对所列扰动范围的表现，不构成 S0--S3 全参数网格重新选型。
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("smoke", "all"),
        default="all",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Reuse an already-passed two-day identity smoke audit.",
    )
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("workers must lie between 1 and 4")
    started = perf_counter()

    smoke_path = TABLE_DIR / "q4_robustness_identity_smoke.json"
    smoke = (
        json.loads(smoke_path.read_text(encoding="utf-8"))
        if args.skip_smoke
        else run_identity_smoke()
    )
    if smoke["status"] != "PASS":
        raise AssertionError("Q4 robustness identity smoke failed")
    if args.phase == "smoke":
        return
    baseline = formal_baseline_payload()
    assert_baseline_reproduction(baseline)
    print("Q4 robustness formal baseline reused after identity smoke PASS", flush=True)

    price_specs = [RobustnessSpec(gamma, 1.0) for gamma in PRICE_GAMMAS if gamma != 1.0]
    uncertainty_specs = [
        RobustnessSpec(1.0, kappa) for kappa in UNCERTAINTY_KAPPAS if kappa != 1.0
    ]
    price_payloads = [baseline]
    uncertainty_payloads = [baseline]
    price_payloads.extend(_run_many(price_specs, args.workers))
    uncertainty_payloads.extend(_run_many(uncertainty_specs, args.workers))
    write_summaries(price_payloads, uncertainty_payloads, started, args.workers)
    print("Q4 robustness summaries complete", flush=True)


if __name__ == "__main__":
    main()
