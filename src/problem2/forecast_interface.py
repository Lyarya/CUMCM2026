"""Read-only Stage 2B interface for the frozen Q2 forecast artifacts.

The interface deliberately exposes forecasts and paired scenarios, never the
future actual columns that are retained in the Stage 2A evaluation table.
It performs no forecasting, scenario generation, or dispatch optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.paths import PROCESSED_DATA_DIR, PROJECT_ROOT, RESULTS_DIR


HORIZON = 144
DT_HOURS = 1.0 / 6.0
FORMAL_LOAD_FORECASTER = "Last Week"
FORMAL_PV_FORECASTER = "7-day same-slot mean"

FORECAST_DIR = RESULTS_DIR / "tables" / "forecasting"
FORMAL_FORECAST_PATH = FORECAST_DIR / "q2_forecast_predictions.csv"
SCENARIO_PATH = FORECAST_DIR / "q2_scenarios.npz"
SCENARIO_MANIFEST_PATH = FORECAST_DIR / "q2_scenario_manifest.csv"
SELECTION_PATH = FORECAST_DIR / "forecast_selection_summary.csv"
MODEL_DECISION_PATH = FORECAST_DIR / "q2_model_decision.json"
LEAKAGE_AUDIT_PATH = FORECAST_DIR / "q2_leakage_audit.json"
FINAL_AUDIT_PATH = FORECAST_DIR / "forecast_final_leakage_audit.json"
PRICE_PATH = PROCESSED_DATA_DIR / "C题" / "problem1_day.csv"
RAW_ATTACHMENT_DIR = PROJECT_ROOT / "data" / "raw" / "C题" / "附件"


@dataclass(frozen=True)
class Q2DayInputs:
    """Optimizer inputs for one operating day.

    Units are kW for power, kWh for ``initial_energy``, yuan/kWh for
    ``price``, and hours for ``dt_hours``.  ``scenario_source_dates`` is one
    shared provenance vector: row ``s`` of both scenario arrays comes from
    that same residual-day draw.
    """

    date: Date
    load_forecast: np.ndarray
    pv_forecast: np.ndarray
    load_scenarios: np.ndarray
    pv_scenarios: np.ndarray
    price: np.ndarray
    initial_energy: float
    timestamps: np.ndarray
    scenario_source_dates: np.ndarray
    dt_hours: float = DT_HOURS

    @property
    def horizon(self) -> int:
        return int(self.load_forecast.shape[0])

    @property
    def scenario_count(self) -> int:
        return int(self.load_scenarios.shape[0])


@dataclass(frozen=True)
class _FormalArtifacts:
    forecasts: pd.DataFrame
    target_dates: np.ndarray
    source_residual_dates: np.ndarray
    load_scenarios: np.ndarray
    pv_scenarios: np.ndarray
    price: np.ndarray
    scenario_count: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _readonly(values: np.ndarray, *, dtype: object | None = None) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@lru_cache(maxsize=1)
def _load_formal_artifacts() -> _FormalArtifacts:
    """Load and validate the frozen Stage 2A artifacts exactly once."""
    final_audit = _read_json(FINAL_AUDIT_PATH)
    leakage_audit = _read_json(LEAKAGE_AUDIT_PATH)
    decision = _read_json(MODEL_DECISION_PATH)
    selection = pd.read_csv(SELECTION_PATH)

    expected_hash = final_audit["protected_artifact_hashes_before"][SCENARIO_PATH.name]
    if _sha256(SCENARIO_PATH) != expected_hash:
        raise AssertionError("q2_scenarios.npz differs from the frozen Stage 2A artifact")
    if final_audit["protected_artifact_hashes_before"] != final_audit[
        "protected_artifact_hashes_after"
    ]:
        raise AssertionError("the Stage 2A protected-artifact audit is inconsistent")

    if len(selection) != 1:
        raise AssertionError("forecast_selection_summary.csv must contain one selection row")
    if selection.iloc[0]["selected_generation_forecaster"] != FORMAL_PV_FORECASTER:
        raise AssertionError("the formal PV forecaster is no longer 7-day same-slot mean")
    if decision["selected_load_forecaster"] != FORMAL_LOAD_FORECASTER:
        raise AssertionError("the formal load forecaster changed")
    if decision["selected_generation_forecaster"] != "7-day mean":
        raise AssertionError("the legacy Q2 decision file no longer identifies the 7-day mean")
    if not bool(final_audit["strict_causal_protocol"]):
        raise AssertionError("the final forecasting audit does not certify strict causality")
    if not bool(leakage_audit["scenario_source_dates_strictly_before_target"]):
        raise AssertionError("the scenario audit does not certify causal source dates")

    forecasts = pd.read_csv(
        FORMAL_FORECAST_PATH,
        parse_dates=["operating_date", "datetime"],
    ).sort_values(["operating_date", "datetime"], kind="stable")
    required_columns = {
        "operating_date",
        "datetime",
        "forecast_load",
        "forecast_generation",
        "load_last_week",
        "same_slot_7d_mean",
    }
    missing = required_columns.difference(forecasts.columns)
    if missing:
        raise AssertionError(f"formal forecast table is missing columns: {sorted(missing)}")
    if forecasts["datetime"].duplicated().any():
        raise AssertionError("formal forecast timestamps must be unique")
    if not np.isfinite(
        forecasts[["forecast_load", "forecast_generation"]].to_numpy(dtype=float)
    ).all():
        raise AssertionError("formal point forecasts must be finite")
    if not np.allclose(forecasts["forecast_load"], forecasts["load_last_week"]):
        raise AssertionError("formal load values no longer match the selected Last Week model")
    if not np.allclose(
        forecasts["forecast_generation"], forecasts["same_slot_7d_mean"]
    ):
        raise AssertionError("formal PV values no longer match the selected 7-day mean")

    with np.load(SCENARIO_PATH, allow_pickle=False) as archive:
        required_keys = {
            "target_dates",
            "source_residual_dates",
            "load_kw",
            "generation_kw",
        }
        if set(archive.files) != required_keys:
            raise AssertionError("unexpected q2_scenarios.npz schema")
        target_dates = archive["target_dates"].astype("datetime64[D]", copy=True)
        source_dates = archive["source_residual_dates"].astype("datetime64[D]", copy=True)
        load_scenarios = archive["load_kw"].copy()
        pv_scenarios = archive["generation_kw"].copy()

    if load_scenarios.shape != pv_scenarios.shape or load_scenarios.ndim != 3:
        raise AssertionError("load and PV scenarios must share one [day, scenario, time] shape")
    day_count, scenario_count, horizon = load_scenarios.shape
    if horizon != HORIZON:
        raise AssertionError(f"forecast horizon must be {HORIZON}, got {horizon}")
    if target_dates.shape != (day_count,):
        raise AssertionError("scenario target-date dimension does not match scenario arrays")
    if source_dates.shape != (day_count, scenario_count):
        raise AssertionError("one shared source-date index is required for every paired scenario")
    if not np.isfinite(load_scenarios).all() or not np.isfinite(pv_scenarios).all():
        raise AssertionError("scenario arrays contain NaN or infinite values")
    if (pv_scenarios < 0).any():
        raise AssertionError("PV scenarios violate the physical lower bound zero")
    if not (source_dates < target_dates[:, None]).all():
        raise AssertionError("a residual scenario uses its target day or a future day")

    formal_dates = forecasts["operating_date"].drop_duplicates().to_numpy(dtype="datetime64[D]")
    if not np.array_equal(formal_dates, target_dates):
        raise AssertionError("forecast and scenario target dates are not aligned")
    daily_counts = forecasts.groupby("operating_date", sort=True).size().to_numpy()
    if not (daily_counts == HORIZON).all():
        raise AssertionError("every formal forecast day must contain 144 intervals")
    for target_date, daily in forecasts.groupby("operating_date", sort=True):
        expected = pd.date_range(
            pd.Timestamp(target_date) + pd.Timedelta(minutes=10),
            periods=HORIZON,
            freq="10min",
        ).to_numpy(dtype="datetime64[ns]")
        observed = daily["datetime"].to_numpy(dtype="datetime64[ns]")
        if not np.array_equal(observed, expected):
            raise AssertionError(f"forecast timestamps are misaligned for {target_date.date()}")

    manifest = pd.read_csv(SCENARIO_MANIFEST_PATH, parse_dates=["target_date"])
    manifest_dates = manifest["target_date"].to_numpy(dtype="datetime64[D]")
    if not np.array_equal(manifest_dates, target_dates):
        raise AssertionError("scenario manifest dates do not match the archive")
    if not (manifest["scenario_count"].to_numpy() == scenario_count).all():
        raise AssertionError("scenario count is inconsistent across the manifest")
    if not (manifest["day_steps"].to_numpy() == HORIZON).all():
        raise AssertionError("scenario manifest horizon is not 144")
    if not manifest["all_sources_strictly_before_target"].astype(bool).all():
        raise AssertionError("scenario manifest reports future leakage")
    if scenario_count != int(leakage_audit["scenario_count_per_day"]):
        raise AssertionError("scenario count differs from the leakage audit")

    price_frame = pd.read_csv(PRICE_PATH).sort_values("slot", kind="stable")
    if price_frame["slot"].tolist() != list(range(1, HORIZON + 1)):
        raise AssertionError("Q2 fixed price profile must use slots 1 through 144")
    price = price_frame["price_yuan_per_kwh"].to_numpy(dtype=float)
    if price.shape != (HORIZON,) or not np.isfinite(price).all():
        raise AssertionError("Q2 price profile must contain 144 finite yuan/kWh values")

    return _FormalArtifacts(
        forecasts=forecasts,
        target_dates=target_dates,
        source_residual_dates=source_dates,
        load_scenarios=load_scenarios,
        pv_scenarios=pv_scenarios,
        price=price,
        scenario_count=scenario_count,
    )


def get_q2_day_inputs(date: object, initial_energy: float) -> Q2DayInputs:
    """Return frozen forecasts and paired scenarios for one formal Q2 day.

    ``initial_energy`` is supplied by the caller.  Stage 2B must pass the
    previous day's terminal battery energy after the first day; this function
    intentionally does not impose Q1's daily ``6000 -> 6000`` condition.
    """
    target = pd.Timestamp(date)
    if pd.isna(target):
        raise ValueError("date must be a valid operating date")
    target = target.normalize()
    initial = float(initial_energy)
    if not np.isfinite(initial):
        raise ValueError("initial_energy must be finite and expressed in kWh")

    artifacts = _load_formal_artifacts()
    target_day = np.datetime64(target.date(), "D")
    positions = np.flatnonzero(artifacts.target_dates == target_day)
    if positions.size != 1:
        first = str(artifacts.target_dates[0])
        last = str(artifacts.target_dates[-1])
        raise KeyError(f"{target.date()} is outside the formal Q2 period {first} to {last}")
    position = int(positions[0])
    daily = artifacts.forecasts.loc[
        artifacts.forecasts["operating_date"] == target
    ].sort_values("datetime", kind="stable")

    timestamps = daily["datetime"].to_numpy(dtype="datetime64[ns]")
    load_forecast = daily["forecast_load"].to_numpy(dtype=float)
    pv_forecast = daily["forecast_generation"].to_numpy(dtype=float)
    load_scenarios = artifacts.load_scenarios[position]
    pv_scenarios = artifacts.pv_scenarios[position]
    source_dates = artifacts.source_residual_dates[position]

    if load_forecast.shape != (HORIZON,) or pv_forecast.shape != (HORIZON,):
        raise AssertionError("point forecast shape changed after date selection")
    if load_scenarios.shape != (artifacts.scenario_count, HORIZON):
        raise AssertionError("load scenario shape changed after date selection")
    if pv_scenarios.shape != load_scenarios.shape:
        raise AssertionError("paired load/PV scenario shapes differ")
    if not np.isfinite(
        np.concatenate(
            [
                load_forecast,
                pv_forecast,
                load_scenarios.reshape(-1),
                pv_scenarios.reshape(-1),
                artifacts.price,
            ]
        )
    ).all():
        raise AssertionError("optimizer input contains NaN or infinite values")
    if (pv_scenarios < 0).any():
        raise AssertionError("PV scenario is negative")
    if not (source_dates < target_day).all():
        raise AssertionError("scenario source date is not strictly before its target")

    return Q2DayInputs(
        date=target.date(),
        load_forecast=_readonly(load_forecast, dtype=float),
        pv_forecast=_readonly(pv_forecast, dtype=float),
        load_scenarios=_readonly(load_scenarios),
        pv_scenarios=_readonly(pv_scenarios),
        price=_readonly(artifacts.price, dtype=float),
        initial_energy=initial,
        timestamps=_readonly(timestamps),
        scenario_source_dates=_readonly(source_dates),
    )


def audit_q2_handoff_integrity() -> dict[str, object]:
    """Recompute the Stage 2A hash and causality checks used by handoff tests."""
    artifacts = _load_formal_artifacts()
    final_audit = _read_json(FINAL_AUDIT_PATH)
    protected_expected = final_audit["protected_artifact_hashes_before"]
    protected_current = {
        name: _sha256(FORECAST_DIR / name) for name in protected_expected
    }
    raw_expected = final_audit["raw_excel_hashes_before"]
    raw_current = {
        name: _sha256(RAW_ATTACHMENT_DIR / name) for name in raw_expected
    }
    source_is_causal = bool(
        (
            artifacts.source_residual_dates
            < artifacts.target_dates[:, None]
        ).all()
    )
    return {
        "formal_load_forecaster": FORMAL_LOAD_FORECASTER,
        "formal_pv_forecaster": FORMAL_PV_FORECASTER,
        "formal_start": str(artifacts.target_dates[0]),
        "formal_end": str(artifacts.target_dates[-1]),
        "formal_day_count": int(artifacts.target_dates.size),
        "forecast_horizon": HORIZON,
        "scenario_count": artifacts.scenario_count,
        "paired_scenarios": artifacts.load_scenarios.shape
        == artifacts.pv_scenarios.shape,
        "leakage_audit": source_is_causal
        and bool(final_audit["strict_causal_protocol"]),
        "protected_artifact_hashes_unchanged": protected_current
        == protected_expected,
        "q2_scenario_hash_unchanged": protected_current[SCENARIO_PATH.name]
        == protected_expected[SCENARIO_PATH.name],
        "raw_data_hashes_unchanged": raw_current == raw_expected,
    }


__all__ = [
    "DT_HOURS",
    "FORMAL_LOAD_FORECASTER",
    "FORMAL_PV_FORECASTER",
    "HORIZON",
    "Q2DayInputs",
    "audit_q2_handoff_integrity",
    "get_q2_day_inputs",
]
