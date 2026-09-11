"""Contract tests for the frozen Q2 forecast-to-optimizer handoff."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from src.problem2.forecast_interface import (
    DT_HOURS,
    FINAL_AUDIT_PATH,
    FORMAL_FORECAST_PATH,
    FORMAL_LOAD_FORECASTER,
    FORMAL_PV_FORECASTER,
    HORIZON,
    SCENARIO_PATH,
    audit_q2_handoff_integrity,
    get_q2_day_inputs,
)


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_handoff_hashes_models_and_formal_period_are_frozen() -> None:
    before = _sha256(SCENARIO_PATH)
    audit = audit_q2_handoff_integrity()
    after = _sha256(SCENARIO_PATH)

    assert before == after
    assert audit["q2_scenario_hash_unchanged"] is True
    assert audit["protected_artifact_hashes_unchanged"] is True
    assert audit["raw_data_hashes_unchanged"] is True
    assert audit["formal_load_forecaster"] == FORMAL_LOAD_FORECASTER == "Last Week"
    assert audit["formal_pv_forecaster"] == FORMAL_PV_FORECASTER
    assert FORMAL_PV_FORECASTER == "7-day same-slot mean"
    assert audit["formal_start"] == "2025-02-01"
    assert audit["formal_end"] == "2025-12-31"
    assert audit["formal_day_count"] == 334
    assert audit["forecast_horizon"] == HORIZON == 144
    assert audit["scenario_count"] == 50
    assert audit["paired_scenarios"] is True
    assert audit["leakage_audit"] is True


@pytest.mark.parametrize("target", ["2025-02-01", "2025-07-15", "2025-12-31"])
def test_day_input_shapes_timestamps_units_and_causality(target: str) -> None:
    inputs = get_q2_day_inputs(target, initial_energy=6_000.0)
    expected_timestamps = pd.date_range(
        pd.Timestamp(target) + pd.Timedelta(minutes=10),
        periods=HORIZON,
        freq="10min",
    ).to_numpy(dtype="datetime64[ns]")

    assert inputs.date == pd.Timestamp(target).date()
    assert inputs.horizon == HORIZON
    assert inputs.scenario_count == 50
    assert inputs.load_forecast.shape == (144,)
    assert inputs.pv_forecast.shape == (144,)
    assert inputs.load_scenarios.shape == (50, 144)
    assert inputs.pv_scenarios.shape == (50, 144)
    assert inputs.price.shape == (144,)
    assert inputs.scenario_source_dates.shape == (50,)
    assert inputs.dt_hours == pytest.approx(DT_HOURS)
    assert inputs.initial_energy == pytest.approx(6_000.0)
    assert np.array_equal(inputs.timestamps, expected_timestamps)
    assert np.isfinite(inputs.load_forecast).all()
    assert np.isfinite(inputs.pv_forecast).all()
    assert np.isfinite(inputs.load_scenarios).all()
    assert np.isfinite(inputs.pv_scenarios).all()
    assert np.isfinite(inputs.price).all()
    assert (inputs.pv_scenarios >= 0).all()
    assert (inputs.scenario_source_dates < np.datetime64(target, "D")).all()


def test_interface_preserves_one_shared_paired_scenario_index() -> None:
    target = "2025-10-04"
    inputs = get_q2_day_inputs(target, initial_energy=5_432.1)
    with np.load(SCENARIO_PATH, allow_pickle=False) as archive:
        position = int(
            np.flatnonzero(archive["target_dates"] == np.datetime64(target, "D"))[0]
        )
        assert np.array_equal(inputs.load_scenarios, archive["load_kw"][position])
        assert np.array_equal(inputs.pv_scenarios, archive["generation_kw"][position])
        assert np.array_equal(
            inputs.scenario_source_dates,
            archive["source_residual_dates"][position],
        )

    # A single provenance row indexes both variables; no independently shuffled
    # load/PV source-date arrays exist in the formal archive or the interface.
    assert inputs.load_scenarios.shape[0] == inputs.pv_scenarios.shape[0]
    assert inputs.scenario_source_dates.shape == (inputs.scenario_count,)


def test_optimizer_inputs_exclude_future_actual_columns() -> None:
    inputs = get_q2_day_inputs("2025-02-01", initial_energy=6_000.0)
    formal = pd.read_csv(FORMAL_FORECAST_PATH)
    daily = formal.loc[formal["operating_date"] == "2025-02-01"]

    assert np.allclose(inputs.load_forecast, daily["load_last_week"])
    assert np.allclose(inputs.pv_forecast, daily["same_slot_7d_mean"])
    assert not hasattr(inputs, "actual_load")
    assert not hasattr(inputs, "actual_generation")
    assert inputs.scenario_source_dates.max() < np.datetime64(inputs.date, "D")


def test_initial_energy_is_caller_supplied_not_daily_reset() -> None:
    carried_energy = 7_321.5
    inputs = get_q2_day_inputs("2025-02-02", initial_energy=carried_energy)
    assert inputs.initial_energy == pytest.approx(carried_energy)
    assert inputs.initial_energy != 6_000.0


def test_interface_rejects_nonformal_dates_and_nonfinite_energy() -> None:
    with pytest.raises(KeyError):
        get_q2_day_inputs("2025-01-31", initial_energy=6_000.0)
    with pytest.raises(ValueError):
        get_q2_day_inputs("2025-02-01", initial_energy=np.nan)
