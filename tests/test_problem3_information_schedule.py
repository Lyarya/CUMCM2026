"""Q3 information schedules, update interface and settlement-schema tests."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from src.problem2.forecast_interface import SCENARIO_PATH, audit_q2_handoff_integrity
from src.problem3.information_schedule import (
    SCHEDULES,
    SettlementMode,
    adjustment_settlement,
    empty_voi_table,
    get_q3_forecast_update,
)


def test_information_schedules_are_exact() -> None:
    assert SCHEDULES == {
        "S0": (0,), "S1": (0, 6), "S2": (0, 6, 12), "S3": (0, 6, 12, 18)
    }


def test_update_freezes_executed_intervals_and_requires_realized_soc() -> None:
    with pytest.raises(TypeError):
        get_q3_forecast_update("2025-06-20", 6)  # type: ignore[call-arg]
    update = get_q3_forecast_update("2025-06-20", 6, current_realized_soc=5_432.1)
    assert update.current_realized_soc == pytest.approx(5_432.1)
    assert update.executed_mask.sum() == 36
    assert update.future_mask.sum() == 108
    assert not np.isfinite(update.fused_forecast_kw[update.executed_mask]).any()
    assert np.isfinite(update.fused_forecast_kw[update.future_mask]).all()
    assert update.actual_history_cutoff == np.datetime64("2025-06-20T06:00")


def test_settlement_interface_supports_both_audited_conventions() -> None:
    base = np.array([10.0, 10.0])
    revisions = np.array([[12.0, 8.0], [11.0, 9.0]])
    price = np.array([1.0, 2.0])
    sequential = adjustment_settlement(base, revisions, price, SettlementMode.SEQUENTIAL_PREVIOUS_COMMITMENT)
    original = adjustment_settlement(base, revisions, price, SettlementMode.ORIGINAL_00_COMMITMENT)
    assert sequential["delta_plus_kwh"].shape == (2, 2)
    assert original["delta_plus_kwh"].shape == (1, 2)
    np.testing.assert_array_equal(original["final_commitment_kwh"], revisions[-1])
    assert not np.isclose(sequential["adjustment_cost_yuan"].sum(), original["adjustment_cost_yuan"].sum())


def test_voi_interface_contains_schema_but_no_fake_rows() -> None:
    table = empty_voi_table()
    assert table.empty
    assert {"schedule", "voi_vs_s0_yuan", "incremental_voi_yuan"}.issubset(table.columns)


def test_q2_scenario_and_stage2a_selection_remain_frozen() -> None:
    before = hashlib.sha256(SCENARIO_PATH.read_bytes()).hexdigest()
    audit = audit_q2_handoff_integrity()
    after = hashlib.sha256(SCENARIO_PATH.read_bytes()).hexdigest()
    assert before == after
    assert audit["q2_scenario_hash_unchanged"] is True
    assert audit["formal_pv_forecaster"] == "7-day same-slot mean"
