"""Oracle/causal price-interface and regression-protection tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from src.problem4 import price_interface

from src.problem2.forecast_interface import SCENARIO_PATH, audit_q2_handoff_integrity
from src.problem4.price_interface import (
    PriceInformationMode,
    get_q4_2_price_inputs,
    get_q4_3_price_inputs,
    get_q4_price_inputs,
)


def test_information_classification_is_ambiguous_and_both_modes_exist() -> None:
    audit = json.loads(
        Path("results/problem4/tables/q4_price_information_audit.json").read_text(encoding="utf-8")
    )
    assert audit["classification"] == "AMBIGUOUS"
    assert audit["explicit_future_price_availability_statement_found"] is False
    assert audit["oracle_mode_supported"] is True
    assert audit["causal_forecast_mode_supported"] is True


@pytest.mark.parametrize("release_hour,executed_count", [(0, 0), (6, 36), (12, 72), (18, 108)])
def test_q4_price_interface_has_exact_grid_and_freezes_past(release_hour: int, executed_count: int) -> None:
    inputs = get_q4_price_inputs(
        "2025-06-20",
        release_hour=release_hour,
        mode=PriceInformationMode.CAUSAL_FORECAST,
    )
    assert inputs.interval_end.shape == (144,)
    assert inputs.price_yuan_per_kwh.shape == (144,)
    assert inputs.executed_mask.sum() == executed_count
    assert inputs.future_mask.sum() == 144 - executed_count
    assert not np.isfinite(inputs.price_yuan_per_kwh[inputs.executed_mask]).any()
    assert np.isfinite(inputs.price_yuan_per_kwh[inputs.future_mask]).all()
    assert inputs.actual_history_cutoff == np.datetime64(f"2025-06-20T{release_hour:02d}:00")
    assert inputs.dt_hours == pytest.approx(1 / 6)


def test_q4_2_and_q4_3_wrappers_require_explicit_information_mode() -> None:
    with pytest.raises(TypeError):
        get_q4_price_inputs("2025-06-20")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        get_q4_2_price_inputs("2025-06-20")  # type: ignore[call-arg]
    day_ahead = get_q4_2_price_inputs(
        "2025-06-20", mode=PriceInformationMode.ORACLE_PERFECT_INFORMATION
    )
    rolling = get_q4_3_price_inputs(
        "2025-06-20", 12, mode=PriceInformationMode.CAUSAL_FORECAST
    )
    assert day_ahead.source_method == "附件4完美信息价格"
    assert rolling.executed_mask.sum() == 72


def test_corrupted_timestamps_are_rejected(monkeypatch) -> None:
    frame = price_interface._load_predictions()
    frame.loc[frame["operating_date"].eq(pd.Timestamp("2025-06-20")), "interval_end"] += pd.Timedelta(minutes=1)
    monkeypatch.setattr(price_interface, "_load_predictions", lambda: frame)
    with pytest.raises(AssertionError, match="timestamps"):
        get_q4_price_inputs("2025-06-20", mode="causal_forecast")


def test_oracle_matches_attachment4_and_causal_path_is_frozen_intraday() -> None:
    frame = price_interface._load_predictions()
    observed = frame.loc[frame["operating_date"].eq(pd.Timestamp("2025-06-20")), "price_yuan_per_kwh"]
    oracle = get_q4_price_inputs("2025-06-20", mode="oracle_perfect_information")
    np.testing.assert_array_equal(oracle.price_yuan_per_kwh, observed)
    initial = get_q4_price_inputs("2025-06-20", mode="causal_forecast")
    updated = get_q4_price_inputs("2025-06-20", release_hour=12, mode="causal_forecast")
    np.testing.assert_array_equal(
        updated.price_yuan_per_kwh[updated.future_mask],
        initial.price_yuan_per_kwh[updated.future_mask],
    )


def test_protected_raw_q2_q3_artifacts_match_checkpoint() -> None:
    from src.common.data_validation import file_sha256
    audit = json.loads(Path("results/problem4/tables/q4_price_checkpoint.json").read_text())
    assert audit["q2_q3_files_unchanged"]
    # The Q4 checkpoint proves no Q2/Q3 file changed during that run.  A later
    # reviewed Q2 milestone may legitimately update those files.
    assert audit["q2_q3_protected_hashes_before"] == audit["q2_q3_protected_hashes_after"]
    for name, digest in audit["all_raw_hashes_before"].items():
        assert file_sha256(Path("data/raw/C题/附件") / name) == digest
    assert file_sha256(SCENARIO_PATH) == audit["q2_scenario_sha256_before"]


def test_q2_scenario_and_stage2a_selection_remain_frozen() -> None:
    before = hashlib.sha256(SCENARIO_PATH.read_bytes()).hexdigest()
    audit = audit_q2_handoff_integrity()
    after = hashlib.sha256(SCENARIO_PATH.read_bytes()).hexdigest()
    assert before == after
    assert audit["q2_scenario_hash_unchanged"] is True
    assert audit["formal_pv_forecaster"] == "7-day same-slot mean"
