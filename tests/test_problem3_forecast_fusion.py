"""Q3 fusion selection, weight and leakage tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.problem3.forecast_fusion import causal_online_fusion, fit_convex_weight


TABLE_DIR = Path("results/problem3/tables")


def test_convex_weight_is_bounded() -> None:
    actual = np.array([0.0, 1.0, 2.0])
    official = np.array([0.0, 0.8, 2.2])
    seasonal = np.array([0.2, 1.2, 1.8])
    weight = fit_convex_weight(actual, official, seasonal)
    assert 0.0 <= weight <= 1.0


def test_generated_weights_are_nonnegative_and_sum_to_one() -> None:
    weights = pd.read_csv(TABLE_DIR / "q3_forecast_fusion_weights.csv")
    assert (weights[["official_weight", "seasonal_weight"]] >= 0.0).all().all()
    np.testing.assert_allclose(weights["official_weight"] + weights["seasonal_weight"], 1.0)


def test_selection_uses_pre_evaluation_validation_and_is_evidence_driven() -> None:
    selection = json.loads((TABLE_DIR / "q3_forecast_selection.json").read_text(encoding="utf-8"))
    assert selection["calibration_period"] == "2025-01-08/2025-01-16"
    assert selection["validation_period"] == "2025-01-17/2025-01-30"
    assert selection["evaluation_period"] == "2025-02-01/2025-12-31"
    assert selection["selected_seasonal_method"] == "七日同刻均值"
    assert selection["selected_method"] == "提前期分箱融合"
    assert selection["structural_expert_included"] is False
    assert selection["fusion_is_causal"] is True


def test_online_weights_update_only_after_target_truth_is_available() -> None:
    frame = pd.DataFrame(
        {
            "issue_time": pd.to_datetime(["2025-01-08 00:00", "2025-01-08 00:00", "2025-01-08 06:00"]),
            "target_time": pd.to_datetime(["2025-01-08 01:00", "2025-01-08 07:00", "2025-01-08 07:00"]),
            "lead_hours": [1, 7, 1],
            "forecast_pv_kw": [10.0, 20.0, 30.0],
            "same_slot_7d_mean_kw": [12.0, 18.0, 28.0],
            "actual_pv_kw": [11.0, 19.0, 29.0],
        }
    )
    first, _ = causal_online_fusion(frame, eta=1.0)
    altered = frame.copy()
    altered.loc[altered["target_time"] > pd.Timestamp("2025-01-08 00:00"), "actual_pv_kw"] = 999_999.0
    second, _ = causal_online_fusion(altered, eta=1.0)
    np.testing.assert_array_equal(first[:2], second[:2])
