"""Attachment-4 source, grid and descriptive-analysis tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.common.data_validation import file_sha256
from src.problem4.price_analysis import correlation_statistics
from src.problem4.price_data import RAW_ATTACHMENT4, audit_attachment4, build_price_canonical


EXPECTED_ATTACHMENT4_SHA256 = "20e9c93aeab5e8e21ae4dd15587f9e190f7408692504c1319598461cd654fe71"


def test_attachment4_raw_hash_and_complete_operational_grid() -> None:
    before = file_sha256(RAW_ATTACHMENT4)
    frame = build_price_canonical()
    audit = audit_attachment4(frame)
    after = file_sha256(RAW_ATTACHMENT4)
    assert before == after == EXPECTED_ATTACHMENT4_SHA256
    assert frame.shape[0] == 52_560
    assert frame["interval_end"].min() == np.datetime64("2025-01-01T00:10:00")
    assert frame["interval_end"].max() == np.datetime64("2026-01-01T00:00:00")
    assert audit["timestamp_alignment_pass"] is True
    assert audit["missing_price_count"] == 0
    assert audit["duplicate_timestamp_count"] == 0
    assert audit["non_10min_gap_count"] == 0
    assert audit["minimum_records_per_day"] == 144
    assert audit["maximum_records_per_day"] == 144


def test_attachment4_units_values_and_stage1_alignment() -> None:
    audit = audit_attachment4()
    assert audit["price_unit"] == "yuan/kWh"
    assert audit["negative_price_count"] == 0
    assert audit["zero_price_count"] == 0
    assert audit["minimum_price"] == 0.0076
    assert audit["maximum_price"] == 1.7936
    assert audit["raw_to_stage1_max_absolute_price_gap"] == 0.0


def test_price_net_load_correlations_are_reproducible() -> None:
    correlations = correlation_statistics(build_price_canonical())
    net = correlations.loc[correlations["variable"].eq("净负荷")].set_index("method")
    assert net.loc["Pearson", "coefficient"] == pytest.approx(0.260169920235, abs=1e-12)
    assert net.loc["Spearman", "coefficient"] == pytest.approx(0.262375795603, abs=1e-12)


def test_formal_result4_workbooks_are_separate_from_source_templates() -> None:
    # The formal run writes only to results/problem4; Attachment-5 templates
    # remain the immutable source workbooks.
    generated = [
        path for path in Path(".").rglob("result4*.xlsx")
        if "data/raw/C题/附件/附件5" not in path.as_posix()
    ]
    assert sorted(path.as_posix() for path in generated) == [
        "results/problem4/result4-2.xlsx",
        "results/problem4/result4-3.xlsx",
    ]
    for name in ("result4-2.xlsx", "result4-3.xlsx"):
        template = Path("data/raw/C题/附件/附件5") / name
        output = Path("results/problem4") / name
        assert template.exists() and output.exists()
        assert file_sha256(template) != file_sha256(output)
