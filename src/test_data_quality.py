"""Regression checks for the C-problem step-1 raw-data audit."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.common.data_validation import (
    EXPECTED_DAYS,
    EXPECTED_SLOTS,
    audit_all_attachments,
    parse_interval_end_offset,
)


def main() -> None:
    assert parse_interval_end_offset("00:10:00") == pd.Timedelta(minutes=10)
    assert parse_interval_end_offset("0:00+1") == pd.Timedelta(days=1)

    audit = audit_all_attachments()
    summary = audit.summary.set_index("dataset")
    assert int(summary.loc["attachment1", "observed_points"]) == EXPECTED_SLOTS
    assert int(summary.loc["attachment2_load", "observed_points"]) == EXPECTED_DAYS * EXPECTED_SLOTS
    assert int(summary.loc["attachment2_pv", "observed_points"]) == EXPECTED_DAYS * EXPECTED_SLOTS
    assert int(summary.loc["attachment3_pv_forecast", "observed_points"]) == EXPECTED_DAYS * 4 * 24
    assert int(summary.loc["attachment4_price", "observed_points"]) == EXPECTED_DAYS * EXPECTED_SLOTS
    assert int(summary["measurement_missing"].sum()) == 0
    assert int(summary["duplicate_timestamps"].sum()) == 0
    assert int(summary["unexpected_time_gaps"].sum()) == 0
    assert audit.report["raw_files_unchanged"] is True
    print("Data-quality checks passed.")


if __name__ == "__main__":
    main()
