"""Run only step 1 of C-problem preprocessing: read-only source audit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.data_utils import save_table
from src.common.data_validation import RAW_C_DIR, audit_all_attachments
from src.common.paths import PROCESSED_DATA_DIR, RESULTS_DIR


def main() -> None:
    audit = audit_all_attachments(RAW_C_DIR)
    processed_dir = PROCESSED_DATA_DIR / "C题"
    table_dir = RESULTS_DIR / "tables"
    processed_dir.mkdir(parents=True, exist_ok=True)

    report_path = processed_dir / "data_quality_step1.json"
    report_path.write_text(
        json.dumps(audit.report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_table(audit.summary, table_dir / "table_data_quality_summary.csv")
    save_table(
        audit.attachment1_comparison,
        table_dir / "table_attachment1_annual_mean_comparison.csv",
    )
    print("Step 1 complete: raw-data audit only.")
    print(audit.summary.to_string(index=False))
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
