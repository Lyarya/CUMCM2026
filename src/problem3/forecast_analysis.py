"""Metrics for official Attachment-3 and Q3 candidate forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.problem3.forecast_data import LEAD_BIN_LABELS, RELEASE_HOURS


def metric_row(actual: pd.Series | np.ndarray, forecast: pd.Series | np.ndarray) -> dict[str, float | int]:
    truth = np.asarray(actual, dtype=float)
    pred = np.asarray(forecast, dtype=float)
    valid = np.isfinite(truth) & np.isfinite(pred)
    if not valid.any():
        return {"n": 0, "mae_kw": np.nan, "rmse_kw": np.nan, "bias_kw": np.nan, "nrmse_pct": np.nan}
    error = pred[valid] - truth[valid]
    scale = float(np.max(truth[valid]))
    rmse = float(np.sqrt(np.mean(np.square(error))))
    return {
        "n": int(valid.sum()),
        "mae_kw": float(np.mean(np.abs(error))),
        "rmse_kw": rmse,
        "bias_kw": float(np.mean(error)),
        "nrmse_pct": float(100.0 * rmse / scale) if scale > 0 else np.nan,
    }


def official_metric_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matched = frame.loc[frame["actual_pv_kw"].notna()].copy()
    overall_rows = []
    for subset, rows in (("全部时段", matched), ("光伏出力时段", matched.loc[matched["daylight_flag"]])):
        overall_rows.append({"subset": subset, **metric_row(rows["actual_pv_kw"], rows["forecast_pv_kw"])})
    lead_rows = []
    for label in LEAD_BIN_LABELS:
        rows = matched.loc[matched["lead_bin"] == label]
        lead_rows.append({"lead_bin": label, "subset": "全部时段", **metric_row(rows["actual_pv_kw"], rows["forecast_pv_kw"])})
        active = rows.loc[rows["daylight_flag"]]
        lead_rows.append({"lead_bin": label, "subset": "光伏出力时段", **metric_row(active["actual_pv_kw"], active["forecast_pv_kw"])})
    release_rows = []
    for release in RELEASE_HOURS:
        rows = matched.loc[matched["release_hour"] == release]
        release_rows.append({"release_hour": release, "subset": "全部时段", **metric_row(rows["actual_pv_kw"], rows["forecast_pv_kw"])})
        active = rows.loc[rows["daylight_flag"]]
        release_rows.append({"release_hour": release, "subset": "光伏出力时段", **metric_row(active["actual_pv_kw"], active["forecast_pv_kw"])})
    return pd.DataFrame(overall_rows), pd.DataFrame(lead_rows), pd.DataFrame(release_rows)


def comparison_metrics(frame: pd.DataFrame, method_columns: dict[str, str]) -> pd.DataFrame:
    """Return common-sample overall, daylight, lead-bin and release metrics."""

    rows: list[dict[str, object]] = []
    scopes: list[tuple[str, object, pd.Series]] = [
        ("overall", "all", pd.Series(True, index=frame.index)),
        ("daylight", "actual_pv_gt_0", frame["daylight_flag"].astype(bool)),
    ]
    scopes.extend(("lead_bin", label, frame["lead_bin"].eq(label)) for label in LEAD_BIN_LABELS)
    scopes.extend(("release_hour", str(hour), frame["release_hour"].eq(hour)) for hour in RELEASE_HOURS)
    for period, period_rows in frame.groupby("period", sort=False):
        for scope_type, scope_value, base_mask in scopes:
            mask = base_mask.loc[period_rows.index]
            sample = period_rows.loc[mask]
            for method, column in method_columns.items():
                rows.append(
                    {
                        "period": period,
                        "scope_type": scope_type,
                        "scope_value": scope_value,
                        "method": method,
                        **metric_row(sample["actual_pv_kw"], sample[column]),
                    }
                )
    return pd.DataFrame(rows)


__all__ = ["comparison_metrics", "metric_row", "official_metric_tables"]
