"""Strictly causal Q3 forecast baselines, convex fusion and selection."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.problem3.forecast_analysis import comparison_metrics, metric_row
from src.problem3.forecast_data import LEAD_BIN_LABELS, RELEASE_HOURS


CALIBRATION_START = pd.Timestamp("2025-01-08")
CALIBRATION_END = pd.Timestamp("2025-01-16 23:59:59")
VALIDATION_START = pd.Timestamp("2025-01-17")
VALIDATION_END = pd.Timestamp("2025-01-30 23:59:59")
EVALUATION_START = pd.Timestamp("2025-02-01")
METHOD_COLUMNS = {
    "官方预测": "forecast_pv_kw",
    "昨日同刻": "yesterday_pv_kw",
    "上周同刻": "last_week_pv_kw",
    "七日同刻均值": "same_slot_7d_mean_kw",
    "静态凸融合": "static_fusion_kw",
    "提前期分箱融合": "lead_aware_fusion_kw",
    "发布时间分组融合": "release_aware_fusion_kw",
    "在线因果融合": "online_causal_fusion_kw",
}


def fit_convex_weight(actual: np.ndarray, official: np.ndarray, seasonal: np.ndarray) -> float:
    """Fit the official-model weight for a two-expert convex MSE projection."""

    truth = np.asarray(actual, dtype=float)
    x = np.asarray(official, dtype=float) - np.asarray(seasonal, dtype=float)
    y = truth - np.asarray(seasonal, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    denominator = float(np.dot(x[valid], x[valid]))
    if denominator <= 0:
        return 0.5
    return float(np.clip(np.dot(x[valid], y[valid]) / denominator, 0.0, 1.0))


def _apply_weight(official: pd.Series, seasonal: pd.Series, weight: np.ndarray | float) -> np.ndarray:
    result = np.asarray(weight) * official.to_numpy(float) + (1.0 - np.asarray(weight)) * seasonal.to_numpy(float)
    return np.maximum(result, 0.0)


def causal_online_fusion(frame: pd.DataFrame, eta: float) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate official/seasonal experts, updating only after truth is available."""

    ordered = frame.sort_values(["issue_time", "lead_hours"], kind="stable")
    prediction = pd.Series(np.nan, index=frame.index, dtype=float)
    official_weight = pd.Series(np.nan, index=frame.index, dtype=float)
    log_weights = np.zeros(2, dtype=float)
    pending: list[int] = []
    for issue_time, rows in ordered.groupby("issue_time", sort=True):
        still_pending: list[int] = []
        for index in pending:
            if frame.at[index, "target_time"] <= issue_time:
                experts = frame.loc[index, ["forecast_pv_kw", "same_slot_7d_mean_kw"]].to_numpy(float)
                truth = float(frame.at[index, "actual_pv_kw"])
                if np.isfinite(experts).all() and np.isfinite(truth):
                    log_weights -= float(eta) * np.abs(experts - truth) / 6000.0
            else:
                still_pending.append(index)
        pending = still_pending
        weights = np.exp(log_weights - np.max(log_weights))
        weights /= weights.sum()
        indices = rows.index.to_numpy()
        experts = rows[["forecast_pv_kw", "same_slot_7d_mean_kw"]].to_numpy(float)
        valid = np.isfinite(experts).all(axis=1)
        values = np.full(len(rows), np.nan)
        values[valid] = experts[valid] @ weights
        prediction.loc[indices] = np.maximum(values, 0.0)
        official_weight.loc[indices] = weights[0]
        pending.extend(indices.tolist())
    return prediction.to_numpy(float), official_weight.to_numpy(float)


def _period_labels(frame: pd.DataFrame) -> pd.Series:
    issue = frame["issue_time"]
    return pd.Series(
        np.select(
            [
                issue.between(CALIBRATION_START, CALIBRATION_END),
                issue.between(VALIDATION_START, VALIDATION_END),
                issue.ge(EVALUATION_START),
            ],
            ["calibration", "validation", "evaluation"],
            default="excluded",
        ),
        index=frame.index,
    )


def _bh_fdr(pvalues: np.ndarray) -> np.ndarray:
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.clip(ranked, 0.0, 1.0)
    return adjusted


def paired_significance(frame: pd.DataFrame, selected_column: str) -> pd.DataFrame:
    """Compare daily paired MAE/RMSE on the Feb-Dec evaluation period."""

    evaluation = frame.loc[frame["period"].eq("evaluation")].copy()
    evaluation["target_date"] = evaluation["issue_time"].dt.normalize()
    rows: list[dict[str, object]] = []
    for method, column in METHOD_COLUMNS.items():
        if column == selected_column:
            continue
        for metric in ("MAE", "RMSE"):
            paired = []
            for _, day in evaluation.groupby("target_date", sort=True):
                selected_error = day[selected_column].to_numpy(float) - day["actual_pv_kw"].to_numpy(float)
                comparator_error = day[column].to_numpy(float) - day["actual_pv_kw"].to_numpy(float)
                valid = np.isfinite(selected_error) & np.isfinite(comparator_error)
                if not valid.any():
                    continue
                if metric == "MAE":
                    selected_value = float(np.mean(np.abs(selected_error[valid])))
                    comparator_value = float(np.mean(np.abs(comparator_error[valid])))
                else:
                    selected_value = float(np.sqrt(np.mean(np.square(selected_error[valid]))))
                    comparator_value = float(np.sqrt(np.mean(np.square(comparator_error[valid]))))
                paired.append((selected_value, comparator_value))
            values = np.asarray(paired, dtype=float)
            difference = values[:, 0] - values[:, 1]
            if np.allclose(difference, 0.0):
                statistic, pvalue = 0.0, 1.0
            else:
                statistic, pvalue = wilcoxon(values[:, 0], values[:, 1], zero_method="wilcox")
            rows.append(
                {
                    "selected_method": "提前期分箱融合",
                    "comparator": method,
                    "pairing_unit": "目标日期",
                    "metric": metric,
                    "n_days": int(len(values)),
                    "median_selected_minus_comparator_kw": float(np.median(difference)),
                    "wilcoxon_statistic": float(statistic),
                    "p_value": float(pvalue),
                }
            )
    table = pd.DataFrame(rows)
    table["p_value_bh_fdr"] = _bh_fdr(table["p_value"].to_numpy(float))
    return table


def build_fusion_analysis(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Fit on early January, select on late January and evaluate Feb-Dec."""

    work = frame.loc[frame["actual_pv_kw"].notna()].copy()
    work["period"] = _period_labels(work)
    common = work.loc[
        work[list(METHOD_COLUMNS.values())[:4]].notna().all(axis=1)
        & work["period"].ne("excluded")
    ].copy()
    validation = common.loc[common["period"].eq("validation")]
    seasonal_columns = {
        "昨日同刻": "yesterday_pv_kw",
        "上周同刻": "last_week_pv_kw",
        "七日同刻均值": "same_slot_7d_mean_kw",
    }
    seasonal_scores = {
        name: metric_row(validation["actual_pv_kw"], validation[column])["rmse_kw"]
        for name, column in seasonal_columns.items()
    }
    selected_seasonal = min(seasonal_scores, key=seasonal_scores.get)
    seasonal_column = seasonal_columns[selected_seasonal]
    calibration = common.loc[common["period"].eq("calibration")]
    global_weight = fit_convex_weight(
        calibration["actual_pv_kw"], calibration["forecast_pv_kw"], calibration[seasonal_column]
    )
    lead_weights = {
        label: fit_convex_weight(
            rows["actual_pv_kw"], rows["forecast_pv_kw"], rows[seasonal_column]
        )
        for label, rows in calibration.groupby("lead_bin", observed=True)
    }
    release_weights = {
        int(hour): fit_convex_weight(
            rows["actual_pv_kw"], rows["forecast_pv_kw"], rows[seasonal_column]
        )
        for hour, rows in calibration.groupby("release_hour")
    }
    common["static_fusion_kw"] = _apply_weight(
        common["forecast_pv_kw"], common[seasonal_column], global_weight
    )
    common["lead_aware_fusion_kw"] = _apply_weight(
        common["forecast_pv_kw"],
        common[seasonal_column],
        common["lead_bin"].map(lead_weights).to_numpy(float),
    )
    common["release_aware_fusion_kw"] = _apply_weight(
        common["forecast_pv_kw"],
        common[seasonal_column],
        common["release_hour"].map(release_weights).to_numpy(float),
    )
    eta_candidates = (0.5, 1.0, 2.0, 4.0, 8.0)
    eta_scores: dict[float, float] = {}
    online_cache: dict[float, np.ndarray] = {}
    for eta in eta_candidates:
        values, _ = causal_online_fusion(common, eta)
        online_cache[eta] = values
        mask = common["period"].eq("calibration")
        eta_scores[eta] = float(metric_row(common.loc[mask, "actual_pv_kw"], values[mask])["rmse_kw"])
    selected_eta = min(eta_scores, key=eta_scores.get)
    common["online_causal_fusion_kw"] = online_cache[selected_eta]

    comparison = comparison_metrics(common, METHOD_COLUMNS)
    validation_overall = comparison.loc[
        comparison["period"].eq("validation")
        & comparison["scope_type"].eq("overall")
    ].sort_values(["rmse_kw", "mae_kw"], kind="stable")
    selected_method = str(validation_overall.iloc[0]["method"])
    selected_column = METHOD_COLUMNS[selected_method]
    significance = paired_significance(common, selected_column)
    weights = pd.DataFrame(
        [
            {"weight_type": "global", "group": "all", "official_weight": global_weight},
            *[
                {"weight_type": "lead_bin", "group": label, "official_weight": lead_weights[label]}
                for label in LEAD_BIN_LABELS
            ],
            *[
                {"weight_type": "release_hour", "group": str(hour), "official_weight": release_weights[hour]}
                for hour in RELEASE_HOURS
            ],
        ]
    )
    weights["seasonal_weight"] = 1.0 - weights["official_weight"]
    validation_row = validation_overall.loc[validation_overall["method"].eq(selected_method)].iloc[0]
    evaluation_row = comparison.loc[
        comparison["period"].eq("evaluation")
        & comparison["scope_type"].eq("overall")
        & comparison["method"].eq(selected_method)
    ].iloc[0]
    selection = {
        "calibration_period": "2025-01-08/2025-01-16",
        "validation_period": "2025-01-17/2025-01-30",
        "evaluation_period": "2025-02-01/2025-12-31",
        "selected_seasonal_method": selected_seasonal,
        "selected_method": selected_method,
        "selected_column": selected_column,
        "selected_online_eta": selected_eta,
        "validation_mae_kw": float(validation_row["mae_kw"]),
        "validation_rmse_kw": float(validation_row["rmse_kw"]),
        "evaluation_mae_kw": float(evaluation_row["mae_kw"]),
        "evaluation_rmse_kw": float(evaluation_row["rmse_kw"]),
        "lead_aware": selected_method == "提前期分箱融合",
        "release_aware": selected_method == "发布时间分组融合",
        "structural_expert_included": False,
        "structural_expert_reason": "现有结构化专家没有覆盖四个发布批次的同一整点协议，故不作不公平拼接。",
        "fusion_is_causal": True,
        "lead_weights": lead_weights,
        "release_weights": {str(k): v for k, v in release_weights.items()},
        "global_official_weight": global_weight,
    }
    return common, comparison, significance, weights, selection


__all__ = [
    "EVALUATION_START",
    "METHOD_COLUMNS",
    "VALIDATION_START",
    "build_fusion_analysis",
    "causal_online_fusion",
    "fit_convex_weight",
    "paired_significance",
]
