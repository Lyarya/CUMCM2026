"""Final, strictly causal forecasting checkpoint before Stage 2B."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import torch

from .data import DailyPanel
from .evaluation import forecast_metrics
from .final_models import (
    CausalScaler,
    DLinear,
    HankelPolynomialExpert,
    ResidualMLP,
    StructuralResidualHybrid,
    fit_torch_regressor,
    parameter_count,
)
from .model import last_week, same_slot_mean, seasonal_phase, yesterday
from .ssa import SSARecurrentForecaster


MODEL_ORDER = (
    "Yesterday",
    "Last Week",
    "7-day same-slot mean",
    "Seasonal/Phase",
    "SSA best config",
    "Polynomial analytical-only teacher",
    "DLinear",
    "StructuralResidualHybrid",
    "PhaseStructuralFusion",
)


@dataclass(frozen=True)
class FinalCheckpointConfig:
    day_steps: int = 144
    validation_start_index: int = 17
    validation_end_index: int = 30
    formal_start_index: int = 31
    lookback_days: tuple[int, ...] = (3, 7, 14)
    dlinear_kernels: tuple[int, ...] = (25, 49, 73)
    normalizations: tuple[str, ...] = ("raw_kw", "causal_zscore")
    phase_weights: tuple[float, ...] = tuple(i / 20 for i in range(21))
    fusion_weights: tuple[float, ...] = tuple(i / 20 for i in range(21))
    teacher_geometry: tuple[tuple[int, str], ...] = (
        (3, "144"),
        (3, "half"),
        (7, "144"),
        (7, "half"),
        (14, "144"),
        (14, "half"),
    )
    hidden_units: int = 16
    epochs: int = 40
    patience: int = 6
    learning_rate: float = 1e-3
    seed: int = 2026


@dataclass(frozen=True)
class FinalCheckpointRun:
    tables: dict[str, pd.DataFrame]
    predictions: dict[str, np.ndarray]
    validation_indices: np.ndarray
    evaluation_indices: np.ndarray
    selected_model: str
    best_hindsight_model: str
    metadata: dict[str, object]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metric_bundle(actual: np.ndarray, forecast: np.ndarray) -> dict[str, float | int | str]:
    truth = np.asarray(actual, dtype=float)
    pred = np.asarray(forecast, dtype=float)
    full = forecast_metrics(truth, pred)
    active = truth > 0
    effective = forecast_metrics(truth[active], pred[active])

    def cv(metrics: dict[str, float | int], values: np.ndarray) -> tuple[float, float, str]:
        denominator = float(np.mean(values)) if values.size else np.nan
        if not np.isfinite(denominator) or abs(denominator) <= 1e-12:
            return np.nan, denominator, "mean_actual_near_zero"
        return float(metrics["rmse_kw"]) / denominator, denominator, "ok"

    full_cv, full_mean, full_status = cv(full, truth.reshape(-1))
    active_cv, active_mean, active_status = cv(effective, truth[active])
    return {
        "full_count": int(full["count"]),
        "full_mae_kw": float(full["mae_kw"]),
        "full_rmse_kw": float(full["rmse_kw"]),
        "full_bias_kw": float(full["bias_kw"]),
        "full_mean_actual_kw": full_mean,
        "full_cvrmse": full_cv,
        "full_cvrmse_status": full_status,
        "effective_count": int(effective["count"]),
        "effective_mae_kw": float(effective["mae_kw"]),
        "effective_rmse_kw": float(effective["rmse_kw"]),
        "effective_bias_kw": float(effective["bias_kw"]),
        "effective_mean_actual_kw": active_mean,
        "effective_cvrmse": active_cv,
        "effective_cvrmse_status": active_status,
    }


def _phase_map(values: np.ndarray, indices: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "Yesterday": np.stack([yesterday(values, int(i)) for i in indices]),
        "Last Week": np.stack([last_week(values, int(i)) for i in indices]),
        "7-day same-slot mean": np.stack([same_slot_mean(values, int(i)) for i in indices]),
    }


def _supervised(values: np.ndarray, target_end: int, lookback_days: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    targets = np.arange(lookback_days, int(target_end), dtype=int)
    x = np.stack([values[j - lookback_days : j].reshape(-1) for j in targets])
    y = values[targets].copy()
    return x, y, targets


def _fit_dlinear(
    values: np.ndarray,
    origin: int,
    *,
    lookback_days: int,
    kernel: int,
    normalization: str,
    config: FinalCheckpointConfig,
) -> tuple[DLinear, CausalScaler, dict[str, object]]:
    x, y, target_indices = _supervised(values, origin, lookback_days)
    permitted = values[:origin]
    scaler = CausalScaler.fit(permitted) if normalization == "causal_zscore" else CausalScaler(0.0, 1.0)
    train_target_max = int(target_indices.max())
    if train_target_max >= origin or origin - 1 >= origin:
        raise AssertionError("training or scaler timestamp reaches the target day")
    torch.manual_seed(config.seed)
    model = DLinear(lookback_days * config.day_steps, config.day_steps, kernel)
    fitted = fit_torch_regressor(
        model,
        scaler.transform(x)[..., None],
        scaler.transform(y)[..., None],
        seed=config.seed,
        epochs=config.epochs,
        patience=config.patience,
        learning_rate=config.learning_rate,
    )
    audit = {
        "origin_index": origin,
        "lookback_days": lookback_days,
        "kernel_size": kernel,
        "normalization": normalization,
        "training_samples": len(x),
        "training_target_max_index": train_target_max,
        "scaler_max_day_index": origin - 1,
        "scaler_mean_kw": scaler.mean,
        "scaler_std_kw": scaler.std,
        "best_epoch": fitted.best_epoch,
        "train_loss": fitted.train_loss,
        "internal_validation_loss": fitted.validation_loss,
        "parameter_count": parameter_count(model),
        "causal_assertions_passed": True,
    }
    return model, scaler, audit


def _predict_dlinear(model: DLinear, scaler: CausalScaler, history: np.ndarray) -> np.ndarray:
    x = torch.as_tensor(scaler.transform(history)[None, :, None], dtype=torch.float32)
    with torch.no_grad():
        values = model(x).numpy()[0, :, 0]
    return np.maximum(scaler.inverse(values), 0.0)


def _teacher(values: np.ndarray, target: int, days: int, rows: int) -> tuple[np.ndarray, np.ndarray]:
    history = values[target - days : target].reshape(-1)
    result = HankelPolynomialExpert(
        lookback=len(history), horizon=values.shape[1], hankel_rows=rows
    ).forecast(history)
    return result.forecast, result.reconstructed_history


def _fit_hybrid(
    values: np.ndarray,
    origin: int,
    *,
    lookback_days: int,
    rows: int,
    config: FinalCheckpointConfig,
    teacher_cache: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]],
) -> tuple[StructuralResidualHybrid, dict[str, object]]:
    x, y, target_indices = _supervised(values, origin, lookback_days)
    rx, ry = [], []
    for sample_x, sample_y, target in zip(x, y, target_indices):
        key = (int(target), lookback_days, rows)
        if key not in teacher_cache:
            teacher_cache[key] = _teacher(values, int(target), lookback_days, rows)
        analytical, reconstructed = teacher_cache[key]
        rx.append(sample_x - reconstructed)
        ry.append(sample_y - analytical)
    rx_array, ry_array = np.stack(rx), np.stack(ry)
    scaler = CausalScaler.fit(np.concatenate([rx_array.reshape(-1), ry_array.reshape(-1)]))
    torch.manual_seed(config.seed)
    residual = ResidualMLP(lookback_days * config.day_steps, config.day_steps, config.hidden_units)
    fitted = fit_torch_regressor(
        residual,
        scaler.transform(rx_array),
        scaler.transform(ry_array),
        seed=config.seed,
        epochs=config.epochs,
        patience=config.patience,
        learning_rate=config.learning_rate,
    )
    expert = HankelPolynomialExpert(
        lookback=lookback_days * config.day_steps,
        horizon=config.day_steps,
        hankel_rows=rows,
    )
    audit = {
        "origin_index": origin,
        "lookback_days": lookback_days,
        "hankel_rows": rows,
        "training_samples": len(x),
        "training_target_max_index": int(target_indices.max()),
        "scaler_max_day_index": origin - 1,
        "residual_scaler_mean_kw": scaler.mean,
        "residual_scaler_std_kw": scaler.std,
        "best_epoch": fitted.best_epoch,
        "train_loss": fitted.train_loss,
        "internal_validation_loss": fitted.validation_loss,
        "parameter_count": parameter_count(residual),
        "causal_assertions_passed": bool(int(target_indices.max()) < origin),
    }
    return StructuralResidualHybrid(expert, residual, scaler), audit


def _select_weight(actual: np.ndarray, first: np.ndarray, second: np.ndarray, weights: tuple[float, ...], name: str) -> tuple[float, pd.DataFrame]:
    rows = []
    for weight in weights:
        pred = weight * first + (1 - weight) * second
        rows.append({name: weight, "second_weight": 1 - weight, **forecast_metrics(actual, pred)})
    table = pd.DataFrame(rows)
    selected = float(table.sort_values(["rmse_kw", "mae_kw", name], kind="stable").iloc[0][name])
    return selected, table


def run_final_checkpoint(
    panel: DailyPanel,
    stage2_predictions: pd.DataFrame,
    ssa_predictions: pd.DataFrame,
    config: FinalCheckpointConfig | None = None,
) -> FinalCheckpointRun:
    """Select on January only, freeze once, and evaluate February--December."""
    cfg = FinalCheckpointConfig() if config is None else config
    values = panel.generation_kw
    validation = np.arange(cfg.validation_start_index, cfg.validation_end_index + 1)
    evaluation = np.arange(cfg.formal_start_index, panel.n_days)
    actual_val, actual_oos = values[validation], values[evaluation]
    if panel.dates[validation[0]] != pd.Timestamp("2025-01-18") or len(evaluation) != 334:
        raise ValueError("unexpected fixed calendar")

    # Fixed baselines and January-only phase weight.
    val_preds = _phase_map(values, validation)
    oos_preds = _phase_map(values, evaluation)
    phase_weight, phase_table = _select_weight(
        actual_val, val_preds["Yesterday"], val_preds["Last Week"], cfg.phase_weights, "yesterday_weight"
    )
    val_preds["Seasonal/Phase"] = seasonal_phase(values, int(validation[0]), phase_weight)[None, :]
    val_preds["Seasonal/Phase"] = phase_weight * val_preds["Yesterday"] + (1 - phase_weight) * val_preds["Last Week"]
    oos_preds["Seasonal/Phase"] = phase_weight * oos_preds["Yesterday"] + (1 - phase_weight) * oos_preds["Last Week"]

    # Previously selected SSA configuration, recomputed on the common January dates.
    ssa = SSARecurrentForecaster(horizon=cfg.day_steps, embedding_dimension=144)
    val_preds["SSA best config"] = np.stack([
        ssa.forecast_ranks(values[i - 14 : i].reshape(-1), (5,))[5].forecast for i in validation
    ])
    expected_dates = np.repeat(panel.dates[evaluation].strftime("%Y-%m-%d"), cfg.day_steps)
    if not np.array_equal(ssa_predictions["operating_date"].astype(str), expected_dates):
        raise ValueError("SSA prediction artifact is not aligned to the formal period")
    oos_preds["SSA best config"] = ssa_predictions["ssa_pred"].to_numpy().reshape(len(evaluation), cfg.day_steps)

    # DLinear grid and strict normalization audit.
    dlinear_rows, scaling_rows = [], []
    dlinear_val_predictions: dict[tuple[int, int, str], np.ndarray] = {}
    for days in cfg.lookback_days:
        for kernel in cfg.dlinear_kernels:
            for normalization in cfg.normalizations:
                predictions = []
                for origin in validation:
                    model, scaler, audit = _fit_dlinear(
                        values, int(origin), lookback_days=days, kernel=kernel,
                        normalization=normalization, config=cfg,
                    )
                    audit["origin_date"] = panel.dates[origin].date().isoformat()
                    audit["scaler_max_timestamp"] = str(panel.datetimes[origin - 1, -1])
                    audit["training_target_max_timestamp"] = str(
                        panel.datetimes[int(audit["training_target_max_index"]), -1]
                    )
                    audit["target_start_timestamp"] = str(panel.datetimes[origin, 0])
                    scaling_rows.append(audit)
                    history = values[origin - days : origin].reshape(-1)
                    predictions.append(_predict_dlinear(model, scaler, history))
                stacked = np.stack(predictions)
                dlinear_val_predictions[(days, kernel, normalization)] = stacked
                dlinear_rows.append({
                    "scope": "january_common_origin_validation",
                    "lookback_days": days, "kernel_size": kernel,
                    "normalization": normalization,
                    **forecast_metrics(actual_val, stacked),
                })
    dlinear_validation = pd.DataFrame(dlinear_rows)
    best_z = dlinear_validation.loc[dlinear_validation["normalization"] == "causal_zscore"].sort_values(
        ["rmse_kw", "mae_kw", "lookback_days", "kernel_size"], kind="stable"
    ).iloc[0]
    corresponding_raw = dlinear_validation.loc[
        (dlinear_validation["normalization"] == "raw_kw")
        & (dlinear_validation["lookback_days"] == best_z["lookback_days"])
        & (dlinear_validation["kernel_size"] == best_z["kernel_size"])
    ].iloc[0]
    zscore_material_harm = bool(best_z["rmse_kw"] > 1.05 * corresponding_raw["rmse_kw"])
    selected_norm = "raw_kw" if zscore_material_harm else "causal_zscore"
    best_dl = dlinear_validation.loc[dlinear_validation["normalization"] == selected_norm].sort_values(
        ["rmse_kw", "mae_kw", "lookback_days", "kernel_size"], kind="stable"
    ).iloc[0]
    dl_key = (int(best_dl["lookback_days"]), int(best_dl["kernel_size"]), str(best_dl["normalization"]))
    val_preds["DLinear"] = dlinear_val_predictions[dl_key]
    final_dl, final_dl_scaler, final_dl_audit = _fit_dlinear(
        values, cfg.formal_start_index, lookback_days=dl_key[0], kernel=dl_key[1],
        normalization=dl_key[2], config=cfg,
    )
    oos_preds["DLinear"] = np.stack([
        _predict_dlinear(final_dl, final_dl_scaler, values[i - dl_key[0] : i].reshape(-1)) for i in evaluation
    ])
    dlinear_oos = pd.DataFrame([{
        "scope": "february_december_frozen_model", "lookback_days": dl_key[0],
        "kernel_size": dl_key[1], "normalization": dl_key[2],
        "parameter_count": parameter_count(final_dl), **_metric_bundle(actual_oos, oos_preds["DLinear"]),
    }])

    # January-only teacher geometry and hybrid geometry audit.
    teacher_cache: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}
    geometry_rows, hybrid_audits = [], []
    geometry_predictions: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for days, row_mode in cfg.teacher_geometry:
        lookback = days * cfg.day_steps
        rows = 144 if row_mode == "144" else lookback // 2
        analytical_predictions, hybrid_predictions = [], []
        for origin in validation:
            key = (int(origin), days, rows)
            if key not in teacher_cache:
                teacher_cache[key] = _teacher(values, int(origin), days, rows)
            analytical_predictions.append(teacher_cache[key][0])
            hybrid, audit = _fit_hybrid(
                values, int(origin), lookback_days=days, rows=rows,
                config=cfg, teacher_cache=teacher_cache,
            )
            audit["origin_date"] = panel.dates[origin].date().isoformat()
            hybrid_audits.append(audit)
            history = values[origin - days : origin].reshape(-1)
            hybrid_predictions.append(hybrid.predict(history))
        analytical_array, hybrid_array = np.stack(analytical_predictions), np.stack(hybrid_predictions)
        geometry_predictions[(days, rows)] = (analytical_array, hybrid_array)
        analytical_metrics = forecast_metrics(actual_val, analytical_array)
        hybrid_metrics = forecast_metrics(actual_val, hybrid_array)
        geometry_rows.append({
            "scope": "january_common_origin_validation", "lookback_days": days,
            "hankel_rows": rows, "row_mode": row_mode, "rank": 1,
            "polynomial_degree": 2, "polynomial_ridge": 5.0,
            "analytical_mae_kw": analytical_metrics["mae_kw"],
            "analytical_rmse_kw": analytical_metrics["rmse_kw"],
            "hybrid_mae_kw": hybrid_metrics["mae_kw"],
            "hybrid_rmse_kw": hybrid_metrics["rmse_kw"],
        })
    geometry = pd.DataFrame(geometry_rows)
    best_geometry = geometry.sort_values(
        ["hybrid_rmse_kw", "hybrid_mae_kw", "lookback_days", "hankel_rows"], kind="stable"
    ).iloc[0]
    geometry_key = (int(best_geometry["lookback_days"]), int(best_geometry["hankel_rows"]))
    val_preds["Polynomial analytical-only teacher"] = geometry_predictions[geometry_key][0]
    val_preds["StructuralResidualHybrid"] = geometry_predictions[geometry_key][1]
    final_hybrid, final_hybrid_audit = _fit_hybrid(
        values, cfg.formal_start_index, lookback_days=geometry_key[0], rows=geometry_key[1],
        config=cfg, teacher_cache=teacher_cache,
    )
    teacher_final = final_hybrid.expert
    teacher_oos, hybrid_oos = [], []
    for origin in evaluation:
        history = values[origin - geometry_key[0] : origin].reshape(-1)
        teacher_oos.append(teacher_final.forecast(history).forecast)
        hybrid_oos.append(final_hybrid.predict(history))
    oos_preds["Polynomial analytical-only teacher"] = np.stack(teacher_oos)
    oos_preds["StructuralResidualHybrid"] = np.stack(hybrid_oos)
    hybrid_oos_table = pd.DataFrame([{
        "scope": "february_december_frozen_model", "lookback_days": geometry_key[0],
        "hankel_rows": geometry_key[1], "rank": 1, "hidden_units": cfg.hidden_units,
        "parameter_count": parameter_count(final_hybrid.residual_model),
        **_metric_bundle(actual_oos, oos_preds["StructuralResidualHybrid"]),
    }])

    fusion_weight, fusion_table = _select_weight(
        actual_val, val_preds["Seasonal/Phase"], val_preds["StructuralResidualHybrid"],
        cfg.fusion_weights, "phase_weight",
    )
    val_preds["PhaseStructuralFusion"] = fusion_weight * val_preds["Seasonal/Phase"] + (1 - fusion_weight) * val_preds["StructuralResidualHybrid"]
    oos_preds["PhaseStructuralFusion"] = fusion_weight * oos_preds["Seasonal/Phase"] + (1 - fusion_weight) * oos_preds["StructuralResidualHybrid"]

    # Compact 9-row final table, with selection based on January only.
    metadata = {
        "Yesterday": (0, "none"), "Last Week": (0, "none"),
        "7-day same-slot mean": (0, "none"), "Seasonal/Phase": (0, "none"),
        "SSA best config": (0, "local_mean_centering"),
        "Polynomial analytical-only teacher": (0, "local_mean_centering"),
        "DLinear": (parameter_count(final_dl), dl_key[2]),
        "StructuralResidualHybrid": (parameter_count(final_hybrid.residual_model), "causal_residual_zscore"),
        "PhaseStructuralFusion": (parameter_count(final_hybrid.residual_model), "component_specific"),
    }
    comparison_rows = []
    for model in MODEL_ORDER:
        vm = _metric_bundle(actual_val, val_preds[model])
        om = _metric_bundle(actual_oos, oos_preds[model])
        comparison_rows.append({
            "model": model, "parameter_count": metadata[model][0], "normalization": metadata[model][1],
            "validation_start": "2025-01-18", "validation_end": "2025-01-31",
            "validation_mae_kw": vm["full_mae_kw"], "validation_rmse_kw": vm["full_rmse_kw"],
            "validation_bias_kw": vm["full_bias_kw"], "validation_cvrmse": vm["full_cvrmse"],
            "validation_effective_mae_kw": vm["effective_mae_kw"],
            "validation_effective_rmse_kw": vm["effective_rmse_kw"],
            "validation_effective_bias_kw": vm["effective_bias_kw"],
            "validation_effective_cvrmse": vm["effective_cvrmse"],
            "oos_mae_kw": om["full_mae_kw"], "oos_rmse_kw": om["full_rmse_kw"],
            "oos_bias_kw": om["full_bias_kw"], "oos_cvrmse": om["full_cvrmse"],
            "oos_effective_mae_kw": om["effective_mae_kw"],
            "oos_effective_rmse_kw": om["effective_rmse_kw"],
            "oos_effective_bias_kw": om["effective_bias_kw"],
            "oos_effective_cvrmse": om["effective_cvrmse"],
            "selected_by_january": False,
        })
    comparison = pd.DataFrame(comparison_rows)
    selected = str(comparison.sort_values(["validation_rmse_kw", "validation_mae_kw"], kind="stable").iloc[0]["model"])
    comparison.loc[comparison["model"] == selected, "selected_by_january"] = True
    hindsight = str(comparison.sort_values(["oos_rmse_kw", "oos_mae_kw"], kind="stable").iloc[0]["model"])

    daily_rows = []
    baseline = oos_preds["7-day same-slot mean"]
    for model in ("DLinear", "StructuralResidualHybrid", "PhaseStructuralFusion"):
        model_daily, base_daily = [], []
        for position, day in enumerate(panel.dates[evaluation]):
            mm = forecast_metrics(actual_oos[position], oos_preds[model][position])
            bm = forecast_metrics(actual_oos[position], baseline[position])
            model_daily.append(float(mm["rmse_kw"])); base_daily.append(float(bm["rmse_kw"]))
            daily_rows.append({
                "date": day.date().isoformat(), "model": model,
                "model_rmse_kw": mm["rmse_kw"], "seven_day_rmse_kw": bm["rmse_kw"],
                "rmse_difference_model_minus_7day_kw": float(mm["rmse_kw"]) - float(bm["rmse_kw"]),
                "model_wins": bool(float(mm["rmse_kw"]) < float(bm["rmse_kw"])),
            })
    daily = pd.DataFrame(daily_rows)
    pairwise_rows = []
    for model, group in daily.groupby("model", sort=False):
        statistic, pvalue = wilcoxon(group["model_rmse_kw"], group["seven_day_rmse_kw"], alternative="two-sided")
        diff = group["rmse_difference_model_minus_7day_kw"].to_numpy()
        pairwise_rows.append({
            "model": model, "paired_days": len(group),
            "mean_daily_rmse_kw": group["model_rmse_kw"].mean(),
            "median_daily_rmse_kw": group["model_rmse_kw"].median(),
            "seven_day_mean_daily_rmse_kw": group["seven_day_rmse_kw"].mean(),
            "seven_day_median_daily_rmse_kw": group["seven_day_rmse_kw"].median(),
            "win_rate": group["model_wins"].mean(), "median_difference_kw": np.median(diff),
            "wilcoxon_statistic": statistic, "wilcoxon_p_value_two_sided": pvalue,
        })

    normalization_audit = pd.DataFrame([{
        "selected_normalization": selected_norm,
        "best_zscore_rmse_kw": best_z["rmse_kw"],
        "same_config_raw_rmse_kw": corresponding_raw["rmse_kw"],
        "material_harm_threshold": 0.05, "zscore_material_harm": zscore_material_harm,
        "final_scaler_fit_end": str(panel.datetimes[cfg.formal_start_index - 1, -1]),
        "final_training_target_end": str(panel.datetimes[cfg.formal_start_index - 1, -1]),
        "formal_target_start": str(panel.datetimes[cfg.formal_start_index, 0]),
        "formal_scaler_frozen": True, "formal_gradients_disabled": True,
        "hybrid_residual_scaler_fit_end": str(panel.datetimes[cfg.formal_start_index - 1, -1]),
        "hybrid_residual_scaler_frozen": True,
    }])
    selection = pd.DataFrame([{
        "selection_period": "2025-01-18/2025-01-31", "formal_period": "2025-02-01/2025-12-31",
        "selection_rule": "minimum pooled January RMSE; MAE breaks exact ties",
        "selected_generation_forecaster": selected,
        "selected_validation_rmse_kw": float(comparison.loc[comparison.model == selected, "validation_rmse_kw"].iloc[0]),
        "selected_oos_rmse_kw": float(comparison.loc[comparison.model == selected, "oos_rmse_kw"].iloc[0]),
        "best_hindsight_oos_model": hindsight,
        "best_hindsight_oos_rmse_kw": float(comparison.loc[comparison.model == hindsight, "oos_rmse_kw"].iloc[0]),
        "formal_model_and_scaler_frozen": True, "online_learning": False,
        "official_attachment3_used": False, "regenerate_q2_scenarios": selected != "7-day same-slot mean",
    }])
    hybrid_config = pd.DataFrame([{
        "lookback_days": geometry_key[0], "hankel_rows": geometry_key[1], "rank": 1,
        "polynomial_degree": 2, "polynomial_ridge": 5.0, "hidden_units": cfg.hidden_units,
        "activation": "GELU", "training_loss": "MSE", "optimizer": "AdamW",
        "seed": cfg.seed, "formal_parameter_count": parameter_count(final_hybrid.residual_model),
        "generic_fixed_hybrid": True, "reproduces_unpublished_architecture": False,
    }])
    tables = {
        "dlinear_validation_metrics.csv": dlinear_validation,
        "dlinear_oos_metrics.csv": dlinear_oos,
        "dlinear_scaling_audit.csv": pd.DataFrame(scaling_rows),
        "analytical_geometry_audit.csv": geometry,
        "hybrid_validation_metrics.csv": geometry[["lookback_days", "hankel_rows", "hybrid_mae_kw", "hybrid_rmse_kw"]].copy(),
        "hybrid_oos_metrics.csv": hybrid_oos_table,
        "hybrid_config.csv": hybrid_config,
        "phase_validation_metrics.csv": phase_table,
        "phase_fusion_weights.csv": fusion_table,
        "forecast_model_final_comparison.csv": comparison,
        "forecast_daily_pairwise_tests.csv": pd.DataFrame(pairwise_rows),
        "forecast_daily_errors.csv": daily,
        "normalization_audit.csv": normalization_audit,
        "forecast_selection_summary.csv": selection,
    }
    meta = {
        "phase_weight": phase_weight, "fusion_phase_weight": fusion_weight,
        "dlinear_config": dl_key, "teacher_geometry": geometry_key,
        "dlinear_final_audit": final_dl_audit, "hybrid_final_audit": final_hybrid_audit,
        "validation_dates": panel.dates[validation], "evaluation_dates": panel.dates[evaluation],
    }
    return FinalCheckpointRun(tables, {"validation": val_preds, "oos": oos_preds}, validation, evaluation, selected, hindsight, meta)
