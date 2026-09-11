"""Descriptive and association analysis for the dynamic-price layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def price_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return competition-facing descriptive statistics in yuan/kWh."""

    price = frame["price_yuan_per_kwh"].astype(float)
    quantiles = price.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return pd.DataFrame(
        [
            {
                "n": int(price.notna().sum()),
                "mean_yuan_per_kwh": float(price.mean()),
                "std_yuan_per_kwh": float(price.std(ddof=1)),
                "min_yuan_per_kwh": float(price.min()),
                "p01_yuan_per_kwh": float(quantiles.loc[0.01]),
                "p05_yuan_per_kwh": float(quantiles.loc[0.05]),
                "q1_yuan_per_kwh": float(quantiles.loc[0.25]),
                "median_yuan_per_kwh": float(quantiles.loc[0.5]),
                "q3_yuan_per_kwh": float(quantiles.loc[0.75]),
                "p95_yuan_per_kwh": float(quantiles.loc[0.95]),
                "p99_yuan_per_kwh": float(quantiles.loc[0.99]),
                "max_yuan_per_kwh": float(price.max()),
            }
        ]
    )


def monthly_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize monthly location, scale and range."""

    rows: list[dict[str, float | int]] = []
    for month, group in frame.groupby("month", sort=True):
        values = group["price_yuan_per_kwh"].astype(float)
        rows.append(
            {
                "month": int(month),
                "n": int(values.size),
                "mean_yuan_per_kwh": float(values.mean()),
                "std_yuan_per_kwh": float(values.std(ddof=1)),
                "coefficient_of_variation": float(values.std(ddof=1) / values.mean()),
                "min_yuan_per_kwh": float(values.min()),
                "q1_yuan_per_kwh": float(values.quantile(0.25)),
                "median_yuan_per_kwh": float(values.median()),
                "q3_yuan_per_kwh": float(values.quantile(0.75)),
                "max_yuan_per_kwh": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def intraday_profile(frame: pd.DataFrame) -> pd.DataFrame:
    """Describe every ten-minute slot and its data-derived price band."""

    grouped = frame.groupby("slot", sort=True)["price_yuan_per_kwh"]
    profile = grouped.agg(["count", "mean", "std", "min", "median", "max"]).reset_index()
    profile["q1_yuan_per_kwh"] = grouped.quantile(0.25).to_numpy()
    profile["q3_yuan_per_kwh"] = grouped.quantile(0.75).to_numpy()
    profile["p05_yuan_per_kwh"] = grouped.quantile(0.05).to_numpy()
    profile["p95_yuan_per_kwh"] = grouped.quantile(0.95).to_numpy()
    profile = profile.rename(
        columns={
            "count": "n",
            "mean": "mean_yuan_per_kwh",
            "std": "std_yuan_per_kwh",
            "min": "min_yuan_per_kwh",
            "median": "median_yuan_per_kwh",
            "max": "max_yuan_per_kwh",
        }
    )
    lower, upper = profile["mean_yuan_per_kwh"].quantile([0.25, 0.75])
    profile["data_driven_band"] = np.select(
        [profile["mean_yuan_per_kwh"].le(lower), profile["mean_yuan_per_kwh"].ge(upper)],
        ["低价时段", "高价时段"],
        default="平价时段",
    )
    profile["interval_start"] = pd.to_timedelta((profile["slot"] - 1) * 10, unit="min")
    profile["interval_end"] = pd.to_timedelta(profile["slot"] * 10, unit="min")
    return profile


def peak_offpeak_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    """Characterize data-derived high/low hours without claiming a tariff rule."""

    hourly_mean = frame.groupby("hour", sort=True)["price_yuan_per_kwh"].mean()
    lower, upper = hourly_mean.quantile([0.25, 0.75])
    band_by_hour = pd.Series("平价时段", index=hourly_mean.index)
    band_by_hour.loc[hourly_mean.le(lower)] = "低价时段"
    band_by_hour.loc[hourly_mean.ge(upper)] = "高价时段"
    work = frame.assign(data_driven_band=frame["hour"].map(band_by_hour))
    rows: list[dict[str, object]] = []
    for band in ("低价时段", "平价时段", "高价时段"):
        group = work.loc[work["data_driven_band"].eq(band)]
        hours = sorted(group["hour"].unique().tolist())
        rows.append(
            {
                "band": band,
                "hours": ",".join(f"{hour:02d}:00" for hour in hours),
                "hour_count": len(hours),
                "n": int(len(group)),
                "mean_price_yuan_per_kwh": float(group["price_yuan_per_kwh"].mean()),
                "median_price_yuan_per_kwh": float(group["price_yuan_per_kwh"].median()),
                "mean_load_kw": float(group["load_kw"].mean()),
                "mean_pv_kw": float(group["pv_actual_kw"].mean()),
                "mean_net_load_kw": float(group["net_load_kw"].mean()),
            }
        )
    return pd.DataFrame(rows)


def volatility_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute ten-minute changes and daily ranges overall and by month."""

    rows: list[dict[str, object]] = []
    groups: list[tuple[str, pd.DataFrame]] = [("全年", frame)]
    groups.extend((f"{month}月", group) for month, group in frame.groupby("month", sort=True))
    for period, group in groups:
        ordered = group.sort_values("interval_end", kind="stable")
        changes = ordered.groupby("operating_date")["price_yuan_per_kwh"].diff()
        daily_range = ordered.groupby("operating_date")["price_yuan_per_kwh"].agg(
            lambda values: values.max() - values.min()
        )
        price = ordered["price_yuan_per_kwh"]
        rows.append(
            {
                "period": period,
                "n": int(len(ordered)),
                "price_std_yuan_per_kwh": float(price.std(ddof=1)),
                "coefficient_of_variation": float(price.std(ddof=1) / price.mean()),
                "mean_absolute_10min_change": float(changes.abs().mean()),
                "std_10min_change": float(changes.std(ddof=1)),
                "maximum_absolute_10min_change": float(changes.abs().max()),
                "mean_daily_range": float(daily_range.mean()),
                "median_daily_range": float(daily_range.median()),
                "maximum_daily_range": float(daily_range.max()),
            }
        )
    return pd.DataFrame(rows)


def correlation_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    """Measure association with load, PV and net load; no causal claim is made."""

    variables = (
        ("实际负荷", "load_kw"),
        ("实际光伏", "pv_actual_kw"),
        ("净负荷", "net_load_kw"),
    )
    price = frame["price_yuan_per_kwh"].to_numpy(dtype=float)
    rows: list[dict[str, object]] = []
    for label, column in variables:
        values = frame[column].to_numpy(dtype=float)
        mask = np.isfinite(price) & np.isfinite(values)
        pearson = pearsonr(price[mask], values[mask])
        spearman = spearmanr(price[mask], values[mask])
        rows.extend(
            [
                {
                    "variable": label,
                    "column": column,
                    "method": "Pearson",
                    "n": int(mask.sum()),
                    "coefficient": float(pearson.statistic),
                    "p_value": float(pearson.pvalue),
                },
                {
                    "variable": label,
                    "column": column,
                    "method": "Spearman",
                    "n": int(mask.sum()),
                    "coefficient": float(spearman.statistic),
                    "p_value": float(spearman.pvalue),
                },
            ]
        )
    return pd.DataFrame(rows)


__all__ = [
    "correlation_statistics",
    "intraday_profile",
    "monthly_statistics",
    "peak_offpeak_statistics",
    "price_summary",
    "volatility_statistics",
]
