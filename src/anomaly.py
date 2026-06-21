"""
Anomaly Detection Engine.

Detects zones that are behaving unexpectedly for the current time of day
and day of week, using two complementary approaches:

1. Statistical baseline: mean ± 2σ per (zone, hour, day_of_week)
   - Flags zones where historical counts significantly exceed the norm

2. Volatility anomalies: zones with high coefficient of variation (CV)
   - High CV = erratic enforcement pattern = unpredictable hotspot

Both types surface in the dashboard as real-time alerts.
"""

import numpy as np
import pandas as pd


def compute_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (h3_index, hour, day_of_week) compute:
      - mean_count, std_count, upper_threshold (mean + 2*std)
    Based on training data only (Nov 2023 – Apr 2024, months 11,12,1,2,3,4).
    """
    train = df[df["month"].isin([11, 12, 1, 2, 3, 4])].copy()

    daily_hourly = (
        train.groupby(["h3_index", "date", "hour", "day_of_week"])
        .size()
        .reset_index(name="count")
    )

    baselines = (
        daily_hourly.groupby(["h3_index", "hour", "day_of_week"])
        .agg(
            mean_count = ("count", "mean"),
            std_count  = ("count", "std"),
            n_obs      = ("count", "count"),
        )
        .reset_index()
    )
    baselines["std_count"]        = baselines["std_count"].fillna(0)
    baselines["upper_threshold"]  = baselines["mean_count"] + 2 * baselines["std_count"]
    return baselines


def compute_zone_volatility(df: pd.DataFrame, zones_df: pd.DataFrame) -> pd.DataFrame:
    """
    Coefficient of variation (std/mean) of daily violation counts per zone.
    High CV zones are unpredictable and warrant attention.
    """
    daily = (
        df.groupby(["h3_index", "date"])
        .size()
        .reset_index(name="daily_count")
    )
    cv = (
        daily.groupby("h3_index")["daily_count"]
        .agg(mean=("mean"), std=("std"), n_days=("count"))
        .reset_index()
    )
    cv.columns = ["h3_index", "mean_daily", "std_daily", "n_days"]
    cv["cv"]   = (cv["std_daily"] / (cv["mean_daily"] + 1e-9)).round(3)

    merged = zones_df[["h3_index", "lat", "lng", "violation_count",
                        "top_violation", "top_vehicle"]].merge(cv, on="h3_index", how="left")
    return merged.sort_values("cv", ascending=False).reset_index(drop=True)


def detect_anomalies_now(df: pd.DataFrame,
                          zones_df: pd.DataFrame,
                          baselines: pd.DataFrame,
                          top_n: int = 10) -> pd.DataFrame:
    """
    For the current hour and day of week, identify zones where historical
    counts exceeded the baseline threshold (mean + 2σ).

    Uses actual historical data at this same (hour, day_of_week) combination
    to find which zones repeatedly spiked — these are the 'anomaly alert' zones.

    Returns up to top_n anomaly zones with explanation strings.
    """
    now         = pd.Timestamp.now()
    cur_hour    = now.hour
    cur_dow     = now.dayofweek

    # Find baselines for current hour × day_of_week
    slot = baselines[
        (baselines["hour"] == cur_hour) & (baselines["day_of_week"] == cur_dow)
    ].copy()

    if slot.empty:
        # Fallback: any hour on this day of week
        slot = baselines[baselines["day_of_week"] == cur_dow].copy()
        slot = slot.groupby("h3_index", as_index=False).agg(
            mean_count       = ("mean_count", "mean"),
            upper_threshold  = ("upper_threshold", "max"),
        )

    # Actual max count at this slot from historical data
    hist = df[(df["hour"] == cur_hour) & (df["day_of_week"] == cur_dow)].copy()
    if hist.empty:
        hist = df[df["day_of_week"] == cur_dow].copy()

    actual_counts = (
        hist.groupby(["h3_index", "date"])
        .size()
        .groupby("h3_index")
        .max()
        .reset_index(name="historical_peak")
    )

    anomalies = slot.merge(actual_counts, on="h3_index", how="inner")
    anomalies  = anomalies[
        anomalies["historical_peak"] > anomalies["upper_threshold"]
    ].copy()

    anomalies["spike_ratio"] = (
        anomalies["historical_peak"] / (anomalies["mean_count"] + 1e-9)
    ).round(2)

    anomalies = anomalies.sort_values("spike_ratio", ascending=False).head(top_n)
    anomalies = anomalies.merge(
        zones_df[["h3_index", "lat", "lng", "top_violation", "top_vehicle", "violation_count"]],
        on="h3_index", how="left"
    )

    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    anomalies["alert_reason"] = anomalies.apply(
        lambda r: (
            f"Historically spikes to {r['historical_peak']:.0f} violations "
            f"(vs normal {r['mean_count']:.1f}) on {day_names[cur_dow]}s at {cur_hour:02d}:00 "
            f"— {r['spike_ratio']:.1f}× above baseline"
        ),
        axis=1
    )

    return anomalies.reset_index(drop=True)
