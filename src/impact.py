"""
Traffic Disruption Model — quantifies the traffic impact of parking violations.

Methodology:
  For every violation record we compute a disruption score:

    disruption = lane_blockage × violation_severity × peak_multiplier × road_factor

  where:
    - lane_blockage   : fraction of a lane physically occupied (vehicle-type specific)
    - violation_severity : how obstructive the parking type is (double-parking > wrong parking)
    - peak_multiplier : violations during rush hour affect exponentially more vehicles
    - road_factor     : junctions and main roads amplify impact vs residential streets

  Aggregated to zone level, disruption is converted to:
    - vehicle_delay_min_per_day  : extra minutes of delay caused across all passing vehicles
    - economic_cost_inr_per_day  : rupee cost using MoRTH value-of-time standard (₹75/veh-hr)

  TomTom congestion (freeflow vs current speed) is used as independent validation:
  high disruption-score zones should show higher observed congestion — this
  correlation is what proves the model, rather than being the model itself.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import (
    LANE_BLOCKAGE, VIOLATION_SEVERITY, PEAK_HOURS,
    PEAK_MULTIPLIER, ROAD_FACTOR_JUNCTION, ROAD_FACTOR_MAIN_ROAD, ROAD_FACTOR_OTHER,
    TRAFFIC_VOLUME_PER_HOUR, VALUE_OF_TIME_INR,
)


def compute_record_disruption(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a per-record disruption score to the raw violation dataframe.

    disruption = lane_blockage × severity × peak_multiplier × road_factor
    """
    df = df.copy()

    df["lane_blockage"] = df["vehicle_type"].str.upper().map(LANE_BLOCKAGE).fillna(0.5)

    df["peak_multiplier"] = df["hour"].apply(
        lambda h: PEAK_MULTIPLIER if h in PEAK_HOURS else 1.0
    )

    def _road_factor(row):
        if getattr(row, "near_junction", False):
            return ROAD_FACTOR_JUNCTION
        vtype = str(getattr(row, "primary_violation", "")).upper()
        if "MAIN ROAD" in vtype:
            return ROAD_FACTOR_MAIN_ROAD
        return ROAD_FACTOR_OTHER

    df["road_factor"] = [_road_factor(r) for r in df.itertuples()]

    df["disruption"] = (
        df["lane_blockage"]
        * df["severity_score"]
        * df["peak_multiplier"]
        * df["road_factor"]
    )

    return df


def compute_zone_impact(df: pd.DataFrame, zones_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates per-record disruption to zone level and converts to
    vehicle delay and economic cost.

    Returns zones_df enriched with:
      - disruption_score       : sum of all violation disruptions in zone
      - disruption_per_day     : average daily disruption (over observation period)
      - vehicle_delay_min_day  : estimated vehicle-minutes of delay per day
      - economic_cost_inr_day  : economic cost in INR per day (MoRTH ₹75/veh-hr)
      - peak_disruption_share  : fraction of disruption during peak hours
    """
    df_d = compute_record_disruption(df)

    _dates = pd.to_datetime(df_d["date"].dropna())
    n_days = max((_dates.max() - _dates.min()).days + 1, 1) if len(_dates) > 0 else 180

    zone_agg = (
        df_d.groupby("h3_index")
        .agg(
            disruption_score     = ("disruption", "sum"),
            peak_disruption      = ("disruption", lambda x: x[df_d.loc[x.index, "hour"].isin(PEAK_HOURS)].sum()),
            n_records            = ("disruption", "count"),
        )
        .reset_index()
    )

    zone_agg["disruption_per_day"]    = zone_agg["disruption_score"] / n_days
    zone_agg["peak_disruption_share"] = (
        zone_agg["peak_disruption"] / (zone_agg["disruption_score"] + 1e-9)
    ).round(3)

    # Convert disruption to vehicle-delay
    # Each unit of disruption blocks fraction of one lane for ~15 min avg dwell time
    # vehicles delayed = TRAFFIC_VOLUME_PER_HOUR × lane_fraction × dwell_hours
    AVG_DWELL_HOURS = 15 / 60   # average illegal parking dwell time

    zone_agg["vehicles_affected_per_day"] = (
        zone_agg["disruption_per_day"] * TRAFFIC_VOLUME_PER_HOUR * AVG_DWELL_HOURS
    ).round(0)

    # Delay per affected vehicle: proportional to blockage intensity
    # Conservative: each affected vehicle loses avg 2 minutes
    AVG_DELAY_MIN_PER_VEHICLE = 2.0

    zone_agg["vehicle_delay_min_day"] = (
        zone_agg["vehicles_affected_per_day"] * AVG_DELAY_MIN_PER_VEHICLE
    ).round(0)

    zone_agg["economic_cost_inr_day"] = (
        zone_agg["vehicle_delay_min_day"] / 60 * VALUE_OF_TIME_INR
    ).round(0)

    merged = zones_df.merge(zone_agg, on="h3_index", how="left")
    for col in ["disruption_score", "disruption_per_day", "vehicle_delay_min_day",
                "economic_cost_inr_day", "vehicles_affected_per_day", "peak_disruption_share"]:
        merged[col] = merged[col].fillna(0)

    return merged.sort_values("economic_cost_inr_day", ascending=False).reset_index(drop=True)


def validate_against_traffic(impact_df: pd.DataFrame,
                               traffic_df: pd.DataFrame) -> dict:
    """
    Validates the disruption model against observed Mappls congestion.

    A good model: zones with high disruption_score should have high congestion_pct.
    Measures Spearman ρ between the two — this is independent validation since
    disruption_score is computed from violation records, not from traffic data.
    """
    merged = impact_df.merge(
        traffic_df[["h3_index", "congestion_pct"]],
        on="h3_index", how="inner"
    ).dropna(subset=["disruption_score", "congestion_pct"])

    if len(merged) < 10:
        return {"rho": None, "p_value": None, "n": len(merged),
                "interpretation": "Insufficient data for validation."}

    rho, p = spearmanr(merged["disruption_score"], merged["congestion_pct"])

    sig = "Significant" if p < 0.05 else "Not significant"

    return {
        "rho":   round(rho, 4),
        "p_value": round(p, 6),
        "n":     len(merged),
        "interpretation": (
            f"Correlation between disruption_score and congestion_pct: "
            f"r={rho:.3f}, p={p:.4f}. {sig} at p<0.05."
        ),
    }


def get_city_totals(impact_df: pd.DataFrame) -> dict:
    """Aggregate economic impact across all zones."""
    return {
        "total_economic_cost_inr_day":  int(impact_df["economic_cost_inr_day"].sum()),
        "total_vehicle_delay_min_day":  int(impact_df["vehicle_delay_min_day"].sum()),
        "total_vehicles_affected_day":  int(impact_df["vehicles_affected_per_day"].sum()),
        "worst_zone_cost_inr_day":      int(impact_df["economic_cost_inr_day"].iloc[0]),
        "top_zone":                     impact_df["h3_index"].iloc[0] if len(impact_df) > 0 else "N/A",
    }
