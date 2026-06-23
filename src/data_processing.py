import json
import logging
import numpy as np
import pandas as pd
import h3

from config import (DATA_PATH, H3_RESOLUTION, MIN_VIOLATIONS,
                    PEAK_HOURS, VIOLATION_SEVERITY, VEHICLE_WEIGHT)

logger = logging.getLogger(__name__)


def _safe_parse_json(val):
    if pd.isna(val):
        return []
    try:
        return json.loads(str(val).replace("'", '"'))
    except Exception:
        return []


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)

    df["violation_type"] = df["violation_type"].apply(_safe_parse_json)
    df["offence_code"]   = df["offence_code"].apply(_safe_parse_json)

    df["created_datetime"] = pd.to_datetime(df["created_datetime"], utc=True, errors="coerce")
    ist = df["created_datetime"].dt.tz_convert("Asia/Kolkata")

    df["hour"]        = ist.dt.hour
    df["day_of_week"] = ist.dt.dayofweek
    df["month"]       = ist.dt.month
    df["date"]        = ist.dt.date
    df["ds"]          = ist.dt.normalize().dt.tz_localize(None)   # for Prophet
    df["week"]        = ist.dt.isocalendar().week.astype("Int64").fillna(0).astype(int)
    df["is_peak"]     = df["hour"].isin(PEAK_HOURS)

    df = df.dropna(subset=["latitude", "longitude"])
    df = df[df["latitude"].between(12.7, 13.2) & df["longitude"].between(77.3, 77.9)]

    df["near_junction"] = df["junction_name"].notna() & (df["junction_name"] != "No Junction")
    df["is_approved"]   = df["validation_status"] == "approved"

    df["primary_violation"] = df["violation_type"].apply(
        lambda x: x[0] if x else "WRONG PARKING"
    )
    df["severity_score"] = df["violation_type"].apply(
        lambda x: max((VIOLATION_SEVERITY.get(v, 1.0) for v in x), default=1.0) if x else 1.0
    )
    df["vehicle_weight"] = df["vehicle_type"].str.upper().map(VEHICLE_WEIGHT).fillna(1.0)
    df["impact_score"]   = df["severity_score"] * df["vehicle_weight"]

    logger.info("Loaded %d records", len(df))
    return df


def add_h3_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["h3_index"] = [
        h3.latlng_to_cell(r.latitude, r.longitude, H3_RESOLUTION)
        for r in df[["latitude", "longitude"]].itertuples()
    ]
    return df


def compute_zones(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per H3 cell into zone-level stats."""
    agg = (
        df.groupby("h3_index")
        .agg(
            violation_count    = ("impact_score", "count"),
            impact_sum         = ("impact_score", "sum"),
            impact_mean        = ("impact_score", "mean"),
            peak_hour_rate     = ("is_peak", "mean"),
            junction_rate      = ("near_junction", "mean"),
            avg_severity       = ("severity_score", "mean"),
            avg_vehicle_weight = ("vehicle_weight", "mean"),
        )
        .reset_index()
    )
    agg = agg[agg["violation_count"] >= MIN_VIOLATIONS].copy()

    agg["lat"] = agg["h3_index"].apply(lambda c: h3.cell_to_latlng(c)[0])
    agg["lng"] = agg["h3_index"].apply(lambda c: h3.cell_to_latlng(c)[1])

    top_violation = (
        df.groupby("h3_index")["primary_violation"]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
        .rename(columns={"primary_violation": "top_violation"})
    )
    top_vehicle = (
        df.groupby("h3_index")["vehicle_type"]
        .agg(lambda x: x.value_counts().index[0] if len(x) > 0 else "CAR")
        .reset_index()
        .rename(columns={"vehicle_type": "top_vehicle"})
    )

    agg = agg.merge(top_violation, on="h3_index", how="left")
    agg = agg.merge(top_vehicle,   on="h3_index", how="left")

    return agg.sort_values("violation_count", ascending=False).reset_index(drop=True)


def get_daily_series(df: pd.DataFrame, zone_ids: list) -> dict:
    """Return {zone_id: DataFrame(ds, y)} for Prophet — daily violation counts."""
    sub = df[df["h3_index"].isin(zone_ids)].copy()
    series = {}
    for zid in zone_ids:
        daily = (
            sub[sub["h3_index"] == zid]
            .groupby("ds")
            .size()
            .reset_index(name="y")
        )
        if len(daily) < 2:
            continue
        date_range = pd.date_range(daily["ds"].min(), daily["ds"].max(), freq="D")
        daily = (
            daily.set_index("ds")
            .reindex(date_range, fill_value=0)
            .reset_index()
            .rename(columns={"index": "ds"})
        )
        series[zid] = daily
    return series


def get_hourly_distribution(df: pd.DataFrame, zone_ids: list) -> dict:
    """Fraction of daily violations per hour (0-23) for each zone."""
    sub = df[df["h3_index"].isin(zone_ids)]
    result = {}
    for zid in zone_ids:
        counts = sub[sub["h3_index"] == zid]["hour"].value_counts().sort_index()
        dist = counts.reindex(range(24), fill_value=0).astype(float)
        total = dist.sum()
        result[zid] = (dist / total).values if total > 0 else np.full(24, 1 / 24)
    return result


def get_summary_stats(df: pd.DataFrame) -> dict:
    return {
        "total_violations": len(df),
        "unique_zones":     df["h3_index"].nunique() if "h3_index" in df.columns else 0,
        "date_range":       (str(df["date"].dropna().min()), str(df["date"].dropna().max())),
        "peak_hour_share":  round(df["is_peak"].mean() * 100, 1),
        "junction_share":   round(df["near_junction"].mean() * 100, 1),
        "top_vehicle":      df["vehicle_type"].value_counts().index[0] if "vehicle_type" in df.columns else "N/A",
        "top_violation":    df["primary_violation"].value_counts().index[0] if "primary_violation" in df.columns else "N/A",
    }
