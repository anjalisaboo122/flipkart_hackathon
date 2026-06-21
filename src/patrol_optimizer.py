"""
Patrol Route Optimizer.

Given the top-K high-risk zones, computes an efficient patrol sequence
using a nearest-neighbor TSP heuristic.

Travel times come from:
  1. MapMyIndia Distance Matrix ETA API (traffic-aware)  — if MMI key is set
  2. Euclidean distance × 1.4 road factor                — fallback

For multiple officers, zones are split into geographic clusters (K-Means)
and each officer gets their own optimised route.
"""

import logging
import time
import math
import numpy as np
import pandas as pd
import requests
from sklearn.cluster import KMeans

from config import TOMTOM_API_KEY, PATROL_ZONES

logger = logging.getLogger(__name__)

TOMTOM_MATRIX_URL = "https://api.tomtom.com/routing/1/matrix/sync/json"

AVG_SPEED_KMPH = 25.0   # fallback avg patrol speed in Bengaluru traffic
ROAD_FACTOR    = 1.4    # Euclidean → road distance multiplier


# ---------------------------------------------------------------------------
# Travel time matrix
# ---------------------------------------------------------------------------

def _euclidean_minutes(lat1, lng1, lat2, lng2) -> float:
    """Haversine distance → road distance → minutes at avg speed."""
    R       = 6371.0
    dlat    = math.radians(lat2 - lat1)
    dlng    = math.radians(lng2 - lng1)
    a       = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng / 2) ** 2
    dist_km = 2 * R * math.asin(math.sqrt(a)) * ROAD_FACTOR
    return round((dist_km / AVG_SPEED_KMPH) * 60, 1)


def _tomtom_travel_matrix(zones: pd.DataFrame) -> np.ndarray | None:
    """
    TomTom Matrix Routing API — returns real traffic-aware travel times.
    Free tier supports up to 100 pairs per request (10×10 max).
    Returns N×N matrix in minutes, or None on failure.
    """
    if not TOMTOM_API_KEY or len(zones) > 10:
        return None

    points = [{"point": {"latitude": r.lat, "longitude": r.lng}}
              for r in zones.itertuples()]
    body   = {"origins": points, "destinations": points}

    try:
        r = requests.post(
            TOMTOM_MATRIX_URL,
            params={"key": TOMTOM_API_KEY, "routeType": "fastest",
                    "traffic": "true", "travelMode": "car"},
            json=body,
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()

        cells = data.get("matrix", [])
        n     = len(zones)
        mx    = np.zeros((n, n))
        for cell in cells:
            i   = cell.get("originIndex", 0)
            j   = cell.get("destinationIndex", 0)
            dur = cell.get("response", {}).get("routeSummary", {}).get("travelTimeInSeconds", 0)
            if i < n and j < n:
                mx[i, j] = round(dur / 60.0, 1)

        if mx.sum() > 0:
            logger.info("Using TomTom Matrix Routing for patrol travel times")
            return mx
    except Exception as e:
        logger.debug("TomTom Matrix Routing failed: %s", e)

    return None


def _build_travel_matrix(zones: pd.DataFrame) -> np.ndarray:
    """Build N×N travel-time matrix (minutes). Try TomTom, else Euclidean fallback."""
    tt = _tomtom_travel_matrix(zones)
    if tt is not None and tt.shape == (len(zones), len(zones)):
        return tt

    n    = len(zones)
    mx   = np.zeros((n, n))
    lats = zones["lat"].values
    lngs = zones["lng"].values
    for i in range(n):
        for j in range(n):
            if i != j:
                mx[i, j] = _euclidean_minutes(lats[i], lngs[i], lats[j], lngs[j])
    return mx


# ---------------------------------------------------------------------------
# TSP nearest-neighbour
# ---------------------------------------------------------------------------

def _nearest_neighbor_tsp(dist_matrix: np.ndarray) -> list[int]:
    """Start at node 0 (highest risk), always go to nearest unvisited."""
    n       = len(dist_matrix)
    visited = [False] * n
    route   = [0]
    visited[0] = True

    for _ in range(n - 1):
        curr  = route[-1]
        best  = -1
        best_d = float("inf")
        for j in range(n):
            if not visited[j] and dist_matrix[curr, j] < best_d:
                best_d = dist_matrix[curr, j]
                best   = j
        route.append(best)
        visited[best] = True

    return route


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plan_patrol(risk_df: pd.DataFrame,
                n_officers: int = 1,
                start_time_str: str = "08:00") -> list[dict]:
    """
    Given a ranked DataFrame of zones (by dynamic_risk), produce patrol plans.

    Returns a list of patrol plans (one per officer), each a dict:
      {
        "officer": int,
        "route": DataFrame with columns:
            stop, h3_index, lat, lng, top_violation, top_vehicle,
            dynamic_risk, congestion_pct, arrival_min, arrival_time
      }
    """
    n_zones  = min(PATROL_ZONES * n_officers, len(risk_df))
    top      = risk_df.head(n_zones).reset_index(drop=True)

    if top.empty:
        return []

    plans = []

    if n_officers == 1:
        groups = [top]
    else:
        # Cluster zones geographically for multi-officer dispatch
        k      = min(n_officers, len(top))
        km     = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(top[["lat", "lng"]])
        top["_cluster"] = labels
        groups = [top[top["_cluster"] == i].reset_index(drop=True) for i in range(k)]

    h, m    = map(int, start_time_str.split(":"))
    start_m = h * 60 + m

    for officer_idx, group in enumerate(groups):
        if group.empty:
            continue

        dist_mx = _build_travel_matrix(group)
        order   = _nearest_neighbor_tsp(dist_mx)

        route_rows = []
        cum_min    = 0
        for stop_num, idx in enumerate(order):
            row = group.iloc[idx]
            if stop_num > 0:
                cum_min += dist_mx[order[stop_num - 1], idx]

            arrival_m   = int(start_m + cum_min)
            arrival_str = f"{(arrival_m // 60) % 24:02d}:{arrival_m % 60:02d}"

            route_rows.append({
                "stop":           stop_num + 1,
                "h3_index":       row["h3_index"],
                "lat":            round(row["lat"], 5),
                "lng":            round(row["lng"], 5),
                "top_violation":  row.get("top_violation", "N/A"),
                "top_vehicle":    row.get("top_vehicle", "N/A"),
                "dynamic_risk":   round(row.get("dynamic_risk", 0), 3),
                "congestion_pct": round(row.get("congestion_pct", 0), 1),
                "travel_min":     round(dist_mx[order[stop_num - 1], idx], 1) if stop_num > 0 else 0,
                "arrival_min":    round(cum_min, 1),
                "arrival_time":   arrival_str,
            })

        plans.append({
            "officer": officer_idx + 1,
            "route":   pd.DataFrame(route_rows),
            "total_travel_min": round(cum_min, 1),
            "n_zones": len(order),
        })

    return plans
