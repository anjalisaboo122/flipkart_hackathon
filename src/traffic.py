"""
Real-time traffic integration.

Priority order:
  1. MapMyIndia Route ETA API       (traffic-aware travel time)
  2. Simulation fallback            (violation-correlated congestion)

Outputs:
  - congestion_pct per zone  (used to amplify violation risk scores)
  - freeflow_speed_kmh / peak_speed_kmh per zone
"""

import time
import logging
import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)




# ---------------------------------------------------------------------------
# Simulation fallback
# ---------------------------------------------------------------------------

def _simulate_congestion(zones_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng        = np.random.default_rng(seed)
    n          = len(zones_df)
    freeflow   = rng.uniform(25, 55, n)
    norm       = (
        (zones_df["violation_count"] - zones_df["violation_count"].min()) /
        (zones_df["violation_count"].max() - zones_df["violation_count"].min() + 1e-9)
    ).values
    cong_factor = (0.65 * norm + 0.35 * rng.uniform(0, 1, n)).clip(0, 0.85)
    peak_speed  = (freeflow * (1 - cong_factor)).clip(5)
    return pd.DataFrame({
        "h3_index":           zones_df["h3_index"].values,
        "freeflow_speed_kmh": freeflow.round(1),
        "peak_speed_kmh":     peak_speed.round(1),
        "congestion_pct":     ((freeflow - peak_speed) / freeflow * 100).round(1),
        "source":             "simulated",
    })


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_congestion(zones_df: pd.DataFrame, top_n: int = 100) -> pd.DataFrame:
    import os
    import time
    import urllib.request
    import urllib.error
    import json
    import pandas as pd

    token = os.environ.get("MAPPLS_TOKEN")
    if not token:
        raise EnvironmentError("MAPPLS_TOKEN environment variable not set")

    OFFSET = 0.004
    BASE = "https://route.mappls.com/route/direction"

    def fetch(endpoint, lat, lon):
        coords = f"{lon},{lat};{lon + OFFSET},{lat + OFFSET}"
        url    = f"{BASE}/{endpoint}/driving/{coords}?access_token={token}"
        try:
            resp = urllib.request.urlopen(url, timeout=12)
            data = json.loads(resp.read())
            r    = data["routes"][0]
            return round(r["distance"], 1), round(r["duration"], 1)
        except urllib.error.HTTPError as e:
            try:
                err_body = e.read().decode('utf-8')
            except Exception:
                err_body = ""
            raise Exception(f"Mappls API failed with status {e.code}. Response: {err_body}")
        except Exception as e:
            raise Exception(f"Mappls API request failed: {e}")

    subset = zones_df.head(top_n).copy()
    records = []

    for _, row in subset.iterrows():
        lat = row["lat"]
        lon = row["lng"]

        dist_adv, dur_adv = fetch("route_adv", lat, lon)
        time.sleep(0.3)
        dist_eta, dur_eta = fetch("route_eta", lat, lon)
        time.sleep(0.3)

        if not dur_adv or dur_adv <= 0:
            raise Exception(f"Mappls API returned invalid dur_adv: {dur_adv}")

        congestion_pct = ((dur_eta - dur_adv) / dur_adv) * 100
        
        spd_adv = round((dist_adv / dur_adv) * 3.6, 2)
        spd_eta = round((dist_eta / dur_eta) * 3.6, 2) if dur_eta else None

        if spd_adv is None or spd_eta is None:
            raise Exception("Mappls API missing valid speed data.")

        records.append({
            "h3_index": row["h3_index"],
            "freeflow_speed_kmh": spd_adv,
            "peak_speed_kmh": spd_eta,
            "congestion_pct": round(max(0.0, congestion_pct), 1),
            "source": "mappls"
        })

    return pd.DataFrame(records)


def compute_dynamic_risk(current_risk: pd.DataFrame,
                          traffic_df: pd.DataFrame) -> pd.DataFrame:
    """
    Multiply predicted violation risk by a traffic amplifier.
    Zones with high congestion get a higher effective risk score.

    Amplifier scale:
      congestion_pct >= 60  → 1.5×
      congestion_pct >= 40  → 1.25×
      congestion_pct >= 20  → 1.0×
      congestion_pct <  20  → 0.75×  (low traffic = less impact)
    """
    merged = current_risk.merge(
        traffic_df[["h3_index", "congestion_pct", "peak_speed_kmh",
                    "freeflow_speed_kmh", "source"]],
        on="h3_index", how="left"
    )
    merged["congestion_pct"] = merged["congestion_pct"].fillna(30.0)

    def _amp(cong):
        if cong >= 60:   return 1.5
        if cong >= 40:   return 1.25
        if cong >= 20:   return 1.0
        return 0.75

    merged["traffic_amplifier"] = merged["congestion_pct"].apply(_amp)
    merged["dynamic_risk"]      = (
        merged["predicted_hourly"] * merged["traffic_amplifier"]
    ).round(4)

    return merged.sort_values("dynamic_risk", ascending=False).reset_index(drop=True)
