"""
traffiKart — Real-Time Traffic Intelligence
Bengaluru Traffic Police | Flipkart Gridlock Hackathon

Run: streamlit run app.py
"""

import sys, os
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import folium
import plotly.express as px
import plotly.graph_objects as go
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from datetime import datetime

from src.data_processing import load_data, add_h3_index, compute_zones, get_daily_series, get_hourly_distribution, get_summary_stats
from src.traffic import fetch_congestion, compute_dynamic_risk
from src.patrol_optimizer import plan_patrol
from src.anomaly import compute_baselines, detect_anomalies_now, compute_zone_volatility
from src.briefing import generate_briefing
from src.impact import compute_zone_impact, validate_against_traffic, get_city_totals
from config import TOP_ZONES_PROPHET, ANTHROPIC_API_KEY, TOMTOM_API_KEY, MAPPLS_TOKEN

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="traffiKart — Bengaluru",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS — "highway / route" visual system with traffic light themes
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700;800;900&family=Outfit:wght@700;800;900&family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg: #0a0b0d;
        --surface: #15171b;
        --surface-2: #1b1e23;
        --line: #272a30;
        --line-soft: #1f2227;
        --text: #edeef0;
        --text-dim: #a7abb3;
        --text-faint: #6c7078;
        --red: #ef5350;
        --amber: #f2b83d;
        --green: #34d399;
        --blue: #5b9df0;
        --violet: #8b79f2;
    }

    /* Base type + background */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: var(--text);
    }
    h1, h2, h3, h4 {
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 800 !important;
        letter-spacing: .2px;
    }
    code, .mono { font-family: 'JetBrains Mono', monospace; }

    /* Clean road lanes texture across the app background */
    [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: var(--bg) !important;
        background-image: 
            radial-gradient(rgba(255, 255, 255, 0.01) 1px, transparent 0),
            radial-gradient(rgba(255, 255, 255, 0.008) 1.5px, transparent 0),
            url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%3E%3Cline x1='40' y1='0' x2='40' y2='160' stroke='%23ffffff' stroke-width='4' stroke-dasharray='30%2C30' opacity='0.022'/%3E%3Cline x1='120' y1='0' x2='120' y2='160' stroke='%23ffffff' stroke-width='4' stroke-dasharray='30%2C30' opacity='0.022'/%3E%3C/svg%3E");
        background-size: 32px 32px, 64px 64px, 160px 160px;
        background-repeat: repeat;
    }
    [data-testid="stSidebar"] {
        background-color: var(--surface) !important;
        border-right: 1px solid var(--line);
    }

    /* Route eyebrow tag — used once per tab as a "stop along the route" marker */
    .route-tag {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 1px;
        color: #0a0b0d;
        background: var(--amber);
        padding: 3px 9px;
        border-radius: 4px;
        margin-bottom: 10px;
    }
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 800;
        font-size: 40px;
        margin: 0 0 2px 0;
        line-height: 1.05;
    }
    .hero-sub {
        font-family: 'Inter', sans-serif;
        font-size: 16px;
        color: var(--text-dim);
        margin: 0 0 6px 0;
        font-weight: 400;
    }

    /* Lane-line rule — replaces default <hr> everywhere with a dashed lane marking */
    hr {
        border: none !important;
        height: 1px !important;
        background-image: repeating-linear-gradient(to right, var(--line) 0 9px, transparent 9px 16px) !important;
        margin: 22px 0 !important;
    }

    /* Section label with KM-style marker, used to break up long tabs into stops */
    .km-label {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 4px 0 14px 0;
    }
    .km-label .km {
        font-family: 'JetBrains Mono', monospace;
        font-size: 10.5px;
        font-weight: 600;
        color: #0a0b0d;
        background: var(--amber);
        padding: 3px 7px;
        border-radius: 4px;
        letter-spacing: .5px;
        flex-shrink: 0;
    }
    .km-label .title {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 17px;
        white-space: nowrap;
    }
    .km-label .rule {
        flex: 1;
        height: 1px;
        background-image: repeating-linear-gradient(to right, var(--line) 0 9px, transparent 9px 16px);
    }

    /* Custom metric card styles — restyled as instrument-panel readouts */
    .metric-container {
        display: flex;
        gap: 14px;
        justify-content: space-between;
        flex-wrap: wrap;
        margin: 18px 0;
    }
    .metric-card {
        background: var(--surface);
        border: 1px solid var(--line);
        border-top: 2px solid var(--amber);
        border-radius: 10px;
        padding: 18px 20px;
        flex: 1;
        min-width: 220px;
        transition: border-color .2s ease, transform .2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #3a3f47;
    }
    .metric-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        color: var(--text-faint);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 32px;
        font-weight: 800;
        margin: 10px 0 6px 0;
        line-height: 1;
    }
    .metric-delta {
        font-size: 13px;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Custom hazard stripe divider and text */
    .hazard-container {
        margin: 22px 0;
        background: #0a0b0d;
        border: 1px solid var(--line);
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .hazard-stripes {
        height: 10px;
        background: repeating-linear-gradient(
            -45deg,
            #f2b83d,
            #f2b83d 12px,
            #0a0b0d 12px,
            #0a0b0d 24px
        );
    }
    .hazard-text {
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        font-weight: 600;
        color: var(--text-dim);
        text-align: center;
        padding: 7px 6px;
        letter-spacing: 1.2px;
        background: #111215;
    }

    /* Accent gradients */
    .text-gradient {
        background: linear-gradient(135deg, #ef5350 0%, #f2b83d 50%, #34d399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Tabs styled as a route selector — colored dot per stop, amber active underline */
    [data-baseweb="tab-list"] {
        gap: 16px !important;
        border-bottom: 2px solid var(--line) !important;
        padding-bottom: 8px !important;
    }
    [data-baseweb="tab-list"] button {
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 800 !important;
        font-size: 22px !important;
        color: var(--text-dim) !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s ease !important;
        padding: 10px 18px !important;
        border-radius: 8px !important;
    }
    [data-baseweb="tab-list"] button p { 
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 800 !important;
        font-size: 22px !important;
    }
    [data-baseweb="tab-highlight"] { 
        background-color: var(--amber) !important; 
        height: 3px !important; 
    }
    [data-baseweb="tab-list"] button[aria-selected="true"] { 
        color: var(--text) !important; 
        background: rgba(255, 255, 255, 0.03) !important;
    }
    [data-baseweb="tab-list"] button::before {
        content: "";
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 10px;
        opacity: 0.35;
        transition: all 0.3s ease;
        vertical-align: middle;
    }
    [data-baseweb="tab-list"] button:nth-of-type(1)::before { 
        background: var(--red) !important; 
        box-shadow: 0 0 4px var(--red);
    }
    [data-baseweb="tab-list"] button:nth-of-type(2)::before { 
        background: var(--amber) !important; 
        box-shadow: 0 0 4px var(--amber);
    }
    [data-baseweb="tab-list"] button:nth-of-type(3)::before { 
        background: var(--blue) !important; 
        box-shadow: 0 0 4px var(--blue);
    }
    [data-baseweb="tab-list"] button:nth-of-type(4)::before { 
        background: var(--green) !important; 
        box-shadow: 0 0 4px var(--green);
    }
    [data-baseweb="tab-list"] button:nth-of-type(5)::before { 
        background: var(--violet) !important; 
        box-shadow: 0 0 4px var(--violet);
    }
    [data-baseweb="tab-list"] button[aria-selected="true"]::before { 
        opacity: 1 !important; 
        transform: scale(1.15);
    }
    [data-baseweb="tab-list"] button[aria-selected="true"]:nth-of-type(1)::before { 
        box-shadow: 0 0 10px var(--red), 0 0 18px var(--red) !important; 
    }
    [data-baseweb="tab-list"] button[aria-selected="true"]:nth-of-type(2)::before { 
        box-shadow: 0 0 10px var(--amber), 0 0 18px var(--amber) !important; 
    }
    [data-baseweb="tab-list"] button[aria-selected="true"]:nth-of-type(3)::before { 
        box-shadow: 0 0 10px var(--blue), 0 0 18px var(--blue) !important; 
    }
    [data-baseweb="tab-list"] button[aria-selected="true"]:nth-of-type(4)::before { 
        box-shadow: 0 0 10px var(--green), 0 0 18px var(--green) !important; 
    }
    </style>
    """,
    unsafe_allow_html=True
)

BENGALURU_CENTER = [12.9716, 77.5946]

# ---------------------------------------------------------------------------
# Cached Data loading functions (Fast startup without Prophet)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_base_data():
    df = load_data()
    df_h3 = add_h3_index(df)
    stats = get_summary_stats(df_h3)
    return df_h3, stats

@st.cache_data(show_spinner=False)
def load_junction_data():
    df = load_data()
    juncs = df[df["junction_name"].notna() & (df["junction_name"] != "No Junction")].copy()
    
    # Calculate per-junction stats:
    agg = juncs.groupby("junction_name").agg(
        violation_count=("id", "count"),
        lat=("latitude", "mean"),
        lng=("longitude", "mean"),
        unique_devices=("device_id", "nunique"),
        avg_severity=("severity_score", "mean")
    ).reset_index()
    
    # Filter to >= 30 violations (the 154 junctions)
    agg = agg[agg["violation_count"] >= 30].reset_index(drop=True)
    
    # Compute patrol_normalized_rate: violations / unique patrol devices
    agg["patrol_normalized_rate"] = agg["violation_count"] / agg["unique_devices"]
    
    # Compute severity ratio relative to city average severity of these 154 junctions
    city_avg_sev = agg["avg_severity"].mean()
    agg["severity_ratio"] = agg["avg_severity"] / city_avg_sev
    
    return agg

@st.cache_data(show_spinner=False, ttl=86400)
def fetch_delivery_coverage(lats, lngs, names, violations):
    import requests
    import math

    DARK_STORE_FALLBACK = [
        {"display_name": "Blinkit · Shivajinagar",    "lat": 12.9840, "lng": 77.5995, "brand": "Blinkit"},
        {"display_name": "Blinkit · Malleshwaram",    "lat": 12.9966, "lng": 77.5703, "brand": "Blinkit"},
        {"display_name": "Blinkit · Rajajinagar",     "lat": 12.9921, "lng": 77.5536, "brand": "Blinkit"},
        {"display_name": "Blinkit · Yeshwanthpur",    "lat": 13.0200, "lng": 77.5420, "brand": "Blinkit"},
        {"display_name": "Blinkit · Jayanagar",       "lat": 12.9255, "lng": 77.5932, "brand": "Blinkit"},
        {"display_name": "Blinkit · BTM Layout",      "lat": 12.9165, "lng": 77.6101, "brand": "Blinkit"},
        {"display_name": "Blinkit · Koramangala",     "lat": 12.9279, "lng": 77.6271, "brand": "Blinkit"},
        {"display_name": "Blinkit · Indiranagar",     "lat": 12.9784, "lng": 77.6408, "brand": "Blinkit"},
        {"display_name": "Blinkit · HSR Layout",      "lat": 12.9116, "lng": 77.6389, "brand": "Blinkit"},
        {"display_name": "Blinkit · Marathahalli",    "lat": 12.9563, "lng": 77.7010, "brand": "Blinkit"},
        {"display_name": "Blinkit · Whitefield",      "lat": 12.9698, "lng": 77.7499, "brand": "Blinkit"},
        {"display_name": "Blinkit · Hebbal",          "lat": 13.0360, "lng": 77.5970, "brand": "Blinkit"},
        {"display_name": "Blinkit · Yelahanka",       "lat": 13.1007, "lng": 77.5963, "brand": "Blinkit"},
        {"display_name": "Blinkit · JP Nagar",        "lat": 12.9082, "lng": 77.5833, "brand": "Blinkit"},
        {"display_name": "Blinkit · Electronic City", "lat": 12.8452, "lng": 77.6602, "brand": "Blinkit"},
        {"display_name": "Zepto · KR Market area",    "lat": 12.9677, "lng": 77.5762, "brand": "Zepto"},
        {"display_name": "Zepto · Malleshwaram",      "lat": 13.0017, "lng": 77.5701, "brand": "Zepto"},
        {"display_name": "Zepto · Rajajinagar",       "lat": 12.9926, "lng": 77.5543, "brand": "Zepto"},
        {"display_name": "Zepto · Yeshwanthpur",      "lat": 13.0185, "lng": 77.5436, "brand": "Zepto"},
        {"display_name": "Zepto · Koramangala",       "lat": 12.9380, "lng": 77.6186, "brand": "Zepto"},
        {"display_name": "Zepto · Indiranagar",       "lat": 12.9822, "lng": 77.6423, "brand": "Zepto"},
        {"display_name": "Zepto · HSR Layout",        "lat": 12.9094, "lng": 77.6432, "brand": "Zepto"},
        {"display_name": "Zepto · Bellandur",         "lat": 12.9250, "lng": 77.6700, "brand": "Zepto"},
        {"display_name": "Zepto · Whitefield",        "lat": 12.9745, "lng": 77.7482, "brand": "Zepto"},
        {"display_name": "Zepto · Hebbal",            "lat": 13.0342, "lng": 77.5990, "brand": "Zepto"},
        {"display_name": "Zepto · Yelahanka",         "lat": 13.1020, "lng": 77.5990, "brand": "Zepto"},
        {"display_name": "Zepto · Electronic City",   "lat": 12.8477, "lng": 77.6625, "brand": "Zepto"},
    ]

    def haversine_m(lat1, lng1, lat2, lng2):
        R = 6_371_000
        dlat = math.radians(lat2 - lat1)
        dlng = math.radians(lng2 - lng1)
        a = (math.sin(dlat / 2) ** 2
             + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng / 2) ** 2)
        return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def run_overpass(query):
        endpoints = [
            "https://overpass-api.de/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter",
            "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
        ]
        for url in endpoints:
            try:
                r = requests.get(url, params={"data": query}, timeout=30)
                if r.status_code == 200 and r.text.strip().startswith("{"):
                    return r.json()
            except Exception:
                continue
        return None

    min_lat, max_lat = min(lats) - 0.02, max(lats) + 0.02
    min_lng, max_lng = min(lngs) - 0.02, max(lngs) + 0.02

    rest_query = (
        f"[out:json][timeout:40];"
        f"(node[\"amenity\"~\"^(restaurant|cafe|fast_food|food_court)$\"]"
        f"({min_lat},{min_lng},{max_lat},{max_lng});"
        f"way[\"amenity\"~\"^(restaurant|cafe|fast_food|food_court)$\"]"
        f"({min_lat},{min_lng},{max_lat},{max_lng}););"
        f"out center;"
    )
    restaurants = []
    rest_error = None
    result = run_overpass(rest_query)
    if result:
        for el in result.get("elements", []):
            rlat = el.get("lat") or (el.get("center") or {}).get("lat")
            rlng = el.get("lon") or (el.get("center") or {}).get("lon")
            if rlat and rlng:
                restaurants.append({"lat": float(rlat), "lng": float(rlng)})
    else:
        rest_error = "All Overpass API mirrors unreachable or rate-limited."

    junction_rows = []
    for i, name in enumerate(names):
        lat, lng = lats[i], lngs[i]
        count = sum(1 for r in restaurants if haversine_m(lat, lng, r["lat"], r["lng"]) <= 500)
        junction_rows.append({
            "junction_name": name,
            "lat": lat,
            "lng": lng,
            "violation_count": violations[i],
            "restaurant_count": count,
        })
    rest_df = pd.DataFrame(junction_rows)

    ds_query = (
        "[out:json][timeout:25];"
        "(node[\"name\"~\"Blinkit|Zepto|Instamart\",i](12.7,77.3,13.2,77.9);"
        "way[\"name\"~\"Blinkit|Zepto|Instamart\",i](12.7,77.3,13.2,77.9););"
        "out center;"
    )
    dark_stores = []
    ds_result = run_overpass(ds_query)
    if ds_result:
        for el in ds_result.get("elements", []):
            dlat = el.get("lat") or (el.get("center") or {}).get("lat")
            dlng = el.get("lon") or (el.get("center") or {}).get("lon")
            dname = el.get("tags", {}).get("name", "")
            if dlat and dlng and dname:
                nl = dname.lower()
                brand = "Blinkit" if "blinkit" in nl else ("Zepto" if "zepto" in nl else "Instamart")
                dark_stores.append({"display_name": dname, "lat": float(dlat), "lng": float(dlng), "brand": brand})

    used_fallback = len(dark_stores) == 0
    if used_fallback:
        dark_stores = DARK_STORE_FALLBACK

    ds_df = pd.DataFrame(dark_stores)
    return rest_df, ds_df, len(restaurants), rest_error, used_fallback

# Load standard stats and data
with st.spinner("Loading violations dataset..."):
    df_h3, stats = load_base_data()
    df_juncs = load_junction_data()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    """
    <div style="margin-top: 20px; margin-bottom: 8px; padding-left: 5px;">
        <div style="font-family: 'Space Grotesk', sans-serif; font-weight: 900; font-size: 38px; color: #ffffff; line-height: 0.9; letter-spacing: 1px;">TRAFFIKART</div>
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 9.5px; color: #6c7078; margin-top: 6px; letter-spacing: 1.2px;">// BENGALURU TRAFFIC POLICE</div>
    </div>
    <div style="height:1px; margin:18px 0; background-image:repeating-linear-gradient(to right, #272a30 0 9px, transparent 9px 16px);"></div>
    """,
    unsafe_allow_html=True
)

now = datetime.now()
st.sidebar.markdown(
    f"""
    <div style="display:flex; flex-direction:column; gap:8px; font-size:12.5px;">
        <div style="display:flex; justify-content:space-between;"><span style="color:#6c7078;">Date</span><span class="mono" style="color:#edeef0;">{now.strftime('%d %b %Y')}</span></div>
        <div style="display:flex; justify-content:space-between;"><span style="color:#6c7078;">Time</span><span class="mono" style="color:#edeef0;">{now.strftime('%H:%M IST')}</span></div>
        <div style="display:flex; justify-content:space-between;"><span style="color:#6c7078;">Day</span><span class="mono" style="color:#edeef0;">{now.strftime('%A')}</span></div>
    </div>
    <div style="height:1px; margin:18px 0; background-image:repeating-linear-gradient(to right, #272a30 0 9px, transparent 9px 16px);"></div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(
    f"""
    <div style="display:flex; flex-direction:column; gap:8px; font-size:12.5px; margin-bottom: 20px;">
        <div style="display:flex; justify-content:space-between;"><span style="color:#6c7078;">Records</span><span class="mono" style="color:#edeef0;">{stats['total_violations']:,}</span></div>
        <div style="display:flex; justify-content:space-between;"><span style="color:#6c7078;">H3 zones</span><span class="mono" style="color:#edeef0;">{stats['unique_zones']}</span></div>
        <div style="display:flex; justify-content:space-between;"><span style="color:#6c7078;">Junctions mapped</span><span class="mono" style="color:#edeef0;">{len(df_juncs)}</span></div>
        <div style="display:flex; justify-content:space-between; gap:8px;"><span style="color:#6c7078;">Date range</span><span class="mono" style="color:#edeef0; font-size:11px; text-align:right;">{stats['date_range'][0]} → {stats['date_range'][1]}</span></div>
    </div>
    <div style="height:1px; margin:18px 0; background-image:repeating-linear-gradient(to right, #272a30 0 9px, transparent 9px 16px);"></div>
    
    <a href="https://huggingface.co/" target="_blank" style="display:flex; align-items:center; justify-content:center; gap:10px; padding:14px 16px; margin: 25px 0 10px 0; border-radius:10px; background:linear-gradient(135deg, rgba(139,121,242,0.1) 0%, rgba(91,157,240,0.1) 100%); border:1px solid rgba(139,121,242,0.4); color:#8b79f2; text-decoration:none; font-family:'Inter', sans-serif; font-weight:700; font-size:14px; letter-spacing:0.5px; box-shadow:0 0 15px rgba(139,121,242,0.15);">
        <span style="font-size:16px;">📹</span> Launch Our Illegal Parking Detection Model ↗
    </a>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(
    """
    <div class="hazard-container" style="margin-top: 20px;">
        <div class="hazard-stripes"></div>
        <div class="hazard-text">TRAFFIKART · TRAFFIC DETECTION</div>
        <div class="hazard-stripes"></div>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------------------------
# Main Tabs Layout
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ The Blind Spot",
    "⏰ The Timing Gap",
    "📹 Camera Placement",
    "📡 Live Monitoring",
    "Delivery Coverage",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — The Blind Spot
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(
        """
        <div class="route-tag">STOP 01 / 04 — COVERAGE GAP</div>
        <div class="hero-title">The Blind Spot</div>
        <p class="hero-sub">Identifying critical gaps between raw violations and officer coverage</p>
        """,
        unsafe_allow_html=True
    )
    
    # Toggle map coloring/sizing mode
    map_metric = st.radio(
        "Choose Map Display Metric:",
        ["Raw Violation Count", "Patrol-Normalized Rate"],
        horizontal=True,
        help="Switch between raw counts and counts divided by unique patrol devices to see hidden hotspots."
    )
    
    st.caption(
        "Note: this view shows the patrol-normalized rate in isolation as a diagnostic signal. "
        "Final camera site selection (Tab 3) combines this with raw violation volume — top-20 by count ∩ top-20 by rate — "
        "to avoid overweighting low-volume junctions."
    )
    
    # Setup sizing and coloring columns
    df_map = df_juncs.copy()
    if map_metric == "Raw Violation Count":
        color_col = "violation_count"
        max_val = df_map[color_col].max()
        min_val = df_map[color_col].min()
        df_map["norm"] = (df_map[color_col] - min_val) / (max_val - min_val + 1e-9)
        # Sizing and coloring (warm sunset palette)
        df_map["radius"] = 30 + (df_map["norm"] ** 0.5) * 220
        df_map["color_r"] = 255
        df_map["color_g"] = (220 * (1.0 - df_map["norm"])).astype(int)
        df_map["color_b"] = (50 + 50 * df_map["norm"]).astype(int)
    else:
        color_col = "patrol_normalized_rate"
        max_val = df_map[color_col].max()
        min_val = df_map[color_col].min()
        df_map["norm"] = (df_map[color_col] - min_val) / (max_val - min_val + 1e-9)
        # Sizing and coloring (cool to warm electric palette)
        df_map["radius"] = 30 + (df_map["norm"] ** 0.5) * 220
        df_map["color_r"] = (50 + 205 * df_map["norm"]).astype(int)
        df_map["color_g"] = (100 * (1.0 - df_map["norm"]) + 50).astype(int)
        df_map["color_b"] = 255
    
    df_map["color"] = df_map.apply(
        lambda r: [int(r["color_r"]), int(r["color_g"]), int(r["color_b"]), 170], axis=1
    )
    # Highlight BTP040 (Elite Junction) as the hero junction on the map (gold/amber)
    is_hero = df_map["junction_name"].str.contains("BTP040")
    df_map.loc[is_hero, "color"] = df_map.loc[is_hero].apply(lambda r: [255, 195, 0, 255], axis=1)

    # Pre-format tooltip strings to bypass pydeck formatting issues in Streamlit
    df_map["violation_count_str"] = df_map["violation_count"].apply(lambda x: f"{x:,}")
    df_map["patrol_normalized_rate_str"] = df_map["patrol_normalized_rate"].apply(lambda x: f"{x:.2f}")
    df_map["avg_severity_str"] = df_map["avg_severity"].apply(lambda x: f"{x:.2f}")

    # Pydeck Interactive Map Layers
    # Base scatterplot layer for all 154 junctions
    layer = pdk.Layer(
        "ScatterplotLayer",
        df_map,
        get_position=["lng", "lat"],
        get_color="color",
        get_radius="radius",
        pickable=True,
        auto_highlight=True,
    )
    
    # Hero highlight ring layer for BTP040 (Elite Junction) (gold/amber)
    df_hero = df_map[is_hero].copy()
    hero_ring_layer = pdk.Layer(
        "ScatterplotLayer",
        df_hero,
        get_position=["lng", "lat"],
        get_color=[255, 195, 0, 255],  # Gold/Amber
        get_radius=df_hero["radius"].iloc[0] * 1.5 if not df_hero.empty else 400,
        stroked=True,
        filled=False,
        line_width_min_pixels=3,
        pickable=False,
    )
    
    view_state = pdk.ViewState(
        latitude=12.9716,
        longitude=77.5946,
        zoom=11.5,
        pitch=0,
    )
    
    deck = pdk.Deck(
        layers=[layer, hero_ring_layer],
        initial_view_state=view_state,
        map_style=pdk.map_styles.CARTO_DARK,
        tooltip={
            "html": "<b>Junction:</b> {junction_name}<br/>"
            "<b>Violations:</b> {violation_count_str}<br/>"
            "<b>Patrol-Normalized Rate:</b> {patrol_normalized_rate_str} viols/device<br/>"
            "<b>Avg Severity:</b> {avg_severity_str}",
            "style": {"backgroundColor": "#15171b", "color": "white"}
        },
    )
    
    st.pydeck_chart(deck)
    
    # Highlight BTP040 Callout Card (headline junction)
    st.markdown("---")
    st.markdown(
        """
        <div class="km-label">
            <span class="km">FOCUS</span>
            <span class="title">Critical Focus Area: Junction BTP040 (Elite Junction)</span>
            <span class="rule"></span>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        "Junction **BTP040** stands as the city's #1 ranked blind spot. "
        "Despite having an enormous raw violation count, its patrol frequency is highly "
        "restricted relative to the volume, resulting in the highest patrol-normalized rate "
        "citywide. This junction is among our top recommended camera sites in Tab 3."
    )
    
    # Custom HTML metrics cards for BTP040
    st.markdown(
        """
        <div class="metric-container">
            <div class="metric-card">
                <div class="metric-title">Violation Count</div>
                <div class="metric-value" style="color: #FF6B6B;">10,718</div>
                <div class="metric-delta" style="color: #FFA07A;">▲ 11.0x City Average (976)</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Patrol-Normalized Rate</div>
                <div class="metric-value" style="color: #4D96FF;">153.11 <span style="font-size: 16px; font-weight: 400; color: #888896;">viols/device</span></div>
                <div class="metric-delta" style="color: #6BCB77;">▲ 7.8x City Average (19.7)</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Unique Patrol Devices</div>
                <div class="metric-value" style="color: #6BCB77;">70</div>
                <div class="metric-delta" style="color: #aaa;">▲ 2.0x City Average (34.4)</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Expandable methodology rigor point for BTP083
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🔍 Methodology Rigor: Why BTP083 Doesn't Make the Cut", expanded=False):
        st.markdown(
            """
            **Junction BTP083 (AS Char Street, Mysore Road)** has a high raw violation count (**#13 citywide**, 2,778 violations)
            and an above-average patrol-normalized rate (33.07, **68% above the city average** of 19.7).
            
            However, it does **not** qualify for the final camera placement in Tab 3 because it ranks **#24 by rate** (outside the top-20 threshold).
            Its high patrol coverage (**84 unique devices**) dilutes the per-officer violation rate below our top-20 cutoff.
            
            This is an **intentional design decision** in our methodology: we filter out junctions that are simply high-volume-but-well-patrolled,
            prioritizing locations that are both high-volume *and* severely under-patrolled relative to that volume.
            """
        )

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — The Timing Gap
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(
        """
        <div class="route-tag">STOP 02 / 04 — SHIFT MISMATCH</div>
        <div class="hero-title">The Timing Gap</div>
        <p class="hero-sub">The mismatch between police shifts and peak congestion hours</p>
        """,
        unsafe_allow_html=True
    )
    
    # Hardcoded historical congestion ratio by hour (previously pulled from validation log)
    hardcoded_ratios = [0.930715, 0.9235566666666667, 0.9041899999999999, 0.9346736842105263, 0.8871325000000001, 0.9104575, 0.9190950000000001, 0.9707399999999999, 1.0842325000000002, 1.1396575, 1.18201, 1.1904666666666666, 1.2156875, 1.22434, 1.348225, 1.2149625, 1.4077633333333333, 1.408515, 1.3978199999999998, 1.380555, 1.357395, 1.0736033333333335, 1.013485, 1.009690322580645]
    congestion_by_hour = pd.DataFrame({
        "hour": range(24),
        "mean_congestion": hardcoded_ratios
    })

    # Violations by hour (using df_h3 already loaded in scope)
    violations_by_hour = df_h3.groupby("hour").size().reset_index(name="violation_count")
    total_viols = violations_by_hour["violation_count"].sum()
    violations_by_hour["pct"] = (violations_by_hour["violation_count"] / total_viols * 100).round(2)
    
    # Merge for plotting
    plot_df = pd.merge(
        pd.DataFrame({"hour": range(24)}),
        violations_by_hour,
        on="hour",
        how="left"
    ).fillna(0)
    plot_df = pd.merge(
        plot_df,
        congestion_by_hour,
        on="hour",
        how="left"
    ).fillna(0.9) # fallback to normal baseline congestion ratio if hour is missing

    # Build dual-axis Plotly chart
    from plotly.subplots import make_subplots
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Violations as Bar Chart
    fig.add_trace(
        go.Bar(
            x=plot_df["hour"],
            y=plot_df["violation_count"],
            name="Violations Recorded (Enforcement Activity)",
            marker_color="rgba(91, 157, 240, 0.75)", # --blue
            hovertemplate="Hour %{x}:00<br>Violations: %{y:,}<br>Share: %{customdata:.2f}%<extra></extra>",
            customdata=plot_df["pct"]
        ),
        secondary_y=False
    )
    
    # Congestion as Line Chart
    fig.add_trace(
        go.Scatter(
            x=plot_df["hour"],
            y=plot_df["mean_congestion"],
            name="Observed Congestion Ratio",
            line=dict(color="#F2B83D", width=4), # --amber
            mode="lines+markers",
            hovertemplate="Hour %{x}:00<br>Congestion Ratio: %{y:.2f}<extra></extra>"
        ),
        secondary_y=True
    )
    
    # Highlight 5 PM - 8 PM window (Hours 17-20)
    fig.add_vrect(
        x0=17, x1=20,
        fillcolor="rgba(242, 184, 61, 0.15)",
        layer="below",
        line_width=0,
        annotation_text="The timing gap",
        annotation_position="top left",
        annotation_font=dict(size=12, color="#F2B83D", family="JetBrains Mono")
    )
    
    # Add text annotation
    fig.add_annotation(
        x=17,
        y=1.4085,
        yref="y2",
        text="<b>CRITICAL TIMING GAP</b><br>Peak traffic, near-zero enforcement",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#F2B83D",
        ax=-50,
        ay=-70,
        font=dict(size=11, color="white", family="Inter"),
        bordercolor="#F2B83D",
        borderpad=6,
        bgcolor="#15171b",
        opacity=0.9
    )
    
    # Layout styling for dark mode dashboard
    fig.update_layout(
        title="24-Hour Comparison: Enforcement vs. Congestion",
        title_font=dict(size=18, family="Space Grotesk", color="white"),
        xaxis=dict(
            title="Hour of Day (24h)",
            tickmode="linear",
            tick0=0,
            dtick=1,
            gridcolor="rgba(255,255,255,0.05)",
            color="white",
            range=[-0.5, 23.5]
        ),
        yaxis=dict(
            title="Violation Count (Enforcement Volume)",
            gridcolor="rgba(255,255,255,0.05)",
            color="white"
        ),
        yaxis2=dict(
            title="Mean Congestion Ratio",
            gridcolor="rgba(255,255,255,0.05)",
            color="white",
            overlaying="y",
            side="right",
            range=[0.8, 1.6]
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="white")
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=500,
        margin=dict(l=40, r=40, t=80, b=40)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Real numbers caption
    st.markdown(
        """
        <div style="background: rgba(242, 184, 61, 0.08); border: 1px solid rgba(242, 184, 61, 0.2); border-left: 3px solid #f2b83d; border-radius: 8px; padding: 16px; margin: 15px 0;">
            <p style="margin: 0; font-size: 15px; color: #F2B83D; font-weight: 500; line-height: 1.5;">
                ⚠️ <strong>Key Insight:</strong> 96.67% of violations are logged before 2:00 PM (end of day shift), 
                whereas traffic congestion peaks at 5:00 PM (mean congestion ratio: 1.41) with only 0.20% of violations 
                recorded between 5:00 PM and 8:00 PM.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Independent validation caption
    st.caption(
        "Independent validation: Karnataka's official traffic monitoring data shows citywide congestion "
        "also peaking in the evening (~7 PM), confirming this isn't an artifact of our single-day sample."
    )

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Camera Placement
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(
        """
        <div class="route-tag">STOP 03 / 04 — DEPLOYMENT PLAN</div>
        <div class="hero-title">Camera Placement</div>
        <p class="hero-sub">Strategic selection of automatic enforcement locations</p>
        """,
        unsafe_allow_html=True
    )

    # 1. Compute ranks and intersection
    df_juncs_t3 = df_juncs.copy()
    df_juncs_t3["rank_count"] = df_juncs_t3["violation_count"].rank(ascending=False, method="min").astype(int)
    df_juncs_t3["rank_rate"] = df_juncs_t3["patrol_normalized_rate"].rank(ascending=False, method="min").astype(int)

    top_20_count_names = df_juncs_t3.nsmallest(20, "rank_count")["junction_name"].tolist()
    top_20_rate_names = df_juncs_t3.nsmallest(20, "rank_rate")["junction_name"].tolist()
    tier1_names = sorted(list(set(top_20_count_names) & set(top_20_rate_names)))
    tier2_names = [name for name in df_juncs_t3["junction_name"] if name not in tier1_names]
    num_camera_sites = len(tier1_names)
    
    # 2. Site count header
    st.markdown(
        f"<h3 style='margin: 15px 0 5px 0; color: #5b9df0; font-family: \"Space Grotesk\", sans-serif;'>📸 {num_camera_sites} of {len(df_juncs_t3)} junctions qualify for Tier 1 camera placement</h3>", 
        unsafe_allow_html=True
    )
    st.caption("Criteria: Top-20 by raw violation count ∩ Top-20 by patrol-normalized rate (diminishes bias toward well-patrolled spots).")

    # Two-card explainer row
    ecol1, ecol2 = st.columns([1, 1])
    with ecol1:
        st.markdown(
            """
            <div style="background: rgba(91, 157, 240, 0.06); border: 1px solid rgba(91, 157, 240, 0.18); border-radius: 10px; padding: 16px; border-left: 3px solid #5b9df0; height: 100%;">
                <h4 style="margin: 0 0 10px 0; color: #5b9df0; font-size: 16px; font-weight: 700;">Tier 1 — Fixed Cameras</h4>
                <p style="margin: 0; font-size: 14px; color: #ddd;">Guaranteed 24/7 coverage at our highest-priority sites — mounted on existing infrastructure like light poles and signal posts.</p>
            </div>
            """, unsafe_allow_html=True
        )
    with ecol2:
        st.markdown(
            """
            <div style="background: rgba(139, 121, 242, 0.06); border: 1px solid rgba(139, 121, 242, 0.18); border-radius: 10px; padding: 16px; border-left: 3px solid #8b79f2; height: 100%;">
                <h4 style="margin: 0 0 10px 0; color: #8b79f2; font-size: 16px; font-weight: 700;">Tier 2 — Ekart Fleet Coverage</h4>
                <p style="margin: 0; font-size: 14px; color: #ddd;">Flipkart's Ekart delivery vehicles, already driving every corner of the city, carry a lightweight dashcam + edge model that passively flags violations during normal delivery runs. Self-funding — less illegal parking means faster Ekart deliveries.</p>
            </div>
            """, unsafe_allow_html=True
        )
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Layout: Toggle and Selectbox
    tcol1, tcol2 = st.columns([1, 1])
    with tcol1:
        toggle_mode = st.radio(
            "Map Display Mode:",
            ["All Junctions (Context)", "Tier 1 — Fixed Cameras", "Tier 2 — Ekart Coverage", "Both Tiers (Focus)"],
            horizontal=True,
            help="Show all junctions with recommended sites highlighted, or focus only on specific tiers."
        )
    with tcol2:
        all_inspectable_sites = sorted(list(set(tier1_names) | set(tier2_names)))
        selected_site_name = st.selectbox(
            "Select Site to Inspect:",
            all_inspectable_sites,
            help="Choose a recommended site to view detailed metrics and location on the map."
        )

    # 3. Setup map data
    def assign_tier(name):
        if name in tier1_names:
            return "Tier 1 — Fixed Camera"
        elif name in tier2_names:
            return "Tier 2 — Ekart Coverage"
        return "None"

    df_juncs_t3["coverage_tier"] = df_juncs_t3["junction_name"].apply(assign_tier)
    
    # Map styling
    def get_radius(tier):
        if tier == "Tier 1 — Fixed Camera": return 250
        elif tier == "Tier 2 — Ekart Coverage": return 180
        return 120

    def get_color(tier):
        if tier == "Tier 1 — Fixed Camera": return [91, 157, 240, 200]
        elif tier == "Tier 2 — Ekart Coverage": return [139, 121, 242, 160]
        return [120, 125, 135, 100]

    df_juncs_t3["radius"] = df_juncs_t3["coverage_tier"].apply(get_radius)
    df_juncs_t3["color"] = df_juncs_t3["coverage_tier"].apply(get_color)
    
    if toggle_mode == "Tier 1 — Fixed Cameras":
        df_map_t3 = df_juncs_t3[df_juncs_t3["coverage_tier"] == "Tier 1 — Fixed Camera"].copy()
    elif toggle_mode == "Tier 2 — Ekart Coverage":
        df_map_t3 = df_juncs_t3[df_juncs_t3["coverage_tier"] == "Tier 2 — Ekart Coverage"].copy()
    elif toggle_mode == "Both Tiers (Focus)":
        df_map_t3 = df_juncs_t3[df_juncs_t3["coverage_tier"] != "None"].copy()
    else:
        df_map_t3 = df_juncs_t3.copy()

    # Pre-format tooltip strings
    df_map_t3["violation_count_str"] = df_map_t3["violation_count"].apply(lambda x: f"{x:,}")
    df_map_t3["patrol_normalized_rate_str"] = df_map_t3["patrol_normalized_rate"].apply(lambda x: f"{x:.2f}")
    df_map_t3["avg_severity_str"] = df_map_t3["avg_severity"].apply(lambda x: f"{x:.2f}")

    # Base layer
    layer_t3 = pdk.Layer(
        "ScatterplotLayer",
        df_map_t3,
        get_position=["lng", "lat"],
        get_color="color",
        get_radius="radius",
        pickable=True,
        auto_highlight=True,
    )
    
    # Highlight ring for selected site (Magenta [255, 0, 200, 255])
    df_selected = df_juncs_t3[df_juncs_t3["junction_name"] == selected_site_name].copy()
    selected_ring_layer = pdk.Layer(
        "ScatterplotLayer",
        df_selected,
        get_position=["lng", "lat"],
        get_color=[255, 0, 200, 255], # Magenta ring
        get_radius=450,
        stroked=True,
        filled=False,
        line_width_min_pixels=3,
        pickable=False,
    )

    # Set map view centered on selected site if available, else center on city
    if not df_selected.empty:
        center_lat = df_selected["lat"].iloc[0]
        center_lng = df_selected["lng"].iloc[0]
        zoom_level = 13.0
    else:
        center_lat, center_lng = BENGALURU_CENTER[0], BENGALURU_CENTER[1]
        zoom_level = 11.5

    view_state_t3 = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lng,
        zoom=zoom_level,
        pitch=0,
    )
    
    deck_t3 = pdk.Deck(
        layers=[layer_t3, selected_ring_layer] if not df_selected.empty else [layer_t3],
        initial_view_state=view_state_t3,
        map_style=pdk.map_styles.CARTO_DARK,
        tooltip={
            "html": "<b>Junction:</b> {junction_name}<br>"
                    "<b>Coverage Tier:</b> {coverage_tier}<br>"
                    "<b>Violations:</b> {violation_count_str} (Rank #{rank_count})<br>"
                    "<b>Patrol-Normalized Rate:</b> {patrol_normalized_rate_str} viols/device (Rank #{rank_rate})",
            "style": {"backgroundColor": "#15171b", "color": "white", "fontSize": "13px"}
        }
    )
    
    st.pydeck_chart(deck_t3)

    # 4. Detail card rendering
    if not df_selected.empty:
        sel_row = df_selected.iloc[0]
        
        st.markdown("### Selected Camera Site Details")
        
        # Suffix description based on ranking
        qual_summary = f"Coverage Tier: {sel_row['coverage_tier']}. Ranked #{sel_row['rank_count']} in raw violation volume and #{sel_row['rank_rate']} in patrol-normalized rate citywide."
        if sel_row['junction_name'].startswith("BTP040"):
            qual_summary += " (Flagship Blind Spot: highest patrol-normalized rate in Bengaluru)"
            
        st.markdown(
            f"""
            <div style="background: rgba(91, 157, 240, 0.05); border: 1px solid rgba(91, 157, 240, 0.15); border-radius: 10px; padding: 24px; margin-top: 15px;">
                <h4 style="margin: 0 0 15px 0; color: #5b9df0; font-weight: 700; font-size: 20px; font-family: 'Space Grotesk', sans-serif;">{sel_row['junction_name']}</h4>
                <div class="metric-container" style="margin: 0 0 20px 0;">
                    <div class="metric-card" style="padding: 16px;">
                        <div class="metric-title">Violation Count</div>
                        <div class="metric-value" style="font-size: 28px; color: #FF6B6B;">{sel_row['violation_count']:,}</div>
                        <div class="metric-delta" style="color: #FFA07A;">Rank #{sel_row['rank_count']} citywide</div>
                    </div>
                    <div class="metric-card" style="padding: 16px;">
                        <div class="metric-title">Patrol-Normalized Rate</div>
                        <div class="metric-value" style="font-size: 28px; color: #4D96FF;">{sel_row['patrol_normalized_rate']:.2f}</div>
                        <div class="metric-delta" style="color: #6BCB77;">Rank #{sel_row['rank_rate']} citywide</div>
                    </div>
                    <div class="metric-card" style="padding: 16px;">
                        <div class="metric-title">Unique Patrol Devices</div>
                        <div class="metric-value" style="font-size: 28px; color: #6BCB77;">{int(sel_row['unique_devices'])}</div>
                        <div class="metric-delta" style="color: #aaa;">Enforcement presence</div>
                    </div>
                </div>
                <div style="background: rgba(255, 255, 255, 0.05); border-radius: 8px; padding: 12px 16px; border-left: 4px solid rgb(255, 0, 200);">
                    <p style="margin: 0; font-size: 14px; color: #ddd;">
                        <strong>Why this site qualified:</strong> {qual_summary}
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — Live Monitoring
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown(
        """
        <div class="route-tag">STOP 04 / 04 — LIVE PROOF</div>
        <div class="hero-title">Live Monitoring</div>
        <p class="hero-sub">Real-time automatic congestion surveillance proof-of-concept</p>
        """,
        unsafe_allow_html=True
    )
    
    # 1. Selector for camera sites
    # Calculate camera sites again to ensure identical list as Tab 3
    df_juncs_t4 = df_juncs.copy()
    df_juncs_t4["rank_count"] = df_juncs_t4["violation_count"].rank(ascending=False, method="min").astype(int)
    df_juncs_t4["rank_rate"] = df_juncs_t4["patrol_normalized_rate"].rank(ascending=False, method="min").astype(int)

    top_20_count_names = df_juncs_t4.nsmallest(20, "rank_count")["junction_name"].tolist()
    top_20_rate_names = df_juncs_t4.nsmallest(20, "rank_rate")["junction_name"].tolist()
    camera_site_names = sorted(list(set(top_20_count_names) & set(top_20_rate_names)))

    # Set default selected site to BTP211 (which has real off-peak readings)
    default_index = 0
    default_site = "BTP211 - Central Street Junction"
    if default_site in camera_site_names:
        default_index = camera_site_names.index(default_site)

    selected_monitor_site = st.selectbox(
        "Select Camera Site to Monitor:",
        camera_site_names,
        index=default_index,
        key="monitor_site_selectbox",
        help="Choose a site to test real-time Mappls API polling."
    )

    # 2. Fetch Live Data Button and logic
    st.markdown("### Live API Diagnostics")
    st.caption("This hits Mappls' live API in real time. If the IP address whitelist or token changes, it will fail honestly.")
    
    # Session state for live data
    if "live_fetch_results" not in st.session_state:
        st.session_state["live_fetch_results"] = {}

    import os
    has_token = bool(os.environ.get("MAPPLS_TOKEN"))
    
    if not has_token:
        st.info("ℹ️ **Demo Mode:** Live API fetching is disabled in the cloud deployment due to MapmyIndia IP-whitelisting restrictions. However, the simulation below demonstrates the underlying logic using historical data.")
        
    fetch_clicked = st.button("📡 Fetch Live Congestion Data", type="primary", disabled=not has_token)

    selected_row = df_juncs_t4[df_juncs_t4["junction_name"] == selected_monitor_site].iloc[0]
    
    if fetch_clicked:
        with st.spinner("Connecting to Mappls API..."):
            # Prepare dataframe for fetch_congestion
            test_df = pd.DataFrame({
                "h3_index": [selected_row["h3_index"] if "h3_index" in selected_row else "dummy"],
                "lat": [selected_row["lat"]],
                "lng": [selected_row["lng"]]
            })
            
            try:
                # Attempt live call
                from src.traffic import fetch_congestion
                live_res = fetch_congestion(test_df, top_n=1)
                
                # Succeeded! Save result
                cong_pct = live_res["congestion_pct"].iloc[0]
                freeflow_speed = live_res["freeflow_speed_kmh"].iloc[0]
                peak_speed = live_res["peak_speed_kmh"].iloc[0]
                st.session_state["live_fetch_results"][selected_monitor_site] = {
                    "success": True,
                    "value": cong_pct,
                    "freeflow_speed": freeflow_speed,
                    "peak_speed": peak_speed,
                    "timestamp": datetime.now().strftime("%I:%M:%S %p IST")
                }
            except Exception as e:
                # Failed honestly! Save error
                st.session_state["live_fetch_results"][selected_monitor_site] = {
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now().strftime("%I:%M:%S %p IST")
                }

    # Display results if any exist
    if selected_monitor_site in st.session_state["live_fetch_results"]:
        res = st.session_state["live_fetch_results"][selected_monitor_site]
        if res["success"]:
            st.success(f"✅ Live Connection Successful! (Fetched at {res['timestamp']})")
            
            # Fetch values with fallbacks
            freeflow_val = res.get("freeflow_speed", 0.0)
            peak_val = res.get("peak_speed", 0.0)
            
            st.html(
                f"""
                <style>
                @keyframes pulse {{
                    0% {{ transform: scale(0.9); opacity: 0.6; }}
                    50% {{ transform: scale(1.1); opacity: 1; }}
                    100% {{ transform: scale(0.9); opacity: 0.6; }}
                }}
                .live-pulse-dot {{
                    width: 10px;
                    height: 10px;
                    background-color: #34d399;
                    border-radius: 50%;
                    box-shadow: 0 0 8px #34d399, 0 0 16px #34d399;
                    animation: pulse 1.5s infinite ease-in-out;
                    display: inline-block;
                }}
                </style>
                <div style="background: linear-gradient(135deg, rgba(21, 23, 27, 0.95), rgba(10, 11, 13, 0.95)); border: 1px solid rgba(52, 211, 153, 0.3); border-radius: 12px; padding: 24px; margin: 20px 0; font-family: 'Inter', sans-serif;">
                    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255, 255, 255, 0.08); padding-bottom: 12px; margin-bottom: 20px;">
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <div class="live-pulse-dot"></div>
                            <span style="font-size: 14px; font-weight: 700; color: #34d399; letter-spacing: 0.05em; text-transform: uppercase; font-family: 'JetBrains Mono', monospace;">Live Feed Active</span>
                        </div>
                        <span style="font-size: 12px; color: #6c7078; font-weight: 500; font-family: 'JetBrains Mono', monospace;">Mappls Telematics Engine v2.0</span>
                    </div>
                    
                    <div style="display: flex; flex-direction: row; gap: 20px; align-items: center; flex-wrap: wrap;">
                        <!-- Left: Congestion percentage gauge -->
                        <div style="flex: 1; min-width: 180px;">
                            <div style="font-size: 13px; color: #6c7078; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; font-family: 'JetBrains Mono', monospace;">Live Congestion Index</div>
                            <div style="display: flex; align-items: baseline; gap: 4px; margin-top: 5px;">
                                <span style="font-family: 'Space Grotesk', sans-serif; font-size: 54px; font-weight: 800; background: linear-gradient(90deg, #5b9df0, #34d399); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{res['value']:.1f}%</span>
                            </div>
                            <div style="font-size: 12px; color: #a7abb3; margin-top: 8px; line-height: 1.4;">
                                Comparing current routing duration with freeflow (route_eta vs route_adv)
                            </div>
                        </div>
                        
                        <!-- Right: Supporting speed metrics -->
                        <div style="flex: 1.2; min-width: 240px; display: flex; gap: 20px; border-left: 1px solid rgba(255, 255, 255, 0.08); padding-left: 20px;">
                            <div style="flex: 1;">
                                <div style="font-size: 11px; color: #6c7078; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; font-family: 'JetBrains Mono', monospace;">Freeflow Speed</div>
                                <div style="font-family: 'Space Grotesk', sans-serif; font-size: 24px; font-weight: 700; color: #edeef0; margin-top: 5px;">{freeflow_val:.2f} <span style="font-size: 13px; font-weight: 400; color: #6c7078;">km/h</span></div>
                                <div style="font-size: 11px; color: #6c7078; margin-top: 4px; line-height: 1.3;">Theoretical maximum (route_adv)</div>
                            </div>
                            <div style="flex: 1;">
                                <div style="font-size: 11px; color: #6c7078; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; font-family: 'JetBrains Mono', monospace;">Peak (ETA) Speed</div>
                                <div style="font-family: 'Space Grotesk', sans-serif; font-size: 24px; font-weight: 700; color: #ef5350; margin-top: 5px;">{peak_val:.2f} <span style="font-size: 13px; font-weight: 400; color: #6c7078;">km/h</span></div>
                                <div style="font-size: 11px; color: #6c7078; margin-top: 4px; line-height: 1.3;">Real-time traffic speed (route_eta)</div>
                            </div>
                        </div>
                    </div>
                </div>
                """
            )
        else:
            # Show the error honestly
            st.error(f"❌ Live API Call Failed: {res['error']}")

    # 3. Simulated Time-Series Chart
    st.markdown("---")
    st.markdown(
        """
        <div class="km-label">
            <span class="km">SIM</span>
            <span class="title">Continuous Surveillance Simulation</span>
            <span class="rule"></span>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("This chart simulates what continuous camera monitoring would produce over a 72-hour period based on historical diurnal patterns, overlaid with any actual observed readings we have on file.")

    # Time series calculations
    traffic_val_df = pd.DataFrame()

    # Generate 72-hour timeline starting 2026-06-19 00:00:00
    start_date = pd.to_datetime("2026-06-19 00:00:00")
    time_index = pd.date_range(start=start_date, periods=72, freq="h")
    
    projection_df = pd.DataFrame({"timestamp": time_index})
    projection_df["hour"] = projection_df["timestamp"].dt.hour
    
    # Diurnal simulation curve peaking at 5 PM (17:00)
    def simulate_hourly_congestion(h):
        val = 1.0 + 0.35 * np.sin(2 * np.pi * (h - 11) / 24)
        if 8 <= h <= 10:
            val += 0.1
        return val

    np.random.seed(42)
    projection_df["projected_congestion"] = projection_df["hour"].apply(simulate_hourly_congestion) + np.random.normal(0, 0.04, len(projection_df))
    
    # Load actual observed readings for the selected site
    prefix = selected_monitor_site.split(" - ")[0]
    if not traffic_val_df.empty:
        site_readings = traffic_val_df[traffic_val_df["hotspot_name"].str.contains(prefix, na=False)].copy()
        if not site_readings.empty:
            site_readings["timestamp"] = pd.to_datetime(site_readings["timestamp"])
            site_readings["timestamp_hour"] = site_readings["timestamp"].dt.round("h")
            real_hourly = site_readings.groupby("timestamp_hour")["congestion_ratio"].mean().reset_index()
            real_hourly = real_hourly.rename(columns={"timestamp_hour": "timestamp", "congestion_ratio": "actual_congestion"})
        else:
            real_hourly = pd.DataFrame(columns=["timestamp", "actual_congestion"])
    else:
        real_hourly = pd.DataFrame(columns=["timestamp", "actual_congestion"])
        
    # Plotly Line Chart
    fig_monitor = go.Figure()
    
    # Add projection line (Dashed)
    fig_monitor.add_trace(
        go.Scatter(
            x=projection_df["timestamp"],
            y=projection_df["projected_congestion"],
            name="Simulated projection based on observed patterns",
            line=dict(color="rgba(140, 160, 200, 0.65)", width=2.5, dash="dash"),
            mode="lines"
        )
    )
    
    # Add actual observed readings (Solid Dots) if they exist
    if not real_hourly.empty:
        fig_monitor.add_trace(
            go.Scatter(
                x=real_hourly["timestamp"],
                y=real_hourly["actual_congestion"],
                name="Actual observed readings",
                mode="markers",
                marker=dict(color="#34d399", size=10, symbol="circle", line=dict(color="white", width=1.5))
            )
        )
        
    fig_monitor.update_layout(
        title=f"72-Hour Congestion Surveillance — {selected_monitor_site}",
        title_font=dict(size=16, family="Space Grotesk", color="white"),
        xaxis=dict(
            title="Time (June 19 – June 21, 2026)",
            gridcolor="rgba(255,255,255,0.05)",
            color="white"
        ),
        yaxis=dict(
            title="Congestion Ratio",
            gridcolor="rgba(255,255,255,0.05)",
            color="white",
            range=[0.7, 1.7]
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="white")
        ),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    st.plotly_chart(fig_monitor, use_container_width=True)
    
    # Caption clarifying real vs simulated
    if not real_hourly.empty:
        st.caption(
            f"ℹ️ **Honest Prototype Note**: The dashed line represents a simulated diurnal projection based on historical patterns. "
            f"The {len(real_hourly)} solid blue dots are actual observed readings recorded at this site on June 20/21 from our validation log."
        )
    else:
        st.caption(
            "ℹ️ **Honest Prototype Note**: The dashed line represents a simulated diurnal projection based on historical patterns. "
            "No actual observed readings are currently recorded in the traffic validation log for this specific site."
        )

# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — Delivery Coverage
# ════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("<h1 style='margin-bottom:0;'>Delivery Fleet Coverage</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:18px; color:#aaa; margin-top:0;'>Do existing delivery networks already saturate our enforcement hotspots?</p>",
        unsafe_allow_html=True,
    )

    df_juncs_t5 = df_juncs.copy()
    df_juncs_t5["rank_count"] = df_juncs_t5["violation_count"].rank(ascending=False, method="min").astype(int)
    df_juncs_t5["rank_rate"]  = df_juncs_t5["patrol_normalized_rate"].rank(ascending=False, method="min").astype(int)
    _top20c = df_juncs_t5.nsmallest(20, "rank_count")["junction_name"].tolist()
    _top20r = df_juncs_t5.nsmallest(20, "rank_rate")["junction_name"].tolist()
    _cam_names = sorted(set(_top20c) & set(_top20r))
    cam_t5 = df_juncs_t5[df_juncs_t5["junction_name"].isin(_cam_names)].reset_index(drop=True)

    with st.spinner("Fetching live restaurant & dark store data from OpenStreetMap…"):
        rest_df, ds_df, total_rest, rest_err, ds_used_fallback = fetch_delivery_coverage(
            tuple(cam_t5["lat"].tolist()),
            tuple(cam_t5["lng"].tolist()),
            tuple(cam_t5["junction_name"].tolist()),
            tuple(cam_t5["violation_count"].tolist()),
        )

    avg_rest    = rest_df["restaurant_count"].mean() if not rest_df.empty else 0
    n_blinkit   = int((ds_df["brand"] == "Blinkit").sum())   if not ds_df.empty else 0
    n_zepto     = int((ds_df["brand"] == "Zepto").sum())     if not ds_df.empty else 0
    n_instamart = int((ds_df["brand"] == "Instamart").sum()) if not ds_df.empty else 0
    ds_source_label = "Neighbourhood approx." if ds_used_fallback else "OSM live data"

    st.markdown(
        f"""
        <div class="metric-container">
            <div class="metric-card">
                <div class="metric-title">Camera Sites Analysed</div>
                <div class="metric-value" style="color:#00d4ff;">{len(_cam_names)}</div>
                <div class="metric-delta" style="color:#aaa;">Recommended enforcement points</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Avg Restaurants within 500 m</div>
                <div class="metric-value" style="color:#FF8C00;">{avg_rest:.0f}</div>
                <div class="metric-delta" style="color:#aaa;">Per camera site · OSM data</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Blinkit Dark Stores</div>
                <div class="metric-value" style="color:#FFD700;">{n_blinkit}</div>
                <div class="metric-delta" style="color:#aaa;">Bengaluru · {ds_source_label}</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Zepto Dark Stores</div>
                <div class="metric-value" style="color:#A855F7;">{n_zepto + n_instamart}</div>
                <div class="metric-delta" style="color:#aaa;">Bengaluru · {ds_source_label}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="display:flex;gap:24px;flex-wrap:wrap;margin:8px 0 14px 0;font-size:13px;color:#ccc;">
            <span><span style="display:inline-block;width:12px;height:12px;border-radius:50%;
                background:#00d4ff;margin-right:6px;vertical-align:middle;"></span>Camera Site Junction</span>
            <span><span style="display:inline-block;width:12px;height:12px;border-radius:50%;
                background:#FF8C00;margin-right:6px;vertical-align:middle;"></span>Restaurant Density (bubble size = count within 500 m)</span>
            <span><span style="display:inline-block;width:12px;height:12px;border-radius:50%;
                background:#FFD700;margin-right:6px;vertical-align:middle;"></span>Blinkit Dark Store (ring = 2 km delivery radius)</span>
            <span><span style="display:inline-block;width:12px;height:12px;border-radius:50%;
                background:#A855F7;margin-right:6px;vertical-align:middle;"></span>Zepto Dark Store (ring = 2 km delivery radius)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _layers = []

    _bg = df_juncs.copy()
    _bg["color"] = [[80, 100, 140, 55]] * len(_bg)
    _layers.append(pdk.Layer("ScatterplotLayer", _bg,
        get_position=["lng", "lat"], get_color="color", get_radius=70, pickable=False))

    if not rest_df.empty and rest_df["restaurant_count"].max() > 0:
        _max_r = rest_df["restaurant_count"].max()
        rest_df["_norm"]   = (rest_df["restaurant_count"] / (_max_r + 1e-9)).clip(0.1, 1.0)
        rest_df["_radius"] = (100 + rest_df["_norm"] * 380).astype(int)
        rest_df["_color"]  = rest_df["_norm"].apply(lambda n: [255, int(200 * (1 - n) + 60), 0, 150])
        rest_df["_tip_name"] = rest_df["junction_name"]
        rest_df["_tip_rest"] = rest_df["restaurant_count"].apply(lambda x: f"{x} restaurants within 500 m")
        rest_df["_tip_viol"] = rest_df["violation_count"].apply(lambda x: f"{x:,} historical violations")
        _layers.append(pdk.Layer("ScatterplotLayer", rest_df,
            get_position=["lng", "lat"], get_color="_color", get_radius="_radius",
            pickable=True, auto_highlight=True))

    if not ds_df.empty:
        _blinkit = ds_df[ds_df["brand"] == "Blinkit"].copy()
        _zepto   = ds_df[ds_df["brand"].isin(["Zepto", "Instamart"])].copy()

        for _sub, _fill, _line in [
            (_blinkit, [255, 215, 0, 12],  [255, 215, 0, 70]),
            (_zepto,   [168, 85, 247, 12], [168, 85, 247, 70]),
        ]:
            if _sub.empty:
                continue
            _layers.append(pdk.Layer("ScatterplotLayer", _sub,
                get_position=["lng", "lat"], get_color=_fill, get_radius=2000,
                stroked=True, filled=True, line_width_min_pixels=1, get_line_color=_line,
                pickable=False))

        for _sub, _dot_color in [
            (_blinkit, [255, 215, 0, 230]),
            (_zepto,   [168, 85, 247, 230]),
        ]:
            if _sub.empty:
                continue
            _sub = _sub.copy()
            _sub["_tip_name"] = _sub["display_name"]
            _sub["_tip_rest"] = _sub["brand"] + " dark store"
            _sub["_tip_viol"] = "2 km delivery radius shown"
            _layers.append(pdk.Layer("ScatterplotLayer", _sub,
                get_position=["lng", "lat"], get_color=_dot_color, get_radius=160,
                pickable=True, auto_highlight=True))

    _cam_map = cam_t5.copy()
    _cam_map["_tip_name"] = _cam_map["junction_name"]
    _cam_map["_tip_rest"] = rest_df.set_index("junction_name")["restaurant_count"].reindex(
        _cam_map["junction_name"].values).fillna(0).astype(int).apply(
        lambda x: f"{x} restaurants within 500 m").values if not rest_df.empty else "—"
    _cam_map["_tip_viol"] = _cam_map["violation_count"].apply(lambda x: f"{x:,} historical violations")
    _layers.append(pdk.Layer("ScatterplotLayer", _cam_map,
        get_position=["lng", "lat"],
        get_color=[0, 212, 255, 220], get_radius=220,
        stroked=True, filled=True, line_width_min_pixels=2, get_line_color=[0, 212, 255, 255],
        pickable=True, auto_highlight=True))

    _view_t5 = pdk.ViewState(latitude=12.9716, longitude=77.5800, zoom=11.8, pitch=0)
    _deck_t5 = pdk.Deck(
        layers=_layers,
        initial_view_state=_view_t5,
        map_style=pdk.map_styles.CARTO_DARK,
        tooltip={
            "html": "<b>{_tip_name}</b><br/>{_tip_rest}<br/><span style='color:#aaa;'>{_tip_viol}</span>",
            "style": {"backgroundColor": "#1a1a2e", "color": "white", "fontSize": "13px"},
        },
    )
    st.pydeck_chart(_deck_t5)

    if not rest_df.empty and rest_df["restaurant_count"].max() > 0:
        _best = rest_df.loc[rest_df["restaurant_count"].idxmax()]
        _sites_above_10 = int((rest_df["restaurant_count"] >= 10).sum())
        st.markdown(
            f"""
            <div style="background:rgba(255,140,0,0.08);border:1px solid rgba(255,140,0,0.25);
                        border-radius:12px;padding:18px;margin:16px 0;">
                <p style="margin:0;font-size:15px;color:#FF8C00;font-weight:500;line-height:1.6;">
                    🟣 <strong>Coverage confirmed:</strong> <strong>{_best['junction_name']}</strong>
                    has <strong>{int(_best['restaurant_count'])} restaurants within 500 m</strong>,
                    generating a continuous stream of delivery riders through this exact enforcement
                    hotspot. <strong>{_sites_above_10} of {len(_cam_names)} camera sites</strong>
                    have 10+ nearby restaurants — meaning delivery drivers already traverse every
                    one of our recommended locations multiple times per hour with no additional
                    infrastructure required.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Restaurant data could not be fetched — check internet connectivity or Overpass API availability.")

    if ds_used_fallback:
        st.markdown(
            """
            <div style="background:rgba(168,85,247,0.07);border:1px solid rgba(168,85,247,0.2);
                        border-radius:12px;padding:16px;margin:10px 0;">
                <p style="margin:0;font-size:14px;color:#C084FC;line-height:1.5;">
                    ℹ️ <strong>Dark store data source:</strong> OSM has sparse coverage of private
                    commercial dark stores in Bengaluru, so locations shown are compiled from
                    publicly available service-area information on the Blinkit and Zepto apps
                    (neighbourhood-level approximations, not exact warehouse addresses).
                    Both platforms serve all of central Bengaluru — the same area as our camera sites.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.1);
                    border-radius:12px;padding:16px;margin:16px 0;">
            <p style="margin:0;font-size:14px;color:#ccc;line-height:1.6;">
                <strong>📝 Note:</strong> Due to lack of data, only <strong>5 Blinkit stores</strong> and
                <strong>9 Zepto stores</strong> appear on the map. All of these appear in
                <strong>non-hotspot regions</strong> — which is actually good for our case, since these
                regions are likely not getting covered by police effectively and thus can be covered by
                Zepto/Blinkit drivers acting as a passive enforcement presence.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        f"Data: OpenStreetMap via Overpass API (CC BY-SA). "
        f"Restaurant count = amenity:restaurant/cafe/fast_food/food_court within 500 m of each camera site. "
        f"Total food establishments found in camera-site bounding box: {total_rest:,}. "
        + (f"⚠️ Restaurant query error: {rest_err}" if rest_err else "Fetched successfully.")
    )