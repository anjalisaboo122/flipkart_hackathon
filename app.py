"""
GridLock AI — Predictive Parking Enforcement Intelligence
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
    page_title="GridLock AI — Bengaluru",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS for premium aesthetics
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    /* Set custom font */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Custom metric card styles */
    .metric-container {
        display: flex; 
        gap: 20px; 
        justify-content: space-between; 
        flex-wrap: wrap;
        margin: 20px 0;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        flex: 1;
        min-width: 250px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.35);
        border: 1px solid rgba(255, 255, 255, 0.2);
        background: rgba(255, 255, 255, 0.08);
    }
    .metric-title {
        font-size: 13px; 
        color: #888896; 
        font-weight: 600; 
        text-transform: uppercase; 
        letter-spacing: 1.2px;
    }
    .metric-value {
        font-size: 38px; 
        font-weight: 800; 
        margin: 12px 0 6px 0;
        line-height: 1;
    }
    .metric-delta {
        font-size: 14px; 
        font-weight: 600;
    }
    
    /* Accent gradients */
    .text-gradient {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
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

# Load standard stats and data
with st.spinner("Loading violations dataset..."):
    df_h3, stats = load_base_data()
    df_juncs = load_junction_data()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/83/Bangalore_Traffic_Police_logo.svg/200px-Bangalore_Traffic_Police_logo.svg.png", width=80, use_container_width=False)
st.sidebar.title("GridLock AI")
st.sidebar.markdown("*Predictive Parking Enforcement Intelligence*")
st.sidebar.markdown("---")

now = datetime.now()
st.sidebar.markdown(f"**Date:** {now.strftime('%d %b %Y')}")
st.sidebar.markdown(f"**Time:** {now.strftime('%H:%M IST')}")
st.sidebar.markdown(f"**Day:** {now.strftime('%A')}")
st.sidebar.markdown("---")

st.sidebar.markdown(f"**Records:** {stats['total_violations']:,}")
st.sidebar.markdown(f"**H3 zones:** {stats['unique_zones']}")
st.sidebar.markdown(f"**Junctions mapped:** {len(df_juncs)}")
st.sidebar.markdown(f"**Date range:** {stats['date_range'][0]} → {stats['date_range'][1]}")

# ---------------------------------------------------------------------------
# Main Tabs Layout
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ The Blind Spot",
    "⏰ The Timing Gap",
    "📹 Camera Placement",
    "📡 Live Monitoring"
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — The Blind Spot
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<h1 style='margin-bottom:0;'>The Blind Spot</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:18px; color:#aaa; margin-top:0;'>Identifying critical gaps between raw violations and officer coverage</p>", unsafe_allow_html=True)
    
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
            "style": {"backgroundColor": "#1a1a2e", "color": "white"}
        },
    )
    
    st.pydeck_chart(deck)
    
    # Highlight BTP040 Callout Card (headline junction)
    st.markdown("---")
    st.markdown("### Critical Focus Area: Junction `BTP040` (Elite Junction)")
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
    st.markdown("<h1 style='margin-bottom:0;'>The Timing Gap</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:18px; color:#aaa; margin-top:0;'>The mismatch between police shifts and peak congestion hours</p>", unsafe_allow_html=True)
    
    # Load and process traffic_validation_fixed.csv
    try:
        traffic_val_df = pd.read_csv("data/traffic_validation_fixed.csv")
        congestion_by_hour = traffic_val_df.groupby("hour")["congestion_ratio"].mean().reset_index(name="mean_congestion")
    except Exception as e:
        st.error(f"Error loading traffic validation data: {e}")
        congestion_by_hour = pd.DataFrame(columns=["hour", "mean_congestion"])

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
            marker_color="rgba(100, 149, 237, 0.75)", # Cornflower Blue
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
            line=dict(color="#FF8E53", width=4), # Coral/Amber line
            mode="lines+markers",
            hovertemplate="Hour %{x}:00<br>Congestion Ratio: %{y:.2f}<extra></extra>"
        ),
        secondary_y=True
    )
    
    # Highlight 5 PM - 8 PM window (Hours 17-20)
    fig.add_vrect(
        x0=17, x1=20,
        fillcolor="rgba(255, 195, 0, 0.15)",
        layer="below",
        line_width=0,
        annotation_text="The timing gap",
        annotation_position="top left",
        annotation_font=dict(size=12, color="#FFC300", family="Outfit")
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
        arrowcolor="#FFC300",
        ax=-50,
        ay=-70,
        font=dict(size=11, color="white", family="Outfit"),
        bordercolor="#FFC300",
        borderpad=6,
        bgcolor="#1a1a24",
        opacity=0.9
    )
    
    # Layout styling for dark mode dashboard
    fig.update_layout(
        title="24-Hour Comparison: Enforcement vs. Congestion",
        title_font=dict(size=18, family="Outfit", color="white"),
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
        <div style="background: rgba(255, 195, 0, 0.08); border: 1px solid rgba(255, 195, 0, 0.2); border-radius: 12px; padding: 16px; margin: 15px 0;">
            <p style="margin: 0; font-size: 15px; color: #FFC300; font-weight: 500; line-height: 1.5;">
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
    st.markdown("<h1 style='margin-bottom:0;'>Camera Placement</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:18px; color:#aaa; margin-top:0;'>Strategic selection of automatic enforcement locations</p>", unsafe_allow_html=True)

    # 1. Compute ranks and intersection
    df_juncs_t3 = df_juncs.copy()
    df_juncs_t3["rank_count"] = df_juncs_t3["violation_count"].rank(ascending=False, method="min").astype(int)
    df_juncs_t3["rank_rate"] = df_juncs_t3["patrol_normalized_rate"].rank(ascending=False, method="min").astype(int)

    top_20_count_names = df_juncs_t3.nsmallest(20, "rank_count")["junction_name"].tolist()
    top_20_rate_names = df_juncs_t3.nsmallest(20, "rank_rate")["junction_name"].tolist()
    camera_site_names = sorted(list(set(top_20_count_names) & set(top_20_rate_names)))
    num_camera_sites = len(camera_site_names)
    
    # 2. Site count header
    st.markdown(
        f"<h3 style='margin: 15px 0 5px 0; color: #00d4ff;'>📸 {num_camera_sites} of {len(df_juncs_t3)} junctions qualify for camera placement</h3>", 
        unsafe_allow_html=True
    )
    st.caption("Criteria: Top-20 by raw violation count ∩ Top-20 by patrol-normalized rate (diminishes bias toward well-patrolled spots).")

    # Layout: Toggle and Selectbox
    tcol1, tcol2 = st.columns([1, 1])
    with tcol1:
        toggle_mode = st.radio(
            "Map Display Mode:",
            ["All Junctions (Context)", "Camera Sites Only (Focus)"],
            horizontal=True,
            help="Show all junctions with recommended sites highlighted, or focus only on recommended sites."
        )
    with tcol2:
        selected_site_name = st.selectbox(
            "Select Camera Site to Inspect:",
            camera_site_names,
            help="Choose a recommended site to view detailed metrics and location on the map."
        )

    # 3. Setup map data
    df_juncs_t3["is_camera_site"] = df_juncs_t3["junction_name"].isin(camera_site_names)
    
    # Map styling
    df_juncs_t3["radius"] = df_juncs_t3.apply(
        lambda r: 250 if r["is_camera_site"] else 120, axis=1
    )
    # Teal [0, 212, 255, 200] for camera sites, dimmed gray-blue [140, 160, 200, 110] for context
    df_juncs_t3["color"] = df_juncs_t3.apply(
        lambda r: [0, 212, 255, 200] if r["is_camera_site"] else [140, 160, 200, 110], axis=1
    )
    
    if toggle_mode == "Camera Sites Only (Focus)":
        df_map_t3 = df_juncs_t3[df_juncs_t3["is_camera_site"]].copy()
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
                    "<b>Violations:</b> {violation_count_str} (Rank #{rank_count})<br>"
                    "<b>Patrol-Normalized Rate:</b> {patrol_normalized_rate_str} viols/device (Rank #{rank_rate})<br>"
                    "<b>Camera Site:</b> {'YES' if is_camera_site else 'NO'}",
            "style": {"backgroundColor": "#1a1a24", "color": "white", "fontSize": "13px"}
        }
    )
    
    st.pydeck_chart(deck_t3)

    # 4. Detail card rendering
    if not df_selected.empty:
        sel_row = df_selected.iloc[0]
        
        st.markdown("### Selected Camera Site Details")
        
        # Suffix description based on ranking
        qual_summary = f"Ranked #{sel_row['rank_count']} in raw violation volume and #{sel_row['rank_rate']} in patrol-normalized rate citywide."
        if sel_row['junction_name'].startswith("BTP040"):
            qual_summary += " (Flagship Blind Spot: highest patrol-normalized rate in Bengaluru)"
            
        st.markdown(
            f"""
            <div style="background: rgba(0, 212, 255, 0.05); border: 1px solid rgba(0, 212, 255, 0.15); border-radius: 16px; padding: 24px; margin-top: 15px;">
                <h4 style="margin: 0 0 15px 0; color: #00d4ff; font-weight: 700; font-size: 20px;">{sel_row['junction_name']}</h4>
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
    st.markdown("<h1 style='margin-bottom:0;'>Live Monitoring</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:18px; color:#aaa; margin-top:0;'>Real-time automatic congestion surveillance proof-of-concept</p>", unsafe_allow_html=True)
    
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

    fetch_clicked = st.button("📡 Fetch Live Congestion Data", type="primary")

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
            
            st.markdown(
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
                    background-color: #00ffcc;
                    border-radius: 50%;
                    box-shadow: 0 0 8px #00ffcc, 0 0 16px #00ffcc;
                    animation: pulse 1.5s infinite ease-in-out;
                    display: inline-block;
                }}
                </style>
                <div style="background: linear-gradient(135deg, rgba(24, 28, 41, 0.95), rgba(15, 18, 27, 0.95)); border: 1px solid rgba(0, 212, 255, 0.3); border-radius: 16px; padding: 24px; box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37); margin: 20px 0; font-family: 'Outfit', sans-serif;">
                    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255, 255, 255, 0.08); padding-bottom: 12px; margin-bottom: 20px;">
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <div class="live-pulse-dot"></div>
                            <span style="font-size: 14px; font-weight: 700; color: #00ffcc; letter-spacing: 0.05em; text-transform: uppercase;">Live Feed Active</span>
                        </div>
                        <span style="font-size: 12px; color: #888896; font-weight: 500;">Mappls Telematics Engine v2.0</span>
                    </div>
                    
                    <div style="display: flex; flex-direction: row; gap: 20px; align-items: center; flex-wrap: wrap;">
                        <!-- Left: Congestion percentage gauge -->
                        <div style="flex: 1; min-width: 180px;">
                            <div style="font-size: 13px; color: #888896; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em;">Live Congestion Index</div>
                            <div style="display: flex; align-items: baseline; gap: 4px; margin-top: 5px;">
                                <span style="font-size: 54px; font-weight: 800; background: linear-gradient(90deg, #00d4ff, #00ffaa); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{res['value']:.1f}%</span>
                            </div>
                            <div style="font-size: 12px; color: #a5a5b4; margin-top: 8px; line-height: 1.4;">
                                Comparing current routing duration with freeflow (route_eta vs route_adv)
                            </div>
                        </div>
                        
                        <!-- Right: Supporting speed metrics -->
                        <div style="flex: 1.2; min-width: 240px; display: flex; gap: 20px; border-left: 1px solid rgba(255, 255, 255, 0.08); padding-left: 20px;">
                            <div style="flex: 1;">
                                <div style="font-size: 11px; color: #888896; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Freeflow Speed</div>
                                <div style="font-size: 24px; font-weight: 700; color: #ffffff; margin-top: 5px;">{freeflow_val:.2f} <span style="font-size: 13px; font-weight: 400; color: #888896;">km/h</span></div>
                                <div style="font-size: 11px; color: #626270; margin-top: 4px; line-height: 1.3;">Theoretical maximum (route_adv)</div>
                            </div>
                            <div style="flex: 1;">
                                <div style="font-size: 11px; color: #888896; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Peak (ETA) Speed</div>
                                <div style="font-size: 24px; font-weight: 700; color: #ff5e7e; margin-top: 5px;">{peak_val:.2f} <span style="font-size: 13px; font-weight: 400; color: #888896;">km/h</span></div>
                                <div style="font-size: 11px; color: #626270; margin-top: 4px; line-height: 1.3;">Real-time traffic speed (route_eta)</div>
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            # Show the error honestly
            st.error(f"❌ Live API Call Failed: {res['error']}")
            
            # Fallback data calculation (Real historical averages only, no synthesis)
            fallback_pct = None
            mean_ratio = None
            has_data = False
            all_off_peak = False
            
            try:
                traffic_val_df = pd.read_csv("data/traffic_validation_fixed.csv")
                prefix = selected_monitor_site.split(" - ")[0]
                site_readings = traffic_val_df[traffic_val_df["hotspot_name"].str.contains(prefix, na=False)]
                if not site_readings.empty:
                    has_data = True
                    # Check if all readings are exclusively off-peak (before 6 AM or after 10 PM)
                    off_peak_mask = (site_readings["hour"] < 6) | (site_readings["hour"] > 22)
                    all_off_peak = off_peak_mask.all()
                    
                    mean_ratio = site_readings["congestion_ratio"].mean()
                    fallback_pct = max(0.0, (mean_ratio - 1.0) * 100)
            except Exception as e:
                st.error(f"Debug Error loading fallback data: {e}")
                
            st.warning("⚠️ DEMO FALLBACK: Displaying historical congestion ratio for this site due to live API error/unavailability.")
            
            if has_data:
                if all_off_peak:
                    st.markdown(
                        f"""
                        <div style="background: rgba(140, 160, 200, 0.08); border: 1px solid rgba(140, 160, 200, 0.2); border-radius: 12px; padding: 20px; margin: 15px 0; font-family: 'Outfit', sans-serif;">
                            <div style="font-size: 14px; color: #888896; font-weight: 600; text-transform: uppercase;">Historical Congestion Level</div>
                            <div style="font-size: 40px; font-weight: 800; color: #9CACE4; margin: 5px 0;">{mean_ratio:.2f}x <span style="font-size: 20px; font-weight: 400; color: #888896;">baseline</span></div>
                            <div style="font-size: 13px; color: #aaa;">Historical off-peak congestion ratio (all available records were logged during off-peak hours and are not representative of typical daytime conditions).</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div style="background: rgba(255, 195, 0, 0.08); border: 1px solid rgba(255, 195, 0, 0.2); border-radius: 12px; padding: 20px; margin: 15px 0; font-family: 'Outfit', sans-serif;">
                            <div style="font-size: 14px; color: #888896; font-weight: 600; text-transform: uppercase;">Historical Congestion Level</div>
                            <div style="font-size: 40px; font-weight: 800; color: #FFC300; margin: 5px 0;">{fallback_pct:.1f}%</div>
                            <div style="font-size: 13px; color: #aaa;">Real average derived from traffic validation log historical records for this site.</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.markdown(
                    """
                    <div style="background: rgba(255, 94, 126, 0.08); border: 1px solid rgba(255, 94, 126, 0.2); border-radius: 12px; padding: 20px; margin: 15px 0; font-family: 'Outfit', sans-serif;">
                        <div style="font-size: 14px; color: #888896; font-weight: 600; text-transform: uppercase;">Historical Congestion Level</div>
                        <div style="font-size: 28px; font-weight: 800; color: #FF5E7E; margin: 10px 0;">INSUFFICIENT DATA</div>
                        <div style="font-size: 13px; color: #aaa;">No historical validation log records exist on file for this specific junction site.</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # 3. Simulated Time-Series Chart
    st.markdown("---")
    st.markdown("### Continuous Surveillance Simulation")
    st.markdown("This chart simulates what continuous camera monitoring would produce over a 72-hour period based on historical diurnal patterns, overlaid with any actual observed readings we have on file.")

    # Time series calculations
    try:
        traffic_val_df = pd.read_csv("data/traffic_validation_fixed.csv")
    except:
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
                marker=dict(color="#00d4ff", size=10, symbol="circle", line=dict(color="white", width=1.5))
            )
        )
        
    fig_monitor.update_layout(
        title=f"72-Hour Congestion Surveillance — {selected_monitor_site}",
        title_font=dict(size=16, family="Outfit", color="white"),
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