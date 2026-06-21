"""
GridLock AI — Predictive Parking Enforcement Intelligence
Bengaluru Traffic Police | Flipkart Gridlock Hackathon

Run: streamlit run app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import folium
import plotly.express as px
import plotly.graph_objects as go
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from datetime import datetime

from src.data_processing  import load_data, add_h3_index, compute_zones, get_daily_series, get_hourly_distribution, get_summary_stats
from src.predictor        import train_prophet_models, validate_may, get_zone_predictions, get_current_hour_risk, get_validation_chart_data
from src.traffic          import fetch_congestion, compute_dynamic_risk
from src.patrol_optimizer import plan_patrol
from src.anomaly          import compute_baselines, detect_anomalies_now, compute_zone_volatility
from src.briefing         import generate_briefing
from src.impact           import compute_zone_impact, validate_against_traffic, get_city_totals
from config               import TOP_ZONES_PROPHET, ANTHROPIC_API_KEY, TOMTOM_API_KEY, MAPPLS_TOKEN

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title = "GridLock AI — Bengaluru",
    page_icon  = "🚔",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

BENGALURU_CENTER = [12.9716, 77.5946]


# ---------------------------------------------------------------------------
# Pipeline (cached — runs once, then cached across interactions)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_pipeline():
    df     = load_data()
    df_h3  = add_h3_index(df)
    zones  = compute_zones(df_h3)

    top_ids    = zones.head(TOP_ZONES_PROPHET)["h3_index"].tolist()
    daily_ser  = get_daily_series(df_h3, top_ids)
    hour_dist  = get_hourly_distribution(df_h3, top_ids)

    models     = train_prophet_models(daily_ser)
    val_result = validate_may(models, daily_ser)
    pred_df    = get_zone_predictions(models, daily_ser, hour_dist, zones)

    baselines  = compute_baselines(df_h3)
    volatility = compute_zone_volatility(df_h3, zones)
    impact_df  = compute_zone_impact(df_h3, zones)

    stats      = get_summary_stats(df_h3)

    return {
        "df":         df_h3,
        "zones":      zones,
        "models":     models,
        "daily_ser":  daily_ser,
        "hour_dist":  hour_dist,
        "val_result": val_result,
        "pred_df":    pred_df,
        "baselines":  baselines,
        "volatility": volatility,
        "impact_df":  impact_df,
        "stats":      stats,
    }


# ---------------------------------------------------------------------------
# Load with progress bar
# ---------------------------------------------------------------------------

with st.spinner("Training Prophet models on 298,282 violation records… (first load only)"):
    data = load_pipeline()

df        = data["df"]
zones     = data["zones"]
models    = data["models"]
daily_ser = data["daily_ser"]
hour_dist = data["hour_dist"]
val_result= data["val_result"]
pred_df   = data["pred_df"]
baselines = data["baselines"]
volatility= data["volatility"]
impact_df = data["impact_df"]
stats     = data["stats"]

# ---------------------------------------------------------------------------
# Fetch live traffic (not cached — refreshes each session)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def get_live_traffic(_zones):
    return fetch_congestion(_zones, top_n=100)

traffic_df     = get_live_traffic(zones)
traffic_source = traffic_df["source"].iloc[0] if not traffic_df.empty else "simulated"

# Current risk = Prophet predictions × live traffic amplifier
current_risk = get_current_hour_risk(pred_df)

# Fallback: if Prophet produced no predictions, rank by historical violation count
if current_risk.empty:
    st.warning("Prophet predictions unavailable — showing historical violation rankings.", icon="⚠️")
    current_risk = zones.copy()
    current_risk["predicted_hourly"] = current_risk["violation_count"] / (180 * 24)
    current_risk["predicted_daily"]  = current_risk["violation_count"] / 180

dynamic_risk = compute_dynamic_risk(current_risk, traffic_df)

# Traffic impact quantification
impact_with_traffic = impact_df.merge(
    traffic_df[["h3_index", "congestion_pct"]], on="h3_index", how="left"
)
validation_result = validate_against_traffic(impact_df, traffic_df)
city_totals       = get_city_totals(impact_df)

# Anomalies for current time slot
anomalies = detect_anomalies_now(df, zones, baselines, top_n=10)


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

source_icon = "🟢" if traffic_source in ("tomtom", "mmi") else "🟡"
source_label = {"tomtom": "TomTom (Live)", "mmi": "MapMyIndia (Live)", "simulated": "Simulated"}.get(traffic_source, traffic_source)
st.sidebar.markdown(f"**Traffic:** {source_icon} {source_label}")
st.sidebar.markdown(f"**Prophet zones:** {len(models)}")
st.sidebar.markdown(f"**Anomalies now:** {len(anomalies)}")
st.sidebar.markdown("---")

st.sidebar.markdown(f"**Records:** {stats['total_violations']:,}")
st.sidebar.markdown(f"**H3 zones:** {len(zones)}")
st.sidebar.markdown(f"**Date range:** {stats['date_range'][0]} → {stats['date_range'][1]}")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ Live Prediction Map",
    "🚔 Patrol Planner",
    "⚠️ Anomaly Alerts",
    "📋 Officer Briefing",
    "📊 Model Validation",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Live Prediction Map
# ════════════════════════════════════════════════════════════════════════════

with tab1:
    st.header("Live Violation Risk — Next 6 Hours")
    st.markdown(
        f"Predicted hotspots for **{now.strftime('%A %d %b, %H:%M')}** "
        f"amplified by **{source_label}** traffic congestion."
    )

    # City-wide impact headline metrics
    total_cost = city_totals["total_economic_cost_inr_day"]
    total_delay = city_totals["total_vehicle_delay_min_day"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Daily Economic Loss", f"₹{total_cost/1e5:.1f}L", help="Rupee cost of commuter delay caused by illegal parking across all zones")
    col2.metric("Vehicle-Minutes Lost/Day", f"{total_delay/1000:.0f}K", help="Total extra delay experienced by all affected vehicles")
    col3.metric("Predicted Violations Today (top zone)", f"{dynamic_risk.iloc[0]['predicted_daily']:.0f}" if not dynamic_risk.empty else "N/A")
    col4.metric("Active Anomaly Alerts", len(anomalies))

    # Folium heatmap
    m = folium.Map(location=BENGALURU_CENTER, zoom_start=12, tiles="CartoDB positron")

    if not dynamic_risk.empty:
        heat_data = []
        for _, row in dynamic_risk.iterrows():
            if pd.notna(row.get("lat")) and pd.notna(row.get("lng")):
                heat_data.append([row["lat"], row["lng"], float(row["dynamic_risk"])])

        HeatMap(
            heat_data,
            radius=20, blur=15, max_zoom=13,
            gradient={0.2: "blue", 0.5: "yellow", 0.8: "orange", 1.0: "red"},
        ).add_to(m)

        # Top 10 markers
        for i, row in dynamic_risk.head(10).iterrows():
            folium.CircleMarker(
                location=[row["lat"], row["lng"]],
                radius=8,
                color="crimson",
                fill=True,
                fill_color="crimson",
                fill_opacity=0.8,
                popup=folium.Popup(
                    f"<b>Rank #{i+1}</b><br>"
                    f"Risk: {row['dynamic_risk']:.4f}<br>"
                    f"Predicted today: {row['predicted_daily']:.0f} violations<br>"
                    f"Congestion: {row.get('congestion_pct', 0):.1f}%<br>"
                    f"Watch for: {row.get('top_violation','N/A')}<br>"
                    f"Vehicle: {row.get('top_vehicle','N/A')}",
                    max_width=250,
                ),
                tooltip=f"#{i+1} Risk: {row['dynamic_risk']:.4f}",
            ).add_to(m)

    st_folium(m, width=None, height=520)

    # Risk table
    st.subheader("Top 20 High-Risk Zones Right Now")
    display_cols = ["h3_index", "predicted_daily", "predicted_hourly",
                    "congestion_pct", "traffic_amplifier", "dynamic_risk",
                    "top_violation", "top_vehicle"]
    display_cols = [c for c in display_cols if c in dynamic_risk.columns]
    st.dataframe(
        dynamic_risk[display_cols].head(20).rename(columns={
            "h3_index":         "Zone ID",
            "predicted_daily":  "Predicted Today",
            "predicted_hourly": "Predicted This Hour",
            "congestion_pct":   "Congestion %",
            "traffic_amplifier":"Traffic Multiplier",
            "dynamic_risk":     "Dynamic Risk Score",
            "top_violation":    "Primary Violation",
            "top_vehicle":      "Primary Vehicle",
        }),
        use_container_width=True, hide_index=True,
    )

    st.divider()
    st.subheader("Traffic Impact by Zone — Economic Cost of Illegal Parking")
    st.markdown(
        "Each zone's disruption score is computed from violation records directly: "
        "`lane_blockage × severity × peak_multiplier × road_factor`. "
        "This isolates violation-attributable delay from general congestion."
    )

    # Validation result
    if validation_result["rho"] is not None:
        v = validation_result
        sig_icon = "✅" if v["p_value"] < 0.05 else "⚠️"
        st.info(
            f"{sig_icon} **Model Validation:** {v['interpretation']}",
            icon="📊"
        )

    # Impact scatter: disruption score vs observed congestion
    impact_plot = impact_with_traffic.dropna(subset=["disruption_score", "congestion_pct"])
    if len(impact_plot) > 5:
        fig_impact = px.scatter(
            impact_plot.head(200),
            x="disruption_score", y="congestion_pct",
            size="violation_count", color="economic_cost_inr_day",
            color_continuous_scale="Reds",
            hover_data=["top_violation", "top_vehicle", "vehicle_delay_min_day"],
            labels={
                "disruption_score":      "Violation Disruption Score (model)",
                "congestion_pct":        "Observed Congestion % (TomTom)",
                "economic_cost_inr_day": "Economic Cost (₹/day)",
            },
            title="Disruption Score (from violations) vs Observed Congestion (TomTom) — Independent Validation",
        )
        fig_impact.update_layout(height=400)
        st.plotly_chart(fig_impact, use_container_width=True)

    # Impact table
    impact_show = impact_df[["h3_index", "disruption_score", "disruption_per_day",
                              "vehicles_affected_per_day", "vehicle_delay_min_day",
                              "economic_cost_inr_day", "peak_disruption_share",
                              "top_violation", "top_vehicle"]].head(20).copy()
    impact_show["economic_cost_inr_day"] = impact_show["economic_cost_inr_day"].apply(lambda x: f"₹{x:,.0f}")
    impact_show["vehicle_delay_min_day"] = impact_show["vehicle_delay_min_day"].apply(lambda x: f"{x:,.0f} min")
    impact_show["peak_disruption_share"] = impact_show["peak_disruption_share"].apply(lambda x: f"{x*100:.0f}%")
    st.dataframe(
        impact_show.rename(columns={
            "h3_index":                "Zone",
            "disruption_score":        "Disruption Score",
            "disruption_per_day":      "Daily Disruption",
            "vehicles_affected_per_day": "Vehicles Affected/Day",
            "vehicle_delay_min_day":   "Delay Caused/Day",
            "economic_cost_inr_day":   "Economic Cost/Day",
            "peak_disruption_share":   "Peak Hour Share",
            "top_violation":           "Primary Violation",
            "top_vehicle":             "Primary Vehicle",
        }),
        use_container_width=True, hide_index=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Patrol Planner
# ════════════════════════════════════════════════════════════════════════════

with tab2:
    st.header("Smart Patrol Route Planner")
    st.markdown(
        "AI-optimised patrol routes based on predicted violation risk and live traffic. "
        "Uses a nearest-neighbour TSP algorithm with TomTom Matrix Routing "
        "for traffic-aware travel time estimates."
    )

    pcol1, pcol2, pcol3 = st.columns(3)
    with pcol1:
        n_officers   = st.slider("Number of Officers", 1, 3, 1)
    with pcol2:
        start_time   = st.time_input("Patrol Start Time", value=now.replace(minute=0, second=0, microsecond=0))
    with pcol3:
        st.markdown("&nbsp;")
        run_patrol   = st.button("Generate Patrol Plan", type="primary", use_container_width=True)

    if run_patrol or "patrol_plans" not in st.session_state:
        with st.spinner("Optimising patrol routes..."):
            start_str = start_time.strftime("%H:%M")
            plans = plan_patrol(dynamic_risk, n_officers=n_officers, start_time_str=start_str)
            st.session_state["patrol_plans"] = plans

    plans = st.session_state.get("patrol_plans", [])

    if plans:
        # Map with routes
        pm = folium.Map(location=BENGALURU_CENTER, zoom_start=12, tiles="CartoDB positron")
        colors = ["crimson", "dodgerblue", "forestgreen"]

        for plan in plans:
            color = colors[(plan["officer"] - 1) % len(colors)]
            route = plan["route"]
            coords = [[r.lat, r.lng] for r in route.itertuples()]

            if len(coords) > 1:
                folium.PolyLine(coords, color=color, weight=3, opacity=0.8,
                                tooltip=f"Officer {plan['officer']}").add_to(pm)

            for _, row in route.iterrows():
                folium.Marker(
                    location=[row["lat"], row["lng"]],
                    icon=folium.DivIcon(
                        html=f'<div style="background:{color};color:white;border-radius:50%;'
                             f'width:24px;height:24px;display:flex;align-items:center;'
                             f'justify-content:center;font-weight:bold;font-size:12px;">'
                             f'{row["stop"]}</div>',
                        icon_size=(24, 24), icon_anchor=(12, 12),
                    ),
                    popup=folium.Popup(
                        f"<b>Stop {row['stop']} — {row['arrival_time']}</b><br>"
                        f"Violation: {row.get('top_violation','N/A')}<br>"
                        f"Vehicle: {row.get('top_vehicle','N/A')}<br>"
                        f"Congestion: {row.get('congestion_pct',0):.0f}%<br>"
                        f"Travel: {row.get('travel_min',0):.0f} min from prev stop",
                        max_width=220,
                    ),
                ).add_to(pm)

        st_folium(pm, width=None, height=480)

        # Route tables
        for plan in plans:
            st.subheader(f"Officer {plan['officer']} — {plan['n_zones']} stops, ~{plan['total_travel_min']:.0f} min total")
            st.dataframe(
                plan["route"].rename(columns={
                    "stop":           "Stop",
                    "arrival_time":   "Arrival",
                    "travel_min":     "Travel (min)",
                    "top_violation":  "Watch For",
                    "top_vehicle":    "Vehicle Type",
                    "congestion_pct": "Congestion %",
                    "dynamic_risk":   "Risk Score",
                }),
                use_container_width=True, hide_index=True,
            )
    else:
        st.info("Click 'Generate Patrol Plan' to compute optimised routes.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Anomaly Alerts
# ════════════════════════════════════════════════════════════════════════════

with tab3:
    st.header(f"Anomaly Alerts — {now.strftime('%A %H:%M')}")
    st.markdown(
        "Zones that historically spike **significantly above normal** at this "
        "exact time of day and day of week. These warrant immediate attention."
    )

    if anomalies.empty:
        st.success("No anomalies detected for the current time slot. Patterns are within normal range.")
    else:
        for _, row in anomalies.iterrows():
            with st.container(border=True):
                acol1, acol2 = st.columns([3, 1])
                with acol1:
                    st.markdown(f"**Zone:** `{row['h3_index'][:12]}...`")
                    st.markdown(f"⚠️ {row['alert_reason']}")
                    st.markdown(f"Primary violation: **{row.get('top_violation','N/A')}** | Vehicle: **{row.get('top_vehicle','N/A')}**")
                with acol2:
                    st.metric("Spike Ratio", f"{row['spike_ratio']:.1f}×")

    st.divider()

    # Volatility chart
    st.subheader("Zone Volatility — Most Erratic Hotspots")
    st.markdown("Zones with the highest coefficient of variation in daily violations — unpredictable patterns that need flexible enforcement.")

    vol_top = volatility.head(15)
    if not vol_top.empty:
        fig = px.bar(
            vol_top, x="h3_index", y="cv",
            color="cv",
            color_continuous_scale="Reds",
            labels={"h3_index": "Zone", "cv": "Coefficient of Variation"},
            title="Top 15 Most Volatile Zones",
        )
        fig.update_xaxes(tickangle=45, tickfont_size=9)
        fig.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Temporal heatmap — when do violations peak?
    st.subheader("Violation Intensity by Hour & Day")
    hourly = (
        df.groupby(["day_of_week", "hour"])
        .size()
        .reset_index(name="count")
    )
    pivot = hourly.pivot(index="day_of_week", columns="hour", values="count").fillna(0)
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    pivot.index = [day_labels[i] for i in pivot.index if i < 7]

    fig2 = px.imshow(
        pivot,
        labels={"x": "Hour of Day", "y": "Day of Week", "color": "Violations"},
        color_continuous_scale="YlOrRd",
        title="Historical Violation Heatmap (Nov 2023–May 2024)",
        aspect="auto",
    )
    fig2.update_layout(height=320)
    st.plotly_chart(fig2, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — Officer Briefing
# ════════════════════════════════════════════════════════════════════════════

with tab4:
    st.header("AI Officer Briefing")

    llm_status = "🟢 Claude AI" if ANTHROPIC_API_KEY else "🟡 Template (add ANTHROPIC_API_KEY for AI-generated briefings)"
    st.markdown(f"Briefing engine: **{llm_status}**")

    plans_for_briefing = st.session_state.get("patrol_plans", [])

    if st.button("Generate Briefing", type="primary"):
        with st.spinner("Generating briefing..."):
            briefing_text = generate_briefing(
                risk_df        = dynamic_risk,
                anomalies      = anomalies,
                plans          = plans_for_briefing,
                traffic_source = traffic_source,
            )
            st.session_state["briefing"] = briefing_text

    briefing = st.session_state.get("briefing", "")
    if briefing:
        st.markdown("---")
        st.code(briefing, language=None)
        st.download_button(
            label    = "Download Briefing",
            data     = briefing,
            file_name= f"patrol_briefing_{now.strftime('%Y%m%d_%H%M')}.txt",
            mime     = "text/plain",
        )
    else:
        st.info("Click 'Generate Briefing' to create an AI-powered patrol briefing for the current shift.")

        # Show patrol plan summary regardless
        if plans_for_briefing:
            st.markdown("**Current Patrol Plan:**")
            for plan in plans_for_briefing:
                st.markdown(f"- Officer {plan['officer']}: {plan['n_zones']} stops, ~{plan['total_travel_min']:.0f} min")
        else:
            st.markdown("*(Generate a patrol plan in the Patrol Planner tab first for best results)*")


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — Model Validation
# ════════════════════════════════════════════════════════════════════════════

with tab5:
    st.header("Model Validation — May 2024 Holdout")
    st.markdown(
        "Prophet models were trained on **Nov 2023 – Apr 2024** and validated "
        "on **May 2024** (held out, never seen during training)."
    )

    if val_result.empty:
        st.warning("Validation results not available — insufficient data in May 2024.")
    else:
        # Summary metrics
        median_mape = val_result["mape"].median()
        mean_mae    = val_result["mae"].mean()
        n_zones_val = len(val_result)
        beat_naive  = (val_result["mape"] < 50).sum()   # naive baseline MAPE ≈ 50%

        vcol1, vcol2, vcol3, vcol4 = st.columns(4)
        vcol1.metric("Median MAPE", f"{median_mape:.1f}%", help="Lower is better. Naive baseline ≈ 50%")
        vcol2.metric("Mean MAE",    f"{mean_mae:.1f} violations/day")
        vcol3.metric("Zones Validated", n_zones_val)
        vcol4.metric("Beat Naive Baseline", f"{beat_naive}/{n_zones_val} zones")

        st.markdown(
            f"Prophet achieves **{median_mape:.1f}% median MAPE** on the May 2024 holdout. "
            f"A naive model (predict the mean) scores ~50% MAPE. "
            f"**{beat_naive} of {n_zones_val} zones** ({beat_naive/n_zones_val*100:.0f}%) "
            f"beat the naive baseline."
        )

        # MAPE distribution
        fig_mape = px.histogram(
            val_result, x="mape", nbins=20,
            title="MAPE Distribution Across Zones (May 2024)",
            labels={"mape": "MAPE (%)", "count": "Zones"},
            color_discrete_sequence=["steelblue"],
        )
        fig_mape.add_vline(x=50, line_dash="dash", line_color="red",
                           annotation_text="Naive baseline (50%)")
        fig_mape.update_layout(height=340)
        st.plotly_chart(fig_mape, use_container_width=True)

        # Predicted vs Actual for best zone
        best_zone = val_result.sort_values("mape").iloc[0]["zone_id"]
        chart_data = get_validation_chart_data(models, daily_ser, best_zone)

        if not chart_data.empty:
            st.subheader(f"Predicted vs Actual — Best Zone (MAPE: {val_result.sort_values('mape').iloc[0]['mape']:.1f}%)")
            fig_chart = go.Figure()
            fig_chart.add_trace(go.Scatter(
                x=chart_data["ds"], y=chart_data["y"],
                mode="lines+markers", name="Actual",
                line=dict(color="crimson", width=2),
            ))
            fig_chart.add_trace(go.Scatter(
                x=chart_data["ds"], y=chart_data["yhat"],
                mode="lines", name="Predicted",
                line=dict(color="steelblue", width=2, dash="dash"),
            ))
            if "yhat_lower" in chart_data.columns:
                fig_chart.add_trace(go.Scatter(
                    x=pd.concat([chart_data["ds"], chart_data["ds"][::-1]]),
                    y=pd.concat([chart_data["yhat_upper"], chart_data["yhat_lower"][::-1]]),
                    fill="toself", fillcolor="rgba(70,130,180,0.15)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="80% Confidence",
                ))
            fig_chart.update_layout(
                xaxis_title="Date (May 2024)", yaxis_title="Daily Violations",
                height=380, legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig_chart, use_container_width=True)

        # Full validation table
        st.subheader("Per-Zone Validation Results")
        st.dataframe(
            val_result.rename(columns={
                "zone_id":        "Zone",
                "mape":           "MAPE (%)",
                "mae":            "MAE",
                "n_days":         "Days",
                "mean_actual":    "Avg Actual",
                "mean_predicted": "Avg Predicted",
            }),
            use_container_width=True, hide_index=True,
        )
