"""
4_dashboard.py
Streamlit UI for Vidarbha Flood & Drought Localized Early Warning System.
Features:
- 100% Marker-Free Dynamic GIS Visualization (No Pin Markers)
- Continuous Gradient Color Schemas for Land Surface Temperature & Flood Risk
- Highlighted Selected Region & Taluka Boundaries with Illuminated Glowing Borders
- Hyper-Local Micro-Area Grid Resolution (Down to few hundred square meters / hectares)
- Multi-Modal Time-Series Telemetry & Localized Vulnerability Analytics
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Import regional registry and spatial polygon helpers
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from geo_registry import (
    VIDARBHA_GEO_DATA, 
    generate_taluka_boundary_polygon, 
    generate_hyper_local_parcels
)

# Page Configuration
st.set_page_config(
    page_title="Vidarbha Hydro-Climate AI — Localized Early Warning",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    /* Metric Card Enhancements */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1f77b4;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0f2027;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 1.2rem;
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 8px;
    }
    .badge-blue { background-color: #e1f5fe; color: #0288d1; border: 1px solid #b3e5fc; }
    .badge-orange { background-color: #fff3e0; color: #f57c00; border: 1px solid #ffe0b2; }
    .badge-red { background-color: #ffebee; color: #d32f2f; border: 1px solid #ffcdd2; }
    .badge-green { background-color: #e8f5e9; color: #388e3c; border: 1px solid #c8e6c9; }
</style>
""", unsafe_allow_html=True)

API_URL = "http://127.0.0.1:8000"

st.markdown('<div class="main-title">Vidarbha Hydro-Climate AI 🛰️🌊🔥</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Dynamic Thermal & Localized Flood/Drought Boundary Analytics — <b>Version 1.0 (Marker-Free GIS)</b></div>', unsafe_allow_html=True)

# Session state initialization
if "batch_results" not in st.session_state:
    st.session_state.batch_results = None
if "selected_district_state" not in st.session_state:
    st.session_state.selected_district_state = "Nagpur"

# --- Gradient Color Mapping Functions ---
def get_lst_gradient_color(lst_c):
    """Returns a continuous hex color for Land Surface Temperature (°C)."""
    if lst_c < 32.0:
        return "#2b83ba"  # Deep Cool Blue (<32°C)
    elif lst_c < 35.0:
        return "#64abb0"  # Teal (32-35°C)
    elif lst_c < 37.5:
        return "#abdda4"  # Sage Green (35-37.5°C)
    elif lst_c < 40.0:
        return "#ffffbf"  # Solar Yellow (37.5-40°C)
    elif lst_c < 42.5:
        return "#fdae61"  # Warm Amber (40-42.5°C)
    elif lst_c < 44.5:
        return "#f46d43"  # Fiery Coral (42.5-44.5°C)
    else:
        return "#d7191c"  # Extreme Heat Red (>44.5°C)

def get_flood_gradient_color(flood_prob):
    """Returns a continuous hex color for Flood Probability (0 to 1)."""
    if flood_prob < 0.20:
        return "#e0f3f8"  # Very Dry/Safe
    elif flood_prob < 0.45:
        return "#abd9e9"  # Low Saturation
    elif flood_prob < 0.70:
        return "#74add1"  # Moderate Moisture
    elif flood_prob < 0.85:
        return "#4575b4"  # High Water Index
    else:
        return "#d73027"  # Critical Inundation Alert

def get_compound_gradient_color(flood_prob, drought_sev, lst_c):
    """Multi-hazard gradient blending flood, drought severity, and LST."""
    if flood_prob > 0.85:
        return "#d73027"  # Inundation Red
    elif drought_sev >= 2:
        return "#e6550d"  # Drought Warning Orange
    elif flood_prob > 0.60:
        return "#3182bd"  # High Moisture Blue
    elif drought_sev == 1:
        return "#fdae6b"  # Mild Dryness Amber
    else:
        return get_lst_gradient_color(lst_c)

# --- Sidebar Controls ---
st.sidebar.header("🗺️ Geographic Navigation")
district_list = list(VIDARBHA_GEO_DATA.keys())
selected_district = st.sidebar.selectbox(
    "Select District", 
    district_list, 
    index=district_list.index(st.session_state.selected_district_state) if st.session_state.selected_district_state in district_list else 0
)
st.session_state.selected_district_state = selected_district

talukas_in_district = ["All Sub-Districts (Overview)"] + list(VIDARBHA_GEO_DATA[selected_district]["talukas"].keys())
selected_taluka = st.sidebar.selectbox("Target Sub-District / Taluka", talukas_in_district)

st.sidebar.markdown("---")
st.sidebar.header("🎨 GIS Layer Controls")
gradient_theme = st.sidebar.radio(
    "Gradient Color Schema",
    [
        "Land Surface Temperature (LST °C)", 
        "Flood Inundation Probability (%)", 
        "Compound Multi-Hazard (Flood & Drought)"
    ]
)

view_resolution = st.sidebar.radio(
    "Spatial Resolution",
    [
        "Sub-District Boundary Polygons", 
        "Hyper-Local Micro-Grid (~400m / 16ha Parcels)"
    ]
)

basemap_style = st.sidebar.selectbox(
    "Basemap Style",
    ["OpenStreetMap", "Esri Satellite Imagery", "CartoDB Light"]
)

simulation_mode = st.sidebar.radio(
    "Telemetry Scenario", 
    ["Random Scenario", "Simulate Flood Risk", "Simulate Drought Risk", "Simulate Normal Condition"]
)

run_btn = st.sidebar.button("🔄 Execute Localized Analysis", type="primary")

# SMS Alert Testing Panel
st.sidebar.markdown("---")
st.sidebar.subheader("📲 Emergency Alert Dispatch")
alert_phone = st.sidebar.text_input("Emergency Contact Number", value="+919876543210")

# Fetch batch predictions for district
def fetch_district_analysis(district_name, scenario_name):
    try:
        resp = requests.post(
            f"{API_URL}/predict_district_zones",
            json={"district": district_name, "scenario": scenario_name},
            timeout=10
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.sidebar.warning(f"Connecting to AI backend: {e}")
        # Local fallback simulation if backend is launching
        return None

if run_btn or st.session_state.batch_results is None or st.session_state.batch_results.get("district") != selected_district:
    with st.spinner(f"Computing localized satellite telemetry and hydro-thermal models for {selected_district}..."):
        data = fetch_district_analysis(selected_district, simulation_mode)
        if data:
            st.session_state.batch_results = data

# If data is available, render dashboard
if st.session_state.batch_results is not None:
    data = st.session_state.batch_results
    zones = data["zones"]
    
    total_talukas = len(zones)
    flood_count = sum(1 for z in zones if z["flood_probability"] > 0.85)
    drought_count = sum(1 for z in zones if z["drought_severity"] >= 2)
    avg_lst = float(np.mean([z["mean_lst_c"] for z in zones]))
    max_lst = float(np.max([z["mean_lst_c"] for z in zones]))
    min_lst = float(np.min([z["mean_lst_c"] for z in zones]))
    
    # KPI Row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Selected District", selected_district, f"{total_talukas} Sub-Regions")
    k2.metric("Mean Surface Temp", f"{avg_lst:.1f} °C", f"Range: {min_lst:.1f} – {max_lst:.1f} °C", delta_color="inverse")
    k3.metric("Flood Inundation Alert", f"{flood_count} Sub-Districts", "CRITICAL" if flood_count > 0 else "STABLE", delta_color="inverse")
    k4.metric("Drought Stress Alert", f"{drought_count} Sub-Districts", "CRITICAL" if drought_count > 0 else "NORMAL", delta_color="inverse")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Navigation Tabs
    tab_map, tab_matrix, tab_series, tab_alert = st.tabs([
        "🗺️ Dynamic GIS Map & Boundaries", 
        "📊 Localized Risk & Topography Matrix", 
        "📈 14-Day Multi-Modal Sensor Feeds",
        "📲 Alert Dispatch Simulator"
    ])
    
    with tab_map:
        # Determine center coordinates and zoom
        if selected_taluka != "All Sub-Districts (Overview)" and selected_taluka in VIDARBHA_GEO_DATA[selected_district]["talukas"]:
            center_coords = VIDARBHA_GEO_DATA[selected_district]["talukas"][selected_taluka]["coords"]
            zoom_lvl = 11 if view_resolution.startswith("Hyper-Local") else 10
        else:
            center_coords = data["center"]
            zoom_lvl = 9
            
        # Select folium tiles
        if basemap_style == "Esri Satellite Imagery":
            tiles_url = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
            attr = "Esri World Imagery"
        elif basemap_style == "CartoDB Light":
            tiles_url = "CartoDB positron"
            attr = None
        else:
            tiles_url = "OpenStreetMap"
            attr = None
            
        # Initialize Marker-Free Base Map
        f_map = folium.Map(
            location=center_coords, 
            zoom_start=zoom_lvl, 
            tiles=tiles_url,
            attr=attr,
            control_scale=True
        )
        
        # Draw Sub-District Polygons and Micro-Grid Parcels
        for z in zones:
            t_name = z["taluka"]
            lat, lon = z["coords"]
            f_prob = z["flood_probability"]
            d_sev = z["drought_severity"]
            lst_val = z["mean_lst_c"]
            elev = z["elevation_m"]
            river = z["river_proximity"]
            
            is_selected = (selected_taluka == t_name)
            
            # Gradient Color Selection
            if gradient_theme.startswith("Land Surface"):
                fill_color = get_lst_gradient_color(lst_val)
            elif gradient_theme.startswith("Flood"):
                fill_color = get_flood_gradient_color(f_prob)
            else:
                fill_color = get_compound_gradient_color(f_prob, d_sev, lst_val)
                
            # If in Hyper-Local Micro-Grid view and this taluka is active/selected
            if view_resolution.startswith("Hyper-Local") and (is_selected or selected_taluka == "All Sub-Districts (Overview)"):
                micro_cells = generate_hyper_local_parcels(
                    lat, lon, base_lst=lst_val, flood_prob=f_prob, drought_sev=d_sev, grid_dim=6, cell_size_m=400
                )
                for cell in micro_cells:
                    c_lst = cell["lst_c"]
                    c_flood = cell["flood_prob"]
                    c_drought = cell["drought_sev"]
                    c_area = cell["area_m2"]
                    c_ha = cell["area_ha"]
                    
                    if gradient_theme.startswith("Land Surface"):
                        c_fill = get_lst_gradient_color(c_lst)
                    elif gradient_theme.startswith("Flood"):
                        c_fill = get_flood_gradient_color(c_flood)
                    else:
                        c_fill = get_compound_gradient_color(c_flood, c_drought, c_lst)
                        
                    cell_popup_html = f"""
                    <div style='font-family: Arial, sans-serif; font-size: 12px; width: 195px;'>
                        <div style='background-color: #1e3c72; color: white; padding: 4px 6px; font-weight: bold; border-radius: 3px;'>
                            🔬 Micro-Area Parcel
                        </div>
                        <div style='padding-top: 5px;'>
                            <b>Taluka:</b> {t_name}<br>
                            <b>Parcel Area:</b> {c_area:,} m² ({c_ha} ha)<br>
                            <b>Surface Temp:</b> <b style='color: #d9534f;'>{c_lst}°C</b><br>
                            <b>Flood Risk:</b> <b>{c_flood:.1%}</b><br>
                            <b>Drought Level:</b> Level {c_drought}<br>
                            <b>Elevation:</b> {elev} m
                        </div>
                    </div>
                    """
                    folium.Rectangle(
                        bounds=cell["bounds"],
                        color="#ffffff",
                        weight=0.9,
                        fill=True,
                        fill_color=c_fill,
                        fill_opacity=0.75,
                        popup=folium.Popup(cell_popup_html, max_width=220),
                        tooltip=f"Parcel {t_name} | LST: {c_lst}°C | Flood: {c_flood:.1%}"
                    ).add_to(f_map)
            
            # Boundary Polygon
            taluka_poly = generate_taluka_boundary_polygon(lat, lon, radius_km=10.5)
            
            # Selected region has glowing cyan/neon border
            if is_selected:
                border_color = "#00FFFF"
                border_weight = 4.0
                border_dash = "5, 5"
                fill_op = 0.20 if view_resolution.startswith("Hyper-Local") else 0.55
            else:
                border_color = "#d73027" if f_prob > 0.85 else ("#e6550d" if d_sev >= 2 else "#2c3e50")
                border_weight = 2.0
                border_dash = None
                fill_op = 0.15 if view_resolution.startswith("Hyper-Local") else 0.50
                
            popup_html = f"""
            <div style='font-family: Arial, sans-serif; font-size: 12px; width: 220px;'>
                <div style='background: {fill_color}; color: black; padding: 5px 8px; font-weight: bold; border-radius: 4px 4px 0 0;'>
                    🏛️ {t_name} {'[SELECTED]' if is_selected else ''}
                </div>
                <div style='padding: 6px; border: 1px solid #ccc; border-top: none; background: white;'>
                    <b>District:</b> {selected_district}<br>
                    <b>Mean LST:</b> <span style='color: #d9534f; font-weight: bold;'>{lst_val}°C</span><br>
                    <b>Flood Probability:</b> <b>{f_prob:.1%}</b><br>
                    <b>Drought Severity:</b> Level {d_sev}<br>
                    <b>River Proximity:</b> {river}<br>
                    <b>Elevation:</b> {elev} m ASL
                </div>
            </div>
            """
            
            folium.Polygon(
                locations=taluka_poly,
                color=border_color,
                weight=border_weight,
                dash_array=border_dash,
                fill=True,
                fill_color=fill_color,
                fill_opacity=fill_op,
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"{t_name} — LST: {lst_val}°C | Flood: {f_prob:.1%} | Drought: L{d_sev}"
            ).add_to(f_map)
            
        # Render Map with Clean Frame
        st_folium(f_map, width=1050, height=520, returned_objects=[])
        
        # Color Legend
        st.markdown("---")
        if gradient_theme.startswith("Land Surface"):
            st.markdown(
                """
                **🌡️ Land Surface Temperature (LST °C) Thermal Scale:**
                <div style="background: linear-gradient(to right, #2b83ba, #64abb0, #abdda4, #ffffbf, #fdae61, #f46d43, #d7191c); height: 16px; border-radius: 8px; margin: 6px 0;"></div>
                <div style="display: flex; justify-content: space-between; font-size: 12px; color: #555;">
                    <span><b>&lt; 32°C</b> (Cool Riparian/Forest)</span>
                    <span><b>35°C - 37.5°C</b> (Moderate Vegetative)</span>
                    <span><b>38°C - 42°C</b> (Semi-Arid/Urban)</span>
                    <span><b>&gt; 44.5°C</b> (Thermal Stress Hotspot)</span>
                </div>
                """, unsafe_allow_html=True
            )
        elif gradient_theme.startswith("Flood"):
            st.markdown(
                """
                **🌊 Flood Inundation Probability Scale:**
                <div style="background: linear-gradient(to right, #e0f3f8, #abd9e9, #74add1, #4575b4, #d73027); height: 16px; border-radius: 8px; margin: 6px 0;"></div>
                <div style="display: flex; justify-content: space-between; font-size: 12px; color: #555;">
                    <span><b>0% - 20%</b> (Dry / Low Risk)</span>
                    <span><b>20% - 50%</b> (Moderate Moisture)</span>
                    <span><b>50% - 85%</b> (Elevated Runoff)</span>
                    <span><b>&gt; 85%</b> (Critical Inundation Hazard)</span>
                </div>
                """, unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                **⚡ Multi-Hazard Risk Color Legend:**
                - 🟥 **Deep Crimson**: Severe Flood Hazard (>85% saturation / river confluence).
                - 🟧 **Vivid Orange**: Critical Drought Stress (Level 2–3 Severity, LST > 41°C).
                - 🟨 **Solar Yellow**: Moderate Heat / Moisture Watch.
                - 🟦 **Cool Blue / Green**: Hydro-Climatic Stability.
                - 🔲 **Illuminated Cyan Boundary (`#00FFFF`)**: Active Selected Region.
                """
            )
            
    with tab_matrix:
        st.subheader(f"Localized Sub-District & Micro-Area Vulnerability Matrix — {selected_district}")
        
        matrix_rows = []
        for z in zones:
            matrix_rows.append({
                "Sub-District (Taluka)": z["taluka"],
                "Hazard Status": z["status"],
                "Land Surface Temp (°C)": z["mean_lst_c"],
                "Flood Inundation Risk": f"{z['flood_probability']:.1%}",
                "Drought Severity": f"Level {z['drought_severity']}",
                "River Basin": z["river_proximity"],
                "Elevation (m)": f"{z['elevation_m']} m"
            })
            
        df_matrix = pd.DataFrame(matrix_rows)
        st.dataframe(df_matrix, hide_index=True)
        
        c_chart1, c_chart2 = st.columns(2)
        with c_chart1:
            st.markdown("#### Land Surface Temp by Sub-District (°C)")
            df_chart_lst = pd.DataFrame({
                "Taluka": [z["taluka"] for z in zones],
                "LST (°C)": [z["mean_lst_c"] for z in zones]
            }).set_index("Taluka").sort_values(by="LST (°C)", ascending=False)
            st.bar_chart(df_chart_lst)
            
        with c_chart2:
            st.markdown("#### Flood Inundation Probability by Sub-District (%)")
            df_chart_flood = pd.DataFrame({
                "Taluka": [z["taluka"] for z in zones],
                "Flood Risk (%)": [round(z["flood_probability"] * 100, 1) for z in zones]
            }).set_index("Taluka").sort_values(by="Flood Risk (%)", ascending=False)
            st.bar_chart(df_chart_flood)
            
    with tab_series:
        st.subheader("14-Day Multi-Modal Remote Sensing Sequences")
        active_t_name = selected_taluka if selected_taluka != "All Sub-Districts (Overview)" else zones[0]["taluka"]
        target_z = next((z for z in zones if z["taluka"] == active_t_name), zones[0])
        
        st.markdown(f"Displaying 14-day temporal multi-sensor telemetry for **{active_t_name}** ({selected_district}):")
        
        dates = [datetime.now().date() - timedelta(days=14 - i) for i in range(14)]
        seq_array = np.array(target_z["sequence"])
        
        df_temporal = pd.DataFrame({
            "Date": dates,
            "Precipitation (mm)": seq_array[:, 0],
            "NDVI (Vegetation Index)": seq_array[:, 1],
            "NDWI (Water Index)": seq_array[:, 2],
            "LST (°C)": seq_array[:, 3]
        }).set_index("Date")
        
        st1, st2 = st.columns(2)
        with st1:
            st.markdown("**1. IMD Precipitation (mm) & MODIS Land Surface Temp (°C)**")
            st.line_chart(df_temporal[["Precipitation (mm)", "LST (°C)"]])
        with st2:
            st.markdown("**2. Sentinel-2 Multi-Spectral Indices (NDVI & NDWI)**")
            st.line_chart(df_temporal[["NDVI (Vegetation Index)", "NDWI (Water Index)"]])
            
        st.info(f"📍 **Summary for {active_t_name}:** Flood Probability: `{target_z['flood_probability']:.1%}` | Drought Severity: `Level {target_z['drought_severity']}` | Mean LST: `{target_z['mean_lst_c']}°C` | River: `{target_z['river_proximity']}`.")

    with tab_alert:
        st.subheader("📲 Emergency Alert & SMS Broadcast Verification")
        st.markdown("Test early warning notification broadcast to localized disaster management officials and farmers:")
        
        active_zone = next((z for z in zones if z["taluka"] == (selected_taluka if selected_taluka != "All Sub-Districts (Overview)" else zones[0]["taluka"])), zones[0])
        
        st.write(f"**Target Zone:** {active_zone['taluka']}, {selected_district}")
        st.write(f"**Flood Risk:** {active_zone['flood_probability']:.1%} | **Drought Level:** Level {active_zone['drought_severity']} | **Surface Temp:** {active_zone['mean_lst_c']}°C")
        
        if st.button("🚨 Trigger Test Emergency Dispatch"):
            try:
                resp = requests.post(
                    f"{API_URL}/predict",
                    json={
                        "district": selected_district,
                        "taluka": active_zone["taluka"],
                        "phone": alert_phone,
                        "sequence": active_zone["sequence"]
                    },
                    timeout=10
                )
                if resp.status_code == 200:
                    res_json = resp.json()
                    st.success(f"**Alert Engine Response:** {res_json['message']}")
                    st.json(res_json)
                else:
                    st.error(f"API Error ({resp.status_code}): {resp.text}")
            except Exception as e:
                st.error(f"Failed to communicate with API backend: {e}")

