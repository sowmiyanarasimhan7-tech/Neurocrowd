"""
NeuroCrowd: AI-Powered Crowd Intelligence & Early Risk Prediction System
Streamlit Main Dashboard Application (Supports Multi-Camera & Multiple Video Switching)
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time
import tempfile
import os

from config import DEFAULT_VENUE_SETTINGS, RISK_COLORS, AUTHORITY_ADVISORIES
from spatial_calibration import (
    calculate_net_usable_area, calculate_real_density, 
    get_fruin_risk_level, generate_zone_grid, map_detections_to_zones,
    estimate_area_from_video_perspective
)
from cv_engine import CrowdCVEngine
from predictive_model import CrowdRiskPredictor

st.set_page_config(
    page_title="NeuroCrowd - AI Crowd Risk Safety System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0B0E14; color: #E2E8F0; }
    .main-header {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155; border-radius: 10px; padding: 12px 20px; margin-bottom: 15px;
    }
    .header-title {
        font-size: 22px; font-weight: 800;
        background: linear-gradient(90deg, #38BDF8, #818CF8);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .kpi-container { display: flex; gap: 12px; margin-bottom: 15px; width: 100%; }
    .kpi-card {
        flex: 1; background: rgba(30, 41, 59, 0.8); border: 1px solid #334155;
        border-radius: 8px; padding: 10px 14px; text-align: center;
    }
    .kpi-label { font-size: 11px; color: #94A3B8; text-transform: uppercase; }
    .kpi-value { font-size: 20px; font-weight: 700; color: #F8FAFC; margin-top: 4px; }
    .kpi-unit { font-size: 12px; color: #94A3B8; font-weight: 400; }
    .badge-safe { background: rgba(0, 230, 118, 0.2); color: #00E676; padding: 3px 10px; border-radius: 12px; font-weight: 700; }
    .badge-moderate { background: rgba(255, 214, 0, 0.2); color: #FFD600; padding: 3px 10px; border-radius: 12px; font-weight: 700; }
    .badge-highrisk { background: rgba(255, 145, 0, 0.2); color: #FF9100; padding: 3px 10px; border-radius: 12px; font-weight: 700; }
    .badge-critical { background: rgba(255, 23, 68, 0.2); color: #FF1744; padding: 3px 10px; border-radius: 12px; font-weight: 700; }
    .advisory-box { background: rgba(15, 23, 42, 0.9); border: 1px solid #334155; border-left: 4px solid #38BDF8; border-radius: 8px; padding: 12px 16px; margin-top: 10px; }
</style>
""", unsafe_allow_html=True)

if "danger_history" not in st.session_state:
    st.session_state.danger_history = []
if "time_history" not in st.session_state:
    st.session_state.time_history = []
if "cv_engine" not in st.session_state:
    st.session_state.cv_engine = CrowdCVEngine()
if "ml_predictor" not in st.session_state:
    st.session_state.ml_predictor = CrowdRiskPredictor()

st.sidebar.markdown("### 🛡️ NeuroCrowd Control Panel")

with st.sidebar.expander("🎯 Detection Accuracy & Head Mode", expanded=True):
    conf_thresh = st.slider("Detection Sensitivity (Conf)", min_value=0.05, max_value=0.60, value=0.15, step=0.05)
    dense_head_mode = st.checkbox("Dense Crowd Head Mode", value=True)

with st.sidebar.expander("📐 Spatial & Area Calibration", expanded=True):
    venue_name = st.text_input("Venue Name", value=DEFAULT_VENUE_SETTINGS["venue_name"])
    auto_estimate_area = st.checkbox("⚡ Auto-Estimate Area from Camera Perspective", value=True)
    manual_total_area = st.number_input("Total Physical Area (m²)", min_value=50.0, max_value=10000.0, value=600.0, step=50.0, disabled=auto_estimate_area)
    obstacle_pct = st.slider("Static Obstacle Deduction %", min_value=0.0, max_value=50.0, value=20.0)

with st.sidebar.expander("🧩 Zone Layout Grid", expanded=False):
    grid_preset = st.selectbox("Grid Layout", ["2x3 (6 Zones)", "2x2 (4 Zones)", "3x3 (9 Zones)"], index=0)
    grid_rows, grid_cols = (2, 3) if "2x3" in grid_preset else ((2, 2) if "2x2" in grid_preset else (3, 3))

with st.sidebar.expander("👁️ Visual Overlays & Transparency", expanded=True):
    show_heatmap = st.checkbox("Show Density Heatmap", value=True)
    heatmap_opacity = st.slider("Heatmap Opacity / Transparency", min_value=0.05, max_value=1.0, value=0.25, step=0.05)
    show_zone_hud = st.checkbox("Show Dynamic Zone HUD Grid", value=True)
    show_boxes = st.checkbox("Show Bounding Boxes / Centroids", value=True)

# Multi-Video Input Selector
st.sidebar.markdown("---")
st.sidebar.markdown("📹 **Multi-Camera & Video Feed Manager**")
video_source_type = st.sidebar.radio(
    "Select Input Mode",
    ["Upload Multiple Video Files", "Videos Folder Presets", "Webcam / Live Stream", "Synthetic Demo Generator"]
)

cap = None
active_video_name = "Synthetic Video Feed"

videos_dir = "videos"
if not os.path.exists(videos_dir):
    os.makedirs(videos_dir, exist_ok=True)

if video_source_type == "Upload Multiple Video Files":
    uploaded_files = st.sidebar.file_uploader(
        "Upload Crowd Videos (.mp4, .avi, .mov)", 
        type=["mp4", "avi", "mov", "mkv"], 
        accept_multiple_files=True
    )
    if uploaded_files:
        file_map = {}
        for f in uploaded_files:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(f.read())
            file_map[f.name] = tfile.name
            
        selected_file_name = st.sidebar.selectbox("📷 Select Active Camera / Video Stream", list(file_map.keys()))
        if selected_file_name:
            active_video_name = selected_file_name
            cap = cv2.VideoCapture(file_map[selected_file_name])
    else:
        st.sidebar.info("💡 Upload one or multiple video files above to switch between camera angles!")

elif video_source_type == "Videos Folder Presets":
    local_files = [f for f in os.listdir(videos_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    if local_files:
        selected_preset = st.sidebar.selectbox("📷 Select Preset Camera Video", local_files)
        active_video_name = selected_preset
        video_path = os.path.join(videos_dir, selected_preset)
        cap = cv2.VideoCapture(video_path)
    else:
        st.sidebar.warning(f"⚠️ No video files found in `{videos_dir}/` folder. Place `.mp4` videos inside `{videos_dir}/` directory!")

elif video_source_type == "Webcam / Live Stream":
    rtsp_url = st.sidebar.text_input("RTSP / Stream URL (Blank for Webcam 0)", value="")
    active_video_name = "Live Security Camera / RTSP Stream"
    cap = cv2.VideoCapture(rtsp_url if rtsp_url.strip() != "" else 0)

elif video_source_type == "Synthetic Demo Generator":
    sim_density = st.sidebar.slider("Simulated Density (p/m²)", min_value=0.2, max_value=5.0, value=1.4, step=0.1)

header_placeholder = st.empty()
kpi_placeholder = st.empty()

col_l, col_r = st.columns([7, 5])
with col_l:
    video_placeholder = st.empty()
    motion_placeholder = st.empty()
with col_r:
    chart_placeholder = st.empty()
    advisory_placeholder = st.empty()
    zone_table_placeholder = st.empty()

def process_video_frame(frame):
    h, w = frame.shape[:2]
    boxes, centers = st.session_state.cv_engine.detect_people(frame, conf_thresh=conf_thresh, dense_head_mode=dense_head_mode)
    people_count = len(centers)

    if auto_estimate_area:
        total_area_sqm = estimate_area_from_video_perspective(w, h, person_boxes=boxes)
    else:
        total_area_sqm = manual_total_area

    net_usable_area = calculate_net_usable_area(total_area_sqm, obstacle_pct)
    real_density = calculate_real_density(people_count, net_usable_area)
    current_risk = get_fruin_risk_level(real_density)
    
    with header_placeholder.container():
        st.markdown(f'''
        <div class="main-header">
            <div class="header-title">NEUROCROWD AI | MULTI-CAMERA CROWD INTELLIGENCE</div>
            <div style="font-size: 12px; color: #94A3B8;">
                📷 ACTIVE CAMERA: <strong>{active_video_name}</strong> ({w}x{h}px) | USABLE AREA: <strong>{net_usable_area:.1f} m²</strong> (Auto-Calibrated)
            </div>
        </div>
        ''', unsafe_allow_html=True)

    motion_mag, turbulence = st.session_state.cv_engine.compute_optical_flow(frame)
    ml_pred = st.session_state.ml_predictor.predict(
        density=real_density, density_delta=0.08 if real_density > 2.0 else -0.02,
        occupancy_ratio=min(1.0, real_density / 4.0), motion_mag=motion_mag, turbulence=turbulence
    )

    zones = generate_zone_grid(w, h, grid_rows, grid_cols, total_area_sqm, obstacle_pct)
    zones = map_detections_to_zones(centers, zones, total_area_sqm, obstacle_pct)

    processed_frame = frame.copy()
    if show_heatmap and len(centers) > 0:
        processed_frame = st.session_state.cv_engine.draw_density_heatmap(
            processed_frame, centers, opacity=heatmap_opacity
        )
    if show_boxes:
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 230, 118), 1)
        for cx, cy in centers:
            cv2.circle(processed_frame, (cx, cy), 3, (255, 255, 255), -1)
    if show_zone_hud:
        processed_frame = st.session_state.cv_engine.draw_zone_grid_overlay(processed_frame, zones)

    st.session_state.danger_history.append(ml_pred["danger_index"])
    st.session_state.time_history.append(time.strftime("%H:%M:%S"))
    if len(st.session_state.danger_history) > 30:
        st.session_state.danger_history.pop(0)
        st.session_state.time_history.pop(0)

    with kpi_placeholder.container():
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-card"><div class="kpi-label">TOTAL PEOPLE</div><div class="kpi-value">{people_count}</div></div>
            <div class="kpi-card"><div class="kpi-label">USABLE AREA</div><div class="kpi-value">{net_usable_area:.0f} <span class="kpi-unit">m²</span></div></div>
            <div class="kpi-card"><div class="kpi-label">REAL DENSITY</div><div class="kpi-value" style="color:{RISK_COLORS[current_risk]};">{real_density:.2f} <span class="kpi-unit">p/m²</span></div></div>
            <div class="kpi-card"><div class="kpi-label">CURRENT RISK</div><div style="margin-top:4px;"><span class="badge-{current_risk.lower().replace("_","")}">{current_risk.replace('_',' ')}</span></div></div>
            <div class="kpi-card"><div class="kpi-label">5-MIN FORECAST</div><div style="margin-top:4px;"><span class="badge-{ml_pred['forecast_risk'].lower().replace("_","")}">{ml_pred['forecast_risk'].replace('_',' ')}</span></div></div>
        </div>
        """, unsafe_allow_html=True)

    video_placeholder.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

    with motion_placeholder.container():
        mc1, mc2 = st.columns(2)
        mc1.caption(f"⚡ Flow Velocity: `{motion_mag} m/s`")
        mc1.progress(min(1.0, motion_mag / 4.0))
        mc2.caption(f"🌪️ Panic Turbulence Index: `{turbulence}`")
        mc2.progress(min(1.0, turbulence / 4.0))

    with chart_placeholder.container():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=st.session_state.time_history, y=st.session_state.danger_history, mode="lines+markers", line=dict(color="#38BDF8", width=3)))
        fig.update_layout(title=dict(text="30-Second Danger Index Trend (0 - 100)", font=dict(size=12, color="#E2E8F0")), paper_bgcolor="rgba(15, 23, 42, 0.8)", plot_bgcolor="rgba(15, 23, 42, 0.8)", font=dict(color="#94A3B8", size=10), margin=dict(l=10, r=10, t=30, b=10), height=190, yaxis=dict(range=[0, 100]))
        st.plotly_chart(fig, use_container_width=True)

    with advisory_placeholder.container():
        st.markdown("#### 🚨 Tactical Advisories for Ground Commanders")
        advisories = AUTHORITY_ADVISORIES.get(current_risk, AUTHORITY_ADVISORIES["SAFE"])
        st.markdown(f'<div class="advisory-box"><ul>' + ''.join([f'<li>{a}</li>' for a in advisories]) + '</ul></div>', unsafe_allow_html=True)

    with zone_table_placeholder.container():
        st.markdown("#### 🧩 Zone Spatial Density Breakdown")
        zone_data = [{"Zone": z["name"], "Count": f"{z['people_count']} p", "Density": f"{z['density']:.2f} p/m²", "Risk": z["risk"]} for z in zones]
        st.dataframe(pd.DataFrame(zone_data), hide_index=True, use_container_width=True)

if cap is not None and cap.isOpened():
    stop_button = st.sidebar.button("⏹️ Stop Stream")
    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        process_video_frame(frame)
        time.sleep(0.03)
    cap.release()

elif video_source_type == "Synthetic Demo Generator":
    w, h = 854, 480
    sim_frame = np.ones((h, w, 3), dtype=np.uint8) * 35
    stage_w = int(w * (obstacle_pct / 100.0))
    cv2.rectangle(sim_frame, (0, 0), (stage_w, h), (50, 40, 60), -1)
    target_count = int(sim_density * 480.0)
    np.random.seed(int(time.time() * 10) % 1000)
    centers = [(np.random.randint(stage_w + 20, w - 20), np.random.randint(20, h - 20)) for _ in range(target_count)]
    process_video_frame(sim_frame)
    time.sleep(1.0)
    st.rerun()