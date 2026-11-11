"""Aplikasi Web Streamlit: Enterprise Smart Face Attendance & Mask Sentinel System
Pengembang: Lukman & Andika
Proyek: Tugas Besar Deep Learning - Masked Face Detection (MFD)
Arsitektur: YOLOv8 (Ultralytics) + FaceNet InceptionResnetV1 (CUDA Accelerated)
Desain UI: Ultra-Clean SaaS Studio (Light Mode Minimalist, Refined Layout & Spacing)
"""

import sys
import os
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import cv2
import pandas as pd
from PIL import Image
import torch
import streamlit as st

# Setup direktori root proyek
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from attendance_web import db_manager
from attendance_web.face_engine import FaceEngine

# ==============================================================================
# 1. KONFIGURASI HALAMAN STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="Sentinel AI - Smart Attendance System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inisialisasi Database SQLite
db_manager.init_db()

# Cache FaceEngine agar hanya dimuat 1 kali ke GPU
@st.cache_resource(show_spinner="Menghubungkan Pipeline AI ke GPU CUDA...")
def get_face_engine():
    return FaceEngine()

face_engine = get_face_engine()

if "streaming" not in st.session_state:
    st.session_state.streaming = False
if "attendance_cooldown" not in st.session_state:
    st.session_state.attendance_cooldown = {}

# ==============================================================================
# 2. DESIGN SYSTEM & REFINED LAYOUT CSS
# ==============================================================================
st.markdown("""
<style>
    /* Font 'Plus Jakarta Sans' */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
    
    /* Global Reset - Canvas Neutral Grey */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif !important;
        background-color: #F1F4F8 !important;
        color: #0F172A !important;
    }
    
    /* Hilangkan Dekorasi Bawaan Streamlit, Tapi PERTAHANKAN Tombol Buka/Tutup Sidebar! */
    #MainMenu, .stDeployButton, footer, [data-testid="stDecoration"] {
        display: none !important;
        visibility: hidden !important;
    }
    
    /* Header Streamlit transparan */
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 2.8rem !important;
        z-index: 99 !important;
    }
    
    /* Tombol Collapse / Expand Sidebar */
    [data-testid="stSidebarCollapseButton"] {
        background: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
        color: #0F172A !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08) !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stSidebarCollapseButton"]:hover {
        background: #F8FAFC !important;
        color: #0F172A !important;
        border-color: #94A3B8 !important;
    }
    
    .block-container {
        padding: 0.8rem 2.5rem 2.5rem 2.5rem !important;
        max-width: 1480px !important;
    }
    
    /* ========================================================= */
    /* 1. TYPOGRAPHY & HIGH-CONTRAST TEXT OVERRIDES (ANTI-BLIND) */
    /* ========================================================= */
    p, span, div, label, h1, h2, h3, h4, h5, h6 {
        color: #0F172A;
    }
    
    /* Label widget & form */
    [data-testid="stWidgetLabel"] label, [data-testid="stWidgetLabel"] p {
        color: #0F172A !important;
        font-weight: 700 !important;
        font-size: 0.86rem !important;
        margin-bottom: 4px !important;
    }
    
    /* Input Text & Textarea */
    input, textarea, [data-baseweb="input"] input {
        color: #0F172A !important;
        -webkit-text-fill-color: #0F172A !important;
        background-color: #FFFFFF !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
    }
    input::placeholder, textarea::placeholder {
        color: #94A3B8 !important;
        -webkit-text-fill-color: #94A3B8 !important;
        font-weight: 400 !important;
    }
    div[data-baseweb="input"] {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 10px !important;
        transition: all 0.2s ease !important;
    }
    div[data-baseweb="input"]:focus-within {
        border-color: #0F172A !important;
        box-shadow: 0 0 0 2px rgba(15, 23, 42, 0.12) !important;
    }
    
    /* Selectbox & Dropdown Popover */
    div[data-baseweb="select"] {
        background-color: #FFFFFF !important;
        border-radius: 10px !important;
    }
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 10px !important;
    }
    div[data-baseweb="select"] * {
        color: #0F172A !important;
        -webkit-text-fill-color: #0F172A !important;
        font-weight: 600 !important;
    }
    div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"] {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 10px !important;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1) !important;
    }
    li[role="option"], div[role="option"] {
        color: #0F172A !important;
        -webkit-text-fill-color: #0F172A !important;
        background-color: #FFFFFF !important;
        font-size: 0.86rem !important;
        font-weight: 500 !important;
        padding: 9px 14px !important;
    }
    li[role="option"]:hover, div[role="option"]:hover, li[aria-selected="true"] {
        background-color: #F1F5F9 !important;
        color: #0F172A !important;
        font-weight: 700 !important;
    }
    
    /* Tabs Component (Pilihan Kamera vs File) */
    button[data-baseweb="tab"] {
        color: #475569 !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 10px 16px !important;
    }
    button[data-baseweb="tab"]:hover {
        color: #0F172A !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #0F172A !important;
        font-weight: 800 !important;
        border-bottom: 2px solid #0F172A !important;
    }
    
    /* Camera Input Styling */
    div[data-testid="stCameraInput"] {
        border: 1px solid #CBD5E1 !important;
        border-radius: 14px !important;
        background: #FFFFFF !important;
        padding: 12px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
    }
    div[data-testid="stCameraInput"] * {
        color: #0F172A !important;
    }
    div[data-testid="stCameraInput"] button {
        background: #0F172A !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    
    /* File Uploader */
    div[data-testid="stFileUploader"] section {
        background-color: #FFFFFF !important;
        border: 1.5px dashed #CBD5E1 !important;
        border-radius: 12px !important;
    }
    div[data-testid="stFileUploader"] * {
        color: #0F172A !important;
    }
    div[data-testid="stFileUploader"] small {
        color: #64748B !important;
    }
    
    /* Sliders & Ticks */
    div[data-testid="stSlider"] label {
        color: #0F172A !important;
        font-weight: 600 !important;
    }
    div[data-testid="stSlider"] * {
        color: #0F172A !important;
    }
    
    /* Expanders */
    [data-testid="stExpander"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
    }
    [data-testid="stExpander"] summary {
        color: #0F172A !important;
        font-weight: 700 !important;
    }
    [data-testid="stExpander"] summary * {
        color: #0F172A !important;
    }
    
    /* Dataframe Table High Contrast */
    div[data-testid="stDataFrame"] {
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        background: #FFFFFF !important;
    }
    div[data-testid="stDataFrame"] * {
        color: #0F172A !important;
        font-size: 0.84rem !important;
    }
    
    /* Captions & Alerts */
    div[data-testid="stCaptionContainer"] p {
        color: #64748B !important;
    }
    .stAlert p {
        color: #0F172A !important;
        font-weight: 600 !important;
    }
    
    /* ========================================================= */
    /* 2. SIDEBAR DESIGN SYSTEM                                  */
    /* ========================================================= */
    [data-testid="stSidebar"] {
        background-color: #F8FAFC !important;
        border-right: 1px solid #E2E8F0 !important;
        padding-top: 10px !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: #E2E8F0 !important;
        margin: 14px 0 !important;
    }
    
    .brand-header {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 8px 18px 8px;
    }
    .brand-logo-squircle {
        width: 38px;
        height: 38px;
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #FFFFFF;
        font-size: 1.25rem;
        box-shadow: 0 4px 10px rgba(16, 185, 129, 0.25);
    }
    .brand-text {
        font-weight: 800;
        font-size: 1.15rem;
        color: #0F172A;
        letter-spacing: -0.02em;
    }
    
    .nav-category {
        font-size: 0.72rem;
        font-weight: 800;
        color: #64748B;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 10px 8px 6px 8px;
    }
    
    /* Radio Nav Items Styling */
    div[data-testid="stRadio"] > div {
        gap: 5px !important;
    }
    div[data-testid="stRadio"] input {
        display: none !important;
    }
    div[data-testid="stRadio"] label > div:first-child {
        display: none !important;
    }
    div[data-testid="stRadio"] label {
        padding: 10px 14px !important;
        border-radius: 10px !important;
        cursor: pointer !important;
        transition: all 0.15s ease !important;
        display: flex !important;
        align-items: center !important;
        background: transparent !important;
        border: 1px solid transparent !important;
        margin: 0 !important;
    }
    div[data-testid="stRadio"] label p, div[data-testid="stRadio"] label span {
        color: #475569 !important;
        font-weight: 600 !important;
        font-size: 0.90rem !important;
    }
    div[data-testid="stRadio"] label:hover {
        background: #EDF2F7 !important;
    }
    div[data-testid="stRadio"] label:hover p {
        color: #0F172A !important;
    }
    div[data-testid="stRadio"] label:has(input:checked) {
        background: #FFFFFF !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05), 0 1px 2px rgba(0, 0, 0, 0.04) !important;
        border: 1px solid #CBD5E1 !important;
    }
    div[data-testid="stRadio"] label:has(input:checked) p {
        color: #0F172A !important;
        font-weight: 800 !important;
    }
    
    /* User Profile Pill at Bottom */
    .profile-card {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 10px 12px;
        margin-top: 24px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }
    .profile-avatar {
        width: 34px;
        height: 34px;
        background: #DCFCE7;
        color: #15803D;
        font-weight: 800;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.85rem;
    }
    
    /* ========================================================= */
    /* 3. MAIN WORKSPACE CONTAINER                               */
    /* ========================================================= */
    .main-workspace {
        background: #FFFFFF;
        border-radius: 20px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.04), 0 2px 6px -1px rgba(0, 0, 0, 0.02);
        border: 1px solid #E2E8F0;
        padding: 26px 32px 32px 32px;
        margin-top: 4px;
    }
    
    .workspace-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 22px;
        padding-bottom: 16px;
        border-bottom: 1px solid #F1F5F9;
    }
    .ws-title {
        font-size: 1.55rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin: 0;
    }
    .ws-sub {
        font-size: 0.88rem;
        color: #475569;
        margin-top: 4px;
    }
    
    .status-badge {
        background: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 6px 14px;
        font-size: 0.80rem;
        font-weight: 700;
        color: #0F172A;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    
    /* Dashed Viewport Container */
    .studio-viewport {
        border: 2px dashed #CBD5E1;
        border-radius: 18px;
        background: #F8FAFC;
        min-height: 420px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 32px 20px;
        transition: border-color 0.2s ease;
    }
    .studio-viewport:hover {
        border-color: #94A3B8;
    }
    .studio-squircle {
        width: 58px;
        height: 58px;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.7rem;
        margin-bottom: 12px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.04);
    }
    
    /* Side Analytics & Tiles */
    .side-panel-header {
        font-size: 0.72rem;
        font-weight: 800;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 10px;
    }
    .stat-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 14px 16px;
    }
    .stat-val {
        font-size: 1.75rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.1;
    }
    
    /* Activity Feed Item */
    .feed-card-item {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 10px 12px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
    }
    
    /* Buttons Styling */
    .stButton > button[kind="primary"] {
        background: #0F172A !important;
        border: 1px solid #0F172A !important;
        border-radius: 10px !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        padding: 8px 20px !important;
        height: 42px !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
        transition: all 0.15s ease !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #1E293B !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.12) !important;
    }
    .stButton > button[kind="secondary"] {
        background: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 10px !important;
        color: #0F172A !important;
        font-weight: 700 !important;
        padding: 8px 18px !important;
        height: 42px !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background: #F8FAFC !important;
        border-color: #94A3B8 !important;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. SIDEBAR NAVIGATION
# ==============================================================================
with st.sidebar:
    st.markdown("""
    <div class="brand-header">
        <div class="brand-logo-squircle">🛡️</div>
        <div>
            <div class="brand-text">SentinelAI</div>
            <div style="font-size: 0.72rem; color: #64748B;">Mask & Attendance Studio</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="nav-category">MENU UTAMA</div>', unsafe_allow_html=True)
    
    nav_selection = st.radio(
        "Pilih Menu:",
        [
            "📸  Live Monitoring",
            "👤  Registrasi Anggota",
            "📊  Riwayat Presensi"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    st.markdown('<div class="nav-category">KONFIGURASI AI</div>', unsafe_allow_html=True)
    with st.expander("⚙️ Kalibrasi Sensitivitas", expanded=False):
        conf_slider = st.slider("Confidence Deteksi Masker", 0.10, 0.90, 0.25, 0.05,
                                help="Ambang batas deteksi YOLOv8 (Rekomendasi: 0.25)")
        dist_slider = st.slider("Toleransi Jarak Biometrik", 0.35, 0.70, 0.48, 0.01,
                                help="Batas baku optimal anti-nama-tertukar: 0.48")
        cooldown_mins = st.number_input("Cooldown Presensi (Menit)", 1, 60, 5)
        
    st.markdown('<div class="nav-category">STATUS KOMPUTASI</div>', unsafe_allow_html=True)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU Mode"
    st.markdown(f"""
    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 10px 12px; font-size: 0.78rem; color: #475569;">
        <div style="font-weight: 700; color: #0F172A; margin-bottom: 2px;">⚡ {gpu_name}</div>
        <div>Model: <code>{Path(face_engine.yolo_path).name}</code></div>
        <div style="color: #10B981; font-weight: 600; margin-top: 4px;">● CUDA Engine Ready</div>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# 4. MAIN WORKSPACE CONTAINER
# ==============================================================================

# ------------------------------------------------------------------------------
# VIEW 1: LIVE MONITORING
# ------------------------------------------------------------------------------
if "Live Monitoring" in nav_selection:
    st.markdown("""
    <div class="main-workspace">
        <div class="workspace-header">
            <div>
                <h1 class="ws-title">Live Monitoring & Mask Protocol</h1>
                <div class="ws-sub">Sistem pengawasan deteksi spasial kepatuhan masker dan pencocokan biometrik presensi secara real-time.</div>
            </div>
            <div>
                <span class="status-badge">🟢 Sistem Aktif • CUDA 12.4</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    col_viewport, col_side_analytics = st.columns([68, 32], gap="large")
    
    with col_viewport:
        demo_video_path = ROOT_DIR / "runs" / "detect" / "demo_webcam" / "0.avi"
        has_demo = demo_video_path.exists()
        
        src_list = ["Video Demo Bawaan (0.avi)", "Upload File Video Sendiri (.mp4, .avi)", "Live Kamera Fisik (Webcam Andika)"]
        if not has_demo:
            src_list.remove("Video Demo Bawaan (0.avi)")
            
        # Toolbar Rapi Sejajar dengan Tombol
        t_col1, t_col2, t_col3 = st.columns([5, 3, 2])
        with t_col1:
            chosen_source = st.selectbox("Sumber Masukan Stream:", src_list, label_visibility="collapsed")
        with t_col2:
            start_btn = st.button("▶️ Mulai Pemantauan", type="primary", use_container_width=True)
        with t_col3:
            stop_btn = st.button("⏹️ Hentikan", type="secondary", use_container_width=True)
            
        # Jika memilih upload, kotak uploader diletakkan rapi di atas viewport
        uploaded_video = None
        if chosen_source == "Upload File Video Sendiri (.mp4, .avi)":
            uploaded_video = st.file_uploader("Pilih Berkas Video Uji:", type=["mp4", "avi", "mov", "mkv"])
            
        video_screen = st.empty()
        status_banner = st.empty()
        
        # State Idle Mengadopsi Persis Dashed Box Referensi Gambar!
        if not st.session_state.streaming:
            video_screen.markdown("""
            <div class="studio-viewport">
                <div class="studio-squircle">📹</div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #0F172A; margin-bottom: 4px;">
                    Viewport Pemantauan Kamera Siap
                </div>
                <div style="font-size: 0.85rem; color: #64748B; max-width: 420px; line-height: 1.4; margin-bottom: 16px;">
                    Pilih sumber video pada toolbar di atas lalu klik tombol <b>Mulai Pemantauan</b> untuk menjalankan pipeline inferensi.
                </div>
                <div style="display: inline-flex; align-items: center; gap: 8px; background: #FFFFFF; border: 1px solid #E2E8F0; padding: 6px 14px; border-radius: 8px; font-size: 0.78rem; font-weight: 600; color: #475569; box-shadow: 0 1px 2px rgba(0,0,0,0.03);">
                    ⚡ Akselerasi Hardware: NVIDIA RTX 3050 Laptop GPU
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_side_analytics:
        # Side Analytics Header yang Sejajar Rapi
        st.markdown('<div class="side-panel-header">RINGKASAN PRESENSI HARI INI</div>', unsafe_allow_html=True)
        summary = db_manager.get_today_summary()
        
        # 2 Kartu Metrik Sejajar
        st.markdown(f"""
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 12px;">
            <div class="stat-card">
                <div style="font-size: 0.72rem; font-weight: 700; color: #64748B; text-transform: uppercase;">HADIR UNIK</div>
                <div class="stat-val">{summary['total_present']}</div>
                <div style="font-size: 0.75rem; color: #94A3B8;">Personil</div>
            </div>
            <div class="stat-card">
                <div style="font-size: 0.72rem; font-weight: 700; color: #64748B; text-transform: uppercase;">TOTAL LOG</div>
                <div class="stat-val">{summary['total_logs']}</div>
                <div style="font-size: 0.75rem; color: #94A3B8;">Verifikasi AI</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Kartu Kepatuhan Masker
        pct = summary['patuh_pct']
        color_bar = "#10B981" if pct >= 75 else ("#F59E0B" if pct >= 50 else "#EF4444")
        st.markdown(f"""
        <div class="stat-card" style="margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="font-size: 0.72rem; font-weight: 700; color: #64748B; text-transform: uppercase;">KEPATUHAN MASKER</div>
                <div style="font-weight: 800; color: {color_bar}; font-size: 1.05rem;">{pct}%</div>
            </div>
            <div style="background: #E2E8F0; border-radius: 999px; height: 7px; overflow: hidden; margin: 8px 0;">
                <div style="background: {color_bar}; width: {pct}%; height: 100%; border-radius: 999px;"></div>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #64748B;">
                <span>🟢 Patuh: <b>{summary['total_patuh']}</b></span>
                <span>🔴 Melanggar: <b>{summary['total_melanggar']}</b></span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="side-panel-header">AKTIVITAS VERIFIKASI TERKINI</div>', unsafe_allow_html=True)
        recent_feed = st.empty()
        
        def render_recent_feed():
            recent = db_manager.get_attendance_logs(limit=5)
            if not recent:
                recent_feed.markdown("""
                <div style="text-align: center; padding: 22px 10px; color: #94A3B8; font-size: 0.82rem; background: #F8FAFC; border: 1px dashed #E2E8F0; border-radius: 10px;">
                    Belum ada riwayat verifikasi hari ini.
                </div>
                """, unsafe_allow_html=True)
                return
            
            feed_html = ""
            for r in recent:
                is_patuh = (r["compliance"] == "PATUH")
                badge_style = "background: #DCFCE7; color: #166534;" if is_patuh else "background: #FEE2E2; color: #991B1B;"
                t_str = r["timestamp"].split(" ")[-1] if " " in r["timestamp"] else r["timestamp"]
                
                feed_html += f"""
                <div class="feed-card-item">
                    <div style="display: flex; align-items: center; gap: 8px; min-width: 0;">
                        <div style="width: 26px; height: 26px; border-radius: 50%; background: #E2E8F0; display: flex; align-items: center; justify-content: center; font-size: 0.75rem;">
                            {'👤' if is_patuh else '⚠️'}
                        </div>
                        <div style="min-width: 0;">
                            <div style="font-weight: 700; font-size: 0.80rem; color: #0F172A; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                                {r['name']}
                            </div>
                            <div style="font-size: 0.70rem; color: #64748B;">
                                {t_str} • {r['identifier'] or '-'}
                            </div>
                        </div>
                    </div>
                    <span style="{badge_style} font-weight: 700; font-size: 0.68rem; padding: 3px 7px; border-radius: 5px;">
                        {r['compliance']}
                    </span>
                </div>
                """
            recent_feed.markdown(feed_html, unsafe_allow_html=True)
            
        render_recent_feed()

    # Kontrol State Streaming
    if start_btn:
        st.session_state.streaming = True
    if stop_btn:
        st.session_state.streaming = False
        st.rerun()

    # Streaming Loop
    if st.session_state.streaming:
        cap = None
        temp_file = None
        
        if chosen_source == "Video Demo Bawaan (0.avi)":
            if demo_video_path.exists():
                cap = cv2.VideoCapture(str(demo_video_path))
                status_banner.info(f"▶️ Streaming Aktif: Memutar demo `{demo_video_path.name}`")
            else:
                st.error("File demo tidak ditemukan.")
                st.session_state.streaming = False
        elif chosen_source == "Upload File Video Sendiri (.mp4, .avi)":
            if uploaded_video:
                temp_file = ROOT_DIR / "attendance_web" / f"temp_{uploaded_video.name}"
                with open(temp_file, "wb") as f:
                    f.write(uploaded_video.read())
                cap = cv2.VideoCapture(str(temp_file))
                status_banner.info(f"▶️ Streaming Aktif: Memutar `{uploaded_video.name}`")
            else:
                st.warning("Silakan unggah video terlebih dahulu.")
                st.session_state.streaming = False
        elif chosen_source == "Live Kamera Fisik (Webcam Andika)":
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                status_banner.error("❌ Kamera fisik tidak terdeteksi. Pastikan webcam terpasang atau gunakan opsi Video Demo.")
                st.session_state.streaming = False
            else:
                status_banner.info("▶️ Mengakses Kamera Fisik. Tekan tombol 'Hentikan' untuk selesai.")

        if cap and cap.isOpened():
            # Set resolusi 640x480 agar enteng dan adem di laptop
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            registered_users = db_manager.get_all_users()
            frame_idx = 0
            cached_annotated = None
            cached_events = []
            
            while cap.isOpened() and st.session_state.streaming:
                ret, frame = cap.read()
                if not ret:
                    if chosen_source != "Live Kamera Fisik (Webcam Andika)":
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        break
                        
                frame_idx += 1
                # Optimasi Cerdas: Eksekusi pipeline AI tiap 2 frame (beban turun 50%, FPS naik)
                if frame_idx % 2 == 1 or cached_annotated is None:
                    annotated_frame, events = face_engine.process_frame(
                        frame, 
                        registered_users=registered_users, 
                        conf_threshold=conf_slider,
                        distance_threshold=dist_slider
                    )
                    cached_annotated = annotated_frame
                    cached_events = events
                else:
                    annotated_frame = cached_annotated
                    events = cached_events
                
                # Catat Presensi
                for ev in events:
                    name = ev["name"]
                    ident = ev["identifier"]
                    comp = ev["compliance"]
                    status_mask = ev["mask_status"]
                    conf = ev["confidence"]
                    crop_img = ev["crop"]
                    user_id = ev["user_id"]
                    
                    if name != "Tamu / Belum Terdaftar" and ident != "Unknown" and user_id is not None:
                        now_ts = time.time()
                        last_ts = st.session_state.attendance_cooldown.get(ident, 0)
                        cooldown_secs = float(cooldown_mins) * 60.0
                        
                        # Dual-Layer Anti-Spam: Cek in-memory session dan database
                        if (now_ts - last_ts) >= cooldown_secs:
                            if db_manager.can_attend(ident, cooldown_minutes=cooldown_mins):
                                st.session_state.attendance_cooldown[ident] = now_ts
                                now_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                snap_name = f"{ident}_{now_stamp}.jpg"
                                snap_path = db_manager.SNAPSHOTS_DIR / snap_name
                                cv2.imwrite(str(snap_path), crop_img)
                                
                                db_manager.record_attendance(
                                    name=name,
                                    identifier=ident,
                                    mask_status=status_mask,
                                    compliance=comp,
                                    confidence=conf,
                                    snapshot_path=str(snap_path),
                                    user_id=user_id
                                )
                                st.toast(f"✅ Presensi Berhasil: {name} ({comp})", icon="🎉")
                                render_recent_feed()
                
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                video_screen.image(frame_rgb, channels="RGB", use_container_width=True)
                time.sleep(0.02)
                
            cap.release()
            if temp_file and temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception:
                    pass
            status_banner.success("⏹️ Pemantauan video selesai.")
            st.session_state.streaming = False
            
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# VIEW 2: REGISTRASI ANGGOTA
# ------------------------------------------------------------------------------
elif "Registrasi Anggota" in nav_selection:
    st.markdown("""
    <div class="main-workspace">
        <div class="workspace-header">
            <div>
                <h1 class="ws-title">Registrasi Biometrik Anggota</h1>
                <div class="ws-sub">Daftarkan identitas personil baru dan unggah pasfoto untuk ekstraksi vektor wajah FaceNet (512 dimensi).</div>
            </div>
            <div>
                <span class="status-badge">👤 Formulir Pendaftaran</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    col_reg_form, col_reg_dir = st.columns([5, 5], gap="large")
    
    with col_reg_form:
        st.markdown("<p style='font-size:0.85rem; font-weight:700; color:#0F172A; margin-bottom:12px;'>📝 Formulir Anggota Baru</p>", unsafe_allow_html=True)
        
        tab_cam, tab_file = st.tabs(["📸 Ambil Foto Langsung (Webcam)", "📁 Unggah File Pasfoto"])
        with tab_cam:
            cam_photo = st.camera_input("Posisikan wajah lurus dan klik Ambil Foto:", key="reg_cam_input")
        with tab_file:
            uploaded_photo = st.file_uploader("Pilih berkas pasfoto (.jpg / .png):", type=["jpg", "jpeg", "png"],
                                              help="Pastikan wajah tampak jelas menghadap kamera.", key="reg_file_input")
            
        active_photo = cam_photo if cam_photo is not None else uploaded_photo
        
        with st.form("form_register_member"):
            r_name = st.text_input("Nama Lengkap:*", placeholder="Contoh: Muhammad Lukman")
            r_id = st.text_input("Nomor Induk (NIM / NIP):*", placeholder="Contoh: 41524010017")
            r_role = st.selectbox("Peran / Jabatan:", ["Mahasiswa", "Dosen Pengampu", "Asisten Laboratorium", "Tenaga Kependidikan", "Tamu VIP"])
            
            submit_member = st.form_submit_button("💾 Daftarkan Anggota", type="primary", use_container_width=True)
            
        if submit_member:
            if not r_name.strip() or not r_id.strip() or not active_photo:
                st.error("⚠️ Nama, NIM, dan Foto Wajah (jepret langsung dari webcam atau unggah file) wajib diisi lengkap!")
            else:
                with st.spinner("Mengekstrak biometrik wajah ke CUDA..."):
                    img_bytes = active_photo.getvalue()
                    success, emb, face_crop, err_msg = face_engine.register_face_from_image(img_bytes)
                    
                    if not success or emb is None:
                        st.error(f"❌ Registrasi Ditolak: {err_msg}")
                    else:
                        clean_name = "".join(c for c in r_name if c.isalnum() or c in (" ", "_")).strip()
                        file_dest = db_manager.REGISTERED_FACES_DIR / f"{clean_name}_{r_id}.jpg"
                        
                        if face_crop is not None:
                            cv2.imwrite(str(file_dest), face_crop)
                        else:
                            with open(file_dest, "wb") as f:
                                f.write(img_bytes)
                                
                        ok_db, msg_db = db_manager.add_user(
                            name=r_name,
                            identifier=r_id,
                            role=r_role,
                            photo_path=str(file_dest),
                            embedding=emb
                        )
                        
                        if ok_db:
                            st.success(f"🎉 **Sukses!** {msg_db}")
                            if face_crop is not None:
                                crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                                st.image(crop_rgb, caption="Potongan Wajah Terdaftar (FaceNet 512D)", width=110)
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"❌ Gagal Simpan Database: {msg_db}")

    with col_reg_dir:
        st.markdown("<p style='font-size:0.85rem; font-weight:700; color:#0F172A; margin-bottom:12px;'>📋 Direktori Anggota Terverifikasi</p>", unsafe_allow_html=True)
        users_list = db_manager.get_all_users()
        
        if not users_list:
            st.info("Belum ada anggota yang terdaftar dalam sistem.")
        else:
            search_query = st.text_input("Cari personil:", placeholder="Cari nama atau NIM...", label_visibility="collapsed")
            if search_query:
                users_list = [u for u in users_list if search_query.lower() in u["name"].lower() or search_query.lower() in u["identifier"].lower()]
                
            st.caption(f"Menampilkan {len(users_list)} anggota terdaftar")
            
            st_box = st.container(height=380)
            with st_box:
                for u in users_list:
                    u_c1, u_c2, u_c3 = st.columns([2, 6, 2])
                    with u_c1:
                        p_file = Path(u["photo_path"])
                        if p_file.exists():
                            st.image(str(p_file), width=48)
                        else:
                            st.markdown("👤")
                    with u_c2:
                        st.markdown(f"<div style='font-weight:700; font-size:0.90rem; color:#0F172A;'>{u['name']}</div><div style='font-size:0.75rem; color:#475569; font-weight:500;'>NIM: <code style='color:#0F172A; background:#E2E8F0; padding:1px 4px; border-radius:4px;'>{u['identifier']}</code> • {u['role']}</div>", unsafe_allow_html=True)
                    with u_c3:
                        if st.button("🗑️", key=f"del_{u['id']}", help=f"Hapus {u['name']}"):
                            db_manager.delete_user(u["id"])
                            st.toast(f"Anggota {u['name']} dihapus.")
                            st.rerun()
                    st.markdown("<hr style='margin: 8px 0; border: none; border-top: 1px solid #F1F5F9;'>", unsafe_allow_html=True)
                    
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# VIEW 3: RIWAYAT PRESENSI & LAPORAN
# ------------------------------------------------------------------------------
elif "Riwayat Presensi" in nav_selection:
    st.markdown("""
    <div class="main-workspace">
        <div class="workspace-header">
            <div>
                <h1 class="ws-title">Audit Trail & Riwayat Kehadiran</h1>
                <div class="ws-sub">Rekapitulasi lengkap riwayat presensi masuk, status kepatuhan protokol, dan bukti verifikasi AI.</div>
            </div>
            <div>
                <span class="status-badge">📊 Database SQLite Connected</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    f_c1, f_c2, f_c3 = st.columns([3, 3, 4], gap="medium")
    with f_c1:
        f_date = st.date_input("Filter Berdasarkan Tanggal:", value=None)
    with f_c2:
        f_comp = st.selectbox("Status Kepatuhan:", ["Semua", "PATUH", "MELANGGAR"])
    with f_c3:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        reset_filter = st.button("🔄 Reset Filter", use_container_width=True)
        if reset_filter:
            f_date = None
            f_comp = "Semua"
            st.rerun()
            
    date_str = f_date.strftime("%Y-%m-%d") if f_date else None
    logs_data = db_manager.get_attendance_logs(date_filter=date_str, compliance_filter=f_comp, limit=1000)
    
    if not logs_data:
        st.info("ℹ️ Tidak ditemukan riwayat presensi yang sesuai dengan kriteria filter.")
    else:
        df_raw = pd.DataFrame(logs_data)
        
        tot_shown = len(df_raw)
        tot_patuh_shown = len(df_raw[df_raw["compliance"] == "PATUH"])
        tot_melanggar_shown = len(df_raw[df_raw["compliance"] == "MELANGGAR"])
        
        st.markdown(f"""
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-bottom: 18px;">
            <div class="stat-card">
                <div style="font-size: 0.72rem; font-weight: 700; color: #64748B; text-transform: uppercase;">LOG DITAMPILKAN</div>
                <div class="stat-val">{tot_shown}</div>
                <div style="font-size: 0.75rem; color: #94A3B8;">Entri Data</div>
            </div>
            <div class="stat-card">
                <div style="font-size: 0.72rem; font-weight: 700; color: #64748B; text-transform: uppercase;">TOTAL PATUH</div>
                <div class="stat-val" style="color: #059669;">{tot_patuh_shown}</div>
                <div style="font-size: 0.75rem; color: #94A3B8;">Masker Sesuai Protokol</div>
            </div>
            <div class="stat-card">
                <div style="font-size: 0.72rem; font-weight: 700; color: #64748B; text-transform: uppercase;">PELANGGARAN</div>
                <div class="stat-val" style="color: #DC2626;">{tot_melanggar_shown}</div>
                <div style="font-size: 0.75rem; color: #94A3B8;">Tanpa / Salah Masker</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        df_display = df_raw[["id", "timestamp", "name", "identifier", "mask_status", "compliance", "confidence"]].copy()
        
        label_map = {
            "masked_face": "Masker Benar (Hidung & Mulut)",
            "unmasked_face": "Wajah Tanpa Masker",
            "incorrectly_masked_face": "Masker Melorot / Salah Pasang"
        }
        df_display["mask_status"] = df_display["mask_status"].map(lambda x: label_map.get(x, x))
        df_display["confidence"] = df_display["confidence"].apply(lambda v: f"{v*100:.1f}%")
        
        df_display.columns = ["ID Log", "Waktu Presensi", "Nama Personil", "NIM / ID", "Kategori Masker", "Status Kepatuhan", "Keyakinan AI"]
        
        st.dataframe(df_display, use_container_width=True, hide_index=True)
        
        csv_export = df_display.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Unduh Laporan Resmi (CSV / Excel)",
            data=csv_export,
            file_name=f"laporan_presensi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            type="primary"
        )
        
    st.markdown("</div>", unsafe_allow_html=True)
