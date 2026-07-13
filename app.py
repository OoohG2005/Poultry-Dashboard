import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import re
import numpy as np

# Import custom modules
from data_generator import PoultrySensorData, get_latest_reading, get_recent_data
from model import PoultryDiseasePredictor

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Smart Poultry Monitoring System",
    page_icon="🐔",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== PWA META TAGS & MANIFEST ====================
st.markdown("""
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <link rel="manifest" href="data:application/json;base64,eyJuYW1lIjoiU21hcnQgUG91bHRyeSBNb25pdG9yaW5nIFN5c3RlbSIsInNob3J0X25hbWUiOiJTbWFydCBQb3VsdHJ5Iiwic3RhcnRfdXJsIjoiLiIsImRpc3BsYXkiOiJzdGFuZGFsb25lIiwiYmFja2dyb3VuZF9jb2xvciI6IiMxYTNjNWUiLCJ0aGVtZV9jb2xvciI6IiMxYTNjNWUifQ==">
""", unsafe_allow_html=True)

# ==================== SCROLL TO TOP ON PAGE LOAD ====================
st.markdown(
    """
    <script>
        window.scrollTo(0, 0);
    </script>
    """,
    unsafe_allow_html=True
)

# ==================== SESSION STATE ====================
if 'generator' not in st.session_state:
    st.session_state.generator = PoultrySensorData()

if 'data' not in st.session_state:
    with st.spinner("Loading data..."):
        st.session_state.data = st.session_state.generator.generate_data(days=30)

if 'predictor' not in st.session_state:
    st.session_state.predictor = PoultryDiseasePredictor()
    with st.spinner("Training AI..."):
        st.session_state.predictor.train(st.session_state.data)

if 'latest_reading' not in st.session_state:
    st.session_state.latest_reading = get_latest_reading(st.session_state.data)

if 'action_log' not in st.session_state:
    st.session_state.action_log = []

if 'notifications_sent' not in st.session_state:
    st.session_state.notifications_sent = []

if 'data_logged' not in st.session_state:
    st.session_state.data_logged = 0

# App Navigation State
if 'page' not in st.session_state:
    st.session_state.page = "Home"

# Farm Profile
if 'farm_name' not in st.session_state:
    st.session_state.farm_name = "Green Valley Farm"
if 'bird_count' not in st.session_state:
    st.session_state.bird_count = 2000
if 'bird_age' not in st.session_state:
    st.session_state.bird_age = "4 weeks"
if 'bird_type' not in st.session_state:
    st.session_state.bird_type = "Broiler"

# Contacts
if 'farmer_name' not in st.session_state:
    st.session_state.farmer_name = "John Banda"
if 'farmer_email' not in st.session_state:
    st.session_state.farmer_email = "farmer@example.com"
if 'farmer_phone' not in st.session_state:
    st.session_state.farmer_phone = "+265 999 111 111"

if 'vet_name' not in st.session_state:
    st.session_state.vet_name = "Dr. Mary Phiri"
if 'vet_email' not in st.session_state:
    st.session_state.vet_email = "vet@example.com"
if 'vet_phone' not in st.session_state:
    st.session_state.vet_phone = "+265 999 222 222"

# Thresholds
if 'thresholds' not in st.session_state:
    st.session_state.thresholds = {
        'temp_min': 20, 'temp_max': 30,
        'humidity_min': 60, 'humidity_max': 80,
        'ammonia_max': 10,
        'co2_max': 1000,
        'water_min': 30,
        'feed_min': 25,
        'light_min': 200, 'light_max': 800
    }

# ==================== CSS: RESPONSIVE APP STYLE ====================
st.markdown("""
<style>
    /* ===== COMPLETELY HIDE ALL STREAMLIT UI ===== */
    /* Hide menu, header, footer, deploy button, toolbar, and everything else */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    .stAppToolbar {display: none;}
    
    /* ---- ADDED: Remove "Manage app" text and toolbar ---- */
    [data-testid="stToolbar"] {
        display: none !important;
    }
    .stAppHeader {
        display: none !important;
    }
    .stApp > header {
        display: none !important;
    }
    
    /* Remove extra padding caused by hidden header */
    .stApp > div:first-child {
        padding: 0 !important;
    }
    .stMain {
        padding-top: 0 !important;
    }
    
    /* ---- Responsive container ---- */
    .stApp {
        background: #f8f9fa;
        height: 100vh;
        overflow-y: auto;
        overflow-x: hidden;
    }
    .block-container {
        padding: 0.8rem 1rem 5.5rem 1rem !important;
        max-width: 100% !important;
    }
    
    @media (max-width: 768px) {
        .stApp {
            max-width: 480px;
            margin: 0 auto;
        }
        .block-container {
            padding: 0.8rem 1rem 5.5rem 1rem !important;
        }
    }
    
    /* Hide scrollbar */
    ::-webkit-scrollbar {width: 0; background: transparent;}
    
    /* ---- App Header ---- */
    .app-header {
        display: flex;
        flex-direction: column;
        padding: 0.8rem 0.5rem;
        background: #1a3c5e;
        color: white;
        border-radius: 0 0 16px 16px;
        margin-bottom: 1rem;
        text-align: center;
    }
    .app-header h1 {
        font-size: 1.4rem;
        margin: 0;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .app-header .subtitle {
        font-size: 0.8rem;
        opacity: 0.85;
        margin: 0.2rem 0 0 0;
        font-weight: 300;
    }
    .app-header .farm-name {
        font-size: 1.2rem;
        font-weight: 600;
        opacity: 0.9;
        margin-top: 0.4rem;
    }
    
    /* ---- Cards ---- */
    .card {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        border: 1px solid #ecf0f1;
    }
    .card-title {
        font-size: 0.8rem;
        color: #7f8c8d;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    
    /* ---- Status Boxes ---- */
    .status-healthy {
        background: linear-gradient(135deg, #27ae60, #2ecc71);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        width: 100%;
    }
    .status-warning {
        background: linear-gradient(135deg, #f39c12, #f1c40f);
        padding: 1.2rem;
        border-radius: 12px;
        color: #1a1a2e;
        text-align: center;
        width: 100%;
    }
    .status-danger {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        width: 100%;
    }
    .status-text {
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0;
    }
    .status-sub {
        font-size: 0.9rem;
        opacity: 0.9;
        margin: 0.2rem 0 0 0;
    }
    
    /* ---- Sensor Mini Cards ---- */
    .sensor-mini {
        background: white;
        border-radius: 10px;
        padding: 0.6rem;
        text-align: center;
        border: 1px solid #ecf0f1;
        margin-bottom: 0.5rem;
    }
    .sensor-mini .label {
        font-size: 0.65rem;
        color: #7f8c8d;
    }
    .sensor-mini .value {
        font-size: 1.2rem;
        font-weight: 700;
        color: #2c3e50;
    }
    .sensor-mini .unit {
        font-size: 0.65rem;
        color: #95a5a6;
    }
    .sensor-mini .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 4px;
    }
    .dot-good { background: #27ae60; }
    .dot-warning { background: #f39c12; }
    .dot-danger { background: #e74c3c; }
    
    /* ---- Actuator Badge ---- */
    .actuator-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    .actuator-on {
        background: #27ae60;
        color: white;
    }
    .actuator-off {
        background: #7f8c8d;
        color: white;
    }
    
    /* ---- Notification ---- */
    .notif-item {
        background: #f8f9fa;
        padding: 0.5rem;
        border-radius: 8px;
        margin-bottom: 0.4rem;
        border-left: 3px solid #2980b9;
        font-size: 0.8rem;
    }
    .notif-item .time {
        font-size: 0.6rem;
        color: #95a5a6;
    }
    
    /* ---- Inputs & Buttons ---- */
    .stTextInput > div > div > input {
        border-radius: 8px !important;
        border: 1px solid #ddd !important;
        padding: 0.5rem !important;
        font-size: 0.9rem !important;
    }
    .stButton > button {
        border-radius: 10px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 600 !important;
        background: #1a3c5e !important;
        color: white !important;
        border: none !important;
        width: 100% !important;
        font-size: 0.9rem !important;
    }
    .stButton > button:hover {
        background: #2c5f87 !important;
    }
    .stSelectbox > div > div {
        border-radius: 8px !important;
    }
    
    /* ---- Plotly ---- */
    .js-plotly-plot .plotly .main-svg {
        border-radius: 10px !important;
    }
    
    /* ---- Bottom Navigation ---- */
    .nav-btn {
        background: transparent !important;
        border: none !important;
        text-align: center;
        padding: 0.3rem 0.5rem;
        color: #7f8c8d;
        font-size: 0.7rem;
        font-weight: 500;
        width: 100%;
        border-radius: 8px;
        transition: all 0.2s;
        cursor: pointer;
    }
    .nav-btn:hover {
        background: #f0f0f0 !important;
    }
    .nav-btn.active {
        color: #1a3c5e;
        font-weight: 600;
    }
    .nav-icon {
        font-size: 1.5rem;
        display: block;
        line-height: 1.2;
    }
</style>
""", unsafe_allow_html=True)

# ==================== FUNCTIONS ====================
def get_actuator_states(latest, thresholds):
    states = {}
    t = thresholds
    states['Cooling Fan'] = 'ON' if latest['temperature'] > t['temp_max'] else 'OFF'
    states['Heater'] = 'ON' if latest['temperature'] < t['temp_min'] else 'OFF'
    states['Ventilation'] = 'ON' if (latest['humidity'] > t['humidity_max'] or 
                                      latest['ammonia'] > t['ammonia_max'] or 
                                      latest['co2'] > t['co2_max']) else 'OFF'
    states['Water Pump'] = 'ON' if latest['water_level'] < t['water_min'] else 'OFF'
    states['Alarm'] = 'ON' if latest.get('risk_label', 0) == 2 else 'OFF'
    states['Lighting'] = 'ON' if latest['light_intensity'] < t['light_min'] else 'OFF'
    return states

def log_action(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.action_log.insert(0, f"{timestamp} - {message}")
    if len(st.session_state.action_log) > 20:
        st.session_state.action_log.pop()

def send_notification(recipient_type, recipient_name, email, phone, subject, message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    notification = {
        'timestamp': timestamp,
        'recipient_type': recipient_type,
        'recipient_name': recipient_name,
        'email': email,
        'phone': phone,
        'subject': subject,
        'message': message
    }
    st.session_state.notifications_sent.insert(0, notification)
    if len(st.session_state.notifications_sent) > 50:
        st.session_state.notifications_sent.pop()
    log_action(f"📧 Notification sent to {recipient_type}: {recipient_name}")
    return True

def get_sensor_status(value, good_range):
    if value < good_range[0] or value > good_range[1]:
        return 'danger'
    elif value < good_range[0] + (good_range[1] - good_range[0]) * 0.15 or value > good_range[1] - (good_range[1] - good_range[0]) * 0.15:
        return 'warning'
    else:
        return 'good'

def render_nav():
    nav_items = [
        {"label": "Home", "icon": "🏠", "page": "Home"},
        {"label": "Sensors", "icon": "📊", "page": "Sensors"},
        {"label": "Control", "icon": "⚙️", "page": "Actuators"},
        {"label": "Alerts", "icon": "🔔", "page": "Alerts"},
        {"label": "Settings", "icon": "👤", "page": "Settings"}
    ]
    cols = st.columns(len(nav_items))
    for col, item in zip(cols, nav_items):
        with col:
            active = st.session_state.page == item["page"]
            if st.button(
                f"{item['icon']}\n{item['label']}",
                key=f"nav_{item['page']}",
                use_container_width=True,
                type="secondary" if not active else "primary"
            ):
                st.session_state.page = item["page"]
                st.rerun()

# ==================== DATA ====================
latest = st.session_state.latest_reading
prediction = st.session_state.predictor.predict(latest)
actuator_states = get_actuator_states(latest, st.session_state.thresholds)

# ==================== APP HEADER ====================
st.markdown(f"""
<div class="app-header">
    <h1>Smart Poultry House Monitoring System</h1>
    <div class="subtitle">AI-Powered Environmental Monitoring & Disease Prediction</div>
    <div class="farm-name">{st.session_state.farm_name}</div>
</div>
""", unsafe_allow_html=True)

# ==================== TOP ROW: STATUS ====================
status_icon = " " if prediction['status'] == 'Healthy' else (" " if prediction['status'] == 'Warning' else " ")
status_class = "status-healthy" if prediction['status'] == 'Healthy' else ("status-warning" if prediction['status'] == 'Warning' else "status-danger")
st.markdown(f"""
<div class="{status_class}">
    <div class="status-text">{status_icon} {prediction['status']}</div>
    <div class="status-sub">Confidence: {prediction['confidence']:.1f}%</div>
    <div class="status-sub">{datetime.now().strftime('%H:%M:%S')}</div>
</div>
""", unsafe_allow_html=True)

# ==================== PAGE RENDER ====================
page = st.session_state.page

if page == "Home":
    st.markdown('<div class="card-title">Quick Stats</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="card" style="text-align:center;padding:0.5rem;"><div style="font-size:0.6rem;color:#7f8c8d;">Birds</div><div style="font-weight:700;">{st.session_state.bird_count}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="card" style="text-align:center;padding:0.5rem;"><div style="font-size:0.6rem;color:#7f8c8d;">Age</div><div style="font-weight:700;font-size:0.9rem;">{st.session_state.bird_age}</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="card" style="text-align:center;padding:0.5rem;"><div style="font-size:0.6rem;color:#7f8c8d;">Type</div><div style="font-weight:700;font-size:0.9rem;">{st.session_state.bird_type}</div></div>', unsafe_allow_html=True)
    with col4:
        active_actuators = sum(1 for s in actuator_states.values() if s == 'ON')
        st.markdown(f'<div class="card" style="text-align:center;padding:0.5rem;"><div style="font-size:0.6rem;color:#7f8c8d;">Active</div><div style="font-weight:700;">{active_actuators}/6</div></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card-title">Key Readings</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        status = get_sensor_status(latest['temperature'], (20, 30))
        dot = "dot-good" if status == 'good' else ("dot-warning" if status == 'warning' else "dot-danger")
        st.markdown(f"""
        <div class="sensor-mini">
            <div class="label">Temperature</div>
            <div class="value">{latest['temperature']:.1f}°</div>
            <div><span class="status-dot {dot}"></span> <span style="font-size:0.6rem;">{'Good' if status == 'good' else status.title()}</span></div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        status = get_sensor_status(latest['humidity'], (60, 80))
        dot = "dot-good" if status == 'good' else ("dot-warning" if status == 'warning' else "dot-danger")
        st.markdown(f"""
        <div class="sensor-mini">
            <div class="label">Humidity</div>
            <div class="value">{latest['humidity']:.1f}%</div>
            <div><span class="status-dot {dot}"></span> <span style="font-size:0.6rem;">{'Good' if status == 'good' else status.title()}</span></div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        status = get_sensor_status(latest['ammonia'], (0, 10))
        dot = "dot-good" if status == 'good' else ("dot-warning" if status == 'warning' else "dot-danger")
        st.markdown(f"""
        <div class="sensor-mini">
            <div class="label">Ammonia</div>
            <div class="value">{latest['ammonia']:.1f}</div>
            <div><span class="status-dot {dot}"></span> <span style="font-size:0.6rem;">{'Good' if status == 'good' else status.title()}</span></div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="card-title">Recent Alerts</div>', unsafe_allow_html=True)
    alerts = []
    t = st.session_state.thresholds
    if latest['temperature'] > t['temp_max']:
        alerts.append("🔥 High Temperature")
    if latest['humidity'] > t['humidity_max']:
        alerts.append("💧 High Humidity")
    if latest['ammonia'] > t['ammonia_max']:
        alerts.append("☣️ Dangerous Ammonia")
    if latest['water_level'] < t['water_min']:
        alerts.append("💦 Low Water")
    if latest['feed_level'] < t['feed_min']:
        alerts.append("🌾 Low Feed")
    if prediction['status'] == 'Disease Risk':
        alerts.append("🚨 Disease Risk Detected")
    
    if alerts:
        for alert in alerts[:3]:
            st.markdown(f'<div class="notif-item">⚠️ {alert}</div>', unsafe_allow_html=True)
        if len(alerts) > 3:
            st.caption(f"+ {len(alerts)-3} more alerts...")
    else:
        st.success(" All clear! No alerts.")

elif page == "Sensors":
    st.markdown('<div class="card-title">All Sensors</div>', unsafe_allow_html=True)
    
    sensor_configs = [
        {'key': 'temperature', 'label': 'Temperature', 'unit': '°C', 'good': (20, 30)},
        {'key': 'humidity', 'label': 'Humidity', 'unit': '%', 'good': (60, 80)},
        {'key': 'ammonia', 'label': 'Ammonia', 'unit': 'ppm', 'good': (0, 10)},
        {'key': 'co2', 'label': 'CO2', 'unit': 'ppm', 'good': (300, 1000)},
        {'key': 'water_level', 'label': 'Water Level', 'unit': '%', 'good': (30, 100)},
        {'key': 'feed_level', 'label': 'Feed Level', 'unit': '%', 'good': (25, 100)},
        {'key': 'light_intensity', 'label': 'Light', 'unit': 'lux', 'good': (200, 800)},
        {'key': 'air_quality', 'label': 'Air Quality', 'unit': '', 'good': (0, 300)}
    ]
    
    for config in sensor_configs:
        value = latest[config['key']]
        status = get_sensor_status(value, config['good'])
        dot = "dot-good" if status == 'good' else ("dot-warning" if status == 'warning' else "dot-danger")
        status_text = "Good" if status == 'good' else ("Warning" if status == 'warning' else "Danger")
        
        st.markdown(f"""
        <div class="card" style="display:flex;justify-content:space-between;align-items:center;">
            <div>
                <div style="font-weight:600;font-size:0.9rem;">{config['label']}</div>
                <div style="font-size:0.7rem;color:#7f8c8d;">Target: {config['good'][0]}–{config['good'][1]} {config['unit']}</div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:1.5rem;font-weight:700;">{value:.1f} <span style="font-size:0.8rem;font-weight:400;color:#7f8c8d;">{config['unit']}</span></div>
                <div><span class="status-dot {dot}"></span> <span style="font-size:0.7rem;font-weight:500;">{status_text}</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="card-title">24-Hour Trends (Discrete Steps)</div>', unsafe_allow_html=True)
    recent_data = get_recent_data(st.session_state.data, hours=24)
    
    fig = px.line(recent_data, x='timestamp', y='temperature',
                  title='Temperature Trend', labels={'value': '°C', 'timestamp': ''})
    fig.update_traces(line=dict(color='#e74c3c', width=2), line_shape='hv')
    fig.update_layout(height=150, margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    fig = px.line(recent_data, x='timestamp', y='humidity',
                  title='Humidity Trend', labels={'value': '%', 'timestamp': ''})
    fig.update_traces(line=dict(color='#2980b9', width=2), line_shape='hv')
    fig.update_layout(height=150, margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

elif page == "Actuators":
    st.markdown('<div class="card-title">Control Status</div>', unsafe_allow_html=True)
    
    for name, state in actuator_states.items():
        status_class = "actuator-on" if state == 'ON' else "actuator-off"
        status_text = "ACTIVE" if state == 'ON' else "STANDBY"
        icon = "🟢" if state == 'ON' else "⚪"
        st.markdown(f"""
        <div class="card" style="display:flex;justify-content:space-between;align-items:center;">
            <div style="font-weight:600;">{name}</div>
            <div><span class="actuator-badge {status_class}">{icon} {status_text}</span></div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="card-title">Manual Control</div>', unsafe_allow_html=True)
    st.caption("Simulated manual override (for demonstration)")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💡 Toggle Lighting"):
            log_action("Manual: Lighting toggled")
            st.toast("Lighting toggled (simulated)")
    with col2:
        if st.button("💨 Toggle Ventilation"):
            log_action("Manual: Ventilation toggled")
            st.toast("Ventilation toggled (simulated)")
    
    st.markdown('<div class="card-title">Current Thresholds</div>', unsafe_allow_html=True)
    t = st.session_state.thresholds
    st.markdown(f"""
    <div class="card">
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;font-size:0.8rem;">
            <div><strong>Temperature:</strong> {t['temp_min']}–{t['temp_max']}°C</div>
            <div><strong>Humidity:</strong> {t['humidity_min']}–{t['humidity_max']}%</div>
            <div><strong>Ammonia:</strong> < {t['ammonia_max']} ppm</div>
            <div><strong>CO2:</strong> < {t['co2_max']} ppm</div>
            <div><strong>Water:</strong> > {t['water_min']}%</div>
            <div><strong>Feed:</strong> > {t['feed_min']}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif page == "Alerts":
    st.markdown('<div class="card-title">Notifications</div>', unsafe_allow_html=True)
    
    alerts = []
    t = st.session_state.thresholds
    if latest['temperature'] > t['temp_max']:
        alerts.append("🔥 High Temperature")
    if latest['humidity'] > t['humidity_max']:
        alerts.append("💧 High Humidity")
    if latest['ammonia'] > t['ammonia_max']:
        alerts.append("☣️ Dangerous Ammonia")
    if latest['water_level'] < t['water_min']:
        alerts.append("💦 Low Water")
    if latest['feed_level'] < t['feed_min']:
        alerts.append("🌾 Low Feed")
    if prediction['status'] == 'Disease Risk':
        alerts.append("🚨 Disease Risk Detected")
    
    if alerts:
        st.markdown("#### Active Alerts")
        for alert in alerts:
            st.error(f"⚠️ {alert}")
    else:
        st.success(" No active alerts")
    
    st.markdown("---")
    st.markdown("#### Recent Notifications")
    
    if len(st.session_state.notifications_sent) == 0:
        st.info("No notifications sent yet.")
    else:
        for notif in st.session_state.notifications_sent[:5]:
            st.markdown(f"""
            <div class="notif-item">
                <div><strong>📧 {notif['recipient_type']}:</strong> {notif['recipient_name']}</div>
                <div style="font-size:0.7rem;color:#7f8c8d;">{notif['subject'][:50]}...</div>
                <div class="time">{notif['timestamp']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("#### System Action Log")
    if len(st.session_state.action_log) == 0:
        st.info("No actions logged yet.")
    else:
        for entry in st.session_state.action_log[:10]:
            st.markdown(f'<div class="log-entry" style="padding:0.3rem 0;border-bottom:1px solid #ecf0f1;font-size:0.8rem;">{entry}</div>', unsafe_allow_html=True)

elif page == "Settings":
    st.markdown('<div class="card-title">Farm Settings</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown("#### Farm Profile")
        farm_name = st.text_input("Farm Name", value=st.session_state.farm_name)
        bird_count = st.number_input("Bird Count", min_value=0, max_value=100000, value=st.session_state.bird_count, step=100)
        bird_age = st.selectbox("Bird Age", ["Day-old", "1 week", "2 weeks", "3 weeks", "4 weeks", "5 weeks", "6 weeks+"],
                                index=["Day-old", "1 week", "2 weeks", "3 weeks", "4 weeks", "5 weeks", "6 weeks+"].index(st.session_state.bird_age))
        bird_type = st.selectbox("Bird Type", ["Broiler", "Layer", "Dual Purpose"],
                                  index=["Broiler", "Layer", "Dual Purpose"].index(st.session_state.bird_type))
        
        if st.button("💾 Save Profile"):
            st.session_state.farm_name = farm_name
            st.session_state.bird_count = bird_count
            st.session_state.bird_age = bird_age
            st.session_state.bird_type = bird_type
            log_action("Farm profile updated")
            st.success("Profile saved!")
            st.rerun()
        
        st.markdown("---")
        st.markdown("#### Notification Contacts")
        
        st.markdown("**Farmer**")
        farmer_name = st.text_input("Farmer Name", value=st.session_state.farmer_name, key="farmer_name_input")
        farmer_email = st.text_input("Email", value=st.session_state.farmer_email, key="farmer_email_input")
        farmer_phone = st.text_input("Phone", value=st.session_state.farmer_phone, key="farmer_phone_input")
        
        st.markdown("**Veterinarian**")
        vet_name = st.text_input("Veterinarian Name", value=st.session_state.vet_name, key="vet_name_input")
        vet_email = st.text_input("Email", value=st.session_state.vet_email, key="vet_email_input")
        vet_phone = st.text_input("Phone", value=st.session_state.vet_phone, key="vet_phone_input")
        
        if st.button("💾 Save Contacts"):
            st.session_state.farmer_name = farmer_name
            st.session_state.farmer_email = farmer_email
            st.session_state.farmer_phone = farmer_phone
            st.session_state.vet_name = vet_name
            st.session_state.vet_email = vet_email
            st.session_state.vet_phone = vet_phone
            log_action("Contacts updated")
            st.success("Profiles saved!")
            st.rerun()
        
        st.markdown("---")
        st.markdown("#### Data")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Export Data"):
                csv = st.session_state.data.to_csv(index=False)
                st.download_button(label="Download CSV", data=csv, file_name=f"sensor_data_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")
        with col2:
            if st.button("🔄 Refresh Data"):
                st.session_state.data = st.session_state.generator.generate_data(days=30)
                st.session_state.latest_reading = get_latest_reading(st.session_state.data)
                log_action("Manual data refresh")
                st.rerun()

# ==================== BOTTOM NAVIGATION ====================
st.markdown("---")
render_nav()

# ==================== FOOTER ====================
st.markdown("""
<div style="text-align:center;font-size:0.6rem;color:#95a5a6;padding:0.5rem 0;">
    Smart Poultry Monitoring System v2.0 | AI-Powered
</div>
""", unsafe_allow_html=True)