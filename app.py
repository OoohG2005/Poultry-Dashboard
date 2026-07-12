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

# Page Configuration
st.set_page_config(
    page_title="Smart Poultry Monitoring System",
    page_icon="🐔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Professional, Clean Design
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        color: #1a3c5e;
        text-align: center;
        padding: 1rem 0;
        font-weight: 700;
        letter-spacing: 1px;
    }
    .sub-header {
        font-size: 1rem;
        color: #6B7B8D;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }
    .status-healthy {
        background: linear-gradient(135deg, #27ae60, #2ecc71);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 2px 8px rgba(39,174,96,0.3);
    }
    .status-warning {
        background: linear-gradient(135deg, #f39c12, #f1c40f);
        padding: 1.2rem;
        border-radius: 12px;
        color: #1a1a2e;
        text-align: center;
        box-shadow: 0 2px 8px rgba(243,156,18,0.3);
    }
    .status-danger {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 2px 8px rgba(231,76,60,0.3);
    }
    .actuator-on {
        background: #27ae60;
        color: white;
        padding: 0.6rem;
        border-radius: 8px;
        text-align: center;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .actuator-off {
        background: #7f8c8d;
        color: white;
        padding: 0.6rem;
        border-radius: 8px;
        text-align: center;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .log-entry {
        padding: 0.4rem 0;
        border-bottom: 1px solid #ecf0f1;
        font-size: 0.85rem;
        color: #2c3e50;
    }
    .sensor-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #ecf0f1;
        transition: all 0.2s;
        height: 100%;
    }
    .sensor-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-color: #bdc3c7;
    }
    .sensor-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2c3e50;
        margin: 0.2rem 0;
    }
    .sensor-label {
        font-size: 0.85rem;
        color: #7f8c8d;
        font-weight: 500;
    }
    .sensor-unit {
        font-size: 0.75rem;
        color: #95a5a6;
    }
    .sensor-status-good { color: #27ae60; }
    .sensor-status-warning { color: #f39c12; }
    .sensor-status-danger { color: #e74c3c; }
    .footer {
        text-align: center;
        color: #95a5a6;
        font-size: 0.8rem;
        padding: 1rem 0;
        border-top: 1px solid #ecf0f1;
        margin-top: 2rem;
    }
    .contact-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 0.8rem;
        border-left: 3px solid #2980b9;
        margin: 0.3rem 0;
    }
    .notification-sent {
        background: #d4edda;
        border-radius: 4px;
        padding: 0.3rem 0.6rem;
        color: #155724;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'generator' not in st.session_state:
    st.session_state.generator = PoultrySensorData()

if 'data' not in st.session_state:
    with st.spinner("Loading sensor data from Google Sheets..."):
        st.session_state.data = st.session_state.generator.generate_data(days=30)

if 'predictor' not in st.session_state:
    st.session_state.predictor = PoultryDiseasePredictor()
    with st.spinner("Training AI model..."):
        st.session_state.predictor.train(st.session_state.data)

if 'latest_reading' not in st.session_state:
    st.session_state.latest_reading = get_latest_reading(st.session_state.data)

if 'action_log' not in st.session_state:
    st.session_state.action_log = []

if 'data_logged' not in st.session_state:
    st.session_state.data_logged = 0

if 'notifications_sent' not in st.session_state:
    st.session_state.notifications_sent = []

# Farm Profile
if 'farm_name' not in st.session_state:
    st.session_state.farm_name = "Green Valley Poultry Farm"
if 'bird_count' not in st.session_state:
    st.session_state.bird_count = 2000
if 'bird_age' not in st.session_state:
    st.session_state.bird_age = "4 weeks"
if 'bird_type' not in st.session_state:
    st.session_state.bird_type = "Broiler"

# Notification Contacts
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

def get_actuator_states(latest, thresholds):
    """Determine actuator states based on sensor readings and thresholds"""
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
    """Simulate sending notification and log it"""
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
    """Determine sensor status: good, warning, danger"""
    if value < good_range[0] or value > good_range[1]:
        return 'danger'
    elif value < good_range[0] + (good_range[1] - good_range[0]) * 0.15 or value > good_range[1] - (good_range[1] - good_range[0]) * 0.15:
        return 'warning'
    else:
        return 'good'

# ================== SIDEBAR ==================
with st.sidebar:
    st.markdown("### Smart Poultry")
    st.markdown("**AI Monitoring System**")
    st.markdown("---")
    
    # Farm Profile
    st.markdown("#### Farm Profile")
    farm_name_input = st.text_input("Farm Name", value=st.session_state.farm_name)
    bird_count_input = st.number_input("Bird Count", min_value=0, max_value=100000, value=st.session_state.bird_count, step=100)
    bird_age_input = st.selectbox("Bird Age", ["Day-old", "1 week", "2 weeks", "3 weeks", "4 weeks", "5 weeks", "6 weeks+"], 
                                   index=["Day-old", "1 week", "2 weeks", "3 weeks", "4 weeks", "5 weeks", "6 weeks+"].index(st.session_state.bird_age))
    bird_type_input = st.selectbox("Bird Type", ["Broiler", "Layer", "Dual Purpose"], 
                                    index=["Broiler", "Layer", "Dual Purpose"].index(st.session_state.bird_type))
    
    if farm_name_input != st.session_state.farm_name or bird_count_input != st.session_state.bird_count or bird_age_input != st.session_state.bird_age or bird_type_input != st.session_state.bird_type:
        st.session_state.farm_name = farm_name_input
        st.session_state.bird_count = bird_count_input
        st.session_state.bird_age = bird_age_input
        st.session_state.bird_type = bird_type_input
        st.rerun()
    
    st.info(f"{bird_count_input:,} birds | Age: {bird_age_input} | Type: {bird_type_input}")
    
    st.markdown("---")
    
    # ================== NOTIFICATION CONTACTS ==================
    st.markdown("#### 📧 Notification Contacts")
    st.caption("These contacts will receive alerts")
    
    with st.expander("Farmer Contact", expanded=True):
        farmer_name = st.text_input("Farmer Name", value=st.session_state.farmer_name)
        farmer_email = st.text_input("Farmer Email", value=st.session_state.farmer_email)
        farmer_phone = st.text_input("Farmer Phone", value=st.session_state.farmer_phone)
        farmer_sms = st.checkbox("Send SMS to Farmer", value=True)
        farmer_email_check = st.checkbox("Send Email to Farmer", value=True)
    
    with st.expander("Veterinarian Contact", expanded=True):
        vet_name = st.text_input("Veterinarian Name", value=st.session_state.vet_name)
        vet_email = st.text_input("Veterinarian Email", value=st.session_state.vet_email)
        vet_phone = st.text_input("Veterinarian Phone", value=st.session_state.vet_phone)
        vet_sms = st.checkbox("Send SMS to Vet", value=True)
        vet_email_check = st.checkbox("Send Email to Vet", value=True)
    
    # Save contacts button
    if st.button("💾 Save Contacts", use_container_width=True):
        st.session_state.farmer_name = farmer_name
        st.session_state.farmer_email = farmer_email
        st.session_state.farmer_phone = farmer_phone
        st.session_state.vet_name = vet_name
        st.session_state.vet_email = vet_email
        st.session_state.vet_phone = vet_phone
        log_action("Contacts updated successfully")
        st.success("✅ Contacts saved!")
        st.rerun()
    
    st.markdown("---")
    
    # Controls
    st.markdown("#### Controls")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh", use_container_width=True):
            st.session_state.data = st.session_state.generator.generate_data(days=30)
            st.session_state.latest_reading = get_latest_reading(st.session_state.data)
            log_action("Manual data refresh")
            st.rerun()
    with col2:
        if st.button("🤖 Retrain AI", use_container_width=True):
            with st.spinner("Retraining..."):
                st.session_state.predictor.train(st.session_state.data)
                log_action("AI model retrained")
                st.rerun()
    
    st.markdown("---")
    
    # Data Range
    st.markdown("#### Data Range")
    days = st.slider("View past days:", 1, 30, 7)
    
    st.markdown("---")
    
    # System Status
    st.markdown("#### System Status")
    st.success("✅ Online")
    st.info(f"Records: {len(st.session_state.data)}")
    st.info(f"Last Update: {datetime.now().strftime('%H:%M:%S')}")
    
    st.markdown("---")
    
    # AI Model Info
    st.markdown("#### AI Model")
    model_info = st.session_state.predictor.get_model_info()
    if model_info and model_info.get('accuracy'):
        st.metric("Accuracy", f"{model_info['accuracy']:.1%}")
    
    st.markdown("---")
    
    # Export
    if st.button("📥 Export Data", use_container_width=True):
        csv = st.session_state.data.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"sensor_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

# ================== MAIN CONTENT ==================
st.markdown('<h1 class="main-header">Smart Poultry House Monitoring System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Environmental Monitoring & Automated Control</p>', unsafe_allow_html=True)

# Get data
latest = st.session_state.latest_reading
prediction = st.session_state.predictor.predict(latest)
actuator_states = get_actuator_states(latest, st.session_state.thresholds)

# ================== FARM SNAPSHOT ==================
st.markdown("#### Farm Snapshot")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Farm", st.session_state.farm_name)
with col2:
    st.metric("Birds", f"{st.session_state.bird_count:,}")
with col3:
    st.metric("Age", st.session_state.bird_age)
with col4:
    st.metric("Type", st.session_state.bird_type)
with col5:
    status_icon = "✅" if prediction['status'] == 'Healthy' else ("⚠️" if prediction['status'] == 'Warning' else "🚨")
    st.metric("Status", f"{status_icon} {prediction['status']}")

st.markdown("---")

# ================== AI STATUS & HEALTH SCORE ==================
col_status, col_gauge = st.columns([1.2, 1])

with col_status:
    st.markdown("#### Current Status")
    status_class = {
        'Healthy': 'status-healthy',
        'Warning': 'status-warning',
        'Disease Risk': 'status-danger'
    }[prediction['status']]
    status_icon = {
        'Healthy': '✅',
        'Warning': '⚠️',
        'Disease Risk': '🚨'
    }[prediction['status']]
    st.markdown(f"""
    <div class="{status_class}">
        <h2 style="margin:0;font-size:1.8rem;">{status_icon} {prediction['status']}</h2>
        <p style="margin:0;font-size:1rem;">Confidence: {prediction['confidence']:.1f}%</p>
        <p style="margin:0;font-size:0.85rem;opacity:0.9;">{datetime.now().strftime('%H:%M:%S')}</p>
    </div>
    """, unsafe_allow_html=True)

with col_gauge:
    health_score = 100 - (prediction['probabilities']['Disease Risk'] / 1)
    health_score = max(0, min(100, health_score))
    
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=health_score,
        title={'text': "Health Score", 'font': {'size': 14}},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 40], 'color': "#27ae60"},
                {'range': [40, 70], 'color': "#f39c12"},
                {'range': [70, 100], 'color': "#e74c3c"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    fig_gauge.update_layout(height=200, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_gauge, use_container_width=True)

st.markdown("---")

# ================== CONTACT SUMMARY ==================
st.markdown("#### Notification Contacts")
col_farmer, col_vet = st.columns(2)

with col_farmer:
    st.markdown(f"""
    <div class="contact-card">
        <strong>👨‍🌾 Farmer</strong><br>
        Name: {st.session_state.farmer_name}<br>
        Email: {st.session_state.farmer_email}<br>
        Phone: {st.session_state.farmer_phone}
    </div>
    """, unsafe_allow_html=True)

with col_vet:
    st.markdown(f"""
    <div class="contact-card">
        <strong>👨‍⚕️ Veterinarian</strong><br>
        Name: {st.session_state.vet_name}<br>
        Email: {st.session_state.vet_email}<br>
        Phone: {st.session_state.vet_phone}
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ================== SENSOR READINGS ==================
st.markdown("#### Environmental Readings")

sensor_configs = [
    {'key': 'temperature', 'label': 'Temperature', 'unit': '°C', 'min': 15, 'max': 40, 'good': (20, 30)},
    {'key': 'humidity', 'label': 'Humidity', 'unit': '%', 'min': 30, 'max': 95, 'good': (60, 80)},
    {'key': 'ammonia', 'label': 'Ammonia', 'unit': 'ppm', 'min': 0, 'max': 30, 'good': (0, 10)},
    {'key': 'co2', 'label': 'CO2', 'unit': 'ppm', 'min': 300, 'max': 2000, 'good': (300, 1000)},
    {'key': 'water_level', 'label': 'Water Level', 'unit': '%', 'min': 0, 'max': 100, 'good': (30, 100)},
    {'key': 'feed_level', 'label': 'Feed Level', 'unit': '%', 'min': 0, 'max': 100, 'good': (25, 100)},
    {'key': 'light_intensity', 'label': 'Light Intensity', 'unit': 'lux', 'min': 0, 'max': 1000, 'good': (200, 800)},
    {'key': 'air_quality', 'label': 'Air Quality', 'unit': '', 'min': 0, 'max': 500, 'good': (0, 300)}
]

for i in range(0, len(sensor_configs), 4):
    cols = st.columns(4)
    for j in range(4):
        idx = i + j
        if idx < len(sensor_configs):
            config = sensor_configs[idx]
            value = latest[config['key']]
            status = get_sensor_status(value, config['good'])
            
            with cols[j]:
                status_text = "Good" if status == 'good' else ("Warning" if status == 'warning' else "Danger")
                status_color = "sensor-status-good" if status == 'good' else ("sensor-status-warning" if status == 'warning' else "sensor-status-danger")
                
                st.markdown(f"""
                <div class="sensor-card">
                    <div class="sensor-label">{config['label']}</div>
                    <div class="sensor-value">{value:.1f} <span class="sensor-unit">{config['unit']}</span></div>
                    <div class="{status_color}">● {status_text}</div>
                    <div style="font-size:0.7rem;color:#95a5a6;margin-top:0.3rem;">
                        Target: {config['good'][0]}–{config['good'][1]} {config['unit']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

st.markdown("---")

# ================== ACTUATOR STATUS ==================
st.markdown("#### Automated Control Status")

actuator_names = list(actuator_states.keys())
cols = st.columns(len(actuator_names))
for i, (name, state) in enumerate(actuator_states.items()):
    col = cols[i % len(actuator_names)]
    with col:
        if state == 'ON':
            st.markdown(f'<div class="actuator-on">{name}: ACTIVE</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="actuator-off">{name}: STANDBY</div>', unsafe_allow_html=True)

st.markdown("---")

# ================== AI PREDICTION DETAILS ==================
st.markdown("#### AI Risk Analysis")

col_prob, col_risk = st.columns([1.2, 1])

with col_prob:
    prob_data = pd.DataFrame({
        'Status': ['Healthy', 'Warning', 'Disease Risk'],
        'Probability': [
            prediction['probabilities']['Healthy'],
            prediction['probabilities']['Warning'],
            prediction['probabilities']['Disease Risk']
        ]
    })
    colors = ['#27ae60', '#f39c12', '#e74c3c']
    fig = px.bar(prob_data, x='Status', y='Probability', 
                 color='Status',
                 color_discrete_sequence=colors,
                 text='Probability')
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(height=200, showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)

with col_risk:
    st.markdown("#### Key Risk Factors")
    importance = st.session_state.predictor.feature_importance
    if importance:
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:4]
        for feature, score in sorted_importance:
            label = feature.replace('_', ' ').title()
            st.progress(score, text=f"{label}: {score:.1%}")

st.markdown("---")

# ================== TRENDS CHARTS ==================
st.markdown("#### Environmental Trends")

recent_data = get_recent_data(st.session_state.data, hours=days*24)

tab1, tab2, tab3 = st.tabs(["All Parameters", "Temperature & Humidity", "Gas & Air Quality"])

with tab1:
    fig = px.line(recent_data, x='timestamp', 
                  y=['temperature', 'humidity', 'ammonia', 'co2', 'air_quality'],
                  title="All Environmental Parameters",
                  labels={'value': 'Reading', 'variable': 'Parameter'})
    fig.update_layout(height=350, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recent_data['timestamp'], 
        y=recent_data['temperature'],
        name='Temperature °C', 
        line=dict(color='#e74c3c', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=recent_data['timestamp'], 
        y=recent_data['humidity'],
        name='Humidity %', 
        line=dict(color='#2980b9', width=2),
        yaxis='y2'
    ))
    fig.add_hline(y=st.session_state.thresholds['temp_max'], line_dash="dash", line_color="#e74c3c", opacity=0.5)
    fig.add_hline(y=st.session_state.thresholds['humidity_max'], line_dash="dash", line_color="#f39c12", opacity=0.5)
    fig.update_layout(
        title='Temperature and Humidity Trends',
        height=350,
        yaxis=dict(title='Temperature (°C)', side='left'),
        yaxis2=dict(title='Humidity (%)', overlaying='y', side='right'),
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    fig = px.area(recent_data, x='timestamp', y=['ammonia', 'co2', 'air_quality'],
                  title="Gas Levels & Air Quality",
                  labels={'value': 'Level', 'variable': 'Parameter'})
    fig.add_hline(y=st.session_state.thresholds['ammonia_max'], line_dash="dash", line_color="#e74c3c", opacity=0.5)
    fig.add_hline(y=st.session_state.thresholds['co2_max'], line_dash="dash", line_color="#f39c12", opacity=0.5)
    fig.update_layout(height=350, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ================== ALERTS & NOTIFICATIONS ==================
col_alerts, col_log = st.columns([1.2, 1])

with col_alerts:
    st.markdown("#### Alerts & Recommendations")
    
    alerts = []
    recommendations = []
    should_notify = False
    notification_message = ""
    
    t = st.session_state.thresholds
    
    # Check parameters
    if latest['temperature'] > t['temp_max']:
        alerts.append("High Temperature - Above threshold")
        recommendations.append("Cooling fans activated automatically")
        should_notify = True
        notification_message = f"Temperature is {latest['temperature']:.1f}°C (threshold: {t['temp_max']}°C)"
    elif latest['temperature'] < t['temp_min']:
        alerts.append("Low Temperature - Below threshold")
        recommendations.append("Heaters activated automatically")
        should_notify = True
        notification_message = f"Temperature is {latest['temperature']:.1f}°C (threshold: {t['temp_min']}°C)"
    
    if latest['humidity'] > t['humidity_max']:
        alerts.append("High Humidity - Above threshold")
        recommendations.append("Ventilation system activated")
        should_notify = True
        notification_message += f" | Humidity: {latest['humidity']:.1f}%"
    elif latest['humidity'] < t['humidity_min']:
        alerts.append("Low Humidity - Below threshold")
        recommendations.append("Consider adding moisture")
    
    if latest['ammonia'] > t['ammonia_max']:
        alerts.append("Dangerous Ammonia Level")
        recommendations.append("Immediate ventilation and litter management")
        should_notify = True
        notification_message = f"Ammonia: {latest['ammonia']:.1f} ppm (threshold: {t['ammonia_max']} ppm)"
    
    if latest['co2'] > t['co2_max']:
        alerts.append("High CO2 Level")
        recommendations.append("Increase fresh air intake")
        should_notify = True
        notification_message = f"CO2: {latest['co2']:.0f} ppm (threshold: {t['co2_max']} ppm)"
    
    if latest['water_level'] < t['water_min']:
        alerts.append("Low Water Level")
        recommendations.append("Farmer action: Refill water supply")
        should_notify = True
        notification_message = f"Water Level: {latest['water_level']:.1f}% (threshold: {t['water_min']}%)"
    
    if latest['feed_level'] < t['feed_min']:
        alerts.append("Low Feed Level")
        recommendations.append("Farmer action: Refill feeders")
        should_notify = True
        notification_message = f"Feed Level: {latest['feed_level']:.1f}% (threshold: {t['feed_min']}%)"
    
    if prediction['status'] == 'Disease Risk':
        alerts.append("🚨 Disease Risk Detected")
        recommendations.append("Notify veterinarian immediately")
        recommendations.append(f"Veterinarian: {st.session_state.vet_name} ({st.session_state.vet_phone})")
        should_notify = True
        notification_message = f"Disease Risk detected! Confidence: {prediction['confidence']:.1f}%"
    
    # Send notifications if needed
    if should_notify and notification_message:
        subject = f"ALERT: {st.session_state.farm_name} - {alerts[0] if alerts else 'Condition Alert'}"
        message = f"""
Farm: {st.session_state.farm_name}
Birds: {st.session_state.bird_count} ({st.session_state.bird_type} - {st.session_state.bird_age})
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Status: {prediction['status']}

Alerts:
{chr(10).join('- ' + a for a in alerts)}

Recommendations:
{chr(10).join('- ' + r for r in recommendations)}

Current Readings:
- Temperature: {latest['temperature']:.1f}°C
- Humidity: {latest['humidity']:.1f}%
- Ammonia: {latest['ammonia']:.1f} ppm
- CO2: {latest['co2']:.0f} ppm
- Water Level: {latest['water_level']:.1f}%
- Feed Level: {latest['feed_level']:.1f}%

This is an automated alert from the Smart Poultry Monitoring System.
"""
        
        # Send to farmer
        send_notification("Farmer", st.session_state.farmer_name, 
                         st.session_state.farmer_email, st.session_state.farmer_phone,
                         subject, message)
        
        # Send to vet if disease risk
        if prediction['status'] == 'Disease Risk':
            send_notification("Veterinarian", st.session_state.vet_name,
                             st.session_state.vet_email, st.session_state.vet_phone,
                             f"URGENT: {subject}", message)
        elif "Dangerous" in notification_message or "Ammonia" in notification_message:
            send_notification("Veterinarian", st.session_state.vet_name,
                             st.session_state.vet_email, st.session_state.vet_phone,
                             f"ALERT: {subject}", message)
    
    # Display alerts
    if len(alerts) == 0:
        st.success("✅ All conditions are within normal range. System is maintaining optimal environment.")
    else:
        for alert in alerts:
            if "Disease" in alert or "Dangerous" in alert:
                st.error(f"🚨 {alert}")
            elif "Low" in alert:
                st.warning(f"⚠️ {alert}")
            else:
                st.info(f"ℹ️ {alert}")
        
        if recommendations:
            st.markdown("**Recommended Actions:**")
            for rec in set(recommendations):
                if "Farmer" in rec:
                    st.warning(rec)
                elif "Veterinarian" in rec:
                    st.info(rec)
                else:
                    st.success(f"✓ {rec}")

with col_log:
    st.markdown("#### System Activity")
    
    st.markdown("**Action Log**")
    if len(st.session_state.action_log) == 0:
        st.info("No actions logged yet.")
    else:
        for entry in st.session_state.action_log[:8]:
            st.markdown(f'<div class="log-entry">{entry}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("**Recent Notifications**")
    if len(st.session_state.notifications_sent) == 0:
        st.info("No notifications sent yet.")
    else:
        for notif in st.session_state.notifications_sent[:5]:
            st.markdown(f"""
            <div class="log-entry">
                📧 {notif['timestamp']} - {notif['recipient_type']}: {notif['recipient_name']}
                <br><span style="font-size:0.75rem;color:#6c757d;">{notif['subject'][:40]}...</span>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("#### Data Logging")
    st.info(f"Data points logged: {st.session_state.data_logged}")
    if st.button("📝 Simulate Data Log"):
        st.session_state.data_logged += 1
        log_action(f"Data point logged (ID: {st.session_state.data_logged})")
        st.rerun()

# ================== FOOTER ==================
st.markdown("""
<div class="footer">
    Smart Poultry House Monitoring System v2.0 | AI-Powered Environmental Control<br>
    Data updated in real-time | Last update: {timestamp} | Total Records: {records}
</div>
""".format(
    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    records=len(st.session_state.data)
), unsafe_allow_html=True)