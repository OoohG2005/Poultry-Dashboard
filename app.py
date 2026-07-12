import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import json
import re
import numpy as np

# Import custom modules
from data_generator import PoultrySensorData, get_latest_reading, get_recent_data
from model import PoultryDiseasePredictor

# Page Configuration
st.set_page_config(
    page_title="🐔 Smart Poultry Monitoring System",
    page_icon="🐔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E4057;
        text-align: center;
        padding: 1rem 0;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6B7B8D;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-healthy {
        background: linear-gradient(135deg, #28a745, #20c997);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .status-warning {
        background: linear-gradient(135deg, #ffc107, #fd7e14);
        padding: 1.5rem;
        border-radius: 15px;
        color: black;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .status-danger {
        background: linear-gradient(135deg, #dc3545, #e74c3c);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .actuator-on {
        background-color: #28a745;
        color: white;
        padding: 0.5rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
    .actuator-off {
        background-color: #6c757d;
        color: white;
        padding: 0.5rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
    .log-entry {
        padding: 0.3rem;
        border-bottom: 1px solid #eee;
        font-size: 0.9rem;
    }
    /* Sensor card styling */
    .sensor-card {
        background: white;
        border-radius: 12px;
        padding: 0.8rem;
        text-align: center;
        border: 1px solid #e8e8e8;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        transition: transform 0.2s;
        height: 100%;
    }
    .sensor-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }
    .sensor-value {
        font-size: 1.8rem;
        font-weight: bold;
        margin: 0.2rem 0;
    }
    .sensor-label {
        font-size: 0.85rem;
        color: #6B7B8D;
        font-weight: 500;
    }
    .sensor-unit {
        font-size: 0.8rem;
        color: #9AA6B2;
    }
    .sensor-status-ok {
        color: #28a745;
    }
    .sensor-status-warning {
        color: #ffc107;
    }
    .sensor-status-danger {
        color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'generator' not in st.session_state:
    st.session_state.generator = PoultrySensorData()

if 'data' not in st.session_state:
    with st.spinner("🔄 Generating sensor data..."):
        st.session_state.data = st.session_state.generator.generate_data(days=30)

if 'predictor' not in st.session_state:
    st.session_state.predictor = PoultryDiseasePredictor()
    with st.spinner("🤖 Training AI model..."):
        st.session_state.predictor.train(st.session_state.data)

if 'latest_reading' not in st.session_state:
    st.session_state.latest_reading = get_latest_reading(st.session_state.data)

if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

if 'action_log' not in st.session_state:
    st.session_state.action_log = []

if 'data_logged' not in st.session_state:
    st.session_state.data_logged = 0

# Farm Profile
if 'farm_name' not in st.session_state:
    st.session_state.farm_name = "My Poultry Farm"
if 'bird_count' not in st.session_state:
    st.session_state.bird_count = 1500

# Function to determine actuator states
def get_actuator_states(latest):
    states = {}
    states['Cooling Fan'] = 'ON' if latest['temperature'] > 30 else 'OFF'
    states['Heater'] = 'ON' if latest['temperature'] < 20 else 'OFF'
    states['Ventilation'] = 'ON' if (latest['humidity'] > 85 or latest['ammonia'] > 10 or latest['co2'] > 1000) else 'OFF'
    states['Water Pump'] = 'ON' if latest['water_level'] < 30 else 'OFF'
    states['Alarm'] = 'ON' if latest.get('risk_label', 0) == 2 else 'OFF'
    states['Lighting'] = 'ON' if latest['light_intensity'] < 200 else 'OFF'
    return states

def log_action(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.action_log.insert(0, f"[{timestamp}] {message}")
    if len(st.session_state.action_log) > 20:
        st.session_state.action_log.pop()

def log_to_google_sheets(latest):
    st.session_state.data_logged += 1
    log_action(f"Data point logged to Google Sheets (ID: {st.session_state.data_logged})")

# Function to create sensor gauge
def create_sensor_gauge(value, min_val, max_val, title, unit, good_range):
    """Create a gauge chart for a sensor reading"""
    # Determine color based on value position
    if value < good_range[0] or value > good_range[1]:
        color = "#dc3545"  # Red - danger
    elif value < good_range[0] + (good_range[1] - good_range[0]) * 0.2 or value > good_range[1] - (good_range[1] - good_range[0]) * 0.2:
        color = "#ffc107"  # Yellow - warning
    else:
        color = "#28a745"  # Green - good
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 14}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 0,
            'steps': [
                {'range': [min_val, good_range[0]], 'color': 'rgba(220, 53, 69, 0.2)'},
                {'range': [good_range[0], good_range[1]], 'color': 'rgba(40, 167, 69, 0.15)'},
                {'range': [good_range[1], max_val], 'color': 'rgba(220, 53, 69, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 2},
                'thickness': 0.75,
                'value': good_range[1]
            }
        }
    ))
    fig.update_layout(
        height=140,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/590/590167.png", width=80)
    st.title("🐔 Smart Poultry")
    st.subheader("AI Monitoring System v2.0")
    
    st.markdown("---")
    
    st.markdown("### 🏠 Farm Profile")
    farm_name_input = st.text_input("Farm Name", value=st.session_state.farm_name)
    bird_count_input = st.number_input("Bird Count", min_value=0, max_value=100000, value=st.session_state.bird_count, step=100)
    if farm_name_input != st.session_state.farm_name or bird_count_input != st.session_state.bird_count:
        st.session_state.farm_name = farm_name_input
        st.session_state.bird_count = bird_count_input
    st.info(f"📊 {st.session_state.bird_count:,} birds on {st.session_state.farm_name}")
    
    st.markdown("---")
    
    st.markdown("### 🎮 Controls")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh", use_container_width=True):
            with st.spinner("Refreshing data..."):
                st.session_state.data = st.session_state.generator.generate_data(days=30)
                st.session_state.latest_reading = get_latest_reading(st.session_state.data)
                log_action("Data refreshed manually")
                st.rerun()
    
    with col2:
        if st.button("🤖 Retrain AI", use_container_width=True):
            with st.spinner("Retraining model..."):
                st.session_state.predictor.train(st.session_state.data)
                log_action("AI model retrained")
                st.rerun()
    
    st.markdown("---")
    
    st.markdown("### 📊 Data Range")
    days = st.slider("Select days to view:", 1, 30, 7, key="days_slider")
    
    st.markdown("---")
    
    st.markdown("### 🧪 Test Conditions")
    st.caption("Click to force specific conditions")
    
    test_col1, test_col2, test_col3 = st.columns(3)
    
    with test_col1:
        if st.button("✅ Healthy", use_container_width=True):
            with st.spinner("Setting Healthy conditions..."):
                df = st.session_state.data.copy()
                df['temperature'] = 24 + np.random.normal(0, 1, len(df))
                df['humidity'] = 60 + np.random.normal(0, 5, len(df))
                df['ammonia'] = 3 + np.random.normal(0, 1, len(df))
                df['co2'] = 500 + np.random.normal(0, 50, len(df))
                df['water_level'] = 80 + np.random.normal(0, 5, len(df))
                df['feed_level'] = 75 + np.random.normal(0, 5, len(df))
                df['light_intensity'] = 500 + np.random.normal(0, 50, len(df))
                df['air_quality'] = 200 + np.random.normal(0, 20, len(df))
                df['risk_label'] = 0
                st.session_state.data = df
                st.session_state.latest_reading = get_latest_reading(df)
                log_action("Test: Healthy conditions applied")
                st.rerun()
    
    with test_col2:
        if st.button("⚠️ Warning", use_container_width=True):
            with st.spinner("Setting Warning conditions..."):
                df = st.session_state.data.copy()
                df['temperature'] = 29 + np.random.normal(0, 1, len(df))
                df['humidity'] = 80 + np.random.normal(0, 3, len(df))
                df['ammonia'] = 8 + np.random.normal(0, 1, len(df))
                df['co2'] = 800 + np.random.normal(0, 50, len(df))
                df['water_level'] = 50 + np.random.normal(0, 5, len(df))
                df['feed_level'] = 45 + np.random.normal(0, 5, len(df))
                df['light_intensity'] = 400 + np.random.normal(0, 50, len(df))
                df['air_quality'] = 300 + np.random.normal(0, 20, len(df))
                df['risk_label'] = 1
                st.session_state.data = df
                st.session_state.latest_reading = get_latest_reading(df)
                log_action("Test: Warning conditions applied")
                st.rerun()
    
    with test_col3:
        if st.button("🚨 Disease", use_container_width=True):
            with st.spinner("Setting Disease Risk conditions..."):
                df = st.session_state.data.copy()
                df['temperature'] = 33 + np.random.normal(0, 1, len(df))
                df['humidity'] = 90 + np.random.normal(0, 3, len(df))
                df['ammonia'] = 15 + np.random.normal(0, 2, len(df))
                df['co2'] = 1200 + np.random.normal(0, 100, len(df))
                df['water_level'] = 30 + np.random.normal(0, 5, len(df))
                df['feed_level'] = 25 + np.random.normal(0, 5, len(df))
                df['light_intensity'] = 300 + np.random.normal(0, 50, len(df))
                df['air_quality'] = 400 + np.random.normal(0, 20, len(df))
                df['risk_label'] = 2
                st.session_state.data = df
                st.session_state.latest_reading = get_latest_reading(df)
                log_action("Test: Disease Risk conditions applied")
                st.rerun()
    
    st.markdown("---")
    
    st.markdown("### ⏱️ Auto Refresh")
    auto_refresh = st.checkbox("Enable Auto Refresh (every 10 seconds)")
    if auto_refresh:
        st.session_state.data = st.session_state.generator.generate_data(days=30)
        st.session_state.latest_reading = get_latest_reading(st.session_state.data)
        log_to_google_sheets(st.session_state.latest_reading)
        time.sleep(10)
        st.rerun()
    
    st.markdown("---")
    
    st.markdown("### 📋 System Status")
    st.success("✅ System Running")
    st.info(f"📡 {len(st.session_state.data)} Records")
    st.info(f"🕒 Updated: {datetime.now().strftime('%H:%M:%S')}")
    st.info(f"📤 Data Logged: {st.session_state.data_logged} entries")
    
    st.markdown("---")
    
    st.markdown("### 🤖 AI Model Info")
    model_info = st.session_state.predictor.get_model_info()
    if model_info and model_info.get('accuracy'):
        st.metric("Model Accuracy", f"{model_info['accuracy']:.1%}")
    
    st.markdown("---")
    if st.button("📥 Export Data", use_container_width=True):
        csv = st.session_state.data.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"poultry_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

# ================== MAIN CONTENT ==================
st.markdown('<h1 class="main-header">🐔 Smart Poultry House Monitoring System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Environmental Monitoring, Automated Control & Data Logging</p>', unsafe_allow_html=True)

# --- HEALTH GAUGE & FARM STATS ---
latest = st.session_state.latest_reading
prediction = st.session_state.predictor.predict(latest)

health_score = 100 - (prediction['probabilities']['Disease Risk'] / 1)
health_score = max(0, min(100, health_score))

col_gauge1, col_gauge2, col_gauge3 = st.columns([1, 1, 1])
with col_gauge1:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=health_score,
        title={'text': "🏥 Farm Health Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 40], 'color': "green"},
                {'range': [40, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        },
        delta={'reference': 80, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}}
    ))
    fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_gauge2:
    st.markdown("#### 📋 Farm Snapshot")
    st.metric("Farm Name", st.session_state.farm_name)
    st.metric("Birds", f"{st.session_state.bird_count:,}")
    st.metric("House Status", "✅ Active")

with col_gauge3:
    st.markdown("#### 📊 Quick Stats")
    risk_label = prediction['status']
    st.metric("Current Risk", risk_label, delta="⚠️" if risk_label != "Healthy" else "✅")
    st.metric("Confidence", f"{prediction['confidence']:.1f}%")
    st.metric("Records", len(st.session_state.data))

st.markdown("---")

# --- IMPROVED SENSOR READINGS WITH GAUGES ---
st.markdown("### 🌡️ Sensor Readings Dashboard")

# Define sensor configurations
sensor_configs = [
    {'key': 'temperature', 'label': '🌡️ Temperature', 'unit': '°C', 'min': 15, 'max': 40, 'good': (20, 30)},
    {'key': 'humidity', 'label': '💧 Humidity', 'unit': '%', 'min': 30, 'max': 95, 'good': (60, 80)},
    {'key': 'ammonia', 'label': '💨 Ammonia', 'unit': 'ppm', 'min': 0, 'max': 30, 'good': (0, 10)},
    {'key': 'co2', 'label': '🫧 CO2', 'unit': 'ppm', 'min': 300, 'max': 2000, 'good': (300, 1000)},
    {'key': 'water_level', 'label': '💦 Water Level', 'unit': '%', 'min': 0, 'max': 100, 'good': (30, 100)},
    {'key': 'feed_level', 'label': '🌾 Feed Level', 'unit': '%', 'min': 0, 'max': 100, 'good': (25, 100)},
    {'key': 'light_intensity', 'label': '💡 Light Intensity', 'unit': 'lux', 'min': 0, 'max': 1000, 'good': (200, 800)},
    {'key': 'air_quality', 'label': '🌬️ Air Quality', 'unit': '', 'min': 0, 'max': 500, 'good': (0, 300)}
]

# Display sensors in a 4-column grid with gauges
for i in range(0, len(sensor_configs), 4):
    cols = st.columns(4)
    for j in range(4):
        idx = i + j
        if idx < len(sensor_configs):
            config = sensor_configs[idx]
            value = latest[config['key']]
            
            with cols[j]:
                # Create gauge for this sensor
                fig = create_sensor_gauge(
                    value=value,
                    min_val=config['min'],
                    max_val=config['max'],
                    title=config['label'],
                    unit=config['unit'],
                    good_range=config['good']
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show value with status indicator
                if value < config['good'][0] or value > config['good'][1]:
                    status_icon = "🔴"
                elif value < config['good'][0] + (config['good'][1] - config['good'][0]) * 0.2 or value > config['good'][1] - (config['good'][1] - config['good'][0]) * 0.2:
                    status_icon = "🟡"
                else:
                    status_icon = "🟢"
                
                st.caption(f"{status_icon} Current: **{value:.1f}** {config['unit']}  |  Ideal: {config['good'][0]}–{config['good'][1]} {config['good'][0] != 0 and ' ' or ''}{config['unit']}")

st.markdown("---")

# Current Status Row (simplified)
col1, col2, col3 = st.columns([1, 1.5, 1])

with col1:
    st.markdown("### 📊 Current Status")
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
        <h2>{status_icon} {prediction['status']}</h2>
        <p style="font-size: 1.1rem;">Confidence: {prediction['confidence']:.1f}%</p>
        <p style="font-size: 0.9rem; opacity: 0.9;">{datetime.now().strftime('%H:%M:%S')}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("### 🤖 AI Prediction")
    prob_data = pd.DataFrame({
        'Status': ['Healthy', 'Warning', 'Disease Risk'],
        'Probability': [
            prediction['probabilities']['Healthy'],
            prediction['probabilities']['Warning'],
            prediction['probabilities']['Disease Risk']
        ]
    })
    colors = ['#28a745', '#ffc107', '#dc3545']
    fig = px.bar(prob_data, x='Status', y='Probability', 
                 color='Status',
                 color_discrete_sequence=colors,
                 title="Risk Probabilities",
                 text='Probability')
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(height=200, showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)

with col3:
    st.markdown("#### 🔑 Key Risk Factors")
    importance = st.session_state.predictor.feature_importance
    if importance:
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:4]
        for feature, score in sorted_importance:
            st.progress(score, text=f"{feature.replace('_', ' ').title()}: {score:.1%}")

# Automated Control Section
st.markdown("---")
st.markdown("### ⚙️ Automated Control Actions")

actuator_states = get_actuator_states(latest)
cols = st.columns(6)
for i, (name, state) in enumerate(actuator_states.items()):
    col = cols[i % 6]
    with col:
        if state == 'ON':
            st.markdown(f'<div class="actuator-on">{name}: 🔴 ON</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="actuator-off">{name}: ⚪ OFF</div>', unsafe_allow_html=True)

# Log automatic actions
if latest['temperature'] > 30 and actuator_states['Cooling Fan'] == 'ON':
    log_action("Cooling Fan turned ON (temperature high)")
elif latest['temperature'] < 20 and actuator_states['Heater'] == 'ON':
    log_action("Heater turned ON (temperature low)")
if latest['humidity'] > 85 and actuator_states['Ventilation'] == 'ON':
    log_action("Ventilation turned ON (high humidity)")
if latest['ammonia'] > 10 and actuator_states['Ventilation'] == 'ON':
    log_action("Ventilation turned ON (high ammonia)")
if latest['co2'] > 1000 and actuator_states['Ventilation'] == 'ON':
    log_action("Ventilation turned ON (high CO2)")
if latest['water_level'] < 30 and actuator_states['Water Pump'] == 'ON':
    log_action("Water Pump turned ON (low water level)")
if latest.get('risk_label', 0) == 2 and actuator_states['Alarm'] == 'ON':
    log_action("⚠️ ALARM ACTIVATED - Disease Risk Detected!")

# Charts Section
st.markdown("---")
st.markdown("### 📈 Environmental Trends")

recent_data = get_recent_data(st.session_state.data, hours=days*24)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 All Parameters", 
    "🌡️ Temperature & Humidity", 
    "💨 Gas Levels", 
    "📋 Risk Analysis"
])

with tab1:
    fig = px.line(recent_data, x='timestamp', 
                  y=['temperature', 'humidity', 'ammonia', 'co2', 'air_quality'],
                  title="All Environmental Parameters",
                  labels={'value': 'Reading', 'variable': 'Parameter'})
    fig.update_layout(height=400, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recent_data['timestamp'], 
        y=recent_data['temperature'],
        name='Temperature (°C)', 
        line=dict(color='red', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=recent_data['timestamp'], 
        y=recent_data['humidity'],
        name='Humidity (%)', 
        line=dict(color='blue', width=2),
        yaxis='y2'
    ))
    fig.add_hline(y=30, line_dash="dash", line_color="red", opacity=0.5)
    fig.add_hline(y=85, line_dash="dash", line_color="orange", opacity=0.5)
    fig.update_layout(
        title='Temperature and Humidity Trends',
        height=400,
        yaxis=dict(title='Temperature (°C)', side='left'),
        yaxis2=dict(title='Humidity (%)', overlaying='y', side='right'),
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    fig = px.area(recent_data, x='timestamp', y=['ammonia', 'co2'],
                  title="Gas Levels (Ammonia & CO2)",
                  labels={'value': 'Level', 'variable': 'Gas'})
    fig.add_hline(y=10, line_dash="dash", line_color="red", opacity=0.5)
    fig.add_hline(y=1000, line_dash="dash", line_color="orange", opacity=0.5)
    fig.update_layout(height=400, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    col1, col2 = st.columns(2)
    with col1:
        risk_counts = recent_data['risk_label'].value_counts()
        risk_labels = {0: 'Healthy', 1: 'Warning', 2: 'Disease Risk'}
        risk_df = pd.DataFrame({
            'Status': [risk_labels[i] for i in risk_counts.index],
            'Count': risk_counts.values
        })
        fig = px.pie(risk_df, values='Count', names='Status',
                     title='Risk Distribution',
                     color='Status',
                     color_discrete_map={
                         'Healthy': '#28a745',
                         'Warning': '#ffc107',
                         'Disease Risk': '#dc3545'
                     })
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        risk_over_time = recent_data[['timestamp', 'risk_label']].copy()
        risk_over_time['Status'] = risk_over_time['risk_label'].map(risk_labels)
        fig = px.scatter(risk_over_time, x='timestamp', y='risk_label',
                         title='Risk Status Over Time',
                         color='Status',
                         color_discrete_map={
                             'Healthy': '#28a745',
                             'Warning': '#ffc107',
                             'Disease Risk': '#dc3545'
                         })
        fig.update_layout(height=350, yaxis=dict(tickmode='array', tickvals=[0,1,2], ticktext=['Healthy', 'Warning', 'Disease Risk']))
        st.plotly_chart(fig, use_container_width=True)

# Alerts and Action Log
st.markdown("---")
col_alerts, col_log = st.columns(2)

with col_alerts:
    st.markdown("### 🔔 Alerts & Recommendations")
    alerts = []
    recommendations = []
    
    if latest['temperature'] > 30:
        alerts.append("⚠️ **High Temperature**: Temperature is above 30°C")
        recommendations.append("👉 Cooling fan activated automatically")
    elif latest['temperature'] < 20:
        alerts.append("⚠️ **Low Temperature**: Temperature is below 20°C")
        recommendations.append("👉 Heater activated automatically")
    
    if latest['humidity'] > 85:
        alerts.append("⚠️ **High Humidity**: Humidity is above 85%")
        recommendations.append("👉 Ventilation system activated")
    elif latest['humidity'] < 40:
        alerts.append("⚠️ **Low Humidity**: Humidity is below 40%")
        recommendations.append("👉 Water mist system recommended")
    
    if latest['ammonia'] > 10:
        alerts.append("🚨 **Dangerous Ammonia**: Level above 10 ppm")
        recommendations.append("👉 Immediate ventilation and litter management")
    
    if latest['co2'] > 1000:
        alerts.append("⚠️ **High CO2**: Level above 1000 ppm")
        recommendations.append("👉 Increased fresh air intake")
    
    if latest['water_level'] < 30:
        alerts.append("⚠️ **Low Water**: Water level below 30%")
        recommendations.append("👉 Water pump activated; check supply")
    
    if latest['feed_level'] < 25:
        alerts.append("⚠️ **Low Feed**: Feed level below 25%")
        recommendations.append("👉 Farmer: Refill feeders and check for spoilage")
    
    if latest['air_quality'] > 300:
        alerts.append("⚠️ **Poor Air Quality**: Level above 300")
        recommendations.append("👉 Ventilation and air circulation improved")
    
    if prediction['status'] == 'Disease Risk':
        alerts.append("🚨 **Disease Risk Detected**: AI predicts potential outbreak")
        recommendations.append("👉 Alarm activated; consult veterinarian immediately")
    
    if len(alerts) == 0:
        st.success("✅ **All Clear**: All conditions are within normal range. System is maintaining optimal environment.")
    else:
        for alert in alerts:
            if "🚨" in alert:
                st.error(alert)
            elif "⚠️" in alert:
                st.warning(alert)
            else:
                st.info(alert)
        if recommendations:
            st.markdown("#### 📋 Actions Taken / Recommendations:")
            for rec in set(recommendations):
                if "Farmer" in rec:
                    st.warning(rec)
                else:
                    st.success(rec)

with col_log:
    st.markdown("### 📋 System Action Log")
    st.caption("Recent automated actions and data logging events")
    if len(st.session_state.action_log) == 0:
        st.info("No actions logged yet.")
    else:
        for entry in st.session_state.action_log[:10]:
            st.markdown(f'<div class="log-entry">{entry}</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("#### 📤 Data Logging")
    st.info(f"📊 Data points logged to Google Sheets: {st.session_state.data_logged}")
    if st.button("📝 Simulate Log Now"):
        log_to_google_sheets(latest)
        st.rerun()

# Footer
st.markdown("---")
st.markdown(f"""
<center>
    <p>🐔 Smart Poultry House Monitoring System v2.0 | Powered by AI & IoT</p>
    <p style="color: gray; font-size: 0.8rem;">
        Data updated in real-time | Last update: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    </p>
    <p style="color: gray; font-size: 0.7rem;">
        Total Records: {len(st.session_state.data)} | 
        Disease Risk Events: {len(st.session_state.data[st.session_state.data['risk_label'] == 2])}
    </p>
</center>
""", unsafe_allow_html=True)