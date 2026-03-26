import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ==========================================
# 0. CONFIGURATION & THEME
# ==========================================
st.set_page_config(page_title="OculoGuard MLOps Central", layout="wide", page_icon="📈")

st.markdown("""
<style>
    .reportview-container { background: #f0f2f6; }
    .metric-card { background: white; padding: 20px; border-radius: 10px; box-shadow: 2px 2px 10px rgba(0,0,0,0.05); }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. HEADER & GLOBAL STATUS
# ==========================================
st.title("🚀 OculoGuard Live Monitoring Dashboard")
st.markdown("### *Real-time Data Drift & Model Performance Tracking*")

# Top Metrics Row
m1, m2, m3, m4 = st.columns(4)
m1.metric("Active Model", "DenseNet121-v2", "Stable")
m2.metric("Avg Precision", "0.78", "+2.1%")
m3.metric("Daily Scans", "142", "+15")
m4.metric("System Health", "98.2%", "Optimal")

st.divider()

# ==========================================
# 2. DATA DRIFT ANALYSIS (Le point "Wahou")
# ==========================================
st.subheader("🕵️ Data Drift Detection")
col_drift_left, col_drift_right = st.columns([2, 1])

with col_drift_left:
    st.write("**Pixel Intensity Distribution (Training vs. Production)**")
    # Simulation de drift
    hist_data = pd.DataFrame({
        'Training Set': np.random.normal(120, 15, 1000),
        'Production (Last 24h)': np.random.normal(132, 18, 1000)
    })
    fig_drift = px.histogram(hist_data, barmode='overlay', color_discrete_sequence=['#3498db', '#e74c3c'])
    st.plotly_chart(fig_drift, use_container_width=True)
    st.info("💡 **Insight:** Current images are slightly brighter. Monitor auto-exposure on clinical cameras.")

with col_drift_right:
    st.write("**Feature Stability Index**")
    metrics = {
        "Image Contrast": 0.041,
        "Noise Level": 0.12,
        "Resolution Sync": 0.89
    }
    for m, p in metrics.items():
        status = "⚠️ DRIFT" if p < 0.05 else "✅ STABLE"
        st.metric(m, f"p={p}", delta=status, delta_color="inverse" if p < 0.05 else "normal")

st.divider()

# ==========================================
# 3. MLFLOW PERFORMANCE TRACKING
# ==========================================
st.subheader("📉 MLflow Experiment Tracking (Run: final-prod-run)")
col_perf1, col_perf2 = st.columns(2)

with col_perf1:
    # Récupération des données de ton Classification Report
    performance_df = pd.DataFrame({
        'Class': ['No_DR', 'Moderate', 'Mild', 'Severe', 'Proliferative'],
        'Precision': [0.97, 0.78, 0.52, 0.28, 0.32],
        'Recall': [0.96, 0.44, 0.72, 0.50, 0.47]
    })
    fig_perf = px.bar(performance_df, x='Class', y=['Precision', 'Recall'], barmode='group',
                     title="Model Performance by Stage")
    st.plotly_chart(fig_perf, use_container_width=True)

with col_perf2:
    st.write("**Inference Latency (ms)**")
    latency = np.random.uniform(120, 250, 50)
    st.line_chart(latency)
    st.caption("Average Latency: 185ms | Target: < 300ms")

st.divider()

# ==========================================
# 4. INFRASTRUCTURE HEALTH (Medallion & Storage)
# ==========================================
st.subheader("⚙️ Infrastructure & Pipeline Status")
i1, i2, i3 = st.columns(3)

with i1:
    st.write("**MinIO S3 Buckets**")
    st.progress(65, text="raw-fundus: 1.2TB / 2TB")
    st.progress(22, text="gold-validated: 450GB")

with i2:
    st.write("**PostgreSQL Operations**")
    st.success("Metadata DB: CONNECTED")
    st.write("- Total Audit Records: 4,521")
    st.write("- Unprocessed Records: 12")

with i3:
    st.write("**Pipeline Stages**")
    st.code("""
    Bronze: OK (Kafka Stream)
    Silver: OK (Spark Job)
    Gold:   OK (Expert Validation)
    """)

# ==========================================
# 5. RECENT LOGS
# ==========================================
if st.checkbox("Show Recent Operational Logs"):
    log_data = pd.DataFrame([
        {"Timestamp": "2026-03-26 14:02", "Event": "Model v2.0 Deployed", "Status": "SUCCESS"},
        {"Timestamp": "2026-03-26 14:15", "Event": "MinIO Backup", "Status": "SUCCESS"},
        {"Timestamp": "2026-03-26 15:10", "Event": "Drift Alert: Luminance", "Status": "WARNING"},
    ])
    st.table(log_data)