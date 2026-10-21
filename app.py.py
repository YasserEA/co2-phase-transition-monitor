import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Page configuration
st.set_page_config(
    page_title="CO₂ Phase Transition Monitor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1e2130;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #00d2ff;
        margin-bottom: 10px;
    }
    .status-alert {
        background-color: #3b1414;
        border: 1px solid #ff4b4b;
        color: #ff4b4b;
        padding: 12px;
        border-radius: 6px;
        font-weight: bold;
    }
    .status-ok {
        background-color: #0f3822;
        border: 1px solid #00cc66;
        color: #00cc66;
        padding: 12px;
        border-radius: 6px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to generate fallback dataset if CSV is missing
def generate_synthetic_data():
    t = np.arange(0, 1802, 2)
    n = len(t)
    
    # Simulate Scenario S2-M2 (Pressure drop with 5% N2)
    P_true = np.where(t < 60, 11.0, np.maximum(5.0, 11.0 - (t - 60) * (6.0 / 1740)))
    T_true = np.where(t < 60, 30.0, np.maximum(8.0, 30.0 - (t - 60) * (22.0 / 1740)))
    
    # Bubble pressure calculation approximation
    P_bubble = 0.15 * T_true + 4.2
    margin = P_true - P_bubble
    
    # GVF and state
    state_true = []
    gvf_true = []
    for m in margin:
        if m > 0.5:
            state_true.append("dense")
            gvf_true.append(0.0)
        elif m >= 0:
            state_true.append("dense")
            gvf_true.append(0.005)
        else:
            state_true.append("two-phase")
            gvf_true.append(min(1.0, abs(m) * 0.12))
            
    # Add noise
    rng = np.random.default_rng(42)
    P_meas = P_true + rng.normal(0, 0.02, n)
    T_meas = T_true + rng.normal(0, 0.1, n)
    gvf_meas = np.clip(np.array(gvf_true) + rng.normal(0, 0.005, n), 0, 1)
    rho_true = np.where(np.array(state_true) == "dense", 850 - T_true * 5 + P_true * 15, 250 - T_true * 2)
    rho_meas = rho_true * (1 + rng.normal(0, 0.005, n))
    
    detect_flag = (pd.Series(gvf_meas) > 0.03).rolling(5).sum().fillna(0) >= 5
    
    df = pd.DataFrame({
        'run_id': 'S2-M2',
        'scenario': 'S2',
        'mixture': 'M2',
        't_s': t,
        'P_true_MPa': P_true,
        'T_true_C': T_true,
        'N2_mol_frac': 0.05,
        'state_true': state_true,
        'gvf_true': gvf_true,
        'rho_true_kgm3': rho_true,
        'margin_MPa': margin,
        'P_meas_MPa': P_meas,
        'T_meas_C': T_meas,
        'gvf_meas': gvf_meas,
        'rho_meas_kgm3': rho_meas,
        'detect_flag': detect_flag
    })
    return df

@st.cache_data
def load_data():
    if os.path.exists("COOLPROP_dataset.csv"):
        df = pd.read_csv("COOLPROP_dataset.csv")
    else:
        df = generate_synthetic_data()
    return df

# Load Data
df = load_data()

# Sidebar Controls
st.sidebar.title("🎛️ Control Panel")
st.sidebar.markdown("---")

run_list = df['run_id'].unique().tolist()
selected_run = st.sidebar.selectbox("Select Scenario Run", run_list, index=0)

run_df = df[df['run_id'] == selected_run].copy().reset_index(drop=True)

# Time Step Slider for Live/Playback Simulation
st.sidebar.markdown("### ⏱️ Time Playback Control")
time_step = st.sidebar.slider(
    "Simulation Time (s)",
    min_value=int(run_df['t_s'].min()),
    max_value=int(run_df['t_s'].max()),
    value=int(run_df['t_s'].max()),
    step=int(run_df['t_s'].iloc[1] - run_df['t_s'].iloc[0]) if len(run_df) > 1 else 2
)

# Current Slice
current_idx = run_df[run_df['t_s'] <= time_step].index[-1]
current_row = run_df.iloc[current_idx]

# Header Section
st.title("⚡ CO₂ Injection Phase Transition Monitor")
st.caption("Real-Time Multi-Phase Flow Monitoring & Early Detection Dashboard | Abbon PM3 Engine")
st.markdown("---")

# Top Alert System & KPIs
col1, col2, col3, col4, col5 = st.columns(5)

# Status Assessment
is_twophase = current_row['state_true'] in ['two-phase', 'gas', 'two-phase(sat)'] or current_row['gvf_meas'] > 0.03
margin_val = current_row['margin_MPa']

with col1:
    st.markdown("##### Phase State")
    if is_twophase:
        st.markdown("<div class='status-alert'>⚠️ TWO-PHASE / GAS</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='status-ok'>✅ DENSE PHASE</div>", unsafe_allow_html=True)

with col2:
    st.metric(
        label="Pressure (P)",
        value=f"{current_row['P_meas_MPa']:.2f} MPa",
        delta=f"{current_row['P_meas_MPa'] - run_df.iloc[0]['P_meas_MPa']:.2f} MPa"
    )

with col3:
    st.metric(
        label="Temperature (T)",
        value=f"{current_row['T_meas_C']:.1f} °C",
        delta=f"{current_row['T_meas_C'] - run_df.iloc[0]['T_meas_C']:.1f} °C"
    )

with col4:
    st.metric(
        label="Gas Volume Fraction (GVF)",
        value=f"{current_row['gvf_meas']*100:.1f} %",
        delta="CRITICAL" if current_row['gvf_meas'] > 0.03 else "NORMAL"
    )

with col5:
    margin_disp = f"{margin_val:.2f} MPa" if pd.notna(margin_val) else "N/A"
    st.metric(
        label="Pressure Margin (P - P_up)",
        value=margin_disp,
        delta="RISK" if pd.notna(margin_val) and margin_val < 0.5 else "SAFE"
    )

st.markdown("<br>", unsafe_allow_html=True)

# Main Dashboard Layout
tab1, tab2, tab3 = st.tabs(["📈 Real-Time Charts", "📊 Phase Diagram (P-T)", "📜 Compliance Report Generator"])

with tab1:
    # Subplots for Pressure/Temperature and Density/GVF
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Pressure & Pressure Margin Over Time", "Measured Density & Gas Volume Fraction (GVF)")
    )

    # Historical slice for playback
    hist_df = run_df.iloc[:current_idx+1]

    # Row 1: Pressure & Margin
    fig.add_trace(go.Scatter(x=hist_df['t_s'], y=hist_df['P_meas_MPa'], name="P Measured (MPa)", line=dict(color='#00d2ff', width=2)), row=1, col=1)
    if 'margin_MPa' in hist_df.columns and hist_df['margin_MPa'].notna().any():
        fig.add_trace(go.Scatter(x=hist_df['t_s'], y=hist_df['margin_MPa'], name="Margin P-Pup (MPa)", line=dict(color='#ff9900', dash='dash')), row=1, col=1)

    # Detection marker
    detected_times = hist_df[hist_df['detect_flag'] == True]
    if not detected_times.empty:
        fig.add_trace(go.Scatter(
            x=[detected_times['t_s'].iloc[0]],
            y=[detected_times['P_meas_MPa'].iloc[0]],
            mode='markers+text',
            marker=dict(color='red', size=12, symbol='x'),
            text=["ALERT TRIGGERED"],
            textposition="top center",
            name="Detection Event"
        ), row=1, col=1)

    # Row 2: Density & GVF
    fig.add_trace(go.Scatter(x=hist_df['t_s'], y=hist_df['rho_meas_kgm3'], name="Density (kg/m³)", line=dict(color='#00cc66', width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=hist_df['t_s'], y=hist_df['gvf_meas']*100, name="GVF (%)", line=dict(color='#ff4b4b', width=2), yaxis="y4"), row=2, col=1)

    fig.update_layout(
        height=550,
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)
    fig.update_yaxes(title_text="Pressure (MPa)", row=1, col=1)
    fig.update_yaxes(title_text="Density (kg/m³)", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Phase Envelope Trajectory (P-T Space)")
    
    fig_pt = go.Figure()
    
    # Plot Trajectory up to selected time
    fig_pt.add_trace(go.Scatter(
        x=hist_df['T_meas_C'],
        y=hist_df['P_meas_MPa'],
        mode='lines+markers',
        name='Trajectory',
        line=dict(color='#00d2ff', width=3),
        marker=dict(size=4)
    ))
    
    # Current state highlight
    fig_pt.add_trace(go.Scatter(
        x=[current_row['T_meas_C']],
        y=[current_row['P_meas_MPa']],
        mode='markers',
        name='Current State',
        marker=dict(color='red' if is_twophase else '#00cc66', size=14, symbol='diamond')
    ))

    fig_pt.update_layout(
        title="Pressure vs. Temperature Operational Path",
        xaxis_title="Temperature (°C)",
        yaxis_title="Pressure (MPa)",
        template="plotly_dark",
        height=500
    )
    
    st.plotly_chart(fig_pt, use_container_width=True)

with tab3:
    st.subheader("📋 Automated Compliance & Event Report")
    st.markdown("Generates an instant operational report for regulatory compliance and audit logs.")
    
    # Calculations for report
    total_duration = run_df['t_s'].max()
    twophase_count = (run_df['state_true'].isin(['two-phase', 'gas', 'two-phase(sat)'])).sum()
    pct_twophase = (twophase_count / len(run_df)) * 100
    
    first_detect = run_df[run_df['detect_flag'] == True]
    time_detect = first_detect['t_s'].iloc[0] if not first_detect.empty else "N/A"

    report_text = f"""
    ================================================================================
    OFFSHORE / ONSHORE CO2 INJECTION COMPLIANCE REPORT
    ================================================================================
    Report ID         : RPT-{selected_run}-{int(time_step)}
    Scenario Run ID   : {selected_run}
    Mixture Profile   : {current_row['mixture']} (N2 Content: {current_row.get('N2_mol_frac', 0.0)*100:.2f} mol%)
    Timestamp Limit   : {time_step} s / {total_duration} s
    --------------------------------------------------------------------------------
    OPERATIONAL SUMMARY:
    - Initial Pressure & Temp : {run_df.iloc[0]['P_true_MPa']:.2f} MPa | {run_df.iloc[0]['T_true_C']:.2f} °C
    - Minimum Pressure Reached: {run_df['P_meas_MPa'].min():.2f} MPa
    - Phase Transition Status : {"EXCURSION DETECTED" if pct_twophase > 0 else "NOMINAL (DENSE PHASE)"}
    - Time in Two-Phase Region: {pct_twophase:.1f} % of total run time
    - First Detection Timestamp: {time_detect} s
    --------------------------------------------------------------------------------
    REGULATORY / SAFETY COMPLIANCE CHECK:
    [{"FAIL" if pct_twophase > 5 else "PASS"}] Phase Stability Standard (Excursion < 5% duration limit)
    [{"FAIL" if current_row['gvf_meas'] > 0.05 else "PASS"}] GVF Operational Threshold Limit (GVF < 5%)
    ================================================================================
    """
    
    st.code(report_text, language="text")
    
    st.download_button(
        label="📥 Download Official Compliance Report (.txt)",
        data=report_report_text if 'report_report_text' in locals() else report_text,
        file_name=f"CO2_Compliance_Report_{selected_run}.txt",
        mime="text/plain"
    )

st.markdown("---")
st.caption("Abbon PM3 Engine Integration | R&D Demonstration Module")
