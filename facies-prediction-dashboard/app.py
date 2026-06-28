import streamlit as st
import lasio
import pandas as pd
import numpy as np
import joblib
import io
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors

st.set_page_config(page_title="Facies Predictor", layout="wide")

# Load trained pipeline assets


@st.cache_resource
def load_pipeline():
    # Dynamically find the exact folder where app.py is currently running on the cloud server
    base_path = os.path.dirname(__file__)

    scaler_path = os.path.join(base_path, 'scaler.joblib')
    model_path = os.path.join(base_path, 'best_baseline_rf.joblib')
    
    scaler = joblib.load('scaler.joblib')
    model = joblib.load('best_baseline_rf.joblib')
    return scaler, model


try:
    scaler, model = load_pipeline()
    st.sidebar.success("Model & Scaler loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading model files: {e}")
    st.stop()

# App User Interface
st.title("Subsurface Facies Prediction Dashboard")
st.markdown("Upload a standard `.las` well log file to generate automated machine learning lithofacies classifications.")

uploaded_file = st.file_uploader("Choose a LAS file", type=['las'])

if uploaded_file is not None:
    # Read LAS file from memory buffer
    bytes_data = uploaded_file.read()
    str_io = io.StringIO(bytes_data.decode('utf-8', errors='ignore'))

    try:
        las = lasio.read(str_io)
        df_las = las.df().reset_index()  # Extract curve data and make 'Depth' a column
        st.success(f"✓ Successfully parsed: {uploaded_file.name}")
    except Exception as e:
        st.error(f"Failed to parse LAS file: {e}")
        st.stop()

    # Dynamic Column mapping and Validation
    st.subheader("Dataset Verification & Processing")

    # Check for required core logs; provide dropdown fallback selectors if exact match fails
    required_curves = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE']
    mapped_columns = {}

    col_selectors = st.columns(len(required_curves))
    for idx, curve in enumerate(required_curves):
        with col_selectors[idx]:
            # Guess mapping based on string match, default to first available curve if missing
            default_idx = df_las.columns.get_loc(
                curve) if curve in df_las.columns else 0
            mapped_columns[curve] = st.selectbox(
                f"Map curve for {curve}:", df_las.columns, index=default_idx)

    # Handle dataset-specific geological metadata targets (NM_M and RELPOS)
    # If missing from raw logs, generate standard baseline geologic assumptions
    if 'NM_M' not in df_las.columns:
        df_las['NM_M'] = 1  # Default to non-marine environment proxy fallback
    if 'RELPOS' not in df_las.columns:
        df_las['RELPOS'] = 0.5  # Default to mid-formation coordinate positions

    # Preprocessing Pipeline
    df_proc = pd.DataFrame()
    df_proc['Depth'] = df_las['Depth']

    # Map selection inputs back to pipeline arrays
    for curve in required_curves:
        df_proc[curve] = df_las[mapped_columns[curve]]

    df_proc['NM_M'] = df_las['NM_M']
    df_proc['RELPOS'] = df_las['RELPOS']

    # Step 1 Fallback: Local/Global Median Imputation for PE gaps
    df_proc['PE'] = df_proc['PE'].fillna(df_proc['PE'].median())

    # Step 2: Feature Engineering (Ratios & 3-Sample Windows)
    df_proc['GR_PHIND_ratio'] = df_proc['GR'] / (df_proc['PHIND'] + 0.001)

    for curve in required_curves:
        df_proc[f'{curve}_roll_mean'] = df_proc[curve].rolling(
            window=3, min_periods=1).mean()
        df_proc[f'{curve}_roll_std'] = df_proc[curve].rolling(
            window=3, min_periods=1).std().fillna(0)

    # Order columns exactly matching training phase features dictionary layout
    features_ordered = [
        'GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS',
        'GR_PHIND_ratio', 'GR_roll_mean', 'GR_roll_std', 'ILD_log10_roll_mean',
        'ILD_log10_roll_std', 'DeltaPHI_roll_mean', 'DeltaPHI_roll_std',
        'PHIND_roll_mean', 'PHIND_roll_std', 'PE_roll_mean', 'PE_roll_std'
    ]

    # Ensure all required features are present
    X_live_raw = df_proc[features_ordered]

    # Scale and Predict
    X_live_scaled = scaler.transform(X_live_raw)
    df_proc['Predicted_Facies'] = model.predict(X_live_scaled)

    # Render Log Visualization Track
    st.subheader("Machine Learning Log Interpretation Log Strip")

    facies_colors = ['#F4D03F', '#F5B041', '#DC7633', '#A11D33',
                     '#1B4F72', '#2E4053', '#7D6608', '#117A65', '#145A32']
    cmap_facies = colors.ListedColormap(facies_colors, 'indexed')

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 8), sharey=True)

    # Track 1: Gamma Ray
    ax[0].plot(df_proc['GR'], df_proc['Depth'], color='black', lw=1.0)
    ax[0].set_title("Gamma Ray (GR)")
    ax[0].set_xlabel("API")
    ax[0].grid(True, linestyle=':', alpha=0.5)

    # Track 2: Resistivity
    ax[1].plot(df_proc['ILD_log10'], df_proc['Depth'], color='blue', lw=1.0)
    ax[1].set_title("Resistivity (ILD)")
    ax[1].set_xlabel("Log10 Ohmm")
    ax[1].grid(True, linestyle=':', alpha=0.5)

    # Track 3: Predicted Facies Log Strip
    pred_strip = np.repeat(
        df_proc['Predicted_Facies'].values, 100).reshape(-1, 100)
    ax[2].imshow(pred_strip, cmap=cmap_facies, aspect='auto',
                 extent=[0, 20, df_proc['Depth'].max(), df_proc['Depth'].min()], vmin=1, vmax=9)
    ax[2].set_title("Predicted Facies")
    ax[2].set_xticks([])

    # Globally invert Y-axis so depth values increase going downward
    ax[0].invert_yaxis()
    plt.tight_layout()

    # Display the static matplotlib image layout container within the web app interface
    st.pyplot(fig)
