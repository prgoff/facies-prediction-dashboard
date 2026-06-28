import streamlit as st
import lasio
import pandas as pd
import numpy as np
import joblib
import io
import matplotlib.pyplot as plt
import matplotlib.colors as colors

st.set_page_config(page_title="Facies Predictor", layout="wide")

# Load trained pipeline assets


@st.cache_resource
def load_pipeline():
    scaler = joblib.load('scaler.joblib')
    model = joblib.load('best_baseline_rf.joblib')
    return scaler, model


try:
    scaler, model = load_pipeline()
    st.sidebar.success("✓ Model & Scaler loaded successfully!")
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
        
        # Force whatever the first column is (the depth index) to be named 'Depth'
        df_las.rename(columns={df_las.columns[0]: 'Depth'}, inplace=True)
        
        st.success(f"✓ Successfully parsed: {uploaded_file.name}")
    except Exception as e:
        st.error(f"Failed to parse LAS file: {e}")
        st.stop()

    # Dynamic Column mapping and Validation
    # --- 3. DYNAMIC COLUMN MAPPING & AUTOMATIC ALIASING ---
    st.subheader("Dataset Verification & Processing")
    
    # Define an industry-standard alias dictionary for common log names
    curve_aliases = {
        'GR': ['GR', 'GR_ED', 'GAM', 'CGR', 'SGR', 'GRD'],
        'ILD_log10': ['ILD_LOG10', 'ILD', 'LL3', 'RT', 'AHT90', 'AT90', 'RILD'],
        'DeltaPHI': ['DELTAPHI', 'DPHI', 'DEPT_PHI', 'DPHI_NPHI'],
        'PHIND': ['PHIND', 'NPHI', 'PHIN', 'NPHI_HL', 'POROSITY'],
        'PE': ['PE', 'PEF', 'PDPE', 'DEN_COR']
    }
    
    required_curves = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE']
    mapped_columns = {}
    
    col_selectors = st.columns(len(required_curves))
    
    for idx, curve in enumerate(required_curves):
        with col_selectors[idx]:
            # Step 1: Try to automatically find a matching alias (case-insensitive)
            detected_column = None
            possible_aliases = curve_aliases[curve]
            
            for col in df_las.columns:
                if col.upper() in [alias.upper() for alias in possible_aliases]:
                    detected_column = col
                    break
            
            # Step 2: Determine the default dropdown index position
            if detected_column:
                default_idx = df_las.columns.get_loc(detected_column)
            else:
                # Fallback to index 0 (Depth) if no alias matches
                default_idx = 0
            
            # Step 3: Render the dropdown pre-selected to the auto-detected curve
            mapped_columns[curve] = st.selectbox(
                f"Map curve for {curve}:", 
                df_las.columns, 
                index=default_idx
            )
            
    # Handle dataset-specific geological metadata targets (NM_M and RELPOS)
    # If missing from raw logs, generate standard baseline geologic assumptions
    if 'NM_M' not in df_las.columns:
        df_las['NM_M'] = 1  # Default to non-marine environment proxy fallback
    if 'RELPOS' not in df_las.columns:
        df_las['RELPOS'] = 0.5  # Default to mid-formation coordinate positions

    # Preprocessing Pipeline
    df_proc = pd.DataFrame()
    df_proc['Depth'] = df_las['Depth']
    
    # Map selection inputs back to pipeline arrays safely
    for curve in required_curves:
        # Force the column to be strictly numeric. Convert any rogue text strings into NaNs safely.
        raw_series = pd.to_numeric(df_las[mapped_columns[curve]], errors='coerce')
        df_proc[curve] = raw_series
        
    # Check if the user passed raw resistivity numbers by checking the data median.
    # Log10 resistivity rarely exceeds 3.5, whereas raw resistivity values regularly go past 10.
    if df_proc['ILD_log10'].median() > 5.0:
        # Prevent taking the log of zero or negative numbers by clipping at a small positive floor
        df_proc['ILD_log10'] = np.log10(df_proc['ILD_log10'].clip(lower=0.001))

    # Bring in fallback targets
    df_proc['NM_M'] = df_las['NM_M'] if 'NM_M' in df_las.columns else 1
    df_proc['RELPOS'] = df_las['RELPOS'] if 'RELPOS' in df_las.columns else 0.5
    
    # If a tool failed or has missing rows, fill them cleanly using column medians
    for col in required_curves:
        df_proc[col] = df_proc[col].fillna(df_proc[col].median())
        
    df_proc['GR_PHIND_ratio'] = df_proc['GR'] / (df_proc['PHIND'] + 0.001)
    
    # Replace any rogue infinite math calculations with standard NaN markers, then fill them
    df_proc.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_proc['GR_PHIND_ratio'] = df_proc['GR_PHIND_ratio'].fillna(df_proc['GR_PHIND_ratio'].median())

    # Generate rolling windows sequentially
    for curve in required_curves:
        df_proc[f'{curve}_roll_mean'] = df_proc[curve].rolling(window=3, min_periods=1).mean()
        df_proc[f'{curve}_roll_std'] = df_proc[curve].rolling(window=3, min_periods=1).std().fillna(0)

    # Order columns exactly matching training phase features layout
    features_ordered = [
        'GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS',
        'GR_PHIND_ratio', 'GR_roll_mean', 'GR_roll_std', 'ILD_log10_roll_mean',
        'ILD_log10_roll_std', 'DeltaPHI_roll_mean', 'DeltaPHI_roll_std',
        'PHIND_roll_mean', 'PHIND_roll_std', 'PE_roll_mean', 'PE_roll_std'
    ]
    
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
