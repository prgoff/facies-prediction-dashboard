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

    # Automatic dictionary mapping and notifications
    st.subheader("Dataset Verification & Processing")
    
    # Comprehensive dictionary mapping standard features to all known vendor aliases
    curve_aliases = {
        'GR': ['GR', 'GGCE', 'GR_ED', 'GAM', 'CGR', 'SGR', 'GRD'],
        'ILD_log10': ['ILD_LOG10', 'RTAO', 'ILD', 'LL3', 'RT', 'AHT90', 'AT90', 'RILD'],
        'DeltaPHI': ['DELTAPHI', 'DPHI', 'DPOR', 'DEPT_PHI', 'DPHI_NPHI'],
        'PHIND': ['PHIND', 'XPOR', 'NPHI', 'PHIN', 'NPHI_HL', 'POROSITY'],
        'PE': ['PE', 'PDPE', 'PEF', 'DEN_COR']
    }
    
    required_curves = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE']
    mapped_columns = {}
    auto_mapped_log = []

    # Hidden Auto-Detection Logic
    for curve in required_curves:
        detected_column = None
        for col in df_las.columns:
            if col.upper() in [alias.upper() for alias in curve_aliases[curve]]:
                detected_column = col
                break
        
        # If detected, lock it in. If completely missing, default to first column.
        mapped_columns[curve] = detected_column if detected_column else df_las.columns[0]
        
        if detected_column:
            auto_mapped_log.append(f"{curve} → {detected_column}")

    # Display a sleek notification banner just like your friend's app
    if len(auto_mapped_log) == len(required_curves):
        st.info(f"**Auto-mapped all curves successfully:** {', '.join(auto_mapped_log)}")
    else:
        st.warning("⚠️ Some curves could not be automatically matched. Please verify mappings below.")

    # Hide the dropdown menus inside a clean, collapsible menu
    with st.expander("⚙️ Advanced Curve Mapping Overrides"):
        st.write("If the auto-detection missed a curve, correct it manually here:")
        col_selectors = st.columns(len(required_curves))
        for idx, curve in enumerate(required_curves):
            with col_selectors[idx]:
                current_default = mapped_columns[curve]
                default_idx = df_las.columns.get_loc(current_default) if current_default in df_las.columns else 0
                mapped_columns[curve] = st.selectbox(f"{curve}:", df_las.columns, index=default_idx)
            
    # Handle dataset-specific geological metadata targets (NM_M and RELPOS)
    # If missing from raw logs, generate standard baseline geologic assumptions
    if 'NM_M' not in df_las.columns:
        df_las['NM_M'] = 1  # Default to non-marine environment proxy fallback
    if 'RELPOS' not in df_las.columns:
        df_las['RELPOS'] = 0.5  # Default to mid-formation coordinate positions
        
    # Failures and Warnings
    # Check if any tool fell back to 'Depth' because it was missing from the file
    for curve, mapped_col in mapped_columns.items():
        if mapped_col in ['Depth', 'DEPT']:
            st.sidebar.error(f"🚨 Missing Tool: This file does not contain a {curve} log. The app will use background averages, but model accuracy will decrease.")

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

    # Plotting engine with advanced tracks toggle
    st.subheader("Subsurface Facies Interpretation Model")
    
    # 1. Add the toggle switch to the sidebar
    show_advanced = st.sidebar.checkbox("👁️ Show Advanced Engineering Tracks", value=False)
    
    # 2. Dynamically set up columns based on the toggle switch
    if show_advanced:
        fig, ax = plt.subplots(1, 5, figsize=(15, 10), sharey=True)
        # Track 1: GR, Track 2: Resistivity, Track 3: NM_M, Track 4: RELPOS, Track 5: Facies
        facies_track_idx = 4
    else:
        fig, ax = plt.subplots(1, 3, figsize=(11, 10), sharey=True)
        # Track 1: GR, Track 2: Resistivity, Track 3: Facies
        facies_track_idx = 2

    # Invert Y-axis so depth goes down into the earth
    ax[0].invert_yaxis()
    
    # Track 1: Gamma Ray (Standard)
    ax[0].plot(df_proc['GR'], df_proc['Depth'], color='black', linewidth=1)
    ax[0].set_title("Gamma Ray (GR)")
    ax[0].set_xlabel("API")
    ax[0].grid(True, linestyle=':', alpha=0.5)
    
    # Track 2: Deep Resistivity (Standard)
    ax[1].plot(df_proc['ILD_log10'], df_proc['Depth'], color='blue', linewidth=1)
    ax[1].set_title("Resistivity (ILD)")
    ax[1].set_xlabel("Log10 Ohmm")
    ax[1].grid(True, linestyle=':', alpha=0.5)

    # If the user checks the box, inject your friend's extra engineering tracks!
    if show_advanced:
        # Track 3: Marine vs Non-Marine indicator block
        ax[2].plot(df_proc['NM_M'], df_proc['Depth'], color='purple', linewidth=1.5)
        ax[2].set_title("Marine Block (NM_M)")
        ax[2].set_xlabel("Code")
        ax[2].grid(True, linestyle=':', alpha=0.5)
        
        # Track 4: Relative Position slope line
        ax[3].plot(df_proc['RELPOS'], df_proc['Depth'], color='brown', linewidth=1)
        ax[3].set_title("Rel Position (RELPOS)")
        ax[3].set_xlabel("Slope Index")
        ax[3].grid(True, linestyle=':', alpha=0.5)

    # Final Track: Your clean, superior multi-colored Facies Strip chart
    # Create a vertical strip chart by repeating the 1D prediction array horizontally
    # Final Track: Your clean multi-colored structural Facies Strip
    pred_strip = np.repeat(df_proc['Predicted_Facies'].values, 100).reshape(-1, 100)
    ax[facies_track_idx].imshow(pred_strip, cmap=cmap_facies, aspect='auto', 
                                extent=[0, 20, df_proc['Depth'].max(), df_proc['Depth'].min()], vmin=1, vmax=9)
    ax[facies_track_idx].set_title("Predicted Facies")
    ax[facies_track_idx].set_xticks([])

    plt.tight_layout()
    st.pyplot(fig)

    # --- DOWNLOAD PREDICTIONS AS CSV ---
    st.markdown("---")
    st.write("### 💾 Export Interpretation Results")
    st.write("Download the processed well log data along with your model's continuous facies predictions as a standard CSV spreadsheet.")

    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

    csv_bytes = convert_df_to_csv(df_proc)

    st.download_button(
        label="📥 Download Facies Predictions (.csv)",
        data=csv_bytes,
        file_name=f"Facies_Predictions_{uploaded_file.name.replace('.las', '')}.csv",
        mime="text/csv",
        key='download-csv'
    )

    # --- CROSSPLOT ANALYSIS ---
    st.markdown("---")
    st.write("### 📊 Facies Crossplot Clustering")
    st.write("Examine the machine learning model's partitions. This graph plots Gamma Ray directly against Resistivity, with every depth point colored by its predicted facies classification.")

    fig_cross, ax_cross = plt.subplots(figsize=(8, 5))
    scatter = ax_cross.scatter(
        df_proc['GR'], 
        df_proc['ILD_log10'], 
        c=df_proc['Predicted_Facies'], 
        cmap=cmap_facies, 
        alpha=0.7, 
        edgecolors='none',
        vmin=1,
        vmax=9
    )
    
    ax_cross.set_xlabel("Gamma Ray (GR) - API Units")
    ax_cross.set_ylabel("Resistivity (ILD) - Log10 Ohm-m")
    ax_cross.set_title("AI Decision Domains: GR vs. Resistivity")
    ax_cross.grid(True, linestyle=':', alpha=0.5)
    
    cbar = fig_cross.colorbar(scatter, ax=ax_cross, ticks=range(1, 10))
    cbar.set_label('Predicted Facies ID Number')
    
    st.pyplot(fig_cross)
    
