import streamlit as st
import lasio
import pandas as pd
import numpy as np
import joblib
import io
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import plotly.express as px
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="Facies Predictor", layout="wide")

# Load trained pipeline assets
@st.cache_resource
def load_all_models():
    scaler = joblib.load('scaler.joblib')
    rf_model = joblib.load('best_baseline_rf.joblib')
    
    # Safely load optional models as you complete training phases
    try:
        xgb_model = joblib.load('best_baseline_xgb.joblib')
    except Exception:
        xgb_model = None
        
    try:
        from tensorflow.keras.models import load_model
        cnn_model = load_model('best_1d_cnn.h5')
    except Exception:
        cnn_model = None
        
    return scaler, rf_model, xgb_model, cnn_model

scaler, rf_model, xgb_model, cnn_model = load_all_models()

def create_live_sequences(scaled_data, window_size=5):
    """
    Transforms a 2D feature matrix into consecutive 3D rolling windows 
    for 1D-CNN sequential tracking. Uses edge padding to preserve exact row count.
    """
    half_w = window_size // 2
    padded = np.pad(scaled_data, ((half_w, half_w), (0, 0)), mode='edge')
    
    sequences = []
    for i in range(len(scaled_data)):
        window = padded[i : i + window_size, :]
        sequences.append(window)
        
    return np.array(sequences)

# App User Interface
st.title("Subsurface Facies Prediction Dashboard")
st.markdown("Upload one or multiple standard `.las` well log files to generate automated machine learning lithofacies classifications.")

# Global Configuration Sidebar
st.sidebar.write("### ⚙️ Global App Configurations")
selected_model = st.sidebar.selectbox(
    "Choose Interpretation Architecture:",
    ["Random Forest Baseline", "XGBoost Classifier", "Sequential 1D-CNN Architecture"]
)
show_advanced = st.sidebar.checkbox(" Show Advanced Engineering Tracks", value=False)

uploaded_files = st.file_uploader("Upload one or multiple well log (.las) files", type=["las"], accept_multiple_files=True)

if uploaded_files:
    # Build clean interface tabs dynamically for each well log uploaded
    well_tabs = st.tabs([f"📄 {f.name}" for f in uploaded_files])
    
    for tab_idx, uploaded_file in enumerate(uploaded_files):
        with well_tabs[tab_idx]:
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
                st.error(f"Failed to parse LAS file {uploaded_file.name}: {e}")
                continue

            # Automatic dictionary mapping and notifications
            st.subheader("Dataset Verification & Processing")
            
            # Comprehensive dictionary mapping standard features to all known vendor aliases
            curve_aliases = {
                'GR': [
                    'GR', 'GGCE', 'GR_ED', 'GAM', 'CGR', 'SGR', 'GRD', 'GR_S', 
                    'ECGR', 'NGAM', 'HGR', 'EDTC_GR', 'GRS', 'GRMAIN', 'GGR', 'GAMMA'
                ],
                'ILD_log10': [
                    'ILD_LOG10', 'RTAO', 'ILD', 'LL3', 'RT', 'AHT90', 'AT90', 'RILD', 
                    'RLA5', 'RD', 'RDEEP', 'RESDEEP', 'HILD', 'RT90', 'M2R9', 'HDIL'
                ],
                'DeltaPHI': [
                    'DELTAPHI', 'DPHI', 'DPOR', 'DEPT_PHI', 'DPHI_NPHI', 'DPH', 
                    'DPHZ', 'POR_DENS', 'DPOR_SAN', 'DPOR_LIM', 'DPHZ_L'
                ],
                'PHIND': [
                    'PHIND', 'XPOR', 'NPHI', 'PHIN', 'NPHI_HL', 'POROSITY', 'NPOR', 
                    'CNC', 'CNLS', 'HNPHI', 'NPOR_SAN', 'NPOR_LIM', 'PHIN_L', 'XPOR_LS'
                ],
                'PE': [
                    'PE', 'PDPE', 'PEF', 'DEN_COR', 'PEFZ', 'PECO', 'PFE', 
                    'PEF_SLB', 'PEFZ_EDTC', 'PDPE_L'
                ]
            }
            
            required_curves = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE']
            mapped_columns = {}
            auto_mapped_log = []

            # Auto-Detection Logic
            for curve in required_curves:
                detected_column = None
                for col in df_las.columns:
                    if col.upper().strip() in [alias.upper() for alias in curve_aliases[curve]]:
                        detected_column = col
                        break
                
                mapped_columns[curve] = detected_column if detected_column else df_las.columns[0]
                if detected_column:
                    auto_mapped_log.append(f"{curve} → {detected_column}")

            # Display notification banner per tab workspace
            if len(auto_mapped_log) == len(required_curves):
                st.info(f"**Auto-mapped all curves successfully for this well:** {', '.join(auto_mapped_log)}")
            else:
                st.warning("⚠️ Some curves could not be automatically matched. Please verify mappings below.")

            # Collapsible mapping editor with unique keys per well instance
            with st.expander(f"Advanced Curve Mapping Overrides ({uploaded_file.name})"):
                st.write("If the auto-detection missed a curve, correct it manually here:")
                col_selectors = st.columns(len(required_curves))
                for idx, curve in enumerate(required_curves):
                    with col_selectors[idx]:
                        current_default = mapped_columns[curve]
                        default_idx = df_las.columns.get_loc(current_default) if current_default in df_las.columns else 0
                        mapped_columns[curve] = st.selectbox(
                            f"{curve}:", df_las.columns, index=default_idx, key=f"select-{curve}-{uploaded_file.name}"
                        )
                    
            # Handle metadata targets fallback configurations
            if 'NM_M' not in df_las.columns:
                df_las['NM_M'] = 1  
            if 'RELPOS' not in df_las.columns:
                df_las['RELPOS'] = 0.5  
                
            # Missing Tool Alert Indicators
            for curve, mapped_col in mapped_columns.items():
                if mapped_col in ['Depth', 'DEPT']:
                    st.warning(f"⚠️ Missing Tool: This file does not contain a {curve} log array. Falling back to background averages.")

            # Preprocessing Pipeline Configuration
            df_proc = pd.DataFrame()
            df_proc['Depth'] = df_las['Depth']
            
            for curve in required_curves:
                raw_series = pd.to_numeric(df_las[mapped_columns[curve]], errors='coerce')
                df_proc[curve] = raw_series
                
            # Automatic Resistivity Log Transformation Safeguard
            if df_proc['ILD_log10'].median() > 5.0:
                df_proc['ILD_log10'] = np.log10(df_proc['ILD_log10'].clip(lower=0.001))

            df_proc['NM_M'] = df_las['NM_M'] if 'NM_M' in df_las.columns else 1
            df_proc['RELPOS'] = df_las['RELPOS'] if 'RELPOS' in df_las.columns else 0.5
            
            # Null values median fill routine
            for col in required_curves:
                df_proc[col] = df_proc[col].fillna(df_proc[col].median())
                
            df_proc['GR_PHIND_ratio'] = df_proc['GR'] / (df_proc['PHIND'] + 0.001)
            df_proc.replace([np.inf, -np.inf], np.nan, inplace=True)
            df_proc['GR_PHIND_ratio'] = df_proc['GR_PHIND_ratio'].fillna(df_proc['GR_PHIND_ratio'].median())

            # Generate rolling feature extensions
            for curve in required_curves:
                df_proc[f'{curve}_roll_mean'] = df_proc[curve].rolling(window=3, min_periods=1).mean()
                df_proc[f'{curve}_roll_std'] = df_proc[curve].rolling(window=3, min_periods=1).std().fillna(0)

            features_ordered = [
                'GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS',
                'GR_PHIND_ratio', 'GR_roll_mean', 'GR_roll_std', 'ILD_log10_roll_mean',
                'ILD_log10_roll_std', 'DeltaPHI_roll_mean', 'DeltaPHI_roll_std',
                'PHIND_roll_mean', 'PHIND_roll_std', 'PE_roll_mean', 'PE_roll_std'
            ]
            
            X_live_raw = df_proc[features_ordered]
            X_live_scaled = scaler.transform(X_live_raw)
            
            # Central Matrix Execution Router Logic
            if selected_model == "Random Forest Baseline":
                df_proc['Predicted_Facies'] = rf_model.predict(X_live_scaled)
                
            elif selected_model == "XGBoost Classifier":
                if xgb_model is not None:
                    df_proc['Predicted_Facies'] = xgb_model.predict(X_live_scaled)
                else:
                    st.error("⚠️ XGBoost model file ('best_baseline_xgb.joblib') not found. Defaulting to Random Forest.")
                    df_proc['Predicted_Facies'] = rf_model.predict(X_live_scaled)
                    
            elif selected_model == "Sequential 1D-CNN Architecture":
                if cnn_model is not None:
                    try:
                        X_live_seq = create_live_sequences(X_live_scaled, window_size=5)
                        cnn_probs = cnn_model.predict(X_live_seq)
                        df_proc['Predicted_Facies'] = np.argmax(cnn_probs, axis=1) + 1
                    except Exception as e:
                        st.error(f"⚠️ 1D-CNN Sequence execution error: {e}. Defaulting to Random Forest.")
                        df_proc['Predicted_Facies'] = rf_model.predict(X_live_scaled)
                else:
                    st.error("⚠️ Deep Learning asset ('best_1d_cnn.h5') not found. Defaulting to Random Forest.")
                    df_proc['Predicted_Facies'] = rf_model.predict(X_live_scaled)
                    
            # Automated Performance Scorecard Engine
            if 'FACIES' in df_las.columns:
                st.markdown("---")
                st.write("### Automated AI Performance Scorecard")
                
                true_labels = df_las['FACIES'].fillna(-1).astype(int)
                valid_idx = true_labels != -1
                
                if valid_idx.sum() > 0:
                    acc = accuracy_score(true_labels[valid_idx], df_proc['Predicted_Facies'][valid_idx])
                    
                    m1, m2 = st.columns(2)
                    with m1:
                        st.metric(label="Overall Model Alignment Accuracy", value=f"{acc:.1%}")
                    with m2:
                        st.metric(label="Validated Depth Samples Evaluated", value=f"{valid_idx.sum()} intervals")
                        
                    with st.expander("View Detailed Granular Precision Report"):
                        report = classification_report(true_labels[valid_idx], df_proc['Predicted_Facies'][valid_idx], output_dict=False)
                        st.code(report)

            # Dynamic Log Plotting Subplots Canvas
            st.subheader("Machine Learning Log Interpretation Log Strip")
            
            facies_colors = ['#F4D03F', '#F5B041', '#DC7633', '#A11D33', '#1B4F72', '#2E4053', '#7D6608', '#117A65', '#145A32']
            cmap_facies = colors.ListedColormap(facies_colors, 'indexed')

            if show_advanced:
                fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(15, 10), sharey=True)
                facies_track_idx = 4
            else:
                fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(11, 8), sharey=True)
                facies_track_idx = 2

            ax[0].invert_yaxis()
            
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

            # Extended Engineering Geologic Track Views
            if show_advanced:
                nm_m_data = df_las['NM_M'] if 'NM_M' in df_las.columns else np.ones(len(df_proc))
                relpos_data = df_las['RELPOS'] if 'RELPOS' in df_las.columns else np.linspace(0, 1, len(df_proc))
                
                ax[2].plot(nm_m_data, df_proc['Depth'], color='purple', lw=1.5)
                ax[2].set_title("Marine Block (NM_M)")
                ax[2].set_xlabel("Code")
                ax[2].grid(True, linestyle=':', alpha=0.5)
                
                ax[3].plot(relpos_data, df_proc['Depth'], color='brown', lw=1.0)
                ax[3].set_title("Rel Position (RELPOS)")
                ax[3].set_xlabel("Slope Index")
                ax[3].grid(True, linestyle=':', alpha=0.5)

            # Final Track: Predicted Facies Colored Strip
            pred_strip = np.repeat(df_proc['Predicted_Facies'].values, 100).reshape(-1, 100)
            ax[facies_track_idx].imshow(
                pred_strip, cmap=cmap_facies, aspect='auto', 
                extent=[0, 20, df_proc['Depth'].max(), df_proc['Depth'].min()], vmin=1, vmax=9
            )
            ax[facies_track_idx].set_title("Predicted Facies")
            ax[facies_track_idx].set_xticks([])

            plt.tight_layout()
            st.pyplot(fig)

            # Export Pipeline Metrics Engine
            st.markdown("---")
            st.write("### Export Interpretation Results")
            st.write("Download the fully annotated well logs containing continuous facies layer predictions.")

            @st.cache_data
            def convert_df_to_csv(df):
                return df.to_csv(index=False).encode('utf-8')

            csv_bytes = convert_df_to_csv(df_proc)

            st.download_button(
                label=" Download Facies Predictions (.csv)",
                data=csv_bytes,
                file_name=f"Facies_Predictions_{uploaded_file.name.replace('.las', '')}.csv",
                mime="text/csv",
                key=f"download-csv-{uploaded_file.name}"
            )

            # Interactive Plotly Domain Clustering Scatter Matrix
            st.markdown("---")
            st.write("### Interactive Facies Crossplot Clustering")
            
            df_proc['Hover_Text'] = (
                "Depth: " + df_proc['Depth'].astype(str) + " ft<br>" +
                "Gamma Ray: " + df_proc['GR'].round(1).astype(str) + " API<br>" +
                "Resistivity: " + df_proc['ILD_log10'].round(2).astype(str) + " Log10"
            )

            fig_cross = px.scatter(
                df_proc, x='GR', y='ILD_log10', color='Predicted_Facies',
                hover_name='Hover_Text', color_continuous_scale=facies_colors, range_color=[1, 9],
                labels={'GR': 'Gamma Ray (API)', 'ILD_log10': 'Resistivity (Log10 Ohmm)'},
                title=f"Decision Domains: GR vs. Resistivity - Well: {uploaded_file.name}"
            )

            fig_cross.update_layout(
                template='plotly_white', plot_bgcolor='white', paper_bgcolor='white', font=dict(color='black'),
                coloraxis_colorbar=dict(title="Facies ID", tickvals=list(range(1, 10)), title_font=dict(color='black'), tickfont=dict(color='black')),
                xaxis=dict(gridcolor='rgba(200,200,200,0.5)', linecolor='black', title_font=dict(color='black'), tickfont=dict(color='black')),
                yaxis=dict(gridcolor='rgba(200,200,200,0.5)', linecolor='black', title_font=dict(color='black'), tickfont=dict(color='black'))
            )

            st.plotly_chart(fig_cross, use_container_width=True, key=f"plotly-{uploaded_file.name}")
            
    st.
