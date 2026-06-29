import streamlit as st
import lasio
import pandas as pd
import numpy as np
import joblib
import io
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import plotly.express as px

st.set_page_config(
    page_title="Facies Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS 
st.markdown("""
<style>
    /* Sidebar */
    [data-testid="stSidebar"] { background: #0f1923; }
    [data-testid="stSidebar"] * { color: #c9d8e8 !important; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #5ab4d6 !important; }

    /* Main title */
    h1 { color: #1a3a52 !important; letter-spacing: -0.5px; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #f0f6fb;
        border: 1px solid #c5dde8;
        border-radius: 8px;
        padding: 12px 16px;
    }

    /* Section dividers */
    .section-header {
        font-size: 1.05rem;
        font-weight: 600;
        color: #1a3a52;
        border-left: 4px solid #5ab4d6;
        padding-left: 10px;
        margin: 1.5rem 0 0.75rem 0;
    }
</style>
""", unsafe_allow_html=True)


#  Facies taxonomy 
FACIES_INFO = {
    1: {"name": "Nonmarine Sandstone",    "color": "#F4D03F"},
    2: {"name": "Nonmarine Coarse Silt",  "color": "#F5B041"},
    3: {"name": "Nonmarine Fine Silt",    "color": "#DC7633"},
    4: {"name": "Marine Siltstone",       "color": "#A11D33"},
    5: {"name": "Mudstone",               "color": "#1B4F72"},
    6: {"name": "Wackestone",             "color": "#2E4053"},
    7: {"name": "Dolomite",               "color": "#7D6608"},
    8: {"name": "Packstone-Grainstone",   "color": "#117A65"},
    9: {"name": "Phylloid Algal Bafflestone", "color": "#145A32"},
}
FACIES_COLORS = [FACIES_INFO[i]["color"] for i in range(1, 10)]
CMAP_FACIES   = mcolors.ListedColormap(FACIES_COLORS, "indexed")

CURVE_ALIASES = {
    "GR":        ["GR", "GGCE", "GR_ED", "GAM", "CGR", "SGR", "GRD"],
    "ILD_log10": ["ILD_LOG10", "RTAO", "ILD", "LL3", "RT", "AHT90", "AT90", "RILD"],
    "DeltaPHI":  ["DELTAPHI", "DPHI", "DPOR", "DEPT_PHI", "DPHI_NPHI"],
    "PHIND":     ["PHIND", "XPOR", "NPHI", "PHIN", "NPHI_HL", "POROSITY"],
    "PE":        ["PE", "PDPE", "PEF", "DEN_COR"],
}
REQUIRED_CURVES = list(CURVE_ALIASES.keys())

FEATURES_ORDERED = [
    "GR", "ILD_log10", "DeltaPHI", "PHIND", "PE", "NM_M", "RELPOS",
    "GR_PHIND_ratio",
    "GR_roll_mean", "GR_roll_std",
    "ILD_log10_roll_mean", "ILD_log10_roll_std",
    "DeltaPHI_roll_mean", "DeltaPHI_roll_std",
    "PHIND_roll_mean", "PHIND_roll_std",
    "PE_roll_mean", "PE_roll_std",
]


#  Model loading 
@st.cache_resource
def load_models():
    """Load all trained model assets once and cache in memory."""
    scaler = joblib.load("scaler.joblib")
    rf_model = joblib.load("best_baseline_rf.joblib")

    xgb_model = None
    try:
        xgb_model = joblib.load("best_baseline_xgb.joblib")
    except Exception:
        pass

    cnn_model = None
    try:
        from tensorflow.keras.models import load_model as keras_load
        cnn_model = keras_load("best_1d_cnn.h5")
    except Exception:
        pass

    return scaler, rf_model, xgb_model, cnn_model


scaler, rf_model, xgb_model, cnn_model = load_models()


# Helpers 
def create_live_sequences(scaled_data: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Pad and roll a 2-D feature matrix into 3-D windows for 1-D CNN inference."""
    half_w = window_size // 2
    padded  = np.pad(scaled_data, ((half_w, half_w), (0, 0)), mode="edge")
    return np.array([padded[i : i + window_size] for i in range(len(scaled_data))])


def auto_map_curves(df: pd.DataFrame) -> dict:
    """Return {canonical_name: detected_column_name} for every required curve."""
    mapping = {}
    for curve, aliases in CURVE_ALIASES.items():
        upper_aliases = [a.upper() for a in aliases]
        detected = next(
            (col for col in df.columns if col.upper() in upper_aliases), None
        )
        mapping[curve] = detected  # None if not found
    return mapping


def preprocess(df_las: pd.DataFrame, mapped_columns: dict) -> pd.DataFrame:
    """Build the model-ready feature frame from a raw LAS dataframe."""
    df = pd.DataFrame()
    df["Depth"] = df_las["Depth"]

    for curve in REQUIRED_CURVES:
        col = mapped_columns.get(curve)
        if col and col in df_las.columns:
            df[curve] = pd.to_numeric(df_las[col], errors="coerce")
        else:
            df[curve] = np.nan  # will be median-filled below

    # Auto-convert raw resistivity to log10 scale when needed
    if df["ILD_log10"].median() > 5.0:
        df["ILD_log10"] = np.log10(df["ILD_log10"].clip(lower=0.001))

    # Geological metadata – fall back to safe defaults when absent
    df["NM_M"]   = pd.to_numeric(df_las.get("NM_M",   pd.Series([1]   * len(df))), errors="coerce").fillna(1)
    df["RELPOS"] = pd.to_numeric(df_las.get("RELPOS", pd.Series([0.5] * len(df))), errors="coerce").fillna(0.5)

    # Median-fill missing tool readings
    for col in REQUIRED_CURVES:
        df[col] = df[col].fillna(df[col].median())

    # Engineered features
    df["GR_PHIND_ratio"] = df["GR"] / (df["PHIND"] + 0.001)

    for curve in REQUIRED_CURVES:
        roll = df[curve].rolling(window=3, min_periods=1)
        df[f"{curve}_roll_mean"] = roll.mean()
        df[f"{curve}_roll_std"]  = roll.std().fillna(0)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)

    return df


def run_inference(model_name: str, X_scaled: np.ndarray) -> np.ndarray:
    """Route scaled features to the selected model and return integer predictions."""
    if model_name == "Random Forest":
        return rf_model.predict(X_scaled)

    elif model_name == "XGBoost":
        if xgb_model is not None:
            return xgb_model.predict(X_scaled)
        st.sidebar.warning("XGBoost model not found — falling back to Random Forest.")
        return rf_model.predict(X_scaled)

    elif model_name == "1D-CNN":
        if cnn_model is not None:
            X_seq   = create_live_sequences(X_scaled, window_size=5)
            probs   = cnn_model.predict(X_seq)
            return np.argmax(probs, axis=1) + 1  # map 0-8 → 1-9
        st.sidebar.warning("1D-CNN model not found — falling back to Random Forest.")
        return rf_model.predict(X_scaled)

    # Fallback
    return rf_model.predict(X_scaled)


def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def render_facies_legend():
    """Render a compact colour legend for all 9 facies classes."""
    cols = st.columns(3)
    for i, (fid, info) in enumerate(FACIES_INFO.items()):
        with cols[i % 3]:
            st.markdown(
                f'<span style="display:inline-block;width:14px;height:14px;'
                f'background:{info["color"]};border-radius:2px;margin-right:6px;'
                f'vertical-align:middle;"></span>'
                f'<span style="font-size:0.82rem;">{fid} – {info["name"]}</span>',
                unsafe_allow_html=True,
            )


def render_log_strip(df: pd.DataFrame, df_las: pd.DataFrame, show_advanced: bool):
    """Draw the multi-track well log strip."""
    n_tracks = 5 if show_advanced else 3
    fig, axes = plt.subplots(
        nrows=1, ncols=n_tracks,
        figsize=(14 if show_advanced else 10, 10),
        sharey=True,
    )
    axes[0].invert_yaxis()

    depth = df["Depth"]

    # Track 0 – Gamma Ray
    axes[0].plot(df["GR"], depth, color="#1a3a52", lw=0.9)
    axes[0].set_title("GR", fontsize=9, fontweight="bold")
    axes[0].set_xlabel("API", fontsize=8)
    axes[0].set_ylabel("Depth", fontsize=8)
    axes[0].grid(True, linestyle=":", alpha=0.4)

    # Track 1 – Resistivity
    axes[1].plot(df["ILD_log10"], depth, color="#1565c0", lw=0.9)
    axes[1].set_title("ILD", fontsize=9, fontweight="bold")
    axes[1].set_xlabel("Log₁₀ Ω·m", fontsize=8)
    axes[1].grid(True, linestyle=":", alpha=0.4)

    if show_advanced:
        # Track 2 – PHIND
        axes[2].plot(df["PHIND"], depth, color="#2e7d32", lw=0.9)
        axes[2].set_title("PHIND", fontsize=9, fontweight="bold")
        axes[2].set_xlabel("v/v", fontsize=8)
        axes[2].grid(True, linestyle=":", alpha=0.4)

        # Track 3 – NM_M / RELPOS overlay
        nm   = df_las["NM_M"].values   if "NM_M"   in df_las.columns else np.ones(len(df))
        rpos = df_las["RELPOS"].values  if "RELPOS" in df_las.columns else np.linspace(0, 1, len(df))
        ax3 = axes[3]
        ax3.plot(nm,   depth, color="#6a1b9a", lw=1.2, label="NM_M")
        ax3.plot(rpos, depth, color="#e65100", lw=0.8, ls="--", label="RELPOS")
        ax3.set_title("Env / Pos", fontsize=9, fontweight="bold")
        ax3.set_xlabel("Code / Index", fontsize=8)
        ax3.legend(fontsize=7, loc="lower right")
        ax3.grid(True, linestyle=":", alpha=0.4)

        facies_ax = axes[4]
    else:
        facies_ax = axes[2]

    # Final track – Predicted Facies strip
    pred  = df["Predicted_Facies"].values
    strip = np.repeat(pred, 100).reshape(-1, 100)
    facies_ax.imshow(
        strip, cmap=CMAP_FACIES, aspect="auto",
        extent=[0, 1, depth.max(), depth.min()],
        vmin=1, vmax=9,
    )
    facies_ax.set_title("Facies", fontsize=9, fontweight="bold")
    facies_ax.set_xticks([])

    plt.tight_layout(w_pad=0.3)
    st.pyplot(fig)
    plt.close(fig)



# SIDEBAR
with st.sidebar:
    st.markdown("## Controls")
    st.markdown("---")

    st.markdown("### Model")
    available_models = ["Random Forest"]
    if xgb_model  is not None: available_models.append("XGBoost")
    if cnn_model  is not None: available_models.append("1D-CNN")

    selected_model = st.selectbox("Interpretation engine", available_models)

    st.markdown("---")
    st.markdown("###  Display")
    show_advanced = st.checkbox("Show advanced tracks (PHIND, NM_M, RELPOS)", value=False)

    st.markdown("---")
    st.markdown("###  Model status")
    st.markdown(
        f"{'✅' if True         else '❌'} Random Forest\n\n"
        f"{'✅' if xgb_model    else '⬜'} XGBoost\n\n"
        f"{'✅' if cnn_model    else '⬜'} 1D-CNN",
    )



# MAIN AREA
st.title("Subsurface Facies Prediction")
st.caption(
    "Upload standard `.las` well log files to generate automated machine-learning "
    "lithofacies classifications across all loaded wells."
)

uploaded_files = st.file_uploader(
    "Drop one or more well log files here",
    type=["las"],
    accept_multiple_files=True,
    help="LAS 2.0 and LAS 3.0 formats are supported.",
)

if not uploaded_files:
    st.info("Upload at least one `.las` file to begin.")
    st.stop()

# Process each uploaded well 
for uploaded_file in uploaded_files:
    st.markdown("---")
    st.markdown(f"<div class='section-header'>Well: {uploaded_file.name}</div>", unsafe_allow_html=True)

    # Parse LAS
    try:
        bytes_data = uploaded_file.read()
        str_io     = io.StringIO(bytes_data.decode("utf-8", errors="ignore"))
        las        = lasio.read(str_io)
        df_las     = las.df().reset_index()
        df_las.rename(columns={df_las.columns[0]: "Depth"}, inplace=True)
    except Exception as e:
        st.error(f"Could not parse **{uploaded_file.name}**: {e}")
        continue

    # Show LAS header metadata
    with st.expander(" Well header metadata"):
        header_rows = []
        for item in las.well:
            header_rows.append({"Mnemonic": item.mnemonic, "Unit": item.unit, "Value": item.value, "Description": item.descr})
        if header_rows:
            st.dataframe(pd.DataFrame(header_rows), use_container_width=True, hide_index=True)
        else:
            st.write("No header metadata found in this file.")

    # Curve mapping 
    auto_mapping = auto_map_curves(df_las)
    mapped_columns = dict(auto_mapping)  # will be overridden by manual selects below

    mapped_ok   = [c for c, col in auto_mapping.items() if col is not None]
    mapped_fail = [c for c, col in auto_mapping.items() if col is None]

    if mapped_fail:
        st.warning(
            f"⚠️ Could not auto-detect: **{', '.join(mapped_fail)}**. "
            "Set them manually in the override panel below."
        )
    else:
        st.success(
            f" Auto-mapped: " +
            ", ".join(f"**{c}** → `{col}`" for c, col in auto_mapping.items())
        )

    with st.expander("🔧 Curve mapping overrides"):
        cols = st.columns(len(REQUIRED_CURVES))
        for idx, curve in enumerate(REQUIRED_CURVES):
            with cols[idx]:
                default_col = mapped_columns.get(curve) or df_las.columns[0]
                default_idx = list(df_las.columns).index(default_col) if default_col in df_las.columns else 0
                mapped_columns[curve] = st.selectbox(
                    curve, df_las.columns, index=default_idx,
                    key=f"{uploaded_file.name}_{curve}",
                )

    # Warn about tools that fell back to Depth column
    for curve, col in mapped_columns.items():
        if col in ("Depth", "DEPT"):
            st.sidebar.error(f"⚠️ {uploaded_file.name}: no {curve} log found — accuracy will be reduced.")

    # Preprocessing & feature engineering 
    df_proc = preprocess(df_las, mapped_columns)

    # Depth range filter
    depth_min, depth_max = float(df_proc["Depth"].min()), float(df_proc["Depth"].max())
    with st.expander("Filter depth range"):
        d_lo, d_hi = st.slider(
            "Depth interval to predict",
            min_value=depth_min, max_value=depth_max,
            value=(depth_min, depth_max),
            step=0.5,
            key=f"{uploaded_file.name}_depth",
        )
    mask    = (df_proc["Depth"] >= d_lo) & (df_proc["Depth"] <= d_hi)
    df_sub  = df_proc[mask].copy().reset_index(drop=True)
    df_las_sub = df_las[mask].reset_index(drop=True)

    # Inference
    X_raw    = df_sub[FEATURES_ORDERED]
    X_scaled = scaler.transform(X_raw)
    df_sub["Predicted_Facies"] = run_inference(selected_model, X_scaled)

    # Accuracy scorecard (if ground-truth labels present)
    if "FACIES" in df_las.columns:
        from sklearn.metrics import accuracy_score, classification_report

        true_labels = pd.to_numeric(df_las_sub.get("FACIES", pd.Series()), errors="coerce").fillna(-1).astype(int)
        valid       = true_labels != -1

        if valid.sum() > 0:
            acc = accuracy_score(true_labels[valid], df_sub["Predicted_Facies"][valid])
            st.markdown("<div class='section-header'> Model scorecard</div>", unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Overall accuracy",          f"{acc:.1%}")
            c2.metric("Validated intervals",       f"{valid.sum():,}")
            c3.metric("Depth range",               f"{d_lo:.0f} – {d_hi:.0f} ft")

            with st.expander(" Full classification report"):
                report = classification_report(
                    true_labels[valid], df_sub["Predicted_Facies"][valid]
                )
                st.code(report)

    # Facies distribution bar chart
    st.markdown("<div class='section-header'> Predicted facies distribution</div>", unsafe_allow_html=True)
    counts = df_sub["Predicted_Facies"].value_counts().sort_index()
    bar_df = pd.DataFrame({
        "Facies": [f"{i} – {FACIES_INFO[i]['name']}" for i in counts.index],
        "Count":  counts.values,
        "Color":  [FACIES_INFO[i]["color"]            for i in counts.index],
    })
    fig_bar = px.bar(
        bar_df, x="Facies", y="Count", color="Facies",
        color_discrete_map={row["Facies"]: row["Color"] for _, row in bar_df.iterrows()},
        template="plotly_white",
        labels={"Count": "Depth intervals"},
    )
    fig_bar.update_layout(showlegend=False, xaxis_tickangle=-30, margin=dict(t=20, b=60))
    st.plotly_chart(fig_bar, use_container_width=True)

    # Log strip 
    st.markdown("<div class='section-header'> Log strip</div>", unsafe_allow_html=True)
    render_log_strip(df_sub, df_las_sub, show_advanced)
    render_facies_legend()

    # Crossplot
    st.markdown("<div class='section-header'>GR vs Resistivity crossplot</div>", unsafe_allow_html=True)
    df_sub["Facies_Label"] = df_sub["Predicted_Facies"].map(
        lambda i: f"{i} – {FACIES_INFO.get(i, {}).get('name', str(i))}"
    )
    fig_cross = px.scatter(
        df_sub,
        x="GR", y="ILD_log10",
        color="Facies_Label",
        color_discrete_map={
            f"{i} – {v['name']}": v["color"] for i, v in FACIES_INFO.items()
        },
        hover_data={"Depth": True, "PE": True, "PHIND": True, "Facies_Label": False},
        labels={"GR": "Gamma Ray (API)", "ILD_log10": "Resistivity (Log₁₀ Ω·m)", "Facies_Label": "Facies"},
        template="plotly_white",
        opacity=0.75,
    )
    fig_cross.update_traces(marker_size=4)
    fig_cross.update_layout(
        legend_title_text="Facies",
        margin=dict(t=30),
    )
    st.plotly_chart(fig_cross, use_container_width=True)

    # CSV download
    st.markdown("<div class='section-header'>Export results</div>", unsafe_allow_html=True)
    csv_bytes = convert_df_to_csv(df_sub)
    stem      = uploaded_file.name.rsplit(".", 1)[0]
    st.download_button(
        label=f"Download {stem}_facies.csv",
        data=csv_bytes,
        file_name=f"{stem}_facies.csv",
        mime="text/csv",
        key=f"dl_{uploaded_file.name}",
    )

st.markdown("---")
st.caption("Facies Predictor · built with Streamlit · Anthropic Claude")
